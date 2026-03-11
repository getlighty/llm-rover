"""Graph-based navigation orchestrator for rover_brain_v2."""

from __future__ import annotations

import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import room_context
import topo_nav

from rover_brain_v2.json_utils import extract_json_dict
from rover_brain_v2.models import NavigatorResult, OrchestratorResult, OrchestratorTask
from rover_brain_v2.prompts import load_prompt, room_reorg_prompt, scene_prompt


LEG_RETRY_ATTEMPTS = 4


def _ra(room, key, default=""):
    """Get room attribute whether room is a dict or an object."""
    if isinstance(room, dict):
        return room.get(key, default)
    return getattr(room, key, default)


class GraphNavigationOrchestrator:
    def __init__(self, *, llm_client, navigator, event_bus):
        self.llm = llm_client
        self.navigator = navigator
        self.events = event_bus
        self.topo = self._load_topo_map()
        self._stop_event = threading.Event()
        self._orchestrator_tasks: queue.Queue[OrchestratorTask] = queue.Queue()
        self._orchestrator_results: queue.Queue[OrchestratorResult] = queue.Queue()
        self._orchestrator_result_cache: dict[int, OrchestratorResult] = {}
        self._orchestrator_task_seq = 0
        self._orchestrator_worker_shutdown = threading.Event()
        self._orchestrator_worker = threading.Thread(
            target=self._orchestrator_loop,
            daemon=True,
            name="rover-v2-orchestrator-worker",
        )
        self._orchestrator_worker.start()

    def cancel(self):
        self._stop_event.set()
        self.navigator.cancel()

    def shutdown(self):
        self._orchestrator_worker_shutdown.set()
        self._stop_event.set()

    def navigate(self, target_query: str) -> bool:
        return self.run_navigation_task(target_query).reached

    def run_navigation_task(self, target_query: str, *, topological: bool = True) -> OrchestratorResult:
        self._stop_event.clear()
        task = self._new_task(target_query=target_query, topological=topological)
        self._orchestrator_tasks.put(task)
        return self._wait_for_task_result(task.task_id, target_query)

    def _new_task(self, *, target_query: str, topological: bool) -> OrchestratorTask:
        self._orchestrator_task_seq += 1
        return OrchestratorTask(
            task_id=self._orchestrator_task_seq,
            target_query=target_query,
            topological=topological,
        )

    def _wait_for_task_result(self, task_id: int, target_query: str,
                              timeout_s: float | None = None) -> OrchestratorResult:
        started = time.time()
        while True:
            cached = self._orchestrator_result_cache.pop(task_id, None)
            if cached is not None:
                return cached
            if timeout_s is not None and time.time() - started >= timeout_s:
                return OrchestratorResult(
                    task_id=task_id,
                    status="timeout",
                    target_query=target_query,
                    summary="orchestrator task timed out",
                )
            if self._stop_event.is_set():
                return OrchestratorResult(
                    task_id=task_id,
                    status="aborted",
                    target_query=target_query,
                    summary="orchestrator task aborted",
                )
            try:
                result = self._orchestrator_results.get(timeout=0.2)
            except queue.Empty:
                continue
            if result.task_id == task_id:
                self.events.publish(
                    "plan",
                    f"Orchestrator result #{result.task_id}: {result.status} | {result.summary}",
                )
                return result
            self._orchestrator_result_cache[result.task_id] = result

    def _orchestrator_loop(self):
        while not self._orchestrator_worker_shutdown.is_set():
            try:
                task = self._orchestrator_tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            result = self._execute_task(task)
            self._orchestrator_results.put(result)

    def _execute_task(self, task: OrchestratorTask) -> OrchestratorResult:
        self._stop_event.clear()
        try:
            if not task.topological:
                return self._execute_direct_reactive_task(task)
            return self._execute_topological_task(task)
        except Exception as exc:
            return OrchestratorResult(
                task_id=task.task_id,
                status="error",
                target_query=task.target_query,
                summary=f"orchestrator crashed: {exc}",
            )

    def _execute_direct_reactive_task(self, task: OrchestratorTask) -> OrchestratorResult:
        self.events.publish("plan", f"Direct reactive navigation: {task.target_query}")
        nav_result = self.navigator.run_reactive_task(task.target_query)
        return OrchestratorResult(
            task_id=task.task_id,
            status="completed" if nav_result.reached else nav_result.status,
            target_query=task.target_query,
            current_room=getattr(self.topo, "current_room", None),
            reached=nav_result.reached,
            summary=nav_result.summary,
            payload={"navigator_result": self._navigator_result_payload(nav_result)},
        )

    def _execute_topological_task(self, task: OrchestratorTask) -> OrchestratorResult:
        target_room = self._resolve_room(task.target_query)
        if not target_room:
            self.events.publish("plan", f"Unknown room '{task.target_query}', using reactive navigation")
            return self._execute_direct_reactive_task(task)

        # Always re-identify current room before navigation to avoid stale state
        fresh_room = self._identify_current_room()
        current_room = fresh_room or self.topo.current_room
        if fresh_room and fresh_room != self.topo.current_room:
            self.events.publish("plan", f"Room re-identified: {self.topo.current_room} → {fresh_room}")
            self.topo.current_room = fresh_room
            self._save_topo()
        if self._stop_event.is_set():
            return self._aborted_result(task, target_room=target_room)
        if not current_room:
            self.events.publish("plan", "Current room unknown, using reactive navigation")
            nav_result = self.navigator.run_reactive_task(
                target_room,
                plan_context=f"Reach {target_room} without graph localization.",
            )
            if nav_result.reached:
                self.topo.current_room = target_room
                self._save_topo()
            return OrchestratorResult(
                task_id=task.task_id,
                status="completed" if nav_result.reached else nav_result.status,
                target_query=task.target_query,
                target_room=target_room,
                current_room=getattr(self.topo, "current_room", None),
                reached=nav_result.reached,
                summary=nav_result.summary,
                payload={"navigator_result": self._navigator_result_payload(nav_result)},
            )
        if current_room == target_room:
            self.events.publish("plan", f"Already in {target_room}")
            return OrchestratorResult(
                task_id=task.task_id,
                status="completed",
                target_query=task.target_query,
                target_room=target_room,
                current_room=current_room,
                reached=True,
                summary=f"already in {target_room}",
            )

        legs = self.topo.plan_route(current_room, target_room)
        if not legs:
            self.events.publish(
                "plan",
                f"No graph route from {current_room} to {target_room}, using reactive navigation",
            )
            nav_result = self.navigator.run_reactive_task(
                target_room,
                plan_context=f"Reach room {target_room} without graph route.",
            )
            if nav_result.reached:
                self.topo.current_room = target_room
                self._save_topo()
            return OrchestratorResult(
                task_id=task.task_id,
                status="completed" if nav_result.reached else nav_result.status,
                target_query=task.target_query,
                target_room=target_room,
                current_room=getattr(self.topo, "current_room", None),
                reached=nav_result.reached,
                summary=nav_result.summary,
                payload={"navigator_result": self._navigator_result_payload(nav_result)},
            )

        self.events.publish("plan", f"Route: {self.topo.route_summary(legs)}")
        completed_legs = 0
        leg_history: list[dict] = []
        leg_index = 0
        while leg_index < len(legs):
            if self._stop_event.is_set():
                self._save_topo()
                return self._aborted_result(
                    task,
                    target_room=target_room,
                    current_room=getattr(self.topo, "current_room", None),
                    completed_legs=completed_legs,
                    payload={"legs": leg_history},
                )
            leg = legs[leg_index]
            next_leg = legs[leg_index + 1] if leg_index + 1 < len(legs) else None
            instruction = self.topo.leg_instruction(leg, next_leg=next_leg)
            self.events.publish(
                "plan",
                f"Leg {leg_index + 1}/{len(legs)}: {leg.from_room} -> {leg.to_room} via {leg.transition}",
            )
            leg_success, actual_room, scene, leg_payload = self._complete_leg(leg, instruction)
            leg_history.append(leg_payload)
            if not leg_success:
                self._save_topo()
                return OrchestratorResult(
                    task_id=task.task_id,
                    status="incomplete",
                    target_query=task.target_query,
                    target_room=target_room,
                    current_room=getattr(self.topo, "current_room", None),
                    reached=False,
                    completed_legs=completed_legs,
                    summary=f"failed to complete leg {leg.from_room} -> {leg.to_room}",
                    payload={"legs": leg_history},
                )

            actual_room = actual_room or leg.to_room
            self.topo.current_room = actual_room
            self.topo.current_confidence = 0.65
            completed_legs += 1
            self._persist_semantic_memory(
                nav_target=task.target_query,
                actual_room=actual_room,
                from_room=leg.from_room,
                transition_id=leg.transition,
                scene_text=scene,
            )
            if actual_room == target_room:
                self._save_topo()
                return OrchestratorResult(
                    task_id=task.task_id,
                    status="completed",
                    target_query=task.target_query,
                    target_room=target_room,
                    current_room=actual_room,
                    reached=True,
                    completed_legs=completed_legs,
                    summary=f"reached {target_room}",
                    payload={"legs": leg_history},
                )
            if actual_room != leg.to_room:
                replan = self.topo.plan_route(actual_room, target_room)
                if not replan:
                    self.events.publish("plan", f"Unexpected room {actual_room}, no replan available")
                    self._save_topo()
                    return OrchestratorResult(
                        task_id=task.task_id,
                        status="incomplete",
                        target_query=task.target_query,
                        target_room=target_room,
                        current_room=actual_room,
                        reached=False,
                        completed_legs=completed_legs,
                        summary=f"entered {actual_room} but could not replan to {target_room}",
                        payload={"legs": leg_history},
                    )
                legs = replan
                leg_index = 0
                self.events.publish("plan", f"Replanned: {self.topo.route_summary(legs)}")
                continue
            leg_index += 1

        self._save_topo()
        reached = self.topo.current_room == target_room
        return OrchestratorResult(
            task_id=task.task_id,
            status="completed" if reached else "incomplete",
            target_query=task.target_query,
            target_room=target_room,
            current_room=getattr(self.topo, "current_room", None),
            reached=reached,
            completed_legs=completed_legs,
            summary=f"reached {target_room}" if reached else f"stopped before reaching {target_room}",
            payload={"legs": leg_history},
        )

    def _visual_guidance(self, leg, instruction: dict, route_plan: str,
                         attempt_scenes: list[str]) -> str:
        """Ask the LLM to look at the camera and describe where the target is.

        Returns a visual guidance string to prepend to the navigator's context.
        """
        frame = self.navigator.camera.snap(max_dim=512, quality=60)
        if frame is None:
            return ""

        target_room = instruction.get("target_room", leg.to_room)
        transition = leg.transition
        visual_cues = instruction.get("visual_cues", [])
        cues_text = ", ".join(visual_cues[:4]) if visual_cues else "unknown"

        history_text = ""
        if attempt_scenes:
            history_text = "\nPrevious attempts saw: " + "; ".join(
                s[:80] for s in attempt_scenes[-3:] if s)

        prompt = (
            f"You are guiding a ground rover. Look at this camera image.\n\n"
            f"FULL ROUTE PLAN: {route_plan}\n"
            f"CURRENT LEG: {leg.from_room} → {leg.to_room} via '{transition}'\n"
            f"VISUAL CUES for doorway: {cues_text}\n"
            f"{history_text}\n\n"
            f"Answer these questions about what you see RIGHT NOW:\n"
            f"1. Where is the doorway/exit to {target_room}? (left, center, right, not visible)\n"
            f"2. What obstacles are between the rover and the doorway?\n"
            f"3. What is the best path to reach the doorway from here?\n"
            f"4. Approximate angle to the doorway (-90=far left, 0=center, +90=far right)?\n\n"
            f"Respond with ONLY JSON:\n"
            f'{{"direction": "center-right", "angle_deg": 30, '
            f'"obstacles": "dining chair, pet basket", '
            f'"best_path": "go right to avoid chair, then turn left toward arch", '
            f'"doorway_visible": true}}'
        )
        try:
            raw = self.llm.complete(
                prompt=prompt,
                system="You are a visual navigation guide. Respond with JSON only.",
                image_bytes=frame,
                max_tokens=150,
                temperature=0.2,
            )
            data = extract_json_dict(raw)
            if isinstance(data, dict):
                direction = data.get("direction", "unknown")
                angle = data.get("angle_deg", 0)
                obstacles = data.get("obstacles", "none")
                best_path = data.get("best_path", "drive forward")
                visible = data.get("doorway_visible", False)
                guidance = (
                    f"ORCHESTRATOR VISUAL GUIDANCE: "
                    f"Target doorway is {direction} (approx {angle}°). "
                    f"Doorway visible: {visible}. "
                    f"Obstacles: {obstacles}. "
                    f"Recommended path: {best_path}."
                )
                self.events.publish("plan", f"Visual guidance: {guidance[:120]}")
                return guidance
        except Exception as exc:
            self.events.publish("error", f"Visual guidance failed: {exc}")
        return ""

    def _complete_leg(self, leg, instruction: dict):
        max_attempts = max(1, int(getattr(self.navigator.config, "navigation_leg_attempts", LEG_RETRY_ATTEMPTS)))
        waypoint_budget = max(getattr(self.navigator.config, "navigation_waypoint_budget", 40) * 2, 60)
        attempt_history: list[dict] = []
        last_scene = ""
        last_room_guess = None
        attempt_scenes: list[str] = []

        # Build full route plan text for context
        route_plan = self.topo.route_summary(
            self.topo.plan_route(leg.from_room, instruction.get("target_room", leg.to_room)) or []
        )

        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return False, last_room_guess, last_scene, {
                    "from_room": leg.from_room,
                    "to_room": leg.to_room,
                    "transition": leg.transition,
                    "status": "aborted",
                    "attempts": attempt_history,
                }

            # Give navigator the orchestrator's LLM and route context
            # so dot confirmation checks strategic value, not just safety
            self.navigator.orchestrator_llm = self.llm
            self.navigator._nav_route_context = (
                f"Route: {route_plan}. Current leg: {leg.from_room} → {leg.to_room} "
                f"via {leg.transition}."
            )

            # Orchestrator looks at the camera and provides visual guidance
            visual_guidance = self._visual_guidance(
                leg, instruction, route_plan, attempt_scenes)

            # Adapt instruction for retries
            adapted = self._adapt_instruction_for_retry(
                instruction, attempt, max_attempts, attempt_scenes,
            )
            # Inject visual guidance into the instruction
            if visual_guidance:
                prev_hints = adapted.get("room_nav_hints", "")
                adapted["room_nav_hints"] = f"{visual_guidance} {prev_hints}".strip()

            self.events.publish(
                "plan",
                f"Navigator round {attempt + 1}/{max_attempts} for {leg.transition}: "
                f"insist on {leg.to_room} (azimuth {adapted.get('expected_azimuth_deg', '?')}°)",
            )
            nav_result = self.navigator.run_doorway_task(
                adapted,
                attempt=attempt,
                total_attempts=max_attempts,
                waypoint_budget=waypoint_budget,
            )
            attempt_history.append(self._navigator_result_payload(nav_result))
            last_scene = nav_result.scene or last_scene
            attempt_scenes.append(last_scene)
            room_guess, confidence = self._room_check(last_scene) if last_scene else (None, 0.0)
            if room_guess:
                last_room_guess = room_guess
                self.events.publish(
                    "room",
                    f"Navigator scene suggests {room_guess} ({confidence:.2f}) after {leg.transition}",
                )
            if nav_result.status == "aborted":
                return False, room_guess, last_scene, {
                    "from_room": leg.from_room,
                    "to_room": leg.to_room,
                    "transition": leg.transition,
                    "status": "aborted",
                    "attempts": attempt_history,
                }
            if nav_result.status == "error":
                self.events.publish(
                    "plan",
                    f"Navigator error on {leg.transition}; reissuing same doorway task",
                )
                continue
            if nav_result.reached or room_guess == leg.to_room:
                actual_room = room_guess or leg.to_room
                self.events.publish(
                    "plan",
                    f"Leg satisfied via navigator: {leg.transition} -> {actual_room}",
                )
                return True, actual_room, last_scene, {
                    "from_room": leg.from_room,
                    "to_room": leg.to_room,
                    "transition": leg.transition,
                    "status": "completed",
                    "attempts": attempt_history,
                }
            self.events.publish(
                "plan",
                f"Navigator incomplete on {leg.transition}; adapting for retry {attempt + 2}",
            )
        return False, last_room_guess, last_scene, {
            "from_room": leg.from_room,
            "to_room": leg.to_room,
            "transition": leg.transition,
            "status": "incomplete",
            "attempts": attempt_history,
        }

    def _adapt_instruction_for_retry(self, instruction: dict, attempt: int,
                                      total_attempts: int,
                                      attempt_scenes: list[str]) -> dict:
        """Adapt the doorway instruction based on previous failed attempts.
        Widens the search angle and adds failure context so the navigator
        tries a different strategy each round."""
        adapted = dict(instruction)
        original_azimuth = instruction.get("expected_azimuth_deg")
        if attempt == 0:
            return adapted

        # Build failure context from previous attempts
        failure_context = []
        for i, scene in enumerate(attempt_scenes):
            if scene:
                failure_context.append(f"Attempt {i + 1} saw: {scene[:100]}")

        prev_hint = adapted.get("room_nav_hints", "")
        retry_hint = (
            f"Previous {attempt} attempt(s) failed to reach the doorway. "
            f"Try a DIFFERENT approach: move to a new position, try the opposite side of the room, "
            f"or navigate around furniture from a different angle. "
        )
        if failure_context:
            retry_hint += "Prior observations: " + "; ".join(failure_context[-2:]) + ". "
        adapted["room_nav_hints"] = f"{retry_hint}{prev_hint}".strip()

        # Rotate the expected azimuth on each retry
        if original_azimuth is not None:
            azimuth = float(original_azimuth)
            # Alternate: try opposite side, then wider angles
            if attempt == 1:
                azimuth = -azimuth  # try opposite side
            elif attempt == 2:
                azimuth = azimuth * 0.5  # try center-ish
            elif attempt >= 3:
                azimuth = 0  # full sweep, no bias
            adapted["expected_azimuth_deg"] = azimuth
            self.events.publish(
                "plan",
                f"Retry {attempt + 1}: azimuth adjusted {original_azimuth}° -> {azimuth}°",
            )
        else:
            # No original azimuth — add sweep hints
            sweep_angles = [90, -90, 45, -45]
            if attempt <= len(sweep_angles):
                adapted["expected_azimuth_deg"] = sweep_angles[attempt - 1]
                self.events.publish(
                    "plan",
                    f"Retry {attempt + 1}: blind sweep at {sweep_angles[attempt - 1]}°",
                )

        return adapted

    def _navigator_result_payload(self, result: NavigatorResult) -> dict:
        return {
            "task_id": result.task_id,
            "mode": result.mode,
            "status": result.status,
            "summary": result.summary,
            "scene": result.scene,
            "reached": result.reached,
            "room_guess": result.room_guess,
            "payload": dict(result.payload),
        }

    def _aborted_result(self, task: OrchestratorTask, *, target_room: str | None = None,
                        current_room: str | None = None, completed_legs: int = 0,
                        payload: dict | None = None) -> OrchestratorResult:
        return OrchestratorResult(
            task_id=task.task_id,
            status="aborted",
            target_query=task.target_query,
            target_room=target_room,
            current_room=current_room,
            reached=False,
            completed_legs=completed_legs,
            summary="orchestrator task aborted",
            payload=dict(payload or {}),
        )

    def _resolve_room(self, target_query: str) -> Optional[str]:
        target = self._normalize_room_text(target_query)
        if not target:
            return None
        target_tokens = set(target.split())
        best_room = None
        best_score = -1
        for room in self.topo.rooms():
            for alias in self._room_aliases(room):
                alias_tokens = set(alias.split())
                score = -1
                if target == alias:
                    score = 100 + len(alias)
                elif alias in target:
                    score = 80 + len(alias)
                elif target in alias:
                    score = 60 + len(target)
                elif alias_tokens and alias_tokens.issubset(target_tokens):
                    score = 40 + len(alias_tokens)
                if score > best_score:
                    best_score = score
                    best_room = _ra(room, "id")
        return best_room

    def _room_aliases(self, room) -> list[str]:
        rid = _ra(room, "id", "")
        rlabel = _ra(room, "label", "")
        aliases = {
            self._normalize_room_text(rid),
            self._normalize_room_text(rlabel),
            self._normalize_room_text(rid.replace("_", " ")),
            self._normalize_room_text(rlabel.replace("_", " ")),
        }
        return [alias for alias in aliases if alias]

    def _normalize_room_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s_]+", " ", (text or "").lower())
        normalized = normalized.replace("_", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _identify_current_room(self) -> Optional[str]:
        # Trust existing topo state if confidence is reasonable
        if self.topo.current_room and getattr(self.topo, 'current_confidence', 0) >= 0.5:
            self.events.publish("room", f"Current room (topo): {self.topo.current_room} "
                                f"({getattr(self.topo, 'current_confidence', 0):.2f})")
            return self.topo.current_room
        # Low confidence or unknown — use navigator's last scene if available
        last_scene = getattr(self.navigator, '_last_scene', '') or ''
        if last_scene:
            room, confidence = self._room_check(last_scene)
            if room:
                self.topo.current_room = room
                self.topo.current_confidence = confidence
                self.events.publish("room", f"Current room (scene): {room} ({confidence:.2f})")
                return room
        # Last resort: ask navigator to describe the scene (text only, no image LLM call)
        try:
            frame = self.navigator.camera.snap()
            if frame is not None:
                scene = self.navigator.describe_scene(frame, "Identify the room")
                if scene:
                    room, confidence = self._room_check(scene)
                    if room:
                        self.topo.current_room = room
                        self.topo.current_confidence = confidence
                        self.events.publish("room", f"Current room (scene fallback): {room} ({confidence:.2f})")
                        return room
        except Exception as exc:
            self.events.publish("error", f"Current-room check failed: {exc}")
        return None

    def _room_check(self, scene_text: str):
        ranked = room_context.identify_room(scene_text)
        if ranked and ranked[0][1] > 0:
            return ranked[0][0], ranked[0][2]
        return None, 0.0

    def _persist_semantic_memory(self, *, nav_target: str, actual_room: str,
                                 from_room: Optional[str], transition_id: Optional[str],
                                 scene_text: str):
        room_context.merge_room_observation(
            actual_room,
            scene_text=scene_text,
            mark_current=True,
            confidence=self.topo.current_confidence or 0.65,
            connected_to=from_room,
        )
        if not from_room or from_room == actual_room:
            self._save_topo()
            return
        prompt = room_reorg_prompt(
            nav_target=nav_target,
            actual_room=actual_room,
            from_room=from_room,
            transition_id=transition_id,
            scene_text=scene_text,
        )
        doorway_landmarks = []
        inside_features = []
        navigational_hint = ""
        confidence = self.topo.current_confidence or 0.65
        try:
            raw = self.llm.complete(
                prompt=prompt,
                system=load_prompt("json_only.system.md"),
                max_tokens=400,
            )
            parsed = extract_json_dict(raw) or {}
            transition = parsed.get("transition_update", {})
            if isinstance(transition, dict):
                doorway_landmarks = list(transition.get("doorway_landmarks", [])[:6])
                inside_features = list(transition.get("inside_features", [])[:6])
                navigational_hint = str(transition.get("navigational_hint", "")).strip()[:200]
                if transition.get("confidence") is not None:
                    confidence = float(transition.get("confidence") or confidence)
            room_features = parsed.get("room_features", [])
            entry_landmarks = parsed.get("entry_landmarks", [])
            room_context.merge_room_observation(
                actual_room,
                features=room_features,
                entry_landmarks=entry_landmarks,
                scene_text=scene_text,
                mark_current=True,
                confidence=confidence,
                connected_to=from_room,
            )
        except Exception as exc:
            self.events.publish("error", f"Semantic memory update failed: {exc}")
        transition = self.topo.ensure_transition_between(
            from_room,
            actual_room,
            transition_id=transition_id,
        )
        self.topo.update_transition_semantics(
            _ra(transition, "id"),
            from_room,
            actual_room,
            doorway_landmarks=doorway_landmarks,
            inside_features=inside_features,
            navigational_hint=navigational_hint,
            inside_room_guess=actual_room,
            confidence=confidence,
            scene_text=scene_text,
        )
        self._save_topo()
        self.events.publish(
            "room",
            f"Semantic memory updated: {from_room} -> {actual_room}",
        )

    def _load_topo_map(self):
        source = Path(topo_nav.MAP_FILE)
        target = Path(__file__).resolve().parents[1] / "data" / "topo_map.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
        topo_nav.MAP_FILE = str(target)
        topo = topo_nav.TopoMap()
        self.events.publish("system", f"V2 topo map path: {target}")
        return topo

    def _save_topo(self):
        try:
            self.topo.save()
        except Exception as exc:
            self.events.publish("error", f"Topo save failed: {exc}")
