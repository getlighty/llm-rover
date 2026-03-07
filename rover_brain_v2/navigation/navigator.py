"""Depth-vector local navigation driven by a navigator LLM."""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time

from rover_brain_v2.json_utils import as_float, extract_json_dict, clamp
from rover_brain_v2.models import NavigatorResult, NavigatorTask
from rover_brain_v2.navigation.depth_vectors import DepthVectorMap
from rover_brain_v2.prompts import navigation_prompt, scene_prompt


TURN_SPEED = 0.35
TURN_RATE_DPS = 200.0
DRIVE_COMMAND_REFRESH_S = 0.35
LEG_SEARCH_ATTEMPTS = 4
BLOCKED_ESCAPE_REVERSE_M = 0.18
BLOCKED_ESCAPE_FURNITURE_REVERSE_M = 0.22
BLOCKED_ESCAPE_MIN_TURN_DEG = 35.0
BLOCKED_ESCAPE_MAX_TURN_DEG = 65.0
FURNITURE_TRAP_HINTS = (
    "chair leg",
    "chair legs",
    "office chair",
    "chair base",
    "chair wheel",
    "chair wheels",
    "caster",
    "desk leg",
    "desk edge",
    "table leg",
    "table legs",
    "under desk",
    "under chair",
)
ZONE_SIGNATURE_STOPWORDS = {
    "the", "and", "with", "from", "into", "toward", "towards", "this", "that", "there",
    "here", "just", "very", "near", "ahead", "visible", "beyond", "through", "inside",
    "outside", "front", "left", "right", "center", "small", "large", "bright", "dark",
    "room", "area", "space", "same", "still", "looks", "look", "like", "open", "close",
    "closed", "floor", "wall", "walls", "door", "doorway",
}
ZONE_REPEAT_SIMILARITY = 0.58
GOAL_CUE_GENERIC_TOKENS = {"door", "doorway", "arch", "arched", "frame", "threshold", "opening"}


class DepthVectorNavigator:
    def __init__(self, *, rover, camera, llm_client, event_bus, flags, config):
        self.rover = rover
        self.camera = camera
        self.llm = llm_client
        self.events = event_bus
        self.flags = flags
        self.config = config
        self.stop_event = threading.Event()
        self.depth_vectors = DepthVectorMap(
            num_columns=21,
            corridor_width=7,
            floor_top=0.40,
            floor_bottom=0.88,
            depth_max_clearance_m=2.4,
            passable_clearance_m=0.45,
        )
        self._scan_index = 0
        self._last_scene = ""
        self._recent_observations: list[str] = []
        self._last_depth_summary = None
        self._blocked_escape_dir = 1
        self._goal_heading_bias_deg: float | None = None
        self._goal_heading_confidence = 0.0
        self._goal_reacquire_steps = 0
        self._goal_target_label = ""
        self._goal_recently_seen_steps = 0
        self._goal_cue_tokens: set[str] = set()
        self._navigator_tasks: queue.Queue[NavigatorTask] = queue.Queue()
        self._navigator_results: queue.Queue[NavigatorResult] = queue.Queue()
        self._navigator_result_cache: dict[int, NavigatorResult] = {}
        self._navigator_task_seq = 0
        self._navigator_worker_shutdown = threading.Event()
        self._navigator_worker = threading.Thread(
            target=self._navigator_loop,
            daemon=True,
            name="rover-v2-navigator-worker",
        )
        self._navigator_worker.start()

    @property
    def last_scene(self) -> str:
        return self._last_scene

    def cancel(self):
        self.stop_event.set()
        self.rover.stop()

    def shutdown(self):
        self._navigator_worker_shutdown.set()

    def reset(self):
        self.stop_event.clear()
        self._scan_index = 0
        self._recent_observations = []
        self._last_scene = ""
        self._last_depth_summary = None

    def run_reactive_task(self, target: str, *,
                          plan_context: str = "",
                          leg_hint: str = "",
                          waypoint_budget: int | None = None) -> NavigatorResult:
        self.stop_event.clear()
        task = self._new_task(
            mode="reactive_goal",
            target=target,
            plan_context=plan_context,
            leg_hint=leg_hint,
            waypoint_budget=waypoint_budget,
        )
        self._navigator_tasks.put(task)
        return self._wait_for_task_result(task.task_id)

    def run_doorway_task(self, instruction: dict, *,
                         attempt: int,
                         total_attempts: int = LEG_SEARCH_ATTEMPTS,
                         immediate_target: str = "",
                         plan_context: str = "",
                         leg_hint: str = "",
                         waypoint_budget: int | None = None) -> NavigatorResult:
        self.stop_event.clear()
        instruction = dict(instruction or {})
        immediate_target = immediate_target or self._leg_target_phrase(instruction)
        plan_context = plan_context or self._leg_plan_context(instruction, attempt, total_attempts)
        leg_hint = leg_hint or self._format_leg_hint(instruction)
        task = self._new_task(
            mode="doorway_search",
            target=immediate_target,
            plan_context=plan_context,
            leg_hint=leg_hint,
            waypoint_budget=waypoint_budget,
            instruction=instruction,
            attempt=attempt,
        )
        self._navigator_tasks.put(task)
        return self._wait_for_task_result(task.task_id)

    def _new_task(self, *, mode: str, target: str, plan_context: str = "",
                  leg_hint: str = "", waypoint_budget: int | None = None,
                  instruction: dict | None = None, attempt: int = 0) -> NavigatorTask:
        self._navigator_task_seq += 1
        return NavigatorTask(
            task_id=self._navigator_task_seq,
            mode=mode,
            target=target,
            plan_context=plan_context,
            leg_hint=leg_hint,
            waypoint_budget=waypoint_budget,
            instruction=dict(instruction or {}),
            attempt=attempt,
        )

    def _wait_for_task_result(self, task_id: int, timeout_s: float | None = None) -> NavigatorResult:
        started = time.time()
        while True:
            cached = self._navigator_result_cache.pop(task_id, None)
            if cached is not None:
                return cached
            if timeout_s is not None and time.time() - started >= timeout_s:
                return NavigatorResult(
                    task_id=task_id,
                    mode="unknown",
                    status="timeout",
                    summary="navigator task timed out",
                )
            try:
                result = self._navigator_results.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set():
                    return NavigatorResult(
                        task_id=task_id,
                        mode="unknown",
                        status="aborted",
                        summary="navigator task aborted",
                    )
                continue
            if result.task_id == task_id:
                self.events.publish(
                    "nav",
                    f"Navigator result #{result.task_id}: {result.status} | {result.summary}",
                )
                return result
            self._navigator_result_cache[result.task_id] = result

    def _navigator_loop(self):
        while not self._navigator_worker_shutdown.is_set():
            try:
                task = self._navigator_tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            result = self._execute_task(task)
            self._navigator_results.put(result)

    def _execute_task(self, task: NavigatorTask) -> NavigatorResult:
        try:
            if task.mode == "doorway_search":
                return self._execute_doorway_task(task)
            reached = self.navigate_reactive(
                task.target,
                plan_context=task.plan_context,
                leg_hint=task.leg_hint,
                waypoint_budget=task.waypoint_budget,
            )
            return NavigatorResult(
                task_id=task.task_id,
                mode=task.mode,
                status="completed" if reached else "incomplete",
                summary="reactive target reached" if reached else "reactive target not reached",
                scene=self.last_scene,
                reached=reached,
                payload={"target": task.target},
            )
        except Exception as exc:
            return NavigatorResult(
                task_id=task.task_id,
                mode=task.mode,
                status="error",
                summary=f"navigator crashed: {exc}",
                scene=self.last_scene,
                payload={"target": task.target},
            )

    def _execute_doorway_task(self, task: NavigatorTask) -> NavigatorResult:
        instruction = dict(task.instruction or {})
        self._begin_leg_goal_tracking(instruction, task.target)
        try:
            if task.attempt > 0:
                self._door_search_reposition(instruction, task.attempt)
            reached = self.navigate_reactive(
                task.target,
                plan_context=task.plan_context,
                leg_hint=task.leg_hint,
                waypoint_budget=task.waypoint_budget,
            )
            scene = self.last_scene
            summary = "doorway step reached" if reached else "doorway step incomplete"
            return NavigatorResult(
                task_id=task.task_id,
                mode=task.mode,
                status="completed" if reached else "incomplete",
                summary=f"{summary} (round {task.attempt + 1})",
                scene=scene,
                reached=reached,
                payload={
                    "target": task.target,
                    "attempt": task.attempt,
                    "instruction": instruction,
                },
            )
        finally:
            self._clear_leg_goal_tracking()

    def navigate_reactive(self, target: str, *,
                          plan_context: str = "",
                          leg_hint: str = "",
                          waypoint_budget: int | None = None) -> bool:
        self.reset()
        if not self._goal_target_label:
            self._goal_target_label = target
            self._goal_cue_tokens = self._build_goal_cue_tokens({}, target)
        budget = waypoint_budget or self.config.navigation_waypoint_budget
        indecisive_streak = 0
        self.events.publish("nav", f"Reactive nav start: {target}")
        for waypoint in range(budget):
            if self.stop_event.is_set():
                self.events.publish("nav", "Reactive nav aborted")
                return False
            frame = self.camera.snap()
            depth_map = self.camera.get_depth_map()
            if frame is None or depth_map is None:
                time.sleep(0.2)
                continue
            depth_summary = self._analyze_depth_for_goal(depth_map)
            self._last_depth_summary = depth_summary
            response = self._assess_waypoint(
                target=target,
                frame=frame,
                depth_summary=depth_summary,
                plan_context=plan_context,
                leg_hint=leg_hint,
            )
            scene = str(response.get("scene", "")).strip()
            if scene:
                self._last_scene = scene
                self._remember(scene)
            action = str(response.get("action", "inspect")).strip().lower()
            self._update_goal_heading_bias(
                action=action,
                response=response,
                depth_summary=depth_summary,
                scene=scene,
            )
            action, response, override_note = self._stabilize_action(
                action=action,
                response=response,
                scene=scene,
                depth_summary=depth_summary,
                indecisive_streak=indecisive_streak,
                zone_repeat_count=self._recent_zone_repeat_count(scene),
            )
            if action in {"inspect", "turn"}:
                indecisive_streak += 1
            else:
                indecisive_streak = 0
            note = f" ({override_note})" if override_note else ""
            self.events.publish(
                "nav",
                f"Waypoint {waypoint + 1}/{budget}: {action}{note} | {scene or 'no scene'}",
            )
            if action == "arrived":
                self.rover.stop()
                self._clear_leg_goal_tracking()
                return True
            if action == "inspect":
                self._inspect()
                self._tick_goal_reacquire()
                continue
            if action == "turn":
                turn_deg = clamp(as_float(response.get("turn_degrees"), 30.0), -120.0, 120.0)
                if abs(turn_deg) < 8:
                    turn_deg = 20 if depth_summary.recommended_heading_deg >= 0 else -20
                self._turn(turn_deg)
                self._tick_goal_reacquire()
                continue
            if action == "reverse":
                self._reverse(clamp(as_float(response.get("drive_distance"), 0.16), 0.10, 0.24))
                self._tick_goal_reacquire()
                continue
            if action == "drive_forward":
                drive_angle = clamp(
                    as_float(response.get("drive_angle"), depth_summary.recommended_heading_deg),
                    -30.0, 30.0,
                )
                drive_distance = clamp(
                    as_float(response.get("drive_distance"), 0.4),
                    0.20, 1.20,
                )
                drive_result = self._drive_forward(drive_distance, drive_angle)
                if drive_result == "aborted":
                    return False
                if drive_result == "blocked":
                    indecisive_streak = 0
                    if not self._recover_from_close_obstacle(
                        drive_angle_deg=drive_angle,
                        depth_summary=depth_summary,
                        scene=scene,
                    ):
                        if self.stop_event.is_set():
                            return False
                        self._inspect()
                self._tick_goal_reacquire()
                continue
            self._inspect()
            self._tick_goal_reacquire()
        self.rover.stop()
        self.events.publish("nav", f"Reactive nav exhausted budget for {target}")
        self._clear_leg_goal_tracking()
        return False

    def navigate_leg(self, instruction: dict, room_check_fn=None):
        target_room = instruction.get("target_room", "")
        scene = ""
        room_guess = None
        per_attempt_budget = max(self.config.navigation_waypoint_budget * 2, 30)
        reached = False
        for attempt in range(LEG_SEARCH_ATTEMPTS):
            if self.stop_event.is_set():
                break
            result = self.run_doorway_task(
                instruction,
                attempt=attempt,
                total_attempts=LEG_SEARCH_ATTEMPTS,
                waypoint_budget=per_attempt_budget,
            )
            scene = result.scene or scene
            reached = result.reached
            if reached:
                break
            self.events.publish(
                "nav",
                f"Door search round {attempt + 1}/{LEG_SEARCH_ATTEMPTS} incomplete for {target_room or 'transition'}",
            )
        if reached and room_check_fn and scene:
            try:
                room_guess, confidence = room_check_fn(scene)
            except Exception:
                room_guess, confidence = None, 0.0
            if room_guess:
                self.events.publish(
                    "room",
                    f"Leg verification suggests {room_guess} ({confidence:.2f})",
                )
        return reached, room_guess, scene

    def describe_scene(self, frame: bytes, context: str = "") -> str:
        try:
            return self.llm.complete(
                prompt=scene_prompt(context),
                system="Describe the image plainly. Do not output JSON.",
                image_bytes=frame,
                max_tokens=220,
            ).strip()
        except Exception as exc:
            self.events.publish("error", f"Scene description failed: {exc}")
            return ""

    def _assess_waypoint(self, *, target: str, frame: bytes, depth_summary,
                         plan_context: str, leg_hint: str) -> dict:
        plan_context = self._merge_goal_context(plan_context)
        heuristic_context = self._zone_exit_heuristic_context(
            target=target,
            depth_summary=depth_summary,
        )
        prompt = navigation_prompt(
            target=target,
            plan_context=plan_context,
            leg_hint=leg_hint,
            depth_context=depth_summary.to_prompt_dict(),
            recent_observations=self._recent_observations,
            heuristic_context=heuristic_context,
        )
        try:
            raw = self.llm.complete(
                prompt=prompt,
                system="Reply with only the requested JSON object.",
                image_bytes=frame,
                max_tokens=260,
            )
            data = extract_json_dict(raw)
            if isinstance(data, dict):
                return data
            self.events.publish("error", f"Navigator JSON parse failed: {raw[:160]}")
        except Exception as exc:
            self.events.publish("error", f"Navigator LLM failed: {exc}")
        return {
            "action": "drive_forward",
            "scene": "",
            "reason": "fallback",
            "drive_angle": depth_summary.recommended_heading_deg,
            "drive_distance": min(0.5, max(0.25, depth_summary.center_distance())),
        }

    def _format_leg_hint(self, instruction: dict) -> str:
        parts = []
        if instruction.get("exit_hint"):
            parts.append(f"Exit hint: {instruction['exit_hint']}")
        if instruction.get("room_nav_hints"):
            parts.append(f"Room nav hints: {instruction['room_nav_hints']}")
        if instruction.get("visual_cues"):
            parts.append("Visual cues: " + ", ".join(instruction["visual_cues"][:4]))
        if instruction.get("doorway_landmarks"):
            parts.append("Doorway landmarks: " + ", ".join(instruction["doorway_landmarks"][:4]))
        if instruction.get("inside_features"):
            parts.append("Inside features: " + ", ".join(instruction["inside_features"][:4]))
        if instruction.get("expected_floor"):
            parts.append(f"Expected floor after crossing: {instruction['expected_floor']}")
        if instruction.get("expected_azimuth_deg") is not None:
            azimuth = as_float(instruction.get("expected_azimuth_deg"), 0.0)
            direction = "right" if azimuth > 0 else "left"
            parts.append(f"Expected azimuth: {azimuth:+.0f}° from room entry ({direction})")
        if instruction.get("doorway_width_m"):
            parts.append(f"Doorway width: {as_float(instruction['doorway_width_m'], 0.8):.2f}m")
        if instruction.get("relationship_hint"):
            parts.append(f"Relationship hint: {instruction['relationship_hint']}")
        if instruction.get("verify_features"):
            parts.append("Verify room with: " + ", ".join(instruction["verify_features"][:4]))
        return " | ".join(parts)

    def _leg_target_phrase(self, instruction: dict) -> str:
        target_room = instruction.get("target_room") or "next room"
        doorway_landmarks = list(instruction.get("doorway_landmarks") or [])
        visual_cues = list(instruction.get("visual_cues") or [])
        cue_parts = doorway_landmarks[:2] or visual_cues[:2]
        if cue_parts:
            return f"the doorway to {target_room} with " + ", ".join(cue_parts)
        exit_hint = str(instruction.get("exit_hint") or "").strip()
        if exit_hint:
            return exit_hint
        return f"the doorway to {target_room}"

    def _leg_plan_context(self, instruction: dict, attempt: int, total_attempts: int) -> str:
        target_room = instruction.get("target_room") or "next room"
        parts = [
            f"Immediate goal: find the doorway to {target_room} in this room right now.",
            f"Door search round {attempt + 1}/{total_attempts}.",
            "Do not reason about the whole trip. Only solve this doorway step.",
            "If the doorway is not visible yet, keep turning or shifting until it is.",
            "If furniture blocks the rover, back up a little, rotate, and reacquire the doorway.",
        ]
        if instruction.get("expected_floor"):
            parts.append(f"After crossing, expect {instruction['expected_floor']}.")
        return " ".join(parts)

    def _begin_leg_goal_tracking(self, instruction: dict, immediate_target: str):
        self._goal_target_label = immediate_target
        self._goal_cue_tokens = self._build_goal_cue_tokens(instruction, immediate_target)
        azimuth = instruction.get("expected_azimuth_deg")
        if azimuth is None:
            self._goal_heading_bias_deg = None
            self._goal_heading_confidence = 0.0
        else:
            self._goal_heading_bias_deg = clamp(as_float(azimuth, 0.0), -28.0, 28.0)
            self._goal_heading_confidence = 0.45
        self._goal_reacquire_steps = 0
        self._goal_recently_seen_steps = 0

    def _clear_leg_goal_tracking(self):
        self._goal_heading_bias_deg = None
        self._goal_heading_confidence = 0.0
        self._goal_reacquire_steps = 0
        self._goal_target_label = ""
        self._goal_recently_seen_steps = 0
        self._goal_cue_tokens = set()

    def _merge_goal_context(self, plan_context: str) -> str:
        extra = self._goal_reacquire_context()
        if not extra:
            return plan_context
        return f"{plan_context} {extra}".strip()

    def _goal_reacquire_context(self) -> str:
        if self._goal_reacquire_steps <= 0 or self._goal_heading_bias_deg is None:
            return ""
        if self._goal_heading_bias_deg >= 6.0:
            side = "right"
        elif self._goal_heading_bias_deg <= -6.0:
            side = "left"
        else:
            side = "front"
        return (
            f"After avoiding an obstacle, reacquire {self._goal_target_label or 'the doorway'} "
            f"on the {side}. Do not keep exploring away from it. "
            "Preserve the last good bearing until the goal is reacquired."
        )

    def _preferred_goal_heading(self) -> float | None:
        if self._goal_heading_bias_deg is None:
            return None
        if self._goal_recently_seen_steps > 0 or self._goal_reacquire_steps > 0 or self._goal_heading_confidence >= 0.55:
            return clamp(self._goal_heading_bias_deg, -28.0, 28.0)
        return None

    def _analyze_depth_for_goal(self, depth_map):
        preferred_heading = self._preferred_goal_heading()
        if preferred_heading is None:
            return self.depth_vectors.analyze(depth_map)
        return self.depth_vectors.analyze(
            depth_map,
            preferred_heading_deg=preferred_heading,
            search_half_window_deg=20.0,
        )

    def _update_goal_heading_bias(self, *, action: str, response: dict, depth_summary, scene: str = ""):
        if not self._goal_target_label:
            return
        candidate = None
        confidence = 0.0
        goal_cues_visible = self._scene_has_goal_cues(scene)
        if bool(response.get("target_visible")) or goal_cues_visible:
            if action == "drive_forward":
                candidate = as_float(response.get("drive_angle"), depth_summary.recommended_heading_deg)
            elif action == "turn":
                candidate = clamp(
                    as_float(response.get("turn_degrees"), depth_summary.recommended_heading_deg),
                    -35.0, 35.0,
                )
            else:
                candidate = depth_summary.recommended_heading_deg
            confidence = 0.84 if goal_cues_visible else 0.80
            self._goal_recently_seen_steps = max(self._goal_recently_seen_steps, 6)
            self._goal_reacquire_steps = max(self._goal_reacquire_steps, 4)
        elif self._goal_reacquire_steps > 0 and action == "drive_forward":
            # During reacquire after obstacle avoidance, hold the preserved
            # heading instead of adopting the LLM's drive angle which may
            # drift away from the original target.
            if self._goal_heading_bias_deg is not None:
                candidate = self._goal_heading_bias_deg
            else:
                candidate = as_float(response.get("drive_angle"), depth_summary.recommended_heading_deg)
            confidence = max(0.62, self._goal_heading_confidence)
        elif self._goal_recently_seen_steps > 0 and action == "drive_forward":
            candidate = as_float(response.get("drive_angle"), depth_summary.recommended_heading_deg)
            confidence = max(0.62, self._goal_heading_confidence)
        if candidate is None:
            return
        candidate = clamp(candidate, -28.0, 28.0)
        if self._goal_heading_bias_deg is None:
            updated = candidate
        else:
            updated = self._goal_heading_bias_deg * 0.55 + candidate * 0.45
        self._goal_heading_bias_deg = clamp(updated, -28.0, 28.0)
        self._goal_heading_confidence = max(self._goal_heading_confidence * 0.75, confidence)

    def _tick_goal_reacquire(self):
        if self._goal_reacquire_steps > 0:
            self._goal_reacquire_steps -= 1
        if self._goal_recently_seen_steps > 0:
            self._goal_recently_seen_steps -= 1

    def _remember(self, scene: str):
        entry = scene[:160]
        self._recent_observations.append(entry)
        if len(self._recent_observations) > 10:
            self._recent_observations = self._recent_observations[-10:]

    def _goal_tokens_from_text(self, text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) >= 3 and token not in {"the", "and", "with", "into", "from", "visible", "doorway"}
        }

    def _build_goal_cue_tokens(self, instruction: dict, immediate_target: str) -> set[str]:
        tokens = set(GOAL_CUE_GENERIC_TOKENS)
        for text in (
            immediate_target,
            instruction.get("target_room"),
            instruction.get("exit_hint"),
            instruction.get("room_nav_hints"),
        ):
            tokens.update(self._goal_tokens_from_text(str(text or "")))
        for key in ("doorway_landmarks", "visual_cues", "inside_features", "verify_features"):
            for item in instruction.get(key) or []:
                tokens.update(self._goal_tokens_from_text(str(item)))
        return tokens

    def _scene_goal_evidence(self, scene: str) -> float:
        if not scene or not self._goal_cue_tokens:
            return 0.0
        scene_tokens = self._goal_tokens_from_text(scene) | set(re.findall(r"[a-z0-9]+", scene.lower()))
        cue_hits = scene_tokens & self._goal_cue_tokens
        doorway_hits = scene_tokens & GOAL_CUE_GENERIC_TOKENS
        hit_score = min(len(cue_hits), 4) * 0.18
        doorway_bonus = 0.18 if doorway_hits else 0.0
        return min(1.0, hit_score + doorway_bonus)

    def _scene_has_goal_cues(self, scene: str) -> bool:
        return self._scene_goal_evidence(scene) >= 0.36

    def _scene_zone_tokens(self, scene: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", (scene or "").lower())
        return {
            word for word in words
            if len(word) >= 3 and word not in ZONE_SIGNATURE_STOPWORDS
        }

    def _scene_zone_similarity(self, left_scene: str, right_scene: str) -> float:
        left_tokens = self._scene_zone_tokens(left_scene)
        right_tokens = self._scene_zone_tokens(right_scene)
        if not left_tokens or not right_tokens:
            return 0.0
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)

    def _recent_zone_repeat_count(self, current_scene: str = "") -> int:
        scenes = [scene for scene in self._recent_observations[-5:] if scene]
        current_scene = (current_scene or "").strip()[:160]
        if current_scene and (not scenes or scenes[-1] != current_scene):
            scenes.append(current_scene)
        if not scenes:
            return 0
        anchor = scenes[-1]
        repeats = 1
        for other in reversed(scenes[:-1]):
            if self._scene_zone_similarity(anchor, other) >= ZONE_REPEAT_SIMILARITY:
                repeats += 1
                continue
            break
        return repeats

    def _zone_exit_heading(self, depth_summary) -> float:
        preferred = self._preferred_goal_heading()
        if preferred is not None:
            return clamp(preferred, -30.0, 30.0)
        return clamp(depth_summary.recommended_heading_deg, -30.0, 30.0)

    def _zone_exit_heuristic_context(self, *, target: str, depth_summary) -> str:
        repeat_threshold = max(2, int(getattr(self.config, "navigation_zone_repeat_threshold", 3)))
        repeat_count = self._recent_zone_repeat_count()
        if repeat_count < repeat_threshold:
            return ""
        if self._goal_recently_seen_steps > 0:
            return (
                f"Doorway cues for {self._goal_target_label or target} were seen recently. "
                "Do not drift into unrelated local exploration; reacquire the goal on the preserved bearing instead."
            )
        corridor_clear = depth_summary.closest_corridor_distance() or depth_summary.center_distance()
        exit_heading = self._zone_exit_heading(depth_summary)
        target_label = self._goal_target_label or target
        if corridor_clear >= getattr(self.config, "navigation_zone_exit_min_clearance_m", 0.50):
            return (
                f"The last {repeat_count} observations look like the same local zone. "
                f"If {target_label} is not visible right now, leave this zone immediately by driving "
                f"through the safest open corridor near {exit_heading:+.0f} degrees."
            )
        return (
            f"The last {repeat_count} observations look like the same local zone. "
            f"If {target_label} is not visible right now, stop inspecting this area and turn decisively "
            f"toward the best exit near {exit_heading:+.0f} degrees."
        )

    def _door_search_reposition(self, instruction: dict, attempt: int):
        if self.stop_event.is_set():
            return
        azimuth = instruction.get("expected_azimuth_deg")
        if attempt == 1 and azimuth is not None:
            turn_deg = clamp(as_float(azimuth, 0.0), -75.0, 75.0)
            if abs(turn_deg) >= 12.0:
                self.events.publish("nav", f"Door search turning toward expected azimuth {turn_deg:+.0f}°")
                self._turn(turn_deg)
                return
        depth_map = self.camera.get_depth_map()
        if depth_map is not None:
            try:
                depth_summary = self._analyze_depth_for_goal(depth_map)
            except Exception:
                depth_summary = None
            if depth_summary is not None:
                center_clear = depth_summary.center_distance()
                corridor_clear = depth_summary.closest_corridor_distance() or center_clear
                if corridor_clear >= 0.55:
                    drive_angle = clamp(depth_summary.recommended_heading_deg, -25.0, 25.0)
                    drive_distance = min(0.55, self._fluid_drive_distance(center_clear, corridor_clear))
                    self.events.publish(
                        "nav",
                        f"Door search reposition {drive_distance:.2f}m at {drive_angle:+.0f}°",
                    )
                    drive_result = self._drive_forward(drive_distance, drive_angle)
                    if drive_result == "ok":
                        return
                    if drive_result == "aborted":
                        return
                    if drive_result == "blocked":
                        self._recover_from_close_obstacle(
                            drive_angle_deg=drive_angle,
                            depth_summary=depth_summary,
                            scene="",
                        )
                        return
        sweep_deg = 55 if attempt % 2 else -55
        if azimuth is not None and attempt <= 2:
            sweep_deg = 60 if as_float(azimuth, 0.0) > 0 else -60
        self.events.publish("nav", f"Door search sweep {sweep_deg:+.0f}°")
        self._turn(sweep_deg)

    def _stabilize_action(self, *, action: str, response: dict, scene: str, depth_summary,
                          indecisive_streak: int, zone_repeat_count: int = 0):
        center_clear = depth_summary.center_distance()
        corridor_clear = depth_summary.closest_corridor_distance() or center_clear
        drive_angle = clamp(
            as_float(response.get("drive_angle"), depth_summary.recommended_heading_deg),
            -30.0, 30.0,
        )
        fluid_distance = self._fluid_drive_distance(center_clear, corridor_clear)
        override_note = ""
        target_visible = bool(response.get("target_visible"))
        repeat_threshold = max(2, int(getattr(self.config, "navigation_zone_repeat_threshold", 3)))
        exit_clearance = float(getattr(self.config, "navigation_zone_exit_min_clearance_m", 0.50))
        exit_heading = self._zone_exit_heading(depth_summary)
        goal_evidence = self._scene_goal_evidence(scene)
        low_goal_evidence = goal_evidence < 0.22
        has_preserved_bearing = self._goal_recently_seen_steps > 0 and self._preferred_goal_heading() is not None

        if not target_visible and has_preserved_bearing and low_goal_evidence:
            if action == "drive_forward" and abs(drive_angle - exit_heading) <= 10.0:
                response["drive_angle"] = exit_heading
                response["drive_distance"] = clamp(max(as_float(response.get("drive_distance"), 0.35), fluid_distance), 0.30, 0.85)
                override_note = "lost goal evidence -> hold bearing"
                return action, response, override_note
            forced_turn = exit_heading
            if abs(forced_turn) < 18.0:
                forced_turn = 28.0 if forced_turn >= 0 else -28.0
            action = "turn"
            response["turn_degrees"] = clamp(forced_turn, -75.0, 75.0)
            override_note = "lost goal evidence -> reacquire bearing"
            return action, response, override_note

        if not target_visible and self._goal_recently_seen_steps <= 0 and zone_repeat_count >= repeat_threshold:
            if corridor_clear >= exit_clearance:
                action = "drive_forward"
                response["drive_angle"] = exit_heading
                response["drive_distance"] = clamp(max(0.45, fluid_distance), 0.45, 0.95)
                override_note = f"zone loop x{zone_repeat_count} -> leave zone"
                return action, response, override_note
            if action == "inspect":
                forced_turn = exit_heading
                if abs(forced_turn) < 18.0:
                    forced_turn = 30.0 if forced_turn >= 0 else -30.0
                action = "turn"
                response["turn_degrees"] = clamp(forced_turn, -75.0, 75.0)
                override_note = f"zone loop x{zone_repeat_count} -> force exit turn"
                return action, response, override_note

        if action == "drive_forward":
            response["drive_distance"] = clamp(
                max(as_float(response.get("drive_distance"), 0.4), fluid_distance),
                0.25, 1.0,
            )
            if abs(drive_angle) < 4.0 and abs(depth_summary.recommended_heading_deg) >= 8.0:
                drive_angle = depth_summary.recommended_heading_deg
            if not target_visible and self._goal_recently_seen_steps > 0 and abs(exit_heading - drive_angle) >= 12.0:
                drive_angle = exit_heading
                override_note = "bias back to doorway cue"
            response["drive_angle"] = clamp(drive_angle, -30.0, 30.0)
            return action, response, override_note

        if action == "turn":
            turn_deg = clamp(as_float(response.get("turn_degrees"), 30.0), -120.0, 120.0)
            if abs(turn_deg) <= 18.0 and corridor_clear >= 0.55:
                action = "drive_forward"
                response["drive_angle"] = depth_summary.recommended_heading_deg
                response["drive_distance"] = fluid_distance
                override_note = "small turn -> fluid drive"
                return action, response, override_note

        if action == "reverse":
            response["drive_distance"] = clamp(
                as_float(response.get("drive_distance"), 0.16),
                0.10, 0.24,
            )
            if corridor_clear >= 0.50 and center_clear >= max(self.config.depth_guard_stop_m + 0.07, 0.42):
                action = "drive_forward"
                response["drive_angle"] = depth_summary.recommended_heading_deg
                response["drive_distance"] = clamp(max(0.35, fluid_distance), 0.35, 0.65)
                override_note = "reverse -> doorway search"
            return action, response, override_note

        if action == "inspect" and corridor_clear >= 0.75 and indecisive_streak >= 1:
            action = "drive_forward"
            response["drive_angle"] = depth_summary.recommended_heading_deg
            response["drive_distance"] = fluid_distance
            override_note = "repeat inspect -> commit drive"
            return action, response, override_note

        if action in {"inspect", "turn"} and indecisive_streak >= 3 and corridor_clear >= 0.50:
            action = "drive_forward"
            response["drive_angle"] = depth_summary.recommended_heading_deg
            response["drive_distance"] = fluid_distance
            override_note = "indecisive loop -> commit drive"
            return action, response, override_note

        return action, response, override_note

    def _fluid_drive_distance(self, center_clear: float, corridor_clear: float) -> float:
        usable_clear = max(0.0, min(center_clear, corridor_clear))
        if usable_clear >= 1.30:
            return 0.90
        if usable_clear >= 1.00:
            return 0.75
        if usable_clear >= 0.75:
            return 0.60
        if usable_clear >= 0.55:
            return 0.45
        return 0.30

    def _turn(self, degrees: float):
        sign = 1 if degrees > 0 else -1
        peak_speed = clamp(
            as_float(getattr(self.config, "navigation_turn_speed", 0.24), 0.24),
            0.16, TURN_SPEED,
        )
        turn_rate_dps = max(60.0, TURN_RATE_DPS * (peak_speed / TURN_SPEED))
        duration = abs(degrees) / turn_rate_dps
        self.events.publish("nav", f"Turning {degrees:+.0f}°")
        for segment_speed, segment_duration in self._turn_speed_profile(duration, peak_speed):
            if self.stop_event.is_set():
                break
            self.rover.send({
                "T": 1,
                "L": round(segment_speed * sign, 3),
                "R": round(-segment_speed * sign, 3),
            })
            self._sleep(segment_duration)
        self.rover.stop()

    def _turn_speed_profile(self, duration_s: float, peak_speed: float):
        duration_s = max(0.06, float(duration_s))
        step_s = clamp(
            as_float(getattr(self.config, "navigation_turn_step_s", 0.04), 0.04),
            0.03, 0.10,
        )
        segment_count = max(3, int(round(duration_s / step_s)))
        if segment_count <= 3:
            fractions = [0.72, 1.0, 0.72]
        elif segment_count == 4:
            fractions = [0.68, 0.9, 0.9, 0.68]
        else:
            plateau_count = max(1, segment_count - 4)
            fractions = [0.62, 0.82] + ([1.0] * plateau_count) + [0.82, 0.62]
        segment_duration = duration_s / len(fractions)
        min_speed = max(0.14, peak_speed * 0.62)
        return [
            (max(min_speed, peak_speed * fraction), segment_duration)
            for fraction in fractions
        ]

    def _reverse(self, distance_m: float):
        duration = max(0.3, abs(distance_m) / max(self.config.navigation_drive_speed, 0.05))
        self.events.publish("nav", f"Reversing {distance_m:.2f}m")
        self.rover.send({"T": 1, "L": -0.12, "R": -0.12})
        self._sleep(duration)
        self.rover.stop()

    def _inspect(self):
        if self.flags.gimbal_pan_enabled:
            sweep = [0, -55, 55, 0]
            pan = sweep[self._scan_index % len(sweep)]
            self._scan_index += 1
            self.events.publish("nav", f"Inspect sweep pan={pan:+d}°")
            self.rover.send({"T": 133, "X": pan, "Y": 0, "SPD": 260, "ACC": 20})
            self._sleep(0.6)
            self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 260, "ACC": 20})
        else:
            self._turn(25 if (self._scan_index % 2 == 0) else -25)
            self._scan_index += 1

    def _drive_forward(self, distance_m: float, drive_angle_deg: float) -> str:
        duration = distance_m / max(self.config.navigation_drive_speed, 0.05)
        steer = clamp(drive_angle_deg / 90.0 * 0.10, -0.10, 0.10)
        left = self.config.navigation_drive_speed + steer
        right = self.config.navigation_drive_speed - steer
        command = {"T": 1, "L": round(left, 3), "R": round(right, 3)}
        self.events.publish(
            "nav",
            f"Driving {distance_m:.2f}m at {drive_angle_deg:+.0f}° "
            f"[L={left:.2f}, R={right:.2f}]",
        )
        start = time.time()
        last_send = 0.0
        while time.time() - start < duration:
            if self.stop_event.is_set():
                self.rover.stop()
                return "aborted"
            if not self._depth_clear_for_motion(turning=False):
                self.events.publish("nav", "Depth guard stopped forward motion")
                self.rover.stop()
                return "blocked"
            now = time.time()
            if now - last_send >= DRIVE_COMMAND_REFRESH_S:
                self.rover.send(command)
                last_send = now
            time.sleep(0.05)
        self.rover.stop()
        return "ok"

    def _recover_from_close_obstacle(self, *, drive_angle_deg: float, depth_summary, scene: str) -> bool:
        if self.stop_event.is_set():
            return False
        plan = self._blocked_escape_plan(
            drive_angle_deg=drive_angle_deg,
            depth_summary=depth_summary,
            scene=scene,
        )
        self.events.publish(
            "nav",
            f"Blocked recovery: reverse {plan['reverse_distance_m']:.2f}m, "
            f"turn {plan['turn_degrees']:+.0f}° ({plan['reason']})",
        )
        self._reverse(plan["reverse_distance_m"])
        if self.stop_event.is_set():
            return False
        self._turn(plan["turn_degrees"])
        if self._goal_target_label:
            self._goal_reacquire_steps = max(self._goal_reacquire_steps, 8)
            self._goal_recently_seen_steps = max(self._goal_recently_seen_steps, 6)
            if self._goal_heading_bias_deg is not None:
                # Preserve the original goal heading — do NOT blend with escape
                # turn direction. Flip the sign so reacquire steers back toward
                # the goal (opposite of the escape turn).
                escape_sign = 1 if plan["turn_degrees"] >= 0 else -1
                goal_sign = 1 if self._goal_heading_bias_deg >= 0 else -1
                if escape_sign == goal_sign:
                    # Escaped in the same direction as the goal — keep heading
                    pass
                else:
                    # Escaped away from goal — boost heading confidence to
                    # ensure reacquire logic kicks in strongly
                    self._goal_heading_confidence = max(self._goal_heading_confidence, 0.82)
            else:
                # No prior heading — use opposite of escape turn as best guess
                self._goal_heading_bias_deg = clamp(-plan["turn_degrees"] * 0.6, -28.0, 28.0)
            self._goal_heading_confidence = max(self._goal_heading_confidence, 0.70)
        return not self.stop_event.is_set()

    def _blocked_escape_plan(self, *, drive_angle_deg: float, depth_summary, scene: str) -> dict:
        stop_threshold = max(0.05, float(self.config.depth_guard_stop_m))
        center_clear = depth_summary.center_distance() if depth_summary is not None else stop_threshold
        corridor_clear = (
            depth_summary.closest_corridor_distance() or center_clear
            if depth_summary is not None else center_clear
        )
        furniture_trap, furniture_reason = self._scene_furniture_trap(scene)
        reverse_distance = BLOCKED_ESCAPE_REVERSE_M
        if furniture_trap or center_clear <= stop_threshold * 0.75:
            reverse_distance = BLOCKED_ESCAPE_FURNITURE_REVERSE_M
        elif center_clear <= stop_threshold + 0.05:
            reverse_distance = max(reverse_distance, 0.22)

        goal_heading = self._preferred_goal_heading()
        heading = goal_heading if goal_heading is not None else (
            depth_summary.recommended_heading_deg if depth_summary is not None else 0.0
        )
        if abs(heading) < 10.0:
            if abs(drive_angle_deg) >= 8.0:
                heading = drive_angle_deg
            else:
                heading = float(self._blocked_escape_dir * BLOCKED_ESCAPE_MIN_TURN_DEG)
                self._blocked_escape_dir *= -1
        turn_magnitude = max(BLOCKED_ESCAPE_MIN_TURN_DEG, min(abs(heading), BLOCKED_ESCAPE_MAX_TURN_DEG))
        if corridor_clear <= stop_threshold + 0.10:
            turn_magnitude = max(turn_magnitude, 45.0)
        turn_degrees = turn_magnitude if heading >= 0 else -turn_magnitude

        reasons = []
        if furniture_trap:
            reasons.append(furniture_reason)
        if center_clear <= stop_threshold:
            reasons.append(f"center clearance {center_clear:.2f}m")
        elif corridor_clear <= stop_threshold:
            reasons.append(f"corridor clearance {corridor_clear:.2f}m")
        elif corridor_clear <= stop_threshold + 0.10:
            reasons.append(f"tight corridor {corridor_clear:.2f}m")

        return {
            "reverse_distance_m": reverse_distance,
            "turn_degrees": turn_degrees,
            "reason": ", ".join(reasons) or "close obstacle ahead",
        }

    def _scene_furniture_trap(self, scene: str):
        text = (scene or "").lower()
        hits = [hint for hint in FURNITURE_TRAP_HINTS if hint in text]
        if not hits and "chair" in text and ("leg" in text or "wheel" in text or "base" in text):
            hits = ["chair clutter"]
        if not hits and "desk" in text and ("leg" in text or "edge" in text):
            hits = ["desk clutter"]
        if not hits:
            return False, ""
        return True, hits[0]

    def _depth_clear_for_motion(self, *, turning: bool) -> bool:
        depth_map = self.camera.get_depth_map()
        if depth_map is None:
            return True
        try:
            summary = self.depth_vectors.analyze(depth_map)
        except Exception:
            return True
        stop_threshold = (
            self.config.depth_guard_turn_stop_m if turning
            else self.config.depth_guard_stop_m
        )
        center_distance = summary.center_distance()
        corridor_distance = summary.closest_corridor_distance() or center_distance
        return min(center_distance, corridor_distance) >= stop_threshold

    def _sleep(self, seconds: float):
        end = time.time() + max(seconds, 0.0)
        while time.time() < end:
            if self.stop_event.is_set():
                break
            time.sleep(0.05)
