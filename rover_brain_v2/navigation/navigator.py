"""Depth-vector local navigation driven by a navigator LLM."""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pathlib import Path

from rover_brain_v2.json_utils import as_float, extract_json_dict, clamp
from rover_brain_v2.models import NavigatorResult, NavigatorTask
from rover_brain_v2.navigation.depth_vectors import DepthVectorMap
from rover_brain_v2.navigation.spatial_map import LocalSpatialMap
from rover_brain_v2.navigation.waypoint_follower import ContinuousNavigator
from rover_brain_v2.prompts import load_prompt, navigation_prompt, scene_prompt


TURN_SPEED = 0.35
TURN_RATE_DPS = 200.0
DRIVE_COMMAND_REFRESH_S = 0.35
LEG_SEARCH_ATTEMPTS = 4
BLOCKED_ESCAPE_REVERSE_M = 0.18
BLOCKED_ESCAPE_FURNITURE_REVERSE_M = 0.22
BLOCKED_ESCAPE_MIN_TURN_DEG = 55.0
BLOCKED_ESCAPE_MAX_TURN_DEG = 100.0
AVOID_STEER = 0.13
DRIVE_CHECK_INTERVAL_S = 0.20
WALL_CLOSE_FRACTION = 0.60
WALL_DEPTH_RANGE_MIN = 0.02
WALL_IMAGE_STD_MAX = 35.0
WALL_IMAGE_DOMINANT_MIN = 0.30
AREA_ESCAPE_CHECKPOINT_STEPS = 8
AREA_ESCAPE_MAX = 3
FRAME_STUCK_THRESHOLD = 4

# YOLO visual servo navigation
YOLO_GIMBAL_GAIN = 40.0
YOLO_GIMBAL_DEADZONE = 0.05
YOLO_CENTER_TOL = 0.12
YOLO_ARRIVE_BW = 0.40
YOLO_STEER_GAIN = 0.005
YOLO_MAX_STEER = 0.12
YOLO_LOST_TIMEOUT = 4.0
YOLO_DRIVE_SPEED = 0.14
YOLO_MIN_BW = 0.03
YOLO_LOOP_HZ = 10.0
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
    def __init__(self, *, rover, camera, llm_client, event_bus, flags, config,
                 speak_fn=None, listen_fn=None, place_db=None):
        self.rover = rover
        self.camera = camera
        self.llm = llm_client
        self.events = event_bus
        self.flags = flags
        self.config = config
        self.speak_fn = speak_fn
        self.listen_fn = listen_fn
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
        self.spatial_map = LocalSpatialMap()
        self._blocked_escape_dir = 1
        self._goal_heading_bias_deg: float | None = None
        self._goal_heading_confidence = 0.0
        self._goal_reacquire_steps = 0
        self._goal_target_label = ""
        self._goal_recently_seen_steps = 0
        self._goal_cue_tokens: set[str] = set()
        self._last_drive_angle: float = 0.0
        self._last_gimbal_pan: float = 0.0
        self._last_gimbal_tilt: float = 0.0
        self._area_escape_dir = 1
        self._area_escape_count = 0
        self._frame_unchanged_count = 0
        self._scene_repeat_count = 0
        self._prev_snap: bytes | None = None
        self._steps_in_current_room = 0
        self._current_room_id = ""
        self.place_db = place_db
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future = None
        self._prefetch_context: dict | None = None
        self._continuous_nav = ContinuousNavigator(
            rover=rover,
            camera=camera,
            depth_vectors=self.depth_vectors,
            event_bus=event_bus,
            flags=flags,
            config=config,
        )
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
        self._map_poll_step = 0
        self._map_poll_active = False  # True when navigate_reactive is running
        self._map_poll_thread = threading.Thread(
            target=self._map_poll_loop,
            daemon=True,
            name="rover-v2-map-poll",
        )
        self._map_poll_thread.start()

    def _gimbal_send(self, pan: float, tilt: float, *, spd: int = 600, acc: int = 80):
        """Send gimbal command with calibration offset applied."""
        pc = self.config.gimbal_pan_center
        tc = self.config.gimbal_tilt_center
        self.rover.send({"T": 133, "X": round(pan + pc, 1), "Y": round(tilt + tc, 1), "SPD": spd, "ACC": acc})
        self._last_gimbal_pan = pan
        self._last_gimbal_tilt = tilt

    def _gimbal_center(self, *, spd: int = 600, acc: int = 80):
        """Send gimbal to calibrated center and reset tracking."""
        self._gimbal_send(0, 0, spd=spd, acc=acc)

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
        self._scene_repeat_count = 0
        self._steps_in_current_room = 0
        self._current_room_id = ""
        self._nav_target_room = None
        self._room_just_changed = False
        self.spatial_map.reset()

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

    def run_vlm_guided_task(self, target: str, *,
                            plan_context: str = "",
                            max_vlm_cycles: int = 30) -> NavigatorResult:
        """Continuous VLM-guided navigation using pixel waypoints and pure pursuit.

        Instead of stop-go, the VLM marks pixel waypoints on keyframes and the
        follower smoothly drives toward them at 10Hz.
        """
        self.stop_event.clear()
        task = self._new_task(
            mode="vlm_guided",
            target=target,
            plan_context=plan_context,
        )
        task.instruction = {"max_vlm_cycles": max_vlm_cycles}
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
            if task.mode == "vlm_guided":
                return self._execute_vlm_guided_task(task)
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

    def _execute_vlm_guided_task(self, task: NavigatorTask) -> NavigatorResult:
        """Continuous depth-reactive navigation with async LLM steering.

        The rover drives immediately using depth and never stops to think.
        The LLM fires in the background and adjusts steering bias.
        """
        self.reset()
        target = task.target
        max_cycles = task.instruction.get("max_vlm_cycles", 30)
        nav = self._continuous_nav
        stop = self.stop_event

        self.events.publish("nav", f"Continuous nav start: {target}")
        self._gimbal_center()
        self._sleep(0.3)

        # Start LLM advisor in background — it sets nav.set_bias() while
        # the driver loop runs in the foreground.
        llm_thread = threading.Thread(
            target=self._llm_advisor_loop,
            args=(target, stop, nav, max_cycles),
            daemon=True,
            name="vlm-advisor",
        )
        llm_thread.start()

        # Run 10Hz driver — this blocks until stop or arrived
        try:
            arrived = nav.run(stop)
        except Exception as exc:
            self.events.publish("error", f"Continuous nav crashed: {exc}")
            arrived = False
        finally:
            stop.set()
            self.rover.stop()
            llm_thread.join(timeout=3.0)
            self._last_scene = nav.scene

        self.events.publish("nav", f"Continuous nav done: arrived={arrived}")
        return NavigatorResult(
            task_id=task.task_id,
            mode="vlm_guided",
            status="completed" if arrived else "incomplete",
            summary=f"Continuous: {'arrived' if arrived else 'did not arrive'} at {target}",
            scene=nav.scene,
            reached=arrived,
            payload={"target": target},
        )

    def _llm_advisor_loop(self, target: str, stop: threading.Event,
                          nav: ContinuousNavigator, max_cycles: int):
        """Background LLM thread: periodically captures frame, calls VLM,
        and sets steering bias.  The driver keeps moving the whole time."""
        for cycle in range(max_cycles):
            if stop.is_set():
                break

            frame = self.camera.snap_with_yolo(max_dim=512, quality=60)
            if frame is None:
                time.sleep(1.0)
                continue

            # Get detection summary without holding the lock long
            try:
                with self.camera._lock:
                    det_summary = self.camera._detection_summary or "none"
            except Exception:
                det_summary = "none"

            # Call LLM — this is the slow part (~3-60s depending on provider)
            # The driver keeps moving the whole time.
            try:
                raw = self.llm.complete(
                    prompt=load_prompt("vlm_waypoints.md",
                                       target=target,
                                       detections=det_summary),
                    system="You are a navigation assistant for a ground rover. Respond with JSON only.",
                    image_bytes=frame,
                    max_tokens=200,
                    temperature=0.2,
                )
                data = extract_json_dict(raw)
            except Exception as exc:
                self.events.publish("error", f"LLM advisor failed: {exc}")
                continue

            if not isinstance(data, dict):
                continue

            arrived = bool(data.get("arrived", False))
            scene = str(data.get("scene", ""))
            confidence = float(data.get("confidence", 0.5))

            # Extract steering target — use first waypoint's x-coordinate
            pixel_x = None
            waypoints = data.get("waypoints", [])
            if waypoints and isinstance(waypoints[0], (list, tuple)) and len(waypoints[0]) >= 2:
                pixel_x = int(waypoints[0][0])

            nav.set_bias(
                pixel_x=pixel_x,
                arrived=arrived,
                scene=scene,
                confidence=confidence,
            )

            if arrived:
                self.events.publish("nav", f"LLM says arrived at {target}")
                break

    def _execute_doorway_task(self, task: NavigatorTask) -> NavigatorResult:
        instruction = dict(task.instruction or {})
        self._nav_target_room = instruction.get("target_room") or None
        try:
            plan_ctx = task.plan_context
            if task.attempt > 0:
                plan_ctx = (
                    f"{plan_ctx} This is retry {task.attempt + 1}. "
                    f"Previous attempts failed. Try a COMPLETELY different approach: "
                    f"different direction, different path through the room."
                )
            leg_hint = task.leg_hint or self._format_leg_hint(instruction)
            reached = self.navigate_reactive(
                task.target,
                plan_context=plan_ctx,
                leg_hint=leg_hint,
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
        except Exception as exc:
            return NavigatorResult(
                task_id=task.task_id,
                mode=task.mode,
                status="error",
                summary=f"doorway nav crashed: {exc}",
                scene=self.last_scene,
                payload={"target": task.target, "attempt": task.attempt},
            )

    # ===== YOLO Visual Servo Navigation (unused, kept as fallback) =====

    def navigate_yolo(self, target: str, yolo_labels: tuple[str, ...],
                      timeout_s: float = 90.0) -> bool:
        """Navigate toward a target using YOLO detection + proportional control.
        Gimbal tracks the target, body aligns, wheels drive forward.
        Returns True if arrived (target fills enough of the frame)."""
        self.reset()
        self.events.publish("nav", f"YOLO nav start: {target} (labels={yolo_labels})")
        self.camera.set_detector_enabled(True)
        pan = 0.0
        tilt = 0.0
        last_cx = 0.5
        last_cy = 0.5
        found_ever = False
        last_seen = 0.0
        last_llm_check = 0.0
        llm_override_cx: float | None = None
        llm_override_cy: float | None = None
        start = time.time()
        self._gimbal_center(spd=600, acc=80)
        try:
            while time.time() - start < timeout_s:
                if self.stop_event.is_set():
                    self.events.publish("nav", "YOLO nav aborted")
                    return False

                dets, _summary, _age = self.camera.get_detections()
                match = self._yolo_select(dets, yolo_labels, last_cx, last_cy)

                if match is None or match.get("bw", 0.0) < YOLO_MIN_BW:
                    # YOLO can't see target — ask LLM to mark it
                    if llm_override_cx is None and time.time() - last_llm_check >= 3.0:
                        last_llm_check = time.time()
                        frame = self.camera.snap()
                        if frame is not None:
                            marker = self._llm_find_target_in_frame(target, frame)
                            if marker is not None:
                                llm_override_cx = marker[0]
                                llm_override_cy = marker[1]
                                self.events.publish("nav", f"LLM marker at ({llm_override_cx:.2f}, {llm_override_cy:.2f})")

                    # If LLM gave us a position, navigate toward it
                    if llm_override_cx is not None:
                        err_x = llm_override_cx - 0.5
                        if abs(err_x) > YOLO_GIMBAL_DEADZONE:
                            pan += err_x * YOLO_GIMBAL_GAIN
                            pan = max(-180, min(180, pan))
                            self._gimbal_send(pan, tilt, spd=600, acc=100)
                        if abs(pan) > 20:
                            self._turn(pan * 0.6)
                            pan *= 0.4
                        # Drive toward LLM marker
                        self.rover.send({"T": 1, "L": round(YOLO_DRIVE_SPEED, 3),
                                         "R": round(YOLO_DRIVE_SPEED, 3)})
                        self._sleep(0.5)
                        self.rover.stop()
                        llm_override_cx = None  # re-check next iteration
                        last_seen = time.time()
                        continue

                    # No YOLO, no LLM marker — stop and search
                    self.rover.stop()
                    lost_for = (time.time() - last_seen) if found_ever else (time.time() - start)
                    if lost_for >= YOLO_LOST_TIMEOUT:
                        # Spin search for target
                        found = self._yolo_spin_search(yolo_labels)
                        if found is None:
                            # Gimbal wide scan
                            scan_pan = self._yolo_gimbal_search(yolo_labels)
                            if scan_pan is not None:
                                self.events.publish("nav", f"YOLO gimbal found target at {scan_pan:+.0f}°")
                                self._turn(scan_pan)
                                pan = 0.0
                                last_seen = time.time()
                                continue
                            self.events.publish("nav", "YOLO nav: target lost after search")
                            return False
                        last_seen = time.time()
                        continue
                    time.sleep(1.0 / YOLO_LOOP_HZ)
                    continue

                # Target found
                found_ever = True
                last_seen = time.time()
                llm_override_cx = None
                llm_override_cy = None
                cx = match["cx"]
                cy = match["cy"]
                bw = match["bw"]
                last_cx = cx
                last_cy = cy

                # Periodic LLM verification (~every 6 seconds)
                now = time.time()
                if now - last_llm_check >= 6.0:
                    last_llm_check = now
                    correction = self._llm_verify_yolo_target(target, match)
                    if correction:
                        if correction.get("wrong_target"):
                            self.events.publish("nav", f"LLM says wrong target: {correction.get('reason', '')}")
                            # Use LLM marker position instead
                            if correction.get("target_cx") is not None:
                                llm_override_cx = float(correction["target_cx"])
                                llm_override_cy = float(correction.get("target_cy", 0.5))
                                cx = llm_override_cx
                                cy = llm_override_cy
                        elif correction.get("arrived"):
                            self.rover.stop()
                            self.events.publish("nav", "LLM confirms arrival")
                            self._last_scene = f"LLM confirmed arrival at {target}"
                            return True

                # Check arrival
                if bw >= YOLO_ARRIVE_BW:
                    self.rover.stop()
                    self.events.publish("nav", f"YOLO nav arrived! bw={bw:.2f} >= {YOLO_ARRIVE_BW}")
                    self._last_scene = f"Arrived at {target} (YOLO bw={bw:.2f})"
                    return True

                # Gimbal: center target in frame
                err_x = cx - 0.5
                err_y = cy - 0.5
                if abs(err_x) > YOLO_GIMBAL_DEADZONE:
                    pan += err_x * YOLO_GIMBAL_GAIN
                if abs(err_y) > YOLO_GIMBAL_DEADZONE:
                    tilt -= err_y * YOLO_GIMBAL_GAIN
                pan = max(-180, min(180, pan))
                tilt = max(-30, min(45, tilt))
                self._gimbal_send(pan, tilt, spd=600, acc=100)

                # If target not centered yet, just track with gimbal
                if abs(err_x) > YOLO_CENTER_TOL:
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    time.sleep(1.0 / YOLO_LOOP_HZ)
                    continue

                # Body align: if gimbal pan is large, turn body to reduce it
                if abs(pan) > 20:
                    body_turn = pan * 0.6
                    self.events.publish("nav", f"YOLO body align: pan={pan:+.0f}° → turn {body_turn:+.0f}°")
                    self._turn(body_turn)
                    pan -= body_turn
                    pan = max(-180, min(180, pan))
                    self._gimbal_send(pan, tilt, spd=600, acc=100)
                    continue

                # Drive forward with steering based on remaining pan offset
                steer = max(-YOLO_MAX_STEER, min(YOLO_MAX_STEER, pan * YOLO_STEER_GAIN))
                left = YOLO_DRIVE_SPEED + steer
                right = YOLO_DRIVE_SPEED - steer

                # Depth guard
                depth_map = self.camera.get_depth_map()
                if depth_map is not None:
                    steer_dir = self._depth_steer_check(pan)
                    if steer_dir == "blocked":
                        self.events.publish("nav", "YOLO depth blocked — recovering")
                        self.rover.stop()
                        self._reverse(0.15)
                        # Scan for best opening after reversing
                        escape = self._find_best_escape_turn(pan)
                        self._turn(escape)
                        self._gimbal_center(spd=600, acc=80)
                        self._sleep(0.3)
                        pan -= escape * 0.5
                        continue
                    if steer_dir == "left":
                        left -= AVOID_STEER
                        right += AVOID_STEER
                    elif steer_dir == "right":
                        left += AVOID_STEER
                        right -= AVOID_STEER

                self.rover.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})
                time.sleep(1.0 / YOLO_LOOP_HZ)

        finally:
            self.rover.stop()
            self._gimbal_center(spd=600, acc=80)

        self.events.publish("nav", "YOLO nav timeout")
        return False

    def _yolo_select(self, detections, labels: tuple[str, ...], last_cx: float, last_cy: float):
        """Select best matching detection from YOLO results."""
        candidates = [
            det for det in detections
            if det.get("name", "").lower() in labels and det.get("bw", 0.0) >= YOLO_MIN_BW
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda d: ((d["cx"] - last_cx) ** 2 + (d["cy"] - last_cy) ** 2, -d.get("conf", 0.0)),
        )

    def _yolo_spin_search(self, labels: tuple[str, ...]):
        """Spin body slowly looking for target with YOLO."""
        self.events.publish("nav", "YOLO spin search")
        for i in range(8):
            if self.stop_event.is_set():
                return None
            self.rover.send({"T": 1, "L": 0.20, "R": -0.20})
            self._sleep(0.4)
            self.rover.stop()
            self._sleep(0.2)
            dets, _, _ = self.camera.get_detections()
            match = self._yolo_select(dets, labels, 0.5, 0.5)
            if match and match.get("bw", 0.0) >= YOLO_MIN_BW:
                self.events.publish("nav", f"YOLO spin found: {match['name']} at cx={match['cx']:.2f}")
                return match
        return None

    def _yolo_gimbal_search(self, labels: tuple[str, ...]):
        """Sweep gimbal looking for target. Returns pan angle or None."""
        self.events.publish("nav", "YOLO gimbal sweep search")
        scan_positions = [0, -60, 60, -120, 120, -160, 160]
        for pan in scan_positions:
            if self.stop_event.is_set():
                return None
            self._gimbal_send(pan, 0, spd=600, acc=80)
            self._sleep(0.5)
            dets, _, _ = self.camera.get_detections()
            match = self._yolo_select(dets, labels, 0.5, 0.5)
            if match and match.get("bw", 0.0) >= YOLO_MIN_BW:
                self.events.publish("nav", f"YOLO gimbal found: {match['name']} at pan={pan}°")
                self._gimbal_center(spd=600, acc=80)
                self._sleep(0.2)
                return pan
        self._gimbal_center(spd=600, acc=80)
        return None

    def _resolve_yolo_labels(self, target: str) -> tuple[str, ...]:
        """Ask LLM which YOLO labels match the target, or use simple mapping."""
        target_lower = target.lower().strip()
        # Direct matches for common targets
        direct_map = {
            "person": ("person", "legs"),
            "kitchen": ("refrigerator", "oven"),
            "desk": ("desk_chair", "monitor", "laptop"),
            "door": ("door_frame",),
            "couch": ("couch",),
            "bed": ("bed",),
            "toilet": ("toilet",),
            "tv": ("tv", "monitor"),
            "chair": ("chair", "desk_chair", "dining_chair"),
            "refrigerator": ("refrigerator",),
        }
        for key, labels in direct_map.items():
            if key in target_lower:
                return labels
        # Try LLM to resolve target to YOLO labels
        if self.llm is not None:
            try:
                available = self._get_detector_classes()
                raw = self.llm.complete(
                    prompt=load_prompt("resolve_yolo_labels.md", target=target, available=available),
                    system=load_prompt("json_only.system.md"),
                    max_tokens=80,
                )
                data = extract_json_dict(raw)
                if isinstance(data, dict) and data.get("labels"):
                    labels = tuple(str(l).lower() for l in data["labels"][:5])
                    self.events.publish("nav", f"LLM resolved '{target}' -> YOLO {labels}")
                    return labels
            except Exception as exc:
                self.events.publish("error", f"YOLO label resolution failed: {exc}")
        return ()

    def _resolve_yolo_labels_from_instruction(self, instruction: dict) -> tuple[str, ...]:
        """Extract YOLO labels from doorway instruction landmarks."""
        cues = list(instruction.get("doorway_landmarks") or [])
        cues += list(instruction.get("visual_cues") or [])
        target_room = instruction.get("target_room", "")
        # Combine instruction text for label resolution
        combined = f"{target_room} {' '.join(cues)}"
        if combined.strip():
            return self._resolve_yolo_labels(combined)
        return ()

    def _llm_verify_yolo_target(self, target: str, detection: dict) -> dict | None:
        """Ask LLM to verify: is the YOLO detection the right target?"""
        frame = self.camera.snap()
        if frame is None:
            return None
        det_name = detection.get("name", "unknown")
        det_cx = detection.get("cx", 0.5)
        det_bw = detection.get("bw", 0.0)
        try:
            raw = self.llm.complete(
                prompt=load_prompt("verify_yolo_target.md",
                    target=target, det_name=det_name,
                    det_cx=f"{det_cx:.2f}", det_bw=f"{det_bw:.2f}"),
                system=load_prompt("json_only.system.md"),
                image_bytes=frame,
                max_tokens=100,
            )
            data = extract_json_dict(raw)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            self.events.publish("error", f"LLM verify failed: {exc}")
        return None

    def _llm_find_target_in_frame(self, target: str, frame: bytes) -> tuple[float, float] | None:
        """Ask LLM to locate the target in the current frame.
        Returns (cx, cy) as 0-1 fractions, or None if not visible."""
        try:
            raw = self.llm.complete(
                prompt=load_prompt("find_target_in_frame.md", target=target),
                system=load_prompt("json_only.system.md"),
                image_bytes=frame,
                max_tokens=60,
            )
            data = extract_json_dict(raw)
            if isinstance(data, dict) and data.get("visible"):
                cx = float(data.get("cx", 0.5))
                cy = float(data.get("cy", 0.5))
                if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                    return (cx, cy)
        except Exception as exc:
            self.events.publish("error", f"LLM find target failed: {exc}")
        return None

    def _get_detector_classes(self) -> list[str]:
        """Get list of classes the YOLO detector knows."""
        if self.camera.detector is not None:
            try:
                return list(self.camera.detector.class_names)
            except Exception:
                pass
        return []

    # ===== LLM Reactive Navigation =====

    def navigate_reactive(self, target: str, *,
                          plan_context: str = "",
                          leg_hint: str = "",
                          waypoint_budget: int | None = None) -> bool:
        """LLM-driven navigation. LLM sees the image with YOLO overlay + depth
        + detections, decides what to do, rover executes. Simple loop."""
        self.reset()
        budget = waypoint_budget or self.config.navigation_waypoint_budget
        consecutive_non_drive = 0
        total_rotation_deg = 0.0
        last_actions: list[str] = []
        pending_depth: dict = {}
        # pending_depth_image removed — depth grid is text-based now
        nav_notes: dict = {}
        pending_notes_read: bool = False
        # Target sighting tracker — remembers where the target was last seen
        target_sighting: dict | None = None  # {step, pan_deg, scene_snippet, steps_since}
        blocked_bearings: list[float] = []  # Bearings where drives got blocked
        self._map_poll_active = True
        self.events.publish("nav", f"Reactive nav start: {target}")
        for step in range(budget):
            if self.stop_event.is_set():
                self.events.publish("nav", "Reactive nav aborted")
                self._map_poll_active = False
                return False

            # Gather sensor data: clean image
            frame = self.camera.snap()
            if frame is None:
                time.sleep(0.2)
                continue

            # Always get depth clearance + 8x8 grid for the LLM
            depth_dict = {}
            depth_map = self.camera.get_depth_map()
            if depth_map is not None:
                # Grid first — always works even if analyze() fails
                try:
                    grid = self._depth_to_grid(depth_map)
                except Exception as exc:
                    self.events.publish("error", f"Depth grid failed: {exc}")
                    grid = []
                try:
                    ds = self.depth_vectors.analyze(depth_map)
                    depth_dict = ds.to_prompt_dict()
                except Exception:
                    pass
                if grid:
                    depth_dict["depth_grid_8x8"] = grid
                    # Estimate straight-ahead clearance from center columns (3,4) floor rows (5-7)
                    center_col = [row[3] for row in grid] if len(grid) >= 8 else []
                    center_floor = [grid[r][c] for r in range(5, 8) for c in (3, 4)
                                    if r < len(grid) and c < len(grid[r])]
                    if center_floor:
                        ahead_clearance = min(center_floor)
                        safe_drive = max(0.0, ahead_clearance - 0.20)
                        depth_dict["ahead_clearance_m"] = round(ahead_clearance, 2)
                        depth_dict["safe_drive_m"] = round(safe_drive, 2)
                    self.events.publish("nav", f"Depth grid ok, center col={center_col}")
                    # Add narrow passage advisory for LLM
                    steer_hint = self._depth_steer_check(0.0)
                    if steer_hint in ("left", "right"):
                        depth_dict["narrow_passage"] = True
                        depth_dict["passage_hint"] = f"Walls close on sides — drive slow and straight, avoid wide angles"
                    elif steer_hint == "blocked":
                        depth_dict["blocked_ahead"] = True
                else:
                    self.events.publish("error", "Depth grid empty after _depth_to_grid")
            else:
                self.events.publish("nav", "No depth map available")
            # Override with check_depth results if pending
            if pending_depth:
                depth_dict = pending_depth
            pending_depth = {}

            # Inject notes into context if read was requested or notes exist
            notes_ctx = ""
            if pending_notes_read or nav_notes:
                notes_ctx = f" Your notes: {json.dumps(nav_notes, ensure_ascii=False)[:400]}"
                pending_notes_read = False

            # Build action history context for LLM
            action_ctx = plan_context + notes_ctx
            if last_actions:
                recent = ", ".join(last_actions[-5:])
                action_ctx = f"{plan_context} Your last actions: [{recent}]. Total body rotation so far: {total_rotation_deg:+.0f}°."
                if consecutive_non_drive >= 3:
                    action_ctx += (
                        f" WARNING: You have done {consecutive_non_drive} non-drive actions in a row! "
                        "You MUST drive forward NOW. Pick the clearest path and GO."
                    )
                if consecutive_non_drive >= 5:
                    safest = self._get_safest_heading()
                    action_ctx += (
                        f" CRITICAL: {consecutive_non_drive} non-drive steps! "
                        f"Safest heading is {safest:+.0f}°. Drive NOW or you will waste your budget."
                    )

            # Target sighting bearing hint
            if target_sighting is not None:
                sight = target_sighting
                bearing_offset = sight["pan_deg"] + (sight["rotation_deg"] - total_rotation_deg)
                bearing_offset = (bearing_offset + 180) % 360 - 180
                safe_drive = depth_dict.get("safe_drive_m", 0)
                age = sight["steps_since"]
                # Check if the direct path to target has been blocked recently
                direct_blocked = any(
                    abs(((b - total_rotation_deg + 180) % 360) - 180) < 30
                    for b in blocked_bearings[-4:]
                )
                if direct_blocked:
                    action_ctx += (
                        f" TARGET AT {bearing_offset:+.0f}° BUT DIRECT PATH BLOCKED "
                        f"(blocked {len(blocked_bearings)}x). "
                        f"DO NOT drive toward {bearing_offset:+.0f}° again. "
                        f"Go AROUND: turn ±90° from target bearing, drive to clear the obstacle, "
                        f"then re-approach from a different angle."
                    )
                elif abs(bearing_offset) < 10:
                    action_ctx += f" TARGET IS AHEAD ({bearing_offset:+.0f}°)."
                    if safe_drive >= 0.3:
                        action_ctx += f" Path clear {safe_drive:.1f}m. DRIVE NOW: drive({safe_drive:.1f})."
                else:
                    action_ctx += f" TARGET AT {bearing_offset:+.0f}°."
                    if safe_drive >= 0.3:
                        action_ctx += f" turn_body({bearing_offset:+.0f}) then drive({safe_drive:.1f})."
                    else:
                        action_ctx += " Path blocked — go around."

            # New room scan instruction
            if self._room_just_changed:
                action_ctx += (
                    " NEW ROOM ENTERED. STOP driving. Scan the room FIRST: "
                    "gimbal(-90,0)+wait(0.5), gimbal(+90,0)+wait(0.5), gimbal(0,0)+wait(0.3). "
                    "Study the spatial map after scanning to find the exit toward your target."
                )
                self._room_just_changed = False

            # Scene repetition context
            if self._scene_repeat_count >= 3:
                action_ctx += (
                    f" WARNING: The scene has NOT changed for {self._scene_repeat_count} steps. "
                    "You are stuck or looping. Try reversing, turning >90°, or using area_escape."
                )

            # Low budget context
            remaining = budget - step
            if remaining <= 5:
                action_ctx += (
                    f" CRITICAL: Only {remaining} steps left! "
                    "Try something drastically different — reverse far, turn 120°+, or area_escape."
                )

            # Compact history when context gets too large (~10K chars)
            est_size = len(action_ctx) + len(leg_hint) + sum(len(o) for o in self._recent_observations[-4:])
            if est_size > 10000:
                compacted = self._compact_history(
                    target=target,
                    observations=self._recent_observations,
                    actions=last_actions,
                    notes=nav_notes,
                    total_rotation=total_rotation_deg,
                )
                if compacted:
                    plan_context = compacted
                    self._recent_observations = self._recent_observations[-2:]
                    last_actions = last_actions[-3:]
                    nav_notes = {k: v for k, v in list(nav_notes.items())[-3:]}
                    action_ctx = plan_context
                    self.events.publish("nav", f"Context compacted to {len(plan_context)} chars")

            # Get YOLO detections for LLM context
            dets, det_summary, det_age = self.camera.get_detections()
            yolo_dets = dets if det_age < 2.0 else None

            # --- Place recognition: capture + match ---
            if self.place_db and depth_dict and frame:
                try:
                    if self.place_db.should_capture(step, total_rotation_deg):
                        depth_dists = depth_dict.get("smoothed_distances_m", [])
                        if not depth_dists:
                            # Fall back to raw grid center column
                            grid = depth_dict.get("depth_grid_8x8")
                            depth_dists = [row[3] for row in grid] if grid else []
                        sig = self.place_db.create_signature(
                            room_id=self._current_room_id or "unknown",
                            heading_deg=total_rotation_deg,
                            step_index=step,
                            detections=dets if det_age < 2.0 else [],
                            depth_distances=depth_dists,
                            jpeg_bytes=frame if isinstance(frame, bytes) else self.camera.get_jpeg() or b"",
                        )
                        if sig:
                            self.place_db.add(sig)
                            self.place_db.mark_captured(step, total_rotation_deg)
                            matches = self.place_db.find_closest(sig, top_k=1)
                            if matches:
                                matched_sig, score = matches[0]
                                self.events.publish("place",
                                    f"Place match: {matched_sig.room_id} sig#{matched_sig.sig_id} "
                                    f"(score={score:.2f}, db={self.place_db.size})")
                                if (matched_sig.room_id and score > 0.70
                                        and matched_sig.room_id != self._current_room_id
                                        and matched_sig.match_count >= 2):
                                    self.events.publish("place",
                                        f"Room correction: {self._current_room_id} → {matched_sig.room_id}")
                                    self._update_current_room(matched_sig.room_id)
                except Exception as exc:
                    self.events.publish("error", f"Place recognition error: {exc}")

            response = self._assess_waypoint(
                target=target,
                frame=frame,
                plan_context=action_ctx,
                leg_hint=leg_hint,
                depth_dict_override=depth_dict if depth_dict else None,
                yolo_detections=yolo_dets,
            )

            # Parse response
            scene = str(response.get("scene", "")).strip()
            reason = str(response.get("reason", "")).strip()
            room_guess = str(response.get("current_room", "")).strip()
            if scene:
                self._last_scene = scene
                self._remember(scene)
            if room_guess:
                self._update_current_room(room_guess)

            # --- Target sighting tracker ---
            target_visible = bool(response.get("target_visible"))
            if target_visible:
                sight_pan = self._last_gimbal_pan
                target_sighting = {
                    "step": step,
                    "pan_deg": sight_pan,
                    "rotation_deg": total_rotation_deg,
                    "scene_snippet": (scene or "")[:100],
                    "steps_since": 0,
                }
                self.events.publish("nav", f"Target sighted at pan={sight_pan:+.0f}° rotation={total_rotation_deg:+.0f}°")
                # Auto-face: if target is off-center, turn body to face it
                if abs(sight_pan) > 20:
                    self.events.publish("nav", f"Auto-facing target: turn_body({sight_pan:+.0f}°)")
                    self._turn(sight_pan)
                    total_rotation_deg += sight_pan
                    self.spatial_map.on_turn(sight_pan)
                    self._gimbal_center(spd=800, acc=80)
                    self._sleep(0.3)
            elif target_sighting is not None:
                target_sighting["steps_since"] += 1
                if target_sighting["steps_since"] > 6:
                    self.events.publish("nav", "Target sighting expired (6 steps without re-sighting)")
                    target_sighting = None

            # --- Feed spatial map ---
            self.spatial_map.set_step(step)
            # Depth center for distance estimate
            depth_center = 1.0
            grid = depth_dict.get("depth_grid_8x8")
            if grid and len(grid) >= 8:
                center_vals = [row[3] for row in grid[4:8] if len(row) > 3]
                if center_vals:
                    depth_center = min(center_vals)
                self.spatial_map.observe_depth_grid(
                    grid=grid, gimbal_pan_deg=self._last_gimbal_pan, step=step,
                )
            if scene:
                self.spatial_map.observe_scene(
                    gimbal_pan_deg=self._last_gimbal_pan,
                    scene=scene,
                    depth_center_m=depth_center,
                    step=step,
                )
            if yolo_dets:
                self.spatial_map.observe_yolo(
                    detections=yolo_dets,
                    gimbal_pan_deg=self._last_gimbal_pan,
                    step=step,
                )

            # Build tool list (supports single or chained tools)
            tools = response.get("tools")
            if not tools or not isinstance(tools, list):
                # Backwards compat: single tool in flat response
                tool_name = str(response.get("tool") or response.get("action", "look")).strip().lower()
                tools = [{"tool": tool_name, **{k: v for k, v in response.items()
                          if k not in ("tool", "action", "tools", "scene", "reason",
                                       "target_visible", "current_room")}}]

            tool_names = "+".join(t.get("tool", "?") for t in tools)
            self.events.publish(
                "nav",
                f"Step {step + 1}/{budget}: {tool_names} | {scene} | {reason}",
            )

            # === Execute tool chain ===
            step_drove = False
            abort = False
            for tc in tools:
                if self.stop_event.is_set():
                    abort = True
                    break
                tool = str(tc.get("tool", "")).strip().lower()

                if tool == "arrived":
                    self.rover.stop()
                    self._gimbal_center(spd=600, acc=80)
                    self.events.publish("nav", "LLM declares arrived!")
                    if self.speak_fn:
                        self.speak_fn("I'm here")
                    self._map_poll_active = False
                    return True

                if tool == "get_pose":
                    pose_info = (
                        f"gimbal_pan={self._last_gimbal_pan:+.0f}°, "
                        f"gimbal_tilt={self._last_gimbal_tilt:+.0f}°, "
                        f"body_rotation={total_rotation_deg:+.0f}°"
                    )
                    self.events.publish("nav", f"Pose: {pose_info}")
                    last_actions.append(f"get_pose()->{pose_info}")
                    continue

                if tool == "set_room":
                    room_id = str(tc.get("room_id", "")).strip()
                    if room_id:
                        self._update_current_room(room_id)
                        last_actions.append(f"set_room({room_id})")
                    continue

                if tool == "note":
                    key = str(tc.get("key", "")).strip()
                    value = tc.get("value", "")
                    if key:
                        nav_notes[key] = value
                        self.events.publish("nav", f"Note: {key}={json.dumps(value)[:80]}")
                        last_actions.append(f"note({key})")
                    continue

                if tool == "read_notes":
                    pending_notes_read = True
                    notes_str = json.dumps(nav_notes, ensure_ascii=False)[:300] if nav_notes else "empty"
                    self.events.publish("nav", f"Notes: {notes_str}")
                    last_actions.append(f"read_notes()->{len(nav_notes)} keys")
                    continue

                if tool == "ask_user":
                    question = str(tc.get("question", "Which way should I go?")).strip()
                    answer = self._ask_user(question)
                    last_actions.append(f"ask_user('{question[:30]}')->'{answer[:40]}'")
                    if answer:
                        # Keep only the last user answer, not all of them
                        plan_context = re.sub(r' User answered: ".*?"', '', plan_context)
                        plan_context += f' User answered: "{answer}"'
                    continue

                if tool == "check_depth":
                    depth_map = self.camera.get_depth_map()
                    if depth_map is not None:
                        try:
                            ds = self.depth_vectors.analyze(depth_map)
                            pending_depth = ds.to_prompt_dict()
                            pending_depth["depth_grid_8x8"] = self._depth_to_grid(depth_map)
                            from rover_brain_v2.prompts import _depth_summary_text
                            self.events.publish("nav", f"Depth: {_depth_summary_text(pending_depth)}")
                        except Exception:
                            pending_depth = {}
                    last_actions.append("check_depth()")
                    continue

                if tool == "correct_label":
                    yolo_label = str(tc.get("yolo_label", "")).strip()
                    correct = str(tc.get("correct_label", "")).strip()
                    if yolo_label and correct:
                        self._apply_label_correction(yolo_label, correct)
                        last_actions.append(f"correct_label({yolo_label}→{correct})")
                    continue

                if tool == "wheels":
                    lw = clamp(as_float(tc.get("left"), 0), -0.20, 0.20)
                    rw = clamp(as_float(tc.get("right"), 0), -0.20, 0.20)
                    self.rover.send({"T": 1, "L": round(lw, 3), "R": round(rw, 3)})
                    last_actions.append(f"wheels(L={lw:.2f},R={rw:.2f})")
                    if lw != 0 or rw != 0:
                        step_drove = True
                    continue

                if tool in ("gimbal", "look"):
                    gp = clamp(as_float(tc.get("pan") or tc.get("gimbal_pan"), 0), -180, 180)
                    gt = clamp(as_float(tc.get("tilt") or tc.get("gimbal_tilt"), 0), -30, 45)
                    self._gimbal_send(gp, gt, spd=600, acc=80)
                    last_actions.append(f"gimbal({gp:+.0f}°,{gt:+.0f}°)")
                    continue

                if tool == "wait":
                    secs = clamp(as_float(tc.get("seconds"), 0.5), 0.1, 3.0)
                    self._sleep(secs)
                    last_actions.append(f"wait({secs:.1f}s)")
                    continue

                if tool == "stop":
                    self.rover.stop()
                    last_actions.append("stop()")
                    continue

                if tool == "turn_body":
                    turn_deg = clamp(as_float(tc.get("angle"), 30), -120, 120)
                    self._turn(turn_deg)
                    total_rotation_deg += turn_deg
                    self.spatial_map.on_turn(turn_deg)
                    last_actions.append(f"turn_body({turn_deg:+.0f}°)")
                    continue

                if tool == "reverse":
                    dist = clamp(as_float(tc.get("distance"), 0.15), 0.10, 0.25)
                    speed = clamp(as_float(tc.get("speed"), 0.12), 0.12, 0.20)
                    self._reverse(dist, speed=speed)
                    self.spatial_map.on_drive(-dist, 0)
                    last_actions.append(f"reverse({dist:.2f}m,spd={speed:.2f})")
                    continue

                if tool == "read_map":
                    map_arr = self.spatial_map.to_array()
                    map_json = json.dumps(map_arr, ensure_ascii=False)[:600]
                    doors = self.spatial_map.door_summary()
                    self.events.publish("nav", f"Map: {len(map_arr)} entries. {doors}")
                    last_actions.append(f"read_map()->{len(map_arr)} entries")
                    continue

                if tool == "drive":
                    gp = as_float(tc.get("gimbal_pan"), None)
                    gt = as_float(tc.get("gimbal_tilt"), None)
                    if gp is not None or gt is not None:
                        gp_c = clamp(gp or 0, -180, 180)
                        gt_c = clamp(gt or 0, -30, 45)
                        self._gimbal_send(gp_c, gt_c, spd=600, acc=80)
                    requested_angle = as_float(tc.get("angle"), 0)
                    # No arced drives — turn body first, then drive straight
                    if abs(requested_angle) > 5:
                        self._turn(requested_angle)
                        total_rotation_deg += requested_angle
                        self.spatial_map.on_turn(requested_angle)
                        self.events.publish("nav", f"Auto turn_body({requested_angle:+.0f}°) before drive")
                    drive_angle = 0.0
                    drive_dist = clamp(as_float(tc.get("distance"), 0.4), 0.15, 2.0)
                    drive_speed = clamp(as_float(tc.get("speed"), self.config.navigation_drive_speed), 0.12, 0.30)
                    self._last_drive_angle = drive_angle
                    drive_result = self._drive_forward(drive_dist, drive_angle, speed=drive_speed)
                    if drive_result == "ok":
                        self.spatial_map.on_drive(drive_dist, drive_angle)
                    last_actions.append(f"drive({drive_dist:.2f}m,{drive_angle:+.0f}°,spd={drive_speed:.2f})={drive_result}")
                    step_drove = True
                    if drive_result == "aborted":
                        abort = True
                        break
                    if drive_result == "blocked":
                        blocked_bearings.append(total_rotation_deg)
                        self.events.publish("nav", f"Drive blocked at heading {total_rotation_deg:+.0f}° — LLM will reassess")
                        break
                    if drive_result == "stuck":
                        blocked_bearings.append(total_rotation_deg)
                        self.events.publish("nav", f"Drive stuck at heading {total_rotation_deg:+.0f}° — LLM will reassess")
                        break
                    continue

                last_actions.append(f"?({tool})")

            if abort and self.stop_event.is_set():
                return False
            consecutive_non_drive = 0 if step_drove else consecutive_non_drive + 1

            # --- Scene repetition detection (P0 #3) ---
            current_snap = self.camera.get_jpeg()
            if self._prev_snap and current_snap:
                if self._frames_similar(self._prev_snap, current_snap, threshold=0.08):
                    self._scene_repeat_count += 1
                else:
                    self._scene_repeat_count = 0
            self._prev_snap = current_snap

            # --- Step-per-room counter (P0 #6) ---
            if room_guess and room_guess != self._current_room_id:
                self._current_room_id = room_guess
                self._steps_in_current_room = 0
            else:
                self._steps_in_current_room += 1
            if self._steps_in_current_room >= 10:
                action_ctx += f" URGENT: You have been in {self._current_room_id} for {self._steps_in_current_room} steps. LEAVE THIS ROOM NOW."

        self.rover.stop()
        self._map_poll_active = False
        self.events.publish("nav", f"Reactive nav exhausted {budget} steps for {target}")
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
                system=load_prompt("scene.system.md"),
                image_bytes=frame,
                max_tokens=220,
            ).strip()
        except Exception as exc:
            self.events.publish("error", f"Scene description failed: {exc}")
            return ""

    def _update_current_room(self, room_id: str):
        if room_id and room_id != self._current_room_id:
            self._room_just_changed = True
            if self.speak_fn:
                self.speak_fn(room_id.replace("_", " "))
        # Update in-memory topo object (shared with orchestrator)
        topo = getattr(self, "topo", None)
        if topo is not None:
            try:
                known = {r.id for r in topo.rooms()}
                if room_id in known and getattr(topo, "current_room", None) != room_id:
                    topo.current_room = room_id
                    topo.save()
                    self.events.publish("room", f"Current room updated: {room_id}")
                    return
            except Exception:
                pass
        # Fallback: write directly to file
        topo_path = Path(__file__).resolve().parents[1] / "data" / "topo_map.json"
        try:
            data = json.loads(topo_path.read_text(encoding="utf-8"))
            if data.get("current_room") != room_id:
                known = {n["id"] for n in data.get("nodes", []) if n.get("type") == "room"}
                if room_id in known:
                    data["current_room"] = room_id
                    topo_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    self.events.publish("room", f"Current room updated: {room_id}")
        except Exception:
            pass

    def _load_topo_data(self) -> dict | None:
        topo_path = Path(__file__).resolve().parents[1] / "data" / "topo_map.json"
        try:
            return json.loads(topo_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _assess_waypoint(self, *, target: str, frame: bytes,
                         plan_context: str, leg_hint: str,
                         depth_summary=None,
                         yolo_detections: list[dict] | None = None,
                         depth_dict_override: dict | None = None) -> dict:
        depth_dict = depth_dict_override or (depth_summary.to_prompt_dict() if depth_summary else {})
        topo_data = self._load_topo_data()
        prompt = navigation_prompt(
            target=target,
            plan_context=plan_context,
            leg_hint=leg_hint,
            depth_context=depth_dict,
            recent_observations=self._recent_observations,
            yolo_detections=yolo_detections,
            topo_data=topo_data,
            current_room=self._current_room_id or None,
            target_room=getattr(self, '_nav_target_room', None),
            spatial_map_text=self.spatial_map.to_prompt_text(),
        )
        try:
            raw = self.llm.complete(
                prompt=prompt,
                system=load_prompt("navigation.system.md"),
                image_bytes=frame,
                max_tokens=3000,
            )
            data = extract_json_dict(raw)
            if isinstance(data, dict):
                return data
            self.events.publish("error", f"Navigator JSON parse failed: {raw[:160]}")
        except Exception as exc:
            self.events.publish("error", f"Navigator LLM failed: {exc}")
        # Fallback: look around instead of blindly driving forward
        return {
            "tool": "look",
            "scene": "",
            "reason": "fallback — LLM response failed, scanning instead of driving blind",
            "pan": 0,
            "tilt": 0,
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
            return f"the opening to {target_room} (look for: " + ", ".join(cue_parts) + ")"
        exit_hint = str(instruction.get("exit_hint") or "").strip()
        if exit_hint:
            return exit_hint
        return f"the opening to {target_room}"

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

    def _compact_history(self, *, target: str, observations: list[str],
                         actions: list[str], notes: dict,
                         total_rotation: float) -> str | None:
        """Truncate navigation history to keep context small. No LLM call."""
        kept_obs = observations[-2:]
        kept_acts = actions[-3:]
        kept_notes = {k: v for k, v in list(notes.items())[-3:]} if notes else {}
        parts = [f"Navigating to {target}. Rotation: {total_rotation:+.0f}°."]
        if kept_obs:
            parts.append("Recent: " + "; ".join(kept_obs))
        if kept_acts:
            parts.append("Actions: " + ", ".join(kept_acts))
        if kept_notes:
            parts.append("Notes: " + json.dumps(kept_notes, ensure_ascii=False)[:200])
        return " ".join(parts)

    def _remember(self, scene: str):
        entry = scene[:160]
        self._recent_observations.append(entry)
        if len(self._recent_observations) > 10:
            self._recent_observations = self._recent_observations[-10:]

    def _map_poll_loop(self):
        """Background thread: feed spatial map from sensors when not navigating."""
        while not self._navigator_worker_shutdown.is_set():
            try:
                if not self._map_poll_active:
                    self._poll_sensors_for_map()
                time.sleep(2.0)
            except Exception:
                time.sleep(5.0)

    def _poll_sensors_for_map(self):
        """Read current YOLO + depth and feed into spatial map."""
        dets, _, det_age = self.camera.get_detections()
        depth_map = self.camera.get_depth_map()
        self._map_poll_step += 1
        step = self._map_poll_step
        self.spatial_map.set_step(step)
        if dets and det_age < 2.0:
            self.spatial_map.observe_yolo(
                detections=dets,
                gimbal_pan_deg=self._last_gimbal_pan,
                step=step,
            )
        if depth_map is not None:
            try:
                grid = self._depth_to_grid(depth_map)
                if grid:
                    self.spatial_map.observe_depth_grid(
                        grid=grid,
                        gimbal_pan_deg=self._last_gimbal_pan,
                        step=step,
                    )
            except Exception:
                pass

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
        # On every retry, do a gimbal scan first to reacquire the target
        scan_result = self._gimbal_scan_for_target(self._goal_target_label, instruction)
        if scan_result is not None:
            self.events.publish("nav", f"Gimbal scan found cue at pan={scan_result:+.0f}° — turning body")
            self._turn(scan_result)
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
            # After 2 consecutive reacquire turns, commit to driving forward
            # instead of spinning endlessly looking for the goal.
            if indecisive_streak >= 2 and corridor_clear >= 0.45:
                action = "drive_forward"
                response["drive_angle"] = clamp(exit_heading, -30.0, 30.0)
                response["drive_distance"] = clamp(max(0.40, fluid_distance), 0.40, 0.90)
                override_note = "reacquire -> commit drive toward goal"
                return action, response, override_note
            # After 4+ reacquire turns with no progress, the heading bias is
            # probably wrong. Reset it and let the LLM decide freely.
            if indecisive_streak >= 4:
                self._goal_heading_bias_deg = None
                self._goal_heading_confidence = 0.0
                self._goal_recently_seen_steps = 0
                self._goal_reacquire_steps = 0
                override_note = "reacquire failed -> heading reset"
                # Fall through to normal LLM action
                return action, response, override_note
            # Use gimbal pan to search instead of body turn
            forced_pan = exit_heading
            if abs(forced_pan) < 18.0:
                forced_pan = 28.0 if forced_pan >= 0 else -28.0
            action = "turn"
            response["turn_degrees"] = clamp(forced_pan, -75.0, 75.0)
            response["target_visible"] = False  # signal: gimbal only
            override_note = "lost goal evidence -> gimbal reacquire"
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
        # Center gimbal before turning so camera faces forward
        if abs(self._last_gimbal_pan) > 5:
            self._gimbal_center(spd=800, acc=80)
        sign = 1 if degrees > 0 else -1
        peak_speed = clamp(
            as_float(getattr(self.config, "navigation_turn_speed", 0.24), 0.24),
            0.16, TURN_SPEED,
        )
        base_rate = getattr(self.config, "turn_rate_dps", TURN_RATE_DPS)
        turn_rate_dps = max(60.0, base_rate * (peak_speed / TURN_SPEED))
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

    def _reverse(self, distance_m: float, *, speed: float = 0.12):
        # Center gimbal before reversing
        if abs(self._last_gimbal_pan) > 5:
            self._gimbal_center(spd=800, acc=80)
        rev_speed = clamp(speed, 0.05, 0.20)
        duration = max(0.3, abs(distance_m) / max(rev_speed, 0.05))
        look_behind = getattr(self.flags, "reverse_look_behind", False)
        if look_behind:
            # Look behind before reversing — wait for gimbal to reach 180° and camera to refresh
            self._gimbal_send(180, 0, spd=600, acc=80)
            self.events.publish("nav", f"Looking behind (pan 180°) before reversing {distance_m:.2f}m at speed={rev_speed:.2f}")
            # Wait for gimbal to reach 180° (~0.8s at SPD 600) + camera frame refresh
            self._sleep(1.0)
            # Analyze what's behind: depth + YOLO
            rear_depth_map = self.camera.get_depth_map()
            if rear_depth_map is not None:
                try:
                    ds = self.depth_vectors.analyze(rear_depth_map)
                    rear_center = ds.center_distance()
                    rear_corridor = ds.closest_corridor_distance() or rear_center
                    self.events.publish("nav", f"Rear depth: center={rear_center:.2f}m, corridor={rear_corridor:.2f}m")
                    stop_m = max(0.05, float(self.config.depth_guard_stop_m))
                    if min(rear_center, rear_corridor) < stop_m:
                        self.events.publish("nav", f"REAR DEPTH GUARD: obstacle at {min(rear_center, rear_corridor):.2f}m behind, aborting reverse")
                        self._gimbal_center()
                        self._sleep(1.2)
                        return
                except Exception:
                    pass
            dets, det_summary, det_age = self.camera.get_detections()
            if det_age < 2.0 and dets:
                det_names = [f"{d['name']}({d.get('bh', 0):.0%})" for d in dets[:5]]
                self.events.publish("nav", f"Rear YOLO: {', '.join(det_names)}")
                for d in dets:
                    if 0.25 < d.get("cx", 0) < 0.75 and d.get("bh", 0) > 0.4:
                        self.events.publish("nav", f"YOLO GUARD: {d['name']} behind rover (bh={d['bh']:.0%}), aborting reverse")
                        self._gimbal_center()
                        self._sleep(1.2)
                        return
        else:
            self.events.publish("nav", f"Reversing {distance_m:.2f}m at speed={rev_speed:.2f} (look-behind disabled)")
        self.rover.send({"T": 1, "L": round(-rev_speed, 3), "R": round(-rev_speed, 3)})
        if look_behind:
            # YOLO guard while reversing (gimbal is pointed behind)
            start = time.time()
            while time.time() - start < duration:
                if self.stop_event.is_set():
                    self.rover.stop()
                    return
                dets, _, det_age = self.camera.get_detections()
                if det_age < 1.0:
                    for d in dets:
                        if 0.25 < d.get("cx", 0) < 0.75 and d.get("bh", 0) > 0.4:
                            self.rover.stop()
                            self.events.publish("nav", f"YOLO GUARD: {d['name']} behind rover (bh={d['bh']:.0%}), stopping reverse")
                            self._gimbal_center()
                            self._sleep(1.2)
                            return
                time.sleep(0.05)
        else:
            # Simple timed reverse without look-behind
            start = time.time()
            while time.time() - start < duration:
                if self.stop_event.is_set():
                    self.rover.stop()
                    return
                time.sleep(0.05)
        self.rover.stop()
        if look_behind:
            # Re-center gimbal after reverse (was at 180°, needs ~1s at SPD 600)
            self._gimbal_center()
            self._sleep(1.2)

    def _reverse_escape_angle(self) -> float:
        """Compute a turn angle after reversing based on depth clearance."""
        depth_map = self.camera.get_depth_map()
        if depth_map is None:
            return 0.0
        try:
            ds = self.depth_vectors.analyze(depth_map)
            heading = ds.recommended_heading_deg
            # If the safest heading is roughly forward, check left vs right clearance
            if abs(heading) < 10:
                dists = ds.smoothed_distances_m
                n = len(dists)
                if n < 3:
                    return 0.0
                third = max(1, n // 3)
                left_avg = sum(dists[:third]) / third
                right_avg = sum(dists[2 * third:]) / max(1, n - 2 * third)
                if left_avg > right_avg + 0.15:
                    return -30.0
                elif right_avg > left_avg + 0.15:
                    return 30.0
                return 0.0
            # Clamp to reasonable escape turn
            return clamp(heading * 1.2, -75.0, 75.0)
        except Exception:
            return 0.0

    def _ask_user(self, question: str, timeout_s: float = 15.0) -> str:
        """Speak a question and wait for user voice/text response."""
        self.events.publish("nav", f"Asking user: {question}")
        # Speak the question
        if self.speak_fn:
            try:
                self.speak_fn(question)
            except Exception:
                pass
        # Listen for response
        if self.listen_fn:
            try:
                answer = self.listen_fn(timeout_s=timeout_s)
                if answer:
                    self.events.publish("nav", f"User replied: {answer}")
                    return answer
            except Exception as exc:
                self.events.publish("error", f"listen_fn failed: {exc}")
        self.events.publish("nav", "No user response (timeout or no mic)")
        return ""

    def _inspect(self):
        if self.flags.gimbal_pan_enabled:
            sweep = [0, -55, 55, 0]
            pan = sweep[self._scan_index % len(sweep)]
            self._scan_index += 1
            self.events.publish("nav", f"Inspect sweep pan={pan:+d}°")
            self._gimbal_send(pan, 0, spd=600, acc=80)
            self._sleep(0.6)
            self._gimbal_center(spd=600, acc=80)
        else:
            self._turn(25 if (self._scan_index % 2 == 0) else -25)
            self._scan_index += 1

    def _drive_forward(self, distance_m: float, drive_angle_deg: float, *, speed: float | None = None) -> str:
        """Drive forward with hard depth guard before sending wheel commands."""
        # ALWAYS center gimbal before driving — no exceptions
        old_pan = self._last_gimbal_pan
        self._gimbal_center(spd=800, acc=80)
        if abs(old_pan) > 5:
            wait_s = max(0.3, abs(old_pan) * 0.006)
            self._sleep(wait_s)
            self.events.publish("nav", f"Gimbal centered from {old_pan:+.0f}° before drive")
        # Hard depth guard: refuse to drive into obstacles
        steer = self._depth_steer_check(drive_angle_deg)
        if steer == "blocked":
            self.events.publish("nav", "DEPTH GUARD: blocked at center, refusing drive")
            return "blocked"
        # Snap before driving for stuck detection
        snap_before = self.camera.get_jpeg()
        base_speed = speed if speed is not None else self.config.navigation_drive_speed
        duration = distance_m / max(base_speed, 0.05)
        steer = clamp(drive_angle_deg / 60.0, -1.0, 1.0) * base_speed * 0.8
        left = base_speed + steer
        right = base_speed - steer
        self.events.publish("nav", f"Driving {distance_m:.2f}m at {drive_angle_deg:+.0f}° speed={base_speed:.2f}")
        self.rover.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})
        start = time.time()
        last_depth_check = 0.0
        while time.time() - start < duration:
            if self.stop_event.is_set():
                self.rover.stop()
                return "aborted"
            # Real-time collision guard: check every ~200ms while driving
            # Only use center_distance (not corridor, which picks up floor noise)
            now = time.time()
            if now - last_depth_check >= 0.20:
                last_depth_check = now
                depth_map = self.camera.get_depth_map()
                if depth_map is not None:
                    try:
                        ds = self.depth_vectors.analyze(depth_map)
                        center_d = ds.center_distance()
                        if center_d < 0.15:
                            self.rover.stop()
                            self.events.publish("nav", f"DEPTH GUARD: obstacle at {center_d:.2f}m while driving, emergency stop")
                            return "blocked"
                    except Exception:
                        pass
            time.sleep(0.05)
        self.rover.stop()
        # Image-based stuck detection: if frame barely changed, we didn't move
        snap_after = self.camera.get_jpeg()
        if snap_before and snap_after and self._frames_similar(snap_before, snap_after, threshold=0.05):
            self._frame_unchanged_count += 1
            self.events.publish(
                "nav",
                f"STUCK detected: image changed <5% after drive "
                f"(streak={self._frame_unchanged_count})",
            )
            if self._frame_unchanged_count >= 2:
                self.events.publish("nav", "Auto-reversing after repeated stuck detection")
                self._reverse(0.15)
                self._frame_unchanged_count = 0
            return "stuck"
        self._frame_unchanged_count = 0
        return "ok"

    def _recover_from_close_obstacle(self, *, drive_angle_deg: float, depth_summary, scene: str) -> bool:
        if self.stop_event.is_set():
            return False
        # Determine reverse distance based on how close the obstacle is
        stop_threshold = max(0.05, float(self.config.depth_guard_stop_m))
        center_clear = depth_summary.center_distance() if depth_summary is not None else stop_threshold
        furniture_trap, _ = self._scene_furniture_trap(scene)
        reverse_distance = BLOCKED_ESCAPE_REVERSE_M
        if furniture_trap or center_clear <= stop_threshold * 0.75:
            reverse_distance = BLOCKED_ESCAPE_FURNITURE_REVERSE_M

        self.events.publish("nav", f"Blocked recovery: reversing {reverse_distance:.2f}m, then scanning for opening")
        self._reverse(reverse_distance)
        if self.stop_event.is_set():
            return False

        # After reversing, get FRESH depth from the new position and find the best opening
        turn_deg = self._find_best_escape_turn(drive_angle_deg)
        self.events.publish("nav", f"Escape turn: {turn_deg:+.0f}°")
        self._turn(turn_deg)
        # Re-send gimbal center after turn (rover.stop() sends T:135 which kills servos)
        self._gimbal_center()
        self._sleep(0.3)

        if self._goal_target_label:
            self._goal_reacquire_steps = max(self._goal_reacquire_steps, 8)
            self._goal_recently_seen_steps = max(self._goal_recently_seen_steps, 6)
            if self._goal_heading_bias_deg is not None:
                escape_sign = 1 if turn_deg >= 0 else -1
                goal_sign = 1 if self._goal_heading_bias_deg >= 0 else -1
                if escape_sign != goal_sign:
                    self._goal_heading_confidence = max(self._goal_heading_confidence, 0.82)
            else:
                self._goal_heading_bias_deg = clamp(-turn_deg * 0.6, -28.0, 28.0)
            self._goal_heading_confidence = max(self._goal_heading_confidence, 0.70)
        return not self.stop_event.is_set()

    def _find_best_escape_turn(self, last_drive_angle: float) -> float:
        """After reversing, scan depth for the widest passable opening.

        Returns a turn angle in degrees toward the best gap.  The depth camera
        only covers ~65° FOV, so if the best gap is at the frame edge the
        real opening is probably further out — we extrapolate.  Alternates
        tie-break direction to avoid always picking the same side.
        """
        depth_map = self.camera.get_depth_map()
        if depth_map is None:
            turn = float(self._blocked_escape_dir * BLOCKED_ESCAPE_MIN_TURN_DEG)
            self._blocked_escape_dir *= -1
            return turn

        try:
            ds = self.depth_vectors.analyze(depth_map)
        except Exception:
            turn = float(self._blocked_escape_dir * BLOCKED_ESCAPE_MIN_TURN_DEG)
            self._blocked_escape_dir *= -1
            return turn

        dists = ds.smoothed_distances_m
        n = len(dists)
        passable_m = max(0.05, float(self.config.depth_guard_stop_m)) + 0.15

        # Find the widest contiguous gap of passable columns
        best_start, best_len, best_center = 0, 0, n // 2
        run_start = None
        for i, d in enumerate(dists):
            if d >= passable_m:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_len = i - run_start
                    if run_len > best_len:
                        best_len = run_len
                        best_start = run_start
                        best_center = run_start + run_len // 2
                    run_start = None
        if run_start is not None:
            run_len = n - run_start
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
                best_center = run_start + run_len // 2

        if best_len == 0:
            # Nothing passable in the entire FOV — need a big turn to find open space
            turn = float(self._blocked_escape_dir * BLOCKED_ESCAPE_MAX_TURN_DEG)
            self._blocked_escape_dir *= -1
            self.events.publish("nav", f"No passable gap in FOV, blind escape {turn:+.0f}°")
            return turn

        # Convert gap center column to heading degrees within FOV
        # Column 0 = far left (~-32.5°), column n-1 = far right (~+32.5°)
        center_norm = (best_center - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
        heading_deg = center_norm * 32.5

        # If the gap touches the frame edge, the real opening extends beyond
        # what we can see.  Extrapolate outward: the opening is at LEAST at
        # the FOV edge, probably further.  Add 50% of the edge heading as
        # extra turn so we actually face the opening.
        gap_end = best_start + best_len - 1
        edge_margin = max(1, n // 6)  # ~3 columns for 21-col map
        at_left_edge = best_start <= edge_margin
        at_right_edge = gap_end >= n - 1 - edge_margin
        if at_left_edge and not at_right_edge:
            # Gap extends off the left edge — turn more left
            heading_deg = min(heading_deg, -20.0)
            heading_deg *= 1.5
        elif at_right_edge and not at_left_edge:
            # Gap extends off the right edge — turn more right
            heading_deg = max(heading_deg, 20.0)
            heading_deg *= 1.5

        # Clamp and enforce minimum
        turn = clamp(heading_deg, -BLOCKED_ESCAPE_MAX_TURN_DEG, BLOCKED_ESCAPE_MAX_TURN_DEG)
        if abs(turn) < BLOCKED_ESCAPE_MIN_TURN_DEG:
            sign = 1.0 if turn >= 0 else -1.0
            if abs(turn) < 5.0:
                # Nearly centered gap — use alternating direction
                sign = float(self._blocked_escape_dir)
                self._blocked_escape_dir *= -1
            turn = sign * BLOCKED_ESCAPE_MIN_TURN_DEG

        self.events.publish(
            "nav",
            f"Depth scan: gap cols {best_start}-{gap_end} "
            f"(width={best_len}/{n}), raw={heading_deg:+.1f}° → turn={turn:+.0f}°",
        )
        return turn

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

    # --- Gimbal look (search without body turn) ---

    def _gimbal_look(self, pan_deg: float):
        """Pan the gimbal to look in a direction without turning the body.
        If the target is spotted, turn the body to face it."""
        pan_deg = clamp(pan_deg, -160.0, 160.0)
        self.events.publish("nav", f"Gimbal look pan={pan_deg:+.0f}°")
        self._gimbal_send(pan_deg, 0, spd=600, acc=80)
        self._sleep(0.5)
        # Snap a frame at this angle — the next waypoint's LLM call will see it
        # If pan is large enough and target is visible, next step will drive toward it
        # Return gimbal to center after a brief pause
        self._sleep(0.3)
        # If gimbal is far off center, check if target is there before returning
        if abs(pan_deg) >= 40:
            frame = self.camera.snap()
            if frame is not None and self._goal_target_label:
                goal_visible = self._quick_check_target_visible(frame)
                if goal_visible:
                    self.events.publish("nav", f"Target spotted at gimbal {pan_deg:+.0f}° — turning body")
                    self._gimbal_center()
                    self._sleep(0.2)
                    self._turn(pan_deg)
                    self._goal_heading_bias_deg = clamp(pan_deg * 0.2, -28.0, 28.0)
                    self._goal_heading_confidence = 0.85
                    self._goal_recently_seen_steps = 6
                    self._goal_reacquire_steps = 0
                    return
        self._gimbal_center()

    def _quick_check_target_visible(self, frame: bytes) -> bool:
        """Quick LLM check if the target is visible in the current frame."""
        target = self._goal_target_label or "target"
        try:
            raw = self.llm.complete(
                prompt=load_prompt("quick_check_target.md", target=target),
                system=load_prompt("json_only.system.md"),
                image_bytes=frame,
                max_tokens=40,
            )
            data = extract_json_dict(raw)
            return isinstance(data, dict) and bool(data.get("visible"))
        except Exception:
            return False

    # --- Gimbal scan for doorway reacquisition ---

    def _gimbal_scan_for_target(self, target_label: str = "", instruction: dict | None = None) -> float | None:
        """Sweep the gimbal across multiple positions and ask the LLM where
        the target is.  Returns the pan angle (degrees) where the best cue
        was found, or None if nothing was spotted."""
        instruction = instruction or {}
        cues = list(instruction.get("doorway_landmarks") or instruction.get("visual_cues") or [])
        if cues:
            cue_text = ", ".join(cues[:3])
        elif target_label:
            cue_text = target_label
        else:
            target_room = instruction.get("target_room", "target")
            cue_text = f"doorway to {target_room}"
        scan_positions = [0, -70, 70, -140, 140]
        best_pan = None
        best_score = 0.0
        self.events.publish("nav", f"Gimbal scan: looking for {cue_text}")
        for pan in scan_positions:
            if self.stop_event.is_set():
                break
            self._gimbal_send(pan, 0, spd=600, acc=80)
            self._sleep(0.55)
            frame = self.camera.snap()
            if frame is None:
                continue
            # Quick LLM check: is the doorway visible at this pan?
            try:
                raw = self.llm.complete(
                    prompt=load_prompt("gimbal_scan_for_target.md", cue_text=cue_text),
                    system=load_prompt("json_only.system.md"),
                    image_bytes=frame,
                    max_tokens=80,
                )
                data = extract_json_dict(raw)
                if isinstance(data, dict) and data.get("visible"):
                    conf = as_float(data.get("confidence"), 0.5)
                    if conf > best_score:
                        best_score = conf
                        pos = str(data.get("position", "center")).lower()
                        # Adjust pan based on position within frame
                        offset = -15 if pos == "left" else 15 if pos == "right" else 0
                        best_pan = pan + offset
                        self.events.publish("nav", f"Gimbal scan: cue at pan={pan}° ({pos}, conf={conf:.2f})")
            except Exception as exc:
                self.events.publish("error", f"Gimbal scan LLM failed: {exc}")
        # Return gimbal to center
        self._gimbal_center(spd=600, acc=80)
        self._sleep(0.3)
        if best_pan is not None and best_score >= 0.4:
            return clamp(best_pan, -160.0, 160.0)
        return None

    # --- Wall detection ---

    def _detect_wall(self, frame: bytes, depth_map) -> bool:
        if depth_map is not None and self._depth_wall_check(depth_map):
            return True
        if frame is not None and self._image_wall_check(frame):
            return True
        return False

    def _depth_wall_check(self, depth_map) -> bool:
        h, w = depth_map.shape[:2]
        x0, x1 = int(w * 0.10), int(w * 0.90)
        y0, y1 = int(h * 0.20), int(h * 0.80)
        region = depth_map[y0:y1, x0:x1]
        if region.size == 0:
            return False
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max - d_min < WALL_DEPTH_RANGE_MIN:
            self.events.publish("nav", f"Wall (depth): uniform range={d_max - d_min:.4f}")
            return True
        close_threshold = d_min + (d_max - d_min) * 0.85
        close_fraction = float(np.mean(region > close_threshold))
        if close_fraction > WALL_CLOSE_FRACTION:
            self.events.publish("nav", f"Wall (depth): {close_fraction:.0%} very close")
            return True
        return False

    def _image_wall_check(self, jpeg: bytes) -> bool:
        try:
            arr = np.frombuffer(jpeg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            small = cv2.resize(img, (80, 60))
            std = float(small.std())
            hist = cv2.calcHist([small], [0], None, [16], [0, 256])
            dominant = float(hist.max() / hist.sum())
            if std < WALL_IMAGE_STD_MAX and dominant > WALL_IMAGE_DOMINANT_MIN:
                self.events.publish("nav", f"Wall (image): std={std:.0f}, dominant={dominant:.0%}")
                return True
        except Exception:
            pass
        return False

    def _recover_from_wall(self):
        self._reverse(0.20)
        if self.stop_event.is_set():
            return
        # Scan for best opening after reversing
        turn_deg = self._find_best_escape_turn(self._last_drive_angle)
        # Wall recovery needs a bigger turn — at least 60°
        if abs(turn_deg) < 60:
            turn_deg = 60.0 * (1 if turn_deg >= 0 else -1)
        self.events.publish("nav", f"Wall recovery: turning {turn_deg:+.0f}°")
        self._turn(turn_deg)
        self._gimbal_center()
        self._sleep(0.3)
        if self._goal_target_label:
            self._goal_reacquire_steps = max(self._goal_reacquire_steps, 8)
            self._goal_heading_confidence = max(self._goal_heading_confidence, 0.70)

    # --- Reactive steering during drive ---

    def _depth_steer_check(self, drive_angle_deg: float = 0.0) -> str:
        depth_map = self.camera.get_depth_map()
        if depth_map is None:
            return "clear"
        try:
            summary = self.depth_vectors.analyze(depth_map)
        except Exception:
            return "clear"
        stop_m = max(0.05, float(self.config.depth_guard_stop_m))
        center_d = summary.center_distance()
        corridor_d = summary.closest_corridor_distance() or center_d
        if min(center_d, corridor_d) < stop_m:
            # Check if we can steer around
            cols = summary.smoothed_distances_m
            n = len(cols)
            mid = n // 2
            left_avg = sum(cols[:mid]) / max(mid, 1)
            right_avg = sum(cols[mid + 1:]) / max(n - mid - 1, 1)
            if left_avg < stop_m and right_avg < stop_m:
                return "blocked"
            return "right" if right_avg > left_avg else "left"
        # Softer avoidance: steer if corridor is tight on one side
        steer_threshold = stop_m + 0.15
        if corridor_d < steer_threshold:
            rec_heading = summary.recommended_heading_deg
            if rec_heading > 8.0:
                return "right"
            if rec_heading < -8.0:
                return "left"
        return "clear"

    # --- Image-based stuck detection ---

    def _depth_to_grid(self, depth_map, rows: int = 8, cols: int = 8) -> list[list[float]]:
        """Convert a depth map to an 8x8 grid of depth values in meters."""
        h, w = depth_map.shape[:2]
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max <= d_min:
            return [[0.0] * cols for _ in range(rows)]
        scale = getattr(self.depth_vectors, "depth_max_clearance_m", 2.4)
        grid = []
        for r in range(rows):
            y0 = int(r * h / rows)
            y1 = int((r + 1) * h / rows)
            row = []
            for c in range(cols):
                x0 = int(c * w / cols)
                x1 = int((c + 1) * w / cols)
                patch = depth_map[y0:y1, x0:x1]
                # DepthAnything: higher value = closer, invert to get distance
                mean_val = float(np.mean(patch))
                normalized = (mean_val - d_min) / (d_max - d_min)
                distance_m = (1.0 - normalized) * scale
                row.append(round(distance_m, 1))
            grid.append(row)
        return grid

    def _frames_similar(self, frame1: bytes, frame2: bytes, threshold: float = 0.08) -> bool:
        try:
            img1 = cv2.imdecode(np.frombuffer(frame1, np.uint8), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imdecode(np.frombuffer(frame2, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                return False
            img1 = cv2.resize(img1, (160, 120))
            img2 = cv2.resize(img2, (160, 120))
            diff = cv2.absdiff(img1, img2)
            percent_change = float(np.mean(diff)) / 255.0
            return percent_change < threshold
        except Exception:
            return False

    # --- Area escape ---

    def _apply_label_correction(self, yolo_label: str, correct_label: str):
        """Apply an LLM label correction to the YOLO detector and persist it."""
        try:
            from local_detector import LABEL_OVERRIDES, _save_label_overrides
            old = LABEL_OVERRIDES.get(yolo_label)
            LABEL_OVERRIDES[yolo_label] = correct_label
            _save_label_overrides()
            self.events.publish("nav", f"YOLO label corrected: '{yolo_label}' → '{correct_label}' (was: {old!r})")
        except Exception as exc:
            self.events.publish("error", f"Label correction failed: {exc}")

    def _area_escape(self):
        self.events.publish("nav", "AREA ESCAPE: reversing 0.4m + turning 120°")
        self.rover.stop()
        self._reverse(0.40)
        if self.stop_event.is_set():
            return
        self._sleep(0.2)
        escape_deg = 120 if self._area_escape_dir > 0 else -120
        self._area_escape_dir = -self._area_escape_dir
        self._turn(escape_deg)
        # Don't set reacquire here — let LLM see the new view first.
        # The goal heading is preserved but we don't force turns immediately.

    def _get_safest_heading(self) -> float:
        """Return the recommended heading toward the most open direction."""
        depth_map = self.camera.get_depth_map()
        if depth_map is not None:
            try:
                ds = self.depth_vectors.analyze(depth_map)
                return float(ds.recommended_heading_deg)
            except Exception:
                pass
        return 0.0

    # --- LLM prefetch ---

    def _submit_prefetch(self, target: str, plan_context: str, leg_hint: str):
        self._cancel_prefetch()
        self._prefetch_context = {
            "target": target,
            "plan_context": plan_context,
            "leg_hint": leg_hint,
        }
        self._prefetch_future = self._prefetch_executor.submit(
            self._run_prefetch, target, plan_context, leg_hint,
        )

    def _run_prefetch(self, target: str, plan_context: str, leg_hint: str) -> dict | None:
        try:
            frame = self.camera.snap()
            depth_map = self.camera.get_depth_map()
            if frame is None or depth_map is None:
                return None
            depth_summary = self._analyze_depth_for_goal(depth_map)
            return self._assess_waypoint(
                target=target,
                frame=frame,
                depth_summary=depth_summary,
                plan_context=plan_context,
                leg_hint=leg_hint,
            )
        except Exception as exc:
            self.events.publish("error", f"Prefetch LLM error: {exc}")
            return None

    def _consume_or_assess(self, *, target, frame, depth_summary, plan_context, leg_hint) -> dict:
        if self._prefetch_future is not None:
            try:
                result = self._prefetch_future.result(timeout=0.3)
                self._prefetch_future = None
                if result is not None:
                    self.events.publish("nav", "Using prefetched LLM assessment")
                    return result
            except Exception:
                pass
            self._prefetch_future = None
        return self._assess_waypoint(
            target=target,
            frame=frame,
            depth_summary=depth_summary,
            plan_context=plan_context,
            leg_hint=leg_hint,
        )

    def _cancel_prefetch(self):
        if self._prefetch_future is not None:
            self._prefetch_future.cancel()
            self._prefetch_future = None

    def _sleep(self, seconds: float):
        end = time.time() + max(seconds, 0.0)
        while time.time() < end:
            if self.stop_event.is_set():
                break
            time.sleep(0.05)
