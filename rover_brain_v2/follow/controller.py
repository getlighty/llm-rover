"""YOLO-based follow-me controller with DepthAnything crash guarding."""

from __future__ import annotations

import math
import random
import time

from rover_brain_v2.follow.depth_guard import DepthCrashGuard
from rover_brain_v2.prompts import follow_callout_prompt


GIMBAL_GAIN = 40.0
GIMBAL_DEADZONE = 0.05
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 45
BW_DEADZONE = 0.05
FWD_GAIN = 5.0
REV_GAIN = 3.0
STEER_GAIN = 0.005
MAX_STEER = 0.15
CENTERING_TOL = 0.15
LOST_TIMEOUT = 3.0
INITIAL_TILT = 30.0
MIN_BW = 0.03
CALLOUT_LINES = [
    "Where did you go?",
    "Come back here.",
    "I lost you.",
    "Still following if you come back.",
]


class FollowMeController:
    def __init__(self, *, rover, camera, event_bus, config, depth_guard: DepthCrashGuard | None = None,
                 llm_client=None, speak_fn=None, flags=None, imu=None):
        self.rover = rover
        self.camera = camera
        self.events = event_bus
        self.config = config
        self.depth_guard = depth_guard or DepthCrashGuard(
            stop_distance_m=config.depth_guard_stop_m,
            turn_stop_distance_m=config.depth_guard_turn_stop_m,
        )
        self.llm = llm_client
        self.speak_fn = speak_fn
        self.flags = flags
        self.imu = imu
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.rover.stop()

    def follow(self, target: str = "person", duration: float = 60.0,
               target_bw: float = 0.25) -> dict:
        if self.camera.detector is None:
            return {"status": "detector_unavailable"}
        self._cancelled = False
        self.camera.set_follow_mode(True)
        labels = self._target_labels(target)
        pan = 0.0
        tilt = INITIAL_TILT
        found_ever = False
        last_seen = 0.0
        last_cx = 0.5
        last_cy = 0.5
        callout_done = False
        start = time.time()
        bw_sum = 0.0
        bw_count = 0
        self.rover.send({"T": 133, "X": 0, "Y": INITIAL_TILT, "SPD": 200, "ACC": 10})
        self.events.publish("follow", f"Following {target} for {duration:.0f}s")
        try:
            while time.time() - start < duration:
                if self._cancelled:
                    return self._result("cancelled", time.time() - start, bw_sum, bw_count)
                dets, _summary, _age = self.camera.get_detections()
                match = self._select_target(dets, labels, last_cx, last_cy)
                if match is None or match.get("bw", 0.0) < MIN_BW:
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    lost_for = (time.time() - last_seen) if found_ever else (time.time() - start)
                    if lost_for >= LOST_TIMEOUT and not callout_done:
                        self._callout(target)
                        callout_done = True
                    if found_ever and lost_for >= LOST_TIMEOUT:
                        found = self._spin_search(labels)
                        if found is None:
                            return self._result("lost", time.time() - start, bw_sum, bw_count)
                        callout_done = False
                        last_seen = time.time()
                        continue
                    time.sleep(1.0 / self.config.follow_loop_hz)
                    continue
                found_ever = True
                callout_done = False
                last_seen = time.time()
                cx = match["cx"]
                cy = match["cy"]
                bw = match["bw"]
                last_cx = cx
                last_cy = cy
                bw_sum += bw
                bw_count += 1
                err_x = cx - 0.5
                err_y = cy - 0.5
                if abs(err_x) > GIMBAL_DEADZONE:
                    pan += err_x * GIMBAL_GAIN
                if abs(err_y) > GIMBAL_DEADZONE:
                    tilt -= err_y * GIMBAL_GAIN
                pan = max(PAN_MIN, min(PAN_MAX, pan))
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
                self.rover.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1), "SPD": 500, "ACC": 200})
                if abs(err_x) > CENTERING_TOL:
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    time.sleep(1.0 / self.config.follow_loop_hz)
                    continue
                bw_error = target_bw - bw
                if abs(bw_error) < BW_DEADZONE:
                    speed = 0.0
                elif bw_error > 0:
                    speed = bw_error * FWD_GAIN
                else:
                    speed = bw_error * REV_GAIN
                steer = max(-MAX_STEER, min(MAX_STEER, pan * STEER_GAIN))
                left = speed + steer
                right = speed - steer
                if speed > 0:
                    guard = self.depth_guard.assess(
                        self.camera.get_depth_map(),
                        steering_angle_deg=pan,
                    )
                    if not guard["safe"]:
                        self.events.publish(
                            "follow",
                            f"Depth guard stop at {guard.get('clearance_m', 0):.2f}m",
                        )
                        self.rover.send({"T": 1, "L": 0, "R": 0})
                        self._depth_recover(guard)
                        time.sleep(0.2)
                        continue
                if self.imu is not None:
                    try:
                        state, magnitude = self.imu.check_tilt()
                    except Exception:
                        state, magnitude = "ok", 0.0
                    if state == "stop":
                        self.events.publish("follow", f"IMU stop during follow ({magnitude:.2f})")
                        self.rover.stop()
                        return self._result("blocked", time.time() - start, bw_sum, bw_count)
                self.rover.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})
                time.sleep(1.0 / self.config.follow_loop_hz)
        finally:
            self.camera.set_follow_mode(False)
            self.rover.send({"T": 1, "L": 0, "R": 0})
            self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
        return self._result("completed", time.time() - start, bw_sum, bw_count)

    def _depth_recover(self, guard: dict):
        heading = float(guard.get("recommended_heading_deg") or 0.0)
        if abs(heading) >= 8:
            turn_time = min(0.7, abs(heading) / 180.0)
            if heading > 0:
                self.rover.send({"T": 1, "L": 0.25, "R": -0.25})
            else:
                self.rover.send({"T": 1, "L": -0.25, "R": 0.25})
            time.sleep(turn_time)
            self.rover.stop()
        else:
            self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
            time.sleep(0.4)
            self.rover.stop()

    def _callout(self, target: str):
        text = None
        if self.llm is not None:
            try:
                text = self.llm.complete(
                    prompt=follow_callout_prompt(target),
                    system="Reply with only one short sentence.",
                    max_tokens=40,
                ).strip()
            except Exception:
                text = None
        if not text:
            text = random.choice(CALLOUT_LINES)
        self.events.publish("follow", f"Callout: {text}")
        if self.speak_fn is not None and self.flags is not None and self.flags.tts_enabled:
            try:
                self.speak_fn(text)
            except Exception:
                pass

    def _spin_search(self, labels):
        spin_speed = 0.35
        for _ in range(12):
            if self._cancelled:
                return None
            self.rover.send({"T": 1, "L": spin_speed, "R": -spin_speed})
            time.sleep(0.3)
            self.rover.stop()
            time.sleep(0.15)
            dets, _, _ = self.camera.get_detections()
            match = self._select_target(dets, labels, 0.5, 0.5)
            if match and match.get("bw", 0.0) >= MIN_BW:
                return match
        return None

    def _target_labels(self, target: str):
        value = target.lower().strip()
        if value in ("person", "me", "human"):
            return ("person", "legs", "human")
        return (value,)

    def _select_target(self, detections, labels, last_cx, last_cy):
        candidates = [
            det for det in detections
            if det.get("name", "").lower() in labels and det.get("bw", 0.0) >= MIN_BW
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda det: ((det["cx"] - last_cx) ** 2 + (det["cy"] - last_cy) ** 2, -det.get("conf", 0.0)),
        )

    def _result(self, status: str, duration: float, bw_sum: float, bw_count: int):
        avg_bw = round(bw_sum / bw_count, 3) if bw_count else 0.0
        return {
            "status": status,
            "duration_tracked": round(duration, 1),
            "avg_bw": avg_bw,
        }
