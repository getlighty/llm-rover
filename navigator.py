#!/usr/bin/env python3
"""
Shoot-and-Go LLM Navigator

Optimistic navigation: estimate driveable distance via depth map, drive 80%
blind with YOLO safety only, then stop and ask the LLM what to do next.
Uses subtask reasoning when target isn't directly visible.
"""

import os
import math
import time
import json
import threading
from collections import deque

import cv2
import numpy as np

# ── Config Persistence ───────────────────────────────────────────────────
NAV_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "nav_config.json")

def _load_nav_config():
    try:
        with open(NAV_CONFIG_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_nav_config(cfg):
    with open(NAV_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ── Constants ────────────────────────────────────────────────────────────
TURN_SPEED = 0.35           # wheel speed during body rotation (m/s)
TURN_RATE_DPS = 200.0       # degrees/sec at TURN_SPEED (calibrate via web UI)

# Load calibrated value if available
_cfg = _load_nav_config()
if "turn_rate_dps" in _cfg:
    TURN_RATE_DPS = float(_cfg["turn_rate_dps"])
    print(f"[nav] Loaded calibrated TURN_RATE_DPS={TURN_RATE_DPS:.1f}")

DRIVE_SPEED = 0.15          # forward driving speed (m/s)
MAX_STEER = 0.05            # max steering differential
STEER_GAIN = 0.3            # P-control gain (steering = err_x * gain)

ARRIVE_BW = 0.40            # bbox width fraction → arrival
ARRIVE_FRAMES = 5           # consecutive frames to confirm arrival
LOST_FRAMES = 15            # frames without target before giving up (~3s at 5Hz)
DRIVE_LOOP_HZ = 5.0         # YOLO P-control loop rate

GIMBAL_SETTLE_S = 0.4       # wait after gimbal move
GIMBAL_SPD = 300
GIMBAL_ACC = 20

OBSERVATION_MAX_AGE_S = 60
OBSERVATION_MAX_TURNS = 2
MAX_OBSERVATIONS = 20
MAX_WAYPOINTS = 30          # safety limit on shoot-and-go iterations
MAX_STUCK_EVENTS = 3
MAX_SUBTASK_DEPTH = 3
SUBTASK_WAYPOINT_BUDGET = 10  # pop subtask if no progress after this many waypoints

DEFAULT_CLEAR_DIST = 0.3    # meters when depth unavailable
MAX_CLEAR_DIST = 2.0        # cap on estimated clear distance
DRIVE_FRACTION = 0.80       # drive 80% of estimated clear distance

FRAME_DIFF_THRESH = 0.20
BLIND_DRIVE_CHECK_HZ = 7.0  # obstacle check rate during blind drive
BLIND_STUCK_FRAMES = 10     # ~1.5s of unchanged frames = stuck

SCAN_POSITIONS = [
    (0, 0), (-65, 0), (65, 0), (-130, 0), (130, 0),
    (0, -15), (-65, -15), (65, -15),
]

# ── Utilities ────────────────────────────────────────────────────────────

def _frame_changed(jpeg_a, jpeg_b, threshold=FRAME_DIFF_THRESH):
    """Quick check if two frames differ significantly (80x60 grayscale)."""
    if jpeg_a is None or jpeg_b is None:
        return True
    try:
        a = cv2.imdecode(np.frombuffer(jpeg_a, np.uint8), cv2.IMREAD_GRAYSCALE)
        b = cv2.imdecode(np.frombuffer(jpeg_b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return True
        a = cv2.resize(a, (80, 60))
        b = cv2.resize(b, (80, 60))
        diff = cv2.absdiff(a, b)
        pct = float(diff.sum()) / (255.0 * 80 * 60)
        return pct > threshold
    except Exception:
        return True


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Navigator ────────────────────────────────────────────────────────────

class Navigator:
    """Shoot-and-go navigation: drive optimistically, check with LLM at stops."""

    def __init__(self, rover, detector, tracker, llm_vision_fn, parse_fn,
                 pose, voice_fn=None, emergency_event=None):
        self.rover = rover
        self.detector = detector
        self.tracker = tracker
        self.llm_vision = llm_vision_fn
        self.parse = parse_fn
        self.pose = pose
        self.voice_fn = voice_fn
        self.emergency = emergency_event
        self.observations = deque(maxlen=MAX_OBSERVATIONS)
        self._body_turns_since_start = 0
        self._subtask_stack = []

    # ── Public API ───────────────────────────────────────────────────────

    def navigate(self, target):
        """Shoot-and-go navigation to target.  Returns True if arrived."""
        self._body_turns_since_start = 0
        self._subtask_stack = []
        stuck_count = 0
        subtask_wp_count = 0  # waypoints since last subtask change

        for wp in range(MAX_WAYPOINTS):
            if self._aborted():
                return False

            effective = self._current_subtask() or target
            print(f"[nav] Waypoint {wp}: goal='{effective}'"
                  + (f" (subtask of '{target}')" if effective != target else ""))

            # 1. Center gimbal, settle
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)

            # 2. Gather sensor data
            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                time.sleep(0.5)
                continue

            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            dets, yolo_summary, det_age = self._get_detections()

            # 3. YOLO fast check — target visible? → P-control approach
            if self.detector and det_age < 1.0:
                target_det = self.detector.find(target, dets)
                if target_det and target_det["bw"] > 0.05:
                    print(f"[nav] YOLO sees '{target}' "
                          f"(bw={target_det['bw']:.2f}), approaching")
                    self._subtask_stack.clear()
                    result = self._drive_to_target(target)
                    if result == "arrived":
                        print(f"[nav] Arrived at '{target}'")
                        return True
                    if result == "stuck":
                        stuck_count += 1
                        if stuck_count >= MAX_STUCK_EVENTS:
                            self._say("I keep getting stuck")
                            return False
                        self._recover_stuck(target)
                    continue

            # 4. Estimate clear distance
            clear_dist = self._estimate_clear_distance(depth_map, dets)
            dist_str = f"{clear_dist:.1f}m" if clear_dist else "unknown"

            # 5. LLM assessment
            assessment = self._llm_assess_waypoint(
                target, self._current_subtask(), jpeg, yolo_summary, clear_dist)

            if assessment is None:
                # LLM failed — drive forward conservatively
                print(f"[nav] LLM unavailable, driving {DEFAULT_CLEAR_DIST}m")
                self._execute_blind_drive(DEFAULT_CLEAR_DIST, stuck_count)
                continue

            scene = assessment.get("scene", "?")
            action = assessment.get("action", "drive_forward")
            print(f"[nav] LLM: '{scene}' → {action}")

            # 6. Subtask management
            if assessment.get("subtask_achieved"):
                self._pop_subtask()
                subtask_wp_count = 0

            if action == "subtask" and assessment.get("subtask"):
                if len(self._subtask_stack) < MAX_SUBTASK_DEPTH:
                    self._push_subtask(
                        assessment["subtask"],
                        assessment.get("subtask_reason", ""))
                    subtask_wp_count = 0

            # Subtask budget: auto-pop if stuck in a subtask too long
            subtask_wp_count += 1
            if (self._current_subtask() and
                    subtask_wp_count > SUBTASK_WAYPOINT_BUDGET):
                old = self._pop_subtask()
                print(f"[nav] Subtask '{old}' budget exhausted, popping")
                subtask_wp_count = 0

            # 7. Execute action
            if action == "approach_target":
                result = self._drive_to_target(target)
                if result == "arrived":
                    print(f"[nav] Arrived at '{target}'")
                    return True
                if result == "stuck":
                    stuck_count += 1
                    if stuck_count >= MAX_STUCK_EVENTS:
                        self._say("I keep getting stuck")
                        return False
                    self._recover_stuck(target)

            elif action in ("turn_left", "turn_right"):
                degrees = assessment.get("turn_degrees", 45)
                if action == "turn_left":
                    degrees = -abs(degrees)
                else:
                    degrees = abs(degrees)
                self._spin_body(degrees)

            elif action in ("drive_forward", "subtask"):
                drive_dist = DEFAULT_CLEAR_DIST
                if clear_dist and clear_dist > 0.2:
                    drive_dist = min(clear_dist * DRIVE_FRACTION, MAX_CLEAR_DIST)
                drive_dist = max(drive_dist, 0.15)
                result = self._execute_blind_drive(drive_dist, stuck_count)
                if result == "stuck_abort":
                    return False

            else:
                # Unknown action, drive forward conservatively
                self._execute_blind_drive(DEFAULT_CLEAR_DIST, stuck_count)

        self._say(f"Could not reach {target}")
        return False

    def _execute_blind_drive(self, distance, stuck_count):
        """Helper: blind drive + handle stuck.  Returns "ok" or "stuck_abort"."""
        print(f"[nav] Driving forward {distance:.1f}m")
        result = self._blind_drive(distance)
        if result == "stuck":
            stuck_count += 1
            if stuck_count >= MAX_STUCK_EVENTS:
                self._say("I keep getting stuck")
                return "stuck_abort"
            self._recover_stuck("")
        return "ok"

    def search(self, target):
        """Search only (no drive).  Returns direction dict or None."""
        return self._search(target)

    # ── Depth Estimation ─────────────────────────────────────────────────

    def _estimate_clear_distance(self, depth_map, detections):
        """Estimate driveable distance (meters) from depth map + YOLO.
        Returns float meters, or None if unavailable."""
        # Check YOLO detections in driving corridor first (calibrated distances)
        if detections:
            corridor_dets = [d for d in detections
                             if 0.25 < d["cx"] < 0.75
                             and d.get("dist_m") is not None
                             and d["dist_m"] > 0.1]
            if corridor_dets:
                nearest = min(corridor_dets, key=lambda d: d["dist_m"])
                return nearest["dist_m"]

        if depth_map is None:
            return None

        h, w = depth_map.shape[:2]
        # Center corridor: middle 30% horizontal, bottom 60% vertical
        x0, x1 = int(w * 0.35), int(w * 0.65)
        y0, y1 = int(h * 0.40), h
        corridor = depth_map[y0:y1, x0:x1]
        if corridor.size == 0:
            return None

        # 5th percentile = conservative nearest point in corridor
        near = float(np.percentile(corridor, 5))
        far = float(np.percentile(corridor, 95))
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())

        if d_max - d_min < 0.01:
            return 1.0  # uniform depth = likely open space

        # Map relative depth to [0.3m, 2.0m]
        relative = (near - d_min) / (d_max - d_min + 1e-6)
        dist_m = 0.3 + relative * (MAX_CLEAR_DIST - 0.3)
        return max(0.3, dist_m)

    # ── Blind Drive ──────────────────────────────────────────────────────

    def _blind_drive(self, distance_m):
        """Drive forward with YOLO safety only (no LLM).
        Returns: "ok", "obstacle", or "stuck"."""
        drive_time = distance_m / DRIVE_SPEED
        t_start = time.time()
        check_interval = 1.0 / BLIND_DRIVE_CHECK_HZ
        prev_jpeg = None
        same_count = 0

        self.rover.send({"T": 1, "L": DRIVE_SPEED, "R": DRIVE_SPEED})

        try:
            while time.time() - t_start < drive_time:
                if self._aborted():
                    return "obstacle"

                time.sleep(check_interval)

                # YOLO obstacle check (use cached detections from camera loop)
                dets, _, age = self._get_detections()
                if age < 1.0 and dets:
                    for d in dets:
                        in_path = 0.20 < d["cx"] < 0.80
                        close = (d["bw"] > 0.30 or
                                 d.get("dist_m", 999) < 0.40)
                        if in_path and close:
                            self._stop_wheels()
                            print(f"[nav] Blind drive stopped: "
                                  f"'{d['name']}' (bw={d['bw']:.2f})")
                            return "obstacle"

                # Frame-change stuck detection
                jpeg = self.tracker.get_jpeg()
                if jpeg and prev_jpeg:
                    if not _frame_changed(jpeg, prev_jpeg, threshold=0.08):
                        same_count += 1
                        if same_count >= BLIND_STUCK_FRAMES:
                            self._stop_wheels()
                            print("[nav] Blind drive stuck (no visual change)")
                            return "stuck"
                    else:
                        same_count = 0
                prev_jpeg = jpeg

            self._stop_wheels()
            return "ok"
        finally:
            self._stop_wheels()

    # ── LLM Waypoint Assessment ──────────────────────────────────────────

    def _llm_assess_waypoint(self, target, subtask, jpeg, yolo_summary,
                              clear_dist):
        """LLM assessment at each stop.  Returns action dict or None."""
        subtask_ctx = ""
        if subtask and subtask != target:
            subtask_ctx = (
                f"Current subtask: '{subtask}' (ultimate goal: '{target}')\n"
                f"Complete the subtask first, then continue to the goal.\n")

        mem = self._memory_summary()
        depth_info = (f"Estimated clear distance ahead: {clear_dist:.1f}m\n"
                      if clear_dist else "")

        prompt = (
            f"I'm a ground rover navigating to '{target}'.\n"
            f"{subtask_ctx}"
            f"YOLO detections: {yolo_summary}\n"
            f"{depth_info}"
            f"{mem}\n\n"
            f"Look at this image. What should I do next? Reply ONLY JSON:\n"
            f'{{"target_visible": bool, '
            f'"scene": "brief description of what you see", '
            f'"action": "drive_forward"/"turn_left"/"turn_right"/'
            f'"approach_target"/"subtask", '
            f'"turn_degrees": <number if turning>, '
            f'"subtask": "<intermediate goal if action=subtask>", '
            f'"subtask_reason": "<why this helps reach the goal>", '
            f'"subtask_achieved": false}}'
        )
        try:
            raw = self.llm_vision(prompt, jpeg)
            result = self.parse(raw)
            if result:
                # Store observation from LLM scene description
                scene = result.get("scene", "")
                if scene:
                    self._store_observation(0, 0, llm_obs={
                        "objects": [], "obstacles": [],
                        "open_space": "center" if result.get("action") == "drive_forward" else "none"
                    })
            return result
        except Exception as e:
            print(f"[nav] LLM waypoint assess error: {e}")
            return None

    # ── YOLO P-Control Final Approach ────────────────────────────────────

    def _drive_to_target(self, target):
        """YOLO P-control approach (no LLM).  For final approach only.
        Returns: "arrived", "lost", or "stuck"."""
        arrive_count = 0
        lost_count = 0
        prev_jpeg = None
        same_count = 0
        loop_period = 1.0 / DRIVE_LOOP_HZ

        print(f"[nav] P-control approach to '{target}'")

        try:
            while not self._aborted():
                t0 = time.time()

                jpeg = self.tracker.get_jpeg()
                if not jpeg:
                    time.sleep(loop_period)
                    continue

                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                dets = self.detector.detect(frame) if self.detector else []
                target_det = (self.detector.find(target, dets)
                              if self.detector else None)

                if target_det:
                    lost_count = 0
                    cx = target_det["cx"]
                    bw = target_det["bw"]

                    # Arrival check
                    if bw >= ARRIVE_BW:
                        arrive_count += 1
                        if arrive_count >= ARRIVE_FRAMES:
                            self._stop_wheels()
                            return "arrived"
                    else:
                        arrive_count = 0

                    # Obstacle check
                    blocked = False
                    for d in dets:
                        if d is target_det:
                            continue
                        in_path = 0.15 < d["cx"] < 0.85
                        close = (d["bw"] > 0.30 or
                                 d.get("dist_m", 999) < 0.40)
                        if in_path and close:
                            self._stop_wheels()
                            blocked = True
                            break

                    if not blocked:
                        err_x = cx - 0.5
                        steer = _clamp(err_x * STEER_GAIN, -MAX_STEER,
                                       MAX_STEER)
                        L = DRIVE_SPEED + steer
                        R = DRIVE_SPEED - steer
                        self.rover.send({"T": 1, "L": round(L, 3),
                                         "R": round(R, 3)})
                else:
                    lost_count += 1
                    if lost_count > LOST_FRAMES:
                        self._stop_wheels()
                        return "lost"

                # Stuck detection (frame comparison)
                if prev_jpeg:
                    if not _frame_changed(jpeg, prev_jpeg, threshold=0.08):
                        same_count += 1
                        if same_count >= 15:  # ~3s at 5Hz
                            self._stop_wheels()
                            return "stuck"
                    else:
                        same_count = 0
                prev_jpeg = jpeg

                elapsed = time.time() - t0
                if elapsed < loop_period:
                    time.sleep(loop_period - elapsed)

        finally:
            self._stop_wheels()

        return "lost"

    # ── Search (gimbal scan) ─────────────────────────────────────────────

    def _search(self, target):
        """Gimbal scan to find target.  Returns direction dict or None."""
        if hasattr(self.tracker, 'pause'):
            self.tracker.pause(300)

        # Quick YOLO check on current frame
        jpeg = self.tracker.get_jpeg()
        if jpeg and self.detector:
            dets = self.detector.detect(
                cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
            hit = self.detector.find(target, dets)
            if hit:
                print(f"[nav] YOLO found '{target}' immediately "
                      f"(cx={hit['cx']:.2f})")
                return {"gimbal_pan": self.pose.cam_pan,
                        "source": "yolo", "det": hit}

        # Scan front hemisphere
        result = self._scan_positions(target, SCAN_POSITIONS)
        if result:
            return result

        # Rotate 180° and scan rear
        print("[nav] Front scan exhausted, rotating 180°")
        self._spin_body(180)
        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)

        result = self._scan_positions(target, SCAN_POSITIONS)
        if result:
            return result

        print(f"[nav] Could not find '{target}' in 360° scan")
        return None

    def _scan_positions(self, target, positions):
        """Scan gimbal positions looking for target."""
        for pan, tilt in positions:
            if self._aborted():
                return None

            self._move_gimbal(pan, tilt)
            time.sleep(GIMBAL_SETTLE_S)

            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                continue

            if self.detector:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                dets = self.detector.detect(frame)
                hit = self.detector.find(target, dets)
                if hit:
                    print(f"[nav] YOLO found '{target}' at pan={pan}° "
                          f"(cx={hit['cx']:.2f}, bw={hit['bw']:.2f})")
                    self._store_observation(pan, tilt, dets=dets)
                    return {"gimbal_pan": pan, "source": "yolo", "det": hit}
                if dets:
                    self._store_observation(pan, tilt, dets=dets)

            obs = self._llm_observe(target, jpeg, pan, tilt)
            if obs and obs.get("found"):
                print(f"[nav] LLM found '{target}' at pan={pan}°")
                return {"gimbal_pan": pan, "source": "llm", "obs": obs}

        return None

    def _llm_observe(self, target, jpeg, pan, tilt):
        """Ask LLM what it sees at this gimbal position."""
        mem = self._memory_summary()
        prompt = (
            f"I'm a ground rover searching for: {target}.\n"
            f"Gimbal: pan={pan}°, tilt={tilt}°.\n"
            f"{mem}\n\n"
            f"Look at this image. Reply ONLY JSON:\n"
            f'{{"found": bool, "objects": ["list","of","visible","objects"], '
            f'"obstacles": ["floor-blocking objects"], '
            f'"open_space": "left"/"center"/"right"/"none"}}'
        )
        try:
            raw = self.llm_vision(prompt, jpeg)
            result = self.parse(raw)
            if result:
                self._store_observation(pan, tilt, llm_obs=result)
            return result
        except Exception as e:
            print(f"[nav] LLM observe error: {e}")
            return None

    # ── Stuck Recovery ───────────────────────────────────────────────────

    def _recover_stuck(self, target):
        """Spin first, reverse only if object very close."""
        print("[nav] Recovering from stuck...")

        jpeg = self.tracker.get_jpeg()
        very_close = False
        if jpeg and self.detector:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                 cv2.IMREAD_COLOR)
            dets = self.detector.detect(frame)
            for d in dets:
                if d["bw"] > 0.50 or d.get("dist_m", 999) < 0.20:
                    very_close = True
                    print(f"[nav] Object '{d['name']}' very close "
                          f"(bw={d['bw']:.2f})")
                    break

        if very_close:
            print("[nav] Reversing 0.5s (object very close)")
            self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
            time.sleep(0.5)
            self._stop_wheels()
            time.sleep(0.2)
            self._spin_body(90)
            return

        for _ in range(4):  # up to 180°
            if self._aborted():
                return
            self._spin_body(45)
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)

            jpeg = self.tracker.get_jpeg()
            if jpeg and self._is_path_clear(jpeg):
                print("[nav] Found clear path after spin")
                return

        print("[nav] Stuck after 180° spin, reversing 1s")
        self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
        time.sleep(1.0)
        self._stop_wheels()
        time.sleep(0.2)
        self._spin_body(90)

    def _is_path_clear(self, jpeg):
        """Check if path ahead is clear using YOLO (fast, no LLM)."""
        if not self.detector:
            return True
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        dets = self.detector.detect(frame)
        for d in dets:
            in_path = 0.20 < d["cx"] < 0.80
            close = d["bw"] > 0.30 or d.get("dist_m", 999) < 0.40
            if in_path and close:
                return False
        return True

    # ── Subtask Stack ────────────────────────────────────────────────────

    def _current_subtask(self):
        return self._subtask_stack[-1] if self._subtask_stack else None

    def _push_subtask(self, goal, reason=""):
        self._subtask_stack.append(goal)
        print(f"[nav] Subtask pushed: '{goal}' ({reason})")
        self._say(f"Looking for {goal} first")

    def _pop_subtask(self):
        if self._subtask_stack:
            done = self._subtask_stack.pop()
            print(f"[nav] Subtask achieved: '{done}'")
            if self._current_subtask():
                print(f"[nav] Back to: '{self._current_subtask()}'")
            return done
        return None

    # ── Primitives ───────────────────────────────────────────────────────

    def _get_detections(self):
        """Get cached YOLO detections from camera.
        Returns (list, summary_str, age_secs)."""
        if hasattr(self.tracker, 'get_detections'):
            return self.tracker.get_detections()
        return ([], "nothing", 999)

    def _spin_body(self, degrees):
        """Timed body rotation.  Positive = CW (right)."""
        if abs(degrees) < 3:
            return
        sign = 1 if degrees > 0 else -1
        self.rover.send({"T": 1,
                         "L": TURN_SPEED * sign,
                         "R": -TURN_SPEED * sign})
        rotation_time = abs(degrees) / TURN_RATE_DPS
        time.sleep(rotation_time)
        self._stop_wheels()
        self.pose.after_body_turn(degrees)
        print(f"[nav] Spun {degrees:+.0f}° (timed {rotation_time:.2f}s)")
        self._body_turns_since_start += 1
        self._invalidate_old_observations()
        time.sleep(0.2)

    def _move_gimbal(self, pan, tilt):
        self.rover.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1),
                         "SPD": GIMBAL_SPD, "ACC": GIMBAL_ACC})
        self.pose.cam_pan = pan
        self.pose.cam_tilt = tilt

    def _stop_wheels(self):
        self.rover.send({"T": 1, "L": 0, "R": 0})

    def _say(self, msg):
        print(f"[nav] {msg}")
        if self.voice_fn:
            try:
                self.voice_fn(msg)
            except Exception:
                pass

    def _aborted(self):
        return self.emergency and self.emergency.is_set()

    # ── Scene Memory ─────────────────────────────────────────────────────

    def _store_observation(self, pan, tilt, dets=None, llm_obs=None):
        objects = []
        if dets:
            for d in dets:
                objects.append({
                    "name": d["name"],
                    "cx": round(d["cx"], 2),
                    "bw": round(d["bw"], 2),
                    "dist_m": round(d.get("dist_m", 0), 2),
                    "source": "yolo",
                })
        if llm_obs:
            for obj_name in llm_obs.get("objects", []):
                objects.append({"name": obj_name, "source": "llm"})

        obs = {
            "time": time.time(),
            "gimbal_pan": pan,
            "gimbal_tilt": tilt,
            "body_turns": self._body_turns_since_start,
            "objects": objects,
            "obstacles": llm_obs.get("obstacles", []) if llm_obs else [],
            "open_directions": [],
        }
        if llm_obs:
            open_space = llm_obs.get("open_space", "")
            if open_space and open_space != "none":
                obs["open_directions"] = [open_space]
        self.observations.append(obs)

    def _invalidate_old_observations(self):
        now = time.time()
        valid = deque(maxlen=MAX_OBSERVATIONS)
        for obs in self.observations:
            age = now - obs["time"]
            turns = self._body_turns_since_start - obs["body_turns"]
            if age < OBSERVATION_MAX_AGE_S and turns < OBSERVATION_MAX_TURNS:
                valid.append(obs)
        self.observations = valid

    def _memory_summary(self):
        self._invalidate_old_observations()
        if not self.observations:
            return ""
        lines = ["Recent observations:"]
        now = time.time()
        for obs in list(self.observations)[-5:]:
            age = int(now - obs["time"])
            age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
            obj_names = [o["name"] for o in obs["objects"]]
            pan = obs["gimbal_pan"]
            lines.append(f"  pan={pan}°: {', '.join(obj_names) or 'nothing'} "
                         f"({age_str})")
            if obs["obstacles"]:
                lines.append(f"    obstacles: {', '.join(obs['obstacles'])}")
            if obs["open_directions"]:
                lines.append(f"    open: {', '.join(obs['open_directions'])}")
        return "\n".join(lines)
