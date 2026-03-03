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
NAV_SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "nav_snapshots")
SNAPSHOT_INTERVAL_S = 3.0  # save a snapshot every N seconds during navigation

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

# Load calibrated values if available
GIMBAL_PAN_OFFSET = 0.0     # degrees to add to all gimbal pan commands
_cfg = _load_nav_config()
if "turn_rate_dps" in _cfg:
    TURN_RATE_DPS = float(_cfg["turn_rate_dps"])
    print(f"[nav] Loaded calibrated TURN_RATE_DPS={TURN_RATE_DPS:.1f}")
if "gimbal_pan_offset" in _cfg:
    GIMBAL_PAN_OFFSET = float(_cfg["gimbal_pan_offset"])
    print(f"[nav] Loaded gimbal pan offset={GIMBAL_PAN_OFFSET:+.1f}°")

# Clear snapshot dir on import (= service restart)
import shutil
if os.path.exists(NAV_SNAPSHOTS_DIR):
    shutil.rmtree(NAV_SNAPSHOTS_DIR)
os.makedirs(NAV_SNAPSHOTS_DIR, exist_ok=True)
print(f"[nav] Snapshot dir: {NAV_SNAPSHOTS_DIR}")

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
BLIND_DRIVE_CHECK_HZ = 5.0  # obstacle check rate during blind drive
ENCODER_STUCK_RATIO = 0.30  # if avg encoder speed < 30% of commanded → stuck
ENCODER_STUCK_READS = 5     # consecutive slow reads = stuck (~1s at 5Hz)

SCAN_POSITIONS = [
    (0, 0), (-65, 0), (65, 0), (-130, 0), (130, 0),
    (0, -15), (-65, -15), (65, -15),
]

# Module-level ref for web_ui access (set by rover_brain_llm.py)
_exploration_grid = None

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
                 pose, voice_fn=None, emergency_event=None,
                 exploration_grid=None):
        self.rover = rover
        self.detector = detector
        self.tracker = tracker
        self.llm_vision = llm_vision_fn
        self.parse = parse_fn
        self.pose = pose
        self.voice_fn = voice_fn
        self.emergency = emergency_event
        self.exploration = exploration_grid
        self.observations = deque(maxlen=MAX_OBSERVATIONS)
        self._body_turns_since_start = 0
        self._subtask_stack = []
        self._last_snapshot_time = 0
        self._snapshot_idx = 0
        self._nav_target = ""  # current navigation target
        self._last_drive_angle = 0  # last drive direction (degrees, neg=left)

    # ── Public API ───────────────────────────────────────────────────────

    def navigate(self, target, plan_context="", step_budget=None):
        """Shoot-and-go navigation to target.  Returns True if arrived.

        Args:
            plan_context: strategic context from orchestrator (injected into
                          every waypoint LLM call). Skips panoramic scan.
            step_budget: override MAX_WAYPOINTS for this step.
        """
        self._nav_target = target
        self._plan_context = plan_context
        self._body_turns_since_start = 0
        self._subtask_stack = []
        self._last_result = None
        self._wp_used = 0
        self._last_scene = ""
        self._last_yolo = ""
        stuck_count = 0
        consecutive_turns = 0  # track turn-without-driving loops
        total_rotation = 0     # cumulative degrees turned (detect circling)
        subtask_wp_count = 0  # waypoints since last subtask change
        max_wp = step_budget if step_budget else MAX_WAYPOINTS

        # Panoramic scan — skip when orchestrator provides plan context
        if not plan_context:
            self._scan_summary = self._panoramic_scan()
        else:
            self._scan_summary = ""  # plan context replaces the scan
            print(f"[nav] Skipping panoramic scan (plan context provided)")

        for wp in range(max_wp):
            if self._aborted():
                self._store_step_result(False, "aborted", wp)
                return False

            self._wp_used = wp + 1
            self._last_drive_avoidances = 0
            effective = self._current_subtask() or target
            print(f"[nav] Waypoint {wp}: goal='{effective}'"
                  + (f" (subtask of '{target}')" if effective != target else ""))

            # 1. Align body to gimbal heading
            self._align_body()
            time.sleep(GIMBAL_SETTLE_S)

            # 2. Gather sensor data (camera now facing body's forward)
            self._maybe_snapshot()
            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                time.sleep(0.5)
                continue

            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            dets, yolo_summary, det_age = self._get_detections()
            self._last_yolo = yolo_summary

            # Update exploration grid with depth observation
            if self.exploration and depth_map is not None:
                body_yaw = (self.pose.body_yaw
                            if hasattr(self.pose, 'body_yaw') else 0)
                cam_pan = self.pose.cam_pan
                self.exploration.update_from_depth(depth_map, body_yaw,
                                                   cam_pan)

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
                        self._store_step_result(True, "arrived", wp + 1)
                        return True
                    if result == "stuck":
                        stuck_count += 1
                        if stuck_count >= MAX_STUCK_EVENTS:
                            self._say("I keep getting stuck")
                            self._store_step_result(False, "stuck", wp + 1)
                            return False
                        self._recover_stuck(target)
                    continue

            # 4. Estimate clear distance
            clear_dist = self._estimate_clear_distance(depth_map, dets)
            dist_str = f"{clear_dist:.1f}m" if clear_dist else "unknown"

            # 5. LLM assessment
            assessment = self._llm_assess_waypoint(
                target, self._current_subtask(), jpeg, yolo_summary, clear_dist,
                consecutive_turns, total_rotation)

            if assessment is None:
                # LLM failed — drive forward conservatively
                print(f"[nav] LLM unavailable, driving {DEFAULT_CLEAR_DIST}m")
                self._execute_blind_drive(DEFAULT_CLEAR_DIST, stuck_count)
                continue

            scene = assessment.get("scene", "?")
            self._last_scene = scene
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
            # Force drive after too many consecutive turns (break spin loops)
            if consecutive_turns >= 3 and action in ("turn_left", "turn_right"):
                print(f"[nav] Forced drive: {consecutive_turns} turns without progress")
                action = "drive_forward"

            if action == "arrived":
                # LLM declares arrival (for non-YOLO targets like doorways)
                print(f"[nav] LLM declares arrived at '{target}'")
                self._store_step_result(True, "arrived", wp + 1)
                return True

            elif action == "approach_target":
                consecutive_turns = 0
                result = self._drive_to_target(target)
                if result == "arrived":
                    print(f"[nav] Arrived at '{target}'")
                    self._store_step_result(True, "arrived", wp + 1)
                    return True
                if result == "stuck":
                    stuck_count += 1
                    if stuck_count >= MAX_STUCK_EVENTS:
                        self._say("I keep getting stuck")
                        self._store_step_result(False, "stuck", wp + 1)
                        return False
                    self._recover_stuck(target)

            elif action in ("turn_left", "turn_right"):
                consecutive_turns += 1
                degrees = assessment.get("turn_degrees", 45)
                if action == "turn_left":
                    degrees = -abs(degrees)
                else:
                    degrees = abs(degrees)
                total_rotation += degrees
                # Gimbal leads, body follows: move gimbal first, then
                # _align_body() at next iteration rotates body to match
                new_pan = _clamp(self.pose.cam_pan + degrees, -180, 180)
                self._move_gimbal(new_pan, 0)
                print(f"[nav] Gimbal turned to {new_pan:.0f}° "
                      f"(total rotation: {total_rotation:+.0f}°)")

            elif action in ("drive_forward", "subtask"):
                consecutive_turns = 0
                drive_dist = DEFAULT_CLEAR_DIST
                if clear_dist and clear_dist > 0.2:
                    drive_dist = min(clear_dist * DRIVE_FRACTION, MAX_CLEAR_DIST)
                drive_dist = max(drive_dist, 0.15)
                drive_angle = assessment.get("drive_angle", 0) if assessment else 0
                drive_angle = _clamp(drive_angle, -30, 30)
                result = self._execute_blind_drive(
                    drive_dist, stuck_count, drive_angle)
                if result == "stuck_abort":
                    self._store_step_result(False, "stuck", wp + 1)
                    return False

            else:
                consecutive_turns = 0
                self._execute_blind_drive(DEFAULT_CLEAR_DIST, stuck_count)

        self._say(f"Could not reach {target}")
        self._store_step_result(False, "budget", max_wp)
        return False

    def _execute_blind_drive(self, distance, stuck_count, drive_angle=0):
        """Helper: blind drive + handle stuck.  Returns "ok" or "stuck_abort"."""
        self._last_drive_angle = drive_angle
        angle_str = f" at {drive_angle:+.0f}°" if abs(drive_angle) > 3 else ""
        print(f"[nav] Driving forward {distance:.1f}m{angle_str}")
        result = self._blind_drive(distance, drive_angle)

        # Update exploration grid after drive
        if self.exploration and result in ("ok", "obstacle"):
            heading = self.pose.body_yaw if hasattr(self.pose, 'body_yaw') else 0
            drive_heading = heading + drive_angle
            driven = distance if result == "ok" else distance * 0.3
            self.exploration.update_after_drive(driven, drive_heading)

        # LLM verify before entering stuck recovery — YOLO may be
        # hallucinating obstacles in empty space
        if result in ("obstacle", "stuck"):
            if self._llm_verify_obstacle():
                stuck_count += 1
                if stuck_count >= MAX_STUCK_EVENTS:
                    self._say("I keep getting stuck")
                    return "stuck_abort"
                self._recover_stuck("")
            else:
                print("[nav] LLM says path is clear — YOLO ghost, continuing")
                result = "ok"
        return "ok"

    def search(self, target):
        """Search only (no drive).  Returns direction dict or None."""
        return self._search(target)

    def _store_step_result(self, success, reason, waypoints_used):
        """Build and store a StepResult for the orchestrator to read."""
        explore_summary = ""
        if self.exploration:
            body_yaw = (self.pose.body_yaw
                        if hasattr(self.pose, 'body_yaw') else 0)
            explore_summary = self.exploration.summarize_for_llm(body_yaw) or ""
        from orchestrator import StepResult
        self._last_result = StepResult(
            success=success,
            reason=reason,
            waypoints_used=waypoints_used,
            final_scene=getattr(self, '_last_scene', ''),
            final_yolo=getattr(self, '_last_yolo', ''),
            exploration_summary=explore_summary,
        )

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

        # Depth Anything: higher = closer. 95th pctl = nearest obstacle.
        near = float(np.percentile(corridor, 95))
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())

        if d_max - d_min < 0.01:
            return 1.0  # uniform depth = likely open space

        # Invert: high depth = close = small distance
        relative_far = 1.0 - (near - d_min) / (d_max - d_min + 1e-6)
        dist_m = 0.3 + relative_far * (MAX_CLEAR_DIST - 0.3)
        return max(0.3, dist_m)

    def _depth_obstacle_check(self, depth_map):
        """Check depth map for close obstacles in driving corridor.
        Returns description string if obstacle found, None if clear."""
        h, w = depth_map.shape[:2]
        # Full rover width corridor: center 50% horizontal, bottom 55% vertical
        x0, x1 = int(w * 0.25), int(w * 0.75)
        y0 = int(h * 0.45)
        corridor = depth_map[y0:h, x0:x1]
        if corridor.size == 0:
            return None

        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max - d_min < 0.01:
            return None  # uniform = no obstacles

        # Depth Anything: higher = closer. 99th percentile = nearest object.
        near_val = float(np.percentile(corridor, 99))

        # If nearest point is in top 20% of depth range, something is close
        relative_closeness = (near_val - d_min) / (d_max - d_min + 1e-6)
        if relative_closeness > 0.80:
            return f"depth {relative_closeness:.0%} of range"

        return None

    def _depth_steer_direction(self, depth_map):
        """Determine which side has more open space from depth map.
        Depth Anything: higher value = CLOSER. So lower mean = more open.
        Returns "left", "right", or "straight"."""
        h, w = depth_map.shape[:2]
        y0 = int(h * 0.40)
        left = depth_map[y0:h, :w // 2]
        right = depth_map[y0:h, w // 2:]
        left_mean = float(np.mean(left)) if left.size else 999
        right_mean = float(np.mean(right)) if right.size else 999
        # Lower mean = farther = more open space (steer toward lower)
        if left_mean < right_mean * 0.85:
            return "left"
        elif right_mean < left_mean * 0.85:
            return "right"
        return "straight"

    # ── Blind Drive ──────────────────────────────────────────────────────

    def _blind_drive(self, distance_m, drive_angle=0):
        """Drive forward with depth + YOLO reactive avoidance.
        drive_angle: degrees offset (-30 to +30, negative=left, positive=right).
        Steers around obstacles when possible, stops only when blocked.
        Returns: "ok", "obstacle", or "stuck".
        Sets self._last_drive_avoidances (int) with count of steer events."""
        drive_time = distance_m / DRIVE_SPEED
        t_start = time.time()
        check_interval = 1.0 / BLIND_DRIVE_CHECK_HZ
        enc_stuck_count = 0
        steering = False  # True when actively avoiding an obstacle
        avoidance_count = 0  # how many times we had to steer around something

        # Convert drive_angle to wheel differential
        angle_steer = _clamp(drive_angle / 90.0 * 0.10, -0.10, 0.10)
        base_L = DRIVE_SPEED + angle_steer
        base_R = DRIVE_SPEED - angle_steer
        self.rover.send({"T": 1, "L": round(base_L, 3),
                         "R": round(base_R, 3)})
        if abs(drive_angle) > 3:
            print(f"[nav] Driving at {drive_angle:+.0f}° angle")

        try:
            while time.time() - t_start < drive_time:
                if self._aborted():
                    return "obstacle"

                time.sleep(check_interval)

                obstacle_detected = False
                steer_dir = "straight"

                # DEPTH MAP obstacle check — sees everything (chair legs,
                # toys, door frames) regardless of YOLO class
                depth_map = (self.tracker.get_depth_map()
                             if hasattr(self.tracker, 'get_depth_map')
                             else None)
                if depth_map is not None:
                    obstacle = self._depth_obstacle_check(depth_map)
                    if obstacle:
                        obstacle_detected = True
                        steer_dir = self._depth_steer_direction(depth_map)

                # YOLO obstacle check — cross-validate with depth map
                # If depth says corridor is clear, YOLO ghost → ignore
                depth_clear = (depth_map is not None
                               and not obstacle_detected)
                dets, _, age = self._get_detections()
                if age < 1.0 and dets:
                    for d in dets:
                        # Emergency stop: object dead center AND very close
                        # (only trust if has calibrated distance or huge bbox)
                        dead_center = 0.30 < d["cx"] < 0.70
                        has_real_dist = d.get("dist_m") is not None
                        dangerously_close = (
                            (d["bw"] > 0.60 and has_real_dist) or
                            d.get("dist_m", 999) < 0.15)
                        if dead_center and dangerously_close:
                            self._stop_wheels()
                            print(f"[nav] EMERGENCY stop: '{d['name']}' "
                                  f"(bw={d['bw']:.2f}, cx={d['cx']:.2f})")
                            return "obstacle"
                        # Steer around: anything in wider path that's close
                        in_path = 0.15 < d["cx"] < 0.85
                        close = (d["bw"] > 0.20 or
                                 d.get("dist_m", 999) < 0.50)
                        if in_path and close:
                            # If depth map says corridor is clear, skip
                            # this YOLO detection — likely a ghost
                            if depth_clear and not has_real_dist:
                                continue
                            obstacle_detected = True
                            if d["cx"] < 0.5:
                                steer_dir = "right"
                            else:
                                steer_dir = "left"
                            break

                # Apply steering or resume straight
                if obstacle_detected:
                    AVOID_STEER = 0.13  # sharp avoidance — near-pivot turn
                    if steer_dir == "left":
                        self.rover.send({"T": 1,
                                         "L": DRIVE_SPEED - AVOID_STEER,
                                         "R": DRIVE_SPEED + AVOID_STEER})
                    elif steer_dir == "right":
                        self.rover.send({"T": 1,
                                         "L": DRIVE_SPEED + AVOID_STEER,
                                         "R": DRIVE_SPEED - AVOID_STEER})
                    else:
                        # Obstacle dead center, no clear side → stop
                        self._stop_wheels()
                        print("[nav] Blind drive: obstacle dead center, "
                              "stopping for LLM replan")
                        return "obstacle"
                    if not steering:
                        avoidance_count += 1
                        print(f"[nav] Avoiding obstacle: steer {steer_dir}")
                    steering = True
                elif steering:
                    # Was steering, obstacle cleared → resume drive angle
                    self.rover.send({"T": 1, "L": round(base_L, 3),
                                     "R": round(base_R, 3)})
                    steering = False

                # Periodic snapshot during drive
                self._maybe_snapshot()

                # Encoder-based stuck detection: if encoder speed drops
                # well below commanded speed, we're pushing against something
                if hasattr(self.rover, 'read_imu'):
                    imu = self.rover.read_imu()
                    if imu and "L" in imu and "R" in imu:
                        enc_avg = (abs(float(imu["L"])) +
                                   abs(float(imu["R"]))) / 2.0
                        expected = DRIVE_SPEED
                        if enc_avg < expected * ENCODER_STUCK_RATIO:
                            enc_stuck_count += 1
                            if enc_stuck_count >= ENCODER_STUCK_READS:
                                self._stop_wheels()
                                print(f"[nav] Stuck: encoder avg "
                                      f"{enc_avg:.3f} < {expected * ENCODER_STUCK_RATIO:.3f} "
                                      f"({enc_stuck_count} reads)")
                                return "stuck"
                        else:
                            enc_stuck_count = 0

            self._stop_wheels()
            self._last_drive_avoidances = avoidance_count
            if avoidance_count >= 3:
                print(f"[nav] Path heavily obstructed "
                      f"({avoidance_count} avoidance events)")
                return "obstacle"
            return "ok"
        finally:
            self._stop_wheels()

    # ── LLM Waypoint Assessment ──────────────────────────────────────────

    def _panoramic_scan(self):
        """Gimbal scan with LLM vision at each position.
        Returns text summary of what each direction looks like."""
        scan_dirs = [(-90, "far left"), (-45, "left"), (0, "center"),
                     (45, "right"), (90, "far right")]
        summaries = []
        for pan, label in scan_dirs:
            if self._aborted():
                break
            self._move_gimbal(pan, 0)
            time.sleep(GIMBAL_SETTLE_S)

            # Collect YOLO
            dets, det_summary, age = self._get_detections()
            yolo_part = ""
            if age < 1.0 and dets:
                obj_names = [f"{d['name']}({d['bw']:.0%})" for d in dets[:4]]
                yolo_part = f"YOLO: {', '.join(obj_names)}"
                self._store_observation(pan, 0, dets=dets)

            # LLM vision call for scene understanding
            llm_part = ""
            result = self._llm_call(
                f"Briefly describe what you see (1 sentence). "
                f"Is there a door, exit, or open path in this direction?\n"
                f"Reply ONLY JSON: "
                f'{{"description": "...", "has_exit": bool, "open_space": bool}}')
            if result:
                llm_part = result.get("description", "")
                if result.get("has_exit"):
                    llm_part += " [EXIT VISIBLE]"
                if result.get("open_space"):
                    llm_part += " [OPEN]"

            line = f"  {label} (pan={pan}°): {llm_part or yolo_part or 'unclear'}"
            summaries.append(line)
            print(f"[nav] Scan {label}: {llm_part or yolo_part or 'no data'}")

        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)
        result = "Initial room scan:\n" + "\n".join(summaries)
        print(f"[nav] Panoramic scan complete")
        return result

    def _llm_assess_waypoint(self, target, subtask, jpeg, yolo_summary,
                              clear_dist, consecutive_turns=0,
                              total_rotation=0):
        """LLM assessment at each stop.  Returns action dict or None."""
        subtask_ctx = ""
        if subtask and subtask != target:
            subtask_ctx = (
                f"Current subtask: '{subtask}' (ultimate goal: '{target}')\n"
                f"Complete the subtask first, then continue to the goal.\n")

        turn_warning = ""
        if consecutive_turns >= 2:
            turn_warning = (
                f"WARNING: I've turned {consecutive_turns} times without driving. "
                f"Stop turning and drive forward toward the most open space.\n")
        if abs(total_rotation) >= 270:
            turn_warning += (
                f"CRITICAL: Total rotation is {total_rotation:+.0f}° — "
                f"I'm going in circles! Drive forward or try a completely "
                f"different direction. Consider reversing.\n")

        obstacle_warning = ""
        avoids = getattr(self, '_last_drive_avoidances', 0)
        if avoids >= 2:
            obstacle_warning = (
                f"WARNING: Last drive had {avoids} obstacle avoidance events. "
                f"The path ahead is BLOCKED. Do NOT keep driving forward "
                f"into the same obstacle. Turn to find a clear path, "
                f"or use a subtask to navigate around the obstruction.\n")

        mem = self._memory_summary()
        depth_info = (f"Estimated clear distance ahead: {clear_dist:.1f}m\n"
                      if clear_dist else "")
        plan_ctx = getattr(self, '_plan_context', "")
        has_plan = bool(plan_ctx)
        if plan_ctx:
            plan_ctx = (
                f"== ORCHESTRATOR PLAN (follow this) ==\n"
                f"{plan_ctx}\n"
                f"== END PLAN ==\n"
            )
        scan_ctx = (f"{self._scan_summary}\n"
                    if hasattr(self, '_scan_summary') and self._scan_summary
                    else "")
        explore_ctx = ""
        if self.exploration:
            body_yaw = (self.pose.body_yaw
                        if hasattr(self.pose, 'body_yaw') else 0)
            explore_ctx = self.exploration.summarize_for_llm(body_yaw)
            if explore_ctx:
                explore_ctx = f"Exploration: {explore_ctx}\n"

        if has_plan:
            strategy = (
                f"FOCUS on reaching your CURRENT STEP target. "
                f"The orchestrator handles the big picture — "
                f"you just need to get to '{target}'.\n"
                f"For drive_forward: set drive_angle to aim toward your target "
                f"(negative=left, positive=right, 0=straight).\n")
        else:
            strategy = (
                f"THINK STRATEGICALLY: Where is the door/exit? Plan a route.\n"
                f"Subtasks must be SPATIAL (e.g. 'reach the doorway on the left').\n"
                f"For drive_forward: set drive_angle to aim toward your target "
                f"(negative=left, positive=right, 0=straight). "
                f"You can drive at an angle to follow a path near obstacles.\n")

        prompt = (
            f"I'm a 26cm wide ground rover at floor level. Goal: '{target}'.\n"
            f"{plan_ctx}"
            f"{scan_ctx}"
            f"{subtask_ctx}"
            f"{turn_warning}"
            f"{obstacle_warning}"
            f"YOLO: {yolo_summary}\n"
            f"{depth_info}"
            f"{explore_ctx}"
            f"{mem}\n\n"
            f"{strategy}\n"
            f"Reply ONLY JSON:\n"
            f'{{"target_visible": bool, '
            f'"scene": "what you see + where the exit/door likely is", '
            f'"action": "arrived"/"drive_forward"/"turn_left"/"turn_right"/'
            f'"approach_target"/"subtask", '
            f'"drive_angle": <degrees offset while driving: -30 to +30, 0=straight>, '
            f'"turn_degrees": <number if turning>, '
            f'"subtask": "<specific spatial goal>", '
            f'"subtask_reason": "<why>", '
            f'"subtask_achieved": false}}\n'
            f'Use "arrived" when the target fills most of the frame or '
            f'you are at/inside it (doorway, room, area). '
            f'Use "approach_target" only for YOLO-detectable objects.'
        )
        result = self._llm_call(prompt, jpeg)
        if result:
            scene = result.get("scene", "")
            if scene:
                self._store_observation(0, 0, llm_obs={
                    "objects": [], "obstacles": [],
                    "open_space": "center" if result.get("action") == "drive_forward" else "none"
                })
        return result

    # ── YOLO P-Control Final Approach ────────────────────────────────────

    def _drive_to_target(self, target):
        """YOLO P-control approach (no LLM).  For final approach only.
        Returns: "arrived", "lost", or "stuck"."""
        arrive_count = 0
        lost_count = 0
        enc_stuck_count = 0
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

                # Encoder-based stuck detection
                if hasattr(self.rover, 'read_imu'):
                    imu = self.rover.read_imu()
                    if imu and "L" in imu and "R" in imu:
                        enc_avg = (abs(float(imu["L"])) +
                                   abs(float(imu["R"]))) / 2.0
                        expected = DRIVE_SPEED
                        if enc_avg < expected * ENCODER_STUCK_RATIO:
                            enc_stuck_count += 1
                            if enc_stuck_count >= ENCODER_STUCK_READS:
                                self._stop_wheels()
                                print(f"[nav] P-control stuck: encoder avg "
                                      f"{enc_avg:.3f} < {expected * ENCODER_STUCK_RATIO:.3f}")
                                return "stuck"
                        else:
                            enc_stuck_count = 0

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
        result = self._llm_call(prompt, jpeg)
        if result:
            self._store_observation(pan, tilt, llm_obs=result)
        return result

    # ── Stuck Recovery ───────────────────────────────────────────────────

    def _recover_stuck(self, target):
        """Back off slowly, then gimbal-scan for clear path.
        Leaves gimbal pointing at the clear direction so the next
        waypoint's _align_body() rotates the body to match."""
        print("[nav] Recovering from stuck — backing off...")

        # 1. Reverse slowly to create clearance
        self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
        time.sleep(0.8)
        self._stop_wheels()
        time.sleep(0.2)

        # 2. Gimbal scan for clear path — try opposite to wanted heading first
        wanted = self._last_drive_angle + self.pose.cam_pan
        if wanted <= 0:
            scan_pans = [90, 135, 45, 180, -90]   # prefer right (opposite of left)
        else:
            scan_pans = [-90, -135, -45, -180, 90]  # prefer left (opposite of right)

        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)

        for pan in scan_pans:
            if self._aborted():
                return
            self._move_gimbal(pan, 0)
            time.sleep(GIMBAL_SETTLE_S)
            jpeg = self.tracker.get_jpeg()
            if jpeg and self._is_path_clear(jpeg):
                print(f"[nav] Clear path at gimbal pan={pan}° "
                      f"(opposite to heading {wanted:+.0f}°)")
                # Leave gimbal here — _align_body() will rotate body to match
                return

        # 3. No clear path found — point gimbal opposite to wanted heading
        #    as best guess, _align_body() will rotate body
        fallback = 90 if wanted <= 0 else -90
        print(f"[nav] No clear path found, defaulting gimbal to {fallback}°")
        self._move_gimbal(fallback, 0)

    def _llm_verify_obstacle(self):
        """Quick LLM check: is there actually an obstacle ahead?
        Returns True if obstacle confirmed, False if path looks clear."""
        result = self._llm_call(
            "I stopped because my sensors detected an obstacle ahead. "
            "Look at this image: is there ACTUALLY a physical object "
            "blocking my path on the floor within 0.5m? "
            "Ignore walls/objects that are far away or to the side.\n"
            "Reply ONLY JSON: "
            '{"obstacle": true/false, "what": "brief description"}')
        if result is None:
            return True  # LLM unavailable — assume obstacle
        blocked = result.get("obstacle", True)
        what = result.get("what", "?")
        print(f"[nav] LLM obstacle check: {'YES' if blocked else 'NO'} — {what}")
        return blocked

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

    def _align_body(self):
        """Rotate body to match gimbal heading, bringing gimbal back to 0°.
        After this call, body faces where camera was pointing and gimbal ≈ 0°."""
        pan = self.pose.cam_pan
        if abs(pan) < 5:
            # Already aligned — just ensure gimbal is exactly centered
            self._move_gimbal(0, self.pose.cam_tilt)
            return
        # Spin body by the gimbal pan amount — _spin_body() will
        # counter-rotate the gimbal, bringing it back toward 0°
        print(f"[nav] Aligning body to gimbal (pan={pan:.0f}°)")
        self._spin_body(pan)
        # Ensure gimbal is exactly centered after alignment
        self._move_gimbal(0, 0)

    def _llm_call(self, prompt, jpeg=None):
        """LLM vision call with task context always included.
        Uses annotated JPEG (with YOLO boxes) so LLM sees detections.
        Returns parsed dict or None."""
        if jpeg is None:
            jpeg = self._get_annotated_jpeg()
        task_json = json.dumps({
            "task": self._nav_target,
            "subtask": self._current_subtask(),
            "pose": self.pose.get_pose() if hasattr(self.pose, 'get_pose') else {},
        })
        full_prompt = f"TASK: {task_json}\n\n{prompt}"
        try:
            raw = self.llm_vision(full_prompt, jpeg)
            return self.parse(raw)
        except Exception as e:
            print(f"[nav] LLM call error: {e}")
            return None

    def _get_detections(self):
        """Get cached YOLO detections from camera.
        Returns (list, summary_str, age_secs)."""
        if hasattr(self.tracker, 'get_detections'):
            return self.tracker.get_detections()
        return ([], "nothing", 999)

    def _get_annotated_jpeg(self):
        """Get JPEG with YOLO boxes drawn (for LLM), fallback to raw."""
        if hasattr(self.tracker, 'get_overlay_jpeg'):
            return self.tracker.get_overlay_jpeg()
        return self.tracker.get_jpeg()

    def _spin_body(self, degrees):
        """Timed body rotation.  Positive = CW (right).
        Gimbal counter-rotates to keep camera at the same world heading."""
        if abs(degrees) < 3:
            return
        sign = 1 if degrees > 0 else -1

        # Counter-rotate gimbal: body turns +deg → gimbal moves -deg
        old_pan = self.pose.cam_pan
        new_pan = old_pan - degrees
        new_pan = max(-180, min(180, new_pan))

        # Start both simultaneously: body spin + gimbal counter-rotation
        self.rover.send({"T": 133, "X": round(new_pan, 1),
                         "Y": self.pose.cam_tilt,
                         "SPD": GIMBAL_SPD, "ACC": GIMBAL_ACC})
        self.rover.send({"T": 1,
                         "L": TURN_SPEED * sign,
                         "R": -TURN_SPEED * sign})

        rotation_time = abs(degrees) / TURN_RATE_DPS
        time.sleep(rotation_time)
        self._stop_wheels()

        self.pose.after_body_turn(degrees)
        self.pose.cam_pan = new_pan
        print(f"[nav] Spun {degrees:+.0f}° (timed {rotation_time:.2f}s), "
              f"gimbal {old_pan:.0f}→{new_pan:.0f}°")

        if self.exploration:
            self.exploration.update_after_turn(degrees)
        self._body_turns_since_start += 1
        self._invalidate_old_observations()
        time.sleep(0.2)

    def _move_gimbal(self, pan, tilt):
        # Offset is applied at serial layer (_nav_send_hook in rover_brain_llm.py)
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

    def _maybe_snapshot(self):
        """Save a JPEG snapshot every SNAPSHOT_INTERVAL_S seconds.
        Logs the file path so it can be reviewed later."""
        now = time.time()
        if now - self._last_snapshot_time < SNAPSHOT_INTERVAL_S:
            return
        jpeg = self.tracker.get_jpeg()
        if not jpeg:
            return
        self._last_snapshot_time = now
        self._snapshot_idx += 1
        fname = f"nav_{self._snapshot_idx:04d}.jpg"
        fpath = os.path.join(NAV_SNAPSHOTS_DIR, fname)
        try:
            with open(fpath, "wb") as f:
                f.write(jpeg)
            print(f"[nav] Snapshot: {fpath}")
        except Exception as e:
            print(f"[nav] Snapshot error: {e}")

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
