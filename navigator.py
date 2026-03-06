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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import room_context

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
MAX_STUCK_EVENTS = 5
MAX_AREA_ESCAPES = 5
AREA_STUCK_RADIUS = 0.45    # meters — tolerate smaller progress before calling it area-stuck
AREA_STUCK_WAYPOINTS = 10   # give exploration more time before declaring area-stuck
LLM_429_BACKOFF_S = 5.0     # seconds to wait after a 429 rate limit
DEFAULT_LLM_RTT_S = 2.5
LLM_RTT_EMA_ALPHA = 0.25
LLM_PREFETCH_MARGIN_S = 0.6
LLM_PREFETCH_MIN_LEAD_S = 0.8
LLM_PREFETCH_MAX_LEAD_S = 6.0
MAX_SUBTASK_DEPTH = 3
SUBTASK_WAYPOINT_BUDGET = 10  # pop subtask if no progress after this many waypoints

DEFAULT_CLEAR_DIST = 0.45   # meters when depth unavailable
MAX_CLEAR_DIST = 2.4        # cap — encoder stops at exact distance
DRIVE_FRACTION = 0.90       # drive 90% of estimated clear distance
COURAGE_FALLBACK_DRIVE_M = 0.45
FORCED_EXPLORE_DRIVE_M = 0.7
TARGET_LOST_DRIVE_M = 0.6
MEDIUM_OBSTACLE_CAP_M = 0.8
FORCED_TURN_LIMIT = 2
MIN_COMMIT_DRIVE_M = 0.25
MAX_DRIVE_AVOIDANCES = 5
AREA_ESCAPE_REVERSE_S = 3.0
AREA_ESCAPE_TURN_DEG = 100
RECOVER_REVERSE_S = 2.1
RECOVER_TURN_DEG = 75
IMMINENT_STOP_CLEARANCE_M = 0.22
IMMINENT_REPLAN_MARGIN_M = 0.12
IMMINENT_LOOKAHEAD_S = 0.8
IMMINENT_TREND_MIN_DROP_MPS = 0.10
UNDER_FURNITURE_CLOSE_CLEAR_M = 0.55
UNDER_FURNITURE_REVERSE_S = 3.4
UNDER_FURNITURE_ESCAPE_TURN_DEG = 90
UNDER_FURNITURE_FLOOR_NAV_S = 4.5

FRAME_DIFF_THRESH = 0.20
BLIND_DRIVE_CHECK_HZ = 5.0  # obstacle check rate during blind drive
ENCODER_STUCK_RATIO = 0.30  # if avg encoder speed < 30% of commanded → stuck
ENCODER_STUCK_READS = 3     # consecutive slow reads = stuck (~0.6s at 5Hz)
ENCODER_RAMPUP_S = 0.6      # ignore encoder stuck during motor ramp-up
FLOOR_ESCAPE_SPEED = 0.12
FLOOR_ESCAPE_TIMEOUT_S = 5.0
DOORWAY_ESCAPE_TIMEOUT_S = 6.5

# YOLO classes to IGNORE for obstacle avoidance.
# These are either above rover height, background furniture seen from ground
# level, or consistent false positives from the wide-angle low camera.
NAV_IGNORE_CLASSES = {
    # Background furniture (above ground / not in collision path)
    "couch", "bed", "dining table", "tv", "laptop", "monitor",
    "clock", "wall_clock",
    # People and animals (not ground obstacles for a low rover)
    "person", "cat", "dog",
    # Common false positives from ground-level wide-angle view
    "skateboard", "surfboard", "snowboard", "sports ball", "frisbee",
    "kite", "baseball bat", "baseball glove", "tennis racket",
    # Animals / outdoor (impossible indoors)
    "elephant", "bear", "zebra", "giraffe", "horse", "sheep", "cow",
    "bird", "airplane", "bus", "train", "truck", "boat",
    # Small objects that can't block the rover
    "fork", "knife", "spoon", "toothbrush", "scissors",
    "mouse", "remote", "cell phone", "keyboard",
    "apple", "orange", "banana", "donut", "cake", "pizza",
    "sandwich", "hot dog", "broccoli", "carrot",
    # Mapped labels that are false positives
    "router", "plant_stem", "cables", "multimeter", "tools",
}
# Dynamic ignore set — LLM-confirmed false positives during this session
_nav_learned_ignore = set()
ROVER_WIDTH_M = 0.26        # body width for corridor checks
UNDER_FURNITURE_TEXT_HINTS = (
    "under a desk", "under desk", "under a table", "under table",
    "under furniture", "chair leg", "chair legs", "office chair",
    "dining chair", "desk structure", "table leg", "table legs",
    "person's legs", "person legs", "human legs", "blocked by chair",
    "blocked by furniture", "under a chair", "under chair",
)
UNDER_FURNITURE_LABEL_HINTS = {
    "chair", "dining chair", "person", "person_legs", "chair_legs",
    "desk", "dining table", "table", "legs", "office_chair",
}


def _clean_phrase_list(items, limit=6):
    phrases = []
    seen = set()
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(text[:64])
        if len(phrases) >= limit:
            break
    return phrases

SCAN_POSITIONS = [
    (0, 0), (-65, 0), (65, 0), (-130, 0), (130, 0),
    (0, -15), (-65, -15), (65, -15),
]

# Module-level ref for web_ui access (set by rover_brain_llm.py)
_exploration_grid = None

# ── Fisheye correction ────────────────────────────────────────────────
# The USB camera has wide-angle fisheye distortion: objects near frame edges
# appear stretched (bbox wider than reality) and positions are compressed
# outward.  These helpers correct cx/bw from pixel-space to real-world.

def _fisheye_cx(cx):
    """Correct a normalized x-center (0-1) for fisheye radial distortion.
    Edges are compressed outward; this maps them closer to true angle.
    Uses a simple cubic model: undistorted = 0.5 + k*(cx-0.5)
    where k < 1 near edges (pull inward)."""
    offset = cx - 0.5  # -0.5 .. +0.5
    # Cubic pull: edges get compressed toward center by ~15%
    corrected = offset * (1.0 - 0.3 * offset * offset * 4)
    return 0.5 + corrected


def _fisheye_bw(bw, cx):
    """Correct a normalized bbox width for fisheye stretch.
    Objects near edges appear wider than they really are.
    Returns de-stretched bw."""
    # Stretch factor increases with distance from center
    dist_from_center = abs(cx - 0.5) * 2  # 0 at center, 1 at edge
    stretch = 1.0 + 0.35 * dist_from_center * dist_from_center
    return bw / stretch


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
                 exploration_grid=None, log_fn=None, room_map=None,
                 yolo_correction_fn=None, floor_nav=None):
        self.rover = rover
        self.detector = detector
        self.tracker = tracker
        self.llm_vision = llm_vision_fn
        self.parse = parse_fn
        self.pose = pose
        self.voice_fn = voice_fn
        self.emergency = emergency_event
        self.exploration = exploration_grid
        self._log_fn = log_fn  # callable(category, message) for web UI
        self.room_map = room_map  # RoomMap for 3D object positions
        self._yolo_correction_fn = yolo_correction_fn  # callable(dict) for LLM corrections
        self.floor_nav = floor_nav
        self.observations = deque(maxlen=MAX_OBSERVATIONS)
        self._body_turns_since_start = 0
        self._subtask_stack = []
        self._last_snapshot_time = 0
        self._snapshot_idx = 0
        self._nav_target = ""  # current navigation target
        self._last_drive_angle = 0  # last drive direction (degrees, neg=left)
        self._target_world_bearing = None  # world bearing where target was last seen
        self._mid_drive_future = None  # LLM prefetch submitted during drive
        self._plan = None  # current NavigationPlan (for journey logging)
        self._last_transition_peek = None
        self._llm_last_rtt_s = None
        self._llm_rtt_ema_s = DEFAULT_LLM_RTT_S

    def _log(self, msg):
        """Print to stdout and send to web UI log stream."""
        print(f"[nav] {msg}")
        if self._log_fn:
            self._log_fn("nav", msg)

    def _record_llm_rtt(self, elapsed_s):
        self._llm_last_rtt_s = max(0.0, float(elapsed_s))
        prev = self._llm_rtt_ema_s
        alpha = LLM_RTT_EMA_ALPHA
        self._llm_rtt_ema_s = ((1.0 - alpha) * prev) + (alpha * self._llm_last_rtt_s)

    def _prefetch_lead_time(self):
        base = self._llm_rtt_ema_s if self._llm_rtt_ema_s else DEFAULT_LLM_RTT_S
        return _clamp(
            base + LLM_PREFETCH_MARGIN_S,
            LLM_PREFETCH_MIN_LEAD_S,
            LLM_PREFETCH_MAX_LEAD_S,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def navigate(self, target, plan_context="", step_budget=None, plan=None):
        """Shoot-and-go navigation to target.  Returns True if arrived.

        Args:
            plan_context: strategic context from orchestrator (injected into
                          every waypoint LLM call). Skips panoramic scan.
            step_budget: override MAX_WAYPOINTS for this step.
            plan: NavigationPlan instance for journey logging (optional).
        """
        self._nav_target = target
        self._plan_context = plan_context
        self._plan = plan
        self._body_turns_since_start = 0
        self._subtask_stack = []
        self._last_result = None
        self._wp_used = 0
        self._last_scene = ""
        self._last_yolo = ""
        self._last_transition_peek = None
        stuck_count = 0
        consecutive_turns = 0  # track turn-without-driving loops
        total_rotation = 0     # cumulative degrees turned (detect circling)
        subtask_wp_count = 0  # waypoints since last subtask change
        max_wp = step_budget if step_budget else MAX_WAYPOINTS
        prev_drive_jpeg = None  # frame before last drive (for stuck detection)
        image_stuck_count = 0   # consecutive image-unchanged drives
        prefetch_future = None  # background LLM call
        llm_pool = ThreadPoolExecutor(max_workers=1)
        self._consecutive_429s = 0

        # Position-based stuck detection
        _checkpoint_x = self.pose.x if hasattr(self.pose, 'x') else 0
        _checkpoint_y = self.pose.y if hasattr(self.pose, 'y') else 0
        _checkpoint_wp = 0
        _area_escape_count = 0

        # Panoramic scan — or directed scan when plan provides context
        if not plan_context:
            self._scan_summary = self._panoramic_scan()
        else:
            self._scan_summary = ""
            self._directed_scan(target, plan_context)

        for wp in range(max_wp):
            if self._aborted():
                self._store_step_result(False, "aborted", wp)
                llm_pool.shutdown(wait=False)
                return False

            self._wp_used = wp + 1
            self._last_drive_avoidances = 0
            effective = self._current_subtask() or target
            self._log(f"Waypoint {wp}: goal='{effective}'"
                  + (f" (subtask of '{target}')" if effective != target else ""))

            # 0. Position-based area stuck detection
            if wp - _checkpoint_wp >= AREA_STUCK_WAYPOINTS:
                cur_x = self.pose.x if hasattr(self.pose, 'x') else 0
                cur_y = self.pose.y if hasattr(self.pose, 'y') else 0
                displacement = math.sqrt(
                    (cur_x - _checkpoint_x)**2 + (cur_y - _checkpoint_y)**2)
                if displacement < AREA_STUCK_RADIUS:
                    _area_escape_count += 1
                    self._log(f"AREA STUCK: moved only {displacement:.2f}m in "
                              f"{AREA_STUCK_WAYPOINTS} waypoints "
                              f"(escape #{_area_escape_count})")
                    if _area_escape_count >= MAX_AREA_ESCAPES:
                        self._log(
                            f"Area stuck {MAX_AREA_ESCAPES}x — giving up on this step")
                        self._store_step_result(False,
                            f"area_stuck ({displacement:.1f}m in "
                            f"{AREA_STUCK_WAYPOINTS}wp)", wp + 1)
                        llm_pool.shutdown(wait=False)
                        return False
                    # Big escape: back out, turn hard, and re-enter open space.
                    self._area_escape()
                    prefetch_future = None
                    total_rotation = 0
                    consecutive_turns = 0
                else:
                    _area_escape_count = max(0, _area_escape_count - 1)
                _checkpoint_x = cur_x
                _checkpoint_y = cur_y
                _checkpoint_wp = wp

            # 1. Align body to gimbal heading
            self._align_body()
            time.sleep(GIMBAL_SETTLE_S)

            # 2. If no prefetch is pending, fire one now so LLM runs
            #    while we gather sensors below (saves 2-3s idle time)
            if prefetch_future is None:
                _ct, _tr = consecutive_turns, total_rotation
                prefetch_future = llm_pool.submit(
                    self._prefetch_assessment, target, _ct, _tr)

            # 3. Gather sensor data (camera now facing body's forward)
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

            # 2b. Circling escape — if we've spun >360° we're trapped
            if abs(total_rotation) >= 360:
                self._log(f"Circling detected ({total_rotation:+.0f}°) "
                      f"— reversing to escape")
                # Auto-learn: save what we see so future plans avoid this
                scene = getattr(self, '_last_scene', '')
                last_yolo = getattr(self, '_last_yolo', '')
                room_name = room_context.load_rooms().get("current_room")
                if room_name and scene:
                    room_context.learn_nav_failure(
                        room_name, scene,
                        f"Got trapped spinning near: {scene[:120]}. "
                        f"YOLO saw: {last_yolo[:80]}. "
                        f"Avoid driving toward these objects.")
                self.rover.send({"T": 1, "L": -0.12, "R": -0.12})
                time.sleep(2.0)
                self._stop_wheels()
                time.sleep(0.3)
                total_rotation = 0
                consecutive_turns = 0
                self._move_gimbal(0, 0)
                continue

            # 3. YOLO fast check — target visible? → P-control approach
            if self.detector and det_age < 1.0:
                target_det = self.detector.find(target, dets)
                if target_det and target_det["bw"] > 0.05:
                    self._log(f"YOLO sees '{target}' "
                          f"(bw={target_det['bw']:.2f}), approaching")
                    self._subtask_stack.clear()
                    # Cancel any prefetch — switching to P-control
                    prefetch_future = None
                    result = self._drive_to_target(target)
                    if result == "arrived":
                        self._log(f"Arrived at '{target}'")
                        self._store_step_result(True, "arrived", wp + 1)
                        llm_pool.shutdown(wait=False)
                        return True
                    if result == "stuck":
                        stuck_count += 1
                        if stuck_count >= MAX_STUCK_EVENTS:
                            self._say("I keep getting stuck")
                            self._store_step_result(False, "stuck", wp + 1)
                            llm_pool.shutdown(wait=False)
                            return False
                        self._recover_stuck(target, encoder_stuck=True)
                    continue

            # 4. Estimate clear distance
            clear_dist = self._estimate_clear_distance(depth_map, dets)
            dist_str = f"{clear_dist:.1f}m" if clear_dist else "unknown"

            # 4b. Wall proximity check — back away and turn 90°
            wall_detected = (
                (depth_map is not None and self._depth_wall_check(depth_map))
                or self._image_wall_check(jpeg))
            if wall_detected:
                prefetch_future = None  # stale after wall recovery
                last_angle = getattr(self, '_last_drive_angle', 0)
                turn_deg = -90 if last_angle > 0 else 90
                self._log(f"Wall detected — backing up + turning "
                      f"{turn_deg:+d}° (was driving {last_angle:+.0f}°)")
                self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
                time.sleep(1.5)
                self._stop_wheels()
                time.sleep(0.3)
                new_pan = _clamp(self.pose.cam_pan + turn_deg, -180, 180)
                self._move_gimbal(new_pan, 0)
                consecutive_turns = 0
                continue

            # 5. LLM assessment — use prefetch if available
            assessment = None
            if prefetch_future is not None:
                try:
                    assessment = prefetch_future.result(timeout=10)
                    if assessment:
                        self._log(f"Using prefetched LLM result")
                except Exception:
                    assessment = None
                prefetch_future = None

            if assessment is None:
                self._log(f"→ LLM input: YOLO={yolo_summary}, "
                      f"clear={dist_str}, turns={consecutive_turns}, "
                      f"rot={total_rotation:+.0f}°")
                assessment = self._llm_assess_waypoint(
                    target, self._current_subtask(), jpeg, yolo_summary,
                    clear_dist, consecutive_turns, total_rotation)

            if assessment is None:
                # LLM failed — keep exploring through the best available space.
                self._log(
                    f"LLM unavailable, driving {COURAGE_FALLBACK_DRIVE_M:.2f}m")
                result, stuck_count = self._execute_blind_drive(
                    COURAGE_FALLBACK_DRIVE_M, stuck_count)
                if result == "stuck_abort":
                    self._store_step_result(False, "stuck", wp + 1)
                    llm_pool.shutdown(wait=False)
                    return False
                # Prefetch for next iteration
                _ct, _tr = consecutive_turns, total_rotation
                prefetch_future = llm_pool.submit(
                    self._prefetch_assessment, target, _ct, _tr)
                continue

            scene = assessment.get("scene", "?")
            self._last_scene = scene
            action = assessment.get("action", "drive_forward")
            # Full thinking log
            extras = []
            if assessment.get("target_visible"):
                extras.append("target_visible=YES")
                # Record world bearing where target was last seen
                drive_angle = assessment.get("drive_angle", 0) or 0
                self._target_world_bearing = (
                    self.pose.body_yaw + self.pose.cam_pan + drive_angle)
            angle = assessment.get("drive_angle")
            if angle and angle != 0:
                extras.append(f"angle={angle:+.0f}°")
            if assessment.get("turn_degrees"):
                extras.append(f"turn={assessment['turn_degrees']}°")
            if assessment.get("subtask"):
                extras.append(f"subtask='{assessment['subtask']}'")
            if assessment.get("subtask_reason"):
                extras.append(f"because='{assessment['subtask_reason']}'")
            if assessment.get("subtask_achieved"):
                extras.append("subtask_achieved=YES")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            self._log(f"LLM: '{scene}' → {action}{extra_str}")

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
                self._log(f"Subtask '{old}' budget exhausted, popping")
                subtask_wp_count = 0

            # 7. Execute action
            # Force drive after too many consecutive turns (break spin loops)
            if (consecutive_turns >= FORCED_TURN_LIMIT
                    and action in ("turn_left", "turn_right")):
                self._log(f"Forced drive: {consecutive_turns} turns without progress")
                action = "drive_forward"

            if action == "arrived":
                # LLM declares arrival (for non-YOLO targets like doorways)
                self._log(f"LLM declares arrived at '{target}'")
                self._store_step_result(True, "arrived", wp + 1)
                llm_pool.shutdown(wait=False)
                return True

            elif action == "approach_target":
                consecutive_turns = 0
                prefetch_future = None  # P-control takes over
                result = self._drive_to_target(target)
                if result == "arrived":
                    self._log(f"Arrived at '{target}'")
                    self._store_step_result(True, "arrived", wp + 1)
                    llm_pool.shutdown(wait=False)
                    return True
                if result == "lost":
                    # YOLO can't find the target — don't re-ask LLM,
                    # just drive forward toward where it was seen
                    angle = (assessment.get("drive_angle", 0)
                             if assessment else 0)
                    self._log(
                        f"P-control lost target, blind drive {TARGET_LOST_DRIVE_M:.2f}m"
                              f" at {angle:+.0f}° instead")
                    pre_drive_jpeg = self.tracker.get_jpeg()
                    _ct, _tr = consecutive_turns, total_rotation
                    result2, stuck_count = self._execute_blind_drive(
                        TARGET_LOST_DRIVE_M, stuck_count, angle,
                        prefetch_pool=llm_pool,
                        prefetch_fn=self._prefetch_assessment,
                        prefetch_args=(target, _ct, _tr))
                    if result2 == "stuck_abort":
                        self._store_step_result(False, "stuck", wp + 1)
                        llm_pool.shutdown(wait=False)
                        return False
                    # Pick up mid-drive prefetch
                    if self._mid_drive_future is not None:
                        prefetch_future = self._mid_drive_future
                        self._mid_drive_future = None
                    else:
                        prefetch_future = None
                elif result == "stuck":
                    stuck_count += 1
                    if stuck_count >= MAX_STUCK_EVENTS:
                        self._say("I keep getting stuck")
                        self._store_step_result(False, "stuck", wp + 1)
                        llm_pool.shutdown(wait=False)
                        return False
                    self._recover_stuck(target, encoder_stuck=True)

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
                self._log(f"Gimbal turned to {new_pan:.0f}° "
                      f"(total rotation: {total_rotation:+.0f}°)")
                # No prefetch after turns — body alignment at next waypoint
                # changes the view, making any prefetched frame stale.
                prefetch_future = None

            elif action in ("drive_forward", "subtask"):
                consecutive_turns = 0
                # Use LLM's distance estimate if provided, else fall back to depth
                llm_dist = assessment.get("drive_distance") if assessment else None
                if llm_dist and isinstance(llm_dist, (int, float)) and 0.1 < llm_dist <= 5.0:
                    drive_dist = min(float(llm_dist), MAX_CLEAR_DIST)
                elif clear_dist and clear_dist > 0.2:
                    drive_dist = min(clear_dist * DRIVE_FRACTION, MAX_CLEAR_DIST)
                else:
                    drive_dist = DEFAULT_CLEAR_DIST
                drive_dist = max(drive_dist, MIN_COMMIT_DRIVE_M)
                # Cap drive distance when depth shows obstacle ahead
                if depth_map is not None:
                    h, w = depth_map.shape[:2]
                    far_strip = depth_map[int(h*0.2):int(h*0.4),
                                          int(w*0.3):int(w*0.7)]
                    near_strip = depth_map[int(h*0.6):h,
                                           int(w*0.3):int(w*0.7)]
                    if (float(np.mean(far_strip))
                            > float(np.mean(near_strip)) * 1.3):
                        drive_dist = min(drive_dist, MEDIUM_OBSTACLE_CAP_M)
                        self._log(f"Capped drive to {MEDIUM_OBSTACLE_CAP_M:.1f}m "
                              f"(obstacle ahead at medium distance)")
                drive_angle = assessment.get("drive_angle", 0) if assessment else 0
                drive_angle = _clamp(drive_angle, -30, 30)
                # Grab frame before driving for stuck detection
                pre_drive_jpeg = self.tracker.get_jpeg()
                # Drive with mid-drive prefetch timed from measured LLM RTT.
                _ct, _tr = consecutive_turns, total_rotation
                result, stuck_count = self._execute_blind_drive(
                    drive_dist, stuck_count, drive_angle,
                    prefetch_pool=llm_pool,
                    prefetch_fn=self._prefetch_assessment,
                    prefetch_args=(target, _ct, _tr))
                if result == "stuck_abort":
                    self._store_step_result(False, "stuck", wp + 1)
                    llm_pool.shutdown(wait=False)
                    return False
                # Pick up mid-drive prefetch if it was submitted
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                else:
                    prefetch_future = None
                # Image-based stuck: did the scene change after driving?
                post_drive_jpeg = self.tracker.get_jpeg()
                if (pre_drive_jpeg and post_drive_jpeg
                        and not _frame_changed(pre_drive_jpeg,
                                               post_drive_jpeg, 0.12)):
                    image_stuck_count += 1
                    if image_stuck_count >= 2:
                        self._log("Stuck: image unchanged after drive"
                              " — backing up and scanning")
                        self._recover_stuck(target)
                        image_stuck_count = 0
                        prefetch_future = None
                    else:
                        self._log("Image unchanged after drive (1/2)")
                else:
                    image_stuck_count = 0

            else:
                consecutive_turns = 0
                _ct, _tr = consecutive_turns, total_rotation
                result, stuck_count = self._execute_blind_drive(
                    COURAGE_FALLBACK_DRIVE_M, stuck_count,
                    prefetch_pool=llm_pool,
                    prefetch_fn=self._prefetch_assessment,
                    prefetch_args=(target, _ct, _tr))
                if result == "stuck_abort":
                    self._store_step_result(False, "stuck", wp + 1)
                    llm_pool.shutdown(wait=False)
                    return False
                # Pick up mid-drive prefetch
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                else:
                    prefetch_future = None

        llm_pool.shutdown(wait=False)
        # Auto-learn from failure: save what went wrong
        scene = getattr(self, '_last_scene', '')
        last_yolo = getattr(self, '_last_yolo', '')
        room_name = room_context.load_rooms().get("current_room")
        if room_name and scene:
            room_context.learn_nav_failure(
                room_name, scene,
                f"Failed to reach '{target}' after {max_wp} waypoints. "
                f"Last scene: {scene[:100]}. "
                f"Avoid this path in future attempts.")
        self._say(f"Could not reach {target}")
        self._store_step_result(False, "budget", max_wp)
        return False

    def _prefetch_assessment(self, target, consecutive_turns, total_rotation):
        """Background LLM call — grab fresh frame + sensors, run assessment.
        Called from ThreadPoolExecutor so it runs while the rover moves."""
        try:
            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                return None
            dets, yolo, _ = self._get_detections()
            depth = (self.tracker.get_depth_map()
                     if hasattr(self.tracker, 'get_depth_map') else None)
            clear = self._estimate_clear_distance(depth, dets)
            return self._llm_assess_waypoint(
                target, self._current_subtask(), jpeg, yolo, clear,
                consecutive_turns, total_rotation)
        except Exception as e:
            self._log(f"Prefetch LLM error: {e}")
            return None

    def _execute_blind_drive(self, distance, stuck_count, drive_angle=0,
                             prefetch_pool=None, prefetch_fn=None,
                             prefetch_args=None):
        """Helper: blind drive + handle stuck.
        Returns tuple: ("ok"|"stuck_abort", updated_stuck_count)."""
        self._last_drive_angle = drive_angle
        self._mid_drive_future = None
        angle_str = f" at {drive_angle:+.0f}°" if abs(drive_angle) > 3 else ""
        self._log(f"Driving forward {distance:.1f}m{angle_str}")
        result = self._blind_drive(distance, drive_angle,
                                   prefetch_pool=prefetch_pool,
                                   prefetch_fn=prefetch_fn,
                                   prefetch_args=prefetch_args)

        # Update exploration grid after drive
        if self.exploration and result in ("ok", "obstacle"):
            heading = self.pose.body_yaw if hasattr(self.pose, 'body_yaw') else 0
            drive_heading = heading + drive_angle
            driven = distance if result == "ok" else distance * 0.3
            self.exploration.update_after_drive(driven, drive_heading)

        # LLM verify before entering stuck recovery — YOLO may be
        # hallucinating obstacles in empty space.
        # Only skip LLM verify if encoders confirmed physical contact.
        if result in ("obstacle", "stuck"):
            # Check if encoders confirmed physical contact
            encoder_stuck = (result == "stuck")
            if encoder_stuck:
                # Physical evidence — don't second-guess with LLM
                stuck_count += 1
                if stuck_count >= MAX_STUCK_EVENTS:
                    self._say("I keep getting stuck")
                    return "stuck_abort", stuck_count
                self._recover_stuck("", encoder_stuck=True)
            elif self._llm_verify_obstacle():
                stuck_count += 1
                if stuck_count >= MAX_STUCK_EVENTS:
                    self._say("I keep getting stuck")
                    return "stuck_abort", stuck_count
                self._recover_stuck("")
            else:
                self._log("LLM says path is clear — YOLO ghost, continuing")
                result = "ok"
        elif result == "ok":
            # Successful movement reduces stale stuck pressure.
            stuck_count = max(0, stuck_count - 1)
        return "ok", stuck_count

    def search(self, target):
        """Search only (no drive).  Returns direction dict or None."""
        return self._search(target)

    def navigate_leg(self, instruction, room_check_fn=None, max_steps=30):
        """Navigate one leg: find and cross a specific transition (doorway).

        The navigator pursues the transition described in `instruction` until
        `room_check_fn` confirms the room has changed, or max_steps is reached.

        Args:
            instruction: dict from TopoMap.leg_instruction() with keys:
                - target_transition: id of doorway
                - visual_cues: what the doorway looks like
                - exit_hint: human-readable guidance
                - expected_floor: floor type after crossing
                - verify_features: room features to check after crossing
                - expected_azimuth_deg: optional heading hint
                - room_nav_hints: optional hints for navigating current room
            room_check_fn: callable(scene_text) -> (room_id, confidence)
                           If room changes, leg is complete.
            max_steps: safety limit

        Returns:
            (success: bool, new_room: str or None, scene: str)
        """
        self._nav_target = instruction.get("exit_hint", "doorway")
        self._plan_context = ""
        self._plan = None
        self._subtask_stack = []
        self._consecutive_429s = 0
        self._last_transition_peek = None

        cues = instruction.get("visual_cues", [])
        exit_hint = instruction.get("exit_hint", "find the exit")
        target_room = instruction.get("target_room", "next room")
        expected_floor = instruction.get("expected_floor", "")
        azimuth = instruction.get("expected_azimuth_deg")
        room_hints = instruction.get("room_nav_hints", "")
        verify_feats = instruction.get("verify_features", [])
        doorway_landmarks = instruction.get("doorway_landmarks", [])
        inside_features = instruction.get("inside_features", [])
        relationship_hint = instruction.get("relationship_hint", "")

        cues_str = ", ".join(cues) if cues else "a doorway"
        verify_str = ", ".join(verify_feats[:3]) if verify_feats else ""

        azimuth_hint = ""
        if azimuth is not None:
            azimuth_hint = (f"The doorway is approximately {azimuth}° from "
                            f"your entry heading "
                            f"({'right' if azimuth > 0 else 'left'}). ")
        self._log(f"Leg: find [{cues_str}] → {target_room}"
                  + (f" (azimuth {azimuth}°)" if azimuth else ""))

        # Don't blind-spin or glance — let the LLM see the forward view
        # and decide to turn based on the azimuth hint in the prompt.
        # Gimbal must always face forward so drive commands match body heading.
        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)

        checkpoint_x = self.pose.x if hasattr(self.pose, 'x') else 0
        checkpoint_y = self.pose.y if hasattr(self.pose, 'y') else 0
        checkpoint_step = 0
        area_escapes = 0
        consecutive_turns = 0
        _prev_jpeg = None
        _img_same_count = 0
        last_scene = ""
        obstacle_info = ""  # feedback from last drive
        last_peek_step = -99
        prefetch_future = None
        llm_pool = ThreadPoolExecutor(max_workers=1)

        for step in range(max_steps):
            if self._aborted():
                llm_pool.shutdown(wait=False)
                return False, None, last_scene

            # Area stuck check after a larger exploration window.
            if step - checkpoint_step >= AREA_STUCK_WAYPOINTS and step > 0:
                cx = self.pose.x if hasattr(self.pose, 'x') else 0
                cy = self.pose.y if hasattr(self.pose, 'y') else 0
                disp = math.sqrt((cx - checkpoint_x)**2 +
                                 (cy - checkpoint_y)**2)
                if disp < AREA_STUCK_RADIUS:
                    self._log(f"AREA STUCK in leg ({disp:.2f}m)")
                    if self._try_floor_nav_escape(doorway_mode=True):
                        consecutive_turns = 0
                        _img_same_count = 0
                        _prev_jpeg = None
                        checkpoint_x, checkpoint_y = cx, cy
                        checkpoint_step = step
                        continue
                    area_escapes += 1
                    self._log(f"Area escape #{area_escapes}")
                    if area_escapes >= MAX_AREA_ESCAPES:
                        llm_pool.shutdown(wait=False)
                        return False, None, last_scene
                    self._area_escape()
                    prefetch_future = None
                    consecutive_turns = 0
                checkpoint_x, checkpoint_y = cx, cy
                checkpoint_step = step

            # Get current view
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)
            jpeg = self._get_annotated_jpeg()
            if not jpeg:
                time.sleep(0.5)
                continue

            # Image variance stuck detection
            if _prev_jpeg and jpeg:
                if not _frame_changed(_prev_jpeg, jpeg, 0.10):
                    _img_same_count += 1
                    if _img_same_count >= 4:
                        self._log("IMAGE STUCK in leg — escaping")
                        if not self._try_floor_nav_escape(doorway_mode=True):
                            self._area_escape()
                        prefetch_future = None
                        _img_same_count = 0
                        _prev_jpeg = None
                        consecutive_turns = 0
                        continue
                else:
                    _img_same_count = 0
            _prev_jpeg = jpeg

            # YOLO + depth
            dets, yolo_summary, _ = self._get_detections()
            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            clear_dist = self._estimate_clear_distance(
                depth_map, dets) if depth_map is not None else None
            depth_info = (f"Depth: {clear_dist:.1f}m clear ahead.\n"
                          if clear_dist else "")

            # YOLO obstruction check: large object covering center = stuck
            obstruction = self._yolo_obstruction_check(dets)
            if obstruction:
                self._log(f"YOLO obstruction: '{obstruction}' blocking center")
                under_furniture, uf_evidence = self._detect_under_furniture(
                    detections=dets, clear_dist=clear_dist)
                if under_furniture and self._under_furniture_escape(
                        self._nav_target, doorway_mode=True,
                        evidence=uf_evidence):
                    obstacle_info = (
                        "Backed out from chair/desk clutter. "
                        "Reassess the doorway from here.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                if self._try_floor_nav_escape(doorway_mode=True):
                    obstacle_info = (
                        "FloorNavigator moved toward a doorway/open gap. "
                        "Reassess from the new position.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                area_escapes += 1
                self._log(f"Legacy escape #{area_escapes}")
                if area_escapes >= MAX_AREA_ESCAPES:
                    llm_pool.shutdown(wait=False)
                    return False, None, last_scene
                self._recover_stuck(self._nav_target)
                prefetch_future = None
                consecutive_turns = 0
                _img_same_count = 0
                _prev_jpeg = None
                continue

            room_ctx = self._room_map_context(6)
            result = None
            if prefetch_future is not None:
                try:
                    result = prefetch_future.result(timeout=10)
                    if result:
                        self._log("Using prefetched leg assessment")
                except Exception:
                    result = None
                prefetch_future = None

            if result is None:
                result = self._llm_assess_leg(
                    instruction, step, max_steps, jpeg, yolo_summary,
                    clear_dist, obstacle_info=obstacle_info, room_ctx=room_ctx)
            obstacle_info = ""  # clear after use

            if result is None:
                self._log(
                    f"Leg step {step+1}: LLM unavailable, blind {COURAGE_FALLBACK_DRIVE_M:.2f}m")
                prefetch_fn = (self._prefetch_leg_assessment
                               if step + 1 < max_steps else None)
                prefetch_args = ((instruction, step + 1, max_steps)
                                 if prefetch_fn else None)
                self._blind_drive(
                    COURAGE_FALLBACK_DRIVE_M, 0,
                    prefetch_pool=llm_pool,
                    prefetch_fn=prefetch_fn,
                    prefetch_args=prefetch_args)
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                continue

            action = result.get("action", "drive")
            scene = result.get("scene", "")
            reason = result.get("reason", "")
            target_vis = result.get("target_visible", False)
            llm_stuck = result.get("stuck", False) or action == "stuck"
            last_scene = scene
            self._log(f"Leg step {step+1}: {scene[:70]} → {action} "
                      f"(target={'Y' if target_vis else 'N'}, "
                      f"stuck={'Y' if llm_stuck else 'N'}) "
                      f"({reason[:50]})")

            if target_vis and (step - last_peek_step) >= 2:
                peek = self._peek_transition_view(instruction, jpeg=jpeg)
                if peek and peek.get("doorway_seen"):
                    self._last_transition_peek = peek
                    last_peek_step = step
                    peek_feats = ", ".join(peek.get("inside_features", [])[:3])
                    self._log(
                        f"Door peek: {peek.get('room_guess') or 'unknown'} "
                        f"({peek.get('confidence', 0.0):.2f})"
                        + (f" via {peek_feats}" if peek_feats else "")
                    )

            # LLM says we're stuck — trigger recovery immediately
            if llm_stuck:
                self._log("LLM detected stuck")
                under_furniture, uf_evidence = self._detect_under_furniture(
                    scene, reason, dets, clear_dist)
                if under_furniture and self._under_furniture_escape(
                        self._nav_target, doorway_mode=True,
                        evidence=uf_evidence):
                    obstacle_info = (
                        "Backed out from under furniture into a clearer spot. "
                        "Reassess the doorway from here.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                if self._try_floor_nav_escape(doorway_mode=True):
                    obstacle_info = (
                        "FloorNavigator followed the widest opening. "
                        "Reassess the doorway from here.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                area_escapes += 1
                self._log(f"Legacy escape #{area_escapes}")
                if area_escapes >= MAX_AREA_ESCAPES:
                    llm_pool.shutdown(wait=False)
                    return False, None, last_scene
                self._recover_stuck(self._nav_target)
                prefetch_future = None
                consecutive_turns = 0
                _img_same_count = 0
                _prev_jpeg = None
                continue

            # YOLO corrections
            yolo_corr = result.get("yolo_corrections")
            if isinstance(yolo_corr, dict) and yolo_corr:
                for wrong, correct in yolo_corr.items():
                    if correct == "_false":
                        _nav_learned_ignore.add(wrong.lower().strip())
                        self._log(f"YOLO ignore: '{wrong}'")
                if self._yolo_correction_fn:
                    self._yolo_correction_fn(yolo_corr)

            # Room map update
            if (self.room_map and self.detector and
                    self.detector.last_detections):
                self.room_map.record(
                    self.detector.last_detections,
                    self.pose.x if hasattr(self.pose, 'x') else 0,
                    self.pose.y if hasattr(self.pose, 'y') else 0,
                    self.pose.body_yaw, self.pose.cam_pan,
                    self.pose.cam_tilt)

            if action == "crossed":
                conf = result.get("confidence", 0.5)
                self._log(f"Leg crossed! conf={conf:.2f}")
                # Verify with room_check_fn if available
                if room_check_fn and conf < 0.9:
                    room_id, room_conf = room_check_fn(scene)
                    if room_id == target_room:
                        self._log(f"Room verified: {room_id} ({room_conf})")
                        llm_pool.shutdown(wait=False)
                        return True, room_id, scene
                    elif room_conf > 0.6:
                        self._log(f"Wrong room: {room_id} ({room_conf}), "
                                  f"expected {target_room}")
                        llm_pool.shutdown(wait=False)
                        return False, room_id, scene
                llm_pool.shutdown(wait=False)
                return True, target_room, scene

            elif action == "turn":
                consecutive_turns += 1
                if consecutive_turns >= FORCED_TURN_LIMIT:
                    self._log(
                        f"{FORCED_TURN_LIMIT} turns in leg — forcing drive {FORCED_EXPLORE_DRIVE_M:.1f}m")
                    consecutive_turns = 0
                    self._blind_drive(FORCED_EXPLORE_DRIVE_M, 0)
                    prefetch_future = None
                else:
                    degrees = int(result.get("turn_degrees", 0))
                    if abs(degrees) > 5:
                        self._spin_body(degrees)
                        self._move_gimbal(0, 0)
                        prefetch_future = None

            elif action == "drive":
                consecutive_turns = 0
                angle = int(result.get("angle", 0))
                dist = float(result.get("distance", 0.5))
                dist = max(MIN_COMMIT_DRIVE_M, min(MAX_CLEAR_DIST, dist))
                if clear_dist is not None:
                    dist = min(dist, clear_dist * DRIVE_FRACTION)
                    dist = max(MIN_COMMIT_DRIVE_M, dist)
                prefetch_fn = (self._prefetch_leg_assessment
                               if step + 1 < max_steps else None)
                prefetch_args = ((instruction, step + 1, max_steps)
                                 if prefetch_fn else None)
                drive_result = self._blind_drive(
                    dist, angle,
                    prefetch_pool=llm_pool,
                    prefetch_fn=prefetch_fn,
                    prefetch_args=prefetch_args)
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                else:
                    prefetch_future = None
                if drive_result in ("obstacle", "stuck"):
                    obstacle_info = (
                        f"WARNING: Last drive was BLOCKED by obstacle "
                        f"at angle {angle}°. Try a DIFFERENT angle or "
                        f"turn to find a clear path around it.\n")
                    encoder_stuck = (drive_result == "stuck")
                    if encoder_stuck:
                        self._recover_stuck(self._nav_target, encoder_stuck=True)
                        prefetch_future = None
                    elif self._llm_verify_obstacle():
                        if self._try_floor_nav_escape(doorway_mode=True):
                            obstacle_info = (
                                "FloorNavigator advanced toward a doorway "
                                "or widest gap. Reassess before driving again.\n")
                            prefetch_future = None
                            _img_same_count = 0
                            _prev_jpeg = None
                            continue
                        self._recover_stuck(self._nav_target)
                        prefetch_future = None
                    else:
                        self._log("YOLO ghost — continuing")

            # After each drive, check room change via scene description
            if room_check_fn and action == "drive" and scene:
                room_id, room_conf = room_check_fn(scene)
                if room_id == target_room and room_conf > 0.7:
                    self._log(f"Room changed to {room_id} ({room_conf}) "
                              f"— leg complete!")
                    llm_pool.shutdown(wait=False)
                    return True, room_id, scene

        self._log(f"Leg exhausted ({max_steps} steps)")
        llm_pool.shutdown(wait=False)
        return False, None, last_scene

    def navigate_reactive(self, goal, max_steps=40):
        """Reactive navigation: look → decide → drive → repeat.

        No pre-planned route. Each step the LLM sees the camera and decides:
        - Which direction has the clearest path toward the goal
        - How far to drive
        - Whether we've arrived

        Returns True if arrived, False if gave up.
        """
        self._nav_target = goal
        self._plan_context = ""
        self._plan = None
        self._subtask_stack = []
        self._consecutive_429s = 0

        checkpoint_x = self.pose.x if hasattr(self.pose, 'x') else 0
        checkpoint_y = self.pose.y if hasattr(self.pose, 'y') else 0
        checkpoint_step = 0
        area_escapes = 0

        room_ctx = self._room_map_context(8)
        self._log(f"Reactive nav to '{goal}' (max {max_steps} steps)")
        consecutive_turns = 0
        _prev_jpeg = None         # for image variance stuck detection
        _img_same_count = 0       # consecutive similar images
        obstacle_info = ""
        prefetch_future = None
        llm_pool = ThreadPoolExecutor(max_workers=1)

        for step in range(max_steps):
            if self._aborted():
                self._log("Reactive nav aborted")
                llm_pool.shutdown(wait=False)
                return False

            # Area stuck check after a larger exploration window.
            if step - checkpoint_step >= AREA_STUCK_WAYPOINTS and step > 0:
                cx = self.pose.x if hasattr(self.pose, 'x') else 0
                cy = self.pose.y if hasattr(self.pose, 'y') else 0
                disp = math.sqrt((cx - checkpoint_x)**2 +
                                 (cy - checkpoint_y)**2)
                if disp < AREA_STUCK_RADIUS:
                    self._log(f"AREA STUCK: {disp:.2f}m in {AREA_STUCK_WAYPOINTS} steps")
                    if self._try_floor_nav_escape(doorway_mode=False):
                        checkpoint_x, checkpoint_y = cx, cy
                        checkpoint_step = step
                        prefetch_future = None
                        consecutive_turns = 0
                        _img_same_count = 0
                        _prev_jpeg = None
                        continue
                    area_escapes += 1
                    self._log(f"Legacy escape #{area_escapes}")
                    if area_escapes >= MAX_AREA_ESCAPES:
                        self._log(f"Giving up — stuck {MAX_AREA_ESCAPES} times")
                        self._say("I can't find a way out")
                        llm_pool.shutdown(wait=False)
                        return False
                    self._area_escape()
                    prefetch_future = None
                checkpoint_x, checkpoint_y = cx, cy
                checkpoint_step = step

            # Refresh room map context every 5 steps
            if step % 5 == 0 and self.room_map:
                room_ctx = self._room_map_context(8)

            # Get current view
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)
            jpeg = self._get_annotated_jpeg()
            if not jpeg:
                time.sleep(0.5)
                continue

            # Get YOLO + depth info
            dets, yolo_summary, _ = self._get_detections()
            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            clear_dist = self._estimate_clear_distance(
                depth_map, dets) if depth_map is not None else None

            # YOLO obstruction check: large object covering center = stuck
            obstruction = self._yolo_obstruction_check(dets)
            if obstruction:
                self._log(f"YOLO obstruction: '{obstruction}' blocking center")
                under_furniture, uf_evidence = self._detect_under_furniture(
                    detections=dets, clear_dist=clear_dist)
                if under_furniture and self._under_furniture_escape(
                        goal, doorway_mode=False, evidence=uf_evidence):
                    obstacle_info = (
                        "Backed out from chair/desk clutter. "
                        "Reassess from the new position.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                if self._try_floor_nav_escape(doorway_mode=False):
                    obstacle_info = (
                        "FloorNavigator moved toward the widest clear space. "
                        "Reassess from the new position.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                area_escapes += 1
                self._log(f"Legacy escape #{area_escapes}")
                if area_escapes >= MAX_AREA_ESCAPES:
                    self._log(f"Giving up — stuck {MAX_AREA_ESCAPES} times")
                    self._say("I can't find a way out")
                    llm_pool.shutdown(wait=False)
                    return False
                self._recover_stuck(goal)
                prefetch_future = None
                continue

            result = None
            if prefetch_future is not None:
                try:
                    result = prefetch_future.result(timeout=10)
                    if result:
                        self._log("Using prefetched reactive assessment")
                except Exception:
                    result = None
                prefetch_future = None

            if result is None:
                result = self._llm_assess_reactive(
                    goal, step, max_steps, jpeg, yolo_summary, clear_dist,
                    obstacle_info=obstacle_info, room_ctx=room_ctx)
            if result is None:
                # LLM unavailable — keep pressing through visible free space.
                self._log(
                    f"Step {step+1}: LLM unavailable, blind {COURAGE_FALLBACK_DRIVE_M:.2f}m")
                prefetch_fn = (self._prefetch_reactive_assessment
                               if step + 1 < max_steps else None)
                prefetch_args = ((goal, step + 1, max_steps)
                                 if prefetch_fn else None)
                self._blind_drive(
                    COURAGE_FALLBACK_DRIVE_M, 0,
                    prefetch_pool=llm_pool,
                    prefetch_fn=prefetch_fn,
                    prefetch_args=prefetch_args)
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                continue

            action = result.get("action", "drive")
            scene = result.get("scene", "")
            reason = result.get("reason", "")
            target_vis = result.get("target_visible", False)
            llm_stuck = result.get("stuck", False) or action == "stuck"
            self._log(f"Step {step+1}: {scene[:70]} → {action} "
                      f"(target={'Y' if target_vis else 'N'}, "
                      f"stuck={'Y' if llm_stuck else 'N'}) "
                      f"({reason[:50]})")

            # LLM says we're stuck — trigger recovery immediately
            if llm_stuck:
                self._log("LLM detected stuck")
                under_furniture, uf_evidence = self._detect_under_furniture(
                    scene, reason, dets, clear_dist)
                if under_furniture and self._under_furniture_escape(
                        goal, doorway_mode=False, evidence=uf_evidence):
                    obstacle_info = (
                        "Backed out from under furniture into a clearer spot. "
                        "Reassess from the new position.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                if self._try_floor_nav_escape(doorway_mode=False):
                    obstacle_info = (
                        "FloorNavigator took the widest passable gap. "
                        "Reassess from the new position.\n")
                    prefetch_future = None
                    consecutive_turns = 0
                    _img_same_count = 0
                    _prev_jpeg = None
                    continue
                area_escapes += 1
                self._log(f"Legacy escape #{area_escapes}")
                if area_escapes >= MAX_AREA_ESCAPES:
                    self._log(f"Giving up — stuck {MAX_AREA_ESCAPES} times")
                    self._say("I can't find a way out")
                    llm_pool.shutdown(wait=False)
                    return False
                self._recover_stuck(goal)
                prefetch_future = None
                continue

            # Process YOLO corrections from LLM
            yolo_corr = result.get("yolo_corrections")
            if isinstance(yolo_corr, dict) and yolo_corr:
                for wrong, correct in yolo_corr.items():
                    if correct == "_false":
                        _nav_learned_ignore.add(wrong.lower().strip())
                        self._log(f"YOLO ignore: '{wrong}' (LLM says false)")
                if self._yolo_correction_fn:
                    self._yolo_correction_fn(yolo_corr)

            # Image variance stuck detection
            if _prev_jpeg and jpeg:
                if not _frame_changed(_prev_jpeg, jpeg, 0.10):
                    _img_same_count += 1
                    if _img_same_count >= 4:
                        self._log(f"IMAGE STUCK: scene unchanged for "
                                  f"{_img_same_count} steps — escaping")
                        if not self._try_floor_nav_escape(doorway_mode=False):
                            self._area_escape()
                        prefetch_future = None
                        _img_same_count = 0
                        _prev_jpeg = None
                        consecutive_turns = 0
                        continue
                else:
                    _img_same_count = 0
            _prev_jpeg = jpeg

            # Feed into room map
            if (self.room_map and self.detector and
                    self.detector.last_detections):
                self.room_map.record(
                    self.detector.last_detections,
                    self.pose.x if hasattr(self.pose, 'x') else 0,
                    self.pose.y if hasattr(self.pose, 'y') else 0,
                    self.pose.body_yaw, self.pose.cam_pan,
                    self.pose.cam_tilt)

            if action == "arrived":
                self._log(f"Arrived at '{goal}'!")
                self._say(f"Reached {goal}")
                llm_pool.shutdown(wait=False)
                return True

            elif action == "turn":
                consecutive_turns += 1
                if consecutive_turns >= FORCED_TURN_LIMIT:
                    # Force a drive — turning in circles
                    self._log(
                        f"{FORCED_TURN_LIMIT} consecutive turns — forcing drive {FORCED_EXPLORE_DRIVE_M:.1f}m")
                    consecutive_turns = 0
                    self._blind_drive(FORCED_EXPLORE_DRIVE_M, 0)
                    prefetch_future = None
                else:
                    degrees = int(result.get("turn_degrees", 0))
                    if abs(degrees) > 5:
                        self._spin_body(degrees)
                        self._move_gimbal(0, 0)
                        prefetch_future = None

            elif action == "drive":
                consecutive_turns = 0
                obstacle_info = ""
                angle = int(result.get("angle", 0))
                dist = float(result.get("distance", 0.5))
                dist = max(MIN_COMMIT_DRIVE_M, min(MAX_CLEAR_DIST, dist))
                if clear_dist is not None:
                    dist = min(dist, clear_dist * DRIVE_FRACTION)
                    dist = max(MIN_COMMIT_DRIVE_M, dist)
                prefetch_fn = (self._prefetch_reactive_assessment
                               if step + 1 < max_steps else None)
                prefetch_args = ((goal, step + 1, max_steps)
                                 if prefetch_fn else None)
                drive_result = self._blind_drive(
                    dist, angle,
                    prefetch_pool=llm_pool,
                    prefetch_fn=prefetch_fn,
                    prefetch_args=prefetch_args)
                if self._mid_drive_future is not None:
                    prefetch_future = self._mid_drive_future
                    self._mid_drive_future = None
                else:
                    prefetch_future = None
                if drive_result in ("obstacle", "stuck"):
                    obstacle_info = (
                        f"WARNING: Last drive BLOCKED at angle {angle}°. "
                        f"Try a DIFFERENT angle to go around.\n")
                    encoder_stuck = (drive_result == "stuck")
                    if encoder_stuck:
                        self._recover_stuck(goal, encoder_stuck=True)
                        prefetch_future = None
                    elif self._llm_verify_obstacle():
                        if self._try_floor_nav_escape(doorway_mode=False):
                            obstacle_info = (
                                "FloorNavigator moved toward the widest gap. "
                                "Reassess before the next drive.\n")
                            prefetch_future = None
                            _img_same_count = 0
                            _prev_jpeg = None
                            continue
                        self._recover_stuck(goal)
                        prefetch_future = None
                    else:
                        self._log("YOLO ghost — continuing")

        self._log(f"Max steps ({max_steps}) reached")
        self._say("Couldn't reach it")
        llm_pool.shutdown(wait=False)
        return False

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

    def _estimate_depth_clear_distance(self, depth_map, drive_angle=0):
        """Estimate clear forward distance from the depth corridor alone."""
        if depth_map is None:
            return None

        # Shift corridor based on drive angle — check where we're actually going
        # 30° angle shifts the corridor center by ~0.17 of frame width
        angle_shift = _clamp(drive_angle / 90.0 * 0.5, -0.25, 0.25)
        corridor_center = 0.5 + angle_shift
        h, w = depth_map.shape[:2]
        # Corridor adjusted for drive angle
        cx0 = int(w * max(0.05, corridor_center - 0.15))
        cx1 = int(w * min(0.95, corridor_center + 0.15))
        y0, y1 = int(h * 0.40), h
        corridor = depth_map[y0:y1, cx0:cx1]
        if corridor.size == 0:
            return None

        # Depth Anything: higher = closer. 95th pctl = nearest obstacle.
        near = float(np.percentile(corridor, 95))
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())

        if d_max - d_min < 0.01:
            mid = (d_min + d_max) / 2.0
            if mid > 0.5:
                return 0.15  # wall right in front
            return 1.0

        # Check depth gradient: if the bottom rows are much closer than
        # top rows, we're approaching something fast
        top_strip = corridor[:corridor.shape[0] // 3, :]
        bot_strip = corridor[2 * corridor.shape[0] // 3:, :]
        top_mean = float(np.mean(top_strip))
        bot_mean = float(np.mean(bot_strip))
        # If bottom much closer than top → floor/object approaching
        gradient_penalty = 0.0
        if bot_mean > top_mean * 1.4 and bot_mean > d_min + (d_max - d_min) * 0.5:
            gradient_penalty = 0.3  # something close at ground level

        # Invert: high depth = close = small distance
        relative_far = 1.0 - (near - d_min) / (d_max - d_min + 1e-6)
        dist_m = 0.3 + relative_far * (MAX_CLEAR_DIST - 0.3)
        dist_m -= gradient_penalty
        return max(0.15, dist_m)

    def _estimate_clear_distance(self, depth_map, detections,
                                 drive_angle=0):
        """Estimate driveable distance from the safest available cues.

        Metric object distances are only trusted when they come from known
        class geometry. Depth remains the primary free-space cue.
        """
        # Shift corridor based on drive angle — check where we're actually going
        angle_shift = _clamp(drive_angle / 90.0 * 0.5, -0.25, 0.25)
        corridor_center = 0.5 + angle_shift

        nearest_metric = None
        if detections:
            half_w = 0.22  # rover is 26cm, ~22% of 65° FOV at 0.5m
            lo = corridor_center - half_w
            hi = corridor_center + half_w
            corridor_dets = [
                d for d in detections
                if lo < _fisheye_cx(d["cx"]) < hi
                and d.get("dist_m") is not None
                and d.get("dist_source") == "bbox_width"
                and d.get("dist_conf", 0.0) >= 0.55
                and d["dist_m"] > 0.1
            ]
            if corridor_dets:
                nearest_metric = min(d["dist_m"] for d in corridor_dets)
            for d in detections:
                real_cx = _fisheye_cx(d["cx"])
                real_bw = _fisheye_bw(d["bw"], d["cx"])
                obj_left = real_cx - real_bw / 2
                obj_right = real_cx + real_bw / 2
                if (obj_right > lo and obj_left < hi
                        and d.get("dist_m") is not None
                        and d.get("dist_source") == "bbox_width"
                        and d.get("dist_conf", 0.0) >= 0.55
                        and d["dist_m"] < 0.5):
                    nearest_metric = (d["dist_m"] if nearest_metric is None
                                      else min(nearest_metric, d["dist_m"]))

        depth_clear = self._estimate_depth_clear_distance(depth_map,
                                                          drive_angle)
        if nearest_metric is not None and depth_clear is not None:
            return min(nearest_metric, depth_clear)
        if nearest_metric is not None:
            return nearest_metric
        return depth_clear

    def _predict_imminent_blockage(self, clear_ahead_m, remaining_dist_m,
                                   clear_history):
        """Detect when the current drive is likely to end in a block/stuck."""
        if clear_ahead_m is None:
            return None
        if clear_ahead_m <= IMMINENT_STOP_CLEARANCE_M:
            return (f"clearance only {clear_ahead_m:.2f}m ahead")
        if (remaining_dist_m > 0.10
                and clear_ahead_m + IMMINENT_REPLAN_MARGIN_M < remaining_dist_m):
            return (f"remaining drive {remaining_dist_m:.2f}m exceeds "
                    f"safe clearance {clear_ahead_m:.2f}m")
        if len(clear_history) >= 2:
            t0, c0 = clear_history[-2]
            t1, c1 = clear_history[-1]
            dt = t1 - t0
            if dt > 1e-3:
                drop_rate = (c0 - c1) / dt
                projected = c1 - drop_rate * IMMINENT_LOOKAHEAD_S
                if (drop_rate >= IMMINENT_TREND_MIN_DROP_MPS
                        and projected <= IMMINENT_STOP_CLEARANCE_M + 0.05):
                    return (f"clearance collapsing ({c1:.2f}m now, "
                            f"{drop_rate:.2f}m/s)")
        return None

    def _depth_obstacle_check(self, depth_map, drive_angle=0):
        """Check depth map for close obstacles in driving corridor.
        Adjusts corridor based on drive_angle.
        Returns description string if obstacle found, None if clear."""
        h, w = depth_map.shape[:2]
        # Shift corridor based on drive angle
        angle_shift = _clamp(drive_angle / 90.0 * 0.3, -0.15, 0.15)
        center = 0.5 + angle_shift
        # Rover-width corridor: 26cm rover ≈ 22% of FOV at typical distance
        x0 = int(w * max(0.05, center - 0.25))
        x1 = int(w * min(0.95, center + 0.25))
        y0 = int(h * 0.45)
        corridor = depth_map[y0:h, x0:x1]
        if corridor.size == 0:
            return None

        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max - d_min < 0.01:
            mid = (d_min + d_max) / 2.0
            if mid > 0.5:
                return "wall (uniform close depth)"
            return None

        # Depth Anything: higher = closer. 97th percentile = nearest object
        # (lowered from 99th to catch smaller obstacles sooner)
        near_val = float(np.percentile(corridor, 97))

        relative_closeness = (near_val - d_min) / (d_max - d_min + 1e-6)
        if relative_closeness > 0.55:
            # Check if it's a ground-level obstacle (bottom rows are close)
            bot_third = corridor[2 * corridor.shape[0] // 3:, :]
            bot_near = float(np.percentile(bot_third, 95))
            bot_closeness = (bot_near - d_min) / (d_max - d_min + 1e-6)
            detail = "ground" if bot_closeness > 0.65 else "mid-level"
            return f"{detail} obstacle at {relative_closeness:.0%} depth"

        return None

    def _depth_wall_check(self, depth_map):
        """Check if a wall/surface fills most of the frame (rover pressed
        against it). Depth Anything: higher = closer.
        Returns True if wall detected, False otherwise."""
        h, w = depth_map.shape[:2]
        # Wide corridor: center 80% horizontal, middle 60% vertical
        x0, x1 = int(w * 0.10), int(w * 0.90)
        y0, y1 = int(h * 0.20), int(h * 0.80)
        region = depth_map[y0:y1, x0:x1]
        if region.size == 0:
            return False

        d_min = float(depth_map.min())
        d_max = float(depth_map.max())

        # Uniform depth = featureless surface (wall, ceiling, floor very close)
        if d_max - d_min < 0.02:
            self._log(f"Wall detected: uniform depth "
                  f"(range={d_max - d_min:.4f})")
            return True

        # What fraction of the region is in the top 15% of depth (= very close)?
        close_threshold = d_min + (d_max - d_min) * 0.85
        close_fraction = float(np.mean(region > close_threshold))

        if close_fraction > 0.60:
            self._log(f"Wall detected: {close_fraction:.0%} of frame "
                  f"is very close (depth threshold {close_threshold:.2f})")
            return True
        return False

    def _image_wall_check(self, jpeg):
        """Check if the raw camera image is a featureless surface (wall).
        Uses color uniformity — a wall has low brightness variance and
        one dominant color, a real scene has varied brightness.
        Returns True if wall/surface detected."""
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            small = cv2.resize(img, (80, 60))
            # Color uniformity: real scenes have std > 45, walls < 35
            std = float(small.std())
            # Dominant color: bin into 16 levels, check max bin fraction
            hist = cv2.calcHist([small], [0], None, [16], [0, 256])
            dominant = float(hist.max() / hist.sum())
            if std < 35 and dominant > 0.30:
                self._log(f"Wall detected (image): "
                      f"std={std:.0f}, dominant_bin={dominant:.0%}")
                return True
            return False
        except Exception:
            return False

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

    def _pick_visual_anchors(self):
        """Pick 1-2 YOLO detections from the edges of the frame as visual anchors.
        These are used to re-orient after avoidance maneuvers.
        Returns list of {"name", "cx", "side"} dicts."""
        dets, _, age = self._get_detections()
        if age > 1.5 or not dets:
            return []
        # Pick objects NOT in the driving corridor (edges of frame)
        # that are reasonably confident and not too large (not a wall)
        anchors = []
        for d in dets:
            cx = d["cx"]
            if 0.25 < cx < 0.75:
                continue  # in driving corridor, skip
            if d["conf"] < 0.30 or d["bw"] > 0.40:
                continue  # low confidence or too large
            side = "left" if cx < 0.5 else "right"
            anchors.append({"name": d["name"], "cx": cx, "side": side})
        # Keep at most 2, prefer high confidence
        anchors.sort(key=lambda a: -dets[0]["conf"]
                     if a["name"] == dets[0]["name"] else 0)
        return anchors[:2]

    def _reorient_to_anchors(self, anchors):
        """After avoidance, scan for anchor objects and correct heading.
        Returns True if correction was applied."""
        if not anchors or not self.detector:
            return False
        # Get current detections
        dets, _, age = self._get_detections()
        if age > 1.5 or not dets:
            return False
        for anchor in anchors:
            for d in dets:
                if d["name"] != anchor["name"]:
                    continue
                # Found the anchor — compute how much it shifted
                old_cx = anchor["cx"]
                new_cx = d["cx"]
                shift = new_cx - old_cx  # positive = anchor moved right = we turned left
                if abs(shift) < 0.05:
                    continue  # negligible shift
                # Convert frame shift to degrees (~65° FOV)
                correction_deg = -shift * 65.0
                if abs(correction_deg) > 5:
                    self._log(f"Anchor '{anchor['name']}' shifted "
                          f"{shift:+.2f} in frame → correcting {correction_deg:+.0f}°")
                    self._spin_body(correction_deg)
                    self._move_gimbal(0, 0)
                    return True
        return False

    def _sleep_abortable(self, secs, tick=0.1):
        deadline = time.time() + max(0.0, secs)
        while time.time() < deadline:
            if self._aborted():
                self._stop_wheels()
                return False
            time.sleep(min(tick, max(0.0, deadline - time.time())))
        return True

    def _detect_under_furniture(self, scene_text="", reason_text="",
                                detections=None, clear_dist=None):
        """Detect a desk/chair-leg trap from scene text and nearby clutter."""
        text = f"{scene_text} {reason_text}".lower()
        score = 0
        evidence = []

        for phrase in UNDER_FURNITURE_TEXT_HINTS:
            if phrase in text:
                score += 2 if phrase.startswith("under ") else 1
                evidence.append(phrase)

        close_furniture = 0
        has_chair = False
        has_table = False
        has_legs = False
        for d in detections or []:
            name = str(d.get("name", "")).strip().lower()
            if not name:
                continue
            if (name in UNDER_FURNITURE_LABEL_HINTS
                    or "chair" in name
                    or "desk" in name
                    or "table" in name
                    or "legs" in name):
                if name not in evidence:
                    evidence.append(name)
                has_chair = has_chair or ("chair" in name)
                has_table = has_table or ("desk" in name or "table" in name)
                has_legs = has_legs or ("person" in name or "legs" in name)
                if (d.get("bw", 0.0) >= 0.12
                        or d.get("dist_m", 999.0) <= 0.6
                        or d.get("depth_closeness", 0.0) >= 0.65):
                    close_furniture += 1

        if has_chair and has_legs:
            score += 2
            evidence.append("chair+legs")
        elif has_chair and has_table:
            score += 2
            evidence.append("chair+desk")
        elif close_furniture >= 2:
            score += 1
            evidence.append("tight clutter")

        if clear_dist is not None and clear_dist <= UNDER_FURNITURE_CLOSE_CLEAR_M:
            score += 1
            evidence.append(f"clear={clear_dist:.2f}m")

        summary = ", ".join(dict.fromkeys(evidence))
        return score >= 3, summary

    def _under_furniture_escape(self, target="", doorway_mode=False,
                                evidence=""):
        """Back out from desk/chair-leg clutter before normal replanning."""
        if self._aborted():
            return False

        self._log("UNDER-FURNITURE ESCAPE"
                  + (f": {evidence}" if evidence else ""))
        self._stop_wheels()
        self._move_gimbal(0, 0)
        time.sleep(0.2)

        # Step 1: reverse straight to clear chair legs / desk edges.
        self._log("  1/3: reversing out of clutter")
        self.rover.send({"T": 1, "L": -0.12, "R": -0.12})
        if not self._sleep_abortable(UNDER_FURNITURE_REVERSE_S):
            return False
        self._stop_wheels()
        time.sleep(0.2)

        # Step 2: rotate to present a fresh forward view.
        escape_turn = UNDER_FURNITURE_ESCAPE_TURN_DEG
        if getattr(self, "_under_escape_dir", 1) < 0:
            escape_turn = -escape_turn
        self._under_escape_dir = -getattr(self, "_under_escape_dir", 1)
        self._log(f"  2/3: turning {escape_turn:+.0f}°")
        self._spin_body(escape_turn)
        self._move_gimbal(0, 0)
        time.sleep(0.2)

        # Step 3: move into the widest opening if FloorNavigator can see one.
        moved = False
        if self.floor_nav is not None and not self._aborted():
            try:
                self._log("  3/3: widest-gap escape")
                moved = bool(self.floor_nav.drive_toward(
                    target_direction="center",
                    speed=FLOOR_ESCAPE_SPEED,
                    timeout=UNDER_FURNITURE_FLOOR_NAV_S,
                    prefer_widest=True))
            except Exception as e:
                self._log(f"Under-furniture floor escape error: {e}")
                moved = False
            finally:
                self._stop_wheels()
                self._move_gimbal(0, 0)
                time.sleep(GIMBAL_SETTLE_S)

        dets, _, age = self._get_detections()
        depth_map = (self.tracker.get_depth_map()
                     if hasattr(self.tracker, 'get_depth_map') else None)
        clear_after = (self._estimate_clear_distance(
            depth_map, dets) if depth_map is not None else None)
        still_trapped, trap_evidence = self._detect_under_furniture(
            detections=dets if age < 1.5 else [], clear_dist=clear_after)
        if still_trapped and not moved:
            self._log("Under-furniture escape still trapped"
                      + (f" ({trap_evidence})" if trap_evidence else ""))
            return False

        target_name = target or getattr(self, "_nav_target", "exit")
        if target_name and not self._aborted():
            for pan in (-90, -45, 0, 45, 90):
                self._move_gimbal(pan, 0)
                if not self._sleep_abortable(0.2, tick=0.05):
                    return False
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)
            jpeg = self._get_annotated_jpeg()
            if jpeg:
                hint = ("Look for a doorway or brightest open exit."
                        if doorway_mode else
                        "Look for the clearest way toward the goal.")
                result = self._llm_call(
                    f"I just backed out from under chair/desk clutter. "
                    f"I need to get to '{target_name}'. {hint}\n"
                    f"Reply JSON: "
                    f'{{"best_direction": <degrees, neg=left, pos=right>, '
                    f'"reason": "why"}}',
                    jpeg)
                if result and "best_direction" in result:
                    best = int(result["best_direction"])
                    best = max(-180, min(180, best))
                    self._log(f"Under-furniture reacquire: {best}° — "
                              f"{result.get('reason', '')[:50]}")
                    if abs(best) > 10:
                        self._spin_body(best)
                        self._move_gimbal(0, 0)

        return True

    def _blind_drive(self, distance_m, drive_angle=0,
                     prefetch_pool=None, prefetch_fn=None,
                     prefetch_args=None):
        """Drive forward with depth + YOLO reactive avoidance.
        drive_angle: degrees offset (-30 to +30, negative=left, positive=right).
        Steers around obstacles when possible, stops only when blocked.
        Uses encoder feedback to measure actual distance (timer fallback).
        Submits a mid-drive LLM prefetch when the measured time remaining is
        close to the observed LLM round-trip time, so the next assessment is
        ready when the drive finishes.
        Returns: "ok", "obstacle", or "stuck".
        Sets self._last_drive_avoidances (int) with count of steer events."""
        drive_time = distance_m / DRIVE_SPEED  # timer fallback
        t_start = time.time()
        check_interval = 1.0 / BLIND_DRIVE_CHECK_HZ
        enc_stuck_count = 0
        steering = False  # True when actively avoiding an obstacle
        avoidance_count = 0  # how many times we had to steer around something

        # Encoder-based distance tracking
        enc_distance = 0.0      # meters traveled (from encoders)
        enc_last_time = None     # last encoder read timestamp
        enc_has_data = False     # got at least one valid encoder reading
        clear_history = deque(maxlen=4)

        # Mid-drive prefetch: fire when the remaining drive time is within the
        # observed LLM RTT plus a small safety margin.
        prefetch_submitted = False

        # Capture visual anchors before driving (for post-avoidance correction)
        visual_anchors = self._pick_visual_anchors()
        if visual_anchors:
            self._log(f"Visual anchors: {[a['name'] + '(' + a['side'] + ')' for a in visual_anchors]}")

        # Keep camera level while driving — tilting down blocks forward vision
        self._move_gimbal(self.pose.cam_pan, 0)

        # Convert drive_angle to wheel differential
        angle_steer = _clamp(drive_angle / 90.0 * 0.10, -0.10, 0.10)
        base_L = DRIVE_SPEED + angle_steer
        base_R = DRIVE_SPEED - angle_steer
        self.rover.send({"T": 1, "L": round(base_L, 3),
                         "R": round(base_R, 3)})
        if abs(drive_angle) > 3:
            self._log(f"Driving at {drive_angle:+.0f}° angle")

        try:
            while time.time() - t_start < drive_time:
                # Encoder distance check — stop when we've driven far enough
                if enc_has_data and enc_distance >= distance_m:
                    break
                if self._aborted():
                    return "obstacle"

                time.sleep(check_interval)

                obstacle_detected = False
                steer_dir = "straight"
                progress = enc_distance if enc_has_data else (
                    (time.time() - t_start) * DRIVE_SPEED)
                remaining_dist = max(0.0, distance_m - progress)

                # DEPTH MAP obstacle check — sees everything (chair legs,
                # toys, door frames) regardless of YOLO class
                depth_map = (self.tracker.get_depth_map()
                             if hasattr(self.tracker, 'get_depth_map')
                             else None)
                # Wall check: featureless surface filling the frame
                cur_jpeg = self.tracker.get_jpeg()
                if cur_jpeg and self._image_wall_check(cur_jpeg):
                    self._stop_wheels()
                    self._last_drive_avoidances = avoidance_count
                    self._log("Blind drive: wall ahead (image), stopping")
                    return "obstacle"

                if depth_map is not None:
                    if self._depth_wall_check(depth_map):
                        self._stop_wheels()
                        self._last_drive_avoidances = avoidance_count
                        self._log("Blind drive: wall ahead (depth), stopping")
                        return "obstacle"
                    obstacle = self._depth_obstacle_check(depth_map,
                                                            drive_angle)
                    if obstacle:
                        obstacle_detected = True
                        steer_dir = self._depth_steer_direction(depth_map)

                # YOLO obstacle check — cross-validate with depth map
                # If depth says corridor is clear, YOLO ghost → ignore
                depth_clear = (depth_map is not None
                               and not obstacle_detected)
                dets, _, age = self._get_detections()
                if age < 1.0 and dets:
                    # Shift corridor check based on current drive angle
                    angle_shift = _clamp(drive_angle / 90.0 * 0.3,
                                         -0.15, 0.15)
                    path_center = 0.5 + angle_shift
                    for d in dets:
                        # Skip classes that aren't real ground-level obstacles
                        if d["name"] in NAV_IGNORE_CLASSES or d["name"] in _nav_learned_ignore:
                            continue
                        # Correct for fisheye distortion
                        real_cx = _fisheye_cx(d["cx"])
                        real_bw = _fisheye_bw(d["bw"], d["cx"])
                        # Object edges in corrected coordinates
                        obj_left = real_cx - real_bw / 2
                        obj_right = real_cx + real_bw / 2
                        # Rover corridor: ~26cm wide → ~22% of FOV at 0.5m
                        rover_left = path_center - 0.22
                        rover_right = path_center + 0.22
                        overlaps_path = (obj_right > rover_left
                                         and obj_left < rover_right)

                        has_real_dist = d.get("dist_m") is not None
                        # Emergency stop: overlaps path AND very close
                        dangerously_close = (
                            d.get("dist_m", 999) < 0.15
                            or (real_bw > 0.50 and d["bh"] > 0.30))
                        if overlaps_path and dangerously_close:
                            self._stop_wheels()
                            self._log(f"EMERGENCY stop: '{d['name']}' "
                                  f"(bw={d['bw']:.2f}→{real_bw:.2f}, "
                                  f"cx={d['cx']:.2f}→{real_cx:.2f}"
                                  f", dist={d.get('dist_m', '?')})")
                            return "obstacle"
                        # Steer around: object in path and close
                        close = (real_bw > 0.15 or
                                 d.get("dist_m", 999) < 0.40)
                        if overlaps_path and close:
                            if depth_clear and not has_real_dist:
                                continue
                            obstacle_detected = True
                            if real_cx < path_center:
                                steer_dir = "right"
                            else:
                                steer_dir = "left"
                            break

                clear_ahead = self._estimate_clear_distance(
                    depth_map, dets if age < 1.0 else [], drive_angle)
                if clear_ahead is not None:
                    clear_history.append((time.time(), clear_ahead))
                    imminent_reason = self._predict_imminent_blockage(
                        clear_ahead, remaining_dist, clear_history)
                    if imminent_reason and not obstacle_detected:
                        obstacle_detected = True
                        if depth_map is not None:
                            steer_dir = self._depth_steer_direction(depth_map)
                        self._log(f"Imminent blockage: {imminent_reason}")

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
                        self._log("Blind drive: obstacle dead center, "
                              "stopping for LLM replan")
                        return "obstacle"
                    if not steering:
                        avoidance_count += 1
                        self._log(f"Avoiding obstacle: steer {steer_dir}")
                    steering = True
                elif steering:
                    # Was steering, obstacle cleared → resume drive angle
                    self.rover.send({"T": 1, "L": round(base_L, 3),
                                     "R": round(base_R, 3)})
                    steering = False

                # Periodic snapshot during drive
                self._maybe_snapshot()

                # Encoder distance tracking: integrate speed to measure distance
                if hasattr(self.rover, 'read_imu'):
                    imu = self.rover.read_imu()
                    if imu and "L" in imu and "R" in imu:
                        now = time.time()
                        enc_spd = (abs(float(imu["L"])) +
                                   abs(float(imu["R"]))) / 2.0
                        if enc_last_time is not None:
                            dt = now - enc_last_time
                            if 0 < dt < 1.0:
                                enc_distance += enc_spd * dt
                                enc_has_data = True
                        enc_last_time = now

                # Mid-drive prefetch: submit while still moving, timed so the
                # LLM answer lands close to the planned stop instead of long
                # before or after it.
                progress = enc_distance if enc_has_data else (
                    (time.time() - t_start) * DRIVE_SPEED)
                if (not prefetch_submitted and prefetch_pool is not None
                        and prefetch_fn is not None):
                    remaining_dist = max(0.0, distance_m - progress)
                    remaining_time = remaining_dist / max(DRIVE_SPEED, 0.01)
                    lead_time = self._prefetch_lead_time()
                    if remaining_time <= lead_time:
                        submit_args = tuple(prefetch_args or ())
                        self._mid_drive_future = prefetch_pool.submit(
                            prefetch_fn, *submit_args)
                        self._log(
                            f"Prefetch next LLM call at T-{remaining_time:.1f}s "
                            f"(lead={lead_time:.1f}s, "
                            f"rtt_avg={self._llm_rtt_ema_s:.1f}s)")
                        prefetch_submitted = True

            self._stop_wheels()
            if enc_has_data:
                self._log(f"Drive done: encoder={enc_distance:.2f}m, "
                          f"target={distance_m:.2f}m")
            self._move_gimbal(self.pose.cam_pan, 0)
            self._last_drive_avoidances = avoidance_count
            if avoidance_count >= MAX_DRIVE_AVOIDANCES:
                self._log(f"Path heavily obstructed "
                      f"({avoidance_count} avoidance events)")
                return "obstacle"
            # Re-orient using visual anchors if avoidance shifted heading
            if avoidance_count > 0 and visual_anchors:
                time.sleep(0.3)  # let camera settle
                self._reorient_to_anchors(visual_anchors)
            return "ok"
        finally:
            self._stop_wheels()
            self._move_gimbal(self.pose.cam_pan, 0)

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

            # Collect YOLO (positions only — LLM identifies from image)
            dets, det_summary, age = self._get_detections()
            yolo_part = ""
            if age < 1.0 and dets:
                yolo_part = f"YOLO: {det_summary}"
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
            self._log(f"Scan {label}: {llm_part or yolo_part or 'no data'}")

        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)
        result = "Initial room scan:\n" + "\n".join(summaries)
        self._log(f"Panoramic scan complete")
        return result

    def _directed_scan(self, target, plan_context):
        """Quick scan toward the most likely direction of the first target.

        Instead of a full panoramic scan (5 positions × LLM call each),
        takes one look at center, asks the LLM which direction the target
        is most likely in, then turns the gimbal there so the first
        waypoint iteration starts facing the right way.
        """
        # 1. Look at center and ask LLM where the target probably is
        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)
        jpeg = self._get_annotated_jpeg()
        if not jpeg:
            self._log("Directed scan: no camera frame")
            return

        result = self._llm_call(
            f"I need to navigate to: '{target}'\n"
            f"Plan context:\n{plan_context}\n\n"
            f"Based on what you see, which direction should I look to find "
            f"the target or the path toward it?\n"
            f"Reply ONLY JSON: "
            f'{{"pan": <degrees -130 to 130>, '
            f'"reason": "brief explanation"}}')

        if not result or "pan" not in result:
            self._log("Directed scan: LLM gave no direction, staying center")
            return

        pan = _clamp(int(result["pan"]), -130, 130)
        reason = result.get("reason", "")
        self._log(f"Directed scan: look {pan}° — {reason}")

        # 2. Turn gimbal to the suggested direction
        if abs(pan) > 10:
            self._move_gimbal(pan, 0)
            time.sleep(GIMBAL_SETTLE_S)

    def _room_map_context(self, max_objects=6):
        if not self.room_map:
            return ""
        rx = self.pose.x if hasattr(self.pose, 'x') else 0
        ry = self.pose.y if hasattr(self.pose, 'y') else 0
        yaw = self.pose.body_yaw if hasattr(self.pose, 'body_yaw') else 0
        data = self.room_map.room_json(rx, ry, yaw, max_objects=max_objects)
        if not data:
            return ""
        return f"ROOM_MAP: {json.dumps(data)}\n"

    def _llm_assess_leg(self, instruction, step, max_steps, jpeg,
                        yolo_summary, clear_dist, obstacle_info="",
                        room_ctx=None):
        cues = instruction.get("visual_cues", [])
        exit_hint = instruction.get("exit_hint", "find the exit")
        target_room = instruction.get("target_room", "next room")
        expected_floor = instruction.get("expected_floor", "")
        azimuth = instruction.get("expected_azimuth_deg")
        room_hints = instruction.get("room_nav_hints", "")
        verify_feats = instruction.get("verify_features", [])
        doorway_landmarks = instruction.get("doorway_landmarks", [])
        inside_features = instruction.get("inside_features", [])
        relationship_hint = instruction.get("relationship_hint", "")

        cues_str = ", ".join(cues) if cues else "a doorway"
        azimuth_hint = ""
        if azimuth is not None:
            azimuth_hint = (f"The doorway is approximately {azimuth}° from "
                            f"your entry heading "
                            f"({'right' if azimuth > 0 else 'left'}). ")

        depth_info = (f"Depth: {clear_dist:.1f}m clear ahead.\n"
                      if clear_dist else "")
        room_ctx = room_ctx if room_ctx is not None else self._room_map_context(6)

        room_verify = ""
        expected_inside = _clean_phrase_list(
            list(inside_features or []) + list(verify_feats or []), limit=6)
        if expected_inside or expected_floor:
            parts = []
            if expected_floor:
                parts.append(f"{expected_floor} floor")
            parts.extend(expected_inside)
            room_verify = (
                f"VERIFY: The correct doorway leads to {target_room} "
                f"which contains: {', '.join(parts)}. "
                f"Look THROUGH the doorway to confirm you see these "
                f"elements inside before entering.")

        relationship_ctx = ""
        rel_parts = []
        if doorway_landmarks:
            rel_parts.append(
                f"Near the correct doorway from here: {', '.join(doorway_landmarks[:3])}.")
        if inside_features:
            rel_parts.append(
                f"Just inside or through it you may see: {', '.join(inside_features[:4])}.")
        if relationship_hint:
            rel_parts.append(f"RELATIONSHIP RULE: {relationship_hint}")
        if self._last_transition_peek:
            peek = self._last_transition_peek
            peek_room = peek.get("room_guess") or "unknown"
            peek_feats = ", ".join(peek.get("inside_features", [])[:3])
            peek_reason = peek.get("reason", "")
            rel_parts.append(
                f"LAST DOOR PEEK: likely {peek_room}"
                + (f" with {peek_feats}" if peek_feats else "")
                + (f" ({peek_reason})" if peek_reason else ""))
        if rel_parts:
            relationship_ctx = "\n".join(rel_parts) + "\n"

        if step < 3:
            mission = (
                f"MISSION: Look around and find the exit from this room. "
                f"I'm looking for: {cues_str}\n"
                f"GUIDANCE: {exit_hint}\n"
                f"{azimuth_hint}"
                f"{'ROOM HINT: ' + room_hints + chr(10) if room_hints else ''}"
                f"{relationship_ctx}"
                f"{room_verify}\n\n"
                f"Describe what you ACTUALLY see. If the exit is not "
                f"visible, turn to search for it. When you see a doorway "
                f"candidate, inspect what is near it and what is visible "
                f"inside before entering. Don't drive toward something "
                f"you can't see yet."
            )
        else:
            mission = (
                f"MISSION: Navigate toward and cross through: {cues_str}\n"
                f"GUIDANCE: {exit_hint}\n"
                f"{azimuth_hint}"
                f"{'ROOM HINT: ' + room_hints + chr(10) if room_hints else ''}"
                f"{relationship_ctx}"
                f"{room_verify}"
            )

        prompt = (
            f"I'm a small ground rover (26cm wide, 22mm ground clearance) "
            f"navigating indoors. Step {step+1}/{max_steps}.\n\n"
            f"{mission}\n\n"
            f"ALWAYS CHECK:\n"
            f"1. Can you see the target ({cues_str})? Report in 'target_visible'.\n"
            f"2. Am I stuck? (same view as before, blocked by obstacle, "
            f"driving into wall/furniture, not making progress). "
            f"Report in 'stuck'.\n\n"
            f"{obstacle_info}"
            f"YOLO: {yolo_summary}\n"
            f"{depth_info}"
            f"{room_ctx}\n"
            f"Look at this image. What do you see?\n"
            f"Actions:\n"
            f"- 'drive': drive toward the doorway. angle=-30..+30, "
            f"distance=0.4-2.4m. Use LONG distance (0.8-2.4m) when "
            f"the path is clear. Keep SHORT (0.4-0.7m) only near "
            f"obstacles. Prefer decisive forward progress through visible open "
            f"space instead of repeated tiny turns.\n"
            f"- 'turn': rotate to look/face a different direction. "
            f"degrees=-180..+180\n"
            f"- 'crossed': I have passed through the doorway and am "
            f"now in the next room.\n"
            f"- 'stuck': I appear stuck (same view, blocked, no progress). "
            f"Will trigger recovery maneuver.\n\n"
            f"Reply ONLY JSON:\n"
            f'{{"scene": "what you see", '
            f'"target_visible": true/false, '
            f'"stuck": true/false, '
            f'"action": "drive"|"turn"|"crossed"|"stuck", '
            f'"angle": <drive angle>, "distance": <meters>, '
            f'"turn_degrees": <degrees>, '
            f'"confidence": <0-1 how sure about crossing>, '
            f'"reason": "why", '
            f'"yolo_corrections": {{"wrong": "correct_or__false"}}}}'
        )
        return self._llm_call(prompt, jpeg)

    def _prefetch_leg_assessment(self, instruction, step, max_steps):
        try:
            jpeg = self._get_annotated_jpeg()
            if not jpeg:
                return None
            dets, yolo_summary, _ = self._get_detections()
            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            clear_dist = self._estimate_clear_distance(
                depth_map, dets) if depth_map is not None else None
            return self._llm_assess_leg(
                instruction, step, max_steps, jpeg, yolo_summary,
                clear_dist, room_ctx=self._room_map_context(6))
        except Exception as e:
            self._log(f"Leg prefetch error: {e}")
            return None

    def _llm_assess_reactive(self, goal, step, max_steps, jpeg, yolo_summary,
                             clear_dist, obstacle_info="", room_ctx=None):
        depth_info = (f"Depth: {clear_dist:.1f}m clear ahead.\n"
                      if clear_dist else "")
        room_ctx = room_ctx if room_ctx is not None else self._room_map_context(8)
        prompt = (
            f"I'm a small ground rover (26cm wide, 22mm ground clearance) "
            f"navigating indoors. Goal: '{goal}'.\n"
            f"Step {step+1}/{max_steps}. "
            f"{obstacle_info}"
            f"YOLO detections: {yolo_summary}\n"
            f"{depth_info}"
            f"{room_ctx}"
            f"ALWAYS CHECK:\n"
            f"1. Can you see the goal ('{goal}')? Report in 'target_visible'.\n"
            f"2. Am I stuck? (same view as before, blocked by obstacle, "
            f"driving into wall/furniture, not making progress). "
            f"Report in 'stuck'.\n\n"
            f"Look at this image. Pick the BEST action:\n"
            f"- 'drive': drive toward goal. Set angle (-30 to +30) and "
            f"distance (0.4-2.4m). Use LONG (0.8-2.4m) when path is "
            f"clear, SHORT (0.4-0.7m) only near obstacles. Prefer a "
            f"committed move through open space over another small turn.\n"
            f"- 'turn': rotate to face a better direction. "
            f"Set degrees (-180 to +180).\n"
            f"- 'arrived': I'm at or inside the goal.\n"
            f"- 'stuck': I appear stuck (same view, blocked, no progress). "
            f"Will trigger recovery maneuver.\n\n"
            f"YOLO labels are often WRONG. If any YOLO label is incorrect, "
            f"add 'yolo_corrections' to fix them. Use '_false' for "
            f"hallucinated objects, or the correct name.\n"
            f"Reply ONLY JSON:\n"
            f'{{"scene": "brief description", '
            f'"target_visible": true/false, '
            f'"stuck": true/false, '
            f'"action": "drive"|"turn"|"arrived"|"stuck", '
            f'"angle": <drive angle degrees>, '
            f'"distance": <meters>, '
            f'"turn_degrees": <degrees if turning>, '
            f'"reason": "why", '
            f'"yolo_corrections": {{"wrong_label": "correct_or__false"}}}}'
        )
        return self._llm_call(prompt, jpeg)

    def _prefetch_reactive_assessment(self, goal, step, max_steps):
        try:
            jpeg = self._get_annotated_jpeg()
            if not jpeg:
                return None
            dets, yolo_summary, _ = self._get_detections()
            depth_map = (self.tracker.get_depth_map()
                         if hasattr(self.tracker, 'get_depth_map') else None)
            clear_dist = self._estimate_clear_distance(
                depth_map, dets) if depth_map is not None else None
            return self._llm_assess_reactive(
                goal, step, max_steps, jpeg, yolo_summary,
                clear_dist, room_ctx=self._room_map_context(8))
        except Exception as e:
            self._log(f"Reactive prefetch error: {e}")
            return None

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
        if consecutive_turns >= FORCED_TURN_LIMIT:
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
        # Detect surface/wall at medium distance via far vs near depth strips
        depth_map = (self.tracker.get_depth_map()
                     if hasattr(self.tracker, 'get_depth_map') else None)
        if depth_map is not None:
            h, w = depth_map.shape[:2]
            far_strip = depth_map[int(h*0.2):int(h*0.4), int(w*0.3):int(w*0.7)]
            near_strip = depth_map[int(h*0.6):h, int(w*0.3):int(w*0.7)]
            far_mean = float(np.mean(far_strip))
            near_mean = float(np.mean(near_strip))
            if far_mean > near_mean * 1.3:
                depth_info += ("WARNING: Surface/wall visible ahead at medium "
                               "distance. Do NOT drive forward — turn to find "
                               "a clear path.\n")
                self._log(f"Depth: wall/surface ahead "
                      f"(far={far_mean:.2f} > near={near_mean:.2f}×1.3)")
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

        room_ctx = room_context.format_room_clues(target_room=target)

        # 3D room map context — structured JSON for navigation
        room_map_ctx = ""
        if self.room_map:
            rx = self.pose.x if hasattr(self.pose, 'x') else 0
            ry = self.pose.y if hasattr(self.pose, 'y') else 0
            yaw = self.pose.body_yaw if hasattr(self.pose, 'body_yaw') else 0
            nav_data = self.room_map.nav_json(target, rx, ry, yaw)
            if nav_data:
                room_map_ctx = f"TARGET_MAP: {json.dumps(nav_data)}\n"
            else:
                room_data = self.room_map.room_json(rx, ry, yaw, max_objects=10)
                if room_data:
                    room_map_ctx = f"ROOM_MAP: {json.dumps(room_data)}\n"

        if has_plan:
            strategy = (
                f"FOCUS on reaching your CURRENT STEP target. "
                f"The orchestrator handles the big picture — "
                f"you just need to get to '{target}'.\n"
                f"For drive_forward: set drive_angle to aim toward your target "
                f"(negative=left, positive=right, 0=straight). "
                f"If a corridor is visibly open, commit to it.\n")
        else:
            strategy = (
                f"THINK STRATEGICALLY: Where is the door/exit? Plan a route.\n"
                f"Subtasks must be SPATIAL (e.g. 'reach the doorway on the left').\n"
                f"For drive_forward: set drive_angle to aim toward your target "
                f"(negative=left, positive=right, 0=straight). "
                f"You can drive at an angle to follow a path near obstacles. "
                f"Prefer bold forward progress through any clearly open corridor.\n")

        # Target bearing hint — tells LLM where target was last seen
        target_bearing_hint = ""
        if self._target_world_bearing is not None:
            relative = self._target_world_bearing - self.pose.body_yaw
            # Normalize to -180..180
            relative = ((relative + 180) % 360) - 180
            if abs(relative) < 10:
                target_bearing_hint = "Target was last seen STRAIGHT AHEAD.\n"
            elif abs(relative) > 20:
                direction = "RIGHT" if relative > 0 else "LEFT"
                target_bearing_hint = (
                    f"** TARGET BEARING: ~{abs(relative):.0f}° to your "
                    f"{direction} ** If you can't see the target in frame, "
                    f"turn {direction.lower()} ~{abs(relative):.0f}° to face it. "
                    f"Do NOT wander in other directions.\n")
            else:
                direction = "RIGHT" if relative > 0 else "LEFT"
                target_bearing_hint = (
                    f"Target bearing: ~{abs(relative):.0f}° to your {direction}. "
                    f"Set drive_angle={int(relative)} to aim toward it.\n")

        prompt = (
            f"I'm a 6-wheel ground rover (28cm long, 26cm wide, 32cm tall, "
            f"camera at ~30cm height, 22mm ground clearance). "
            f"Wide-angle fisheye camera — objects near frame edges appear "
            f"stretched/larger and closer than they are. "
            f"Trust center of frame more than edges. Goal: '{target}'.\n"
            f"{target_bearing_hint}"
            f"{plan_ctx}"
            f"{scan_ctx}"
            f"{subtask_ctx}"
            f"{turn_warning}"
            f"{obstacle_warning}"
            f"YOLO: {yolo_summary}\n"
            f"{depth_info}"
            f"{explore_ctx}"
            f"{room_ctx}"
            f"{room_map_ctx}"
            f"{mem}\n\n"
            f"{strategy}\n"
            f"Reply ONLY JSON:\n"
            f'{{"target_visible": bool, '
            f'"scene": "what you see + where the exit/door likely is", '
            f'"action": "arrived"/"drive_forward"/"turn_left"/"turn_right"/'
            f'"approach_target"/"subtask", '
            f'"drive_angle": <degrees offset while driving: -30 to +30, 0=straight>, '
            f'"drive_distance": <meters to drive, 0.4-2.4>, '
            f'"turn_degrees": <number if turning>, '
            f'"subtask": "<specific spatial goal>", '
            f'"subtask_reason": "<why>", '
            f'"subtask_achieved": false, '
            f'"yolo_corrections": {{"wrong_label": "correct_or__false"}}}}\n'
            f'For drive_distance: I have obstacle avoidance and encoder-measured stopping. '
            f'If the path ahead is clear, drive the full distance (up to 2.4m). '
            f'If cluttered or uncertain, keep it short (0.4-0.7m). '
            f'Do not waste steps on timid micro-moves when a route is visibly open.\n'
            f'Use "arrived" when the target fills most of the frame or '
            f'you are at/inside it (doorway, room, area). '
            f'Use "approach_target" only for YOLO-detectable objects.\n'
            f'If any YOLO label is wrong, add yolo_corrections. '
            f'Use "_false" for hallucinations.'
        )
        result = self._llm_call(prompt, jpeg)
        if result:
            scene = result.get("scene", "")
            if scene:
                self._last_scene = scene
                self._store_observation(0, 0, llm_obs={
                    "objects": [], "obstacles": [],
                    "open_space": "center" if result.get("action") == "drive_forward" else "none"
                })
                room_name, _ = room_context.get_current_room(scene, yolo_summary)
                # Process YOLO corrections from waypoint LLM
                yolo_corr = result.get("yolo_corrections")
                if isinstance(yolo_corr, dict) and yolo_corr:
                    for wrong, correct in yolo_corr.items():
                        if correct == "_false":
                            _nav_learned_ignore.add(wrong.lower().strip())
                            self._log(f"YOLO ignore: '{wrong}' (LLM says false)")
                    if self._yolo_correction_fn:
                        self._yolo_correction_fn(yolo_corr)
                # Feed YOLO detections into room map
                if (self.room_map and self.detector and
                        self.detector.last_detections):
                    self.room_map.record(
                        self.detector.last_detections,
                        self.pose.x if hasattr(self.pose, 'x') else 0,
                        self.pose.y if hasattr(self.pose, 'y') else 0,
                        self.pose.body_yaw, self.pose.cam_pan,
                        self.pose.cam_tilt)
                # Log to journey if plan is active
                if getattr(self, '_plan', None) is not None:
                    heading = (self.pose.body_yaw
                               if hasattr(self.pose, 'body_yaw') else 0)
                    self._plan.log_observation(
                        scene, yolo=yolo_summary,
                        room=room_name or "", heading=heading)
        return result

    # ── YOLO P-Control Final Approach ────────────────────────────────────

    def _drive_to_target(self, target):
        """YOLO P-control approach (no LLM).  For final approach only.
        Returns: "arrived", "lost", or "stuck"."""
        arrive_count = 0
        lost_count = 0
        enc_stuck_count = 0
        wheels_on_time = 0.0  # when wheels were last commanded on
        loop_period = 1.0 / DRIVE_LOOP_HZ

        self._log(f"P-control approach to '{target}'")

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
                        if d["name"] in NAV_IGNORE_CLASSES or d["name"] in _nav_learned_ignore:
                            continue
                        in_path = 0.15 < d["cx"] < 0.85
                        close = (d["bw"] > 0.30 or
                                 d.get("dist_m", 999) < 0.40)
                        if in_path and close:
                            self._stop_wheels()
                            wheels_on_time = 0.0
                            blocked = True
                            break

                    if not blocked:
                        err_x = cx - 0.5
                        steer = _clamp(err_x * STEER_GAIN, -MAX_STEER,
                                       MAX_STEER)
                        L = DRIVE_SPEED + steer
                        R = DRIVE_SPEED - steer
                        if wheels_on_time == 0.0:
                            wheels_on_time = time.time()
                        self.rover.send({"T": 1, "L": round(L, 3),
                                         "R": round(R, 3)})
                else:
                    lost_count += 1
                    if lost_count > LOST_FRAMES:
                        self._stop_wheels()
                        return "lost"

                # Encoder-based stuck detection DISABLED — encoders report 0
                # with current firmware. Re-enable after confirming T:1001
                # L/R fields show actual wheel speeds.
                pass  # was: encoder stuck check (p-control)

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
                self._log(f"YOLO found '{target}' immediately "
                      f"(cx={hit['cx']:.2f})")
                return {"gimbal_pan": self.pose.cam_pan,
                        "source": "yolo", "det": hit}

        # Scan front hemisphere
        result = self._scan_positions(target, SCAN_POSITIONS)
        if result:
            return result

        # Rotate 180° and scan rear
        self._log("Front scan exhausted, rotating 180°")
        self._spin_body(180)
        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)

        result = self._scan_positions(target, SCAN_POSITIONS)
        if result:
            return result

        self._log(f"Could not find '{target}' in 360° scan")
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
                    self._log(f"YOLO found '{target}' at pan={pan}° "
                          f"(cx={hit['cx']:.2f}, bw={hit['bw']:.2f})")
                    self._store_observation(pan, tilt, dets=dets)
                    return {"gimbal_pan": pan, "source": "yolo", "det": hit}
                if dets:
                    self._store_observation(pan, tilt, dets=dets)

            obs = self._llm_observe(target, jpeg, pan, tilt)
            if obs and obs.get("found"):
                self._log(f"LLM found '{target}' at pan={pan}°")
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

    def _area_escape(self):
        """3-point maneuver to escape a stuck area, then relocate target.

        Like a driver doing a 3-point turn:
          1. Reverse hard to create space
          2. Turn to face a new direction
          3. Drive forward into open space
        Then: gimbal sweep + LLM call to reacquire the target doorway.
        """
        target = getattr(self, '_nav_target', 'exit')
        self._log(f"3-POINT ESCAPE: reverse → turn → drive → relocate '{target}'")
        self._stop_wheels()
        self._move_gimbal(0, 0)
        time.sleep(0.2)

        # 1. Reverse hard to create space
        # Negative values = forward after _send_raw swap, so positive = reverse
        self._log("  1/3: reversing hard")
        self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
        if not self._sleep_abortable(AREA_ESCAPE_REVERSE_S):
            return
        self._stop_wheels()
        time.sleep(0.3)

        # 2. Turn hard into a fresh heading (alternating direction)
        escape_dir = (AREA_ESCAPE_TURN_DEG
                      if (getattr(self, '_escape_dir', 1) > 0)
                      else -AREA_ESCAPE_TURN_DEG)
        self._escape_dir = -getattr(self, '_escape_dir', 1)
        self._log(f"  2/3: turning {escape_dir}°")
        self._spin_body(escape_dir)
        self._move_gimbal(0, 0)
        time.sleep(0.3)

        # 3. Drive forward assertively into the new opening
        self._log(f"  3/3: driving {FORCED_EXPLORE_DRIVE_M:.1f}m forward")
        self._blind_drive(FORCED_EXPLORE_DRIVE_M, 0)
        self._stop_wheels()
        time.sleep(0.3)

        self._subtask_stack.clear()

        # Relocate: gimbal sweep to find the target again
        self._log(f"Relocating '{target}'...")
        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)

        # Scan 5 directions with gimbal, then ask LLM which way
        for pan in [-120, -60, 0, 60, 120]:
            if self._aborted():
                return
            self._move_gimbal(pan, 0)
            time.sleep(0.3)

        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)
        jpeg = self._get_annotated_jpeg()
        if not jpeg:
            return

        result = self._llm_call(
            f"I just did a 3-point turn to escape being stuck. "
            f"I need to find: '{target}'. "
            f"Look at this image — where should I go now? "
            f"Look for doorways, exits, bright openings, or the target.\n"
            f"Reply JSON: {{\"best_direction\": <degrees, "
            f"negative=left, positive=right, 0=straight ahead>, "
            f"\"reason\": \"what you see\"}}", jpeg)

        if result and "best_direction" in result:
            best = int(result["best_direction"])
            best = max(-180, min(180, best))
            reason = result.get('reason', '')[:60]
            self._log(f"Reacquired: turn {best}° — {reason}")
            if abs(best) > 10:
                self._spin_body(best)
                self._move_gimbal(0, 0)
        else:
            self._log("Could not relocate target — continuing forward")

    def _recover_stuck(self, target, encoder_stuck=False):
        """Mini 3-point maneuver to clear an obstacle, then reacquire target.

        1. Reverse to create space
        2. Turn away from blocked direction
        3. Gimbal scan to find the target again and face it
        """
        blocked_angle = getattr(self, '_last_drive_angle', 0)

        if encoder_stuck:
            self._log("Recovering from physical stuck — reversing")
        else:
            self._log(f"Recovering — obstacle at {blocked_angle}° angle")

        # 1. Reverse to create space
        # Negative values = forward after _send_raw swap, so positive = reverse
        self._move_gimbal(0, 0)
        self.rover.send({"T": 1, "L": -0.10, "R": -0.10})
        if not self._sleep_abortable(RECOVER_REVERSE_S):
            return
        self._stop_wheels()
        time.sleep(0.2)

        # 2. Turn farther away from the blocked direction.
        turn = RECOVER_TURN_DEG if blocked_angle <= 0 else -RECOVER_TURN_DEG
        self._log(f"Turning {turn}° away from obstacle")
        self._spin_body(turn)
        self._move_gimbal(0, 0)
        time.sleep(0.2)

        # 3. Gimbal scan to reacquire target
        target_name = target or getattr(self, '_nav_target', 'exit')
        for pan in [-90, -45, 0, 45, 90]:
            if self._aborted():
                return
            self._move_gimbal(pan, 0)
            time.sleep(0.25)

        self._move_gimbal(0, 0)
        time.sleep(GIMBAL_SETTLE_S)
        jpeg = self._get_annotated_jpeg()
        if jpeg and target_name:
            result = self._llm_call(
                f"I just backed up and turned to avoid an obstacle. "
                f"I need to get to: '{target_name}'. "
                f"Which direction has the best path toward it? "
                f"Look for doorways, open space, bright light.\n"
                f"Reply JSON: {{\"best_direction\": <degrees, "
                f"neg=left, pos=right>, \"reason\": \"why\"}}",
                jpeg)
            if result and "best_direction" in result:
                best = int(result["best_direction"])
                best = max(-180, min(180, best))
                self._log(f"Reacquired: {best}° — "
                          f"{result.get('reason', '')[:50]}")
                if abs(best) > 10:
                    # Turn body to face the target, not just gimbal
                    self._spin_body(best)
                    self._move_gimbal(0, 0)
            else:
                self._log("Could not reacquire — continuing")

    def _try_floor_nav_escape(self, doorway_mode=False):
        """Use FloorNavigator to move into the best available opening.

        Returns True when FloorNavigator made forward progress, False when it
        found no usable route or is unavailable.
        """
        if self._aborted() or self.floor_nav is None:
            return False

        mode = "doorway" if doorway_mode else "widest-gap"
        self._log(f"FloorNavigator recovery: {mode}")
        self._stop_wheels()
        self._move_gimbal(0, 0)
        time.sleep(0.2)

        moved = False
        try:
            if doorway_mode:
                moved = bool(self.floor_nav.drive_through_opening(
                    timeout=DOORWAY_ESCAPE_TIMEOUT_S,
                    speed=FLOOR_ESCAPE_SPEED))
                if not moved:
                    self._log("FloorNavigator doorway pass failed; trying widest gap")
                    moved = bool(self.floor_nav.drive_toward(
                        target_direction="center",
                        speed=FLOOR_ESCAPE_SPEED,
                        timeout=FLOOR_ESCAPE_TIMEOUT_S,
                        prefer_widest=True))
            else:
                moved = bool(self.floor_nav.drive_toward(
                    target_direction="center",
                    speed=FLOOR_ESCAPE_SPEED,
                    timeout=FLOOR_ESCAPE_TIMEOUT_S,
                    prefer_widest=True))
        except Exception as e:
            self._log(f"FloorNavigator recovery error: {e}")
            moved = False
        finally:
            self._stop_wheels()
            self._move_gimbal(0, 0)
            time.sleep(GIMBAL_SETTLE_S)

        if moved:
            self._log("FloorNavigator recovery succeeded")
            self._last_drive_angle = 0
        else:
            self._log("FloorNavigator recovery found no usable opening")
        return moved

    def _llm_verify_obstacle(self):
        """Quick LLM check: is there actually an obstacle ahead?
        Returns True if obstacle confirmed, False if path looks clear."""
        result = self._llm_call(
            "I stopped because my sensors detected an obstacle ahead. "
            "Look at this image: is there ANYTHING blocking my forward path "
            "within 0.5m? This includes walls, furniture, doors, bags, "
            "large surfaces, or any solid object. "
            "A wall or surface filling most of the frame = YES, obstacle. "
            "Ignore objects that are far away or clearly to the side.\n"
            "Reply ONLY JSON: "
            '{"obstacle": true/false, "what": "brief description"}')
        if result is None:
            return True  # LLM unavailable — assume obstacle
        blocked = result.get("obstacle", True)
        what = result.get("what", "?")
        self._log(f"LLM obstacle check: {'YES' if blocked else 'NO'} — {what}")
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
        self._log(f"Subtask pushed: '{goal}' ({reason})")
        self._say(f"Looking for {goal} first")

    def _pop_subtask(self):
        if self._subtask_stack:
            done = self._subtask_stack.pop()
            self._log(f"Subtask achieved: '{done}'")
            if self._current_subtask():
                self._log(f"Back to: '{self._current_subtask()}'")
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
        self._log(f"Aligning body to gimbal (pan={pan:.0f}°)")
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
            started = time.time()
            raw = self.llm_vision(full_prompt, jpeg)
            self._record_llm_rtt(time.time() - started)
            self._consecutive_429s = 0
            return self.parse(raw)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                self._consecutive_429s = getattr(self, '_consecutive_429s', 0) + 1
                wait = min(LLM_429_BACKOFF_S * self._consecutive_429s, 15.0)
                self._log(f"Rate limited (429), waiting {wait:.0f}s "
                          f"(#{self._consecutive_429s})")
                time.sleep(wait)
            else:
                self._log(f"LLM call error: {e}")
            return None

    def _peek_transition_view(self, instruction, jpeg=None):
        """Infer what room seems to lie beyond a doorway candidate."""
        target_room = instruction.get("target_room", "next room")
        cues = instruction.get("visual_cues", [])
        doorway_landmarks = instruction.get("doorway_landmarks", [])
        inside_features = instruction.get("inside_features", [])
        expected_floor = instruction.get("expected_floor", "")
        relationship_hint = instruction.get("relationship_hint", "")

        prompt = (
            f"You are close to a doorway candidate but have NOT crossed it yet.\n"
            f"Goal room beyond the doorway: {target_room}.\n"
            f"Doorway cues: {', '.join(cues) if cues else 'doorway'}.\n"
            f"Nearby doorway landmarks from this side: "
            f"{', '.join(doorway_landmarks) if doorway_landmarks else 'unknown'}.\n"
            f"Expected features inside: "
            f"{', '.join(inside_features) if inside_features else 'unknown'}.\n"
            f"Expected floor inside: {expected_floor or 'unknown'}.\n"
            f"{'Relationship rule: ' + relationship_hint + chr(10) if relationship_hint else ''}"
            f"Look THROUGH the opening and infer what room is likely on the other side.\n"
            f"Reply ONLY JSON:\n"
            f'{{"doorway_seen": true/false, '
            f'"room_guess": "<room name or empty>", '
            f'"confidence": <0-1>, '
            f'"doorway_landmarks": ["short phrases"], '
            f'"inside_features": ["short phrases"], '
            f'"navigational_hint": "one short relationship rule", '
            f'"reason": "brief why"}}'
        )
        result = self._llm_call(prompt, jpeg)
        if not isinstance(result, dict):
            return None
        return {
            "doorway_seen": bool(result.get("doorway_seen", False)),
            "room_guess": str(result.get("room_guess", "")).strip().lower().replace(" ", "_"),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "doorway_landmarks": _clean_phrase_list(
                result.get("doorway_landmarks", []), limit=5),
            "inside_features": _clean_phrase_list(
                result.get("inside_features", []), limit=6),
            "navigational_hint": str(result.get("navigational_hint", "")).strip()[:180],
            "reason": str(result.get("reason", "")).strip()[:160],
        }

    def _get_detections(self):
        """Get cached YOLO detections from camera.
        Suppressed when gimbal is tilted below 0° (seeing own body).
        Returns (list, summary_str, age_secs)."""
        if self.pose.cam_tilt < 0:
            return ([], "", 999)
        if hasattr(self.tracker, 'get_detections'):
            return self.tracker.get_detections()
        return ([], "nothing", 999)

    def _get_annotated_jpeg(self):
        """Get JPEG with YOLO boxes drawn (for LLM), fallback to raw."""
        if hasattr(self.tracker, 'get_overlay_jpeg'):
            return self.tracker.get_overlay_jpeg()
        return self.tracker.get_jpeg()

    def _yolo_obstruction_check(self, dets):
        """Check if a YOLO detection covers the center of the frame at a
        large size, indicating we're jammed against an object.
        Returns the obstructing label or None."""
        if not dets:
            return None
        # Frame is 640x480.  Center band: x in [160..480], y in [120..360]
        # "Large" = bbox covers >30% of frame area
        frame_area = 640 * 480
        for d in dets:
            if not hasattr(d, 'xyxy') or d.xyxy is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in d.xyxy)
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bbox_area = bw * bh
            # Large bbox centered in frame
            if (bbox_area > 0.30 * frame_area and
                    160 < cx < 480 and 120 < cy < 360):
                label = getattr(d, 'label', 'object')
                return label
        return None

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
        self._log(f"Spun {degrees:+.0f}° (timed {rotation_time:.2f}s), "
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
        self._log(f"{msg}")
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
            self._log(f"Snapshot: {fpath}")
        except Exception as e:
            self._log(f"Snapshot error: {e}")

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
