"""
Floor-aware continuous navigator.

Drives forward while analyzing the bottom portion of the camera frame for
obstacles (via existing YOLO detections).  The floor zone is divided into
columns; consecutive clear columns form passages.  The rover steers toward
the widest passage near its target direction — only stopping when no
passage is wide enough for the 25 cm body.

No new model required — piggybacks on LocalDetector (YOLO-World v2).
No occupancy grid — works directly in camera pixel space.

Usage:
    nav = FloorNavigator(rover, detector, tracker)
    nav.drive_toward("center", speed=0.15, timeout=5.0)
    nav.drive_to_object("chair", timeout=10.0)
    nav.drive_through_opening(timeout=8.0)
"""

import cv2
import numpy as np
import time
import threading


class FloorNavigator:
    """Continuous navigation with real-time obstacle avoidance.
    Uses existing YOLO detections + floor region analysis."""

    # --- Geometry ---
    ROVER_WIDTH_M = 0.25        # physical body width
    CAMERA_HFOV_DEG = 65.0      # horizontal FOV of the wide-angle USB cam
    FRAME_W = 640
    FRAME_H = 480

    # --- Floor zone ---
    FLOOR_HORIZON = 0.55        # floor region starts at 55% down the frame
    NUM_COLUMNS = 16            # clearance map resolution
    MIN_GAP_COLS = 2            # minimum consecutive clear columns to fit through

    # --- Drive ---
    DRIVE_SPEED = 0.15          # m/s default forward speed
    STEER_GAIN = 0.004          # wheel differential per pixel offset from target col
    MAX_STEER = 0.12            # max steering differential (m/s)
    OBSTACLE_CLOSE_BW = 0.50    # bbox width fraction — object is dangerously close

    # --- Arrival ---
    ARRIVE_BW = 0.40            # target object bbox width fraction = arrived
    ARRIVE_FRAMES = 4           # consecutive frames above ARRIVE_BW to confirm

    def __init__(self, rover, detector, tracker, emergency_event=None):
        """
        Args:
            rover:  RoverSerial (sends JSON commands to ESP32)
            detector:  LocalDetector (runs YOLO, provides .detect() / .find())
            tracker:  HumanTracker (owns camera, provides .get_jpeg())
            emergency_event:  threading.Event — abort immediately when set
        """
        self.rover = rover
        self.detector = detector
        self.tracker = tracker
        self._emergency = emergency_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def drive_toward(self, target_direction="center", speed=None,
                     timeout=10.0, stop_condition=None):
        """Drive forward avoiding obstacles.  ~10 Hz control loop.

        Args:
            target_direction: "center", "left", "right", or an int pixel column
            speed:  m/s forward (default DRIVE_SPEED)
            timeout:  seconds before giving up
            stop_condition:  callable() -> bool — return True when done

        Returns:
            True  if stop_condition was met
            False if blocked or timed out
        """
        speed = speed if speed is not None else self.DRIVE_SPEED
        target_col = self._direction_to_col(target_direction)
        t0 = time.time()

        while time.time() - t0 < timeout:
            if self._emergency and self._emergency.is_set():
                self._stop()
                return False

            if stop_condition and stop_condition():
                self._stop()
                return True

            frame, detections = self._sense()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            floor_obs = self._get_floor_obstacles(detections, w, h)

            # Check for dangerously close obstacles
            if self._anything_too_close(floor_obs):
                self._stop()
                print("[floor_nav] Obstacle too close — stopped")
                return False

            clearance = self._build_clearance_map(floor_obs, w)
            gap_col = self._find_best_gap(clearance, target_col)

            if gap_col is None:
                self._stop()
                print("[floor_nav] No passable gap — stopped")
                return False

            L, R = self._col_to_steer(gap_col, speed)
            self.rover.send({"T": 1, "L": round(L, 3), "R": round(R, 3)})
            time.sleep(0.1)

        self._stop()
        return False

    def drive_to_object(self, target_name, speed=None, timeout=15.0):
        """Drive toward a YOLO-detected object while avoiding obstacles.

        Combines target tracking with floor clearance checking.
        Stops when object bbox is large enough (arrived).

        Returns:
            True  if arrived at the object
            False if lost, blocked, or timed out
        """
        speed = speed if speed is not None else self.DRIVE_SPEED
        t0 = time.time()
        lost_count = 0
        arrive_count = 0
        MAX_LOST = 15  # ~1.5 s at 10 Hz

        print(f"[floor_nav] Driving to '{target_name}'")

        while time.time() - t0 < timeout:
            if self._emergency and self._emergency.is_set():
                self._stop()
                return False

            frame, detections = self._sense()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # Find the target
            target = self.detector.find(target_name, detections)
            if target is None:
                lost_count += 1
                if lost_count > MAX_LOST:
                    self._stop()
                    print(f"[floor_nav] Lost '{target_name}'")
                    return False
                self._stop()
                time.sleep(0.1)
                continue

            lost_count = 0

            # Check arrival
            if target["bw"] >= self.ARRIVE_BW:
                arrive_count += 1
                if arrive_count >= self.ARRIVE_FRAMES:
                    self._stop()
                    print(f"[floor_nav] Arrived at '{target_name}' "
                          f"(bw={target['bw']:.2f})")
                    return True
            else:
                arrive_count = 0

            # Target pixel column
            target_col = int(target["cx"] * self.NUM_COLUMNS)
            target_col = max(0, min(self.NUM_COLUMNS - 1, target_col))

            # Floor obstacle analysis
            floor_obs = self._get_floor_obstacles(detections, w, h)

            if self._anything_too_close(floor_obs):
                self._stop()
                print("[floor_nav] Obstacle too close while approaching target")
                return False

            clearance = self._build_clearance_map(floor_obs, w)
            gap_col = self._find_best_gap(clearance, target_col)

            if gap_col is None:
                self._stop()
                print("[floor_nav] Path to target blocked")
                return False

            L, R = self._col_to_steer(gap_col, speed)
            self.rover.send({"T": 1, "L": round(L, 3), "R": round(R, 3)})
            time.sleep(0.1)

        self._stop()
        print(f"[floor_nav] Timeout reaching '{target_name}'")
        return False

    def drive_through_opening(self, timeout=10.0, speed=0.12):
        """Drive through a doorway by centering in the opening.

        Uses vertical edge detection (Hough lines) to find door frame edges,
        steers toward the midpoint between them.  Falls back to the widest
        gap in the clearance map if no vertical edges are found.

        Returns:
            True  if drove through (floor zone becomes fully clear)
            False if blocked or timed out
        """
        t0 = time.time()
        clear_streak = 0
        THROUGH_FRAMES = 8  # consecutive all-clear frames = we're through

        print("[floor_nav] Driving through opening")

        while time.time() - t0 < timeout:
            if self._emergency and self._emergency.is_set():
                self._stop()
                return False

            frame, detections = self._sense()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # Try to find door frame edges via Hough lines
            midpoint_col = self._find_opening_center(frame, w, h)

            floor_obs = self._get_floor_obstacles(detections, w, h)

            if self._anything_too_close(floor_obs):
                self._stop()
                print("[floor_nav] Blocked in doorway")
                return False

            clearance = self._build_clearance_map(floor_obs, w)

            # Check if we're through (floor zone all clear)
            if all(clearance):
                clear_streak += 1
                if clear_streak >= THROUGH_FRAMES:
                    self._stop()
                    print("[floor_nav] Through the opening")
                    return True
            else:
                clear_streak = 0

            if midpoint_col is not None:
                # Steer toward the detected opening center
                target_col = midpoint_col
            else:
                # Fall back to widest gap
                gap_col = self._find_best_gap(clearance, self.NUM_COLUMNS // 2)
                if gap_col is None:
                    self._stop()
                    print("[floor_nav] No gap found in doorway")
                    return False
                target_col = gap_col

            L, R = self._col_to_steer(target_col, speed)
            self.rover.send({"T": 1, "L": round(L, 3), "R": round(R, 3)})
            time.sleep(0.1)

        self._stop()
        print("[floor_nav] Timeout in doorway")
        return False

    def check_floor_clear(self, detections, frame_w, frame_h, direction_col=None):
        """Quick check: is the floor ahead clear in the given direction?

        Useful for visual_servo to query before driving forward.

        Args:
            detections: list of detection dicts from LocalDetector
            frame_w, frame_h: frame dimensions
            direction_col: column index (0..NUM_COLUMNS-1) to check, or None for center

        Returns:
            (clear: bool, best_col: int or None)
            clear=True means the rover can drive in direction_col.
            best_col is the recommended column if direction is blocked.
        """
        if direction_col is None:
            direction_col = self.NUM_COLUMNS // 2

        floor_obs = self._get_floor_obstacles(detections, frame_w, frame_h)

        if self._anything_too_close(floor_obs):
            return False, None

        clearance = self._build_clearance_map(floor_obs, frame_w)

        # Check if the requested direction is in a passable gap
        gap_col = self._find_best_gap(clearance, direction_col)
        if gap_col is None:
            return False, None

        # Is the requested column itself clear and in a wide-enough gap?
        if clearance[direction_col]:
            # Check that there's a wide-enough gap around it
            run_start = direction_col
            run_end = direction_col
            while run_start > 0 and clearance[run_start - 1]:
                run_start -= 1
            while run_end < self.NUM_COLUMNS - 1 and clearance[run_end + 1]:
                run_end += 1
            if (run_end - run_start + 1) >= self.MIN_GAP_COLS:
                return True, direction_col

        # Direction blocked but there's an alternative
        return False, gap_col

    # ------------------------------------------------------------------
    # Internal: sensing
    # ------------------------------------------------------------------

    def _sense(self):
        """Grab a frame + run detection.  Returns (frame, detections) or (None, [])."""
        jpeg = self.tracker.get_jpeg()
        if not jpeg:
            return None, []
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            return None, []
        detections = self.detector.detect(frame)
        return frame, detections

    # ------------------------------------------------------------------
    # Internal: floor obstacle analysis
    # ------------------------------------------------------------------

    # Classes that are structural / wall-mounted — never floor obstacles
    _IGNORE_FLOOR = {
        # Structural
        "floor", "ceiling", "wall", "rug", "window", "door", "doorway",
        "archway", "light", "lamp", "light fixture",
        # Wall-mounted / elevated
        "shelf", "shelves", "cabinet", "monitor", "tv", "clock",
        "speaker", "camera", "fan",
        # Common YOLO misidentifications of doorways/walls
        "couch", "bed", "bench", "dining table",
    }

    def _get_floor_obstacles(self, detections, frame_w, frame_h):
        """Filter detections that are genuine floor-level obstacles.

        Filters out:
        - Structural/wall-mounted classes (shelves, doorways, walls)
        - Detections that barely graze the floor zone (need substantial overlap)
        - Very wide detections (>55% of frame) — likely doorways/walls, not obstacles
        - Low-confidence detections (<0.40)

        Returns list of (x1_px, x2_px, name, bbox_width_frac).
        """
        floor_y = int(frame_h * self.FLOOR_HORIZON)
        obstacles = []

        # Import label overrides to use corrected names
        try:
            from local_detector import LABEL_OVERRIDES
        except ImportError:
            LABEL_OVERRIDES = {}

        for d in detections:
            name = d["name"]

            # Apply label overrides (corrected names from LLM calibration)
            name = LABEL_OVERRIDES.get(name, name)

            # Skip structural / wall-mounted classes
            if name in self._IGNORE_FLOOR:
                continue

            # Skip low confidence (likely false positives near doorways)
            if d["conf"] < 0.40:
                continue

            x1, y1, x2, y2 = d["bbox"]
            bw_frac = (x2 - x1) / frame_w
            bh_px = y2 - y1

            # Skip very wide detections — doorways/walls, not point obstacles
            if bw_frac > 0.55:
                continue

            # Require substantial overlap with floor zone — at least 20% of
            # the bbox height must be in the floor zone (not just bottom edge)
            overlap = max(0, y2 - floor_y)
            if bh_px > 0 and overlap / bh_px < 0.20:
                continue

            # Must actually intrude into floor zone
            if y2 > floor_y:
                obstacles.append((x1, x2, name, bw_frac))

        return obstacles

    def _anything_too_close(self, floor_obstacles):
        """True if any floor obstacle has a bbox width >= OBSTACLE_CLOSE_BW."""
        for _, _, name, bw in floor_obstacles:
            if bw >= self.OBSTACLE_CLOSE_BW:
                return True
        return False

    def _build_clearance_map(self, floor_obstacles, frame_w):
        """Build column-based clearance map.

        Returns list of bools (True=clear) for each of NUM_COLUMNS columns.
        A column is blocked if any obstacle overlaps it.
        """
        col_w = frame_w / self.NUM_COLUMNS
        clearance = [True] * self.NUM_COLUMNS

        for x1, x2, name, bw in floor_obstacles:
            c_start = max(0, int(x1 / col_w))
            c_end = min(self.NUM_COLUMNS - 1, int(x2 / col_w))
            for c in range(c_start, c_end + 1):
                clearance[c] = False

        return clearance

    def _find_best_gap(self, clearance, target_col):
        """Find the widest gap of consecutive clear columns nearest to target_col.

        Returns center column of the best gap, or None if no gap is wide enough.
        """
        # Find all runs of consecutive True columns
        gaps = []
        start = None
        for i, clear in enumerate(clearance):
            if clear:
                if start is None:
                    start = i
            else:
                if start is not None:
                    gaps.append((start, i - 1))
                    start = None
        if start is not None:
            gaps.append((start, len(clearance) - 1))

        # Filter by minimum width
        passable = [(s, e) for s, e in gaps if (e - s + 1) >= self.MIN_GAP_COLS]
        if not passable:
            return None

        # Pick the gap whose center is nearest to target_col, breaking ties by width
        def score(gap):
            center = (gap[0] + gap[1]) / 2.0
            dist = abs(center - target_col)
            width = gap[1] - gap[0] + 1
            return (dist, -width)  # prefer closer, then wider

        best = min(passable, key=score)
        return int((best[0] + best[1]) / 2)

    # ------------------------------------------------------------------
    # Internal: steering
    # ------------------------------------------------------------------

    def _col_to_steer(self, target_col, speed):
        """Convert target column to (L, R) wheel speeds with proportional steering.

        Center column = straight.  Offset from center → differential steering.
        """
        center_col = self.NUM_COLUMNS / 2.0
        # Pixel-equivalent offset (map column to frame pixels for gain calc)
        col_w = self.FRAME_W / self.NUM_COLUMNS
        offset_px = (target_col - center_col) * col_w

        steer = offset_px * self.STEER_GAIN
        steer = max(-self.MAX_STEER, min(self.MAX_STEER, steer))

        L = speed + steer
        R = speed - steer
        # Clamp to safe range
        L = max(-0.3, min(0.5, L))
        R = max(-0.3, min(0.5, R))
        return L, R

    def _direction_to_col(self, direction):
        """Map a direction string or int to a column index."""
        if isinstance(direction, (int, float)):
            return max(0, min(self.NUM_COLUMNS - 1, int(direction)))
        d = str(direction).lower().strip()
        if d == "left":
            return self.NUM_COLUMNS // 4          # 25% from left
        elif d == "right":
            return 3 * self.NUM_COLUMNS // 4      # 75% from left
        else:
            return self.NUM_COLUMNS // 2           # center

    def _stop(self):
        """Send wheel stop command."""
        self.rover.send({"T": 1, "L": 0, "R": 0})

    # ------------------------------------------------------------------
    # Internal: doorway detection
    # ------------------------------------------------------------------

    def _find_opening_center(self, frame, frame_w, frame_h):
        """Detect vertical door frame edges via Hough lines.

        Returns the center column (0..NUM_COLUMNS-1) of the opening,
        or None if no clear pair of vertical edges is found.
        """
        # Work on the middle vertical band of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Look for strong vertical lines using probabilistic Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=int(frame_h * 0.3),
                                maxLineGap=20)
        if lines is None:
            return None

        # Filter for near-vertical lines (within 15° of vertical)
        vertical_xs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > 0 and dx / dy < 0.27:  # tan(15°) ≈ 0.27
                mid_x = (x1 + x2) / 2.0
                vertical_xs.append(mid_x)

        if len(vertical_xs) < 2:
            return None

        # Cluster the vertical lines into left-group and right-group
        vertical_xs.sort()
        center_x = frame_w / 2.0

        left_edges = [x for x in vertical_xs if x < center_x]
        right_edges = [x for x in vertical_xs if x >= center_x]

        if not left_edges or not right_edges:
            return None

        # Take the innermost edges (closest to center) as the door frame
        left_x = max(left_edges)
        right_x = min(right_edges)

        if right_x - left_x < frame_w * 0.08:
            # Opening too narrow to be a real doorway
            return None

        midpoint_px = (left_x + right_x) / 2.0
        col_w = frame_w / self.NUM_COLUMNS
        return int(midpoint_px / col_w)
