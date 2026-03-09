"""Continuous depth-reactive navigator with async LLM steering.

The rover drives continuously at 10Hz using depth vectors to avoid obstacles.
An async LLM thread periodically looks at the camera and sets a steering bias
(target pixel column) that the driver follows.  The rover NEVER stops to think.

Architecture:
  Driver thread (10Hz):  depth → steer toward LLM bias column → collision avoid
  LLM thread (~0.1-0.3Hz): capture → VLM call → set bias column + arrived flag
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from rover_brain_v2.navigation.depth_vectors import DepthVectorMap


# ── Constants ────────────────────────────────────────────────────────────

CONTROL_HZ = 10
DRIVE_SPEED = 0.15          # m/s — user preference
MAX_STEER = 0.12            # max wheel speed differential
STOP_CLEARANCE_M = 0.28     # emergency stop
PASSABLE_M = 0.45           # consider passable above this
REVERSE_DISTANCE_M = 0.18   # how far to back up
REVERSE_SPEED = 0.12
ESCAPE_TURN_SPEED = 0.24
ESCAPE_MIN_DEG = 55.0
ESCAPE_MAX_DEG = 100.0
STUCK_TIMEOUT_S = 3.0       # if blocked this long, reverse+escape
STALE_BIAS_S = 15.0         # ignore LLM bias if older than this
ARRIVED_CONFIRM_S = 1.0     # LLM must say arrived for this long


class ContinuousNavigator:
    """10Hz depth-reactive driver with async LLM goal steering.

    Usage:
        nav = ContinuousNavigator(rover=..., camera=..., ...)
        nav.set_goal("kitchen")   # starts LLM advisor thread
        nav.run(stop_event)       # blocks on 10Hz drive loop
        nav.arrived               # True if LLM declared arrival
    """

    def __init__(self, *, rover, camera, depth_vectors: DepthVectorMap,
                 event_bus, flags, config):
        self.rover = rover
        self.camera = camera
        self.depth_vectors = depth_vectors
        self.events = event_bus
        self.flags = flags
        self.config = config

        # LLM sets these atomically — driver reads them
        self._lock = threading.Lock()
        self._bias_col: int | None = None       # target column (0..20) in depth map
        self._bias_time: float = 0.0
        self._arrived = False
        self._arrived_time: float = 0.0
        self._scene: str = ""
        self._confidence: float = 0.0

        # Driver state
        self._blocked_since: float = 0.0
        self._escape_dir: int = 1               # alternates left/right
        self._last_steer_col: int | None = None

    @property
    def arrived(self) -> bool:
        with self._lock:
            return self._arrived

    @property
    def scene(self) -> str:
        with self._lock:
            return self._scene

    # ── LLM advisor interface (called from LLM thread) ──────────────

    def set_bias(self, *, pixel_x: int | None, arrived: bool = False,
                 scene: str = "", confidence: float = 0.5):
        """Called by LLM thread to steer the driver.

        pixel_x: target x-coordinate on the 640px image (0=left, 640=right).
                 None means "just follow depth, no preference".
        arrived: True → stop driving, goal reached.
        """
        now = time.time()
        # Convert pixel_x to depth column (0..num_columns-1)
        n = self.depth_vectors.num_columns
        if pixel_x is not None:
            col = int(round(pixel_x / 640.0 * (n - 1)))
            col = max(0, min(n - 1, col))
        else:
            col = None

        with self._lock:
            self._bias_col = col
            self._bias_time = now
            self._scene = scene
            self._confidence = confidence
            if arrived and not self._arrived:
                self._arrived = True
                self._arrived_time = now
            elif not arrived:
                self._arrived = False

        direction = f"col {col}/{n}" if col is not None else "none"
        self.events.publish(
            "drive",
            f"LLM bias: {direction}, arrived={arrived}, "
            f"conf={confidence:.1f}, scene={scene[:50]}",
        )

    # ── 10Hz driver loop ────────────────────────────────────────────

    def run(self, stop_event: threading.Event) -> bool:
        """Continuous depth-reactive driving. Returns True if arrived.

        Runs at CONTROL_HZ. The LLM sets bias asynchronously.
        This method handles all obstacle avoidance autonomously.
        """
        interval = 1.0 / CONTROL_HZ
        self._blocked_since = 0.0
        speed = getattr(self.config, "navigation_drive_speed", DRIVE_SPEED)

        while not stop_event.is_set():
            t0 = time.time()

            # Check arrived
            with self._lock:
                if self._arrived:
                    self.rover.stop()
                    self.events.publish("drive", "LLM declared arrived — stopping")
                    return True

            # Get depth
            depth_map = self.camera.get_depth_map()
            if depth_map is None:
                # No depth — crawl forward slowly
                self.rover.send({"T": 1, "L": round(speed * 0.5, 3),
                                 "R": round(speed * 0.5, 3)})
                self._sleep_tick(t0, interval)
                continue

            try:
                ds = self.depth_vectors.analyze(depth_map)
            except Exception:
                self._sleep_tick(t0, interval)
                continue

            center_d = ds.center_distance()
            corridor_d = ds.closest_corridor_distance() or center_d

            # ── BLOCKED: obstacle too close ──
            if min(center_d, corridor_d) < STOP_CLEARANCE_M:
                self.rover.stop()
                if self._blocked_since == 0.0:
                    self._blocked_since = time.time()
                    self.events.publish("drive",
                        f"Blocked: center={center_d:.2f}m, corridor={corridor_d:.2f}m")

                # If blocked for too long, reverse + escape
                if time.time() - self._blocked_since > STUCK_TIMEOUT_S:
                    self.events.publish("drive", "Stuck — reversing and escaping")
                    self._reverse_escape(ds, stop_event)
                    self._blocked_since = 0.0

                self._sleep_tick(t0, interval)
                continue

            self._blocked_since = 0.0

            # ── STEERING: blend LLM bias with depth safety ──
            target_col = self._pick_steer_column(ds)
            self._last_steer_col = target_col

            # Convert column to steering differential
            n = self.depth_vectors.num_columns
            center_col = (n - 1) / 2.0
            col_offset = (target_col - center_col) / max(center_col, 1.0)
            # Proportional steering: offset ∈ [-1, 1] → steer ∈ [-MAX_STEER, MAX_STEER]
            steer = col_offset * MAX_STEER

            left = speed + steer
            right = speed - steer
            # Clamp to avoid negative speeds (no reversing while driving)
            left = max(0.0, min(speed * 1.3, left))
            right = max(0.0, min(speed * 1.3, right))

            self.rover.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})
            self._sleep_tick(t0, interval)

        self.rover.stop()
        return self.arrived

    def _pick_steer_column(self, ds) -> int:
        """Decide which column to steer toward.

        Priority:
        1. LLM bias column (if fresh and passable)
        2. Best passable column near LLM bias (if bias exists but exact col blocked)
        3. Depth-recommended column (farthest open space)
        """
        n = self.depth_vectors.num_columns
        dists = ds.smoothed_distances_m

        with self._lock:
            bias_col = self._bias_col
            bias_age = time.time() - self._bias_time

        # Check if LLM bias is fresh
        if bias_col is not None and bias_age < STALE_BIAS_S:
            # Is the bias column passable?
            if dists[bias_col] >= PASSABLE_M:
                return bias_col
            # Bias column blocked — find nearest passable column
            best_col = None
            best_dist = n
            for i, d in enumerate(dists):
                if d >= PASSABLE_M and abs(i - bias_col) < best_dist:
                    best_dist = abs(i - bias_col)
                    best_col = i
            if best_col is not None:
                return best_col

        # No usable bias — follow depth: pick farthest column
        return ds.farthest_col

    def _reverse_escape(self, ds, stop_event: threading.Event):
        """Back up and turn toward the widest opening."""
        # Reverse
        duration = REVERSE_DISTANCE_M / max(REVERSE_SPEED, 0.05)
        self.rover.send({"T": 1, "L": round(-REVERSE_SPEED, 3),
                         "R": round(-REVERSE_SPEED, 3)})
        t0 = time.time()
        while time.time() - t0 < duration and not stop_event.is_set():
            time.sleep(0.05)
        self.rover.stop()
        time.sleep(0.1)

        # Get fresh depth after reversing
        depth_map = self.camera.get_depth_map()
        turn_deg = self._compute_escape_angle(depth_map)

        self.events.publish("drive", f"Escape turn: {turn_deg:+.0f}°")
        self._spin(turn_deg, stop_event)

    def _compute_escape_angle(self, depth_map) -> float:
        """Find the widest gap and return a turn angle toward it."""
        if depth_map is None:
            turn = float(self._escape_dir * ESCAPE_MIN_DEG)
            self._escape_dir *= -1
            return turn

        try:
            ds = self.depth_vectors.analyze(depth_map)
        except Exception:
            turn = float(self._escape_dir * ESCAPE_MIN_DEG)
            self._escape_dir *= -1
            return turn

        dists = ds.smoothed_distances_m
        n = len(dists)

        # Find widest contiguous passable gap
        best_start, best_len, best_center = 0, 0, n // 2
        run_start = None
        for i, d in enumerate(dists):
            if d >= PASSABLE_M:
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
            # Fully blocked — big blind turn
            turn = float(self._escape_dir * ESCAPE_MAX_DEG)
            self._escape_dir *= -1
            return turn

        # Column to heading (column space is ~65° FOV)
        center_norm = (best_center - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
        heading = center_norm * 32.5

        # If gap is at FOV edge, the real opening is further out — extrapolate
        gap_end = best_start + best_len - 1
        edge = max(1, n // 6)
        if best_start <= edge and gap_end < n - 1 - edge:
            heading = min(heading, -20.0) * 1.5
        elif gap_end >= n - 1 - edge and best_start > edge:
            heading = max(heading, 20.0) * 1.5

        turn = max(-ESCAPE_MAX_DEG, min(ESCAPE_MAX_DEG, heading))
        if abs(turn) < ESCAPE_MIN_DEG:
            sign = 1.0 if turn >= 0 else -1.0
            if abs(turn) < 5.0:
                sign = float(self._escape_dir)
                self._escape_dir *= -1
            turn = sign * ESCAPE_MIN_DEG

        return turn

    def _spin(self, degrees: float, stop_event: threading.Event):
        """In-place body rotation."""
        if abs(degrees) < 3.0:
            return
        turn_rate = getattr(self.config, "turn_rate_dps", 200.0)
        speed = min(ESCAPE_TURN_SPEED, getattr(self.config, "navigation_turn_speed", 0.24))
        effective_rate = max(60.0, turn_rate * (speed / 0.35))
        duration = abs(degrees) / effective_rate
        sign = 1 if degrees > 0 else -1
        self.rover.send({"T": 1, "L": round(speed * sign, 3),
                         "R": round(-speed * sign, 3)})
        t0 = time.time()
        while time.time() - t0 < duration and not stop_event.is_set():
            time.sleep(0.03)
        self.rover.stop()

    def stop(self):
        self.rover.stop()

    @staticmethod
    def _sleep_tick(t0: float, interval: float):
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
