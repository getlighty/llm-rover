"""
Path planning and navigation for UGV Rover PT.

Converts camera detections into a 2D world map, plans vectorized paths
as waypoint sequences, and follows them with dead reckoning + visual feedback.

Architecture:
  WorldMap     - 2D occupancy grid in XY meters, built from detection sweeps
  DoorDetector - Finds doorways using edge detection + geometry
  PathPlanner  - A* search on occupancy grid, returns waypoint list
  PathFollower - Executes waypoint path with slow, safe movement

Coordinate system:
  - Origin (0,0) = rover's position at map creation time
  - X = right (positive), Y = forward (positive)
  - Heading 0° = forward (positive Y), 90° = right (positive X)
  - All distances in meters

Usage:
    world = WorldMap(rover, detector, tracker, pose)
    world.survey()                      # 360° scan to build map
    door = world.find_door()            # detect door location
    path = PathPlanner(world).plan(goal=(door.x, door.y))
    PathFollower(rover, pose, detector, tracker).follow(path)
"""

import cv2
import numpy as np
import json
import math
import time
import os
import threading

# Grid resolution
CELL_SIZE = 0.10  # meters per cell (10cm)
MAP_SIZE = 100    # cells per side → 10m x 10m map
MAP_CENTER = MAP_SIZE // 2  # rover starts at center

# Safety constants
SAFE_SPEED = 0.15          # m/s - very slow
BACKUP_SPEED = 0.10        # m/s - even slower for reverse
TURN_SPEED = 0.20          # m/s wheel differential for turning
WAYPOINT_TOLERANCE = 0.15  # meters - close enough to waypoint
OBSTACLE_CLEARANCE = 0.30  # meters - minimum clearance from obstacles
ROVER_WIDTH = 0.25         # meters - rover body width


class WorldMap:
    """2D occupancy grid built from detection sweeps.

    Cell values:
      0   = unknown
      1   = free space (observed, no obstacle)
      -1  = obstacle
      2   = door (passable opening)
      3   = waypoint/path marker
    """

    def __init__(self, rover, detector, tracker, pose):
        self.rover = rover
        self.detector = detector
        self.tracker = tracker
        self.pose = pose

        self.grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        self.landmarks = {}  # name -> {"x": m, "y": m, "conf": float, ...}
        self.doors = []      # list of Door objects
        self.origin_x = 0.0  # rover's starting X in world coords
        self.origin_y = 0.0

        # Mark rover's starting position as free
        self._mark_free(0, 0, radius=0.15)

    def survey(self, voice=None):
        """Do a full 360° sweep, detect objects, build map.

        Sweeps gimbal at multiple pan/tilt positions, then rotates body
        and repeats. Builds occupancy grid from detection distances.
        """
        PAN_STEPS = [-120, -80, -40, 0, 40, 80, 120]
        TILT_STEPS = [0, 15]

        print("[map] Starting room survey...")
        if voice:
            voice.speak("Surveying room.")

        detections_by_pos = {}

        # Phase 1: Forward-facing sweep
        for tilt in TILT_STEPS:
            for pan in PAN_STEPS:
                self.rover.send({"T": 133, "X": pan, "Y": tilt, "SPD": 300, "ACC": 20})
                time.sleep(0.7)

                dets = self._capture_and_detect()
                if dets:
                    body_yaw = self.pose.body_yaw
                    self._integrate_detections(dets, pan, tilt, body_yaw)
                    detections_by_pos[(pan, tilt, "fwd")] = dets

        # Phase 2: Rotate 180° and sweep again
        print("[map] Rotating 180° for rear sweep...")
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 300, "ACC": 20})
        time.sleep(0.3)
        self._rotate_degrees(180)

        for tilt in TILT_STEPS:
            for pan in PAN_STEPS:
                self.rover.send({"T": 133, "X": pan, "Y": tilt, "SPD": 300, "ACC": 20})
                time.sleep(0.7)

                dets = self._capture_and_detect()
                if dets:
                    body_yaw = self.pose.body_yaw
                    self._integrate_detections(dets, pan, tilt, body_yaw)
                    detections_by_pos[(pan, tilt, "rear")] = dets

        # Return to original heading
        print("[map] Returning to original heading...")
        self._rotate_degrees(180)
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

        # Also look for doors using edge detection
        print("[map] Scanning for doors...")
        self._scan_for_doors()

        n_obstacles = np.sum(self.grid == -1)
        n_free = np.sum(self.grid == 1)
        n_landmarks = len(self.landmarks)
        n_doors = len(self.doors)
        print(f"[map] Survey complete: {n_landmarks} landmarks, {n_obstacles} obstacle cells, "
              f"{n_free} free cells, {n_doors} doors")

        if voice:
            voice.speak("Survey done.")

        return self.landmarks

    def _capture_and_detect(self):
        """Capture a frame and run detection."""
        jpeg = self.tracker.get_jpeg()
        if not jpeg:
            return []
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return []
        return self.detector.detect(frame)

    # Objects that are solid obstacles (can't drive through/over)
    OBSTACLE_CLASSES = {
        "chair", "couch", "bed", "dining table", "toilet", "refrigerator",
        "oven", "microwave", "sink", "bench", "potted plant", "tv",
        "suitcase", "bicycle", "motorcycle", "car", "truck", "bus",
    }
    # Objects that are small/movable targets (navigate TO, not avoid)
    TARGET_CLASSES = {
        "person", "cup", "bottle", "bowl", "cell phone", "remote",
        "book", "mouse", "keyboard", "laptop", "scissors", "toothbrush",
        "apple", "banana", "orange", "sandwich", "pizza", "donut", "cake",
        "fork", "knife", "spoon", "vase", "clock", "teddy bear",
        "backpack", "handbag", "umbrella", "tie", "frisbee",
    }

    def _integrate_detections(self, detections, gimbal_pan, gimbal_tilt, body_yaw):
        """Convert detections to XY coordinates and update grid."""
        for det in detections:
            dist = det.get("dist_m")
            if dist is None or dist < 0.1 or dist > 8.0:
                continue  # skip unknown or unreliable distances

            # World angle = body heading + gimbal pan
            world_angle_deg = body_yaw + gimbal_pan
            world_angle_rad = math.radians(world_angle_deg)

            # Convert polar (angle, distance) to cartesian (x, y)
            x = dist * math.sin(world_angle_rad)
            y = dist * math.cos(world_angle_rad)

            name = det["name"]
            conf = det["conf"]

            # Update landmark
            if name not in self.landmarks or self.landmarks[name]["conf"] < conf:
                self.landmarks[name] = {
                    "x": round(x, 2), "y": round(y, 2),
                    "dist": round(dist, 2),
                    "angle": round(world_angle_deg, 1),
                    "conf": conf,
                    "time": time.time(),
                }

            # Only mark large/immovable objects as obstacles
            if name in self.OBSTACLE_CLASSES and conf > 0.25:
                self._mark_obstacle(x, y)
            elif name not in self.TARGET_CLASSES and conf > 0.30:
                # Unknown objects: mark as obstacle if confident
                self._mark_obstacle(x, y, radius=0.10)

            # Mark free space along the ray from rover to object
            self._mark_ray_free(0, 0, x, y)

    def _mark_obstacle(self, x, y, radius=0.15):
        """Mark cells near (x, y) as obstacles."""
        cx, cy = self._world_to_grid(x, y)
        r = max(1, int(radius / CELL_SIZE))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                    self.grid[gy, gx] = -1

    def _mark_free(self, x, y, radius=0.10):
        """Mark cells near (x, y) as free."""
        cx, cy = self._world_to_grid(x, y)
        r = max(1, int(radius / CELL_SIZE))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                    if self.grid[gy, gx] == 0:  # don't override obstacles
                        self.grid[gy, gx] = 1

    def _mark_ray_free(self, x0, y0, x1, y1):
        """Mark cells along ray from (x0,y0) to (x1,y1) as free (Bresenham)."""
        gx0, gy0 = self._world_to_grid(x0, y0)
        gx1, gy1 = self._world_to_grid(x1, y1)

        # Stop slightly before the obstacle
        dx = gx1 - gx0
        dy = gy1 - gy0
        steps = max(abs(dx), abs(dy), 1)

        for i in range(int(steps * 0.85)):  # stop at 85% of distance
            t = i / steps
            gx = int(gx0 + dx * t)
            gy = int(gy0 + dy * t)
            if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                if self.grid[gy, gx] != -1:  # don't clear obstacles
                    self.grid[gy, gx] = 1

    def _world_to_grid(self, x, y):
        """Convert world meters to grid cell coordinates."""
        gx = int(x / CELL_SIZE) + MAP_CENTER
        gy = int(y / CELL_SIZE) + MAP_CENTER
        return max(0, min(MAP_SIZE - 1, gx)), max(0, min(MAP_SIZE - 1, gy))

    def _grid_to_world(self, gx, gy):
        """Convert grid cell to world meters."""
        x = (gx - MAP_CENTER) * CELL_SIZE
        y = (gy - MAP_CENTER) * CELL_SIZE
        return x, y

    def _rotate_degrees(self, degrees):
        """Rotate the rover body by the specified degrees. Positive = clockwise."""
        duration = abs(degrees) / 120.0  # ~120 deg/s at TURN_SPEED
        if degrees > 0:
            self.rover.send({"T": 1, "L": TURN_SPEED, "R": -TURN_SPEED})
        else:
            self.rover.send({"T": 1, "L": -TURN_SPEED, "R": TURN_SPEED})
        time.sleep(duration)
        self.rover.send({"T": 1, "L": 0, "R": 0})
        time.sleep(0.5)

    def _scan_for_doors(self):
        """Sweep and look for doorways using edge detection."""
        PAN_STEPS = [-120, -80, -40, 0, 40, 80, 120]

        for pan in PAN_STEPS:
            self.rover.send({"T": 133, "X": pan, "Y": 0, "SPD": 300, "ACC": 20})
            time.sleep(0.7)

            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                continue
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            door = DoorDetector.detect_door(frame)
            if door:
                body_yaw = self.pose.body_yaw
                world_angle = body_yaw + pan + door["offset_deg"]
                # Estimate door distance (use frame center distance or default)
                dist = door.get("dist_m", 2.0)
                x = dist * math.sin(math.radians(world_angle))
                y = dist * math.cos(math.radians(world_angle))
                door_entry = {
                    "x": round(x, 2), "y": round(y, 2),
                    "width_m": door["width_m"],
                    "angle": round(world_angle, 1),
                    "confidence": door["confidence"],
                }
                self.doors.append(door_entry)
                # Mark door area as passable
                gx, gy = self._world_to_grid(x, y)
                if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                    self.grid[gy, gx] = 2
                print(f"[map] Door found at angle={world_angle:.0f}° dist={dist:.1f}m "
                      f"width={door['width_m']:.2f}m")

        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

    def find_landmark(self, name):
        """Find a landmark by name. Returns (x, y) or None."""
        name = name.lower().strip()
        if name in self.landmarks:
            lm = self.landmarks[name]
            return lm["x"], lm["y"]
        # Substring match
        for key, lm in self.landmarks.items():
            if name in key or key in name:
                return lm["x"], lm["y"]
        return None

    def find_door(self):
        """Return the best detected door, or None."""
        if not self.doors:
            return None
        return max(self.doors, key=lambda d: d["confidence"])

    def save(self, path="/tmp/world_map.json"):
        """Save map state to disk."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "landmarks": self.landmarks,
            "doors": self.doors,
            "grid_obstacles": int(np.sum(self.grid == -1)),
            "grid_free": int(np.sum(self.grid == 1)),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # Also save grid as image
        img = self._render_grid()
        cv2.imwrite(path.replace(".json", ".png"), img)
        print(f"[map] Saved to {path}")

    def _render_grid(self, path_waypoints=None):
        """Render the occupancy grid as a color image."""
        img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

        # Color coding
        img[self.grid == 0] = [40, 40, 40]      # unknown = dark gray
        img[self.grid == 1] = [200, 200, 200]    # free = light gray
        img[self.grid == -1] = [0, 0, 180]       # obstacle = red
        img[self.grid == 2] = [0, 200, 0]        # door = green

        # Mark rover position
        rx, ry = self._world_to_grid(0, 0)
        cv2.circle(img, (rx, ry), 3, (255, 200, 0), -1)

        # Mark landmarks
        for name, lm in self.landmarks.items():
            lx, ly = self._world_to_grid(lm["x"], lm["y"])
            cv2.circle(img, (lx, ly), 2, (0, 255, 255), -1)

        # Mark path
        if path_waypoints:
            for i, wp in enumerate(path_waypoints):
                wx, wy = self._world_to_grid(wp[0], wp[1])
                color = (255, 0, 255) if i == len(path_waypoints) - 1 else (255, 100, 0)
                cv2.circle(img, (wx, wy), 2, color, -1)
                if i > 0:
                    prev = path_waypoints[i - 1]
                    px, py = self._world_to_grid(prev[0], prev[1])
                    cv2.line(img, (px, py), (wx, wy), (255, 100, 0), 1)

        # Scale up for visibility
        img = cv2.resize(img, (MAP_SIZE * 4, MAP_SIZE * 4), interpolation=cv2.INTER_NEAREST)
        return img


class DoorDetector:
    """Detect doorways in camera frames using edge detection + geometry."""

    @staticmethod
    def detect_door(frame):
        """Analyze a frame for doorway-like vertical structures.

        Looks for:
          - Two roughly parallel vertical lines (door frame edges)
          - Appropriate spacing (0.6m-1.2m door width)
          - Vertical extent (door is tall)

        Returns dict with door info or None.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect vertical lines using Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=int(h * 0.3),  # at least 30% of frame height
                                maxLineGap=20)
        if lines is None:
            return None

        # Filter for near-vertical lines (within 15° of vertical)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                angle = 90
            else:
                angle = abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))
            if angle > 75:  # near vertical
                cx = (x1 + x2) / 2
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                vertical_lines.append({"x": cx, "length": length, "angle": angle,
                                       "coords": (x1, y1, x2, y2)})

        if len(vertical_lines) < 2:
            return None

        # Sort by x position
        vertical_lines.sort(key=lambda l: l["x"])

        # Look for pairs that could be door frame edges
        best_door = None
        best_score = 0

        for i in range(len(vertical_lines)):
            for j in range(i + 1, len(vertical_lines)):
                left = vertical_lines[i]
                right = vertical_lines[j]
                gap_px = right["x"] - left["x"]
                gap_frac = gap_px / w

                # Door should be 15-50% of frame width (depends on distance)
                if gap_frac < 0.10 or gap_frac > 0.65:
                    continue

                # Both lines should be reasonably long
                min_len = min(left["length"], right["length"])
                if min_len < h * 0.25:
                    continue

                # Score: prefer larger, more parallel, more centered doors
                center = (left["x"] + right["x"]) / 2
                center_offset = abs(center / w - 0.5)
                parallelism = 1.0 - abs(left["angle"] - right["angle"]) / 90.0
                size_score = gap_frac * 2  # bigger is better
                height_score = min_len / h

                score = (parallelism * 0.3 + size_score * 0.3 +
                         height_score * 0.3 + (1 - center_offset) * 0.1)

                if score > best_score:
                    best_score = score
                    # Estimate door width assuming ~65° FOV horizontal
                    fov_h = 65.0  # horizontal FOV in degrees
                    door_width_deg = gap_frac * fov_h
                    # Very rough distance estimate: typical door is 0.8m wide
                    est_width_m = 0.80  # assume standard door
                    est_dist_m = est_width_m / (2 * math.tan(math.radians(door_width_deg / 2)))

                    offset_frac = (center / w) - 0.5  # -0.5..0.5
                    offset_deg = offset_frac * fov_h

                    best_door = {
                        "confidence": min(score, 0.95),
                        "width_m": round(est_width_m, 2),
                        "dist_m": round(est_dist_m, 1),
                        "offset_deg": round(offset_deg, 1),
                        "gap_frac": round(gap_frac, 3),
                        "left_x": left["coords"],
                        "right_x": right["coords"],
                    }

        if best_door and best_door["confidence"] > 0.3:
            return best_door
        return None


class PathPlanner:
    """A* path planner on the occupancy grid."""

    def __init__(self, world_map):
        self.world = world_map

    def plan(self, goal_xy, start_xy=(0, 0)):
        """Plan a path from start to goal. Returns list of (x, y) waypoints in meters.

        Args:
            goal_xy: (x, y) tuple in meters
            start_xy: (x, y) tuple in meters, default is rover start position

        Returns:
            List of (x, y) waypoints, or empty list if no path found
        """
        sx, sy = self.world._world_to_grid(start_xy[0], start_xy[1])
        gx, gy = self.world._world_to_grid(goal_xy[0], goal_xy[1])

        # Inflate obstacles for safety margin
        inflated = self._inflate_grid(int(OBSTACLE_CLEARANCE / CELL_SIZE))

        # A* search
        path_cells = self._astar(inflated, (sx, sy), (gx, gy))

        if not path_cells:
            print("[planner] No path found!")
            return []

        # Convert grid cells to world coordinates
        raw_path = [self.world._grid_to_world(cx, cy) for cx, cy in path_cells]

        # Simplify path — remove collinear points, keep turns
        waypoints = self._simplify_path(raw_path)

        print(f"[planner] Path: {len(path_cells)} cells → {len(waypoints)} waypoints")
        return waypoints

    def _inflate_grid(self, radius):
        """Create inflated obstacle grid for safe path planning."""
        grid = self.world.grid.copy()
        obstacles = (grid == -1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        inflated_obs = cv2.dilate(obstacles, kernel)
        # Keep doors passable even if near obstacles
        doors = (grid == 2)
        inflated_obs[doors] = 0
        result = grid.copy()
        result[inflated_obs == 1] = -1
        result[doors] = 2
        return result

    def _astar(self, grid, start, goal):
        """A* search on grid. Returns list of (gx, gy) cells or empty list."""
        import heapq

        sx, sy = start
        gx, gy = goal

        # Check goal is reachable — search wider area if needed
        if grid[gy, gx] == -1:
            print("[planner] Goal is in obstacle, finding nearest free cell...")
            best_dist = float('inf')
            search_r = 30  # search up to 3m radius
            for dx in range(-search_r, search_r + 1):
                for dy in range(-search_r, search_r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE and grid[ny, nx] != -1:
                        d = dx * dx + dy * dy
                        if d < best_dist:
                            best_dist = d
                            gx, gy = nx, ny
            if best_dist == float('inf'):
                return []
            print(f"[planner] Adjusted goal to nearest free cell ({best_dist ** 0.5 * CELL_SIZE:.2f}m away)")

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, sx, sy)]
        came_from = {}
        g_score = {(sx, sy): 0}
        visited = set()

        # 8-directional movement
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if cx == gx and cy == gy:
                # Reconstruct path
                path = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return path

            for (dx, dy), cost in zip(dirs, costs):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE):
                    continue
                if grid[ny, nx] == -1:
                    continue
                if (nx, ny) in visited:
                    continue

                new_g = g_score[(cx, cy)] + cost
                if new_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = new_g
                    f = new_g + heuristic((nx, ny), (gx, gy))
                    heapq.heappush(open_set, (f, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

            # Limit search to prevent long stalls
            if len(visited) > 5000:
                print("[planner] A* search exceeded 5000 nodes, giving up")
                return []

        return []

    def _simplify_path(self, path, tolerance=0.15):
        """Remove collinear/redundant waypoints. Keep turns and endpoints."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            next_pt = path[i + 1]

            # Check if curr is collinear with prev and next
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_pt[0] - curr[0]
            dy2 = next_pt[1] - curr[1]

            # Cross product for collinearity
            cross = abs(dx1 * dy2 - dy1 * dx2)
            if cross > tolerance:
                simplified.append(curr)

        simplified.append(path[-1])

        # Further reduce: merge waypoints that are very close
        merged = [simplified[0]]
        for wp in simplified[1:]:
            prev = merged[-1]
            dist = math.sqrt((wp[0] - prev[0]) ** 2 + (wp[1] - prev[1]) ** 2)
            if dist > 0.20:  # minimum 20cm between waypoints
                merged.append(wp)
            else:
                merged[-1] = wp  # replace with later waypoint

        # Always include the goal
        if merged[-1] != simplified[-1]:
            merged.append(simplified[-1])

        return merged


class PathFollower:
    """Execute a waypoint path with safe, slow movement.

    Safety features:
    - Very slow speeds (0.15 m/s forward, 0.10 m/s reverse)
    - Look around (gimbal sweep) before backing up
    - Periodic obstacle checks during movement
    - Emergency stop on detection of close obstacles
    """

    def __init__(self, rover, pose, detector, tracker):
        self.rover = rover
        self.pose = pose
        self.detector = detector
        self.tracker = tracker
        self._running = True
        self._emergency = False

    def follow(self, waypoints, voice=None):
        """Follow a sequence of (x, y) waypoints.

        Returns True if reached final waypoint, False if aborted.
        """
        if not waypoints:
            print("[follow] No waypoints to follow")
            return False

        self._running = True
        self._emergency = False

        print(f"[follow] Following path with {len(waypoints)} waypoints")
        if voice:
            voice.speak("Following path.")

        # Track position using dead reckoning from pose tracker
        current_x = 0.0
        current_y = 0.0

        for i, wp in enumerate(waypoints):
            if not self._running:
                self._stop()
                return False

            goal_x, goal_y = wp
            print(f"[follow] Waypoint {i + 1}/{len(waypoints)}: ({goal_x:.2f}, {goal_y:.2f})")

            success = self._drive_to(current_x, current_y, goal_x, goal_y)

            if not success:
                print(f"[follow] Failed to reach waypoint {i + 1}")
                self._stop()
                return False

            # Update estimated position (dead reckoning is imprecise, but ok for short paths)
            current_x = goal_x
            current_y = goal_y

        self._stop()
        print("[follow] Path complete!")
        if voice:
            voice.speak("Path complete.")
        return True

    def _drive_to(self, from_x, from_y, to_x, to_y):
        """Drive from current position to target position.

        1. Calculate heading to target
        2. Rotate to face target
        3. Drive forward until distance is covered
        4. Periodically check for obstacles
        """
        dx = to_x - from_x
        dy = to_y - from_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < WAYPOINT_TOLERANCE:
            return True  # already there

        # Target heading (0° = forward/+Y, 90° = right/+X)
        target_heading = math.degrees(math.atan2(dx, dy))

        # Current heading from pose tracker
        current_heading = self.pose.body_yaw

        # Rotation needed
        turn = target_heading - current_heading
        # Normalize to -180..180
        turn = ((turn + 180) % 360) - 180

        # Decide if we need to go backwards (turn > 135°)
        go_reverse = abs(turn) > 135

        if go_reverse:
            # Look around before backing up
            print(f"[follow] Need to reverse — looking around first...")
            self._look_around_for_backup()

            # Adjust heading to reverse direction
            if turn > 0:
                turn -= 180
            else:
                turn += 180

        # Rotate to face target (or reverse direction)
        if abs(turn) > 5:
            print(f"[follow] Turning {turn:.0f}°")
            self._rotate(turn)

        # Drive forward (or backward)
        speed = BACKUP_SPEED if go_reverse else SAFE_SPEED
        if go_reverse:
            speed = -speed  # negative = reverse

        # Calculate drive time
        drive_time = dist / abs(speed)
        drive_time = min(drive_time, 10.0)  # cap at 10s per segment

        print(f"[follow] {'Reversing' if go_reverse else 'Driving'} {dist:.2f}m "
              f"(~{drive_time:.1f}s at {abs(speed):.2f} m/s)")

        return self._drive_straight(speed, drive_time)

    def _rotate(self, degrees):
        """Rotate rover by specified degrees. Positive = clockwise."""
        if abs(degrees) < 3:
            return

        duration = abs(degrees) / 120.0  # ~120 deg/s at TURN_SPEED

        if degrees > 0:
            self.rover.send({"T": 1, "L": TURN_SPEED, "R": -TURN_SPEED})
        else:
            self.rover.send({"T": 1, "L": -TURN_SPEED, "R": TURN_SPEED})

        # Wait with periodic obstacle checks
        elapsed = 0
        while elapsed < duration and self._running:
            time.sleep(0.1)
            elapsed += 0.1

        self.rover.send({"T": 1, "L": 0, "R": 0})
        time.sleep(0.3)

    def _drive_straight(self, speed, duration):
        """Drive at constant speed for duration. Check obstacles every 0.5s.

        Returns True if completed, False if stopped for obstacle.
        """
        self.rover.send({"T": 1, "L": speed, "R": speed})

        elapsed = 0
        check_interval = 0.5  # check obstacles every 0.5s

        while elapsed < duration and self._running:
            time.sleep(min(0.1, duration - elapsed))
            elapsed += 0.1

            # Periodic obstacle check
            if elapsed % check_interval < 0.15:
                if self._check_path_blocked():
                    print("[follow] Obstacle ahead! Stopping.")
                    self._stop()
                    return False

        self._stop()
        return True

    def _check_path_blocked(self):
        """Check if there's a close obstacle ahead using detection."""
        jpeg = self.tracker.get_jpeg()
        if not jpeg:
            return False
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False

        dets = self.detector.detect(frame)
        for det in dets:
            dist = det.get("dist_m")
            if dist and dist < 0.3:  # object closer than 30cm
                cx = det["cx"]
                # Only worry about objects in front (center 60% of frame)
                if 0.2 < cx < 0.8:
                    print(f"[follow] Close obstacle: {det['name']} at {dist:.2f}m")
                    return True
        return False

    def _look_around_for_backup(self):
        """Sweep gimbal to check behind before reversing."""
        print("[follow] Checking behind...")
        # Look right
        self.rover.send({"T": 133, "X": 120, "Y": 10, "SPD": 400, "ACC": 20})
        time.sleep(0.8)
        self._check_and_log("right-rear")

        # Look left
        self.rover.send({"T": 133, "X": -120, "Y": 10, "SPD": 400, "ACC": 20})
        time.sleep(0.8)
        self._check_and_log("left-rear")

        # Look directly behind (if gimbal allows)
        self.rover.send({"T": 133, "X": 180, "Y": 10, "SPD": 400, "ACC": 20})
        time.sleep(0.8)
        self._check_and_log("behind")

        # Return to forward
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 300, "ACC": 20})
        time.sleep(0.5)

    def _check_and_log(self, direction):
        """Detect objects and log what we see in a direction."""
        jpeg = self.tracker.get_jpeg()
        if not jpeg:
            return
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        dets = self.detector.detect(frame)
        if dets:
            names = [f"{d['name']}({d.get('dist_m', '?')}m)" for d in dets]
            print(f"[follow] {direction}: {', '.join(names)}")
        else:
            print(f"[follow] {direction}: clear")

    def _stop(self):
        """Stop all movement."""
        self.rover.send({"T": 1, "L": 0, "R": 0})

    def stop(self):
        """Abort path following."""
        self._running = False
        self._stop()
