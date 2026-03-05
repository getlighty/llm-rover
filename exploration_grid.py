#!/usr/bin/env python3
"""
Exploration Grid — 2-layer voxel grid for tracking visited/free/occupied space.

80×80×2 cells at 20cm resolution = 16m×16m coverage.
Layer 0 = floor plane (0–20cm), Layer 1 = obstacle height (20–40cm).
Rover starts at grid center (40, 40).

Coordinate system:
  X = right (positive), Y = forward (positive)
  Heading 0° = forward (+Y), 90° = right (+X)
"""

import math
import time

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

GRID_SIZE = 80           # cells per axis
CELL_M = 0.20            # meters per cell
NUM_LAYERS = 2           # floor + obstacle
DECAY_S = 600            # 10 minutes — cells older than this revert to UNKNOWN

# Cell states
UNKNOWN = 0
VISITED = 1
FREE = 2
OCCUPIED = 3

# Rover physical params
ROVER_WIDTH_M = 0.26     # 26cm rover width
ROVER_HALF_CELLS = max(1, int(math.ceil(ROVER_WIDTH_M / 2.0 / CELL_M)))

# Camera
CAMERA_FOV_DEG = 65.0    # horizontal FOV
DEPTH_COLS = 16           # angular columns for depth projection
MAX_DEPTH_M = 4.0         # max range for depth projection
MIN_DEPTH_M = 0.3         # min range

# LLM summary
SECTOR_NAMES = ["ahead", "right", "behind", "left"]
SUMMARY_MIN_M = 0.3
SUMMARY_MAX_M = 4.0


class ExplorationGrid:
    """3D voxel grid tracking explored space around the rover."""

    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, NUM_LAYERS), dtype=np.uint8)
        self.timestamps = np.zeros((GRID_SIZE, GRID_SIZE, NUM_LAYERS),
                                   dtype=np.float64)
        # Rover position in grid coords (center of grid)
        self.rx = GRID_SIZE / 2.0
        self.ry = GRID_SIZE / 2.0
        self.heading = 0.0  # degrees, 0=+Y (forward), 90=+X (right)

    def reset(self):
        """Clear grid for new session."""
        self.grid[:] = UNKNOWN
        self.timestamps[:] = 0.0
        self.rx = GRID_SIZE / 2.0
        self.ry = GRID_SIZE / 2.0
        self.heading = 0.0

    # ── Update Methods ───────────────────────────────────────────────

    def update_after_drive(self, distance_m, heading_deg):
        """Mark cells along driven path as VISITED (rover-width band).
        Uses Bresenham-style ray along heading direction."""
        now = time.time()
        rad = math.radians(heading_deg)
        # Direction: heading 0° = +Y, 90° = +X
        dx = math.sin(rad)
        dy = math.cos(rad)

        steps = max(1, int(distance_m / CELL_M))
        for i in range(steps + 1):
            frac = i * CELL_M
            if frac > distance_m:
                frac = distance_m
            cx = self.rx + dx * (frac / CELL_M)
            cy = self.ry + dy * (frac / CELL_M)

            # Mark rover-width band
            for ox in range(-ROVER_HALF_CELLS, ROVER_HALF_CELLS + 1):
                for oy in range(-ROVER_HALF_CELLS, ROVER_HALF_CELLS + 1):
                    gx = int(round(cx + ox))
                    gy = int(round(cy + oy))
                    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                        self.grid[gy, gx, 0] = VISITED
                        self.timestamps[gy, gx, 0] = now

        # Update rover position
        self.rx += dx * (distance_m / CELL_M)
        self.ry += dy * (distance_m / CELL_M)
        # Clamp to grid bounds
        self.rx = max(1.0, min(GRID_SIZE - 2.0, self.rx))
        self.ry = max(1.0, min(GRID_SIZE - 2.0, self.ry))

    def update_from_depth(self, depth_map, body_yaw, cam_pan):
        """Project depth map into grid: FREE up to obstacle, OCCUPIED at obstacle.
        body_yaw: current body heading in degrees
        cam_pan: gimbal pan angle in degrees (added to body_yaw)
        """
        if depth_map is None:
            return

        now = time.time()
        h, w = depth_map.shape[:2]

        # Depth Anything: higher value = closer. Normalize to [0, 1] range.
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max - d_min < 0.01:
            return  # uniform depth, no useful info

        # Camera direction: body_yaw + cam_pan
        cam_heading = body_yaw + cam_pan
        half_fov = CAMERA_FOV_DEG / 2.0

        # Sample DEPTH_COLS angular columns
        for col in range(DEPTH_COLS):
            # Angle for this column relative to camera center
            col_frac = (col + 0.5) / DEPTH_COLS  # 0..1
            angle_offset = (col_frac - 0.5) * CAMERA_FOV_DEG
            world_angle = cam_heading + angle_offset

            # Pixel column range in depth map
            px_start = int(col_frac * w - w / DEPTH_COLS / 2)
            px_end = int(col_frac * w + w / DEPTH_COLS / 2)
            px_start = max(0, px_start)
            px_end = min(w, px_end)
            if px_end <= px_start:
                continue

            # Use bottom 60% of image (floor-level obstacles)
            row_start = int(h * 0.40)
            stripe = depth_map[row_start:h, px_start:px_end]
            if stripe.size == 0:
                continue

            # 95th percentile = nearest object in this column
            near_val = float(np.percentile(stripe, 95))

            # Convert relative depth to meters (inverted: high = close)
            relative_far = 1.0 - (near_val - d_min) / (d_max - d_min + 1e-6)
            dist_m = MIN_DEPTH_M + relative_far * (MAX_DEPTH_M - MIN_DEPTH_M)
            dist_m = max(MIN_DEPTH_M, min(MAX_DEPTH_M, dist_m))

            # Ray-march from rover to obstacle
            rad = math.radians(world_angle)
            ray_dx = math.sin(rad)
            ray_dy = math.cos(rad)
            ray_steps = max(1, int(dist_m / CELL_M))

            for s in range(ray_steps + 1):
                step_m = s * CELL_M
                if step_m > dist_m:
                    break
                gx = int(round(self.rx + ray_dx * (step_m / CELL_M)))
                gy = int(round(self.ry + ray_dy * (step_m / CELL_M)))

                if not (0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE):
                    break

                layer = 0 if step_m < dist_m * 0.9 else 1

                if s < ray_steps:
                    # Before obstacle → FREE
                    if self.grid[gy, gx, 0] != VISITED:
                        self.grid[gy, gx, 0] = FREE
                        self.timestamps[gy, gx, 0] = now
                else:
                    # At obstacle distance → OCCUPIED
                    self.grid[gy, gx, 1] = OCCUPIED
                    self.timestamps[gy, gx, 1] = now

    def update_after_turn(self, degrees):
        """Sync internal heading after body rotation."""
        self.heading = (self.heading + degrees) % 360

    # ── LLM Summary ──────────────────────────────────────────────────

    def summarize_for_llm(self, body_yaw):
        """Divide space into 4 sectors relative to current heading.
        Returns string like 'Explored: ahead(72%), right(45%); Unexplored: left(8%), behind(3%)'"""
        now = time.time()

        # Decay old cells
        stale = (self.timestamps > 0) & (now - self.timestamps > DECAY_S)
        self.grid[stale] = UNKNOWN
        self.timestamps[stale] = 0.0

        # Build coordinate arrays relative to rover position
        ys, xs = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
        dx = xs.astype(np.float64) - self.rx
        dy = ys.astype(np.float64) - self.ry

        # Distance in cells → meters
        dist_cells = np.sqrt(dx * dx + dy * dy)
        dist_m = dist_cells * CELL_M

        # Angle from rover (0° = +Y forward, 90° = +X right)
        angles = np.degrees(np.arctan2(dx, dy))  # atan2(x, y) for heading convention

        # Relative to body heading
        rel_angles = (angles - body_yaw + 360) % 360  # 0-360

        # Distance mask: only cells within summary range
        range_mask = (dist_m >= SUMMARY_MIN_M) & (dist_m <= SUMMARY_MAX_M)

        # Sector boundaries (centered on each cardinal):
        # ahead: 315-45, right: 45-135, behind: 135-225, left: 225-315
        sector_bounds = [
            (315, 45),    # ahead
            (45, 135),    # right
            (135, 225),   # behind
            (225, 315),   # left
        ]

        explored = []
        unexplored = []

        for name, (lo, hi) in zip(SECTOR_NAMES, sector_bounds):
            if lo > hi:
                # Wraps around 0° (ahead sector)
                sector_mask = range_mask & ((rel_angles >= lo) | (rel_angles < hi))
            else:
                sector_mask = range_mask & (rel_angles >= lo) & (rel_angles < hi)

            total = int(sector_mask.sum())
            if total == 0:
                unexplored.append(f"{name}(0%)")
                continue

            # Count known cells (any layer) in this sector
            known = 0
            for layer in range(NUM_LAYERS):
                layer_known = sector_mask & (self.grid[:, :, layer] != UNKNOWN)
                known += int(layer_known.sum())
            # Avoid double-counting: cap at total
            known = min(known, total)

            pct = int(100 * known / total)
            if pct >= 30:
                explored.append(f"{name}({pct}%)")
            else:
                unexplored.append(f"{name}({pct}%)")

        parts = []
        if explored:
            parts.append("Explored: " + ", ".join(explored))
        if unexplored:
            parts.append("Unexplored: " + ", ".join(unexplored))
        return "; ".join(parts) if parts else "No exploration data"

    # ── Visualization ────────────────────────────────────────────────

    def render_image(self, scale=4):
        """Render grid as a color-coded top-down JPEG image.
        Returns JPEG bytes or None."""
        import cv2

        size = GRID_SIZE * scale
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Background: dark gray
        img[:] = (30, 26, 26)  # BGR for #1a1a1e

        # Color map (BGR): UNKNOWN=bg, VISITED=green, FREE=blue, OCCUPIED=red
        colors = {
            VISITED: (113, 204, 46),   # #2ecc71
            FREE:    (219, 152, 52),    # #3498db
            OCCUPIED:(60, 76, 231),     # #e74c3c
        }

        # Draw cells (layer 0 = floor, layer 1 = obstacles on top)
        for layer in range(NUM_LAYERS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    state = self.grid[gy, gx, layer]
                    if state == UNKNOWN:
                        continue
                    color = colors.get(state)
                    if color is None:
                        continue
                    # Grid Y=0 is top of array but Y+ is forward,
                    # so flip Y for display (forward = up)
                    py = (GRID_SIZE - 1 - gy) * scale
                    px = gx * scale
                    img[py:py + scale, px:px + scale] = color

        # Draw rover position as white triangle
        rx_px = int(self.rx * scale)
        ry_px = int((GRID_SIZE - 1 - self.ry) * scale)
        heading_rad = math.radians(self.heading)
        tri_size = scale * 2
        # Triangle points: tip forward, two rear corners
        tip_x = rx_px + int(math.sin(heading_rad) * tri_size)
        tip_y = ry_px - int(math.cos(heading_rad) * tri_size)
        left_x = rx_px + int(math.sin(heading_rad - 2.4) * tri_size * 0.6)
        left_y = ry_px - int(math.cos(heading_rad - 2.4) * tri_size * 0.6)
        right_x = rx_px + int(math.sin(heading_rad + 2.4) * tri_size * 0.6)
        right_y = ry_px - int(math.cos(heading_rad + 2.4) * tri_size * 0.6)
        pts = np.array([[tip_x, tip_y], [left_x, left_y],
                         [right_x, right_y]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))

        # Grid lines (subtle)
        for i in range(0, size, scale * 10):
            cv2.line(img, (i, 0), (i, size - 1), (50, 50, 50), 1)
            cv2.line(img, (0, i), (size - 1, i), (50, 50, 50), 1)

        # Scale indicator: 1m = 5 cells = 5*scale px
        bar_px = int(1.0 / CELL_M) * scale
        cv2.rectangle(img, (8, size - 16), (8 + bar_px, size - 12),
                      (200, 200, 200), -1)
        cv2.putText(img, "1m", (8, size - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()

    # ── Debug ─────────────────────────────────────────────────────────

    def stats(self):
        """Return dict of grid statistics for debugging."""
        visited = int((self.grid == VISITED).sum())
        free = int((self.grid == FREE).sum())
        occupied = int((self.grid == OCCUPIED).sum())
        unknown = int((self.grid == UNKNOWN).sum())
        total = GRID_SIZE * GRID_SIZE * NUM_LAYERS
        return {
            "visited": visited,
            "free": free,
            "occupied": occupied,
            "unknown": unknown,
            "total": total,
            "rover_pos": (round(self.rx * CELL_M, 2),
                          round(self.ry * CELL_M, 2)),
            "heading": round(self.heading, 1),
        }
