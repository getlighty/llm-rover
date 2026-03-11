"""
CLIP Spatial Map — lightweight VLMaps replacement for ground rovers.

Builds a 2D grid of CLIP embeddings using:
  - Camera frames encoded with CLIP ViT-B/32
  - Ground-plane geometry (camera height + tilt → distance estimate)
  - Encoder odometry (x, y, yaw) from PoseTracker

Query with natural language: "where is the couch?" → (x, y) grid coordinates.
"""

import math
import threading
import time
import logging
import json
import numpy as np

log = logging.getLogger("clip_map")

# Camera geometry
CAMERA_HEIGHT = 0.15       # meters above ground
CAMERA_HFOV = 65.0         # horizontal FOV degrees
CAMERA_VFOV = 50.0         # vertical FOV degrees
IMG_W = 640
IMG_H = 480

# Grid parameters
GRID_SIZE = 200            # cells per side
GRID_RES = 0.05            # meters per cell (5cm)
CLIP_DIM = 512             # CLIP ViT-B/32 embedding dimension
MAX_RANGE = 3.0            # max reliable depth estimate (meters)
MIN_RANGE = 0.15           # min range (camera height basically)

# Grid covers GRID_SIZE * GRID_RES = 10m x 10m
# Origin is at center of grid (100, 100)
GRID_ORIGIN = GRID_SIZE // 2


def _ground_plane_depth(pixel_row, cam_tilt_deg):
    """Estimate distance to a pixel assuming it's on the ground plane.

    Uses pinhole model: distance = camera_height / tan(angle_below_horizon)

    Args:
        pixel_row: y pixel coordinate (0=top, IMG_H-1=bottom)
        cam_tilt_deg: gimbal tilt angle (positive=up, negative=down)

    Returns:
        Estimated distance in meters, or None if geometry invalid
    """
    # Pixel angle relative to image center (positive = below center)
    pixel_angle = (pixel_row - IMG_H / 2) * (CAMERA_VFOV / IMG_H)

    # Total angle below horizon: pixel angle - camera tilt
    # (negative tilt = looking down = more below horizon)
    angle_below_horizon = pixel_angle - cam_tilt_deg

    if angle_below_horizon <= 2.0:  # nearly horizontal or looking up
        return None

    angle_rad = math.radians(angle_below_horizon)
    distance = CAMERA_HEIGHT / math.tan(angle_rad)

    if distance < MIN_RANGE or distance > MAX_RANGE:
        return None
    return distance


def _pixel_to_world(pixel_col, pixel_row, cam_tilt_deg, robot_x, robot_y, world_heading_deg):
    """Project a pixel to world (x, y) coordinates via ground-plane geometry.

    Args:
        pixel_col, pixel_row: image coordinates
        cam_tilt_deg: gimbal tilt
        robot_x, robot_y: robot world position (meters)
        world_heading_deg: body_yaw + cam_pan (where camera points)

    Returns:
        (world_x, world_y) or None if projection invalid
    """
    dist = _ground_plane_depth(pixel_row, cam_tilt_deg)
    if dist is None:
        return None

    # Horizontal angle from image center
    h_angle = (pixel_col - IMG_W / 2) * (CAMERA_HFOV / IMG_W)

    # World angle (heading + horizontal offset)
    world_angle = math.radians(world_heading_deg + h_angle)

    wx = robot_x + dist * math.sin(world_angle)
    wy = robot_y + dist * math.cos(world_angle)
    return (wx, wy)


def _world_to_grid(wx, wy):
    """Convert world coordinates (meters) to grid indices."""
    gx = int(round(wx / GRID_RES)) + GRID_ORIGIN
    gy = int(round(wy / GRID_RES)) + GRID_ORIGIN
    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
        return (gx, gy)
    return None


def _grid_to_world(gx, gy):
    """Convert grid indices back to world coordinates (meters)."""
    wx = (gx - GRID_ORIGIN) * GRID_RES
    wy = (gy - GRID_ORIGIN) * GRID_RES
    return (wx, wy)


class CLIPMap:
    """2D spatial grid of CLIP embeddings for semantic querying."""

    def __init__(self, device="cuda"):
        import clip as openai_clip
        import torch

        self.device = device
        self.torch = torch

        # Load CLIP model
        log.info("Loading CLIP ViT-B/32 on %s...", device)
        self.model, self.preprocess = openai_clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.clip = openai_clip
        log.info("CLIP loaded.")

        # Grid: embeddings (float16 to save memory ~39MB)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, CLIP_DIM), dtype=np.float16)
        self.counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint16)
        self.labels = {}  # (gx, gy) -> list of label strings

        self._lock = threading.Lock()
        self._total_updates = 0

    def update_from_frame(self, frame_bgr, robot_x, robot_y, body_yaw, cam_pan, cam_tilt):
        """Process a camera frame and update the grid.

        Divides the frame into a grid of tiles, encodes each with CLIP,
        and projects onto the spatial map.

        Args:
            frame_bgr: OpenCV BGR frame (640x480)
            robot_x, robot_y: robot position in meters
            body_yaw: robot heading in degrees
            cam_pan: gimbal pan in degrees
            cam_tilt: gimbal tilt in degrees
        """
        import torch
        from PIL import Image

        world_heading = body_yaw + cam_pan

        # Divide frame into 4x3 tiles (160x160 each)
        tile_w, tile_h = 160, 160
        tiles = []
        tile_centers = []

        for row in range(3):  # 3 rows
            for col in range(4):  # 4 columns
                x0 = col * tile_w
                y0 = row * tile_h
                cx = x0 + tile_w // 2
                cy = y0 + tile_h // 2

                world_pos = _pixel_to_world(cx, cy, cam_tilt, robot_x, robot_y, world_heading)
                if world_pos is None:
                    continue

                grid_pos = _world_to_grid(*world_pos)
                if grid_pos is None:
                    continue

                tile = frame_bgr[y0:y0 + tile_h, x0:x0 + tile_w]
                # Convert BGR to RGB PIL
                tile_rgb = tile[:, :, ::-1]
                pil_img = Image.fromarray(tile_rgb)
                tiles.append(self.preprocess(pil_img))
                tile_centers.append(grid_pos)

        if not tiles:
            return 0

        # Batch encode all tiles
        with torch.no_grad():
            batch = torch.stack(tiles).to(self.device)
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy().astype(np.float16)

        # Update grid with running average
        updated = 0
        with self._lock:
            for feat, (gx, gy) in zip(features, tile_centers):
                n = self.counts[gx, gy]
                if n == 0:
                    self.grid[gx, gy] = feat
                else:
                    # Running average
                    alpha = 1.0 / (n + 1)
                    self.grid[gx, gy] = (
                        self.grid[gx, gy].astype(np.float32) * (1 - alpha)
                        + feat.astype(np.float32) * alpha
                    ).astype(np.float16)
                self.counts[gx, gy] = min(n + 1, 65535)
                updated += 1

            self._total_updates += updated

        return updated

    def query(self, text, top_k=5, min_score=0.15):
        """Find grid locations matching a text query.

        Args:
            text: Natural language query (e.g., "red couch", "doorway")
            top_k: Number of results to return
            min_score: Minimum cosine similarity threshold

        Returns:
            List of dicts: [{"x": float, "y": float, "score": float, "gx": int, "gy": int}]
        """
        import torch

        with torch.no_grad():
            tokens = self.clip.tokenize([text]).to(self.device)
            text_feat = self.model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu().numpy().astype(np.float16).flatten()

        with self._lock:
            # Only search cells that have been observed
            mask = self.counts > 0
            if not mask.any():
                return []

            # Cosine similarity (embeddings are already normalized)
            observed = self.grid[mask]  # (N, 512)
            scores = (observed.astype(np.float32) @ text_feat.astype(np.float32))

            # Get indices in the masked array
            indices = np.where(mask)
            gx_all = indices[0]
            gy_all = indices[1]

        # Filter by min_score and get top_k
        valid = scores >= min_score
        if not valid.any():
            return []

        valid_scores = scores[valid]
        valid_gx = gx_all[valid]
        valid_gy = gy_all[valid]

        top_idx = np.argsort(valid_scores)[-top_k:][::-1]

        results = []
        for idx in top_idx:
            gx, gy = int(valid_gx[idx]), int(valid_gy[idx])
            wx, wy = _grid_to_world(gx, gy)
            results.append({
                "x": round(wx, 2),
                "y": round(wy, 2),
                "score": round(float(valid_scores[idx]), 3),
                "gx": gx,
                "gy": gy,
            })

        return results

    def label_location(self, text, gx, gy):
        """Attach a text label to a grid cell."""
        with self._lock:
            key = (gx, gy)
            if key not in self.labels:
                self.labels[key] = []
            if text not in self.labels[key]:
                self.labels[key].append(text)

    def get_stats(self):
        """Return map statistics."""
        with self._lock:
            observed = int(np.count_nonzero(self.counts))
            return {
                "observed_cells": observed,
                "total_updates": self._total_updates,
                "grid_size": GRID_SIZE,
                "grid_res_m": GRID_RES,
                "coverage_m2": round(observed * GRID_RES * GRID_RES, 2),
                "memory_mb": round(self.grid.nbytes / 1e6, 1),
                "labels": len(self.labels),
            }

    def save(self, path="clip_map.npz"):
        """Save grid to disk."""
        with self._lock:
            np.savez_compressed(
                path,
                grid=self.grid,
                counts=self.counts,
            )
            # Save labels separately as JSON
            label_path = path.replace(".npz", "_labels.json")
            serializable = {f"{k[0]},{k[1]}": v for k, v in self.labels.items()}
            with open(label_path, "w") as f:
                json.dump(serializable, f)
        log.info("Saved CLIP map to %s (%d observed cells)", path, np.count_nonzero(self.counts))

    def load(self, path="clip_map.npz"):
        """Load grid from disk."""
        try:
            data = np.load(path)
            with self._lock:
                self.grid = data["grid"]
                self.counts = data["counts"]
            label_path = path.replace(".npz", "_labels.json")
            try:
                with open(label_path) as f:
                    raw = json.load(f)
                    self.labels = {
                        tuple(int(x) for x in k.split(",")): v
                        for k, v in raw.items()
                    }
            except FileNotFoundError:
                pass
            observed = int(np.count_nonzero(self.counts))
            log.info("Loaded CLIP map from %s (%d observed cells)", path, observed)
            return True
        except FileNotFoundError:
            log.info("No existing CLIP map at %s", path)
            return False


class CLIPMapUpdater:
    """Background thread that periodically updates the CLIP map from camera frames."""

    def __init__(self, clip_map, frame_getter, pose_getter, interval=2.0):
        """
        Args:
            clip_map: CLIPMap instance
            frame_getter: callable() -> JPEG bytes or None
            pose_getter: callable() -> dict with body_yaw, cam_pan, cam_tilt, x, y
            interval: seconds between updates
        """
        self.clip_map = clip_map
        self.frame_getter = frame_getter
        self.pose_getter = pose_getter
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="clip-map-updater")
        self._thread.start()
        log.info("CLIP map updater started (interval=%.1fs)", self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        import cv2

        while not self._stop.is_set():
            try:
                jpeg = self.frame_getter()
                if jpeg is None:
                    self._stop.wait(1.0)
                    continue

                pose = self.pose_getter()

                # Decode JPEG to BGR
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._stop.wait(1.0)
                    continue

                updated = self.clip_map.update_from_frame(
                    frame,
                    robot_x=pose.get("x", 0),
                    robot_y=pose.get("y", 0),
                    body_yaw=pose.get("body_yaw", 0),
                    cam_pan=pose.get("cam_pan", 0),
                    cam_tilt=pose.get("cam_tilt", 0),
                )

                if updated:
                    log.debug("CLIP map: updated %d cells (total observed: %d)",
                              updated, self.clip_map.get_stats()["observed_cells"])

            except Exception as e:
                log.error("CLIP map update error: %s", e)

            self._stop.wait(self.interval)


if __name__ == "__main__":
    import sys
    import cv2

    # Quick test: encode a single image and query it
    cmap = CLIPMap(device="cuda")

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
    else:
        try:
            import requests
            resp = requests.get("http://localhost:8765/api/snap", timeout=5)
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            cap.release()
            if not ret:
                print("No image source")
                sys.exit(1)

    print(f"Image: {img.shape}")

    # Simulate robot at origin looking forward, camera level
    t0 = time.time()
    n = cmap.update_from_frame(img, robot_x=0, robot_y=0, body_yaw=0, cam_pan=0, cam_tilt=-10)
    elapsed = time.time() - t0
    print(f"Updated {n} grid cells in {elapsed:.2f}s")
    print(f"Stats: {cmap.get_stats()}")

    # Interactive query
    while True:
        q = input("\nQuery (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        t0 = time.time()
        results = cmap.query(q, top_k=3)
        elapsed = time.time() - t0
        print(f"Results ({elapsed:.3f}s):")
        for r in results:
            print(f"  ({r['x']:.2f}, {r['y']:.2f}) score={r['score']:.3f}")

    print("\nDone.")
