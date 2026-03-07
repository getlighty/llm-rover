#!/usr/bin/env python3
"""DepthAnything guidance extractor for programmatic navigation.

Produces a column-distance profile from a DepthAnything depth map and a
smaller corridor array centered on the farthest opening. The result can be
used directly by nav code without involving the LLM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CorridorGuidance:
    """Programmatic guidance result derived from a depth map."""

    distances_m: List[float]
    smoothed_distances_m: List[float]
    passable_columns: List[bool]
    corridor_distances_m: List[Optional[float]]
    corridor_columns: List[Optional[int]]
    corridor_passable: List[Optional[bool]]
    farthest_col: int
    requested_col: Optional[int]
    recommended_col: int
    recommended_heading_norm: float
    recommended_heading_deg: float
    usable_row_start: int
    usable_row_end: int

    def nav_profile(self) -> List[float]:
        """Smoothed full-width distance array for nav decisions."""
        return list(self.smoothed_distances_m)

    def nav_corridor(self) -> List[Optional[float]]:
        """Corridor array centered on the farthest detected opening."""
        return list(self.corridor_distances_m)


class DepthCorridorExtractor:
    """Extract a navigation-friendly corridor from a DepthAnything map."""

    def __init__(self,
                 num_columns: int = 21,
                 corridor_width: int = 7,
                 floor_top: float = 0.40,
                 floor_bottom: float = 0.88,
                 depth_max_clearance_m: float = 2.4,
                 passable_clearance_m: float = 0.45,
                 depth_near_percentile: float = 95.0,
                 smooth_kernel: int = 3):
        if num_columns < 3:
            raise ValueError("num_columns must be >= 3")
        if corridor_width < 3 or corridor_width % 2 == 0:
            raise ValueError("corridor_width must be odd and >= 3")
        if corridor_width > num_columns:
            raise ValueError("corridor_width must be <= num_columns")
        if smooth_kernel < 1 or smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be odd and >= 1")
        self.num_columns = num_columns
        self.corridor_width = corridor_width
        self.floor_top = floor_top
        self.floor_bottom = floor_bottom
        self.depth_max_clearance_m = depth_max_clearance_m
        self.passable_clearance_m = passable_clearance_m
        self.depth_near_percentile = depth_near_percentile
        self.smooth_kernel = smooth_kernel

    def analyze_depth_map(self, depth_map: np.ndarray) -> CorridorGuidance:
        """Convert a DepthAnything map into a centered corridor guidance array."""
        return self.analyze_depth_map_toward(depth_map)

    def analyze_depth_map_toward(self,
                                 depth_map: np.ndarray,
                                 preferred_heading_deg: Optional[float] = None,
                                 preferred_col: Optional[int] = None,
                                 search_half_window_deg: float = 18.0
                                 ) -> CorridorGuidance:
        """Convert a depth map into a corridor, optionally biased to a target heading."""
        if depth_map is None or getattr(depth_map, "size", 0) == 0:
            raise ValueError("depth_map is empty")

        h, w = depth_map.shape[:2]
        y0 = int(h * self.floor_top)
        y1 = int(h * self.floor_bottom)
        if y1 <= y0:
            raise ValueError("invalid floor band")

        region = depth_map[y0:y1, :]
        if region.size == 0:
            raise ValueError("empty floor region")

        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            raise ValueError("depth map contains non-finite values")

        distances = []
        col_w = w / self.num_columns
        for c in range(self.num_columns):
            x0 = int(c * col_w)
            x1 = w if c == self.num_columns - 1 else int((c + 1) * col_w)
            patch = region[:, x0:x1]
            dist = self._depth_patch_to_clearance(patch, d_min, d_max)
            distances.append(float(dist) if dist is not None else 0.0)

        distances_arr = np.asarray(distances, dtype=np.float32)
        smoothed = self._smooth(distances_arr)
        passable = [
            bool(dist >= self.passable_clearance_m)
            for dist in smoothed
        ]
        requested_col = self._requested_col(preferred_heading_deg, preferred_col)
        tie_preference = (
            requested_col if requested_col is not None
            else int(round((self.num_columns - 1) / 2.0))
        )
        farthest_col = self._argmax_with_preference(
            smoothed, preferred_idx=tie_preference)
        recommended_col = farthest_col
        if requested_col is not None:
            window_cols = self._heading_window_to_columns(search_half_window_deg)
            lo = max(0, requested_col - window_cols)
            hi = min(self.num_columns - 1, requested_col + window_cols)
            recommended_col = self._argmax_with_preference(
                smoothed[lo:hi + 1], preferred_idx=requested_col - lo) + lo

        center_col = (self.num_columns - 1) / 2.0
        heading_norm = (recommended_col - center_col) / max(center_col, 1.0)
        heading_norm = float(np.clip(heading_norm, -1.0, 1.0))
        heading_deg = heading_norm * 32.5

        corridor_columns: List[Optional[int]] = []
        corridor_distances: List[Optional[float]] = []
        corridor_passable: List[Optional[bool]] = []
        half = self.corridor_width // 2
        for offset in range(-half, half + 1):
            idx = recommended_col + offset
            if 0 <= idx < self.num_columns:
                corridor_columns.append(idx)
                corridor_distances.append(float(smoothed[idx]))
                corridor_passable.append(passable[idx])
            else:
                corridor_columns.append(None)
                corridor_distances.append(None)
                corridor_passable.append(None)

        return CorridorGuidance(
            distances_m=[round(float(v), 3) for v in distances_arr],
            smoothed_distances_m=[round(float(v), 3) for v in smoothed],
            passable_columns=passable,
            corridor_distances_m=[
                None if v is None else round(float(v), 3)
                for v in corridor_distances
            ],
            corridor_columns=corridor_columns,
            corridor_passable=corridor_passable,
            farthest_col=farthest_col,
            requested_col=requested_col,
            recommended_col=recommended_col,
            recommended_heading_norm=round(heading_norm, 4),
            recommended_heading_deg=round(float(heading_deg), 2),
            usable_row_start=y0,
            usable_row_end=y1,
        )

    def analyze_frame(self, estimator, bgr_frame: np.ndarray,
                      preferred_heading_deg: Optional[float] = None,
                      preferred_col: Optional[int] = None,
                      search_half_window_deg: float = 18.0) -> CorridorGuidance:
        """Run depth estimation on a frame, then extract guidance."""
        depth_map = estimator.estimate(bgr_frame)
        return self.analyze_depth_map_toward(
            depth_map,
            preferred_heading_deg=preferred_heading_deg,
            preferred_col=preferred_col,
            search_half_window_deg=search_half_window_deg,
        )

    def draw_debug(self, bgr_frame: np.ndarray, guidance: CorridorGuidance) -> np.ndarray:
        """Overlay the chosen corridor on a camera frame."""
        vis = bgr_frame.copy()
        h, w = vis.shape[:2]
        y0 = guidance.usable_row_start
        y1 = guidance.usable_row_end
        col_w = w / self.num_columns

        for idx, dist in enumerate(guidance.smoothed_distances_m):
            x0 = int(idx * col_w)
            x1 = w if idx == self.num_columns - 1 else int((idx + 1) * col_w)
            intensity = max(0, min(255, int((dist / self.depth_max_clearance_m) * 255)))
            color = (0, intensity, 255 - intensity)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

        cx0 = int(guidance.recommended_col * col_w)
        cx1 = w if guidance.recommended_col == self.num_columns - 1 else int((guidance.recommended_col + 1) * col_w)
        cv2.rectangle(vis, (cx0, y0), (cx1, y1), (0, 255, 255), 3)
        cv2.putText(
            vis,
            f"peak={guidance.recommended_col} steer={guidance.recommended_heading_deg:+.1f}deg",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _smooth(self, distances: np.ndarray) -> np.ndarray:
        if self.smooth_kernel == 1:
            return distances
        pad = self.smooth_kernel // 2
        kernel = np.ones(self.smooth_kernel, dtype=np.float32) / self.smooth_kernel
        padded = np.pad(distances, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _requested_col(self, preferred_heading_deg: Optional[float],
                       preferred_col: Optional[int]) -> Optional[int]:
        if preferred_col is not None:
            return max(0, min(self.num_columns - 1, int(preferred_col)))
        if preferred_heading_deg is None:
            return None
        center_col = (self.num_columns - 1) / 2.0
        heading_norm = float(np.clip(preferred_heading_deg / 32.5, -1.0, 1.0))
        col = int(round(center_col + heading_norm * center_col))
        return max(0, min(self.num_columns - 1, col))

    def _heading_window_to_columns(self, half_window_deg: float) -> int:
        if half_window_deg <= 0:
            return 0
        center_col = max((self.num_columns - 1) / 2.0, 1.0)
        span = center_col * min(float(half_window_deg), 32.5) / 32.5
        return max(1, int(round(span)))

    def _argmax_with_preference(self, values: np.ndarray, preferred_idx: int) -> int:
        values = np.asarray(values, dtype=np.float32)
        max_val = float(np.max(values))
        candidates = np.flatnonzero(np.isclose(values, max_val, rtol=1e-5, atol=1e-6))
        if candidates.size == 0:
            return int(np.argmax(values))
        return int(min(candidates, key=lambda idx: abs(int(idx) - int(preferred_idx))))

    def _depth_patch_to_clearance(self, patch: np.ndarray,
                                  d_min: float, d_max: float) -> Optional[float]:
        if patch is None or patch.size == 0:
            return None
        if d_max - d_min < 0.01:
            mid = (d_min + d_max) / 2.0
            return 0.15 if mid > 0.5 else 1.0

        near = float(np.percentile(patch, self.depth_near_percentile))
        if not np.isfinite(near):
            return None

        top = patch[:max(1, patch.shape[0] // 3), :]
        bottom = patch[-max(1, patch.shape[0] // 3):, :]
        gradient_penalty = 0.0
        if top.size and bottom.size:
            top_mean = float(np.mean(top))
            bottom_mean = float(np.mean(bottom))
            if (bottom_mean > top_mean * 1.35
                    and bottom_mean > d_min + (d_max - d_min) * 0.55):
                gradient_penalty = 0.25

        relative_far = 1.0 - (near - d_min) / (d_max - d_min + 1e-6)
        dist_m = 0.3 + relative_far * (self.depth_max_clearance_m - 0.3)
        dist_m -= gradient_penalty
        return max(0.15, dist_m)


def _load_depth_map(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    raise ValueError("Only .npy depth maps are supported for --depth-map")


def extract_nav_profile(depth_map: np.ndarray, **kwargs) -> List[float]:
    """Return the smoothed full-width distance profile for nav code."""
    return DepthCorridorExtractor(**kwargs).analyze_depth_map(depth_map).nav_profile()


def extract_nav_corridor(depth_map: np.ndarray, **kwargs) -> List[Optional[float]]:
    """Return a corridor array for nav, optionally biased to a target heading."""
    extractor_kwargs = {
        key: value for key, value in kwargs.items()
        if key not in ("preferred_heading_deg", "preferred_col",
                       "search_half_window_deg")
    }
    guidance_kwargs = {
        key: kwargs.get(key)
        for key in ("preferred_heading_deg", "preferred_col",
                    "search_half_window_deg")
        if key in kwargs
    }
    return DepthCorridorExtractor(**extractor_kwargs).analyze_depth_map_toward(
        depth_map, **guidance_kwargs).nav_corridor()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="RGB image to run through DepthAnything.")
    parser.add_argument("--depth-map", help="Precomputed depth map (.npy).")
    parser.add_argument("--engine", help="TensorRT engine for --image mode.")
    parser.add_argument("--columns", type=int, default=21)
    parser.add_argument("--corridor-width", type=int, default=7)
    parser.add_argument("--target-heading", type=float,
                        help="Preferred corridor heading in degrees (-30..+30).")
    parser.add_argument("--target-col", type=int,
                        help="Preferred column index to bias corridor selection.")
    parser.add_argument("--target-window", type=float, default=18.0,
                        help="Half-window in degrees around target heading.")
    parser.add_argument("--json-out")
    parser.add_argument("--vis-out")
    args = parser.parse_args()

    if bool(args.image) == bool(args.depth_map):
        raise SystemExit("Provide exactly one of --image or --depth-map")

    extractor = DepthCorridorExtractor(
        num_columns=args.columns,
        corridor_width=args.corridor_width,
    )

    if args.depth_map:
        depth_map = _load_depth_map(Path(args.depth_map))
        guidance = extractor.analyze_depth_map_toward(
            depth_map,
            preferred_heading_deg=args.target_heading,
            preferred_col=args.target_col,
            search_half_window_deg=args.target_window,
        )
        vis = None
    else:
        repo_root = Path(__file__).resolve().parent
        import sys
        sys.path.insert(0, str(repo_root))
        from local_detector import DepthEstimator  # noqa: E402

        if not args.engine:
            raise SystemExit("--engine is required with --image")
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        est = DepthEstimator(engine_path=args.engine)
        try:
            depth_map = est.estimate(frame)
            guidance = extractor.analyze_depth_map_toward(
                depth_map,
                preferred_heading_deg=args.target_heading,
                preferred_col=args.target_col,
                search_half_window_deg=args.target_window,
            )
        finally:
            est.close()
        vis = extractor.draw_debug(frame, guidance)

    payload = asdict(guidance)
    print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
    if args.vis_out and vis is not None:
        cv2.imwrite(args.vis_out, vis)


if __name__ == "__main__":
    main()
