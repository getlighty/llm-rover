"""DepthAnything vector-map extraction for local navigation and safety."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass(slots=True)
class DepthVectorSummary:
    distances_m: list[float]
    smoothed_distances_m: list[float]
    passable_columns: list[bool]
    corridor_columns: list[Optional[int]]
    corridor_distances_m: list[Optional[float]]
    corridor_passable: list[Optional[bool]]
    farthest_col: int
    requested_col: Optional[int]
    recommended_col: int
    recommended_heading_deg: float
    usable_row_start: int
    usable_row_end: int

    def to_prompt_dict(self) -> dict:
        return asdict(self)

    def closest_corridor_distance(self) -> float | None:
        values = [value for value in self.corridor_distances_m if value is not None]
        return min(values) if values else None

    def center_distance(self) -> float:
        center = len(self.smoothed_distances_m) // 2
        return float(self.smoothed_distances_m[center])


class DepthVectorMap:
    def __init__(self, *, num_columns: int = 21, corridor_width: int = 7,
                 floor_top: float = 0.40, floor_bottom: float = 0.88,
                 depth_max_clearance_m: float = 2.4,
                 passable_clearance_m: float = 0.45,
                 depth_near_percentile: float = 95.0,
                 smooth_kernel: int = 3):
        self.num_columns = num_columns
        self.corridor_width = corridor_width
        self.floor_top = floor_top
        self.floor_bottom = floor_bottom
        self.depth_max_clearance_m = depth_max_clearance_m
        self.passable_clearance_m = passable_clearance_m
        self.depth_near_percentile = depth_near_percentile
        self.smooth_kernel = smooth_kernel

    def analyze(self, depth_map: np.ndarray,
                preferred_heading_deg: float | None = None,
                preferred_col: int | None = None,
                search_half_window_deg: float = 18.0) -> DepthVectorSummary:
        if depth_map is None or getattr(depth_map, "size", 0) == 0:
            raise ValueError("depth_map is empty")
        h, w = depth_map.shape[:2]
        y0 = int(h * self.floor_top)
        y1 = int(h * self.floor_bottom)
        if y1 <= y0:
            raise ValueError("invalid floor band")
        region = depth_map[y0:y1, :]
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        distances = []
        col_w = w / self.num_columns
        for col in range(self.num_columns):
            x0 = int(col * col_w)
            x1 = w if col == self.num_columns - 1 else int((col + 1) * col_w)
            patch = region[:, x0:x1]
            clearance = self._patch_to_clearance(patch, d_min, d_max)
            distances.append(float(clearance) if clearance is not None else 0.0)
        raw = np.asarray(distances, dtype=np.float32)
        smoothed = self._smooth(raw)
        passable = [bool(val >= self.passable_clearance_m) for val in smoothed]
        requested_col = self._requested_col(preferred_heading_deg, preferred_col)
        preferred_idx = requested_col if requested_col is not None else int(round((self.num_columns - 1) / 2.0))
        farthest_col = self._argmax_with_preference(smoothed, preferred_idx)
        recommended_col = farthest_col
        if requested_col is not None:
            window_cols = self._heading_window_to_columns(search_half_window_deg)
            lo = max(0, requested_col - window_cols)
            hi = min(self.num_columns - 1, requested_col + window_cols)
            recommended_col = self._argmax_with_preference(
                smoothed[lo:hi + 1], requested_col - lo
            ) + lo
        center_col = (self.num_columns - 1) / 2.0
        heading_norm = (recommended_col - center_col) / max(center_col, 1.0)
        heading_deg = float(np.clip(heading_norm, -1.0, 1.0)) * 32.5
        corridor_columns: list[Optional[int]] = []
        corridor_distances: list[Optional[float]] = []
        corridor_passable: list[Optional[bool]] = []
        half = self.corridor_width // 2
        for offset in range(-half, half + 1):
            idx = recommended_col + offset
            if 0 <= idx < self.num_columns:
                corridor_columns.append(idx)
                corridor_distances.append(round(float(smoothed[idx]), 3))
                corridor_passable.append(passable[idx])
            else:
                corridor_columns.append(None)
                corridor_distances.append(None)
                corridor_passable.append(None)
        return DepthVectorSummary(
            distances_m=[round(float(v), 3) for v in raw],
            smoothed_distances_m=[round(float(v), 3) for v in smoothed],
            passable_columns=passable,
            corridor_columns=corridor_columns,
            corridor_distances_m=corridor_distances,
            corridor_passable=corridor_passable,
            farthest_col=int(farthest_col),
            requested_col=requested_col,
            recommended_col=int(recommended_col),
            recommended_heading_deg=round(heading_deg, 2),
            usable_row_start=y0,
            usable_row_end=y1,
        )

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        if self.smooth_kernel <= 1:
            return values
        pad = self.smooth_kernel // 2
        kernel = np.ones(self.smooth_kernel, dtype=np.float32) / self.smooth_kernel
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _requested_col(self, heading_deg: float | None,
                       preferred_col: int | None) -> int | None:
        if preferred_col is not None:
            return max(0, min(self.num_columns - 1, int(preferred_col)))
        if heading_deg is None:
            return None
        center = (self.num_columns - 1) / 2.0
        heading_norm = float(np.clip(float(heading_deg) / 32.5, -1.0, 1.0))
        return max(0, min(self.num_columns - 1, int(round(center + heading_norm * center))))

    def _heading_window_to_columns(self, half_window_deg: float) -> int:
        if half_window_deg <= 0:
            return 0
        center = max((self.num_columns - 1) / 2.0, 1.0)
        span = center * min(float(half_window_deg), 32.5) / 32.5
        return max(1, int(round(span)))

    def _argmax_with_preference(self, values: np.ndarray, preferred_idx: int) -> int:
        values = np.asarray(values, dtype=np.float32)
        max_val = float(np.max(values))
        candidates = np.flatnonzero(np.isclose(values, max_val, rtol=1e-5, atol=1e-6))
        if candidates.size == 0:
            return int(np.argmax(values))
        return int(min(candidates, key=lambda idx: abs(int(idx) - int(preferred_idx))))

    def _patch_to_clearance(self, patch: np.ndarray,
                            d_min: float, d_max: float) -> float | None:
        if patch is None or patch.size == 0:
            return None
        if d_max - d_min < 0.01:
            mid = (d_min + d_max) / 2.0
            return 0.15 if mid > 0.5 else 1.0
        near = float(np.percentile(patch, self.depth_near_percentile))
        normalized = (near - d_min) / max(d_max - d_min, 1e-6)
        clearance = (1.0 - normalized) * self.depth_max_clearance_m
        return float(np.clip(clearance, 0.05, self.depth_max_clearance_m))


def extract_nav_corridor(depth_map: np.ndarray, **kwargs):
    extractor = DepthVectorMap(**kwargs)
    return extractor.analyze(depth_map).corridor_distances_m

