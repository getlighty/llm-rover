"""Focused checks for rover_brain_v2 depth-vector extraction."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rover_brain_v2.navigation.depth_vectors import DepthVectorMap, extract_nav_corridor


def _depth_map(num_columns, column_values):
    depth = np.full((480, 640), 0.90, dtype=np.float32)
    y0 = int(depth.shape[0] * 0.40)
    y1 = int(depth.shape[0] * 0.88)
    col_w = depth.shape[1] / num_columns
    for col, value in column_values.items():
        x0 = int(col * col_w)
        x1 = depth.shape[1] if col == num_columns - 1 else int((col + 1) * col_w)
        depth[y0:y1, x0:x1] = value
    return depth


def test_farthest_corridor_is_selected():
    extractor = DepthVectorMap(num_columns=9, corridor_width=5, smooth_kernel=1)
    summary = extractor.analyze(_depth_map(9, {7: 0.08, 8: 0.08, 4: 0.20}))
    assert summary.farthest_col == 7
    assert summary.recommended_col == 7
    assert summary.corridor_columns == [5, 6, 7, 8, None]
    assert summary.recommended_heading_deg > 0


def test_requested_side_can_override_global_best():
    extractor = DepthVectorMap(num_columns=9, corridor_width=5, smooth_kernel=1)
    summary = extractor.analyze(
        _depth_map(9, {1: 0.05, 2: 0.05, 7: 0.16, 8: 0.16}),
        preferred_heading_deg=24.0,
        search_half_window_deg=10.0,
    )
    assert summary.farthest_col == 2
    assert summary.requested_col == 7
    assert summary.recommended_col == 7


def test_extract_nav_corridor_pads_edges():
    corridor = extract_nav_corridor(
        _depth_map(7, {0: 0.05}),
        num_columns=7,
        corridor_width=5,
        smooth_kernel=1,
    )
    assert corridor[:2] == [None, None]
    assert corridor[2] is not None


if __name__ == "__main__":
    test_farthest_corridor_is_selected()
    test_requested_side_can_override_global_best()
    test_extract_nav_corridor_pads_edges()
    print("depth vector tests passed")
