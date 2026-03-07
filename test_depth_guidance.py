#!/usr/bin/env python3
"""Focused checks for programmatic DepthAnything corridor extraction."""

import numpy as np

from depth_guidance import DepthCorridorExtractor, extract_nav_corridor


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


def _uniform_depth(value=0.10):
    return np.full((480, 640), value, dtype=np.float32)


def test_corridor_centers_on_global_farthest_opening():
    extractor = DepthCorridorExtractor(
        num_columns=9,
        corridor_width=5,
        smooth_kernel=1,
    )
    guidance = extractor.analyze_depth_map(
        _depth_map(9, {7: 0.08, 8: 0.08, 4: 0.20})
    )
    assert guidance.farthest_col == 7
    assert guidance.recommended_col == 7
    assert guidance.corridor_columns == [5, 6, 7, 8, None]
    assert guidance.recommended_heading_deg > 0


def test_corridor_can_stay_inside_requested_direction_window():
    extractor = DepthCorridorExtractor(
        num_columns=9,
        corridor_width=5,
        smooth_kernel=1,
    )
    guidance = extractor.analyze_depth_map_toward(
        _depth_map(9, {1: 0.05, 2: 0.05, 7: 0.16, 8: 0.16}),
        preferred_heading_deg=24.0,
        search_half_window_deg=10.0,
    )
    assert guidance.farthest_col == 2
    assert guidance.requested_col == 7
    assert guidance.recommended_col == 7
    assert guidance.corridor_columns[2] == guidance.recommended_col
    assert guidance.recommended_heading_deg > 0


def test_uniform_open_prefers_center_without_target():
    extractor = DepthCorridorExtractor(
        num_columns=9,
        corridor_width=5,
        smooth_kernel=1,
    )
    guidance = extractor.analyze_depth_map(_uniform_depth())
    assert guidance.farthest_col == 4
    assert guidance.recommended_col == 4
    assert guidance.recommended_heading_deg == 0.0


def test_edge_corridor_pads_with_none_outside_image():
    corridor = extract_nav_corridor(
        _depth_map(7, {0: 0.05}),
        num_columns=7,
        corridor_width=5,
        smooth_kernel=1,
    )
    assert corridor[:2] == [None, None]
    assert corridor[2] is not None
    assert corridor[-1] is not None


if __name__ == "__main__":
    test_corridor_centers_on_global_farthest_opening()
    test_corridor_can_stay_inside_requested_direction_window()
    test_uniform_open_prefers_center_without_target()
    test_edge_corridor_pads_with_none_outside_image()
    print("depth guidance tests passed")
