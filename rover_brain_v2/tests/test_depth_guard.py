"""Checks for depth-based crash prevention during follow mode."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rover_brain_v2.follow.depth_guard import DepthCrashGuard


def _open_depth():
    return np.full((480, 640), 0.10, dtype=np.float32)


def _blocked_center_depth():
    depth = np.full((480, 640), 0.10, dtype=np.float32)
    y0 = int(depth.shape[0] * 0.40)
    y1 = int(depth.shape[0] * 0.88)
    depth[y0:y1, 220:420] = 0.95
    return depth


def test_depth_guard_allows_open_corridor():
    guard = DepthCrashGuard(stop_distance_m=0.35, turn_stop_distance_m=0.25)
    result = guard.assess(_open_depth(), steering_angle_deg=0.0)
    assert result["safe"] is True
    assert result["clearance_m"] >= 1.0


def test_depth_guard_stops_when_center_corridor_is_blocked():
    guard = DepthCrashGuard(stop_distance_m=0.35, turn_stop_distance_m=0.25)
    result = guard.assess(_blocked_center_depth(), steering_angle_deg=0.0)
    assert result["safe"] is False
    assert result["clearance_m"] < result["threshold_m"]


if __name__ == "__main__":
    test_depth_guard_allows_open_corridor()
    test_depth_guard_stops_when_center_corridor_is_blocked()
    print("depth guard tests passed")
