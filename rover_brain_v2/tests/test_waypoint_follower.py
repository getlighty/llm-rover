"""Unit tests for ContinuousNavigator."""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rover_brain_v2.navigation.depth_vectors import DepthVectorMap
from rover_brain_v2.navigation.waypoint_follower import (
    ContinuousNavigator, STOP_CLEARANCE_M, PASSABLE_M,
)


def _make_depth_vectors():
    return DepthVectorMap(
        num_columns=21, corridor_width=7,
        floor_top=0.40, floor_bottom=0.88,
        depth_max_clearance_m=2.4, passable_clearance_m=0.45,
    )


def _make_nav(depth_map=None):
    rover = MagicMock()
    camera = MagicMock()
    camera.get_depth_map.return_value = depth_map
    events = MagicMock()
    flags = MagicMock()
    flags.desk_mode = False
    config = MagicMock()
    config.navigation_drive_speed = 0.15
    config.turn_rate_dps = 200.0
    config.navigation_turn_speed = 0.24
    config.depth_guard_stop_m = 0.25
    dv = _make_depth_vectors()
    return ContinuousNavigator(
        rover=rover, camera=camera, depth_vectors=dv,
        event_bus=events, flags=flags, config=config,
    )


def _open_depth_map():
    """Depth map with everything far away (low values = far in DA v2)."""
    return np.full((480, 640), 0.1, dtype=np.float32)


def _blocked_depth_map():
    """Depth map with obstacle dead ahead (high values = close)."""
    dm = np.full((480, 640), 0.1, dtype=np.float32)
    # Make floor region (rows 192-422) very close
    dm[192:422, :] = 0.95
    return dm


class TestSetBias:
    def test_set_bias_stores_column(self):
        nav = _make_nav()
        nav.set_bias(pixel_x=320, scene="hallway ahead", confidence=0.9)
        assert nav._bias_col is not None
        assert nav._bias_col == 10  # center of 21 columns
        assert nav.scene == "hallway ahead"

    def test_set_bias_left(self):
        nav = _make_nav()
        nav.set_bias(pixel_x=0, scene="door on left")
        assert nav._bias_col == 0

    def test_set_bias_right(self):
        nav = _make_nav()
        nav.set_bias(pixel_x=640, scene="door on right")
        assert nav._bias_col == 20

    def test_set_bias_none(self):
        nav = _make_nav()
        nav.set_bias(pixel_x=None, scene="no idea")
        assert nav._bias_col is None

    def test_arrived(self):
        nav = _make_nav()
        nav.set_bias(pixel_x=320, arrived=True, scene="at the door")
        assert nav.arrived is True


class TestPickSteerColumn:
    def test_follows_bias_when_passable(self):
        nav = _make_nav(_open_depth_map())
        ds = nav.depth_vectors.analyze(_open_depth_map())
        nav.set_bias(pixel_x=100, scene="door left")  # col ~3
        col = nav._pick_steer_column(ds)
        # Should be near the bias column
        assert col <= 5, f"Expected left-ish column, got {col}"

    def test_follows_depth_without_bias(self):
        nav = _make_nav(_open_depth_map())
        ds = nav.depth_vectors.analyze(_open_depth_map())
        col = nav._pick_steer_column(ds)
        # With uniform depth, should pick farthest (near center)
        assert 8 <= col <= 12, f"Expected center column, got {col}"


class TestEscapeAngle:
    def test_blocked_gives_large_turn(self):
        nav = _make_nav()
        angle = nav._compute_escape_angle(None)
        assert abs(angle) >= 55, f"Expected >=55°, got {angle}"

    def test_open_right_turns_right(self):
        nav = _make_nav()
        # Depth map: blocked left/center, open on right
        dm = np.full((480, 640), 0.95, dtype=np.float32)  # close everywhere
        dm[192:422, 400:] = 0.05  # far on right side
        angle = nav._compute_escape_angle(dm)
        assert angle > 0, f"Expected positive (right) turn, got {angle}"

    def test_open_left_turns_left(self):
        nav = _make_nav()
        dm = np.full((480, 640), 0.95, dtype=np.float32)
        dm[192:422, :240] = 0.05  # far on left side
        angle = nav._compute_escape_angle(dm)
        assert angle < 0, f"Expected negative (left) turn, got {angle}"


class TestRunLoop:
    def test_stops_on_arrived(self):
        nav = _make_nav(_open_depth_map())
        stop = threading.Event()
        nav.set_bias(pixel_x=320, arrived=True, scene="done")
        result = nav.run(stop)
        assert result is True
        assert nav.rover.stop.called

    def test_drives_with_open_space(self):
        """With open depth, the driver should send wheel commands."""
        nav = _make_nav(_open_depth_map())
        stop = threading.Event()
        call_count = [0]

        def mock_send(cmd):
            if isinstance(cmd, dict) and cmd.get("T") == 1:
                L = cmd.get("L", 0)
                R = cmd.get("R", 0)
                if L > 0 or R > 0:
                    call_count[0] += 1
                    if call_count[0] >= 3:
                        stop.set()  # stop after 3 drive commands

        nav.rover.send = mock_send
        nav.rover.stop = MagicMock()
        nav.run(stop)
        assert call_count[0] >= 3, f"Expected >=3 drive commands, got {call_count[0]}"


if __name__ == "__main__":
    print("Running continuous navigator tests...")
    for cls in [TestSetBias, TestPickSteerColumn, TestEscapeAngle, TestRunLoop]:
        inst = cls()
        for name in sorted(dir(inst)):
            if name.startswith("test_"):
                print(f"  {cls.__name__}.{name}...", end=" ")
                try:
                    getattr(inst, name)()
                    print("OK")
                except Exception as e:
                    print(f"FAIL: {e}")
    print("\nDone!")
