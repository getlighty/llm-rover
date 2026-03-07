#!/usr/bin/env python3
"""Focused regression checks for depth-driven FloorNavigator behavior."""

import cv2
import numpy as np

from floor_nav import FloorNavigator


class _DummyRover:
    def __init__(self):
        self.commands = []

    def send(self, cmd):
        self.commands.append(dict(cmd))


class _DummyTracker:
    def __init__(self, depth_map, detections=None):
        self._depth_map = depth_map
        self._detections = list(detections or [])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        self._jpeg = buf.tobytes()

    def get_jpeg(self):
        return self._jpeg

    def get_depth_map(self):
        return self._depth_map

    def get_detections(self):
        return list(self._detections), "", 0.0


class _DoorHintNavigator(FloorNavigator):
    def _find_opening_center(self, frame, frame_w, frame_h):
        return self.NUM_COLUMNS // 2


def _center_blocking_detection():
    return [{
        "name": "chair",
        "conf": 0.95,
        "cx": 0.5,
        "bw": 0.55,
        "bh": 0.45,
        "bbox": (180, 210, 460, 470),
    }]


def _open_depth():
    return np.full((480, 640), 0.1, dtype=np.float32)


def _close_depth():
    return np.full((480, 640), 0.9, dtype=np.float32)


def _left_open_doorway_depth():
    depth = np.full((480, 640), 0.15, dtype=np.float32)
    y0 = int(480 * 0.40)
    # Center obstacle: close wall/door jamb.
    depth[y0:, 230:420] = 0.95
    # Right side is only barely usable; left side is the widest opening.
    depth[y0:, 500:610] = 0.95
    return depth


def test_check_floor_clear_prefers_depth_over_yolo():
    nav = FloorNavigator(
        _DummyRover(),
        detector=None,
        tracker=_DummyTracker(_open_depth(), _center_blocking_detection()),
    )
    clear, best_col = nav.check_floor_clear(
        _center_blocking_detection(), 640, 480, direction_col=nav.NUM_COLUMNS // 2)
    assert clear is True
    assert best_col == nav.NUM_COLUMNS // 2


def test_drive_toward_returns_progress_on_open_depth():
    rover = _DummyRover()
    nav = FloorNavigator(
        rover,
        detector=None,
        tracker=_DummyTracker(_open_depth(), _center_blocking_detection()),
    )
    nav.MIN_PROGRESS_S = 0.15
    result = nav.drive_toward(timeout=0.25)
    assert result is True
    assert any(cmd.get("L") or cmd.get("R") for cmd in rover.commands)


def test_drive_toward_stops_when_depth_is_close():
    rover = _DummyRover()
    nav = FloorNavigator(
        rover,
        detector=None,
        tracker=_DummyTracker(_close_depth(), _center_blocking_detection()),
    )
    nav.MIN_PROGRESS_S = 0.15
    result = nav.drive_toward(timeout=0.25)
    assert result is False
    assert not any(cmd.get("L") or cmd.get("R") for cmd in rover.commands)


def test_drive_through_opening_uses_real_gap_not_door_hint():
    rover = _DummyRover()
    nav = _DoorHintNavigator(
        rover,
        detector=None,
        tracker=_DummyTracker(_left_open_doorway_depth()),
    )
    nav.MIN_PROGRESS_S = 0.15
    result = nav.drive_through_opening(timeout=0.25)
    assert result is True

    drive_cmds = [
        cmd for cmd in rover.commands
        if cmd.get("T") == 1 and (cmd.get("L") or cmd.get("R"))
    ]
    assert drive_cmds, "expected at least one drive command"
    first = drive_cmds[0]
    assert first["L"] < first["R"], first


if __name__ == "__main__":
    test_check_floor_clear_prefers_depth_over_yolo()
    test_drive_toward_returns_progress_on_open_depth()
    test_drive_toward_stops_when_depth_is_close()
    test_drive_through_opening_uses_real_gap_not_door_hint()
    print("floor_nav depth tests passed")
