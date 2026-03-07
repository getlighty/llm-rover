#!/usr/bin/env python3
"""Regression checks for stale/direct-YOLO ghost paths."""

from navigator import Navigator
from search_engine import SearchEngine


class _DummyPose:
    x = 0.0
    y = 0.0
    body_yaw = 0.0
    cam_pan = 0.0
    cam_tilt = 0.0
    world_pan = 0.0


class _DetectorNoDetect:
    last_detections = [{"name": "ghost"}]

    def detect(self, frame):
        raise AssertionError("direct detector.detect() should not be called")

    def find(self, target, detections=None):
        target = str(target).strip().lower()
        for det in detections or []:
            if det.get("name") == target:
                return det
        return None


class _TrackerWithDetections:
    def __init__(self, dets, age=0.1):
        self._dets = list(dets)
        self._age = age

    def get_detections(self):
        return list(self._dets), "summary", self._age

    def get_jpeg(self):
        return b"jpeg"


def _make_nav(tracker, detector):
    return Navigator(
        rover=None,
        detector=detector,
        tracker=tracker,
        llm_vision_fn=lambda *args, **kwargs: None,
        parse_fn=lambda *args, **kwargs: None,
        pose=_DummyPose(),
    )


def test_navigator_path_clear_uses_tracker_cache_only():
    tracker = _TrackerWithDetections([
        {"name": "chair", "cx": 0.5, "bw": 0.35, "dist_m": 0.3}
    ])
    nav = _make_nav(tracker, _DetectorNoDetect())
    assert nav._is_path_clear(b"jpeg") is False


def test_search_engine_yolo_check_uses_tracker_cache_only():
    tracker = _TrackerWithDetections([
        {"name": "bottle", "cx": 0.5, "bw": 0.15, "conf": 0.9}
    ])
    search = SearchEngine(
        rover=None,
        tracker=tracker,
        pose=_DummyPose(),
        spatial_map=None,
        llm_vision_fn=lambda *args, **kwargs: None,
        parse_fn=lambda *args, **kwargs: None,
        detector=_DetectorNoDetect(),
    )
    hit = search._yolo_check("bottle")
    assert hit is not None
    assert hit["name"] == "bottle"


if __name__ == "__main__":
    test_navigator_path_clear_uses_tracker_cache_only()
    test_search_engine_yolo_check_uses_tracker_cache_only()
    print("yolo ghost cleanup tests passed")
