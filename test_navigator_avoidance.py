#!/usr/bin/env python3
"""Focused checks for obstacle-avoidance curve geometry."""

import numpy as np

from local_detector import DepthEstimator, DEPTH_SELF_FLOOR_CUTOFF
from navigator import Navigator


class _DummyPose:
    x = 0.0
    y = 0.0
    body_yaw = 0.0
    cam_pan = 0.0
    cam_tilt = 0.0


class _DummyFloorNav:
    FRAME_W = 640
    FRAME_H = 480

    def __init__(self, clear=False, best_col=None):
        self._clear = clear
        self._best_col = best_col

    def check_floor_clear(self, detections, frame_w, frame_h, direction_col=None,
                          prefer_widest=False):
        return self._clear, self._best_col


class _DummyTracker:
    def __init__(self, depth_map=None):
        self._depth_map = depth_map
        self.depth_estimator = None

    def get_depth_map(self):
        return self._depth_map

    def get_overlay_jpeg(self):
        return b"overlay-jpeg"

    def get_jpeg(self):
        return b"raw-jpeg"


class _TrackerWithDetections(_DummyTracker):
    def __init__(self, depth_map=None, detections=None, age=0.1):
        super().__init__(depth_map=depth_map)
        self._detections = list(detections or [])
        self._age = age

    def get_detections(self):
        return list(self._detections), "summary", self._age


class _PromptCapture:
    def __init__(self, response='{"action":"turn","turn_degrees":25}'):
        self.prompt = None
        self.jpeg = None
        self.aux_images = None
        self.response = response

    def __call__(self, prompt, jpeg, aux_images=None):
        self.prompt = prompt
        self.jpeg = jpeg
        self.aux_images = aux_images
        return self.response


def _parse_json(raw):
    import json
    return json.loads(raw)


def _make_nav(floor_nav=None, llm_vision_fn=None, parse_fn=None, tracker=None):
    return Navigator(
        rover=_DummyRover(),
        detector=None,
        tracker=tracker,
        llm_vision_fn=llm_vision_fn or (lambda *args, **kwargs: None),
        parse_fn=parse_fn or (lambda *args, **kwargs: None),
        pose=_DummyPose(),
        floor_nav=floor_nav,
    )


class _DummyRover:
    def __init__(self):
        self.commands = []

    def send(self, cmd):
        self.commands.append(dict(cmd))


class _DummyRoomMap:
    def __init__(self):
        self.records = []

    def record(self, detections, x, y, body_yaw, cam_pan, cam_tilt):
        self.records.append((list(detections), x, y, body_yaw, cam_pan, cam_tilt))

    def nav_json(self, target, rx, ry, yaw):
        return None

    def room_json(self, rx, ry, yaw, max_objects=10):
        return None


def _bottom_glow_depth():
    depth = np.full((480, 640), 0.10, dtype=np.float32)
    glow_row = int(depth.shape[0] * 0.92)
    depth[glow_row:, :] = 0.95
    return depth


def _open_depth_map():
    return np.full((480, 640), 0.10, dtype=np.float32)


def _sector_depth_map():
    depth = np.full((480, 640), 0.18, dtype=np.float32)
    y0 = int(depth.shape[0] * 0.40)
    y1 = int(round(depth.shape[0] * DEPTH_SELF_FLOOR_CUTOFF))
    depth[y0:y1, 250:395] = 0.92  # center blocked / close
    depth[y0:y1, :180] = 0.05     # left very open
    depth[y0:y1, 470:640] = 0.28  # right somewhat open
    return depth


def _targeted_corridor_depth_map():
    depth = np.full((480, 640), 0.90, dtype=np.float32)
    y0 = int(depth.shape[0] * 0.40)
    y1 = int(round(depth.shape[0] * DEPTH_SELF_FLOOR_CUTOFF))
    depth[y0:y1, 40:210] = 0.05   # globally farthest left opening
    depth[y0:y1, 440:620] = 0.18  # right opening should win when requested
    return depth


def test_curve_right_for_left_obstacle():
    nav = _make_nav()
    side, angle, reason = nav._compute_avoidance_curve(
        {"name": "chair", "cx": 0.38, "bw": 0.24, "dist_m": 0.32},
        path_center=0.5,
        depth_map=None,
    )
    assert side == "right"
    assert angle > 0
    assert "chair" in reason


def test_curve_left_for_right_obstacle():
    nav = _make_nav()
    side, angle, reason = nav._compute_avoidance_curve(
        {"name": "bin", "cx": 0.63, "bw": 0.22, "dist_m": 0.35},
        path_center=0.5,
        depth_map=None,
    )
    assert side == "left"
    assert angle < 0
    assert "bin" in reason


def test_wheel_command_scales_with_curve():
    nav = _make_nav()
    left_slow, right_fast, target_angle = nav._curve_wheel_command(0.0, -20.0)
    assert target_angle < 0
    assert left_slow < right_fast

    left_fast, right_slow, target_angle = nav._curve_wheel_command(0.0, 20.0)
    assert target_angle > 0
    assert left_fast > right_slow


def test_depth_logic_ignores_bottom_glow():
    nav = _make_nav()
    depth_map = _bottom_glow_depth()
    assert nav._depth_obstacle_check(depth_map) is None
    assert nav._estimate_depth_clear_distance(depth_map) > 1.5


def test_depth_colorize_masks_bottom_glow():
    depth_map = np.tile(
        np.linspace(0.10, 0.50, 480, dtype=np.float32)[:, None],
        (1, 640),
    )
    glow_row = int(depth_map.shape[0] * 0.92)
    depth_map[glow_row:, :] = 0.95
    color = DepthEstimator.colorize(None, depth_map)
    usable_bottom = int(round(depth_map.shape[0] * DEPTH_SELF_FLOOR_CUTOFF))
    bottom_mean = float(np.mean(color[usable_bottom:, :]))
    upper_mean = float(np.mean(color[max(0, usable_bottom - 40):usable_bottom, :]))
    assert bottom_mean < upper_mean


def test_depth_llm_context_reports_open_side_and_blockage():
    nav = _make_nav()
    depth_map = _sector_depth_map()
    clear_dist = nav._estimate_clear_distance(depth_map, [])
    info = nav._depth_llm_context(clear_dist, depth_map)
    assert "DepthAnything: clear ahead" in info
    assert "Depth sectors:" in info
    assert "Most open side: left." in info
    assert "Blocked/near: center." in info
    assert "more reliable than YOLO" in info


def test_depth_target_corridor_prefers_requested_side():
    nav = _make_nav()
    guidance = nav._depth_target_corridor(
        _targeted_corridor_depth_map(),
        drive_angle=24.0,
    )
    assert guidance is not None
    assert guidance.farthest_col < guidance.recommended_col
    assert guidance.requested_col is not None
    assert guidance.recommended_heading_deg > 0


def test_fluid_motion_converts_small_turn_to_drive_on_open_depth():
    nav = _make_nav()
    result = nav._prefer_fluid_motion(
        {"action": "turn", "turn_degrees": 12},
        depth_map=_open_depth_map(),
        clear_dist=1.4,
    )
    assert result["action"] == "drive"
    assert result["angle"] > 0
    assert result["distance"] >= 0.55
    assert "turn->drive" in result["_fluid_motion_note"]


def test_fluid_motion_extends_short_drive_on_open_depth():
    nav = _make_nav()
    result = nav._prefer_fluid_motion(
        {"action": "drive", "angle": 5, "distance": 0.3},
        depth_map=_open_depth_map(),
        clear_dist=1.4,
    )
    assert result["action"] == "drive"
    assert result["distance"] >= 0.55
    assert "extended drive" in result["_fluid_motion_note"]


def test_under_furniture_ignores_case_with_open_gap():
    nav = _make_nav(floor_nav=_DummyFloorNav(clear=False, best_col=12))
    detected, evidence = nav._detect_under_furniture(
        scene_text=("View is obstructed by table legs and a person's legs "
                    "standing under a desk"),
        reason_text="Current view is blocked by furniture and human legs",
        detections=[{"name": "legs", "bw": 0.18, "depth_closeness": 0.72}],
        clear_dist=0.38,
    )
    assert detected is False
    assert "gap_available" in evidence


def test_under_furniture_triggers_when_no_gap_and_close_clutter():
    nav = _make_nav(floor_nav=_DummyFloorNav(clear=False, best_col=None))
    detected, evidence = nav._detect_under_furniture(
        scene_text=("View is obstructed by table legs and a person's legs "
                    "standing under a desk"),
        reason_text="Current view is blocked by furniture and human legs",
        detections=[
            {"name": "desk", "bw": 0.20, "depth_closeness": 0.76},
            {"name": "legs", "bw": 0.18, "depth_closeness": 0.72},
        ],
        clear_dist=0.32,
    )
    assert detected is True
    assert "clear=0.32m" in evidence


def test_normalize_llm_result_ignores_stuck_action():
    nav = _make_nav()
    result = nav._normalize_llm_result(
        {"action": "stuck", "stuck": True, "scene": "blocked"},
        depth_map=_sector_depth_map(),
    )
    assert result["action"] == "turn"
    assert result["stuck"] is False
    assert result["_llm_stuck_ignored"] is True
    assert abs(result["turn_degrees"]) >= 20


def test_reactive_prompt_no_longer_requests_stuck():
    capture = _PromptCapture()
    nav = _make_nav(llm_vision_fn=capture, parse_fn=_parse_json)
    nav._nav_target = "kitchen"
    nav._llm_assess_reactive(
        goal="kitchen",
        step=0,
        max_steps=5,
        jpeg=b"fake-jpeg",
        yolo_summary="nothing",
        clear_dist=1.2,
        depth_map=_sector_depth_map(),
        room_ctx="",
    )
    assert capture.prompt is not None
    assert '"stuck": true/false' not in capture.prompt
    assert '"action": "drive"|"turn"|"arrived"|"stuck"' not in capture.prompt
    assert "Do NOT return action='stuck'" in capture.prompt
    assert "prefer 'drive' with angle set to the center of that corridor" in capture.prompt
    assert "Use 'turn' first only" in capture.prompt


def test_llm_call_includes_separate_depth_image_and_guide():
    capture = _PromptCapture()
    nav = _make_nav(
        llm_vision_fn=capture,
        parse_fn=_parse_json,
        tracker=_DummyTracker(depth_map=_sector_depth_map()),
    )
    nav._nav_target = "doorway"
    result = nav._llm_call("Reply ONLY JSON:\n{\"action\":\"turn\",\"turn_degrees\":25}",
                           jpeg=b"camera-jpeg")
    assert result["action"] == "turn"
    assert capture.jpeg == b"camera-jpeg"
    assert capture.aux_images is not None
    assert len(capture.aux_images) == 1
    label, data = capture.aux_images[0]
    assert label == "depth_anything"
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 100
    assert "Image 2 is a separate DepthAnything map aligned to Image 1." in capture.prompt
    assert "bright/warm/yellow-white means CLOSER" in capture.prompt


def test_navigate_leg_room_map_update_does_not_crash():
    tracker = _TrackerWithDetections(
        depth_map=_open_depth_map(),
        detections=[{"name": "door", "cx": 0.5, "bw": 0.2, "conf": 0.9}],
        age=0.1,
    )
    room_map = _DummyRoomMap()
    nav = Navigator(
        rover=_DummyRover(),
        detector=None,
        tracker=tracker,
        llm_vision_fn=lambda *args, **kwargs: (
            '{"scene":"doorway ahead","target_visible":true,'
            '"action":"crossed","confidence":0.95,"reason":"inside hall"}'
        ),
        parse_fn=_parse_json,
        pose=_DummyPose(),
        room_map=room_map,
    )
    ok, room_id, scene = nav.navigate_leg(
        {"visual_cues": ["doorway"], "target_room": "hallway"},
        max_steps=1,
    )
    assert ok is True
    assert room_id == "hallway"
    assert "doorway" in scene
    assert len(room_map.records) == 1


def test_llm_assess_waypoint_room_map_update_does_not_crash():
    tracker = _TrackerWithDetections(
        depth_map=_open_depth_map(),
        detections=[{"name": "door", "cx": 0.5, "bw": 0.2, "conf": 0.9}],
        age=0.1,
    )
    room_map = _DummyRoomMap()
    nav = Navigator(
        rover=_DummyRover(),
        detector=None,
        tracker=tracker,
        llm_vision_fn=lambda *args, **kwargs: (
            '{"target_visible":true,"scene":"doorway ahead",'
            '"action":"drive_forward","drive_angle":0,"drive_distance":0.8}'
        ),
        parse_fn=_parse_json,
        pose=_DummyPose(),
        room_map=room_map,
    )
    result = nav._llm_assess_waypoint(
        target="hallway",
        subtask=None,
        jpeg=b"jpeg",
        yolo_summary="door(90%)",
        clear_dist=1.2,
    )
    assert result["action"] == "drive_forward"
    assert len(room_map.records) == 1


if __name__ == "__main__":
    test_curve_right_for_left_obstacle()
    test_curve_left_for_right_obstacle()
    test_wheel_command_scales_with_curve()
    test_depth_logic_ignores_bottom_glow()
    test_depth_colorize_masks_bottom_glow()
    test_depth_llm_context_reports_open_side_and_blockage()
    test_depth_target_corridor_prefers_requested_side()
    test_fluid_motion_converts_small_turn_to_drive_on_open_depth()
    test_fluid_motion_extends_short_drive_on_open_depth()
    test_under_furniture_ignores_case_with_open_gap()
    test_under_furniture_triggers_when_no_gap_and_close_clutter()
    test_normalize_llm_result_ignores_stuck_action()
    test_reactive_prompt_no_longer_requests_stuck()
    test_llm_call_includes_separate_depth_image_and_guide()
    test_navigate_leg_room_map_update_does_not_crash()
    test_llm_assess_waypoint_room_map_update_does_not_crash()
    print("navigator avoidance tests passed")
