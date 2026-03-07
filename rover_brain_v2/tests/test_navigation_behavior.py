"""Checks for smoother rover_brain_v2 navigation behavior."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rover_brain_v2.models import NavigatorResult
from rover_brain_v2.navigation.depth_vectors import DepthVectorSummary
from rover_brain_v2.navigation.navigator import DepthVectorNavigator
from rover_brain_v2.navigation.orchestrator import GraphNavigationOrchestrator
from rover_brain_v2.prompts import navigation_prompt


class _DummyRover:
    def send(self, _command):
        pass

    def stop(self):
        pass


class _DummyCamera:
    def snap(self):
        return None

    def get_depth_map(self):
        return None


class _DummyEvents:
    def publish(self, *_args, **_kwargs):
        pass


class _QueuedNavigator:
    def __init__(self, results):
        self.camera = _DummyCamera()
        self.config = SimpleNamespace(
            navigation_waypoint_budget=18,
            navigation_leg_attempts=3,
        )
        self._results = list(results)
        self.calls = []

    def run_doorway_task(self, instruction, *, attempt, total_attempts, waypoint_budget=None, **_kwargs):
        self.calls.append({
            "attempt": attempt,
            "total_attempts": total_attempts,
            "target_room": instruction.get("target_room"),
            "waypoint_budget": waypoint_budget,
        })
        return self._results.pop(0)

    def run_reactive_task(self, *_args, **_kwargs):
        raise AssertionError("reactive fallback was not expected in this test")

    def cancel(self):
        pass


class _FakeTopo:
    def __init__(self):
        self.current_room = "office"
        self.current_confidence = 0.0

    def rooms(self):
        return [
            SimpleNamespace(id="office", label="office"),
            SimpleNamespace(id="kitchen", label="kitchen"),
        ]

    def plan_route(self, current_room, target_room):
        assert current_room == "office"
        assert target_room == "kitchen"
        return [
            SimpleNamespace(
                from_room="office",
                to_room="kitchen",
                transition="office_kitchen_door",
            ),
        ]

    def route_summary(self, _legs):
        return "office -> kitchen"

    def leg_instruction(self, leg):
        return {
            "target_room": leg.to_room,
            "exit_hint": "find the kitchen doorway",
            "doorway_landmarks": ["wooden frame", "warm light"],
        }

    def ensure_transition_between(self, _from_room, _to_room, transition_id=None):
        return SimpleNamespace(id=transition_id or "office_kitchen_door")

    def update_transition_semantics(self, *_args, **_kwargs):
        pass

    def save(self):
        pass


def _navigator():
    return DepthVectorNavigator(
        rover=_DummyRover(),
        camera=_DummyCamera(),
        llm_client=None,
        event_bus=_DummyEvents(),
        flags=SimpleNamespace(gimbal_pan_enabled=False),
        config=SimpleNamespace(
            navigation_drive_speed=0.15,
            navigation_turn_speed=0.24,
            navigation_turn_step_s=0.04,
            navigation_zone_repeat_threshold=3,
            navigation_zone_exit_min_clearance_m=0.50,
            depth_guard_stop_m=0.35,
            depth_guard_turn_stop_m=0.25,
            navigation_waypoint_budget=18,
        ),
    )


def _depth_summary(center=1.2, corridor=1.1, heading=9.0):
    values = [0.9] * 21
    values[10] = center
    corridor_values = [corridor] * 7
    return DepthVectorSummary(
        distances_m=values,
        smoothed_distances_m=values,
        passable_columns=[True] * 21,
        corridor_columns=[7, 8, 9, 10, 11, 12, 13],
        corridor_distances_m=corridor_values,
        corridor_passable=[True] * 7,
        farthest_col=13,
        requested_col=None,
        recommended_col=13,
        recommended_heading_deg=heading,
        usable_row_start=100,
        usable_row_end=420,
    )


def test_orchestrator_resolves_natural_room_command():
    orchestrator = GraphNavigationOrchestrator(
        llm_client=None,
        navigator=SimpleNamespace(camera=_DummyCamera()),
        event_bus=_DummyEvents(),
    )
    assert orchestrator._resolve_room("go to kitchen") == "kitchen"
    assert orchestrator._resolve_room("take me to the living room") == "living_room"
    orchestrator.shutdown()


def test_orchestrator_retries_same_leg_until_navigator_reports_success():
    navigator = _QueuedNavigator([
        NavigatorResult(
            task_id=1,
            mode="doorway_search",
            status="incomplete",
            summary="chair leg avoided but doorway not reacquired yet",
            scene="chair leg near office doorway",
            reached=False,
        ),
        NavigatorResult(
            task_id=2,
            mode="doorway_search",
            status="completed",
            summary="doorway reacquired and crossed",
            scene="kitchen doorway reached",
            reached=True,
        ),
    ])
    orchestrator = GraphNavigationOrchestrator(
        llm_client=None,
        navigator=navigator,
        event_bus=_DummyEvents(),
    )
    orchestrator.topo = _FakeTopo()
    orchestrator._persist_semantic_memory = lambda **_kwargs: None

    result = orchestrator.run_navigation_task("go to kitchen")

    assert result.reached is True
    assert result.completed_legs == 1
    assert [call["attempt"] for call in navigator.calls] == [0, 1]
    assert all(call["target_room"] == "kitchen" for call in navigator.calls)
    orchestrator.shutdown()


def test_small_turn_is_smoothed_into_forward_drive():
    navigator = _navigator()
    summary = _depth_summary(center=1.4, corridor=1.1, heading=11.0)
    action, response, note = navigator._stabilize_action(
        action="turn",
        response={"action": "turn", "turn_degrees": 12, "scene": "open hallway"},
        scene="open hallway",
        depth_summary=summary,
        indecisive_streak=1,
    )
    assert action == "drive_forward"
    assert response["drive_distance"] >= 0.6
    assert response["drive_angle"] >= 8.0
    assert "fluid drive" in note


def test_navigation_prompt_mentions_zone_exit_heuristic():
    prompt = navigation_prompt(
        target="the doorway to kitchen",
        plan_context="Immediate goal: find the doorway now.",
        leg_hint="none",
        depth_context={"recommended_heading_deg": 18},
        recent_observations=["chair and desk corner", "same chair and desk corner"],
        heuristic_context="The last 3 observations look like the same local zone. If the doorway is not visible right now, leave this zone immediately.",
    )
    assert "Heuristic estimate" in prompt
    assert "leave this zone immediately" in prompt
    assert "preserve the successful bearing" in prompt


def test_repeated_inspect_commits_to_forward_motion():
    navigator = _navigator()
    summary = _depth_summary(center=0.9, corridor=0.85, heading=6.0)
    action, response, note = navigator._stabilize_action(
        action="inspect",
        response={"action": "inspect", "scene": "same doorway ahead"},
        scene="same doorway ahead",
        depth_summary=summary,
        indecisive_streak=3,
        zone_repeat_count=1,
    )
    assert action == "drive_forward"
    assert response["drive_distance"] >= 0.5
    assert "commit drive" in note


def test_zone_loop_forces_exit_drive_when_target_not_visible():
    navigator = _navigator()
    navigator._recent_observations = [
        "chair and desk corner with clutter",
        "same chair and desk corner with clutter",
        "chair and desk corner close ahead",
    ]
    summary = _depth_summary(center=0.95, corridor=0.82, heading=17.0)
    action, response, note = navigator._stabilize_action(
        action="inspect",
        response={"action": "inspect", "scene": "chair and desk corner", "target_visible": False},
        scene="chair and desk corner",
        depth_summary=summary,
        indecisive_streak=0,
        zone_repeat_count=navigator._recent_zone_repeat_count("chair and desk corner"),
    )
    assert action == "drive_forward"
    assert response["drive_angle"] >= 14.0
    assert response["drive_distance"] >= 0.45
    assert "leave zone" in note


def test_recent_goal_evidence_reacquires_preserved_bearing():
    navigator = _navigator()
    navigator._goal_target_label = "the doorway to kitchen"
    navigator._goal_cue_tokens = {"doorway", "kitchen", "arch", "counter"}
    navigator._goal_heading_bias_deg = 24.0
    navigator._goal_heading_confidence = 0.82
    navigator._goal_recently_seen_steps = 5
    navigator._goal_reacquire_steps = 4
    summary = _depth_summary(center=0.75, corridor=0.78, heading=-22.0)

    action, response, note = navigator._stabilize_action(
        action="drive_forward",
        response={"action": "drive_forward", "drive_angle": -24, "drive_distance": 0.8, "target_visible": False},
        scene="close-up of unrelated side surface and floor clutter",
        depth_summary=summary,
        indecisive_streak=0,
        zone_repeat_count=1,
    )

    assert action == "turn"
    assert response["turn_degrees"] >= 18.0
    assert "reacquire bearing" in note


def test_zone_exit_context_is_suppressed_when_doorway_seen_recently():
    navigator = _navigator()
    navigator._goal_target_label = "the doorway to kitchen"
    navigator._goal_recently_seen_steps = 4
    navigator._recent_observations = [
        "orange arch and kitchen counter visible",
        "same orange arch and kitchen counter visible",
        "orange arch and kitchen counter visible",
        "orange arch and kitchen counter visible",
    ]
    summary = _depth_summary(center=0.92, corridor=0.88, heading=-18.0)

    context = navigator._zone_exit_heuristic_context(
        target="the doorway to kitchen",
        depth_summary=summary,
    )

    assert "Do not drift" in context
    assert "preserved bearing" in context


def test_leg_hint_carries_graph_search_context():
    navigator = _navigator()
    hint = navigator._format_leg_hint({
        "exit_hint": "find the bright hallway doorway",
        "room_nav_hints": "go around the chair, not through it",
        "expected_floor": "stone tiles",
        "expected_azimuth_deg": -45,
        "doorway_width_m": 0.9,
    })
    assert "go around the chair" in hint
    assert "stone tiles" in hint
    assert "Expected azimuth" in hint
    assert "0.90m" in hint


def test_leg_target_phrase_prioritizes_immediate_doorway():
    navigator = _navigator()
    target = navigator._leg_target_phrase({
        "target_room": "hallway",
        "doorway_landmarks": ["wooden door frame", "bright hallway light"],
        "exit_hint": "find the office hall door",
    })
    assert "doorway to hallway" in target
    assert "wooden door frame" in target


def test_blocked_escape_plan_backs_up_from_chair_leg():
    navigator = _navigator()
    summary = _depth_summary(center=0.18, corridor=0.24, heading=-14.0)
    plan = navigator._blocked_escape_plan(
        drive_angle_deg=0.0,
        depth_summary=summary,
        scene="office chair leg and chair wheel close in front of the rover",
    )
    assert 0.20 <= plan["reverse_distance_m"] <= 0.24
    assert plan["turn_degrees"] <= -35.0
    assert "chair" in plan["reason"]


def test_blocked_escape_plan_uses_open_side_when_available():
    navigator = _navigator()
    summary = _depth_summary(center=0.28, corridor=0.42, heading=22.0)
    plan = navigator._blocked_escape_plan(
        drive_angle_deg=6.0,
        depth_summary=summary,
        scene="close obstacle ahead",
    )
    assert 0.18 <= plan["reverse_distance_m"] <= 0.22
    assert plan["turn_degrees"] >= 35.0


def test_blocked_escape_plan_keeps_goal_side_after_avoidance():
    navigator = _navigator()
    navigator._goal_target_label = "the doorway to hallway"
    navigator._goal_heading_bias_deg = -24.0
    navigator._goal_heading_confidence = 0.8
    navigator._goal_reacquire_steps = 3
    summary = _depth_summary(center=0.24, corridor=0.55, heading=18.0)
    plan = navigator._blocked_escape_plan(
        drive_angle_deg=10.0,
        depth_summary=summary,
        scene="table leg close ahead",
    )
    assert plan["turn_degrees"] <= -35.0


def test_reverse_action_is_kept_short_or_converted_to_search_drive():
    navigator = _navigator()
    summary = _depth_summary(center=0.60, corridor=0.72, heading=18.0)
    action, response, note = navigator._stabilize_action(
        action="reverse",
        response={"action": "reverse", "drive_distance": 0.5},
        scene="open doorway area",
        depth_summary=summary,
        indecisive_streak=0,
        zone_repeat_count=1,
    )
    assert action == "drive_forward"
    assert response["drive_distance"] <= 0.65
    assert "doorway search" in note


def test_turn_speed_profile_ramps_up_and_down():
    navigator = _navigator()
    profile = navigator._turn_speed_profile(duration_s=0.28, peak_speed=0.24)
    speeds = [speed for speed, _ in profile]
    assert len(profile) >= 3
    assert speeds[0] < max(speeds)
    assert speeds[-1] < max(speeds)
    assert max(speeds) <= 0.24


if __name__ == "__main__":
    test_orchestrator_resolves_natural_room_command()
    test_orchestrator_retries_same_leg_until_navigator_reports_success()
    test_small_turn_is_smoothed_into_forward_drive()
    test_navigation_prompt_mentions_zone_exit_heuristic()
    test_repeated_inspect_commits_to_forward_motion()
    test_zone_loop_forces_exit_drive_when_target_not_visible()
    test_recent_goal_evidence_reacquires_preserved_bearing()
    test_zone_exit_context_is_suppressed_when_doorway_seen_recently()
    test_leg_hint_carries_graph_search_context()
    test_leg_target_phrase_prioritizes_immediate_doorway()
    test_blocked_escape_plan_backs_up_from_chair_leg()
    test_blocked_escape_plan_uses_open_side_when_available()
    test_blocked_escape_plan_keeps_goal_side_after_avoidance()
    test_reverse_action_is_kept_short_or_converted_to_search_drive()
    test_turn_speed_profile_ramps_up_and_down()
    print("navigation behavior tests passed")
