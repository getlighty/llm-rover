"""Prompt builders used by rover_brain_v2."""

from __future__ import annotations

import json


def command_system_prompt(gimbal_pan_enabled: bool = True) -> str:
    extra = (
        "You may use gimbal pan and tilt." if gimbal_pan_enabled
        else "Do not rotate the gimbal horizontally."
    )
    return (
        "You are Jasper, an indoor rover. Reply with ONLY one JSON object.\n"
        "Use short speech. Prefer safe, simple movements.\n"
        "Allowed commands:\n"
        '- wheels: {"T":1,"L":-0.2..0.2,"R":-0.2..0.2}\n'
        '- gimbal: {"T":133,"X":pan_deg,"Y":tilt_deg,"SPD":300,"ACC":20}\n'
        '- lights: {"T":132,"IO4":0..255,"IO5":0..255}\n'
        '- oled: {"T":3,"lineNum":0..3,"Text":"message"}\n'
        'Reply schema: {"commands":[...],"speak":"...",'
        '"observe":false,"remember":"optional note"}\n'
        "If wheel motion is used, keep it brief and cautious. "
        + extra
    )


def navigation_prompt(*, target: str, plan_context: str, leg_hint: str,
                      depth_context: dict, recent_observations: list[str],
                      heuristic_context: str = "") -> str:
    memory = "\n".join(f"- {line}" for line in recent_observations[-4:])
    return (
        "You are the navigator LLM for a small indoor rover.\n"
        "Use the camera image and the structured DepthAnything vector map to pick the next safe motion.\n"
        "Do not use YOLO. Use only what you see and the depth vectors.\n"
        "Focus only on the immediate next target for this step.\n"
        "Available actions: arrived, inspect, turn, drive_forward, reverse.\n"
        "Constraints:\n"
        "- drive_angle must stay within [-30, 30]\n"
        "- drive_distance must stay within [0.20, 1.20]\n"
        "- if you choose reverse, keep drive_distance within [0.10, 0.24]\n"
        "- turn_degrees should stay within [10, 120]\n"
        "- prefer smooth forward segments through the safest open corridor\n"
        "- if this step is about a doorway or exit, ignore the final destination and find that doorway now\n"
        "- if doorway cues, threshold, bright hall light, or floor change are visible, orient to them immediately\n"
        "- if depth shows a clear corridor, commit to drive_forward instead of tiny turns\n"
        "- do not repeat inspect/turn from the same spot unless the path is genuinely blocked\n"
        "- after one inspect and one turn, choose the safest forward corridor and move\n"
        "- if the heuristic estimate says the current zone likely does not contain the target, obey it\n"
        "- when the target is not visible in the current frame, leave the wrong zone immediately instead of re-exploring it\n"
        "- once a local cluster has been checked and target cues are absent, stop scanning that same area and exit it\n"
        "- if goal cues were seen recently, preserve the successful bearing and reacquire the goal instead of drifting to unrelated local details\n"
        "- if the target or doorway is visible and close, you may declare arrived\n"
        "- if the scene is ambiguous, inspect instead of guessing\n"
        '- scene and reason must each be one short phrase, not a long explanation\n'
        "Reply ONLY JSON with this schema:\n"
        '{"action":"arrived|inspect|turn|drive_forward|reverse",'
        '"scene":"brief visual summary","reason":"brief reasoning",'
        '"drive_angle":0,"drive_distance":0.0,"turn_degrees":0,'
        '"target_visible":false,"target_room_guess":""}\n\n'
        f"Mission target: {target}\n"
        f"Plan context: {plan_context or 'none'}\n"
        f"Leg hint: {leg_hint or 'none'}\n"
        f"Heuristic estimate: {heuristic_context or 'none'}\n"
        f"Depth vector map: {json.dumps(depth_context, ensure_ascii=True)}\n"
        f"Recent observations:\n{memory or '- none'}"
    )


def scene_prompt(context: str = "") -> str:
    suffix = f" Context: {context}" if context else ""
    return (
        "Describe this indoor rover camera frame in 2-4 short sentences. "
        "Mention open space, doorways, floor type, and landmarks." + suffix
    )


def room_reorg_prompt(*, nav_target: str, actual_room: str, from_room: str | None,
                      transition_id: str | None, scene_text: str) -> str:
    return (
        "Compress this successful room transition into navigation memory. "
        "Reply ONLY JSON with relationship-based doorway cues.\n"
        '{"arrived_room":"","room_features":[""],"entry_landmarks":[""],'
        '"transition_update":{"transition_id":"","doorway_landmarks":[""],'
        '"inside_features":[""],"inside_room_guess":"","navigational_hint":"",'
        '"confidence":0.0}}\n'
        f"Target: {nav_target}\n"
        f"Arrived room: {actual_room}\n"
        f"From room: {from_room or 'unknown'}\n"
        f"Transition: {transition_id or 'unknown'}\n"
        f"Scene: {scene_text or 'unknown'}"
    )


def follow_callout_prompt(target: str) -> str:
    return (
        f"The {target} I was following disappeared. "
        "Generate one short playful sentence calling out to it. "
        "Reply with the sentence only."
    )
