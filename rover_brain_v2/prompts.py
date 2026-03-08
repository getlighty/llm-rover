"""Prompt builders used by rover_brain_v2."""

from __future__ import annotations

import json
from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """Load a .md prompt template, substitute %var% markers.

    Reads from disk every call (no cache) so edits take effect
    without restarting the service.
    """
    path = _PROMPTS_DIR / name
    text = path.read_text(encoding="utf-8").strip()
    for key, value in kwargs.items():
        text = text.replace(f"%{key}%", str(value))
    return text


def command_system_prompt(gimbal_pan_enabled: bool = True) -> str:
    extra = (
        "You may use gimbal pan and tilt." if gimbal_pan_enabled
        else "Do not rotate the gimbal horizontally."
    )
    return load_prompt("command_system.md", extra=extra)


def _depth_summary_text(depth_context: dict) -> str:
    """Summarize depth as left/center/right clearances + 8x8 grid for the LLM."""
    parts = []
    distances = depth_context.get("smoothed_distances_m", [])
    if distances:
        n = len(distances)
        third = max(1, n // 3)
        left_avg = sum(distances[:third]) / third
        center_avg = sum(distances[third:2 * third]) / max(1, 2 * third - third)
        right_avg = sum(distances[2 * third:]) / max(1, n - 2 * third)
        heading = depth_context.get("recommended_heading_deg", 0)
        parts.append(f"left={left_avg:.2f}m, center={center_avg:.2f}m, right={right_avg:.2f}m")
        parts.append(f"safest heading={heading:+.0f}°")
        min_d = min(distances)
        if min_d < 0.40:
            min_col = distances.index(min(distances))
            side = "left" if min_col < n // 3 else "right" if min_col >= 2 * n // 3 else "center"
            parts.append(f"WARNING: obstacle at {min_d:.2f}m on {side}")
    # 8x8 depth grid — always include if available
    grid = depth_context.get("depth_grid_8x8")
    if grid:
        parts.append("Depth grid (8x8, meters, rows=top→bottom, cols=left→right):")
        for row in grid:
            parts.append("  " + " ".join(f"{v:.1f}" for v in row))
    if not parts:
        return "no depth data"
    return "\n".join(parts)


def _house_map_text(topo_data: dict | None) -> str:
    """Build a concise text description of the house layout for the LLM."""
    if not topo_data:
        return "no house map available"
    nodes = topo_data.get("nodes", [])
    edges = topo_data.get("edges", [])
    current_room = topo_data.get("current_room", "unknown")

    rooms = {}
    transitions = {}
    for n in nodes:
        nid = n.get("id", "")
        if n.get("type") == "room":
            features = ", ".join(n.get("features", [])[:6])
            floor = n.get("floor_type", "")
            rooms[nid] = f"{n.get('label', nid)} [{features}] floor={floor}"
        elif n.get("type") == "transition":
            cues = ", ".join(n.get("visual_cues", [])[:3])
            azimuth = n.get("azimuth_from", {})
            az_text = ", ".join(f"{k}: {v}°" for k, v in azimuth.items())
            hint = n.get("nav_hints", "")
            transitions[nid] = f"{n.get('label', nid)} cues=[{cues}] azimuth=[{az_text}] hint={hint}"

    # Build adjacency: which rooms connect via which transition
    connections = []
    edge_map: dict[str, list[str]] = {}
    for e in edges:
        a, b = e.get("a", ""), e.get("b", "")
        edge_map.setdefault(b, []).append(a)
        edge_map.setdefault(a, []).append(b)

    for tid, tdesc in transitions.items():
        connected_rooms = [r for r in edge_map.get(tid, []) if r in rooms]
        if len(connected_rooms) >= 2:
            connections.append(f"  {connected_rooms[0]} <--[{tid}]--> {connected_rooms[1]}: {tdesc}")
        elif connected_rooms:
            connections.append(f"  {connected_rooms[0]} --[{tid}]-->: {tdesc}")

    lines = [f"You are currently in: {current_room}"]
    lines.append("Rooms:")
    for rid, rdesc in rooms.items():
        marker = " (YOU ARE HERE)" if rid == current_room else ""
        lines.append(f"  {rid}: {rdesc}{marker}")
    lines.append("Connections (how rooms link via doorways/passages):")
    lines.extend(connections)
    return "\n".join(lines)


def navigation_prompt(*, target: str, plan_context: str, leg_hint: str,
                      depth_context: dict, recent_observations: list[str],
                      heuristic_context: str = "",
                      yolo_detections: list[dict] | None = None,
                      topo_data: dict | None = None) -> str:
    memory = "\n".join(f"- {line}" for line in recent_observations[-4:])
    yolo_text = "none"
    if yolo_detections:
        yolo_lines = []
        for d in yolo_detections[:10]:
            dist = f", dist={d['dist_m']:.2f}m" if d.get("dist_m") else ""
            yolo_lines.append(
                f"  {d.get('name','?')} at cx={d.get('cx',0):.2f} cy={d.get('cy',0):.2f} "
                f"bw={d.get('bw',0):.2f}{dist} conf={d.get('conf',0):.2f}"
            )
        yolo_text = "\n".join(yolo_lines)
    return load_prompt("navigation.md",
        target=target,
        plan_context=plan_context or "none",
        leg_hint=leg_hint or "none",
        heuristic=heuristic_context or "none",
        depth_text=_depth_summary_text(depth_context),
        yolo_text=yolo_text,
        house_map=_house_map_text(topo_data),
        memory=memory or "- none",
    )


def scene_prompt(context: str = "") -> str:
    suffix = f" Context: {context}" if context else ""
    return load_prompt("scene.md", suffix=suffix)


def room_reorg_prompt(*, nav_target: str, actual_room: str, from_room: str | None,
                      transition_id: str | None, scene_text: str) -> str:
    return load_prompt("room_reorg.md",
        nav_target=nav_target,
        actual_room=actual_room,
        from_room=from_room or "unknown",
        transition_id=transition_id or "unknown",
        scene_text=scene_text or "unknown",
    )


def follow_callout_prompt(target: str) -> str:
    return load_prompt("follow_callout.md", target=target)
