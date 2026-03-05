"""room_context.py — Room database with exclusion-based identification.

Standalone module, no dependencies on rover_brain_llm.
Stores room data in rooms.json alongside this file.
"""

import json
import os
import time

ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOMS_FILE = os.path.join(ROVER_DIR, "rooms.json")


def load_rooms():
    """Read rooms.json, return full dict. Handles missing/corrupt."""
    if not os.path.exists(ROOMS_FILE):
        return {"current_room": None, "current_confidence": 0.0, "rooms": []}
    try:
        with open(ROOMS_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and "rooms" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"current_room": None, "current_confidence": 0.0, "rooms": []}


def _write_rooms(data):
    """Write rooms dict to disk."""
    with open(ROOMS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def identify_room(scene_text, yolo_summary=""):
    """Score each room using exclusion logic. Returns sorted list of
    (room_name, score, confidence) tuples, best first.

    Uses LLM scene description only — YOLO labels are unreliable.

    Scoring:
      +1  per positive_feature found in scene text
      -2  per negative_feature found (exclusion — strong penalty)
      +2  for floor_type match
    """
    data = load_rooms()
    combined = scene_text.lower()
    results = []

    for room in data.get("rooms", []):
        score = 0

        # Positive features: +1 each
        pos_hits = 0
        for feat in room.get("positive_features", []):
            if feat.lower() in combined:
                score += 1
                pos_hits += 1

        # Negative features: -2 each (exclusion)
        neg_hits = 0
        for feat in room.get("negative_features", []):
            if feat.lower() in combined:
                score -= 2
                neg_hits += 1

        # Floor type: +2 (most diagnostic)
        floor = room.get("floor_type", "")
        if floor and floor.lower() in combined:
            score += 2

        results.append((room["name"], score, pos_hits, neg_hits))

    if not results:
        return []

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Compute confidence: gap between #1 and #2
    best_score = results[0][1]
    second_score = results[1][1] if len(results) > 1 else 0
    gap = best_score - second_score
    total_features = results[0][2] + results[0][3]  # pos + neg hits

    if best_score <= 0:
        confidence = 0.0
    elif gap >= 4:
        confidence = 0.95
    elif gap >= 2:
        confidence = 0.75
    elif gap >= 1:
        confidence = 0.55
    else:
        confidence = 0.3

    return [(name, sc, round(confidence if i == 0 else max(0, confidence - 0.3 * (i)), 2))
            for i, (name, sc, _, _) in enumerate(results)]


def get_current_room(scene_text, yolo_summary=""):
    """Identify current room, persist to rooms.json, log result.
    Stores latest scene description for LLM context.
    Returns (room_name, confidence) or (None, 0.0).
    """
    ranked = identify_room(scene_text, yolo_summary)
    if not ranked:
        return None, 0.0

    best_name, best_score, best_conf = ranked[0]

    # Only update if we have some positive signal
    if best_score <= 0:
        scores_str = ", ".join(f"{n}={s:+d}" for n, s, _ in ranked[:4])
        print(f"[room] No match (all scores ≤0) | scores: {scores_str}")
        return None, 0.0

    # Persist
    data = load_rooms()
    data["current_room"] = best_name
    data["current_confidence"] = best_conf

    for room in data.get("rooms", []):
        if room["name"] == best_name:
            room["last_visited"] = time.strftime("%Y-%m-%d %H:%M")
            room["visit_count"] = room.get("visit_count", 0) + 1
            # Store latest scene observation (what the LLM actually saw)
            if scene_text and len(scene_text) > 20:
                room["last_scene"] = scene_text[:300]
            if yolo_summary:
                room["last_yolo"] = yolo_summary[:200]
            break

    _write_rooms(data)

    # Log
    scores_str = ", ".join(f"{n}={s:+d}" for n, s, _ in ranked[:4])
    print(f"[room] Identified: {best_name} (conf={best_conf}) | scores: {scores_str}")

    return best_name, best_conf


def format_room_clues():
    """Compact one-liner for navigator waypoint prompt (replaces hardcoded room_ctx)."""
    data = load_rooms()
    rooms = data.get("rooms", [])
    if not rooms:
        return ""

    clues = []
    for room in rooms:
        name = room["name"].replace("_", " ")
        key_features = room.get("positive_features", [])[:3]
        floor = room.get("floor_type", "")
        landmarks = room.get("entry_landmarks", [])

        parts = []
        if landmarks:
            parts.append(landmarks[0].get("landmark", ""))
        if floor:
            parts.append(floor)
        if key_features:
            parts.append("+".join(key_features[:2]))

        clues.append(f"{name}={', '.join(p for p in parts if p)}")

    current = data.get("current_room")
    prefix = f"Currently in: {current}. " if current else ""
    # Include nav hints and last scene for current room
    hint = ""
    if current:
        for room in rooms:
            if room["name"] == current:
                if room.get("nav_hints"):
                    hint = f" NAV: {room['nav_hints']}"
                break
    return f"ROOM CLUES: {prefix}{'. '.join(clues)}.{hint}\n"


def format_home_layout():
    """Multi-line block for orchestrator route planner."""
    data = load_rooms()
    rooms = data.get("rooms", [])
    if not rooms:
        return "No room data available."

    lines = ["HOME LAYOUT (rooms the rover knows):"]
    for room in rooms:
        name = room["name"].replace("_", " ").title()
        floor = room.get("floor_type", "")
        floor_low = floor.lower()
        # Filter out floor-type duplicates from features
        feats = [f for f in room.get("positive_features", [])[:8]
                 if floor_low not in f.lower() and f.lower() not in floor_low][:5]
        connections = ", ".join(r.replace("_", " ") for r in room.get("connections", []))
        landmarks = room.get("entry_landmarks", [])

        desc_parts = []
        if feats:
            desc_parts.append(", ".join(feats))
        if floor:
            desc_parts.append(floor)
        entry = ""
        if landmarks:
            entry = f". Enter via {landmarks[0].get('landmark', '')}"

        nav_hint = room.get("nav_hints", "")
        hint_str = f" WARNING: {nav_hint}" if nav_hint else ""
        last_scene = room.get("last_scene", "")
        scene_str = f" Last seen: {last_scene}" if last_scene else ""
        line = (f"- {name}: {', '.join(desc_parts)}{entry}{hint_str}."
                f" Connects to: {connections}.{scene_str}")
        lines.append(line)

    current = data.get("current_room")
    if current:
        lines.append(f"\nRover is currently in: {current.replace('_', ' ')}")
        for room in rooms:
            if room["name"] == current and room.get("nav_hints"):
                lines.append(f"NAVIGATION WARNING: {room['nav_hints']}")

    lines.append("\nUse this layout to plan routes with specific room names and visual landmarks.")
    lines.append('For "go to kitchen": exit current room → hallway → find orange arched doorway → enter.')
    return "\n".join(lines)


def format_for_prompt():
    """Return '## Room Knowledge' block for the system prompt."""
    data = load_rooms()
    rooms = data.get("rooms", [])
    if not rooms:
        return ""

    lines = ["## Room Knowledge"]
    current = data.get("current_room")
    if current:
        conf = data.get("current_confidence", 0)
        lines.append(f"Current room: {current.replace('_', ' ')} (confidence: {conf})")

    lines.append("Known rooms:")
    for room in rooms:
        name = room["name"].replace("_", " ")
        floor = room.get("floor_type", "")
        connections = ", ".join(r.replace("_", " ") for r in room.get("connections", []))
        key = ", ".join(room.get("positive_features", [])[:4])
        lines.append(f"- {name}: {key}. Floor: {floor}. → {connections}")

    return "\n".join(lines)


def update_room_features(room_name, features):
    """Add newly observed features to a room's positive_features list."""
    data = load_rooms()
    for room in data.get("rooms", []):
        if room["name"] == room_name:
            existing = set(f.lower() for f in room.get("positive_features", []))
            for feat in features:
                if feat.lower() not in existing:
                    room["positive_features"].append(feat)
            _write_rooms(data)
            return True
    return False


def learn_nav_failure(room_name, failure_scene, failure_reason):
    """Log navigation failures but do NOT save as lessons or nav_hints.
    Auto-learned 'avoid this path' lessons were poisoning route planning
    by telling the orchestrator to avoid correct routes after transient failures."""
    print(f"[room] Nav failure in {room_name}: {failure_reason[:100]} (not saved)")
