"""room_context.py — Room database with LLM-first identification.

Standalone module, no dependencies on rover_brain_llm.
Stores room data in rooms.json alongside this file.

Room identification is now done by the LLM (it sees the camera image and
knows the room list). This module only persists and formats room knowledge —
no more hand-rolled feature-scoring.
"""

import json
import os
import time

ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOMS_FILE = os.path.join(ROVER_DIR, "rooms.json")
ROOM_GRAPH_FILE = os.path.join(ROVER_DIR, "room_graph.json")
TOPO_MAP_FILE = os.path.join(ROVER_DIR, "topo_map.json")


# ── Internal helpers ─────────────────────────────────────────────────

def _normalize_room_name(name):
    return str(name or "").strip().lower().replace(" ", "_")


def _merge_unique_text(existing, new_items, limit=12):
    merged = []
    seen = set()
    for item in list(existing or []) + list(new_items or []):
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
        if limit and len(merged) >= limit:
            break
    return merged


def _merge_landmarks(existing, new_landmarks, limit=6):
    existing_text = []
    for entry in existing or []:
        if isinstance(entry, dict):
            existing_text.append(entry.get("landmark", ""))
        else:
            existing_text.append(str(entry))
    merged = _merge_unique_text(existing_text, new_landmarks, limit=limit)
    return [{"landmark": text} for text in merged]


def _ensure_room_entry(data, room_name):
    room_id = _normalize_room_name(room_name)
    if not room_id:
        return None
    for room in data.get("rooms", []):
        if room.get("name") == room_id:
            return room
    room = {
        "name": room_id,
        "positive_features": [],
        "negative_features": [],
        "floor_type": "",
        "connections": [],
        "entry_landmarks": [],
        "nav_hints": "",
        "last_visited": "",
        "visit_count": 0,
        "last_scene": "",
        "last_yolo": "",
    }
    data.setdefault("rooms", []).append(room)
    return room


def _graph_to_rooms(graph):
    """Convert graph JSON (nodes/edges) to legacy rooms list shape."""
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    rooms = []
    edge_map = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        dst = e.get("target")
        if not src or not dst:
            continue
        edge_map.setdefault(src, set()).add(dst)
        edge_map.setdefault(dst, set()).add(src)

    for n in nodes:
        if not isinstance(n, dict):
            continue
        rid = n.get("id")
        if not rid:
            continue
        rooms.append({
            "name": rid,
            "positive_features": n.get("features", []),
            "negative_features": n.get("negative_features", []),
            "floor_type": n.get("floor_type", ""),
            "connections": sorted(list(edge_map.get(rid, set()))),
            "entry_landmarks": n.get("entry_landmarks", []),
            "nav_hints": n.get("nav_hints", ""),
            "last_visited": n.get("last_visited", ""),
            "visit_count": n.get("visit_count", 0),
            "last_scene": n.get("last_scene", ""),
            "last_yolo": n.get("last_yolo", ""),
        })
    return {
        "current_room": graph.get("current_room"),
        "current_confidence": graph.get("current_confidence", 0.0),
        "rooms": rooms,
    }


def _rooms_to_graph(data):
    """Convert legacy rooms list shape into explicit graph JSON."""
    rooms = data.get("rooms", []) if isinstance(data, dict) else []
    nodes = []
    edges = []
    seen_edges = set()

    for room in rooms:
        if not isinstance(room, dict):
            continue
        name = room.get("name")
        if not name:
            continue
        nodes.append({
            "id": name,
            "label": name.replace("_", " ").title(),
            "features": room.get("positive_features", []),
            "negative_features": room.get("negative_features", []),
            "floor_type": room.get("floor_type", ""),
            "entry_landmarks": room.get("entry_landmarks", []),
            "nav_hints": room.get("nav_hints", ""),
            "last_visited": room.get("last_visited", ""),
            "visit_count": room.get("visit_count", 0),
            "last_scene": room.get("last_scene", ""),
            "last_yolo": room.get("last_yolo", ""),
        })
        for dst in room.get("connections", []):
            if not dst:
                continue
            a, b = sorted([name, dst])
            key = (a, b)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({
                "source": a, "target": b,
                "type": "connected", "weight": 1.0,
            })

    return {
        "version": 1,
        "current_room": data.get("current_room"),
        "current_confidence": data.get("current_confidence", 0.0),
        "nodes": nodes,
        "edges": edges,
    }


def _load_topo_data():
    if not os.path.exists(TOPO_MAP_FILE):
        return {}
    try:
        with open(TOPO_MAP_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and "nodes" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _relationship_lines(current_room=None, target_room=None, max_lines=4):
    topo = _load_topo_data()
    if not topo:
        return []

    current_room = _normalize_room_name(current_room)
    target_room = _normalize_room_name(target_room)
    lines = []

    for node in topo.get("nodes", []):
        if node.get("type") != "transition":
            continue
        views = node.get("semantic_views", {})
        if not isinstance(views, dict) or not views:
            continue

        if current_room and current_room in views:
            pairs = [(current_room, views[current_room])]
        else:
            pairs = list(views.items())

        for from_room, view in pairs:
            if not isinstance(view, dict):
                continue
            to_room = _normalize_room_name(view.get("to_room"))
            if target_room and to_room and to_room != target_room:
                continue

            doorway = ", ".join(view.get("doorway_landmarks", [])[:2])
            inside = ", ".join(view.get("inside_features", [])[:3])
            hint = str(view.get("navigational_hint", "")).strip()
            parts = [f"{from_room.replace('_', ' ')} -> {to_room.replace('_', ' ')}"]
            if doorway:
                parts.append(f"door near {doorway}")
            if inside:
                parts.append(f"inside {inside}")
            if hint:
                parts.append(hint)
            lines.append("; ".join(parts))
            if len(lines) >= max_lines:
                return lines

    return lines


# ── Public API — Load / Save ─────────────────────────────────────────

def load_rooms():
    """Read rooms.json, return full dict. Handles missing/corrupt."""
    if not os.path.exists(ROOMS_FILE):
        if os.path.exists(ROOM_GRAPH_FILE):
            try:
                with open(ROOM_GRAPH_FILE) as f:
                    return _graph_to_rooms(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        return {"current_room": None, "current_confidence": 0.0, "rooms": []}
    try:
        with open(ROOMS_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and "rooms" in data:
            if not os.path.exists(ROOM_GRAPH_FILE):
                try:
                    with open(ROOM_GRAPH_FILE, "w") as gf:
                        json.dump(_rooms_to_graph(data), gf, indent=2)
                except OSError:
                    pass
            return data
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            return _graph_to_rooms(data)
    except (json.JSONDecodeError, OSError):
        pass
    return {"current_room": None, "current_confidence": 0.0, "rooms": []}


def _write_rooms(data):
    """Write rooms dict to disk and mirror graph representation."""
    with open(ROOMS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    try:
        graph = _rooms_to_graph(data)
        with open(ROOM_GRAPH_FILE, "w") as f:
            json.dump(graph, f, indent=2)
    except OSError:
        pass


def load_room_graph():
    """Return explicit room graph JSON (nodes + edges)."""
    if os.path.exists(ROOM_GRAPH_FILE):
        try:
            with open(ROOM_GRAPH_FILE) as f:
                graph = json.load(f)
            if isinstance(graph, dict) and "nodes" in graph and "edges" in graph:
                return graph
        except (json.JSONDecodeError, OSError):
            pass
    data = load_rooms()
    graph = _rooms_to_graph(data)
    try:
        with open(ROOM_GRAPH_FILE, "w") as f:
            json.dump(graph, f, indent=2)
    except OSError:
        pass
    return graph


# ── Room identification — now LLM-driven ─────────────────────────────

def identify_room(scene_text, yolo_summary=""):
    """Return known rooms as candidates for LLM to pick from.

    OLD behaviour: hand-rolled scoring (+1 per feature, -2 per negative).
    NEW behaviour: return room list with features so the LLM (which sees
    the actual camera image) can decide. We still return the same
    (name, score, confidence) tuple format for compatibility, but scores
    are now just based on simple keyword overlap — the LLM makes the
    real decision via get_current_room().
    """
    data = load_rooms()
    combined = scene_text.lower() if scene_text else ""
    results = []

    for room in data.get("rooms", []):
        # Simple keyword overlap — not the decision-maker, just a hint
        hits = sum(1 for feat in room.get("positive_features", [])
                   if feat.lower() in combined)
        floor = room.get("floor_type", "")
        if floor and floor.lower() in combined:
            hits += 1
        results.append((room["name"], hits, 0.5 if hits > 0 else 0.1))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_current_room(scene_text, yolo_summary=""):
    """Persist current room identified by the LLM.

    The LLM now includes "room":"<name>" in its response JSON.
    This function is called after the LLM has already decided — it just
    persists the result. Falls back to keyword hints if no LLM decision.

    Returns (room_name, confidence) or (None, 0.0).
    """
    ranked = identify_room(scene_text, yolo_summary)
    if not ranked:
        return None, 0.0

    best_name, best_score, best_conf = ranked[0]
    if best_score <= 0:
        return None, 0.0

    data = load_rooms()
    data["current_room"] = best_name
    data["current_confidence"] = best_conf

    for room in data.get("rooms", []):
        if room["name"] == best_name:
            room["last_visited"] = time.strftime("%Y-%m-%d %H:%M")
            room["visit_count"] = room.get("visit_count", 0) + 1
            if scene_text and len(scene_text) > 20:
                room["last_scene"] = scene_text[:300]
            if yolo_summary:
                room["last_yolo"] = yolo_summary[:200]
            break

    _write_rooms(data)
    print(f"[room] Identified: {best_name} (conf={best_conf})")
    return best_name, best_conf


# ── Format for prompts ───────────────────────────────────────────────

def room_list_for_llm():
    """Return compact room list for LLM to pick from in its response.

    This replaces the old scoring logic — the LLM sees this list plus
    the camera image and decides which room it's in.
    """
    data = load_rooms()
    rooms = data.get("rooms", [])
    if not rooms:
        return ""
    lines = []
    for room in rooms:
        name = room["name"].replace("_", " ")
        feats = ", ".join(room.get("positive_features", [])[:4])
        floor = room.get("floor_type", "")
        parts = [feats] if feats else []
        if floor:
            parts.append(f"floor: {floor}")
        lines.append(f"  - {name}: {'; '.join(parts)}")
    return "Known rooms:\n" + "\n".join(lines)


def format_room_clues(target_room=None):
    """Compact one-liner for navigator waypoint prompt."""
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
    hint = ""
    if current:
        for room in rooms:
            if room["name"] == current:
                if room.get("nav_hints"):
                    hint = f" NAV: {room['nav_hints']}"
                break
    rel_lines = _relationship_lines(current_room=current, target_room=target_room,
                                    max_lines=3)
    rel = ""
    if rel_lines:
        rel = " RELATION CLUES: " + " | ".join(rel_lines)
    return f"ROOM CLUES: {prefix}{'. '.join(clues)}.{hint}{rel}\n"


def format_relationship_clues(current_room=None, target_room=None, max_lines=3):
    """Compact relationship-oriented doorway hints for prompts."""
    lines = _relationship_lines(current_room=current_room, target_room=target_room,
                                max_lines=max_lines)
    if not lines:
        return ""
    return "RELATION CLUES: " + " | ".join(lines) + "\n"


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

    rel_lines = _relationship_lines(current_room=current, max_lines=6)
    if rel_lines:
        lines.append("\nLEARNED DOORWAY RELATIONSHIPS:")
        lines.extend(f"- {line}" for line in rel_lines)

    lines.append("\nUse this layout to plan routes with specific room names and visual landmarks.")
    lines.append('For "go to kitchen": exit current room → hallway → find orange arched doorway → enter.')
    return "\n".join(lines)


def format_for_prompt():
    """Return '## Room Knowledge' block for the system prompt.

    Now includes the room list so the LLM can identify rooms from the
    camera image without needing a separate scoring function.
    """
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
        scene = room.get("last_scene", "")
        scene_str = f" | Last seen: {scene[:120]}" if scene else ""
        lines.append(f"- {name}: {key}. Floor: {floor}. → {connections}{scene_str}")

    rel_lines = _relationship_lines(current_room=current, max_lines=4)
    if rel_lines:
        lines.append("Known doorway relationships:")
        lines.extend(f"- {line}" for line in rel_lines)

    lines.append("\nWhen you see a room, include \"room\":\"<name>\" in your "
                 "response to help track location.")

    return "\n".join(lines)


# ── Room mutation ────────────────────────────────────────────────────

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


def link_rooms(room_a, room_b):
    """Ensure two rooms are connected in rooms.json."""
    a_name = _normalize_room_name(room_a)
    b_name = _normalize_room_name(room_b)
    if not a_name or not b_name or a_name == b_name:
        return False

    data = load_rooms()
    a_room = _ensure_room_entry(data, a_name)
    b_room = _ensure_room_entry(data, b_name)

    changed = False
    if b_name not in a_room.get("connections", []):
        a_room.setdefault("connections", []).append(b_name)
        a_room["connections"] = sorted(_merge_unique_text(a_room["connections"], [], limit=20))
        changed = True
    if a_name not in b_room.get("connections", []):
        b_room.setdefault("connections", []).append(a_name)
        b_room["connections"] = sorted(_merge_unique_text(b_room["connections"], [], limit=20))
        changed = True

    if changed:
        _write_rooms(data)
    return changed


def merge_room_observation(room_name, features=None, entry_landmarks=None,
                           scene_text="", yolo_summary="", floor_type="",
                           connected_to=None, mark_current=False,
                           confidence=None):
    """Merge a successful arrival observation into rooms.json."""
    room_id = _normalize_room_name(room_name)
    if not room_id:
        return False

    data = load_rooms()
    room = _ensure_room_entry(data, room_id)
    changed = False

    merged_features = _merge_unique_text(
        room.get("positive_features", []), features or [], limit=40)
    if merged_features != room.get("positive_features", []):
        room["positive_features"] = merged_features
        changed = True

    merged_landmarks = _merge_landmarks(
        room.get("entry_landmarks", []), entry_landmarks or [], limit=8)
    if merged_landmarks != room.get("entry_landmarks", []):
        room["entry_landmarks"] = merged_landmarks
        changed = True

    if floor_type and not room.get("floor_type"):
        room["floor_type"] = str(floor_type).strip()
        changed = True

    if scene_text:
        clipped = str(scene_text).strip()[:300]
        if clipped and clipped != room.get("last_scene"):
            room["last_scene"] = clipped
            changed = True

    if yolo_summary:
        clipped = str(yolo_summary).strip()[:200]
        if clipped and clipped != room.get("last_yolo"):
            room["last_yolo"] = clipped
            changed = True

    if mark_current:
        data["current_room"] = room_id
        if confidence is not None:
            data["current_confidence"] = round(float(confidence), 2)
        room["last_visited"] = time.strftime("%Y-%m-%d %H:%M")
        room["visit_count"] = room.get("visit_count", 0) + 1
        changed = True

    if connected_to:
        other = _ensure_room_entry(data, connected_to)
        if other and other["name"] != room_id:
            if other["name"] not in room.get("connections", []):
                room.setdefault("connections", []).append(other["name"])
                room["connections"] = sorted(_merge_unique_text(room["connections"], [], limit=20))
                changed = True
            if room_id not in other.get("connections", []):
                other.setdefault("connections", []).append(room_id)
                other["connections"] = sorted(_merge_unique_text(other["connections"], [], limit=20))
                changed = True

    if changed:
        _write_rooms(data)
    return changed


def learn_nav_failure(room_name, failure_scene, failure_reason):
    """Log navigation failures but do NOT save as lessons or nav_hints."""
    print(f"[room] Nav failure in {room_name}: {failure_reason[:100]} (not saved)")
