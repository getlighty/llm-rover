"""Topological navigation — simplified, LLM-first.

Models the home as a graph of rooms connected by transitions (doorways).
Plans routes as sequences of legs and provides rich context for the LLM
to navigate each leg visually.

Simplified from the original:
- TopoNode class → plain dicts
- Same BFS routing (it's 20 lines and works)
- Same semantic_views learning (genuinely useful)
- Removed redundant class boilerplate

Node types:
  room       — a space the rover can be inside (office, hallway, kitchen)
  transition — a doorway/threshold between two rooms

Each transition stores semantic_views: what the doorway looks like from
each side, what's visible inside, and a navigational hint. These are
learned by the LLM after each successful room crossing.
"""

import json
import os
import time
from collections import deque

MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo_map.json")


def _merge_unique(existing, new_items, limit=8):
    merged = []
    seen = set()
    for value in list(existing or []) + list(new_items or []):
        text = str(value).strip()
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


class Leg:
    """One segment of a route: cross from current room through a transition."""

    def __init__(self, from_room, transition, to_room):
        self.from_room = from_room
        self.transition = transition
        self.to_room = to_room

    def __repr__(self):
        return f"{self.from_room} →[{self.transition}]→ {self.to_room}"

    def __eq__(self, other):
        if not isinstance(other, Leg):
            return NotImplemented
        return (self.from_room == other.from_room
                and self.transition == other.transition
                and self.to_room == other.to_room)


class TopoMap:
    """Topological map of the home — rooms connected by doorways.

    Nodes and edges are stored as simple dicts internally.
    """

    def __init__(self):
        self.nodes = {}      # id -> dict
        self.edges = {}      # id -> set of connected node ids
        self.current_room = None
        self.current_confidence = 0.0
        self._load()

    # ── Graph construction ───────────────────────────────────────────

    def _add_node(self, node_dict):
        """Add a node dict to the graph."""
        nid = node_dict["id"]
        self.nodes[nid] = node_dict
        if nid not in self.edges:
            self.edges[nid] = set()

    def add_edge(self, a_id, b_id):
        self.edges.setdefault(a_id, set()).add(b_id)
        self.edges.setdefault(b_id, set()).add(a_id)

    def rooms(self):
        return [n for n in self.nodes.values() if n.get("type") == "room"]

    def transitions(self):
        return [n for n in self.nodes.values() if n.get("type") == "transition"]

    def neighbors(self, node_id):
        return self.edges.get(node_id, set())

    def transition_between(self, room_a, room_b):
        """Find the transition node connecting two rooms, or None."""
        for t in self.transitions():
            nbrs = self.neighbors(t["id"])
            if room_a in nbrs and room_b in nbrs:
                return t
        return None

    def ensure_room(self, room_id, label="", features=None, floor_type=""):
        """Ensure a room node exists and optionally enrich it."""
        node = self.nodes.get(room_id)
        if node and node.get("type") == "room":
            if features:
                node["features"] = _merge_unique(node.get("features", []), features, limit=20)
            if floor_type and not node.get("floor_type"):
                node["floor_type"] = floor_type
            if label and not node.get("label"):
                node["label"] = label
            return node

        node = {
            "id": room_id,
            "type": "room",
            "label": label or room_id.replace("_", " ").title(),
            "features": list(features or []),
            "floor_type": floor_type or "",
        }
        self._add_node(node)
        return node

    def ensure_transition_between(self, room_a, room_b, transition_id=None,
                                  label=""):
        """Ensure a transition node exists between two rooms."""
        existing = self.transition_between(room_a, room_b)
        if existing:
            return existing

        if not transition_id:
            a, b = sorted([room_a, room_b])
            transition_id = f"{a}_{b}_transition"

        node = {
            "id": transition_id,
            "type": "transition",
            "label": label or transition_id.replace("_", " ").title(),
            "visual_cues": [],
            "semantic_views": {},
        }
        self._add_node(node)
        self.add_edge(room_a, transition_id)
        self.add_edge(transition_id, room_b)
        return node

    # ── Learning ─────────────────────────────────────────────────────

    def update_room_semantics(self, room_id, features=None, floor_type=""):
        """Merge newly learned room features."""
        room = self.ensure_room(room_id, features=features, floor_type=floor_type)
        if features:
            room["features"] = _merge_unique(room.get("features", []), features, limit=20)
        if floor_type and not room.get("floor_type"):
            room["floor_type"] = floor_type
        return room

    def update_transition_semantics(self, transition_id, from_room, to_room,
                                    doorway_landmarks=None, inside_features=None,
                                    navigational_hint="", inside_room_guess="",
                                    confidence=0.0, scene_text=""):
        """Store learned relationship cues for a transition."""
        if not from_room or not to_room:
            return None

        self.ensure_room(from_room)
        self.ensure_room(to_room)
        node = self.nodes.get(transition_id)
        if not node or node.get("type") != "transition":
            node = self.ensure_transition_between(
                from_room, to_room, transition_id=transition_id)

        views = node.setdefault("semantic_views", {})
        view = views.get(from_room, {})
        if not isinstance(view, dict):
            view = {}

        view["to_room"] = to_room
        view["doorway_landmarks"] = _merge_unique(
            view.get("doorway_landmarks", []), doorway_landmarks, limit=6)
        view["inside_features"] = _merge_unique(
            view.get("inside_features", []), inside_features, limit=6)
        if inside_room_guess:
            view["inside_room_guess"] = inside_room_guess
        if navigational_hint:
            view["navigational_hint"] = str(navigational_hint).strip()[:200]
        if scene_text:
            view["last_scene"] = str(scene_text).strip()[:240]
        if confidence:
            view["confidence"] = round(max(
                float(view.get("confidence", 0.0) or 0.0),
                min(1.0, float(confidence))), 2)
        view["last_observed"] = time.strftime("%Y-%m-%d %H:%M")
        view["observation_count"] = int(view.get("observation_count", 0) or 0) + 1

        views[from_room] = view
        node["observation_count"] = max(
            int(node.get("observation_count", 0) or 0),
            view["observation_count"])

        combined_cues = list(doorway_landmarks or []) + list(inside_features or [])
        if combined_cues:
            node["visual_cues"] = _merge_unique(
                node.get("visual_cues", []), combined_cues, limit=10)
        if navigational_hint and not node.get("nav_hints"):
            node["nav_hints"] = str(navigational_hint).strip()[:200]
        return view

    # ── Route planning ───────────────────────────────────────────────

    def plan_route(self, from_room, to_room):
        """BFS shortest path. Returns list of Leg objects."""
        if from_room == to_room:
            return []
        if from_room not in self.nodes or to_room not in self.nodes:
            return []

        visited = {from_room}
        queue = deque([(from_room, [])])

        while queue:
            current, path = queue.popleft()
            for t_id in self.neighbors(current):
                t_node = self.nodes.get(t_id)
                if not t_node or t_node.get("type") != "transition":
                    continue
                for r_id in self.neighbors(t_id):
                    r_node = self.nodes.get(r_id)
                    if not r_node or r_node.get("type") != "room" or r_id in visited:
                        continue
                    new_path = path + [Leg(current, t_id, r_id)]
                    if r_id == to_room:
                        return new_path
                    visited.add(r_id)
                    queue.append((r_id, new_path))

        return []

    def leg_instruction(self, leg, *, next_leg=None):
        """Build navigation instruction for a single leg.

        Returns dict with everything the LLM needs to navigate this leg.
        If next_leg is provided, includes directional context for where to
        go after entering the target room.
        """
        t_node = self.nodes.get(leg.transition, {})
        to_node = self.nodes.get(leg.to_room, {})
        from_node = self.nodes.get(leg.from_room, {})

        if not t_node or not to_node:
            return {"target": leg.to_room, "hint": f"go to {leg.to_room}"}

        instruction = {
            "target_transition": leg.transition,
            "target_room": leg.to_room,
            "visual_cues": t_node.get("visual_cues", []),
            "exit_hint": t_node.get("nav_hints") or f"find the {t_node.get('label', leg.transition)}",
            "expected_floor": to_node.get("floor_type", ""),
            "verify_features": to_node.get("features", [])[:5],
            "doorway_width_m": t_node.get("width_m", 0.8),
        }

        azimuth = t_node.get("azimuth_from", {}).get(leg.from_room)
        if azimuth is not None:
            instruction["expected_azimuth_deg"] = azimuth

        if from_node.get("nav_hints"):
            instruction["room_nav_hints"] = from_node["nav_hints"]

        semantic_view = t_node.get("semantic_views", {}).get(leg.from_room, {})
        if isinstance(semantic_view, dict):
            doorway_landmarks = semantic_view.get("doorway_landmarks", [])
            inside_features = semantic_view.get("inside_features", [])
            relationship_hint = semantic_view.get("navigational_hint", "")
            if doorway_landmarks:
                instruction["doorway_landmarks"] = doorway_landmarks[:4]
            if inside_features:
                instruction["inside_features"] = inside_features[:5]
            if relationship_hint:
                instruction["relationship_hint"] = relationship_hint
            if semantic_view.get("confidence"):
                instruction["relationship_confidence"] = semantic_view["confidence"]

        # Add next-leg direction context so the LLM knows where to go
        # after entering the target room (prevents wrong turns in hub rooms)
        if next_leg:
            next_t_node = self.nodes.get(next_leg.transition, {})
            next_azimuth = next_t_node.get("azimuth_from", {}).get(leg.to_room)
            next_label = next_t_node.get("label", next_leg.transition)
            next_nav_hints = next_t_node.get("nav_hints", "")
            direction_text = self._azimuth_to_direction(next_azimuth)
            hint = (
                f"AFTER entering {leg.to_room}: turn {direction_text} "
                f"to find '{next_label}' leading to {next_leg.to_room}."
            )
            if next_nav_hints:
                hint += f" {next_nav_hints}"
            instruction["next_leg_direction"] = hint
            if next_azimuth is not None:
                instruction["next_leg_azimuth_deg"] = next_azimuth

        return instruction

    @staticmethod
    def _azimuth_to_direction(azimuth):
        """Convert azimuth degrees to a human-readable direction word."""
        if azimuth is None:
            return "to find"
        a = float(azimuth)
        if -20 <= a <= 20:
            return "STRAIGHT AHEAD"
        elif 20 < a <= 70:
            return "SLIGHTLY RIGHT"
        elif 70 < a <= 110:
            return "RIGHT"
        elif a > 110:
            return "HARD RIGHT / BEHIND-RIGHT"
        elif -70 <= a < -20:
            return "SLIGHTLY LEFT"
        elif -110 <= a < -70:
            return "LEFT"
        else:
            return "HARD LEFT / BEHIND-LEFT"

    def route_summary(self, legs):
        """Human-readable route description with turn directions."""
        if not legs:
            return "Already there."
        parts = []
        for i, leg in enumerate(legs):
            t = self.nodes.get(leg.transition, {})
            label = t.get("label", leg.transition)
            azimuth = t.get("azimuth_from", {}).get(leg.from_room)
            direction = self._azimuth_to_direction(azimuth) if azimuth is not None else ""
            dir_tag = f" ({direction})" if direction else ""
            parts.append(f"{leg.from_room} →[{label}{dir_tag}]→ {leg.to_room}")
        return " | ".join(parts)

    def rooms_through(self, transition_id):
        """Return room ids connected by a transition."""
        nbrs = self.neighbors(transition_id)
        return [n for n in nbrs if self.nodes.get(n, {}).get("type") == "room"]

    # ── Persistence ──────────────────────────────────────────────────

    def save(self):
        data = {
            "current_room": self.current_room,
            "current_confidence": self.current_confidence,
            "nodes": list(self.nodes.values()),
            "edges": [],
        }
        seen = set()
        for a, neighbors in self.edges.items():
            for b in neighbors:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    data["edges"].append({"a": a, "b": b})
        with open(MAP_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(MAP_FILE):
            self._build_default()
            self.save()
            return
        try:
            with open(MAP_FILE) as f:
                data = json.load(f)
            self.current_room = data.get("current_room")
            self.current_confidence = data.get("current_confidence", 0.0)
            for nd in data.get("nodes", []):
                self._add_node(nd)
            for ed in data.get("edges", []):
                self.add_edge(ed["a"], ed["b"])
            print(f"[topo] Loaded {len(self.rooms())} rooms, "
                  f"{len(self.transitions())} transitions")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[topo] Load error: {e}, building default")
            self._build_default()
            self.save()

    def _build_default(self):
        """Build the default home topology."""
        # --- Rooms ---
        for rid, label, features, floor, hints in [
            ("office", "Office",
             ["desk", "monitor", "office chair", "wooden floor",
              "yellow walls", "cables", "shelving"],
             "wooden floor",
             "Exit is the doorway with bright hallway light. "
             "Go around the chair, not through it."),
            ("hallway", "Hallway",
             ["stone tiles", "narrow corridor", "plant pot",
              "yellow walls", "arched doorway"],
             "stone tiles", ""),
            ("kitchen", "Kitchen",
             ["stove", "oven", "refrigerator", "sink", "counter", "cabinets"],
             "tiles", ""),
            ("living_room", "Living Room",
             ["sofa", "couch", "tv", "coffee table", "parquet floor", "rug"],
             "parquet floor", ""),
            ("dining", "Dining Room",
             ["dining table", "chairs", "tableware"],
             "tiles", ""),
            ("bathroom", "Bathroom",
             ["toilet", "shower", "bathtub", "sink", "tiles", "mirror"],
             "tiles", ""),
        ]:
            self._add_node({
                "id": rid, "type": "room", "label": label,
                "features": features, "floor_type": floor,
                "nav_hints": hints,
            })

        # --- Transitions ---
        for tid, label, cues, azimuth, hints, width in [
            ("office_hall_door", "Office-Hallway Door",
             ["wooden door frame", "bright hallway light beyond",
              "floor changes from wood to stone tiles"],
             {"office": -45, "hallway": 180},
             "Look for the doorway with bright light. "
             "Floor changes from wooden to stone tiles.", 0.8),
            ("hall_kitchen_arch", "Hallway-Kitchen Arch",
             ["orange arched doorway", "kitchen counter visible",
              "tiled floor beyond"],
             {"hallway": 90},
             "Turn right in the hallway. "
             "Look for the orange arched doorway.", 1.0),
            ("hall_living_arch", "Hallway-Living Room Arch",
             ["arched doorway", "parquet floor beyond",
              "couch visible through opening"],
             {"hallway": -90},
             "Turn left in the hallway. "
             "Look for the arch with wooden floor beyond.", 1.0),
            ("kitchen_dining_pass", "Kitchen-Dining Passage",
             ["open passage", "dining table visible"],
             {"kitchen": 0},
             "Straight through the kitchen.", 1.2),
            ("living_dining_pass", "Living-Dining Passage",
             ["open passage", "dining table visible"],
             {"living_room": 90},
             "Through the opening on the right.", 1.2),
            ("hall_bathroom_door", "Hallway-Bathroom Door",
             ["bathroom door", "tiled floor beyond", "toilet visible"],
             {"hallway": 0},
             "Look for the bathroom door in the hallway.", 0.7),
        ]:
            self._add_node({
                "id": tid, "type": "transition", "label": label,
                "visual_cues": cues, "azimuth_from": azimuth,
                "nav_hints": hints, "width_m": width,
                "semantic_views": {},
            })

        # --- Edges ---
        for a, b in [
            ("office", "office_hall_door"),
            ("office_hall_door", "hallway"),
            ("hallway", "hall_kitchen_arch"),
            ("hall_kitchen_arch", "kitchen"),
            ("hallway", "hall_living_arch"),
            ("hall_living_arch", "living_room"),
            ("kitchen", "kitchen_dining_pass"),
            ("kitchen_dining_pass", "dining"),
            ("living_room", "living_dining_pass"),
            ("living_dining_pass", "dining"),
            ("hallway", "hall_bathroom_door"),
            ("hall_bathroom_door", "bathroom"),
        ]:
            self.add_edge(a, b)

        self.current_room = "office"
        self.current_confidence = 0.5
        print(f"[topo] Built default map: {len(self.rooms())} rooms, "
              f"{len(self.transitions())} transitions")
