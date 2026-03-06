"""Topological navigation orchestrator.

Models the home as a graph of rooms connected by transitions (doorways).
Plans routes as sequences of legs (transition crossings) and dispatches
each leg to the reactive navigator with image-grounded instructions.

Architecture:
  TopoMap       — graph of rooms + transitions, loaded from topo_map.json
  RoutePlanner  — BFS shortest path, returns list of Leg objects
  Orchestrator  — drives the navigator through each leg, verifies crossings

Node types:
  room       — a space the rover can be inside (office, hallway, kitchen)
  transition — a doorway/threshold between two rooms

Each transition stores:
  - visual_cues: what the doorway looks like from the source room
  - azimuth_deg: expected heading when facing the doorway (relative to room entry)
  - floor_change: expected floor type on the other side
  - width_m: approximate doorway width (for clearance)
"""

import json
import os
import time
from collections import deque

MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo_map.json")


class TopoNode:
    """A node in the topological graph (room or transition)."""

    def __init__(self, node_id, node_type="room", label="", **attrs):
        self.id = node_id
        self.type = node_type  # "room" or "transition"
        self.label = label or node_id.replace("_", " ").title()
        self.features = attrs.get("features", [])
        self.floor_type = attrs.get("floor_type", "")
        self.visual_cues = attrs.get("visual_cues", [])
        self.azimuth_from = attrs.get("azimuth_from", {})  # {room_id: degrees}
        self.width_m = attrs.get("width_m", 0.8)
        self.nav_hints = attrs.get("nav_hints", "")
        sem = attrs.get("semantic_views", {})
        self.semantic_views = sem if isinstance(sem, dict) else {}
        self.observation_count = int(attrs.get("observation_count", 0) or 0)

    def to_dict(self):
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
        }
        if self.features:
            d["features"] = self.features
        if self.floor_type:
            d["floor_type"] = self.floor_type
        if self.visual_cues:
            d["visual_cues"] = self.visual_cues
        if self.azimuth_from:
            d["azimuth_from"] = self.azimuth_from
        if self.width_m != 0.8:
            d["width_m"] = self.width_m
        if self.nav_hints:
            d["nav_hints"] = self.nav_hints
        if self.semantic_views:
            d["semantic_views"] = self.semantic_views
        if self.observation_count:
            d["observation_count"] = self.observation_count
        return d


class Leg:
    """One segment of a route: cross from current room through a transition."""

    def __init__(self, from_room, transition, to_room):
        self.from_room = from_room        # room node id
        self.transition = transition      # transition node id
        self.to_room = to_room            # room node id (destination)

    def __repr__(self):
        return f"{self.from_room} →[{self.transition}]→ {self.to_room}"


class TopoMap:
    """Transition-aware topological map of the home."""

    def __init__(self):
        self.nodes = {}      # id -> TopoNode
        self.edges = {}      # id -> set of connected node ids
        self.current_room = None
        self.current_confidence = 0.0
        self._load()

    def add_node(self, node):
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = set()

    def add_edge(self, a_id, b_id):
        self.edges.setdefault(a_id, set()).add(b_id)
        self.edges.setdefault(b_id, set()).add(a_id)

    @staticmethod
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

    def rooms(self):
        return [n for n in self.nodes.values() if n.type == "room"]

    def transitions(self):
        return [n for n in self.nodes.values() if n.type == "transition"]

    def neighbors(self, node_id):
        return self.edges.get(node_id, set())

    def transition_between(self, room_a, room_b):
        """Find the transition node connecting two rooms, or None."""
        for t in self.transitions():
            nbrs = self.neighbors(t.id)
            if room_a in nbrs and room_b in nbrs:
                return t
        return None

    def ensure_room(self, room_id, label="", features=None, floor_type=""):
        """Ensure a room node exists and optionally enrich it."""
        room = self.nodes.get(room_id)
        if room and room.type == "room":
            if features:
                room.features = self._merge_unique(room.features, features, limit=20)
            if floor_type and not room.floor_type:
                room.floor_type = floor_type
            if label and not room.label:
                room.label = label
            return room

        room = TopoNode(
            room_id, "room",
            label or room_id.replace("_", " ").title(),
            features=list(features or []),
            floor_type=floor_type,
        )
        self.add_node(room)
        return room

    def ensure_transition_between(self, room_a, room_b, transition_id=None,
                                  label=""):
        """Ensure a transition node exists between two rooms."""
        existing = self.transition_between(room_a, room_b)
        if existing:
            return existing

        if not transition_id:
            a, b = sorted([room_a, room_b])
            transition_id = f"{a}_{b}_transition"

        transition = TopoNode(
            transition_id, "transition",
            label or transition_id.replace("_", " ").title(),
        )
        self.add_node(transition)
        self.add_edge(room_a, transition.id)
        self.add_edge(transition.id, room_b)
        return transition

    def update_room_semantics(self, room_id, features=None, floor_type=""):
        """Merge newly learned room features into a room node."""
        room = self.ensure_room(room_id, features=features, floor_type=floor_type)
        if features:
            room.features = self._merge_unique(room.features, features, limit=20)
        if floor_type and not room.floor_type:
            room.floor_type = floor_type
        return room

    def update_transition_semantics(self, transition_id, from_room, to_room,
                                    doorway_landmarks=None, inside_features=None,
                                    navigational_hint="", inside_room_guess="",
                                    confidence=0.0, scene_text=""):
        """Store learned relationship cues for a transition as seen from one room."""
        if not from_room or not to_room:
            return None

        self.ensure_room(from_room)
        self.ensure_room(to_room)
        transition = self.nodes.get(transition_id)
        if not transition or transition.type != "transition":
            transition = self.ensure_transition_between(
                from_room, to_room, transition_id=transition_id)

        view = transition.semantic_views.get(from_room, {})
        if not isinstance(view, dict):
            view = {}

        view["to_room"] = to_room
        view["doorway_landmarks"] = self._merge_unique(
            view.get("doorway_landmarks", []), doorway_landmarks, limit=6)
        view["inside_features"] = self._merge_unique(
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

        transition.semantic_views[from_room] = view
        transition.observation_count = max(
            transition.observation_count, view["observation_count"])

        combined_cues = list(doorway_landmarks or []) + list(inside_features or [])
        if combined_cues:
            transition.visual_cues = self._merge_unique(
                transition.visual_cues, combined_cues, limit=10)
        if navigational_hint and not transition.nav_hints:
            transition.nav_hints = str(navigational_hint).strip()[:200]
        return view

    def rooms_through(self, transition_id):
        """Return the two room ids connected by a transition."""
        nbrs = self.neighbors(transition_id)
        return [n for n in nbrs if self.nodes.get(n, TopoNode("")).type == "room"]

    def plan_route(self, from_room, to_room):
        """BFS shortest path from one room to another.
        Returns list of Leg objects, or empty list if no path."""
        if from_room == to_room:
            return []
        if from_room not in self.nodes or to_room not in self.nodes:
            return []

        # BFS over rooms only, via transitions
        visited = {from_room}
        queue = deque([(from_room, [])])

        while queue:
            current, path = queue.popleft()

            # Find transitions from current room
            for t_id in self.neighbors(current):
                t_node = self.nodes.get(t_id)
                if not t_node or t_node.type != "transition":
                    continue
                # Find room on other side of transition
                for r_id in self.neighbors(t_id):
                    r_node = self.nodes.get(r_id)
                    if not r_node or r_node.type != "room" or r_id in visited:
                        continue
                    new_path = path + [Leg(current, t_id, r_id)]
                    if r_id == to_room:
                        return new_path
                    visited.add(r_id)
                    queue.append((r_id, new_path))

        return []

    def leg_instruction(self, leg):
        """Build navigation instruction for a single leg.

        Returns dict with everything the navigator needs:
        - target_transition: id of the doorway to find and cross
        - visual_cues: what to look for
        - exit_hint: human-readable hint
        - expected_floor: what floor to expect after crossing
        - verify_room: room features to check after crossing
        """
        t_node = self.nodes.get(leg.transition)
        to_node = self.nodes.get(leg.to_room)
        from_node = self.nodes.get(leg.from_room)

        if not t_node or not to_node:
            return {"target": leg.to_room, "hint": f"go to {leg.to_room}"}

        instruction = {
            "target_transition": leg.transition,
            "target_room": leg.to_room,
            "visual_cues": t_node.visual_cues,
            "exit_hint": t_node.nav_hints or f"find the {t_node.label}",
            "expected_floor": to_node.floor_type,
            "verify_features": to_node.features[:5],
            "doorway_width_m": t_node.width_m,
        }

        # Add azimuth hint if available
        azimuth = t_node.azimuth_from.get(leg.from_room)
        if azimuth is not None:
            instruction["expected_azimuth_deg"] = azimuth

        # Add from-room nav hints
        if from_node and from_node.nav_hints:
            instruction["room_nav_hints"] = from_node.nav_hints

        semantic_view = t_node.semantic_views.get(leg.from_room, {})
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

        return instruction

    def route_summary(self, legs):
        """Human-readable route description."""
        if not legs:
            return "Already there."
        parts = []
        for leg in legs:
            t = self.nodes.get(leg.transition)
            label = t.label if t else leg.transition
            parts.append(f"{leg.from_room} →[{label}]→ {leg.to_room}")
        return " | ".join(parts)

    def save(self):
        data = {
            "current_room": self.current_room,
            "current_confidence": self.current_confidence,
            "nodes": [n.to_dict() for n in self.nodes.values()],
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
                self.add_node(TopoNode(
                    nd["id"], nd.get("type", "room"), nd.get("label", ""),
                    features=nd.get("features", []),
                    floor_type=nd.get("floor_type", ""),
                    visual_cues=nd.get("visual_cues", []),
                    azimuth_from=nd.get("azimuth_from", {}),
                    width_m=nd.get("width_m", 0.8),
                    nav_hints=nd.get("nav_hints", ""),
                    semantic_views=nd.get("semantic_views", {}),
                    observation_count=nd.get("observation_count", 0),
                ))
            for ed in data.get("edges", []):
                self.add_edge(ed["a"], ed["b"])
            print(f"[topo] Loaded {len(self.rooms())} rooms, "
                  f"{len(self.transitions())} transitions")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[topo] Load error: {e}, building default")
            self._build_default()
            self.save()

    def _build_default(self):
        """Build the default home topology from known layout."""
        # --- Rooms ---
        self.add_node(TopoNode("office", "room", "Office",
            features=["desk", "monitor", "office chair", "wooden floor",
                       "yellow walls", "cables", "shelving"],
            floor_type="wooden floor",
            nav_hints="Exit is the doorway with bright hallway light. "
                      "Go around the chair, not through it."))

        self.add_node(TopoNode("hallway", "room", "Hallway",
            features=["stone tiles", "narrow corridor", "plant pot",
                       "yellow walls", "arched doorway"],
            floor_type="stone tiles"))

        self.add_node(TopoNode("kitchen", "room", "Kitchen",
            features=["stove", "oven", "refrigerator", "sink",
                       "counter", "cabinets"],
            floor_type="tiles"))

        self.add_node(TopoNode("living_room", "room", "Living Room",
            features=["sofa", "couch", "tv", "coffee table",
                       "parquet floor", "rug"],
            floor_type="parquet floor"))

        self.add_node(TopoNode("dining", "room", "Dining Room",
            features=["dining table", "chairs", "tableware"],
            floor_type="tiles"))

        # --- Transitions (doorways) ---
        self.add_node(TopoNode("office_hall_door", "transition",
            "Office-Hallway Door",
            visual_cues=["wooden door frame", "bright hallway light beyond",
                         "floor changes from wood to stone tiles"],
            azimuth_from={"office": -45, "hallway": 180},
            nav_hints="Look for the doorway with bright light. "
                      "Floor changes from wooden to stone tiles.",
            width_m=0.8))

        self.add_node(TopoNode("hall_kitchen_arch", "transition",
            "Hallway-Kitchen Arch",
            visual_cues=["orange arched doorway", "kitchen counter visible",
                         "tiled floor beyond"],
            azimuth_from={"hallway": 90},
            nav_hints="Turn right in the hallway. "
                      "Look for the orange arched doorway.",
            width_m=1.0))

        self.add_node(TopoNode("hall_living_arch", "transition",
            "Hallway-Living Room Arch",
            visual_cues=["arched doorway", "parquet floor beyond",
                         "couch visible through opening"],
            azimuth_from={"hallway": -90},
            nav_hints="Turn left in the hallway. "
                      "Look for the arch with wooden floor beyond.",
            width_m=1.0))

        self.add_node(TopoNode("kitchen_dining_pass", "transition",
            "Kitchen-Dining Passage",
            visual_cues=["open passage", "dining table visible"],
            azimuth_from={"kitchen": 0},
            nav_hints="Straight through the kitchen.",
            width_m=1.2))

        self.add_node(TopoNode("living_dining_pass", "transition",
            "Living-Dining Passage",
            visual_cues=["open passage", "dining table visible"],
            azimuth_from={"living_room": 90},
            nav_hints="Through the opening on the right.",
            width_m=1.2))

        self.add_node(TopoNode("bathroom", "room", "Bathroom",
            features=["toilet", "shower", "bathtub", "sink",
                       "tiles", "mirror"],
            floor_type="tiles"))

        self.add_node(TopoNode("hall_bathroom_door", "transition",
            "Hallway-Bathroom Door",
            visual_cues=["bathroom door", "tiled floor beyond",
                         "toilet visible"],
            azimuth_from={"hallway": 0},
            nav_hints="Look for the bathroom door in the hallway.",
            width_m=0.7))

        # --- Edges (room ↔ transition) ---
        self.add_edge("office", "office_hall_door")
        self.add_edge("office_hall_door", "hallway")

        self.add_edge("hallway", "hall_kitchen_arch")
        self.add_edge("hall_kitchen_arch", "kitchen")

        self.add_edge("hallway", "hall_living_arch")
        self.add_edge("hall_living_arch", "living_room")

        self.add_edge("kitchen", "kitchen_dining_pass")
        self.add_edge("kitchen_dining_pass", "dining")

        self.add_edge("living_room", "living_dining_pass")
        self.add_edge("living_dining_pass", "dining")

        self.add_edge("hallway", "hall_bathroom_door")
        self.add_edge("hall_bathroom_door", "bathroom")

        self.current_room = "office"
        self.current_confidence = 0.5
        print(f"[topo] Built default map: {len(self.rooms())} rooms, "
              f"{len(self.transitions())} transitions")
