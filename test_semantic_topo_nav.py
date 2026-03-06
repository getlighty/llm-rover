import json

import room_context
import topo_nav


def test_topo_transition_semantics_feed_leg_instruction(tmp_path, monkeypatch):
    topo_file = tmp_path / "topo_map.json"
    monkeypatch.setattr(topo_nav, "MAP_FILE", str(topo_file))

    topo = topo_nav.TopoMap()
    topo.update_transition_semantics(
        "hall_kitchen_arch",
        "hallway",
        "kitchen",
        doorway_landmarks=["orange arched doorway", "plant pot"],
        inside_features=["refrigerator", "red floor"],
        navigational_hint="choose the archway with appliances visible inside",
        confidence=0.82,
        scene_text="Orange arch with appliances visible beyond it.",
    )
    topo.save()

    reloaded = topo_nav.TopoMap()
    instruction = reloaded.leg_instruction(
        topo_nav.Leg("hallway", "hall_kitchen_arch", "kitchen"))

    assert instruction["doorway_landmarks"] == [
        "orange arched doorway",
        "plant pot",
    ]
    assert "refrigerator" in instruction["inside_features"]
    assert instruction["relationship_hint"].startswith("choose the archway")
    assert instruction["relationship_confidence"] == 0.82


def test_room_context_merges_observation_and_formats_relationships(tmp_path, monkeypatch):
    rooms_file = tmp_path / "rooms.json"
    room_graph_file = tmp_path / "room_graph.json"
    topo_file = tmp_path / "topo_map.json"

    monkeypatch.setattr(room_context, "ROOMS_FILE", str(rooms_file))
    monkeypatch.setattr(room_context, "ROOM_GRAPH_FILE", str(room_graph_file))
    monkeypatch.setattr(room_context, "TOPO_MAP_FILE", str(topo_file))

    rooms_file.write_text(json.dumps({
        "current_room": "hallway",
        "current_confidence": 0.7,
        "rooms": [
            {
                "name": "hallway",
                "positive_features": ["stone tiles", "plant pot"],
                "negative_features": [],
                "floor_type": "stone tiles",
                "connections": ["kitchen"],
                "entry_landmarks": [{"landmark": "stone-tiled floor transition"}],
                "nav_hints": "",
                "last_visited": "",
                "visit_count": 1,
                "last_scene": "",
                "last_yolo": "",
            },
            {
                "name": "kitchen",
                "positive_features": ["counter"],
                "negative_features": [],
                "floor_type": "tiles",
                "connections": ["hallway"],
                "entry_landmarks": [],
                "nav_hints": "",
                "last_visited": "",
                "visit_count": 1,
                "last_scene": "",
                "last_yolo": "",
            },
        ],
    }), encoding="utf-8")

    topo_file.write_text(json.dumps({
        "current_room": "hallway",
        "current_confidence": 0.7,
        "nodes": [
            {
                "id": "hall_kitchen_arch",
                "type": "transition",
                "label": "Hallway-Kitchen Arch",
                "semantic_views": {
                    "hallway": {
                        "to_room": "kitchen",
                        "doorway_landmarks": ["orange arched doorway", "plant pot"],
                        "inside_features": ["counter", "refrigerator"],
                        "navigational_hint": "pick the arch that shows counters inside",
                        "confidence": 0.84,
                    }
                },
            }
        ],
        "edges": [],
    }), encoding="utf-8")

    changed = room_context.merge_room_observation(
        "kitchen",
        features=["refrigerator", "red floor"],
        entry_landmarks=["orange arched doorway"],
        connected_to="hallway",
        mark_current=True,
        confidence=0.9,
    )

    assert changed is True

    data = room_context.load_rooms()
    kitchen = next(room for room in data["rooms"] if room["name"] == "kitchen")

    assert "refrigerator" in kitchen["positive_features"]
    assert "hallway" in kitchen["connections"]
    assert any(
        entry.get("landmark") == "orange arched doorway"
        for entry in kitchen["entry_landmarks"]
    )

    clues = room_context.format_room_clues(target_room="kitchen")
    assert "RELATION CLUES:" in clues
    assert "hallway -> kitchen" in clues
    assert "orange arched doorway" in clues
