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


def test_room_map_narrative_memory(tmp_path, monkeypatch):
    """Test the new narrative room map."""
    import room_map
    map_file = tmp_path / "room_map.json"
    monkeypatch.setattr(room_map, "MAP_FILE", str(map_file))

    rm = room_map.RoomMap()

    # Record narrative observation
    rm.record_narrative("Desk with monitor ahead, chair to the right",
                        room="office", objects=["desk", "monitor", "chair"])

    # Legacy record interface (YOLO detections)
    rm.record([{"name": "cup"}, {"name": "keyboard"}], 0, 0, 0, 0, 0)

    assert rm.object_count() == 5  # desk, monitor, chair, cup, keyboard

    # room_json returns narrative data
    rj = rm.room_json()
    assert rj is not None
    assert "recent_observations" in rj
    assert "known_objects" in rj
    assert any(o["name"] == "desk" for o in rj["known_objects"])

    # Find works
    name, info = rm.find("monitor")
    assert name == "monitor"
    assert info["room"] == "office"

    # nav_json works
    nav = rm.nav_json("desk")
    assert nav is not None
    assert nav["room"] == "office"

    # Save and reload
    rm.save()
    rm2 = room_map.RoomMap()
    assert rm2.object_count() == 5

    # describe_room returns text
    desc = rm2.describe_room()
    assert "Room Map" in desc
    assert "desk" in desc.lower()


def test_exploration_grid_stub():
    """Test that the exploration grid stub doesn't crash."""
    from exploration_grid import ExplorationGrid
    grid = ExplorationGrid()

    # These should be no-ops
    grid.update_after_drive(1.0, 90.0)
    grid.update_after_turn(45.0)
    grid.update_from_depth(None, 0, 0)

    # summarize_for_llm returns empty for fresh grid
    assert grid.summarize_for_llm() == ""

    # Add a note and get a summary
    grid.add_note("Checked left side, wall visible", room="office")
    summary = grid.summarize_for_llm()
    assert "Checked left side" in summary
    assert "office" in summary
