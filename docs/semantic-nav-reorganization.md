# Semantic Navigation Reorganization

## Goal

Shift inter-room navigation away from primarily fixed cues and shortest-path routing, and toward relationship-based reasoning that is closer to how a human navigates:

- identify a target room by what is near its doorway
- identify it by what is visible just inside
- inspect a doorway before entering
- reorganize learned knowledge after each successful reach

Low-level obstacle avoidance and local safe motion remain unchanged. This work adds a semantic layer on top of the existing navigation stack.

## Previous Behavior

Before this change:

- `rover_brain_llm.py` used `_topo_navigate()` for room-to-room routing
- `topo_nav.py` stored a static room/transition graph
- `room_context.py` stored room features and room-to-room connections
- `navigator.py` could prompt the LLM to look through a doorway during `navigate_leg()`

The limitation was that doorway reasoning was mostly transient. The rover did not persist a relationship like:

- "from hallway, the correct kitchen doorway is the orange arch near the plant pot"
- "through that doorway I should see counters, refrigerator, red floor"
- "next time, choose the opening that shows kitchen appliances inside"

## New Behavior

After a successful room reach, the rover now runs a semantic reorganization pass that:

1. merges newly observed room features into room memory
2. stores entry landmarks for the room it just entered
3. links the source room to the destination room if needed
4. updates the transition with source-side doorway landmarks
5. updates the transition with "inside" features visible through or just beyond the doorway
6. stores a short relationship-based navigation hint for future room selection

During future room-to-room legs, the rover now uses those learned relationship cues in the prompt, not just the original static transition description.

## Data Model Changes

### `topo_map.json`

Transition nodes can now store:

- `semantic_views`
- `observation_count`

`semantic_views` is keyed by source room. Each view can contain:

- `to_room`
- `doorway_landmarks`
- `inside_features`
- `inside_room_guess`
- `navigational_hint`
- `confidence`
- `last_scene`
- `last_observed`
- `observation_count`

This is the main new persistent structure for relationship-based navigation.

### `rooms.json`

Room memory is now updated after successful arrivals via a merge path that can persist:

- new `positive_features`
- `entry_landmarks`
- `connections`
- `last_scene`
- `last_yolo`
- `current_room`
- `current_confidence`

## Runtime Flow

### 1. Topological leg planning

`_topo_navigate()` still plans room-to-room legs using `TopoMap.plan_route()`.

### 2. Relationship-aware leg instructions

`TopoMap.leg_instruction()` now injects learned semantic transition data into each leg when available:

- `doorway_landmarks`
- `inside_features`
- `relationship_hint`
- `relationship_confidence`

### 3. Doorway peek before crossing

`Navigator.navigate_leg()` now does more than treat a doorway as a geometric target.

When the target doorway becomes visible:

- it asks the LLM to inspect what is near the doorway
- it asks what seems to be inside or beyond the opening
- it stores that transient result in `_last_transition_peek`
- it uses that result to reinforce the same leg prompt

This makes doorway selection more semantic and less purely positional.

### 4. Semantic reorganization after success

After a successful room reach, `rover_brain_llm.py` now calls `_semantic_reorganize_after_reach()`.

That function:

- collects the final room scene
- collects the last YOLO summary
- collects the last doorway peek
- collects room-map context if available
- asks the LLM to compress the result into relationship-oriented memory
- writes the merged room data into `rooms.json`
- writes the merged transition semantics into `topo_map.json`

This runs on:

- normal successful topological leg completion
- successful reactive fallback to a destination room
- successful fallback when the current room could not be identified or no topo route existed

## Files Changed

### `topo_nav.py`

Added support for transition semantic memory:

- `TopoNode.semantic_views`
- `TopoNode.observation_count`
- `TopoMap.ensure_room()`
- `TopoMap.ensure_transition_between()`
- `TopoMap.update_room_semantics()`
- `TopoMap.update_transition_semantics()`
- relationship-aware fields in `TopoMap.leg_instruction()`

### `room_context.py`

Added merge and prompt helpers for semantic room/doorway knowledge:

- room/landmark merge helpers
- topo relationship loading
- `format_room_clues(target_room=...)`
- `format_relationship_clues(...)`
- relationship sections in prompt formatting
- `link_rooms()`
- `merge_room_observation()`

### `navigator.py`

Extended room-leg behavior:

- target-specific relationship clues in waypoint prompts
- `_last_transition_peek`
- `_peek_transition_view()`
- relationship-oriented prompt fields inside `navigate_leg()`
- periodic doorway peek when a doorway candidate is visible

### `rover_brain_llm.py`

Added the high-level semantic reorganization hook:

- `_canonical_room_id()`
- `_semantic_text_list()`
- `_semantic_reorganize_after_reach()`
- success-path calls from `_topo_navigate()`

### `test_semantic_topo_nav.py`

Added focused checks for:

- transition semantic persistence into leg instructions
- merged room observation persistence
- relationship-clue formatting

## Validation

Validated in this environment with:

- `python3 -m py_compile` on the edited files
- direct Python smoke tests with assertions covering the new semantic persistence path

`pytest` was not available in the environment, so the new test file was not executed through pytest here.

## Important Design Notes

- This does not remove low-level local planning or obstacle avoidance.
- The rover still needs distance/depth for safe motion.
- The change is about how it chooses which doorway or room to pursue at the semantic level.
- The new memory is source-room specific. The same doorway can mean different things depending on which side the rover is on.

## Known Limitations

- The semantic reorganization still depends on LLM output quality.
- The rover learns after successful reaches; failed doorway inspections are not yet used to explicitly suppress wrong doorway choices.
- The relationship memory is stored in prompt-friendly text, not a stricter ontology.

## Recommended Next Improvements

1. Reject doorway candidates during `navigate_leg()` when the peek strongly contradicts the target room.
2. Track negative transition evidence, for example "arched doorway with couch beyond is not kitchen."
3. Weight transition choice by semantic confidence, recency, and repeated success count.
4. Expose learned transition semantics in the web UI for inspection and manual cleanup.
