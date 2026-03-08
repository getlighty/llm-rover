The rover is scanning with its gimbal to find a navigation target. Camera is ground-level (~15cm height).

Is '%cue_text%' visible in this image?

What counts as visible:
- A doorway: door frame, floor transition (wood→tile), opening in wall, light from adjacent room
- A room: distinctive furniture, appliances, or features of that room visible through an opening
- Landmarks: specific objects mentioned in the cue text (bookshelf, counter, fridge, etc.)
- Even partial visibility counts if you can confidently identify the cue

What does NOT count:
- Reflections in mirrors, windows, or shiny surfaces
- Similar-looking but different features (e.g., a closet door is not a room doorway)
- Your own rover parts (wheels, shadow, cables)

Reply JSON: {"visible": true/false, "confidence": 0.0-1.0, "position": "left"/"center"/"right"}
- confidence: 0.9+ = clearly visible, 0.5-0.8 = likely but partial, <0.5 = uncertain
- position: where in the frame the cue appears (left/center/right third)
