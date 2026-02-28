## scout

I'm a ground rover searching for: {target}.
Look at this image and tell me what you see.

Reply ONLY with JSON:
{{"found": true/false, "objects": ["list","all","visible","objects"], "hint": "left"/"right"/"behind"/"unknown"}}

- "found": true if {target} is visible in this image
- "objects": list ALL notable objects you can identify (furniture, items, people, etc.)
- "hint": if {target} is NOT found, your best guess which direction to look based on context clues
  - "left" = try looking further left
  - "right" = try looking further right
  - "behind" = might be behind me
  - "unknown" = no clues

## check

I'm searching for: {target}.
Camera at pan={pan}°, tilt={tilt}°, world_pan={world_pan}°.
Look carefully at this image. Is {target} visible?

Reply ONLY with JSON:
{{"found": true/false, "objects": ["list","all","visible","objects"]}}

List ALL notable objects you can identify, not just the target.

## center

You are looking at '{target}'. Your head is at pan={pan}°, tilt={tilt}°.
Is '{target}' centered in this image?

If yes: {{"centered": true}}
If no, adjust gimbal to center it: {{"centered": false, "commands": [{{"T":133,"X":<pan>,"Y":<tilt>,"SPD":100,"ACC":10}}]}}

Reply with ONLY the JSON.
