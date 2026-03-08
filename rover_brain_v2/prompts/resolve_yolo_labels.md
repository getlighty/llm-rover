You are helping an indoor rover's navigation system map a target name to YOLO object detection classes.

The rover needs to navigate to: '%target%'.
Available YOLO detection classes: %available%

Which classes should the rover look for to find or get close to this target?
Think about:
- Direct matches: "kitchen" → look for "oven", "refrigerator", "sink", "microwave"
- Indirect cues: "bathroom" → look for "toilet", "sink"; "office" → look for "keyboard", "monitor"
- Doorway navigation: if the target is a room, look for objects visible NEAR or THROUGH its doorway
- Include the target itself if it's a class name (e.g., "person" → ["person"])
- Include related objects that would help identify the area (max 5 labels)

Reply JSON: {"labels": ["class1", "class2"]}
Only include classes from the available list. Prefer specific, reliable detections over vague ones.
