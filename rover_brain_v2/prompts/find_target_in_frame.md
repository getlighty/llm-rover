You are helping an indoor rover locate a target in its camera frame. The camera is mounted low (~15cm off the ground) on a pan-tilt gimbal.

Is '%target%' visible in this image? If yes, where is it?

Guidelines:
- The rover sees from floor level — furniture towers above, you see undersides of tables, chair legs, etc.
- A "room" target (kitchen, office) is visible if you can see distinctive features or a doorway leading to it.
- A "doorway" is visible if you can see the door frame, a floor transition, or an opening in a wall.
- An object is "visible" even if partially occluded, as long as you can identify it.
- If the target is a room and you're INSIDE it, it's visible (cx=0.5, cy=0.5).
- If you see the target reflected in a mirror or window, that does NOT count.

Reply JSON: {"visible": bool, "cx": float (0=left edge, 0.5=center, 1=right edge), "cy": float (0=top, 0.5=middle, 1=bottom)}
If not visible, set cx and cy to 0.5 (they will be ignored).
