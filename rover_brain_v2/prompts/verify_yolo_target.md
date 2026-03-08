You are verifying a YOLO detection for an indoor rover's navigation system.

The rover is navigating to: '%target%'.
YOLO detected '%det_name%' at horizontal position %det_cx% (0=left edge, 0.5=center, 1=right edge), size %det_bw% of frame width.

Look at the image and answer:
1) Is this detection the correct target or something related/useful for navigation? YOLO can misidentify indoor objects (e.g., ceiling light as "frisbee", recycle bin as "cup", cable organizer as "toilet").
2) Has the rover arrived? The target is "arrived" if it fills a significant portion of the frame (bw > 0.35) and the rover is clearly close to it.
3) If the detection is wrong, can you see the actual target elsewhere in the image? Provide its approximate position.

Consider:
- A detection of a related object near the target still counts (e.g., detecting "oven" when navigating to "kitchen" is useful).
- If bw > 0.4, the rover is very close to whatever was detected.
- Objects at the edge of frame (cx < 0.1 or cx > 0.9) may be partially visible and less reliable.

Reply JSON: {"wrong_target": bool, "arrived": bool, "reason": "brief explanation", "target_cx": float|null, "target_cy": float|null}
target_cx/target_cy are 0-1 fractions of frame position. Set to null if target not visible.
