## steer

I'm a ground rover driving toward '{target}' right now.
Camera pan={pan}°.

Reply ONLY with JSON:
{{"v": true/false, "close": true/false, "dir": "left"/"right"/"center", "size": "small"/"medium"/"large"}}

- "v": true if {target} is visible in this image
- "close": true if {target} is within about 0.5 meters (very close, filling most of the frame)
- "dir": which direction to steer to keep {target} centered
- "size": how large {target} appears ("small"=far, "medium"=few meters, "large"=very close)

## assess

I'm a ground rover. I need to drive toward '{target}'.
Look at this image. Can you see {target}?

Reply ONLY with JSON:
{{"visible": true/false, "distance": "far"/"medium"/"close", "direction": "left"/"center"/"right"}}
