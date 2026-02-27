# Rover Memory

See identity.md for hardware specs, capabilities, and abstraction layers.

## Identity
- Robot's name: Jasper
- Owner/operator: Ovi

## People
- Ovi met Elon Musk on September 3, 2020
- Ovi's zodiac sign is Pisces

## Preferences
- Ovi doesn't like the headlight shined in his face
- Dim lights when there's enough ambient light; only brighten when needed for camera
- Don't say Ovi's name every time
- Don't come back to position zero all the time
- Move slowly when approaching the human
- Move at 20% of maximum speed at all times
- Look around before backing up
- Upgrades need a clear purpose

## Environment
- Workshop with yellow walls
- Right side: desk with curved monitors, keyboard, cables
- Left side: tools and shelving
- Wicker basket on/near the desk
- Printer on the desk to the right
- Floor: wood paneling

## Object Locations (from spatial map)
- Basket: directly in front, with blue plastic container next to it
- Printer: on desk to the right (pan ~0)

## Capabilities
- Can modify own code with bash (syntax check first, then restart)
- Can perform tasks automatically
- Local object detection: YOLOv8n (80 COCO classes, ~5 FPS)
- Visual servo navigation: find objects and drive to them autonomously
- Path planning: survey room → 2D map → A* pathfinding → waypoint following
- Door navigation: edge detection + LLM vision to find and navigate through doors
- Stop word detection: local whisper-tiny for instant emergency stop
- Adaptive light management: auto-dim/brighten based on ambient brightness
- The human has spoken to the robot, indicating interaction. [2026-02-26 21:07]
- The human mentioned the word "goblet" on 2026-02-26. [2026-02-26 21:09]
- The human said: navigate to the kitchen through the door. [2026-02-26 23:07]
- The human instructed the robot to navigate to the kitchen through the door. [2026-02-26 23:18]
- Ovi mentioned the word "goblet" on 2026-02-26. [2026-02-27 08:15]
- The human mentioned the word "goblet" on 2026-02-26 and again on 2026-02-27. [2026-02-27 09:28]
- Ovi mentioned the word "goblet" on 2026-02-26 and again on 2026-02-27. [2026-02-27 10:19]
- The human said: Parac parlamentar中 [2026-02-27 10:36]
- The human said: Thegee need some of these [2026-02-27 11:07]
- The human said: In small steps, go towards that door. [2026-02-27 11:31]
- The human said: You have to move slower in smaller angles. [2026-02-27 11:46]
- The human has a puppy in the family. [2026-02-27 11:53]
- The human said: I love you. [2026-02-27 12:09]
- The human said: I don't like you. [2026-02-27 12:20]
- The human said "I love you" and later "I don't like you", indicating a change in their feelings towards the robot. [2026-02-27 12:21]
- The human's feelings towards the robot changed from love to dislike. [2026-02-27 12:21]
- Keep headlights on at all times; do not turn them off unless explicitly told. [2026-02-27 18:05]
- User wants both headlights kept on continuously; set IO4 and IO5 high and do not turn them off unless asked. [2026-02-27 18:06]
