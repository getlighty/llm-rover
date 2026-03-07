# rover_brain_v2

`rover_brain_v2` is a new modular runtime that sits alongside the existing `rover_brain_llm.py` stack without modifying it.

It currently includes:

- A new provider registry with independent `command_llm`, `navigator_llm`, `orchestrator_llm`, `stt`, and `tts` roles.
- A new navigation stack that uses:
  - a navigator LLM for local waypoint decisions
  - an orchestrator LLM for room verification and semantic memory updates
  - DepthAnything vector maps for local motion guidance
  - the existing `topo_nav.py` / `room_context.py` graph memory and room data
- A new follow-me controller that keeps the current YOLO-style visual servo behavior, but adds continuous DepthAnything crash guarding.
- A separate web UI with direct teleop, quick room navigation, follow controls, provider selection, runtime flags, and live logs.

What this package intentionally reuses from the existing repo:

- `local_detector.py` for YOLO detections and DepthAnything inference
- `imu.py` for IMU polling and heading/tilt telemetry
- `room_context.py` and `topo_nav.py` for room memory and graph routing
- `audio.py` for mic/speaker device discovery and audio capture/playback

Run it with:

```bash
cd /home/jasper/rover-control
python3 -m rover_brain_v2
```

Optional flags:

```bash
python3 -m rover_brain_v2 --web-port 8765 --camera 0 --serial-port /dev/ttyTHS0
```
