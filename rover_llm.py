#!/usr/bin/env python3
"""
LLM-controlled Waveshare UGV Rover PT via Ollama cloud models.

Sends natural language commands to an Ollama cloud model, which returns
JSON commands that are forwarded to the ESP32 sub-controller over serial.
"""

import serial
import json
import time
import sys
import threading
import queue
import requests
import readline
import os

# --- Configuration ---
SERIAL_PORT = os.environ.get("ROVER_SERIAL_PORT", "/dev/ttyTHS1")
BAUD_RATE = 115200
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:cloud")
HEARTBEAT_INTERVAL = 2.0  # seconds between heartbeat commands during motion

SYSTEM_PROMPT = """\
You are the brain of a Waveshare UGV Rover PT, a 6-wheel rover robot with a 2-axis pan-tilt camera gimbal. You receive natural language instructions and translate them into one or more JSON commands to control the robot.

## Your capabilities
- Drive the rover (forward, backward, turn, spin, curve, stop)
- Control the pan-tilt camera gimbal (look left/right/up/down, center)
- Control LED lights (base lights, head lights)
- Display text on the onboard OLED screen
- Read sensor feedback (IMU, voltage, wheel speeds)

## JSON Command Reference

### Motion (wheels)
Closed-loop speed control (m/s, max ~1.3):
{"T":1, "L":<left_speed>, "R":<right_speed>}
- Forward: L and R both positive and equal
- Backward: L and R both negative and equal
- Spin left in place: L negative, R positive
- Spin right in place: L positive, R negative
- Curve: different magnitudes for L and R
- Stop: L=0, R=0
- Typical slow speed: 0.2, medium: 0.5, fast: 1.0

ROS-style velocity (linear X m/s, angular Z rad/s):
{"T":13, "X":<linear>, "Z":<angular>}

### Pan-Tilt Gimbal
Absolute position (X: pan -180..180 deg, Y: tilt -30..90 deg):
{"T":133, "X":<pan>, "Y":<tilt>, "SPD":<speed>, "ACC":<accel>}
SPD=0 means maximum speed. Typical: SPD=200, ACC=10

Directional continuous move (X/Y: -1, 0, or 1):
{"T":141, "X":<dir>, "Y":<dir>, "SPD":300}
X=2, Y=2 means return to center.

Stop gimbal:
{"T":135}

### Lights
{"T":132, "IO4":<base_0-255>, "IO5":<head_0-255>}

### OLED Display (4 lines, 0-3)
{"T":3, "lineNum":<0-3>, "Text":"<message>"}
Restore default display: {"T":-3}

### Feedback request
{"T":130}
Continuous feedback on: {"T":131, "cmd":1}
Continuous feedback off: {"T":131, "cmd":0}

### Emergency stop (all motors)
{"T":0}

## Response Format
You MUST respond with a JSON object containing:
- "commands": array of JSON command objects to send to the rover, in order
- "speak": a short natural language description of what you're doing (1 sentence)
- "duration": optional float, seconds to maintain the last motion command before auto-stopping (default: 0, meaning no auto-stop). Use this for timed movements like "drive forward for 3 seconds".

Example response for "drive forward slowly for 2 seconds then stop":
```json
{"commands": [{"T":1, "L":0.2, "R":0.2}], "speak": "Driving forward slowly for 2 seconds.", "duration": 2.0}
```

Example response for "look left and turn on the headlights":
```json
{"commands": [{"T":133, "X":-90, "Y":0, "SPD":200, "ACC":10}, {"T":132, "IO4":0, "IO5":255}], "speak": "Looking left and turning on headlights."}
```

Example response for "spin around":
```json
{"commands": [{"T":1, "L":0.5, "R":-0.5}], "speak": "Spinning clockwise.", "duration": 3.0}
```

## Rules
- Always respond with valid JSON only. No markdown, no explanation outside the JSON.
- When the user says "stop" or "halt", immediately send {"T":1, "L":0, "R":0} and {"T":135}.
- For safety, never exceed speed 1.0 m/s unless explicitly told to go fast/max speed.
- If the user's intent is unclear, ask for clarification in the "speak" field and send no commands.
- When doing timed movements, set "duration" so the system auto-stops after that time.
- /no_think
"""


class RoverSerial:
    """Handles serial communication with the ESP32 sub-controller."""

    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.cmd_queue = queue.Queue()
        self._running = False
        self._writer_thread = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self._running = True
            self._writer_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._writer_thread.start()
            print(f"[serial] Connected to {self.port} @ {self.baud}")
            return True
        except serial.SerialException as e:
            print(f"[serial] Failed to connect to {self.port}: {e}")
            print(f"[serial] Running in DRY-RUN mode (commands will be printed but not sent)")
            return False

    def send(self, cmd_dict):
        """Queue a JSON command for sending."""
        # Swap forward/backward by negating wheel speeds
        if isinstance(cmd_dict, dict) and cmd_dict.get("T") == 1:
            cmd_dict = dict(cmd_dict, L=-cmd_dict.get("L", 0), R=-cmd_dict.get("R", 0))
        self.cmd_queue.put(cmd_dict)

    def _process_queue(self):
        while self._running:
            try:
                cmd = self.cmd_queue.get(timeout=0.5)
                payload = json.dumps(cmd) + "\n"
                self.ser.write(payload.encode("utf-8"))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[serial] Write error: {e}")

    def read_feedback(self):
        """Try to read a JSON line from the ESP32."""
        if not self.ser or not self.ser.in_waiting:
            return None
        try:
            line = self.ser.readline().decode("utf-8").strip()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return None

    def close(self):
        self._running = False
        if self.ser:
            self.ser.close()


class OllamaClient:
    """Calls the local Ollama API to reach cloud-hosted models."""

    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.history = []

    def chat(self, user_message):
        """Send a message and get the assistant response."""
        self.history.append({"role": "user", "content": user_message})

        # Keep history manageable (last 20 exchanges)
        if len(self.history) > 40:
            self.history = self.history[-40:]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.history,
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            assistant_msg = data["message"]["content"]
            self.history.append({"role": "assistant", "content": assistant_msg})
            return assistant_msg
        except requests.RequestException as e:
            return json.dumps({
                "commands": [],
                "speak": f"Error communicating with Ollama: {e}",
            })


def parse_llm_response(raw_text):
    """Extract the JSON object from the LLM response, handling markdown fences."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def main():
    print("=" * 60)
    print("  UGV Rover PT - LLM Control Interface")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Serial: {SERIAL_PORT} @ {BAUD_RATE}")
    print("=" * 60)

    # Connect serial
    rover = RoverSerial(SERIAL_PORT, BAUD_RATE)
    serial_ok = rover.connect()

    # Init Ollama client
    llm = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL)

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            print(f"[warn] Model '{OLLAMA_MODEL}' not found. Available: {models}")
        else:
            print(f"[ollama] Model '{OLLAMA_MODEL}' ready")
    except Exception as e:
        print(f"[ollama] Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        sys.exit(1)

    # Auto-stop timer state
    stop_timer = None

    def auto_stop():
        """Send stop command after timed movement."""
        rover.send({"T": 1, "L": 0, "R": 0})
        if serial_ok:
            print("\n[auto-stop] Motion stopped")
        else:
            print(f"\n[dry-run] Would send: {json.dumps({'T':1,'L':0,'R':0})}")
            print("[auto-stop] Motion stopped")

    print("\nType commands in natural language. Type 'quit' to exit.")
    print("Type 'status' to request rover feedback.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            # Emergency stop before exit
            rover.send({"T": 1, "L": 0, "R": 0})
            rover.send({"T": 135})
            break

        if user_input.lower() == "status":
            rover.send({"T": 130})
            time.sleep(0.5)
            fb = rover.read_feedback()
            if fb:
                print(f"[feedback] {json.dumps(fb, indent=2)}")
            else:
                print("[feedback] No data received")
            continue

        # Send to LLM
        print("[thinking...]", end="", flush=True)
        raw_response = llm.chat(user_input)
        print("\r" + " " * 20 + "\r", end="")

        parsed = parse_llm_response(raw_response)
        if not parsed:
            print(f"[llm] Could not parse response:\n{raw_response}")
            continue

        speak = parsed.get("speak", "")
        commands = parsed.get("commands", [])
        duration = parsed.get("duration", 0)

        if speak:
            print(f"[rover] {speak}")

        if not commands:
            continue

        # Cancel any pending auto-stop
        if stop_timer and stop_timer.is_alive():
            stop_timer.cancel()

        # Execute commands
        for cmd in commands:
            if serial_ok:
                rover.send(cmd)
                print(f"  -> {json.dumps(cmd)}")
            else:
                print(f"  [dry-run] {json.dumps(cmd)}")

        # Set auto-stop timer if duration specified
        if duration and duration > 0:
            stop_timer = threading.Timer(duration, auto_stop)
            stop_timer.start()
            print(f"  [timer] Auto-stop in {duration}s")

    # Cleanup
    if stop_timer and stop_timer.is_alive():
        stop_timer.cancel()
    rover.send({"T": 1, "L": 0, "R": 0})
    rover.send({"T": 135})
    time.sleep(0.3)
    rover.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
