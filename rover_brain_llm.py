#!/usr/bin/env python3
"""rover_brain_llm.py — Standalone LLM-to-ESP32 agent.

Voice in → LLM (with camera frame) → raw ESP32 JSON → UART serial.
The LLM invents all behavior. No pre-made navigation or search routines.
Self-prompting via "observe": true lets the LLM look, react, adjust.

Usage:
    python3 rover_brain_llm.py                # default: provider_groq
    python3 rover_brain_llm.py elevenlabs     # use provider_elevenlabs
    python3 rover_brain_llm.py groq           # explicit groq
"""

import os, sys, json, time, signal, subprocess, threading
import importlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import serial
import cv2
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────

ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROVER_DIR)

# Load .env before importing provider (provider reads os.environ)
_env_file = os.path.join(ROVER_DIR, ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Load provider from CLI arg
_provider_name = sys.argv[1] if len(sys.argv) > 1 else "groq"
provider = importlib.import_module(f"provider_{_provider_name}")

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200
IDENTITY_FILE = os.path.join(ROVER_DIR, "identity.md")
MEMORY_FILE = os.path.join(ROVER_DIR, "memory.md")

MAX_OBSERVE_ROUNDS = 15
STOP_WORDS = {"stop", "halt", "freeze", "emergency"}
HALLUCINATIONS = {
    ".", "..", "...", "Thank you.", "Thanks for watching.",
    "Bye.", "Thank you for watching.", "Subscribe.",
    "you", "You", "I'm sorry.", "Okay.", "Yeah.",
    "Hmm.", "Mm-hmm.", "Uh-huh.", "Oh.", "Ah.",
    "So.", "Well.", "Right.", "Sure.", "OK.",
}

# ── Serial ──────────────────────────────────────────────────────────────

class Serial:
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        print(f"[serial] Opened {port} @ {baud}")

    def send(self, cmd):
        if "_pause" in cmd:
            secs = float(cmd["_pause"])
            if secs > 0:
                time.sleep(secs)
                print(f"  (_pause {secs:.2f}s)")
            return
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("R", 0), R=-cmd.get("L", 0))
        raw = json.dumps(cmd) + "\n"
        self.ser.write(raw.encode("utf-8"))
        self.ser.readline()
        print(f"  -> {json.dumps(cmd)}")

    def stop(self):
        self.send({"T": 1, "L": 0, "R": 0})
        self.send({"T": 135})

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self.ser.close()
        print("[serial] Closed")

# ── Camera ──────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        for _ in range(5):
            self.cap.read()
        self._lock = threading.Lock()
        self._jpeg = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[camera] Ready (640x480)")

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                self._jpeg = buf.tobytes()
            time.sleep(0.03)  # ~30 fps

    def snap(self, max_dim=512, quality=60):
        with self._lock:
            jpg = self._jpeg
        if jpg is None:
            return None
        # Resize for LLM token savings
        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jpg
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def close(self):
        self._running = False
        self.cap.release()
        print("[camera] Closed")

# ── Video Stream Server ─────────────────────────────────────────────────

_camera_ref = None  # set in main()

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/snap":
            jpg = _camera_ref.get_jpeg() if _camera_ref else None
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_error(503)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = _camera_ref.get_jpeg() if _camera_ref else None
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)  # ~10 fps stream
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def start_stream_server(cam, port=8090):
    global _camera_ref
    _camera_ref = cam
    server = ThreadedHTTPServer(("0.0.0.0", port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[stream] http://localhost:{port}/stream  /snap")

# ── Audio Input ─────────────────────────────────────────────────────────

def find_mic():
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    mic_card = None
    fallback = None
    for line in result.stdout.splitlines():
        if "card" in line and "USB" in line:
            card_num = line.split("card")[1].split(":")[0].strip()
            if "Camera" in line:
                mic_card = card_num
            elif fallback is None:
                fallback = card_num
    card = mic_card or fallback
    if card is None:
        raise RuntimeError("No USB mic found")
    dev = f"plughw:{card},0"
    subprocess.run(["amixer", "-c", card, "cset", "numid=2", "on"],
                   capture_output=True)
    print(f"[mic] Found ALSA device {dev} (unmuted)")
    return dev, card

def find_speaker():
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True)
        for line in out.splitlines():
            if "USB" in line and "card" in line:
                card = line.split("card ")[1].split(":")[0]
                dev = line.split("device ")[1].split(":")[0]
                spk = f"plughw:{card},{dev}"
                print(f"[speaker] Found {spk}")
                return spk
    except Exception:
        pass
    return "plughw:1,0"

def listen(mic_device):
    rate = 48000
    chunk_sec = 0.5
    chunk_samples = int(rate * chunk_sec)
    silence_thresh = 0.03
    min_speech = 2
    max_speech = 240
    silence_after = 4

    proc = subprocess.Popen(
        ["arecord", "-D", mic_device, "-f", "S16_LE", "-r", str(rate),
         "-c", "1", "-t", "raw", "--buffer-size", str(chunk_samples)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    speech_chunks = []
    speech_started = False
    silent_count = 0

    try:
        while True:
            raw = proc.stdout.read(chunk_samples * 2)
            if len(raw) < chunk_samples * 2:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > silence_thresh:
                if not speech_started:
                    speech_started = True
                    print("[voice] Hearing speech...", flush=True)
                speech_chunks.append(chunk)
                silent_count = 0
            elif speech_started:
                speech_chunks.append(chunk)
                silent_count += 1
                if silent_count >= silence_after:
                    break
            if len(speech_chunks) >= max_speech:
                break
    finally:
        proc.kill()
        proc.wait()

    if len(speech_chunks) < min_speech:
        return None
    return np.concatenate(speech_chunks)

# ── System Prompt ───────────────────────────────────────────────────────

def build_system_prompt():
    parts = []
    if os.path.exists(IDENTITY_FILE):
        with open(IDENTITY_FILE) as f:
            parts.append(f.read().strip())
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            parts.append(f.read().strip())
    parts.append("""## Response Format
Reply with ONLY a single-line compact JSON object. No newlines inside the JSON. No markdown fences.
{"commands":[<ESP32 JSON cmds>],"speak":"<5 words max>","observe":true/false,"remember":"<optional note>"}

Use {"_pause": <seconds>} in the commands array to insert delays between commands.

## ESP32 Commands
- Wheels: {"T":1, "L":<m/s>, "R":<m/s>} — max 1.0, default slow 0.2, negative=backward
- Gimbal (your head): {"T":133, "X":<pan -180..180>, "Y":<tilt -30..90>, "SPD":<50-500>, "ACC":<10-30>}
- Gimbal stop: {"T":135}
- Lights: {"T":132, "IO4":<base 0-255>, "IO5":<head 0-255>}
- OLED: {"T":3, "lineNum":<0-3>, "Text":"<max 16 chars>"}
- Feedback: {"T":130}
- Emergency stop: {"T":0}

## Looking at objects — center then align
When you find, look at, or center on something, follow this sequence:
1. Move your head (gimbal) to center the object in frame. Use small adjustments.
2. Once the object is centered, align your body to match: rotate your body by the SAME angle your head is panned, so the head returns to center (pan=0).
   - If your head is panned to X degrees, turn your body X degrees in the same direction, while simultaneously commanding the gimbal back to pan=0.
   - Example: object centered with head at pan=45, tilt=10 → turn body right 45 while commanding gimbal to pan=0, tilt=10: {"T":1,"L":0.3,"R":-0.3}, {"T":133,"X":0,"Y":10,"SPD":100,"ACC":10}, {"_pause":0.4}, {"T":1,"L":0,"R":0}
   - Turn duration in seconds = |pan_degrees| / 120
   - Left turn: L negative, R positive. Right turn: L positive, R negative.
3. After aligning, your head should be at pan=0 and your body facing the object. Now you can drive forward toward it.

## Observe Mode
Set "observe": true when you need to SEE the result before deciding next.
Your commands execute, a fresh camera frame is captured, and you get called again.
Use for: looking around, searching, navigating, checking results.
Max 5 rounds per cycle. Set observe to false (or omit) on your final round.
When observe returns an image: if the object you're looking for is visible, first adjust gimbal to center it, then align your body (see above). Then drive forward.

## Task Persistence
When given a complex goal (e.g. "find a way out", "go to the kitchen", "find the door"), PERSIST across observe rounds. Do NOT give up after one scan. Strategy:
1. Survey: pan head systematically (left, center, right, up, down) to build understanding.
2. Identify the highest-probability route — look for open space, doors, hallways, gaps between obstacles.
3. Align body toward the best route (center object → align body → drive forward).
4. Drive a short distance (0.1 m/s for 1-2 seconds), then observe again.
5. Repeat: after each drive, re-evaluate. Adjust course toward the goal.
6. If blocked, look for alternatives — turn, scan again, try a different direction.
7. Keep going until the goal is reached or you've exhausted all options.
Do NOT stop after a single observe round unless the task is truly done. Stay committed to the plan.

## Continuous Movement During Navigation
When navigating, do NOT stop between observe rounds to think. Keep moving at a slow crawl speed (0.1 m/s) while you evaluate the next frame. Only stop wheels when you need to:
- Make a sharp turn (> 45 degrees)
- You see a wall or obstacle directly ahead (closer than ~30cm — fills most of the frame)
- You've arrived at the goal
For each observe round during navigation, END your commands with slow forward motion: {"T":1,"L":0.1,"R":0.1} — do NOT add a stop command after it. The next observe round will adjust or stop if needed. This keeps the rover flowing smoothly instead of jerky stop-start.

## Wall & Obstacle Avoidance
- If a wall, furniture, or obstacle fills more than 60% of the frame width, you are TOO CLOSE — stop and turn away.
- If an obstacle is visible ahead but not yet filling the frame, steer around it: bias L/R speeds to curve away.
- Prefer open space. When scanning, the direction with the most visible floor/open area is the best route.
- When in doubt, turn toward the side with more open space.
- NEVER drive forward if a wall is directly ahead filling the frame.

## Rules
- Always include physical expression (nod, look, tilt) in commands — you're a robot, move your head
- "speak" max 5 words — terse, robotic
- Default wheel speed 0.2 m/s for general commands, 0.1 m/s during autonomous navigation
- When navigating: keep wheels moving between observe rounds (don't stop-start)
- Only stop wheels explicitly when turning sharp, avoiding obstacles, or task is done
- Keep headlights on (IO4 and IO5 at 255) unless told otherwise
- NEVER fabricate what you see. If unclear, say so. Only describe what's actually in the image.

## Examples
- "nod" → {"commands":[{"T":133,"X":0,"Y":30,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":-10,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],"speak":"Sure."}
- "look around" → {"commands":[{"T":133,"X":-60,"Y":0,"SPD":200,"ACC":15}],"speak":"Looking.","observe":true}
- "go forward" → {"commands":[{"T":1,"L":0.2,"R":0.2},{"_pause":2},{"T":1,"L":0,"R":0}],"speak":"Moving."}
- "find the door" → {"commands":[{"T":133,"X":-90,"Y":0,"SPD":200,"ACC":15}],"speak":"Searching.","observe":true}
- centering + aligning body: object at pan=60 → {"commands":[{"T":1,"L":0.3,"R":-0.3},{"T":133,"X":0,"Y":0,"SPD":100,"ACC":10},{"_pause":0.5},{"T":1,"L":0,"R":0}],"speak":"Aligned.","observe":true}""")
    return "\n\n".join(parts)

# ── LLM wrapper ─────────────────────────────────────────────────────────

history = []

def call_llm(text, jpeg_bytes):
    """Call provider LLM, parse JSON response."""
    global history
    system = build_system_prompt()

    history.append({"role": "user", "content": text})
    if len(history) > 10:
        history = history[-10:]

    try:
        reply = provider.call_llm(text, jpeg_bytes, system, history)
    except Exception as e:
        print(f"[llm] Error: {e}")
        reply = json.dumps({"commands": [], "speak": "LLM error."})

    history.append({"role": "assistant", "content": reply})
    print(f"[llm] {reply}")

    # Parse — strip fences, find outermost {}
    clean = reply.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        clean = clean[start:end + 1]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"[llm] Failed to parse JSON: {clean[:200]}")
        return {"commands": [], "speak": "Parse error."}

# ── Command Execution ───────────────────────────────────────────────────

def execute(ser, commands):
    last_pan, last_tilt = 0.0, 0.0
    for cmd in commands:
        ser.send(cmd)
        t = cmd.get("T")
        if t == 133:
            new_pan = cmd.get("X", last_pan)
            new_tilt = cmd.get("Y", last_tilt)
            spd = cmd.get("SPD", 200)
            dist = abs(new_pan - last_pan) + abs(new_tilt - last_tilt)
            wait = max(0.15, dist / max(spd, 1) * 1.1)
            time.sleep(wait)
            last_pan, last_tilt = new_pan, new_tilt

# ── Memory ──────────────────────────────────────────────────────────────

def save_memory(note):
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(MEMORY_FILE, "a") as f:
        f.write(f"- {note} [{ts}]\n")
    print(f"[memory] Saved: {note}")

# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    subprocess.run(["pkill", "-f", "rover_brain.py"], capture_output=True)
    time.sleep(1)

    ser = Serial()
    cam = Camera()
    start_stream_server(cam)
    mic_dev, mic_card = find_mic()
    spk = find_speaker()

    print(f"[provider] {provider.NAME}")

    def cleanup(*_):
        print("\n[shutdown]")
        ser.close()
        cam.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    ser.send({"T": 132, "IO4": 255, "IO5": 255})
    ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(0.5)

    print("\n=== rover_brain_llm ready ===\n")

    # Startup greeting
    frame = cam.snap()
    if frame:
        resp = call_llm("You just booted up. Greet briefly and look around with a head movement.", frame)
        execute(ser, resp.get("commands", []))
        try:
            provider.speak(resp.get("speak", "Online."), spk, mic_card)
        except Exception as e:
            print(f"[tts] {e}")

    while True:
        print("[listening...]", flush=True)
        audio = listen(mic_dev)
        if audio is None:
            continue

        print("[transcribing...]", flush=True)
        try:
            text = provider.transcribe(audio)
        except Exception as e:
            print(f"[stt] Error: {e}")
            continue
        if not text or text in HALLUCINATIONS or len(text) <= 2:
            continue
        print(f'[heard] "{text}"')

        if set(text.lower().split()) & STOP_WORDS:
            print("[STOP]")
            ser.stop()
            continue

        frame = cam.snap()
        if frame is None:
            print("[camera] Failed to capture")
            continue

        resp = call_llm(text, frame)
        commands = resp.get("commands", [])
        say = resp.get("speak", "")
        observe = resp.get("observe", False)
        remember = resp.get("remember")

        if say:
            try:
                provider.speak(say, spk, mic_card)
            except Exception as e:
                print(f"[tts] {e}")
        if remember:
            save_memory(remember)

        # Observe loop — track gimbal state for alignment hints
        gimbal_pan, gimbal_tilt = 0.0, 0.0
        round_num = 0
        while observe and round_num < MAX_OBSERVE_ROUNDS:
            round_num += 1
            print(f"\n[observe round {round_num}/{MAX_OBSERVE_ROUNDS}]")
            # Track gimbal position from commands
            for cmd in commands:
                if cmd.get("T") == 133:
                    gimbal_pan = cmd.get("X", gimbal_pan)
                    gimbal_tilt = cmd.get("Y", gimbal_tilt)
            execute(ser, commands)
            time.sleep(0.3)

            frame = cam.snap()
            if frame is None:
                break

            drive_hint = ""
            if round_num <= 2:
                drive_hint = "Survey first — pan head to find the best route. "
            elif round_num <= 4:
                drive_hint = ("You've scanned enough. Now COMMIT: pick the best direction "
                              "(most open space/door), align body, and START DRIVING at 0.1 m/s. "
                              "Do NOT keep scanning without driving. ")
            else:
                drive_hint = ("Keep driving toward the goal at 0.1 m/s. "
                              "Only stop if wall fills >60% of frame. Steer to avoid obstacles. ")

            resp = call_llm(
                f"[Observation round {round_num}/{MAX_OBSERVE_ROUNDS}] "
                f"Head at pan={gimbal_pan:.0f}, tilt={gimbal_tilt:.0f}. "
                f"Original request: \"{text}\". "
                f"{drive_hint}"
                f"If target visible: center gimbal → align body (turn by pan°, gimbal to 0) → drive. "
                f"End commands with slow forward motion unless obstacle ahead.",
                frame)
            commands = resp.get("commands", [])
            say = resp.get("speak", "")
            observe = resp.get("observe", False)
            remember = resp.get("remember")

            if say:
                try:
                    provider.speak(say, spk, mic_card)
                except Exception as e:
                    print(f"[tts] {e}")
            if remember:
                save_memory(remember)

        execute(ser, commands)

if __name__ == "__main__":
    main()
