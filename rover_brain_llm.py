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

import os, sys, json, re, time, signal, subprocess, threading, queue
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
        self._lock = threading.Lock()
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        print(f"[serial] Opened {port} @ {baud}")

    def _send_raw(self, cmd):
        """Send a command without acquiring the lock (caller must hold it)."""
        if "_pause" in cmd:
            # Pauses are handled in execute() to keep them interruptible
            return
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("R", 0), R=-cmd.get("L", 0))
        raw = json.dumps(cmd) + "\n"
        self.ser.write(raw.encode("utf-8"))
        self.ser.readline()
        print(f"  -> {json.dumps(cmd)}")

    def send(self, cmd):
        with self._lock:
            self._send_raw(cmd)

    def stop(self):
        with self._lock:
            self._send_raw({"T": 1, "L": 0, "R": 0})
            self._send_raw({"T": 135})

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

# ── Smart Flash Snapshot ────────────────────────────────────────────────

LOW_LIGHT_THRESHOLD = 40  # mean brightness below this = too dark

def snap_with_flash(cam, ser):
    """Take a snapshot. If dark, flash lights on briefly for the capture."""
    # Quick brightness check from current frame
    jpg = cam.get_jpeg()
    if jpg is None:
        return cam.snap()
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    brightness = float(np.mean(img)) if img is not None else 255.0

    if brightness < LOW_LIGHT_THRESHOLD:
        print(f"[flash] Low light ({brightness:.0f}), lights on for snap")
        ser.send({"T": 132, "IO4": 255, "IO5": 255})
        time.sleep(0.15)  # let lights illuminate
        frame = cam.snap()
        ser.send({"T": 132, "IO4": 0, "IO5": 0})
        return frame
    return cam.snap()

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

def listen(mic_device, abort_event=None):
    rate = 48000
    chunk_sec = 0.5
    chunk_samples = int(rate * chunk_sec)
    silence_thresh = 0.03
    min_speech = 2
    max_speech = 240
    silence_after_long = 4   # 2s silence for longer utterances
    silence_after_short = 2  # 1s silence for short bursts (e.g. "stop")
    short_threshold = 4      # <= 4 speech chunks (2s) = short utterance

    proc = subprocess.Popen(
        ["arecord", "-D", mic_device, "-f", "S16_LE", "-r", str(rate),
         "-c", "1", "-t", "raw", "--buffer-size", str(chunk_samples)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    speech_chunks = []
    speech_started = False
    silent_count = 0

    try:
        while True:
            if abort_event and abort_event.is_set():
                return None
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
                # Short utterance = shorter silence wait (faster stop detection)
                needed = silence_after_short if len(speech_chunks) <= short_threshold else silence_after_long
                if silent_count >= needed:
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
{"commands":[<ESP32 JSON cmds>],"speak":"<5 words max>","observe":true/false,"remember":"<optional note>","landmarks":[{"name":"<what>","pan":<gimbal angle when seen>,"dist":"<near/mid/far>"},...]}

Use {"_pause": <seconds>} in the commands array to insert delays between commands.

## Body Size — Know Your Limits
You are 26cm wide, 35cm long, 20cm tall (without gimbal). With gimbal raised: 30cm tall.
- You CANNOT fit through gaps narrower than 30cm.
- Chair legs, narrow spaces between furniture, gaps under low shelves — you will get stuck.
- Before driving into a space, estimate if you fit. If it looks tight, DON'T try — go around.
- If a doorway or corridor looks narrow, slow down and center yourself carefully.
- You are low to the ground — you can go UNDER tables (if >25cm clearance) but NOT between tight chair legs.

## CRITICAL: Camera ≠ Body Direction
Your camera is on the gimbal (your head). When your head is panned, the camera does NOT face the same direction as your body.
- Wheels move the BODY: forward (L+,R+) = body's front direction, backward (L-,R-) = body's rear.
- The camera shows what your HEAD sees, which may be sideways or behind you.
- If gimbal is at pan=0: camera and body face the same way. Wheels work as expected.
- If gimbal is at pan=90 (looking right): the camera sees what's to your RIGHT. Driving "forward" (L+,R+) moves the body PERPENDICULAR to what the camera sees (body goes left relative to camera view).
- If gimbal is at pan=120 (looking back-right): driving "backward" (L-,R-) actually moves TOWARD what the camera sees.

**RULE: Before sending wheel commands, ALWAYS check your current gimbal pan angle.**
- If pan ≠ 0, you MUST first align your body to face the camera direction (incremental turning as described below), THEN drive.
- NEVER send wheel commands based on what you see in the camera unless pan=0. What you see and where your body drives are DIFFERENT when the head is rotated.
- To move AWAY from something you see with pan≠0: first align body to face it (turn toward it), THEN back up.
- To move TOWARD something you see with pan≠0: first align body to face it (turn toward it), THEN drive forward.

## ESP32 Commands
- Wheels: {"T":1, "L":<m/s>, "R":<m/s>} — max 1.0, default slow 0.2, negative=backward
- Gimbal (your head): {"T":133, "X":<pan -180..180>, "Y":<tilt -30..90>, "SPD":<50-500>, "ACC":<10-30>}
- Gimbal stop: {"T":135}
- Lights: {"T":132, "IO4":<base 0-255>, "IO5":<head 0-255>}
- OLED: {"T":3, "lineNum":<0-3>, "Text":"<max 16 chars>"}
- Feedback: {"T":130}
- Emergency stop: {"T":0}

## Turning Toward Objects — BODY follows CAMERA, never the reverse
CRITICAL RULE: The gimbal (camera/head) finds the target. Then the BODY rotates to match. NEVER reset the gimbal to 0 first — that loses sight of the target and the body drives into whatever it was already facing.

### Process:
1. Gimbal finds target at pan=X. KEEP the gimbal there — it's your eyes on the target.
2. ROTATE THE BODY toward where the gimbal is pointing. Turn duration = |X| / 120 seconds.
   - Gimbal panned RIGHT (X>0): body turns right → {"T":1,"L":0.15,"R":-0.15}
   - Gimbal panned LEFT (X<0): body turns left → {"T":1,"L":-0.15,"R":0.15}
3. WHILE the body turns, SIMULTANEOUSLY reduce gimbal pan to keep the target in view.
4. AFTER the body has finished turning, THEN set gimbal to 0. Now body faces the target.
5. Drive forward.

### Example: door seen at pan=50 (right)
Step 1 — body turns right, gimbal compensates to hold target:
  {"T":1,"L":0.15,"R":-0.15}, {"_pause":0.4}, {"T":1,"L":0,"R":0}, {"T":133,"X":0,"Y":0,"SPD":150,"ACC":15} → observe=true
Step 2 — verify target is ahead, then drive:
  {"T":1,"L":0.1,"R":0.1} → observe=true

### WRONG (never do this):
  {"T":133,"X":0,"Y":0,...} THEN {"T":1,"L":0.1,"R":0.1}  ← WRONG: resets camera first, body drives wherever it was facing, target is lost.

### Key rules:
- Body ALWAYS follows camera. Camera NEVER follows body.
- Don't reset gimbal to 0 until the body has turned to match.
- Good enough is good enough — don't over-align, but always turn the body first.

## Gimbal Tracking During Navigation
When navigating toward a goal (door, hallway, object), KEEP the gimbal pointed at the target while driving. Do NOT reset to 0,0 and drive blind.
- If the exit/target is at pan=20 right, keep gimbal at X=20 while steering the body right with differential drive to close the angle.
- Only set gimbal to 0 when the body is already aimed at the target (pan is near 0 naturally).
- This way you ALWAYS have eyes on the target. If you lose sight, stop and scan to reacquire.
- Think of it like looking where you want to go while steering — you don't stare straight ahead and hope.

## Spatial Memory — Landmarks
Every observe round, log what you see in the "landmarks" field. Each entry:
- "name": what it is (door, wall, chair, bag, person, hallway, open space)
- "pan": the gimbal pan angle when you saw it (this is the heading relative to your body)
- "dist": "near" (<50cm), "mid" (50cm-2m), "far" (>2m)

Example: {"landmarks":[{"name":"door","pan":45,"dist":"mid"},{"name":"wall","pan":-30,"dist":"near"},{"name":"open hallway","pan":90,"dist":"far"}]}

Use your accumulated landmarks to navigate:
- You scanned left and saw a wall at pan=-60. You scanned right and saw a door at pan=50. Now you KNOW the door is to your right — head there.
- If you drove forward and the door was at pan=50 last round, it's probably still roughly to your right. Steer toward it.
- Landmarks persist in conversation history, so you can reference what you saw earlier even if you can't see it now.
- Update landmarks each round — distances and angles change as you move.

## Observe Mode
Set "observe": true when you need to SEE the result before deciding next.
Your commands execute, a fresh camera frame is captured, and you get called again.
Use for: looking around, searching, navigating, checking results.
Max 5 rounds per cycle. Set observe to false (or omit) on your final round.
When observe returns an image: if the target is visible, note its pan angle from your landmarks, steer body toward it.

## Task Persistence
When given a complex goal (e.g. "find a way out", "go to the kitchen", "find the door"), PERSIST across observe rounds. Do NOT give up after one scan. Strategy:
1. Quick scan: one look left, center, right to get bearings (1-2 rounds max).
2. Make your BEST GUESS about which direction leads to the goal — even if you can't see it yet. Use common sense:
   - Doors and hallways likely lead to other rooms.
   - Open space = good direction to explore.
   - "Kitchen", "bathroom", "bedroom" — guess based on typical house layout and any clues (sounds, light, floor type).
   - If you see a corridor or doorway, go through it — it probably leads somewhere useful.
3. COMMIT and START MOVING. Do not over-scan. Pick a direction and go.
4. Drive a short distance (0.1 m/s for 1-2 seconds), then observe again.
5. Repeat: after each drive, re-evaluate. Adjust course toward the goal.
6. If blocked, try a different direction — but always keep moving.
7. Keep going until the goal is reached or you've exhausted all options.

IMPORTANT: You will almost NEVER see the target immediately. That's fine — use spatial reasoning and intuition to navigate toward it. Don't say "I can't see it" and stop. Instead, pick the most promising direction and drive. Explore actively. A wrong guess that keeps you moving is better than standing still scanning endlessly.

## Stuck Recovery
The system monitors your camera frames. If the scene hasn't changed for 3+ rounds, you'll see:
** STUCK DETECTED **
This means your wheel commands are NOT working — you're physically blocked or spinning in place.
Recovery strategy:
1. BACK UP first: {"T":1,"L":-0.15,"R":-0.15}, {"_pause":1.5}, {"T":1,"L":0,"R":0}
2. Turn your body 90-180 degrees away from the obstacle.
3. Try a COMPLETELY different direction — not a small adjustment.
4. If stuck twice in a row, do a full 180 and go the opposite way.
NEVER repeat the same commands that got you stuck. Each stuck warning means you must change strategy drastically.

## Continuous Movement During Navigation
When navigating, do NOT stop between observe rounds to think. Keep moving at a slow crawl speed (0.1 m/s) while you evaluate the next frame. Only stop wheels when you need to:
- Make a sharp turn (> 45 degrees)
- You are about to physically collide (obstacle fills >80% of frame, almost touching)
- You've arrived at the goal
For each observe round during navigation, END your commands with slow forward motion: {"T":1,"L":0.1,"R":0.1} — do NOT add a stop command after it. The next observe round will adjust or stop if needed. This keeps the rover flowing smoothly instead of jerky stop-start.

## Wall & Obstacle Avoidance
- Only back up when you are ACTUALLY about to collide — an obstacle fills >80% of the frame AND is clearly within 15cm (you can barely see any floor in front of you).
- People, pets, furniture that are visible but clearly 50cm+ away are NOT a collision threat. Drive past them or steer around — do NOT back away.
- To steer around obstacles: use differential wheel speeds. Curve left: {"T":1,"L":0.05,"R":0.15}. Curve right: {"T":1,"L":0.15,"R":0.05}.
- Prefer open space, but don't be afraid of objects at a distance. Only react when things are VERY close.
- Be OPTIMISTIC: if there's a gap, try it. If something is far away, keep driving toward your goal.
- Backing up is a LAST RESORT — only when physically touching or about to touch something. Otherwise steer around.

## Rules
- Always include physical expression (nod, look, tilt) in commands — you're a robot, move your head
- "speak" max 5 words — terse, robotic
- Default wheel speed 0.2 m/s for general commands, 0.1 m/s during autonomous navigation
- GIMBAL TRACKS THE TARGET while navigating. Keep your eyes on the goal. Body steers to follow.
- Only set gimbal to 0 when body already faces the target (pan angle is near 0).
- Only back up if physically blocked (obstacle fills >80% of frame, no visible floor ahead)
- Lights: off by default. System flashes them automatically for camera in dark conditions. But if the USER asks for lights on/off, obey with {"T":132,"IO4":<0-255>,"IO5":<0-255>}
- NEVER fabricate what you see. If unclear, say so. Only describe what's actually in the image.

## Examples
- "nod" → {"commands":[{"T":133,"X":0,"Y":30,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":-10,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],"speak":"Sure."}
- "look around" → {"commands":[{"T":133,"X":-60,"Y":0,"SPD":200,"ACC":15}],"speak":"Looking.","observe":true}
- "go forward" → {"commands":[{"T":1,"L":0.2,"R":0.2},{"_pause":2},{"T":1,"L":0,"R":0}],"speak":"Moving."}
- "find the door" → {"commands":[{"T":133,"X":-90,"Y":0,"SPD":200,"ACC":15}],"speak":"Searching.","observe":true}
- target at pan=40 (right), body turns right FIRST → {"commands":[{"T":1,"L":0.15,"R":-0.15},{"_pause":0.35},{"T":1,"L":0,"R":0},{"T":133,"X":0,"Y":0,"SPD":150,"ACC":15}],"speak":"Turning.","observe":true}
- target at pan=-60 (left), body turns left FIRST → {"commands":[{"T":1,"L":-0.15,"R":0.15},{"_pause":0.5},{"T":1,"L":0,"R":0},{"T":133,"X":0,"Y":0,"SPD":150,"ACC":15}],"speak":"Turning.","observe":true}
- wall too close → {"commands":[{"T":1,"L":-0.15,"R":-0.15},{"_pause":1.0},{"T":1,"L":0,"R":0}],"speak":"Backing up.","observe":true}

## Mid-Plan User Messages
Sometimes the user will speak to you DURING an observe loop. Their message appears as:
** USER SAID (mid-plan): "..." **

How to handle:
- Minor adjustments ("turn right", "slower", "look up"): incorporate into your next commands without abandoning the plan.
- Questions ("what do you see?", "where are we?"): answer in "speak" AND continue the plan.
- Contradictions ("actually go left"): adjust your plan direction. Don't restart from scratch — just change course.
- The user's original request is still your primary goal unless they explicitly say otherwise.
- If in doubt, keep going with the plan and acknowledge the user's input in "speak".""")
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

    # Parse — strip fences, find outermost {}, attempt repair
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
        pass
    # Attempt repairs for common LLM JSON errors
    repaired = clean
    repaired = re.sub(r'(\d)"(\})', r'\1\2', repaired)  # 15"} → 15}
    repaired = re.sub(r'(\d)"(\])', r'\1\2', repaired)  # 15"] → 15]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)   # trailing comma
    repaired = re.sub(r'"\s*"', '","', repaired)          # missing comma between strings
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    print(f"[llm] Failed to parse JSON: {clean[:200]}")
    return {"commands": [], "speak": "Hmm."}

# ── Command Execution ───────────────────────────────────────────────────

def _interruptible_sleep(secs, stop_event=None):
    """Sleep in small increments, checking stop_event."""
    elapsed = 0.0
    while elapsed < secs:
        if stop_event and stop_event.is_set():
            return True  # interrupted
        time.sleep(min(0.1, secs - elapsed))
        elapsed += 0.1
    return False

def execute(ser, commands, stop_event=None):
    last_pan, last_tilt = 0.0, 0.0
    for cmd in commands:
        if stop_event and stop_event.is_set():
            ser.stop()
            return
        # Handle _pause without holding serial lock
        if "_pause" in cmd:
            secs = float(cmd["_pause"])
            if secs > 0:
                print(f"  (_pause {secs:.2f}s)")
                if _interruptible_sleep(secs, stop_event):
                    ser.stop()
                    return
            continue
        ser.send(cmd)
        t = cmd.get("T")
        if t == 133:
            new_pan = cmd.get("X", last_pan)
            new_tilt = cmd.get("Y", last_tilt)
            spd = cmd.get("SPD", 200)
            dist = abs(new_pan - last_pan) + abs(new_tilt - last_tilt)
            wait = max(0.15, dist / max(spd, 1) * 1.1)
            if _interruptible_sleep(wait, stop_event):
                ser.stop()
                return
            last_pan, last_tilt = new_pan, new_tilt

# ── Memory ──────────────────────────────────────────────────────────────

def save_memory(note):
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(MEMORY_FILE, "a") as f:
        f.write(f"- {note} [{ts}]\n")
    print(f"[memory] Saved: {note}")

# ── Stuck Detection ────────────────────────────────────────────────────

STUCK_SIM_DRIVING = 0.88   # with wheels active, this similar = pushing against something
STUCK_SIM_STATIC = 0.96    # without wheels, this similar = not actually scanning
STUCK_ROUNDS_DRIVING = 1   # trigger immediately when driving into something
STUCK_ROUNDS_STATIC = 3    # need more rounds when just scanning

def frame_similarity(jpg_a, jpg_b):
    """Compare two JPEG frames. Returns 0.0 (different) to 1.0 (identical)."""
    if jpg_a is None or jpg_b is None:
        return 0.0
    try:
        a = cv2.imdecode(np.frombuffer(jpg_a, np.uint8), cv2.IMREAD_GRAYSCALE)
        b = cv2.imdecode(np.frombuffer(jpg_b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return 0.0
        # Resize to small thumbnails for fast comparison
        a = cv2.resize(a, (64, 48))
        b = cv2.resize(b, (64, 48))
        diff = np.mean(np.abs(a.astype(float) - b.astype(float))) / 255.0
        return 1.0 - diff
    except Exception:
        return 0.0

# ── Threading State ─────────────────────────────────────────────────────

CANCEL_WORDS = {"cancel", "forget it", "nevermind", "never mind", "abort"}

interrupt_queue = queue.Queue()   # voice thread → run_plan (inject messages)
command_queue = queue.Queue()     # voice thread → main loop (new commands)
plan_active = threading.Event()   # True while observe loop is running
stop_event = threading.Event()    # Signals immediate stop to executor

# Will be set in main() so voice_thread can call ser.stop() directly
_ser_ref = None

def _clean_words(text):
    """Strip punctuation and return lowercase word set."""
    return set(re.sub(r'[^\w\s]', '', text.lower()).split())

def classify_interrupt(text):
    """Classify an interrupt: stop / cancel / override / inject."""
    words = _clean_words(text)
    if words & STOP_WORDS:
        return "stop"
    if words & CANCEL_WORDS:
        return "cancel"
    # Check multi-word cancel phrases
    lower = re.sub(r'[^\w\s]', '', text.lower())
    for phrase in ("forget it", "never mind"):
        if phrase in lower:
            return "cancel"
    first_word = re.sub(r'[^\w]', '', text.lower().split()[0]) if text.strip() else ""
    action_starters = {"go", "find", "navigate", "drive", "turn", "look",
                       "move", "come", "search", "back", "reverse"}
    if first_word in action_starters:
        return "override"
    return "inject"

def voice_thread(mic_dev, mic_card):
    """Always-on listener. Routes messages to interrupt_queue or command_queue."""
    while True:
        audio = listen(mic_dev)
        if audio is None:
            continue

        try:
            text = provider.transcribe(audio)
        except Exception as e:
            print(f"[stt] Error: {e}")
            continue

        if not text or text in HALLUCINATIONS or len(text) <= 2:
            continue

        print(f'[heard] "{text}"')

        if plan_active.is_set():
            kind = classify_interrupt(text)
            print(f"[interrupt] {kind}: {text}")

            if kind == "stop":
                stop_event.set()
                try:
                    if _ser_ref:
                        _ser_ref.stop()
                except Exception:
                    pass

            elif kind == "cancel":
                stop_event.set()

            elif kind == "override":
                stop_event.set()
                command_queue.put(text)

            elif kind == "inject":
                interrupt_queue.put(text)
        else:
            command_queue.put(text)

def run_plan(text, ser, cam, spk, mic_card):
    """Execute an LLM command cycle with interruptible observe loop."""
    plan_active.set()
    stop_event.clear()
    # Drain any stale interrupts
    while not interrupt_queue.empty():
        try:
            interrupt_queue.get_nowait()
        except queue.Empty:
            break

    frame = snap_with_flash(cam, ser)
    if frame is None:
        plan_active.clear()
        return

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

    gimbal_pan, gimbal_tilt = 0.0, 0.0
    round_num = 0
    prev_frame = frame          # for stuck detection
    similar_count = 0           # consecutive similar frames
    wheels_were_active = False  # did last command set include wheel motion?

    while True:
        if stop_event.is_set():
            print("[plan] Interrupted — stopping.")
            ser.stop()
            plan_active.clear()
            return

        # Track gimbal + check for wheel commands + detect turning
        wheels_were_active = False
        is_turning = False
        for cmd in commands:
            if isinstance(cmd, dict):
                if cmd.get("T") == 133:
                    gimbal_pan = cmd.get("X", gimbal_pan)
                    gimbal_tilt = cmd.get("Y", gimbal_tilt)
                if cmd.get("T") == 1:
                    l, r = cmd.get("L", 0), cmd.get("R", 0)
                    if l != 0 or r != 0:
                        wheels_were_active = True
                    # Opposite signs = body rotation
                    if l != 0 and r != 0 and (l > 0) != (r > 0):
                        is_turning = True

        execute(ser, commands, stop_event)

        if stop_event.is_set():
            ser.stop()
            plan_active.clear()
            return

        if not observe:
            break

        round_num += 1
        if round_num >= MAX_OBSERVE_ROUNDS:
            print("[plan] Max observe rounds reached.")
            break

        # Skip delay during active turning — send frame ASAP
        if not is_turning:
            time.sleep(0.3)

        if stop_event.is_set():
            ser.stop()
            plan_active.clear()
            return

        frame = snap_with_flash(cam, ser)
        if frame is None:
            break

        # ── Stuck detection ──
        sim = frame_similarity(prev_frame, frame)
        if wheels_were_active:
            thresh = STUCK_SIM_DRIVING
            needed = STUCK_ROUNDS_DRIVING
        else:
            thresh = STUCK_SIM_STATIC
            needed = STUCK_ROUNDS_STATIC
        print(f"[sim] {sim:.3f} thresh={thresh} wheels={'ON' if wheels_were_active else 'off'}")
        if sim >= thresh:
            similar_count += 1
        else:
            similar_count = 0
        prev_frame = frame
        stuck = similar_count >= needed
        if stuck:
            print(f"[STUCK] {similar_count} rounds (wheels={'ON' if wheels_were_active else 'off'}, sim={sim:.3f})")

        # Drain injected messages
        injected = []
        while not interrupt_queue.empty():
            try:
                injected.append(interrupt_queue.get_nowait())
            except queue.Empty:
                break

        drive_hint = ""
        if is_turning:
            drive_hint = (
                f"You just turned (gimbal at pan={gimbal_pan:.0f}). "
                f"Keep gimbal tracking the target. Steer body to close the angle. "
                f"Don't reset gimbal to 0 until body faces the target. ")
        elif round_num <= 2:
            drive_hint = "Survey first — pan head to find the best route. "
        elif round_num <= 4:
            drive_hint = ("You've scanned enough. Now COMMIT: pick the best direction, "
                          "align body, and START DRIVING at 0.1 m/s. ")
        else:
            drive_hint = ("Keep driving toward the goal at 0.1 m/s. "
                          "Only stop if wall fills >60% of frame. ")

        stuck_hint = ""
        if stuck and wheels_were_active:
            stuck_hint = (
                f"\n\n** PUSHING AGAINST OBSTACLE ({similar_count} rounds, wheels ON but no movement) ** "
                f"You are driving into something and NOT moving. STOP wheels immediately. "
                f"BACK UP: {'{'}\"T\":1,\"L\":-0.15,\"R\":-0.15{'}'}, then turn 90-180 degrees "
                f"and try a COMPLETELY different direction. Do NOT keep driving forward.")
        elif stuck:
            stuck_hint = (
                f"\n\n** NOT MAKING PROGRESS ({similar_count} rounds, scene unchanged) ** "
                f"You keep scanning without moving. COMMIT to a direction and DRIVE. "
                f"Pick the most open path and start moving at 0.1 m/s.")

        user_context = ""
        if injected:
            joined = "; ".join(injected)
            user_context = (
                f'\n\n** USER SAID (mid-plan): "{joined}" ** '
                f"Incorporate this into your current plan. "
                f"If it's a minor adjustment, adapt. "
                f"If it contradicts your goal, acknowledge and adjust.")

        prompt = (
            f"[Observe round {round_num}/{MAX_OBSERVE_ROUNDS}] "
            f"Head at pan={gimbal_pan:.0f}, tilt={gimbal_tilt:.0f}. "
            f'Original request: "{text}". '
            f"{drive_hint}"
            f"If target visible: keep gimbal on it, steer body toward it. "
            f"End commands with motion toward goal unless blocked."
            f"{stuck_hint}{user_context}")

        resp = call_llm(prompt, frame)
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

    # Execute final commands if not already done in last round
    plan_active.clear()

# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    global _ser_ref
    subprocess.run(["pkill", "-f", "rover_brain.py"], capture_output=True)
    time.sleep(1)

    ser = Serial()
    _ser_ref = ser
    cam = Camera()
    start_stream_server(cam)
    mic_dev, mic_card = find_mic()
    spk = find_speaker()

    print(f"[provider] {provider.NAME}")

    def cleanup(*_):
        print("\n[shutdown]")
        stop_event.set()
        ser.close()
        cam.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    ser.send({"T": 132, "IO4": 0, "IO5": 0})  # lights off by default
    ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(0.5)

    # Start voice listener thread
    vt = threading.Thread(target=voice_thread, args=(mic_dev, mic_card), daemon=True)
    vt.start()

    print("\n=== rover_brain_llm ready ===\n")

    # Startup greeting
    frame = snap_with_flash(cam, ser)
    if frame:
        resp = call_llm("You just booted up. Greet briefly and look around with a head movement.", frame)
        execute(ser, resp.get("commands", []))
        try:
            provider.speak(resp.get("speak", "Online."), spk, mic_card)
        except Exception as e:
            print(f"[tts] {e}")

    # Main loop — wait for commands from voice thread
    while True:
        try:
            text = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Safety check for stop words (voice thread handles these too)
        if _clean_words(text) & STOP_WORDS:
            ser.stop()
            continue

        print(f"[plan] Starting: {text}")
        run_plan(text, ser, cam, spk, mic_card)

if __name__ == "__main__":
    main()
