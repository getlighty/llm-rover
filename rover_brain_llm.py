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
import collections
import importlib
import math

import serial
import cv2
import numpy as np

# ── Extracted modules ──────────────────────────────────────────────────

import audio
import prompts
import reflection
import web_ui

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

# ── Log Event Bus ──────────────────────────────────────────────────────

_log_events = collections.deque(maxlen=500)
_log_lock = threading.Lock()

def log_event(category, data):
    """Append a structured log event and print to stdout."""
    entry = {"ts": time.time(), "cat": category, "data": data}
    with _log_lock:
        _log_events.append(entry)
    print(f"[{category}] {data}", flush=True)

def get_log_events_since(last_ts):
    """Return list of events newer than last_ts."""
    with _log_lock:
        return [e for e in _log_events if e["ts"] > last_ts]

# ── Provider Loading ───────────────────────────────────────────────────

# Load provider from CLI arg (sets initial LLM provider)
_provider_name = sys.argv[1] if len(sys.argv) > 1 else "groq"
provider = importlib.import_module(f"provider_{_provider_name}")

# Decoupled STT / LLM / TTS modules
stt_mod = importlib.import_module("stt_groq")
llm_mod = provider  # reuse provider_*.call_llm()
tts_mod = importlib.import_module("tts_groq")

_stt_name = "groq"
_llm_name = _provider_name
_tts_name = "groq"

AVAILABLE_PROVIDERS = {
    "stt": ["groq"],
    "llm": ["groq", "xai", "anthropic", "ollama"],
    "tts": ["groq", "elevenlabs"],
}

def set_provider(kind, name):
    """Hot-swap a provider at runtime. kind = 'stt'|'llm'|'tts'."""
    global stt_mod, llm_mod, tts_mod, _stt_name, _llm_name, _tts_name
    if kind == "stt":
        mod = importlib.import_module(f"stt_{name}")
        importlib.reload(mod)
        stt_mod = mod
        _stt_name = name
    elif kind == "llm":
        mod = importlib.import_module(f"provider_{name}")
        importlib.reload(mod)
        llm_mod = mod
        _llm_name = name
    elif kind == "tts":
        mod = importlib.import_module(f"tts_{name}")
        importlib.reload(mod)
        tts_mod = mod
        _tts_name = name
    else:
        raise ValueError(f"Unknown provider kind: {kind}")
    log_event("system", f"Switched {kind} to {name}")

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200

MAX_OBSERVE_ROUNDS = 15
STOP_WORDS = {"stop", "halt", "freeze", "emergency"}
desk_mode = False  # When True, all wheel commands (T=1) are blocked
stt_enabled = True  # When False, voice_thread skips STT (mic muted)

def _check_floor(resp):
    """Auto-toggle desk_mode based on LLM's on_floor field."""
    global desk_mode
    on_floor = resp.get("on_floor")
    if on_floor is None:
        return
    new_mode = not on_floor
    if new_mode != desk_mode:
        desk_mode = new_mode
        log_event("system", f"Auto desk mode: {'ON (elevated)' if desk_mode else 'OFF (on floor)'}")

HALLUCINATIONS = {
    ".", "..", "...", "Thank you.", "Thanks for watching.",
    "Bye.", "Thank you for watching.", "Subscribe.",
    "you", "You", "I'm sorry.", "Okay.", "Yeah.",
    "Hmm.", "Mm-hmm.", "Uh-huh.", "Oh.", "Ah.",
    "So.", "Well.", "Right.", "Sure.", "OK.",
}

import re as _re

def _is_gibberish(text):
    """Return True if STT text looks like noise/bleed rather than a real command."""
    stripped = text.strip().rstrip(".")
    # Too short after stripping punctuation
    if len(stripped) < 3:
        return True
    # Single word that isn't a plausible command
    words = stripped.split()
    if len(words) == 1 and stripped.lower() not in (
        "stop", "halt", "go", "forward", "backward", "back", "left", "right",
        "spin", "look", "scan", "hello", "hi", "hey", "help", "status",
        "faster", "slower", "lights", "dance", "explore", "navigate",
        "where", "what", "who", "why", "how", "cancel", "quiet", "reset",
        "yes", "no", "come", "stay", "wait", "move", "turn", "park",
    ):
        return True
    # Mostly non-ASCII → likely non-English STT artifact
    ascii_chars = sum(1 for c in stripped if c.isascii())
    if len(stripped) > 0 and ascii_chars / len(stripped) < 0.7:
        return True
    # Mostly non-alpha (sound effects like "Pfff", "Tss", random chars)
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 3 and alpha_chars / len(stripped) < 0.5:
        return True
    # Repetitive single-char patterns like "aaah", "pfff", "shhh"
    if len(words) == 1 and len(stripped) >= 3 and len(set(stripped.lower())) <= 2:
        return True
    return False

# ── Serial ──────────────────────────────────────────────────────────────

class Serial:
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self._lock = threading.Lock()
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        log_event("system", f"Serial opened {port} @ {baud}")

    def _send_raw(self, cmd):
        """Send a command without acquiring the lock (caller must hold it)."""
        if "_pause" in cmd:
            # Pauses are handled in execute() to keep them interruptible
            return
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            if desk_mode and (cmd.get("L", 0) != 0 or cmd.get("R", 0) != 0):
                log_event("system", "Desk mode: wheel command blocked")
                return
            cmd = dict(cmd, L=-cmd.get("R", 0), R=-cmd.get("L", 0))
        raw = json.dumps(cmd) + "\n"
        self.ser.write(raw.encode("utf-8"))
        self.ser.readline()
        log_event("serial", json.dumps(cmd))

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
        log_event("system", "Serial closed")

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
        log_event("system", "Camera ready (640x480)")

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
        log_event("system", "Camera closed")

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
        log_event("system", f"Flash: low light ({brightness:.0f})")
        ser.send({"T": 132, "IO4": 255, "IO5": 255})
        time.sleep(0.15)  # let lights illuminate
        frame = cam.snap()
        ser.send({"T": 132, "IO4": 0, "IO5": 0})
        return frame
    return cam.snap()

# ── LLM wrapper ─────────────────────────────────────────────────────────

history = []

def call_llm(text, jpeg_bytes):
    """Call provider LLM, parse JSON response."""
    global history
    system = prompts.build_system_prompt()

    history.append({"role": "user", "content": text})
    if len(history) > 10:
        history = history[-10:]

    try:
        reply = llm_mod.call_llm(text, jpeg_bytes, system, history)
    except Exception as e:
        log_event("error", f"LLM error: {e}")
        reply = json.dumps({"commands": [], "speak": "LLM error."})

    history.append({"role": "assistant", "content": reply})
    log_event("llm", reply)

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
    log_event("error", f"JSON parse failed: {clean[:200]}")
    return {"commands": [], "speak": "Hmm."}


# ── Command Execution ───────────────────────────────────────────────────

def _interruptible_sleep(secs, stop_ev=None):
    """Sleep in small increments, checking stop_event."""
    elapsed = 0.0
    while elapsed < secs:
        if stop_ev and stop_ev.is_set():
            return True  # interrupted
        time.sleep(min(0.1, secs - elapsed))
        elapsed += 0.1
    return False

DRIVE_PAN_LIMIT = 30  # max |pan| degrees to allow forward/backward driving
WHEEL_SEP = 0.20      # meters between left/right wheels

def execute(ser, commands, stop_ev=None):
    global _gimbal_pan
    last_pan = _gimbal_pan  # start from actual current gimbal position
    last_tilt = 0.0
    pending_spin = None     # (L, R) of a spin command awaiting its _pause

    for cmd in commands:
        if stop_ev and stop_ev.is_set():
            ser.stop()
            return
        # Handle _pause without holding serial lock
        if "_pause" in cmd:
            secs = float(cmd["_pause"])
            if secs > 0:
                log_event("serial", f"_pause {secs:.2f}s")
                if _interruptible_sleep(secs, stop_ev):
                    ser.stop()
                    return
                # Auto-compensate gimbal after body spin to keep camera steady
                if pending_spin is not None:
                    sp_l, sp_r = pending_spin
                    omega = (sp_r - sp_l) / WHEEL_SEP  # rad/s
                    body_delta_deg = math.degrees(omega * secs)
                    new_pan = last_pan - body_delta_deg
                    new_pan = max(-180, min(180, new_pan))
                    if abs(new_pan - last_pan) > 2:
                        gimbal_cmd = {"T": 133, "X": round(new_pan, 1),
                                      "Y": last_tilt, "SPD": 300, "ACC": 20}
                        ser.send(gimbal_cmd)
                        log_event("system",
                            f"Gimbal compensate: body turned {body_delta_deg:+.0f}°, "
                            f"pan {last_pan:.0f}→{new_pan:.0f}°")
                        last_pan = round(new_pan, 1)
                        _gimbal_pan = last_pan
                    pending_spin = None
            continue
        # Safety: block forward/backward driving when head is turned sideways
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            l, r = cmd.get("L", 0), cmd.get("R", 0)
            if l != 0 or r != 0:
                is_spin = (l != 0 and r != 0 and (l > 0) != (r > 0))
                if is_spin:
                    pending_spin = (l, r)
                elif abs(last_pan) > DRIVE_PAN_LIMIT:
                    log_event("system",
                        f"Blocked drive: head at pan={last_pan:.0f}° "
                        f"(limit ±{DRIVE_PAN_LIMIT}°). Align head first.")
                    continue
        ser.send(cmd)
        t = cmd.get("T")
        if t == 133:
            new_pan = cmd.get("X", last_pan)
            new_tilt = cmd.get("Y", last_tilt)
            spd = cmd.get("SPD", 200)
            dist = abs(new_pan - last_pan) + abs(new_tilt - last_tilt)
            wait = max(0.15, dist / max(spd, 1) * 1.1)
            if _interruptible_sleep(wait, stop_ev):
                ser.stop()
                return
            last_pan, last_tilt = new_pan, new_tilt
            _gimbal_pan = last_pan  # sync to global for radar
        elif t == 1:
            # Non-spin wheel command clears pending_spin
            l, r = cmd.get("L", 0), cmd.get("R", 0)
            if not (l != 0 and r != 0 and (l > 0) != (r > 0)):
                pending_spin = None

# ── Memory ──────────────────────────────────────────────────────────────

MEMORY_FILE = os.path.join(ROVER_DIR, "memory.md")

def save_memory(note):
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(MEMORY_FILE, "a") as f:
        f.write(f"- {note} [{ts}]\n")
    log_event("system", f"Memory saved: {note}")

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

# Negative feedback detection
NEGATIVE_WORDS = {"wrong", "bad", "nope"}
NEGATIVE_PHRASES = ("not that", "other one", "wrong way", "wrong one",
                    "not there", "why did you", "no no", "what are you doing")

interrupt_queue = queue.Queue()   # voice thread → run_plan (inject messages)
command_queue = queue.Queue()     # voice thread → main loop (new commands)
plan_active = threading.Event()   # True while observe loop is running
stop_event = threading.Event()    # Signals immediate stop to executor

# Will be set in main() so voice_thread can call ser.stop() directly
_ser_ref = None

# ── 2D World Map ──────────────────────────────────────────────────────

_map_lock = threading.Lock()
_landmarks = {}       # name → {"name","x","y","type","ts"}
_rover_x = 0.0        # world position meters
_rover_y = 0.0
_rover_heading = 0.0  # radians, 0 = +Y (forward at boot)
_gimbal_pan = 0.0     # degrees
LANDMARK_MAX_AGE = 300  # 5 minutes

def _dist_to_meters(d):
    """Convert dist field to meters. Accepts number or legacy string."""
    if isinstance(d, (int, float)):
        return float(d)
    return {"near": 0.4, "mid": 1.2, "far": 2.5}.get(str(d), 1.2)

def _update_pose_from_commands(commands):
    """Dead-reckon rover position from wheel commands."""
    global _rover_x, _rover_y, _rover_heading
    WHEEL_SEP = 0.20  # meters between left/right wheels
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        if cmd.get("T") == 1:
            L = cmd.get("L", 0)
            R = cmd.get("R", 0)
            if L == 0 and R == 0:
                continue
            # Find the associated _pause duration
            idx = commands.index(cmd)
            dt = 0.0
            for j in range(idx + 1, len(commands)):
                c2 = commands[j]
                if isinstance(c2, dict) and "_pause" in c2:
                    dt = float(c2["_pause"])
                    break
            if dt == 0:
                dt = 0.3  # default if no pause follows
            v = (L + R) / 2.0
            omega = (R - L) / WHEEL_SEP
            _rover_heading += omega * dt
            _rover_x += v * dt * math.sin(_rover_heading)
            _rover_y += v * dt * math.cos(_rover_heading)

def _update_landmarks(resp):
    """Convert LLM body-relative x,y landmarks to world coords and merge."""
    now = time.time()
    commands = resp.get("commands", [])
    lm = resp.get("landmarks")
    if lm and isinstance(lm, list):
        with _map_lock:
            for entry in lm:
                name = entry.get("name", "")
                if not name:
                    continue
                bx = float(entry.get("x", 0))
                by = float(entry.get("y", 0))
                if "pan" in entry and ("x" not in entry or "y" not in entry):
                    pan_deg = entry.get("pan", 0)
                    dist_m = _dist_to_meters(entry.get("dist", 1.2))
                    bx = dist_m * math.sin(math.radians(pan_deg))
                    by = dist_m * math.cos(math.radians(pan_deg))
                cos_h = math.cos(_rover_heading)
                sin_h = math.sin(_rover_heading)
                wx = _rover_x + bx * cos_h + by * sin_h
                wy = _rover_y - bx * sin_h + by * cos_h
                _landmarks[name] = {
                    "name": name,
                    "x": round(wx, 2), "y": round(wy, 2),
                    "type": entry.get("type", "object"),
                    "ts": now,
                }
            stale = [k for k, v in _landmarks.items() if now - v["ts"] > LANDMARK_MAX_AGE]
            for k in stale:
                del _landmarks[k]
    # Update rover pose AFTER placing landmarks at current position
    _update_pose_from_commands(commands)

def _get_map_state():
    now = time.time()
    with _map_lock:
        lm = []
        for v in _landmarks.values():
            entry = dict(v)
            entry["age"] = now - v["ts"]
            lm.append(entry)
        return {
            "landmarks": lm,
            "rover": {"x": _rover_x, "y": _rover_y,
                       "heading": math.degrees(_rover_heading),
                       "gimbal_pan": _gimbal_pan},
        }

def _truncate_after_first_gimbal(commands):
    """Keep only up to the first gimbal move + its pause. Returns truncated list."""
    truncated = []
    gimbal_seen = False
    for cmd in commands:
        truncated.append(cmd)
        if isinstance(cmd, dict) and cmd.get("T") == 133:
            if gimbal_seen:
                truncated.pop()
                break
            gimbal_seen = True
        elif gimbal_seen and "_pause" not in (cmd if isinstance(cmd, dict) else {}):
            break
    return truncated

def _speak(text, spk, mic_card):
    """Speak with TTS via audio module."""
    audio.speak(text, tts_mod, spk, mic_card, log_fn=log_event)

def _clean_words(text):
    """Strip punctuation and return lowercase word set."""
    return set(re.sub(r'[^\w\s]', '', text.lower()).split())

def classify_interrupt(text):
    """Classify an interrupt: stop / cancel / feedback_negative / override / inject."""
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
    # Check for negative feedback
    if words & NEGATIVE_WORDS:
        return "feedback_negative"
    for phrase in NEGATIVE_PHRASES:
        if phrase in lower:
            return "feedback_negative"
    first_word = re.sub(r'[^\w]', '', text.lower().split()[0]) if text.strip() else ""
    action_starters = {"go", "find", "navigate", "drive", "turn", "look",
                       "move", "come", "search", "back", "reverse"}
    if first_word in action_starters:
        return "override"
    return "inject"

def voice_thread(mic_dev, mic_card):
    """Always-on listener. Routes messages to interrupt_queue or command_queue."""
    while True:
        if not stt_enabled:
            time.sleep(0.5)
            continue
        audio_data = audio.listen(mic_dev)
        if audio_data is None:
            continue

        try:
            text = stt_mod.transcribe(audio_data)
        except Exception as e:
            log_event("error", f"STT error: {e}")
            continue

        if not text or text in HALLUCINATIONS or len(text) <= 2:
            continue
        if _is_gibberish(text):
            log_event("heard", f"(ignored gibberish) {text}")
            continue

        log_event("heard", text)

        if plan_active.is_set():
            kind = classify_interrupt(text)
            log_event("interrupt", f"{kind}: {text}")

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

            elif kind == "feedback_negative":
                # Inject into plan AND flag for reflection
                interrupt_queue.put(text)

            elif kind == "inject":
                interrupt_queue.put(text)
        else:
            command_queue.put(text)

def run_plan(text, ser, cam, spk, mic_card):
    """Execute an LLM command cycle with interruptible observe loop.

    Returns metadata dict: {original_text, feedback, stuck_count, rounds, interrupted}
    """
    plan_active.set()
    stop_event.clear()
    # Drain any stale interrupts
    while not interrupt_queue.empty():
        try:
            interrupt_queue.get_nowait()
        except queue.Empty:
            break

    # Tracking for reflection
    plan_feedback = []       # negative feedback texts
    total_stuck_events = 0   # number of stuck detections
    plan_history = []        # brief round summaries

    frame = snap_with_flash(cam, ser)
    if frame is None:
        plan_active.clear()
        return {"original_text": text, "feedback": [], "stuck_count": 0,
                "rounds": 0, "interrupted": False, "history": []}

    resp = call_llm(text, frame)
    _check_floor(resp)
    _update_landmarks(resp)
    commands = resp.get("commands", [])
    say = resp.get("speak", "")
    observe = resp.get("observe", False)
    remember = resp.get("remember")

    if observe and commands:
        commands = _truncate_after_first_gimbal(commands)

    if say:
        _speak(say, spk, mic_card)
        plan_history.append(f"Said: {say}")
    if remember:
        save_memory(remember)

    # Init local gimbal tracking from the first response's commands
    gimbal_pan, gimbal_tilt = 0.0, 0.0
    for cmd in commands:
        if isinstance(cmd, dict) and cmd.get("T") == 133:
            gimbal_pan = cmd.get("X", gimbal_pan)
            gimbal_tilt = cmd.get("Y", gimbal_tilt)
    round_num = 0
    prev_frame = frame          # for stuck detection
    similar_count = 0           # consecutive similar frames
    wheels_were_active = False  # did last command set include wheel motion?
    was_interrupted = False

    while True:
        if stop_event.is_set():
            log_event("plan", "Interrupted — stopping.")
            ser.stop()
            was_interrupted = True
            plan_active.clear()
            return {"original_text": text, "feedback": plan_feedback,
                    "stuck_count": total_stuck_events, "rounds": round_num,
                    "interrupted": True, "history": plan_history}

        # During observe rounds, truncate after the first gimbal move
        if observe and commands:
            commands = _truncate_after_first_gimbal(commands)

        # Track gimbal + check for wheel commands + detect turning
        # Also build action summary for the LLM to understand what happened
        wheels_were_active = False
        is_turning = False
        action_parts = []
        for cmd in commands:
            if isinstance(cmd, dict):
                if cmd.get("T") == 133:
                    gimbal_pan = cmd.get("X", gimbal_pan)
                    gimbal_tilt = cmd.get("Y", gimbal_tilt)
                    action_parts.append(f"Head moved to pan={gimbal_pan:.0f}°")
                if cmd.get("T") == 1:
                    l, r = cmd.get("L", 0), cmd.get("R", 0)
                    if l == 0 and r == 0:
                        action_parts.append("Wheels stopped")
                    elif l != 0 and r != 0 and (l > 0) != (r > 0):
                        is_turning = True
                        wheels_were_active = True
                        direction = "right" if l > 0 else "left"
                        action_parts.append(f"Body spun {direction}")
                    elif l != 0 or r != 0:
                        wheels_were_active = True
                        if l > 0 and r > 0:
                            if abs(l - r) < 0.03:
                                action_parts.append(f"Drove forward at {(l+r)/2:.2f} m/s")
                            elif l > r:
                                action_parts.append(f"Curved right at {(l+r)/2:.2f} m/s")
                            else:
                                action_parts.append(f"Curved left at {(l+r)/2:.2f} m/s")
                        elif l < 0 and r < 0:
                            action_parts.append(f"Backed up at {abs((l+r)/2):.2f} m/s")
        last_action = "; ".join(action_parts) if action_parts else "No commands"

        execute(ser, commands, stop_event)

        if stop_event.is_set():
            ser.stop()
            plan_active.clear()
            return {"original_text": text, "feedback": plan_feedback,
                    "stuck_count": total_stuck_events, "rounds": round_num,
                    "interrupted": True, "history": plan_history}

        if not observe:
            break

        round_num += 1
        if round_num >= MAX_OBSERVE_ROUNDS:
            log_event("plan", "Max observe rounds reached.")
            break

        # Skip delay during active turning — send frame ASAP
        if not is_turning:
            time.sleep(0.3)

        if stop_event.is_set():
            ser.stop()
            plan_active.clear()
            return {"original_text": text, "feedback": plan_feedback,
                    "stuck_count": total_stuck_events, "rounds": round_num,
                    "interrupted": True, "history": plan_history}

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
        log_event("system", f"sim={sim:.3f} thresh={thresh} wheels={'ON' if wheels_were_active else 'off'}")
        if sim >= thresh:
            similar_count += 1
        else:
            similar_count = 0
        prev_frame = frame
        stuck = similar_count >= needed and not desk_mode
        if stuck:
            total_stuck_events += 1
            log_event("stuck", f"{similar_count} rounds (wheels={'ON' if wheels_were_active else 'off'}, sim={sim:.3f})")
            plan_history.append(f"Stuck #{total_stuck_events}")

        # Drain injected messages
        injected = []
        while not interrupt_queue.empty():
            try:
                msg = interrupt_queue.get_nowait()
                injected.append(msg)
                # Check if this is negative feedback
                msg_kind = classify_interrupt(msg)
                if msg_kind == "feedback_negative":
                    plan_feedback.append(msg)
                    plan_history.append(f"Negative feedback: {msg}")
            except queue.Empty:
                break

        drive_hint = ""
        if is_turning:
            drive_hint = (
                f"You just spun the body. Center your head (pan→0), "
                f"then observe to confirm the target is ahead before driving. ")
        elif round_num <= 2:
            drive_hint = "Survey first — pan head to find the best route. "
        elif round_num <= 4:
            drive_hint = ("You've scanned enough. Now COMMIT: align body to target, "
                          "center head, confirm target is ahead, then drive at 0.1 m/s. ")
        else:
            drive_hint = ("Keep driving toward the goal at 0.1 m/s. "
                          "Only stop if wall fills >60% of frame. ")

        stuck_hint = ""
        if stuck and wheels_were_active:
            stuck_hint = (
                f"\n\n** PUSHING AGAINST OBSTACLE ({similar_count} rounds, wheels ON but no movement) ** "
                f"You are driving into something and NOT moving. STOP wheels immediately. "
                f"BACK UP: {{\"T\":1,\"L\":-0.15,\"R\":-0.15}}, then turn 90-180 degrees "
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

        # Use actual gimbal position from execute() (single source of truth)
        actual_pan = _gimbal_pan
        gimbal_pan = actual_pan  # sync local tracking

        # Camera sees in direction: body + gimbal_pan
        # Wheels drive along: body direction (pan=0)
        if abs(actual_pan) <= 30:
            look_vs_drive = "Camera and wheels face the SAME direction. You can drive toward what you see."
        else:
            look_vs_drive = (
                f"Camera is pointed {abs(actual_pan):.0f}° {'right' if actual_pan > 0 else 'left'} "
                f"of your body. Wheels will NOT drive toward what you see. "
                f"You MUST spin body to align, center head, and confirm before driving.")

        # Scene change from stuck detection
        if sim < 0.7:
            scene_change = "Scene changed significantly from last frame."
        elif sim < STUCK_SIM_DRIVING:
            scene_change = "Scene changed slightly from last frame."
        else:
            scene_change = "Scene looks almost identical to last frame."

        prompt = (
            f'** YOUR CURRENT TASK: "{text}" **\n\n'
            f"[Observe round {round_num}/{MAX_OBSERVE_ROUNDS}]\n"
            f"Last action: {last_action}\n"
            f"Result: {scene_change}\n"
            f"Gimbal: pan={actual_pan:.0f}°, tilt={gimbal_tilt:.0f}°\n"
            f"{look_vs_drive}\n\n"
            f"{drive_hint}"
            f"This is a NEW camera frame taken AFTER your last commands. "
            f"Evaluate: does it match your task? Is the path clear? "
            f"Only drive if confirmed."
            f"{stuck_hint}{user_context}")

        resp = call_llm(prompt, frame)
        _check_floor(resp)
        _update_landmarks(resp)
        commands = resp.get("commands", [])
        say = resp.get("speak", "")
        observe = resp.get("observe", False)
        remember = resp.get("remember")

        if say:
            _speak(say, spk, mic_card)
            plan_history.append(f"Round {round_num}: {say}")
        if remember:
            save_memory(remember)

    plan_active.clear()
    return {"original_text": text, "feedback": plan_feedback,
            "stuck_count": total_stuck_events, "rounds": round_num,
            "interrupted": was_interrupted, "history": plan_history}

# ── Provider/state helpers for web_ui ──────────────────────────────────

def _get_providers():
    return {
        "current": {"stt": _stt_name, "llm": _llm_name, "tts": _tts_name},
        "available": AVAILABLE_PROVIDERS,
        "desk_mode": desk_mode,
        "stt_enabled": stt_enabled,
    }

def _set_desk_mode(val):
    global desk_mode
    desk_mode = val
    log_event("system", f"Desk mode: {'ON' if desk_mode else 'OFF'}")

def _set_stt_enabled(val):
    global stt_enabled
    stt_enabled = val
    log_event("system", f"STT: {'ON' if stt_enabled else 'OFF'}")

# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    global _ser_ref
    subprocess.run(["pkill", "-f", "rover_brain.py"], capture_output=True)
    time.sleep(1)

    ser = Serial()
    _ser_ref = ser
    cam = Camera()

    # Initialize and start web UI
    web_ui.init(
        camera=cam, serial=ser,
        get_log_events_since=get_log_events_since,
        get_map_state=_get_map_state,
        log_event=log_event,
        set_provider=set_provider,
        get_providers=_get_providers,
        set_desk_mode=_set_desk_mode,
        set_stt_enabled=_set_stt_enabled,
        plan_active=plan_active,
        stop_event=stop_event,
        command_queue=command_queue,
        interrupt_queue=interrupt_queue,
        classify_interrupt=classify_interrupt,
    )
    web_ui.start_server()

    mic_dev, mic_card = audio.find_mic()
    spk = audio.find_speaker()

    log_event("system", f"LLM provider: {llm_mod.NAME}")
    log_event("system", f"STT: {stt_mod.NAME}, TTS: {tts_mod.NAME}")

    def cleanup(*_):
        log_event("system", "Shutdown")
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

    log_event("system", "rover_brain_llm ready")

    # Startup greeting — runs through run_plan for proper scan truncation
    run_plan("You just booted up. Greet briefly and look around with a head movement.", ser, cam, spk, mic_card)

    # Main loop — wait for commands from voice thread or web UI
    while True:
        try:
            text = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Safety check for stop words (voice thread handles these too)
        if _clean_words(text) & STOP_WORDS:
            ser.stop()
            continue

        log_event("plan", f"Starting: {text}")
        result = run_plan(text, ser, cam, spk, mic_card)

        # ── Post-plan reflection ──
        if result is None:
            continue

        trigger = None
        if result["feedback"]:
            trigger = "feedback_negative"
        elif result["stuck_count"] >= 3:
            trigger = "stuck_repeated"
        elif result["interrupted"] and result["rounds"] > 0:
            trigger = "cancelled"

        if trigger:
            log_event("system", f"Reflecting on plan (trigger={trigger})")
            def _do_reflect(req=result["original_text"], trig=trigger,
                            fb=list(result["feedback"]),
                            sc=result["stuck_count"],
                            hist=list(result.get("history", []))):
                reflection.reflect(
                    req, trig, feedback=fb, stuck_count=sc,
                    history=hist, log_fn=log_event)
            t = threading.Thread(target=_do_reflect, daemon=True)
            t.start()

if __name__ == "__main__":
    main()
