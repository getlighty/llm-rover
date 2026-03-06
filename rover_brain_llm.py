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

import room_context
import audio
import imu as imu_mod
import orchestrator
import prompts
# import reflection  # removed — lessons learned via orchestrator.learn_from_feedback
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

# Re-set API keys on modules imported before .env was loaded
orchestrator.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
# "xai" is special — means "use xAI Realtime voice", not provider_xai
_provider_name = sys.argv[1] if len(sys.argv) > 1 else "groq"
_use_xai_realtime = _provider_name == "xai"
if _use_xai_realtime:
    _provider_name = "ollama"  # keep original LLM for text/web commands
provider = importlib.import_module(f"provider_{_provider_name}")

# Decoupled STT / LLM / TTS modules
stt_mod = importlib.import_module("stt_groq")
llm_mod = provider  # reuse provider_*.call_llm()
tts_mod = importlib.import_module("tts_groq")

_stt_name = "groq"
_llm_name = f"{_provider_name}/{provider.LLM_MODEL}"
_tts_name = "groq"

AVAILABLE_PROVIDERS = {
    "stt": ["groq", "xai-realtime"],
    "llm": [
        "ollama/qwen3.5:9b",
        "ollama/qwen3.5:cloud",
        "ollama/qwen3-vl:2b",
        "ollama/ministral-3:14b-cloud",
        "ollama/gemma3:27b-cloud",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        "xai/grok-4-1-fast-reasoning",
        "xai/grok-4-1-fast-non-reasoning",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "tts": ["groq", "elevenlabs"],
    "orch": [
        "qwen3.5:9b",
        "claude-sonnet-4-6",
        "glm-5:cloud",
        "minimax-m2.5:cloud",
        "kimi-k2.5:cloud",
        "qwen3.5:cloud",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
}

# ── xAI Realtime voice state (module-level for hot-swap) ──
_xai_voice = None       # XAIRealtimeVoice instance (or None)
_shared_refs = {}          # filled by main(): ser, cam, spk, mic_dev, etc.


def _follow_recovery(cam, ser, log_fn):
    """LLM-guided obstacle recovery during follow mode."""
    jpeg = snap_with_flash(cam, ser)
    if not jpeg:
        return False
    resp = call_llm("Obstacle blocking path while following target. "
                     "Navigate around with short maneuvers.", jpeg)
    if resp and resp.get("commands"):
        execute(ser, resp["commands"], stop_event, cam)
        return True
    return False


def _follow_llm_fn(prompt, jpeg=None):
    """Flexible LLM call for follow mode — callout, verification, etc."""
    try:
        resp = call_llm(prompt, jpeg)
        text = resp.get("speak", "") if resp else ""
        if text and "unavailable" in text.lower():
            return ""
        return text
    except Exception:
        return ""


def _follow_label_override(yolo_label, correct_label):
    """Persist a YOLO label correction from follow mode."""
    from local_detector import LABEL_OVERRIDES, _save_label_overrides
    LABEL_OVERRIDES[yolo_label] = correct_label
    _save_label_overrides()
    log_event("detect", f"Follow auto-label: '{yolo_label}' -> '{correct_label}'")


def _xai_tool_dispatch(fn_name, args):
    """Route xAI Realtime tool calls to rover hardware."""
    ser = _shared_refs.get("ser")
    cam = _shared_refs.get("cam")
    try:
        if fn_name == "send_rover_commands":
            cmds = []
            for c in args.get("commands", []):
                if isinstance(c, str):
                    try: c = json.loads(c)
                    except (json.JSONDecodeError, TypeError): continue
                if isinstance(c, dict):
                    cmds.append({k.strip('"'): v for k, v in c.items()})
            # Clamp wheel speeds to 0.20 m/s max
            for c in cmds:
                if isinstance(c, dict) and c.get("T") == 1:
                    for k in ("L", "R"):
                        if k in c:
                            c[k] = max(-0.20, min(0.20, float(c[k])))
            # If duration provided, inject _pause + auto-stop after last wheel cmd
            dur = float(args.get("duration", 0))
            if dur > 0:
                # Find last wheel command and insert pause + stop after it
                last_wheel = -1
                for i, c in enumerate(cmds):
                    if isinstance(c, dict) and c.get("T") == 1:
                        l, r = c.get("L", 0), c.get("R", 0)
                        if l != 0 or r != 0:
                            last_wheel = i
                if last_wheel >= 0:
                    cmds.insert(last_wheel + 1, {"_pause": min(dur, 10)})
                    cmds.insert(last_wheel + 2, {"T": 1, "L": 0, "R": 0})
            execute(ser, cmds, stop_event, cam)
            return json.dumps({"status": "ok", "commands_sent": len(cmds)})

        elif fn_name == "look_at_camera":
            pan = args.get("pan", 0)
            tilt = args.get("tilt", 0)
            question = args.get("question", "Describe what you see briefly.")
            ser.send({"T": 133, "X": pan, "Y": tilt, "SPD": 300, "ACC": 20})
            time.sleep(0.8)
            jpeg = cam.snap()
            if not jpeg:
                return json.dumps({"error": "No camera frame"})
            description = llm_mod.call_llm(question, jpeg,
                "Describe what you see in 1-2 sentences.", [])
            try:
                parsed = json.loads(description)
                description = parsed.get("speak", description)
            except (json.JSONDecodeError, TypeError):
                pass
            return json.dumps({"description": str(description)[:500]})

        elif fn_name == "navigate_to":
            target = args.get("target", "")
            if not target:
                return json.dumps({"error": "No target specified"})
            if _navigator:
                reached = _navigator.navigate(target)
                return json.dumps({"found": reached, "reached": reached, "target": target})
            return json.dumps({"error": "Navigator not available"})

        elif fn_name == "search_for":
            target = args.get("target", "")
            if not target:
                return json.dumps({"error": "No target specified"})
            if _navigator:
                direction = _navigator.search(target)
                return json.dumps({"found": direction is not None, "target": target})
            return json.dumps({"error": "Navigator not available"})

        elif fn_name == "remember":
            note = args.get("note", "")
            if note:
                save_memory(note)
                return json.dumps({"status": "remembered", "note": note})
            return json.dumps({"error": "Empty note"})

        elif fn_name == "get_status":
            return json.dumps({"desk_mode": desk_mode, "stt_enabled": stt_enabled})

        elif fn_name == "set_speed":
            log_event("system", "set_speed not implemented in llm mode")
            return json.dumps({"status": "ok", "note": "speed is LLM-controlled"})

        elif fn_name == "track_object":
            from track_object import track
            target = args.get("target", "")
            duration = float(args.get("duration", 10))
            if not target:
                return json.dumps({"error": "No target specified"})
            imu_ref = _shared_refs.get("imu")
            result = track(target, ser, cam, imu_ref, duration=duration,
                           log_fn=lambda msg: log_event("track", msg))
            return json.dumps(result)

        elif fn_name == "follow_person":
            from follow_target import follow
            target = args.get("target", "person")
            duration = float(args.get("duration", 60))
            cam._follow_mode = True
            try:
                result = follow(target, ser, cam, _shared_refs.get("imu"),
                                duration=duration,
                                stop_event=stop_event,
                                log_fn=lambda msg: log_event("follow", msg),
                                voice=_xai_voice, floor_nav=_floor_nav,
                                recovery_fn=_follow_recovery,
                                speak_fn=lambda t: _speak_async(t,
                                    _shared_refs.get("spk"), _shared_refs.get("mic_card")),
                                llm_fn=_follow_llm_fn,
                                label_override_fn=_follow_label_override)
            finally:
                cam._follow_mode = False
            return json.dumps(result)

        elif fn_name == "correct_label":
            from local_detector import LABEL_OVERRIDES, _save_label_overrides
            yolo_label = args.get("yolo_label", "").strip().lower()
            correct_label = args.get("correct_label", "").strip().lower()
            if not yolo_label or not correct_label:
                return json.dumps({"error": "Need yolo_label and correct_label"})
            if yolo_label == correct_label:
                return json.dumps({"status": "no_change", "label": yolo_label})
            LABEL_OVERRIDES[yolo_label] = correct_label
            _save_label_overrides()
            log_event("detect", f"Label override: '{yolo_label}' -> '{correct_label}'")
            return json.dumps({"status": "corrected",
                               "yolo_label": yolo_label,
                               "correct_label": correct_label})

        else:
            return json.dumps({"error": f"Unknown function: {fn_name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _start_xai_realtime():
    """Create and start an XAIRealtimeVoice instance. Returns it, or None on failure."""
    xai_api_key = os.environ.get("XAI_API_KEY", "")
    if not xai_api_key:
        log_event("error", "XAI_API_KEY not set in .env")
        return None
    if not _shared_refs:
        log_event("error", "xAI Realtime: hardware refs not ready (main not started?)")
        return None
    try:
        from xai_realtime import XAIRealtimeVoice
        from tools import to_openai
        from prompts import build_voice_system_prompt
        voice = XAIRealtimeVoice(
            api_key=xai_api_key,
            mic_device=_shared_refs.get("mic_dev") or "plughw:0,0",
            playback_device=_shared_refs.get("spk"),
            instructions=build_voice_system_prompt(gimbal_pan_enabled=gimbal_pan_enabled),
            tools=to_openai(),
            tool_dispatch_fn=_xai_tool_dispatch,
            emergency_event=stop_event,
        )
        voice.start()
        return voice
    except Exception as e:
        log_event("error", f"xAI Realtime failed to start: {e}")
        import traceback; traceback.print_exc()
        return None


_PROVIDER_PREFS_FILE = os.path.join(ROVER_DIR, ".provider_prefs.json")


def _save_provider_prefs():
    """Persist LLM and orchestrator choices to disk."""
    try:
        prefs = {"llm": _llm_name, "orch": orchestrator.OLLAMA_TEXT_MODEL}
        with open(_PROVIDER_PREFS_FILE, "w") as f:
            json.dump(prefs, f)
    except Exception:
        pass


def _restore_provider_prefs():
    """Restore LLM and orchestrator from saved prefs. Call after main init."""
    try:
        with open(_PROVIDER_PREFS_FILE) as f:
            prefs = json.load(f)
        llm = prefs.get("llm")
        orch = prefs.get("orch")
        if llm and llm != _llm_name:
            set_provider("llm", llm)
        if orch and orch != orchestrator.OLLAMA_TEXT_MODEL:
            set_provider("orch", orch)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    except Exception as e:
        log_event("error", f"Provider prefs restore failed: {e}")


def set_provider(kind, name):
    """Hot-swap a provider at runtime. kind = 'stt'|'llm'|'tts'.

    For LLM, name can be 'provider/model' to set both provider and model.
    """
    global stt_mod, llm_mod, tts_mod, _stt_name, _llm_name, _tts_name
    global stt_enabled, _xai_voice
    if kind == "stt":
        if name == "xai-realtime":
            _xai_voice = _start_xai_realtime()
            if _xai_voice:
                stt_enabled = False
                _stt_name = "xai-realtime"
                log_event("system", "xAI Realtime voice (Grok): ACTIVE — STT disabled")
                return
            else:
                log_event("error", "Failed to start xAI Realtime")
                return
        else:
            # Switching away from xai-realtime: stop it, re-enable STT
            if _xai_voice:
                _xai_voice.stop()
                _xai_voice = None
                stt_enabled = True
                log_event("system", "xAI Realtime stopped — STT re-enabled")
            mod = importlib.import_module(f"stt_{name}")
            importlib.reload(mod)
            stt_mod = mod
            _stt_name = name
    elif kind == "llm":
        # Parse "provider/model" format
        if "/" in name:
            parts = name.split("/", 1)
            provider = parts[0]
            model = parts[1]
        else:
            provider = name
            model = None
        mod = importlib.import_module(f"provider_{provider}")
        importlib.reload(mod)
        if model:
            mod.LLM_MODEL = model
        llm_mod = mod
        _llm_name = name
        log_event("system", f"LLM model: {mod.LLM_MODEL}")
        _save_provider_prefs()
    elif kind == "tts":
        mod = importlib.import_module(f"tts_{name}")
        importlib.reload(mod)
        tts_mod = mod
        _tts_name = name
    elif kind == "orch":
        orchestrator.OLLAMA_MODEL = name
        orchestrator.OLLAMA_TEXT_MODEL = name
        log_event("system", f"Orchestrator model: {name}")
        _save_provider_prefs()
    else:
        raise ValueError(f"Unknown provider kind: {kind}")
    log_event("system", f"Switched {kind} to {name}")

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200

MAX_OBSERVE_ROUNDS = 200
STOP_WORDS = {"stop", "halt", "freeze", "emergency"}
desk_mode = False  # When True, all wheel commands (T=1) are blocked
stt_enabled = True  # When False, voice_thread skips STT (mic muted)
gimbal_pan_enabled = True   # When False, gimbal pan (X) clamped to 0, prompt hides pan
tts_enabled = True          # When False, _speak() is silenced

def _check_floor(resp):
    """Log LLM's on_floor assessment but do NOT auto-toggle desk_mode.
    Desk mode is now manual-only (via web UI checkbox)."""
    on_floor = resp.get("on_floor")
    if on_floor is None:
        return
    if not on_floor:
        log_event("system", "LLM thinks elevated — ignoring (desk mode is manual only)")

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
            # Voltage throttle: scale wheel speeds when battery is low
            if _voltage_throttle < 1.0:
                l = cmd.get("L", 0) * _voltage_throttle
                r = cmd.get("R", 0) * _voltage_throttle
                cmd = dict(cmd, L=round(l, 4), R=round(r, 4))
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

    def read_imu(self):
        """Read IMU from the continuous T:1001 feedback stream.
        Reads two lines (skip potentially partial first), returns parsed
        T:1001 dict or None.  Short lock timeout to avoid blocking commands."""
        if not self._lock.acquire(timeout=0.1):
            return None  # busy — skip this read
        try:
            self.ser.reset_input_buffer()
            self.ser.readline()  # skip partial line after buffer clear
            line = self.ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                return None
            data = json.loads(line)
            if data.get("T") == 1001:
                return data
            return None
        except (json.JSONDecodeError, serial.SerialException):
            return None
        finally:
            self._lock.release()

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

        # YOLO detection state (set detector from main() after construction)
        self.detector = None
        self.depth_estimator = None  # set from main() if available
        self._det_results = []     # list of detection dicts
        self._det_persistent = {}  # name → {det, ts} for additive scanning
        self._det_summary = ""     # one-line summary
        self._det_ts = 0.0         # timestamp of last detection
        self._det_jpeg = None      # JPEG with bounding box overlay
        self._det_frame_count = 0  # frame counter for running every 3rd
        self._depth_map = None     # latest depth map (float32 H x W)
        self._DET_PERSIST_S = 0.8  # keep old detections for this long

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log_event("system", "Camera ready (640x480)")

    def get_depth_map(self):
        """Return latest depth map (float32 H x W) or None."""
        with self._lock:
            return self._depth_map

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            raw_jpeg = buf.tobytes()

            # Run YOLO + depth (every frame if following, else every 3rd)
            overlay_jpeg = None
            if self.detector is not None and yolo_enabled:
                self._det_frame_count += 1
                _det_interval = 1 if getattr(self, '_follow_mode', False) else 3
                if self._det_frame_count >= _det_interval:
                    self._det_frame_count = 0
                    try:
                        dets = self.detector.detect(frame)

                        # Run depth estimation and enrich detections
                        depth_map = None
                        if self.depth_estimator is not None:
                            try:
                                depth_map = self.depth_estimator.estimate(frame)
                                self.depth_estimator.enrich_detections(
                                    dets, depth_map)
                            except Exception:
                                pass

                        # Feed into room map (~1/sec)
                        if (hasattr(self, '_room_map') and self._room_map
                                and hasattr(self, '_nav_pose') and self._nav_pose
                                and dets):
                            self._rm_counter = getattr(self, '_rm_counter', 0) + 1
                            if self._rm_counter >= 10:  # every 10th detection = ~1/s
                                self._rm_counter = 0
                                try:
                                    p = self._nav_pose
                                    self._room_map.record(
                                        dets, p.x, p.y, p.body_yaw,
                                        p.cam_pan, p.cam_tilt)
                                except Exception:
                                    pass

                        # Additive scanning: merge with persistent detections
                        now = time.time()
                        current_names = {d["name"] for d in dets}
                        for d in dets:
                            self._det_persistent[d["name"]] = {
                                "det": d, "ts": now}
                        # Add old detections not seen this frame (if fresh)
                        for name, entry in list(self._det_persistent.items()):
                            if (name not in current_names
                                    and now - entry["ts"] < self._DET_PERSIST_S):
                                dets.append(entry["det"])
                            elif now - entry["ts"] >= self._DET_PERSIST_S * 2:
                                del self._det_persistent[name]

                        summary = self.detector.summary(dets)
                        # Draw overlay on a copy
                        overlay = frame.copy()
                        self.detector.draw(overlay, dets)

                        # Draw depth minimap in bottom-right corner
                        if depth_map is not None:
                            try:
                                dv = self.depth_estimator.colorize(depth_map)
                                mh, mw = 96, 128
                                mini = cv2.resize(dv, (mw, mh))
                                oh, ow = overlay.shape[:2]
                                overlay[oh-mh:oh, ow-mw:ow] = mini
                            except Exception:
                                pass

                        _, obuf = cv2.imencode(".jpg", overlay,
                                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                        overlay_jpeg = obuf.tobytes()
                        with self._lock:
                            self._det_results = dets
                            self._det_summary = summary
                            self._det_ts = time.time()
                            self._det_jpeg = overlay_jpeg
                            self._depth_map = depth_map
                    except Exception:
                        pass  # don't crash capture loop on detection error

            with self._lock:
                self._jpeg = raw_jpeg
            time.sleep(0.03)  # ~30 fps

    def get_detections(self):
        """Return (det_list, summary_str, age_secs).
        Suppresses all detections when gimbal is tilted down (seeing own body)."""
        if _gimbal_tilt < _TILT_YOLO_SUPPRESS:
            return ([], "", 999.0)
        with self._lock:
            return (list(self._det_results), self._det_summary,
                    time.time() - self._det_ts if self._det_ts else 999.0)

    def get_overlay_jpeg(self):
        """Return JPEG with bounding boxes, or raw if YOLO is off."""
        with self._lock:
            if yolo_enabled and self._det_jpeg is not None:
                return self._det_jpeg
            return self._jpeg

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
    system = prompts.build_system_prompt(gimbal_pan_enabled=gimbal_pan_enabled)

    history.append({"role": "user", "content": text})
    if len(history) > 10:
        history = history[-10:]

    reply = None
    for attempt in range(3):
        try:
            reply = llm_mod.call_llm(text, jpeg_bytes, system, history)
            break
        except Exception as e:
            log_event("error", f"LLM error (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2)
    if reply is None:
        reply = json.dumps({"commands": [], "speak": "LLM unavailable."})

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
    # Last resort: ask the LLM to fix its own non-JSON response
    log_event("error", f"JSON parse failed, retrying extraction: {clean[:120]}")
    try:
        fix_prompt = (
            "Your previous response was not valid JSON. Convert it to the required format. "
            "Reply with ONLY a JSON object like: "
            '{"commands":[],"speak":"<short>","observe":false}\n\n'
            f"Your response was:\n{reply[:500]}"
        )
        fix_reply = llm_mod.call_llm(fix_prompt, jpeg_bytes,
            "You MUST reply with ONLY a raw JSON object. Nothing else. No explanation.", [])
        fix_clean = fix_reply.strip()
        if fix_clean.startswith("```"):
            fix_clean = fix_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        fs = fix_clean.find("{")
        fe = fix_clean.rfind("}")
        if fs >= 0 and fe > fs:
            result = json.loads(fix_clean[fs:fe + 1])
            log_event("system", "JSON recovered via retry")
            return result
    except Exception:
        pass
    log_event("error", f"JSON parse failed: {clean[:200]}")
    return {"commands": [], "speak": "Hmm."}


# ── Command Execution ───────────────────────────────────────────────────

def _interruptible_sleep(secs, stop_ev=None):
    """Sleep in small increments, checking stop_event and IMU tilt."""
    elapsed = 0.0
    while elapsed < secs:
        if stop_ev and stop_ev.is_set():
            return True  # interrupted
        # Tilt safety: emergency stop if rover is tipping
        if _imu:
            level, deg = _imu.check_tilt()
            if level == "stop":
                log_event("imu", f"IMPACT/TILT: accel={deg:.2f} — stopping motors")
                if _ser_ref:
                    _ser_ref.stop()
                return True  # interrupted
            elif level == "warn":
                log_event("imu", f"Accel spike: {deg:.2f}")
        time.sleep(min(0.1, secs - elapsed))
        elapsed += 0.1
    return False

DRIVE_PAN_LIMIT = 30  # max |pan| degrees to allow forward/backward driving
WHEEL_SEP = 0.20      # meters between left/right wheels

def _trajectory_hits_obstacle(l, r, duration, cam):
    """Check if a wheel command's trajectory intersects any YOLO detection.

    Projects the rover's path based on differential drive kinematics and
    checks if it passes through any detected object's bounding box in
    image space.

    Camera FOV ~65°. Objects in the image map to angular positions:
      angle_from_center = (cx - 0.5) * FOV_deg
    The rover's trajectory curves based on (R-L)/WHEEL_SEP angular rate.

    Returns (hit, description) — hit is True if collision predicted.
    """
    if cam is None:
        return False, ""
    dets, _, age = cam.get_detections()
    if age > 2.0 or not dets:
        return False, ""

    FOV_DEG = 65.0
    WHEEL_SEP = 0.20
    v = (l + r) / 2.0           # forward speed m/s
    omega = (r - l) / WHEEL_SEP  # yaw rate rad/s (+ = turning left in image)

    if abs(v) < 0.02:
        return False, ""  # not really moving forward

    travel_m = abs(v) * duration
    if travel_m < 0.05:
        return False, ""  # trivial movement

    # Project heading change over the duration
    heading_change_deg = math.degrees(omega * duration)

    # The rover will sweep from current heading (center) by heading_change_deg
    # In image space, center = 0.5. A point at cx maps to
    #   angle = (cx - 0.5) * FOV_DEG  (degrees from center)
    # The rover's trajectory sweeps through angle 0 → heading_change_deg
    # An object is hit if its angular extent overlaps the sweep corridor

    # Half-width of the rover body in angular space at object distance
    BODY_HALF_W = 0.13  # 26cm body / 2

    for d in dets:
        if d["name"] in _IGNORE_OBSTACLE:
            continue
        if d["conf"] < 0.25:
            continue
        # Only check objects in the lower portion of the frame (closer)
        if d["cy"] < 0.35:
            continue  # too far away to matter this command

        # Object angular position relative to camera center
        obj_angle = (d["cx"] - 0.5) * FOV_DEG

        # Object angular half-width
        obj_half_w = (d["bw"] / 2.0) * FOV_DEG

        # Estimate distance to object (use dist_m if available, else from cy)
        if "dist_m" in d:
            obj_dist = d["dist_m"]
        else:
            # Rough estimate: cy=1.0 → ~0.3m, cy=0.5 → ~1.5m
            obj_dist = max(0.2, 2.0 * (1.0 - d["cy"]))

        # Would we reach the object?
        if travel_m < obj_dist * 0.5:
            continue  # won't travel far enough to reach it

        # Rover body angular half-width at object distance
        body_angle = math.degrees(math.atan2(BODY_HALF_W, max(obj_dist, 0.2)))

        # Trajectory sweep corridor: from 0 to heading_change_deg
        sweep_min = min(0, heading_change_deg) - body_angle
        sweep_max = max(0, heading_change_deg) + body_angle

        # Object angular extent
        obj_min = obj_angle - obj_half_w
        obj_max = obj_angle + obj_half_w

        # Check overlap
        if sweep_max >= obj_min and sweep_min <= obj_max:
            dist_str = f"{obj_dist:.1f}m" if "dist_m" in d else f"~{obj_dist:.1f}m"
            desc = f"{d['name']}({d['conf']:.0%},{dist_str})"
            return True, desc

    return False, ""


def _backup_like_car(ser, cam, speed=0.15, duration=1.0, stop_ev=None):
    """Back up like a car: look behind, check clear, reverse, look forward.

    1. Stop wheels
    2. Pan gimbal to 180° (look behind)
    3. Grab frame, run YOLO — check for obstacles behind
    4. If clear: back up at `speed` for `duration` seconds
       If blocked: abort, log the obstacle, return False
    5. Stop wheels
    6. Pan gimbal back to 0° (look forward)

    Returns True if backup was executed, False if blocked.
    """
    global _gimbal_pan

    log_event("backup", "Starting car-style backup — looking behind")

    # 1. Stop wheels
    ser.send({"T": 1, "L": 0, "R": 0})

    # 2. Look behind
    ser.send({"T": 133, "X": 180, "Y": -10, "SPD": 400, "ACC": 20})
    _gimbal_pan = 180
    time.sleep(0.6)  # wait for gimbal to arrive

    if stop_ev and stop_ev.is_set():
        ser.stop()
        return False

    # 3. Check for obstacles behind using YOLO
    rear_clear = True
    obstacle_desc = ""
    if cam is not None:
        # Let camera capture a fresh frame with the rear view
        time.sleep(0.3)
        dets, summary, age = cam.get_detections()
        if age < 2.0 and dets:
            for d in dets:
                if d["name"] in _IGNORE_OBSTACLE:
                    continue
                if d["conf"] < 0.45:
                    continue  # rear view is noisy — need high confidence
                # In the rear view, "close" objects are still at bottom of frame
                # Any large close object = can't back up
                if d["cy"] > 0.50 and d["bh"] > 0.25:
                    rear_clear = False
                    dist = f"{d['dist_m']:.1f}m" if "dist_m" in d else "close"
                    obstacle_desc = f"{d['name']}({d['conf']:.0%},{dist})"
                    break

    if not rear_clear:
        log_event("backup", f"Blocked — obstacle behind: {obstacle_desc}")
        # Look forward again
        ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 400, "ACC": 20})
        _gimbal_pan = 0
        time.sleep(0.5)
        return False

    log_event("backup", f"Clear behind — reversing at {speed} m/s for {duration}s")

    # 4. Back up (keep looking behind to monitor)
    ser.send({"T": 1, "L": -speed, "R": -speed})

    # Drive in small increments so we can check for new obstacles
    elapsed = 0.0
    step = 0.3  # check every 0.3s
    while elapsed < duration:
        chunk = min(step, duration - elapsed)
        if _interruptible_sleep(chunk, stop_ev):
            ser.stop()
            ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 400, "ACC": 20})
            _gimbal_pan = 0
            return False
        elapsed += chunk

        # Re-check YOLO while reversing
        if cam is not None:
            dets, _, age = cam.get_detections()
            if age < 1.5 and dets:
                for d in dets:
                    if d["name"] in _IGNORE_OBSTACLE:
                        continue
                    if d["conf"] < 0.50:
                        continue  # only emergency-stop on solid detections
                    # Something appeared very close behind — emergency stop
                    if d["cy"] > 0.70 and d["bh"] > 0.35:
                        log_event("backup",
                            f"Emergency stop — {d['name']} appeared close behind")
                        ser.send({"T": 1, "L": 0, "R": 0})
                        ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 400, "ACC": 20})
                        _gimbal_pan = 0
                        time.sleep(0.5)
                        return True  # partial backup is still a backup

    # 5. Stop wheels
    ser.send({"T": 1, "L": 0, "R": 0})

    # 6. Look forward again
    log_event("backup", "Done — looking forward")
    ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 400, "ACC": 20})
    _gimbal_pan = 0
    time.sleep(0.5)

    return True


def _floor_nav_sleep(ser, secs, stop_ev, cam):
    """Sleep during forward driving, checking floor obstacles every 100ms.

    If an obstacle is detected in the floor zone, stops wheels immediately
    and returns the remaining time.  Returns 0.0 if the full sleep completed.
    """
    if _floor_nav is None or cam is None:
        # No floor nav — fall back to regular interruptible sleep
        _interruptible_sleep(secs, stop_ev)
        return 0.0

    elapsed = 0.0
    while elapsed < secs:
        if stop_ev and stop_ev.is_set():
            ser.send({"T": 1, "L": 0, "R": 0})
            return secs - elapsed
        # Tilt safety (same as _interruptible_sleep)
        if _imu:
            level, deg = _imu.check_tilt()
            if level == "stop":
                log_event("imu", f"IMPACT/TILT: accel={deg:.2f} — stopping motors")
                ser.send({"T": 1, "L": 0, "R": 0})
                return secs - elapsed

        # Check floor obstacles using cached YOLO detections
        dets, _, age = cam.get_detections()
        if age < 1.0 and dets:
            clear, alt_col = _floor_nav.check_floor_clear(
                dets, 640, 480)
            if not clear:
                ser.send({"T": 1, "L": 0, "R": 0})
                log_event("floor_nav",
                    f"Floor blocked — stopped ({secs - elapsed:.1f}s remaining)")
                return secs - elapsed

        time.sleep(0.1)
        elapsed += 0.1
    return 0.0


def execute(ser, commands, stop_ev=None, cam=None):
    global _gimbal_pan, _gimbal_tilt
    last_pan = _gimbal_pan  # start from actual current gimbal position
    last_tilt = _gimbal_tilt
    pending_spin = None     # (L, R) of a spin command awaiting its _pause
    skip_next_pause = False  # set after car-backup to skip the associated _pause
    driving_forward = False  # True when wheels are driving forward (not spin)

    for cmd in commands:
        if stop_ev and stop_ev.is_set():
            ser.stop()
            return
        # Handle _pause without holding serial lock
        if "_pause" in cmd:
            if skip_next_pause:
                skip_next_pause = False
                continue
            secs = float(cmd["_pause"])
            if secs > 0:
                log_event("serial", f"_pause {secs:.2f}s")
                if driving_forward and cam is not None:
                    # Check for obstacles every 200ms while driving forward
                    _drove = 0.0
                    _hit = False
                    while _drove < secs:
                        if stop_ev and stop_ev.is_set():
                            ser.stop()
                            return
                        time.sleep(0.2)
                        _drove += 0.2
                        # YOLO obstacle check
                        _blocking, _bdesc = _check_path_obstacle(cam)
                        if _blocking:
                            ser.send({"T": 1, "L": 0, "R": 0})
                            log_event("system",
                                f"Obstacle during drive: {_bdesc} — stopped"
                                f" ({secs - _drove:.1f}s remaining)")
                            _hit = True
                            break
                    if _hit:
                        driving_forward = False
                        pending_spin = None
                        continue
                else:
                    if _interruptible_sleep(secs, stop_ev):
                        ser.stop()
                        return
                # Auto-compensate gimbal after body spin to keep camera steady
                if pending_spin is not None and gimbal_pan_enabled:
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
            driving_forward = False
            continue
        # Safety: block forward/backward driving when head is turned sideways
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            l, r = cmd.get("L", 0), cmd.get("R", 0)
            if l != 0 or r != 0:
                is_spin = (l != 0 and r != 0 and (l > 0) != (r > 0))
                is_backward = (l < 0 and r < 0)
                if is_spin:
                    pending_spin = (l, r)
                    driving_forward = False
                elif is_backward and not _backup_allowed:
                    log_event("system",
                        "Blocked backup: not stuck. Spin instead.")
                    continue
                elif is_backward and _backup_allowed:
                    # Car-style backup: look behind, check, reverse
                    driving_forward = False
                    idx = commands.index(cmd)
                    dt = 0.8  # default backup duration
                    for j in range(idx + 1, len(commands)):
                        c2 = commands[j]
                        if isinstance(c2, dict) and "_pause" in c2:
                            dt = float(c2["_pause"])
                            break
                        if isinstance(c2, dict) and c2.get("T") == 1:
                            break
                    speed = min(abs(l), abs(r), 0.20)  # cap at 0.20 m/s
                    _backup_like_car(ser, cam, speed=speed,
                                     duration=dt, stop_ev=stop_ev)
                    skip_next_pause = True  # consume the associated _pause
                    continue
                elif abs(last_pan) > DRIVE_PAN_LIMIT:
                    log_event("system",
                        f"Blocked drive: head at pan={last_pan:.0f}° "
                        f"(limit ±{DRIVE_PAN_LIMIT}°). Align head first.")
                    continue
                # Safety: block driving into YOLO-detected obstacle
                elif not is_backward and cam is not None:
                    # Find the pause duration that follows this wheel command
                    idx = commands.index(cmd)
                    dt = 0.5  # default
                    for j in range(idx + 1, len(commands)):
                        c2 = commands[j]
                        if isinstance(c2, dict) and "_pause" in c2:
                            dt = float(c2["_pause"])
                            break
                        if isinstance(c2, dict) and c2.get("T") == 1:
                            break  # next wheel cmd, no pause found
                    # Floor nav check disabled — too aggressive in cluttered rooms.
                    # The LLM has its own vision-based obstacle avoidance.
                    # Trajectory check (legacy fallback)
                    hit, desc = _trajectory_hits_obstacle(l, r, dt, cam)
                    if hit:
                        log_event("system",
                            f"Blocked drive: trajectory hits {desc}. "
                            f"Spin to avoid.")
                        ser.stop()
                        continue
                    # Mark that we're driving forward (for floor-nav-aware pause)
                    driving_forward = True
            else:
                driving_forward = False
        # Clamp gimbal pan to 0 when pan is disabled
        if not gimbal_pan_enabled and isinstance(cmd, dict) and cmd.get("T") == 133:
            cmd["X"] = 0
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
            _gimbal_tilt = last_tilt
        elif t == 1:
            # Non-spin wheel command clears pending_spin
            l, r = cmd.get("L", 0), cmd.get("R", 0)
            if not (l != 0 and r != 0 and (l > 0) != (r > 0)):
                pending_spin = None
            # IMU: signal wheel activity for stuck detection
            if _imu:
                if l != 0 or r != 0:
                    _imu.wheels_active.set()
                    _imu.reset_stationary()
                else:
                    _imu.wheels_active.clear()

    # End of execute — wheels are done
    if _imu:
        _imu.wheels_active.clear()

# ── Bash Execution ─────────────────────────────────────────────────────

_BASH_BLOCKED = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/sd",
    "shutdown", "reboot", "poweroff", "init 0", "init 6",
    ":(){ :", "fork bomb",
    "chmod -R 777 /", "chown -R",
    "> /etc/", "mv /etc/", "rm /etc/",
    "systemctl disable", "systemctl mask",
    "iptables -F", "ufw disable",
    "passwd", "userdel", "deluser",
    "kill -9 1", "kill -9 -1",
    "curl | sh", "curl | bash", "wget | sh", "wget | bash",
]

_BASH_TIMEOUT = 30  # seconds
_BASH_MAX_OUTPUT = 2000  # characters

# Holds output from the last bash command for injection into the next LLM prompt
_bash_output = ""


def _run_bash(cmd):
    """Run a bash command as user jasper with safety checks.

    Returns (success, output_string).
    """
    global _bash_output

    if not cmd or not isinstance(cmd, str):
        return False, "Empty command"

    cmd = cmd.strip()
    lower = cmd.lower()

    # Block dangerous commands
    for blocked in _BASH_BLOCKED:
        if blocked in lower:
            msg = f"Blocked: contains '{blocked}'"
            log_event("bash", msg)
            _bash_output = msg
            return False, msg

    log_event("bash", f"$ {cmd}")

    try:
        # Run as jasper user (service runs as root)
        result = subprocess.run(
            ["su", "-", "jasper", "-c", cmd],
            capture_output=True, text=True,
            timeout=_BASH_TIMEOUT,
            cwd="/home/jasper",
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout + stderr
        # Truncate long output
        if len(output) > _BASH_MAX_OUTPUT:
            output = output[:_BASH_MAX_OUTPUT] + f"\n... (truncated, {len(stdout)+len(stderr)} chars total)"
        rc = result.returncode
        status = f"exit={rc}"
        if rc != 0 and not output:
            output = f"(exited with code {rc})"
        log_event("bash", f"{status}: {output[:200]}")
        _bash_output = output
        return rc == 0, output

    except subprocess.TimeoutExpired:
        msg = f"Timeout ({_BASH_TIMEOUT}s)"
        log_event("bash", msg)
        _bash_output = msg
        return False, msg
    except Exception as e:
        msg = f"Error: {e}"
        log_event("bash", msg)
        _bash_output = msg
        return False, msg


# ── File Tools (LLM self-access) ─────────────────────────────────────

_FILE_ROOT = ROVER_DIR  # /home/jasper/rover-control/
_FILE_ALLOWED_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini",
    ".env", ".sh", ".log", ".csv", ".toml",
}

def _resolve_file_path(path):
    """Resolve and validate a file path. Returns (abs_path, error_msg)."""
    if not path:
        return None, "Empty path"
    # Allow relative paths (relative to ROVER_DIR)
    if not os.path.isabs(path):
        path = os.path.join(_FILE_ROOT, path)
    path = os.path.realpath(path)
    # Must be under /home/jasper/
    if not path.startswith("/home/jasper/"):
        return None, f"Access denied: {path} (must be under /home/jasper/)"
    return path, None


def _file_read(path, offset=None, limit=None):
    """Read a file and return its contents. Supports offset/limit in lines."""
    global _bash_output
    abs_path, err = _resolve_file_path(path)
    if err:
        _bash_output = err
        log_event("file", f"read DENIED: {err}")
        return
    if not os.path.isfile(abs_path):
        _bash_output = f"File not found: {path}"
        log_event("file", f"read 404: {path}")
        return
    try:
        with open(abs_path, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, (offset or 1) - 1)
        end = start + (limit or 200)
        selected = lines[start:end]
        numbered = "".join(
            f"{start + i + 1:4d}  {line}" for i, line in enumerate(selected))
        if len(numbered) > 8000:
            numbered = numbered[:8000] + "\n... (truncated)"
        header = f"── {os.path.relpath(abs_path, '/home/jasper')} ({total} lines) ──\n"
        _bash_output = header + numbered
        log_event("file", f"read: {path} ({total} lines, showing {start+1}-{min(end, total)})")
    except Exception as e:
        _bash_output = f"Error reading {path}: {e}"
        log_event("file", f"read error: {e}")


def _file_write(path, content, append=False):
    """Write content to a file. Can create new files or overwrite/append."""
    global _bash_output
    abs_path, err = _resolve_file_path(path)
    if err:
        _bash_output = err
        log_event("file", f"write DENIED: {err}")
        return
    # Block writing to critical system files
    basename = os.path.basename(abs_path)
    if basename in (".bashrc", ".profile", ".bash_logout"):
        _bash_output = f"Write denied: {basename} is a protected file"
        log_event("file", f"write DENIED: {basename}")
        return
    ext = os.path.splitext(abs_path)[1].lower()
    if ext and ext not in _FILE_ALLOWED_EXTENSIONS:
        _bash_output = f"Write denied: extension '{ext}' not allowed"
        log_event("file", f"write DENIED: ext {ext}")
        return
    try:
        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, mode) as f:
            f.write(content)
        _bash_output = f"Written {len(content)} chars to {path}" + (" (appended)" if append else "")
        log_event("file", f"write: {path} ({len(content)} chars, {'append' if append else 'overwrite'})")
    except Exception as e:
        _bash_output = f"Error writing {path}: {e}"
        log_event("file", f"write error: {e}")


def _file_list(path=None, pattern=None):
    """List files in a directory, optionally filtered by glob pattern."""
    global _bash_output
    dir_path = path or _FILE_ROOT
    abs_path, err = _resolve_file_path(dir_path)
    if err:
        _bash_output = err
        log_event("file", f"list DENIED: {err}")
        return
    if not os.path.isdir(abs_path):
        _bash_output = f"Not a directory: {dir_path}"
        log_event("file", f"list 404: {dir_path}")
        return
    try:
        import glob as _glob
        if pattern:
            matches = _glob.glob(os.path.join(abs_path, pattern))
            entries = sorted(os.path.relpath(m, abs_path) for m in matches)
        else:
            entries = sorted(os.listdir(abs_path))
        lines = []
        for name in entries[:100]:
            full = os.path.join(abs_path, name)
            if os.path.isdir(full):
                lines.append(f"  {name}/")
            else:
                size = os.path.getsize(full)
                if size > 1024 * 1024:
                    sz = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    sz = f"{size / 1024:.1f}KB"
                else:
                    sz = f"{size}B"
                lines.append(f"  {name}  ({sz})")
        rel = os.path.relpath(abs_path, "/home/jasper")
        header = f"── {rel}/ ({len(entries)} items) ──\n"
        _bash_output = header + "\n".join(lines)
        log_event("file", f"list: {dir_path} ({len(entries)} items)")
    except Exception as e:
        _bash_output = f"Error listing {dir_path}: {e}"
        log_event("file", f"list error: {e}")


def _file_grep(pattern, path=None, max_results=30):
    """Search for a pattern in files under a directory."""
    global _bash_output
    dir_path = path or _FILE_ROOT
    abs_path, err = _resolve_file_path(dir_path)
    if err:
        _bash_output = err
        return
    try:
        import re
        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        for root, dirs, files in os.walk(abs_path):
            # Skip hidden dirs and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _FILE_ALLOWED_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, abs_path)
                                results.append(f"  {rel}:{i}: {line.rstrip()[:120]}")
                                if len(results) >= max_results:
                                    break
                except Exception:
                    pass
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        if results:
            _bash_output = f"── grep '{pattern}' ({len(results)} matches) ──\n" + "\n".join(results)
        else:
            _bash_output = f"No matches for '{pattern}'"
        log_event("file", f"grep: '{pattern}' → {len(results)} matches")
    except Exception as e:
        _bash_output = f"Error: {e}"
        log_event("file", f"grep error: {e}")


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
STUCK_ROUNDS_DRIVING = 3   # need consistent evidence before declaring stuck
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

# Last meaningful task for "retry" support
_last_task = ""  # stores the last non-trivial task for retry/continuation
_last_task_target = ""  # extracted target object for navigation tasks

# Will be set in main() so voice_thread can call ser.stop() directly
_ser_ref = None
_imu = None  # IMUPoller, set in main()

# ── Voltage Watchdog ─────────────────────────────────────────────────
_LOW_VOLTAGE_THRESHOLD = 9.0   # volts
_LOW_VOLTAGE_THROTTLE = 0.20   # 20% of nominal speed
_CHARGE_RISE_THRESHOLD = 0.5   # V rise in 1-min average = charging
_CHARGE_ABS_THRESHOLD = 12.3   # V above this while rising = definitely charging
_voltage_throttle = 1.0        # current multiplier (1.0 = full, 0.2 = throttled)
_voltage_warned = False        # only warn once per low episode
_charging = False              # True when charger detected


def _voltage_watchdog():
    """Background thread: monitor battery voltage, throttle when low,
    detect charging and auto-enable desk mode."""
    global _voltage_throttle, _voltage_warned, _charging, desk_mode
    from collections import deque
    # Rolling buffer: one sample every 2s, 30 samples = 1 minute
    history = deque(maxlen=30)
    prev_avg = None

    while True:
        time.sleep(2.0)
        if _imu is None or not _imu.state.fresh:
            continue
        v = _imu.state.voltage
        if v <= 0:
            continue  # no reading yet

        # ── Rolling average ──
        history.append(v)
        avg = sum(history) / len(history)

        # ── Charging detection ──
        # Need at least 20s of history (10 samples) before deciding
        if len(history) >= 10 and prev_avg is not None:
            rise = avg - prev_avg
            # Charging: average rose significantly, or voltage is high and rising
            if (rise > _CHARGE_RISE_THRESHOLD
                    or (avg > _CHARGE_ABS_THRESHOLD and rise > 0.1)):
                if not _charging:
                    _charging = True
                    desk_mode = True
                    log_event("battery",
                              f"CHARGING detected (avg={avg:.2f}V, "
                              f"rise={rise:+.2f}V/min) — desk mode ON")
            # Discharging: average dropped or stable below charge threshold
            elif rise < -0.1 and _charging:
                _charging = False
                # Don't auto-disable desk mode — user may want it on
                log_event("battery",
                          f"UNPLUGGED (avg={avg:.2f}V, "
                          f"rise={rise:+.2f}V/min) — charging=false")
        prev_avg = avg

        # ── Low voltage throttle ──
        if v < _LOW_VOLTAGE_THRESHOLD:
            if _voltage_throttle != _LOW_VOLTAGE_THROTTLE:
                _voltage_throttle = _LOW_VOLTAGE_THROTTLE
                log_event("battery",
                          f"LOW VOLTAGE {v:.1f}V < {_LOW_VOLTAGE_THRESHOLD}V "
                          f"— throttling to {int(_LOW_VOLTAGE_THROTTLE * 100)}%")
            if not _voltage_warned:
                _voltage_warned = True
        else:
            if _voltage_throttle != 1.0:
                _voltage_throttle = 1.0
                _voltage_warned = False
                log_event("battery",
                          f"Voltage OK {v:.1f}V — full speed restored")
_floor_nav = None  # FloorNavigator, set in main()
_navigator = None  # Navigator, set in main()
_room_scanner = None  # VectorRoomScanner, set in main()

# ── 2D World Map ──────────────────────────────────────────────────────

_map_lock = threading.Lock()
_landmarks = {}       # name → {"name","x","y","type","ts"}
_rover_x = 0.0        # world position meters
_rover_y = 0.0
_rover_heading = 0.0  # radians, 0 = +Y (forward at boot)
_gimbal_pan = 0.0     # degrees
_gimbal_tilt = 0.0    # degrees — updated by execute()
_TILT_YOLO_SUPPRESS = 0  # tilt below 0° = looking at ground/own body, suppress YOLO
_backup_allowed = False  # only True during stuck recovery
LANDMARK_MAX_AGE = 300  # 5 minutes
_room_scan_state = {
    "scan_ts": 0.0,
    "task": "",
    "elements": [],
    "scene_summary": "",
    "room_guess": {"name": None, "confidence": 0.0, "reason": ""},
    "candidates": [],
}

def _dist_to_meters(d):
    """Convert dist field to meters. Accepts number or legacy string."""
    if isinstance(d, (int, float)):
        return float(d)
    return {"near": 0.4, "mid": 1.2, "far": 2.5}.get(str(d), 1.2)

def _update_pose_from_commands(commands):
    """Dead-reckon rover position from wheel commands.
    Uses IMU magnetometer heading when available, falls back to kinematics."""
    global _rover_x, _rover_y, _rover_heading
    WHEEL_SEP = 0.20  # meters between left/right wheels

    # If IMU has fresh data, use magnetometer heading directly
    if _imu and _imu.state.fresh:
        _rover_heading = math.radians(_imu.heading_deg)

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
            # Heading: prefer IMU if fresh, else dead-reckon
            if not (_imu and _imu.state.fresh):
                omega = (R - L) / WHEEL_SEP
                _rover_heading += omega * dt
            # X/Y always dead-reckoned (magnetometer can't help)
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
        rover_data = {"x": _rover_x, "y": _rover_y,
                      "heading": math.degrees(_rover_heading),
                      "gimbal_pan": _gimbal_pan}
        if _imu:
            imu_data = _imu.get_map_data()
            if imu_data:
                rover_data["imu"] = imu_data
        room_scan = dict(_room_scan_state)
        return {
            "landmarks": lm,
            "rover": rover_data,
            "room_scan": room_scan,
        }


def _run_task_room_scan(task_text, find_target=None):
    """Run an LLM vector room scan and persist latest state for UI + routing.

    Args:
        find_target: if set, stop sweep early when this object is spotted.
    """
    global _room_scan_state
    if not _room_scanner:
        return None
    heading_deg = math.degrees(_rover_heading)
    state = _room_scanner.scan_room(task_text=task_text,
                                     body_yaw_deg=heading_deg,
                                     find_target=find_target)
    with _map_lock:
        _room_scan_state = dict(state)
    return state

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
    """Speak with TTS via audio module (blocking)."""
    if not tts_enabled:
        return
    audio.speak(text, tts_mod, spk, mic_card, log_fn=log_event)

def _speak_async(text, spk, mic_card):
    """Fire-and-forget TTS in background thread."""
    if not text:
        return
    threading.Thread(target=_speak, args=(text, spk, mic_card), daemon=True).start()

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

        # ── Wake word: "Jasper" required ──
        # Stop/cancel always work for safety. Everything else needs wake word.
        lower = text.lower().strip()
        words_set = set(lower.split())
        is_stop = bool(words_set & STOP_WORDS)
        is_cancel = bool(words_set & CANCEL_WORDS)
        has_wake = lower.startswith("jasper")

        if not has_wake and not is_stop and not is_cancel:
            log_event("heard", f"(no wake word) {text}")
            continue

        # Strip "Jasper" prefix from the command text
        if has_wake:
            text = text.lstrip()
            # Remove "Jasper" / "Jasper," / "Jasper " from the start
            for prefix in ("jasper,", "jasper ", "jasper."):
                if lower.startswith(prefix):
                    text = text[len(prefix):].strip()
                    break
            else:
                if lower == "jasper":
                    text = ""  # just said the name, nothing else
            if not text:
                log_event("heard", "Jasper (wake word only)")
                continue

        log_event("heard", text)

        if plan_active.is_set():
            kind = classify_interrupt(text)

            # Check if "override" is actually the same task (user repeating)
            if kind == "override" and _last_task:
                try:
                    import requests as _req
                    r = _req.post("http://192.168.0.126:11434/api/chat",
                        json={"model": "qwen3.5:9b",
                              "messages": [
                                  {"role": "system",
                                   "content": "Reply ONLY yes or no."},
                                  {"role": "user",
                                   "content": f'Is "{text}" asking for the same '
                                              f'thing as "{_last_task}"?'}],
                              "stream": False, "think": False,
                              "options": {"num_predict": 5}},
                        timeout=3)
                    answer = r.json()["message"]["content"].strip().lower()
                    if answer.startswith("yes"):
                        log_event("interrupt",
                            f"same_task: '{text}' ≈ '{_last_task}'")
                        continue
                except Exception:
                    pass  # LLM timeout/error → treat as normal override

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

# Objects that are not obstacles (background / surface classes)
# NOTE: "wall" is NOT ignored — a wall ahead is a real obstacle.
_IGNORE_OBSTACLE = {
    "floor", "ceiling", "rug", "window", "light", "lamp",
    "lamp/light fixture", "light_fixture", "light fixture",
    "wooden floor/deck", "pegboard", "tool pegboard",
}

def _check_path_obstacle(cam):
    """Check YOLO detections for an object blocking the driving corridor.

    Returns (blocking, description) — blocking is True if a large, close
    object sits in the center of the frame (the driving corridor).

    Thresholds are conservative to avoid false stuck triggers — e.g. door
    frames misidentified as 'couch' at 45% confidence should NOT trigger.
    """
    dets, summary, age = cam.get_detections()
    if age > 2.0 or not dets:
        return False, ""

    # Classes often misidentified near doorways / at frame edges
    _DOORWAY_FALSE_POSITIVES = {"couch", "bed", "bench", "dining table"}

    for d in dets:
        if d["name"] in _IGNORE_OBSTACLE:
            continue
        if d["conf"] < 0.50:
            continue  # must be fairly confident to trigger instant stuck

        # Doorway false positives need even higher confidence
        if d["name"] in _DOORWAY_FALSE_POSITIVES and d["conf"] < 0.65:
            continue

        # Driving corridor: center 60% of frame width
        if not (0.20 <= d["cx"] <= 0.80):
            continue

        # Truly blocking: bottom of frame + large enough to fill path
        if d["cy"] > 0.55 and d["bh"] > 0.30 and d["bw"] > 0.20:
            dist = f"{d['dist_m']:.1f}m" if "dist_m" in d else "close"
            desc = f"{d['name']}({d['conf']:.0%},{dist})"
            return True, desc
    return False, ""


def _yolo_fingerprint(cam):
    """Build a compact fingerprint of current YOLO detections.

    Returns a list of (name, cx_bin, cy_bin, bh_bin) tuples representing
    what objects are where. Positions are binned to 0.1 resolution so
    minor jitter doesn't break matching.
    """
    dets, _, age = cam.get_detections()
    if age > 2.0 or not dets:
        return []
    fp = []
    for d in dets:
        if d["name"] in _IGNORE_OBSTACLE:
            continue
        if d["conf"] < 0.30:
            continue
        fp.append((
            d["name"],
            round(d["cx"], 1),
            round(d["cy"], 1),
            round(d["bh"], 1),
        ))
    fp.sort()
    return fp


def _yolo_stuck(prev_fp, curr_fp):
    """Compare two YOLO fingerprints. Returns (stuck, description).

    Stuck = same objects at same binned positions means the rover
    isn't actually moving despite wheels being active.
    Requires at least 1 non-ignored detection for a meaningful comparison.
    """
    if not prev_fp or not curr_fp:
        return False, ""
    if prev_fp == curr_fp:
        names = sorted(set(n for n, _, _, _ in curr_fp))
        desc = "+".join(names) + " unchanged"
        return True, desc
    return False, ""

def _obstacle_avoidance_hint(cam):
    """Analyze YOLO detections to suggest the best evasion direction.

    Returns a hint string like "steer LEFT — obstacle is right of center"
    or "" if no obstacles need avoiding.
    """
    dets, _, age = cam.get_detections()
    if age > 2.0 or not dets:
        return ""

    # Collect obstacles in the forward path (not background classes)
    obstacles = []
    for d in dets:
        if d["name"] in _IGNORE_OBSTACLE:
            continue
        if d["conf"] < 0.25:
            continue
        # Only care about close-ish objects (bottom 60% of frame)
        if d["cy"] > 0.40:
            obstacles.append(d)

    if not obstacles:
        return ""

    # Find the most threatening obstacle (closest = largest cy + bh)
    threat = max(obstacles, key=lambda d: d["cy"] + d["bh"])

    # Determine which side is clearer
    # left_clear = space to the left of the obstacle's left edge
    # right_clear = space to the right of the obstacle's right edge
    left_edge = threat["cx"] - threat["bw"] / 2
    right_edge = threat["cx"] + threat["bw"] / 2
    left_clear = left_edge          # 0.0 = no space, 1.0 = all space
    right_clear = 1.0 - right_edge

    dist = f"{threat['dist_m']:.1f}m" if "dist_m" in threat else "close"
    name = threat["name"]

    if left_clear > right_clear and left_clear > 0.2:
        return (f"Steer LEFT to avoid {name}({dist}) — "
                f"it's right of center, left side is clearer.")
    elif right_clear > 0.2:
        return (f"Steer RIGHT to avoid {name}({dist}) — "
                f"it's left of center, right side is clearer.")
    elif threat["bw"] > 0.7:
        return (f"WIDE obstacle {name}({dist}) fills most of the frame — "
                f"BACK UP and turn 90°+ to find a clear path.")
    else:
        return (f"Obstacle {name}({dist}) ahead — "
                f"back up slightly and turn to go around it.")


def _extract_target(text):
    """Pull the search target noun(s) from a task string."""
    lower = text.lower()
    for prefix in ("find the ", "find a ", "find ", "go to the ", "go to ",
                   "navigate to the ", "navigate to ", "look for the ",
                   "look for a ", "look for ", "search for the ",
                   "search for a ", "search for ", "get to the ", "get to "):
        if lower.startswith(prefix):
            return lower[len(prefix):].split(" and ")[0].strip(" .,!?")
    # "where is the X"
    m = re.match(r"where\s+is\s+(?:the\s+)?(.+)", lower)
    if m:
        return m.group(1).strip(" .,!?")
    return ""


def _extract_all_targets(text):
    """Extract all targets from a multi-part command.

    "go to the kitchen and find the fridge" → ["kitchen", "fridge"]
    "find the exit and go to hallway" → ["exit", "hallway"]
    """
    lower = text.lower()
    targets = []
    # Split on "and then", "then", "and"
    parts = re.split(r'\s+and\s+then\s+|\s+then\s+|\s+and\s+', lower)
    for part in parts:
        part = part.strip()
        for prefix in ("find the ", "find a ", "find ", "go to the ", "go to ",
                       "navigate to the ", "navigate to ", "look for the ",
                       "look for a ", "look for ", "search for the ",
                       "search for a ", "search for ", "get to the ",
                       "get to "):
            if part.startswith(prefix):
                t = part[len(prefix):].strip(" .,!?")
                if t:
                    targets.append(t)
                break
    return targets

# Track correction frequency: {("wrong","correct"): count}
_correction_counts = {}
_CORRECTION_PERSIST_THRESHOLD = 3  # same correction must be seen N times to persist

def _apply_yolo_corrections(corrections):
    """Apply VLM label corrections — only persist consistent ones.

    corrections: dict like {"bed": "couch", "vase": "_false"}
    "_false" = false positive (logged, never persisted).
    Other corrections need to be seen 3+ times consistently before
    being saved to label_overrides.json, because COCO labels like
    "bed" get misapplied to different objects in different contexts.
    """
    from local_detector import LABEL_OVERRIDES, _save_label_overrides

    changed = False
    for wrong, correct in corrections.items():
        if not isinstance(wrong, str) or not isinstance(correct, str):
            continue
        wrong = wrong.strip().lower()
        correct = correct.strip().lower()
        if not wrong or not correct or wrong == correct:
            continue
        if correct == "_false":
            log_event("yolo_corr", f"False positive: '{wrong}' (not saved)")
            continue
        # Already mapped to this
        if LABEL_OVERRIDES.get(wrong) == correct:
            continue

        # Track how many times we've seen this specific correction
        key = (wrong, correct)
        _correction_counts[key] = _correction_counts.get(key, 0) + 1
        count = _correction_counts[key]

        # If a DIFFERENT correction for the same source label was seen,
        # that means the label is context-dependent — reset all counts for it
        conflicting = [k for k in _correction_counts if k[0] == wrong and k[1] != correct]
        if conflicting:
            for ck in conflicting:
                _correction_counts.pop(ck, None)
            _correction_counts[key] = 1
            count = 1
            log_event("yolo_corr",
                f"'{wrong}'→'{correct}' (1/{_CORRECTION_PERSIST_THRESHOLD}, "
                f"reset — conflicting corrections seen)")
            continue

        if count >= _CORRECTION_PERSIST_THRESHOLD:
            LABEL_OVERRIDES[wrong] = correct
            changed = True
            log_event("yolo_corr", f"Override saved: '{wrong}' → '{correct}' (confirmed {count}x)")
        else:
            log_event("yolo_corr",
                f"'{wrong}'→'{correct}' ({count}/{_CORRECTION_PERSIST_THRESHOLD})")

    if changed:
        _save_label_overrides()


def _yolo_one_liner(cam):
    """Return a compact one-liner of current YOLO detections (for round memory)."""
    dets, _, age = cam.get_detections()
    if age > 2.0 or not dets:
        return ""
    parts = []
    for d in dets:
        if d["name"] in _IGNORE_OBSTACLE:
            continue
        if d["conf"] < 0.30:
            continue
        s = d["name"]
        if "dist_m" in d:
            s += f"({d['dist_m']:.1f}m)"
        parts.append(s)
    return " ".join(parts[:5])  # max 5 objects to keep it brief


def _build_yolo_context(cam, task_text=""):
    """Build YOLO detection context string for LLM prompts.

    If the task target isn't a YOLO class, tells the LLM to use its own
    vision to find it (YOLO still reports obstacles).
    """
    from local_detector import COCO_NAMES

    dets, summary, age = cam.get_detections()

    # Check if task target is outside YOLO's class list
    target = _extract_target(task_text)
    target_in_yolo = target and target in COCO_NAMES
    target_missing = target and not target_in_yolo

    if age > 2.0 or not dets:
        if target_missing:
            return (f'YOLO cannot detect "{target}" — use YOUR OWN VISION '
                    f"to identify it in the camera frame.")
        return ""

    parts = []
    warnings = []
    for d in dets:
        s = f"{d['name']}({d['conf']:.0%}"
        if "dist_m" in d:
            s += f",{d['dist_m']:.1f}m"
        s += ")"
        parts.append(s)
        # Warn about close obstacles in driving corridor
        if (d["name"] not in _IGNORE_OBSTACLE and
                0.15 <= d["cx"] <= 0.85 and d["cy"] > 0.55 and d["bh"] > 0.20):
            dist = f"{d['dist_m']:.1f}m" if "dist_m" in d else "close"
            warnings.append(f"{d['name']} ({dist})")
    ctx = f"YOLO detects: {' '.join(parts)}"
    ctx += ("\nReview these YOLO labels against what you SEE. "
            'If any label is wrong, add "yolo_corrections" to your response.')
    if warnings:
        ctx += f"\n** WARNING: {', '.join(warnings)} — steer around or stop **"
    if target_missing:
        ctx += (f'\nYOLO cannot detect "{target}" — use YOUR OWN VISION '
                f"to identify it in the camera frame. YOLO shows obstacles only."
                f"\nLook for ANY object that could be a {target} — it may appear as "
                f"a bin, container, bag, box, or similar item. Describe what you "
                f"actually see in the frame. If ANYTHING looks like it could be "
                f'the "{target}", drive toward it.')
    return ctx


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

    # ── Tell orchestrator the current task ──
    orchestrator.set_task(text)
    orchestrator.reset_call_counter()
    _orch_guidance = [None]  # async guidance from orchestrator

    # Send first LLM call immediately
    first_prompt = text
    yolo_ctx = _build_yolo_context(cam, text)
    if yolo_ctx:
        first_prompt = f"{yolo_ctx}\n\n{first_prompt}"
        log_event("system", f"[yolo→llm] {yolo_ctx.splitlines()[0]}")

    resp = call_llm(first_prompt, frame)
    _check_floor(resp)
    _update_landmarks(resp)
    commands = resp.get("commands", [])
    say = resp.get("speak", "")
    observe = resp.get("observe", False)
    remember = resp.get("remember")

    # VLM YOLO corrections from first call
    yolo_corr = resp.get("yolo_corrections")
    if isinstance(yolo_corr, dict) and yolo_corr:
        _apply_yolo_corrections(yolo_corr)

    # Bash command execution
    bash_cmd = resp.get("bash")
    if bash_cmd:
        _run_bash(bash_cmd)
        if not observe:
            observe = True  # force observe so LLM sees the output

    # File tool execution
    _file_tool_used = False
    if resp.get("file_read"):
        fr = resp["file_read"]
        _file_read(fr.get("path", fr) if isinstance(fr, dict) else fr,
                   fr.get("offset") if isinstance(fr, dict) else None,
                   fr.get("limit") if isinstance(fr, dict) else None)
        _file_tool_used = True
    if resp.get("file_write"):
        fw = resp["file_write"]
        if isinstance(fw, dict) and "path" in fw:
            _file_write(fw["path"], fw.get("content", ""), fw.get("append", False))
        _file_tool_used = True
    if resp.get("file_list"):
        fl = resp["file_list"]
        _file_list(fl.get("path") if isinstance(fl, dict) else None,
                   fl.get("pattern") if isinstance(fl, dict) else None)
        _file_tool_used = True
    if resp.get("file_grep"):
        fg = resp["file_grep"]
        if isinstance(fg, dict) and "pattern" in fg:
            _file_grep(fg["pattern"], fg.get("path"))
        _file_tool_used = True
    if _file_tool_used and not observe:
        observe = True  # force observe so LLM sees file output

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
    prev_yolo_fp = _yolo_fingerprint(cam)  # YOLO fingerprint tracking
    yolo_same_count = 0         # consecutive rounds with identical YOLO fingerprint
    wheels_were_active = False  # did last command set include wheel motion?
    scan_only_rounds = 0        # consecutive rounds with gimbal-only (no wheels)
    was_interrupted = False
    repeated_speak = 0          # consecutive rounds with same speak text
    last_speak = ""             # previous round's speak text
    round_memory = []           # compact per-round summaries for LLM spatial memory

    # Capture round 0 (first LLM response, before loop)
    _yolo_brief = _yolo_one_liner(cam)
    round_memory.append(
        f"R0 pan={gimbal_pan:.0f}° | {say or 'no speech'}"
        + (f" | yolo: {_yolo_brief}" if _yolo_brief else ""))

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

        # Track scan-only rounds (gimbal moves but no wheel movement)
        if wheels_were_active or is_turning:
            scan_only_rounds = 0
        else:
            scan_only_rounds += 1

        execute(ser, commands, stop_event, cam)

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

        # No inter-round delay — LLM call latency is the bottleneck

        if stop_event.is_set():
            ser.stop()
            plan_active.clear()
            return {"original_text": text, "feedback": plan_feedback,
                    "stuck_count": total_stuck_events, "rounds": round_num,
                    "interrupted": True, "history": plan_history}

        frame = snap_with_flash(cam, ser)
        if frame is None:
            break

        # ── Stuck detection (YOLO + VLM + pixel, fused) ──
        sim = frame_similarity(prev_frame, frame)
        prev_frame = frame

        # YOLO fingerprint: same objects at same positions = not moving
        curr_yolo_fp = _yolo_fingerprint(cam)
        if wheels_were_active and curr_yolo_fp and prev_yolo_fp:
            yolo_match, yolo_desc = _yolo_stuck(prev_yolo_fp, curr_yolo_fp)
            if yolo_match:
                yolo_same_count += 1
            else:
                yolo_same_count = 0
        else:
            yolo_same_count = 0
            yolo_desc = ""
        prev_yolo_fp = curr_yolo_fp

        # YOLO blocking: large close object in driving corridor (instant)
        det_blocking, det_block_desc = _check_path_obstacle(cam)

        # Pixel similarity
        if wheels_were_active:
            thresh = STUCK_SIM_DRIVING
            needed = STUCK_ROUNDS_DRIVING
        else:
            thresh = STUCK_SIM_STATIC
            needed = STUCK_ROUNDS_STATIC
        if sim >= thresh:
            similar_count += 1
        else:
            similar_count = 0

        # ── Fused stuck decision ──
        # Fastest → slowest:
        #   0. IMU stationary — wheels ON but accel shows no motion (~300ms)
        #   1. YOLO blocking + wheels active → instant (0 rounds)
        #   2. YOLO fp match + pixel high (both agree) → 1 round
        #   3. YOLO fp match alone → 2 rounds
        #   4. VLM says stuck → 1 round (parsed from LLM response below)
        #   5. Pixel similarity alone → 3 rounds (fallback)
        global _backup_allowed
        # IMU stuck only counts if pixel similarity also confirms no movement.
        # The gyro is broken (always 0) so IMU alone is unreliable.
        imu_stuck = (_imu is not None and wheels_were_active
                     and _imu.is_stationary and sim >= thresh)
        det_block_stuck = det_blocking and wheels_were_active
        yolo_pixel_stuck = (yolo_same_count >= 1 and sim >= thresh
                            and wheels_were_active)           # corroborated: 1 round
        yolo_fp_stuck = yolo_same_count >= 2 and wheels_were_active  # standalone: 2 rounds
        pixel_stuck = similar_count >= needed
        # vlm_stuck is checked after LLM response below

        stuck = (imu_stuck or det_block_stuck or yolo_pixel_stuck
                 or yolo_fp_stuck or pixel_stuck) and not desk_mode
        _backup_allowed = stuck

        # "Possibly stuck" flag: YOLO fp matched once but no corroboration yet.
        # We'll hint the VLM so it can confirm via vision.
        possibly_stuck = (yolo_same_count == 1 and wheels_were_active
                          and not stuck and not desk_mode)

        yolo_info = ""
        if curr_yolo_fp:
            yolo_info = f" yolo_same={yolo_same_count}"
        if det_blocking:
            yolo_info += f" yolo_block={det_block_desc}"
        log_event("system", f"sim={sim:.3f} thresh={thresh} wheels={'ON' if wheels_were_active else 'off'}{yolo_info}")

        if stuck:
            total_stuck_events += 1
            if imu_stuck:
                reason = "imu_stationary"
            elif det_block_stuck:
                reason = f"yolo_block:{det_block_desc}"
            elif yolo_pixel_stuck:
                reason = f"yolo+pixel:1round({yolo_desc},sim={sim:.3f})"
            elif yolo_fp_stuck:
                reason = f"yolo_fp:{yolo_same_count}rounds({yolo_desc})"
            else:
                reason = f"pixel:{similar_count}rounds"
            log_event("stuck", f"#{total_stuck_events} {reason} (wheels={'ON' if wheels_were_active else 'off'}, sim={sim:.3f})")
            plan_history.append(f"Stuck #{total_stuck_events} ({reason})")

            # Hard abort after too many stuck events — stop wasting time
            if total_stuck_events >= 8:
                log_event("stuck", f"Aborting plan — stuck {total_stuck_events} times, escape failed")
                _speak("I'm stuck. Can't get out.", spk, mic_card)
                ser.stop()
                plan_active.clear()
                return {"original_text": text, "feedback": plan_feedback,
                        "stuck_count": total_stuck_events, "rounds": round_num,
                        "interrupted": False, "history": plan_history}

        # ── Orchestrator: ask for guidance when stuck ──
        if (stuck and total_stuck_events >= 2
                and orchestrator.get_call_count() < orchestrator.MAX_LLM_CALLS):
            _gd_frame = frame
            _gd_ctx = (
                f"Stuck {total_stuck_events} times. "
                f"Recent history: {'; '.join(plan_history[-4:])}"
            )
            def _ask_orch(jpg=_gd_frame, ctx=_gd_ctx):
                _orch_guidance[0] = orchestrator.ask_guidance(
                    jpg, ctx, log_fn=log_event)
            threading.Thread(target=_ask_orch, daemon=True).start()

        # ── Orchestrator: apply guidance if available ──
        guidance = _orch_guidance[0]
        if guidance:
            _orch_guidance[0] = None  # consume it
            if guidance.get("action") == "redirect":
                hint = guidance.get("guidance", "")
                plan_history.append(f"Orchestrator: {hint}")
            elif guidance.get("action") == "abort":
                reason = guidance.get("reason", "")
                log_event("orchestrator", f"Abort: {reason}")
                plan_history.append(f"Orchestrator abort: {reason}")
                observe = False

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
                    # Learn a generalized lesson from user feedback (bg thread)
                    threading.Thread(
                        target=orchestrator.learn_from_feedback,
                        args=(msg, text, list(plan_history[-6:])),
                        kwargs={"log_fn": log_event},
                        daemon=True,
                    ).start()
            except queue.Empty:
                break

        target = _extract_target(text)
        drive_hint = ""
        if is_turning:
            drive_hint = (
                f"You just spun the body. Center your head (pan→0), "
                f"then observe to confirm the target is ahead before driving. ")
        elif round_num <= 1:
            # Round 0-1: scan front hemisphere with gimbal
            if target:
                drive_hint = (
                    f"Scan: pan head -90° to +90° in one sweep. "
                    f'In "speak", describe what you SEE (not just "Searching"). '
                    f"If view is obstructed (object fills >40% of frame), "
                    f"spin body 90° first to get a clear view. ")
            else:
                drive_hint = "Survey first — pan head to find the best route. "
        elif round_num <= 3:
            # Round 2-3: spin body 180° and scan rear hemisphere
            drive_hint = (
                f"Now spin your body 180° to scan behind you. "
                f"Then pan head -90° to +90° to complete a full 360° survey. "
                f'In "speak", describe what you see in this direction. '
                f"Note anything relevant to the task — landmarks, doors, open paths, objects. ")
        elif round_num <= 5:
            if target:
                drive_hint = (
                    f"360° scan is DONE. COMMIT NOW: spin body to face the direction where you saw "
                    f"the best cue for \"{target}\" and DRIVE there at 0.1 m/s. "
                    f"If nothing matched, drive toward the most open area. "
                    f"Do NOT scan again from the same spot. ")
            else:
                drive_hint = ("360° scan complete. Now COMMIT: align body to the best direction, "
                              "center head, confirm target is ahead, then drive at 0.1 m/s. ")
        else:
            drive_hint = ("Keep driving toward the goal at 0.1 m/s. "
                          "Only stop if wall fills >60% of frame. ")
        if target:
            drive_hint += (
                f'In "speak", say what you actually see. '
                f'Is anything here a "{target}" (or similar shape/color)? '
                f"If yes, drive to it now. ")

        # Stale scan warning: LLM keeps scanning without driving
        if scan_only_rounds >= 5:
            drive_hint += (
                f"\n** WARNING: You have scanned for {scan_only_rounds} rounds "
                f"without driving. STOP scanning. Your NEXT command MUST include "
                f"wheel movement. Drive toward open space at 0.1 m/s. "
                f"If view is blocked by a nearby object, spin body 90° first. **")

        # Proactive obstacle avoidance: warn BEFORE stuck triggers
        if not stuck and not possibly_stuck and det_blocking:
            avoid_hint = _obstacle_avoidance_hint(cam)
            if avoid_hint:
                drive_hint += f"\n** OBSTACLE AHEAD: {avoid_hint} **"

        stuck_hint = ""
        avoidance = _obstacle_avoidance_hint(cam) if (stuck or possibly_stuck) else ""
        if stuck and wheels_were_active:
            if det_block_stuck:
                obstacle_info = f"YOLO: {det_block_desc} blocking path"
            elif yolo_pixel_stuck:
                obstacle_info = f"YOLO+pixel: {yolo_desc}"
            elif yolo_fp_stuck:
                obstacle_info = f"YOLO: {yolo_desc}"
            else:
                obstacle_info = f"pixel similarity {sim:.0%}"
            evasion = f" {avoidance}" if avoidance else (
                " CURVE away from the obstacle using differential wheel speeds.")

            # Escalation based on how many times we've been stuck
            escalation = ""
            if total_stuck_events >= 4:
                escalation = (
                    f" CRITICAL: stuck {total_stuck_events} times. "
                    f"Curving and spinning have failed. Back up briefly, then "
                    f"drive FORWARD toward the WIDEST open space. Commit to it.")
            elif total_stuck_events >= 3:
                escalation = (
                    f" (stuck {total_stuck_events} times — try spinning 90° "
                    f"to face the most open area, then drive forward. "
                    f"Only back up if spinning doesn't help.)")
            # stuck 1-2: default evasion text (curve) is sufficient

            stuck_hint = (
                f"\n\n** STUCK #{total_stuck_events} — {obstacle_info} "
                f"(wheels ON but not moving) ** "
                f"STOP wheels immediately.{evasion}{escalation} "
                f'Set "stuck":true in your response.')
        elif possibly_stuck:
            evasion_note = f" YOLO suggests: {avoidance}" if avoidance else ""
            stuck_hint = (
                f"\n\n** POSSIBLY STUCK — YOLO detections unchanged since last round **{evasion_note} "
                f"Look at this frame: are you moving? Is there an obstacle ahead? "
                f'If stuck, set "stuck":true and evade. If moving, set "stuck":false.')
        elif stuck:
            target_nudge = ""
            if target:
                target_nudge = (
                    f' If your view is obstructed (something close fills the frame), '
                    f'spin 90° to get a clear view. Then look for "{target}" and drive to it.')
            stuck_hint = (
                f"\n\n** NOT MAKING PROGRESS ({similar_count} rounds, scene unchanged) ** "
                f"You keep scanning without moving. STOP scanning and START driving. "
                f"Pick the most open path and drive at 0.12 m/s. Go WIDE around obstacles "
                f"— do not try to squeeze through tight gaps.{target_nudge}")

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

        # ── Orchestrator guidance hint ──
        orch_hint = ""
        if guidance and guidance.get("action") == "redirect":
            orch_hint = f'\n\n** ORCHESTRATOR SAYS: {guidance.get("guidance", "")} **'
            guidance = None  # consumed

        # ── YOLO context for LLM ──
        yolo_ctx = _build_yolo_context(cam, text)
        if yolo_ctx:
            log_event("system", f"[yolo→llm] {yolo_ctx.splitlines()[0]}")
        yolo_line = f"\n{yolo_ctx}" if yolo_ctx else ""

        # ── Bash output injection ──
        global _bash_output
        bash_ctx = ""
        if _bash_output:
            bash_ctx = f"\n\n** BASH OUTPUT **\n{_bash_output}"
            _bash_output = ""  # consume it

        # ── Build spatial memory block from recent rounds ──
        memory_block = ""
        if round_memory:
            memory_block = ("RECENT MEMORY (your past observations):\n"
                            + "\n".join(round_memory[-8:]) + "\n\n")

        imu_line = ""
        if _imu:
            imu_line = _imu.get_prompt_line()
            if imu_line:
                imu_line = imu_line + "\n"

        # ── Build target focus reminder for navigation tasks ──
        target_focus = ""
        if _last_task_target:
            target_focus = f"\n** YOU ARE LOOKING FOR: {_last_task_target.upper()} **\nKeep this target in mind. Avoid obstacles but don't lose focus on finding it.\n"

        prompt = (
            f'** YOUR CURRENT TASK: "{text}" **\n'
            f"{target_focus}\n"
            f"{memory_block}"
            f"[Observe round {round_num}/{MAX_OBSERVE_ROUNDS}]\n"
            f"Last action: {last_action}\n"
            f"Result: {scene_change}\n"
            f"Gimbal: pan={actual_pan:.0f}°, tilt={gimbal_tilt:.0f}°\n"
            f"{imu_line}"
            f"{look_vs_drive}{yolo_line}\n\n"
            f"{drive_hint}"
            f"This is a NEW camera frame taken AFTER your last commands. "
            f"Evaluate: does it match your task? Is the path clear? "
            f"Only drive if confirmed."
            f"{stuck_hint}{user_context}{orch_hint}{bash_ctx}")

        # ── Log round context for debugging ──
        ctx_parts = []
        ctx_parts.append(f"round={round_num}/{MAX_OBSERVE_ROUNDS}")
        ctx_parts.append(f"pan={actual_pan:.0f}°")
        ctx_parts.append(f"last={last_action[:50]}")
        if stuck:
            ctx_parts.append(f"stuck_total={total_stuck_events}")
        if stuck_hint:
            # Extract first line of stuck hint for log
            hint_summary = stuck_hint.strip().split("**")[1] if "**" in stuck_hint else stuck_hint[:80]
            ctx_parts.append(f"hint={hint_summary.strip()[:80]}")
        if orch_hint:
            ctx_parts.append(f"orch=yes")
        log_event("observe", " | ".join(ctx_parts))

        resp = call_llm(prompt, frame)
        _check_floor(resp)
        _update_landmarks(resp)
        commands = resp.get("commands", [])
        say = resp.get("speak", "")
        observe = resp.get("observe", False)
        remember = resp.get("remember")

        # ── Repeated action loop detection ──
        if say and say == last_speak:
            repeated_speak += 1
        else:
            repeated_speak = 0
        last_speak = say
        if repeated_speak >= 3:
            log_event("stuck",
                      f"LLM stuck in loop: repeated '{say}' "
                      f"{repeated_speak + 1} times — aborting plan")
            _speak("I'm repeating myself. Stopping.", spk, mic_card)
            ser.stop()
            plan_active.clear()
            return {"original_text": text, "feedback": plan_feedback,
                    "stuck_count": total_stuck_events, "rounds": round_num,
                    "interrupted": False, "history": plan_history}

        # ── Log LLM decision summary ──
        cmd_types = []
        for c in commands:
            if isinstance(c, dict):
                if c.get("T") == 1:
                    l, r = c.get("L", 0), c.get("R", 0)
                    if l == 0 and r == 0:
                        cmd_types.append("stop")
                    elif (l > 0 and r < 0) or (l < 0 and r > 0):
                        cmd_types.append(f"spin({'L' if l < 0 else 'R'})")
                    elif l < 0 and r < 0:
                        cmd_types.append(f"back({abs(l):.2f})")
                    else:
                        cmd_types.append(f"drive({l:.2f},{r:.2f})")
                elif c.get("T") == 133:
                    cmd_types.append(f"gimbal({c.get('X',0):.0f},{c.get('Y',0):.0f})")
                elif "_pause" in c:
                    cmd_types.append(f"pause({c['_pause']}s)")
        log_event("decide", f"cmds=[{','.join(cmd_types)}] speak=\"{say}\" stuck={resp.get('stuck',False)} observe={observe}")

        # ── VLM stuck detection: LLM sees the frame and confirms stuck ──
        vlm_stuck = resp.get("stuck", False) is True
        if vlm_stuck and not stuck and not desk_mode:
            stuck = True
            _backup_allowed = True
            total_stuck_events += 1
            log_event("stuck", f"vlm_confirmed (wheels={'ON' if wheels_were_active else 'off'}, sim={sim:.3f})")
            plan_history.append(f"Stuck #{total_stuck_events} (vlm_confirmed)")

        # ── VLM YOLO label corrections ──
        yolo_corr = resp.get("yolo_corrections")
        if isinstance(yolo_corr, dict) and yolo_corr:
            _apply_yolo_corrections(yolo_corr)

        # ── Bash command execution ──
        bash_cmd = resp.get("bash")
        if bash_cmd:
            _run_bash(bash_cmd)
            if not observe:
                observe = True  # force observe so LLM sees the output

        # ── File tool execution ──
        _ft_used = False
        if resp.get("file_read"):
            fr = resp["file_read"]
            _file_read(fr.get("path", fr) if isinstance(fr, dict) else fr,
                       fr.get("offset") if isinstance(fr, dict) else None,
                       fr.get("limit") if isinstance(fr, dict) else None)
            _ft_used = True
        if resp.get("file_write"):
            fw = resp["file_write"]
            if isinstance(fw, dict) and "path" in fw:
                _file_write(fw["path"], fw.get("content", ""), fw.get("append", False))
            _ft_used = True
        if resp.get("file_list"):
            fl = resp["file_list"]
            _file_list(fl.get("path") if isinstance(fl, dict) else None,
                       fl.get("pattern") if isinstance(fl, dict) else None)
            _ft_used = True
        if resp.get("file_grep"):
            fg = resp["file_grep"]
            if isinstance(fg, dict) and "pattern" in fg:
                _file_grep(fg["pattern"], fg.get("path"))
            _ft_used = True
        if _ft_used and not observe:
            observe = True

        if say:
            _speak_async(say, spk, mic_card)
            plan_history.append(f"Round {round_num}: {say}")
        if remember:
            save_memory(remember)

        # ── Round memory: capture compact summary for LLM spatial memory ──
        _yolo_brief = _yolo_one_liner(cam)
        _mem_line = f"R{round_num} pan={gimbal_pan:.0f}° | {last_action} | {say or 'no speech'}"
        if _yolo_brief:
            _mem_line += f" | yolo: {_yolo_brief}"
        round_memory.append(_mem_line)
        if len(round_memory) > 10:
            round_memory = round_memory[-10:]

    plan_active.clear()
    return {"original_text": text, "feedback": plan_feedback,
            "stuck_count": total_stuck_events, "rounds": round_num,
            "interrupted": was_interrupted, "history": plan_history}

# ── Provider/state helpers for web_ui ──────────────────────────────────

def _get_providers():
    # When xAI Realtime is active, voice is on (just via xAI, not Groq STT)
    effective_stt = stt_enabled or (_xai_voice is not None)
    return {
        "current": {"stt": _stt_name, "llm": _llm_name, "tts": _tts_name,
                    "orch": orchestrator.OLLAMA_TEXT_MODEL},
        "available": AVAILABLE_PROVIDERS,
        "desk_mode": desk_mode,
        "stt_enabled": effective_stt,
        "gimbal_pan_enabled": gimbal_pan_enabled,
        "tts_enabled": tts_enabled,
        "yolo_enabled": yolo_enabled,
    }

def _set_desk_mode(val):
    global desk_mode
    desk_mode = val
    log_event("system", f"Desk mode: {'ON' if desk_mode else 'OFF'}")

def _set_stt_enabled(val):
    global stt_enabled
    stt_enabled = val
    log_event("system", f"STT: {'ON' if stt_enabled else 'OFF'}")

def _set_gimbal_pan_enabled(val):
    global gimbal_pan_enabled
    gimbal_pan_enabled = val
    log_event("system", f"Gimbal pan: {'ON' if gimbal_pan_enabled else 'OFF'}")

def _set_tts_enabled(val):
    global tts_enabled
    tts_enabled = val
    log_event("system", f"TTS: {'ON' if tts_enabled else 'OFF'}")

yolo_enabled = True  # YOLO detection on/off

def _set_yolo_enabled(val):
    global yolo_enabled
    yolo_enabled = val
    log_event("system", f"YOLO: {'ON' if yolo_enabled else 'OFF'}")

killed = False  # When True, main loop ignores commands

def _set_killed(val):
    """Kill switch: stop all activity, or resume."""
    global killed, stt_enabled
    killed = bool(val)
    if killed:
        # 1. Stop motors immediately
        stop_event.set()
        if _ser_ref:
            try:
                _ser_ref.stop()
            except Exception:
                pass
        # 2. Disable STT
        stt_enabled = False
        # 3. Kill any playing TTS audio
        subprocess.run(["pkill", "-9", "aplay"], capture_output=True)
        subprocess.run(["pkill", "-9", "gst-launch-1.0"], capture_output=True)
        # 4. Drain command queue
        while not command_queue.empty():
            try:
                command_queue.get_nowait()
            except queue.Empty:
                break
        # 5. Drain interrupt queue
        while not interrupt_queue.empty():
            try:
                interrupt_queue.get_nowait()
            except queue.Empty:
                break
        # 6. Lights off
        if _ser_ref:
            try:
                _ser_ref._send_raw({"T": 132, "IO4": 0, "IO5": 0})
            except Exception:
                pass
        log_event("system", "KILL SWITCH ENGAGED — all activity stopped")
    else:
        stop_event.clear()
        stt_enabled = True
        log_event("system", "Kill switch released — resuming")

# ── Topological Navigation ────────────────────────────────────────────

_topo_map = None  # TopoMap, initialized in main()


def _topo_room_check(scene_text):
    """Check which room the scene describes. Returns (room_id, confidence)."""
    ranked = room_context.identify_room(scene_text)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0], ranked[0][2]
    return None, 0.0


def _topo_navigate(nav_target, stop_event):
    """Topological navigation: plan route through room graph, execute each leg.

    1. Resolve nav_target to a destination room
    2. Plan route as sequence of legs (transitions)
    3. Navigate each leg: find doorway → cross → verify room
    4. Fall back to reactive nav if no topo route exists

    Returns True if destination reached.
    """
    global _topo_map
    nav = _navigator
    if not nav or not _topo_map:
        log_event("plan", "Topo nav unavailable, falling back to reactive")
        return nav.navigate_reactive(nav_target) if nav else False

    # Resolve target to a room id
    target_room = None
    target_lower = nav_target.lower().strip()
    for room in _topo_map.rooms():
        if (room.id in target_lower or
                room.label.lower() in target_lower or
                target_lower in room.id or
                target_lower in room.label.lower()):
            target_room = room.id
            break

    if not target_room:
        log_event("plan", f"'{nav_target}' not a known room, "
                          f"using reactive nav")
        return nav.navigate_reactive(nav_target)

    # Determine current room
    from_room = _topo_map.current_room
    if not from_room:
        # Try to identify current room from camera
        jpeg = nav.tracker.get_jpeg()
        if jpeg:
            scene_result = nav._llm_call(
                "Briefly describe this room. What room are you in? "
                "Mention floor type and key furniture.",
                jpeg)
            if scene_result:
                scene = scene_result.get("scene", str(scene_result))
                from_room, conf = _topo_room_check(scene)
                if from_room:
                    _topo_map.current_room = from_room
                    _topo_map.current_confidence = conf
                    log_event("plan", f"Identified starting room: "
                                      f"{from_room} ({conf})")

    if not from_room:
        log_event("plan", "Can't determine current room, "
                          "using reactive nav")
        return nav.navigate_reactive(nav_target)

    if from_room == target_room:
        log_event("plan", f"Already in {target_room}!")
        return True

    # Plan route
    legs = _topo_map.plan_route(from_room, target_room)
    if not legs:
        log_event("plan", f"No topo route from {from_room} to "
                          f"{target_room}, using reactive nav")
        return nav.navigate_reactive(nav_target)

    route_str = _topo_map.route_summary(legs)
    log_event("plan", f"Route: {route_str}")

    # Execute each leg
    for i, leg in enumerate(legs):
        if stop_event.is_set():
            log_event("plan", "Topo nav stopped")
            return False

        instruction = _topo_map.leg_instruction(leg)
        log_event("plan", f"Leg {i+1}/{len(legs)}: "
                          f"{leg.from_room} → {leg.to_room} "
                          f"via [{leg.transition}]")
        log_event("plan", f"  Hint: {instruction.get('exit_hint', '')}")

        success, new_room, scene = nav.navigate_leg(
            instruction,
            room_check_fn=_topo_room_check,
        )

        if success:
            arrived_room = new_room or leg.to_room
            _topo_map.current_room = arrived_room
            _topo_map.current_confidence = 0.8
            log_event("plan", f"Leg {i+1} complete: now in {arrived_room}")

            # Entry verification: panoramic scan to confirm room
            nav._move_gimbal(0, 0)
            time.sleep(0.5)
            jpeg = nav.tracker.get_jpeg()
            if jpeg:
                verify = nav._llm_call(
                    f"I just entered a new room. Describe what you see. "
                    f"Mention the floor type and key objects.",
                    jpeg)
                if verify:
                    vscene = verify.get("scene", str(verify))
                    vroom, vconf = _topo_room_check(vscene)
                    if vroom:
                        _topo_map.current_room = vroom
                        _topo_map.current_confidence = vconf
                        log_event("plan", f"Entry verified: {vroom} "
                                          f"({vconf})")
                        if vroom != leg.to_room:
                            log_event("plan",
                                f"Wrong room! Expected {leg.to_room}, "
                                f"got {vroom}")
                            # Don't abort — the route planner can replan
        else:
            log_event("plan", f"Leg {i+1} failed: could not cross "
                              f"{leg.transition}")
            # Try reactive nav as fallback for this leg
            log_event("plan", f"Falling back to reactive nav for "
                              f"'{leg.to_room}'")
            reached = nav.navigate_reactive(leg.to_room)
            if reached:
                _topo_map.current_room = leg.to_room
                _topo_map.current_confidence = 0.6
                log_event("plan", f"Reactive fallback reached {leg.to_room}")
            else:
                log_event("plan", f"Could not reach {leg.to_room}")
                return False

    # Save updated topo map state
    _topo_map.save()
    log_event("plan", f"Topo nav complete: arrived at {target_room}")
    return True


# ── Orchestrated Navigation (legacy) ─────────────────────────────────

def _orchestrated_navigate(nav_target, prior_history=None):
    """Orchestrator-led navigation: plan route → step through with navigator.

    Args:
        nav_target: destination string
        prior_history: list of step summaries from a failed prior attempt
                       (fed into plan_route so the orchestrator avoids
                       repeating the same approach)

    Returns dict: {"reached": bool, "history": list[str]}
    """
    nav = _navigator
    if not nav:
        return {"reached": False, "history": []}

    # 1. Gather context for route planning
    jpeg = nav.tracker.get_jpeg()
    explore_summary = ""
    if nav.exploration:
        body_yaw = (nav.pose.body_yaw
                    if hasattr(nav.pose, 'body_yaw') else 0)
        explore_summary = nav.exploration.summarize_for_llm(body_yaw) or ""

    # 2. Plan route (1 LLM call → 2-5 spatial steps)
    orchestrator.reset_call_counter()
    prior_ctx = ""
    if prior_history:
        prior_ctx = ("PREVIOUS ATTEMPT FAILED: "
                     + "; ".join(prior_history[-5:])
                     + "\nPlan a DIFFERENT route.")
    # Room map context for route planning
    _rm_json = None
    if nav.room_map:
        rx = nav.pose.x if hasattr(nav.pose, 'x') else 0
        ry = nav.pose.y if hasattr(nav.pose, 'y') else 0
        yaw = nav.pose.body_yaw if hasattr(nav.pose, 'body_yaw') else 0
        _rm_json = nav.room_map.room_json(rx, ry, yaw)

    plan = orchestrator.plan_route(
        nav_target, jpeg_bytes=jpeg,
        exploration_summary=(explore_summary +
                             ("\n" + prior_ctx if prior_ctx else "")),
        log_fn=log_event,
        room_map_json=_rm_json)

    log_event("plan",
              f"[plan] {len(plan.steps)} steps for '{nav_target}'")

    # 3. Execute each step
    max_retries = 1
    while plan.current_index < len(plan.steps):
        if stop_event.is_set():
            log_event("plan", "[plan] Aborted by stop event")
            return {"reached": False, "history": plan.history}

        step = plan.steps[plan.current_index]
        step_num = plan.current_index + 1
        total = len(plan.steps)
        log_event("plan",
                  f"[plan] Step {step_num}/{total}: '{step.target}'")

        # Build context string for navigator
        plan_context = plan.context_for_navigator()

        # Run navigator for this step
        reached = nav.navigate(
            step.target,
            plan_context=plan_context,
            step_budget=step.waypoint_budget,
            plan=plan,
        )

        # Read step result
        result = getattr(nav, '_last_result', None)
        if result is None:
            # Navigator didn't store a result (shouldn't happen)
            from orchestrator import StepResult
            result = StepResult(
                success=reached, reason="arrived" if reached else "unknown",
                waypoints_used=0, final_scene="", final_yolo="",
                exploration_summary="",
            )

        log_event("plan",
                  f"[plan] Step {step_num}/{total} "
                  f"{'succeeded' if result.success else 'failed'}: "
                  f"{result.reason} ({result.waypoints_used} wp)")

        # Orchestrator always evaluates — it decides when the task is done
        # (executor's "arrived" is just input, not the final word)
        eval_jpeg = nav.tracker.get_jpeg()
        # Refresh room map for evaluation
        if nav.room_map:
            rx = nav.pose.x if hasattr(nav.pose, 'x') else 0
            ry = nav.pose.y if hasattr(nav.pose, 'y') else 0
            yaw = nav.pose.body_yaw if hasattr(nav.pose, 'body_yaw') else 0
            _rm_json = nav.room_map.room_json(rx, ry, yaw)
        decision = orchestrator.evaluate_step(
            plan, result, jpeg_bytes=eval_jpeg, log_fn=log_event,
            room_map_json=_rm_json)

        action = decision.get("decision", "continue" if result.success else "abort")

        # Record in history AFTER evaluation — skip/continue override failure
        if action in ("done", "continue", "skip"):
            plan.history.append(f"{step.target} (done)")
        else:
            status = "done" if result.success else f"failed:{result.reason}"
            plan.history.append(f"{step.target} ({status})")

        if action == "done":
            # Orchestrator confirms mission goal is achieved
            reason = decision.get("reason", "goal achieved")
            log_event("plan",
                      f"[plan] Orchestrator confirms done: {reason}")
            return {"reached": True, "history": plan.history}

        elif action == "continue":
            plan.current_index += 1
            # Last step accepted → success
            if plan.current_index >= len(plan.steps):
                log_event("plan",
                          f"[plan] Navigation complete: '{nav_target}'")
                return {"reached": True, "history": plan.history}

        elif action == "skip":
            log_event("plan",
                      f"[plan] Skipping step: {decision.get('reason', '')}")
            plan.current_index += 1
            # Last step skipped → orchestrator says close enough
            if plan.current_index >= len(plan.steps):
                log_event("plan",
                          f"[plan] Navigation complete (skipped last): "
                          f"'{nav_target}'")
                return {"reached": True, "history": plan.history}

        elif action == "retry":
            if max_retries > 0:
                max_retries -= 1
                hint = decision.get("retry_hint", "")
                if hint:
                    plan.history.append(f"retry hint: {hint}")
                # If failed due to budget, increase it for the retry
                if result.reason == "budget" and step:
                    old_budget = step.waypoint_budget
                    step.waypoint_budget = min(30, old_budget + 10)
                    log_event("plan",
                              f"[plan] Budget increase: {old_budget} → {step.waypoint_budget}")
                log_event("plan", f"[plan] Retrying step: {hint}")
                # Loop will re-execute same step
            else:
                log_event("plan", "[plan] No retries left, advancing")
                plan.current_index += 1

        elif action == "replan":
            new_steps_data = decision.get("new_steps", [])
            if new_steps_data:
                from orchestrator import Step
                new_steps = []
                for s in new_steps_data:
                    if isinstance(s, dict) and "target" in s:
                        new_steps.append(Step(
                            target=s["target"],
                            rationale=s.get("rationale", ""),
                            waypoint_budget=max(8, int(
                                s.get("waypoint_budget", 10))),
                        ))
                if new_steps:
                    plan.steps = (plan.steps[:plan.current_index + 1]
                                  + new_steps)
                    plan.current_index += 1
                    log_event("plan",
                              f"[plan] Replanned: {len(new_steps)} new steps")
                else:
                    plan.current_index += 1
            else:
                plan.current_index += 1

        elif action == "abort":
            log_event("plan",
                      f"[plan] Aborted: {decision.get('reason', '')}")
            return {"reached": False, "history": plan.history}

        else:
            # Unknown decision — advance
            plan.current_index += 1

    # All steps exhausted without explicit success/abort
    log_event("plan", f"[plan] All steps exhausted for '{nav_target}'")
    # Save room map after navigation
    if nav.room_map:
        nav.room_map.save()
    return {"reached": False, "history": plan.history}


# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    global _ser_ref
    subprocess.run(["pkill", "-f", "rover_brain.py"], capture_output=True)
    time.sleep(1)

    ser = Serial()
    _ser_ref = ser
    cam = Camera()

    # Initialize YOLO detector for real-time detection during navigation
    try:
        from local_detector import LocalDetector
        cam.detector = LocalDetector()
        log_event("system", f"YOLO detector active: {cam.detector.backend}, "
                  f"{len(cam.detector.class_names)} classes")
    except Exception as e:
        log_event("error", f"YOLO detector failed to load: {e}")

    # Initialize depth estimator for distance estimation
    try:
        from local_detector import DepthEstimator
        cam.depth_estimator = DepthEstimator()
        log_event("system", f"Depth estimator active: {cam.depth_estimator.backend}")
    except Exception as e:
        log_event("system", f"Depth estimator not available: {e}")

    # Initialize FloorNavigator for obstacle-aware driving
    global _floor_nav
    if cam.detector is not None:
        try:
            from floor_nav import FloorNavigator
            _floor_nav = FloorNavigator(ser, cam.detector, cam)
            log_event("system", "FloorNavigator active")
        except Exception as e:
            log_event("system", f"FloorNavigator not available: {e}")

    # Initialize and start web UI
    web_ui.init(
        camera=cam, serial=ser,
        detector=getattr(cam, 'detector', None),
        get_log_events_since=get_log_events_since,
        get_map_state=_get_map_state,
        log_event=log_event,
        set_provider=set_provider,
        get_providers=_get_providers,
        set_desk_mode=_set_desk_mode,
        set_stt_enabled=_set_stt_enabled,
        set_tts_enabled=_set_tts_enabled,
        set_gimbal_pan_enabled=_set_gimbal_pan_enabled,
        set_yolo_enabled=_set_yolo_enabled,
        set_killed=_set_killed,
        get_killed=lambda: killed,
        plan_active=plan_active,
        stop_event=stop_event,
        command_queue=command_queue,
        interrupt_queue=interrupt_queue,
        classify_interrupt=classify_interrupt,
    )
    web_ui.start_server()

    mic_dev, mic_card = audio.find_mic()
    spk = audio.find_speaker()

    # Populate refs for xAI Realtime (allows hot-swap from web UI)
    _shared_refs.update(ser=ser, cam=cam, spk=spk, mic_dev=mic_dev)

    # Initialize Navigator (LLM-centric nav with YOLO fast fallback)
    global _navigator, _room_scanner
    try:
        import navigator
        from navigator import Navigator
        from rover_brain import PoseTracker

        _nav_pose = PoseTracker()
        # Hook pose tracker into serial commands
        _orig_send = ser.send
        def _nav_send_hook(cmd):
            # Apply gimbal pan offset to all T:133 commands system-wide
            if isinstance(cmd, dict) and cmd.get("T") == 133:
                cmd = dict(cmd, X=cmd.get("X", 0) + navigator.GIMBAL_PAN_OFFSET)
            _orig_send(cmd)
            _nav_pose.on_command(cmd)
        ser.send = _nav_send_hook

        def _nav_vision_fn(prompt, jpeg_bytes):
            """Lightweight LLM vision call for Navigator — 30s timeout.
            Uses the orchestrator's current model (respects UI dropdown)."""
            import base64 as _b64
            import requests
            model = orchestrator.OLLAMA_MODEL
            b64 = _b64.b64encode(jpeg_bytes).decode()
            # Route claude models to Anthropic API
            if model.startswith("claude-"):
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "system": "Reply ONLY with the requested JSON.",
                        "messages": [{"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/jpeg",
                                "data": b64}},
                            {"type": "text", "text": prompt},
                        ]}],
                        "max_tokens": 300,
                        "temperature": 0.3,
                    },
                    timeout=30)
                r.raise_for_status()
                return r.json()["content"][0]["text"].strip()
            else:
                r = requests.post(
                    orchestrator.OLLAMA_URL,
                    json={"model": model,
                          "messages": [
                              {"role": "system",
                               "content": "Reply ONLY with the requested JSON."},
                              {"role": "user", "content": prompt,
                               "images": [b64]},
                          ],
                          "stream": False, "think": False,
                          "options": {"temperature": 0.3, "num_predict": 300}},
                    timeout=30)
                r.raise_for_status()
                content = r.json()["message"].get("content", "")
                return content

        def _nav_parse_fn(raw):
            """Parse JSON from LLM response."""
            text = raw.strip()
            if "```" in text:
                lines = [l for l in text.split("\n")
                         if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return None

        # Exploration grid — persists across navigate() calls, cleared on restart
        _exploration_grid = None
        try:
            from exploration_grid import ExplorationGrid
            _exploration_grid = ExplorationGrid()
            navigator._exploration_grid = _exploration_grid  # expose for web_ui
            print("[nav] Exploration grid active (80x80x2, 20cm cells)")
        except Exception as _eg_err:
            print(f"[nav] Exploration grid unavailable: {_eg_err}")

        # Room map — 3D descriptive spatial map of objects
        from room_map import RoomMap
        _room_map = RoomMap()
        log_event("system", f"RoomMap active ({_room_map.object_count()} objects loaded)")

        _navigator = Navigator(
            rover=ser, detector=cam.detector, tracker=cam,
            llm_vision_fn=_nav_vision_fn, parse_fn=_nav_parse_fn,
            pose=_nav_pose, voice_fn=lambda msg: _speak(msg, spk, mic_card),
            emergency_event=stop_event,
            exploration_grid=_exploration_grid,
            log_fn=log_event,
            room_map=_room_map,
            yolo_correction_fn=_apply_yolo_corrections,
        )
        # Expose room map + pose to camera for detection recording
        cam._room_map = _room_map
        cam._nav_pose = _nav_pose
        log_event("system", f"Navigator active (yolo={'ON' if cam.detector else 'OFF'}"
                  f", exploration={'ON' if _exploration_grid else 'OFF'}"
                  f", room_map={_room_map.object_count()} objects)")

        # Topological map for room-graph navigation
        global _topo_map
        from topo_nav import TopoMap
        _topo_map = TopoMap()
        # Sync current room from room_context
        rc_data = room_context.load_rooms()
        if rc_data.get("current_room"):
            _topo_map.current_room = rc_data["current_room"]
            _topo_map.current_confidence = rc_data.get("current_confidence", 0.5)
        log_event("system", f"TopoMap active ({len(_topo_map.rooms())} rooms, "
                  f"{len(_topo_map.transitions())} transitions, "
                  f"current={_topo_map.current_room})")

        # Vector room scanner (LLM-estimated distances + sizes).
        try:
            from room_scanner import VectorRoomScanner

            def _room_scan_vision_fn(prompt, jpeg_bytes):
                raw = _nav_vision_fn(prompt, jpeg_bytes)
                return _nav_parse_fn(raw) if raw else None

            _room_scanner = VectorRoomScanner(
                rover=ser, tracker=cam,
                vision_json_fn=_room_scan_vision_fn,
                log_fn=lambda msg: log_event("room", msg),
            )
            log_event("system", "Room scanner active (vector LLM scan)")
        except Exception as _rs_err:
            _room_scanner = None
            log_event("system", f"Room scanner unavailable: {_rs_err}")
    except Exception as e:
        log_event("system", f"Navigator not available: {e}")
        _room_scanner = None
        import traceback; traceback.print_exc()

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

    # ── IMU initialization ──
    global _imu
    try:
        imu_poller = imu_mod.IMUPoller(ser, log_fn=log_event)
        if imu_poller.read_once():
            imu_poller.set_heading_offset()
            log_event("system", f"IMU active: "
                      f"accel={imu_poller.state.accel_magnitude:.3f}, "
                      f"voltage={imu_poller.state.voltage:.1f}V")
        else:
            log_event("system", "IMU: no T:1001 data (will retry in background)")
        imu_poller.start()
        _imu = imu_poller
    except Exception as e:
        log_event("error", f"IMU init failed: {e}")
        _imu = None

    _shared_refs["imu"] = _imu

    # ── Voltage watchdog thread ──
    threading.Thread(target=_voltage_watchdog, daemon=True,
                     name="voltage-watchdog").start()
    log_event("system", f"Voltage watchdog active (threshold={_LOW_VOLTAGE_THRESHOLD}V)")

    # ── xAI Realtime voice (if requested via CLI) ──
    if _use_xai_realtime:
        set_provider("stt", "xai-realtime")

    # Start voice listener thread (STT disabled when xAI Realtime is active)
    vt = threading.Thread(target=voice_thread, args=(mic_dev, mic_card), daemon=True)
    vt.start()

    # Restore saved LLM/orchestrator preferences
    _restore_provider_prefs()

    log_event("system", "rover_brain_llm ready")

    # Startup greeting — simple head nod, no observe loop
    if _xai_voice:
        time.sleep(3)
        _xai_voice.send_text("You just booted up. Greet briefly.")
    else:
        ser.send({"T": 133, "X": 0, "Y": 20, "SPD": 300, "ACC": 20})
        time.sleep(0.3)
        ser.send({"T": 133, "X": 0, "Y": -10, "SPD": 300, "ACC": 20})
        time.sleep(0.3)
        ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 15})
        _speak("Online.", spk, mic_card)

    # Main loop — wait for commands from voice thread or web UI
    while True:
        if killed:
            time.sleep(0.5)
            continue
        try:
            text = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if killed:
            continue

        # Safety check for stop words
        if _clean_words(text) & STOP_WORDS:
            if not _xai_voice:
                ser.stop()
                continue
            # xAI mode: let Grok handle it (it knows to send emergency stop)

        # Route through xAI Realtime if active (Grok handles + speaks response)
        if _xai_voice:
            log_event("plan", f"→ xAI: {text}")
            _xai_voice.send_text(text)
            continue

        # ── Retry support: restore last meaningful task ──
        global _last_task, _last_task_target
        text_lower = text.strip().lower()
        is_retry = text_lower in ("retry", "try again", "continue", "keep going", "go on")
        if is_retry and _last_task:
            text = _last_task
            log_event("system", f"Retry: restored task '{text}'")
        elif text_lower not in ("stop", "halt", "cancel", "never mind", "nevermind", "abort"):
            # Store this as the last meaningful task
            _last_task = text
            _last_task_target = _extract_target(text)

        # Task-level room scan: build vector map + best room guess before acting.
        # Pass all targets so scan can stop early if it spots any of them.
        _scan_target_found = None
        _scan_target_pan = None
        nav_target = _extract_target(text)
        all_targets = _extract_all_targets(text)
        # Use the last target as the scan find_target (it's usually the
        # most specific — e.g. "fridge" in "go to kitchen and find fridge")
        _scan_find = all_targets[-1] if all_targets else nav_target
        if _room_scanner:
            scan_skip = {"stop", "halt", "cancel", "never mind", "nevermind", "abort"}
            if text.strip().lower() not in scan_skip:
                try:
                    scan_state = _run_task_room_scan(
                        text, find_target=_scan_find)
                    if scan_state:
                        guess = scan_state.get("room_guess", {})
                        g_name = guess.get("name") or "unknown"
                        g_conf = float(guess.get("confidence", 0.0))
                        log_event("room",
                                  f"Best room guess for task '{text}': "
                                  f"{g_name} ({g_conf:.2f})")
                        # Check if target was spotted during sweep
                        if scan_state.get("target_found"):
                            _scan_target_found = scan_state["target_found"]
                            _scan_target_pan = scan_state.get("target_pan", 0)
                            log_event("room",
                                f"Target '{_scan_target_found}' spotted "
                                f"at pan={_scan_target_pan}° during scan!")
                except Exception as e:
                    log_event("error", f"Room vector scan failed: {e}")

        # ── Navigator shortcut for movement commands ──
        if _navigator:
            text_lower_strip = text.strip().lower()
            is_nav = any(text_lower_strip.startswith(p) for p in (
                "go to ", "go out", "go through",
                "navigate to ", "drive to ", "move to ",
                "head toward", "get to ", "get out",
                "exit ", "leave "))
            is_search = nav_target and any(text_lower_strip.startswith(p) for p in (
                "find ", "look for ", "search for ", "where is "))
            # If no explicit target extracted, use the full command as the goal
            if is_nav and not nav_target:
                nav_target = text.strip()
            if is_nav or is_search:
                plan_active.set()
                stop_event.clear()

                # Search-only mode: just find, don't drive
                search_only = is_search and not is_nav and "go " not in text_lower_strip
                if search_only:
                    log_event("plan",
                              f"Navigator: searching for '{nav_target}'")
                    direction = _navigator.search(nav_target)
                    if direction is not None:
                        plan_active.clear()
                        log_event("plan",
                                  f"Navigator: found '{nav_target}'")
                        continue
                    log_event("plan",
                              f"Navigator: '{nav_target}' not visible, "
                              f"exploring")

                # If target was spotted during room scan, orient and go directly
                if _scan_target_found and _scan_target_pan is not None:
                    log_event("plan",
                        f"Target '{_scan_target_found}' visible at "
                        f"pan={_scan_target_pan}° — driving directly")
                    # Turn body to face where we saw it
                    _navigator._spin_body(int(_scan_target_pan))
                    _navigator._move_gimbal(0, 0)
                    import time as _t; _t.sleep(0.3)
                    # Navigate reactively toward it
                    reached = _navigator.navigate_reactive(nav_target)
                    plan_active.clear()
                    stop_event.clear()
                    log_event("plan",
                        f"Navigator: "
                        f"{'reached' if reached else 'could not reach'}"
                        f" '{nav_target}'")
                    continue

                # Topological navigation: plan route through room graph,
                # then navigate each leg (transition) reactively
                reached = _topo_navigate(nav_target, stop_event)

                plan_active.clear()
                stop_event.clear()
                log_event("plan",
                    f"Navigator: "
                    f"{'reached' if reached else 'could not reach'}"
                    f" '{nav_target}'")
                continue

        # ── Follow shortcut: bypass LLM, pure YOLO visual servo ──
        _follow_words = {"follow", "following"}
        if _follow_words & _clean_words(text):
            plan_active.set()
            stop_event.clear()
            # Extract target from text: "follow me" → "person", "follow the dog" → "dog"
            import string as _string
            words = [w.strip(_string.punctuation) for w in text.strip().lower().split()]
            words = [w for w in words if w]  # drop empty after stripping
            _filler = {"the", "that", "this", "a", "an", "my"}
            _rest = [w for w in words if w not in {"follow", "following"} | _filler]
            _ftarget = "person"
            if _rest:
                w0 = _rest[0]
                if w0 in ("me", "person", "human", "owner", "man", "woman"):
                    _ftarget = "person"
                else:
                    _ftarget = w0
            log_event("follow", f"YOLO follow mode: {text} (target={_ftarget})")
            if not yolo_enabled:
                _set_yolo_enabled(True)
                log_event("follow", "Auto-enabled YOLO for follow mode")
            cam._follow_mode = True  # YOLO every frame during follow
            _speak("Following.", spk, mic_card)
            from follow_target import follow
            try:
                result = follow(_ftarget, ser, cam, _shared_refs.get("imu"),
                                duration=300,
                                stop_event=stop_event,
                                log_fn=lambda msg: log_event("follow", msg),
                                voice=_xai_voice, floor_nav=None,
                                recovery_fn=_follow_recovery,
                                speak_fn=lambda t: _speak(t, spk, mic_card),
                                llm_fn=_follow_llm_fn,
                                label_override_fn=_follow_label_override)
            finally:
                cam._follow_mode = False  # restore 3-frame interval
            log_event("follow", f"Done: {result.get('status')}")
            plan_active.clear()
            stop_event.clear()
            continue

        log_event("plan", f"Starting: {text}")
        result = run_plan(text, ser, cam, spk, mic_card)

        if result is None:
            continue

if __name__ == "__main__":
    main()
