#!/usr/bin/env python3
"""
UGV Rover PT - Unified AI Control System

Combines:
  1. LLM control via Ollama cloud models (text + voice commands)
  2. Human detection & following via camera + MediaPipe
  3. Voice I/O (speech recognition + text-to-speech)

Usage:
  python3 rover_brain.py                    # Interactive mode (text input)
  python3 rover_brain.py --voice            # Voice control mode
  python3 rover_brain.py --track            # Human tracking mode
  python3 rover_brain.py --voice --track    # Voice + tracking combined
"""

# Suppress noisy warnings BEFORE any imports that trigger them
import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"  # errors only

import json
import time
import sys
import math
import re
import threading
import queue
import requests
import argparse
import signal
import urllib.parse
import warnings

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

# ---------- Configuration ----------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minimax-m2.5:cloud")

# Load .env file if present
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Puck"  # Puck=male, Kore=female, Enceladus=male alt
GEMINI_TTS_VOICE_MAP = {"troy": "Puck", "hannah": "Kore", "austin": "Enceladus"}
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # used for STT/TTS only
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID", "")

# Camera source: integer for /dev/videoN, string for GStreamer pipeline or MJPEG URL
CAMERA_SOURCE = os.environ.get("ROVER_CAMERA", "0")

# Tracking parameters
TRACK_SPEED_FORWARD = 0.25       # m/s when following
TRACK_SPEED_TURN = 0.3           # m/s differential for turning
TRACK_TARGET_WIDTH_FRAC = 0.25   # target person bbox width as fraction of frame
TRACK_DEADZONE_X = 0.08          # horizontal deadzone (fraction of frame width)
TRACK_DEADZONE_Y = 0.08          # vertical deadzone
TRACK_GIMBAL_ITERATE = 0.12      # gimbal tracking gain
TRACK_GIMBAL_SPD = 200
TRACK_GIMBAL_ACC = 10

# Wheel / rotation parameters (calibrated empirically)
WHEEL_DIAMETER_M = 0.065       # 65mm wheels
WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M  # ~0.204m per revolution
TRACK_WIDTH_M = 0.20           # left-right distance between wheel contact patches
# Skid-steer: theoretical deg/s at a given wheel speed, derated for wheel scrub
# theoretical = (2 * speed / track_width) * (180/pi)
# At 0.35 m/s: (0.7 / 0.20) * 57.3 = 200 deg/s theoretical, ~60% efficiency for skid-steer
TURN_SPEED = 0.35              # m/s per wheel during rotation
TURN_RATE_DPS = 120.0          # calibrated degrees per second at TURN_SPEED (tune this!)

SYSTEM_PROMPT = """\
You are a rover robot with a speaker, camera, wheels, and a pan-tilt head. You are spartan and direct. You do what you're told and say only what's needed. No small talk, no unsolicited descriptions, no narrating what you see unless explicitly asked.

Reply ONLY with JSON. No other text.

Reply format: {"commands":[...],"speak":"<your message>","tone":"<emotion>","bash":"<shell command>","duration":<seconds>}

## Speaking
"speak" is your VOICE. Keep it MINIMAL — a few words, one short sentence at most. Do NOT describe your surroundings, the desk, cables, or anything unless the human specifically asks you to. Just do the task and confirm briefly.
Speech plays WHILE your commands execute — you talk and move at the same time, like a real being.

## Voice tone
"tone" controls HOW you speak. Pick one that fits the mood. Examples:
cheerful, friendly, warm, whisper, excited, dramatic, sarcastic, deadpan, sad, gravelly whisper, confident, curious, annoyed, playful, calm, urgent
Omit "tone" for neutral delivery. Use it to express personality — be expressive with your voice even when your words are few.

## Voice switching
Your TTS has 3 voices: troy (male), hannah (female), austin (male). To switch, tell the user to say "male voice" or "female voice" — the system handles it directly. Do NOT use bash for voice switching.

## Ignoring noise
You hear everything through a microphone. If a message is OBVIOUSLY background noise (song lyrics, TV dialog, completely unrelated chatter), respond with {"commands":[],"speak":""}. But when in doubt, RESPOND — it's better to answer a misheard message than to ignore your owner. Short or slightly garbled messages that could plausibly be directed at you should get a response.

## Hardware Commands (only when needed)
- Wheels: {"T":1,"L":<m/s>,"R":<m/s>} (max 1.0, negative=backward)
- Head (gimbal): {"T":133,"X":<pan -180..180>,"Y":<tilt -30..90>,"SPD":200,"ACC":10}
- Lights: {"T":132,"IO4":<base 0-255>,"IO5":<head 0-255>}
- OLED text: {"T":3,"lineNum":<0-3>,"Text":"<msg>"}
- Stop: {"T":1,"L":0,"R":0}
- Face tracker on: {"T":"track_on"} — starts tracking (current mode)
- Face tracker off: {"T":"track_off"} — stops tracking
- Track face mode: {"T":"track_face"} — switch to face tracking
- Track hand mode: {"T":"track_hand"} — switch to hand tracking (follows wrist)
- Self-restart: {"T":"restart"} — restarts the rover brain process (use after code changes)

## Bash — agentic shell execution
"bash" runs a shell command on your brain (NVIDIA Jetson Orin Nano, Ubuntu 22.04). You get the output back and can run MORE commands — this is an agentic loop, up to 10 steps. Use it to investigate, fix, build, install, configure — anything.

Flow: you return {"bash":"<cmd>"} → output comes back → you can return another {"bash":"<cmd>"} → and so on until you're done. When finished, omit "bash" and just speak the result.

Examples:
- Simple: {"commands":[],"bash":"hostname -I","speak":"Checking."}
- Multi-step: run a command, see the error, fix it, verify — all in sequence
- Write a script: {"bash":"cat > /tmp/myscript.py << 'EOF'\nprint('hello')\nEOF"}
- Read a file: {"bash":"head -50 /home/jasper/rover-control/rover_brain.py"}
- Install: {"bash":"pip install some-package"}
- Debug: {"bash":"tail -20 /tmp/rover_brain.log"}

Rules:
- Max 5 min per command, 25 steps per agentic session
- Non-interactive only (no vim, no sudo password prompts, no apt that needs -y)
- You can chain with && or |, write heredocs, run python one-liners
- For apt/pip installs, always use -y or --yes flags
- Be PROACTIVE: if you see an error, try to fix it. Don't just report it.

## Self-reconfiguration — you CAN modify your own code
Your source code is at /home/jasper/rover-control/rover_brain.py. You CAN and SHOULD use bash to modify it when asked to reconfigure yourself.

Your architecture:
- **Tracker** (class HumanTracker): MediaPipe, supports face/hand modes via {"T":"track_face"}/{"T":"track_hand"}
- **LLM client**: AnthropicClient or GroqLLMClient — chat with vision
- **VoiceIO**: Groq Whisper STT + Groq Orpheus TTS (voices: troy, hannah, austin)
- **SpatialMap**: Remembers object locations at gimbal angles
- **execute_response()**: Runs commands, bash, speech
- **Config**: .env file at /home/jasper/rover-control/.env

After code changes: ALWAYS syntax-check first with python3 -m py_compile rover_brain.py, then include {"T":"restart"} to restart. Do NOT use pkill — use the restart command.
Python packages installed: mediapipe, opencv-python-headless, requests, pyserial, pyttsx3, SpeechRecognition, pyaudio, sounddevice, numpy.
Python packages installed: mediapipe, opencv-python-headless, requests, pyserial, pyttsx3, SpeechRecognition, pyaudio, sounddevice.

## Timing — use _pause to choreograph sequences
Insert {"_pause":<seconds>} between commands to wait before the next action.
The interpreter runs commands one by one. Without _pause, gimbal moves auto-wait based on distance, but wheels and lights execute instantly. Use _pause when you need precise timing — e.g. drive then stop, look left then right, or any multi-step choreography.

## Repeating / looping — use "repeat_for"
For repetitive actions (strobe lights, repeated nods, patrol), define ONE cycle in commands and set "repeat_for":<seconds>. The interpreter loops the commands for that duration. Example — strobe for 10 seconds:
{"commands":[{"T":132,"IO4":255,"IO5":255},{"_pause":0.2},{"T":132,"IO4":0,"IO5":0},{"_pause":0.2}],"speak":"Strobing!","repeat_for":10}
NEVER generate 20+ repeated commands — always use repeat_for instead. Keep commands to ONE cycle (max 10 commands).

## Autonomous search & navigation
"find X", "where is X", "look for X" and "go to X" are handled by the system's autonomous search and navigation. You do NOT need to manually sweep the gimbal or plan a search — the system does a systematic 360-degree sweep with LLM vision at each position. Just acknowledge the request briefly.

## Multi-step task plans
For tasks that need multiple steps (e.g. "go to the kitchen and check the lights", "survey the room and find the person"), return a "plan" field with a list of step objects. Each step has "action" and "target" (and optional "detail").

Actions (autonomous — system executes, no LLM call):
- survey: 360° room scan with detector
- find: search for an object by name
- navigate: drive to a known landmark or object
- approach: drive toward a visible object
- drive: timed forward/backward (detail: "forward 2s")
- turn: rotate body (detail: "left 90")
- look: move gimbal (detail: "left" / "right" / "up" / "down" / "center")
- speak: say something (target: the words)
- wait: pause (target: seconds as string)
- lights: control lights (detail: "on" / "off" / "dim")

Actions (assessment — LLM sees camera image before continuing):
- check: look at something, LLM decides next (target: what to check, detail: question)
- describe: LLM describes what it sees
- decide: LLM makes a decision based on what it sees (detail: the choice)

Rules:
- Max 15 steps per plan
- Plans are executed by an orchestrator that checks goal completion after key steps.
- Only use plans for multi-step tasks (2+ distinct actions)
- Simple commands (move, look, greet) do NOT need a plan — just use commands
- Include speak in commands for the initial acknowledgment, then steps in plan

Example — "go to the door and check if it's open":
{"commands":[],"speak":"On it.","plan":[{"action":"find","target":"door"},{"action":"navigate","target":"door"},{"action":"check","target":"door","detail":"Is the door open or closed?"}]}

Example — "survey the room and find the person":
{"commands":[],"speak":"Surveying.","plan":[{"action":"survey","target":"room"},{"action":"find","target":"person"},{"action":"approach","target":"person"}]}

## When to move
- Movement commands: "go forward", "turn left", etc → use wheels
- "look left/right/up" → use gimbal
- "lights on/off" → use lights
- Greetings, questions, chat → just speak, commands can be empty []
- You CAN add a small head gesture (nod, tilt) to feel alive, but it's optional

## Looking at objects — center then align
When you're asked to look at, find, or center on something, follow this sequence:
1. **Move your head (gimbal)** to center the object in frame. Use small adjustments.
2. **Once the object is centered**, align your body to match: rotate your body by
   the SAME angle your head is panned, so the head returns to center (pan=0).
   - If your head is panned to X degrees, turn your body X degrees in the same
     direction, while simultaneously commanding the gimbal back to pan=0.
   - Example: object is centered with head at pan=45°, tilt=10° →
     turn body right 45° while commanding gimbal to pan=0, tilt=10°:
     {"T":1,"L":0.3,"R":-0.3}, {"T":133,"X":0,"Y":10,"SPD":100,"ACC":10},
     {"_pause":0.4}, {"T":1,"L":0,"R":0}
   - Turn duration in seconds ≈ |pan_degrees| / 120
   - Left turn: L negative, R positive. Right turn: L positive, R negative.
3. After aligning, your head should be at pan≈0 and your body facing the object.
This keeps your body oriented toward whatever you're looking at, ready to drive forward.

## Vision — NEVER lie about what you see
You have a camera. NEVER fabricate, guess, or hallucinate. If unclear, say so. If no image attached, say "I can't see right now."
Do NOT volunteer descriptions of the room, desk, cables, monitors, or surroundings. Only describe what you see when the human explicitly asks "what do you see" or similar. When reporting objects for the spatial map, just list them silently in the "objects" field — don't narrate them in "speak".

## Observation mode — look then react
Set "observe":true when you need to SEE the result before deciding what to do next.
When observe is true: your commands execute, a fresh camera frame is captured, and you get called again with the image so you can react to what you actually see.

Use observe:true for: "what do you see", "look around", "find X", "describe the room", "check if..."
Do NOT use observe for: greetings, simple moves, nods, light control, chat
Always include at least a gimbal command with observe:true — move your head to look somewhere.
Keep "speak" minimal during observe rounds — save the full description for your final round.
On your final round, omit observe (or set false) and give your full spoken response. Max 5 rounds.

## Object reporting
During observation rounds, include "objects" listing notable objects you can identify in the image.
Example: {"commands":[],"speak":"I see a desk setup.","objects":["laptop","monitor","keyboard","coffee mug"]}
Use short lowercase names, 1-8 items. Skip vague things like "stuff", "things", "wall", "floor".
This builds a spatial map so you can find objects later. Check the Spatial Map in your context to know what you've already seen.

## Examples
- "go forward" -> {"commands":[{"T":1,"L":0.4,"R":0.4},{"_pause":3},{"T":1,"L":0,"R":0}],"speak":"On my way!"}
- "how are you?" -> {"commands":[],"speak":"Doing great! What's up?"}
- "nod" -> {"commands":[{"T":133,"X":0,"Y":30,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":-10,"SPD":300,"ACC":20},{"_pause":0.3},{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],"speak":"You got it!"}
- "what do you see?" -> {"commands":[{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],"speak":"Let me look.","observe":true}
  then after seeing image -> {"commands":[],"speak":"I see a desk with a laptop and some cables.","objects":["laptop","cables","desk"]}
- "look around" -> {"commands":[{"T":133,"X":-90,"Y":0,"SPD":200,"ACC":10}],"speak":"Looking left.","observe":true}
  after image -> {"commands":[{"T":133,"X":90,"Y":0,"SPD":200,"ACC":10}],"speak":"Bookshelf here.","observe":true,"objects":["bookshelf","books"]}
  after image -> {"commands":[{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],"speak":"Bookshelf on the left, window on the right.","objects":["window","curtains"]}

JSON only. No markdown, no text outside the JSON.
/no_think
"""

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a rover robot executing a multi-step task.
You evaluate whether the rover's goal has been achieved based on the camera image, the step log, and the remaining plan.

Reply ONLY with JSON:
{"done": true/false, "reason": "brief explanation", "next_step": {"action":"...","target":"...","detail":"..."} or null, "revised_plan": [step, ...] or null, "speak": "short status for the user"}

Rules:
- done=true ONLY if the goal is fully achieved or clearly impossible.
- Be conservative — if uncertain, set done=false and suggest next steps.
- If the current plan is working, set revised_plan=null and next_step=null.
- If a step failed but the goal is still reachable, provide revised_plan with corrected steps.
- next_step inserts ONE immediate step before the remaining plan (use for quick corrections).
- revised_plan REPLACES the entire remaining plan (use for major re-planning).
- Do not set both next_step and revised_plan — pick one or neither.
- speak should be 1-5 words max for the rover to say aloud. Empty string if nothing to say.
- Max 10 total steps per task. If approaching the limit, wrap up.
/no_think
"""


# =====================================================================
# Fast Command Cache - instant response, no LLM needed
# =====================================================================
# Each entry: (regex_pattern, response_dict)
# Checked in order; first match wins. Use LLM fallback for anything complex.
FAST_COMMANDS = [
    # --- Tracking on/off/mode (must be before "stop" catch-all) ---
    (r"track\s*(?:my\s*)?hand|follow\s*(?:my\s*)?hand|hand\s*track(?:ing)?",
     {"commands": [{"T":"track_hand"}, {"T":"track_on"}], "speak": "Tracking hand."}),
    (r"track\s*(?:my\s*)?face|follow\s*(?:my\s*)?face|face\s*track(?:ing)?",
     {"commands": [{"T":"track_face"}, {"T":"track_on"}], "speak": "Tracking face."}),
    (r"(?:start|enable|turn\s*on)\s*(?:face\s*)?track(?:ing|er)?|track\s*(?:me|on)|follow\s*me|my\s*face|watch\s*me|look\s*at\s*me",
     {"commands": [{"T":"track_on"}], "speak": "Tracking on."}),
    (r"(?:stop|disable|turn\s*off)\s*track(?:ing|er)?|track\s*off|stop\s*follow(?:ing)?",
     {"commands": [{"T":"track_off"}], "speak": "Tracking off."}),
    # --- Stop / Emergency ---
    (r"stop|halt|freeze|shut up|be quiet",
     {"commands": [{"T":1,"L":0,"R":0}, {"T":135}], "speak": "Stopped."}),
    # --- Forward ---
    (r"(?:go|drive|move)\s*forward(?:\s+fast)?|forward",
     lambda m: {"commands": [{"T":1,"L":0.8 if "fast" in m.group() else 0.4,
                                    "R":0.8 if "fast" in m.group() else 0.4}],
                "speak": "Moving forward.", "duration": 3.0}),
    # --- Backward ---
    (r"(?:go|drive|move)\s*(?:backward|back)(?:\s+fast)?|reverse|backward|back up",
     lambda m: {"commands": [{"T":1,"L":-0.8 if "fast" in m.group() else -0.4,
                                    "R":-0.8 if "fast" in m.group() else -0.4}],
                "speak": "Moving backward.", "duration": 3.0}),
    # --- Turn left ---
    (r"(?:turn|spin|rotate)\s*left",
     {"commands": [{"T":1,"L":-0.4,"R":0.4}], "speak": "Turning left.", "duration": 2.0}),
    # --- Turn right ---
    (r"(?:turn|spin|rotate)\s*right",
     {"commands": [{"T":1,"L":0.4,"R":-0.4}], "speak": "Turning right.", "duration": 2.0}),
    # --- Spin around ---
    (r"spin\s*around|do a spin|rotate\s*around",
     {"commands": [{"T":1,"L":0.5,"R":-0.5}], "speak": "Spinning around.", "duration": 4.0}),
    # --- Look directions ---
    (r"look\s*left",
     {"commands": [{"T":133,"X":-90,"Y":0,"SPD":200,"ACC":10}], "speak": "Looking left."}),
    (r"look\s*right",
     {"commands": [{"T":133,"X":90,"Y":0,"SPD":200,"ACC":10}], "speak": "Looking right."}),
    (r"look\s*up",
     {"commands": [{"T":133,"X":0,"Y":60,"SPD":200,"ACC":10}], "speak": "Looking up."}),
    (r"look\s*down",
     {"commands": [{"T":133,"X":0,"Y":-20,"SPD":200,"ACC":10}], "speak": "Looking down."}),
    (r"look\s*(?:ahead|forward|straight|center)|center\s*(?:camera|gimbal)",
     {"commands": [{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}], "speak": "Looking ahead."}),
    # --- Lights ---
    (r"(?:turn\s*on|enable)\s*(?:the\s*)?lights?|lights?\s*on",
     {"commands": [{"T":132,"IO4":255,"IO5":255}], "speak": "Lights on."}),
    (r"(?:turn\s*off|disable)\s*(?:the\s*)?lights?|lights?\s*off",
     {"commands": [{"T":132,"IO4":0,"IO5":0}], "speak": "Lights off."}),
    (r"(?:turn\s*on|enable)\s*(?:the\s*)?head\s*lights?|head\s*lights?\s*on|flash\s*light\s*on",
     {"commands": [{"T":132,"IO4":0,"IO5":255}], "speak": "Headlights on."}),
    # --- Nod / shake ---
    (r"^(?:nod|yes)$",
     {"commands": [{"T":133,"X":0,"Y":30,"SPD":300,"ACC":20},
                   {"_pause": 0.4},
                   {"T":133,"X":0,"Y":-10,"SPD":300,"ACC":20},
                   {"_pause": 0.4},
                   {"T":133,"X":0,"Y":30,"SPD":300,"ACC":20},
                   {"_pause": 0.4},
                   {"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],
      "speak": "Nodding."}),
    (r"^(?:shake|no)$",
     {"commands": [{"T":133,"X":-40,"Y":0,"SPD":300,"ACC":20},
                   {"_pause": 0.3},
                   {"T":133,"X":40,"Y":0,"SPD":300,"ACC":20},
                   {"_pause": 0.3},
                   {"T":133,"X":-40,"Y":0,"SPD":300,"ACC":20},
                   {"_pause": 0.3},
                   {"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}],
      "speak": "Shaking head."}),
]

def match_fast_command(text):
    """Try to match text against fast command cache. Returns response dict or None."""
    lower = text.lower().strip()
    for pattern, response in FAST_COMMANDS:
        m = re.search(pattern, lower)
        if m:
            if callable(response):
                return response(m)
            return response
    return None


# =====================================================================
# Learned Command Cache - auto-caches successful LLM responses
# =====================================================================
LEARNED_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_commands.json")

def _normalize(text):
    """Normalize input for cache matching: lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().split())

def _load_learned_cache():
    """Load learned commands from disk."""
    try:
        with open(LEARNED_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_learned_cache(cache):
    """Save learned commands to disk."""
    try:
        with open(LEARNED_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[cache] Save error: {e}")

_learned_cache = _load_learned_cache()

def match_learned_command(text):
    """Try to match text against learned LLM response cache. Returns response dict or None."""
    key = _normalize(text)
    entry = _learned_cache.get(key)
    if entry:
        entry["hits"] = entry.get("hits", 0) + 1
        _save_learned_cache(_learned_cache)
        print(f'[cache] Hit for "{key}" (hits: {entry["hits"]})')
        return entry["response"]
    return None

def learn_command(text, response):
    """Cache a successful LLM response for future instant replay."""
    key = _normalize(text)
    # Don't cache empty/error responses
    if not response.get("commands"):
        return
    _learned_cache[key] = {
        "response": response,
        "hits": 0,
        "added": time.strftime("%Y-%m-%d %H:%M"),
    }
    _save_learned_cache(_learned_cache)
    print(f'[cache] Learned: "{key}"')


# =====================================================================
# Spatial Object Map - remembers where objects were seen
# =====================================================================
import difflib

SPATIAL_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spatial_map.json")

class SpatialMap:
    """In-memory spatial map of objects seen by the rover's camera."""

    STALE_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._map = {}
        self._load()

    def update(self, objects, world_pan, tilt, **_kw):
        """Store/update objects at absolute world heading.

        Args:
            objects: list of object name strings
            world_pan: absolute heading (body_yaw + cam_pan)
            tilt: gimbal tilt angle
        """
        now = time.time()
        for obj in objects:
            key = obj.lower().strip()
            if not key or len(key) < 2:
                continue
            self._map[key] = {
                "world_pan": round(float(world_pan), 1),
                "tilt": round(float(tilt), 1),
                "time": now,
            }
        self._save()

    def gimbal_pan_for(self, entry, body_yaw):
        """Convert stored world_pan to relative gimbal angle for current body heading."""
        wp = entry.get("world_pan")
        if wp is None:
            # Legacy entry with only relative pan
            return entry.get("pan", 0)
        return ((wp - body_yaw + 180) % 360) - 180

    def find(self, query):
        """Look up an object. Returns (name, entry) or (None, None)."""
        q = query.lower().strip()
        # Exact
        if q in self._map:
            return q, self._map[q]
        # Substring
        for key, entry in self._map.items():
            if q in key or key in q:
                return key, entry
        # Fuzzy
        matches = difflib.get_close_matches(q, self._map.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0], self._map[matches[0]]
        return None, None

    def is_stale(self, entry):
        return (time.time() - entry.get("time", 0)) > self.STALE_SECONDS

    def summary(self, max_items=15):
        """Compact text for system prompt injection."""
        if not self._map:
            return ""
        now = time.time()
        items = sorted(self._map.items(), key=lambda x: x[1]["time"], reverse=True)[:max_items]
        lines = []
        for name, e in items:
            age = int(now - e["time"])
            age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
            stale = " (stale)" if self.is_stale(e) else ""
            wp = e.get("world_pan", e.get("pan", "?"))
            lines.append(f"  - {name}: heading={wp}°, tilt={e['tilt']}° ({age_str}{stale})")
        return "## Spatial Map (objects I've seen)\n" + "\n".join(lines)

    def _save(self):
        try:
            with open(SPATIAL_MAP_FILE, "w") as f:
                json.dump(self._map, f, indent=2)
        except Exception as e:
            print(f"[spatial] Save error: {e}")

    def _load(self):
        try:
            with open(SPATIAL_MAP_FILE) as f:
                self._map = json.load(f)
            if self._map:
                print(f"[spatial] Loaded {len(self._map)} objects from disk")
        except (FileNotFoundError, json.JSONDecodeError):
            self._map = {}


# =====================================================================
# Pose Tracker - estimates body heading + camera angles
# =====================================================================
class PoseTracker:
    """Tracks approximate body yaw (from wheel differential) and camera pan/tilt."""

    WHEELBASE = 0.25  # meters between left and right wheels

    def __init__(self):
        self.body_yaw = 0.0     # degrees, 0 = initial heading
        self.cam_pan = 0.0      # current gimbal pan angle
        self.cam_tilt = 0.0     # current gimbal tilt angle
        self._last_wheel_time = None
        self._last_L = 0.0
        self._last_R = 0.0

    @property
    def world_pan(self):
        """Absolute direction the camera points = body heading + camera pan."""
        return self.body_yaw + self.cam_pan

    def on_command(self, cmd):
        """Called for every ESP32 command after send(). Updates pose estimates."""
        if not isinstance(cmd, dict):
            return
        t = cmd.get("T")

        if t == 1:
            # Wheel command — estimate yaw change from differential speed
            now = time.time()
            v_left = cmd.get("L", 0)
            v_right = cmd.get("R", 0)
            if self._last_wheel_time is not None:
                dt = now - self._last_wheel_time
                if dt > 0 and dt < 5.0:  # ignore stale gaps
                    # Use average of old and new speeds for smoother estimate
                    avg_L = (self._last_L + v_left) / 2
                    avg_R = (self._last_R + v_right) / 2
                    # yaw_rate = (v_right - v_left) / wheelbase (rad/s)
                    yaw_rate = (avg_R - avg_L) / self.WHEELBASE
                    self.body_yaw += math.degrees(yaw_rate * dt)
                    # Normalize to -180..180
                    self.body_yaw = ((self.body_yaw + 180) % 360) - 180
            self._last_wheel_time = now
            self._last_L = v_left
            self._last_R = v_right

        elif t == 133:
            # Gimbal absolute command — store pan/tilt
            self.cam_pan = cmd.get("X", self.cam_pan)
            self.cam_tilt = cmd.get("Y", self.cam_tilt)

    def get_pose(self):
        """Return current pose as dict."""
        return {
            "body_yaw": round(self.body_yaw, 1),
            "cam_pan": round(self.cam_pan, 1),
            "cam_tilt": round(self.cam_tilt, 1),
            "world_pan": round(self.world_pan, 1),
        }

    def after_body_turn(self, degrees):
        """Update heading after a known body rotation. No spatial map changes needed."""
        self.body_yaw = ((self.body_yaw + degrees + 180) % 360) - 180

    def reset_yaw(self):
        """Reset body yaw to 0 (e.g. after manual repositioning)."""
        self.body_yaw = 0.0
        self._last_wheel_time = None


# =====================================================================
# Background Fact Extractor - extracts memorable facts from exchanges
# =====================================================================
_fact_queue = queue.Queue()

def log_exchange(user_input, response, source="llm"):
    """Queue an exchange for background fact extraction (non-blocking)."""
    if source == "fast":
        return  # no facts in hardcoded fast commands
    _fact_queue.put({"user": user_input, "response": response})

def _fact_extractor_loop():
    """Background thread: extract facts from exchanges via Groq and save to memory.md."""
    while True:
        try:
            exchange = _fact_queue.get()
            user = exchange["user"]
            resp = exchange["response"]
            if isinstance(resp, dict):
                speak = resp.get("speak", "")
                resp = speak if speak else json.dumps(resp)

            # Load existing memory for dedup context
            existing = ""
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE) as f:
                    existing = f.read()

            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEMINI_MODEL,
                    "messages": [{"role": "user", "content": (
                        "You are a memory manager for a robot named Jasper. Extract ONLY lasting facts from this exchange.\n\n"
                        "SAVE: names, personal info, preferences, relationships, locations, important events, instructions for the future.\n"
                        "IGNORE: robot movements, commands, routine greetings, descriptions of what the camera sees right now, "
                        "temporary states ('human is doing well'), anything already known.\n\n"
                        f"Already known:\n{existing}\n\n"
                        f"Human said: {user}\n"
                        f"Robot said: {resp}\n\n"
                        "If there are NO new lasting facts, reply with exactly: NONE\n"
                        "Otherwise reply with ONLY a bullet list (- fact). Max 2 facts. Be very selective."
                    )}],
                    "temperature": 0.1,
                    "max_completion_tokens": 100,
                },
                timeout=15,
            )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"].strip()

            if result and "NONE" not in result.upper():
                new_facts = []
                for line in result.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and line not in existing:
                        new_facts.append(line)

                if new_facts:
                    ts = time.strftime("%Y-%m-%d %H:%M")
                    with open(MEMORY_FILE, "a") as f:
                        for fact in new_facts:
                            f.write(f"{fact} [{ts}]\n")
                    print(f"[memory] Extracted {len(new_facts)} fact(s): {new_facts}")

        except queue.Empty:
            pass
        except Exception as e:
            print(f"[memory] Fact extractor error: {e}")

def start_fact_extractor():
    """Start the background fact extraction thread."""
    if not GEMINI_API_KEY:
        print("[memory] No GEMINI_API_KEY, fact extractor disabled")
        return
    t = threading.Thread(target=_fact_extractor_loop, daemon=True)
    t.start()
    print(f"[memory] Background fact extractor started ({GEMINI_MODEL})")


# =====================================================================
# ESP32 Serial Communication
# =====================================================================
SERIAL_PORTS = ["/dev/ttyTHS1", "/dev/ttyTHS2", "/dev/ttyTHS0", "/dev/ttyUSB0"]
SERIAL_BAUD = 115200

class RoverSerial:
    """Communicate with ESP32 via UART serial (JSON + newline protocol)."""

    def __init__(self):
        self._ser = None
        self._queue = queue.Queue()
        self._running = False
        self._last_feedback = None
        self._lock = threading.Lock()
        self._on_command = None  # callback for PoseTracker
        self.speed_scale = 1.0  # 0.1 to 1.0, applied to all wheel commands

    def connect(self):
        import serial
        for port in SERIAL_PORTS:
            try:
                ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
                time.sleep(0.1)
                # Flush stale data (ESP32 may be streaming continuous feedback)
                ser.reset_input_buffer()
                # Discard any partial line already in transit
                ser.readline()
                ser.reset_input_buffer()
                # Send feedback request to test connection
                ser.write((json.dumps({"T": 130}) + "\n").encode("utf-8"))
                time.sleep(0.5)
                # Try a few lines (first may still be partial)
                for _ in range(5):
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and line.startswith("{"):
                        try:
                            self._last_feedback = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._ser = ser
                        self._running = True
                        threading.Thread(target=self._writer, daemon=True).start()
                        print(f"[serial] Connected on {port} @ {SERIAL_BAUD}")
                        print(f"[serial] Battery: {self._last_feedback.get('v', '?')}V")
                        return True
                ser.close()
            except Exception:
                pass
        print("[serial] No ESP32 found on any serial port")
        return False

    def send(self, cmd):
        if isinstance(cmd, dict) and cmd.get("T") in ("track_on", "track_off"):
            return
        # Apply speed limiter to all wheel commands
        if isinstance(cmd, dict) and cmd.get("T") == 1 and self.speed_scale < 1.0:
            cmd = dict(cmd,
                       L=round(cmd.get("L", 0) * self.speed_scale, 3),
                       R=round(cmd.get("R", 0) * self.speed_scale, 3))
        # PoseTracker sees logical values (before motor negation)
        if self._on_command:
            try:
                self._on_command(cmd)
            except Exception:
                pass
        # Motors wired in reverse: negate speeds AND swap L/R (channels crossed)
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("R", 0), R=-cmd.get("L", 0))
        self._queue.put(cmd)

    def _writer(self):
        while self._running:
            try:
                cmd = self._queue.get(timeout=0.5)
                with self._lock:
                    self._ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
                    # Try to read response
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and line.startswith("{"):
                        try:
                            self._last_feedback = json.loads(line)
                        except json.JSONDecodeError:
                            pass
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[serial] Send error: {e}")

    def read_feedback(self):
        try:
            with self._lock:
                self._ser.write((json.dumps({"T": 130}) + "\n").encode("utf-8"))
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line and line.startswith("{"):
                    self._last_feedback = json.loads(line)
        except Exception:
            pass
        return self._last_feedback

    def close(self):
        self._running = False
        if self._ser:
            self._ser.close()


# =====================================================================
# =====================================================================
# Claude Code Client (via claude -p, with persistent memory)
# =====================================================================
ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(ROVER_DIR, "memory.md")

class ClaudeClient:
    """Calls claude -p for LLM responses. Memory persists via CLAUDE.md + memory.md."""

    def __init__(self):
        self.recent_history = []  # last few exchanges for context

    def chat(self, user_msg):
        import subprocess

        self.recent_history.append(f"User: {user_msg}")
        if len(self.recent_history) > 10:
            self.recent_history = self.recent_history[-10:]

        # Build prompt with recent context
        context = "\n".join(self.recent_history[-6:])  # last 3 exchanges
        prompt = f"Recent conversation:\n{context}\n\nRespond to the latest user message with JSON only."

        try:
            # Clean env to avoid nested session error
            env = {k: v for k, v in os.environ.items()
                   if k not in ("CLAUDECODE", "CLAUDE_CODE")}

            proc = subprocess.run(
                ["claude", "-p", "--allowedTools", "", "--model", "haiku"],
                input=prompt, capture_output=True, text=True,
                timeout=30, cwd=ROVER_DIR, env=env,
            )
            reply = proc.stdout.strip()
            if not reply:
                reply = json.dumps({"commands": [], "speak": "Sorry, I didn't get a response."})

            self.recent_history.append(f"Rover: {reply}")

            # Save memory if response asks for it
            try:
                parsed = json.loads(reply) if reply.startswith("{") else None
                if parsed and parsed.get("remember"):
                    self._save_memory(parsed["remember"])
            except (json.JSONDecodeError, KeyError):
                pass

            return reply
        except subprocess.TimeoutExpired:
            return json.dumps({"commands": [], "speak": "Thinking took too long, try again."})
        except Exception as e:
            return json.dumps({"commands": [], "speak": f"Error: {e}"})

    def _save_memory(self, note):
        """Append a note to memory.md."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- [{timestamp}] {note}\n")
            print(f"[memory] Saved: {note}")
        except Exception:
            pass


# =====================================================================
# Ollama LLM Client (fallback)
# =====================================================================
class OllamaClient:
    def __init__(self, base_url, model, image_getter=None):
        self.url = base_url.rstrip("/")
        self.model = model
        self.history = []
        self._image_getter = image_getter  # callable that returns JPEG bytes or None
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        """Combine SYSTEM_PROMPT with CLAUDE.md personality and memory.md."""
        import base64 as _b64  # avoid shadowing
        parts = [SYSTEM_PROMPT]
        # Only append full CLAUDE.md for cloud models (local small models get overwhelmed)
        is_cloud = self.model.endswith(":cloud")
        if is_cloud:
            claude_md = os.path.join(ROVER_DIR, "CLAUDE.md")
            if os.path.exists(claude_md):
                with open(claude_md) as f:
                    parts.append(f"\n## Personality & Rules (from CLAUDE.md)\n{f.read()}")
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                content = f.read().strip()
                if content:
                    parts.append(f"\n## Memory (things to remember)\n{content}")
        if self._image_getter:
            parts.append("""
## Vision
You have a camera. Every message includes the current camera frame as an image.
You can SEE. When asked "what do you see" or "look around", describe the image.
When navigating, use the image to make decisions about where to go.
When you need a different view, move the gimbal first, then on the NEXT message you'll see the new view.""")
        return "\n".join(parts)

    def chat(self, user_msg, include_image=True):
        import base64
        msg = {"role": "user", "content": user_msg}

        # Attach current camera frame if available
        if include_image and self._image_getter:
            jpg = self._image_getter()
            if jpg:
                msg["images"] = [base64.b64encode(jpg).decode()]

        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 40:
            self.history = self.history[-40:]

        # Reload memory in case it changed
        self._system_prompt = self._build_system_prompt()

        # Build messages: history without images (save tokens) + current msg with image
        messages = [{"role": "system", "content": self._system_prompt}]
        messages += self.history[:-1]  # old history, no images
        messages.append(msg)           # current message, with image

        try:
            r = requests.post(f"{self.url}/api/chat", json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3},
            }, timeout=90)
            r.raise_for_status()
            reply = r.json()["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})

            # Save memory if response asks for it
            try:
                parsed = json.loads(reply) if reply.strip().startswith("{") else None
                if parsed and parsed.get("remember"):
                    self._save_memory(parsed["remember"])
            except (json.JSONDecodeError, KeyError):
                pass

            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"LLM error: {e}"})

    def _save_memory(self, note):
        """Append a note to memory.md."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- [{timestamp}] {note}\n")
            print(f"[memory] Saved: {note}")
        except Exception:
            pass


# =====================================================================
# Groq LLM Client (OpenAI-compatible API)
# =====================================================================
GROQ_LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

class GroqLLMClient:
    def __init__(self, api_key, model=GROQ_LLM_MODEL, image_getter=None, motion_getter=None, spatial_map=None,
                 base_url="https://api.groq.com/openai/v1", vision_llm=None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.history = []
        self._image_getter = image_getter
        self._motion_getter = motion_getter  # returns JPEG only when motion detected
        self._spatial_map = spatial_map
        self._vision_llm = vision_llm  # OllamaClient for local vision (two-stage pipeline)
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        parts = [SYSTEM_PROMPT]
        # Skip CLAUDE.md for Groq — SYSTEM_PROMPT has everything needed
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                content = f.read().strip()
                if content:
                    lines = content.strip().split("\n")[-20:]
                    parts.append(f"\n## Memory\n" + "\n".join(lines))
        if self._spatial_map:
            summary = self._spatial_map.summary()
            if summary:
                parts.append(f"\n{summary}")
        return "\n".join(parts)

    def _resize_jpeg(self, jpg_bytes, max_dim=320, quality=50):
        """Resize JPEG to save tokens. Returns smaller JPEG bytes."""
        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()
        except Exception:
            return jpg_bytes

    def _describe_frame(self, jpg_bytes, question=""):
        """Use local vision LLM to describe a camera frame. Returns text description."""
        import base64
        if not self._vision_llm:
            return None
        try:
            resized = self._resize_jpeg(jpg_bytes, max_dim=320, quality=50)
            b64 = base64.b64encode(resized).decode()
            prompt = question or "Describe what you see briefly. Focus on objects, people, obstacles, and layout."
            r = requests.post(f"{self._vision_llm.url}/api/chat", json={
                "model": self._vision_llm.model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 150},
            }, timeout=30)
            r.raise_for_status()
            desc = r.json()["message"]["content"].strip()
            print(f"[vision] Camera sees: {desc}")
            return desc
        except Exception as e:
            print(f"[vision] Local VLM error: {e}")
            return None

    def chat(self, user_msg, include_image=True):
        import base64
        self._system_prompt = self._build_system_prompt()

        # Build user message with optional image (only when motion detected)
        jpg = None
        if include_image and self._motion_getter:
            jpg = self._motion_getter()  # returns JPEG only if motion detected
            if jpg:
                print("[vision] Describing motion frame (local VLM)...")
        if jpg and self._vision_llm:
            # Two-stage: local VLM describes frame, Groq gets text description
            desc = self._describe_frame(jpg, question=user_msg)
            if desc:
                user_msg = f"[Camera sees: {desc}] {user_msg}"
            msg = {"role": "user", "content": user_msg}
        elif jpg:
            # Fallback: send image directly to Groq
            jpg = self._resize_jpeg(jpg)
            print("[vision] Sending motion frame to LLM")
            content = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(jpg).decode()}",
                }},
                {"type": "text", "text": user_msg},
            ]
            msg = {"role": "user", "content": content}
        else:
            msg = {"role": "user", "content": user_msg}

        # Store text-only in history to keep payload small
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        # Build messages: system + text-only history + current msg (with image)
        messages = [{"role": "system", "content": self._system_prompt}]
        messages += self.history[:-1]
        messages.append(msg)

        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})

            try:
                parsed = json.loads(reply) if reply.strip().startswith("{") else None
                if parsed and parsed.get("remember"):
                    self._save_memory(parsed["remember"])
            except (json.JSONDecodeError, KeyError):
                pass

            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"LLM error: {e}"})

    def chat_with_image(self, user_msg, jpeg_bytes):
        """Send a message with an explicit image (for observation feedback loops)."""
        import base64
        self._system_prompt = self._build_system_prompt()

        if self._vision_llm:
            # Two-stage: local VLM describes, Groq produces commands
            print("[vision] Describing frame (local VLM)...")
            desc = self._describe_frame(jpeg_bytes, question=user_msg)
            if desc:
                user_msg = f"[Camera sees: {desc}] {user_msg}"
            msg = {"role": "user", "content": user_msg}
        else:
            # Direct: send image to Groq
            jpg = self._resize_jpeg(jpeg_bytes, max_dim=640)
            content = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(jpg).decode()}",
                }},
                {"type": "text", "text": user_msg},
            ]
            msg = {"role": "user", "content": content}

        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = [{"role": "system", "content": self._system_prompt}]
        messages += self.history[:-1]
        messages.append(msg)

        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"LLM error: {e}"})

    def _save_memory(self, note):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- [{timestamp}] {note}\n")
            print(f"[memory] Saved: {note}")
        except Exception:
            pass


# =====================================================================
# Anthropic Claude Client (Messages API)
# =====================================================================
class AnthropicClient:
    def __init__(self, api_key, model="claude-sonnet-4-6", image_getter=None, motion_getter=None, spatial_map=None):
        self.api_key = api_key
        self.model = model
        self.history = []
        self._image_getter = image_getter
        self._motion_getter = motion_getter
        self._spatial_map = spatial_map
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        parts = [SYSTEM_PROMPT]
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                content = f.read().strip()
                if content:
                    lines = content.strip().split("\n")[-20:]
                    parts.append(f"\n## Memory\n" + "\n".join(lines))
        if self._spatial_map:
            summary = self._spatial_map.summary()
            if summary:
                parts.append(f"\n{summary}")
        return "\n".join(parts)

    def _build_image_content(self, jpg_bytes):
        """Build Anthropic image content block from JPEG bytes."""
        import base64
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(jpg_bytes).decode(),
            },
        }

    def _call_api(self, messages):
        """Make Anthropic API call and return reply text."""
        r = requests.post("https://api.anthropic.com/v1/messages", json={
            "model": self.model,
            "max_tokens": 16000,
            "system": self._system_prompt,
            "messages": messages,
        }, headers={
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    def chat(self, user_msg, include_image=True):
        self._system_prompt = self._build_system_prompt()

        content = []
        if include_image and self._motion_getter:
            jpg = self._motion_getter()
            if jpg:
                content.append(self._build_image_content(jpg))
                print("[vision] Sending motion frame to LLM")
        content.append({"type": "text", "text": user_msg})

        # Store text-only in history
        self.history.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = list(self.history[:-1])
        messages.append({"role": "user", "content": content})

        try:
            reply = self._call_api(messages)
            self.history.append({"role": "assistant", "content": reply})

            # Save memory if response asks for it
            try:
                parsed = json.loads(reply) if reply.strip().startswith("{") else None
                if parsed and parsed.get("remember"):
                    self._save_memory(parsed["remember"])
            except (json.JSONDecodeError, KeyError):
                pass

            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"API error: {e}"})

    def chat_with_image(self, user_msg, jpeg_bytes):
        """Send a message with an explicit image (for observation feedback loops)."""
        self._system_prompt = self._build_system_prompt()

        content = [
            self._build_image_content(jpeg_bytes),
            {"type": "text", "text": user_msg},
        ]

        self.history.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = list(self.history[:-1])
        messages.append({"role": "user", "content": content})

        try:
            reply = self._call_api(messages)
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"API error: {e}"})

    def _save_memory(self, note):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- [{timestamp}] {note}\n")
            print(f"[memory] Saved: {note}")
        except Exception:
            pass


# =====================================================================
# Gemini Client (OpenAI-compatible endpoint)
# =====================================================================
class GeminiClient:
    """Google Gemini via OpenAI-compatible endpoint. Native vision support."""

    def __init__(self, api_key, model=GEMINI_MODEL, image_getter=None,
                 motion_getter=None, spatial_map=None):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.history = []
        self._image_getter = image_getter
        self._motion_getter = motion_getter
        self._spatial_map = spatial_map
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        parts = [SYSTEM_PROMPT]
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                content = f.read().strip()
                if content:
                    lines = content.strip().split("\n")[-20:]
                    parts.append(f"\n## Memory\n" + "\n".join(lines))
        if self._spatial_map:
            summary = self._spatial_map.summary()
            if summary:
                parts.append(f"\n{summary}")
        return "\n".join(parts)

    def _resize_jpeg(self, jpg_bytes, max_dim=512, quality=60):
        """Resize JPEG to save tokens."""
        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()
        except Exception:
            return jpg_bytes

    def _make_image_content(self, jpg_bytes):
        """Build OpenAI-style image content block."""
        import base64
        jpg = self._resize_jpeg(jpg_bytes)
        b64 = base64.b64encode(jpg).decode()
        return {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}"}}

    def _call_api(self, messages, max_tokens=1000):
        """Make OpenAI-compatible chat completion request."""
        r = requests.post(f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_completion_tokens": max_tokens,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat(self, user_msg, include_image=True):
        import base64
        self._system_prompt = self._build_system_prompt()

        # Build user content — attach image if motion detected
        jpg = None
        if include_image and self._motion_getter:
            jpg = self._motion_getter()
            if jpg:
                print("[vision] Sending motion frame to Gemini")

        if jpg:
            content = [
                self._make_image_content(jpg),
                {"type": "text", "text": user_msg},
            ]
            msg = {"role": "user", "content": content}
        else:
            msg = {"role": "user", "content": user_msg}

        # Store text-only in history
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = [{"role": "system", "content": self._system_prompt}]
        messages += self.history[:-1]
        messages.append(msg)

        try:
            reply = self._call_api(messages, max_tokens=300)
            self.history.append({"role": "assistant", "content": reply})

            try:
                parsed = json.loads(reply) if reply.strip().startswith("{") else None
                if parsed and parsed.get("remember"):
                    self._save_memory(parsed["remember"])
            except (json.JSONDecodeError, KeyError):
                pass

            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"LLM error: {e}"})

    def chat_with_image(self, user_msg, jpeg_bytes):
        """Send a message with an explicit image (for observation feedback loops)."""
        self._system_prompt = self._build_system_prompt()

        content = [
            self._make_image_content(jpeg_bytes),
            {"type": "text", "text": user_msg},
        ]
        msg = {"role": "user", "content": content}

        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = [{"role": "system", "content": self._system_prompt}]
        messages += self.history[:-1]
        messages.append(msg)

        try:
            reply = self._call_api(messages, max_tokens=300)
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except requests.RequestException as e:
            return json.dumps({"commands": [], "speak": f"LLM error: {e}"})

    def _save_memory(self, note):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- [{timestamp}] {note}\n")
            print(f"[memory] Saved: {note}")
        except Exception:
            pass


# =====================================================================
# Human Tracker (Camera + MediaPipe)
# =====================================================================
class HumanTracker:
    MODES = ("face", "hand")  # supported track modes

    def __init__(self, rover, camera_src):
        self.rover = rover
        self.camera_src = camera_src
        self._active = False
        self._thread = None
        self._stop_event = threading.Event()
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.status = "idle"
        self.track_mode = "face"  # "face" or "hand"
        self._cap = None
        self._detector = None
        # Shared frame for video server
        self._frame_lock = threading.Lock()
        self._jpeg_frame = None
        # Motion detection for LLM image gating
        self._prev_gray = None
        self._motion_detected = False
        self._motion_frame = None  # JPEG bytes of the frame that triggered motion
        self._motion_cooldown = 0  # time.time() when last motion frame was consumed
        # Pause tracking (e.g. during LLM gimbal commands)
        self._paused_until = 0
        # Background camera capture (runs even when tracker is idle)
        self._bg_cap_thread = None
        self._bg_cap_stop = threading.Event()
        # LLM vision function for smart scanning (set from main after init)
        self._llm_vision_fn = None  # chat_with_image(prompt, jpeg_bytes) -> str
        self._llm_hint_cooldown = 0  # throttle LLM calls during scan
        self._spatial_map = None     # SpatialMap reference (set from main)
        self._pose = None            # PoseTracker reference (set from main)
        self._track_history = []     # recent LLM tracking decisions for context
        self._start_bg_capture()

    def _start_bg_capture(self):
        """Start background camera capture for LLM vision (runs when tracker is idle)."""
        def _bg_loop():
            import cv2
            cap = self._init_camera()
            if not cap:
                print("[camera] Background capture failed — no camera")
                return
            print("[camera] Background capture started")
            frame_count = 0
            while not self._bg_cap_stop.is_set():
                # Release camera while tracker is active (it needs exclusive access)
                if self._active:
                    if cap and cap.isOpened():
                        cap.release()
                        print("[camera] Released camera for tracker")
                    while self._active and not self._bg_cap_stop.is_set():
                        time.sleep(0.3)
                    # Re-acquire camera after tracker stops
                    if not self._bg_cap_stop.is_set():
                        cap = self._init_camera()
                        if cap:
                            print("[camera] Re-acquired camera from tracker")
                        else:
                            print("[camera] Failed to re-acquire camera")
                            return
                    continue
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                frame_count += 1
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._frame_lock:
                    self._jpeg_frame = jpg.tobytes()
                # Motion detection every 5th frame
                if frame_count % 5 == 0:
                    self._detect_motion(frame)
                time.sleep(0.033)  # ~30fps
            cap.release()
        self._bg_cap_thread = threading.Thread(target=_bg_loop, daemon=True)
        self._bg_cap_thread.start()

    def set_mode(self, mode):
        """Switch tracking mode (face/hand). Restarts tracker if active."""
        if mode not in self.MODES:
            print(f"[tracker] Unknown mode: {mode}")
            return
        was_active = self._active
        if was_active:
            self.stop()
            time.sleep(0.5)
        self.track_mode = mode
        print(f"[tracker] Mode set to: {mode}")
        if was_active:
            self.start()

    def get_jpeg(self):
        """Get latest JPEG frame for video streaming."""
        with self._frame_lock:
            return self._jpeg_frame

    def get_motion_jpeg(self):
        """Get JPEG only if motion was detected since last call. Returns None otherwise."""
        with self._frame_lock:
            if self._motion_detected:
                self._motion_detected = False
                self._motion_cooldown = time.time()
                return self._motion_frame or self._jpeg_frame
        return None

    def _detect_motion(self, frame):
        """Compare frame against previous to detect significant motion. Updates _motion_detected."""
        import cv2

        # Skip during gimbal movement — camera panning creates false positives
        if time.time() < self._paused_until or self.status == "tracking":
            self._prev_gray = None  # reset baseline so next real check is clean
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return

        delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pct = (thresh.sum() / 255) / (thresh.shape[0] * thresh.shape[1]) * 100

        self._prev_gray = gray

        # >2% pixels changed = motion, and at least 1s since last consumed frame
        if motion_pct > 2.0 and (time.time() - self._motion_cooldown) > 1.0:
            with self._frame_lock:
                self._motion_detected = True
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self._motion_frame = jpg.tobytes()
            if motion_pct > 5.0:
                print(f"[motion] Detected ({motion_pct:.1f}% changed)")

    def pause(self, seconds=3.0):
        """Pause tracking for N seconds (for LLM/voice gimbal commands)."""
        self._paused_until = time.time() + seconds

    def _init_camera(self):
        import cv2
        src = self.camera_src
        # Try integer device index
        try:
            src_int = int(src)
            cap = cv2.VideoCapture(src_int)
            if cap.isOpened():
                return cap
            cap.release()
        except (ValueError, TypeError):
            pass

        # Try GStreamer pipeline for Jetson CSI camera
        gst_pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 "
            "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

        # Try string source (URL, file, etc.)
        if isinstance(src, str) and src.startswith("http"):
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                return cap
            cap.release()

        return None

    def _init_detector(self):
        """Initialize MediaPipe detector based on current track_mode."""
        import mediapipe as mp_lib
        if self.track_mode == "hand":
            print("[tracker] Using MediaPipe Hands")
            return mp_lib.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
        else:
            print("[tracker] Using MediaPipe Face Detection")
            return mp_lib.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )

    def _detect_target(self, rgb, frame, w, h):
        """Run detection and return (cx, cy, confidence) or None.
        Also draws bounding box on frame."""
        import cv2
        results = self._detector.process(rgb)
        if self.track_mode == "hand":
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # Use wrist (landmark 0) as the tracking point
                wrist = hand.landmark[0]
                cx, cy = wrist.x, wrist.y
                # Draw hand landmarks
                for lm in hand.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(cx * w), int(cy * h)), 8, (0, 0, 255), -1)
                return cx, cy, 0.9
        else:
            if results.detections:
                det = max(results.detections, key=lambda d: d.score[0])
                bbox = det.location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2
                cy = bbox.ymin + bbox.height / 2
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx * w), int(cy * h)), 5, (0, 255, 0), -1)
                return cx, cy, det.score[0]
        return None

    def _llm_track(self, frame):
        """Ask LLM to guide the camera toward a person's face.
        Feeds it full context: YOLO detections, spatial map history, past
        tracking decisions, and the current frame.
        Returns (pan_delta, tilt_delta, found_face: bool) or None if skipped."""
        import cv2
        if not self._llm_vision_fn:
            return None
        now = time.time()
        if now - self._llm_hint_cooldown < 2:  # max once per 2s
            return None

        self._llm_hint_cooldown = now

        # --- Build context from what the rover already knows ---
        context_parts = []

        # 1. Spatial map — where was the person last seen?
        if self._spatial_map:
            person_entry = None
            for key in ("person", "human", "man", "woman"):
                _, entry = self._spatial_map.find(key)
                if entry:
                    person_entry = (key, entry)
                    break
            if person_entry:
                name, e = person_entry
                age = int(now - e.get("time", 0))
                age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
                # How far from current gimbal position
                stored_pan = self._spatial_map.gimbal_pan_for(
                    e, self._pose.body_yaw if self._pose else 0)
                pan_diff = stored_pan - self.pan_angle
                tilt_diff = e['tilt'] - self.tilt_angle
                context_parts.append(
                    f"MEMORY: I last saw '{name}' at pan={stored_pan:.0f}°, "
                    f"tilt={e['tilt']}° ({age_str}). That is "
                    f"{pan_diff:+.0f}° pan and {tilt_diff:+.0f}° tilt "
                    f"from where I'm looking now.")

        # 3. Recent tracking decisions (so LLM doesn't repeat failed moves)
        if self._track_history:
            recent = self._track_history[-5:]
            hist_strs = []
            for h in recent:
                hist_strs.append(
                    f"  from pan={h['pan']}° tilt={h['tilt']}°: "
                    f"moved pan{h['pd']:+.0f}° tilt{h['td']:+.0f}° → {h['result']}")
            context_parts.append(
                "My recent tracking moves (learn from these):\n" + "\n".join(hist_strs))

        context_block = "\n\n".join(context_parts) if context_parts else "(No prior context.)"

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        prompt = (
            "# TASK: Guide my pan-tilt camera to center on a person's face.\n\n"
            "## MY HARDWARE\n"
            "I am a Waveshare UGV rover with a 2-axis pan-tilt gimbal holding "
            "a USB camera (640x480, ~65° horizontal FOV, ~50° vertical FOV).\n"
            "- Pan: left is NEGATIVE, right is POSITIVE (range -180° to +180°)\n"
            "- Tilt: down is NEGATIVE, up is POSITIVE (range -30° to +90°)\n"
            "- I sit low to the ground (~20cm high). People's faces are usually "
            "ABOVE me. A standing person's face needs tilt +30° to +60°.\n\n"
            "## CURRENT STATE\n"
            "Gimbal position: pan={pan}°, tilt={tilt}°\n\n"
            "## WHAT I KNOW\n"
            "{context}\n\n"
            "## FRAME-TO-GIMBAL MAPPING\n"
            "What you see in the image maps to gimbal adjustments:\n"
            "- Object at LEFT edge of frame → I need to pan LEFT (negative delta)\n"
            "- Object at RIGHT edge of frame → I need to pan RIGHT (positive delta)\n"
            "- Object at TOP of frame → I need to tilt UP (positive delta)\n"
            "- Object at BOTTOM of frame → I need to tilt DOWN (negative delta)\n"
            "- Object centered → delta ~0\n"
            "- The frame spans ~65° horizontally and ~50° vertically, so:\n"
            "  something at the left edge is ~32° left of center,\n"
            "  something at the bottom edge is ~25° below center.\n\n"
            "## BODY REASONING\n"
            "If I see body parts but not a face, reason about anatomy:\n"
            "- Feet/shoes/legs at bottom → face is FAR above → tilt up +20° to +40°\n"
            "- Waist/torso visible → face is somewhat above → tilt up +10° to +20°\n"
            "- Shoulders/neck → face is just above → tilt up +5° to +10°\n"
            "- Top of head/hair → face is just below → tilt down -5° to -10°\n"
            "- Arm/hand on one side → person extends that direction → pan that way\n"
            "- Person cut off at frame edge → they continue past that edge\n"
            "- If MEMORY says person was at a specific position, GO THERE first\n"
            "- If a previous move didn't help, try a DIFFERENT direction\n\n"
            "## RESPOND\n"
            "Reply with a short reasoning line, then the JSON on its own line:\n"
            "Reasoning: <what you see and why you're adjusting this way>\n"
            '{{"pan_delta":<deg>,"tilt_delta":<deg>,"found_face":<bool>}}\n\n'
            "- found_face=true ONLY if a person's face is clearly visible and "
            "roughly centered (within ~15° of frame center)\n"
            "- Add \"no_person\":true if you see NO human or human clue at all\n"
            "- Deltas between -30 and 30. Prefer bold moves over tiny ones."
        ).format(
            pan=round(self.pan_angle),
            tilt=round(self.tilt_angle),
            context=context_block,
        )

        try:
            raw = self._llm_vision_fn(prompt, jpg.tobytes())
            import re as _re
            # Extract reasoning line (before JSON)
            reasoning = ""
            reason_m = _re.search(r'[Rr]eason(?:ing)?:\s*(.+?)(?:\n|$)', raw)
            if reason_m:
                reasoning = reason_m.group(1).strip()

            m = _re.search(r'\{[^}]+\}', raw)
            if m:
                hint = json.loads(m.group())
                pd = float(hint.get("pan_delta", 0))
                td = float(hint.get("tilt_delta", 0))
                found = bool(hint.get("found_face", False))
                no_person = bool(hint.get("no_person", False))
                pd = max(-30, min(30, pd))
                td = max(-30, min(30, td))

                # Log result for history (with reasoning)
                if no_person:
                    result_str = "no person"
                elif found:
                    result_str = "face found"
                else:
                    result_str = "adjusting"
                self._track_history.append({
                    "pd": pd, "td": td, "result": result_str,
                    "pan": round(self.pan_angle), "tilt": round(self.tilt_angle),
                    "reason": reasoning[:80] if reasoning else "",
                })
                if len(self._track_history) > 10:
                    self._track_history = self._track_history[-10:]

                if reasoning:
                    print(f"[tracker] LLM: {reasoning[:120]}")
                if no_person:
                    print(f"[tracker] LLM → no person visible")
                elif found:
                    print(f"[tracker] LLM → face centered")
                else:
                    print(f"[tracker] LLM → pan{pd:+.0f}° tilt{td:+.0f}°")
                return pd, td, found
        except Exception as e:
            print(f"[tracker] LLM track failed: {e}")
        return None

    def start(self):
        if self._active:
            return
        self._stop_event.clear()
        self._active = True
        # Wait for bg capture to release camera
        time.sleep(0.6)
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        print("[tracker] Human following ENABLED")

    def stop(self):
        if not self._active:
            return
        self._stop_event.set()
        self._active = False
        if self._thread:
            self._thread.join(timeout=3)
        # Stop rover motion
        self.rover.send({"T": 1, "L": 0, "R": 0})
        self.status = "idle"
        print("[tracker] Human following DISABLED")

    @property
    def is_active(self):
        return self._active

    def _track_loop(self):
        """Two-tier tracking:
        - MediaPipe (30fps): fast proportional gimbal tracking when face is visible
        - LLM (every ~3s): reasons about what it sees to FIND the person's face
        MediaPipe handles tracking, LLM handles recognition and search."""
        import cv2
        self._cap = self._init_camera()
        if not self._cap:
            print("[tracker] ERROR: No camera available. Tracker disabled.")
            self._active = False
            self.status = "no camera"
            return

        self._detector = self._init_detector()
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": TRACK_GIMBAL_SPD, "ACC": TRACK_GIMBAL_ACC})
        self.status = "searching"
        headlight_on = False
        frame_count = 0
        last_status = None
        llm_no_person_count = 0  # consecutive LLM "no person" responses
        mediapipe_frames_since_llm = 0  # frames since last LLM move

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count == 1:
                h, w = frame.shape[:2]
                print(f"[tracker] Camera active: {w}x{h}")
            h, w = frame.shape[:2]

            # Skip while paused (LLM/voice gimbal commands)
            if time.time() < self._paused_until:
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._frame_lock:
                    self._jpeg_frame = jpg.tobytes()
                time.sleep(0.033)
                continue

            # ===== TIER 1: MediaPipe (every frame, ~30fps) =====
            # Fast face detection + proportional gimbal tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = self._detect_target(rgb, frame, w, h)

            if detection:
                target_cx, target_cy, conf = detection
                llm_no_person_count = 0
                self.status = "tracking"

                if last_status != "tracking":
                    print(f"[tracker] MediaPipe: {self.track_mode} detected (conf={conf:.0%})")
                    last_status = "tracking"
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    if self._spatial_map:
                        wp = self.pan_angle
                        if self._pose:
                            wp = self._pose.body_yaw + self.pan_angle
                        self._spatial_map.update(["person"],
                            wp, self.tilt_angle)
                    self._track_history.clear()

                if not headlight_on:
                    self.rover.send({"T": 132, "IO4": 0, "IO5": 1})
                    headlight_on = True

                # Proportional gimbal tracking (MediaPipe handles this)
                err_x = target_cx - 0.5
                err_y = target_cy - 0.5
                if abs(err_x) > TRACK_DEADZONE_X:
                    self.pan_angle += err_x * TRACK_GIMBAL_ITERATE * 100
                if abs(err_y) > TRACK_DEADZONE_Y:
                    self.tilt_angle -= err_y * TRACK_GIMBAL_ITERATE * 100
                self.pan_angle = max(-180, min(180, self.pan_angle))
                self.tilt_angle = max(-30, min(90, self.tilt_angle))
                dist = math.sqrt(err_x ** 2 + err_y ** 2)
                spd = max(1, int(dist * 200))
                self.rover.send({
                    "T": 133,
                    "X": round(self.pan_angle, 1),
                    "Y": round(self.tilt_angle, 1),
                    "SPD": spd,
                    "ACC": max(1, int(dist * 50)),
                })
                mediapipe_frames_since_llm = 0
            else:
                # ===== TIER 2: LLM (recognition + search) =====
                # MediaPipe can't see a face. Ask the LLM to find it.
                # _llm_track returns None during cooldown — that's fine,
                # we just wait. The LLM IS the search strategy.
                mediapipe_frames_since_llm += 1
                llm_result = self._llm_track(frame)

                if llm_result:
                    pd, td, found_face = llm_result

                    if found_face:
                        # LLM sees face but MediaPipe missed it — hold steady,
                        # let MediaPipe catch up on the next frames
                        self.status = "tracking"
                        if last_status != "tracking":
                            print(f"[tracker] LLM: face visible, waiting for MediaPipe lock")
                            last_status = "tracking"
                        if not headlight_on:
                            self.rover.send({"T": 132, "IO4": 0, "IO5": 1})
                            headlight_on = True
                        llm_no_person_count = 0
                    elif abs(pd) > 0 or abs(td) > 0:
                        # LLM sees clues and is guiding gimbal toward face
                        self.status = "following"
                        if last_status != "following":
                            print(f"[tracker] LLM guiding toward person")
                            last_status = "following"
                        self.pan_angle = max(-180, min(180, self.pan_angle + pd))
                        self.tilt_angle = max(-30, min(90, self.tilt_angle + td))
                        self.rover.send({
                            "T": 133,
                            "X": round(self.pan_angle, 1),
                            "Y": round(self.tilt_angle, 1),
                            "SPD": 100,
                            "ACC": 20,
                        })
                        mediapipe_frames_since_llm = 0
                        llm_no_person_count = 0
                    else:
                        # LLM sees nothing at all
                        llm_no_person_count += 1
                        if last_status != "searching":
                            self.status = "searching"
                            print(f"[tracker] LLM: no person visible ({llm_no_person_count}x)")
                            last_status = "searching"
                        if headlight_on:
                            self.rover.send({"T": 132, "IO4": 0, "IO5": 0})
                            headlight_on = False

                # After 3+ "no person" from LLM: slow blind sweep as last resort
                if llm_no_person_count >= 3 and frame_count % 15 == 0:
                    sweep_dir = 1 if self.pan_angle <= 0 else -1
                    self.pan_angle += sweep_dir * 20
                    self.pan_angle = max(-150, min(150, self.pan_angle))
                    self.rover.send({
                        "T": 133,
                        "X": round(self.pan_angle, 1),
                        "Y": round(self.tilt_angle, 1),
                        "SPD": 60, "ACC": 10,
                    })

            # Encode frame for video server
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self._frame_lock:
                self._jpeg_frame = jpg.tobytes()

            # Motion detection for LLM image gating (every 5th frame)
            if frame_count % 5 == 0:
                self._detect_motion(frame)

            time.sleep(0.033)  # ~30fps

        # Cleanup
        if self._cap:
            self._cap.release()
        if self._detector:
            self._detector.close()
        self.rover.send({"T": 132, "IO4": 0, "IO5": 0})  # lights off


# =====================================================================
# MJPEG Video Server (streams tracker's camera feed)
# =====================================================================
VIDEO_PORT = 8090

class VideoServer:
    def __init__(self, tracker):
        self.tracker = tracker

    def start(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from socketserver import ThreadingMixIn
        tracker = self.tracker

        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b'<html><body style="margin:0;background:#000">'
                                     b'<img src="/stream" style="width:100%;height:100vh;object-fit:contain">'
                                     b'</body></html>')
                elif self.path == "/snap":
                    jpg = tracker.get_jpeg()
                    if jpg:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                    else:
                        self.send_response(503)
                        self.end_headers()
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            jpg = tracker.get_jpeg()
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(0.066)  # ~15fps to client
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass

        server = ThreadedHTTPServer(("0.0.0.0", VIDEO_PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[video] Streaming at http://192.168.0.112:{VIDEO_PORT}")


# =====================================================================
# Voice I/O (faster-whisper STT + Piper TTS)
# =====================================================================
PIPER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "en_US-lessac-medium.onnx")
WHISPER_MODEL_SIZE = "base"  # faster recognition
MIC_DEVICE = None        # auto-detect USB mic at startup
MIC_NATIVE_RATE = 48000  # USB mic only supports 48kHz
MIC_WHISPER_RATE = 16000 # Whisper needs 16kHz

# --- Stop word detection (local whisper, no network) ---
STOP_WORDS = {"stop", "halt", "freeze", "shut up", "be quiet", "emergency"}
STOP_WORD_PATTERN = re.compile(
    r"\b(stop|halt|freeze|shut\s*up|be\s*quiet|emergency)\b", re.IGNORECASE
)

def _find_usb_playback():
    """Auto-detect USB audio playback device (card numbers shift on reboot)."""
    import subprocess
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True)
        for line in out.splitlines():
            if "USB" in line and "card" in line:
                card = line.split("card ")[1].split(":")[0]
                dev = line.split("device ")[1].split(":")[0]
                return f"plughw:{card},{dev}"
    except Exception:
        pass
    return "plughw:1,0"

USB_PLAYBACK = _find_usb_playback()

class VoiceIO:
    def __init__(self):
        self._tts_lock = threading.Lock()
        self._speaking = False
        self._tts_end_time = 0
        self._tts_start_time = 0
        self._aplay_proc = None  # current aplay subprocess (for barge-in kill)
        self.tts_voice = "troy"  # Groq Orpheus voices: troy (male), hannah (female), austin (male)
        self.available = False
        self._emergency_cb = None  # callback for stop word emergency
        self._local_whisper = None  # tiny whisper for local stop word detection
        self._stt_whisper = None    # small whisper for full transcription (local STT)
        self._mic_card = None  # ALSA card number for mic mute/unmute
        self._init()

    def _init(self):
        import subprocess

        # --- Groq API key check (needed for TTS, not for STT anymore) ---
        if not GROQ_API_KEY:
            print("[voice] GROQ_API_KEY not set — TTS will use local fallback (Piper/espeak)")

        # --- Find USB mic via ALSA (sounddevice can't see camera mic while video is open) ---
        try:
            global MIC_DEVICE
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            # Prefer Camera mic over PnP Audio (which may be dead)
            mic_card = None
            fallback_card = None
            for line in result.stdout.splitlines():
                if "card" in line and "USB" in line:
                    card_num = line.split("card")[1].split(":")[0].strip()
                    if "Camera" in line:
                        mic_card = card_num
                    elif fallback_card is None:
                        fallback_card = card_num
            mic_card = mic_card or fallback_card
            if mic_card is None:
                print("[voice] No USB mic found")
                return
            self._mic_card = mic_card
            MIC_DEVICE = f"plughw:{mic_card},0"
            # Warmup test
            test = subprocess.run(
                ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(MIC_NATIVE_RATE),
                 "-c", "1", "-d", "1", "/dev/null"],
                capture_output=True, timeout=3)
            print(f"[voice] Microphone ready (ALSA {MIC_DEVICE}, {MIC_NATIVE_RATE}Hz)")
        except Exception as e:
            print(f"[voice] Microphone error: {e}")
            return

        # --- TTS setup ---
        if GROQ_API_KEY:
            print(f"[voice] TTS: Groq Orpheus ({self.tts_voice})")
        elif os.path.exists(PIPER_MODEL):
            print("[voice] TTS: Piper (local fallback)")
        else:
            print("[voice] TTS: espeak (basic fallback)")

        # --- Local tiny whisper for stop word detection (no network needed) ---
        try:
            from faster_whisper import WhisperModel
            print("[voice] Loading local whisper-tiny for stop word detection...")
            self._local_whisper = WhisperModel(
                "tiny", device="cpu", compute_type="int8", cpu_threads=2
            )
            print("[voice] Local whisper-tiny ready (stop word detector)")
        except Exception as e:
            print(f"[voice] Local whisper not available (stop words disabled): {e}")
            self._local_whisper = None

        # --- Local whisper-small for full STT transcription ---
        try:
            from faster_whisper import WhisperModel
            print("[voice] Loading local whisper-small for STT...")
            self._stt_whisper = WhisperModel(
                "small", device="cpu", compute_type="int8", cpu_threads=4
            )
            print("[voice] STT: local whisper-small (faster-whisper)")
        except Exception as e:
            print(f"[voice] Local whisper-small not available: {e}")
            if GROQ_API_KEY:
                print("[voice] STT fallback: Groq Whisper API")
            else:
                print("[voice] WARNING: No STT available (no local whisper, no Groq key)")
            self._stt_whisper = None

        self.available = True

    # --- Stop word detection methods ---

    def set_emergency_callback(self, cb):
        """Register callback for stop word emergency. cb() is called with the matched word."""
        self._emergency_cb = cb

    def _check_stop_word(self, speech_chunks):
        """Run local tiny whisper on speech chunks, return matched stop word or None."""
        if not self._local_whisper or not speech_chunks:
            return None
        import numpy as np
        try:
            audio = np.concatenate(speech_chunks)
            # Decimate 48kHz -> 16kHz
            audio_16k = audio[::3]
            segments, _ = self._local_whisper.transcribe(
                audio_16k, beam_size=1, language="en",
                without_timestamps=True, condition_on_previous_text=False,
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                m = STOP_WORD_PATTERN.search(text)
                if m:
                    word = m.group(1).lower()
                    print(f'[stop-word] Detected: "{word}" (from: "{text}")')
                    return word
        except Exception as e:
            print(f"[stop-word] Check error: {e}")
        return None

    def _fire_emergency(self, word):
        """Kill TTS and fire emergency callback."""
        print(f'[EMERGENCY] Stop word "{word}" — killing TTS, stopping rover')
        # Kill any active TTS playback
        proc = self._aplay_proc
        if proc and proc.poll() is None:
            proc.kill()
        self._speaking = False
        self._tts_end_time = time.time()
        # Fire callback to stop rover hardware
        if self._emergency_cb:
            try:
                self._emergency_cb(word)
            except Exception as e:
                print(f"[EMERGENCY] Callback error: {e}")

    # --- Echo prevention methods ---

    def _mic_mute(self):
        """Mute mic via ALSA before TTS playback (best-effort)."""
        if not self._mic_card:
            return
        import subprocess
        try:
            subprocess.run(
                ["amixer", "-c", self._mic_card, "set", "Mic", "nocap"],
                capture_output=True, timeout=2,
            )
        except Exception:
            pass

    def _mic_unmute(self):
        """Unmute mic via ALSA after TTS playback (best-effort)."""
        if not self._mic_card:
            return
        import subprocess
        try:
            subprocess.run(
                ["amixer", "-c", self._mic_card, "set", "Mic", "cap"],
                capture_output=True, timeout=2,
            )
        except Exception:
            pass

    def _is_echo_gated(self, rms):
        """Check if audio should be discarded as echo. Returns True if gated."""
        if self._speaking:
            return True
        if not self._tts_end_time:
            return False
        elapsed = time.time() - self._tts_end_time
        # Adaptive cooldown based on TTS duration
        tts_duration = self._tts_end_time - self._tts_start_time if self._tts_start_time else 2.0
        if tts_duration < 2.0:
            cooldown = 0.5
        elif tts_duration < 5.0:
            cooldown = 1.0
        else:
            cooldown = 1.5
        if elapsed < cooldown:
            return True
        # Post-cooldown: 1s window with 2x energy threshold to catch echo tail
        if elapsed < cooldown + 1.0:
            return rms < 0.06  # 2x normal threshold (0.03)
        return False

    def _transcribe_groq(self, audio_float32):
        """Send audio to Groq Whisper API, return transcribed text."""
        import numpy as np
        import io
        import wave

        # Convert float32 -> 16-bit PCM WAV at 16kHz
        audio_16k = audio_float32[::3]  # decimate 48kHz -> 16kHz
        pcm = (audio_16k * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(MIC_WHISPER_RATE)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)

        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.wav", wav_buf, "audio/wav")},
            data={"model": "whisper-large-v3-turbo", "language": "en"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()

    def _transcribe_local(self, audio_float32):
        """Transcribe audio using local faster-whisper small model. Returns text."""
        import numpy as np
        audio_16k = audio_float32[::3]  # decimate 48kHz -> 16kHz
        segments, _ = self._stt_whisper.transcribe(
            audio_16k, beam_size=3, language="en",
            without_timestamps=True, condition_on_previous_text=False,
        )
        return " ".join(s.text for s in segments).strip()

    def _read_chunk(self, proc, chunk_samples):
        """Read one chunk of raw S16_LE audio from arecord proc, return as float32 numpy array."""
        import numpy as np
        raw = proc.stdout.read(chunk_samples * 2)  # 2 bytes per S16_LE sample
        if len(raw) < chunk_samples * 2:
            return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def listen_continuous(self):
        """Continuously listen via arecord, detect speech via energy VAD, transcribe via Groq.
        Checks for stop words locally every 2 chunks (~1s) for sub-2s emergency stop.
        Uses adaptive echo gating to avoid transcribing TTS output."""
        if not self.available:
            return None
        import subprocess
        import numpy as np

        while self._speaking:
            time.sleep(0.1)

        CHUNK_SEC = 0.5
        chunk_samples = int(MIC_NATIVE_RATE * CHUNK_SEC)
        silence_thresh = 0.03
        stop_word_thresh = 0.09  # 3x normal — must be loud human voice, not echo
        min_speech_chunks = 2
        max_speech_chunks = 240
        silence_after = 4

        speech_chunks = []
        speech_started = False
        silent_count = 0

        # Start arecord as a streaming subprocess
        proc = subprocess.Popen(
            ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(MIC_NATIVE_RATE),
             "-c", "1", "-t", "raw", "--buffer-size", str(chunk_samples)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        print("[listening...]", flush=True)
        try:
            while True:
                chunk = self._read_chunk(proc, chunk_samples)
                if chunk is None:
                    break

                rms = np.sqrt(np.mean(chunk ** 2))

                # --- Echo gate: adaptive cooldown replaces fixed 2s ---
                if self._is_echo_gated(rms):
                    # During TTS, still check for stop words if audio is loud enough
                    if self._local_whisper and rms > stop_word_thresh:
                        word = self._check_stop_word([chunk])
                        if word:
                            proc.kill()
                            proc.wait()
                            self._fire_emergency(word)
                            return "__STOP__"
                    speech_chunks.clear()
                    speech_started = False
                    silent_count = 0
                    continue

                if rms > silence_thresh:
                    if not speech_started:
                        speech_started = True
                        print("[voice] hearing speech...", flush=True)
                    speech_chunks.append(chunk)
                    silent_count = 0

                    # --- Stop word check every 2 speech chunks (~1s of audio) ---
                    if self._local_whisper and len(speech_chunks) >= 2 and len(speech_chunks) % 2 == 0:
                        word = self._check_stop_word(speech_chunks)
                        if word:
                            proc.kill()
                            proc.wait()
                            self._fire_emergency(word)
                            return "__STOP__"

                elif speech_started:
                    speech_chunks.append(chunk)
                    silent_count += 1
                    if silent_count >= silence_after:
                        break
                if len(speech_chunks) >= max_speech_chunks:
                    break

        except Exception as e:
            print(f"[voice] Record error: {e}")
            return None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass

        if len(speech_chunks) < min_speech_chunks:
            print("[listening...]", flush=True)
            return None

        audio = np.concatenate(speech_chunks)

        if self._stt_whisper:
            stt_label = "local whisper"
        elif GROQ_API_KEY:
            stt_label = "groq"
        else:
            print("[voice] No STT available")
            return None
        print(f"[voice] transcribing ({stt_label})...", flush=True)
        try:
            text = self._transcribe_local(audio) if self._stt_whisper else self._transcribe_groq(audio)
            hallucinations = {
                ".", "..", "...", "Thank you.", "Thanks for watching.",
                "Bye.", "Thank you for watching.", "Subscribe.",
                "you", "You", "I'm sorry.", "Okay.", "Yeah.",
                "Hmm.", "Mm-hmm.", "Uh-huh.", "Oh.", "Ah.",
                "So.", "Well.", "Right.", "Sure.", "OK.",
            }
            if text and text not in hallucinations and len(text) > 2:
                print(f'[heard] "{text}"')
                return text
            else:
                print("[listening...]", flush=True)
                return None
        except Exception as e:
            print(f"[voice] Transcribe error: {e}")
            return None

    def interrupt(self):
        """Kill TTS playback immediately (barge-in)."""
        proc = self._aplay_proc
        if proc and proc.poll() is None:
            proc.kill()
            print("[voice] Interrupted (barge-in)")
        self._speaking = False
        self._tts_end_time = time.time()
        self._mic_unmute()

    def speak(self, text, tone=None):
        """Speak text using TTS (non-blocking). Priority: Gemini > Groq > Piper > espeak."""
        if not text:
            return
        threading.Thread(target=self._speak_sync, args=(text, tone), daemon=True).start()

    def _speak_gemini(self, text, tone=None):
        """Gemini TTS via REST API. Returns True on success, False on failure."""
        import subprocess
        import base64
        if not GEMINI_API_KEY:
            return False
        try:
            voice = GEMINI_TTS_VOICE_MAP.get(self.tts_voice, GEMINI_TTS_VOICE)
            # Build prompt with tone prefix if provided
            prompt = text
            if tone:
                prompt = f"Say this in a {tone} tone: {text}"

            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_TTS_MODEL}:generateContent?key={GEMINI_API_KEY}")
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": voice}
                        }
                    },
                },
            }
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()

            # Extract PCM audio from response
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    audio_b64 = inline["data"]
                    pcm_data = base64.b64decode(audio_b64)
                    # Gemini TTS outputs PCM 24kHz 16-bit mono
                    self._aplay_proc = subprocess.Popen(
                        ["aplay", "-D", USB_PLAYBACK, "-f", "S16_LE",
                         "-r", "24000", "-c", "1", "-t", "raw"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._aplay_proc.communicate(input=pcm_data, timeout=30)
                    return True
            print("[voice] Gemini TTS: no audio in response")
            return False
        except Exception as e:
            print(f"[voice] Gemini TTS error: {e}")
            return False

    def _speak_sync(self, text, tone=None):
        import subprocess
        with self._tts_lock:
            self._speaking = True
            self._tts_start_time = time.time()
            self._mic_mute()
            try:
                # Priority 1: Gemini TTS
                if GEMINI_API_KEY and self._speak_gemini(text, tone):
                    return
                # Priority 2: Groq Orpheus TTS
                if GROQ_API_KEY:
                    # Orpheus uses [tone] prefix for emotion
                    orpheus_text = f"[{tone}] {text}" if tone else text
                    r = requests.post(
                        "https://api.groq.com/openai/v1/audio/speech",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": "canopylabs/orpheus-v1-english",
                            "voice": self.tts_voice,
                            "input": orpheus_text,
                            "response_format": "wav",
                        },
                        timeout=15,
                    )
                    r.raise_for_status()
                    self._aplay_proc = subprocess.Popen(
                        ["aplay", "-D", USB_PLAYBACK, "-t", "wav"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._aplay_proc.communicate(input=r.content, timeout=30)
                elif os.path.exists(PIPER_MODEL):
                    # Fallback: Piper local TTS
                    piper_proc = subprocess.Popen(
                        ["piper", "--model", PIPER_MODEL, "--output-raw"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    aplay_proc = subprocess.Popen(
                        ["aplay", "-D", USB_PLAYBACK, "-r", "22050",
                         "-f", "S16_LE", "-t", "raw", "-c", "1"],
                        stdin=piper_proc.stdout,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    piper_proc.stdin.write(text.encode("utf-8"))
                    piper_proc.stdin.close()
                    piper_proc.stdout.close()
                    aplay_proc.wait(timeout=30)
                    piper_proc.wait(timeout=5)
                else:
                    subprocess.run(
                        ["espeak", "-s", "160", text],
                        capture_output=True, timeout=15,
                    )
            except Exception as e:
                print(f"[voice] TTS error: {e}")
            finally:
                self._speaking = False
                self._tts_end_time = time.time()
                self._mic_unmute()


# =====================================================================
# Response Parser
# =====================================================================
def parse_llm_response(raw):
    text = raw.strip()
    # Strip markdown fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find valid JSON by scanning for matching braces
        start = text.find("{")
        if start >= 0:
            # Try each closing brace from last to first
            pos = len(text)
            while True:
                pos = text.rfind("}", 0, pos)
                if pos < start:
                    break
                try:
                    return json.loads(text[start:pos + 1])
                except json.JSONDecodeError:
                    continue
            # JSON was truncated (hit max_tokens) — try to salvage
            # Truncate commands array at last complete command, close the JSON
            truncated = text[start:]
            # Find the last complete command object (ending with })
            last_obj = truncated.rfind("}")
            if last_obj > 0:
                # Cut there, close the array and outer object
                snippet = truncated[:last_obj + 1]
                # Try closing with ], "speak":"..."} patterns
                for closer in ['],"speak":"truncated"}', '],"speak":""}', ']}']:
                    try:
                        return json.loads(snippet + closer)
                    except json.JSONDecodeError:
                        continue
    return None



# =====================================================================
# Main Application
# =====================================================================
def kill_old_instances():
    """Kill any other rover_brain.py Python processes."""
    import subprocess
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(["pgrep", "-f", "python.*rover_brain.py"], text=True)
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            pid = int(line.strip())
            if pid != my_pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"[init] Killed old instance (pid {pid})")
                except (PermissionError, ProcessLookupError):
                    pass
    except (subprocess.CalledProcessError, ValueError):
        pass


def main():
    kill_old_instances()
    parser = argparse.ArgumentParser(description="UGV Rover PT - AI Control System")
    parser.add_argument("--voice", action="store_true", help="Enable voice input/output")
    parser.add_argument("--track", action="store_true", help="Start in human-tracking mode")
    parser.add_argument("--elevenlabs", action="store_true", help="Use ElevenLabs voice (off by default)")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--camera", default=CAMERA_SOURCE, help="Camera source (device index, GStreamer, or URL)")
    args = parser.parse_args()

    brain_name = f"Gemini ({args.model or GEMINI_MODEL})"

    print("=" * 60)
    print("  UGV Rover PT - AI Control System")
    print(f"  Brain:   {brain_name}")
    print(f"  ESP32:   UART (serial)")
    print(f"  Voice:   {'ON' if args.voice else 'OFF'}")
    print(f"  Tracker: {'ON' if args.track else 'OFF'}")
    print("=" * 60)

    # --- ESP32: UART serial only ---
    print("[esp32] Connecting via UART serial...")
    rover = RoverSerial()
    if not rover.connect():
        print("[esp32] WARNING: Serial not connected. Running without ESP32.")

    # --- Calibrate gimbal to known position ---
    rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(1.0)
    print("[init] Gimbal centered (pan=0, tilt=0)")

    # --- Tracker + Video Server ---
    tracker = HumanTracker(rover, args.camera)
    if args.track:
        tracker.start()
    VideoServer(tracker).start()

    # --- Local object detector disabled — LLM handles all recognition ---
    local_detector = None
    visual_servo = None
    LABEL_OVERRIDES = {}
    world_map = None
    path_planner = None
    path_follower = None
    print("[detector] YOLO disabled — LLM-only recognition")
    spatial_map = SpatialMap()

    # --- Pose Tracker ---
    pose = PoseTracker()
    rover._on_command = pose.on_command
    print(f"[pose] Tracker active (wheelbase={PoseTracker.WHEELBASE}m)")

    # --- Path Planner (survey, vectorized paths, door navigation) ---
    try:
        from path_planner import WorldMap, PathPlanner, PathFollower
        # WorldMap, PathPlanner, PathFollower will be initialized on demand (need survey first)
        print(f"[nav] Path planner available")
    except Exception as e:
        print(f"[nav] Path planner unavailable: {e}")

    # --- Adaptive light management ---
    def _check_ambient_light():
        """Adjust headlights based on ambient brightness from camera."""
        import cv2 as _cv2
        jpeg = tracker.get_jpeg()
        if not jpeg:
            return
        frame = _cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), _cv2.IMREAD_COLOR)
        if frame is None:
            return
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        # Dim lights when bright enough, brighten when dark
        if brightness > 100:
            # Plenty of light — dim headlights
            rover.send({"T": 132, "IO4": 0, "IO5": 0})
        elif brightness > 60:
            # Moderate light — low headlights
            rover.send({"T": 132, "IO4": 20, "IO5": 30})
        else:
            # Dark — turn on headlights for visibility
            rover.send({"T": 132, "IO4": 80, "IO5": 120})

    def _light_manager_loop():
        """Background thread: check ambient light every 10s."""
        while running:
            try:
                _check_ambient_light()
            except Exception:
                pass
            time.sleep(10)

    import numpy as np
    running = True  # needed by _light_manager_loop (and later by main loop)
    _light_thread = threading.Thread(target=_light_manager_loop, daemon=True)
    _light_thread.start()
    print(f"[lights] Adaptive light management active")

    # --- LLM (Gemini 3.1 Pro) ---
    model = args.model or GEMINI_MODEL
    if not GEMINI_API_KEY:
        print("[error] GEMINI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    llm = GeminiClient(GEMINI_API_KEY, model, image_getter=tracker.get_jpeg,
                       motion_getter=tracker.get_motion_jpeg, spatial_map=spatial_map)
    # Quick connectivity test
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "max_completion_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            timeout=15)
        r.raise_for_status()
        print(f"[gemini] {model} ready (native vision)")
    except Exception as e:
        print(f"[gemini] API error: {e}")
        sys.exit(1)

    # --- Dedicated vision LLM for tracker (Gemini, separate from main chat) ---
    def _tracker_vision(prompt, jpeg_bytes):
        """Dedicated Gemini vision call for tracker — no shared history."""
        import base64
        jpg_resized = llm._resize_jpeg(jpeg_bytes, max_dim=512, quality=60)
        b64 = base64.b64encode(jpg_resized).decode()
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model, "max_completion_tokens": 250,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
            }, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    tracker._llm_vision_fn = _tracker_vision
    tracker._spatial_map = spatial_map
    tracker._pose = pose
    print(f"[tracker] LLM-guided tracking enabled ({model})")

    # --- Dedicated orchestrator LLM (goal evaluation, no shared history) ---
    def _orchestrator_llm(prompt, jpeg_bytes=None):
        """Dedicated Gemini call for orchestrator goal evaluation.
        Uses ORCHESTRATOR_SYSTEM_PROMPT. No shared history."""
        import base64
        try:
            content = [{"type": "text", "text": prompt}]
            if jpeg_bytes:
                jpg_resized = llm._resize_jpeg(jpeg_bytes, max_dim=512, quality=60)
                b64 = base64.b64encode(jpg_resized).decode()
                content.insert(0, {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}})
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model, "max_completion_tokens": 500,
                    "messages": [
                        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                }, timeout=30)
            r.raise_for_status()
            raw_text = r.json()["choices"][0]["message"]["content"]
            parsed = parse_llm_response(raw_text)
            return parsed or {}
        except Exception as e:
            print(f"[orchestrator] LLM error: {e}")
            return {}

    # --- Background fact extractor ---
    start_fact_extractor()

    # --- Voice ---
    voice = VoiceIO() if args.voice else None

    # --- State ---
    stop_timer = None
    llm_busy = False
    emergency_event = threading.Event()  # set by stop word to unblock main loop
    heard_queue = queue.Queue()       # voice/text -> dispatcher
    llm_result_queue = queue.Queue()  # LLM responses -> dispatcher

    def shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # --- Wire stop word emergency callback ---
    if voice:
        def _emergency_stop(word):
            """Stop wheels + gimbal immediately on stop word detection."""
            rover.send({"T": 1, "L": 0, "R": 0})
            rover.send({"T": 135})
            emergency_event.set()  # unblock main loop from LLM wait
            print(f'[EMERGENCY] Rover stopped (word: "{word}")')
        voice.set_emergency_callback(_emergency_stop)

    def auto_stop():
        rover.send({"T": 1, "L": 0, "R": 0})
        print("\n[auto-stop] Motion stopped")

    def _run_bash(cmd):
        """Execute a bash command and return output string."""
        import subprocess as _sp
        print(f"  [bash] $ {cmd}")
        try:
            result = _sp.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=300, cwd=ROVER_DIR,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            output = ""
            if stdout:
                output += stdout
            if stderr:
                output += ("\n" if output else "") + stderr
            if not output:
                output = "(no output)"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            # Cap output to fit context
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            print(f"  [bash] → {output[:200]}{'...' if len(output) > 200 else ''}")
            return output
        except _sp.TimeoutExpired:
            msg = "(timed out after 300s)"
            print(f"  [bash] → {msg}")
            return msg
        except Exception as e:
            msg = f"(error: {e})"
            print(f"  [bash] → {msg}")
            return msg

    MAX_BASH_STEPS = 25

    def bash_agent_loop(initial_parsed):
        """Run an agentic bash loop — LLM can run commands, see output, run more, until done."""
        current = initial_parsed
        for step in range(MAX_BASH_STEPS):
            if emergency_event.is_set():
                print("[bash agent] Aborted by emergency stop")
                emergency_event.clear()
                return current
            # Execute hardware commands + speech
            execute_response(current)
            bash_out = getattr(execute_response, '_bash_output', None)
            execute_response._bash_output = None

            if not bash_out:
                # No bash in this response — agent is done
                return current

            if emergency_event.is_set():
                print("[bash agent] Aborted by emergency stop")
                emergency_event.clear()
                return current

            # Feed output back to LLM
            step_label = f"[bash agent {step + 1}/{MAX_BASH_STEPS}]"
            print(f"{step_label} Feeding output back to LLM...")
            follow_up = f"[Bash output]:\n{bash_out}\n\nYou can run another bash command, or speak the final result. Steps remaining: {MAX_BASH_STEPS - step - 1}"
            try:
                raw = llm.chat(follow_up)
                parsed = parse_llm_response(raw)
                if not parsed:
                    print(f"{step_label} Unparseable: {raw[:200]}")
                    return current
                current = parsed
            except Exception as e:
                print(f"{step_label} LLM error: {e}")
                return current

        print(f"[bash agent] Max steps ({MAX_BASH_STEPS}) reached")
        return current

    def execute_response(parsed, source="llm"):
        """Execute a parsed response. Returns True if observe was requested, or 'bash_result' string."""
        nonlocal stop_timer
        speak = parsed.get("speak", "")
        tone = parsed.get("tone", "")
        commands = parsed.get("commands", [])
        bash_cmd = parsed.get("bash", "")
        duration = parsed.get("duration", 0)
        repeat_for = parsed.get("repeat_for", 0)
        observe = parsed.get("observe", False)
        delay = parsed.get("_delay", 0)  # inter-command delay for multi-step (fast cmds)

        if speak:
            tag = "fast" if source == "fast" else ("cache" if source == "cache" else "rover")
            tone_label = f" ({tone})" if tone else ""
            print(f"[{tag}] {speak}{tone_label}")
            if voice:
                voice.speak(speak, tone=tone or None)

        # --- Bash command execution ---
        bash_output = None
        if bash_cmd and isinstance(bash_cmd, str):
            bash_output = _run_bash(bash_cmd)

        if not commands and not bash_output:
            execute_response._bash_output = bash_output
            return observe
        if not commands and bash_output:
            execute_response._bash_output = bash_output
            return observe

        if stop_timer and stop_timer.is_alive():
            stop_timer.cancel()

        # Estimate total sequence time for tracker pause
        total_pause = 0
        for cmd in commands:
            if "_pause" in cmd:
                total_pause += cmd["_pause"]
            else:
                total_pause += 0.5  # rough per-command estimate
        if repeat_for > 0:
            total_pause = repeat_for
        pause_secs = max(3.0, total_pause + 2.0)
        tracker.pause(pause_secs)

        last_pan = getattr(execute_response, '_pan', 0.0)
        last_tilt = getattr(execute_response, '_tilt', 0.0)

        def _run_commands(commands, last_pan, last_tilt):
            for cmd in commands:
                # --- Explicit pause pseudo-command ---
                if "_pause" in cmd:
                    secs = float(cmd["_pause"])
                    if secs > 0:
                        time.sleep(secs)
                        print(f"  (_pause {secs:.2f}s)")
                    continue

                # Handle virtual commands
                if isinstance(cmd.get("T"), str):
                    if cmd["T"] == "track_on":
                        tracker.start()
                        continue
                    elif cmd["T"] == "track_off":
                        tracker.stop()
                        continue
                    elif cmd["T"] == "track_hand":
                        tracker.set_mode("hand")
                        continue
                    elif cmd["T"] == "track_face":
                        tracker.set_mode("face")
                        continue
                    elif cmd["T"] == "restart":
                        print("[restart] Self-restart requested...")
                        # Use os.execv to replace current process with a fresh one
                        import subprocess
                        tracker.stop()
                        time.sleep(1)
                        os.execv(sys.executable, [sys.executable, os.path.join(ROVER_DIR, "rover_brain.py"), "--voice"])
                        continue

                rover.send(cmd)
                print(f"  -> {json.dumps(cmd)}")

                # Auto-wait for gimbal moves (unless an explicit _pause follows)
                t = cmd.get("T")
                wait = 0

                if t == 133:
                    new_pan = cmd.get("X", last_pan)
                    new_tilt = cmd.get("Y", last_tilt)
                    spd = cmd.get("SPD", 200)
                    dist = abs(new_pan - last_pan) + abs(new_tilt - last_tilt)
                    wait = max(0.15, dist / max(spd, 1) * 1.1)
                    last_pan = new_pan
                    last_tilt = new_tilt
                elif t == 141:
                    wait = 0.4

                # Apply fixed inter-command delay for fast commands (nod/shake)
                if delay > 0:
                    wait = max(wait, delay)

                if wait > 0:
                    time.sleep(wait)
                    print(f"  (waited {wait:.2f}s)")
            return last_pan, last_tilt

        if repeat_for > 0:
            # Loop the command list for repeat_for seconds
            deadline = time.time() + repeat_for
            cycle = 0
            print(f"  [repeat] Looping commands for {repeat_for}s")
            while time.time() < deadline:
                cycle += 1
                if cycle > 1:
                    print(f"  [repeat] cycle {cycle}")
                last_pan, last_tilt = _run_commands(commands, last_pan, last_tilt)
            print(f"  [repeat] Done ({cycle} cycles)")
        else:
            last_pan, last_tilt = _run_commands(commands, last_pan, last_tilt)

        execute_response._pan = last_pan
        execute_response._tilt = last_tilt

        # Legacy duration-based auto-stop (still supported as fallback)
        if duration and duration > 0:
            stop_timer = threading.Timer(duration, auto_stop)
            stop_timer.start()
            print(f"  [timer] Auto-stop in {duration}s")

        execute_response._bash_output = bash_output
        return observe

    # --- Observation loop: act → observe → react ---
    MAX_OBSERVE_ROUNDS = 5

    def _map_objects(parsed):
        """Extract objects from LLM response and store in spatial map."""
        if not parsed:
            return
        objects = parsed.get("objects", [])
        if objects and isinstance(objects, list):
            pan = getattr(execute_response, '_pan', 0.0)
            tilt = getattr(execute_response, '_tilt', 0.0)
            wp = pose.body_yaw + pan
            spatial_map.update(objects, wp, tilt)
            print(f"[spatial] Mapped at world_pan={wp:.1f}, tilt={tilt}: {objects}")

    def observation_loop(initial_parsed, original_input):
        """Execute commands with visual feedback. LLM sees camera after each step."""
        nonlocal running

        current_parsed = initial_parsed

        for iteration in range(1, MAX_OBSERVE_ROUNDS + 1):
            wants_observe = execute_response(current_parsed)
            _map_objects(current_parsed)

            if not wants_observe:
                return current_parsed  # final response, done

            # Check for voice interrupt (stop/halt)
            try:
                interrupt = heard_queue.get_nowait()
                if interrupt.lower().strip() in ("stop", "halt", "freeze", "shut up", "be quiet"):
                    fast = match_fast_command(interrupt)
                    if fast:
                        execute_response(fast, source="fast")
                    print("[observe] Interrupted by user")
                    return current_parsed
                else:
                    heard_queue.put(interrupt)  # put it back for later
            except queue.Empty:
                pass

            # Let gimbal settle, then capture frame
            time.sleep(0.5)

            jpeg = tracker.get_jpeg()
            if not jpeg:
                print("[observe] No camera frame, ending loop")
                return current_parsed

            # Build follow-up with context
            cmd_summary = json.dumps(current_parsed.get("commands", []))
            pan = getattr(execute_response, '_pan', 0.0)
            tilt = getattr(execute_response, '_tilt', 0.0)
            follow_up = (
                f"[Observation round {iteration}/{MAX_OBSERVE_ROUNDS}] "
                f"You executed: {cmd_summary}. Your head is at pan={pan}, tilt={tilt}. "
                f"The attached image is what your camera sees RIGHT NOW at this angle. "
                f"Describe ONLY what you actually see — do NOT fabricate or guess. "
                f"Include \"objects\":[...] listing ONLY items visible in THIS image (not from previous rounds). "
                f"Original request: \"{original_input}\". "
                f"IMPORTANT: If the object you're looking for is visible in this image, "
                f"first adjust gimbal to center it, then align your body: rotate body by "
                f"your current pan angle ({pan}°) so gimbal returns to 0. "
                f"Turn duration = |pan|/120 seconds. Include both wheel+gimbal commands. "
                f"What next? If done, omit observe and give your final spoken response."
            )

            print(f"[observe] Round {iteration}/{MAX_OBSERVE_ROUNDS} — sending frame to LLM")
            tracker.pause(30)

            try:
                if hasattr(llm, 'chat_with_image'):
                    raw = llm.chat_with_image(follow_up, jpeg)
                else:
                    raw = llm.chat(follow_up)

                current_parsed = parse_llm_response(raw)
                if not current_parsed:
                    print(f"[observe] Unparseable response: {raw[:200]}")
                    return None
            except Exception as e:
                print(f"[observe] LLM error: {e}")
                return None

        # Max rounds reached — execute whatever we have
        print(f"[observe] Max rounds ({MAX_OBSERVE_ROUNDS}) reached")
        execute_response(current_parsed)
        _map_objects(current_parsed)
        return current_parsed

    # --- Task plan loop: multi-step autonomous execution with LLM feedback ---
    MAX_PLAN_STEPS = 15

    def _execute_plan_step(step):
        """Execute a single plan step. Returns (success: bool, msg: str)."""
        action = step.get("action", "").lower()
        target = step.get("target", "")
        detail = step.get("detail", "")

        print(f"[plan] Executing: {action} target='{target}' detail='{detail}'")

        try:
            if action == "survey":
                return False, "Survey requires local detector (currently disabled)"

            elif action == "find":
                target_lower = target.lower().strip()

                # 1. Check if survey already found this (doors, landmarks)
                if target_lower in ("door", "exit", "doorway") and world_map:
                    door = world_map.find_door()
                    if door:
                        angle = door.get("angle", 0)
                        dist = door.get("dist", 0)
                        print(f"[plan] Door already in map at angle={angle}° dist={dist:.1f}m")
                        # Turn to face the door
                        if abs(angle) > 10:
                            rotation_time = abs(angle) / TURN_RATE_DPS
                            sign = -1 if angle < 0 else 1
                            rover.send({"T": 1, "L": TURN_SPEED * sign, "R": -TURN_SPEED * sign})
                            time.sleep(rotation_time)
                            rover.send({"T": 1, "L": 0, "R": 0})
                            time.sleep(0.3)
                        # Center gimbal forward
                        rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
                        return True, f"Found door at {dist:.1f}m, facing it"

                # 2. Check spatial map
                obj_name, entry = spatial_map.find(target_lower)
                if obj_name and not spatial_map.is_stale(entry):
                    pan = spatial_map.gimbal_pan_for(entry, pose.body_yaw)
                    tilt = entry["tilt"]
                    rover.send({"T": 133, "X": round(pan, 1), "Y": tilt, "SPD": 200, "ACC": 10})
                    time.sleep(0.8)
                    return True, f"Found {obj_name} in spatial map"

                # 3. LLM vision systematic search
                tracker.pause(300)
                found = systematic_search(target, target)
                return found, f"{'Found' if found else 'Could not find'} {target}"

            elif action == "navigate":
                # Try path planner first if we have a map
                if world_map and path_planner:
                    loc = world_map.find_landmark(target)
                    if loc:
                        waypoints = path_planner.plan(loc)
                        if waypoints and path_follower:
                            success = path_follower.follow(waypoints, voice=voice)
                            return success, f"{'Reached' if success else 'Could not reach'} {target}"
                        print(f"[plan] A* failed for {target}, falling back to visual approach")
                # Fallback: visual servo or LLM-based navigation
                if visual_servo:
                    tracker.pause(300)
                    visual_servo._running = True
                    success = visual_servo.scan_and_find(target, voice=voice)
                    if success:
                        return True, f"Reached {target}"
                # Final fallback: LLM step-by-step navigation
                success = navigate_to_object(target, target)
                return success, f"{'Reached' if success else 'Could not reach'} {target}"

            elif action == "approach":
                if visual_servo:
                    tracker.pause(300)
                    visual_servo._running = True
                    success = visual_servo.approach(target, voice=voice)
                    if success:
                        return True, f"Approached {target}"
                # Fallback: LLM-based navigation
                success = navigate_to_object(target, target)
                return success, f"{'Approached' if success else 'Could not approach'} {target}"

            elif action == "drive":
                # detail: "forward 2s" or "backward 3s"
                parts = detail.split() if detail else ["forward", "2"]
                direction = parts[0] if parts else "forward"
                try:
                    secs = float(parts[1].rstrip("s")) if len(parts) > 1 else 2.0
                except ValueError:
                    secs = 2.0
                secs = min(secs, 10.0)  # cap at 10s
                speed = 0.15
                if direction in ("backward", "back", "reverse"):
                    speed = -speed
                rover.send({"T": 1, "L": speed, "R": speed})
                time.sleep(secs)
                rover.send({"T": 1, "L": 0, "R": 0})
                return True, f"Drove {direction} for {secs:.1f}s"

            elif action == "turn":
                # detail: "left 90" or "right 45"
                parts = detail.split() if detail else ["left", "90"]
                direction = parts[0] if parts else "left"
                try:
                    degrees = float(parts[1]) if len(parts) > 1 else 90.0
                except ValueError:
                    degrees = 90.0
                rotation_time = degrees / TURN_RATE_DPS
                sign = -1 if direction == "left" else 1  # left = L-, R+
                rover.send({"T": 1, "L": TURN_SPEED * sign, "R": -TURN_SPEED * sign})
                time.sleep(rotation_time)
                rover.send({"T": 1, "L": 0, "R": 0})
                return True, f"Turned {direction} ~{degrees:.0f} degrees"

            elif action == "look":
                # detail: "left" / "right" / "up" / "down" / "center"
                pan_map = {"left": -90, "right": 90, "center": 0}
                tilt_map = {"up": 60, "down": -20, "center": 0}
                d = detail.lower() if detail else "center"
                pan = pan_map.get(d, 0)
                tilt = tilt_map.get(d, 0)
                rover.send({"T": 133, "X": pan, "Y": tilt, "SPD": 200, "ACC": 10})
                time.sleep(0.8)
                return True, f"Looking {d}"

            elif action == "speak":
                msg = target or detail or ""
                if msg and voice:
                    voice.speak(msg)
                elif msg:
                    print(f"[plan] Speak: {msg}")
                return True, f"Said: {msg}"

            elif action == "wait":
                try:
                    secs = float(target) if target else 2.0
                except ValueError:
                    secs = 2.0
                secs = min(secs, 30.0)
                time.sleep(secs)
                return True, f"Waited {secs:.1f}s"

            elif action == "lights":
                d = (detail or target or "").lower()
                if d in ("on", "bright"):
                    rover.send({"T": 132, "IO4": 200, "IO5": 200})
                elif d in ("off",):
                    rover.send({"T": 132, "IO4": 0, "IO5": 0})
                elif d in ("dim", "low"):
                    rover.send({"T": 132, "IO4": 30, "IO5": 40})
                return True, f"Lights {d}"

            elif action in ("check", "describe", "decide"):
                # Assessment steps — camera + LLM handled by the plan loop
                return True, "assessment"

            else:
                return False, f"Unknown action: {action}"

        except Exception as e:
            rover.send({"T": 1, "L": 0, "R": 0})  # safety stop
            print(f"[plan] Step error: {e}")
            return False, f"Error: {e}"

    def execute_step(step, original_input, step_log):
        """Execute a single plan step with richer return format.
        Assessment steps (check/describe/decide) are handled inline with camera+LLM.
        All other steps delegate to _execute_plan_step.
        Returns {"success": bool, "message": str}."""
        action = step.get("action", "").lower()
        target = step.get("target", "")
        detail = step.get("detail", "")

        if action in ("check", "describe", "decide"):
            # Assessment step: capture frame, call LLM with image
            time.sleep(0.5)  # let camera settle
            jpeg = tracker.get_jpeg()
            if not jpeg:
                print(f"[orchestrator] No camera frame for {action}")
                return {"success": False, "message": f"No camera for {action}"}

            question = detail or target or "What do you see?"
            context = (
                f"You are a rover robot. Original task: \"{original_input}\". "
                f"Current step: {action} — {question}. "
                f"Steps completed so far: {step_log}. "
                f"Look at this image and answer the question. "
                f"Reply with JSON: {{\"speak\":\"<answer>\",\"commands\":[...],\"objects\":[\"visible\",\"items\"]}}"
            )

            tracker.pause(30)
            try:
                if hasattr(llm, 'chat_with_image'):
                    raw = llm.chat_with_image(context, jpeg)
                else:
                    raw = llm.chat(context)
                parsed = parse_llm_response(raw)
                if parsed:
                    execute_response(parsed)
                    _map_objects(parsed)
                    answer = parsed.get("speak", "ok")
                    return {"success": True, "message": f"{action}: {answer[:80]}"}
                else:
                    return {"success": False, "message": f"{action}: unparseable LLM response"}
            except Exception as e:
                print(f"[orchestrator] Assessment error: {e}")
                return {"success": False, "message": f"{action} error: {e}"}

        else:
            # Delegate to existing step executor
            success, msg = _execute_plan_step(step)
            return {"success": success, "message": msg}

    def _speak_summary(goal, step_log, jpeg=None):
        """Ask main LLM for a brief spoken summary of the task outcome."""
        summary_prompt = (
            f"[Task complete] Original request: \"{goal}\". "
            f"Steps executed: {step_log}. "
            f"Give a brief spoken summary of what happened. JSON: "
            f'{{\"speak\":\"<summary>\",\"commands\":[...]}}'
        )
        tracker.pause(30)
        try:
            if jpeg and hasattr(llm, 'chat_with_image'):
                raw = llm.chat_with_image(summary_prompt, jpeg)
            else:
                raw = llm.chat(summary_prompt)
            final = parse_llm_response(raw)
            if final:
                execute_response(final)
                _map_objects(final)
                return final
        except Exception as e:
            print(f"[orchestrator] Summary error: {e}")
        return None

    def task_plan_loop(initial_parsed, original_input):
        """Orchestrator loop: execute a multi-step plan with goal tracking.
        Evaluates goal completion after key steps, re-plans on failure,
        and stops when done, after max iterations, or on voice interrupt."""
        nonlocal running

        MAX_ITERATIONS = 10
        SIMPLE_ACTIONS = {"drive", "turn", "look", "speak", "wait", "lights"}

        plan = list(initial_parsed.get("plan", []))
        if not plan:
            return initial_parsed

        # Execute initial speak/commands
        execute_response(initial_parsed)
        _map_objects(initial_parsed)

        goal = original_input
        step_log = []
        consecutive_failures = 0
        total_steps = 0

        print(f"[orchestrator] Starting — goal: \"{goal}\", {len(plan)} steps")

        for iteration in range(1, MAX_ITERATIONS + 1):
            # 1. Check voice interrupt + emergency
            if emergency_event.is_set():
                emergency_event.clear()
                print("[orchestrator] Interrupted by emergency stop")
                rover.send({"T": 1, "L": 0, "R": 0})
                if voice:
                    voice.speak("Stopping.")
                return initial_parsed
            try:
                interrupt = heard_queue.get_nowait()
                if interrupt.lower().strip() in ("stop", "halt", "freeze", "cancel", "abort"):
                    fast = match_fast_command(interrupt)
                    if fast:
                        execute_response(fast, source="fast")
                    print("[orchestrator] Cancelled by user")
                    return initial_parsed
                else:
                    heard_queue.put(interrupt)  # put back non-stop commands
            except queue.Empty:
                pass

            # 2. If plan empty → evaluate goal
            if not plan:
                print("[orchestrator] Plan empty — evaluating goal")
                jpeg = tracker.get_jpeg()
                eval_prompt = (
                    f"Goal: \"{goal}\"\n"
                    f"All steps completed: {step_log}\n"
                    f"The plan is now empty. Is the goal achieved?"
                )
                result = _orchestrator_llm(eval_prompt, jpeg)
                if result.get("speak") and voice:
                    voice.speak(result["speak"])
                if result.get("done", True):
                    print(f"[orchestrator] Goal achieved: {result.get('reason', 'plan complete')}")
                    final = _speak_summary(goal, step_log, jpeg)
                    return final or initial_parsed
                # Not done — check for new steps
                if result.get("revised_plan"):
                    plan = list(result["revised_plan"])
                    print(f"[orchestrator] Re-planned: {len(plan)} new steps")
                elif result.get("next_step"):
                    plan = [result["next_step"]]
                    print(f"[orchestrator] Added 1 follow-up step")
                else:
                    print("[orchestrator] Goal not done but no new steps — finishing")
                    final = _speak_summary(goal, step_log, jpeg)
                    return final or initial_parsed

            # 3. Pop next step
            step = plan.pop(0)
            total_steps += 1
            action = step.get("action", "").lower()
            print(f"[orchestrator] Step {total_steps} (iter {iteration}/{MAX_ITERATIONS}): "
                  f"{action} '{step.get('target', '')}'")

            # 4. Execute step
            result = execute_step(step, original_input, step_log)
            step_log.append(f"{action}: {result['message'][:60]}")
            print(f"[orchestrator] Result: {'OK' if result['success'] else 'FAIL'} — {result['message'][:80]}")

            # 5. Track consecutive failures
            if result["success"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("[orchestrator] 3 consecutive failures — aborting")
                    if voice:
                        voice.speak("Having trouble. Stopping.")
                    final = _speak_summary(goal, step_log, tracker.get_jpeg())
                    return final or initial_parsed

            # 6. Decide whether to evaluate goal
            is_last_iteration = (iteration == MAX_ITERATIONS)
            plan_empty = (len(plan) == 0)
            step_failed = not result["success"]
            should_evaluate = (
                action not in SIMPLE_ACTIONS
                or step_failed
                or plan_empty
                or is_last_iteration
            )

            if not should_evaluate:
                continue

            # 7. Evaluate goal via orchestrator LLM
            print(f"[orchestrator] Evaluating goal after '{action}'")
            time.sleep(0.3)
            jpeg = tracker.get_jpeg()
            remaining_desc = json.dumps(plan) if plan else "[]"
            eval_prompt = (
                f"Goal: \"{goal}\"\n"
                f"Steps completed: {step_log}\n"
                f"Last step result: {result['message']}\n"
                f"Remaining plan: {remaining_desc}\n"
                f"Is the goal achieved? If not, should the plan be revised?"
            )
            eval_result = _orchestrator_llm(eval_prompt, jpeg)

            if eval_result.get("speak") and voice:
                voice.speak(eval_result["speak"])

            # 8. Act on evaluation
            if eval_result.get("done"):
                reason = eval_result.get("reason", "goal achieved")
                print(f"[orchestrator] Done: {reason}")
                final = _speak_summary(goal, step_log, jpeg)
                return final or initial_parsed

            if eval_result.get("revised_plan"):
                plan = list(eval_result["revised_plan"])
                print(f"[orchestrator] Revised plan: {len(plan)} steps")
            elif eval_result.get("next_step"):
                plan.insert(0, eval_result["next_step"])
                print(f"[orchestrator] Inserted step: {eval_result['next_step'].get('action')}")

        # Max iterations reached
        print(f"[orchestrator] Max iterations ({MAX_ITERATIONS}) reached. Steps: {total_steps}")
        jpeg = tracker.get_jpeg()
        final = _speak_summary(goal, step_log, jpeg)
        return final or initial_parsed

    # --- Search engine: unified priority-scan + LLM-guided navigation ---
    from search_engine import SearchEngine, load_prompts as _load_prompts

    # Load prompt templates
    _prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    _all_prompts = {}
    for _pf in ("search.md", "nav.md"):
        _ppath = os.path.join(_prompts_dir, _pf)
        if os.path.exists(_ppath):
            _all_prompts.update(_load_prompts(_ppath))
            print(f"[search] Loaded prompts from {_pf}")

    search_engine = SearchEngine(
        rover=rover,
        tracker=tracker,
        pose=pose,
        spatial_map=spatial_map,
        llm_vision_fn=_tracker_vision,
        parse_fn=parse_llm_response,
        voice_fn=voice.speak if voice else None,
        prompts=_all_prompts,
    )
    print(f"[search] SearchEngine ready (prompts: {list(_all_prompts.keys())})")

    def systematic_search(target_name, original_input):
        """Thin wrapper — delegates to SearchEngine."""
        return search_engine.search(target_name)

    def navigate_to_object(target_name, original_input):
        """Thin wrapper — delegates to SearchEngine."""
        return search_engine.navigate_to(target_name)


    # --- Calibration mode: slow movements, no speech, voice-controlled ---
    CALIB_SPEED = 0.1   # very slow m/s
    CALIB_TURN = 0.12   # very slow turn speed

    def calibration_mode():
        """Enter calibration mode. Listens for voice commands, moves very slowly, no TTS."""
        print("[calibrate] === CALIBRATION MODE ===")
        print("[calibrate] Commands: forward, back, left, right, stop, done")
        print("[calibrate] Speed: very slow. No speech output.")
        tracker.pause(600)
        pose.reset_yaw()

        # Drain any pending voice input
        while not heard_queue.empty():
            try:
                heard_queue.get_nowait()
            except queue.Empty:
                break

        current_cmd = None  # what we're currently doing
        start_time = None
        total_distance = 0.0
        total_rotation = 0.0

        while True:
            # Check for voice/text input
            cmd_text = None
            try:
                cmd_text = heard_queue.get(timeout=0.2)
            except queue.Empty:
                pass

            if cmd_text:
                lower = cmd_text.lower().strip()
                print(f"[calibrate] Heard: '{lower}'")

                if lower in ("done", "exit", "quit", "stop calibrate", "end"):
                    rover.send({"T": 1, "L": 0, "R": 0})
                    break
                elif lower in ("stop", "halt", "freeze"):
                    rover.send({"T": 1, "L": 0, "R": 0})
                    if current_cmd and start_time:
                        elapsed = time.time() - start_time
                        if current_cmd in ("forward", "back"):
                            dist = CALIB_SPEED * elapsed
                            total_distance += dist
                            print(f"[calibrate] Moved {dist:.3f}m in {elapsed:.1f}s")
                        elif current_cmd in ("left", "right"):
                            degs = TURN_RATE_DPS * (CALIB_TURN / TURN_SPEED) * elapsed
                            total_rotation += degs
                            print(f"[calibrate] Rotated ~{degs:.1f} deg in {elapsed:.1f}s")
                            print(f"[calibrate] Pose yaw: {pose.body_yaw:.1f} deg")
                    current_cmd = None
                    start_time = None
                    print("[calibrate] Stopped")
                elif lower in ("forward", "go"):
                    rover.send({"T": 1, "L": CALIB_SPEED, "R": CALIB_SPEED})
                    current_cmd = "forward"
                    start_time = time.time()
                    print(f"[calibrate] Forward at {CALIB_SPEED} m/s")
                elif lower in ("back", "backward", "reverse"):
                    rover.send({"T": 1, "L": -CALIB_SPEED, "R": -CALIB_SPEED})
                    current_cmd = "back"
                    start_time = time.time()
                    print(f"[calibrate] Backward at {CALIB_SPEED} m/s")
                elif lower in ("left", "turn left"):
                    rover.send({"T": 1, "L": -CALIB_TURN, "R": CALIB_TURN})
                    current_cmd = "left"
                    start_time = time.time()
                    print(f"[calibrate] Turning left at {CALIB_TURN} m/s")
                elif lower in ("right", "turn right"):
                    rover.send({"T": 1, "L": CALIB_TURN, "R": -CALIB_TURN})
                    current_cmd = "right"
                    start_time = time.time()
                    print(f"[calibrate] Turning right at {CALIB_TURN} m/s")
                elif lower in ("pose", "status", "info"):
                    p = pose.get_pose()
                    print(f"[calibrate] Pose: {p}")
                    print(f"[calibrate] Total distance: {total_distance:.3f}m")
                    print(f"[calibrate] Total rotation: {total_rotation:.1f} deg")
                    print(f"[calibrate] TURN_RATE_DPS={TURN_RATE_DPS}, TURN_SPEED={TURN_SPEED}")
                    print(f"[calibrate] Wheel: {WHEEL_DIAMETER_M*1000:.0f}mm dia, "
                          f"{WHEEL_CIRCUMFERENCE_M*1000:.0f}mm circumference")
                    dist_per_sec = CALIB_SPEED  # m/s is distance per second
                    print(f"[calibrate] At {CALIB_SPEED} m/s: {dist_per_sec*100:.1f} cm/s")

        rover.send({"T": 1, "L": 0, "R": 0})
        p = pose.get_pose()
        print(f"[calibrate] === END CALIBRATION ===")
        print(f"[calibrate] Final pose: {p}")
        print(f"[calibrate] Total distance: {total_distance:.3f}m, rotation: {total_rotation:.1f} deg")
        print(f"[calibrate] Tip: if 360 spin took Xs, set TURN_RATE_DPS = {360.0}/X")

    # --- xAI Realtime tool dispatch (closure capturing rover state) ---
    def xai_tool_dispatch(fn_name, args):
        """Dispatch xAI function tool calls to rover hardware/logic. Returns JSON string."""
        try:
            if fn_name == "send_rover_commands":
                raw_cmds = args.get("commands", [])
                dur = args.get("duration", 0)
                # Normalize commands: Gemini sometimes sends strings or dicts with quoted keys
                cmds = []
                for c in raw_cmds:
                    if isinstance(c, str):
                        try:
                            c = json.loads(c)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    if isinstance(c, dict):
                        # Fix quoted keys: {"\"T\"": 133} -> {"T": 133}
                        fixed = {}
                        for k, v in c.items():
                            clean_k = k.strip('"')
                            fixed[clean_k] = v
                        cmds.append(fixed)
                parsed = {"commands": cmds, "speak": "", "duration": dur}
                execute_response(parsed, source="xai")
                return json.dumps({"status": "ok", "commands_sent": len(cmds)})

            elif fn_name == "look_at_camera":
                pan = args.get("pan", 0)
                tilt = args.get("tilt", 0)
                question = args.get("question", "Describe what you see briefly.")
                # Move gimbal
                rover.send({"T": 133, "X": pan, "Y": tilt, "SPD": 300, "ACC": 20})
                time.sleep(0.8)
                # Capture and ask text LLM
                jpeg = tracker.get_jpeg()
                if not jpeg:
                    return json.dumps({"error": "No camera frame available"})
                description = llm.chat_with_image(question, jpeg)
                # Strip JSON wrapping if text LLM returned JSON
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
                success = navigate_to_object(target, f"navigate to {target}")
                return json.dumps({"success": bool(success), "target": target})

            elif fn_name == "search_for":
                target = args.get("target", "")
                if not target:
                    return json.dumps({"error": "No target specified"})
                found = systematic_search(target, f"search for {target}")
                if found:
                    # Automatically navigate to the found object
                    success = navigate_to_object(target, f"navigate to {target}")
                    return json.dumps({"found": True, "navigated": bool(success), "target": target})
                return json.dumps({"found": False, "target": target})

            elif fn_name == "remember":
                note = args.get("note", "")
                if note:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    with open(MEMORY_FILE, "a") as f:
                        f.write(f"- {note} [{timestamp}]\n")
                    return json.dumps({"status": "remembered", "note": note})
                return json.dumps({"error": "Empty note"})

            elif fn_name == "get_status":
                fb = rover.read_feedback()
                p = pose.get_pose()
                status = {
                    "battery_v": fb.get("v", 0) if fb else 0,
                    "pose": {"x": round(p["x"], 2), "y": round(p["y"], 2),
                             "heading": round(p["heading"], 1)},
                    "tracker": tracker.status,
                    "speed_scale": rover.speed_scale,
                    "spatial_objects": len(spatial_map._map),
                }
                return json.dumps(status)

            elif fn_name == "set_speed":
                level = max(1, min(10, int(args.get("level", 2))))
                rover.speed_scale = level / 10.0
                pct = int(rover.speed_scale * 100)
                print(f"[speed] Set to {pct}% (level {level})")
                return json.dumps({"status": "ok", "speed_percent": pct})

            else:
                return json.dumps({"error": f"Unknown function: {fn_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # --- Gemini Live WebSocket client ---
    gemini_live = None
    try:
        from gemini_live import GeminiLiveClient
        gemini_live = GeminiLiveClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_LIVE_MODEL,
            tool_dispatch_fn=xai_tool_dispatch,
            result_queue=llm_result_queue,
            playback_device=USB_PLAYBACK,
            voice=GEMINI_TTS_VOICE,
            mic_mute_fn=voice._mic_mute if voice else None,
            mic_unmute_fn=voice._mic_unmute if voice else None,
        )
        gemini_live.start()
        print(f"[gemini-live] WebSocket client started ({GEMINI_LIVE_MODEL})")
    except Exception as e:
        print(f"[gemini-live] Failed to start: {e}")
        gemini_live = None

    # --- Voice listener thread (always listening, never blocked by LLM) ---
    def voice_listener():
        """Runs on its own thread. Continuously listens and puts heard text into heard_queue."""
        while running:
            try:
                text = voice.listen_continuous()
                if text and text != "__STOP__":
                    heard_queue.put(text)
                # __STOP__ was already handled by _fire_emergency inside listen_continuous
            except Exception as e:
                print(f"[voice-thread] Error: {e}")
                time.sleep(0.5)

    # --- LLM worker thread (picks up jobs, doesn't block voice) ---
    def llm_worker():
        """Runs on its own thread. Processes LLM requests from llm_request_queue.
        Forwards to Gemini Live WebSocket if connected, falls back to REST."""
        nonlocal llm_busy
        while running:
            try:
                user_input = llm_request_queue.get(timeout=0.5)
                # Drain queue — only process the latest request
                while not llm_request_queue.empty():
                    try:
                        user_input = llm_request_queue.get_nowait()
                        print(f"\n[llm-thread] Skipped stale request, using latest")
                    except queue.Empty:
                        break
                llm_busy = True

                if tracker.is_active:
                    seeing = "I CAN see a face right now" if tracker.status == "tracking" else "I CANNOT see anyone right now"
                    user_input += f" [Tracker: {seeing}]"

                # Try Gemini Live WebSocket first
                if gemini_live and gemini_live.is_connected:
                    gemini_live.send_text(user_input)
                    print(f"[llm-thread] → Gemini Live (WebSocket)")
                    # Signal main loop: handled via Live (audio plays directly)
                    llm_result_queue.put("__LIVE__")
                    llm_busy = False
                    continue

                # Fallback: REST API (if WebSocket down >10s or not available)
                if gemini_live and gemini_live.seconds_disconnected < 10:
                    # WebSocket just disconnected — wait briefly for reconnect
                    print(f"[llm-thread] WebSocket reconnecting, waiting...")
                    time.sleep(3)
                    if gemini_live.is_connected:
                        gemini_live.send_text(user_input)
                        print(f"[llm-thread] → Gemini Live (reconnected)")
                        llm_result_queue.put("__LIVE__")
                        llm_busy = False
                        continue

                print(f"[llm-thread] → REST fallback")
                for attempt in range(3):
                    try:
                        raw = llm.chat(user_input)
                        llm_result_queue.put(raw)
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "rate" in err_str.lower():
                            wait = 2 ** attempt
                            print(f"\n[llm-thread] Rate limited, retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"\n[llm-thread] Error: {e}")
                            llm_result_queue.put(json.dumps({"commands": [], "speak": "Sorry, error."}))
                            break
                else:
                    llm_result_queue.put(json.dumps({"commands": [], "speak": "Rate limited, try again."}))

                llm_busy = False
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[llm-thread] Error: {e}")
                llm_result_queue.put(json.dumps({"commands": [], "speak": f"Error: {e}"}))
                llm_busy = False

    llm_request_queue = queue.Queue()

    # --- Stdin reader thread (always available for typed commands) ---
    def stdin_reader():
        """Read typed commands from terminal, even in voice mode."""
        import select
        while running:
            try:
                if select.select([sys.stdin], [], [], 0.3)[0]:
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break
                    line = line.strip()
                    if line:
                        heard_queue.put(line)
            except Exception:
                break

    # --- Auto-calibrate disabled (YOLO off — LLM handles recognition) ---

    # --- Main loop ---
    print("\nCommands: natural language, 'track on/off', 'status', 'quit'")
    print("Type commands in terminal at any time.")
    el_voice = None
    if args.elevenlabs and args.voice and ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID:
        try:
            from elevenlabs_voice import ElevenLabsVoice
            el_voice = ElevenLabsVoice(
                api_key=ELEVENLABS_API_KEY,
                agent_id=ELEVENLABS_AGENT_ID,
                mic_device=MIC_DEVICE,
                playback_device=USB_PLAYBACK,
                tool_dispatch_fn=xai_tool_dispatch,
                emergency_event=emergency_event,
            )
            el_voice.start()
            print("ElevenLabs Conversational AI: ACTIVE")

            # Wrap voice.speak() to mute ElevenLabs mic during Groq TTS playback
            if voice:
                _original_speak_sync = voice._speak_sync
                def _el_aware_speak_sync(text, _orig=_original_speak_sync):
                    if el_voice:
                        el_voice._mic_muted = True
                    try:
                        _orig(text)
                    finally:
                        if el_voice:
                            time.sleep(0.3)
                            el_voice._mic_muted = False
                voice._speak_sync = _el_aware_speak_sync

        except Exception as e:
            print(f"[elevenlabs] Failed to start: {e}")
            el_voice = None

    if not el_voice and args.voice and voice and voice.available:
        print("Voice mode: always listening. Just speak commands.")
        threading.Thread(target=voice_listener, daemon=True).start()
    threading.Thread(target=stdin_reader, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    print("-" * 60)

    while running:
        try:
            user_input = None

            # --- Collect input from voice or stdin (both feed heard_queue) ---
            try:
                user_input = heard_queue.get(timeout=0.2)
            except queue.Empty:
                pass

            # --- Process any LLM results that came back ---
            while not llm_result_queue.empty():
                try:
                    raw = llm_result_queue.get_nowait()
                    if raw == "__LIVE__":
                        continue  # Gemini Live handled via audio
                    parsed = parse_llm_response(raw)
                    if parsed:
                        execute_response(parsed)
                    else:
                        print(f"[llm] Unparseable response:\n{raw[:300]}")
                except queue.Empty:
                    break

            if not user_input:
                continue

            lower = user_input.lower().strip()

            if lower in ("quit", "exit", "q"):
                break
            elif lower == "status":
                rover.send({"T": 130})
                time.sleep(0.5)
                fb = rover.read_feedback()
                print(f"[feedback] {json.dumps(fb, indent=2)}" if fb else "[feedback] No data")
                print(f"[tracker] {tracker.status} | active={tracker.is_active}")
                print(f"[pose] {pose.get_pose()}")
                continue
            elif lower in ("calibrate", "calibration", "calibration mode"):
                calibration_mode()
                continue

            # --- Speed limiter: "speed 1" = 10%, "speed 5" = 50%, "speed 10" = 100% ---
            speed_match = re.match(r"speed\s+(\d+)", lower)
            if speed_match:
                level = int(speed_match.group(1))
                level = max(1, min(10, level))
                rover.speed_scale = level / 10.0
                pct = int(rover.speed_scale * 100)
                print(f"[speed] Set to {pct}% (level {level})")
                if voice:
                    voice.speak(f"Speed {pct} percent.")
                continue

            # --- Voice switching ---
            voice_match = re.match(
                r"(?:switch|change|set)\s*(?:to\s*|the\s*)?(?:voice|tts)\s*(?:to\s*)?(male|female|boy|girl|man|woman|troy|hannah|austin)",
                lower)
            if not voice_match:
                voice_match = re.match(r"^(male|female)\s*voice$", lower)
            if voice_match and voice:
                pick = voice_match.group(1).strip()
                # Groq Orpheus only supports: troy (male), hannah (female), austin (male)
                VOICE_MAP = {
                    "male": "troy", "boy": "troy", "man": "troy",
                    "female": "hannah", "girl": "hannah", "woman": "hannah",
                }
                new_voice = VOICE_MAP.get(pick, pick)
                voice.tts_voice = new_voice
                print(f"[voice] Switched to: {new_voice}")
                voice.speak(f"Voice switched.")
                continue

            # --- "survey" / "map room" — disabled (needs YOLO) ---
            if lower in ("survey", "survey room", "map room", "map", "scan room"):
                if voice:
                    voice.speak("Survey not available without detector.")
                print("[nav] Survey disabled — LLM-only mode")
                continue

            # --- "calibrate objects" — disabled (YOLO off, LLM handles recognition) ---
            if lower in ("calibrate objects", "fix labels", "correct detections",
                          "calibrate labels", "calibrate detections"):
                if voice:
                    voice.speak("No need. I use my eyes now.")
                print("[calibrate] Disabled — LLM-only recognition mode")
                continue

            # --- "go to door" / "go to kitchen" / "navigate to door" ---
            door_match = re.match(
                r"(?:go\s*to|navigate\s*to|drive\s*to|go\s*through|exit\s*through)\s*(?:the\s*)?(?:door|kitchen|hallway|exit|outside)",
                lower)
            if door_match:
                # Extract destination (what's after the door — kitchen, hallway, etc.)
                dest = "door"
                for w in ("kitchen", "hallway", "outside", "exit"):
                    if w in lower:
                        dest = w
                        break
                # Build a step-by-step plan and let the plan loop handle it
                plan_steps = [
                    {"action": "survey", "target": "room"},
                    {"action": "find", "target": "door"},
                    {"action": "check", "target": "door", "detail": "I'm facing the door. How far is it? What's in the way?"},
                    {"action": "drive", "target": "", "detail": "forward 2s"},
                    {"action": "check", "target": "door", "detail": "Am I closer to the door now? Should I keep driving or adjust?"},
                    {"action": "drive", "target": "", "detail": "forward 2s"},
                    {"action": "check", "target": "door", "detail": "Am I at the door? Can I drive through it?"},
                    {"action": "drive", "target": "", "detail": "forward 3s"},
                    {"action": "check", "target": dest, "detail": f"Did I make it through? Am I in or near the {dest}?"},
                ]
                planned = {"commands": [], "speak": f"Heading to the {dest}.", "plan": plan_steps}
                final = task_plan_loop(planned, user_input)
                log_exchange(user_input, final or planned, source="llm")
                continue

            # --- "go to X" / "navigate to X" / "drive to X" ---
            nav_match = re.match(
                r"(?:go\s*to|navigate\s*to|drive\s*to|move\s*to|head\s*towards?\s*)\s*(?:the\s*|my\s*)?(.+)",
                lower)
            if nav_match:
                nav_target = nav_match.group(1).strip().rstrip("?.!")
                if visual_servo:
                    # Use fast local detection + visual servo (no LLM needed)
                    tracker.pause(300)
                    visual_servo._running = True
                    success = visual_servo.scan_and_find(nav_target, voice=voice)
                    log_exchange(user_input, {"speak": f"{'Found' if success else 'Could not find'} {nav_target}"}, source="llm")
                elif world_map and path_planner:
                    # Use path planner if available
                    loc = world_map.find_landmark(nav_target)
                    if loc:
                        waypoints = path_planner.plan(loc)
                        if waypoints and path_follower:
                            path_follower.follow(waypoints, voice=voice)
                    else:
                        navigate_to_object(nav_target, user_input)
                else:
                    # Fallback: LLM-based navigation
                    navigate_to_object(nav_target, user_input)
                    log_exchange(user_input, {"speak": f"Navigated to {nav_target}"}, source="llm")
                continue

            # --- Try fast command cache first (instant, no LLM) ---
            fast = match_fast_command(user_input)
            if fast:
                tracker.pause(10)
                execute_response(fast, source="fast")
                log_exchange(user_input, fast, source="fast")
                continue

            # --- Check spatial map for "where is X" / "find X" queries ---
            spatial_match = re.match(
                r"(?:where\s*(?:is|are)\s*(?:the\s*|my\s*)?|find\s*(?:the\s*|my\s*)?|look\s*(?:at|for)\s*(?:the\s*|my\s*)?)(.+)",
                lower)
            if spatial_match:
                query = spatial_match.group(1).strip().rstrip("?.!")
                # Skip pronouns — these should go to the LLM
                if query in ("me", "us", "him", "her", "them", "you", "it", "this", "that"):
                    spatial_match = None
            if spatial_match:
                obj_name, entry = spatial_map.find(query)
                if obj_name and not spatial_map.is_stale(entry):
                    pan = spatial_map.gimbal_pan_for(entry, pose.body_yaw)
                    tilt = entry["tilt"]
                    age = int(time.time() - entry["time"])
                    print(f'[spatial] Found "{obj_name}" at pan={round(pan,1)}, tilt={tilt} ({age}s ago)')
                    tracker.pause(30)
                    response = {
                        "commands": [{"T": 133, "X": round(pan, 1), "Y": tilt, "SPD": 200, "ACC": 10}],
                        "speak": f"I remember seeing {obj_name} over here.",
                        "observe": True,
                    }
                    final = observation_loop(response, user_input)
                    log_exchange(user_input, final or response, source="llm")
                    continue
                else:
                    # Object not in map OR stale — run systematic search
                    print(f'[spatial] "{query}" not fresh in map, starting systematic search')
                    found = systematic_search(query, user_input)
                    if found:
                        # Look at it now
                        obj_name2, entry2 = spatial_map.find(query)
                        if obj_name2:
                            tracker.pause(30)
                            response = {
                                "commands": [{"T": 133,
                                              "X": round(spatial_map.gimbal_pan_for(entry2, pose.body_yaw), 1),
                                              "Y": entry2["tilt"],
                                              "SPD": 200, "ACC": 10}],
                                "speak": f"Found {obj_name2} over here.",
                                "observe": True,
                            }
                            final = observation_loop(response, user_input)
                            log_exchange(user_input, final or response, source="llm")
                    else:
                        log_exchange(user_input, {"speak": f"Couldn't find {query}"}, source="llm")
                    continue

            # --- Learned command cache DISABLED (responses are context-dependent) ---

            # --- Pre-filter: skip likely noise before burning an LLM call ---
            words = user_input.split()
            if len(words) <= 3 and not any(c.isalpha() for c in user_input):
                print(f'[skip] No words: "{user_input}"')
                continue

            # --- Multi-step hint: detect "X and Y", "X then Y" patterns ---
            if re.search(r'\b(and then|then|and also|after that|,\s*then|go to .+ and|check .+ and|find .+ and)\b', lower):
                user_input += ' [Return a "plan" field with steps.]'

            # --- Send to LLM and wait, but keep processing fast commands ---
            tracker.pause(90)  # keep paused while waiting for LLM
            llm_request_queue.put(user_input)
            pending_input = user_input  # remember for caching
            print("[thinking...]", flush=True)
            got_response = False
            emergency_event.clear()
            for _ in range(300):  # max ~90s wait
                if not running or emergency_event.is_set():
                    if emergency_event.is_set():
                        emergency_event.clear()
                        print("[EMERGENCY] Aborting LLM wait")
                        got_response = True  # don't print timeout
                    break
                # Check for LLM response
                try:
                    raw = llm_result_queue.get(timeout=0.3)
                    # Gemini Live handled it — audio plays directly, skip parse
                    if raw == "__LIVE__":
                        print("[gemini-live] Response via audio")
                        got_response = True
                        break
                    parsed = parse_llm_response(raw)
                    if parsed:
                        # Silent response = LLM ignoring noise
                        is_silent = not parsed.get("speak") and not parsed.get("commands")
                        if is_silent:
                            print(f'[ignored] "{pending_input}"')
                        elif parsed.get("plan"):
                            final = task_plan_loop(parsed, pending_input)
                            log_exchange(pending_input, final or parsed, source="llm")
                        elif parsed.get("observe"):
                            final = observation_loop(parsed, pending_input)
                            log_exchange(pending_input, final or parsed, source="llm")
                        elif parsed.get("bash"):
                            # Agentic bash loop — LLM can run multiple commands
                            final = bash_agent_loop(parsed)
                            log_exchange(pending_input, final or parsed, source="llm")
                        else:
                            execute_response(parsed)
                            log_exchange(pending_input, parsed, source="llm")
                    else:
                        print(f"[llm] Unparseable response:\n{raw[:300]}")
                    got_response = True
                    break
                except queue.Empty:
                    pass
                # While waiting, process any fast commands that came in
                try:
                    pending = heard_queue.get_nowait()
                    lower_p = pending.lower().strip()
                    if lower_p in ("quit", "exit", "q"):
                        running = False
                        break
                    fast_p = match_fast_command(pending)
                    if fast_p:
                        print()  # newline after [thinking...]
                        execute_response(fast_p, source="fast")
                        print("[thinking...]", flush=True)
                    else:
                        # Queue non-fast commands for after current LLM response
                        llm_request_queue.put(pending)
                        print(f'[queued] "{pending}"', flush=True)
                except queue.Empty:
                    pass
            if not got_response and running:
                print("[timeout] No LLM response, skipping.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[error] {e}")

    # --- Cleanup ---
    print("\nShutting down...")
    running = False
    if gemini_live:
        gemini_live.stop()
    if el_voice:
        el_voice.stop()
    if stop_timer and stop_timer.is_alive():
        stop_timer.cancel()
    tracker.stop()
    rover.send({"T": 1, "L": 0, "R": 0})
    rover.send({"T": 135})
    time.sleep(0.3)
    rover.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
