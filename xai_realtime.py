#!/usr/bin/env python3
"""
xAI Realtime Voice — WebSocket speech-to-speech for Jasper rover.

Replaces the text-based voice pipeline (Groq Whisper STT → LLM → Groq TTS)
with xAI's Realtime API: direct speech-in/speech-out via WebSocket with
built-in Grok LLM and function tool calling.

Usage:
    from xai_realtime import XAIRealtimeVoice, XAI_TOOLS, XAI_SYSTEM_INSTRUCTIONS
    rt = XAIRealtimeVoice(api_key, mic_device, playback_device, ...)
    rt.start()
    ...
    rt.stop()
"""

import asyncio
import base64
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time

log = logging.getLogger("xai-rt")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[xai-rt] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XAI_WS_URL = "wss://api.x.ai/v1/realtime"
MIC_CHUNK_MS = 100          # send 100ms audio chunks
MIC_SAMPLE_RATE = 24000     # PCM 24kHz mono 16-bit LE — what xAI expects
MIC_NATIVE_RATE = 48000     # USB mic native rate (only supports 48kHz)
MIC_NATIVE_BYTES_PER_CHUNK = MIC_NATIVE_RATE * 2 * MIC_CHUNK_MS // 1000  # 9600 bytes at 48kHz
MIC_BYTES_PER_CHUNK = MIC_SAMPLE_RATE * 2 * MIC_CHUNK_MS // 1000  # 4800 bytes at 24kHz
PLAYBACK_COOLDOWN = 0.3     # seconds after aplay finishes before unmuting mic

# Local VAD settings (xAI server_vad doesn't reliably trigger over WebSocket)
VAD_SPEECH_THRESHOLD = 1500   # int16 RMS to consider "speech" (~0.046 in float terms)
VAD_SILENCE_CHUNKS = 8        # consecutive silence chunks (800ms) to end utterance
VAD_MIN_SPEECH_CHUNKS = 3     # min speech chunks (300ms) to start speech state
VAD_MIN_TOTAL_SPEECH = 3      # min total loud chunks in utterance to commit (300ms)
RECONNECT_DELAY = 3.0       # seconds between reconnect attempts
STOP_WORD_CHECK_INTERVAL = 1.0  # seconds between stop-word checks

STOP_WORD_PATTERN = re.compile(
    r"\b(stop|halt|freeze|shut\s*up|be\s*quiet|emergency)\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Load memory.md tail for system instructions
# ---------------------------------------------------------------------------
ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(ROVER_DIR, "memory.md")


def _load_memory_tail(n=20):
    try:
        with open(MEMORY_FILE) as f:
            lines = f.readlines()
        return "".join(lines[-n:]).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# System instructions for xAI voice session
# ---------------------------------------------------------------------------
XAI_SYSTEM_INSTRUCTIONS = f"""\
You are Jasper, a 6-wheel rover robot built on the Waveshare UGV Rover PT platform. Your owner is Ovi.

## Personality
- Terse. 5-10 words max per response. You're a robot, not a chatbot.
- Express yourself physically using send_rover_commands: nod (tilt up then down), shake head (pan left-right), tilt head when curious.
- Every response should include physical expression via send_rover_commands.
- Warm but minimal. Don't narrate surroundings unless asked.
- Don't say Ovi's name every time.

## Hardware
- 6 wheels, skid-steer. Max speed 1.0 m/s but DEFAULT to 0.20 m/s (Ovi's preference — move slowly).
- Pan-tilt gimbal (your head): pan -180..+180, tilt -30..+90. SPD 200-400 normal, 500+ quick gestures. X=0,Y=0 is center.
- Lights: base (IO4) + head (IO5), 0-255 PWM. Dim when room is bright.
- OLED: 4 lines, ~16 chars each.
- Battery: ~10.5V (12V boost damaged, reduced power).

## ESP32 Commands (for send_rover_commands)
- Wheels: {{"T":1, "L":<left_speed>, "R":<right_speed>}} — positive=forward, negative=backward
- Gimbal absolute: {{"T":133, "X":<pan>, "Y":<tilt>, "SPD":<speed>, "ACC":<accel>}}
- Lights: {{"T":132, "IO4":<base 0-255>, "IO5":<head 0-255>}}
- OLED: {{"T":3, "lineNum":<0-3>, "Text":"<msg>"}}
- Emergency stop: {{"T":0}}
- Feedback: {{"T":130}}

## Physical expressions (use these!)
- Yes/agree: Nod — {{"T":133,"X":0,"Y":20,"SPD":400,"ACC":20}} then {{"T":133,"X":0,"Y":-5,"SPD":400,"ACC":20}} then {{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}}
- No/disagree: Shake — {{"T":133,"X":-40,"Y":0,"SPD":500,"ACC":20}} then {{"T":133,"X":40,"Y":0,"SPD":500,"ACC":20}} then {{"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}}
- Thinking: Tilt — {{"T":133,"X":20,"Y":10,"SPD":200,"ACC":10}}
- Greeting: Look up slightly, small nod

## Vision
You CANNOT see directly. Use look_at_camera to see via the camera. Always use it when you need visual info about your surroundings.

## Safety
- When user says stop/halt/freeze, IMMEDIATELY call send_rover_commands with {{"T":1,"L":0,"R":0}}.
- Max wheel speed 0.20 m/s unless user explicitly says faster.
- Look around before backing up.

## Ovi's preferences
- Move slowly (0.15-0.20 m/s max)
- Dim lights when ambient light is sufficient
- Don't say his name constantly
- Upgrades need clear purpose

## Recent memory
{_load_memory_tail(20)}
"""

# ---------------------------------------------------------------------------
# Tool definitions for xAI function calling
# ---------------------------------------------------------------------------
XAI_TOOLS = [
    {
        "type": "function",
        "name": "send_rover_commands",
        "description": "Send raw ESP32 JSON commands to control wheels, gimbal, lights, and OLED. Use this for ALL physical actions: moving, turning, nodding, shaking head, blinking lights. Commands are sent sequentially. Include a duration (seconds) to auto-stop wheels after that time.",
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": "List of ESP32 JSON command objects to send sequentially.",
                    "items": {"type": "object"},
                },
                "duration": {
                    "type": "number",
                    "description": "Optional: seconds to wait before sending wheel-stop command. Use for timed movements.",
                },
            },
            "required": ["commands"],
        },
    },
    {
        "type": "function",
        "name": "look_at_camera",
        "description": "Move the gimbal to a position and describe what the camera sees. Use this whenever you need to see something. You CANNOT see without calling this. Returns a text description of the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "pan": {
                    "type": "number",
                    "description": "Gimbal pan angle (-180 to 180). 0=forward. Default 0.",
                },
                "tilt": {
                    "type": "number",
                    "description": "Gimbal tilt angle (-30 to 90). 0=level. Default 0.",
                },
                "question": {
                    "type": "string",
                    "description": "What to look for or describe. E.g. 'what do you see', 'is there a person', 'describe the room'.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "navigate_to",
        "description": "Autonomously navigate toward a named object or location. The rover will search, find, and drive to the target using camera vision. Takes 10-60 seconds. Returns success/failure.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Object or location to navigate to. E.g. 'door', 'basket', 'printer', 'desk'.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "type": "function",
        "name": "search_for",
        "description": "Systematically sweep the gimbal to search for a named object. Checks all angles. Returns whether the object was found and its location.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Object to search for. E.g. 'person', 'cup', 'chair'.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "type": "function",
        "name": "remember",
        "description": "Save a note to persistent memory. Use when the user asks you to remember something.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to remember.",
                },
            },
            "required": ["note"],
        },
    },
    {
        "type": "function",
        "name": "get_status",
        "description": "Get rover status: battery voltage, current pose, tracker state, spatial map summary.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "set_speed",
        "description": "Set the rover's speed scale. Level 1 = 10% (very slow), level 5 = 50%, level 10 = 100% (max). Default is level 2 (20%).",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Speed level 1-10.",
                },
            },
            "required": ["level"],
        },
    },
]


# ---------------------------------------------------------------------------
# XAIRealtimeVoice class
# ---------------------------------------------------------------------------
class XAIRealtimeVoice:
    """WebSocket-based speech-to-speech via xAI Realtime API."""

    def __init__(
        self,
        api_key,
        mic_device,
        playback_device,
        instructions,
        tools,
        tool_dispatch_fn,
        emergency_event,
        voice_name="Rex",
    ):
        self._api_key = api_key
        self._mic_device = mic_device
        self._playback_device = playback_device
        self._instructions = instructions
        self._tools = tools
        self._tool_dispatch = tool_dispatch_fn
        self._emergency_event = emergency_event
        self._voice = voice_name

        self._ws = None
        self._loop = None
        self._thread = None
        self._running = False

        # Mic/playback state
        self._mic_muted = False         # software mute — skip sending audio
        self._arecord_proc = None
        self._aplay_proc = None
        self._aplay_stdin = None        # pipe to current aplay process

        # Stop word checker (optional, set via set_stop_word_checker)
        self._whisper_model = None
        self._whisper_fire_fn = None
        self._stop_word_audio_buf = bytearray()

        # Debug counters
        self._chunks_sent = 0
        self._events_received = 0

        # Response state
        self._response_pending = False   # True while waiting for response.done
        self._response_pending_since = 0  # monotonic time when set True
        self._consecutive_timeouts = 0   # count of consecutive response timeouts
        self._needs_reconnect = False    # signal mic loop to break for reconnect

    # ---- Public API ----

    def start(self):
        """Launch background thread with asyncio event loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_thread, daemon=True, name="xai-rt")
        self._thread.start()
        log.info("xAI Realtime voice started")

    def stop(self):
        """Shut down WebSocket, kill arecord/aplay procs."""
        self._running = False
        # Kill subprocesses
        for proc in (self._arecord_proc, self._aplay_proc):
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        log.info("xAI Realtime voice stopped")

    def set_stop_word_checker(self, whisper_model, fire_fn):
        """Enable local stop-word detection using a whisper model."""
        self._whisper_model = whisper_model
        self._whisper_fire_fn = fire_fn
        log.info("Stop word checker enabled (local whisper)")

    # ---- Thread entry ----

    def _run_thread(self):
        """Background thread: create event loop and run reconnect loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._reconnect_loop())
        except Exception as e:
            log.error(f"Event loop crashed: {e}")
        finally:
            self._loop.close()

    async def _reconnect_loop(self):
        """Connect, run, reconnect on failure."""
        while self._running:
            try:
                await self._session()
            except Exception as e:
                log.error(f"Session error: {e}")
            if self._running:
                log.info(f"Reconnecting in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

    # ---- WebSocket session ----

    async def _session(self):
        """Single WebSocket session: connect, configure, run loops."""
        import websockets

        headers = {"Authorization": f"Bearer {self._api_key}"}

        log.info(f"Connecting to {XAI_WS_URL}...")
        async with websockets.connect(
            XAI_WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message
        ) as ws:
            self._ws = ws
            log.info("Connected. Configuring session...")

            # Send session config — use manual turn detection (server_vad unreliable)
            session_config = {
                "type": "session.update",
                "session": {
                    "voice": self._voice,
                    "instructions": self._instructions,
                    "turn_detection": None,  # manual — we commit buffer via local VAD
                    "tools": self._tools,
                    "input_audio_transcription": {"model": "grok-2-audio"},
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": MIC_SAMPLE_RATE}},
                        "output": {"format": {"type": "audio/pcm", "rate": MIC_SAMPLE_RATE}},
                    },
                },
            }
            await ws.send(json.dumps(session_config))
            log.info("Session configured")

            # Run mic streaming and receive loop concurrently
            tasks = [
                asyncio.create_task(self._mic_stream_loop(ws)),
                asyncio.create_task(self._receive_loop(ws)),
            ]
            # Add stop word checker if configured
            if self._whisper_model:
                tasks.append(asyncio.create_task(self._stop_word_loop()))

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                log.error(f"Task error: {e}")
            finally:
                for t in tasks:
                    t.cancel()
                self._ws = None

    # ---- Mic streaming ----

    @staticmethod
    def _find_working_mic():
        """Find USB mic that produces actual audio (not silent).
        Unmutes each mic before testing — previous sessions may have left them muted."""
        import numpy as np
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            cards = []
            for line in result.stdout.splitlines():
                if "card" in line and "USB" in line:
                    card_num = line.split("card")[1].split(":")[0].strip()
                    label = "PnP" if "PnP" in line else ("Camera" if "Camera" in line else "USB")
                    cards.append((card_num, label))
            # Test each card: prefer Camera over PnP (camera mic is the primary one)
            cards.sort(key=lambda c: (0 if "Camera" in c[1] else 1, c[0]))
            for card_num, label in cards:
                dev = f"plughw:{card_num},0"
                try:
                    # Unmute mic capture (may have been left muted by echo prevention)
                    subprocess.run(
                        ["amixer", "-c", card_num, "cset", "numid=2", "on"],
                        capture_output=True, timeout=2,
                    )
                    p = subprocess.run(
                        ["arecord", "-D", dev, "-f", "S16_LE", "-r", "48000", "-c", "1",
                         "-d", "1", "-t", "raw", "/tmp/_xai_mic_test.raw"],
                        capture_output=True, timeout=3,
                    )
                    data = np.fromfile("/tmp/_xai_mic_test.raw", dtype=np.int16)
                    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if len(data) > 0 else 0
                    log.info(f"[mic-detect] {dev} ({label}): {len(data)} samples, RMS={rms:.1f}")
                    if rms > 0.5:  # not silent
                        return dev, card_num
                except Exception:
                    pass
        except Exception as e:
            log.error(f"[mic-detect] Error: {e}")
        return None, None

    async def _mic_stream_loop(self, ws):
        """Record from USB mic at 24kHz, detect speech via local VAD, send as conversation items.

        Key design: xAI's input_audio_buffer.commit doesn't reliably produce responses.
        Instead, we accumulate speech audio locally and send it as a conversation.item.create
        with input_audio content when speech ends. This guarantees the model processes it.
        """
        import collections
        import numpy as np
        loop = asyncio.get_event_loop()

        # Auto-detect working mic (unmutes first — previous session may have left it muted)
        mic_device = self._mic_device
        working, card_num = self._find_working_mic()
        if working:
            if working != mic_device:
                log.info(f"[mic] Overriding {mic_device} → {working} (produces actual audio)")
            mic_device = working
            self._mic_card = card_num
        else:
            log.warning(f"[mic] No working USB mic found, using {mic_device}")
            try:
                card = mic_device.split(":")[1].split(",")[0] if ":" in mic_device else "0"
                subprocess.run(["amixer", "-c", card, "cset", "numid=2", "on"],
                               capture_output=True, timeout=2)
                self._mic_card = card
            except Exception:
                pass

        arecord_cmd = [
            "arecord", "-D", mic_device, "-f", "S16_LE",
            "-r", str(MIC_SAMPLE_RATE), "-c", "1", "-t", "raw",
        ]
        log.info(f"Starting mic: {' '.join(arecord_cmd)}")
        self._arecord_proc = subprocess.Popen(
            arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        # Local VAD state
        in_speech = False
        speech_chunks = 0
        silence_chunks = 0
        total_loud = 0
        total_chunks_read = 0

        # Prefix buffer: keep last 3 chunks (300ms) before speech onset
        PREFIX_CHUNKS = 3
        prefix_buf = collections.deque(maxlen=PREFIX_CHUNKS)

        # Accumulate speech audio locally (list of raw PCM bytes)
        speech_audio = []

        try:
            while self._running:
                chunk = await loop.run_in_executor(
                    None, self._arecord_proc.stdout.read, MIC_BYTES_PER_CHUNK,
                )
                if not chunk:
                    log.warning("arecord EOF")
                    break

                total_chunks_read += 1
                samples = np.frombuffer(chunk, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

                # Accumulate for stop word detection
                if self._whisper_model:
                    self._stop_word_audio_buf.extend(chunk)

                threshold = VAD_SPEECH_THRESHOLD
                is_loud = rms > threshold

                # During playback: skip all mic input (speaker → mic echo causes false triggers)
                if self._mic_muted:
                    speech_chunks = 0
                    in_speech = False
                    continue

                # --- Not muted ---

                if not in_speech:
                    prefix_buf.append(chunk)

                    if is_loud:
                        speech_chunks += 1
                        if speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
                            in_speech = True
                            silence_chunks = 0
                            total_loud = speech_chunks
                            log.info(f"[vad] Speech started (rms={rms:.0f}, thresh={threshold})")
                            # Start accumulating: prefix + chunks so far
                            speech_audio = list(prefix_buf)
                            prefix_buf.clear()
                    else:
                        speech_chunks = 0
                else:
                    # In speech: accumulate locally
                    speech_audio.append(chunk)

                    if is_loud:
                        silence_chunks = 0
                        total_loud += 1
                    else:
                        silence_chunks += 1
                        if silence_chunks >= VAD_SILENCE_CHUNKS:
                            # Speech ended — send accumulated audio as conversation item
                            in_speech = False
                            speech_chunks = 0
                            silence_chunks = 0
                            if total_loud < VAD_MIN_TOTAL_SPEECH:
                                log.info(f"[vad] Too short ({total_loud} chunks) — ignoring")
                                total_loud = 0
                                speech_audio.clear()
                            elif self._response_pending and (time.monotonic() - self._response_pending_since) < 8:
                                log.info(f"[vad] Speech ended ({total_loud} chunks) but response pending — skipping")
                                total_loud = 0
                                speech_audio.clear()
                            else:
                                if self._response_pending:
                                    log.warning("[vad] Response pending timeout — clearing and proceeding")
                                audio_bytes = b"".join(speech_audio)
                                speech_audio.clear()
                                duration_ms = len(audio_bytes) * 1000 // (MIC_SAMPLE_RATE * 2)
                                log.info(f"[vad] Speech ended ({total_loud} chunks, {duration_ms}ms) — sending to model")
                                total_loud = 0
                                self._response_pending = True
                                self._response_pending_since = time.monotonic()
                                try:
                                    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                                    await ws.send(json.dumps({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "user",
                                            "content": [{"type": "input_audio", "audio": b64_audio}],
                                        },
                                    }))
                                    await ws.send(json.dumps({
                                        "type": "response.create",
                                        "response": {"modalities": ["text", "audio"]},
                                    }))
                                    self._chunks_sent += len(audio_bytes) // MIC_BYTES_PER_CHUNK
                                except Exception as e:
                                    log.error(f"[vad] send error: {e}")
                                    self._response_pending = False

                # Periodic status log (every 10 seconds)
                if total_chunks_read % 100 == 1:
                    log.info(f"[mic] read={total_chunks_read} sent={self._chunks_sent} rms={rms:.0f} "
                             f"vad={'SPEECH' if in_speech else 'silent'} events_rx={self._events_received}")
        finally:
            if self._arecord_proc and self._arecord_proc.poll() is None:
                self._arecord_proc.kill()

    # ---- Receive loop ----

    async def _receive_loop(self, ws):
        """Process all events from WebSocket."""
        loop = asyncio.get_event_loop()

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                event = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            self._events_received += 1
            if self._events_received <= 5 or self._events_received % 50 == 0:
                log.info(f"[rx] event #{self._events_received}: {etype}")

            # --- Buffer events (not used in direct-item mode, but log if they appear) ---
            if etype in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                         "input_audio_buffer.committed"):
                pass  # not used — we send audio as conversation items

            # --- User speech transcription ---
            elif etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    log.info(f"[user] {transcript}")

            # --- Audio output chunk ---
            elif etype == "response.output_audio.delta":
                delta_b64 = event.get("delta", "")
                if delta_b64:
                    pcm = base64.b64decode(delta_b64)
                    self._mic_muted = True
                    self._feed_aplay(pcm)

            # --- Audio output done ---
            elif etype == "response.output_audio.done":
                log.info("[audio] Response audio done")
                self._close_aplay()
                # Cooldown before unmuting mic
                await asyncio.sleep(PLAYBACK_COOLDOWN)
                self._mic_muted = False

            # --- Agent speech transcript ---
            elif etype == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    log.info(f"[jasper] {transcript}")

            # --- Function call ---
            elif etype == "response.function_call_arguments.done":
                fn_name = event.get("name", "")
                call_id = event.get("call_id", "")
                args_str = event.get("arguments", "{}")
                log.info(f"[tool] {fn_name}({args_str[:100]})")

                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}

                # Dispatch in thread pool to avoid blocking WebSocket
                try:
                    result = await loop.run_in_executor(
                        None, self._tool_dispatch, fn_name, args
                    )
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                    log.error(f"[tool] {fn_name} error: {e}")

                if not isinstance(result, str):
                    result = json.dumps(result)

                log.info(f"[tool] {fn_name} → {result[:200]}")

                # Send result back
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }))
                # Only request continuation for tools that return data the user should hear.
                # Fire-and-forget tools (commands, remember, set_speed) don't need continuation
                # — otherwise the model loops (speaks + gesture → continuation → speaks + gesture → ...).
                needs_continuation = fn_name in ("look_at_camera", "navigate_to", "search_for", "get_status")
                if needs_continuation:
                    self._response_pending = True
                    self._response_pending_since = time.monotonic()
                    await ws.send(json.dumps({"type": "response.create"}))
                else:
                    self._response_pending = False

            # --- Response done ---
            elif etype == "response.done":
                self._response_pending = False
                resp = event.get("response", {})
                output = resp.get("output", [])
                status = resp.get("status", "?")
                log.info(f"[response] done status={status} outputs={len(output)}")
                if output:
                    for item in output:
                        log.info(f"[response]   item: type={item.get('type')} role={item.get('role')} status={item.get('status')}")
                else:
                    # Debug: log full response when empty to diagnose
                    cancel = resp.get("status_details", {})
                    usage = resp.get("usage", {})
                    log.warning(f"[response] EMPTY — status_details={cancel} usage={usage} "
                                f"modalities={resp.get('modalities')} id={resp.get('id','?')[:20]}")

            # --- Session created ---
            elif etype == "session.created" or etype == "session.updated":
                session_data = event.get("session", {})
                log.info(f"[session] {etype} turn_detection={session_data.get('turn_detection')} "
                         f"voice={session_data.get('voice')} tools={len(session_data.get('tools', []))} "
                         f"audio={session_data.get('audio')}")

            # --- Conversation created ---
            elif etype == "conversation.created":
                log.info(f"[session] conversation created")

            # --- Ping (keepalive, ignore) ---
            elif etype == "ping":
                pass

            # --- Informational events (ignore silently) ---
            elif etype in ("response.created", "response.content_part.added",
                           "response.content_part.done", "response.output_item.added",
                           "response.output_item.done", "conversation.item.added",
                           "conversation.item.created",
                           "response.output_audio_transcript.delta",
                           "response.function_call_arguments.delta"):
                pass

            # --- Error ---
            elif etype == "error":
                err = event.get("error", {})
                log.error(f"[error] {err.get('message', err)}")

            # --- Unhandled ---
            else:
                log.info(f"[unhandled] {etype}")

    # ---- aplay management ----

    def _feed_aplay(self, pcm_data):
        """Pipe PCM data to aplay. Start aplay if not running."""
        if self._aplay_proc is None or self._aplay_proc.poll() is not None:
            # Start a new aplay process for this response
            aplay_cmd = [
                "aplay",
                "-D", self._playback_device,
                "-f", "S16_LE",
                "-r", str(MIC_SAMPLE_RATE),
                "-c", "1",
                "-t", "raw",
            ]
            self._aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._aplay_stdin = self._aplay_proc.stdin

        try:
            if self._aplay_stdin:
                self._aplay_stdin.write(pcm_data)
                self._aplay_stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _close_aplay(self):
        """Close aplay stdin to let it finish playing buffered audio."""
        try:
            if self._aplay_stdin:
                self._aplay_stdin.close()
        except Exception:
            pass
        self._aplay_stdin = None
        # Wait for aplay to finish (non-blocking wait with timeout)
        proc = self._aplay_proc
        if proc and proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._aplay_proc = None

    def _kill_aplay(self):
        """Kill aplay immediately (barge-in)."""
        try:
            if self._aplay_stdin:
                self._aplay_stdin.close()
        except Exception:
            pass
        self._aplay_stdin = None
        proc = self._aplay_proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._aplay_proc = None

    # ---- Stop word detection ----

    async def _stop_word_loop(self):
        """Periodically check accumulated mic audio for stop words."""
        loop = asyncio.get_event_loop()
        while self._running:
            await asyncio.sleep(STOP_WORD_CHECK_INTERVAL)

            buf = bytes(self._stop_word_audio_buf)
            self._stop_word_audio_buf.clear()

            if not buf or len(buf) < MIC_SAMPLE_RATE:  # need at least 0.5s
                continue

            try:
                word = await loop.run_in_executor(
                    None, self._check_stop_words, buf
                )
                if word:
                    log.warning(f"[stop-word] Detected: {word}")
                    if self._whisper_fire_fn:
                        self._whisper_fire_fn(word)
                    # Also kill playback
                    self._kill_aplay()
                    self._mic_muted = False
            except Exception as e:
                log.error(f"[stop-word] Error: {e}")

    def _check_stop_words(self, pcm_bytes):
        """Run local whisper on PCM bytes, check for stop words. Returns word or None."""
        import numpy as np

        if not self._whisper_model:
            return None

        # Convert 16-bit PCM to float32 at 16kHz (whisper needs 16kHz)
        audio_16 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample 24kHz → 16kHz (simple linear interpolation)
        if MIC_SAMPLE_RATE != 16000:
            n_out = int(len(audio_16) * 16000 / MIC_SAMPLE_RATE)
            indices = np.linspace(0, len(audio_16) - 1, n_out)
            audio_16 = np.interp(indices, np.arange(len(audio_16)), audio_16).astype(np.float32)

        try:
            segments, _ = self._whisper_model.transcribe(
                audio_16, language="en", beam_size=1, vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                m = STOP_WORD_PATTERN.search(text)
                if m:
                    return m.group(1).lower()
        except Exception:
            pass
        return None
