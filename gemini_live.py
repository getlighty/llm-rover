#!/usr/bin/env python3
"""
Gemini Live API — persistent WebSocket session for Jasper rover.

Native audio mode: model produces speech directly (PCM 24kHz).
Tool calls dispatched via xai_tool_dispatch closure from rover_brain.
Audio output played via aplay. Text transcripts logged and queued.

Usage:
    from gemini_live import GeminiLiveClient
    client = GeminiLiveClient(api_key, model, tool_dispatch_fn, result_queue,
                              playback_device="plughw:1,0")
    client.start()
    client.send_text("hello")
    ...
    client.stop()
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import threading
import time

log = logging.getLogger("gemini-live")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[gemini-live] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECONNECT_BASE = 3      # initial retry delay (seconds)
RECONNECT_MAX = 120     # max retry delay (seconds)
PLAYBACK_COOLDOWN = 0.3  # seconds after aplay finishes before unmuting mic

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
# System instructions for Gemini Live session
# ---------------------------------------------------------------------------
GEMINI_LIVE_SYSTEM = """\
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
- Wheels: {"T":1, "L":<left_speed>, "R":<right_speed>} — positive=forward, negative=backward
- Gimbal absolute: {"T":133, "X":<pan>, "Y":<tilt>, "SPD":<speed>, "ACC":<accel>}
- Lights: {"T":132, "IO4":<base 0-255>, "IO5":<head 0-255>}
- OLED: {"T":3, "lineNum":<0-3>, "Text":"<msg>"}
- Emergency stop: {"T":0}
- Feedback: {"T":130}

## Physical expressions (use these!)
- Yes/agree: Nod — {"T":133,"X":0,"Y":20,"SPD":400,"ACC":20} then {"T":133,"X":0,"Y":-5,"SPD":400,"ACC":20} then {"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}
- No/disagree: Shake — {"T":133,"X":-40,"Y":0,"SPD":500,"ACC":20} then {"T":133,"X":40,"Y":0,"SPD":500,"ACC":20} then {"T":133,"X":0,"Y":0,"SPD":200,"ACC":10}
- Thinking: Tilt — {"T":133,"X":20,"Y":10,"SPD":200,"ACC":10}
- Greeting: Look up slightly, small nod

## Vision
You CANNOT see directly. Use look_at_camera to see via the camera. Always use it when you need visual info about your surroundings.

## Safety
- When user says stop/halt/freeze, IMMEDIATELY call send_rover_commands with {"T":1,"L":0,"R":0}.
- Max wheel speed 0.20 m/s unless user explicitly says faster.
- Look around before backing up.

## Ovi's preferences
- Move slowly (0.15-0.20 m/s max)
- Dim lights when ambient light is sufficient
- Don't say his name constantly
- Upgrades need clear purpose
"""

# ---------------------------------------------------------------------------
# Tool definitions in Gemini function calling format
# ---------------------------------------------------------------------------
GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
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
                            "description": "What to look for or describe.",
                        },
                    },
                },
            },
            {
                "name": "navigate_to",
                "description": "Autonomously navigate toward a named object or location. The rover will search, find, and drive to the target using camera vision. Takes 10-60 seconds. Returns success/failure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Object or location to navigate to.",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "search_for",
                "description": "Systematically sweep the gimbal to search for a named object. Checks all angles. Returns whether the object was found and its location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Object to search for.",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
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
                "name": "get_status",
                "description": "Get rover status: battery voltage, current pose, tracker state, spatial map summary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
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
    }
]


# ---------------------------------------------------------------------------
# GeminiLiveClient
# ---------------------------------------------------------------------------
class GeminiLiveClient:
    """Persistent WebSocket session to Gemini Live API with native audio output."""

    def __init__(self, api_key, model, tool_dispatch_fn, result_queue,
                 playback_device=None, voice="Puck", system_instruction=None,
                 mic_mute_fn=None, mic_unmute_fn=None):
        """
        Args:
            api_key: Gemini API key
            model: model name (e.g. "gemini-2.5-flash-native-audio-preview-12-2025")
            tool_dispatch_fn: callable(fn_name, args) -> str (JSON result)
            result_queue: queue.Queue — receives text transcripts of model speech
            playback_device: ALSA device for audio output (e.g. "plughw:1,0")
            voice: Gemini voice name (Puck, Kore, Enceladus, etc.)
            system_instruction: override system prompt
            mic_mute_fn: callable() to mute mic during playback
            mic_unmute_fn: callable() to unmute mic after playback
        """
        self._api_key = api_key
        self._model = model
        self._tool_dispatch = tool_dispatch_fn
        self._result_queue = result_queue
        self._playback_device = playback_device or "plughw:1,0"
        self._voice = voice
        self._system_instruction = system_instruction
        self._mic_mute_fn = mic_mute_fn
        self._mic_unmute_fn = mic_unmute_fn

        self._ws = None
        self._loop = None
        self._thread = None
        self._running = False
        self._connected = False
        self._connected_since = 0

        # Audio playback
        self._aplay_proc = None
        self._aplay_stdin = None

        # Outbound text queue
        self._send_queue_sync = None  # threading queue for cross-thread send

        # Stats
        self._messages_sent = 0
        self._messages_received = 0
        self._tool_calls_handled = 0

    # ---- Public API ----

    def start(self):
        """Launch background thread with asyncio event loop."""
        if self._running:
            return
        self._running = True
        self._send_queue_sync = __import__("queue").Queue()
        self._thread = threading.Thread(target=self._run_thread, daemon=True,
                                        name="gemini-live")
        self._thread.start()
        log.info("Gemini Live client started")

    def stop(self):
        """Shut down WebSocket and background thread."""
        self._running = False
        self._kill_aplay()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("Gemini Live client stopped")

    def send_text(self, text):
        """Thread-safe: queue text to send to Gemini Live session."""
        if not self._running:
            log.warning("send_text called but client not running")
            return
        if self._send_queue_sync:
            self._send_queue_sync.put(text)

    @property
    def is_connected(self):
        return self._connected

    @property
    def seconds_disconnected(self):
        if self._connected:
            return 0
        if self._connected_since == 0:
            return 999  # never connected
        return time.monotonic() - self._connected_since

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
        """Connect, run, reconnect on failure with exponential backoff."""
        backoff = RECONNECT_BASE
        while self._running:
            try:
                await self._session()
                backoff = RECONNECT_BASE  # reset on clean session
            except Exception as e:
                log.error(f"Session error: {e}")
            finally:
                self._connected = False
                self._kill_aplay()

            if not self._running:
                break
            log.info(f"Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX)

    # ---- Build setup message ----

    def _build_setup(self):
        """Build the BidiGenerateContent setup message."""
        memory = _load_memory_tail(20)
        system_text = self._system_instruction or GEMINI_LIVE_SYSTEM
        if memory:
            system_text += f"\n\n## Recent memory\n{memory}"

        return {
            "setup": {
                "model": f"models/{self._model}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self._voice
                            }
                        }
                    },
                },
                "system_instruction": {
                    "parts": [{"text": system_text}]
                },
                "tools": GEMINI_TOOLS,
                "output_audio_transcription": {},
            }
        }

    # ---- WebSocket session ----

    async def _session(self):
        """Single WebSocket session: connect, configure, run loops."""
        import websockets

        ws_url = (
            f"wss://generativelanguage.googleapis.com/ws/"
            f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
            f"?key={self._api_key}"
        )

        log.info(f"Connecting to Gemini Live ({self._model})...")
        async with websockets.connect(
            ws_url,
            ping_interval=25,
            ping_timeout=15,
            max_size=10 * 1024 * 1024,
        ) as ws:
            self._ws = ws

            # Send setup message
            setup = self._build_setup()
            await ws.send(json.dumps(setup))
            log.info("Setup sent, waiting for setupComplete...")

            # Wait for setupComplete
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            msg = json.loads(raw)
            if "setupComplete" in msg:
                log.info("Session established (setupComplete)")
                self._connected = True
                self._connected_since = time.monotonic()
            else:
                log.warning(f"Expected setupComplete, got: {list(msg.keys())}")
                return

            # Create async send queue
            send_queue = asyncio.Queue()

            # Run tasks concurrently
            tasks = [
                asyncio.create_task(self._receive_loop(ws)),
                asyncio.create_task(self._send_loop(ws, send_queue)),
                asyncio.create_task(self._bridge_sync_queue(send_queue)),
            ]
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                log.error(f"Task error: {e}")
            finally:
                for t in tasks:
                    t.cancel()
                self._ws = None
                self._connected = False

    async def _bridge_sync_queue(self, async_queue):
        """Bridge thread-safe sync queue -> asyncio queue."""
        import queue as queue_mod
        while self._running:
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._send_queue_sync.get(timeout=0.05)
                )
                await async_queue.put(text)
            except queue_mod.Empty:
                continue

    # ---- Send loop ----

    async def _send_loop(self, ws, send_queue):
        """Send queued text messages to the WebSocket."""
        while self._running:
            text = await send_queue.get()
            if not text:
                continue

            msg = {
                "client_content": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": text}]
                        }
                    ],
                    "turn_complete": True,
                }
            }
            try:
                await ws.send(json.dumps(msg))
                self._messages_sent += 1
                log.info(f"[tx] Sent text ({len(text)} chars): {text[:80]}")
            except Exception as e:
                log.error(f"[tx] Send error: {e}")

    # ---- Audio playback ----

    def _feed_aplay(self, pcm_data):
        """Pipe PCM data to aplay. Start aplay if not running."""
        if self._aplay_proc is None or self._aplay_proc.poll() is not None:
            self._aplay_proc = subprocess.Popen(
                ["aplay", "-D", self._playback_device,
                 "-f", "S16_LE", "-r", "24000", "-c", "1", "-t", "raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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

    # ---- Receive loop ----

    async def _receive_loop(self, ws):
        """Process all messages from the WebSocket."""
        loop = asyncio.get_event_loop()

        # Accumulate transcript parts until turnComplete
        transcript_parts = []
        audio_playing = False

        async for raw_msg in ws:
            if not self._running:
                break

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            self._messages_received += 1

            # --- setupComplete (shouldn't appear here but handle gracefully) ---
            if "setupComplete" in msg:
                continue

            # --- serverContent: audio, transcription, turn completion ---
            if "serverContent" in msg:
                sc = msg["serverContent"]
                turn_complete = sc.get("turnComplete", False)

                # Audio output transcription (text of what the model said)
                ot = sc.get("outputTranscription")
                if ot and "text" in ot:
                    transcript_parts.append(ot["text"])

                # Model turn with audio data or thinking text
                mt = sc.get("modelTurn")
                if mt:
                    for part in mt.get("parts", []):
                        # Thinking text (thought=true) — log but don't queue
                        if part.get("thought") and "text" in part:
                            log.info(f"[think] {part['text'][:150]}")
                            continue

                        # Audio chunk — play it
                        inline = part.get("inlineData")
                        if inline and inline.get("mimeType", "").startswith("audio/"):
                            if not audio_playing:
                                audio_playing = True
                                if self._mic_mute_fn:
                                    try:
                                        self._mic_mute_fn()
                                    except Exception:
                                        pass
                            pcm = base64.b64decode(inline["data"])
                            self._feed_aplay(pcm)

                if turn_complete:
                    # Finish audio playback
                    if audio_playing:
                        self._close_aplay()
                        await asyncio.sleep(PLAYBACK_COOLDOWN)
                        if self._mic_unmute_fn:
                            try:
                                self._mic_unmute_fn()
                            except Exception:
                                pass
                        audio_playing = False

                    # Queue transcript for main loop
                    if transcript_parts:
                        full_transcript = "".join(transcript_parts).strip()
                        transcript_parts.clear()
                        if full_transcript:
                            log.info(f"[rx] Transcript: {full_transcript[:200]}")
                            # Don't put raw transcript in llm_result_queue —
                            # the audio already played. Just log it.
                    else:
                        transcript_parts.clear()
                continue

            # --- toolCall: function calling ---
            if "toolCall" in msg:
                tool_call = msg["toolCall"]
                fn_calls = tool_call.get("functionCalls", [])

                for fc in fn_calls:
                    fn_name = fc.get("name", "")
                    fn_args = fc.get("args", {})
                    fn_id = fc.get("id", "")
                    log.info(f"[tool] {fn_name}({json.dumps(fn_args)[:100]})")

                    # Dispatch in thread pool
                    try:
                        result = await loop.run_in_executor(
                            None, self._tool_dispatch, fn_name, fn_args
                        )
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                        log.error(f"[tool] {fn_name} error: {e}")

                    if not isinstance(result, str):
                        result = json.dumps(result)

                    log.info(f"[tool] {fn_name} -> {result[:200]}")
                    self._tool_calls_handled += 1

                    # Send tool response back
                    tool_response = {
                        "tool_response": {
                            "function_responses": [
                                {
                                    "name": fn_name,
                                    "id": fn_id,
                                    "response": {"result": result},
                                }
                            ]
                        }
                    }
                    try:
                        await ws.send(json.dumps(tool_response))
                    except Exception as e:
                        log.error(f"[tool] Response send error: {e}")

                continue

            # --- toolCallCancellation ---
            if "toolCallCancellation" in msg:
                cancelled = msg["toolCallCancellation"].get("ids", [])
                log.warning(f"[tool] Cancelled: {cancelled}")
                continue

            # --- usageMetadata (ignore) ---
            if "usageMetadata" in msg:
                continue

            # --- Unknown ---
            log.info(f"[rx] Unknown keys: {list(msg.keys())}")

    # ---- Status ----

    def status(self):
        return {
            "connected": self._connected,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "tool_calls": self._tool_calls_handled,
            "model": self._model,
        }
