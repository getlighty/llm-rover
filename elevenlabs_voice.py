"""
ElevenLabs Conversational AI voice module for the Jasper rover.

Replaces xai_realtime.py with ElevenLabs' end-to-end voice pipeline:
  - STT + LLM + TTS handled by ElevenLabs cloud
  - Custom ALSA audio interface (no PyAudio dependency)
  - Client tools dispatch to rover hardware via xai_tool_dispatch closure

Usage:
    from elevenlabs_voice import ElevenLabsVoice
    voice = ElevenLabsVoice(api_key, agent_id, mic_device, playback_device,
                            tool_dispatch_fn, emergency_event)
    voice.start()
    ...
    voice.stop()
"""

import json
import re
import subprocess
import threading
import time

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    AudioInterface,
    ClientTools,
    Conversation,
)

STOP_WORD_PATTERN = re.compile(
    r"\b(stop|halt|freeze|shut\s*up|be\s*quiet|emergency)\b", re.IGNORECASE
)

# Tool names that return data the LLM should incorporate into its response
_DATA_TOOLS = {"look_at_camera", "get_status", "search_for", "navigate_to"}


class ALSAAudioInterface(AudioInterface):
    """Custom AudioInterface using ALSA arecord/aplay (no PyAudio needed)."""

    def __init__(self, mic_device="plughw:0,0", playback_device=None):
        self._mic_device = mic_device
        self._playback_device = playback_device or "plughw:1,0"
        self._input_callback = None
        self._arecord_proc = None
        self._aplay_proc = None
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._aplay_lock = threading.Lock()
        self._mic_muted = False  # external mute flag

    def start(self, input_callback):
        """Start mic capture. Called by Conversation before session begins."""
        self._input_callback = input_callback
        self._stop_event.clear()

        # Unmute mic
        if self._mic_device:
            card = self._mic_device.replace("plughw:", "").split(",")[0]
            try:
                subprocess.run(
                    ["amixer", "-c", card, "cset", "numid=2", "on"],
                    capture_output=True, timeout=3,
                )
            except Exception:
                pass

        # Start arecord: 16kHz 16-bit mono raw PCM
        try:
            self._arecord_proc = subprocess.Popen(
                [
                    "arecord", "-D", self._mic_device,
                    "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[11labs-audio] arecord failed: {e}")
            return

        # Reader thread: sends audio chunks to ElevenLabs
        self._reader_thread = threading.Thread(
            target=self._read_mic, daemon=True, name="11labs-mic"
        )
        self._reader_thread.start()
        print("[11labs-audio] Mic capture started")

    def _read_mic(self):
        """Read 250ms chunks from arecord and forward to ElevenLabs."""
        chunk_bytes = 4000 * 2  # 4000 samples * 2 bytes (S16_LE) = 250ms at 16kHz
        while not self._stop_event.is_set():
            try:
                data = self._arecord_proc.stdout.read(chunk_bytes)
                if not data:
                    break
                if not self._mic_muted and self._input_callback:
                    self._input_callback(data)
            except Exception:
                break

    def stop(self):
        """Stop mic capture and playback. Called by Conversation on session end."""
        self._stop_event.set()
        if self._arecord_proc:
            try:
                self._arecord_proc.kill()
                self._arecord_proc.wait(timeout=2)
            except Exception:
                pass
            self._arecord_proc = None
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None
        self._kill_aplay()
        print("[11labs-audio] Audio stopped")

    def output(self, audio: bytes):
        """Receive TTS audio from ElevenLabs and play via aplay."""
        with self._aplay_lock:
            # Start aplay if not running
            if self._aplay_proc is None or self._aplay_proc.poll() is not None:
                try:
                    self._aplay_proc = subprocess.Popen(
                        [
                            "aplay", "-D", self._playback_device,
                            "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw",
                        ],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    print(f"[11labs-audio] aplay failed: {e}")
                    return
            try:
                self._aplay_proc.stdin.write(audio)
                self._aplay_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._aplay_proc = None

    def interrupt(self):
        """Kill playback immediately (barge-in)."""
        self._kill_aplay()

    def _kill_aplay(self):
        """Terminate current aplay process."""
        with self._aplay_lock:
            if self._aplay_proc:
                try:
                    self._aplay_proc.kill()
                    self._aplay_proc.wait(timeout=1)
                except Exception:
                    pass
                self._aplay_proc = None


class ElevenLabsVoice:
    """
    Wrapper around ElevenLabs Conversational AI SDK.

    Manages a Conversation session with custom ALSA audio and client tools
    that dispatch to the rover's xai_tool_dispatch closure.

    Auto-reconnects on session drop with exponential backoff.
    """

    RECONNECT_BASE = 3      # initial retry delay (seconds)
    RECONNECT_MAX = 120     # max retry delay
    QUOTA_BACKOFF = 300     # 5 min wait on quota errors

    def __init__(
        self,
        api_key,
        agent_id,
        mic_device,
        playback_device,
        tool_dispatch_fn,
        emergency_event,
    ):
        self._client = ElevenLabs(api_key=api_key)
        self._agent_id = agent_id
        self._mic_device = mic_device or "plughw:0,0"
        self._playback_device = playback_device
        self._tool_dispatch = tool_dispatch_fn
        self._emergency_event = emergency_event
        self._conversation = None
        self._audio = None
        self._stopped = False
        self._session_thread = None
        self.__mic_muted = False

    @property
    def _mic_muted(self):
        return self.__mic_muted

    @_mic_muted.setter
    def _mic_muted(self, val):
        self.__mic_muted = val
        if self._audio:
            self._audio._mic_muted = val

    def start(self):
        """Start the session loop in a background thread."""
        self._stopped = False
        self._session_thread = threading.Thread(
            target=self._session_loop, daemon=True, name="11labs-session"
        )
        self._session_thread.start()

    def stop(self):
        """Permanently stop — no more reconnects."""
        self._stopped = True
        self._teardown_session()
        if self._session_thread:
            self._session_thread.join(timeout=5)
            self._session_thread = None
        print("[11labs] Stopped")

    def _session_loop(self):
        """Reconnect loop: start session, wait for it to end, retry."""
        backoff = self.RECONNECT_BASE
        while not self._stopped:
            try:
                self._start_session()
                backoff = self.RECONNECT_BASE  # reset on successful start
                # Block until the session ends (SDK runs in its own thread)
                if self._conversation:
                    self._conversation.wait_for_session_end()
                if self._stopped:
                    break
                print("[11labs] Session ended, reconnecting...")
            except Exception as e:
                err = str(e)
                if self._stopped:
                    break
                if "quota" in err.lower():
                    backoff = self.QUOTA_BACKOFF
                    print(f"[11labs] Quota exceeded — waiting {backoff}s")
                else:
                    print(f"[11labs] Session error: {e} — retrying in {backoff}s")
            finally:
                self._teardown_session()

            if self._stopped:
                break
            # Wait with backoff, but check _stopped every second
            for _ in range(int(backoff)):
                if self._stopped:
                    return
                time.sleep(1)
            backoff = min(backoff * 2, self.RECONNECT_MAX)

    def _start_session(self):
        """Create fresh audio interface, client tools, and conversation."""
        self._audio = ALSAAudioInterface(
            mic_device=self._mic_device,
            playback_device=self._playback_device,
        )
        self._audio._mic_muted = self.__mic_muted

        client_tools = ClientTools()
        for tool_name in [
            "send_rover_commands",
            "look_at_camera",
            "navigate_to",
            "search_for",
            "remember",
            "get_status",
            "set_speed",
        ]:
            client_tools.register(tool_name, self._make_tool_handler(tool_name))

        self._conversation = Conversation(
            client=self._client,
            agent_id=self._agent_id,
            requires_auth=True,
            audio_interface=self._audio,
            client_tools=client_tools,
            callback_agent_response=self._on_agent_response,
            callback_user_transcript=self._on_user_transcript,
        )
        self._conversation.start_session()
        print("[11labs] Conversation session started")

    def _teardown_session(self):
        """Clean up current session without affecting the reconnect loop."""
        conv = self._conversation
        self._conversation = None
        if conv:
            try:
                conv.end_session()
            except Exception:
                pass

    def send_context(self, text):
        """Send contextual update to the agent (non-interrupting)."""
        if self._conversation:
            try:
                self._conversation.send_contextual_update(text)
            except Exception:
                pass

    def _make_tool_handler(self, tool_name):
        """Create a handler closure for a specific client tool."""
        def handler(parameters):
            try:
                result = self._tool_dispatch(tool_name, parameters)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        return handler

    def _on_user_transcript(self, transcript):
        """Called when ElevenLabs STT produces a user transcript."""
        print(f"[11labs] User: {transcript}")
        if STOP_WORD_PATTERN.search(transcript):
            print("[11labs] STOP WORD detected — firing emergency")
            self._emergency_event.set()

    def _on_agent_response(self, response):
        """Called when the agent produces a text response."""
        print(f"[11labs] Agent: {response}")
