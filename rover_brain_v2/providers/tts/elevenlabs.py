"""ElevenLabs streaming text-to-speech."""

from __future__ import annotations

import subprocess

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import require_env


class ElevenLabsTextToSpeech:
    name = "elevenlabs"

    def __init__(self, voice_id: str = "iP95p4xoKVk53GoZ742B",
                 model_id: str = "eleven_turbo_v2"):
        self.voice_id = voice_id
        self.model_id = model_id

    def speak(self, text: str, speaker_dev: str, mic_card: str | None):
        if not text:
            return
        api_key = require_env("ELEVENLABS_API_KEY")
        if mic_card is not None:
            subprocess.run(
                ["amixer", "-c", str(mic_card), "cset", "numid=2", "off"],
                capture_output=True,
            )
        try:
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.model_id,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
                stream=True,
                timeout=15,
            )
            if not response.ok:
                raise ProviderError(response.text)
            proc = subprocess.Popen(
                [
                    "gst-launch-1.0", "-q", "fdsrc", "fd=0", "!",
                    "decodebin", "!", "audioconvert", "!", "audioresample", "!",
                    "alsasink", f"device={speaker_dev}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for chunk in response.iter_content(chunk_size=4096):
                if chunk and proc.stdin is not None:
                    proc.stdin.write(chunk)
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=30)
        finally:
            if mic_card is not None:
                subprocess.run(
                    ["amixer", "-c", str(mic_card), "cset", "numid=2", "on"],
                    capture_output=True,
                )

