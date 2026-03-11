"""Groq Orpheus text-to-speech."""

from __future__ import annotations

import subprocess

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import require_env


class GroqTextToSpeech:
    name = "groq"

    def __init__(self, voice: str = "troy"):
        self.voice = voice

    def speak(self, text: str, speaker_dev: str, mic_card: str | None):
        if not text:
            return
        api_key = require_env("GROQ_API_KEY")
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "canopylabs/orpheus-v1-english",
                "voice": self.voice,
                "input": text,
                "response_format": "wav",
            },
            timeout=15,
        )
        if not response.ok:
            raise ProviderError(response.text)
        # Check for Bluetooth audio sink — use paplay if available
        bt_sink = self._get_bt_sink()
        if bt_sink:
            try:
                proc = subprocess.Popen(
                    ["sudo", "-u", "jasper", "paplay", "--device", bt_sink,
                     "--format=s16le", "--channels=1", "--rate=24000"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=response.content, timeout=30)
            except Exception:
                pass
            return
        if mic_card is not None:
            subprocess.run(
                ["amixer", "-c", str(mic_card), "cset", "numid=2", "off"],
                capture_output=True,
            )
        try:
            proc = subprocess.Popen(
                ["aplay", "-D", speaker_dev, "-t", "wav"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=response.content, timeout=30)
        finally:
            if mic_card is not None:
                subprocess.run(
                    ["amixer", "-c", str(mic_card), "cset", "numid=2", "on"],
                    capture_output=True,
                )

    @staticmethod
    def _get_bt_sink() -> str | None:
        try:
            out = subprocess.check_output(
                ["sudo", "-u", "jasper", "pactl", "list", "sinks", "short"],
                text=True, timeout=3, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "bluez" in line.lower():
                    return line.split("\t")[1] if "\t" in line else line.split()[1]
        except Exception:
            pass
        return None

