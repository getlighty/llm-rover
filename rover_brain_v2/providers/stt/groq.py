"""Groq Whisper speech-to-text client."""

from __future__ import annotations

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import audio_to_wav_bytes, require_env


class GroqSpeechToText:
    name = "groq"

    def transcribe(self, audio_48k):
        api_key = require_env("GROQ_API_KEY")
        wav_buf = audio_to_wav_bytes(audio_48k)
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", wav_buf, "audio/wav")},
            data={"model": "whisper-large-v3-turbo", "language": "en"},
            timeout=15,
        )
        if not response.ok:
            raise ProviderError(response.text)
        return response.json().get("text", "").strip()

