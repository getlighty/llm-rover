"""Shared helpers for provider clients."""

from __future__ import annotations

import base64
import io
import os
import wave

import numpy as np


def require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def b64_image(image_bytes: bytes | None) -> str | None:
    if not image_bytes:
        return None
    return base64.b64encode(image_bytes).decode("ascii")


def audio_to_wav_bytes(audio_48k) -> io.BytesIO:
    audio_16k = audio_48k[::3]
    pcm = (audio_16k * 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    wav_buf.seek(0)
    return wav_buf

