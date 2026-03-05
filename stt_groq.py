"""STT module: Groq Whisper — 48kHz float32 audio → text."""

import os, io, wave
import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NAME = "Groq Whisper"


def transcribe(audio_48k):
    """48kHz float32 → Groq Whisper → text."""
    audio_16k = audio_48k[::3]
    pcm = (audio_16k * 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    wav_buf.seek(0)
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", wav_buf, "audio/wav")},
        data={"model": "whisper-large-v3-turbo", "language": "en"},
        timeout=15)
    r.raise_for_status()
    return r.json().get("text", "").strip()
