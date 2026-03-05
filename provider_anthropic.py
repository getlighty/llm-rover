"""Provider: Groq STT/TTS + Anthropic Claude Haiku 4.5 Vision LLM"""

import os, json, io, wave, base64, subprocess
import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_URL = "https://api.anthropic.com/v1/messages"
TTS_VOICE = "troy"
NAME = "Anthropic Claude Haiku 4.5 + Groq STT/TTS"


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


def call_llm(text, jpeg_bytes, system, history):
    """Text + JPEG → Anthropic Claude Haiku 4.5 → raw reply string."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    user_content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
        {"type": "text", "text": text},
    ]
    # Convert history: Anthropic doesn't use "system" in messages array
    messages = []
    for msg in history[:-1]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_content})
    r = requests.post(LLM_URL,
        headers={"x-api-key": ANTHROPIC_API_KEY,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": LLM_MODEL, "system": system, "messages": messages,
              "temperature": 0.3, "max_tokens": 1000},
        timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def speak(text, speaker_dev, mic_card):
    """Text → Groq Orpheus TTS → aplay. Mutes mic during playback."""
    if not text:
        return
    print(f"[speak] {text}")
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "canopylabs/orpheus-v1-english",
              "voice": TTS_VOICE, "input": text, "response_format": "wav"},
        timeout=15)
    r.raise_for_status()
    subprocess.run(["amixer", "-c", mic_card, "cset", "numid=2", "off"],
                   capture_output=True)
    try:
        proc = subprocess.Popen(
            ["aplay", "-D", speaker_dev, "-t", "wav"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.communicate(input=r.content, timeout=30)
    finally:
        subprocess.run(["amixer", "-c", mic_card, "cset", "numid=2", "on"],
                       capture_output=True)
