"""Provider: Groq STT/TTS + xAI Grok 4.1 Fast Vision LLM"""

import os, json, io, wave, base64, subprocess
import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
LLM_MODEL = "grok-4-1-fast-reasoning"
LLM_URL = "https://api.x.ai/v1/chat/completions"
TTS_VOICE = "troy"
NAME = "xAI Grok 4.1 Fast Reasoning + Groq STT/TTS"


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
    """Text + JPEG → xAI Grok 4.1 Fast → raw reply string."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": text},
    ]
    messages = [{"role": "system", "content": system}]
    messages += history[:-1]
    messages.append({"role": "user", "content": user_content})
    r = requests.post(LLM_URL,
        headers={"Authorization": f"Bearer {XAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "messages": messages,
              "temperature": 0.3, "max_tokens": 1000},
        timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


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
