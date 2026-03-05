"""Provider: Groq STT/TTS + Ollama Vision LLM (qwen3.5:9b)"""

import os, json, io, wave, base64, subprocess
import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "qwen3.5:9b"
LLM_URL = "http://localhost:11434/api/chat"
TTS_VOICE = "troy"
NAME = "Ollama qwen3.5:9b + Groq STT/TTS"


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
    """Text + JPEG → Ollama qwen3.5:9b → raw reply string."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    # Build messages: system + history + user with image
    messages = [{"role": "system", "content": system}]
    for msg in history[:-1]:
        messages.append(msg)
    # Ollama native API: images go as base64 list on the message
    messages.append({
        "role": "user",
        "content": text,
        "images": [b64],
    })
    # Disable thinking mode for qwen3 models (dumps CoT as content otherwise)
    think = not LLM_MODEL.startswith("qwen3")
    r = requests.post(LLM_URL,
        json={"model": LLM_MODEL, "messages": messages,
              "stream": False, "think": think,
              "options": {"temperature": 0.3, "num_predict": 500}},
        timeout=60)
    r.raise_for_status()
    msg = r.json()["message"]
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")
    if thinking:
        print(f"[ollama-thinking] {thinking[:300]}")
    if not content.strip() and thinking:
        # Model put everything in thinking, nothing in content — use thinking
        content = thinking
    print(f"[ollama-raw] {content[:300]}")
    return content


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
