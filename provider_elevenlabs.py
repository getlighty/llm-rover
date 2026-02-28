"""Provider: ElevenLabs TTS + Groq STT/LLM — streaming TTS for low latency."""

import os, json, io, wave, base64, subprocess
import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
VOICE_ID = "iP95p4xoKVk53GoZ742B"  # Chris
TTS_MODEL = "eleven_turbo_v2"
NAME = "ElevenLabs (Groq STT + Maverick LLM + ElevenLabs TTS)"


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
    """Text + JPEG → Groq Maverick → raw reply string."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": text},
    ]
    messages = [{"role": "system", "content": system}]
    messages += history[:-1]
    messages.append({"role": "user", "content": user_content})
    r = requests.post(LLM_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "messages": messages,
              "temperature": 0.3, "max_completion_tokens": 1000},
        timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def speak(text, speaker_dev, mic_card):
    """Text → ElevenLabs streaming TTS (MP3) → GStreamer → ALSA. Mutes mic during playback."""
    if not text:
        return
    print(f"[speak] {text}")
    subprocess.run(["amixer", "-c", mic_card, "cset", "numid=2", "off"],
                   capture_output=True)
    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
            headers={"xi-api-key": ELEVENLABS_API_KEY,
                     "Content-Type": "application/json"},
            json={"text": text, "model_id": TTS_MODEL,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            stream=True, timeout=15)
        r.raise_for_status()
        # Stream MP3 through GStreamer for decoding + ALSA playback
        proc = subprocess.Popen(
            ["gst-launch-1.0", "-q", "fdsrc", "fd=0", "!",
             "decodebin", "!", "audioconvert", "!", "audioresample", "!",
             "alsasink", f"device={speaker_dev}"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                proc.stdin.write(chunk)
        proc.stdin.close()
        proc.wait(timeout=30)
    except Exception as e:
        print(f"[tts] Error: {e}")
    finally:
        subprocess.run(["amixer", "-c", mic_card, "cset", "numid=2", "on"],
                       capture_output=True)
