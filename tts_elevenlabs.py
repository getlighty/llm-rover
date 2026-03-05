"""TTS module: ElevenLabs streaming — text → MP3 stream → GStreamer → ALSA."""

import os, subprocess
import requests

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = "iP95p4xoKVk53GoZ742B"  # Chris
TTS_MODEL = "eleven_turbo_v2"
NAME = "ElevenLabs"


def speak(text, speaker_dev, mic_card):
    """Text → ElevenLabs streaming TTS (MP3) → GStreamer → ALSA. Mutes mic during playback."""
    if not text:
        return
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
