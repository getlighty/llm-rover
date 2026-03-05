"""TTS module: Groq Orpheus — text → WAV → aplay."""

import os, subprocess
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TTS_VOICE = "troy"
NAME = "Groq Orpheus"


def speak(text, speaker_dev, mic_card):
    """Text → Groq Orpheus TTS → aplay. Mutes mic during playback."""
    if not text:
        return
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
