#!/usr/bin/env python3
"""Quick Romanian whisper test - record from mic and transcribe."""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL = "small"
LANG = "ro"
MIC_DEVICE = 0       # USB Camera mic
MIC_RATE = 48000
RECORD_SEC = 5

print(f"Loading whisper '{MODEL}' (int8)...")
model = WhisperModel(MODEL, device="cpu", compute_type="int8")
print("Ready.\n")

while True:
    input(f"Press Enter to record {RECORD_SEC}s (Ctrl+C to quit)...")
    print("Speak now!")
    audio = sd.rec(int(MIC_RATE * RECORD_SEC), samplerate=MIC_RATE,
                   channels=1, dtype="float32", device=MIC_DEVICE)
    sd.wait()
    audio = audio.flatten()[::3]  # 48kHz -> 16kHz

    print("Transcribing...", end=" ", flush=True)
    segs, info = model.transcribe(audio, language=LANG, beam_size=3,
                                  vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    print(f"[{info.language} {info.language_probability:.0%}] {text}\n")
