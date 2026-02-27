#!/usr/bin/env python3
"""Romanian speech recognition using faster-whisper on Jetson Orin Nano."""

import sys
import wave
import tempfile
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- Config ---
MODEL_SIZE = "small"       # tiny, base, small, medium, large-v3
LANGUAGE = "ro"            # Romanian
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 5        # default recording length
DEVICE_INDEX = 0           # USB Camera mic

def load_model():
    print(f"Loading whisper '{MODEL_SIZE}' model (CPU int8)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model ready.")
    return model

def record_audio(seconds=RECORD_SECONDS):
    """Record from microphone, return numpy float32 array."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )
    print(f"Recording {seconds}s... speak now!")
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    print("Done recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def transcribe(model, audio):
    """Transcribe numpy audio array, return text."""
    segments, info = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text, info

def transcribe_file(model, path):
    """Transcribe a wav/mp3/etc file."""
    segments, info = model.transcribe(
        path,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text, info

def main():
    model = load_model()

    # If a file path is given, transcribe that file
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Transcribing file: {path}")
        text, info = transcribe_file(model, path)
        print(f"Language: {info.language} (prob: {info.language_probability:.2f})")
        print(f"Text: {text}")
        return

    # Otherwise, interactive mic loop
    print("\nInteractive mode - press Enter to record, 'q' to quit")
    print(f"Language: Romanian | Model: {MODEL_SIZE} | Device: mic #{DEVICE_INDEX}\n")

    while True:
        try:
            cmd = input(">>> Press Enter to record (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() == "q":
            break

        # Allow custom duration: e.g. "10" for 10 seconds
        try:
            secs = int(cmd) if cmd else RECORD_SECONDS
        except ValueError:
            secs = RECORD_SECONDS

        audio = record_audio(secs)
        print("Transcribing...")
        text, info = transcribe(model, audio)
        print(f"[{info.language} {info.language_probability:.0%}] {text}\n")

if __name__ == "__main__":
    main()
