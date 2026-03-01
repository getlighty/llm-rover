"""audio.py — Mic/speaker discovery + listen/speak.

Extracted from rover_brain_llm.py. No circular imports.
"""

import subprocess
import threading
import time

import numpy as np


# ── TTS state (shared with rover_brain_llm) ──────────────────────────

tts_playing = threading.Event()   # True while TTS is playing (suppress mic)
tts_done_at = 0.0                 # timestamp when TTS last finished
TTS_COOLDOWN = 1.5                # seconds to ignore mic after TTS ends


def find_mic():
    """Discover USB mic. Returns (device_str, card_num)."""
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    mic_card = None
    fallback = None
    for line in result.stdout.splitlines():
        if "card" in line and "USB" in line:
            card_num = line.split("card")[1].split(":")[0].strip()
            if "Camera" in line:
                mic_card = card_num
            elif fallback is None:
                fallback = card_num
    card = mic_card or fallback
    if card is None:
        raise RuntimeError("No USB mic found")
    dev = f"plughw:{card},0"
    subprocess.run(["amixer", "-c", card, "cset", "numid=2", "on"],
                   capture_output=True)
    return dev, card


def find_speaker():
    """Discover USB speaker. Returns device string."""
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True)
        for line in out.splitlines():
            if "USB" in line and "card" in line:
                card = line.split("card ")[1].split(":")[0]
                dev = line.split("device ")[1].split(":")[0]
                return f"plughw:{card},{dev}"
    except Exception:
        pass
    return "plughw:1,0"


def listen(mic_device, abort_event=None):
    """Record speech from mic. Returns numpy array of audio or None."""
    global tts_done_at

    rate = 48000
    chunk_sec = 0.5
    chunk_samples = int(rate * chunk_sec)
    silence_thresh = 0.03
    min_speech = 2
    max_speech = 240
    silence_after_long = 4   # 2s silence for longer utterances
    silence_after_short = 2  # 1s silence for short bursts (e.g. "stop")
    short_threshold = 4      # <= 4 speech chunks (2s) = short utterance

    proc = subprocess.Popen(
        ["arecord", "-D", mic_device, "-f", "S16_LE", "-r", str(rate),
         "-c", "1", "-t", "raw", "--buffer-size", str(chunk_samples)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    speech_chunks = []
    speech_started = False
    silent_count = 0

    try:
        while True:
            if abort_event and abort_event.is_set():
                return None
            raw = proc.stdout.read(chunk_samples * 2)
            if len(raw) < chunk_samples * 2:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # If TTS is playing or just finished, discard audio and reset state
            if tts_playing.is_set() or (time.time() - tts_done_at) < TTS_COOLDOWN:
                speech_chunks.clear()
                speech_started = False
                silent_count = 0
                continue

            if rms > silence_thresh:
                if not speech_started:
                    speech_started = True
                speech_chunks.append(chunk)
                silent_count = 0
            elif speech_started:
                speech_chunks.append(chunk)
                silent_count += 1
                needed = silence_after_short if len(speech_chunks) <= short_threshold else silence_after_long
                if silent_count >= needed:
                    break
            if len(speech_chunks) >= max_speech:
                break
    finally:
        proc.kill()
        proc.wait()

    if len(speech_chunks) < min_speech:
        return None
    return np.concatenate(speech_chunks)


def speak(text, tts_mod, spk, mic_card, log_fn=None):
    """Speak with TTS, suppressing mic input during playback."""
    global tts_done_at
    if not text:
        return
    if log_fn:
        log_fn("speak", text)
    tts_playing.set()
    try:
        tts_mod.speak(text, spk, mic_card)
    except Exception as e:
        if log_fn:
            log_fn("error", f"TTS error: {e}")
    finally:
        tts_playing.clear()
        tts_done_at = time.time()
