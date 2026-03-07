"""Provider: llama.cpp local vision LLM (qwen3-vl) + Groq STT/TTS."""

import io
import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "qwen3-vl:2b"
TTS_VOICE = "troy"
NAME = "llama.cpp qwen3-vl:2b + Groq STT/TTS"

_MODELS_DIR = "/home/jasper/models"
_LLAMACPP_BIN = "/home/jasper/llama.cpp/build/bin/llama-cli"

# model tag -> (gguf, mmproj)
_MODEL_REGISTRY = {
    "qwen3-vl:2b": (
        os.path.join(_MODELS_DIR, "Qwen3VL-2B-Q4_K_M.gguf"),
        os.path.join(_MODELS_DIR, "mmproj-Qwen3VL-2B-Q8_0.gguf"),
    ),
}


def transcribe(audio_48k):
    """48kHz float32 -> Groq Whisper -> text."""
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


def _run_llamacpp(prompt_text, image_bytes=None, max_tokens=300):
    """Call llama-cli with prompt and optional image. Returns raw stdout."""
    paths = _MODEL_REGISTRY.get(LLM_MODEL)
    if not paths:
        raise ValueError(f"Unknown llama.cpp model: {LLM_MODEL}")
    model_path, mmproj_path = paths

    prompt_fd, prompt_path = tempfile.mkstemp(suffix=".txt")
    img_path = None
    try:
        os.write(prompt_fd, prompt_text.encode())
        os.close(prompt_fd)

        cmd = [
            _LLAMACPP_BIN,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-ngl", "99",
            "-n", str(max_tokens),
            "--temp", "0.1",
            "--no-display-prompt",
            "--single-turn",
            "-t", "4",
            "-f", prompt_path,
        ]

        if image_bytes:
            img_fd, img_path = tempfile.mkstemp(suffix=".jpg")
            os.write(img_fd, image_bytes)
            os.close(img_fd)
            cmd.extend(["--image", img_path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120)

        raw = result.stdout
        # Strip ANSI codes
        raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        # The model echoes the prompt then appends its response.
        # Find all top-level JSON objects with "commands" key — take the last one.
        import json as _json
        best = None
        for m in re.finditer(r'\{[^{}]*"commands"[^{}]*\{.*?\}[^{}]*\}', raw, re.DOTALL):
            best = m.group(0)
        if best:
            return best
        # Fallback: find the last {"commands": ...} by scanning backwards
        # for lines starting with {"commands"
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if line.startswith('{"commands'):
                end = line.rfind('}')
                if end > 0:
                    return line[:end + 1]
        # Last resort: return everything after the last spinner line
        parts = re.split(r'[|/\\-]{2,}\s*', raw)
        if len(parts) > 1:
            tail = parts[-1].strip()
            start = tail.find('{')
            end = tail.rfind('}')
            if start >= 0 and end > start:
                return tail[start:end + 1]
        return raw.strip()
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass
        if img_path:
            try:
                os.unlink(img_path)
            except OSError:
                pass


_SHORT_SYSTEM = """You are Jasper, a rover robot. Reply ONLY one raw JSON object.

Examples:
"lights on" → {"commands":[{"T":132,"IO4":200,"IO5":200},{"T":133,"X":0,"Y":20,"SPD":400,"ACC":20},{"T":133,"X":0,"Y":0,"SPD":400,"ACC":20}],"speak":"Lights on"}
"lights off" → {"commands":[{"T":132,"IO4":0,"IO5":0},{"T":133,"X":0,"Y":20,"SPD":400,"ACC":20},{"T":133,"X":0,"Y":0,"SPD":400,"ACC":20}],"speak":"Lights off"}
"go forward" → {"commands":[{"T":1,"L":0.15,"R":0.15}],"speak":"Moving","duration":2}
"stop" → {"commands":[{"T":1,"L":0,"R":0}],"speak":"Stopping"}
"turn left" → {"commands":[{"T":1,"L":-0.2,"R":0.2}],"speak":"Turning","duration":1}
"turn right" → {"commands":[{"T":1,"L":0.2,"R":-0.2}],"speak":"Turning","duration":1}
"back up" → {"commands":[{"T":1,"L":-0.15,"R":-0.15}],"speak":"Backing up","duration":2}
"look left" → {"commands":[{"T":133,"X":-90,"Y":0,"SPD":300,"ACC":20}],"speak":"Looking"}
"look right" → {"commands":[{"T":133,"X":90,"Y":0,"SPD":300,"ACC":20}],"speak":"Looking"}
"look up" → {"commands":[{"T":133,"X":0,"Y":60,"SPD":300,"ACC":20}],"speak":"Looking up"}
"center" → {"commands":[{"T":133,"X":0,"Y":0,"SPD":300,"ACC":20}],"speak":"Centered"}
"nod yes" → {"commands":[{"T":133,"X":0,"Y":30,"SPD":500,"ACC":20},{"T":133,"X":0,"Y":-10,"SPD":500,"ACC":20},{"T":133,"X":0,"Y":0,"SPD":300,"ACC":20}],"speak":"Yes"}
"shake no" → {"commands":[{"T":133,"X":-40,"Y":0,"SPD":500,"ACC":20},{"T":133,"X":40,"Y":0,"SPD":500,"ACC":20},{"T":133,"X":0,"Y":0,"SPD":300,"ACC":20}],"speak":"No"}
"hello" → {"commands":[{"T":132,"IO5":200},{"T":133,"X":0,"Y":30,"SPD":400,"ACC":20},{"T":133,"X":0,"Y":0,"SPD":400,"ACC":20},{"T":132,"IO5":0}],"speak":"Hey there"}
"display hi" → {"commands":[{"T":3,"lineNum":1,"Text":"Hello!"}],"speak":"Hi"}
"dim lights" → {"commands":[{"T":132,"IO4":30,"IO5":40}],"speak":"Dimmed"}
"dance" → {"commands":[{"T":1,"L":0.3,"R":-0.3},{"T":132,"IO5":255},{"T":1,"L":-0.3,"R":0.3},{"T":132,"IO5":0}],"speak":"Dancing","duration":1}

Commands: wheels {"T":1,"L":speed,"R":speed}, gimbal {"T":133,"X":pan,"Y":tilt,"SPD":300,"ACC":20}, lights {"T":132,"IO4":0-255,"IO5":0-255}, oled {"T":3,"lineNum":0-3,"Text":"msg"}
"duration" field = seconds to run wheel commands before auto-stop.
Always include a head gesture (nod/tilt) with your response."""


def call_llm(text, jpeg_bytes, system, history):
    """Text + JPEG -> llama.cpp qwen3-vl -> raw reply string."""
    # Use short system prompt — 2B model can't handle long prompts
    prompt = f"{_SHORT_SYSTEM}\n\nUser: {text}"
    raw = _run_llamacpp(prompt, jpeg_bytes, max_tokens=300)
    print(f"[llamacpp-raw] {raw[:300]}")
    return raw


def speak(text, speaker_dev, mic_card):
    """Text -> Groq Orpheus TTS -> aplay."""
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
    wav = r.content
    # Mute mic during playback
    if mic_card is not None:
        subprocess.run(["amixer", "-c", str(mic_card), "set",
                        "Capture", "nocap"],
                       capture_output=True)
    try:
        proc = subprocess.Popen(
            ["aplay", "-D", speaker_dev, "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        proc.communicate(wav, timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        if mic_card is not None:
            subprocess.run(["amixer", "-c", str(mic_card), "set",
                            "Capture", "cap"],
                           capture_output=True)
