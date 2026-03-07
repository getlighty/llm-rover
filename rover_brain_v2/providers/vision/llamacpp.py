"""llama.cpp local vision client."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from rover_brain_v2.providers.base import ProviderError


class LlamaCppVisionClient:
    name = "llama.cpp"
    _models_dir = "/home/jasper/models"
    _llama_bin = "/home/jasper/llama.cpp/build/bin/llama-cli"
    _registry = {
        "qwen3-vl:2b": (
            "/home/jasper/models/Qwen3VL-2B-Q4_K_M.gguf",
            "/home/jasper/models/mmproj-Qwen3VL-2B-Q8_0.gguf",
        )
    }

    def __init__(self, model: str):
        self.model = model

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.2,
                 max_tokens: int = 400) -> str:
        del history
        paths = self._registry.get(self.model)
        if not paths:
            raise ProviderError(f"Unknown llama.cpp model: {self.model}")
        model_path, mmproj_path = paths
        full_prompt = f"{system.strip()}\n\nUser: {prompt}".strip()
        prompt_fd, prompt_path = tempfile.mkstemp(suffix=".txt")
        image_path = None
        try:
            os.write(prompt_fd, full_prompt.encode("utf-8"))
            os.close(prompt_fd)
            cmd = [
                self._llama_bin,
                "-m", model_path,
                "--mmproj", mmproj_path,
                "-ngl", "99",
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--no-display-prompt",
                "--single-turn",
                "-t", "4",
                "-f", prompt_path,
            ]
            if image_bytes:
                image_fd, image_path = tempfile.mkstemp(suffix=".jpg")
                os.write(image_fd, image_bytes)
                os.close(image_fd)
                cmd.extend(["--image", image_path])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise ProviderError(proc.stderr.strip() or proc.stdout.strip())
            raw = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
            return raw.strip()
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
            if image_path:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

