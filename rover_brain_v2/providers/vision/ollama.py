"""Ollama chat client with optional image support."""

from __future__ import annotations

import requests

from rover_brain_v2.config import ollama_chat_url
from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import b64_image


class OllamaVisionClient:
    name = "ollama"

    def __init__(self, model: str):
        self.model = model
        self.url = ollama_chat_url()

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        for item in history or []:
            messages.append(item)
        user_message = {"role": "user", "content": prompt}
        if image_bytes:
            user_message["images"] = [b64_image(image_bytes)]
        messages.append(user_message)
        think = not self.model.startswith("qwen3")
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": think,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=60,
        )
        if not response.ok:
            raise ProviderError(response.text)
        message = response.json()["message"]
        content = message.get("content", "")
        if not content.strip() and message.get("thinking"):
            content = message["thinking"]
        return content

