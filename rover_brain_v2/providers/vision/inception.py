"""Inception Labs Mercury diffusion LLM client (text-only, non-streaming)."""

from __future__ import annotations

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import require_env


class InceptionVisionClient:
    """OpenAI-compatible client for Inception Labs Mercury models.

    Mercury is a text-only diffusion LLM — image inputs are stripped
    before sending.  Responses are always non-streaming (serialized).
    """

    API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"

    def __init__(self, *, model: str = "mercury-2"):
        self.name = "inception"
        self.model = model

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str:
        api_key = require_env("INCEPTION_API_KEY")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        for item in history or []:
            # Strip any image content from history entries
            messages.append(self._text_only(item))
        # Text-only: ignore image_bytes entirely
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise ProviderError(response.text)
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _text_only(msg: dict) -> dict:
        """Ensure a message contains only text content."""
        content = msg.get("content")
        if isinstance(content, list):
            # Extract text parts, drop image_url parts
            text_parts = [
                p["text"] for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return {**msg, "content": " ".join(text_parts) if text_parts else ""}
        return msg
