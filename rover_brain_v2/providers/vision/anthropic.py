"""Anthropic Claude vision client."""

from __future__ import annotations

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import b64_image, require_env


class AnthropicVisionClient:
    name = "anthropic"

    def __init__(self, model: str):
        self.model = model

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str:
        api_key = require_env("ANTHROPIC_API_KEY")
        content = []
        if image_bytes:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_image(image_bytes),
                },
            })
        content.append({"type": "text", "text": prompt})
        messages = list(history or [])
        messages.append({"role": "user", "content": content})
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=45,
        )
        if not response.ok:
            raise ProviderError(response.text)
        return response.json()["content"][0]["text"]

