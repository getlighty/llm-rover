"""OpenAI-compatible vision chat clients."""

from __future__ import annotations

import requests

from rover_brain_v2.providers.base import ProviderError
from rover_brain_v2.providers.common import b64_image, require_env


class OpenAICompatVisionClient:
    def __init__(self, *, provider_name: str, model: str, url: str,
                 api_key_env: str):
        self.name = provider_name
        self.model = model
        self.url = url
        self.api_key_env = api_key_env

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str:
        api_key = require_env(self.api_key_env)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        for item in history or []:
            messages.append(item)
        if image_bytes:
            b64 = b64_image(image_bytes)
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = prompt
        messages.append({"role": "user", "content": user_content})
        payload = self._payload(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        if not response.ok:
            raise ProviderError(response.text)
        return response.json()["choices"][0]["message"]["content"]

    def _payload(self, **payload):
        if self.name == "groq":
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        return payload
