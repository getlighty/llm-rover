"""Provider base classes and shared exceptions."""

from __future__ import annotations

from typing import Protocol


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class SpeechToTextClient(Protocol):
    name: str

    def transcribe(self, audio_48k): ...


class TextToSpeechClient(Protocol):
    name: str

    def speak(self, text: str, speaker_dev: str, mic_card: str | None): ...


class VisionLanguageClient(Protocol):
    name: str
    model: str

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str: ...

