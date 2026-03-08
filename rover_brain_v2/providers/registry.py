"""Registry that exposes modular STT/TTS/VLM clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rover_brain_v2.models import ProviderSelection

_PERSIST_PATH = Path(__file__).resolve().parents[1] / "data" / "provider_selection.json"
from rover_brain_v2.providers.base import VisionLanguageClient
from rover_brain_v2.providers.stt.groq import GroqSpeechToText
from rover_brain_v2.providers.tts.elevenlabs import ElevenLabsTextToSpeech
from rover_brain_v2.providers.tts.groq import GroqTextToSpeech
from rover_brain_v2.providers.vision.anthropic import AnthropicVisionClient
from rover_brain_v2.providers.vision.llamacpp import LlamaCppVisionClient
from rover_brain_v2.providers.vision.ollama import OllamaVisionClient
from rover_brain_v2.providers.vision.openai_compat import OpenAICompatVisionClient


AVAILABLE_STT = ["groq"]
AVAILABLE_TTS = ["groq", "elevenlabs"]
AVAILABLE_VLMS = [
    "ollama/qwen3.5:9b",
    "ollama/qwen3.5:cloud",
    "ollama/qwen3-vl:2b",
    "ollama/ministral-3:14b-cloud",
    "ollama/gemma3:27b-cloud",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4-1-fast-non-reasoning",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-haiku-4-5-20251001",
    "llama.cpp/qwen3-vl:2b",
]


@dataclass(slots=True)
class ProviderBundle:
    stt: GroqSpeechToText
    command_llm: VisionLanguageClient
    navigator_llm: VisionLanguageClient
    orchestrator_llm: VisionLanguageClient
    tts: object


class ProviderRegistry:
    def __init__(self):
        self.selection = ProviderSelection(
            stt="groq",
            navigator_llm="ollama/qwen3.5:cloud",
            orchestrator_llm="ollama/qwen3.5:cloud",
            command_llm="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
            tts="groq",
        )
        self._load_selection()
        self._stt_clients = {"groq": GroqSpeechToText()}
        self._tts_clients = {
            "groq": GroqTextToSpeech(),
            "elevenlabs": ElevenLabsTextToSpeech(),
        }
        self._vlm_cache: dict[str, VisionLanguageClient] = {}

    def _load_selection(self):
        try:
            data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
            for key in ("stt", "tts", "command_llm", "navigator_llm", "orchestrator_llm"):
                if key in data and hasattr(self.selection, key):
                    setattr(self.selection, key, data[key])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def _save_selection(self):
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(
            json.dumps(self.snapshot(), indent=2) + "\n",
            encoding="utf-8",
        )

    def available(self) -> dict[str, list[str]]:
        return {
            "stt": list(AVAILABLE_STT),
            "tts": list(AVAILABLE_TTS),
            "command_llm": list(AVAILABLE_VLMS),
            "navigator_llm": list(AVAILABLE_VLMS),
            "orchestrator_llm": list(AVAILABLE_VLMS),
        }

    def snapshot(self) -> dict[str, str]:
        return {
            "stt": self.selection.stt,
            "tts": self.selection.tts,
            "command_llm": self.selection.command_llm,
            "navigator_llm": self.selection.navigator_llm,
            "orchestrator_llm": self.selection.orchestrator_llm,
        }

    def set_selection(self, **changes) -> dict[str, str]:
        for key, value in changes.items():
            if not hasattr(self.selection, key):
                continue
            setattr(self.selection, key, value)
        self._save_selection()
        return self.snapshot()

    def bundle(self) -> ProviderBundle:
        return ProviderBundle(
            stt=self._stt_clients[self.selection.stt],
            command_llm=self._get_vlm(self.selection.command_llm),
            navigator_llm=self._get_vlm(self.selection.navigator_llm),
            orchestrator_llm=self._get_vlm(self.selection.orchestrator_llm),
            tts=self._tts_clients[self.selection.tts],
        )

    def _get_vlm(self, spec: str) -> VisionLanguageClient:
        client = self._vlm_cache.get(spec)
        if client is not None:
            return client
        provider, model = spec.split("/", 1)
        if provider == "ollama":
            client = OllamaVisionClient(model=model)
        elif provider == "anthropic":
            client = AnthropicVisionClient(model=model)
        elif provider == "groq":
            client = OpenAICompatVisionClient(
                provider_name="groq",
                model=model,
                url="https://api.groq.com/openai/v1/chat/completions",
                api_key_env="GROQ_API_KEY",
            )
        elif provider == "xai":
            client = OpenAICompatVisionClient(
                provider_name="xai",
                model=model,
                url="https://api.x.ai/v1/chat/completions",
                api_key_env="XAI_API_KEY",
            )
        elif provider == "llama.cpp":
            client = LlamaCppVisionClient(model=model)
        else:
            raise ValueError(f"Unsupported provider spec: {spec}")
        self._vlm_cache[spec] = client
        return client
