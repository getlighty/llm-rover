"""Provider registry smoke tests."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rover_brain_v2.providers.registry import ProviderRegistry


def test_registry_exposes_modular_provider_groups():
    registry = ProviderRegistry()
    available = registry.available()
    assert "command_llm" in available
    assert "navigator_llm" in available
    assert "orchestrator_llm" in available
    assert "tts" in available
    assert "ollama/qwen3.5:cloud" in available["navigator_llm"]


def test_registry_can_switch_independent_llm_roles():
    registry = ProviderRegistry()
    registry.set_selection(
        command_llm="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        navigator_llm="ollama/qwen3.5:9b",
        orchestrator_llm="anthropic/claude-haiku-4-5-20251001",
        tts="elevenlabs",
    )
    snapshot = registry.snapshot()
    assert snapshot["command_llm"].startswith("groq/")
    assert snapshot["navigator_llm"].startswith("ollama/")
    assert snapshot["orchestrator_llm"].startswith("anthropic/")
    assert snapshot["tts"] == "elevenlabs"


if __name__ == "__main__":
    test_registry_exposes_modular_provider_groups()
    test_registry_can_switch_independent_llm_roles()
    print("provider registry tests passed")
