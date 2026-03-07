"""Shared Ollama endpoint configuration.

All Ollama-backed code paths should resolve their endpoint here instead of
hardcoding an IP. The default is the local Ollama daemon.
"""

import os


def _load_env_defaults():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(repo_dir, ".env")
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        pass


def _normalize_base_url(url):
    url = (url or "").strip()
    if not url:
        url = "http://localhost:11434"
    url = url.rstrip("/")
    if url.endswith("/api/chat"):
        url = url[:-9]
    return url


_load_env_defaults()

OLLAMA_BASE_URL = _normalize_base_url(os.environ.get("OLLAMA_URL"))
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

