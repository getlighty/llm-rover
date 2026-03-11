"""Logging wrapper for VisionLanguageClient — records prompts and responses."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class LLMCallLog:
    """Thread-safe ring buffer of LLM call records."""

    def __init__(self, max_entries: int = 80):
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._seq = 0

    def record(self, *, role: str, provider: str, model: str,
               prompt: str, system: str, has_image: bool,
               response: str, elapsed_s: float, error: str = ""):
        with self._lock:
            self._seq += 1
            self._entries.append({
                "id": self._seq,
                "ts": time.time(),
                "role": role,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "system": system,
                "has_image": has_image,
                "response": response,
                "elapsed_s": round(elapsed_s, 3),
                "error": error,
            })

    def entries(self, since_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._entries if e["id"] > since_id]


# Singleton shared across all wrapped clients
llm_call_log = LLMCallLog()


LLM_CALL_TIMEOUT_S = 45  # max seconds for any LLM call


class LoggingVLMWrapper:
    """Wraps a VisionLanguageClient, logging every complete() call.

    Enforces a hard timeout on all LLM calls to prevent stalls.
    """

    def __init__(self, client, *, role: str = "unknown"):
        self._client = client
        self._role = role

    @property
    def name(self) -> str:
        return self._client.name

    @property
    def model(self) -> str:
        return self._client.model

    def complete(self, *, prompt: str, system: str = "",
                 image_bytes: bytes | None = None,
                 history: list[dict] | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800) -> str:
        t0 = time.time()
        error = ""
        response = ""
        try:
            # Run LLM call in a thread with hard timeout
            result_box: list = []
            error_box: list = []

            def _call():
                try:
                    r = self._client.complete(
                        prompt=prompt,
                        system=system,
                        image_bytes=image_bytes,
                        history=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result_box.append(r)
                except Exception as e:
                    error_box.append(e)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=LLM_CALL_TIMEOUT_S)

            if t.is_alive():
                error = f"LLM call timed out after {LLM_CALL_TIMEOUT_S}s"
                raise TimeoutError(error)
            if error_box:
                error = str(error_box[0])
                raise error_box[0]
            response = result_box[0] if result_box else ""
            return response
        except Exception as exc:
            if not error:
                error = str(exc)
            raise
        finally:
            llm_call_log.record(
                role=self._role,
                provider=self._client.name,
                model=self._client.model,
                prompt=prompt,
                system=system,
                has_image=image_bytes is not None,
                response=response,
                elapsed_s=time.time() - t0,
                error=error,
            )

    def __getattr__(self, name):
        return getattr(self._client, name)
