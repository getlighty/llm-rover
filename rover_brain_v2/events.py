"""Small thread-safe structured event bus."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class EventBus:
    def __init__(self, max_events: int = 1200):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def publish(self, category: str, data: Any) -> dict[str, Any]:
        entry = {"ts": time.time(), "cat": category, "data": data}
        with self._lock:
            self._events.append(entry)
        print(f"[{category}] {data}", flush=True)
        return entry

    def since(self, last_ts: float) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if event["ts"] > last_ts]

    def snapshot(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    def compact(self, keep: int = 20):
        """Drop old events, keeping only the most recent `keep` entries."""
        with self._lock:
            if len(self._events) > keep:
                kept = list(self._events)[-keep:]
                self._events.clear()
                self._events.extend(kept)

