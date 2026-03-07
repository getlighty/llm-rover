"""Shared JSON extraction and numeric helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def strip_code_fences(text: str) -> str:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return clean


def extract_json_text(text: str) -> str:
    clean = strip_code_fences(text)
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        return clean[start:end + 1]
    return clean


def extract_json_dict(text: str) -> dict[str, Any] | None:
    clean = extract_json_text(text)
    candidates = [clean]
    candidates.append(re.sub(r",\s*([}\]])", r"\1", clean))
    candidates.append(re.sub(r'"\s*"', '","', clean))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

