"""lessons.py — Learned lesson storage & retrieval.

Standalone module, no dependencies on rover_brain_llm.
Stores lessons in lessons.json alongside this file.
"""

import json
import os
import time

ROVER_DIR = os.path.dirname(os.path.abspath(__file__))
LESSONS_FILE = os.path.join(ROVER_DIR, "lessons.json")
MAX_LESSONS = 20


def load_lessons():
    """Read lessons.json, return list of dicts. Handles missing/corrupt."""
    if not os.path.exists(LESSONS_FILE):
        return []
    try:
        with open(LESSONS_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _write_lessons(lessons):
    """Write lessons list to disk."""
    with open(LESSONS_FILE, "w") as f:
        json.dump(lessons, f, indent=2)


def save_lesson(text, context="general", trigger="unknown", source=""):
    """Append a lesson. Dedup by first 40 chars, cap at MAX_LESSONS (oldest dropped)."""
    lessons = load_lessons()
    prefix = text[:40].lower()
    for existing in lessons:
        if existing.get("lesson", "")[:40].lower() == prefix:
            return  # duplicate
    next_id = max((l.get("id", 0) for l in lessons), default=0) + 1
    lessons.append({
        "id": next_id,
        "lesson": text,
        "context": context,
        "trigger": trigger,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "source_event": source,
    })
    # Cap oldest
    while len(lessons) > MAX_LESSONS:
        lessons.pop(0)
    _write_lessons(lessons)


def delete_lesson(lesson_id):
    """Remove a lesson by ID. Returns True if found and removed."""
    lessons = load_lessons()
    before = len(lessons)
    lessons = [l for l in lessons if l.get("id") != lesson_id]
    if len(lessons) < before:
        _write_lessons(lessons)
        return True
    return False


def format_for_prompt():
    """Return formatted '## Learned Lessons' block for the system prompt."""
    lessons = load_lessons()
    if not lessons:
        return ""
    lines = ["## Learned Lessons"]
    for l in lessons:
        lines.append(f"- {l['lesson']}")
    return "\n".join(lines)
