"""reflection.py — Orchestrator-level post-plan reflection.

Fully self-contained module with its own LLM connection (Anthropic Claude Opus 4.6).
Runs in a separate conversation context from the move-directing LLM.
Receives only processed/summarized data — never raw frames or movement history.
"""

import os
import requests

import lessons

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-6"

# ── Orchestrator system prompt — high-level principles only ────────────

ORCHESTRATOR_SYSTEM = """You are the learning orchestrator for Jasper, a 6-wheel rover robot with a pan-tilt camera gimbal.

Your role: analyze plan outcomes and extract reusable lessons so the rover improves over time.

## Rover context
- Jasper navigates indoor spaces using camera vision + LLM-directed wheel/gimbal commands
- Camera is on a gimbal (the "head") — body and camera can face different directions
- Common failure modes: steering into wrong objects, getting physically stuck, misidentifying targets, losing sight of goals
- The rover sees one camera frame at a time and makes reactive steering decisions

## Your principles
- Extract only ONE lesson per reflection — specific and actionable
- Max 25 words — the lesson gets injected into the rover's system prompt
- Focus on what to DO differently next time, not what went wrong
- If a similar lesson already exists, either refine it or say NONE
- Lessons should generalize beyond the single incident when possible
- Never reference specific rooms, objects, or timestamps — keep lessons universal

## Output format
Respond with ONLY the lesson text, or NONE if no useful lesson can be extracted.
No quotes, no explanation, no preamble."""


def _call_orchestrator(user_prompt):
    """Make a direct text-only LLM call to Anthropic Opus 4.6. Separate from rover's LLM."""
    if not ANTHROPIC_API_KEY:
        return None
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    try:
        r = requests.post(ANTHROPIC_URL,
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL,
                  "system": ORCHESTRATOR_SYSTEM,
                  "messages": messages,
                  "temperature": 0.3, "max_tokens": 100},
            timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception:
        return None


def reflect(request, trigger, feedback=None, stuck_count=0, history=None,
            log_fn=None):
    """Analyze a completed plan and extract a lesson if warranted.

    All inputs are processed/summarized data — no raw frames or LLM transcripts.

    Args:
        request: original user request text (e.g. "go to the door")
        trigger: why reflection was triggered:
                 "feedback_negative" | "stuck_repeated" | "cancelled"
        feedback: list of negative feedback strings from user
        stuck_count: total stuck-detection events during the plan
        history: list of brief round summaries (e.g. ["Said: Door ahead", "Stuck #1"])
        log_fn: optional callable(category, message) for logging

    Returns:
        lesson text (str) or None
    """
    # Build the processed-data summary for the orchestrator
    parts = [f"Plan: \"{request}\"", f"Trigger: {trigger}"]

    if feedback:
        parts.append("User corrections during plan:")
        for f in feedback:
            parts.append(f"  - \"{f}\"")

    if stuck_count > 0:
        parts.append(f"Got physically stuck {stuck_count} time(s)")

    if history:
        parts.append("Execution summary (last 8 events):")
        for h in history[-8:]:
            parts.append(f"  - {h}")

    # Include existing lessons so the orchestrator can avoid duplicates / refine
    existing = lessons.load_lessons()
    if existing:
        parts.append("Existing lessons already learned:")
        for l in existing:
            parts.append(f"  - {l['lesson']}")

    user_prompt = "\n".join(parts)

    result = _call_orchestrator(user_prompt)

    if not result:
        return None

    text = result.strip().strip('"').strip("'")
    if not text or text.upper() == "NONE" or len(text) > 150:
        return None

    # Determine context category
    context = "navigation" if "stuck" in trigger else "general"
    if any(w in request.lower() for w in ("go to", "find", "navigate", "door", "room")):
        context = "navigation"

    source = ""
    if feedback:
        source = f"User said '{feedback[0]}'"
    elif trigger == "stuck_repeated":
        source = f"Got stuck {stuck_count} times"

    lessons.save_lesson(text, context=context, trigger=trigger, source=source)
    if log_fn:
        log_fn("system", f"Learned: {text}")
    return text
