"""orchestrator.py — Strategic advisor for the rover executor.

The orchestrator does NOT control the rover directly. It:
1. Knows the current task (set by the executor when user gives a command)
2. Gives guidance when the executor asks for help (stuck, confused, etc.)
3. Learns from user feedback and saves generalized lessons

The executor sends images and context; the orchestrator responds with JSON.
"""

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

import lessons


# ── Navigation plan dataclasses ──────────────────────────────────────

@dataclass
class Step:
    target: str              # spatial goal: "the doorway on the left"
    rationale: str           # why this step
    waypoint_budget: int = 10  # max navigator waypoints for this step


@dataclass
class StepResult:
    success: bool
    reason: str              # "arrived", "stuck", "budget", "aborted"
    waypoints_used: int
    final_scene: str         # last LLM scene from navigator
    final_yolo: str          # last YOLO summary
    exploration_summary: str  # grid state


@dataclass
class NavigationPlan:
    task: str
    steps: List[Step]
    current_index: int = 0
    history: List[str] = field(default_factory=list)  # completed step summaries

    def context_for_navigator(self) -> str:
        """Concise string for navigator's per-waypoint LLM prompt."""
        step = (self.steps[self.current_index]
                if self.current_index < len(self.steps) else None)
        if not step:
            return ""
        parts = [
            f"MISSION: {self.task}",
            f"CURRENT STEP ({self.current_index+1}/{len(self.steps)}): {step.target}",
            f"WHY: {step.rationale}",
        ]
        if self.current_index + 1 < len(self.steps):
            parts.append(f"NEXT: {self.steps[self.current_index+1].target}")
        if self.history:
            parts.append(f"DONE: {'; '.join(self.history[-3:])}")
        return "\n".join(parts)

# ── Ollama config ─────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:cloud"        # vision-capable (scene description)
OLLAMA_TEXT_MODEL = "qwen3.5:cloud"   # reasoning/guidance

# ── Constants ─────────────────────────────────────────────────────────

MAX_LLM_CALLS = 8             # cost cap per task

# ── Per-task call counter ─────────────────────────────────────────────

_llm_calls = 0
_llm_lock = threading.Lock()


def _increment_calls():
    global _llm_calls
    with _llm_lock:
        _llm_calls += 1
        return _llm_calls


def reset_call_counter():
    global _llm_calls
    with _llm_lock:
        _llm_calls = 0


def get_call_count():
    with _llm_lock:
        return _llm_calls


# ── Current task state ────────────────────────────────────────────────

_current_task = ""
_task_lock = threading.Lock()


def set_task(text):
    """Executor tells the orchestrator what the user's current task is."""
    global _current_task
    with _task_lock:
        _current_task = text


def get_task():
    with _task_lock:
        return _current_task


# ── Task classification ──────────────────────────────────────────────

SIMPLE_PATTERNS = re.compile(
    r"^(nod|shake|look|turn|lights?|say |spin|wiggle|dance|greet|wave|stop|go forward|go back)",
    re.IGNORECASE)

NAV_KEYWORDS = {"find", "go to", "navigate", "search", "explore",
                "get to", "come to", "where is", "look for", "door",
                "room", "kitchen", "hallway", "exit"}


def classify_task(text):
    """Returns 'simple' or 'complex'."""
    if SIMPLE_PATTERNS.match(text.strip()):
        return "simple"
    lower = text.lower()
    if any(kw in lower for kw in NAV_KEYWORDS):
        return "complex"
    if len(text.split()) > 6:
        return "complex"
    return "simple"


# ── Ollama API ────────────────────────────────────────────────────────

def _call_ollama(system_prompt, user_text, jpeg_bytes=None, log_fn=None,
                 model=None):
    """Make an Ollama API call, optionally with an image."""
    count = _increment_calls()
    model = model or (OLLAMA_MODEL if jpeg_bytes else OLLAMA_TEXT_MODEL)
    if log_fn:
        log_fn("orchestrator", f"LLM call #{count} ({model})")

    user_msg = {"role": "user", "content": user_text}
    if jpeg_bytes:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        user_msg["images"] = [b64]

    messages = [
        {"role": "system", "content": system_prompt},
        user_msg,
    ]

    # Disable thinking for qwen3 models
    think = not model.startswith("qwen3")

    try:
        r = requests.post(OLLAMA_URL,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": think,
                "options": {"temperature": 0.3, "num_predict": 1500},
            },
            timeout=60)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        if log_fn:
            log_fn("error", f"Orchestrator LLM call failed: {e}")
        return None


def _parse_json(text):
    """Extract JSON from a response that might have markdown fences."""
    if not text:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── Scene description (vision LLM → text for orchestrator) ──────────

DESCRIBE_SYSTEM = """You are a camera describing what it sees. Be concise and spatial.
Include: what objects are visible, their approximate positions (left/center/right, near/far),
any doorways or openings, floor type, obstacles, and how much open space there is.
Keep it to 3-5 sentences. Do NOT output JSON — just plain text.
Do NOT roleplay, do NOT say "systems online" or similar — just describe the image objectively."""


def describe_scene(jpeg_bytes, context="", log_fn=None):
    """Ask the vision LLM to describe the current camera frame in prose."""
    if not jpeg_bytes:
        return "No camera image available."
    prompt = "Describe what you see in this camera frame."
    if context:
        prompt += f" Context: {context}"
    raw = _call_ollama(DESCRIBE_SYSTEM, prompt, jpeg_bytes, log_fn=log_fn,
                       model=OLLAMA_MODEL)
    if raw:
        if log_fn:
            log_fn("orchestrator", f"Scene: {raw[:150]}")
        return raw
    return "Scene description unavailable."


# ── Route planning ───────────────────────────────────────────────────

PLAN_ROUTE_SYSTEM = """You are the route planner for Jasper, a 26cm-wide 6-wheel indoor rover.
Break the navigation task into 2-5 spatial milestones the rover can see and drive to sequentially.

Each step must be a CONCRETE SPATIAL TARGET the rover can recognize visually:
- Good: "the doorway on the left wall", "end of the hallway", "the kitchen counter"
- Bad: "turn left", "drive 2 meters", "be careful"

Rules:
- Steps should be reachable landmarks, not abstract actions
- First step should be something currently visible or very nearby
- Last step should be the final destination or as close as possible
- 2-5 steps total (fewer if the target is nearby/simple)
- Each step gets a waypoint_budget (5-15) — more for tricky navigation

Respond with ONLY valid JSON:
{"steps": [{"target": "...", "rationale": "...", "waypoint_budget": 10}, ...],
 "route_reasoning": "1-2 sentence route description"}"""


def plan_route(task, jpeg_bytes=None, exploration_summary="", log_fn=None):
    """Break a navigation task into spatial steps. Returns NavigationPlan.

    Makes 1 vision LLM call. Falls back to single-step plan on failure.
    """
    existing_lessons = lessons.load_lessons()
    lessons_block = ""
    if existing_lessons:
        lessons_block = "\nLearned lessons:\n"
        for l in existing_lessons:
            lessons_block += f"- {l['lesson']}\n"

    user_text = f'Navigation task: "{task}"\n'
    if exploration_summary:
        user_text += f"\nExploration grid:\n{exploration_summary}\n"
    if lessons_block:
        user_text += lessons_block

    raw = _call_ollama(PLAN_ROUTE_SYSTEM, user_text, jpeg_bytes,
                       log_fn=log_fn, model=OLLAMA_MODEL)
    data = _parse_json(raw)

    if data and "steps" in data and isinstance(data["steps"], list):
        steps = []
        for s in data["steps"]:
            if isinstance(s, dict) and "target" in s:
                budget = max(8, int(s.get("waypoint_budget", 10)))
                steps.append(Step(
                    target=s["target"],
                    rationale=s.get("rationale", ""),
                    waypoint_budget=budget,
                ))
        if steps:
            reasoning = data.get("route_reasoning", "")
            if log_fn:
                step_names = [s.target for s in steps]
                log_fn("orchestrator",
                       f"Plan: {len(steps)} steps — {', '.join(step_names)}")
                if reasoning:
                    log_fn("orchestrator", f"Route: {reasoning}")
            return NavigationPlan(task=task, steps=steps)

    # Fallback: single-step plan (degrades to current behavior)
    if log_fn:
        log_fn("orchestrator",
               f"Plan fallback: single step '{task}' "
               f"(parse failed: {raw[:100] if raw else 'no response'})")
    return NavigationPlan(
        task=task,
        steps=[Step(target=task, rationale="direct navigation", waypoint_budget=30)],
    )


# ── Step evaluation ──────────────────────────────────────────────────

EVALUATE_STEP_SYSTEM = """You are evaluating navigation progress for Jasper, a 6-wheel indoor rover.
A step in the navigation plan just completed. Decide what to do next.

Decisions:
- "continue": Step succeeded, move to the next step in the plan
- "skip": Step didn't fully succeed but we're close enough, move on
- "retry": Step failed but is worth retrying (provide a hint for the retry)
- "replan": The situation changed significantly — provide new remaining steps
- "abort": Navigation is impossible (blocked, unsafe, or target doesn't exist)

Respond with ONLY valid JSON:
{"decision": "continue|skip|retry|replan|abort",
 "reason": "brief explanation",
 "new_steps": [{"target": "...", "rationale": "...", "waypoint_budget": 10}],
 "retry_hint": "hint for the navigator on retry"}

Only include "new_steps" if decision is "replan".
Only include "retry_hint" if decision is "retry"."""


def evaluate_step(plan, result, jpeg_bytes=None, log_fn=None):
    """Evaluate a completed step and decide next action. Returns decision dict.

    Makes 1 LLM call. Falls back to simple logic if budget exhausted or LLM fails.
    """
    # Budget check — if exhausted, use simple logic
    if get_call_count() >= MAX_LLM_CALLS:
        decision = "continue" if result.success else "abort"
        if log_fn:
            log_fn("orchestrator",
                   f"Evaluate (budget): {decision} (calls={get_call_count()})")
        return {"decision": decision, "reason": "LLM budget exhausted"}

    step = (plan.steps[plan.current_index]
            if plan.current_index < len(plan.steps) else None)
    step_desc = step.target if step else "unknown"

    user_text = (
        f"MISSION: {plan.task}\n"
        f"STEP {plan.current_index+1}/{len(plan.steps)}: '{step_desc}'\n"
        f"RESULT: {'SUCCESS' if result.success else 'FAILED'} — {result.reason}\n"
        f"Waypoints used: {result.waypoints_used}\n"
        f"Scene: {result.final_scene}\n"
        f"YOLO: {result.final_yolo}\n"
    )
    if result.exploration_summary:
        user_text += f"Exploration: {result.exploration_summary}\n"
    if plan.history:
        user_text += f"Completed: {'; '.join(plan.history[-3:])}\n"
    remaining = [s.target for s in plan.steps[plan.current_index + 1:]]
    if remaining:
        user_text += f"Remaining steps: {', '.join(remaining)}\n"

    raw = _call_ollama(EVALUATE_STEP_SYSTEM, user_text, jpeg_bytes,
                       log_fn=log_fn, model=OLLAMA_TEXT_MODEL)
    data = _parse_json(raw)

    if data and "decision" in data:
        decision = data["decision"]
        if decision not in ("continue", "skip", "retry", "replan", "abort"):
            decision = "continue" if result.success else "abort"
            data["decision"] = decision
        if log_fn:
            log_fn("orchestrator",
                   f"Evaluate: {decision} — {data.get('reason', '')[:100]}")
        return data

    # Fallback on parse failure
    decision = "continue" if result.success else "abort"
    if log_fn:
        log_fn("orchestrator",
               f"Evaluate fallback: {decision} "
               f"(parse failed: {raw[:80] if raw else 'no response'})")
    return {"decision": decision, "reason": "evaluation parse failed"}


# ── Guidance (executor asks for help) ────────────────────────────────

GUIDANCE_SYSTEM = """You are the strategic advisor for Jasper, a 6-wheel rover.
The executor is asking for your help because it is stuck or needs guidance.

You receive:
- The current task (what the user asked the rover to do)
- A scene description from the vision system (what the camera sees right now)
- Context from the executor (what happened, why it needs help)

## Rover capabilities
- 6 wheels, differential drive, max 0.2 m/s (slow mover)
- Pan-tilt camera gimbal (the "head") — can look around independently of body
- 26cm wide, 35cm long — can't fit through gaps < 30cm
- Camera is USB 640x480 wide-angle (~65 deg FOV)
- Indoor environment (home/workshop)

## Your job
Analyze the situation and give concrete, actionable guidance. Think about:
- Where is the rover relative to its goal?
- What obstacles are in the way?
- What is the best escape/approach strategy?
- Should the rover try a completely different route?
- Should it give up (abort)?

Respond with ONLY valid JSON:
{"action": "redirect", "guidance": "Specific instructions for what to do next"}
{"action": "abort", "reason": "Why the task cannot be completed"}

Rules:
- "redirect" with clear, specific instructions (e.g. "Back up 2 seconds, spin right 90 degrees, drive forward toward the doorway visible on the right side")
- "abort" only if truly impossible (e.g. blocked by closed door, target doesn't exist)
- Be specific about directions, distances, and what to look for
- Reference what the scene description tells you"""


def ask_guidance(jpeg_bytes, context, log_fn=None):
    """Executor asks orchestrator for help. Sends image + context, gets JSON guidance.

    Returns dict with 'action' and 'guidance'/'reason', or None on failure.
    """
    task = get_task()

    # Get scene description from vision LLM
    scene = describe_scene(jpeg_bytes, context=task, log_fn=log_fn)

    # Include learned lessons
    existing_lessons = lessons.load_lessons()
    lessons_block = ""
    if existing_lessons:
        lessons_block = "\nLearned lessons:\n"
        for l in existing_lessons:
            lessons_block += f"- {l['lesson']}\n"

    user_text = (
        f"Task: \"{task}\"\n\n"
        f"Scene (from vision system):\n{scene}\n\n"
        f"Executor context:\n{context}\n"
        f"{lessons_block}"
    )

    raw = _call_ollama(GUIDANCE_SYSTEM, user_text, log_fn=log_fn)
    data = _parse_json(raw)
    if not data or "action" not in data:
        if log_fn:
            log_fn("orchestrator", f"Guidance parse failed: {raw[:200] if raw else 'no response'}")
        return None

    if log_fn:
        if data["action"] == "redirect":
            log_fn("orchestrator", f"Guidance: {data.get('guidance', '')[:150]}")
        elif data["action"] == "abort":
            log_fn("orchestrator", f"Abort: {data.get('reason', '')[:150]}")

    return data


# ── Learn from user feedback ────────────────────────────────────────

GENERALIZE_SYSTEM = """You are a learning system for Jasper, a 6-wheel rover.
The user gave feedback or a correction during a task. Your job is to extract a GENERAL RULE
that applies broadly — not just to this specific situation.

Examples:
- User says "you hit a bag" during navigation → "When objects are detected nearby on the floor, slow down and steer wide around them. Bags and soft objects on the floor are easy to miss visually."
- User says "don't go there" while heading toward a dark corner → "Avoid driving into dark or poorly-lit areas unless explicitly instructed. Visibility is critical for safe navigation."
- User says "you're too close to the wall" → "Maintain at least 15cm clearance from walls and large surfaces. If a wall fills more than 60% of one side of the frame, steer away."

Rules for generating lessons:
- Make it GENERAL — it should apply to similar situations in any room, not just this specific object/place
- Keep it actionable — describe what the rover should DO differently
- Keep it concise — one or two sentences max
- Include the sensory cue (what the rover would see/detect) and the correct action
- Do NOT reference specific rooms, objects by name, or one-time events

Respond with ONLY valid JSON:
{"lesson": "The general rule text", "context": "navigation|safety|interaction|scanning"}"""


def learn_from_feedback(user_feedback, task_text, round_summaries=None,
                        log_fn=None):
    """Generalize user feedback into a reusable lesson and save it."""
    summary = ""
    if round_summaries:
        summary = "\n".join(round_summaries[-6:])

    user_text = (
        f"Task: \"{task_text}\"\n"
        f"User feedback: \"{user_feedback}\"\n"
    )
    if summary:
        user_text += f"\nRecent execution context:\n{summary}\n"

    raw = _call_ollama(GENERALIZE_SYSTEM, user_text, log_fn=log_fn)
    data = _parse_json(raw)
    if not data or "lesson" not in data:
        if log_fn:
            log_fn("orchestrator", f"Failed to generalize feedback: {raw[:200] if raw else 'no response'}")
        return None

    lesson_text = data["lesson"]
    context = data.get("context", "general")

    lessons.save_lesson(
        lesson_text,
        context=context,
        trigger="user_feedback",
        source=user_feedback[:100],
    )
    if log_fn:
        log_fn("orchestrator", f"Learned: {lesson_text}")
    return lesson_text
