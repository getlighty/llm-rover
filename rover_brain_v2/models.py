"""Shared dataclasses for runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderSelection:
    stt: str
    navigator_llm: str
    orchestrator_llm: str
    command_llm: str
    tts: str


@dataclass(slots=True)
class MotionCommandResult:
    status: str
    detail: str = ""
    rounds: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeFlags:
    desk_mode: bool = True
    stt_enabled: bool = True
    tts_enabled: bool = True
    gimbal_pan_enabled: bool = True
    killed: bool = False
    yolo_overlay_enabled: bool = False
    reverse_look_behind: bool = False


@dataclass(slots=True)
class FollowRequest:
    target: str = "person"
    duration: float = 60.0
    target_bw: float = 0.25


@dataclass(slots=True)
class NavigationRequest:
    target: str
    topological: bool = True


@dataclass(slots=True)
class NavigatorTask:
    task_id: int
    mode: str
    target: str
    plan_context: str = ""
    leg_hint: str = ""
    waypoint_budget: int | None = None
    instruction: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0


@dataclass(slots=True)
class NavigatorResult:
    task_id: int
    mode: str
    status: str
    summary: str
    scene: str = ""
    reached: bool = False
    room_guess: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestratorTask:
    task_id: int
    target_query: str
    topological: bool = True


@dataclass(slots=True)
class OrchestratorResult:
    task_id: int
    status: str
    target_query: str
    target_room: str | None = None
    current_room: str | None = None
    reached: bool = False
    completed_legs: int = 0
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
