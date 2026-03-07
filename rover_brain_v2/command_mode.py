"""General LLM command execution loop for rover_brain_v2."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rover_brain_v2.json_utils import as_float, extract_json_dict
from rover_brain_v2.models import MotionCommandResult
from rover_brain_v2.prompts import command_system_prompt


MEMORY_FILE = Path(__file__).resolve().parent / "memory.md"


class GeneralCommandController:
    def __init__(self, *, rover, camera, event_bus, flags, config, speak_fn):
        self.rover = rover
        self.camera = camera
        self.events = event_bus
        self.flags = flags
        self.config = config
        self.speak = speak_fn
        self._history: list[dict] = []
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()
        self.rover.stop()

    def run(self, text: str, llm_client) -> MotionCommandResult:
        self._cancel.clear()
        rounds = 0
        pending_prompt = text
        self.events.publish("plan", f"General command: {text}")
        while rounds <= self.config.command_observe_rounds:
            if self._cancel.is_set():
                return MotionCommandResult(status="cancelled", rounds=rounds)
            frame = self.camera.snap()
            result = self._call_llm(pending_prompt, frame, llm_client)
            commands = list(result.get("commands", []))
            if result.get("duration"):
                self._inject_duration_stop(commands, as_float(result.get("duration"), 0.0))
            speak_text = str(result.get("speak", "")).strip()
            remember = str(result.get("remember", "")).strip()
            observe = bool(result.get("observe", False))
            if speak_text:
                self.speak(speak_text)
            if remember:
                self._remember(remember)
            self._execute(commands)
            rounds += 1
            if not observe:
                return MotionCommandResult(
                    status="completed",
                    rounds=rounds,
                    payload=result,
                )
            scene = self._observe_scene(llm_client)
            pending_prompt = (
                f"Observe round {rounds}. This is the result after your previous commands.\n"
                f"Current scene: {scene or 'unknown'}\n"
                "If the task is done, stop observing. Otherwise output the next JSON action."
            )
        return MotionCommandResult(status="observe_limit", rounds=rounds)

    def _call_llm(self, text: str, frame: bytes | None, llm_client):
        system = command_system_prompt(gimbal_pan_enabled=self.flags.gimbal_pan_enabled)
        history = self._history[-8:]
        try:
            raw = llm_client.complete(
                prompt=text,
                system=system,
                image_bytes=frame,
                history=history,
                max_tokens=600,
            )
        except Exception as exc:
            self.events.publish("error", f"Command LLM failed: {exc}")
            raw = '{"commands":[],"speak":"LLM unavailable","observe":false}'
        self.events.publish("llm", raw[:600])
        parsed = extract_json_dict(raw)
        if not isinstance(parsed, dict):
            parsed = {"commands": [], "speak": "I could not parse that.", "observe": False}
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": raw})
        self._history = self._history[-12:]
        return parsed

    def _execute(self, commands: list[dict]):
        for command in commands:
            if self._cancel.is_set():
                break
            if not isinstance(command, dict):
                continue
            if "_pause" in command:
                self._sleep(as_float(command["_pause"], 0.0))
                continue
            self.rover.send(command)
            if command.get("T") == 1 and (command.get("L", 0) or command.get("R", 0)):
                self._sleep(0.15)
        self.rover.stop()

    def _inject_duration_stop(self, commands: list[dict], duration: float):
        if duration <= 0:
            return
        for idx in range(len(commands) - 1, -1, -1):
            command = commands[idx]
            if isinstance(command, dict) and command.get("T") == 1 and (command.get("L", 0) or command.get("R", 0)):
                commands.insert(idx + 1, {"_pause": min(duration, 8.0)})
                commands.insert(idx + 2, {"T": 1, "L": 0, "R": 0})
                return

    def _observe_scene(self, llm_client) -> str:
        frame = self.camera.snap()
        if frame is None:
            return ""
        try:
            return llm_client.complete(
                prompt="Describe what changed after the last action in one sentence.",
                system="Reply in one plain sentence. No JSON.",
                image_bytes=frame,
                max_tokens=120,
            ).strip()
        except Exception:
            return ""

    def _remember(self, note: str):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = f"- {time.strftime('%Y-%m-%d %H:%M')} {note.strip()}\n"
        with MEMORY_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self.events.publish("system", f"Memory saved: {note[:120]}")

    def _sleep(self, seconds: float):
        end = time.time() + max(seconds, 0.0)
        while time.time() < end:
            if self._cancel.is_set():
                break
            time.sleep(0.05)
