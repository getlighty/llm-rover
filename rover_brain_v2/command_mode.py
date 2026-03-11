"""General LLM command execution loop for rover_brain_v2 — tool-based."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rover_brain_v2.json_utils import as_float, extract_json_dict, clamp
from rover_brain_v2.models import MotionCommandResult
from rover_brain_v2.prompts import command_system_prompt, load_prompt


MEMORY_FILE = Path(__file__).resolve().parent / "memory.md"

TURN_SPEED = 0.35
TURN_RATE_DPS = 200.0


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
        self._last_gimbal_pan = 0.0
        self._last_gimbal_tilt = 0.0

    def cancel(self):
        self._cancel.set()
        self.rover.stop()

    def run(self, text: str, llm_client, *, speak_allowed: bool = True) -> MotionCommandResult:
        self._cancel.clear()
        self._speak_allowed = speak_allowed
        rounds = 0
        pending_prompt = text
        self.events.publish("plan", f"General command: {text}")
        while rounds <= self.config.command_observe_rounds:
            if self._cancel.is_set():
                return MotionCommandResult(status="cancelled", rounds=rounds)
            frame = self.camera.snap()
            result = self._call_llm(pending_prompt, frame, llm_client)
            tools = list(result.get("tools") or result.get("commands") or [])
            speak_text = str(result.get("speak", "")).strip()
            remember = str(result.get("remember", "")).strip()
            observe = bool(result.get("observe", False))
            navigate = str(result.get("navigate", "")).strip()
            if speak_text and self._speak_allowed:
                self.speak(speak_text)
            if remember:
                self._remember(remember)
            if navigate:
                return MotionCommandResult(
                    status="navigate",
                    rounds=rounds,
                    payload=result,
                )
            self._execute_tools(tools)
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
            raw = '{"tools":[],"speak":"LLM unavailable","observe":false}'
        self.events.publish("llm", raw[:600])
        parsed = extract_json_dict(raw)
        if not isinstance(parsed, dict):
            parsed = {"tools": [], "speak": "I could not parse that.", "observe": False}
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": raw})
        self._history = self._history[-12:]
        return parsed

    def _execute_tools(self, tools: list):
        for tc in tools:
            if self._cancel.is_set():
                break
            if not isinstance(tc, dict):
                continue

            # Legacy ESP32 raw commands (backwards compat)
            if "T" in tc and "tool" not in tc:
                self._execute_raw(tc)
                continue
            if "_pause" in tc:
                self._sleep(as_float(tc["_pause"], 0.0))
                continue

            tool = str(tc.get("tool", "")).strip().lower()

            if tool == "drive":
                angle = clamp(as_float(tc.get("angle"), 0), -60, 60)
                dist = clamp(as_float(tc.get("distance"), 0.3), 0.10, 2.0)
                speed = clamp(as_float(tc.get("speed"), 0.15), 0.10, 0.25)
                self._gimbal_center()
                duration = dist / max(speed, 0.05)
                steer = clamp(angle / 60.0, -1.0, 1.0) * speed * 0.8
                self.rover.send({"T": 1, "L": round(speed + steer, 3), "R": round(speed - steer, 3)})
                self._sleep(duration)
                self.rover.stop()

            elif tool == "reverse":
                dist = clamp(as_float(tc.get("distance"), 0.15), 0.10, 0.25)
                speed = 0.12
                duration = dist / speed
                self._gimbal_center()
                self.rover.send({"T": 1, "L": round(-speed, 3), "R": round(-speed, 3)})
                self._sleep(duration)
                self.rover.stop()

            elif tool == "turn_body":
                degrees = clamp(as_float(tc.get("angle"), 0), -180, 180)
                self._gimbal_center()
                self._turn(degrees)

            elif tool == "stop":
                self.rover.stop()

            elif tool in ("gimbal", "look"):
                pan = clamp(as_float(tc.get("pan"), 0), -180, 180)
                tilt = clamp(as_float(tc.get("tilt"), 0), -30, 45)
                self._gimbal_send(pan, tilt)

            elif tool == "wait":
                self._sleep(clamp(as_float(tc.get("seconds"), 0.3), 0.05, 3.0))

            elif tool == "lights":
                base = int(clamp(as_float(tc.get("base"), 0), 0, 255))
                head = int(clamp(as_float(tc.get("head"), 0), 0, 255))
                self.rover.send({"T": 132, "IO4": base, "IO5": head})

            elif tool == "oled":
                line = int(clamp(as_float(tc.get("line"), 0), 0, 3))
                text = str(tc.get("text", ""))[:16]
                self.rover.send({"T": 3, "lineNum": line, "Text": text})

        self.rover.stop()

    def _execute_raw(self, command: dict):
        """Execute a legacy raw ESP32 command."""
        if command.get("T") == 133:
            pc = getattr(self.config, "gimbal_pan_center", 0.0)
            tc = getattr(self.config, "gimbal_tilt_center", 0.0)
            command = dict(command, X=round(command.get("X", 0) + pc, 1),
                           Y=round(command.get("Y", 0) + tc, 1))
            self._last_gimbal_pan = command.get("X", 0) - pc
            self._last_gimbal_tilt = command.get("Y", 0) - tc
        self.rover.send(command)

    def _gimbal_send(self, pan: float, tilt: float):
        pc = getattr(self.config, "gimbal_pan_center", 0.0)
        tc = getattr(self.config, "gimbal_tilt_center", 0.0)
        self.rover.send({"T": 133, "X": round(pan + pc, 1), "Y": round(tilt + tc, 1),
                         "SPD": 300, "ACC": 20})
        self._last_gimbal_pan = pan
        self._last_gimbal_tilt = tilt

    def _gimbal_center(self):
        if abs(self._last_gimbal_pan) > 5 or abs(self._last_gimbal_tilt) > 5:
            self._gimbal_send(0, 0)
            self._sleep(0.3)

    def _turn(self, degrees: float):
        if abs(degrees) < 1:
            return
        sign = 1 if degrees > 0 else -1
        speed = min(TURN_SPEED, 0.24)
        rate = TURN_RATE_DPS * (speed / TURN_SPEED)
        duration = abs(degrees) / max(rate, 60)
        self.rover.send({"T": 1, "L": round(speed * sign, 3), "R": round(-speed * sign, 3)})
        self._sleep(duration)
        self.rover.stop()

    def _observe_scene(self, llm_client) -> str:
        frame = self.camera.snap()
        if frame is None:
            return ""
        try:
            return llm_client.complete(
                prompt=load_prompt("observe_scene.md"),
                system=load_prompt("observe_scene.system.md"),
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
