"""Main runtime for rover_brain_v2."""

from __future__ import annotations

import argparse
import os
import re
import threading
import time

from imu import IMUPoller
from local_detector import DepthEstimator, LocalDetector
from place_recognition import PlaceDB

from rover_brain_v2.command_mode import GeneralCommandController
from rover_brain_v2.config import load_config, save_calibration
from rover_brain_v2.events import EventBus
from rover_brain_v2.follow.controller import FollowMeController
from rover_brain_v2.hardware.audio_io import AudioIO
from rover_brain_v2.hardware.camera import CameraPipeline
from rover_brain_v2.hardware.serial_link import SerialLink
from rover_brain_v2.models import FollowRequest, NavigationRequest, RuntimeFlags
from rover_brain_v2.navigation.navigator import DepthVectorNavigator
from rover_brain_v2.navigation.orchestrator import GraphNavigationOrchestrator
from rover_brain_v2.providers.registry import ProviderRegistry


FOLLOW_RE = re.compile(r"^\s*follow(?: me| the)?(?: (?P<target>.+))?\s*$", re.IGNORECASE)
STOP_RE = re.compile(r"^\s*(stop|cancel|abort|halt|freeze|hold position)\s*$", re.IGNORECASE)
ACK_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|thanks?|thank you|got it|all right|alright|sure|yes|yep|mm+|uh huh)[\s.!?,;:]*$",
    re.IGNORECASE,
)
PUNCT_RE = re.compile(r"^[\s.!?,;:]+$")
NAV_RE = re.compile(
    r"\b(go to|navigate to|take me to|find|search for|search|look for|head to|bring me to)\b",
    re.IGNORECASE,
)


class RoverBrainV2:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.events = EventBus()
        self.flags = RuntimeFlags()
        self.providers = ProviderRegistry()
        self.audio = AudioIO()
        self.audio.discover()
        self.rover = SerialLink(
            self.config.serial_port,
            self.config.serial_baud,
            self.events,
            self.flags,
        )
        self.camera = CameraPipeline(self.config.camera_source, self.events)
        self.imu = None
        try:
            self.imu = IMUPoller(self.rover, log_fn=self.events.publish)
            self.imu.start()
            self.events.publish("system", "IMU poller active")
        except Exception as exc:
            self.events.publish("system", f"IMU unavailable: {exc}")
        try:
            self.camera.detector = LocalDetector()
            self.events.publish("system", "YOLO detector loaded for follow mode")
        except Exception as exc:
            self.events.publish("error", f"Detector unavailable: {exc}")
        try:
            self.camera.depth_estimator = DepthEstimator()
            self.events.publish("system", "DepthAnything estimator loaded")
        except Exception as exc:
            self.events.publish("error", f"Depth estimator unavailable: {exc}")
        # Center gimbal on startup
        pc = self.config.gimbal_pan_center
        tc = self.config.gimbal_tilt_center
        self.rover.send({"T": 133, "X": round(pc, 1), "Y": round(tc, 1), "SPD": 200, "ACC": 20})
        self.events.publish("system", f"Gimbal centered to ({pc}, {tc})")

        self.command_controller = GeneralCommandController(
            rover=self.rover,
            camera=self.camera,
            event_bus=self.events,
            flags=self.flags,
            config=self.config,
            speak_fn=self.speak,
        )
        bundle = self.providers.bundle()
        self.place_db = PlaceDB(
            db_path=os.path.join(os.path.dirname(__file__), "..", "place_signatures.pkl"),
        )
        self.place_db.load()
        self.navigator = DepthVectorNavigator(
            rover=self.rover,
            camera=self.camera,
            llm_client=bundle.navigator_llm,
            event_bus=self.events,
            flags=self.flags,
            config=self.config,
            speak_fn=self.speak,
            listen_fn=self._listen_once,
            place_db=self.place_db,
        )
        self.orchestrator = GraphNavigationOrchestrator(
            llm_client=bundle.orchestrator_llm,
            navigator=self.navigator,
            event_bus=self.events,
        )
        self.navigator.topo = self.orchestrator.topo
        self.follow = FollowMeController(
            rover=self.rover,
            camera=self.camera,
            event_bus=self.events,
            config=self.config,
            llm_client=bundle.command_llm,
            speak_fn=self.speak,
            flags=self.flags,
            imu=self.imu,
        )
        self._shutdown = threading.Event()
        self._task_lock = threading.Lock()
        self._active_thread: threading.Thread | None = None
        self._active_mode = "idle"
        self._active_cancel = None
        self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self._voice_thread.start()
        self._apply_detector_policy("idle")

    def refresh_provider_bindings(self):
        bundle = self.providers.bundle()
        self.navigator.llm = bundle.navigator_llm
        self.orchestrator.llm = bundle.orchestrator_llm
        self.follow.llm = bundle.command_llm
        self.events.publish(
            "system",
            "Providers updated: "
            f"cmd={self.providers.selection.command_llm}, "
            f"nav={self.providers.selection.navigator_llm}, "
            f"orch={self.providers.selection.orchestrator_llm}, "
            f"tts={self.providers.selection.tts}",
        )

    def speak(self, text: str):
        if not text or not self.flags.tts_enabled:
            return
        bundle = self.providers.bundle()
        if not self.audio.ready:
            self.events.publish("speak", text)
            return
        self.audio.speak(text, bundle.tts, log_fn=self.events.publish)

    def _listen_once(self, timeout_s: float = 15.0) -> str | None:
        """Record one utterance and transcribe it. Returns text or None."""
        if not self.audio.ready:
            return None
        abort = threading.Event()
        timer = threading.Timer(timeout_s, abort.set)
        timer.start()
        try:
            audio_data = self.audio.listen(abort_event=abort)
            if audio_data is None:
                return None
            text = self.providers.bundle().stt.transcribe(audio_data)
            return text.strip() if text else None
        except Exception as exc:
            self.events.publish("error", f"listen_once failed: {exc}")
            return None
        finally:
            timer.cancel()

    def handle_text_command(self, text: str, *, from_voice: bool = False):
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty command"}
        if self.flags.killed and not STOP_RE.match(text):
            self.events.publish("system", f"Ignoring command while killed: {text}")
            return {"ok": False, "error": "killed"}
        with self._task_lock:
            active_mode = self._active_mode
        if active_mode != "idle" and (ACK_RE.match(text) or PUNCT_RE.match(text)):
            self.events.publish("heard", text)
            self.events.publish("system", f"Ignoring filler while {active_mode} is active")
            return {"ok": True, "mode": active_mode, "ignored": True}
        self.events.publish("heard", text)
        # Reserved quick commands — bypass LLM
        if STOP_RE.match(text):
            self.cancel_active_task()
            return {"ok": True, "mode": "stop"}
        follow_match = FOLLOW_RE.match(text)
        if follow_match:
            target = (follow_match.group("target") or "person").strip()
            if target.lower() == "me":
                target = "person"
            self.start_follow(FollowRequest(target=target, duration=60.0, target_bw=self.config.follow_target_bw))
            return {"ok": True, "mode": "follow", "target": target}
        # Everything else goes through the LLM
        self.start_general_command(text, speak_allowed=True)
        return {"ok": True, "mode": "general"}

    def start_follow(self, request: FollowRequest):
        def runner():
            self._apply_detector_policy("follow")
            result = self.follow.follow(
                target=request.target,
                duration=request.duration,
                target_bw=request.target_bw,
            )
            self.events.publish("follow", f"Follow finished: {result}")
            return result
        self._spawn_task("follow", runner, self.follow.cancel)

    def start_navigation(self, request: NavigationRequest):
        def runner():
            self._apply_detector_policy("navigation")
            if request.topological:
                result = self.orchestrator.run_navigation_task(request.target, topological=True)
                self.events.publish("plan", f"Navigation finished: {result.status} | {result.summary}")
                return result.reached
            result = self.navigator.run_reactive_task(request.target)
            self.events.publish("plan", f"Navigation finished: {result.status} | {result.summary}")
            return result.reached
        cancel_fn = self.orchestrator.cancel if request.topological else self.navigator.cancel
        self._spawn_task("navigation", runner, cancel_fn)

    def start_general_command(self, text: str, *, speak_allowed: bool = True):
        def runner():
            self._apply_detector_policy("general")
            bundle = self.providers.bundle()
            result = self.command_controller.run(text, bundle.command_llm, speak_allowed=speak_allowed)
            # If the LLM decided this is a navigation request, hand off
            if result.status == "navigate" and result.payload:
                nav_target = str(result.payload.get("navigate", text)).strip()
                self.events.publish("plan", f"LLM routed to navigation: {nav_target}")
                self._apply_detector_policy("navigation")
                nav_result = self.orchestrator.run_navigation_task(nav_target, topological=True)
                self.events.publish("plan", f"Navigation finished: {nav_result.status} | {nav_result.summary}")
                return nav_result.reached
            self.events.publish("plan", f"General command finished: {result.status}")
            return result
        self._spawn_task("general", runner, self.command_controller.cancel)

    def cancel_active_task(self):
        with self._task_lock:
            cancel = self._active_cancel
            mode = self._active_mode
        if cancel is not None:
            self.events.publish("system", f"Cancelling active {mode} task")
            try:
                cancel()
            except Exception as exc:
                self.events.publish("error", f"Task cancel failed: {exc}")
        self.rover.stop()
        self._apply_detector_policy("idle")
        return {"ok": True, "mode": "idle"}

    def set_killed(self, engage: bool):
        self.flags.killed = bool(engage)
        if self.flags.killed:
            self.flags.stt_enabled = False
            self.cancel_active_task()
            self.events.publish("system", "Kill switch engaged")
        else:
            self.flags.stt_enabled = True
            self.events.publish("system", "Kill switch released")
        return self.flags.killed

    def update_flags(self, **changes):
        for key, value in changes.items():
            if hasattr(self.flags, key):
                setattr(self.flags, key, value)
        if "yolo_overlay_enabled" in changes:
            self._apply_detector_policy(self._active_mode)
        self.events.publish("system", f"Flags updated: {changes}")
        return self.flags

    def set_providers(self, **changes):
        snapshot = self.providers.set_selection(**changes)
        self.refresh_provider_bindings()
        return snapshot

    def direct_teleop(self, action: str):
        action = action.strip().lower()
        wheel = {
            "forward": {"T": 1, "L": 0.14, "R": 0.14},
            "back": {"T": 1, "L": -0.12, "R": -0.12},
            "left": {"T": 1, "L": -0.20, "R": 0.20},
            "right": {"T": 1, "L": 0.20, "R": -0.20},
            "stop": {"T": 1, "L": 0.0, "R": 0.0},
        }
        lights = {
            "lights_on": {"T": 132, "IO4": 180, "IO5": 180},
            "lights_off": {"T": 132, "IO4": 0, "IO5": 0},
        }
        command = wheel.get(action) or lights.get(action)
        if command is None:
            return {"ok": False, "error": f"unknown action {action}"}
        self.rover.send(command)
        if action != "stop" and command.get("T") == 1:
            time.sleep(0.25)
            self.rover.stop()
        return {"ok": True, "action": action}

    def move_gimbal(self, pan: float, tilt: float):
        pc = self.config.gimbal_pan_center
        tc = self.config.gimbal_tilt_center
        self.rover.send({"T": 133, "X": round(float(pan) + pc, 1), "Y": round(float(tilt) + tc, 1), "SPD": 250, "ACC": 20})
        return {"ok": True, "pan": pan, "tilt": tilt}

    def test_turn(self, degrees: float = 90.0):
        """Execute a direct turn for calibration testing. No LLM involved."""
        self.events.publish("system", f"Calibration test turn: {degrees:+.0f}°")
        self.navigator._turn(degrees)
        self.events.publish("system", f"Calibration test turn complete")
        return {"ok": True, "degrees": degrees}

    def set_calibration(self, **values):
        result = {"ok": True}
        if "gimbal_pan_center" in values:
            self.config.gimbal_pan_center = float(values["gimbal_pan_center"])
            result["gimbal_pan_center"] = self.config.gimbal_pan_center
        if "gimbal_tilt_center" in values:
            self.config.gimbal_tilt_center = float(values["gimbal_tilt_center"])
            result["gimbal_tilt_center"] = self.config.gimbal_tilt_center
        if "turn_rate_dps" in values:
            self.config.turn_rate_dps = float(values["turn_rate_dps"])
            result["turn_rate_dps"] = self.config.turn_rate_dps
        save_calibration(**{k: v for k, v in values.items()
                           if k in ("gimbal_pan_center", "gimbal_tilt_center", "turn_rate_dps")})
        # Move gimbal to new center
        pc = self.config.gimbal_pan_center
        tc = self.config.gimbal_tilt_center
        self.rover.send({"T": 133, "X": round(pc, 1), "Y": round(tc, 1), "SPD": 200, "ACC": 20})
        self.events.publish("system", f"Calibration saved: {result}")
        return result

    def status(self):
        current_room = getattr(self.orchestrator.topo, "current_room", None)
        return {
            "flags": {
                "desk_mode": self.flags.desk_mode,
                "stt_enabled": self.flags.stt_enabled,
                "tts_enabled": self.flags.tts_enabled,
                "gimbal_pan_enabled": self.flags.gimbal_pan_enabled,
                "yolo_overlay_enabled": self.flags.yolo_overlay_enabled,
                "killed": self.flags.killed,
            },
            "providers": {
                "current": self.providers.snapshot(),
                "available": self.providers.available(),
            },
            "audio_ready": self.audio.ready,
            "active_mode": self._active_mode,
            "detector_mode": self._detector_mode(),
            "current_room": current_room,
            "known_rooms": [room.id for room in self.orchestrator.topo.rooms()],
            "calibration": {
                "gimbal_pan_center": self.config.gimbal_pan_center,
                "gimbal_tilt_center": self.config.gimbal_tilt_center,
                "turn_rate_dps": self.config.turn_rate_dps,
            },
        }

    def landmarks(self):
        imu_data = self.imu.get_map_data() if self.imu is not None else None
        return {
            "imu": imu_data,
            "current_room": getattr(self.orchestrator.topo, "current_room", None),
            "active_mode": self._active_mode,
        }

    def shutdown(self):
        self._shutdown.set()
        self.cancel_active_task()
        try:
            self.orchestrator.shutdown()
        except Exception:
            pass
        try:
            self.navigator.shutdown()
        except Exception:
            pass
        if self.imu is not None:
            try:
                self.imu.stop()
            except Exception:
                pass
        try:
            self.place_db.save()
        except Exception:
            pass
        self.camera.close()
        self.rover.close()

    def _looks_like_navigation(self, text: str) -> bool:
        if NAV_RE.search(text):
            return True
        lower = text.lower()
        return any(room.id.replace("_", " ") in lower for room in self.orchestrator.topo.rooms())

    def _spawn_task(self, mode: str, target, cancel_fn):
        self.cancel_active_task()

        def wrapped():
            try:
                target()
            except Exception as exc:
                self.events.publish("error", f"{mode} task crashed: {exc}")
            finally:
                with self._task_lock:
                    self._active_mode = "idle"
                    self._active_thread = None
                    self._active_cancel = None
                self._apply_detector_policy("idle")

        thread = threading.Thread(target=wrapped, daemon=True, name=f"rover-v2-{mode}")
        with self._task_lock:
            self._active_mode = mode
            self._active_thread = thread
            self._active_cancel = cancel_fn
        thread.start()

    def _detector_mode(self) -> str:
        with self._task_lock:
            mode = self._active_mode
        if mode == "follow":
            return "enabled_for_follow"
        if mode == "navigation":
            return "disabled_for_navigation"
        return "enabled" if self.flags.yolo_overlay_enabled else "disabled"

    def _apply_detector_policy(self, mode: str):
        if self.camera.detector is None:
            return
        if mode == "follow":
            enabled = True
            reason = "follow mode"
        elif mode == "navigation":
            enabled = True
            reason = "navigation mode"
        else:
            enabled = bool(self.flags.yolo_overlay_enabled)
            reason = f"{mode or 'idle'} mode"
        self.camera.set_detector_enabled(enabled)
        self.events.publish(
            "system",
            f"Detector {'ENABLED' if enabled else 'DISABLED'} for {reason}",
        )

    def _voice_loop(self):
        while not self._shutdown.is_set():
            if self.flags.killed or not self.flags.stt_enabled or not self.audio.ready:
                time.sleep(0.2)
                continue
            audio_data = self.audio.listen(abort_event=self._shutdown)
            if audio_data is None:
                continue
            try:
                text = self.providers.bundle().stt.transcribe(audio_data)
            except Exception as exc:
                self.events.publish("error", f"STT failed: {exc}")
                continue
            if text:
                with self._task_lock:
                    active_mode = self._active_mode
                if active_mode != "idle" and not STOP_RE.match(text) and not ACK_RE.match(text) and not PUNCT_RE.match(text):
                    self.events.publish("heard", text)
                    self.events.publish("system", f"Ignoring voice input while {active_mode} is active")
                    continue
                self.handle_text_command(text, from_voice=True)


def main():
    parser = argparse.ArgumentParser(description="Run rover_brain_v2")
    parser.add_argument("--web-port", type=int, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--serial-port", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.web_port is not None:
        config.web_port = args.web_port
    if args.camera is not None:
        config.camera_source = args.camera
    if args.serial_port is not None:
        config.serial_port = args.serial_port

    brain = RoverBrainV2(config=config)
    from rover_brain_v2.web.server import RoverWebServer

    server = RoverWebServer(brain, host=config.web_host, port=config.web_port)
    server.start()
    brain.events.publish("system", f"rover_brain_v2 web UI on http://{config.web_host}:{config.web_port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        brain.shutdown()
