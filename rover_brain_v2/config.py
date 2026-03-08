"""Configuration helpers for rover_brain_v2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


V2_DIR = Path(__file__).resolve().parent
ROVER_DIR = V2_DIR.parent
ENV_FILE = ROVER_DIR / ".env"


def load_env_defaults() -> None:
    """Populate os.environ from the repo .env file when available."""
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def ollama_chat_url() -> str:
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").strip()
    if base_url.endswith("/api/chat"):
        base_url = base_url[:-9]
    return base_url.rstrip("/") + "/api/chat"


@dataclass(slots=True)
class AppConfig:
    serial_port: str = "/dev/ttyTHS1"
    serial_baud: int = 115200
    camera_source: int = 0
    web_host: str = "0.0.0.0"
    web_port: int = 8765
    command_observe_rounds: int = 6
    command_drive_speed: float = 0.15
    navigation_waypoint_budget: int = 40
    navigation_drive_speed: float = 0.15
    navigation_turn_speed: float = 0.24
    navigation_turn_step_s: float = 0.04
    navigation_zone_repeat_threshold: int = 3
    navigation_zone_exit_min_clearance_m: float = 0.50
    follow_target_bw: float = 0.25
    follow_loop_hz: float = 10.0
    depth_guard_stop_m: float = 0.25
    depth_guard_turn_stop_m: float = 0.18
    gimbal_pan_center: float = 0.0
    gimbal_tilt_center: float = 0.0
    turn_rate_dps: float = 200.0


def load_config() -> AppConfig:
    load_env_defaults()
    cfg = AppConfig(
        serial_port=os.environ.get("ROVER_SERIAL_PORT", "/dev/ttyTHS1"),
        serial_baud=int(os.environ.get("ROVER_SERIAL_BAUD", "115200")),
        camera_source=int(os.environ.get("ROVER_CAMERA", "0")),
        web_host=os.environ.get("ROVER_V2_WEB_HOST", "0.0.0.0"),
        web_port=int(os.environ.get("ROVER_V2_WEB_PORT", "8765")),
        command_observe_rounds=int(os.environ.get("ROVER_V2_OBSERVE_ROUNDS", "6")),
        command_drive_speed=float(os.environ.get("ROVER_V2_COMMAND_SPEED", "0.15")),
        navigation_waypoint_budget=int(os.environ.get("ROVER_V2_NAV_BUDGET", "40")),
        navigation_drive_speed=float(os.environ.get("ROVER_V2_NAV_SPEED", "0.15")),
        navigation_turn_speed=float(os.environ.get("ROVER_V2_NAV_TURN_SPEED", "0.24")),
        navigation_turn_step_s=float(os.environ.get("ROVER_V2_NAV_TURN_STEP_S", "0.04")),
        navigation_zone_repeat_threshold=int(os.environ.get("ROVER_V2_NAV_ZONE_REPEAT_THRESHOLD", "3")),
        navigation_zone_exit_min_clearance_m=float(os.environ.get("ROVER_V2_NAV_ZONE_EXIT_CLEARANCE_M", "0.50")),
        follow_target_bw=float(os.environ.get("ROVER_V2_FOLLOW_BW", "0.25")),
        follow_loop_hz=float(os.environ.get("ROVER_V2_FOLLOW_HZ", "10.0")),
        depth_guard_stop_m=float(os.environ.get("ROVER_V2_DEPTH_STOP_M", "0.25")),
        depth_guard_turn_stop_m=float(os.environ.get("ROVER_V2_DEPTH_TURN_STOP_M", "0.18")),
        gimbal_pan_center=float(os.environ.get("ROVER_V2_GIMBAL_PAN_CENTER", "0.0")),
        gimbal_tilt_center=float(os.environ.get("ROVER_V2_GIMBAL_TILT_CENTER", "0.0")),
    )
    _load_calibration(cfg)
    return cfg


_CALIBRATION_PATH = V2_DIR / "data" / "calibration.json"


def _load_calibration(cfg: AppConfig):
    try:
        data = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
        cfg.gimbal_pan_center = float(data.get("gimbal_pan_center", cfg.gimbal_pan_center))
        cfg.gimbal_tilt_center = float(data.get("gimbal_tilt_center", cfg.gimbal_tilt_center))
        cfg.turn_rate_dps = float(data.get("turn_rate_dps", cfg.turn_rate_dps))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass


def save_calibration(**values):
    """Save calibration values. Merges with existing file."""
    data = {}
    try:
        data = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    data.update(values)
    _CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CALIBRATION_PATH.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
