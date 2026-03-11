"""Bluetooth speaker pairing and management."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

_PERSIST_PATH = Path(__file__).resolve().parents[1] / "data" / "bt_speaker.json"


def _run(cmd: str, timeout: int = 10) -> str:
    """Run a bluetoothctl command."""
    try:
        result = subprocess.run(
            ["bluetoothctl", *cmd.split()],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def _run_expect(commands: list[str], timeout: int = 15) -> str:
    """Run multiple bluetoothctl commands via stdin."""
    script = "\n".join(commands) + "\nquit\n"
    try:
        result = subprocess.run(
            ["bluetoothctl"],
            input=script, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def scan(duration: int = 8) -> list[dict]:
    """Scan for nearby Bluetooth devices. Returns list of {mac, name}."""
    try:
        proc = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        proc.stdin.write("power on\n")
        proc.stdin.write("scan on\n")
        proc.stdin.flush()
        time.sleep(duration)
        proc.stdin.write("devices\n")
        proc.stdin.flush()
        time.sleep(1)
        proc.stdin.write("scan off\n")
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        output, _ = proc.communicate(timeout=5)
    except Exception as e:
        return []
    seen = set()
    devices = []
    for line in output.splitlines():
        if "Device " not in line or ":" not in line:
            continue
        parts = line.split("Device ", 1)
        if len(parts) < 2:
            continue
        rest = parts[1].strip()
        mac = rest[:17]
        name = rest[18:].strip() if len(rest) > 17 else ""
        if not name or name == "(null)" or mac in seen:
            continue
        seen.add(mac)
        devices.append({"mac": mac, "name": name})
    return devices


def paired_devices() -> list[dict]:
    """List already paired devices."""
    output = _run("paired-devices")
    devices = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Device "):
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                devices.append({"mac": parts[1], "name": parts[2]})
    return devices


def pair_and_connect(mac: str) -> dict:
    """Pair with a device and connect. Scans first to make device available."""
    try:
        proc = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        # Scan to discover the device first
        for cmd in ["power on", "pairable on", "agent on", "default-agent", "scan on"]:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
            time.sleep(0.5)
        # Wait for device to appear in scan
        time.sleep(5)
        # Now pair, trust, connect
        for cmd in [f"pair {mac}"]:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        time.sleep(5)
        for cmd in [f"trust {mac}", f"connect {mac}"]:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
            time.sleep(3)
        proc.stdin.write("scan off\nquit\n")
        proc.stdin.flush()
        output, _ = proc.communicate(timeout=10)
    except Exception as e:
        output = str(e)
    time.sleep(2)
    connected = is_connected(mac)
    if connected:
        save_preferred(mac)
    return {"mac": mac, "connected": connected, "output": output[-500:]}


def connect(mac: str) -> dict:
    """Connect to an already paired device."""
    try:
        proc = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        proc.stdin.write("power on\n")
        proc.stdin.flush()
        time.sleep(1)
        proc.stdin.write(f"connect {mac}\n")
        proc.stdin.flush()
        time.sleep(5)
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        output, _ = proc.communicate(timeout=10)
    except Exception as e:
        output = str(e)
    connected = is_connected(mac)
    return {"mac": mac, "connected": connected, "output": output[-300:]}


def disconnect(mac: str) -> dict:
    """Disconnect a device."""
    output = _run_expect([f"disconnect {mac}"], timeout=10)
    return {"mac": mac, "connected": False, "output": output[-200:]}


def remove(mac: str) -> dict:
    """Remove (unpair) a device."""
    _run_expect([f"disconnect {mac}"], timeout=5)
    output = _run_expect([f"remove {mac}"], timeout=10)
    prefs = load_preferred()
    if prefs.get("mac") == mac:
        save_preferred("")
    return {"mac": mac, "removed": True, "output": output[-200:]}


def is_connected(mac: str) -> bool:
    """Check if a specific device is connected."""
    output = _run(f"info {mac}")
    for line in output.splitlines():
        if "Connected:" in line and "yes" in line.lower():
            return True
    return False


def get_bt_audio_sink() -> str | None:
    """Find the PulseAudio sink for a connected Bluetooth device."""
    try:
        out = subprocess.check_output(["pactl", "list", "sinks", "short"], text=True, timeout=5)
        for line in out.splitlines():
            if "bluez" in line.lower() or "bluetooth" in line.lower():
                return line.split("\t")[1] if "\t" in line else line.split()[1]
    except Exception:
        pass
    return None


def save_preferred(mac: str):
    """Save preferred BT speaker MAC for auto-reconnect."""
    _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PERSIST_PATH.write_text(json.dumps({"mac": mac}, indent=2), encoding="utf-8")


def load_preferred() -> dict:
    """Load preferred BT speaker MAC."""
    try:
        return json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def auto_reconnect() -> bool:
    """Try to reconnect to the preferred BT speaker. Returns True if connected."""
    prefs = load_preferred()
    mac = prefs.get("mac", "").strip()
    if not mac:
        return False
    if is_connected(mac):
        return True
    result = connect(mac)
    return result.get("connected", False)
