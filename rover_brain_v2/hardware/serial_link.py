"""Serial wrapper for ESP32 rover control."""

from __future__ import annotations

import json
import threading
import time

import serial


class SerialLink:
    def __init__(self, port: str, baud: int, event_bus, flags):
        self._serial = serial.Serial(port, baud, timeout=0.5)
        self._lock = threading.Lock()
        self._events = event_bus
        self._flags = flags
        time.sleep(0.1)
        self._serial.reset_input_buffer()
        self._events.publish("system", f"Serial opened {port} @ {baud}")

    def send(self, command: dict):
        with self._lock:
            self._send_locked(command)

    def _send_locked(self, command: dict):
        if "_pause" in command:
            return
        cmd = dict(command)
        if cmd.get("T") == 1:
            if self._flags.desk_mode and (cmd.get("L", 0) or cmd.get("R", 0)):
                self._events.publish("system", "Desk mode blocked wheel command")
                return
            cmd = {"T": 1, "L": -cmd.get("R", 0), "R": -cmd.get("L", 0)}
        raw = json.dumps(cmd) + "\n"
        self._serial.write(raw.encode("utf-8"))
        self._serial.readline()
        self._events.publish("serial", json.dumps(cmd))

    def stop(self):
        with self._lock:
            self._send_locked({"T": 1, "L": 0, "R": 0})
            self._send_locked({"T": 135})

    def read_imu(self):
        if not self._lock.acquire(timeout=0.1):
            return None
        try:
            self._serial.reset_input_buffer()
            self._serial.readline()
            line = self._serial.readline().decode("utf-8", errors="replace").strip()
            if not line:
                return None
            payload = json.loads(line)
            if payload.get("T") == 1001:
                return payload
            return None
        except Exception:
            return None
        finally:
            self._lock.release()

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self._serial.close()
        self._events.publish("system", "Serial closed")

