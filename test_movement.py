#!/usr/bin/env python3
"""Quick movement test: drive very slowly forward, then back. Verify basic control."""

import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

import json
import time
import serial

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200

def send(ser, cmd):
    # Motor wires are reversed, negate wheel speeds
    if isinstance(cmd, dict) and cmd.get("T") == 1:
        cmd = dict(cmd, L=-cmd.get("L", 0), R=-cmd.get("R", 0))
    ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
    time.sleep(0.02)
    try:
        return ser.readline().decode("utf-8", errors="ignore").strip()
    except:
        return ""

def main():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(0.2)

    # Check battery
    send(ser, {"T": 130})
    time.sleep(0.3)
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"[test] Feedback: {line}")

    # Center gimbal
    send(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(0.5)

    speed = 0.12  # very slow

    print(f"[test] Driving forward at {speed} m/s for 1.5s (~18cm)...")
    send(ser, {"T": 1, "L": speed, "R": speed})
    time.sleep(1.5)
    send(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(1.0)

    print(f"[test] Driving backward at {speed} m/s for 1.5s (~18cm)...")
    send(ser, {"T": 1, "L": -speed, "R": -speed})
    time.sleep(1.5)
    send(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(0.5)

    print(f"[test] Turning right ~45°...")
    send(ser, {"T": 1, "L": 0.15, "R": -0.15})
    time.sleep(0.4)
    send(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(0.5)

    print(f"[test] Turning left ~45° (back)...")
    send(ser, {"T": 1, "L": -0.15, "R": 0.15})
    time.sleep(0.4)
    send(ser, {"T": 1, "L": 0, "R": 0})

    print("[test] Movement test complete")
    ser.close()

if __name__ == "__main__":
    main()
