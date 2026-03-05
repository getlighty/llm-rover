#!/usr/bin/env python3
"""Test encoder-guided 90° rotation.

Sends a spin command, reads T:1001 encoder speeds, integrates
differential rotation, stops when target reached.
"""
import json
import math
import serial
import time
import sys

PORT = "/dev/ttyTHS1"
BAUD = 115200
TURN_SPEED = 0.35       # m/s per wheel
TRACK_WIDTH = 0.20      # effective track width (calibrate!)
TARGET_DEG = 90.0       # degrees to turn
TOLERANCE = 5.0
TIMEOUT = 4.0
DIRECTION = "right"     # "left" or "right"

if len(sys.argv) > 1:
    DIRECTION = sys.argv[1]
if len(sys.argv) > 2:
    TARGET_DEG = float(sys.argv[2])

print(f"=== Encoder-guided {DIRECTION} turn: {TARGET_DEG}° ===")
print(f"    TURN_SPEED={TURN_SPEED}, TRACK_WIDTH={TRACK_WIDTH}")

ser = serial.Serial(PORT, BAUD, timeout=0.5)
time.sleep(0.2)
# Drain buffer
while ser.in_waiting:
    ser.readline()


def send(cmd):
    """Send command with motor negation (L=-R, R=-L)."""
    if cmd.get("T") == 1:
        raw_l = -cmd.get("R", 0)
        raw_r = -cmd.get("L", 0)
        raw = {"T": 1, "L": raw_l, "R": raw_r}
    else:
        raw = cmd
    ser.write((json.dumps(raw) + "\n").encode())


def read_t1001():
    """Read one T:1001 line, return dict or None."""
    for _ in range(5):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("T") == 1001:
                return d
        except json.JSONDecodeError:
            continue
    return None


# Start spinning
if DIRECTION == "right":
    send({"T": 1, "L": TURN_SPEED, "R": -TURN_SPEED})
else:
    send({"T": 1, "L": -TURN_SPEED, "R": TURN_SPEED})

print(f"Wheels spinning {DIRECTION}...")

accumulated_rad = 0.0
last_time = None
deadline = time.time() + TIMEOUT
timed_out = False
sample_count = 0

while time.time() < deadline:
    data = read_t1001()
    if data is None:
        time.sleep(0.05)
        continue

    now = time.time()
    if last_time is None:
        last_time = now
        v_l = float(data.get("L", 0))
        v_r = float(data.get("R", 0))
        print(f"  First reading: L={v_l:.3f} R={v_r:.3f} m/s")
        time.sleep(0.05)
        continue

    dt = now - last_time
    last_time = now

    v_l = float(data.get("L", 0))
    v_r = float(data.get("R", 0))

    # omega = (R - L) / track: positive = CCW (left)
    omega_rad = (v_r - v_l) / TRACK_WIDTH
    accumulated_rad += omega_rad * dt
    accumulated_deg = math.degrees(accumulated_rad)
    accumulated_abs = abs(accumulated_deg)

    sample_count += 1
    if sample_count % 3 == 0 or accumulated_abs >= TARGET_DEG - TOLERANCE:
        print(f"  L={v_l:+.3f} R={v_r:+.3f} omega={math.degrees(omega_rad):+.1f}°/s "
              f"accumulated={accumulated_deg:+.1f}° (target={TARGET_DEG}°)")

    if accumulated_abs >= TARGET_DEG - TOLERANCE:
        break

    if accumulated_abs > TARGET_DEG + 30:
        print("  OVERSHOOT!")
        break

    time.sleep(0.05)
else:
    timed_out = True

# Stop
send({"T": 1, "L": 0, "R": 0})

actual_deg = math.degrees(accumulated_rad)
label = "TIMEOUT" if timed_out else "OK"
print(f"\n=== Result: {actual_deg:+.1f}° ({label}) ===")
print(f"    Target: {TARGET_DEG}°, Samples: {sample_count}")

ser.close()
