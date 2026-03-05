#!/usr/bin/env python3
"""test_mag_spin.py — Test 90° magnetometer-guided body rotation.

Stops wheels, reads IMU baseline, spins right 90°, reports actual vs expected.
"""

import json
import math
import serial
import time
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imu import IMUPoller

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200
TURN_SPEED = 0.35
WHEELBASE = 0.25

# Mag spin params (same as rover_brain.py)
MAG_SPIN_TOLERANCE_DEG = 5.0
MAG_SPIN_OVERSHOOT_DEG = 30.0
MAG_SPIN_TIMEOUT_S = 4.0
MAG_SPIN_POLL_HZ = 10


def _angle_diff(to_deg, from_deg):
    d = (to_deg - from_deg) % 360
    if d > 180:
        d -= 360
    return d


class SimpleSerial:
    """Minimal serial wrapper with read_imu() for IMUPoller."""

    def __init__(self, port, baud):
        self._ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        self._lock = threading.Lock()

    def send(self, cmd):
        # Motors wired in reverse AND L/R swapped (same as RoverSerial)
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("R", 0), R=-cmd.get("L", 0))
        with self._lock:
            self._ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
            # Read response (may be feedback or T:1001)
            self._ser.readline()

    def read_imu(self):
        with self._lock:
            for _ in range(5):
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("T") == 1001:
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    def close(self):
        self._ser.close()


def mag_spin(ser, imu, target_delta_deg):
    """Run a mag-guided spin. Returns (actual_deg, timed_out) or None."""
    if not imu.state.fresh:
        print("  ERROR: IMU data not fresh")
        return None

    start_heading = imu.heading_deg
    target_heading = (start_heading + target_delta_deg) % 360
    deadline = time.time() + MAG_SPIN_TIMEOUT_S
    poll_interval = 1.0 / MAG_SPIN_POLL_HZ
    timed_out = False

    print(f"  Start heading: {start_heading:.1f}°")
    print(f"  Target heading: {target_heading:.1f}° (delta={target_delta_deg:+.0f}°)")

    responsiveness_checked = False
    spin_t0 = time.time()

    while time.time() < deadline:
        current = imu.heading_deg
        actual_so_far = _angle_diff(current, start_heading)
        remaining = _angle_diff(target_heading, current)
        elapsed = time.time() - spin_t0

        print(f"  [{elapsed:.1f}s] heading={current:.1f}°, actual={actual_so_far:+.1f}°, remaining={remaining:+.1f}°",
              end="\r")

        # Responsiveness check: if heading hasn't moved after 0.5s, bail
        if not responsiveness_checked and elapsed >= 0.5:
            responsiveness_checked = True
            if abs(actual_so_far) < MAG_SPIN_TOLERANCE_DEG:
                print(f"\n  UNRESPONSIVE: heading moved only {actual_so_far:+.1f}° in 0.5s")
                print(f"  Falling back to timed spin for remaining {max(0, target_delta_deg / 80):.1f}s...")
                # Don't stop wheels — let caller handle timed fallback
                remaining_time = abs(target_delta_deg) / 80  # rough estimate
                remaining_time = max(0, remaining_time - elapsed)
                time.sleep(remaining_time)
                ser.send({"T": 1, "L": 0, "R": 0})
                time.sleep(0.2)
                actual_delta = _angle_diff(imu.heading_deg, start_heading)
                return actual_delta, True, start_heading, imu.heading_deg

        if abs(remaining) <= MAG_SPIN_TOLERANCE_DEG:
            print()
            break

        if target_delta_deg > 0:
            overshoot = actual_so_far - target_delta_deg
        else:
            overshoot = target_delta_deg - actual_so_far
        if overshoot > MAG_SPIN_OVERSHOOT_DEG:
            print(f"\n  OVERSHOOT: actual={actual_so_far:.1f}°")
            break

        time.sleep(poll_interval)
    else:
        timed_out = True
        print()

    # Stop wheels
    ser.send({"T": 1, "L": 0, "R": 0})
    time.sleep(0.2)

    actual_delta = _angle_diff(imu.heading_deg, start_heading)
    end_heading = imu.heading_deg
    label = "TIMEOUT" if timed_out else "OK"
    return actual_delta, timed_out, start_heading, end_heading


def main():
    target = 90.0  # degrees, positive = right
    if len(sys.argv) > 1:
        target = float(sys.argv[1])

    print(f"=== Magnetometer-Guided Spin Test: {target:+.0f}° ===\n")

    print("Connecting to ESP32...")
    ser = SimpleSerial(SERIAL_PORT, SERIAL_BAUD)

    # Stop any existing motion
    ser.send({"T": 1, "L": 0, "R": 0})
    time.sleep(0.3)

    print("Starting IMU poller...")
    imu = IMUPoller(ser, log_fn=lambda cat, msg: print(f"  [{cat}] {msg}"))
    imu.start()

    # Warm up IMU — need a few reads
    print("Warming up IMU (1s)...")
    imu.wheels_active.set()  # force polling
    time.sleep(1.0)

    if not imu.state.fresh:
        print("ERROR: No IMU data after 1s warmup")
        ser.close()
        return

    imu.set_heading_offset()
    print(f"Boot heading offset: {imu._heading_offset:.1f}° raw")
    print(f"Current heading: {imu.heading_deg:.1f}° (should be ~0°)\n")

    # Start spin
    sign = 1 if target > 0 else -1  # positive = right (L+, R-)
    print(f"Sending spin command: L={TURN_SPEED * sign:.2f}, R={-TURN_SPEED * sign:.2f}")
    ser.send({"T": 1, "L": TURN_SPEED * sign, "R": -TURN_SPEED * sign})
    time.sleep(0.05)  # let wheels engage

    print("Running mag_spin()...\n")
    result = mag_spin(ser, imu, target)

    if result is None:
        print("\nFAILED: No IMU data")
    else:
        actual, timed_out, start_h, end_h = result
        status = "TIMEOUT" if timed_out else "OK"
        drift = actual - target
        print(f"\n{'='*50}")
        print(f"  Target:    {target:+.1f}°")
        print(f"  Actual:    {actual:+.1f}°")
        print(f"  Drift:     {drift:+.1f}°")
        print(f"  Heading:   {start_h:.1f}° → {end_h:.1f}°")
        print(f"  Status:    {status}")
        print(f"{'='*50}")

    # Cleanup
    imu.stop()
    ser.send({"T": 1, "L": 0, "R": 0})
    ser.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
