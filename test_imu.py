#!/usr/bin/env python3
"""test_imu.py — Validate IMU data from ESP32 T:1001 feedback stream.

The ESP32 continuously streams T:1001 JSON with accel, gyro, magnetometer,
odometry, voltage, and gimbal pan/tilt.  This script reads and prints
parsed data to establish baselines for stuck detection thresholds.

Usage:
    python3 test_imu.py              # 20 readings
    python3 test_imu.py --count 50   # 50 readings
"""

import argparse
import json
import math
import serial
import time

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200


def read_t1001(ser):
    """Read one clean T:1001 line from the stream."""
    for _ in range(5):  # up to 5 attempts
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("T") == 1001:
                return data
        except json.JSONDecodeError:
            continue  # partial line, skip
    return None


def print_imu(data, idx):
    """Pretty-print one T:1001 reading."""
    ax, ay, az = data.get("ax", 0), data.get("ay", 0), data.get("az", 0)
    gx, gy, gz = data.get("gx", 0), data.get("gy", 0), data.get("gz", 0)
    mx, my, mz = data.get("mx", 0), data.get("my", 0), data.get("mz", 0)

    accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)
    heading = math.degrees(math.atan2(my, mx)) % 360

    print(f"[{idx:3d}] Accel: {ax:8.5f} {ay:8.5f} {az:8.5f} (mag={accel_mag:.4f})  |  "
          f"Gyro: {gx:6.2f} {gy:6.2f} {gz:6.2f} (mag={gyro_mag:.2f})  |  "
          f"Heading: {heading:5.1f}°  |  "
          f"V={data.get('v', 0):.2f}  "
          f"L={data.get('L', 0)} R={data.get('R', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Read ESP32 T:1001 IMU stream")
    parser.add_argument("--count", type=int, default=20, help="Number of readings")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="Baud rate")
    args = parser.parse_args()

    print(f"Opening {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    time.sleep(0.3)
    ser.reset_input_buffer()

    print(f"\nReading T:1001 stream ({args.count} samples)...\n")
    readings = []

    for i in range(args.count):
        data = read_t1001(ser)
        if data:
            print_imu(data, i + 1)
            readings.append(data)
        else:
            print(f"[{i+1:3d}] FAILED — no T:1001 received")

    # Summary
    print(f"\n{'='*80}")
    print(f"Results: {len(readings)}/{args.count} successful")

    if readings:
        def stat(key, label):
            vals = [r.get(key, 0) for r in readings]
            return f"  {label:12s}: min={min(vals):10.5f}  max={max(vals):10.5f}  avg={sum(vals)/len(vals):10.5f}"

        print("\nAccelerometer (gravity-subtracted):")
        print(stat("ax", "X"))
        print(stat("ay", "Y"))
        print(stat("az", "Z"))
        mags = [math.sqrt(r["ax"]**2 + r["ay"]**2 + r["az"]**2) for r in readings]
        print(f"  {'magnitude':12s}: min={min(mags):10.5f}  max={max(mags):10.5f}  avg={sum(mags)/len(mags):10.5f}")

        print("\nGyroscope:")
        print(stat("gx", "X"))
        print(stat("gy", "Y"))
        print(stat("gz", "Z"))

        print("\nMagnetometer heading:")
        headings = [math.degrees(math.atan2(r["my"], r["mx"])) % 360 for r in readings]
        print(f"  {'heading':12s}: min={min(headings):10.1f}°  max={max(headings):10.1f}°  "
              f"avg={sum(headings)/len(headings):10.1f}°")

        print(f"\nVoltage: {readings[-1].get('v', 0):.2f}V")

        print("\n--- Suggested stuck detection thresholds ---")
        avg_mag = sum(mags) / len(mags)
        max_mag = max(mags)
        print(f"  Accel rest magnitude: avg={avg_mag:.4f}, max={max_mag:.4f}")
        print(f"  Suggested ACCEL_MOVING_THRESHOLD: {max_mag * 1.5:.3f}")
        print(f"  (Set higher if vibration from wheels exceeds this at rest)")

    ser.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
