#!/usr/bin/env python3
"""test_ahrs.py — Standalone AHRS test using ESP32 T:1001 stream.

Reads raw IMU data from serial, runs the Madgwick AHRS filter in real-time,
and prints heading/pitch/roll, mag_weight, and gyro bias every update.

Usage:
    python3 test_ahrs.py              # continuous output
    python3 test_ahrs.py --count 50   # 50 readings then stop
    python3 test_ahrs.py --cal        # load mag_cal.json if present
"""

import argparse
import json
import math
import serial
import time

from ahrs_filter import MadgwickAHRS, MagCalibration

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200


def read_t1001(ser):
    """Read one clean T:1001 line from the stream."""
    for _ in range(5):
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("T") == 1001:
                return data
        except json.JSONDecodeError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Test AHRS filter with live IMU data")
    parser.add_argument("--count", type=int, default=0, help="Number of readings (0=infinite)")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="Baud rate")
    parser.add_argument("--cal", action="store_true", help="Load mag_cal.json")
    parser.add_argument("--beta", type=float, default=0.033, help="Madgwick beta gain")
    args = parser.parse_args()

    # Set up mag calibration
    mag_cal = None
    if args.cal:
        mag_cal = MagCalibration()
        if mag_cal.load():
            print(f"Loaded mag cal: offset={mag_cal.offset}")
        else:
            print("No mag_cal.json found, running without calibration")
            mag_cal = None

    ahrs = MadgwickAHRS(beta=args.beta, mag_cal=mag_cal)

    print(f"Opening {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    time.sleep(0.3)
    ser.reset_input_buffer()

    print(f"\nReading T:1001 stream, running AHRS filter...\n")
    print(f"{'#':>4s}  {'Heading':>7s}  {'Pitch':>6s}  {'Roll':>6s}  "
          f"{'MagW':>5s}  {'GyroBias':>24s}  {'GyroRaw':>20s}  "
          f"{'AccMag':>7s}  {'Enc':>5s}")
    print("-" * 110)

    i = 0
    try:
        while True:
            data = read_t1001(ser)
            if data is None:
                continue

            i += 1
            ax = float(data.get("ax", 0))
            ay = float(data.get("ay", 0))
            az = float(data.get("az", 0))
            gx = float(data.get("gx", 0))
            gy = float(data.get("gy", 0))
            gz = float(data.get("gz", 0))
            mx = float(data.get("mx", 0))
            my = float(data.get("my", 0))
            mz = float(data.get("mz", 0))
            enc_l = float(data.get("L", 0))
            enc_r = float(data.get("R", 0))
            enc_speed = (abs(enc_l) + abs(enc_r)) / 2.0

            ahrs.update(gx, gy, gz, ax, ay, az, mx, my, mz, enc_speed)

            roll, pitch, yaw = ahrs.euler
            heading = ahrs.heading
            bias = ahrs.gyro_bias
            accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

            print(f"{i:4d}  {heading:7.1f}°  {pitch:+6.1f}°  {roll:+6.1f}°  "
                  f"{ahrs.mag_weight:5.2f}  "
                  f"({bias[0]:+6.2f},{bias[1]:+6.2f},{bias[2]:+6.2f})  "
                  f"({gx:+6.1f},{gy:+6.1f},{gz:+6.1f})  "
                  f"{accel_mag:7.2f}  {enc_speed:5.3f}")

            if args.count > 0 and i >= args.count:
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Final heading: {ahrs.heading:.1f}°")
    roll, pitch, yaw = ahrs.euler
    print(f"Final euler: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
    print(f"Gyro bias: {ahrs.gyro_bias}")
    print(f"Mag weight: {ahrs.mag_weight:.3f}")
    print(f"Quaternion: {ahrs.q}")

    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()
