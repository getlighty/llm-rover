#!/usr/bin/env python3
"""
Static desk test: gimbal tracking of a specific object.
Tests the visual servo's Stage 1 (gimbal P-control) without wheel movement.
The gimbal should follow the target object and keep it centered.
"""

import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

import cv2
import numpy as np
import json
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detector import LocalDetector

import serial

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200

# Gimbal P-control gains (from visual_servo.py)
GIMBAL_GAIN_X = 40.0
GIMBAL_GAIN_Y = 30.0
GIMBAL_DEADZONE = 0.05

def send_serial(ser, cmd):
    ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
    time.sleep(0.02)
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and line.startswith("{"):
            return json.loads(line)
    except Exception:
        pass
    return None


def track_object(ser, cap, detector, target_name, duration=15):
    """Track an object with gimbal for `duration` seconds. No wheel movement."""
    pan = 0.0
    tilt = 0.0
    lost_count = 0
    step = 0
    start = time.time()

    # Center gimbal
    send_serial(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(0.5)

    # First: scan to find the target
    PAN_STEPS = [-120, -80, -40, 0, 40, 80, 120]
    found = False

    print(f"[track] Scanning for '{target_name}'...")
    for scan_pan in PAN_STEPS:
        send_serial(ser, {"T": 133, "X": scan_pan, "Y": 0, "SPD": 300, "ACC": 20})
        time.sleep(0.6)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            continue
        dets = detector.detect(frame)
        target = detector.find(target_name, dets)
        if target and target["bw"] > 0.02:
            pan = scan_pan
            print(f"[track] Found '{target_name}' at pan={scan_pan} (conf={target['conf']:.0%}, bw={target['bw']:.2f})")
            found = True
            break

    if not found:
        print(f"[track] '{target_name}' not found in scan")
        send_serial(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
        return False

    # Now track with P-control
    print(f"[track] Tracking '{target_name}' for {duration}s with gimbal P-control...")
    out_dir = "/tmp/track_test"
    os.makedirs(out_dir, exist_ok=True)

    errors_x = []
    errors_y = []

    while time.time() - start < duration:
        step += 1

        # Flush camera buffer
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            continue

        dets = detector.detect(frame)
        target = detector.find(target_name, dets)

        if target is None:
            lost_count += 1
            if lost_count > 20:
                print(f"[track] Lost '{target_name}' for 20 frames, stopping")
                break
            time.sleep(0.1)
            continue

        lost_count = 0
        cx = target["cx"]
        cy = target["cy"]
        bw = target["bw"]

        if bw < 0.02:
            continue

        # P-control errors
        err_x = cx - 0.5
        err_y = cy - 0.5
        errors_x.append(err_x)
        errors_y.append(err_y)

        # Update gimbal
        if abs(err_x) > GIMBAL_DEADZONE:
            pan += err_x * GIMBAL_GAIN_X
        if abs(err_y) > GIMBAL_DEADZONE:
            tilt -= err_y * GIMBAL_GAIN_Y

        pan = max(-150, min(150, pan))
        tilt = max(-30, min(90, tilt))

        send_serial(ser, {"T": 133, "X": float(round(pan, 1)), "Y": float(round(tilt, 1)),
                          "SPD": 200, "ACC": 10})

        # Log every 10 steps
        if step % 10 == 0:
            dist = target.get("dist_m", "?")
            print(f"[track] step={step} cx={cx:.2f} cy={cy:.2f} err_x={err_x:+.2f} "
                  f"err_y={err_y:+.2f} pan={pan:.0f} tilt={tilt:.0f} bw={bw:.2f} dist={dist}m")
            # Save annotated frame
            annotated = detector.draw(frame.copy(), dets)
            cv2.imwrite(os.path.join(out_dir, f"step_{step:04d}.jpg"), annotated)

        time.sleep(0.1)  # ~10Hz

    # Save final frame
    ret, frame = cap.read()
    if ret:
        dets = detector.detect(frame)
        annotated = detector.draw(frame.copy(), dets)
        cv2.imwrite(os.path.join(out_dir, "final.jpg"), annotated)

    # Return to center
    send_serial(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

    # Stats
    if errors_x:
        avg_err_x = sum(abs(e) for e in errors_x) / len(errors_x)
        avg_err_y = sum(abs(e) for e in errors_y) / len(errors_y)
        max_err_x = max(abs(e) for e in errors_x)
        max_err_y = max(abs(e) for e in errors_y)
        print(f"\n[track] TRACKING STATS for '{target_name}':")
        print(f"  Steps: {step}, Lost: {lost_count}")
        print(f"  Avg |err_x|: {avg_err_x:.3f}  Avg |err_y|: {avg_err_y:.3f}")
        print(f"  Max |err_x|: {max_err_x:.3f}  Max |err_y|: {max_err_y:.3f}")
        print(f"  Final pan: {pan:.0f}  Final tilt: {tilt:.0f}")
        if avg_err_x < 0.10 and avg_err_y < 0.10:
            print(f"  PASS: Good tracking (avg error < 0.10)")
        elif avg_err_x < 0.15 and avg_err_y < 0.15:
            print(f"  OK: Acceptable tracking (avg error < 0.15)")
        else:
            print(f"  NEEDS TUNING: High error (consider adjusting gains)")

    return True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "person"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    print(f"[test] Gimbal tracking test: target='{target}', duration={duration}s")

    # Connect
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    detector = LocalDetector(conf=0.20)

    # Run tracking test
    success = track_object(ser, cap, detector, target, duration)

    if success:
        print(f"\n[test] Gimbal tracking test PASSED")
    else:
        print(f"\n[test] Gimbal tracking test FAILED — target not found or lost")

    cap.release()
    ser.close()


if __name__ == "__main__":
    main()
