#!/usr/bin/env python3
"""
Static desk test: gimbal sweep + YOLOv8 detection at each position.
Captures annotated frames and builds a detection map.
No wheel movement — camera-only test.
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

# Serial setup
import serial

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200

def send_serial(ser, cmd):
    """Send JSON command to ESP32."""
    ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
    time.sleep(0.05)
    # Read response
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and line.startswith("{"):
            return json.loads(line)
    except Exception:
        pass
    return None

def main():
    # Connect serial
    print("[test] Connecting to ESP32...")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(0.2)

    # Test connection
    fb = send_serial(ser, {"T": 130})
    if fb:
        print(f"[test] Connected. Battery: {fb.get('v', '?')}V")
    else:
        print("[test] Connected (no feedback)")

    # Open camera
    print("[test] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[test] ERROR: Cannot open camera")
        ser.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    # Init detector
    print("[test] Loading YOLOv8n...")
    detector = LocalDetector(conf=0.20)

    # Sweep positions
    PAN_STEPS = [-120, -80, -40, 0, 40, 80, 120]
    TILT_STEPS = [0, 25]

    out_dir = "/tmp/sweep_test"
    os.makedirs(out_dir, exist_ok=True)

    all_detections = {}  # (pan, tilt) -> [detections]
    total_objects = set()

    print(f"\n[test] Starting gimbal sweep: {len(PAN_STEPS)}x{len(TILT_STEPS)} = {len(PAN_STEPS)*len(TILT_STEPS)} positions")
    print("=" * 60)

    for tilt in TILT_STEPS:
        for pan in PAN_STEPS:
            # Move gimbal
            send_serial(ser, {"T": 133, "X": pan, "Y": tilt, "SPD": 300, "ACC": 20})
            time.sleep(0.8)  # let gimbal settle

            # Flush old frames from camera buffer
            for _ in range(5):
                cap.read()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print(f"  pan={pan:+4d} tilt={tilt:+3d}: FRAME ERROR")
                continue

            # Run detection
            dets = detector.detect(frame)
            all_detections[(pan, tilt)] = dets

            # Draw and save
            annotated = detector.draw(frame.copy(), dets)
            fname = f"pan{pan:+04d}_tilt{tilt:+03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), annotated)

            # Log results
            if dets:
                names = [f"{d['name']}({d['conf']:.0%})" for d in dets]
                for d in dets:
                    total_objects.add(d['name'])
                print(f"  pan={pan:+4d} tilt={tilt:+3d}: {', '.join(names)}")
            else:
                print(f"  pan={pan:+4d} tilt={tilt:+3d}: (nothing)")

    # Return gimbal to center
    send_serial(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

    # Summary
    print("\n" + "=" * 60)
    print("[test] SWEEP COMPLETE")
    print(f"  Positions scanned: {len(all_detections)}")
    print(f"  Unique object types: {sorted(total_objects)}")

    # Build object location map
    print("\n[test] Object location map:")
    obj_locations = {}
    for (pan, tilt), dets in all_detections.items():
        for d in dets:
            name = d['name']
            if name not in obj_locations:
                obj_locations[name] = []
            obj_locations[name].append({
                "pan": pan, "tilt": tilt,
                "conf": d['conf'],
                "cx": d['cx'], "cy": d['cy'],
                "bw": d['bw'],
                "dist_m": d.get('dist_m', None)
            })

    for name, locs in sorted(obj_locations.items()):
        best = max(locs, key=lambda l: l['conf'])
        dist_str = f" ~{best['dist_m']:.1f}m" if best['dist_m'] else ""
        print(f"  {name}: best at pan={best['pan']:+d} tilt={best['tilt']:+d} "
              f"({best['conf']:.0%}{dist_str}), seen {len(locs)}x")

    # Save map as JSON
    map_file = os.path.join(out_dir, "detection_map.json")
    with open(map_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "positions_scanned": len(all_detections),
            "objects": {
                name: locs for name, locs in obj_locations.items()
            }
        }, f, indent=2)
    print(f"\n[test] Results saved to {out_dir}/")
    print(f"[test] Detection map: {map_file}")

    # Cleanup
    cap.release()
    ser.close()
    print("[test] Done.")

if __name__ == "__main__":
    main()
