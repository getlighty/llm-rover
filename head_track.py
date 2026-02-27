#!/usr/bin/env python3
"""Head tracking - gimbal follows detected face via WiFi ESP32 control.
Uses OpenCV's Haar cascade face detector (no mediapipe needed)."""

import cv2
import requests
import time
import signal
import sys

ESP32_URL = "http://192.168.4.1/js"
session = requests.Session()

pan_angle = 0.0
tilt_angle = 0.0
running = True

def send_cmd(json_str):
    try:
        session.get(ESP32_URL, params={"json": json_str}, timeout=2)
    except:
        pass

def send_gimbal(x, y, spd=200, acc=10):
    send_cmd('{"T":133,"X":%.1f,"Y":%.1f,"SPD":%d,"ACC":%d}' % (x, y, spd, acc))

def set_lights(base=0, head=0):
    send_cmd('{"T":132,"IO4":%d,"IO5":%d}' % (base, head))

def stop(*_):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

print("Head tracker starting...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

# Boost camera brightness/exposure for low light
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # higher = brighter (auto if supported)
cap.set(cv2.CAP_PROP_GAIN, 200)
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# Use OpenCV's built-in Haar cascade
cascade_paths = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]
face_cascade = None
for p in cascade_paths:
    face_cascade = cv2.CascadeClassifier(p)
    if not face_cascade.empty():
        print(f"Loaded face cascade: {p}")
        break

if face_cascade is None or face_cascade.empty():
    print("ERROR: Cannot load face cascade")
    sys.exit(1)

# Center gimbal and turn on headlights
send_gimbal(0, 0)
set_lights(base=200, head=255)
print("Lights ON. Tracking... Ctrl+C to stop.")

GAIN_X = 5.0
GAIN_Y = 4.0
DEADZONE = 0.06
last_send = 0
MIN_SEND_INTERVAL = 0.12
detect_count = 0
frame_count = 0

while running:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.03)
        continue

    frame_count += 1
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30)
    )

    if len(faces) > 0:
        detect_count += 1
        # Pick largest face
        areas = [fw * fh for (_, _, fw, fh) in faces]
        idx = areas.index(max(areas))
        fx, fy, fw, fh = faces[idx]

        face_cx = (fx + fw / 2) / w
        face_cy = (fy + fh / 2) / h

        err_x = face_cx - 0.5
        err_y = face_cy - 0.5

        if abs(err_x) > DEADZONE:
            pan_angle -= err_x * GAIN_X
        if abs(err_y) > DEADZONE:
            tilt_angle -= err_y * GAIN_Y

        pan_angle = max(-180, min(180, pan_angle))
        tilt_angle = max(-30, min(90, tilt_angle))

        now = time.time()
        if now - last_send >= MIN_SEND_INTERVAL:
            send_gimbal(pan_angle, tilt_angle)
            last_send = now
            pct = 100 * detect_count / frame_count if frame_count else 0
            print(f"\r  pan={pan_angle:+6.1f}  tilt={tilt_angle:+6.1f}  face=({face_cx:.2f},{face_cy:.2f})  det={detect_count}/{frame_count} ({pct:.0f}%)", end="", flush=True)
    elif frame_count % 30 == 0:
        pct = 100 * detect_count / frame_count if frame_count else 0
        print(f"\r  [no face] frames={frame_count} det={detect_count} ({pct:.0f}%)          ", end="", flush=True)

    time.sleep(0.033)

cap.release()
send_gimbal(0, 0)
set_lights(base=0, head=0)
print("\nStopped. Gimbal centered. Lights off.")
