#!/usr/bin/env python3
"""Find the door using camera sweep + Groq LLM vision."""

import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

import cv2
import numpy as np
import json
import time
import sys
import base64
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serial

# Load .env
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200


def send(ser, cmd):
    if isinstance(cmd, dict) and cmd.get("T") == 1:
        cmd = dict(cmd, L=-cmd.get("L", 0), R=-cmd.get("R", 0))
    ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
    time.sleep(0.02)
    try:
        ser.readline()
    except:
        pass


def ask_groq_vision(image_bytes, prompt):
    """Send image to Groq LLM vision and get response."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 300,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    # Sweep and capture frames at each position
    PAN_STEPS = [-150, -120, -80, -40, 0, 40, 80, 120, 150]
    frames = {}

    print("[door] Sweeping to find door...")
    for pan in PAN_STEPS:
        send(ser, {"T": 133, "X": pan, "Y": 5, "SPD": 300, "ACC": 20})
        time.sleep(0.7)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames[pan] = jpg.tobytes()
            cv2.imwrite(f"/tmp/door_scan_pan{pan:+04d}.jpg", frame)

    # Return to center
    send(ser, {"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

    # Ask LLM about each frame
    print("[door] Asking LLM about each direction...")
    door_candidates = []

    prompt = (
        "You are a robot looking around a room. "
        "Is there a door or doorway visible in this image? "
        "If YES: describe its position (left/center/right of frame), "
        "estimate how far away it is in meters, and what's beyond it. "
        "If NO: just say 'No door visible' and briefly describe what you see. "
        "Be concise (2-3 sentences max)."
    )

    for pan in PAN_STEPS:
        if pan not in frames:
            continue
        print(f"\n  pan={pan:+4d}°: ", end="", flush=True)
        try:
            response = ask_groq_vision(frames[pan], prompt)
            print(response.strip())
            if any(word in response.lower() for word in ["door", "doorway", "opening", "entrance", "exit", "hallway"]):
                door_candidates.append({"pan": pan, "description": response.strip()})
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n\n[door] Door candidates: {len(door_candidates)}")
    for c in door_candidates:
        print(f"  pan={c['pan']:+d}°: {c['description'][:100]}...")

    if door_candidates:
        # Ask LLM to pick the best one
        best_pan = door_candidates[0]["pan"]
        print(f"\n[door] Best door candidate at pan={best_pan}°")

        # Get more detailed info about the door
        if best_pan in frames:
            detail_prompt = (
                "You are a small wheeled robot on the ground level. "
                "There is a door/doorway in this image. "
                "Estimate: 1) How far away is the door in meters? "
                "2) Is the door open or closed? "
                "3) Is the doorway wide enough for a small robot (25cm wide) to drive through? "
                "4) What obstacles are between you and the door? "
                "5) What direction should you drive to reach the door (left, straight, right)? "
                "Be concise."
            )
            detail = ask_groq_vision(frames[best_pan], detail_prompt)
            print(f"\n[door] Detail: {detail}")
    else:
        print("\n[door] No door found in any direction. May need to drive forward and re-scan.")

    cap.release()
    ser.close()
    print("\n[door] Done.")


if __name__ == "__main__":
    main()
