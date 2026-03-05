#!/usr/bin/env python3
"""pilot.py — Direct rover control via web API.

Sends ESP32 commands via POST /esp and takes snapshots via GET /snap.
Used by Claude Code to manually pilot the rover.
"""

import json
import sys
import time
import urllib.request

BASE = "http://localhost:8090"


def esp(*cmds):
    """Send ESP32 commands. Returns result dict."""
    payload = json.dumps({"commands": list(cmds)}).encode()
    req = urllib.request.Request(f"{BASE}/esp", data=payload,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def snap(path="/tmp/pilot_snap.jpg"):
    """Take a camera snapshot, save to path."""
    urllib.request.urlretrieve(f"{BASE}/snap", path)
    return path


def gimbal(pan=0, tilt=0, spd=200, acc=10):
    """Move gimbal to absolute position."""
    return esp({"T": 133, "X": pan, "Y": tilt, "SPD": spd, "ACC": acc})


def drive(left, right, duration=1.0):
    """Drive wheels for duration then stop."""
    return esp(
        {"T": 1, "L": left, "R": right},
        {"_pause": min(duration, 10)},
        {"T": 1, "L": 0, "R": 0},
    )


def stop():
    """Emergency stop."""
    return esp({"T": 1, "L": 0, "R": 0})


def spin(direction="right", duration=0.4):
    """Spin in place. direction='left' or 'right'."""
    spd = 0.15
    if direction == "right":
        return drive(spd, -spd, duration)
    else:
        return drive(-spd, spd, duration)


def look_and_snap(pan=0, tilt=0, path="/tmp/pilot_snap.jpg", settle=0.6):
    """Move gimbal, wait for it to settle, take snapshot."""
    gimbal(pan, tilt)
    time.sleep(settle)
    return snap(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pilot.py <command> [args]")
        print("  snap [path]")
        print("  gimbal <pan> <tilt>")
        print("  drive <left> <right> <duration>")
        print("  spin <left|right> [duration]")
        print("  stop")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "snap":
        p = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pilot_snap.jpg"
        snap(p)
        print(f"Saved: {p}")
    elif cmd == "gimbal":
        pan = float(sys.argv[2]) if len(sys.argv) > 2 else 0
        tilt = float(sys.argv[3]) if len(sys.argv) > 3 else 0
        print(gimbal(pan, tilt))
    elif cmd == "drive":
        l = float(sys.argv[2])
        r = float(sys.argv[3])
        d = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        print(drive(l, r, d))
    elif cmd == "spin":
        d = sys.argv[2] if len(sys.argv) > 2 else "right"
        dur = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        print(spin(d, dur))
    elif cmd == "stop":
        print(stop())
