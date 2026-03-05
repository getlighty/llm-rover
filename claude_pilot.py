#!/usr/bin/env python3
"""claude_pilot.py — Direct rover control API for Claude Code.

Provides functions to control the rover's gimbal, wheels, and camera
directly via the /esp and /snap HTTP endpoints, bypassing the LLM.
Claude Code can use this to do its own search, navigation, and testing.

Usage from bash:
    python3 claude_pilot.py scan          # 360° scan, save snapshots
    python3 claude_pilot.py center        # center gimbal
    python3 claude_pilot.py look 45 -10   # pan=45, tilt=-10
    python3 claude_pilot.py snap name     # take snapshot as /tmp/pilot_name.jpg
    python3 claude_pilot.py spin_right 0.5  # spin right for 0.5s
    python3 claude_pilot.py spin_left 0.5
    python3 claude_pilot.py forward 0.5 0.1  # forward for 0.5s at 0.1 m/s
    python3 claude_pilot.py stop          # stop wheels
"""

import json
import sys
import time
import urllib.request

BASE = "http://localhost:8090"


def esp(cmd):
    """Send raw ESP32 command via /esp endpoint."""
    if isinstance(cmd, list):
        data = {"commands": cmd}
    else:
        data = cmd
    req = urllib.request.Request(
        f"{BASE}/esp",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
    return resp


def snap(name="frame"):
    """Capture a snapshot to /tmp/pilot_{name}.jpg."""
    path = f"/tmp/pilot_{name}.jpg"
    urllib.request.urlretrieve(f"{BASE}/snap", path)
    return path


def gimbal(pan=0, tilt=0, speed=300, acc=15):
    """Move gimbal to absolute position."""
    return esp({"T": 133, "X": pan, "Y": tilt, "SPD": speed, "ACC": acc})


def wheels(left, right):
    """Set wheel speeds (m/s). Positive = forward."""
    return esp({"T": 1, "L": left, "R": right})


def stop():
    """Stop all wheels."""
    return wheels(0, 0)


def lights(base=0, head=0):
    """Set LED brightness (0-255)."""
    return esp({"T": 132, "IO4": base, "IO5": head})


def scan_360(tilt=-5):
    """Do a full 360° scan in 45° increments. Returns list of (pan, path)."""
    results = []
    pans = [-180, -135, -90, -45, 0, 45, 90, 135]
    for pan in pans:
        gimbal(pan, tilt)
        time.sleep(1.2)
        name = f"scan_{pan:+04d}"
        path = snap(name)
        results.append((pan, path))
        print(f"  pan={pan:+4d}° → {path}")
    gimbal(0, 0)  # return to center
    return results


def spin_right(duration=0.5, speed=0.15):
    """Spin body right for given duration."""
    wheels(speed, -speed)
    time.sleep(duration)
    stop()


def spin_left(duration=0.5, speed=0.15):
    """Spin body left for given duration."""
    wheels(-speed, speed)
    time.sleep(duration)
    stop()


def forward(duration=0.5, speed=0.1):
    """Drive forward for given duration."""
    wheels(speed, speed)
    time.sleep(duration)
    stop()


def search_scan(tilt=-10):
    """Quick search scan: left, center, right. Returns snapshots."""
    results = []
    for pan in [-90, -45, 0, 45, 90]:
        gimbal(pan, tilt)
        time.sleep(1.0)
        path = snap(f"search_{pan:+04d}")
        results.append((pan, path))
        print(f"  pan={pan:+4d}° → {path}")
    gimbal(0, 0)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 claude_pilot.py <command> [args...]")
        print("Commands: scan, center, look, snap, spin_right, spin_left, forward, stop, search")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "scan":
        tilt = int(sys.argv[2]) if len(sys.argv) > 2 else -5
        print(f"Scanning 360° at tilt={tilt}°...")
        scan_360(tilt)
        print("Done.")
    elif cmd == "search":
        tilt = int(sys.argv[2]) if len(sys.argv) > 2 else -10
        print(f"Quick search scan at tilt={tilt}°...")
        search_scan(tilt)
        print("Done.")
    elif cmd == "center":
        gimbal(0, 0)
        print("Centered.")
    elif cmd == "look":
        pan = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        tilt = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        gimbal(pan, tilt)
        print(f"Looking at pan={pan}°, tilt={tilt}°.")
    elif cmd == "snap":
        name = sys.argv[2] if len(sys.argv) > 2 else "frame"
        path = snap(name)
        print(f"Saved: {path}")
    elif cmd == "spin_right":
        dur = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        spin_right(dur)
        print(f"Spun right for {dur}s.")
    elif cmd == "spin_left":
        dur = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        spin_left(dur)
        print(f"Spun left for {dur}s.")
    elif cmd == "forward":
        dur = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        spd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        forward(dur, spd)
        print(f"Forward {dur}s at {spd} m/s.")
    elif cmd == "stop":
        stop()
        print("Stopped.")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
