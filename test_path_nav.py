#!/usr/bin/env python3
"""
Ground test: Room survey + path planning + map visualization.
Tests the full pipeline: survey → map → plan → (optional) follow.

Usage:
    python3 test_path_nav.py                  # Survey only, plan to nearest object
    python3 test_path_nav.py --target person  # Survey + plan path to 'person'
    python3 test_path_nav.py --target door    # Survey + find door + plan path
    python3 test_path_nav.py --go             # Actually drive the planned path
"""

import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

import cv2
import numpy as np
import json
import math
import time
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detector import LocalDetector
from path_planner import WorldMap, PathPlanner, PathFollower, DoorDetector

import serial

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200


class SimpleRover:
    """Minimal rover serial interface for testing."""

    def __init__(self, ser):
        self._ser = ser
        self._on_command = None

    def send(self, cmd):
        # Notify pose tracker
        if self._on_command:
            try:
                self._on_command(cmd)
            except Exception:
                pass
        # Negate wheel speeds (motors wired in reverse)
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("L", 0), R=-cmd.get("R", 0))
        self._ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
        time.sleep(0.02)
        try:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            pass


class SimpleTracker:
    """Minimal camera wrapper for testing."""

    def __init__(self, cap):
        self._cap = cap
        self._jpeg = None

    def get_jpeg(self):
        for _ in range(3):
            self._cap.read()
        ret, frame = self._cap.read()
        if not ret:
            return None
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._jpeg = jpg.tobytes()
        return self._jpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None, help="Target to navigate to")
    parser.add_argument("--go", action="store_true", help="Actually drive the path")
    args = parser.parse_args()

    # Setup
    print("[test] Initializing...")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(0.2)

    rover = SimpleRover(ser)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    tracker = SimpleTracker(cap)
    detector = LocalDetector(conf=0.20)

    # Pose tracker
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Import PoseTracker from rover_brain
    import importlib.util
    spec = importlib.util.spec_from_file_location("rover_brain_module",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rover_brain.py"))
    # Can't import the whole module (too many deps), recreate PoseTracker here
    class PoseTracker:
        WHEELBASE = 0.25
        def __init__(self):
            self.body_yaw = 0.0
            self.cam_pan = 0.0
            self.cam_tilt = 0.0
            self._last_wheel_time = None
            self._last_L = 0.0
            self._last_R = 0.0
        @property
        def world_pan(self):
            return self.body_yaw + self.cam_pan
        def on_command(self, cmd):
            if not isinstance(cmd, dict):
                return
            t = cmd.get("T")
            if t == 1:
                now = time.time()
                v_left = cmd.get("L", 0)
                v_right = cmd.get("R", 0)
                if self._last_wheel_time is not None:
                    dt = now - self._last_wheel_time
                    if 0 < dt < 5.0:
                        avg_L = (self._last_L + v_left) / 2
                        avg_R = (self._last_R + v_right) / 2
                        yaw_rate = (avg_R - avg_L) / self.WHEELBASE
                        self.body_yaw += math.degrees(yaw_rate * dt)
                        self.body_yaw = ((self.body_yaw + 180) % 360) - 180
                self._last_wheel_time = now
                self._last_L = v_left
                self._last_R = v_right
            elif t == 133:
                self.cam_pan = cmd.get("X", self.cam_pan)
                self.cam_tilt = cmd.get("Y", self.cam_tilt)

    pose = PoseTracker()
    rover._on_command = pose.on_command

    # Build world map
    world = WorldMap(rover, detector, tracker, pose)

    print("\n[test] === ROOM SURVEY ===")
    landmarks = world.survey()

    print(f"\n[test] Landmarks found:")
    for name, info in sorted(landmarks.items(), key=lambda x: -x[1]["conf"]):
        print(f"  {name}: ({info['x']:.2f}, {info['y']:.2f}) "
              f"dist={info['dist']:.1f}m angle={info['angle']:.0f}° conf={info['conf']:.0%}")

    if world.doors:
        print(f"\n[test] Doors found:")
        for door in world.doors:
            print(f"  ({door['x']:.2f}, {door['y']:.2f}) width={door['width_m']:.2f}m "
                  f"angle={door['angle']:.0f}° conf={door['confidence']:.0%}")

    # Save map
    world.save("/tmp/world_map.json")
    print(f"\n[test] Map saved to /tmp/world_map.json and /tmp/world_map.png")

    # Plan path
    planner = PathPlanner(world)
    goal = None
    goal_name = args.target

    if goal_name == "door":
        door = world.find_door()
        if door:
            goal = (door["x"], door["y"])
            print(f"\n[test] Planning path to door at ({goal[0]:.2f}, {goal[1]:.2f})")
        else:
            print("\n[test] No door detected. Try --target <object> instead.")
    elif goal_name:
        loc = world.find_landmark(goal_name)
        if loc:
            goal = loc
            print(f"\n[test] Planning path to '{goal_name}' at ({goal[0]:.2f}, {goal[1]:.2f})")
        else:
            print(f"\n[test] '{goal_name}' not found in map. Available: {list(landmarks.keys())}")
    else:
        # Default: plan to nearest significant object
        if landmarks:
            best = min(landmarks.items(),
                       key=lambda x: x[1]["dist"] if x[1]["dist"] > 0.5 else 999)
            goal_name = best[0]
            goal = (best[1]["x"], best[1]["y"])
            print(f"\n[test] Auto-selecting nearest target: '{goal_name}' "
                  f"at ({goal[0]:.2f}, {goal[1]:.2f})")

    if goal:
        waypoints = planner.plan(goal)
        if waypoints:
            print(f"\n[test] Path waypoints:")
            for i, wp in enumerate(waypoints):
                dist = math.sqrt(wp[0] ** 2 + wp[1] ** 2)
                angle = math.degrees(math.atan2(wp[0], wp[1]))
                print(f"  {i + 1}. ({wp[0]:.2f}, {wp[1]:.2f}) "
                      f"dist={dist:.2f}m heading={angle:.0f}°")

            # Render map with path
            img = world._render_grid(path_waypoints=waypoints)
            cv2.imwrite("/tmp/world_map_path.png", img)
            print(f"\n[test] Map with path saved to /tmp/world_map_path.png")

            if args.go:
                print(f"\n[test] === FOLLOWING PATH ===")
                follower = PathFollower(rover, pose, detector, tracker)
                success = follower.follow(waypoints)
                print(f"\n[test] Navigation {'SUCCEEDED' if success else 'FAILED'}")
            else:
                print(f"\n[test] Add --go to actually drive the path")
        else:
            print("[test] No path found to goal")

    # Cleanup
    rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    cap.release()
    ser.close()
    print("\n[test] Done.")


if __name__ == "__main__":
    main()
