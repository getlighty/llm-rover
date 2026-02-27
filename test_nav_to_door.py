#!/usr/bin/env python3
"""
Navigate toward the door.
Based on LLM vision: door is at roughly pan=+130°, ~3m away.

Strategy:
1. Survey room to build map
2. Set door goal at estimated position
3. Plan and follow path
4. Use very slow speeds, look around before reversing
"""

import os
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

import cv2
import numpy as np
import json
import math
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detector import LocalDetector
from path_planner import WorldMap, PathPlanner, PathFollower

import serial

SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUD = 115200


class SimpleRover:
    def __init__(self, ser):
        self._ser = ser
        self._on_command = None
    def send(self, cmd):
        if self._on_command:
            try: self._on_command(cmd)
            except: pass
        if isinstance(cmd, dict) and cmd.get("T") == 1:
            cmd = dict(cmd, L=-cmd.get("L", 0), R=-cmd.get("R", 0))
        self._ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
        time.sleep(0.02)
        try: self._ser.readline()
        except: pass


class SimpleTracker:
    def __init__(self, cap):
        self._cap = cap
    def get_jpeg(self):
        for _ in range(3):
            self._cap.read()
        ret, frame = self._cap.read()
        if not ret:
            return None
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes()


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


def main():
    print("[nav] Door Navigation Test")
    print("=" * 60)

    # Setup
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
    pose = PoseTracker()
    rover._on_command = pose.on_command

    # Dim lights
    rover.send({"T": 132, "IO4": 0, "IO5": 0})

    # Step 1: Survey room
    print("\n[nav] Step 1: Surveying room...")
    world = WorldMap(rover, detector, tracker, pose)
    world.survey()
    world.save("/tmp/nav_door_map.json")

    print(f"\n[nav] Map: {len(world.landmarks)} landmarks, {len(world.doors)} doors")
    for name, info in sorted(world.landmarks.items(), key=lambda x: -x[1]["conf"]):
        print(f"  {name}: ({info['x']:.2f}, {info['y']:.2f}) "
              f"dist={info['dist']:.1f}m angle={info['angle']:.0f}° conf={info['conf']:.0%}")

    # Step 2: Set door goal
    # LLM vision found door at pan=+130°, ~3m away
    # From the images: door is to the right, slightly forward
    # In XY: x = 3 * sin(130°) ≈ 2.3, y = 3 * cos(130°) ≈ -1.9
    # But also visible at pan=+120 which means x = 3*sin(120°) ≈ 2.6, y = 3*cos(120°) ≈ -1.5

    door_angle_deg = 130.0  # estimated from LLM vision
    door_dist_m = 2.5  # conservative estimate
    door_x = door_dist_m * math.sin(math.radians(door_angle_deg))
    door_y = door_dist_m * math.cos(math.radians(door_angle_deg))

    # Check if door was found in map
    door = world.find_door()
    if door:
        door_x, door_y = door["x"], door["y"]
        print(f"\n[nav] Door found in map at ({door_x:.2f}, {door_y:.2f})")
    else:
        print(f"\n[nav] Door not in map, using LLM estimate: ({door_x:.2f}, {door_y:.2f})")
        print(f"  (angle={door_angle_deg}°, dist={door_dist_m}m)")

    # Step 3: Plan path
    print("\n[nav] Step 2: Planning path to door...")
    planner = PathPlanner(world)

    # First navigate to a point 0.5m before the door (approach carefully)
    approach_dist = 0.5
    approach_x = door_x - approach_dist * math.sin(math.radians(door_angle_deg))
    approach_y = door_y - approach_dist * math.cos(math.radians(door_angle_deg))

    waypoints = planner.plan((approach_x, approach_y))

    if not waypoints:
        print("[nav] No path found! Trying direct path...")
        # Fallback: just aim for the door directly
        waypoints = [(door_x * 0.3, door_y * 0.3),  # 30% of the way
                     (door_x * 0.6, door_y * 0.6),  # 60%
                     (approach_x, approach_y)]

    print(f"\n[nav] Path: {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        dist = math.sqrt(wp[0] ** 2 + wp[1] ** 2)
        angle = math.degrees(math.atan2(wp[0], wp[1]))
        print(f"  {i + 1}. ({wp[0]:.2f}, {wp[1]:.2f}) dist={dist:.2f}m heading={angle:.0f}°")

    # Render map with path
    img = world._render_grid(path_waypoints=waypoints)
    cv2.imwrite("/tmp/nav_door_path.png", img)

    # Step 4: Follow path (auto-start with 3s countdown)
    print("\n[nav] Starting navigation in 3 seconds... (Ctrl+C to abort)")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n[nav] Step 3: Following path...")
    follower = PathFollower(rover, pose, detector, tracker)
    success = follower.follow(waypoints)

    if success:
        print("\n[nav] Reached door approach point!")
        # Take a frame to see what's ahead
        time.sleep(0.5)
        jpeg = tracker.get_jpeg()
        if jpeg:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                dets = detector.detect(frame)
                detector.draw(frame, dets)
                cv2.imwrite("/tmp/nav_at_door.jpg", frame)
                print("[nav] Arrival frame saved to /tmp/nav_at_door.jpg")
    else:
        print("\n[nav] Navigation stopped (obstacle or path issue)")

    # Cleanup
    rover.send({"T": 1, "L": 0, "R": 0})
    rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    cap.release()
    ser.close()
    print("\n[nav] Done.")


if __name__ == "__main__":
    main()
