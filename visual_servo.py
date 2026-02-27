"""
Visual servoing: navigate toward a detected object using local detection + proportional control.

Architecture follows NVIDIA JetBot object_following pattern:
  - Proportional steering from bounding box center offset
  - Two-stage: gimbal tracks object, wheels steer based on gimbal pan
  - Stops when bbox width exceeds approach threshold

Usage:
    servo = VisualServo(rover, detector, tracker)
    success = servo.approach("cup")
"""

import cv2
import numpy as np
import time
import threading


class VisualServo:
    """Drive toward a detected object using local YOLOv8 + proportional control.

    Two-stage control (from Waveshare ugv_jetson gimbal_track pattern):
    1. Gimbal tracking: P-control keeps the object centered in frame
    2. Wheel driving: forward speed + differential steering from gimbal pan offset
    """

    def __init__(self, rover, detector, tracker):
        """
        Args:
            rover: RoverSerial instance (sends JSON commands to ESP32)
            detector: LocalDetector instance (runs YOLOv8)
            tracker: HumanTracker instance (owns camera, provides get_jpeg())
        """
        self.rover = rover
        self.detector = detector
        self.tracker = tracker

        # --- Gimbal P-control gains ---
        self.gimbal_gain_x = 40.0   # degrees per unit error (0..1 range)
        self.gimbal_gain_y = 30.0
        self.gimbal_deadzone = 0.05  # fraction of frame, below this don't move gimbal

        # --- Wheel control ---
        self.drive_speed = 0.20       # m/s base forward speed (20% of max, per Ovi's preference)
        self.steer_gain = 0.005       # wheel speed diff per degree of gimbal pan
        self.max_steer = 0.15         # max differential applied to wheels

        # --- Approach thresholds ---
        self.approach_bw = 0.45       # stop when bbox width > this fraction of frame
        self.min_detect_bw = 0.03     # ignore detections smaller than this (noise)
        self.centering_tolerance = 0.15  # object must be this centered before driving
        self.arrive_confirm = 5       # need this many consecutive frames above threshold

        # --- State ---
        self.pan = 0.0
        self.tilt = 0.0
        self._running = False
        self._lost_count = 0
        self._arrive_count = 0        # consecutive frames above approach_bw
        self.MAX_LOST = 20   # frames without detection before giving up (~2s at 10Hz)
        self.MAX_STEPS = 300  # max frames (~30s at 10Hz)

    def approach(self, target_name, voice=None):
        """Navigate toward target_name. Blocks until arrival, loss, or abort.

        Returns:
            True if arrived (bbox large enough)
            False if lost object or max steps reached
        """
        self._running = True
        self._lost_count = 0
        self._arrive_count = 0
        self.pan = 0.0
        self.tilt = 0.0
        step = 0

        # Center gimbal
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
        time.sleep(0.5)

        print(f"[servo] Approaching '{target_name}'")
        if voice:
            voice.speak(f"Going to {target_name}.")

        while self._running and step < self.MAX_STEPS:
            step += 1

            # Get frame
            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                time.sleep(0.05)
                continue

            # Decode JPEG -> numpy BGR
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run detection
            detections = self.detector.detect(frame)
            target = self.detector.find(target_name, detections)

            if target is None:
                self._lost_count += 1
                if self._lost_count > self.MAX_LOST:
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    print(f"[servo] Lost '{target_name}' ({self.MAX_LOST} frames)")
                    if voice:
                        voice.speak(f"Lost {target_name}.")
                    return False
                # Slow down while lost but don't stop immediately
                self.rover.send({"T": 1, "L": 0, "R": 0})
                time.sleep(0.1)
                continue

            self._lost_count = 0
            cx = target["cx"]  # 0..1
            cy = target["cy"]  # 0..1
            bw = target["bw"]  # bbox width fraction

            # Skip tiny detections
            if bw < self.min_detect_bw:
                continue

            # --- Check arrival (require consecutive frames to filter noise) ---
            if bw >= self.approach_bw:
                self._arrive_count += 1
                if self._arrive_count >= self.arrive_confirm:
                    self.rover.send({"T": 1, "L": 0, "R": 0})
                    dist = target.get("dist_m", "?")
                    print(f"[servo] Arrived at '{target_name}' (bw={bw:.2f}, dist={dist}m)")
                    if voice:
                        voice.speak(f"I'm at the {target_name}.")
                    return True
            else:
                self._arrive_count = 0

            # --- Stage 1: Gimbal tracking (keep object centered) ---
            err_x = cx - 0.5   # positive = object is right of center
            err_y = cy - 0.5   # positive = object is below center

            if abs(err_x) > self.gimbal_deadzone:
                self.pan += err_x * self.gimbal_gain_x
            if abs(err_y) > self.gimbal_deadzone:
                self.tilt -= err_y * self.gimbal_gain_y  # tilt inverted

            self.pan = max(-150, min(150, self.pan))
            self.tilt = max(-30, min(90, self.tilt))

            self.rover.send({"T": 133, "X": round(self.pan, 1),
                             "Y": round(self.tilt, 1), "SPD": 200, "ACC": 10})

            # --- Stage 2: Wheel driving ---
            if abs(err_x) < self.centering_tolerance:
                # Object roughly centered — drive forward, steer by gimbal pan
                steer = self.pan * self.steer_gain
                steer = max(-self.max_steer, min(self.max_steer, steer))
                L = self.drive_speed + steer
                R = self.drive_speed - steer
                L = max(-0.3, min(0.5, L))
                R = max(-0.3, min(0.5, R))
                self.rover.send({"T": 1, "L": round(L, 3), "R": round(R, 3)})
            else:
                # Object too far off-center — stop wheels, let gimbal catch up
                self.rover.send({"T": 1, "L": 0, "R": 0})

            # Log every 10 frames
            if step % 10 == 0:
                dist = target.get("dist_m", "?")
                print(f"[servo] step={step} target={target_name} cx={cx:.2f} "
                      f"bw={bw:.2f} pan={self.pan:.0f} dist={dist}m")

            time.sleep(0.1)  # ~10Hz control loop

        # Max steps
        self.rover.send({"T": 1, "L": 0, "R": 0})
        print(f"[servo] Max steps ({self.MAX_STEPS}) reached")
        if voice:
            voice.speak(f"Couldn't reach {target_name}.")
        return False

    def stop(self):
        """Abort the approach."""
        self._running = False

    def scan_and_find(self, target_name, voice=None):
        """Sweep gimbal to find target, then approach it.

        Combines systematic gimbal sweep with local detection (no LLM needed).
        Returns True if found and arrived, False otherwise.
        """
        PAN_STEPS = [-120, -80, -40, 0, 40, 80, 120]
        TILT_STEPS = [0, 25]

        print(f"[servo] Scanning for '{target_name}'")
        if voice:
            voice.speak(f"Looking for {target_name}.")

        for tilt in TILT_STEPS:
            for pan in PAN_STEPS:
                if not self._running:
                    return False

                self.rover.send({"T": 133, "X": pan, "Y": tilt,
                                 "SPD": 300, "ACC": 20})
                time.sleep(0.6)  # let gimbal settle

                # Capture and detect
                jpeg = self.tracker.get_jpeg()
                if not jpeg:
                    continue
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                detections = self.detector.detect(frame)
                target = self.detector.find(target_name, detections)

                if target:
                    bw = target["bw"]
                    dist = target.get("dist_m", "?")
                    print(f"[servo] Found '{target_name}' at pan={pan} tilt={tilt} "
                          f"bw={bw:.2f} dist={dist}m")
                    if voice:
                        voice.speak(f"Found {target_name}.")
                    # Set gimbal state and start approach
                    self.pan = pan
                    self.tilt = tilt
                    return self.approach(target_name, voice=voice)

                # Log what we did see
                if detections:
                    names = [d["name"] for d in detections]
                    print(f"[servo] pan={pan} tilt={tilt}: {names} (no {target_name})")

        # Not found in sweep — try rotating body 180 and sweep again
        print(f"[servo] '{target_name}' not found forward, rotating 180deg")
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 300, "ACC": 20})
        time.sleep(0.3)
        self.rover.send({"T": 1, "L": -0.35, "R": 0.35})
        time.sleep(1.5)  # ~180 degrees at 120 deg/s
        self.rover.send({"T": 1, "L": 0, "R": 0})
        time.sleep(0.5)

        for tilt in TILT_STEPS:
            for pan in PAN_STEPS:
                if not self._running:
                    return False

                self.rover.send({"T": 133, "X": pan, "Y": tilt,
                                 "SPD": 300, "ACC": 20})
                time.sleep(0.6)

                jpeg = self.tracker.get_jpeg()
                if not jpeg:
                    continue
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                detections = self.detector.detect(frame)
                target = self.detector.find(target_name, detections)

                if target:
                    print(f"[servo] Found '{target_name}' at pan={pan} tilt={tilt} (rear)")
                    if voice:
                        voice.speak(f"Found {target_name}.")
                    self.pan = pan
                    self.tilt = tilt
                    return self.approach(target_name, voice=voice)

        print(f"[servo] '{target_name}' not found after 360 sweep")
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
        if voice:
            voice.speak(f"Couldn't find {target_name}.")
        return False
