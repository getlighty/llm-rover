"""track_object.py — YOLO + gimbal object tracker.

Locks the pan-tilt gimbal onto a named object using YOLO detections
and proportional control.  Gimbal only — no wheel driving.

Importable:
    from track_object import track
    result = track("cup", ser, cam, imu, duration=10)

Standalone:
    python3 track_object.py --target cup --duration 10
"""

import json
import time

# ── P-control gains (from visual_servo.py) ────────────────────────────
GAIN_X = 40.0       # degrees per unit error (0..1 range)
GAIN_Y = 30.0
DEADZONE = 0.05     # fraction of frame
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90
LOOP_HZ = 10
NOT_FOUND_TIMEOUT = 2.0   # seconds before "not_found"
LOST_FRAMES = 20          # consecutive misses before "lost"


def track(target, ser, cam, imu, duration=10.0, log_fn=None):
    """Track a named object with the gimbal for `duration` seconds.

    Args:
        target:   object label string (e.g. "cup", "person")
        ser:      RoverSerial — sends {T:133} gimbal commands
        cam:      Camera — has .get_detections(), .detector.find()
        imu:      IMUPoller — has .heading_deg, .state.voltage (or None)
        duration: seconds to track (default 10)
        log_fn:   optional callback(msg) for logging

    Returns:
        dict with status, heading_deg, gimbal_pan, gimbal_tilt, etc.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(f"[track] {msg}")

    pan = 0.0
    tilt = 0.0
    lost_count = 0
    found_ever = False
    last_conf = 0.0
    last_bw = 0.0
    t_start = time.time()

    # Center gimbal first
    ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
    time.sleep(0.3)

    log(f"Tracking '{target}' for {duration}s")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration:
            break

        # Get detections from camera's background detection loop
        dets, _summary, det_age = cam.get_detections()

        # Find target in detections
        match = cam.detector.find(target, dets) if dets else None

        if match is None:
            lost_count += 1
            # Not found timeout — never saw it
            if not found_ever and elapsed >= NOT_FOUND_TIMEOUT:
                log(f"'{target}' not found within {NOT_FOUND_TIMEOUT}s")
                _reset_gimbal(ser)
                return _result("not_found", target, elapsed, pan, tilt, imu,
                               last_conf, last_bw)
            # Lost — had it but lost for too many frames
            if found_ever and lost_count >= LOST_FRAMES:
                log(f"Lost '{target}' ({LOST_FRAMES} frames)")
                _reset_gimbal(ser)
                return _result("lost", target, elapsed, pan, tilt, imu,
                               last_conf, last_bw)
            time.sleep(1.0 / LOOP_HZ)
            continue

        # Target found
        found_ever = True
        lost_count = 0
        last_conf = match["conf"]
        last_bw = match["bw"]

        # Compute error (0.5 = center)
        err_x = match["cx"] - 0.5   # positive = object right of center
        err_y = match["cy"] - 0.5   # positive = object below center

        # Apply P-gain with deadzone
        if abs(err_x) > DEADZONE:
            pan += err_x * GAIN_X
        if abs(err_y) > DEADZONE:
            tilt -= err_y * GAIN_Y   # tilt inverted

        # Clamp
        pan = max(PAN_MIN, min(PAN_MAX, pan))
        tilt = max(TILT_MIN, min(TILT_MAX, tilt))

        # Send gimbal command
        ser.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1),
                  "SPD": 200, "ACC": 10})

        # Log every ~1 second
        frame_num = int(elapsed * LOOP_HZ)
        if frame_num % LOOP_HZ == 0:
            log(f"t={elapsed:.1f}s cx={match['cx']:.2f} cy={match['cy']:.2f} "
                f"pan={pan:.1f} tilt={tilt:.1f} conf={last_conf:.0%}")

        time.sleep(1.0 / LOOP_HZ)

    # Successfully tracked for full duration
    log(f"Tracked '{target}' for {duration}s, pan={pan:.1f}")
    return _result("tracked", target, duration, pan, tilt, imu,
                   last_conf, last_bw)


def _reset_gimbal(ser):
    """Return gimbal to center."""
    ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})


def _result(status, target, duration, pan, tilt, imu, conf, bw):
    """Build result dict."""
    # World heading = body heading + gimbal pan
    heading = None
    voltage = None
    if imu is not None:
        try:
            heading = round((imu.heading_deg + pan) % 360, 1)
        except Exception:
            pass
        try:
            voltage = round(float(imu.state.voltage), 1)
        except Exception:
            pass

    result = {
        "status": status,
        "target": target,
        "duration": round(duration, 1),
        "gimbal_pan": round(pan, 1),
        "gimbal_tilt": round(tilt, 1),
        "confidence": round(conf, 2),
        "bbox_width": round(bw, 3),
    }
    if heading is not None:
        result["heading_deg"] = heading
    if voltage is not None:
        result["voltage"] = voltage
    return result


# ── Standalone CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Track an object with the gimbal")
    parser.add_argument("--target", required=True, help="Object label to track")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds to track (default 10)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--port", default="/dev/ttyTHS0",
                        help="Serial port (default /dev/ttyTHS0)")
    args = parser.parse_args()

    # Import rover modules
    from rover_brain_llm import RoverSerial, Camera
    import imu as imu_mod

    ser = RoverSerial(args.port)
    cam = Camera(args.camera)
    cam.start()
    time.sleep(1.0)  # let detector warm up

    imu_poller = imu_mod.IMUPoller(ser, log_fn=lambda cat, msg: print(f"[{cat}] {msg}"))
    imu_poller.read_once()
    imu_poller.set_heading_offset()
    imu_poller.start()

    try:
        result = track(args.target, ser, cam, imu_poller, duration=args.duration)
        print(json.dumps(result, indent=2))
    finally:
        imu_poller.stop()
        cam.stop()
        ser.close()
