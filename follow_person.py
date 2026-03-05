"""follow_person.py — Person follower (YOLO visual servo).

Same smooth gimbal feel as the old HumanTracker face tracker in
rover_brain.py, plus proportional wheel driving to maintain distance.
YOLO detection drives the 10Hz control loop directly — no LLM in the loop.

Distance control uses bbox width directly — no fragile distance estimation.
Bigger bbox = closer = slow down. Smaller bbox = farther = speed up.

Importable:
    from follow_person import follow
    result = follow(ser, cam, imu, duration=60, target_bw=0.25)

Standalone:
    python3 follow_person.py --duration 30
"""

import math
import json
import time

# ── Gimbal P-control (matches old HumanTracker gains) ────────────────
GIMBAL_GAIN = 40.0        # degrees per unit error (fast response)
GIMBAL_DEADZONE = 0.05    # fraction of frame
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90

# ── Drive control (bbox-width based) ─────────────────────────────────
# Target bbox width — when person/legs fills this fraction of frame, stop.
# ~0.25 ≈ person at ~2m with 65° FOV. Works for "legs" too.
TARGET_BW = 0.25
BW_DEADZONE = 0.05       # no drive when within ±0.05 of target_bw
FWD_GAIN = 5.0            # speed per unit bw error (person too small → drive)
REV_GAIN = 3.0            # speed per unit bw error (person too big → reverse)
STEER_GAIN = 0.005        # differential per degree of pan
MAX_STEER = 0.15          # max differential applied to wheels
CENTERING_TOL = 0.15      # don't drive until person is roughly centered

# ── Person labels ────────────────────────────────────────────────────
PERSON_LABELS = ("person", "legs", "human")

LOOP_HZ = 10
LOST_TIMEOUT = 3.0        # seconds without detection → stop
INITIAL_TILT = 30         # tilt up to see person (rover is low to the ground)
MIN_BW = 0.03             # ignore tiny detections (noise)


def follow(ser, cam, imu, duration=60.0, target_bw=TARGET_BW,
           stop_event=None, log_fn=None):
    """Follow a person for up to `duration` seconds.

    Two-stage control:
      1. Gimbal P-control keeps person centered in frame
      2. Wheels drive forward/back based on bbox width,
         steer from gimbal pan offset

    Args:
        ser:         RoverSerial
        cam:         Camera (has .get_detections(), .detector.find())
        imu:         IMUPoller or None
        duration:    max seconds to follow
        target_bw:   target bbox width fraction (default 0.25)
        stop_event:  threading.Event — set to abort
        log_fn:      optional callback(msg)

    Returns:
        dict with status, duration_tracked, avg_bw
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(f"[follow] {msg}")

    pan = 0.0
    tilt = float(INITIAL_TILT)
    found_ever = False
    last_seen = 0.0
    spun = False
    t_start = time.time()
    bw_sum = 0.0
    bw_count = 0

    # Centre gimbal, tilt up to see person
    ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT, "SPD": 200, "ACC": 10})
    time.sleep(0.3)
    log(f"Following person for {duration}s, target_bw={target_bw}")

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration:
                break
            if stop_event and stop_event.is_set():
                break

            dets, _summary, _age = cam.get_detections()
            # Match person OR legs (YOLO-World often sees legs from low camera)
            match = None
            if dets:
                for label in PERSON_LABELS:
                    match = cam.detector.find(label, dets)
                    if match:
                        break

            if match is None or match["bw"] < MIN_BW:
                # Stop wheels immediately when person not visible
                ser.send({"T": 1, "L": 0, "R": 0})
                lost_secs = time.time() - last_seen if found_ever else elapsed
                if lost_secs >= LOST_TIMEOUT and not spun:
                    # Spin in place to find the person
                    log("Person lost — spinning to search")
                    ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT,
                              "SPD": 200, "ACC": 10})
                    pan = 0.0
                    found = _spin_search(ser, cam, stop_event)
                    spun = True
                    if found:
                        log("Found person during spin")
                        last_seen = time.time()
                        continue
                    else:
                        log("Person not found after spin")
                        return _result("lost", elapsed, bw_sum, bw_count)
                if not found_ever and elapsed >= LOST_TIMEOUT:
                    log("Person not found within 3s")
                    return _result("not_found", elapsed, bw_sum, bw_count)
                time.sleep(1.0 / LOOP_HZ)
                continue

            # ── Person detected ──────────────────────────────────────
            found_ever = True
            last_seen = time.time()
            spun = False  # reset so we can spin again next time we lose them

            cx = match["cx"]
            cy = match["cy"]
            bw = match["bw"]
            bw_sum += bw
            bw_count += 1

            # ── Stage 1: Gimbal P-control (old face tracker feel) ────
            err_x = cx - 0.5
            err_y = cy - 0.5

            if abs(err_x) > GIMBAL_DEADZONE:
                pan += err_x * GIMBAL_GAIN
            if abs(err_y) > GIMBAL_DEADZONE:
                tilt -= err_y * GIMBAL_GAIN
            pan = max(PAN_MIN, min(PAN_MAX, pan))
            tilt = max(TILT_MIN, min(TILT_MAX, tilt))

            # Dynamic gimbal speed — proportional to error
            dist_err = math.sqrt(err_x ** 2 + err_y ** 2)
            spd = max(50, int(dist_err * 500))
            acc = max(10, int(dist_err * 200))
            ser.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1),
                      "SPD": spd, "ACC": acc})

            # ── Stage 2: Wheel driving ───────────────────────────────
            # Don't drive until person is roughly centered (let gimbal catch up)
            if abs(err_x) > CENTERING_TOL:
                ser.send({"T": 1, "L": 0, "R": 0})
                time.sleep(1.0 / LOOP_HZ)
                continue

            # bbox width error: positive = too far (bw too small), negative = too close
            bw_error = target_bw - bw

            if abs(bw_error) < BW_DEADZONE:
                speed = 0.0
            elif bw_error > 0:
                # Too far — drive forward
                speed = bw_error * FWD_GAIN
            else:
                # Too close — reverse
                speed = bw_error * REV_GAIN

            # Steering from gimbal pan
            steer = pan * STEER_GAIN
            steer = max(-MAX_STEER, min(MAX_STEER, steer))
            left = speed + steer
            right = speed - steer

            ser.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})

            # Log every ~1s
            frame_num = int(elapsed * LOOP_HZ)
            if frame_num % LOOP_HZ == 0:
                log(f"bw={bw:.2f} target={target_bw:.2f} speed={speed:.2f} pan={pan:.1f}° [{match['name']}]")

            time.sleep(1.0 / LOOP_HZ)

    finally:
        # Always stop wheels and reset gimbal
        ser.send({"T": 1, "L": 0, "R": 0})
        ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})

    log(f"Followed person for {elapsed:.1f}s")
    return _result("completed", elapsed, bw_sum, bw_count)


def _spin_search(ser, cam, stop_event):
    """Spin 360° in place, checking for person at each step.
    Returns True if person found, False if full rotation with nothing."""
    SPIN_SPEED = 0.35
    STEP_TIME = 0.3       # seconds per step
    # ~205°/s at 0.35 → need about 360/205 ≈ 1.75s → ~6 steps
    STEPS = 12            # overshoot to be safe
    for _ in range(STEPS):
        if stop_event and stop_event.is_set():
            ser.send({"T": 1, "L": 0, "R": 0})
            return False
        ser.send({"T": 1, "L": SPIN_SPEED, "R": -SPIN_SPEED})
        time.sleep(STEP_TIME)
        ser.send({"T": 1, "L": 0, "R": 0})
        time.sleep(0.15)  # let detection catch up
        dets, _, _ = cam.get_detections()
        if dets:
            for label in PERSON_LABELS:
                match = cam.detector.find(label, dets)
                if match and match["bw"] >= MIN_BW:
                    return True
    ser.send({"T": 1, "L": 0, "R": 0})
    return False


def _result(status, duration, bw_sum, bw_count):
    avg_bw = round(bw_sum / bw_count, 3) if bw_count > 0 else 0.0
    return {
        "status": status,
        "duration_tracked": round(duration, 1),
        "avg_bw": avg_bw,
    }


# ── Standalone CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Follow a person")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target-bw", type=float, default=TARGET_BW)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", default="/dev/ttyTHS0")
    args = parser.parse_args()

    from rover_brain_llm import RoverSerial, Camera

    ser = RoverSerial(args.port)
    cam = Camera(args.camera)
    cam.start()
    time.sleep(1.0)

    try:
        result = follow(ser, cam, None, duration=args.duration,
                        target_bw=args.target_bw)
        print(json.dumps(result, indent=2))
    finally:
        cam.stop()
        ser.close()
