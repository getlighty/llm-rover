"""follow_target.py — Smart target follower (YOLO visual servo).

Follows any YOLO-detectable target (person, dog, car, bike, etc.) with:
  - Identity tracking (HSV histogram + position proximity)
  - Voice conversation when stationary (mic mute/unmute)
  - Collision detection via floor_nav + IMU
  - LLM-generated callouts when target lost
  - LLM visual verification + auto-labeling on lock-on

Same smooth gimbal feel as the old HumanTracker face tracker in
rover_brain.py, plus proportional wheel driving to maintain distance.
YOLO detection drives the 10Hz control loop directly — no LLM in the loop.

Distance control uses bbox width directly — no fragile distance estimation.
Bigger bbox = closer = slow down. Smaller bbox = farther = speed up.

Importable:
    from follow_target import follow
    result = follow("person", ser, cam, imu, duration=60, target_bw=0.25)

Standalone:
    python3 follow_target.py --target person --duration 30
"""

import cv2
import math
import json
import time
import random
import threading
import numpy as np

# ── Gimbal P-control (matches old HumanTracker gains) ────────────────
GIMBAL_GAIN = 40.0        # degrees per unit error (fast response)
GIMBAL_DEADZONE = 0.05    # fraction of frame
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 45  # cap at 45° — beyond that is ceiling

# ── Drive control (bbox-width based) ─────────────────────────────────
TARGET_BW = 0.25
BW_DEADZONE = 0.05       # no drive when within ±0.05 of target_bw
FWD_GAIN = 5.0            # speed per unit bw error (person too small → drive)
REV_GAIN = 3.0            # speed per unit bw error (person too big → reverse)
STEER_GAIN = 0.005        # differential per degree of pan
MAX_STEER = 0.15          # max differential applied to wheels
CENTERING_TOL = 0.15      # don't drive until target is roughly centered

LOOP_HZ = 10
LOST_TIMEOUT = 3.0        # seconds without detection → callout + spin
INITIAL_TILT = 30         # tilt up to see target (rover is low to the ground)
MIN_BW = 0.03             # ignore tiny detections (noise)

# ── Bump detection via accelerometer ──────────────────────────────────
BUMP_ACCEL_THRESHOLD = 0.5  # accel magnitude spike indicating a bump/collision
BUMP_COOLDOWN_S = 2.0       # ignore bumps for this long after realignment

# ── Histogram identity tracking ──────────────────────────────────────
HIST_BINS = 16            # HSV histogram bins per channel
HIST_SIMILARITY_WEIGHT = 0.3
POSITION_WEIGHT = 0.7

# ── Fallback callout lines (when no LLM available) ──────────────────
_CALLOUT_LINES = [
    "Where'd you go?",
    "Hey, come back!",
    "I lost you!",
    "Where are you?",
    "Don't leave me!",
    "Come back here!",
]


def _get_target_labels(target):
    """Return tuple of YOLO labels to match for this target."""
    t = target.lower()
    if t in ("person", "human", "me"):
        return ("person", "legs", "human")
    return (t,)


def _compute_histogram(jpeg_bytes, bbox):
    """Compute HSV 16x16 histogram of target crop from JPEG bytes."""
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        # bbox is normalized (cx, cy, bw, bh) — convert to pixel coords
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [HIST_BINS, HIST_BINS],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    except Exception:
        return None


def _compare_histogram(a, b):
    """Compare two histograms via correlation. Returns 0.0-1.0."""
    if a is None or b is None:
        return 0.0
    val = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
    return max(0.0, val)


def _find_target(dets, labels, cam):
    """Find best detection matching target labels. Returns detection dict or None."""
    for label in labels:
        match = cam.detector.find(label, dets)
        if match and match["bw"] >= MIN_BW:
            return match
    return None


def _find_all_targets(dets, labels):
    """Find ALL detections matching target labels. Returns list of dets."""
    matches = []
    for d in dets:
        if d["name"].lower() in labels and d.get("bw", 0) >= MIN_BW:
            matches.append(d)
    return matches


def _find_by_identity(dets, labels, last_cx, last_cy, ref_hist, jpeg_bytes):
    """Identity-aware target matching using position + histogram.

    Single match: return it directly (position is enough).
    Multiple matches: score by weighted position proximity + histogram similarity.
    """
    candidates = _find_all_targets(dets, labels)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates — use identity scoring
    best = None
    best_score = -1.0
    for c in candidates:
        # Position proximity (inverse distance, normalized to 0-1)
        dx = c["cx"] - last_cx
        dy = c["cy"] - last_cy
        dist = math.sqrt(dx * dx + dy * dy)
        pos_score = max(0.0, 1.0 - dist * 2.0)  # 0.5 apart → 0.0

        # Histogram similarity
        hist_score = 0.0
        if ref_hist is not None and jpeg_bytes is not None:
            bbox = (c["cx"], c["cy"], c["bw"], c.get("bh", c["bw"]))
            c_hist = _compute_histogram(jpeg_bytes, bbox)
            hist_score = _compare_histogram(ref_hist, c_hist)

        score = POSITION_WEIGHT * pos_score + HIST_SIMILARITY_WEIGHT * hist_score
        if score > best_score:
            best_score = score
            best = c
    return best


def _set_driving(voice, driving):
    """Mute mic when driving, unmute when stationary."""
    if voice is None:
        return
    try:
        if hasattr(voice, '_audio') and voice._audio is not None:
            voice._audio._mic_muted = driving
    except Exception:
        pass


def _check_bump(imu, ser, log):
    """Check for bump/collision via accelerometer.

    Reads accel magnitude from IMU poller if available, otherwise
    tries to read T:1001 directly from serial.

    Returns True if bump detected, False otherwise.
    """
    mag = None

    # Try IMU poller first
    if imu is not None:
        try:
            result = imu.check_tilt()
            if isinstance(result, tuple):
                status, m = result
                mag = m
            elif result == "stop":
                mag = 99.0  # force bump
        except Exception:
            pass

    # Fallback: read accel from serial T:1001 stream
    if mag is None and ser is not None:
        try:
            fb = ser.read_imu() if hasattr(ser, 'read_imu') else None
            if fb and isinstance(fb, dict):
                ax = fb.get("ax", 0)
                ay = fb.get("ay", 0)
                az = fb.get("az", 0)
                mag = (ax**2 + ay**2 + az**2) ** 0.5
        except Exception:
            pass

    if mag is not None and mag >= BUMP_ACCEL_THRESHOLD:
        log(f"BUMP detected: accel={mag:.2f}")
        return True
    return False


def _check_collision(imu, floor_nav, dets, log, target_labels=None):
    """Check for collision via IMU tilt and floor_nav obstacle detection.
    Excludes detections matching target_labels from obstacle check
    (the target IS what we're following, not an obstacle).
    Returns "tilt", "obstacle", or None."""
    if imu is not None:
        try:
            tilt_result = imu.check_tilt()
            if isinstance(tilt_result, tuple) and tilt_result[0] == "stop":
                log("Collision: IMU accel spike")
                return "tilt"
            elif tilt_result == "stop":
                log("Collision: IMU accel spike")
                return "tilt"
        except Exception:
            pass
    if floor_nav is not None:
        try:
            # Filter out the follow target from obstacle detection
            filtered = dets
            if target_labels and dets:
                filtered = [d for d in dets
                            if d["name"].lower() not in target_labels]
            clear, _ = floor_nav.check_floor_clear(filtered, 640, 480)
            if not clear:
                log("Obstacle: floor obstacle")
                return "obstacle"
        except Exception:
            pass
    return None


def follow(target, ser, cam, imu, duration=60.0, target_bw=TARGET_BW,
           stop_event=None, log_fn=None,
           voice=None, floor_nav=None, recovery_fn=None,
           speak_fn=None, llm_fn=None, label_override_fn=None):
    """Follow a target for up to `duration` seconds.

    Two-stage control:
      1. Gimbal P-control keeps target centered in frame
      2. Wheels drive forward/back based on bbox width,
         steer from gimbal pan offset

    Args:
        target:       label string ("person", "dog", "car", etc.)
        ser:          RoverSerial
        cam:          Camera (has .get_detections(), .detector.find())
        imu:          IMUPoller or None
        duration:     max seconds to follow
        target_bw:    target bbox width fraction (default 0.25)
        stop_event:   threading.Event — set to abort
        log_fn:       optional callback(msg)
        voice:        ElevenLabsVoice — toggle _mic_muted
        floor_nav:    FloorNavigator — check_floor_clear()
        recovery_fn:  callable(cam, ser, log) -> bool
        speak_fn:     callable(text) — TTS
        llm_fn:       callable(prompt, jpeg|None) -> str
        label_override_fn:  callable(yolo_label, correct_label)

    Returns:
        dict with status, duration_tracked, avg_bw
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(f"[follow] {msg}")

    labels = _get_target_labels(target)
    pan = 0.0
    tilt = float(INITIAL_TILT)
    found_ever = False
    last_seen = 0.0
    spun = False
    t_start = time.time()
    bw_sum = 0.0
    bw_count = 0
    callout_done = False  # only callout once per lost episode

    # Identity tracking state
    ref_hist = None
    last_cx = 0.5
    last_cy = 0.5
    locked = False
    verified = False

    # Auto-enable YOLO if detector available
    if hasattr(cam, '_yolo_enabled'):
        cam._yolo_enabled = True
        log("YOLO auto-enabled for follow mode")

    # Centre gimbal, tilt up to see target
    ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT, "SPD": 200, "ACC": 10})
    time.sleep(0.3)
    last_bump_time = 0.0  # cooldown tracking
    log(f"Following {target} for {duration}s, target_bw={target_bw}")

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration:
                break
            if stop_event and stop_event.is_set():
                break

            dets, _summary, _age = cam.get_detections()

            # ── Target matching (simple label match) ─────────────────
            match = None
            if dets:
                match = _find_target(dets, labels, cam)

            if match is None or match["bw"] < MIN_BW:
                # Stop wheels immediately when target not visible
                ser.send({"T": 1, "L": 0, "R": 0})
                _set_driving(voice, False)

                lost_secs = time.time() - last_seen if found_ever else elapsed
                if lost_secs >= LOST_TIMEOUT and not spun:
                    # ── Voice callout (non-blocking) ─────────────
                    if not callout_done:
                        callout_done = True

                        def _do_callout():
                            callout_text = None
                            if llm_fn:
                                try:
                                    prompt = (
                                        f"The {target} I was following disappeared. "
                                        f"Generate a single short playful sentence "
                                        f"calling out to it. Just the sentence, nothing else."
                                    )
                                    callout_text = llm_fn(prompt, None)
                                except Exception:
                                    pass
                            if not callout_text:
                                callout_text = random.choice(_CALLOUT_LINES)
                            if speak_fn:
                                try:
                                    speak_fn(callout_text)
                                except Exception:
                                    pass
                            log(f"Callout: {callout_text}")

                        threading.Thread(target=_do_callout, daemon=True).start()

                    # ── Spin search ────────────────────────────────
                    log(f"{target.title()} lost — spinning to search")
                    ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT,
                              "SPD": 200, "ACC": 10})
                    pan = 0.0
                    tilt = float(INITIAL_TILT)
                    found_det = _spin_search(ser, cam, labels, stop_event)
                    spun = True
                    if found_det:
                        log(f"Found {target} during spin")
                        last_seen = time.time()
                        pan = 0.0
                        tilt = float(INITIAL_TILT)
                        ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT,
                                  "SPD": 200, "ACC": 10})
                        last_cx = found_det["cx"]
                        last_cy = found_det["cy"]
                        callout_done = False
                        continue
                    else:
                        log(f"{target.title()} not found after spin")
                        return _result("lost", elapsed, bw_sum, bw_count)
                if not found_ever and elapsed >= LOST_TIMEOUT:
                    log(f"{target.title()} not found within 3s")
                    return _result("not_found", elapsed, bw_sum, bw_count)
                time.sleep(1.0 / LOOP_HZ)
                continue

            # ── Target detected ────────────────────────────────────
            found_ever = True
            last_seen = time.time()
            spun = False
            callout_done = False

            cx = match["cx"]
            cy = match["cy"]
            bw = match["bw"]
            bw_sum += bw
            bw_count += 1
            last_cx = cx
            last_cy = cy

            # Identity tracking disabled for now

            # ── Collision check ────────────────────────────────────
            collision = _check_collision(imu, floor_nav, dets, log,
                                         target_labels=labels)
            if collision:
                ser.send({"T": 1, "L": 0, "R": 0})
                _set_driving(voice, False)
                if recovery_fn:
                    try:
                        recovered = recovery_fn(cam, ser, log)
                        if recovered:
                            log("Recovery succeeded, resuming follow")
                            locked = False  # re-lock after recovery
                            ref_hist = None
                            pan = 0.0      # reset gimbal state
                            tilt = float(INITIAL_TILT)
                            ser.send({"T": 133, "X": 0, "Y": INITIAL_TILT,
                                      "SPD": 200, "ACC": 10})
                            continue
                        else:
                            log("Recovery failed, aborting")
                            return _result("blocked", elapsed,
                                           bw_sum, bw_count)
                    except Exception as e:
                        log(f"Recovery error: {e}")
                        return _result("blocked", elapsed, bw_sum, bw_count)
                else:
                    log("No recovery_fn, stopping")
                    return _result("blocked", elapsed, bw_sum, bw_count)

            # ── Stage 1: Gimbal P-control ──────────────────────────
            err_x = cx - 0.5
            err_y = cy - 0.5

            if abs(err_x) > GIMBAL_DEADZONE:
                pan += err_x * GIMBAL_GAIN
            if abs(err_y) > GIMBAL_DEADZONE:
                tilt -= err_y * GIMBAL_GAIN
            pan = max(PAN_MIN, min(PAN_MAX, pan))
            tilt = max(TILT_MIN, min(TILT_MAX, tilt))

            # Max gimbal speed — snap to position so servo keeps up with loop
            ser.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1),
                      "SPD": 500, "ACC": 200})

            # ── Stage 2: Wheel driving ─────────────────────────────
            # Don't drive until target is roughly centered
            if abs(err_x) > CENTERING_TOL:
                ser.send({"T": 1, "L": 0, "R": 0})
                _set_driving(voice, False)
                time.sleep(1.0 / LOOP_HZ)
                continue

            # bbox width error: positive = too far, negative = too close
            bw_error = target_bw - bw

            if abs(bw_error) < BW_DEADZONE:
                speed = 0.0
                _set_driving(voice, False)  # in deadzone, unmute
            elif bw_error > 0:
                speed = bw_error * FWD_GAIN
                _set_driving(voice, True)
            else:
                speed = bw_error * REV_GAIN
                _set_driving(voice, True)

            # Steering from gimbal pan
            steer = pan * STEER_GAIN
            steer = max(-MAX_STEER, min(MAX_STEER, steer))
            left = speed + steer
            right = speed - steer

            ser.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})

            # ── Bump detection: body hit something while driving ──
            if (speed != 0.0 and
                    time.time() - last_bump_time > BUMP_COOLDOWN_S and
                    _check_bump(imu, ser, log)):
                last_bump_time = time.time()
                # Stop wheels immediately
                ser.send({"T": 1, "L": 0, "R": 0})
                _set_driving(voice, False)

                # Body realignment: spin body to match gimbal pan,
                # keeping the camera pointed at the target
                if abs(pan) > 10:
                    log(f"Bump! Realigning body {pan:.0f}° to match camera")
                    # Spin body by current pan angle
                    body_turn_time = abs(pan) / 200.0  # ~200°/s at turn speed
                    if pan > 0:
                        ser.send({"T": 1, "L": 0.30, "R": -0.30})
                    else:
                        ser.send({"T": 1, "L": -0.30, "R": 0.30})
                    time.sleep(body_turn_time)
                    ser.send({"T": 1, "L": 0, "R": 0})
                    # Reset gimbal pan to 0 (body now faces where camera was)
                    pan = 0.0
                    ser.send({"T": 133, "X": 0, "Y": round(tilt, 1),
                              "SPD": 300, "ACC": 20})
                    time.sleep(0.3)
                    log("Body realigned — resuming follow")
                else:
                    log("Bump! Body already aligned, backing up slightly")
                    ser.send({"T": 1, "L": 0.08, "R": 0.08})
                    time.sleep(0.5)
                    ser.send({"T": 1, "L": 0, "R": 0})
                continue

            # Log every ~1s
            frame_num = int(elapsed * LOOP_HZ)
            if frame_num % LOOP_HZ == 0:
                id_tag = " [locked]" if locked else ""
                log(f"bw={bw:.2f} target={target_bw:.2f} "
                    f"speed={speed:.2f} pan={pan:.1f}° "
                    f"[{match['name']}]{id_tag}")

            time.sleep(1.0 / LOOP_HZ)

    finally:
        # Always stop wheels, reset gimbal, and unmute mic
        ser.send({"T": 1, "L": 0, "R": 0})
        ser.send({"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10})
        _set_driving(voice, False)

    log(f"Followed {target} for {elapsed:.1f}s")
    return _result("completed", elapsed, bw_sum, bw_count)


def _spin_search(ser, cam, labels, stop_event):
    """Spin 360° in place, checking for target at each step.
    Returns detection dict if found, None if full rotation with nothing."""
    SPIN_SPEED = 0.35
    STEP_TIME = 0.3
    STEPS = 12
    for _ in range(STEPS):
        if stop_event and stop_event.is_set():
            ser.send({"T": 1, "L": 0, "R": 0})
            return None
        ser.send({"T": 1, "L": SPIN_SPEED, "R": -SPIN_SPEED})
        time.sleep(STEP_TIME)
        ser.send({"T": 1, "L": 0, "R": 0})
        time.sleep(0.15)
        dets, _, _ = cam.get_detections()
        if dets:
            for label in labels:
                match = cam.detector.find(label, dets)
                if match and match["bw"] >= MIN_BW:
                    return match
    ser.send({"T": 1, "L": 0, "R": 0})
    return None


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

    parser = argparse.ArgumentParser(description="Follow a target")
    parser.add_argument("--target", type=str, default="person")
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
        result = follow(args.target, ser, cam, None,
                        duration=args.duration,
                        target_bw=args.target_bw)
        print(json.dumps(result, indent=2))
    finally:
        cam.stop()
        ser.close()
