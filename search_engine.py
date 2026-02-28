"""
Simplified Search & Navigate — Waveshare UGV Rover
===================================================
Replaces the 4-phase search system with a single priority-scan loop.
Drops iterative LLM centering in favor of geometry-based alignment.
Uses absolute world headings throughout (no spatial map translation).

Key changes from original:
  - 4 phases → 1 loop over priority-sorted positions
  - 3-call centering → 1 confirmation call + geometry
  - Spatial map stores world_pan (absolute), never translated
  - Body rotation is just "turn to face world_pan", not a separate phase
  - Navigation uses gimbal-as-tracker: gimbal tracks object, body follows gimbal
"""

import time
import json
import math
import difflib
import cv2
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────────────────────
# CONSTANTS (tune these on your rover)
# ──────────────────────────────────────────────────────────────

STALE_SECONDS       = 300      # 5 min spatial map freshness
GIMBAL_SETTLE       = 0.5      # seconds after gimbal move
GIMBAL_SPD          = 300      # gimbal motor speed (0-500)
GIMBAL_ACC          = 20       # gimbal acceleration
FRAME_DIFF_THRESH   = 0.20     # 20% pixel change = new scene
MAX_SCAN_POSITIONS  = 20       # hard cap on LLM calls per search
TURN_SPEED          = 0.35     # m/s wheel speed during rotation
TURN_RATE_DPS       = 120.0    # degrees per second (calibrate!)
NAV_SPEED           = 0.12     # m/s forward creep
NAV_STEER           = 0.06     # m/s steering differential
NAV_CHECK_INTERVAL  = 1.0      # seconds between nav LLM checks
NAV_MAX_STEPS       = 30       # max navigation iterations
CLOSE_THRESHOLD     = 5.0      # degrees — body alignment threshold


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def normalize_angle(deg: float) -> float:
    """Normalize to [-180, +180]."""
    return ((deg + 180) % 360) - 180


def frame_changed(jpeg_a: bytes, jpeg_b: bytes,
                  threshold=FRAME_DIFF_THRESH) -> bool:
    """Quick check if two frames are meaningfully different (80x60 grayscale)."""
    if jpeg_a is None or jpeg_b is None:
        return True
    try:
        a = cv2.imdecode(np.frombuffer(jpeg_a, np.uint8), cv2.IMREAD_GRAYSCALE)
        b = cv2.imdecode(np.frombuffer(jpeg_b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return True
        a = cv2.resize(a, (80, 60))
        b = cv2.resize(b, (80, 60))
        diff = cv2.absdiff(a, b)
        pct = float(diff.sum()) / (255.0 * 80 * 60)
        return pct > threshold
    except Exception:
        return True


def load_prompts(path):
    """Load prompts/{name}.md → dict of section → template."""
    text = open(path).read()
    sections = {}
    current = None
    for line in text.split('\n'):
        if line.startswith('## '):
            current = line[3:].strip().lower()
            sections[current] = ''
        elif current is not None:
            sections[current] += line + '\n'
    return {k: v.strip() for k, v in sections.items()}


# ──────────────────────────────────────────────────────────────
# SCAN POSITION GENERATOR — replaces 4 phases with 1 priority list
# ──────────────────────────────────────────────────────────────

def generate_scan_positions(
    hint_world_pan: Optional[float] = None,
    hints: list = None,
    body_yaw: float = 0.0,
) -> list:
    """
    Return gimbal (pan, tilt) positions in priority order.
    Replaces Phases 2/3/4 with a single ordered list.

    Priority:
      1. Spatial map hint (stale object position, converted to relative pan)
      2. Scout: [-90, 0, +90] at tilt 0 (wide coverage fast)
      3. LLM directional hints from scout results
      4. Intermediates: [-45, +45, -135, +135] at tilt 0
      5. Floor sweep: same angles at tilt 30 (objects on ground)
    """
    positions = []
    seen = set()

    def _add(pan, tilt):
        key = (int(pan), int(tilt))
        if key not in seen:
            seen.add(key)
            positions.append((pan, tilt))

    # 1. Spatial map hint — convert world heading to relative gimbal angle
    if hint_world_pan is not None:
        relative = normalize_angle(hint_world_pan - body_yaw)
        if -180 <= relative <= 180:
            _add(relative, 0)

    # 2. Scout positions (wide coverage)
    for pan in [-90, 0, 90]:
        _add(pan, 0)

    # 3. LLM hint-based positions
    hint_map = {"left": -135, "right": 135, "behind": 180}
    for hint in (hints or []):
        if hint in hint_map:
            _add(hint_map[hint], 0)

    # 4. Intermediate fill
    for pan in [-45, 45, -135, 135]:
        _add(pan, 0)

    # 5. Floor-level sweep (tilt 30 = looking down)
    for pan in [-135, -90, -45, 0, 45, 90, 135]:
        _add(pan, 30)

    return positions


# ──────────────────────────────────────────────────────────────
# SEARCH ENGINE — single unified loop
# ──────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Finds objects using the gimbal camera + LLM vision.

    Instead of 4 discrete phases, runs a single loop over
    priority-sorted positions. Stops as soon as the target is found.
    Falls back to body rotation only if the entire front hemisphere fails.
    """

    def __init__(self, rover, tracker, pose, spatial_map,
                 llm_vision_fn, parse_fn, voice_fn=None, prompts=None):
        """
        Args:
            rover: Serial command sender (rover.send(json_dict))
            tracker: Camera frame provider (.get_jpeg() -> bytes, .pause(s))
            pose: Pose tracker with .body_yaw, .cam_pan, .cam_tilt, .world_pan
            spatial_map: SpatialMap with .find(), .update(), .is_stale()
            llm_vision_fn: Callable(prompt: str, jpeg: bytes) -> raw string
            parse_fn: Callable(raw: str) -> dict or None (JSON parser)
            voice_fn: Optional callable(text: str) for speech output
            prompts: Dict of prompt templates from load_prompts()
        """
        self.rover = rover
        self.tracker = tracker
        self.pose = pose
        self.spatial_map = spatial_map
        self._llm_raw = llm_vision_fn
        self._parse = parse_fn
        self.voice = voice_fn or (lambda t: None)
        self.prompts = prompts or {}

    def _llm(self, prompt, jpeg):
        """Call LLM vision and parse response. Returns dict or None."""
        try:
            raw = self._llm_raw(prompt, jpeg)
            return self._parse(raw)
        except Exception as e:
            print(f"[search] LLM error: {e}")
            return None

    # ── PUBLIC API ─────────────────────────────────────────────

    def search(self, target: str) -> bool:
        """
        Find target object. Returns True if found and body aligned.

        Flow:
          1. Check spatial map (0 LLM calls)
          2. Scan front hemisphere with priority positions
          3. If not found, rotate body 180° and scan again
        """
        print(f"[search] Starting smart search for '{target}'")
        self.voice(f"Searching for {target}.")
        self.tracker.pause(300)

        # ── Step 1: Spatial map lookup (free) ──
        name, entry = self.spatial_map.find(target)
        hint_world_pan = None

        if entry and not self.spatial_map.is_stale(entry):
            # Fresh hit — just look there
            wp = entry.get("world_pan")
            if wp is not None:
                relative_pan = normalize_angle(wp - self.pose.body_yaw)
            else:
                relative_pan = entry.get("pan", 0)
            self._move_gimbal(relative_pan, entry["tilt"])
            result = self._check_frame(target, "check")
            if result:
                found_flag = result.get("found", False)
                if not found_flag:
                    for obj in result.get("objects", []):
                        if target.lower() in obj.lower() or obj.lower() in target.lower():
                            found_flag = True
                            break
                if found_flag:
                    print(f"[search] Phase 1: fresh spatial map hit for '{name}'")
                    return self._finalize(target)
            # Not actually there anymore — fall through to scan
            print(f"[search] Spatial map hit but not confirmed, scanning")

        elif entry:
            # Stale hit — use as priority hint
            hint_world_pan = entry.get("world_pan")
            if hint_world_pan is None:
                hint_world_pan = self.pose.body_yaw + entry.get("pan", 0)
            print(f"[search] Stale hint at world_pan={hint_world_pan:.0f}°")

        # ── Step 2: Priority scan (front hemisphere) ──
        found, hints = self._scan_hemisphere(target, hint_world_pan)
        if found:
            return True

        # ── Step 3: Rotate 180° and scan rear hemisphere ──
        print("[search] Front hemisphere clear, rotating 180°")
        self._rotate_body(180)
        found, _ = self._scan_hemisphere(target, hint_world_pan, extra_hints=hints)
        if found:
            return True

        print(f"[search] '{target}' not found after full search")
        self.voice(f"Couldn't find {target}.")
        self._move_gimbal(0, 0)
        return False

    def navigate_to(self, target: str) -> bool:
        """
        Find target, align body, then drive toward it with LLM steering.
        Returns True if reached or max steps hit, False if lost.
        """
        print(f"[nav] Navigate to '{target}'")

        # Find it first (search aligns body too)
        if not self.search(target):
            return False

        # Drive loop
        self.voice(f"Going to {target}.")
        self._send_wheels(NAV_SPEED, NAV_SPEED)
        prev_jpeg = None

        for step in range(NAV_MAX_STEPS):
            time.sleep(NAV_CHECK_INTERVAL)

            jpeg = self.tracker.get_jpeg()
            if not jpeg:
                print("[nav] No camera frame")
                self._send_wheels(0, 0)
                break

            if not frame_changed(prev_jpeg, jpeg):
                print(f"[nav] Step {step}: frame unchanged, keeping course")
                continue  # Scene unchanged, stay on course
            prev_jpeg = jpeg

            result = self._check_frame(target, "steer")
            if result is None:
                continue

            visible = result.get("v", result.get("visible", False))
            close = result.get("close", False)
            direction = result.get("dir", result.get("direction", "center"))
            size = result.get("size", "small")
            print(f"[nav] Step {step}: vis={visible} close={close} "
                  f"dir={direction} size={size}")

            if close:
                self._send_wheels(0, 0)
                print(f"[nav] Arrived at {target}!")
                self.voice(f"I'm at the {target}.")
                return True

            if not visible:
                self._send_wheels(0, 0)
                print(f"[nav] Lost {target}, stopping")
                self.voice(f"I lost sight of {target}.")
                return False

            # Steer based on direction
            if direction == "left":
                self._send_wheels(NAV_SPEED - NAV_STEER,
                                  NAV_SPEED + NAV_STEER)
            elif direction == "right":
                self._send_wheels(NAV_SPEED + NAV_STEER,
                                  NAV_SPEED - NAV_STEER)
            else:
                self._send_wheels(NAV_SPEED, NAV_SPEED)

        # Max steps reached
        self._send_wheels(0, 0)
        print("[nav] Max steps reached")
        self.voice(f"Drove toward {target} as far as I could.")
        return True

    # ── SCAN LOOP (replaces phases 2/3/4) ──────────────────────

    def _scan_hemisphere(self, target, hint_world_pan=None, extra_hints=None):
        """
        Scan current hemisphere with priority-sorted positions.
        Returns (found: bool, hints: list of directional clues).
        """
        positions = generate_scan_positions(
            hint_world_pan=hint_world_pan,
            hints=extra_hints,
            body_yaw=self.pose.body_yaw,
        )
        checked = set()
        hints_collected = []
        llm_calls = 0

        for pan, tilt in positions:
            if llm_calls >= MAX_SCAN_POSITIONS:
                print(f"[search] Hit max scan positions ({MAX_SCAN_POSITIONS})")
                break

            key = (int(pan), int(tilt))
            if key in checked:
                continue
            checked.add(key)

            self._move_gimbal(pan, tilt)
            result = self._check_frame(target, "scout")
            llm_calls += 1

            if result is None:
                continue

            # Map everything we see (absolute world heading)
            objects = result.get("objects", [])
            if objects:
                wp = round(self.pose.world_pan, 1)
                self.spatial_map.update(objects, wp, tilt)
                print(f"[search] Mapped at world_pan={wp}°: {objects}")

            # Check if found — either LLM flagged it, or target appears
            # in the objects list (LLM sometimes lists it but forgets to
            # set found=true)
            found_flag = result.get("found", False)
            if not found_flag and objects:
                tgt = target.lower()
                for obj in objects:
                    if tgt in obj.lower() or obj.lower() in tgt:
                        found_flag = True
                        print(f"[search] Target '{target}' detected in objects "
                              f"list as '{obj}' (LLM missed found=true)")
                        break

            if found_flag:
                print(f"[search] FOUND '{target}' at pan={pan} tilt={tilt} "
                      f"(llm_call #{llm_calls})")
                if self._finalize(target):
                    print(f"[search] Done in {llm_calls} LLM calls")
                    return True, hints_collected

            # Collect directional hints for next pass
            hint = result.get("hint", "unknown")
            if hint != "unknown":
                hints_collected.append(hint)
                print(f"[search] Hint at pan={pan}: {hint}")

        return False, hints_collected

    # ── CENTERING + ALIGNMENT ──────────────────────────────────

    def _finalize(self, target):
        """
        Confirm target visible, 1 LLM centering call max, align body.
        """
        self.voice(f"Found {target}.")

        # One confirmation + centering call
        jpeg = self.tracker.get_jpeg()
        if jpeg:
            prompt = self._build_prompt(
                "center", target=target,
                pan=self.pose.cam_pan, tilt=self.pose.cam_tilt)
            result = self._llm(prompt, jpeg)

            if result and not result.get("centered", True):
                cmds = result.get("commands", [])
                if cmds:
                    cmd = cmds[0]
                    self._move_gimbal(
                        cmd.get("X", self.pose.cam_pan),
                        cmd.get("Y", self.pose.cam_tilt),
                        spd=cmd.get("SPD", 100))

        # Align body to face where gimbal is pointing (pure geometry)
        self._align_body_to_gimbal()

        # Save final position with absolute heading
        wp = round(self.pose.world_pan, 1)
        self.spatial_map.update([target], wp, self.pose.cam_tilt)
        print(f"[search] Saved '{target}' at world_pan={wp}°")
        return True

    def _align_body_to_gimbal(self):
        """
        Rotate body to face where the gimbal is pointing.
        Simultaneously counter-rotate gimbal to keep view stable.
        No LLM calls — pure geometry.
        """
        if abs(self.pose.cam_pan) <= CLOSE_THRESHOLD:
            return  # Already aligned

        degrees = self.pose.cam_pan
        duration = max(0.2, min(5.0, abs(degrees) / TURN_RATE_DPS))
        sign = 1 if degrees > 0 else -1

        # Wheels turn body, gimbal counter-rotates to stay on target
        gimbal_spd = max(50, int(abs(degrees) / duration))
        print(f"[search] Aligning body {degrees:.0f}° ({duration:.1f}s)")
        self.rover.send({"T": 1, "L": TURN_SPEED * sign,
                         "R": -TURN_SPEED * sign})
        self.rover.send({"T": 133, "X": 0, "Y": self.pose.cam_tilt,
                         "SPD": gimbal_spd, "ACC": GIMBAL_ACC})
        time.sleep(duration)
        self._send_wheels(0, 0)
        time.sleep(0.3)

        # Update pose — body yaw changes, gimbal resets to center
        self.pose.after_body_turn(degrees)
        self.pose.cam_pan = 0
        print("[search] Body aligned, gimbal centered")

    # ── BODY ROTATION ──────────────────────────────────────────

    def _rotate_body(self, degrees):
        """
        Rotate body by N degrees. Gimbal centers during rotation.
        No spatial map translation needed — we store world_pan.
        """
        duration = abs(degrees) / TURN_RATE_DPS
        sign = 1 if degrees > 0 else -1

        self._move_gimbal(0, 0)  # Center gimbal first
        self.rover.send({"T": 1, "L": -TURN_SPEED * sign,
                         "R": TURN_SPEED * sign})
        time.sleep(duration)
        self._send_wheels(0, 0)
        time.sleep(0.3)

        self.pose.after_body_turn(degrees)
        print(f"[search] Rotated body {degrees}°, "
              f"new heading={self.pose.body_yaw:.0f}°")

    # ── HARDWARE COMMANDS ──────────────────────────────────────

    def _move_gimbal(self, pan, tilt, spd=GIMBAL_SPD):
        """Move gimbal and wait for settle."""
        pan = max(-180, min(180, pan))
        tilt = max(-45, min(90, tilt))
        self.rover.send({"T": 133, "X": round(pan, 1), "Y": round(tilt, 1),
                         "SPD": spd, "ACC": GIMBAL_ACC})
        self.pose.cam_pan = pan
        self.pose.cam_tilt = tilt
        time.sleep(GIMBAL_SETTLE)

    def _send_wheels(self, left, right):
        """Send wheel velocity command."""
        self.rover.send({"T": 1, "L": round(left, 2),
                         "R": round(right, 2)})

    # ── LLM INTERFACE ──────────────────────────────────────────

    def _check_frame(self, target, prompt_key):
        """Capture frame, build prompt, call LLM, return parsed result."""
        jpeg = self.tracker.get_jpeg()
        if jpeg is None:
            return None
        prompt = self._build_prompt(
            prompt_key, target=target,
            pan=self.pose.cam_pan, tilt=self.pose.cam_tilt,
            world_pan=round(self.pose.world_pan, 1))
        return self._llm(prompt, jpeg)

    def _build_prompt(self, key, **kwargs):
        """Format a prompt template with kwargs. Falls back to generic."""
        template = self.prompts.get(key)
        if template:
            try:
                return template.format(**kwargs)
            except KeyError:
                pass  # Fall through to hardcoded

        # Fallback prompts (no external file needed)
        fallbacks = {
            "scout": (
                "I'm a ground rover searching for: {target}.\n"
                "Camera at pan={pan}°, tilt={tilt}°.\n"
                "Reply ONLY JSON: "
                '{{"found": true/false, "objects": ["all","visible","objects"], '
                '"hint": "left"/"right"/"behind"/"unknown"}}'
            ),
            "check": (
                "I'm searching for: {target}. "
                "Camera at pan={pan}°, tilt={tilt}°.\n"
                "Is it visible? Reply ONLY JSON: "
                '{{"found": true/false, "objects": ["all","visible","objects"]}}'
            ),
            "center": (
                "You see '{target}'. Gimbal at pan={pan}°, tilt={tilt}°.\n"
                "Is it centered? Reply ONLY JSON:\n"
                'If yes: {{"centered": true}}\n'
                'If no: {{"centered": false, "commands": '
                '[{{"T":133,"X":<pan>,"Y":<tilt>,"SPD":100,"ACC":10}}]}}'
            ),
            "steer": (
                "I'm driving toward '{target}'. Camera pan={pan}°.\n"
                "Reply ONLY JSON: "
                '{{"v": true/false, "close": true/false, '
                '"dir": "left"/"right"/"center", '
                '"size": "small"/"medium"/"large"}}'
            ),
        }
        template = fallbacks.get(key, fallbacks["check"])
        return template.format(**kwargs)
