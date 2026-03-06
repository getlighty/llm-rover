"""room_scanner.py — LLM-based vector room scan + room guess.

Runs a gimbal sweep, asks the vision model to estimate distances/sizes of
visible scene elements, projects them into rover-centric XY, and outputs a
compact vector map for UI + room-context inference.
"""

import math
import threading
import time

import room_context

SCAN_PANS = [-140, -95, -50, 0, 50, 95, 140]
SCAN_TILT = 0
GIMBAL_SETTLE_S = 0.45


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _to_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class VectorRoomScanner:
    """Vector room scanner using LLM-estimated distance + object dimensions."""

    def __init__(self, rover, tracker, vision_json_fn, log_fn=None):
        self.rover = rover
        self.tracker = tracker
        self.vision_json_fn = vision_json_fn  # fn(prompt, jpeg_bytes) -> dict|None
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._last_scan = {
            "scan_ts": 0.0,
            "task": "",
            "elements": [],
            "scene_summary": "",
            "room_guess": {"name": None, "confidence": 0.0, "reason": ""},
            "candidates": [],
        }

    def _log(self, msg):
        if self.log_fn:
            try:
                self.log_fn(msg)
            except Exception:
                pass

    def get_state(self):
        with self._lock:
            return dict(self._last_scan)

    def scan_room(self, task_text="", body_yaw_deg=0.0, find_target=None):
        """Run a panoramic gimbal scan and return vector map state.

        Args:
            find_target: if set, stop sweep early when this object/landmark
                         is spotted. Sets 'target_found' in returned state.
        """
        t0 = time.time()
        observations = []
        scene_bits = []
        target_found_pan = None

        self._log("Room scan: starting vector sweep")
        for pan in SCAN_PANS:
            self.rover.send({"T": 133, "X": pan, "Y": SCAN_TILT,
                             "SPD": 300, "ACC": 20})
            time.sleep(GIMBAL_SETTLE_S)

            jpeg = self._get_scan_jpeg()
            if not jpeg:
                continue

            frame_data = self._estimate_frame(task_text, pan, jpeg)
            if not frame_data:
                continue

            scene = str(frame_data.get("scene_summary", "")).strip()
            if scene:
                scene_bits.append(scene)

            elems = frame_data.get("elements", [])
            if not isinstance(elems, list):
                continue
            for elem in elems[:8]:
                obs = self._normalize_observation(elem, pan, body_yaw_deg)
                if obs:
                    observations.append(obs)

            # Early exit: target spotted during sweep
            if find_target and target_found_pan is None:
                target_lower = find_target.lower()
                scene_lower = scene.lower()
                elem_names = " ".join(
                    str(e.get("name", "")).lower() for e in elems)
                if (target_lower in scene_lower or
                        target_lower in elem_names):
                    target_found_pan = pan
                    self._log(f"Room scan: TARGET '{find_target}' spotted "
                              f"at pan={pan}° — stopping sweep")
                    break

        # Return gimbal to center after scan.
        self.rover.send({"T": 133, "X": 0, "Y": 0, "SPD": 250, "ACC": 20})

        merged = self._merge_observations(observations)
        room_guess, candidates, scene_summary = self._guess_room(
            task_text, merged, scene_bits)

        state = {
            "scan_ts": time.time(),
            "scan_duration_s": round(time.time() - t0, 2),
            "task": task_text,
            "elements": merged,
            "scene_summary": scene_summary,
            "room_guess": room_guess,
            "candidates": candidates,
        }
        if target_found_pan is not None:
            state["target_found"] = find_target
            state["target_pan"] = target_found_pan
        with self._lock:
            self._last_scan = state

        self._log(
            f"Room scan: {len(merged)} elements, "
            f"guess={room_guess.get('name') or 'unknown'} "
            f"({room_guess.get('confidence', 0):.2f})")
        return state

    def _get_scan_jpeg(self):
        if hasattr(self.tracker, "get_overlay_jpeg"):
            jpg = self.tracker.get_overlay_jpeg()
            if jpg:
                return jpg
        if hasattr(self.tracker, "get_jpeg"):
            return self.tracker.get_jpeg()
        return None

    def _estimate_frame(self, task_text, pan_deg, jpeg):
        prompt = (
            f"You are estimating a rover room scan. "
            f"Current camera pan={pan_deg} degrees (0=forward, left negative, right positive). "
            f"Task context: '{task_text or 'none'}'.\n"
            f"Estimate distances and physical sizes from this frame.\n"
            f"Reply ONLY JSON with this schema:\n"
            f'{{"scene_summary":"1 short sentence",'
            f'"elements":[{{"name":"object name","bearing_deg":-30..30,'
            f'"distance_m":0.2..8.0,"width_m":0.05..4.0,'
            f'"depth_m":0.05..4.0,"height_m":0.05..3.0,'
            f'"confidence":0.0..1.0}}]}}\n'
            f"Rules: include up to 6 most useful room landmarks/obstacles/openings; "
            f"use best-effort metric estimates."
        )
        data = self.vision_json_fn(prompt, jpeg)
        return data if isinstance(data, dict) else None

    def _normalize_observation(self, elem, pan_deg, body_yaw_deg):
        if not isinstance(elem, dict):
            return None
        name = str(elem.get("name", "")).strip().lower()
        if not name:
            return None

        dist = _clamp(_to_float(elem.get("distance_m", 1.5), 1.5), 0.2, 8.0)
        bearing = _clamp(_to_float(elem.get("bearing_deg", 0.0), 0.0), -45.0, 45.0)
        conf = _clamp(_to_float(elem.get("confidence", 0.45), 0.45), 0.05, 1.0)

        width_m = _clamp(_to_float(elem.get("width_m", 0.4), 0.4), 0.05, 4.0)
        depth_m = _clamp(_to_float(elem.get("depth_m", width_m * 0.7), width_m * 0.7),
                         0.05, 4.0)
        height_m = _clamp(_to_float(elem.get("height_m", 0.7), 0.7), 0.05, 3.0)

        world_bearing = body_yaw_deg + pan_deg + bearing
        rad = math.radians(world_bearing)
        x = dist * math.sin(rad)
        y = dist * math.cos(rad)
        return {
            "name": name[:48],
            "x": x,
            "y": y,
            "distance_m": dist,
            "bearing_deg": world_bearing,
            "width_m": width_m,
            "depth_m": depth_m,
            "height_m": height_m,
            "confidence": conf,
            "_w": conf,
            "_samples": 1,
        }

    def _merge_observations(self, observations):
        merged = []
        for obs in observations:
            match = None
            for cur in merged:
                if cur["name"] != obs["name"]:
                    continue
                dx = cur["x"] - obs["x"]
                dy = cur["y"] - obs["y"]
                if math.hypot(dx, dy) < 0.75:
                    match = cur
                    break
            if not match:
                merged.append(dict(obs))
                continue

            w_old = match["_w"]
            w_new = obs["_w"]
            w_tot = w_old + w_new
            for key in ("x", "y", "distance_m", "bearing_deg",
                        "width_m", "depth_m", "height_m"):
                match[key] = (match[key] * w_old + obs[key] * w_new) / w_tot
            match["_w"] = w_tot
            match["_samples"] += 1
            match["confidence"] = max(match["confidence"], obs["confidence"])

        cleaned = []
        for m in merged:
            cleaned.append({
                "name": m["name"],
                "x": round(m["x"], 2),
                "y": round(m["y"], 2),
                "distance_m": round(m["distance_m"], 2),
                "bearing_deg": round(((m["bearing_deg"] + 180) % 360) - 180, 1),
                "width_m": round(m["width_m"], 2),
                "depth_m": round(m["depth_m"], 2),
                "height_m": round(m["height_m"], 2),
                "confidence": round(m["confidence"], 2),
                "samples": m["_samples"],
            })
        cleaned.sort(key=lambda e: (-e["confidence"], e["distance_m"]))
        return cleaned[:40]

    def _guess_room(self, task_text, elements, scene_bits):
        scene_parts = [s for s in scene_bits if s]
        top = elements[:10]
        if top:
            vec = ", ".join(
                f"{e['name']} {e['distance_m']:.1f}m"
                f" (size {e['width_m']:.1f}x{e['depth_m']:.1f}m)"
                for e in top
            )
            scene_parts.append(f"Vector scan landmarks: {vec}.")
        if task_text:
            scene_parts.append(f"Task context: {task_text}.")
        scene_summary = " ".join(scene_parts)[:1000]

        yolo_like = ", ".join(f"{e['name']}({int(e['confidence']*100)}%)"
                              for e in top[:8])
        room_name, conf = room_context.get_current_room(scene_summary, yolo_like)
        ranked = room_context.identify_room(scene_summary, yolo_like)
        candidates = [
            {"name": n, "score": s, "confidence": c}
            for n, s, c in ranked[:4]
        ]

        reason = "No strong room match from current scan."
        if room_name:
            hits = [e["name"] for e in top[:5]]
            reason = ("Matched room clues from panoramic vector scan: "
                      + ", ".join(hits))
        return (
            {"name": room_name, "confidence": conf, "reason": reason},
            candidates,
            scene_summary,
        )
