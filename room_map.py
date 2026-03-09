"""Narrative room map for the rover — LLM-first spatial memory.

Instead of computing XY coordinates from bearing + distance (fragile with
imprecise odometry), the map stores natural-language observations that the
LLM produces and consumes directly.

Each observation is a short sentence describing what's around the rover.
The LLM generates these; the code only persists them.

Data is persisted to room_map.json and survives restarts.
"""

import json
import os
import threading
import time

MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "room_map.json")

STALE_SECONDS = 600  # 10 min — observations older than this are demoted
MAX_OBSERVATIONS = 30  # rolling buffer
MAX_OBJECTS = 50  # dedup object sightings


class RoomMap:
    """Narrative spatial memory — stores what the LLM sees in plain text.

    Old interface (record / room_json / nav_json / describe_room /
    describe_for_navigation) is preserved so callers don't break, but
    the internal representation is now a list of narrative observations
    instead of XY coordinates.
    """

    def __init__(self):
        self._observations = []   # [{text, room, time, objects}]
        self._objects = {}        # name -> {last_seen, room, description, seen_count}
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    # ── Recording ────────────────────────────────────────────────────

    def record_narrative(self, text, room=None, objects=None):
        """Store a narrative observation from the LLM.

        Args:
            text: natural-language description of what is visible
            room: current room name (if known)
            objects: list of object names spotted
        """
        if not text or len(text.strip()) < 5:
            return
        with self._lock:
            self._observations.append({
                "text": text.strip()[:300],
                "room": room or "",
                "time": time.time(),
                "objects": list(objects or []),
            })
            if len(self._observations) > MAX_OBSERVATIONS:
                self._observations = self._observations[-MAX_OBSERVATIONS:]

            for name in (objects or []):
                name = name.lower().strip()
                if not name or len(name) < 2:
                    continue
                entry = self._objects.get(name, {
                    "last_seen": 0, "room": "", "description": "",
                    "seen_count": 0,
                })
                entry["last_seen"] = time.time()
                entry["room"] = room or entry.get("room", "")
                entry["seen_count"] = entry.get("seen_count", 0) + 1
                self._objects[name] = entry

            # Trim objects dict
            if len(self._objects) > MAX_OBJECTS:
                by_time = sorted(self._objects.items(),
                                 key=lambda kv: kv[1]["last_seen"])
                self._objects = dict(by_time[-MAX_OBJECTS:])

            self._dirty = True

    def record(self, detections, rover_x, rover_y, body_yaw, cam_pan, cam_tilt):
        """Legacy interface — convert YOLO detections into a narrative entry.

        Old callers pass structured detections; we just store the object
        names. No more trigonometry.
        """
        if not detections:
            return
        names = []
        for det in detections:
            name = det.get("name", "").lower().strip()
            if name and len(name) >= 2:
                names.append(name)
        if names:
            text = f"Detected: {', '.join(names)}"
            self.record_narrative(text, objects=names)

    # ── Query — what callers actually need ───────────────────────────

    def room_json(self, rover_x=0.0, rover_y=0.0, body_yaw=0.0, max_objects=15):
        """Return room context as dict for LLM prompt injection.

        Returns recent observations + known objects — no coordinates.
        """
        with self._lock:
            if not self._observations and not self._objects:
                return None

            now = time.time()
            recent = [o for o in self._observations
                      if (now - o["time"]) < STALE_SECONDS][-8:]

            fresh_objects = []
            for name, info in sorted(self._objects.items(),
                                     key=lambda kv: kv[1]["last_seen"],
                                     reverse=True)[:max_objects]:
                age = now - info["last_seen"]
                if age > STALE_SECONDS:
                    continue
                fresh_objects.append({
                    "name": name,
                    "room": info.get("room", ""),
                    "seen_count": info.get("seen_count", 1),
                    "age_s": int(age),
                })

            if not recent and not fresh_objects:
                return None

            return {
                "recent_observations": [o["text"] for o in recent],
                "known_objects": fresh_objects,
            }

    def nav_json(self, target, rover_x=0.0, rover_y=0.0, body_yaw=0.0):
        """Return what we know about a specific object — for navigation."""
        q = target.lower().strip()
        with self._lock:
            info = self._objects.get(q)
            if not info:
                # Fuzzy match
                for key, val in self._objects.items():
                    if q in key or key in q:
                        info = val
                        q = key
                        break
            if not info:
                return None

            age = int(time.time() - info["last_seen"])
            result = {
                "name": q,
                "room": info.get("room", ""),
                "seen_count": info.get("seen_count", 1),
                "age_s": age,
            }
            if age > STALE_SECONDS:
                result["warning"] = f"last seen {age}s ago — may have moved"

            # Find observations that mention this object
            mentions = [o["text"] for o in self._observations
                        if q in " ".join(o.get("objects", [])).lower()][-3:]
            if mentions:
                result["recent_sightings"] = mentions
            return result

    def find(self, query):
        """Find an object by name. Returns (name, info_dict) or (None, None).

        Kept for compatibility — returns a dict instead of RoomObject.
        """
        q = query.lower().strip()
        with self._lock:
            if q in self._objects:
                return q, self._objects[q]
            for key, val in self._objects.items():
                if q in key or key in q:
                    return key, val
        return None, None

    def describe_room(self, rover_x=0.0, rover_y=0.0, body_yaw=0.0, max_objects=20):
        """Generate natural language room description for LLM prompt."""
        with self._lock:
            if not self._observations and not self._objects:
                return ""

            now = time.time()
            recent = [o for o in self._observations
                      if (now - o["time"]) < STALE_SECONDS][-6:]
            stale = [o for o in self._observations
                     if (now - o["time"]) >= STALE_SECONDS][-3:]

            lines = ["## Room Map"]
            if recent:
                lines.append("Recent observations:")
                for obs in recent:
                    age = int(now - obs["time"])
                    age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
                    lines.append(f"  - {obs['text']} ({age_str})")

            # Fresh objects summary
            fresh = [(n, i) for n, i in self._objects.items()
                     if (now - i["last_seen"]) < STALE_SECONDS]
            fresh.sort(key=lambda x: x[1]["last_seen"], reverse=True)
            if fresh:
                names = [f"{n} (in {i.get('room', '?')})"
                         if i.get("room") else n
                         for n, i in fresh[:max_objects]]
                lines.append(f"Objects nearby: {', '.join(names)}")

            if stale:
                lines.append("Previously seen:")
                for obs in stale:
                    lines.append(f"  - {obs['text']}")

            return "\n".join(lines)

    def describe_for_navigation(self, target, rover_x=0.0, rover_y=0.0,
                                body_yaw=0.0):
        """Focused description for navigating to a specific target."""
        nav = self.nav_json(target)
        if not nav:
            return f"'{target}' not in room map. Survey the room first."
        lines = [f"Navigation to '{nav['name']}':"]
        if nav.get("room"):
            lines.append(f"  Last seen in: {nav['room']}")
        lines.append(f"  Seen {nav['seen_count']}x, last {nav['age_s']}s ago")
        if nav.get("warning"):
            lines.append(f"  WARNING: {nav['warning']}")
        if nav.get("recent_sightings"):
            lines.append("  Recent sightings:")
            for s in nav["recent_sightings"]:
                lines.append(f"    - {s}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self):
        """Persist to disk."""
        with self._lock:
            if not self._dirty:
                return
            data = {
                "observations": self._observations,
                "objects": self._objects,
                "saved": time.time(),
            }
            self._dirty = False
        try:
            with open(MAP_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[room_map] Save error: {e}")

    def _load(self):
        try:
            with open(MAP_FILE) as f:
                data = json.load(f)
            self._observations = data.get("observations", [])
            self._objects = data.get("objects", {})
            # Migrate from old XY-based format
            if "objects" in data and isinstance(data["objects"], dict):
                first_val = next(iter(data["objects"].values()), None)
                if isinstance(first_val, dict) and "x" in first_val:
                    # Old format — convert
                    self._objects = {}
                    for name, obj in data["objects"].items():
                        self._objects[name] = {
                            "last_seen": obj.get("last_seen", 0),
                            "room": "",
                            "description": "",
                            "seen_count": obj.get("seen_count", 1),
                        }
                    self._observations = []
                    self._dirty = True
                    print(f"[room_map] Migrated {len(self._objects)} objects "
                          f"from XY format to narrative format")
                    return
            if self._objects:
                print(f"[room_map] Loaded {len(self._objects)} objects, "
                      f"{len(self._observations)} observations")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def object_count(self):
        with self._lock:
            return len(self._objects)

    def clear(self):
        """Clear all map data."""
        with self._lock:
            self._objects.clear()
            self._observations.clear()
            self._dirty = True
        self.save()
        print("[room_map] Cleared")
