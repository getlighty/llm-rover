"""3D descriptive room map for the rover.

Builds a spatial graph of objects observed from different positions.
Each observation point records what was visible, at what bearing and distance.
Object positions are estimated in a 2D coordinate frame (X=right, Y=forward)
with rough height from tilt angle.

The map generates natural language room descriptions that can be injected
into LLM prompts so the rover understands spatial layout during navigation.

Data is persisted to room_map.json and survives restarts.
"""

import json
import math
import os
import time
import threading

MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "room_map.json")

STALE_SECONDS = 600  # 10 minutes — objects older than this are marked stale
MERGE_DISTANCE = 0.5  # meters — objects closer than this are merged


class RoomObject:
    """A single object with estimated 3D position."""

    def __init__(self, name, x, y, height=0.0, dist_m=None, confidence=0.5):
        self.name = name
        self.x = x              # meters, right of origin
        self.y = y              # meters, forward of origin
        self.height = height    # meters, rough estimate from tilt
        self.dist_m = dist_m    # raw distance from last observation
        self.confidence = confidence
        self.seen_count = 1
        self.last_seen = time.time()
        self.first_seen = time.time()
        self.observer_positions = []  # list of (ox, oy) where seen from

    def update(self, x, y, height, dist_m, confidence, observer_xy):
        """Merge a new observation with exponential moving average."""
        alpha = 0.4  # weight of new observation
        self.x = self.x * (1 - alpha) + x * alpha
        self.y = self.y * (1 - alpha) + y * alpha
        self.height = self.height * (1 - alpha) + height * alpha
        if dist_m is not None:
            self.dist_m = dist_m
        self.confidence = max(self.confidence, confidence)
        self.seen_count += 1
        self.last_seen = time.time()
        if observer_xy not in self.observer_positions[-3:]:
            self.observer_positions.append(observer_xy)
            if len(self.observer_positions) > 5:
                self.observer_positions = self.observer_positions[-5:]

    @property
    def is_stale(self):
        return (time.time() - self.last_seen) > STALE_SECONDS

    def to_dict(self):
        return {
            "name": self.name,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "height": round(self.height, 2),
            "dist_m": round(self.dist_m, 2) if self.dist_m else None,
            "confidence": round(self.confidence, 2),
            "seen_count": self.seen_count,
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "observer_positions": [(round(ox, 2), round(oy, 2))
                                   for ox, oy in self.observer_positions],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["name"], d["x"], d["y"],
                  d.get("height", 0), d.get("dist_m"),
                  d.get("confidence", 0.5))
        obj.seen_count = d.get("seen_count", 1)
        obj.last_seen = d.get("last_seen", 0)
        obj.first_seen = d.get("first_seen", obj.last_seen)
        obj.observer_positions = [tuple(p) for p in d.get("observer_positions", [])]
        return obj


class ObservationPoint:
    """A position where the rover took observations."""

    def __init__(self, x, y, heading, label=None):
        self.x = x
        self.y = y
        self.heading = heading  # body yaw at this point
        self.label = label or f"pos_{int(time.time()) % 10000}"
        self.time = time.time()
        self.objects_seen = []  # list of object names seen from here

    def to_dict(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading": round(self.heading, 1),
            "label": self.label,
            "time": self.time,
            "objects_seen": self.objects_seen,
        }

    @classmethod
    def from_dict(cls, d):
        pt = cls(d["x"], d["y"], d["heading"], d.get("label"))
        pt.time = d.get("time", 0)
        pt.objects_seen = d.get("objects_seen", [])
        return pt


class RoomMap:
    """3D descriptive map of objects in the room.

    Stores object positions estimated from bearing + distance observations
    taken at known rover positions. Generates natural language descriptions.
    """

    def __init__(self):
        self._objects = {}          # name -> RoomObject
        self._points = []           # list of ObservationPoint
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    def record(self, detections, rover_x, rover_y, body_yaw, cam_pan, cam_tilt):
        """Record objects detected from the current rover position.

        Args:
            detections: list of dicts with at minimum {"name": str}
                        optional: {"dist_m": float, "conf": float,
                                   "cx": float (0-1), "cy": float (0-1)}
            rover_x, rover_y: rover position in meters (from PoseTracker)
            body_yaw: rover body heading in degrees
            cam_pan: gimbal pan angle in degrees
            cam_tilt: gimbal tilt angle in degrees
        """
        if not detections:
            return

        with self._lock:
            observer_xy = (round(rover_x, 2), round(rover_y, 2))
            names_seen = []

            for det in detections:
                name = det.get("name", "").lower().strip()
                if not name or len(name) < 2:
                    continue

                dist_m = det.get("dist_m")
                conf = det.get("conf", 0.5)
                cx = det.get("cx", 0.5)  # horizontal center 0-1

                # Compute bearing: body_yaw + cam_pan + offset from frame center
                # cx=0.5 is center, FOV ~65 degrees
                fov_h = 65.0
                frame_offset_deg = (cx - 0.5) * fov_h
                world_bearing = body_yaw + cam_pan + frame_offset_deg

                if dist_m and dist_m > 0.1 and dist_m < 10.0:
                    # Compute world XY from bearing + distance
                    bearing_rad = math.radians(world_bearing)
                    obj_x = rover_x + dist_m * math.sin(bearing_rad)
                    obj_y = rover_y + dist_m * math.cos(bearing_rad)

                    # Rough height estimate from tilt angle
                    # Camera height ~0.15m, tilt down = negative
                    cam_height = 0.15
                    height = cam_height + dist_m * math.tan(math.radians(-cam_tilt))
                    height = max(0.0, min(3.0, height))  # clamp
                else:
                    # No distance — use bearing only, place at 2m default
                    dist_m_est = 2.0
                    bearing_rad = math.radians(world_bearing)
                    obj_x = rover_x + dist_m_est * math.sin(bearing_rad)
                    obj_y = rover_y + dist_m_est * math.cos(bearing_rad)
                    height = 0.5  # default mid-height
                    conf *= 0.5  # lower confidence without distance

                # Update or create object
                if name in self._objects:
                    existing = self._objects[name]
                    d = math.sqrt((existing.x - obj_x)**2 + (existing.y - obj_y)**2)
                    if d < MERGE_DISTANCE or existing.seen_count < 3:
                        existing.update(obj_x, obj_y, height, dist_m, conf, observer_xy)
                    else:
                        # Same name but far away — might be a different instance
                        # Keep the more confident one, or create numbered variant
                        if conf > existing.confidence:
                            existing.update(obj_x, obj_y, height, dist_m, conf, observer_xy)
                        else:
                            # Store as "name_2" etc
                            for i in range(2, 10):
                                alt = f"{name}_{i}"
                                if alt not in self._objects:
                                    self._objects[alt] = RoomObject(
                                        alt, obj_x, obj_y, height, dist_m, conf)
                                    self._objects[alt].observer_positions.append(observer_xy)
                                    break
                else:
                    self._objects[name] = RoomObject(
                        name, obj_x, obj_y, height, dist_m, conf)
                    self._objects[name].observer_positions.append(observer_xy)

                names_seen.append(name)

            # Record observation point
            if names_seen:
                pt = ObservationPoint(rover_x, rover_y, body_yaw)
                pt.objects_seen = names_seen
                self._points.append(pt)
                if len(self._points) > 50:
                    self._points = self._points[-50:]
                self._dirty = True

    def save(self):
        """Persist to disk (call periodically, not on every update)."""
        with self._lock:
            if not self._dirty:
                return
            data = {
                "objects": {k: v.to_dict() for k, v in self._objects.items()},
                "points": [p.to_dict() for p in self._points],
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
            for name, d in data.get("objects", {}).items():
                self._objects[name] = RoomObject.from_dict(d)
            for pd in data.get("points", []):
                self._points.append(ObservationPoint.from_dict(pd))
            if self._objects:
                print(f"[room_map] Loaded {len(self._objects)} objects, "
                      f"{len(self._points)} observation points")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def find(self, query):
        """Find an object by name. Returns (name, RoomObject) or (None, None)."""
        q = query.lower().strip()
        with self._lock:
            if q in self._objects:
                return q, self._objects[q]
            for key, obj in self._objects.items():
                if q in key or key in q:
                    return key, obj
        return None, None

    def direction_word(self, dx, dy):
        """Convert dx, dy offset to a compass-like direction word."""
        angle = math.degrees(math.atan2(dx, dy))  # 0=forward, 90=right
        if angle < 0:
            angle += 360
        if angle < 10 or angle >= 350:
            return "directly ahead"
        elif angle < 55:
            return "ahead-right"
        elif angle < 125:
            return "to the right"
        elif angle < 170:
            return "behind-right"
        elif angle < 190:
            return "directly behind"
        elif angle < 235:
            return "behind-left"
        elif angle < 305:
            return "to the left"
        else:
            return "ahead-left"

    def height_word(self, h):
        """Describe object height."""
        if h < 0.15:
            return "on the floor"
        elif h < 0.5:
            return "low"
        elif h < 1.0:
            return "at table height"
        elif h < 1.8:
            return "at eye level"
        else:
            return "high up"

    def _obj_relative(self, obj, rover_x, rover_y, body_yaw):
        """Compute relative position data for an object. Returns dict."""
        dx = obj.x - rover_x
        dy = obj.y - rover_y
        dist = math.sqrt(dx*dx + dy*dy)
        yaw_rad = math.radians(body_yaw)
        rel_dx = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
        rel_dy = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        bearing = math.degrees(math.atan2(dx, dy))
        turn = ((bearing - body_yaw + 180) % 360) - 180
        return {
            "name": obj.name,
            "dist_m": round(dist, 1),
            "direction": self.direction_word(rel_dx, rel_dy),
            "turn_deg": round(turn),
            "height": self.height_word(obj.height),
            "stale": obj.is_stale,
            "seen_count": obj.seen_count,
            "age_s": int(time.time() - obj.last_seen),
        }

    def room_json(self, rover_x=0.0, rover_y=0.0, body_yaw=0.0, max_objects=15):
        """Return structured room data as a dict (for JSON injection into nav prompts)."""
        with self._lock:
            if not self._objects:
                return None
            items = sorted(self._objects.values(),
                           key=lambda o: o.last_seen, reverse=True)
            fresh = [o for o in items if not o.is_stale][:max_objects]
            if not fresh:
                return None
            objects = [self._obj_relative(o, rover_x, rover_y, body_yaw)
                       for o in fresh]
            # Nearby pairs
            relationships = []
            for i, a in enumerate(fresh[:10]):
                for b in fresh[i+1:10]:
                    dx = b.x - a.x
                    dy = b.y - a.y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < 4.0:
                        relationships.append({
                            "a": a.name, "b": b.name,
                            "dist_m": round(d, 1),
                            "b_relative_to_a": self.direction_word(dx, dy),
                        })
                    if len(relationships) >= 8:
                        break
            return {"objects": objects, "relationships": relationships}

    def nav_json(self, target, rover_x=0.0, rover_y=0.0, body_yaw=0.0):
        """Return structured navigation data for a target as dict, or None."""
        name, obj = self.find(target)
        if not obj:
            return None
        with self._lock:
            info = self._obj_relative(obj, rover_x, rover_y, body_yaw)
            if obj.is_stale:
                info["warning"] = f"last seen {int(time.time() - obj.last_seen)}s ago"
            # Nearby landmarks
            nearby = []
            for other_name, other_obj in self._objects.items():
                if other_name == name or other_obj.is_stale:
                    continue
                odx = other_obj.x - obj.x
                ody = other_obj.y - obj.y
                odist = math.sqrt(odx*odx + ody*ody)
                if odist < 3.0:
                    nearby.append({
                        "name": other_name,
                        "dist_from_target_m": round(odist, 1),
                        "relative_to_target": self.direction_word(odx, ody),
                    })
            nearby.sort(key=lambda n: n["dist_from_target_m"])
            info["nearby_landmarks"] = nearby[:5]
            return info

    def describe_room(self, rover_x=0.0, rover_y=0.0, body_yaw=0.0, max_objects=20):
        """Generate natural language room description from rover's perspective.

        Args:
            rover_x, rover_y: current rover position
            body_yaw: current heading (0=initial forward)
            max_objects: max objects to include

        Returns:
            String description suitable for LLM prompt injection.
        """
        with self._lock:
            if not self._objects:
                return ""

            now = time.time()
            # Sort by recency, filter stale
            items = sorted(self._objects.values(),
                           key=lambda o: o.last_seen, reverse=True)

            # Separate fresh from stale
            fresh = [o for o in items if not o.is_stale][:max_objects]
            stale = [o for o in items if o.is_stale][:5]

            if not fresh and not stale:
                return ""

            lines = ["## Room Map"]

            if fresh:
                lines.append("Objects I can see or recently saw:")
                for obj in fresh:
                    dx = obj.x - rover_x
                    dy = obj.y - rover_y
                    dist = math.sqrt(dx*dx + dy*dy)

                    # Direction relative to current heading
                    # Rotate by -body_yaw to get relative direction
                    yaw_rad = math.radians(body_yaw)
                    rel_dx = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
                    rel_dy = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

                    direction = self.direction_word(rel_dx, rel_dy)
                    h_desc = self.height_word(obj.height)
                    age = int(now - obj.last_seen)
                    age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"

                    line = f"  - {obj.name}: {dist:.1f}m {direction}, {h_desc}"
                    if age > 10:
                        line += f" (seen {age_str})"
                    if obj.seen_count > 1:
                        line += f" [seen {obj.seen_count}x]"
                    lines.append(line)

            if stale:
                lines.append("Previously seen (may have moved):")
                for obj in stale:
                    dx = obj.x - rover_x
                    dy = obj.y - rover_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    yaw_rad = math.radians(body_yaw)
                    rel_dx = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
                    rel_dy = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
                    direction = self.direction_word(rel_dx, rel_dy)
                    lines.append(f"  - {obj.name}: was ~{dist:.1f}m {direction}")

            # Add spatial relationships between nearby objects
            relationships = self._describe_relationships(fresh[:10])
            if relationships:
                lines.append("Spatial relationships:")
                lines.extend(f"  - {r}" for r in relationships)

            return "\n".join(lines)

    def describe_for_navigation(self, target, rover_x=0.0, rover_y=0.0, body_yaw=0.0):
        """Focused description for navigating to a specific target.

        Returns string with target location and nearby landmarks.
        """
        name, obj = self.find(target)
        if not obj:
            return f"'{target}' not in room map. Survey the room first."

        with self._lock:
            dx = obj.x - rover_x
            dy = obj.y - rover_y
            dist = math.sqrt(dx*dx + dy*dy)

            yaw_rad = math.radians(body_yaw)
            rel_dx = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
            rel_dy = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            direction = self.direction_word(rel_dx, rel_dy)

            # Bearing to turn to
            target_bearing = math.degrees(math.atan2(dx, dy))
            turn = ((target_bearing - body_yaw + 180) % 360) - 180

            lines = [f"Navigation to '{name}':"]
            lines.append(f"  Distance: {dist:.1f}m {direction}")
            lines.append(f"  Turn: {turn:+.0f} degrees from current heading")
            lines.append(f"  Height: {self.height_word(obj.height)}")

            if obj.is_stale:
                lines.append(f"  WARNING: Last seen {int(time.time() - obj.last_seen)}s ago, may have moved")

            # Nearby landmarks for reference
            nearby = []
            for other_name, other_obj in self._objects.items():
                if other_name == name:
                    continue
                odx = other_obj.x - obj.x
                ody = other_obj.y - obj.y
                odist = math.sqrt(odx*odx + ody*ody)
                if odist < 3.0 and not other_obj.is_stale:
                    rel_dir = self.direction_word(odx, ody)
                    nearby.append((odist, other_name, rel_dir))

            if nearby:
                nearby.sort()
                lines.append("  Nearby landmarks:")
                for ndist, nname, ndir in nearby[:5]:
                    lines.append(f"    - {nname} is {ndist:.1f}m {ndir} of {name}")

            return "\n".join(lines)

    def _describe_relationships(self, objects):
        """Generate spatial relationship descriptions between objects."""
        if len(objects) < 2:
            return []
        relationships = []
        seen_pairs = set()
        for i, a in enumerate(objects):
            for b in objects[i+1:]:
                pair = tuple(sorted([a.name, b.name]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 4.0:  # only describe nearby pairs
                    direction = self.direction_word(dx, dy)
                    relationships.append(
                        f"{b.name} is {dist:.1f}m {direction} of {a.name}")
                if len(relationships) >= 8:
                    return relationships
        return relationships

    def object_count(self):
        with self._lock:
            return len(self._objects)

    def clear(self):
        """Clear all map data."""
        with self._lock:
            self._objects.clear()
            self._points.clear()
            self._dirty = True
        self.save()
        print("[room_map] Cleared")
