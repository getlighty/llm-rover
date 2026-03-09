"""Local spatial map built from observations during navigation.

Maintains a list of landmarks/observations around the rover with their
current bearing (updated on every turn) and distance (updated on every drive).
Provides structured array output for the LLM via a read_map tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Observation expires after this many steps
STALE_STEPS = 14
# Max entries to keep
MAX_ENTRIES = 30


@dataclass(slots=True)
class MapEntry:
    """A single observation on the spatial map."""
    label: str              # What was seen ("wooden door frame", "chair", "wall")
    bearing_deg: float      # Current bearing relative to body heading (0=ahead, +right, -left)
    distance_m: float       # Estimated distance in meters
    feature: str            # "door", "wall", "furniture", "open", "obstacle", "object"
    step_observed: int      # Step when first/last observed
    source: str             # "scene", "yolo", "depth"
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bearing": round(self.bearing_deg),
            "dist_m": round(self.distance_m, 2),
            "type": self.feature,
            "age": 0,  # Filled in by caller
        }


class LocalSpatialMap:
    """Ego-centric spatial map that updates with rover motion."""

    def __init__(self):
        self.entries: list[MapEntry] = []
        self._heading_deg = 0.0  # Cumulative body heading
        self._x = 0.0
        self._y = 0.0
        self._step = 0

    def reset(self):
        self.entries.clear()
        self._heading_deg = 0.0
        self._x = 0.0
        self._y = 0.0
        self._step = 0

    def set_step(self, step: int):
        self._step = step

    def on_turn(self, angle_deg: float):
        """Body turned by angle_deg. All bearings shift."""
        self._heading_deg += angle_deg
        for e in self.entries:
            e.bearing_deg = _norm(e.bearing_deg - angle_deg)

    def on_drive(self, distance_m: float, angle_deg: float):
        """Body drove forward. Shift distances geometrically."""
        rad = math.radians(angle_deg)
        self._x += distance_m * math.cos(rad)
        self._y += distance_m * math.sin(rad)
        for e in self.entries:
            obs_rad = math.radians(e.bearing_deg)
            approach = distance_m * math.cos(obs_rad - rad)
            e.distance_m = max(0.05, e.distance_m - approach)
        # Re-compute bearings based on new position relative to observation
        # (only a rough approximation — good enough for indoor distances)

    def observe_scene(self, *, gimbal_pan_deg: float, scene: str,
                      depth_center_m: float, step: int):
        """Record an LLM scene observation."""
        if not scene or len(scene) < 5:
            return
        bearing = _norm(gimbal_pan_deg)
        feature = _classify_text(scene)
        short = _simplify_label(scene)
        dist = depth_center_m if depth_center_m > 0.05 else 1.0
        merged = self._merge_nearby(bearing, short, dist, step, threshold_deg=20)
        if not merged:
            self.entries.append(MapEntry(
                label=short,
                bearing_deg=bearing,
                distance_m=dist,
                feature=feature,
                step_observed=step,
                source="scene",
                confidence=0.7,
            ))
        self._cap()

    def observe_yolo(self, *, detections: list[dict], gimbal_pan_deg: float,
                     step: int, fov_deg: float = 65.0):
        """Record YOLO detections as map entries."""
        if not detections:
            return
        for det in detections[:6]:
            name = det.get("name", "?")
            cx = det.get("cx", 0.5)
            dist = det.get("dist_m") or 1.0
            bearing = _norm(gimbal_pan_deg + (cx - 0.5) * fov_deg)
            feature = _classify_yolo(name)
            merged = self._merge_nearby(bearing, name, float(dist), step,
                                         threshold_deg=15, source="yolo")
            if not merged:
                self.entries.append(MapEntry(
                    label=name,
                    bearing_deg=bearing,
                    distance_m=float(dist),
                    feature=feature,
                    step_observed=step,
                    source="yolo",
                    confidence=float(det.get("conf", 0.5)),
                ))
        self._cap()

    def observe_depth_column(self, *, bearing_deg: float, distance_m: float, step: int):
        """Record a depth measurement as a map entry."""
        if distance_m < 0.3:
            label = f"obstacle {distance_m:.1f}m"
            feature = "obstacle"
        elif distance_m < 0.6:
            label = f"surface {distance_m:.1f}m"
            feature = "wall"
        elif distance_m > 1.5:
            label = f"open {distance_m:.1f}m"
            feature = "open"
        else:
            label = f"clearance {distance_m:.1f}m"
            feature = "object"
        merged = self._merge_nearby(bearing_deg, label, distance_m, step,
                                     threshold_deg=20, source="depth")
        if not merged:
            self.entries.append(MapEntry(
                label=label,
                bearing_deg=_norm(bearing_deg),
                distance_m=distance_m,
                feature=feature,
                step_observed=step,
                source="depth",
                confidence=0.9,
            ))
        self._cap()

    def observe_depth_grid(self, *, grid: list[list[float]], gimbal_pan_deg: float,
                           step: int, fov_deg: float = 65.0):
        """Extract depth readings at left/center/right of the FOV."""
        if not grid or len(grid) < 8:
            return
        n_cols = len(grid[0])
        floor_rows = grid[4:8]  # Bottom half = floor level
        # Sample 3 sectors: left, center, right
        sample_cols = [0, n_cols // 2, n_cols - 1]
        for col in sample_cols:
            bearing = _norm(gimbal_pan_deg + (col / max(n_cols - 1, 1) - 0.5) * fov_deg)
            vals = [row[col] for row in floor_rows if col < len(row)]
            if not vals:
                continue
            min_d = min(vals)
            self.observe_depth_column(bearing_deg=bearing, distance_m=min_d, step=step)

    def prune(self):
        """Remove stale entries."""
        cutoff = self._step - STALE_STEPS
        self.entries = [e for e in self.entries if e.step_observed >= cutoff]

    def to_array(self) -> list[dict]:
        """Return map as sorted array of dicts for the LLM."""
        self.prune()
        result = []
        for e in sorted(self.entries, key=lambda e: e.bearing_deg):
            d = e.to_dict()
            d["age"] = self._step - e.step_observed
            result.append(d)
        return result

    def to_prompt_text(self) -> str:
        """Compact text rendition of the map array for injection into the prompt."""
        arr = self.to_array()
        if not arr:
            return "SPATIAL MAP: empty — scan with gimbal to build map"
        lines = ["SPATIAL MAP (bearing: 0°=ahead, +=right, -=left):"]
        for item in arr:
            age_s = "" if item["age"] == 0 else f" {item['age']}steps ago"
            lines.append(
                f"  {item['bearing']:+4d}° {item['dist_m']:.1f}m "
                f"[{item['type']}] {item['label']}{age_s}"
            )
        # Position summary
        if abs(self._x) > 0.1 or abs(self._y) > 0.1:
            lines.append(f"  Position: {self._x:+.1f}m fwd, {self._y:+.1f}m lateral from start")
        lines.append(f"  Body heading: {self._heading_deg:+.0f}° from start")
        return "\n".join(lines)

    def door_summary(self) -> str:
        """Quick summary of known door locations for bearing context."""
        doors = [e for e in self.entries
                 if e.feature == "door" and self._step - e.step_observed < 8]
        if not doors:
            return ""
        parts = []
        for d in sorted(doors, key=lambda e: e.bearing_deg):
            parts.append(f"{d.label} at {d.bearing_deg:+.0f}° ({d.distance_m:.1f}m)")
        return "Known doors: " + "; ".join(parts)

    def _merge_nearby(self, bearing: float, label: str, dist: float, step: int,
                      threshold_deg: float = 20, source: str = "scene") -> bool:
        """Merge with existing entry if close in bearing and similar label."""
        for e in self.entries:
            if abs(_norm(e.bearing_deg - bearing)) < threshold_deg:
                # Same source and similar label — update
                if e.source == source or _labels_similar(e.label, label):
                    e.label = label[:80]
                    e.bearing_deg = bearing
                    e.distance_m = dist
                    e.step_observed = step
                    if source == "yolo":
                        e.feature = _classify_yolo(label)
                    else:
                        e.feature = _classify_text(label)
                    return True
        return False

    def _cap(self):
        """Keep map size bounded."""
        if len(self.entries) > MAX_ENTRIES:
            # Remove oldest low-confidence entries
            self.entries.sort(key=lambda e: (e.step_observed, e.confidence))
            self.entries = self.entries[-MAX_ENTRIES:]


def _norm(deg: float) -> float:
    """Normalize to -180..+180."""
    deg = deg % 360
    return deg - 360 if deg > 180 else deg


def _labels_similar(a: str, b: str) -> bool:
    """Check if two labels likely describe the same thing."""
    a_words = set(a.lower().split()[:4])
    b_words = set(b.lower().split()[:4])
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    return overlap >= min(2, len(a_words), len(b_words))


_DOOR_WORDS = ("door frame", "doorway", "door visible", "archway", "passage",
               "entrance", "threshold", "bright opening", "bright light beyond",
               "light beyond", "opening to", "opening into", "bright arch",
               "hallway beyond", "room beyond", "floor transition")
_WALL_WORDS = ("wall", "radiator", "window", "curtain", "baseboard", "panel")
_FURNITURE_WORDS = ("desk", "chair", "table", "couch", "bed", "cabinet", "shelf",
                    "sofa", "wardrobe", "dresser", "bag", "backpack", "suitcase",
                    "monitor", "basket", "plant", "lamp", "rug")
_OBSTACLE_WORDS = ("blocked", "obstacle", "stuck", "trap", "cable", "under desk",
                   "under table", "under bed", "under chair", "chair legs",
                   "chair base", "desk legs", "furniture leg")
_SEARCH_WORDS = ("scanning", "looking for", "find the", "searching", "escaping",
                 "turning to", "turning left", "turning right")


def _classify_text(scene: str) -> str:
    s = scene.lower()
    is_search = any(w in s for w in _SEARCH_WORDS)
    if not is_search and any(w in s for w in _DOOR_WORDS):
        return "door"
    if any(w in s for w in _OBSTACLE_WORDS):
        return "obstacle"
    if any(w in s for w in _WALL_WORDS):
        return "wall"
    if any(w in s for w in _FURNITURE_WORDS):
        return "furniture"
    if any(w in s for w in ("hallway", "corridor", "open space", "clear")):
        return "open"
    return "object"


def _simplify_label(scene: str) -> str:
    """Extract short landmark label from verbose LLM scene description."""
    s = scene.lower()
    # Priority order: return the most useful landmark
    for word, label in (
        ("door frame", "door frame"), ("doorway", "doorway"), ("archway", "archway"),
        ("bright opening", "doorway (bright)"), ("bright light beyond", "doorway (light)"),
        ("floor transition", "floor transition"), ("threshold", "threshold"),
        ("hallway", "hallway"), ("corridor", "corridor"),
        ("radiator", "radiator"), ("window", "window"), ("curtain", "curtain"),
        ("bookshelf", "bookshelf"), ("shelf", "shelf"),
        ("monitor", "monitor"), ("tv", "TV"), ("lamp", "lamp"),
        ("refrigerator", "fridge"), ("oven", "oven"), ("sink", "sink"),
        ("washing machine", "washer"), ("microwave", "microwave"),
        ("couch", "couch"), ("sofa", "sofa"), ("bed", "bed"),
        ("desk", "desk"), ("dining table", "dining table"), ("table", "table"),
        ("chair base", "chair base"), ("chair legs", "chair legs"), ("chair", "chair"),
        ("cabinet", "cabinet"), ("wardrobe", "wardrobe"), ("dresser", "dresser"),
        ("backpack", "backpack"), ("bag", "bag"), ("suitcase", "suitcase"),
        ("dog", "dog"), ("cat", "cat"), ("pet", "pet"),
        ("plant", "plant"), ("rug", "rug"), ("carpet", "carpet"),
        ("wall", "wall"), ("floor", "floor"),
        ("cable", "cable"), ("obstacle", "obstacle"),
    ):
        if word in s:
            return label
    # Fallback: first 20 chars
    return scene[:20].strip()


def _classify_yolo(name: str) -> str:
    n = name.lower()
    if n in ("door", "doorway"):
        return "door"
    if n in ("chair", "couch", "bed", "dining table", "desk", "table", "bench",
             "potted plant", "vase", "tv", "monitor", "refrigerator", "oven",
             "microwave", "toaster", "sink", "toilet"):
        return "furniture"
    if n in ("person", "dog", "cat"):
        return "living"
    return "object"
