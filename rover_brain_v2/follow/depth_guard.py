"""DepthAnything guard rails for follow mode."""

from __future__ import annotations

from rover_brain_v2.navigation.depth_vectors import DepthVectorMap


class DepthCrashGuard:
    def __init__(self, *, stop_distance_m: float = 0.35,
                 turn_stop_distance_m: float = 0.25):
        self.stop_distance_m = stop_distance_m
        self.turn_stop_distance_m = turn_stop_distance_m
        self.depth_vectors = DepthVectorMap(
            num_columns=21,
            corridor_width=7,
            floor_top=0.40,
            floor_bottom=0.88,
            depth_max_clearance_m=2.4,
            passable_clearance_m=0.45,
        )

    def assess(self, depth_map, steering_angle_deg: float = 0.0) -> dict:
        if depth_map is None:
            return {"safe": True, "reason": "no_depth"}
        summary = self.depth_vectors.analyze(
            depth_map,
            preferred_heading_deg=steering_angle_deg,
            search_half_window_deg=18.0,
        )
        closest = summary.closest_corridor_distance()
        center = summary.center_distance()
        threshold = self.turn_stop_distance_m if abs(steering_angle_deg) > 8 else self.stop_distance_m
        clearance = min(
            center,
            closest if closest is not None else center,
        )
        return {
            "safe": clearance >= threshold,
            "clearance_m": round(float(clearance), 3),
            "center_distance_m": round(float(center), 3),
            "closest_corridor_distance_m": None if closest is None else round(float(closest), 3),
            "recommended_heading_deg": summary.recommended_heading_deg,
            "threshold_m": threshold,
        }

