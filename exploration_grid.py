"""Exploration Grid — lightweight stub replacing the 80×80 voxel grid.

The original implementation maintained an 80×80×2 numpy grid tracking
visited/free/occupied cells at 20cm resolution. This was a traditional
robotics approach that didn't integrate well with LLM-first navigation.

This stub preserves the public interface so navigator.py and web_ui.py
don't break, but the LLM now handles spatial reasoning directly from
camera images rather than relying on a discretized grid.

What's kept:
- summarize_for_llm() — returns exploration notes as text
- render_image() — returns a simple placeholder image
- update_after_drive/turn/depth — no-ops that don't crash

What's removed:
- numpy dependency
- 80×80×2 voxel grid
- Depth projection math
- Cell-based obstacle tracking
"""

import io
import time


class ExplorationGrid:
    """Stub exploration tracker — records what the LLM has explored as text."""

    def __init__(self):
        self._notes = []       # list of (time, text) exploration notes
        self._visited_rooms = set()
        self.heading = 0.0
        self.rx = 40.0  # grid-coords (unused, kept for compat)
        self.ry = 40.0

    def reset(self):
        self._notes.clear()
        self._visited_rooms.clear()
        self.heading = 0.0

    def add_note(self, text, room=None):
        """Record an exploration observation."""
        self._notes.append((time.time(), text))
        if len(self._notes) > 20:
            self._notes = self._notes[-20:]
        if room:
            self._visited_rooms.add(room)

    # ── LLM interface (the only thing that matters) ──────────────────

    def summarize_for_llm(self, body_yaw=0.0):
        """Return exploration summary as text for LLM prompt injection."""
        if not self._notes:
            return ""
        lines = ["Exploration notes:"]
        now = time.time()
        for ts, text in self._notes[-6:]:
            age = int(now - ts)
            age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
            lines.append(f"  - {text} ({age_str})")
        if self._visited_rooms:
            lines.append(f"Visited rooms: {', '.join(sorted(self._visited_rooms))}")
        return "\n".join(lines)

    # ── Stubs — called by navigator.py, do nothing ───────────────────

    def update_after_drive(self, distance_m, heading_deg):
        """Stub — no grid to update."""
        pass

    def update_after_turn(self, degrees):
        """Stub — no grid to update."""
        self.heading = (self.heading + degrees) % 360

    def update_from_depth(self, depth_map, body_yaw, cam_pan):
        """Stub — no depth projection."""
        pass

    # ── Web UI placeholder ───────────────────────────────────────────

    def render_image(self, scale=4):
        """Return a minimal JPEG placeholder for the /nav/grid endpoint."""
        try:
            from PIL import Image
            img = Image.new("RGB", (80 * scale, 80 * scale), (40, 40, 40))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=50)
            return buf.getvalue()
        except ImportError:
            # No PIL — return minimal valid JPEG bytes
            return None
