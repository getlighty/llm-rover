"""Camera capture pipeline with optional YOLO and DepthAnything processing."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class CameraPipeline:
    def __init__(self, source: int, event_bus):
        self._events = event_bus
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera")
        for _ in range(5):
            self._cap.read()
        self._lock = threading.Lock()
        self._jpeg = None
        self._overlay_jpeg = None
        self._depth_map = None
        self._detections = []
        self._detection_summary = ""
        self._detection_ts = 0.0
        self._running = True
        self._follow_mode = False
        self._detector_enabled = True
        self._detector = None
        self._depth_estimator = None
        self._frame_counter = 0
        # Visual-odometry tracked navigation target.
        # LLM sets a pixel target; optical flow tracks it frame-to-frame
        # so it stays pinned to the real-world floor point.
        self._nav_point: np.ndarray | None = None       # shape (1,1,2) float32 for LK
        self._nav_point_ts: float = 0.0
        self._nav_prev_gray: np.ndarray | None = None   # previous grayscale frame
        self._nav_dot_screen: tuple[float, float] | None = None  # (cx, cy) 0-1
        # Second "next" target — prefetched while driving toward the first
        self._nav_next_point: np.ndarray | None = None
        self._nav_next_ts: float = 0.0
        self._nav_next_screen: tuple[float, float] | None = None
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        # Precompute fisheye undistortion maps for 640x480
        self._undistort_map1, self._undistort_map2 = self._build_undistort_maps(640, 480)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._events.publish("system", "Camera ready (640x480, undistort enabled)")

    @staticmethod
    def _build_undistort_maps(w: int, h: int):
        """Build undistortion remap tables for a typical wide-angle USB camera.

        Distortion coefficients are estimated for a ~65° FOV barrel-distortion
        camera.  Tuned empirically: k1=-0.30 corrects the worst barrel bulge
        without cropping too much of the frame.
        """
        fx = w * 0.85   # focal length estimate (~544 px for 640w)
        fy = fx
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)
        # k1=barrel, k2=higher-order, p1/p2=tangential, k3=extra
        dist_coeffs = np.array([-0.30, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=0.4,
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2,
        )
        return map1, map2

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def depth_estimator(self):
        return self._depth_estimator

    @depth_estimator.setter
    def depth_estimator(self, value):
        self._depth_estimator = value

    def set_follow_mode(self, enabled: bool):
        self._follow_mode = bool(enabled)

    def set_nav_target(self, x: int, y: int):
        """Set a real-world navigation target at pixel (x, y).

        Optical flow tracks it frame-to-frame so it stays pinned
        to the same physical spot as the camera moves.
        """
        with self._lock:
            self._nav_point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            self._nav_point_ts = time.time()
            # Reset prev frame so flow starts fresh from next capture
            self._nav_prev_gray = None

    def set_gimbal_pan(self, pan_deg: float):
        """Note gimbal pan change — optical flow handles this naturally."""
        pass  # optical flow tracks the motion directly, no manual shift needed

    def adjust_nav_bearing(self, delta_deg: float):
        """No-op — optical flow tracks body turns via image motion."""
        pass

    def set_nav_next_target(self, x: int, y: int):
        """Set the prefetched next navigation target (shown as blue dot)."""
        with self._lock:
            self._nav_next_point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            self._nav_next_ts = time.time()

    def clear_nav_target(self):
        """Remove the navigation target."""
        with self._lock:
            self._nav_point = None
            self._nav_prev_gray = None
            self._nav_dot_screen = None

    def clear_nav_next_target(self):
        """Remove the next navigation target."""
        with self._lock:
            self._nav_next_point = None
            self._nav_next_screen = None

    def promote_next_target(self):
        """Move the next target to become the current target."""
        with self._lock:
            if self._nav_next_point is not None:
                self._nav_point = self._nav_next_point
                self._nav_point_ts = self._nav_next_ts
                self._nav_dot_screen = self._nav_next_screen
                self._nav_prev_gray = None  # reset flow tracking
                self._nav_next_point = None
                self._nav_next_screen = None

    def get_nav_dot(self) -> tuple[float, float] | None:
        """Return tracked dot screen position as (cx, cy) 0-1, or None."""
        with self._lock:
            return self._nav_dot_screen

    def get_nav_next_dot(self) -> tuple[float, float] | None:
        """Return next dot screen position as (cx, cy) 0-1, or None."""
        with self._lock:
            return self._nav_next_screen

    def _track_nav_point(self, gray_frame):
        """Track navigation points using Lucas-Kanade optical flow.

        Tracks both the current target and the prefetched next target.
        Called every frame. Returns (px, py) of current target or None.
        """
        with self._lock:
            has_current = self._nav_point is not None
            has_next = self._nav_next_point is not None
            if not has_current and not has_next:
                self._nav_prev_gray = gray_frame.copy()
                return None
            if has_current:
                age = time.time() - self._nav_point_ts
                if age > 30.0:
                    self._nav_point = None
                    has_current = False
            if has_next:
                age_next = time.time() - self._nav_next_ts
                if age_next > 30.0:
                    self._nav_next_point = None
                    has_next = False
            prev_gray = self._nav_prev_gray
            point = self._nav_point.copy() if has_current else None
            next_point = self._nav_next_point.copy() if has_next else None

        if prev_gray is None:
            with self._lock:
                self._nav_prev_gray = gray_frame.copy()
            if point is not None:
                return (int(point[0, 0, 0]), int(point[0, 0, 1]))
            return None

        # Track both points in a single optical flow call
        all_pts = []
        idx_current = idx_next = -1
        if point is not None:
            idx_current = len(all_pts)
            all_pts.append(point)
        if next_point is not None:
            idx_next = len(all_pts)
            all_pts.append(next_point)

        if not all_pts:
            with self._lock:
                self._nav_prev_gray = gray_frame.copy()
            return None

        stacked = np.concatenate(all_pts, axis=0)  # shape (N, 1, 2)
        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray_frame, stacked, None, **self._lk_params
        )

        current_result = None
        with self._lock:
            self._nav_prev_gray = gray_frame.copy()

            # Update current target
            if idx_current >= 0 and new_pts is not None and status[idx_current, 0] == 1:
                nx, ny = float(new_pts[idx_current, 0, 0]), float(new_pts[idx_current, 0, 1])
                if 0 <= nx <= 640 and 0 <= ny <= 384:
                    self._nav_point = new_pts[idx_current:idx_current+1]
                    self._nav_dot_screen = (nx / 640.0, ny / 480.0)
                    current_result = (int(nx), int(ny))
                else:
                    self._nav_dot_screen = None
            elif idx_current >= 0:
                self._nav_dot_screen = None

            # Update next target
            if idx_next >= 0 and new_pts is not None and status[idx_next, 0] == 1:
                nx2, ny2 = float(new_pts[idx_next, 0, 0]), float(new_pts[idx_next, 0, 1])
                if 0 <= nx2 <= 640 and 0 <= ny2 <= 384:
                    self._nav_next_point = new_pts[idx_next:idx_next+1]
                    self._nav_next_screen = (nx2 / 640.0, ny2 / 480.0)
                else:
                    self._nav_next_screen = None
            elif idx_next >= 0:
                self._nav_next_screen = None

        return current_result


    def set_detector_enabled(self, enabled: bool):
        self._detector_enabled = bool(enabled)
        if not enabled and self._detector is not None:
            try:
                self._detector.clear_cache()
            except Exception:
                pass

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            # Undistort fisheye
            if self._undistort_map1 is not None:
                frame = cv2.remap(frame, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            raw_jpeg = buffer.tobytes()
            overlay_jpeg = raw_jpeg
            dets = []
            summary = ""
            depth_map = None
            self._frame_counter += 1
            det_interval = 1
            should_process = (self._frame_counter % det_interval) == 0
            if should_process and self._depth_estimator is not None:
                try:
                    depth_map = self._depth_estimator.estimate(frame)
                except Exception:
                    depth_map = None
            if should_process and self._detector_enabled and self._detector is not None:
                try:
                    dets = self._detector.detect(frame)
                    summary = self._detector.summary(dets)
                except Exception:
                    dets = []
                    summary = ""
            # Track navigation dot via optical flow (visual odometry)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nav_target = self._track_nav_point(gray)
            # Draw overlays — nav dot always, YOLO only when follow mode
            show_yolo = dets and self._follow_mode
            needs_overlay = show_yolo or nav_target is not None
            if needs_overlay:
                overlay = frame.copy()
                if show_yolo and self._detector is not None:
                    self._detector.draw(overlay, dets)
                h, w = overlay.shape[:2]
                if nav_target is not None:
                    tx, ty = nav_target
                    cv2.line(overlay, (w // 2, h), (tx, ty),
                             (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(overlay, (tx, ty), 12, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(overlay, (tx, ty), 10, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.line(overlay, (tx - 16, ty), (tx + 16, ty),
                             (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(overlay, (tx, ty - 16), (tx, ty + 16),
                             (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(overlay, "TARGET", (tx + 14, ty - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                _, overlay_buffer = cv2.imencode(
                    ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 72]
                )
                overlay_jpeg = overlay_buffer.tobytes()
            with self._lock:
                self._jpeg = raw_jpeg
                self._overlay_jpeg = overlay_jpeg
                if dets:
                    self._detections = dets
                    self._detection_summary = summary
                    self._detection_ts = time.time()
                elif not self._detector_enabled:
                    self._detections = []
                    self._detection_summary = ""
                    self._detection_ts = 0.0
                self._depth_map = depth_map
            time.sleep(0.03)

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_overlay_jpeg(self):
        with self._lock:
            return self._overlay_jpeg or self._jpeg

    def snap_with_yolo(self, max_dim: int = 512, quality: int = 60):
        """Return resized JPEG with YOLO boxes drawn but NO depth minimap."""
        with self._lock:
            jpeg = self._jpeg
            dets = list(self._detections)
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        if dets and self._detector is not None:
            self._detector.draw(image, dets)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def snap(self, max_dim: int = 512, quality: int = 60):
        with self._lock:
            jpeg = self._jpeg
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def get_depth_map(self):
        with self._lock:
            return None if self._depth_map is None else self._depth_map.copy()

    def get_depth_image(self, max_dim: int = 320, quality: int = 50):
        """Return colorized depth map as JPEG bytes, or None."""
        with self._lock:
            dm = self._depth_map
        if dm is None or self._depth_estimator is None:
            return None
        try:
            colorized = self._depth_estimator.colorize(dm)
            h, w = colorized.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                colorized = cv2.resize(colorized, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", colorized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()
        except Exception:
            return None

    def get_detections(self):
        with self._lock:
            age = time.time() - self._detection_ts if self._detection_ts else 999.0
            return list(self._detections), self._detection_summary, age

    def close(self):
        self._running = False
        self._cap.release()
        self._events.publish("system", "Camera closed")
