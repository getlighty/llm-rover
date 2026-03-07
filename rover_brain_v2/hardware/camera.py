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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._events.publish("system", "Camera ready (640x480)")

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
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            raw_jpeg = buffer.tobytes()
            overlay_jpeg = raw_jpeg
            dets = []
            summary = ""
            depth_map = None
            self._frame_counter += 1
            det_interval = 1 if self._follow_mode else 3
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
            if depth_map is not None or dets:
                overlay = frame.copy()
                if dets and self._detector is not None:
                    self._detector.draw(overlay, dets)
                if depth_map is not None and self._depth_estimator is not None:
                    try:
                        mini = cv2.resize(self._depth_estimator.colorize(depth_map), (128, 96))
                        h, w = overlay.shape[:2]
                        overlay[h - 96:h, w - 128:w] = mini
                    except Exception:
                        pass
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

    def get_detections(self):
        with self._lock:
            age = time.time() - self._detection_ts if self._detection_ts else 999.0
            return list(self._detections), self._detection_summary, age

    def close(self):
        self._running = False
        self._cap.release()
        self._events.publish("system", "Camera closed")
