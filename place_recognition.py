"""Lightweight visual place recognition for indoor rover navigation.

Builds location fingerprints from YOLO detections + depth profiles + CNN embeddings.
Signatures are collected automatically during navigation and matched to recognize
previously-visited locations.
"""

from __future__ import annotations

import math
import os
import pickle
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PlaceSignature:
    """Visual fingerprint of a specific location."""
    sig_id: int
    timestamp: float
    room_id: str
    heading_deg: float
    step_index: int

    # Component 1: YOLO object fingerprint (semantic)
    yolo_objects: dict[str, float]                    # {name: max_conf}
    yolo_layout: list[tuple[str, float, float]]       # [(name, cx, cy)]

    # Component 2: Depth profile (structural)
    depth_profile: np.ndarray                          # shape (21,)

    # Component 3: CNN embedding (visual appearance)
    cnn_embedding: np.ndarray                          # shape (576,), L2-normalized

    # Metadata
    match_count: int = 0
    last_matched: float = 0.0


MAX_SIGNATURES = 500
MAX_PER_ROOM = 120
MERGE_CNN_THRESHOLD = 0.93
MIN_STEP_GAP = 2          # minimum steps between captures
CNN_WEIGHT = 0.55
DEPTH_WEIGHT = 0.25
YOLO_WEIGHT = 0.20
DEPTH_MAX_M = 2.4


class PlaceDB:
    """Database of place signatures with matching and persistence."""

    def __init__(self, db_path: str = "place_signatures.pkl",
                 device: str = "cuda"):
        self._signatures: list[PlaceSignature] = []
        self._db_path = db_path
        self._next_id = 1
        self._lock = threading.Lock()
        self._device = device

        # CNN model (lazy-loaded)
        self._model = None
        self._transform = None
        self._model_loaded = False

        # Pre-computed matrices for fast lookup
        self._cnn_matrix: np.ndarray | None = None
        self._depth_matrix: np.ndarray | None = None
        self._dirty = False

        # Capture gating
        self._last_capture_step = -999
        self._last_capture_heading = 0.0

    # --- Model loading ---

    def _ensure_model(self):
        """Lazy-load MobileNetV3-Small feature extractor."""
        if self._model_loaded:
            return
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            full_model = models.mobilenet_v3_small(weights=weights)
            full_model.eval()
            full_model.to(self._device)
            # We only need features + avgpool (no classifier)
            self._model = full_model
            self._transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            self._model_loaded = True
            print("[place] MobileNetV3-Small loaded for place recognition")
        except Exception as exc:
            print(f"[place] Failed to load CNN model: {exc}")
            self._model_loaded = True  # don't retry
            self._model = None

    # --- Signature creation ---

    def create_signature(self, *, room_id: str, heading_deg: float,
                         step_index: int, detections: list[dict],
                         depth_distances: list[float],
                         jpeg_bytes: bytes) -> PlaceSignature | None:
        """Create a PlaceSignature from current sensor data."""
        self._ensure_model()

        yolo_objects, yolo_layout = self._build_yolo_fingerprint(detections)
        depth_profile = np.array(depth_distances[:21], dtype=np.float32)
        if len(depth_profile) < 21:
            depth_profile = np.pad(depth_profile, (0, 21 - len(depth_profile)))

        cnn_embedding = self._extract_cnn_embedding(jpeg_bytes)
        if cnn_embedding is None:
            return None

        with self._lock:
            sig_id = self._next_id
            self._next_id += 1

        return PlaceSignature(
            sig_id=sig_id,
            timestamp=time.time(),
            room_id=room_id,
            heading_deg=heading_deg,
            step_index=step_index,
            yolo_objects=yolo_objects,
            yolo_layout=yolo_layout,
            depth_profile=depth_profile,
            cnn_embedding=cnn_embedding,
        )

    def _build_yolo_fingerprint(self, detections: list[dict]
                                ) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
        obj_map: dict[str, float] = {}
        layout: list[tuple[str, float, float]] = []
        for d in detections:
            name = d.get("name", "")
            conf = d.get("conf", 0.0)
            if not name:
                continue
            if name not in obj_map or conf > obj_map[name]:
                obj_map[name] = round(conf, 2)
            layout.append((name, round(d.get("cx", 0.5), 2), round(d.get("cy", 0.5), 2)))
        sorted_objs = dict(sorted(obj_map.items(), key=lambda x: -x[1])[:10])
        layout = sorted(layout, key=lambda x: -obj_map.get(x[0], 0))[:8]
        return sorted_objs, layout

    def _extract_cnn_embedding(self, jpeg_bytes: bytes) -> np.ndarray | None:
        if self._model is None or self._transform is None:
            return None
        try:
            import torch
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            tensor = self._transform(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.features(tensor)
                pooled = self._model.avgpool(features).flatten(1)
            vec = pooled.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            return vec
        except Exception as exc:
            print(f"[place] CNN embedding failed: {exc}")
            return None

    # --- Database operations ---

    def add(self, sig: PlaceSignature):
        with self._lock:
            self._signatures.append(sig)
            self._dirty = True
        self._prune()
        # Auto-save every 10 signatures
        if len(self._signatures) % 10 == 0:
            self.save()

    def find_closest(self, query: PlaceSignature, top_k: int = 3,
                     min_score: float = 0.45) -> list[tuple[PlaceSignature, float]]:
        """Find closest known locations. Returns [(sig, score)] sorted by score."""
        with self._lock:
            if not self._signatures:
                return []
            if self._dirty or self._cnn_matrix is None:
                self._rebuild_matrices()

            # CNN cosine similarity (batch)
            q_cnn = query.cnn_embedding.reshape(1, -1)
            cnn_scores = (self._cnn_matrix @ q_cnn.T).flatten()

            # Depth L1 similarity
            q_depth = query.depth_profile.reshape(1, -1)
            depth_diffs = np.abs(self._depth_matrix - q_depth)
            depth_scores = 1.0 - np.mean(depth_diffs, axis=1) / DEPTH_MAX_M

            # YOLO similarity (per-signature, slightly slower but small N)
            yolo_scores = np.array([
                self._yolo_similarity(query.yolo_objects, s.yolo_objects)
                for s in self._signatures
            ], dtype=np.float32)

            combined = (CNN_WEIGHT * cnn_scores +
                        DEPTH_WEIGHT * depth_scores +
                        YOLO_WEIGHT * yolo_scores)

            # Exclude the query itself (if it was just added)
            for i, s in enumerate(self._signatures):
                if s.sig_id == query.sig_id:
                    combined[i] = -1.0

            top_indices = np.argsort(-combined)[:top_k]
            results = []
            for idx in top_indices:
                score = float(combined[idx])
                if score >= min_score:
                    sig = self._signatures[idx]
                    sig.match_count += 1
                    sig.last_matched = time.time()
                    results.append((sig, score))
            return results

    def _yolo_similarity(self, a: dict[str, float], b: dict[str, float]) -> float:
        all_keys = set(a) | set(b)
        if not all_keys:
            return 0.0
        intersection = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in all_keys)
        union = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in all_keys)
        return intersection / max(union, 1e-6)

    def _rebuild_matrices(self):
        n = len(self._signatures)
        if n == 0:
            self._cnn_matrix = None
            self._depth_matrix = None
            self._dirty = False
            return
        self._cnn_matrix = np.stack([s.cnn_embedding for s in self._signatures])
        self._depth_matrix = np.stack([s.depth_profile for s in self._signatures])
        self._dirty = False

    def _prune(self):
        with self._lock:
            n = len(self._signatures)
            if n <= MAX_SIGNATURES:
                return
            # Per-room cap: drop oldest unmatched
            room_counts: dict[str, int] = {}
            for s in self._signatures:
                room_counts[s.room_id] = room_counts.get(s.room_id, 0) + 1
            to_remove = set()
            for room, count in room_counts.items():
                if count > MAX_PER_ROOM:
                    room_sigs = sorted(
                        [s for s in self._signatures if s.room_id == room],
                        key=lambda s: (s.match_count, s.timestamp),
                    )
                    for s in room_sigs[:count - MAX_PER_ROOM]:
                        to_remove.add(s.sig_id)
            if to_remove:
                self._signatures = [s for s in self._signatures if s.sig_id not in to_remove]
                self._dirty = True
            # Global cap
            if len(self._signatures) > MAX_SIGNATURES:
                by_value = sorted(self._signatures, key=lambda s: (s.match_count, s.timestamp))
                self._signatures = by_value[-MAX_SIGNATURES:]
                self._dirty = True

    # --- Capture gating ---

    def should_capture(self, step_index: int, heading_deg: float) -> bool:
        step_gap = step_index - self._last_capture_step
        if step_gap < MIN_STEP_GAP:
            return False
        heading_diff = abs((heading_deg - self._last_capture_heading + 180) % 360 - 180)
        return step_gap >= 3 or heading_diff >= 30.0

    def mark_captured(self, step_index: int, heading_deg: float):
        self._last_capture_step = step_index
        self._last_capture_heading = heading_deg

    # --- Persistence ---

    def save(self):
        try:
            with self._lock:
                data = {
                    "version": 1,
                    "next_id": self._next_id,
                    "signatures": self._signatures,
                }
            with open(self._db_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[place] Saved {len(self._signatures)} signatures to {self._db_path}")
        except Exception as exc:
            print(f"[place] Save failed: {exc}")

    def load(self):
        if not os.path.exists(self._db_path):
            return
        try:
            with open(self._db_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._signatures = data.get("signatures", [])
                self._next_id = data.get("next_id", 1)
                self._dirty = True
            print(f"[place] Loaded {len(self._signatures)} signatures from {self._db_path}")
        except Exception as exc:
            print(f"[place] Load failed: {exc}")

    @property
    def size(self) -> int:
        return len(self._signatures)
