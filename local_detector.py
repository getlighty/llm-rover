"""
Local object detection using Ultralytics YOLO TensorRT.
Defaults to YOLO26n on this rover.
TensorRT FP16 GPU inference (~20+ FPS), falls back to PyTorch when needed.

Based on:
- NVIDIA JetBot object_following example (SSD + proportional control)
- Waveshare ugv_jetson cv_ctrl.py (OpenCV MobileNet SSD)
- Ultralytics YOLOv8-OpenCV-ONNX-Python example

Usage:
    detector = LocalDetector()                    # YOLO26n TRT (default)
    detector = LocalDetector(model_path="...")     # custom model
    detections = detector.detect(bgr_frame)
    target = detector.find("cup", detections)
"""

import cv2
import numpy as np
import os
import threading
import time

# YOLOv8 COCO 80-class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Known real-world widths (meters) for distance estimation
KNOWN_WIDTHS = {
    "person": 0.45, "cup": 0.08, "bottle": 0.07, "chair": 0.50,
    "laptop": 0.35, "cell phone": 0.07, "book": 0.20, "keyboard": 0.45,
    "mouse": 0.06, "tv": 0.80, "bowl": 0.15, "backpack": 0.30,
    "handbag": 0.25, "suitcase": 0.40, "vase": 0.12, "clock": 0.25,
    "potted plant": 0.20, "teddy bear": 0.25, "remote": 0.05,
    # YOLO-World workshop classes
    "door": 0.90, "table": 1.20, "desk": 1.00, "shelf": 0.80,
    "cabinet": 0.60, "box": 0.30, "container": 0.40, "bin": 0.35,
    "recycle bin": 0.35, "trash can": 0.35, "monitor": 0.50,
    "cable": 0.02, "light": 0.30, "lamp": 0.25, "fan": 0.40,
    "phone": 0.07, "bag": 0.30, "plant": 0.20, "window": 1.00,
    "toolbox": 0.40, "tool": 0.15, "screwdriver": 0.03, "drill": 0.15,
    "wheel": 0.15, "robot": 0.30, "speaker": 0.20, "camera": 0.08,
    "blue basket": 0.35,
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO26N_PT = os.path.join(MODEL_DIR, "yolo26n.pt")
YOLO26N_TRT = os.path.join(MODEL_DIR, "yolo26n.engine")

def _pick_default_model():
    if os.path.exists(YOLO26N_TRT):
        return YOLO26N_TRT
    return YOLO26N_PT

DEFAULT_MODEL = _pick_default_model()

# LLM-corrected label overrides (e.g. "cup" -> "recycle bin")
LABEL_OVERRIDES = {}
_OVERRIDES_FILE = os.path.join(MODEL_DIR, "label_overrides.json")

def _load_label_overrides():
    global LABEL_OVERRIDES
    try:
        import json as _json
        with open(_OVERRIDES_FILE) as f:
            LABEL_OVERRIDES = _json.load(f)
        if LABEL_OVERRIDES:
            print(f"[detector] Loaded {len(LABEL_OVERRIDES)} label overrides")
    except (FileNotFoundError, ValueError):
        LABEL_OVERRIDES = {}

def _save_label_overrides():
    try:
        import json as _json
        with open(_OVERRIDES_FILE, "w") as f:
            _json.dump(LABEL_OVERRIDES, f, indent=2)
    except Exception as e:
        print(f"[detector] Failed to save overrides: {e}")

_load_label_overrides()


class LocalDetector:
    """YOLOv8 object detection. PyTorch GPU, TensorRT, or OpenCV DNN CPU."""

    def __init__(self, model_path=None, conf=0.20, nms_thresh=0.45,
                 input_size=640, focal_length=500.0):
        if model_path is None:
            model_path = _pick_default_model()
        self.conf = conf
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.focal_length = focal_length
        self._lock = threading.Lock()
        self.last_detections = []
        self.last_inference_ms = 0
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Pick class names based on model
        basename = os.path.basename(model_path).lower()
        if "yolo26" in basename:
            self.class_names = COCO_NAMES
            self._model_label = "YOLO26n"
        elif "yolo11" in basename or "yolov8" in basename:
            self.class_names = COCO_NAMES
            self._model_label = "Ultralytics YOLO"
        else:
            self.class_names = COCO_NAMES
            self._model_label = "Ultralytics YOLO"

        # Backend state
        self._ultralytics_model = None  # PyTorch (Ultralytics YOLO)
        self._trt_context = None        # TensorRT
        self.net = None                 # OpenCV DNN

        if model_path.endswith(".pt"):
            self._init_pytorch(model_path)
        elif model_path.endswith(".engine"):
            # YOLO26 engines built with trtexec do not carry Ultralytics
            # metadata, so load them through the raw TensorRT path.
            if "yolo26" in basename:
                self._init_tensorrt(model_path)
            else:
                # Try Ultralytics first (handles its own engine wrapper format)
                try:
                    self._init_pytorch(model_path)
                except Exception:
                    self._init_tensorrt(model_path)
        else:
            self._init_opencv_dnn(model_path)

        print(f"[detector] {self._model_label} loaded ({self.backend}, "
              f"{len(self.class_names)} classes, conf={conf})")

    def _attach_metric_distance(self, det, name, bw_px):
        """Attach a metric object distance when class geometry is known.

        This is currently the only defensible metric distance source in the
        detector. Monocular depth is kept as a separate relative cue.
        """
        known_w = KNOWN_WIDTHS.get(name)
        if not known_w or bw_px <= 5:
            return det
        dist_m = (known_w * self.focal_length) / max(float(bw_px), 1.0)
        # Larger boxes are more stable; tiny boxes are noisy.
        size_conf = min(0.35, bw_px / 400.0)
        det["dist_m"] = round(dist_m, 2)
        det["dist_source"] = "bbox_width"
        det["dist_conf"] = round(min(0.95, 0.55 + size_conf), 2)
        return det

    def _init_pytorch(self, model_path):
        """Load model via Ultralytics for PyTorch/TRT GPU inference."""
        try:
            import torch
            from ultralytics import YOLO
            is_engine = model_path.endswith(".engine")
            self._ultralytics_model = YOLO(model_path, task="detect")
            # Read class names from model (handles custom-trained models)
            if hasattr(self._ultralytics_model, "names") and self._ultralytics_model.names:
                self.class_names = list(self._ultralytics_model.names.values())
                if len(self.class_names) == 1:
                    self._model_label = f"YOLOv8-{self.class_names[0]}"
                elif not is_engine:
                    self._model_label = f"YOLOv8-custom({len(self.class_names)}cls)"
            if is_engine:
                # Engine files are already on GPU — don't call .to()
                self.backend = "TensorRT FP16"
            elif torch.cuda.is_available():
                self._ultralytics_model.to("cuda")
                self.backend = "PyTorch GPU"
            else:
                self.backend = "PyTorch CPU"
            # Warmup
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._ultralytics_model.predict(dummy, verbose=False, conf=self.conf,
                                            half=True, device=0 if is_engine else None)
        except Exception as e:
            print(f"[detector] PyTorch init failed ({e}), falling back to ONNX")
            onnx_path = model_path.replace(".pt", ".onnx")
            if os.path.exists(onnx_path):
                self._ultralytics_model = None
                self._init_opencv_dnn(onnx_path)
            else:
                raise

    def _init_tensorrt(self, engine_path):
        """Load TensorRT engine for GPU inference."""
        import tensorrt as trt
        import torch

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        self._trt_context = engine.create_execution_context()
        self._torch_device = torch.device("cuda:0")

        self._trt_input_name = engine.get_tensor_name(0)
        self._trt_output_name = engine.get_tensor_name(1)
        in_shape = engine.get_tensor_shape(self._trt_input_name)
        out_shape = engine.get_tensor_shape(self._trt_output_name)
        self._trt_output_shape = tuple(out_shape)

        self._trt_input = torch.zeros(*in_shape, dtype=torch.float32,
                                       device=self._torch_device)
        self._trt_output = torch.zeros(*out_shape, dtype=torch.float32,
                                        device=self._torch_device)

        self._trt_context.set_tensor_address(
            self._trt_input_name, self._trt_input.data_ptr())
        self._trt_context.set_tensor_address(
            self._trt_output_name, self._trt_output.data_ptr())

        self._trt_stream = torch.cuda.Stream()
        self.backend = "TensorRT FP16"

        self._trt_context.execute_async_v3(self._trt_stream.cuda_stream)
        self._trt_stream.synchronize()

    def _init_opencv_dnn(self, model_path):
        """Load ONNX model via OpenCV DNN (CPU fallback)."""
        self.net = cv2.dnn.readNetFromONNX(model_path)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            self.net.setInput(dummy)
            self.net.forward()
            self.backend = "CUDA FP16"
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "CPU"

    def _preprocess(self, frame):
        """Preprocess BGR frame to NCHW float32 blob."""
        return cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )

    def detect(self, frame):
        """Run detection on a BGR numpy frame. Returns list of detection dicts."""
        h, w = frame.shape[:2]

        # PyTorch (Ultralytics) path — uses its own pre/post-processing
        if self._ultralytics_model is not None:
            return self._detect_ultralytics(frame, h, w)

        # TensorRT / OpenCV DNN path — manual pre/post-processing
        blob = self._preprocess(frame)

        t0 = time.time()
        if self._trt_context is not None:
            outputs = self._infer_trt(blob)
        else:
            self.net.setInput(blob)
            outputs = self.net.forward()
        self.last_inference_ms = (time.time() - t0) * 1000

        return self._postprocess(outputs, h, w)

    def _detect_ultralytics(self, frame, h, w):
        """Run detection via Ultralytics YOLO predict API."""
        t0 = time.time()
        predict_kwargs = dict(
            verbose=False, conf=self.conf, iou=self.nms_thresh,
            half=True, imgsz=self.input_size,
        )
        if self.backend == "TensorRT FP16":
            predict_kwargs["device"] = 0
        results = self._ultralytics_model.predict(frame, **predict_kwargs)
        self.last_inference_ms = (time.time() - t0) * 1000

        detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i])
                cid = int(boxes.cls[i])
                name = self.class_names[cid] if cid < len(self.class_names) else f"cls_{cid}"
                name = LABEL_OVERRIDES.get(name, name)
                bw_px = x2 - x1
                bh_px = y2 - y1
                cx_r = (x1 + x2) / 2.0
                cy_r = (y1 + y2) / 2.0
                det = {
                    "name": name,
                    "conf": round(conf, 2),
                    "bbox": (x1, y1, x2, y2),
                    "cx": round(cx_r / w, 3),
                    "cy": round(cy_r / h, 3),
                    "bw": round(bw_px / w, 3),
                    "bh": round(bh_px / h, 3),
                }
                self._attach_metric_distance(det, name, bw_px)
                detections.append(det)

        with self._lock:
            self.last_detections = detections
        return detections

    def _infer_trt(self, blob):
        """Run TensorRT inference. Returns raw output numpy array."""
        import torch
        blob_tensor = torch.from_numpy(blob).to(self._torch_device)
        self._trt_input.copy_(blob_tensor)
        self._trt_context.execute_async_v3(self._trt_stream.cuda_stream)
        self._trt_stream.synchronize()
        return self._trt_output.cpu().numpy()

    def _postprocess(self, outputs, h, w):
        """Post-process raw YOLO output into detection dicts."""
        # TensorRT exports from recent Ultralytics builds may already include
        # decoded boxes with NMS: (1, N, 6) = [x1, y1, x2, y2, conf, cls].
        if outputs.ndim == 3 and outputs.shape[-1] == 6:
            detections = []
            for row in outputs[0]:
                x1, y1, x2, y2, conf, cls = [float(v) for v in row]
                if conf < self.conf:
                    continue
                cid = int(cls)
                name = self.class_names[cid] if cid < len(self.class_names) else f"cls_{cid}"
                name = LABEL_OVERRIDES.get(name, name)
                x1_i, y1_i = int(max(0, x1)), int(max(0, y1))
                x2_i, y2_i = int(min(w, x2)), int(min(h, y2))
                bw_px = max(0, x2_i - x1_i)
                bh_px = max(0, y2_i - y1_i)
                if bw_px <= 1 or bh_px <= 1:
                    continue
                cx_r = (x1_i + x2_i) / 2.0
                cy_r = (y1_i + y2_i) / 2.0
                det = {
                    "name": name,
                    "conf": round(float(conf), 2),
                    "bbox": (x1_i, y1_i, x2_i, y2_i),
                    "cx": round(cx_r / w, 3),
                    "cy": round(cy_r / h, 3),
                    "bw": round(bw_px / w, 3),
                    "bh": round(bh_px / h, 3),
                }
                self._attach_metric_distance(det, name, bw_px)
                detections.append(det)
            with self._lock:
                self.last_detections = detections
            return detections

        # YOLOv8 output: (1, N+4, 8400) -> transpose to (8400, N+4)
        out = outputs[0].T if outputs.ndim == 3 else outputs.T
        scores = out[:, 4:]
        max_scores = scores.max(axis=1)
        mask = max_scores > self.conf
        filtered = out[mask]
        fscores = max_scores[mask]
        class_ids = scores[mask].argmax(axis=1)

        boxes = []
        confs = []
        names = []
        raw_data = []
        sx = w / self.input_size
        sy = h / self.input_size
        for i in range(len(filtered)):
            cx, cy, bw, bh = filtered[i, :4]
            cx_r = cx * sx
            cy_r = cy * sy
            bw_r = bw * sx
            bh_r = bh * sy
            x1 = int(cx_r - bw_r / 2)
            y1 = int(cy_r - bh_r / 2)
            cid = int(class_ids[i])
            name = self.class_names[cid] if cid < len(self.class_names) else f"cls_{cid}"
            name = LABEL_OVERRIDES.get(name, name)
            boxes.append([x1, y1, int(bw_r), int(bh_r)])
            confs.append(float(fscores[i]))
            names.append(name)
            raw_data.append((cx_r, cy_r, bw_r, bh_r))

        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms_thresh)
            for idx in indices.flatten():
                x1, y1, bw_px, bh_px = boxes[idx]
                cx_r, cy_r, bw_r, bh_r = raw_data[idx]
                det = {
                    "name": names[idx],
                    "conf": round(confs[idx], 2),
                    "bbox": (x1, y1, x1 + bw_px, y1 + bh_px),
                    "cx": round(cx_r / w, 3),
                    "cy": round(cy_r / h, 3),
                    "bw": round(bw_r / w, 3),
                    "bh": round(bh_r / h, 3),
                }
                self._attach_metric_distance(det, names[idx], bw_px)
                detections.append(det)

        with self._lock:
            self.last_detections = detections
        return detections

    def find(self, target_name, detections=None):
        """Find a specific object by name. Returns best match or None."""
        dets = detections if detections is not None else self.last_detections
        target = target_name.lower().strip()
        # Exact match
        matches = [d for d in dets if d["name"] == target]
        if not matches:
            # Substring match
            matches = [d for d in dets if target in d["name"] or d["name"] in target]
        if matches:
            return max(matches, key=lambda d: d["conf"])
        return None

    def clear_cache(self):
        """Drop any stale detections when YOLO is disabled or unavailable."""
        with self._lock:
            self.last_detections = []

    def draw(self, frame, detections=None):
        """Draw bounding boxes + labels on frame. Returns the modified frame."""
        dets = detections if detections is not None else self.last_detections
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['name']} {d['conf']:.0%}"
            if "dist_m" in d:
                label += f" {d['dist_m']:.1f}m"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def calibrate_distance(self, known_width_m, bbox_width_px, actual_distance_m):
        """Calibrate focal length from a known measurement."""
        self.focal_length = (bbox_width_px * actual_distance_m) / known_width_m
        print(f"[detector] Calibrated focal length: {self.focal_length:.0f}px")
        return self.focal_length

    def calibrate_labels(self, frame, llm_vision_fn):
        """Use LLM vision to correct YOLO label misidentifications.

        Args:
            frame: BGR numpy frame
            llm_vision_fn: callable(prompt, jpeg_bytes) -> str (LLM response)

        Returns:
            dict of corrections made {"old_label": "new_label", ...}
        """
        import json as _json

        # Run detection on the frame
        dets = self.detect(frame)
        if not dets:
            print("[calibrate] No detections to calibrate")
            return {}

        # Build a list of detections for the LLM
        det_list = []
        for i, d in enumerate(dets):
            det_list.append(f"{i+1}. \"{d['name']}\" (conf={d['conf']:.0%}, bbox={d['bbox']})")

        prompt = (
            "I have a YOLO object detector running. It detected these objects in the image:\n"
            + "\n".join(det_list) + "\n\n"
            "For each detection, confirm if the YOLO label is correct or provide the correct label. "
            "YOLO often misidentifies indoor objects (e.g. recycle bin as 'cup', ceiling light as 'frisbee', "
            "parts organizer as 'toilet'). Reply ONLY with JSON:\n"
            '{"corrections": [{"index": 1, "yolo": "cup", "correct": "recycle bin"}, ...]}\n'
            "Only include entries where the label is WRONG. If all labels are correct, return "
            '{"corrections": []}'
        )

        # Encode frame as JPEG for LLM
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = buf.tobytes()

        try:
            raw = llm_vision_fn(prompt, jpeg_bytes)
            # Parse LLM response — handle truncated JSON from max_tokens
            text = raw.strip()
            if "```" in text:
                lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            start = text.find("{")
            if start < 0:
                print(f"[calibrate] No JSON in LLM response: {text[:200]}")
                return {}
            snippet = text[start:]
            parsed = None
            # Try clean parse first
            end = text.rfind("}") + 1
            if end > start:
                try:
                    parsed = _json.loads(text[start:end])
                except _json.JSONDecodeError:
                    pass
            # Truncated — salvage complete correction entries
            if parsed is None:
                # Find last complete object in the array (ends with })
                last_obj = snippet.rfind("}")
                while last_obj > 0 and not parsed:
                    chunk = snippet[:last_obj + 1]
                    for closer in (']}\n', ']}'):
                        try:
                            parsed = _json.loads(chunk + closer)
                            break
                        except _json.JSONDecodeError:
                            pass
                    last_obj = snippet.rfind("}", 0, last_obj)
            if not parsed:
                print(f"[calibrate] Could not parse LLM response: {text[:200]}")
                return {}

            # Labels that should never be overridden (YOLO is reliable for these)
            PROTECTED_LABELS = {
                "person", "chair", "laptop", "tv", "keyboard", "mouse",
                "cell phone", "bottle", "bed", "couch", "dining table",
                "dog", "cat", "car", "bicycle",
            }

            corrections = parsed.get("corrections", [])
            changes = {}
            for c in corrections:
                yolo_label = c.get("yolo", "").strip().lower()
                correct_label = c.get("correct", "").strip().lower()
                if yolo_label and correct_label and yolo_label != correct_label:
                    if yolo_label in PROTECTED_LABELS:
                        print(f"[calibrate] Skipping protected label: '{yolo_label}'")
                        continue
                    # Also check if matching detection had high confidence
                    matching = [d for d in dets if d["name"] == yolo_label]
                    if matching and matching[0]["conf"] > 0.7:
                        print(f"[calibrate] Skipping high-conf ({matching[0]['conf']:.0%}): '{yolo_label}'")
                        continue
                    LABEL_OVERRIDES[yolo_label] = correct_label
                    changes[yolo_label] = correct_label
                    print(f"[calibrate] Override: '{yolo_label}' -> '{correct_label}'")

            if changes:
                _save_label_overrides()
                print(f"[calibrate] Saved {len(changes)} label corrections")
            else:
                print("[calibrate] All labels confirmed correct")

            return changes

        except Exception as e:
            print(f"[calibrate] Error during calibration: {e}")
            return {}

    def summary(self, detections=None):
        """One-line summary of detections for logging."""
        dets = detections if detections is not None else self.last_detections
        if not dets:
            return "nothing"
        parts = []
        for d in dets:
            s = f"{d['name']}({d['conf']:.0%}"
            if "dist_m" in d:
                s += f",{d['dist_m']:.1f}m"
            s += ")"
            parts.append(s)
        return " ".join(parts)


# ── Depth Estimation (Depth Anything TensorRT) ───────────────────────

DEPTH_ENGINE_CANDIDATES = [
    os.environ.get("ROVER_DEPTH_ENGINE", "").strip(),
    os.path.join(MODEL_DIR, "depth_anything_v2_vits14_308.trt"),
    os.path.join(MODEL_DIR, "depth_anything_vits14_308.trt"),
]
DEPTH_INPUT_SIZE = 308
DEPTH_SELF_FLOOR_CUTOFF = 0.88

# ImageNet normalization (applied in BGR order to match model training)
_DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DEPTH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _depth_usable_bottom(h):
    """Bottom row limit for depth consumers to ignore self-floor glow."""
    return max(1, min(h, int(round(h * DEPTH_SELF_FLOOR_CUTOFF))))


class DepthEstimator:
    """Monocular depth estimation using a Depth Anything TensorRT engine.

    Produces a relative float32 depth map (higher = closer).
    CUDA resources are lazily initialized on first estimate() call so that
    the pycuda context is created on the correct thread.

    Usage:
        depth = DepthEstimator()             # lightweight — just reads engine bytes
        depth_map = depth.estimate(bgr_frame) # float32 (H, W) at frame resolution
        depth.enrich_detections(dets, depth_map)  # adds/updates dist_m on each det
    """

    def __init__(self, engine_path=None, input_size=DEPTH_INPUT_SIZE,
                 depth_scale=1.0):
        if engine_path is None:
            engine_path = next(
                (p for p in DEPTH_ENGINE_CANDIDATES if p and os.path.exists(p)),
                os.path.join(MODEL_DIR, "depth_anything_vits14_308.trt"),
            )
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Depth engine not found: {engine_path}")

        self.engine_path = engine_path
        self.engine_name = os.path.basename(engine_path)
        self.input_size = input_size
        self.input_h = input_size
        self.input_w = input_size
        self.output_h = input_size
        self.output_w = input_size
        self.depth_scale = depth_scale
        self._lock = threading.Lock()
        self.last_depth_map = None
        self.last_inference_ms = 0
        self.backend = "TensorRT FP16"

        # Read engine bytes now (main thread, no CUDA needed)
        with open(engine_path, 'rb') as f:
            self._engine_bytes = f.read()

        # CUDA resources — initialized lazily on first estimate() call
        self._ready = False
        self._cuda_ctx = None

        print(f"[depth] Depth engine loaded: {self.engine_name} "
              f"({len(self._engine_bytes) / 1024 / 1024:.0f} MB, "
              f"default={input_size}x{input_size}, scale={depth_scale})")

    def _lazy_init(self):
        """Initialize pycuda + TRT resources on the calling thread.

        Uses the CUDA primary context (shared with PyTorch/Ultralytics)
        rather than creating a new context, to avoid conflicts.
        """
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        if hasattr(dev, "retain_primary_context"):
            self._cuda_ctx = dev.retain_primary_context()
        else:
            self._cuda_ctx = dev.retain_primary_ctx()
        self._cuda_ctx.push()
        self._cuda = cuda

        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(self._engine_bytes)
        del self._engine_bytes  # free the raw bytes
        self.context = self.engine.create_execution_context()

        input_name = None
        output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_name = name
        input_shape = tuple(self.engine.get_tensor_shape(input_name))
        output_shape = tuple(self.engine.get_tensor_shape(output_name))
        if len(input_shape) == 4:
            self.input_h = int(input_shape[-2])
            self.input_w = int(input_shape[-1])
            self.input_size = self.input_h
        if len(output_shape) >= 2:
            self.output_h = int(output_shape[-2])
            self.output_w = int(output_shape[-1])

        input_vol = 1 * 3 * self.input_h * self.input_w
        output_vol = 1 * self.output_h * self.output_w
        self.h_input = cuda.pagelocked_empty(input_vol, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_vol, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_tensor_address(name, int(self.d_input))
            else:
                self.context.set_tensor_address(name, int(self.d_output))

        self._ready = True
        print(f"[depth] CUDA context initialized on thread "
              f"{threading.current_thread().name} "
              f"for {self.engine_name} ({self.input_w}x{self.input_h})")

    def close(self):
        ctx = getattr(self, "_cuda_ctx", None)
        if ctx is None:
            return
        try:
            ctx.pop()
        except Exception:
            pass
        try:
            ctx.detach()
        except Exception:
            pass
        self._cuda_ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _preprocess(self, bgr_frame):
        """Resize, normalize, transpose to CHW float32."""
        img = cv2.resize(bgr_frame, (self.input_w, self.input_h),
                         interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - _DEPTH_MEAN) / _DEPTH_STD
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(img[None], dtype=np.float32)

    def estimate(self, bgr_frame):
        """Run depth inference. Returns float32 depth map at frame resolution."""
        if not self._ready:
            self._lazy_init()

        h, w = bgr_frame.shape[:2]
        preprocessed = self._preprocess(bgr_frame)

        cuda = self._cuda
        t0 = time.time()
        np.copyto(self.h_input, preprocessed.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        self.last_inference_ms = (time.time() - t0) * 1000

        depth = self.h_output.reshape(self.output_h, self.output_w)
        depth = cv2.resize(depth, (w, h))

        with self._lock:
            self.last_depth_map = depth
        return depth

    def enrich_detections(self, detections, depth_map):
        """Attach relative depth cues to detections.

        Samples a small patch around each bbox center for robustness.
        Metric object distance is intentionally left to bbox geometry when
        available; raw monocular depth is not treated as meters here.
        """
        if depth_map is None or len(detections) == 0:
            return
        h, w = depth_map.shape[:2]
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            # Sample 5x5 patch around bbox center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = 5
            patch = depth_map[max(0, cy-r):min(h, cy+r+1),
                              max(0, cx-r):min(w, cx+r+1)]
            if patch.size > 0:
                raw_depth = float(np.median(patch))
                d["depth_rel"] = round(raw_depth, 4)
                if d_max - d_min > 1e-6:
                    closeness = (raw_depth - d_min) / (d_max - d_min + 1e-6)
                    d["depth_closeness"] = round(float(closeness), 3)

    def colorize(self, depth_map):
        """Convert depth map to BGR colorized image for visualization."""
        d = depth_map.copy()
        usable_bottom = _depth_usable_bottom(d.shape[0])
        visible = d[:usable_bottom, :] if usable_bottom > 0 else d
        dmin, dmax = visible.min(), visible.max()
        if dmax - dmin > 1e-6:
            d = (d - dmin) / (dmax - dmin) * 255.0
        else:
            d = np.zeros_like(d)
        if usable_bottom < d.shape[0]:
            d[usable_bottom:, :] = 0.0
        return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_INFERNO)
