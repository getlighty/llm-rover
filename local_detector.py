"""
Local object detection using YOLOv8 ONNX / TensorRT.
Supports both YOLOv8n (COCO-80) and YOLO-World (custom classes).
TensorRT FP16 GPU inference (~20+ FPS), falls back to OpenCV DNN CPU.

Based on:
- NVIDIA JetBot object_following example (SSD + proportional control)
- Waveshare ugv_jetson cv_ctrl.py (OpenCV MobileNet SSD)
- Ultralytics YOLOv8-OpenCV-ONNX-Python example

Usage:
    detector = LocalDetector()                    # YOLO-World TRT (default)
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
COCO_MODEL = os.path.join(MODEL_DIR, "yolov8n.onnx")

# YOLO-World workshop model (42 indoor/workshop classes, open-vocabulary baked in)
WORLD_MODEL_PT = os.path.join(MODEL_DIR, "yolov8s-world-workshop.pt")
WORLD_MODEL_ONNX = os.path.join(MODEL_DIR, "yolov8s-world-workshop.onnx")
WORLD_MODEL_TRT = os.path.join(MODEL_DIR, "yolov8s-world-workshop.engine")

# Default: prefer TRT engine > PT (PyTorch GPU) > ONNX World > ONNX COCO
def _pick_default_model():
    if os.path.exists(WORLD_MODEL_TRT):
        return WORLD_MODEL_TRT
    if os.path.exists(WORLD_MODEL_PT):
        return WORLD_MODEL_PT
    if os.path.exists(WORLD_MODEL_ONNX):
        return WORLD_MODEL_ONNX
    return COCO_MODEL

DEFAULT_MODEL = _pick_default_model()
WORLD_NAMES = [
    "person", "door", "chair", "table", "desk",
    "shelf", "cabinet", "box", "container", "bin",
    "recycle bin", "trash can", "bottle", "cup", "bowl",
    "monitor", "laptop", "keyboard", "mouse", "cable",
    "light", "lamp", "fan", "clock", "phone",
    "backpack", "bag", "book", "plant", "window",
    "wall", "floor", "ceiling", "rug",
    "toolbox", "tool", "screwdriver", "drill",
    "wheel", "robot", "speaker", "camera",
]

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
        if "world" in basename or "workshop" in basename:
            self.class_names = WORLD_NAMES
            self._model_label = "YOLO-World"
        else:
            self.class_names = COCO_NAMES
            self._model_label = "YOLOv8n"

        # Backend state
        self._ultralytics_model = None  # PyTorch (Ultralytics YOLO)
        self._trt_context = None        # TensorRT
        self.net = None                 # OpenCV DNN

        if model_path.endswith(".pt"):
            self._init_pytorch(model_path)
        elif model_path.endswith(".engine"):
            self._init_tensorrt(model_path)
        else:
            self._init_opencv_dnn(model_path)

        print(f"[detector] {self._model_label} loaded ({self.backend}, "
              f"{len(self.class_names)} classes, conf={conf})")

    def _init_pytorch(self, model_path):
        """Load model via Ultralytics for PyTorch GPU inference."""
        try:
            import torch
            from ultralytics import YOLO
            self._ultralytics_model = YOLO(model_path)
            # Read class names from model (handles custom-trained models)
            if hasattr(self._ultralytics_model, "names") and self._ultralytics_model.names:
                self.class_names = list(self._ultralytics_model.names.values())
                if len(self.class_names) == 1:
                    self._model_label = f"YOLOv8-{self.class_names[0]}"
                else:
                    self._model_label = f"YOLOv8-custom({len(self.class_names)}cls)"
            if torch.cuda.is_available():
                self._ultralytics_model.to("cuda")
                self.backend = "PyTorch GPU"
            else:
                self.backend = "PyTorch CPU"
            # Warmup
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._ultralytics_model.predict(dummy, verbose=False, conf=self.conf,
                                            half=True)
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
        results = self._ultralytics_model.predict(
            frame, verbose=False, conf=self.conf, iou=self.nms_thresh,
            half=True, imgsz=self.input_size,
        )
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
                known_w = KNOWN_WIDTHS.get(name)
                if known_w and bw_px > 5:
                    det["dist_m"] = round((known_w * self.focal_length) / bw_px, 2)
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
                known_w = KNOWN_WIDTHS.get(names[idx])
                if known_w and bw_px > 5:
                    det["dist_m"] = round((known_w * self.focal_length) / bw_px, 2)
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
