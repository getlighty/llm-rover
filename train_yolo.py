#!/usr/bin/env python3
"""
YOLO Training Pipeline — Capture, Annotate, Train, Test
========================================================
Usage:
  python3 train_yolo.py capture       # Capture images from rover camera
  python3 train_yolo.py annotate      # Auto-annotate using YOLO-World
  python3 train_yolo.py train         # Train YOLOv8n on annotated dataset
  python3 train_yolo.py test [image]  # Test trained model on an image
  python3 train_yolo.py export        # Export to ONNX/TensorRT for production
"""

import os
import sys
import time
import json
import glob
import shutil
import argparse
import requests
import cv2
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_IMAGES = IMAGES_DIR / "train"
TRAIN_LABELS = LABELS_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
VAL_LABELS = LABELS_DIR / "val"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
DATASET_YAML = DATASET_DIR / "dataset.yaml"

SNAP_URL = "http://0.0.0.0:8090/snap"

# ── Classes to train ───────────────────────────────────────────
# Add more classes here as needed
CLASSES = ["blue basket"]

# YOLO-World prompts for auto-annotation (can be broader to catch variants)
YOLO_WORLD_PROMPTS = ["basket", "blue basket", "blue bin", "blue container"]


def setup_dirs():
    """Create dataset directory structure."""
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {DATASET_DIR}")


def write_dataset_yaml():
    """Write YOLO dataset.yaml config."""
    yaml_content = f"""# Auto-generated dataset config
path: {DATASET_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    DATASET_YAML.write_text(yaml_content)
    print(f"Wrote {DATASET_YAML}")


# ══════════════════════════════════════════════════════════════════
# CAPTURE — grab images from rover camera
# ══════════════════════════════════════════════════════════════════

def cmd_capture(args):
    """Capture images from the rover camera stream."""
    setup_dirs()
    target_dir = TRAIN_IMAGES
    existing = len(list(target_dir.glob("*.jpg")))
    count = existing

    print(f"\n=== IMAGE CAPTURE ===")
    print(f"Saving to: {target_dir}")
    print(f"Existing images: {existing}")
    print(f"Source: {SNAP_URL}")
    print(f"\nInstructions:")
    print(f"  - Position the blue basket in different locations/angles")
    print(f"  - Press ENTER to capture a frame")
    print(f"  - Type 'burst N' to capture N frames rapidly (for gimbal sweep)")
    print(f"  - Type 'auto' to auto-capture while you move the basket")
    print(f"  - Type 'done' when finished\n")

    while True:
        try:
            cmd = input(f"[{count} captured] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "done" or cmd == "q":
            break

        elif cmd.startswith("burst"):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            interval = float(parts[2]) if len(parts) > 2 else 0.5
            print(f"Burst capturing {n} frames ({interval}s interval)...")
            for i in range(n):
                if _capture_one(target_dir, count):
                    count += 1
                    print(f"  [{i+1}/{n}] captured #{count}")
                time.sleep(interval)

        elif cmd == "auto":
            print("Auto-capture mode (1 frame/sec). Press Ctrl+C to stop.")
            try:
                while True:
                    if _capture_one(target_dir, count):
                        count += 1
                        print(f"  captured #{count}")
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nStopped auto-capture.")

        else:
            # Single capture on ENTER
            if _capture_one(target_dir, count):
                count += 1
                print(f"  captured #{count}")
            else:
                print("  Failed to capture frame")

    print(f"\nDone! {count - existing} new images captured ({count} total)")


def _capture_one(target_dir, index):
    """Capture a single frame from the rover camera."""
    try:
        r = requests.get(SNAP_URL, timeout=5)
        if r.status_code != 200:
            return False
        img_data = r.content
        fname = target_dir / f"img_{index:04d}.jpg"
        fname.write_bytes(img_data)
        return True
    except Exception as e:
        print(f"  Capture error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# ANNOTATE — auto-annotate using YOLO-World zero-shot
# ══════════════════════════════════════════════════════════════════

def cmd_annotate(args):
    """Auto-annotate images using YOLO-World zero-shot detection."""
    from ultralytics import YOLO

    setup_dirs()

    # Find images to annotate
    images = sorted(TRAIN_IMAGES.glob("*.jpg"))
    if not images:
        print(f"No images found in {TRAIN_IMAGES}")
        print("Run 'python3 train_yolo.py capture' first.")
        return

    # Load YOLO-World for zero-shot detection
    world_model_path = MODELS_DIR / "yolov8s-world-workshop.pt"
    if not world_model_path.exists():
        # Fall back to downloading yolov8s-worldv2
        print("Downloading yolov8s-worldv2 for annotation...")
        world_model_path = "yolov8s-worldv2.pt"

    print(f"Loading YOLO-World from {world_model_path}...")
    model = YOLO(str(world_model_path))

    # Set custom classes for zero-shot detection
    model.set_classes(YOLO_WORLD_PROMPTS)
    print(f"Set detection classes: {YOLO_WORLD_PROMPTS}")

    conf_threshold = args.conf if hasattr(args, 'conf') else 0.15
    annotated = 0
    skipped = 0
    total = len(images)

    print(f"\nAnnotating {total} images (conf >= {conf_threshold})...\n")

    for i, img_path in enumerate(images):
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")

        # Run inference
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            verbose=False,
            imgsz=640,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            skipped += 1
            if args.verbose:
                print(f"  [{i+1}/{total}] {img_path.name}: no detections (skipped)")
            continue

        # Convert to YOLO format: class_id cx cy w h (normalized)
        img_h, img_w = result.orig_shape
        lines = []

        for box in boxes:
            # All detections map to class 0 ("blue basket")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Normalize to 0-1
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # All detections are class 0 (our single target class)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines) + "\n")
        annotated += 1

        if args.verbose or (i + 1) % 10 == 0:
            n_boxes = len(lines)
            print(f"  [{i+1}/{total}] {img_path.name}: {n_boxes} box(es) "
                  f"(conf={float(boxes.conf.max()):.2f})")

    print(f"\nAnnotation complete:")
    print(f"  Annotated: {annotated}/{total}")
    print(f"  Skipped (no detection): {skipped}/{total}")
    print(f"  Labels in: {TRAIN_LABELS}")

    if annotated == 0:
        print("\n⚠  No images were annotated! The basket may not be visible")
        print("   or confidence is too low. Try: --conf 0.10")
        return

    # Auto-split: move ~20% to val
    _split_val(ratio=0.2)

    # Write dataset config
    write_dataset_yaml()

    # Show preview
    if annotated > 0:
        print(f"\nPreview annotations at: http://localhost:8090")
        _preview_annotations(images[0] if annotated > 0 else None)


def _split_val(ratio=0.2):
    """Move a portion of training images+labels to validation set."""
    images = sorted(TRAIN_IMAGES.glob("*.jpg"))
    n_val = max(1, int(len(images) * ratio))

    # Pick evenly spaced images for validation
    indices = np.linspace(0, len(images) - 1, n_val, dtype=int)

    moved = 0
    for idx in indices:
        img_path = images[idx]
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")

        if not label_path.exists():
            continue

        shutil.move(str(img_path), str(VAL_IMAGES / img_path.name))
        shutil.move(str(label_path), str(VAL_LABELS / label_path.name))
        moved += 1

    print(f"  Split: {moved} images moved to validation set")


def _preview_annotations(img_path):
    """Show annotation stats for an image."""
    if img_path is None:
        return
    label_path = TRAIN_LABELS / (img_path.stem + ".txt")
    if label_path.exists():
        lines = label_path.read_text().strip().split("\n")
        print(f"  Example: {img_path.name} → {len(lines)} bounding box(es)")


# ══════════════════════════════════════════════════════════════════
# TRAIN — fine-tune YOLOv8n on the annotated dataset
# ══════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train YOLOv8n on the annotated dataset."""
    from ultralytics import YOLO

    if not DATASET_YAML.exists():
        print(f"No dataset config found at {DATASET_YAML}")
        print("Run 'python3 train_yolo.py annotate' first.")
        return

    # Count images
    n_train = len(list(TRAIN_IMAGES.glob("*.jpg")))
    n_val = len(list(VAL_IMAGES.glob("*.jpg")))
    n_labels = len(list(TRAIN_LABELS.glob("*.txt")))
    print(f"Dataset: {n_train} train images, {n_val} val images, {n_labels} train labels")

    if n_train < 5:
        print("⚠  Very few training images. Capture more for better results.")
        print("   Recommend at least 30-50 images from different angles.")

    # Load pretrained YOLOv8n
    base_model = args.model if hasattr(args, 'model') else "yolov8n.pt"
    print(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)

    epochs = args.epochs if hasattr(args, 'epochs') else 50
    batch = args.batch if hasattr(args, 'batch') else 8
    imgsz = args.imgsz if hasattr(args, 'imgsz') else 640

    print(f"Training: epochs={epochs}, batch={batch}, imgsz={imgsz}")
    print(f"Output: {RUNS_DIR}/\n")

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name="blue_basket",
        exist_ok=True,
        device=0,           # GPU
        half=True,           # FP16 for Orin
        workers=2,           # Low worker count for Jetson
        patience=15,         # Early stopping
        save=True,
        plots=True,
        augment=True,
        # Augmentation for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
    )

    # Copy best model to models/
    best_path = RUNS_DIR / "blue_basket" / "weights" / "best.pt"
    if best_path.exists():
        dest = MODELS_DIR / "yolov8n-blue-basket.pt"
        shutil.copy2(str(best_path), str(dest))
        print(f"\n✓ Best model saved to: {dest}")
        print(f"  To use: update local_detector.py or search_engine.py")
    else:
        print(f"\n⚠  No best.pt found at {best_path}")

    return results


# ══════════════════════════════════════════════════════════════════
# TEST — run inference with trained model
# ══════════════════════════════════════════════════════════════════

def cmd_test(args):
    """Test the trained model on an image or live camera."""
    from ultralytics import YOLO

    model_path = MODELS_DIR / "yolov8n-blue-basket.pt"
    if not model_path.exists():
        print(f"No trained model at {model_path}")
        print("Run 'python3 train_yolo.py train' first.")
        return

    model = YOLO(str(model_path))
    print(f"Loaded model: {model_path}")

    if args.image:
        # Test on specific image
        _test_image(model, args.image)
    else:
        # Test on live camera
        _test_live(model)


def _test_image(model, image_path):
    """Run inference on a single image."""
    results = model.predict(image_path, conf=0.25, verbose=True)
    result = results[0]

    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = CLASSES[cls] if cls < len(CLASSES) else f"class_{cls}"
            print(f"  {name}: conf={conf:.2f} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
    else:
        print("  No detections")

    # Save annotated image
    out_path = Path(image_path).stem + "_detected.jpg"
    result.save(out_path)
    print(f"Saved: {out_path}")


def _test_live(model):
    """Test on live camera frames from rover."""
    print(f"\nTesting on live camera ({SNAP_URL})")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            r = requests.get(SNAP_URL, timeout=5)
            if r.status_code != 200:
                print("No frame")
                time.sleep(1)
                continue

            # Decode JPEG
            img = cv2.imdecode(
                np.frombuffer(r.content, np.uint8),
                cv2.IMREAD_COLOR)

            t0 = time.monotonic()
            results = model.predict(img, conf=0.25, verbose=False)
            dt = (time.monotonic() - t0) * 1000

            result = results[0]
            n = len(result.boxes) if result.boxes else 0

            if n > 0:
                confs = [f"{float(b.conf[0]):.2f}" for b in result.boxes]
                print(f"  {n} detection(s) [{', '.join(confs)}] ({dt:.0f}ms)")
            else:
                print(f"  no detections ({dt:.0f}ms)")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")


# ══════════════════════════════════════════════════════════════════
# EXPORT — convert trained model to ONNX/TensorRT
# ══════════════════════════════════════════════════════════════════

def cmd_export(args):
    """Export trained model to ONNX and optionally TensorRT."""
    from ultralytics import YOLO

    model_path = MODELS_DIR / "yolov8n-blue-basket.pt"
    if not model_path.exists():
        print(f"No trained model at {model_path}")
        return

    model = YOLO(str(model_path))

    # Export to ONNX
    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=640, half=True, simplify=True)
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"  ONNX: {onnx_path} ({size_mb:.1f} MB)")

    # Try TensorRT (may fail on Orin Nano for large models)
    if args.trt:
        print("Exporting to TensorRT (this may take a few minutes)...")
        try:
            model.export(format="engine", imgsz=640, half=True)
            engine_path = model_path.with_suffix(".engine")
            if engine_path.exists():
                size_mb = engine_path.stat().st_size / 1024 / 1024
                print(f"  TensorRT: {engine_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  TensorRT export failed: {e}")
            print("  (YOLOv8n should be small enough for Orin Nano)")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="YOLO Training Pipeline for Rover Object Detection")
    sub = parser.add_subparsers(dest="command")

    # capture
    p_cap = sub.add_parser("capture", help="Capture images from rover camera")

    # annotate
    p_ann = sub.add_parser("annotate", help="Auto-annotate using YOLO-World")
    p_ann.add_argument("--conf", type=float, default=0.15,
                       help="Confidence threshold (default: 0.15)")
    p_ann.add_argument("--verbose", "-v", action="store_true")

    # train
    p_trn = sub.add_parser("train", help="Train YOLOv8n on annotated dataset")
    p_trn.add_argument("--epochs", type=int, default=50)
    p_trn.add_argument("--batch", type=int, default=8)
    p_trn.add_argument("--imgsz", type=int, default=640)
    p_trn.add_argument("--model", default="yolov8n.pt",
                       help="Base model (default: yolov8n.pt)")

    # test
    p_tst = sub.add_parser("test", help="Test trained model")
    p_tst.add_argument("image", nargs="?", help="Image path (or live camera)")

    # export
    p_exp = sub.add_parser("export", help="Export model to ONNX/TensorRT")
    p_exp.add_argument("--trt", action="store_true",
                       help="Also export to TensorRT")

    args = parser.parse_args()

    if args.command == "capture":
        cmd_capture(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()
        print("\nTypical workflow:")
        print("  1. python3 train_yolo.py capture      # Grab 30-50 images")
        print("  2. python3 train_yolo.py annotate      # Auto-label with YOLO-World")
        print("  3. python3 train_yolo.py train          # Fine-tune YOLOv8n")
        print("  4. python3 train_yolo.py test           # Verify on live camera")
        print("  5. python3 train_yolo.py export --trt   # Export for production")


if __name__ == "__main__":
    main()
