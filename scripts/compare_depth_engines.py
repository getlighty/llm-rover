#!/usr/bin/env python3
"""Compare two depth engines on a single image."""

import argparse
from pathlib import Path

import cv2

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from local_detector import DepthEstimator  # noqa: E402


def run_engine(engine_path: Path, image_path: Path, out_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    est = DepthEstimator(engine_path=str(engine_path))
    try:
        depth = est.estimate(img)
        color = est.colorize(depth)
        out_path = out_dir / f"{engine_path.stem}.jpg"
        cv2.imwrite(str(out_path), color)
        print(
            f"{engine_path.name}: {est.last_inference_ms:.1f} ms, "
            f"min={float(depth.min()):.4f}, max={float(depth.max()):.4f}, "
            f"vis={out_path}"
        )
    finally:
        est.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", action="append", required=True,
                        help="Repeat for each engine to compare.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "depth_compare"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for engine in args.engine:
        run_engine(Path(engine), Path(args.image), out_dir)


if __name__ == "__main__":
    main()
