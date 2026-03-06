#!/usr/bin/env python3
"""Build a Depth Anything V2 TensorRT engine on-device.

This script:
1. clones the official Depth-Anything-V2 repo into a temp dir
2. downloads the requested checkpoint if needed
3. exports a static ONNX model for the selected input size
4. builds a TensorRT FP16 engine with trtexec

Example:
    python3 scripts/build_depth_anything_v2.py --encoder vits --input-size 308
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2"
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def download_checkpoint(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Checkpoint exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-L", "--fail", "-o", str(out_path), url])


def clone_repo(dest: Path):
    run(["git", "clone", "--depth", "1", REPO_URL, str(dest)])


def export_onnx(repo_dir: Path, checkpoint: Path, encoder: str,
                input_size: int, onnx_out: Path):
    sys.path.insert(0, str(repo_dir))
    module = importlib.import_module("depth_anything_v2.dpt")
    DepthAnythingV2 = getattr(module, "DepthAnythingV2")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        test_out = model(dummy)
    print(f"Export sanity output: shape={tuple(test_out.shape)}")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        input_names=["input"],
        output_names=["depth"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def build_engine(onnx_path: Path, engine_path: Path):
    if not TRTEXEC.exists():
        raise FileNotFoundError(f"trtexec not found at {TRTEXEC}")
    run([
        str(TRTEXEC),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--skipInference",
    ])


def write_metadata(engine_path: Path, encoder: str, input_size: int, checkpoint: Path,
                   onnx_path: Path):
    meta = {
        "family": "depth_anything_v2",
        "encoder": encoder,
        "input_size": input_size,
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": "fp16",
    }
    sidecar = engine_path.with_suffix(engine_path.suffix + ".json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote metadata: {sidecar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=308)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional local checkpoint path.")
    parser.add_argument("--onnx-out", default=None)
    parser.add_argument("--engine-out", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = (repo_root / args.models_dir).resolve()
    ckpt_name = f"depth_anything_v2_{args.encoder}.pth"
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else models_dir / ckpt_name
    onnx_path = Path(args.onnx_out).resolve() if args.onnx_out else (
        models_dir / f"depth_anything_v2_{args.encoder}14_{args.input_size}.onnx"
    )
    engine_path = Path(args.engine_out).resolve() if args.engine_out else (
        models_dir / f"depth_anything_v2_{args.encoder}14_{args.input_size}.trt"
    )

    download_checkpoint(MODEL_URLS[args.encoder], ckpt_path)

    with tempfile.TemporaryDirectory(prefix="depth-anything-v2.") as tmp:
        repo_dir = Path(tmp) / "Depth-Anything-V2"
        clone_repo(repo_dir)
        export_onnx(repo_dir, ckpt_path, args.encoder, args.input_size, onnx_path)
        build_engine(onnx_path, engine_path)
        write_metadata(engine_path, args.encoder, args.input_size, ckpt_path, onnx_path)

    print(f"Built engine: {engine_path}")


if __name__ == "__main__":
    main()
