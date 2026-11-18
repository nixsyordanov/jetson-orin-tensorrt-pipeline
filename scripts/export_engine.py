# SPDX-License-Identifier: MIT
"""
Model Export & TensorRT Engine Compilation Utility.
Converts PyTorch (.pt) or ONNX models to optimized TensorRT 10 FP16 engines.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("ExportEngine")


def export_with_ultralytics(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    batch: int = 1,
    device: str = "0",
    workspace_gb: int = 4,
) -> str:
    """Exports model using Ultralytics TensorRT export pipeline."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Install via: pip install ultralytics")
        sys.exit(1)

    logger.info("Loading source model: %s", model_path)
    model = YOLO(model_path)

    logger.info(
        "Initiating TensorRT export (FP16=%s, Batch=%d, ImgSz=%d, Workspace=%dGB)...",
        half,
        batch,
        imgsz,
        workspace_gb,
    )
    exported_engine = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        batch=batch,
        device=device,
        workspace=workspace_gb,
        simplify=True,
    )
    logger.info("TensorRT engine compiled successfully: %s", exported_engine)
    return str(exported_engine)


def export_with_trtexec(
    onnx_path: str,
    output_engine: str,
    imgsz: int = 640,
    half: bool = True,
    timing_cache: str = "timing.cache",
) -> None:
    """Directly compiles ONNX model using NVIDIA's trtexec CLI tool."""
    trtexec_bin = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_bin):
        logger.error("trtexec binary not found at %s. Please verify TensorRT installation.", trtexec_bin)
        sys.exit(1)

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine}",
        f"--timingCacheFile={timing_cache}",
    ]
    if half:
        cmd.append("--fp16")

    logger.info("Running trtexec command:\n%s", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error("trtexec export failed with return code %d", res.returncode)
        sys.exit(res.returncode)
    logger.info("Engine built successfully via trtexec: %s", output_engine)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch/ONNX models to TensorRT 10 FP16.")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Path to input weights (.pt or .onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution square dimension")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 half precision")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace limit in GB")
    parser.add_argument("--use-trtexec", action="store_true", help="Compile via trtexec instead of Python API")
    parser.add_argument("--output", type=str, default=None, help="Explicit output path for compiled .engine")

    args = parser.parse_args()
    weights_path = Path(args.weights)

    if not weights_path.is_file():
        logger.error("Input model file not found: %s", weights_path.resolve())
        sys.exit(1)

    if args.use_trtexec or weights_path.suffix == ".onnx":
        out_engine = args.output or str(weights_path.with_suffix(".engine"))
        export_with_trtexec(
            onnx_path=str(weights_path),
            output_engine=out_engine,
            imgsz=args.imgsz,
            half=not args.no_fp16,
        )
    else:
        export_with_ultralytics(
            model_path=str(weights_path),
            imgsz=args.imgsz,
            half=not args.no_fp16,
            batch=args.batch,
            device=args.device,
            workspace_gb=args.workspace,
        )


if __name__ == "__main__":
    main()
