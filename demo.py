# SPDX-License-Identifier: MIT
"""
Turnkey Demo Application for Orin TensorRT Vision Pipeline.
Instantly executable on both desktop development environments and Jetson Orin edge boards.

Usage:
    # Run with default synthetic flight-pattern benchmark (Zero-dependency test)
    python demo.py

    # Run with custom config
    python demo.py --config config.example.yaml

    # Run with RTSP camera and local preview
    python demo.py --source "rtsp://admin:pass@192.168.1.100:554/stream" --display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import OrinVisionPipeline
from src.utils.config_parser import load_config


def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Vision Pipeline on NVIDIA Jetson Orin with TensorRT 10"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.example.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video input URI (e.g. 'synthetic', 'rtsp://...', or '0' for webcam)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Override TensorRT engine or model path (e.g. models/yolov8s.engine)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Force enable local OpenCV display window preview",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional auto-stop duration in seconds",
    )

    args = parser.parse_args()

    # Load base configuration
    cfg = load_config(args.config if Path(args.config).is_file() else None)

    # Apply CLI overrides
    if args.source:
        cfg.input.uri = args.source
        cfg.input.source_type = "synthetic" if args.source == "synthetic" else "rtsp"
    if args.engine:
        cfg.inference.engine_path = args.engine
    if args.display:
        cfg.streamer.enabled = True
        cfg.streamer.output_type = "display"

    print("=" * 65)
    print("  NVIDIA JETSON ORIN - TENSORRT 10 HIGH-THROUGHPUT PIPELINE")
    print("=" * 65)
    print(f" Source Type     : {cfg.input.source_type} ({cfg.input.uri})")
    print(f" Resolution      : {cfg.input.width}x{cfg.input.height} @ {cfg.input.fps} FPS")
    print(f" Inference Model : {cfg.inference.engine_path}")
    print(f" Half Precision  : {cfg.inference.half_precision}")
    print(f" Egress Output   : {cfg.streamer.output_type if cfg.streamer.enabled else 'Disabled'}")
    print("=" * 65)
    print(" Press Ctrl+C in terminal (or ESC on preview window) to terminate.\n")

    pipeline = OrinVisionPipeline(config=cfg)

    # Optional auto-stop timer for automated tests/benchmarks
    if args.duration:
        import threading
        import time

        def auto_stop():
            time.sleep(args.duration)
            print(f"\n[Timer] Reached test duration of {args.duration}s. Halting pipeline...")
            pipeline.stop()

        timer = threading.Thread(target=auto_stop, daemon=True)
        timer.start()

    pipeline.start()


if __name__ == "__main__":
    main()
