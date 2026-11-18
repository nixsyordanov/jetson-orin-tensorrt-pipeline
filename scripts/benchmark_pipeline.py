# SPDX-License-Identifier: MIT
"""
Pipeline & TensorRT Engine Latency Benchmark Tool.
Measures latency percentiles (P50, P95, P99), throughput (FPS), and hardware metrics.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import TensorRTEngine
from src.utils.metrics import JetsonTelemetry

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("Benchmark")


def run_benchmark(engine_path: str, imgsz: int = 640, frames: int = 500, half: bool = True):
    logger.info("Initializing benchmark on engine: %s", engine_path)
    engine = TensorRTEngine(
        engine_path=engine_path,
        input_size=(imgsz, imgsz),
        half_precision=half,
    )

    telemetry = JetsonTelemetry(enabled=True)

    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    logger.info("Running engine warmup (20 iterations)...")
    for _ in range(20):
        _ = engine.infer(dummy_frame)

    latencies_ms = []
    logger.info("Benchmarking %d inference runs...", frames)
    t_start = time.perf_counter()

    for i in range(frames):
        t0 = time.perf_counter()
        _ = engine.infer(dummy_frame)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_time = time.perf_counter() - t_start
    arr = np.array(latencies_ms)

    mean_lat = np.mean(arr)
    median_lat = np.median(arr)
    p95_lat = np.percentile(arr, 95)
    p99_lat = np.percentile(arr, 99)
    std_lat = np.std(arr)
    effective_fps = frames / total_time
    max_throughput_fps = 1000.0 / mean_lat if mean_lat > 0 else 0

    hw_info = telemetry.get_snapshot() if telemetry.enabled else {}
    telemetry.stop()

    print("\n" + "=" * 60)
    print("        ORIN TENSORRT BENCHMARK REPORT")
    print("=" * 60)
    print(f" Target Model        : {engine_path}")
    print(f" Precision           : {'FP16 (Half)' if half else 'FP32'}")
    print(f" Resolution          : {imgsz}x{imgsz}")
    print(f" Sample Count        : {frames} iterations")
    print("-" * 60)
    print(f" Mean Latency        : {mean_lat:6.2f} ms")
    print(f" Median (P50)        : {median_lat:6.2f} ms")
    print(f" 95th Percentile     : {p95_lat:6.2f} ms")
    print(f" 99th Percentile     : {p99_lat:6.2f} ms")
    print(f" Latency StdDev      : {std_lat:6.2f} ms")
    print("-" * 60)
    print(f" Measured Throughput : {effective_fps:6.1f} FPS")
    print(f" Max Theoretical FPS : {max_throughput_fps:6.1f} FPS")
    if hw_info:
        print(f" GPU Utilization     : {hw_info.get('gpu_util_pct', 0.0):.0f}%")
        print(f" Power Draw          : {hw_info.get('power_w', 0.0):.1f} W")
        print(f" Core Temperature    : {hw_info.get('gpu_temp_c', 0.0):.1f} C")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT inference latency on Jetson.")
    parser.add_argument("--engine", type=str, default="models/yolov8s.engine", help="Path to engine file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input dimension")
    parser.add_argument("--frames", type=int, default=300, help="Number of benchmark iterations")
    parser.add_argument("--no-fp16", action="store_true", help="Disable half precision")

    args = parser.parse_args()
    run_benchmark(args.engine, args.imgsz, args.frames, half=not args.no_fp16)


if __name__ == "__main__":
    main()
