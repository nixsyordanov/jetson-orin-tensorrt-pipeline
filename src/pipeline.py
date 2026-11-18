# SPDX-License-Identifier: MIT
"""
Main Pipeline Orchestrator.
Synchronizes threaded video ingestion, TensorRT execution, OSD rendering, and egress streaming.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional

import cv2

from src.engine import TensorRTEngine
from src.utils.config_parser import AppConfig, load_config
from src.utils.metrics import JetsonTelemetry, RollingFPSMeter, StageProfiler
from src.utils.visualizer import TacticalVisualizer
from src.video.grabber import VideoGrabber
from src.video.streamer import VideoStreamer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OrinPipeline")


class OrinVisionPipeline:
    """End-to-end edge AI vision pipeline optimized for NVIDIA Jetson architectures."""

    def __init__(self, config: Optional[AppConfig] = None, config_path: Optional[str] = None):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = load_config(config_path)

        self._running = False

        # Profilers and metrics
        self.fps_meter = RollingFPSMeter(window_size=60)
        self.profiler = StageProfiler(max_history=120)
        self.telemetry = JetsonTelemetry(enabled=self.cfg.telemetry.tegrastats_enabled)

        # Video ingestion
        self.grabber = VideoGrabber(
            source_type=self.cfg.input.source_type,
            uri=self.cfg.input.uri,
            width=self.cfg.input.width,
            height=self.cfg.input.height,
            fps=self.cfg.input.fps,
            gstreamer_pipeline=self.cfg.input.gstreamer_pipeline,
        )

        # Inference engine
        self.engine = TensorRTEngine(
            engine_path=self.cfg.inference.engine_path,
            confidence_threshold=self.cfg.inference.confidence_threshold,
            iou_threshold=self.cfg.inference.iou_threshold,
            input_size=(self.cfg.inference.input_width, self.cfg.inference.input_height),
            half_precision=self.cfg.inference.half_precision,
            device=self.cfg.inference.device,
        )

        # Tactical OSD visualizer
        self.visualizer = TacticalVisualizer(
            show_hud=self.cfg.visualizer.show_hud,
            show_labels=self.cfg.visualizer.show_labels,
            show_conf=self.cfg.visualizer.show_conf,
            show_trails=self.cfg.visualizer.show_trails,
            trail_length=self.cfg.visualizer.trail_length,
            box_thickness=self.cfg.visualizer.box_thickness,
            font_scale=self.cfg.visualizer.font_scale,
        )

        # Egress streamer
        self.streamer: Optional[VideoStreamer] = None
        if self.cfg.streamer.enabled:
            self.streamer = VideoStreamer(
                output_type=self.cfg.streamer.output_type,
                target_uri=self.cfg.streamer.target_uri,
                width=self.cfg.input.width,
                height=self.cfg.input.height,
                fps=self.cfg.streamer.fps,
                bitrate_kbps=self.cfg.streamer.bitrate_kbps,
                preset=self.cfg.streamer.preset,
                tune=self.cfg.streamer.tune,
            )

        # Signal handlers for clean exits
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Termination signal received. Shutting down pipeline cleanly...")
        self.stop()

    def start(self) -> None:
        """Starts background worker threads and launches the primary processing loop."""
        logger.info("Initializing pipeline threads...")
        self.grabber.start()
        if self.streamer:
            self.streamer.start()

        self._running = True
        logger.info("Pipeline active. Ingestion source: %s", self.cfg.input.source_type)

        last_telemetry_log = time.perf_counter()

        try:
            while self._running:
                # Stage 1: Frame Acquisition
                self.profiler.start("grab_ms")
                ret, frame = self.grabber.read()
                self.profiler.stop("grab_ms")

                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                fps = self.fps_meter.tick()

                # Stage 2: TensorRT Inference
                self.profiler.start("infer_ms")
                detections = self.engine.infer(frame)
                self.profiler.stop("infer_ms")

                # Stage 3: Tactical OSD Visualization
                if self.cfg.visualizer.enabled:
                    self.profiler.start("visualize_ms")
                    stage_latencies = self.profiler.summary()
                    telem_data = self.telemetry.get_snapshot() if self.cfg.telemetry.enabled else None
                    annotated_frame = self.visualizer.draw(
                        frame,
                        detections=detections,
                        fps=fps,
                        latencies=stage_latencies,
                        telemetry=telem_data,
                    )
                    self.profiler.stop("visualize_ms")
                else:
                    annotated_frame = frame

                # Stage 4: Egress / Streaming
                if self.streamer:
                    self.profiler.start("stream_ms")
                    self.streamer.push(annotated_frame)
                    self.profiler.stop("stream_ms")

                # Periodic telemetry logging
                now = time.perf_counter()
                if (now - last_telemetry_log) >= self.cfg.telemetry.log_interval_sec:
                    last_telemetry_log = now
                    lat_summary = self.profiler.summary()
                    gpu_info = self.telemetry.get_snapshot() if self.cfg.telemetry.enabled else {}
                    logger.info(
                        "Pipeline Metrics | FPS: %5.1f | Latency: %4.1f ms (Infer: %4.1f ms) | GPU: %2.0f%% | Targets: %d",
                        fps,
                        lat_summary.get("total_ms", 0.0),
                        lat_summary.get("infer_ms", 0.0),
                        gpu_info.get("gpu_util_pct", 0.0),
                        len(detections),
                    )

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt intercepted.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Terminates workers and releases hardware resources."""
        if not self._running:
            return
        self._running = False
        logger.info("Stopping pipeline workers...")
        self.grabber.release()
        if self.streamer:
            self.streamer.release()
        self.telemetry.stop()
        logger.info("Orin Vision Pipeline successfully halted.")
