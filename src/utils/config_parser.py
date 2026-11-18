# SPDX-License-Identifier: MIT
"""
Configuration Parser & Validator for Orin Vision Pipeline.
Loads YAML configurations with fallback defaults and environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class VideoInputConfig:
    source_type: str = "synthetic"  # synthetic, rtsp, v4l2, file
    uri: str = "test-pattern"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    drop_frames: bool = True
    gstreamer_pipeline: Optional[str] = None


@dataclass
class InferenceConfig:
    engine_path: str = "models/yolov8s.engine"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    input_width: int = 640
    input_height: int = 640
    half_precision: bool = True
    device: str = "cuda:0"
    max_batch_size: int = 1


@dataclass
class VisualizerConfig:
    enabled: bool = True
    show_hud: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_trails: bool = True
    trail_length: int = 30
    box_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class StreamerConfig:
    enabled: bool = False
    output_type: str = "display"  # display, srt, udp, rtsp, file
    target_uri: str = "srt://127.0.0.1:9000?mode=caller"
    bitrate_kbps: int = 4000
    fps: int = 30
    preset: str = "ultrafast"
    tune: str = "zerolatency"


@dataclass
class TelemetryConfig:
    enabled: bool = True
    log_interval_sec: float = 2.0
    tegrastats_enabled: bool = True


@dataclass
class AppConfig:
    input: VideoInputConfig = field(default_factory=VideoInputConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    streamer: StreamerConfig = field(default_factory=StreamerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, merging with default values."""
    config = AppConfig()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Parse input block
    inp = raw.get("input", {})
    config.input = VideoInputConfig(
        source_type=inp.get("source_type", config.input.source_type),
        uri=os.path.expandvars(inp.get("uri", config.input.uri)),
        width=int(inp.get("width", config.input.width)),
        height=int(inp.get("height", config.input.height)),
        fps=int(inp.get("fps", config.input.fps)),
        drop_frames=bool(inp.get("drop_frames", config.input.drop_frames)),
        gstreamer_pipeline=inp.get("gstreamer_pipeline", None),
    )

    # Parse inference block
    inf = raw.get("inference", {})
    config.inference = InferenceConfig(
        engine_path=os.path.expandvars(inf.get("engine_path", config.inference.engine_path)),
        confidence_threshold=float(inf.get("confidence_threshold", config.inference.confidence_threshold)),
        iou_threshold=float(inf.get("iou_threshold", config.inference.iou_threshold)),
        input_width=int(inf.get("input_width", config.inference.input_width)),
        input_height=int(inf.get("input_height", config.inference.input_height)),
        half_precision=bool(inf.get("half_precision", config.inference.half_precision)),
        device=str(inf.get("device", config.inference.device)),
        max_batch_size=int(inf.get("max_batch_size", config.inference.max_batch_size)),
    )

    # Parse visualizer block
    vis = raw.get("visualizer", {})
    config.visualizer = VisualizerConfig(
        enabled=bool(vis.get("enabled", config.visualizer.enabled)),
        show_hud=bool(vis.get("show_hud", config.visualizer.show_hud)),
        show_labels=bool(vis.get("show_labels", config.visualizer.show_labels)),
        show_conf=bool(vis.get("show_conf", config.visualizer.show_conf)),
        show_trails=bool(vis.get("show_trails", config.visualizer.show_trails)),
        trail_length=int(vis.get("trail_length", config.visualizer.trail_length)),
        box_thickness=int(vis.get("box_thickness", config.visualizer.box_thickness)),
        font_scale=float(vis.get("font_scale", config.visualizer.font_scale)),
    )

    # Parse streamer block
    stm = raw.get("streamer", {})
    config.streamer = StreamerConfig(
        enabled=bool(stm.get("enabled", config.streamer.enabled)),
        output_type=stm.get("output_type", config.streamer.output_type),
        target_uri=os.path.expandvars(stm.get("target_uri", config.streamer.target_uri)),
        bitrate_kbps=int(stm.get("bitrate_kbps", config.streamer.bitrate_kbps)),
        fps=int(stm.get("fps", config.streamer.fps)),
        preset=stm.get("preset", config.streamer.preset),
        tune=stm.get("tune", config.streamer.tune),
    )

    # Parse telemetry block
    tel = raw.get("telemetry", {})
    config.telemetry = TelemetryConfig(
        enabled=bool(tel.get("enabled", config.telemetry.enabled)),
        log_interval_sec=float(tel.get("log_interval_sec", config.telemetry.log_interval_sec)),
        tegrastats_enabled=bool(tel.get("tegrastats_enabled", config.telemetry.tegrastats_enabled)),
    )

    return config
