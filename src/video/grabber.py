# SPDX-License-Identifier: MIT
"""
Threaded High-Performance Video Ingestion Engine.
Supports NVIDIA GStreamer NVDEC hardware decode, V4L2, RTSP, and synthetic benchmark streams.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_gstreamer_nvdec_pipeline(rtsp_uri: str, width: int = 1920, height: int = 1080) -> str:
    """Constructs an optimized NVIDIA Jetson GStreamer hardware-accelerated decode pipeline."""
    return (
        f"rtspsrc location={rtsp_uri} latency=0 buffer-mode=none drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
        f"nvvidconv ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class SyntheticStreamGenerator:
    """Generates synthetic multi-target flight patterns for zero-dependency benchmarking."""

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_idx = 0

    def generate(self) -> np.ndarray:
        self._frame_idx += 1
        t = self._frame_idx / float(self.fps)

        # Sky gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (140, 110, 85)  # Soft steel blue

        # Horizon / terrain silhouette
        cv2.ellipse(frame, (self.width // 2, self.height + 400), (self.width, 600), 0, 180, 360, (50, 45, 40), -1)

        # Target 1: Sweeping sinusoidal trajectory (simulating bird / UAV)
        cx1 = int((math.sin(t * 0.8) * 0.4 + 0.5) * self.width)
        cy1 = int((math.cos(t * 1.2) * 0.2 + 0.35) * self.height)
        cv2.circle(frame, (cx1, cy1), 16, (20, 20, 20), -1)
        # Wing silhouette
        wing_span = int(24 + 10 * math.sin(t * 12))
        cv2.line(frame, (cx1 - wing_span, cy1 - 4), (cx1 + wing_span, cy1 - 4), (30, 30, 30), 4)

        # Target 2: Fast traversing object
        cx2 = int(((t * 300) % (self.width + 100)) - 50)
        cy2 = int((self.height * 0.2) + math.sin(t * 2.0) * 40)
        if 0 <= cx2 < self.width:
            cv2.circle(frame, (cx2, cy2), 12, (25, 25, 25), -1)

        return frame


class VideoGrabber:
    """Threaded video capture unit with automatic buffer flushing and reconnection."""

    def __init__(
        self,
        source_type: str = "synthetic",
        uri: str = "test-pattern",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        gstreamer_pipeline: Optional[str] = None,
    ):
        self.source_type = source_type.lower()
        self.uri = uri
        self.width = width
        self.height = height
        self.fps = fps
        self.gstreamer_pipeline = gstreamer_pipeline

        self._cap: Optional[cv2.VideoCapture] = None
        self._synthetic_gen: Optional[SyntheticStreamGenerator] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_source()

    def _init_source(self) -> None:
        if self.source_type == "synthetic":
            self._synthetic_gen = SyntheticStreamGenerator(self.width, self.height, self.fps)
            return

        # Build pipeline string if hardware GStreamer is requested
        if self.source_type == "gstreamer":
            pipeline = self.gstreamer_pipeline or build_gstreamer_nvdec_pipeline(self.uri, self.width, self.height)
            logger.info("Opening GStreamer pipeline: %s", pipeline)
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # OpenCV V4L2 or standard RTSP/File
            target = int(self.uri) if self.uri.isdigit() else self.uri
            logger.info("Opening VideoCapture on source: %s", target)
            self._cap = cv2.VideoCapture(target)

        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> "VideoGrabber":
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(1, self.fps)

        while self._running:
            start_time = time.perf_counter()

            if self.source_type == "synthetic" and self._synthetic_gen is not None:
                frame = self._synthetic_gen.generate()
                ret = True
            elif self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Stream read failure, attempting reconnection...")
                    time.sleep(1.0)
                    self._init_source()
                    continue
            else:
                time.sleep(0.1)
                continue

            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame

            # Regulate frame rate for synthetic mode to match target FPS
            if self.source_type == "synthetic":
                elapsed = time.perf_counter() - start_time
                sleep_sec = frame_interval - elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve the most recent frame from the buffer."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("VideoGrabber released.")
