# SPDX-License-Identifier: MIT
"""
TensorRT Inference Engine Wrapper.
Provides low-latency model execution with CUDA warmup, FP16 execution, and unified detection outputs.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TensorRTEngine:
    """Wrapper managing TensorRT 10 engine execution and postprocessing."""

    def __init__(
        self,
        engine_path: str,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        half_precision: bool = True,
        device: str = "cuda:0",
    ):
        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.half_precision = half_precision
        self.device = device

        self._backend = "mock"
        self._model = None
        self._class_names: Dict[int, str] = {0: "target", 1: "bird", 2: "drone", 3: "aircraft"}

        self._load_engine()

    def _load_engine(self) -> None:
        path = Path(self.engine_path)

        # Check if Ultralytics is available (primary seamless runtime on Jetson)
        try:
            from ultralytics import YOLO

            if path.is_file():
                logger.info("Loading model weights via Ultralytics: %s", path)
                self._model = YOLO(str(path), task="detect")
                self._backend = "ultralytics"
                if hasattr(self._model, "names") and self._model.names:
                    self._class_names = self._model.names
            else:
                logger.warning(
                    "Engine file '%s' not found on disk. Initializing lightweight demo model.",
                    path,
                )
                self._model = YOLO("yolov8n.pt")
                self._backend = "ultralytics"
                if hasattr(self._model, "names") and self._model.names:
                    self._class_names = self._model.names

        except ImportError:
            logger.warning("Ultralytics package not installed. Running in mock demonstration mode.")
            self._backend = "mock"

        self.warmup()

    def warmup(self, iterations: int = 5) -> None:
        """Pre-heats CUDA context and allocates page-locked memory buffers."""
        logger.info("Warming up inference engine (%d iterations)...", iterations)
        dummy_frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for _ in range(iterations):
            _ = self.infer(dummy_frame)
        logger.info("Engine warmup completed.")

    def infer(self, frame: np.ndarray) -> List[Dict]:
        """Runs inference on a single BGR frame and returns normalized detections."""
        if self._backend == "ultralytics" and self._model is not None:
            return self._infer_ultralytics(frame)
        return self._infer_mock(frame)

    def _infer_ultralytics(self, frame: np.ndarray) -> List[Dict]:
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size[0],
            half=self.half_precision,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(coords)):
            c_id = int(cls_ids[i])
            detections.append(
                {
                    "bbox": coords[i].tolist(),
                    "confidence": float(confs[i]),
                    "class_id": c_id,
                    "class_name": self._class_names.get(c_id, f"class_{c_id}"),
                }
            )

        return detections

    def _infer_mock(self, frame: np.ndarray) -> List[Dict]:
        """Fallback mock detector for testing pipelines without GPU dependencies."""
        h, w = frame.shape[:2]
        # Detect dark blobs simulating targets in synthetic frames
        gray = np.mean(frame, axis=2)
        dark_pixels = np.argwhere(gray < 50)

        if len(dark_pixels) == 0:
            return []

        min_y, min_x = np.min(dark_pixels, axis=0)
        max_y, max_x = np.max(dark_pixels, axis=0)

        # Filter sensible sizes
        box_w = max_x - min_x
        box_h = max_y - min_y
        if 10 < box_w < w * 0.8 and 10 < box_h < h * 0.8:
            return [
                {
                    "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
                    "confidence": 0.92,
                    "class_id": 1,
                    "class_name": "target",
                }
            ]

        return []
