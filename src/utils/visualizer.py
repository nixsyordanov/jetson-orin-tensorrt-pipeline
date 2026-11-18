# SPDX-License-Identifier: MIT
"""
Tactical On-Screen Display (OSD) & HUD Visualizer.
Renders real-time telemetry HUD, bounding boxes with confidence, and motion tracking trails.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TacticalVisualizer:
    """Renders professional overlays, bounding boxes, and HUD telemetry on video frames."""

    # Curated high-contrast color palette (BGR)
    PALETTE = [
        (0, 255, 128),   # Spring green
        (0, 200, 255),   # Bright amber / gold
        (255, 128, 0),   # Electric blue
        (255, 64, 255),  # Magenta
        (0, 165, 255),   # Vibrant orange
        (128, 255, 0),   # Chartreuse
        (255, 215, 0),   # Cyan
    ]

    def __init__(
        self,
        show_hud: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_trails: bool = True,
        trail_length: int = 30,
        box_thickness: int = 2,
        font_scale: float = 0.55,
    ):
        self.show_hud = show_hud
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.box_thickness = box_thickness
        self.font_scale = font_scale

        # Motion trail histories keyed by track_id: deque of (x, y) centroids
        self._trails: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.trail_length)
        )

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        fps: float = 0.0,
        latencies: Optional[Dict[str, float]] = None,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Render all overlays onto the frame in-place or on a shallow view."""
        h, w = frame.shape[:2]

        # 1. Draw motion trails if tracking IDs are present
        if self.show_trails:
            self._render_trails(frame, detections)

        # 2. Draw bounding boxes and tags
        for det in detections:
            bbox = det.get("bbox")  # [x1, y1, x2, y2]
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_id = int(det.get("class_id", 0))
            cls_name = det.get("class_name", f"id_{cls_id}")
            conf = float(det.get("confidence", 0.0))
            track_id = det.get("track_id")

            color = self.PALETTE[cls_id % len(self.PALETTE)]

            # Draw tactical corner bracket box
            self._draw_corner_box(frame, x1, y1, x2, y2, color, thickness=self.box_thickness)

            # Draw label pill
            if self.show_labels:
                tag = f"{cls_name}"
                if track_id is not None:
                    tag = f"#{track_id} {tag}"
                if self.show_conf:
                    tag += f" {conf * 100:.0f}%"

                self._draw_label_pill(frame, tag, x1, y1, color)

        # 3. Draw top Telemetry HUD Bar
        if self.show_hud:
            self._render_hud_bar(frame, w, fps, latencies, telemetry, len(detections))

        return frame

    def _draw_corner_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """Draws a modern corner-accented bounding box."""
        # Thin base rectangle with low alpha border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness - 1), lineType=cv2.LINE_AA)

        # Prominent corner brackets
        line_len = max(10, min(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2), 24))

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, thickness + 1, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, thickness + 1, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, thickness + 1, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, thickness + 1, cv2.LINE_AA)

    def _draw_label_pill(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, self.font_scale, 1)

        # Position above box, or flip inside if near top boundary
        pad = 4
        y_top = y - th - (pad * 2) if y - th - (pad * 2) > 0 else y + 2
        y_bot = y_top + th + (pad * 2)
        x_right = x + tw + (pad * 2)

        # Solid background pill
        cv2.rectangle(frame, (x, y_top), (x_right, y_bot), color, -1)
        # High-contrast black text on bright pill
        cv2.putText(
            frame,
            text,
            (x + pad, y_bot - pad - baseline // 2),
            font,
            self.font_scale,
            (10, 10, 10),
            1,
            lineType=cv2.LINE_AA,
        )

    def _render_trails(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Draw fading motion trajectories."""
        active_ids = set()
        for det in detections:
            track_id = det.get("track_id")
            bbox = det.get("bbox")
            if track_id is not None and bbox is not None:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                self._trails[track_id].append((cx, cy))
                active_ids.add(track_id)

                history = list(self._trails[track_id])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, history[i - 1], history[i], (0, 220, 255), thickness, cv2.LINE_AA)

        # Prune stale trail IDs
        stale = [k for k in self._trails if k not in active_ids]
        if len(stale) > 100:
            for k in stale[:50]:
                del self._trails[k]

    def _render_hud_bar(
        self,
        frame: np.ndarray,
        width: int,
        fps: float,
        latencies: Optional[Dict[str, float]],
        telemetry: Optional[Dict[str, float]],
        target_count: int,
    ) -> None:
        """Top telemetry bar with semi-transparent background."""
        bar_h = 36
        overlay = frame[:bar_h, :width].copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_h), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.75, frame[:bar_h, :width], 0.25, 0, frame[:bar_h, :width])

        # Bottom accent line
        cv2.line(frame, (0, bar_h - 1), (width, bar_h - 1), (0, 200, 255), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_sz = 0.45
        y_text = 23

        # Section 1: Engine & FPS
        fps_str = f"JETSON ORIN | FPS: {fps:5.1f}"
        cv2.putText(frame, fps_str, (16, y_text), font, font_sz, (0, 255, 128), 1, cv2.LINE_AA)

        # Section 2: Latencies
        offset_x = 220
        if latencies:
            inf_ms = latencies.get("infer_ms", 0.0)
            tot_ms = latencies.get("total_ms", 0.0)
            lat_str = f"LATENCY: {tot_ms:.1f}ms (INF: {inf_ms:.1f}ms)"
            cv2.putText(frame, lat_str, (offset_x, y_text), font, font_sz, (220, 220, 220), 1, cv2.LINE_AA)
            offset_x += 240

        # Section 3: Telemetry (GPU load, temp, power)
        if telemetry:
            gpu_pct = telemetry.get("gpu_util_pct", 0.0)
            pwr_w = telemetry.get("power_w", 0.0)
            temp_c = telemetry.get("gpu_temp_c", 0.0)
            telem_str = f"GPU: {gpu_pct:.0f}% | {temp_c:.0f}C | {pwr_w:.1f}W"
            cv2.putText(frame, telem_str, (offset_x, y_text), font, font_sz, (0, 200, 255), 1, cv2.LINE_AA)

        # Section 4: Target count (right aligned)
        targets_str = f"TARGETS: {target_count}"
        cv2.putText(frame, targets_str, (width - 130, y_text), font, font_sz, (255, 255, 255), 1, cv2.LINE_AA)
