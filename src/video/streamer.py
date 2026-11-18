# SPDX-License-Identifier: MIT
"""
Asynchronous Zero-Latency Video Egress / Streaming Engine.
Pipes annotated frames to FFmpeg for ultra-low latency SRT/UDP/RTSP streaming or local display.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Non-blocking frame egress supporting FFmpeg hardware encoding and local preview."""

    def __init__(
        self,
        output_type: str = "display",
        target_uri: str = "srt://127.0.0.1:9000?mode=caller",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        bitrate_kbps: int = 4000,
        preset: str = "ultrafast",
        tune: str = "zerolatency",
    ):
        self.output_type = output_type.lower()
        self.target_uri = target_uri
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.preset = preset
        self.tune = tune

        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._writer: Optional[cv2.VideoWriter] = None

    def start(self) -> "VideoStreamer":
        if self.output_type in ("srt", "udp", "rtsp", "file"):
            self._init_ffmpeg()
        elif self.output_type == "opencv_file":
            self._init_opencv_writer()

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def _init_ffmpeg(self) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            logger.warning("FFmpeg binary not found on PATH. Falling back to local display.")
            self.output_type = "display"
            return

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-b:v", f"{self.bitrate_kbps}k",
            "-maxrate", f"{self.bitrate_kbps}k",
            "-bufsize", f"{self.bitrate_kbps // 2}k",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),
        ]

        if self.output_type == "srt":
            cmd.extend(["-f", "mpegts", self.target_uri])
        elif self.output_type == "udp":
            cmd.extend(["-f", "mpegts", self.target_uri])
        elif self.output_type == "rtsp":
            cmd.extend(["-f", "rtsp", self.target_uri])
        else:
            cmd.append(self.target_uri)

        logger.info("Spawning FFmpeg stream pipeline: %s", " ".join(cmd))
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _init_opencv_writer(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.target_uri, fourcc, self.fps, (self.width, self.height))

    def push(self, frame: np.ndarray) -> None:
        """Push a processed frame to the streaming worker (drops on congestion)."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop older frame to guarantee zero streaming latency
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def _worker(self) -> None:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    logger.error("FFmpeg pipe broken. Terminating egress worker.")
                    break
            elif self._writer:
                self._writer.write(frame)
            elif self.output_type == "display":
                cv2.imshow("Orin Vision Pipeline Preview", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit
                    self._running = False
                    break

            self._queue.task_done()

    def release(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._ffmpeg_proc:
            if self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                except Exception:
                    pass
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None

        if self._writer:
            self._writer.release()
            self._writer = None

        if self.output_type == "display":
            cv2.destroyAllWindows()

        logger.info("VideoStreamer released.")
