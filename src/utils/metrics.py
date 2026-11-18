# SPDX-License-Identifier: MIT
"""
Performance Profiling and Jetson Hardware Telemetry.
Tracks per-stage latency, rolling FPS, and Tegrastats (GPU load, power, thermals).
"""

from __future__ import annotations

import collections
import os
import subprocess
import threading
import time
from typing import Dict, Optional


class RollingFPSMeter:
    """Calculates instantaneous and smoothed rolling FPS over a moving time window."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._timestamps = collections.deque(maxlen=window_size)
        self._last_time = time.perf_counter()

    def tick(self) -> float:
        """Record a frame event and return the current rolling FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class StageProfiler:
    """High-precision latency profiler for individual stages in the vision pipeline."""

    def __init__(self, max_history: int = 120):
        self.max_history = max_history
        self._stages: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_history)
        )
        self._active_starts: Dict[str, float] = {}

    def start(self, stage_name: str) -> None:
        """Start measuring a specific stage."""
        self._active_starts[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        """Stop measuring a stage and record duration in milliseconds."""
        end = time.perf_counter()
        start = self._active_starts.pop(stage_name, None)
        if start is None:
            return 0.0
        duration_ms = (end - start) * 1000.0
        self._stages[stage_name].append(duration_ms)
        return duration_ms

    def get_stage_avg(self, stage_name: str) -> float:
        """Return average duration in ms for a stage."""
        history = self._stages.get(stage_name)
        if not history:
            return 0.0
        return sum(history) / len(history)

    def summary(self) -> Dict[str, float]:
        """Return average ms for all tracked stages and total latency."""
        res = {k: sum(v) / len(v) for k, v in self._stages.items() if len(v) > 0}
        res["total_ms"] = sum(res.values())
        return res


class JetsonTelemetry:
    """Monitors NVIDIA Jetson hardware state (GPU frequency, temperature, power consumption)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and os.path.exists("/usr/bin/tegrastats")
        self.gpu_util: float = 0.0
        self.gpu_temp_c: float = 0.0
        self.power_mw: float = 0.0
        self.ram_used_mb: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_reader()

    def _start_reader(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            proc = subprocess.Popen(
                ["/usr/bin/tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            while self._running and proc.stdout:
                line = proc.stdout.readline()
                if not line:
                    break
                self._parse_line(line)
        except Exception:
            self.enabled = False

    def _parse_line(self, line: str) -> None:
        # tegrastats output sample:
        # RAM 4512/7620MB ... GR3D_FREQ 72%@1020 ... tj@48C ... POM_5V_IN 5420/5200
        try:
            if "GR3D_FREQ" in line:
                part = line.split("GR3D_FREQ")[1].split()[0]
                if "%" in part:
                    self.gpu_util = float(part.split("%")[0])
            if "tj@" in line:
                part = line.split("tj@")[1].split("C")[0]
                self.gpu_temp_c = float(part)
            if "POM_5V_IN" in line:
                part = line.split("POM_5V_IN")[1].split()[0]
                self.power_mw = float(part.split("/")[0])
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False

    def get_snapshot(self) -> Dict[str, float]:
        return {
            "gpu_util_pct": self.gpu_util,
            "gpu_temp_c": self.gpu_temp_c,
            "power_w": round(self.power_mw / 1000.0, 2),
        }
