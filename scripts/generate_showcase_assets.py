# SPDX-License-Identifier: MIT
"""
Generates high-resolution LinkedIn showcase graphics and benchmark infographics.
Outputs:
  - assets/linkedin_benchmark_infographic.png (1600x900 comparison card)
  - assets/linkedin_tactical_hud_preview.png (1920x1080 live pipeline HUD)
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_benchmark_infographic():
    """Renders a sleek, modern 16:9 benchmark comparison card using Matplotlib."""
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#0B0F14")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0B0F14")
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 900)
    ax.axis("off")

    # Header Accent Line
    ax.plot([80, 1520], [840, 840], color="#76B900", linewidth=3)

    # Title & Subtitle
    ax.text(80, 855, "NVIDIA JETSON ORIN NANO  |  PERFORMANCE OPTIMIZATION",
            fontsize=24, fontweight="bold", color="#FFFFFF", family="sans-serif")
    ax.text(80, 805, "Production Edge Vision Pipeline: GStreamer NVDEC + TensorRT 10.3 FP16 Autotuning",
            fontsize=13, color="#8899A6", family="sans-serif")

    # Badges (Top Right)
    badge_x = 1180
    badges = [("JetPack 6.x", "#76B900"), ("TensorRT 10", "#00D2FF"), ("Zero-Lag RTSP", "#FF9900")]
    for text, color in badges:
        rect = patches.FancyBboxPatch((badge_x, 805), 105, 30, boxstyle="round,pad=3",
                                      facecolor="#151C24", edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(badge_x + 52, 820, text, color=color, fontsize=10, fontweight="bold",
                ha="center", va="center", family="sans-serif")
        badge_x += 115

    # ================= LEFT CARD: BEFORE =================
    left_card = patches.FancyBboxPatch((80, 390), 690, 370, boxstyle="round,pad=10",
                                       facecolor="#12171F", edgecolor="#263238", linewidth=1.5)
    ax.add_patch(left_card)
    ax.text(110, 725, "DEFAULT PROFILE  (Off-Target Export)",
            fontsize=16, fontweight="bold", color="#FF5252", family="sans-serif")
    ax.text(110, 700, "15W Power Mode | Cloud-compiled ONNX | Standard OpenCV Capture",
            fontsize=11, color="#78909C", family="sans-serif")

    before_metrics = [
        ("GPU Frequency", "624 MHz", "#B0BEC5", "Dynamic scaling throttled"),
        ("TRT Compute Latency", "53.6 ms", "#FF5252", "High kernel execution time"),
        ("Inference Throughput", "18.7 FPS", "#FF7043", "Below camera frame rate"),
        ("RTSP Frame Lag", "~2,400 ms", "#FF5252", "Severe buffer backlog"),
    ]
    y_pos = 640
    for title, val, color, note in before_metrics:
        ax.text(110, y_pos, title, fontsize=12, color="#90A4AE", family="sans-serif")
        ax.text(500, y_pos, val, fontsize=15, fontweight="bold", color=color, ha="right", family="sans-serif")
        ax.text(110, y_pos - 18, note, fontsize=9.5, color="#546E7A", family="sans-serif")
        ax.plot([110, 740], [y_pos - 25, y_pos - 25], color="#1E2733", linewidth=1)
        y_pos -= 65

    # ================= RIGHT CARD: AFTER =================
    right_card = patches.FancyBboxPatch((830, 390), 690, 370, boxstyle="round,pad=10",
                                        facecolor="#12171F", edgecolor="#76B900", linewidth=2.0)
    ax.add_patch(right_card)
    ax.text(860, 725, "OPTIMIZED PIPELINE  (MAXN + Native TRT 10)",
            fontsize=16, fontweight="bold", color="#76B900", family="sans-serif")
    ax.text(860, 700, "MAXN Clocks Locked | Native Ampere SM87 Compilation | NVDEC Zero-Drop",
            fontsize=11, color="#81C784", family="sans-serif")

    after_metrics = [
        ("GPU Frequency", "1020 MHz", "#76B900", "+63.4% locked compute headroom"),
        ("TRT Compute Latency", "15.98 ms", "#00E676", "3.35x faster kernel execution"),
        ("Inference Throughput", "62.2 FPS", "#00E676", "+232% throughput boost"),
        ("RTSP Frame Lag", "< 60 ms", "#00E676", "True glass-to-glass real-time"),
    ]
    y_pos = 640
    for title, val, color, note in after_metrics:
        ax.text(860, y_pos, title, fontsize=12, color="#CFD8DC", family="sans-serif")
        ax.text(1490, y_pos, val, fontsize=15, fontweight="bold", color=color, ha="right", family="sans-serif")
        ax.text(860, y_pos - 18, note, fontsize=9.5, color="#76B900", family="sans-serif")
        ax.plot([860, 1490], [y_pos - 25, y_pos - 25], color="#1E2733", linewidth=1)
        y_pos -= 65

    # ================= BOTTOM COMPARISON SECTION =================
    bot_card = patches.FancyBboxPatch((80, 80), 1440, 270, boxstyle="round,pad=10",
                                      facecolor="#10151C", edgecolor="#1E2733", linewidth=1.5)
    ax.add_patch(bot_card)

    ax.text(110, 310, "COMPARATIVE BENCHMARK OVERVIEW",
            fontsize=14, fontweight="bold", color="#FFFFFF", family="sans-serif")

    # Bar 1: Throughput (FPS)
    ax.text(110, 265, "THROUGHPUT (FPS) - HIGHER IS BETTER", fontsize=11, fontweight="bold", color="#B0BEC5")
    # Background tracks
    ax.barh(235, 1200, height=22, left=110, color="#1A222D", edgecolor="none")
    # Values
    b1_w = (18.7 / 70.0) * 1200
    b2_w = (62.2 / 70.0) * 1200
    ax.barh(244, b1_w, height=10, left=110, color="#FF5252", label="Before: 18.7 FPS")
    ax.barh(228, b2_w, height=14, left=110, color="#76B900", label="After: 62.2 FPS (+232%)")
    ax.text(110 + b1_w + 15, 244, "18.7 FPS (Baseline)", fontsize=10, color="#FF8A80", va="center", fontweight="bold")
    ax.text(110 + b2_w + 15, 228, "62.2 FPS (+232% Lift)", fontsize=11, color="#76B900", va="center", fontweight="bold")

    # Bar 2: Latency (ms)
    ax.text(110, 185, "COMPUTE LATENCY (MS) - LOWER IS BETTER", fontsize=11, fontweight="bold", color="#B0BEC5")
    ax.barh(155, 1200, height=22, left=110, color="#1A222D", edgecolor="none")
    l1_w = (53.6 / 60.0) * 1200
    l2_w = (15.98 / 60.0) * 1200
    ax.barh(164, l1_w, height=10, left=110, color="#FF5252")
    ax.barh(148, l2_w, height=14, left=110, color="#00D2FF")
    ax.text(110 + l1_w + 15, 164, "53.6 ms", fontsize=10, color="#FF8A80", va="center", fontweight="bold")
    ax.text(110 + l2_w + 15, 148, "15.98 ms (-70% Latency)", fontsize=11, color="#00D2FF", va="center", fontweight="bold")

    # Bottom Footer note
    ax.text(110, 105, "Hardware: NVIDIA Jetson Orin Nano Developer Kit (8GB)  |  OS: JetPack 6.0 (L4T 36.3)  |  Precision: FP16  |  Resolution: 640x640 / 1080p",
            fontsize=10.5, color="#607D8B", family="sans-serif")

    out_file = OUTPUT_DIR / "linkedin_benchmark_infographic.png"
    plt.savefig(out_file, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Generated infographic: {out_file}")


def generate_tactical_hud_preview():
    """Generates a dramatic 1920x1080 tactical camera view with real-time OSD and tracking trails."""
    w, h = 1920, 1080
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Realistic twilight atmosphere gradient
    for y in range(h):
        ratio = y / h
        b = int(70 + ratio * 20)
        g = int(60 + ratio * 15)
        r = int(50 + ratio * 10)
        frame[y, :] = (b, g, r)

    # Distant mountain silhouettes
    pts1 = np.array([[0, 780], [300, 680], [600, 720], [900, 650], [1300, 710], [1650, 640], [1920, 730], [1920, 1080], [0, 1080]], np.int32)
    cv2.fillPoly(frame, [pts1], (38, 34, 30))

    pts2 = np.array([[0, 840], [400, 790], [800, 820], [1200, 770], [1700, 810], [1920, 790], [1920, 1080], [0, 1080]], np.int32)
    cv2.fillPoly(frame, [pts2], (24, 20, 18))

    # Tactical grid overlay (subtle cross-grid)
    grid_color = (60, 52, 45)
    for x in range(0, w, 160):
        cv2.line(frame, (x, 50), (x, h - 50), grid_color, 1, cv2.LINE_AA)
    for y in range(80, h - 50, 120):
        cv2.line(frame, (50, y), (w - 50, y), grid_color, 1, cv2.LINE_AA)

    # Center tactical reticle
    cx, cy = w // 2, h // 2
    cv2.circle(frame, (cx, cy), 45, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.line(frame, (cx - 70, cy), (cx - 15, cy), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 15, cy), (cx + 70, cy), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 70), (cx, cy - 15), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 15), (cx, cy + 70), (0, 200, 255), 1, cv2.LINE_AA)

    # ---------------- Target 1: Flying Bird with motion trail ----------------
    trail1 = [(480, 460), (520, 440), (570, 415), (630, 390), (700, 360), (780, 330), (870, 305)]
    for i in range(1, len(trail1)):
        alpha = i / len(trail1)
        th = max(1, int(alpha * 4))
        cv2.line(frame, trail1[i - 1], trail1[i], (0, 220, 255), th, cv2.LINE_AA)

    t1_x1, t1_y1, t1_x2, t1_y2 = 820, 255, 930, 345
    # Target shape
    cv2.circle(frame, (875, 300), 14, (15, 15, 15), -1)
    cv2.line(frame, (850, 296), (900, 296), (25, 25, 25), 4)

    # Tactical corner brackets (Emerald)
    col1 = (0, 255, 128)
    corner_len = 22
    # TL
    cv2.line(frame, (t1_x1, t1_y1), (t1_x1 + corner_len, t1_y1), col1, 3, cv2.LINE_AA)
    cv2.line(frame, (t1_x1, t1_y1), (t1_x1, t1_y1 + corner_len), col1, 3, cv2.LINE_AA)
    # TR
    cv2.line(frame, (t1_x2, t1_y1), (t1_x2 - corner_len, t1_y1), col1, 3, cv2.LINE_AA)
    cv2.line(frame, (t1_x2, t1_y1), (t1_x2, t1_y1 + corner_len), col1, 3, cv2.LINE_AA)
    # BL
    cv2.line(frame, (t1_x1, t1_y2), (t1_x1 + corner_len, t1_y2), col1, 3, cv2.LINE_AA)
    cv2.line(frame, (t1_x1, t1_y2), (t1_x1, t1_y2 - corner_len), col1, 3, cv2.LINE_AA)
    # BR
    cv2.line(frame, (t1_x2, t1_y2), (t1_x2 - corner_len, t1_y2), col1, 3, cv2.LINE_AA)
    cv2.line(frame, (t1_x2, t1_y2), (t1_x2, t1_y2 - corner_len), col1, 3, cv2.LINE_AA)
    # Thin frame
    cv2.rectangle(frame, (t1_x1, t1_y1), (t1_x2, t1_y2), col1, 1, cv2.LINE_AA)
    # Label pill
    cv2.rectangle(frame, (t1_x1, t1_y1 - 26), (t1_x1 + 140, t1_y1), col1, -1)
    cv2.putText(frame, "#104 BIRD 95%", (t1_x1 + 6, t1_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 15, 20), 1, cv2.LINE_AA)

    # ---------------- Target 2: Distant UAV / Raptor ----------------
    trail2 = [(1450, 560), (1400, 520), (1340, 485), (1270, 455), (1190, 430)]
    for i in range(1, len(trail2)):
        alpha = i / len(trail2)
        th = max(1, int(alpha * 3))
        cv2.line(frame, trail2[i - 1], trail2[i], (0, 200, 255), th, cv2.LINE_AA)

    t2_x1, t2_y1, t2_x2, t2_y2 = 1140, 395, 1240, 470
    col2 = (0, 200, 255)
    cv2.circle(frame, (1190, 430), 10, (15, 15, 15), -1)
    # Corner brackets
    cv2.line(frame, (t2_x1, t2_y1), (t2_x1 + 18, t2_y1), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x1, t2_y1), (t2_x1, t2_y1 + 18), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x2, t2_y1), (t2_x2 - 18, t2_y1), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x2, t2_y1), (t2_x2, t2_y1 + 18), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x1, t2_y2), (t2_x1 + 18, t2_y2), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x1, t2_y2), (t2_x1, t2_y2 - 18), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x2, t2_y2), (t2_x2 - 18, t2_y2), col2, 3, cv2.LINE_AA)
    cv2.line(frame, (t2_x2, t2_y2), (t2_x2, t2_y2 - 18), col2, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (t2_x1, t2_y1), (t2_x2, t2_y2), col2, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (t2_x1, t2_y1 - 26), (t2_x1 + 145, t2_y1), col2, -1)
    cv2.putText(frame, "#105 RAPTOR 91%", (t2_x1 + 6, t2_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 15, 20), 1, cv2.LINE_AA)

    # ---------------- TOP TELEMETRY HUD BAR ----------------
    hud_h = 44
    overlay = frame[:hud_h, :w].copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 15, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame[:hud_h, :w], 0.12, 0, frame[:hud_h, :w])
    cv2.line(frame, (0, hud_h - 1), (w, hud_h - 1), (0, 200, 255), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_txt = 28
    # Section 1
    cv2.putText(frame, "JETSON ORIN NANO | FPS:  62.2", (24, y_txt), font, 0.6, (0, 255, 128), 1, cv2.LINE_AA)
    # Section 2
    cv2.putText(frame, "LATENCY: 17.5ms (INF: 15.9ms)", (380, y_txt), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    # Section 3
    cv2.putText(frame, "GPU: 96% @ 1020MHz | 49C | 9.8W", (740, y_txt), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    # Section 4
    cv2.putText(frame, "MODE: MAXN (LOCKED) | TRT 10 FP16", (1220, y_txt), font, 0.52, (180, 200, 210), 1, cv2.LINE_AA)
    # Section 5
    cv2.putText(frame, "TARGETS: 2", (w - 150, y_txt), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------------- BOTTOM STATUS BAR ----------------
    bot_h = 36
    bot_y = h - bot_h
    overlay_bot = frame[bot_y:, :w].copy()
    cv2.rectangle(overlay_bot, (0, 0), (w, bot_h), (12, 15, 20), -1)
    cv2.addWeighted(overlay_bot, 0.85, frame[bot_y:, :w], 0.15, 0, frame[bot_y:, :w])
    cv2.line(frame, (0, bot_y), (w, bot_y), (60, 70, 80), 1, cv2.LINE_AA)

    cv2.putText(frame, "INGESTION: GStreamer NVDEC (nvv4l2decoder) 1080p@30fps  |  PIPELINE: Zero-Buffer Queue  |  EGRESS: SRT (zerolatency)",
                (24, h - 12), font, 0.45, (160, 175, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "GLASS-TO-GLASS: < 60ms", (w - 240, h - 12), font, 0.45, (0, 255, 128), 1, cv2.LINE_AA)

    out_file = OUTPUT_DIR / "linkedin_tactical_hud_preview.png"
    cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"Generated tactical HUD preview: {out_file}")


if __name__ == "__main__":
    generate_benchmark_infographic()
    generate_tactical_hud_preview()
