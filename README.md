<div align="center">

# ⚡ Jetson Orin TensorRT Vision Pipeline

**Ultra-low latency, high-throughput edge AI vision pipeline optimized for NVIDIA Jetson Orin Nano, Orin NX, and AGX Orin with TensorRT 10 and hardware-accelerated GStreamer.**

[![NVIDIA JetPack](https://img.shields.io/badge/NVIDIA%20JetPack-6.0%20%7C%206.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetpack)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.x-green?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-Real--Time%20Inference-orange)](https://github.com)

</div>

---

## 📌 Overview

Deploying modern deep learning object detectors on resource-constrained edge platforms often suffers from memory bottlenecks, video buffer lag, and unoptimized engine compilation. 

This repository provides a production-grade, asynchronous computer vision framework engineered specifically for **NVIDIA Jetson Orin** architectures. It achieves real-time inference (up to **62.2 FPS** on Jetson Orin Nano) by leveraging:
- **Hardware-accelerated video decoding (NVDEC)** via GStreamer (`nvv4l2decoder` + `nvvidconv`).
- **TensorRT 10 FP16 engine execution** with asynchronous CUDA stream scheduling.
- **Threaded non-blocking capture and buffer flushing** to eliminate RTSP frame accumulation lag.
- **Tactical On-Screen Display (OSD)** rendering real-time hardware telemetry (GPU frequency, power consumption, thermals via Tegrastats) and motion history trails.
- **Ultra-low latency egress streaming** pushing annotated streams via **SRT / UDP / RTSP** with zero-latency H.264 encoding.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Ingestion [" 🎥 Video Ingestion "]
        RTSP["RTSP Camera / V4L2"] --> GStreamer["GStreamer NVDEC<br/>(nvv4l2decoder)"]
        SYNTH["Synthetic Pattern<br/>Generator"] --> Worker["Threaded Buffer<br/>Flushing Worker"]
        GStreamer --> Worker
    end

    subgraph Compute [" ⚡ TensorRT Acceleration "]
        Worker --> Pre["Letterbox & CUDA Host-to-Device"]
        Pre --> TRT["TensorRT 10 Engine<br/>(FP16 Precision)"]
        TRT --> Post["NMS & Confidence Filtering"]
    end

    subgraph Visualization [" 🎯 Telemetry & OSD "]
        Post --> OSD["Tactical Visualizer<br/>(Corners & Motion Trails)"]
        TELEM["Tegrastats Monitor<br/>(GPU % / Temp / Watts)"] --> OSD
    end

    subgraph Egress [" 📡 Low-Latency Output "]
        OSD --> EgressQueue["Zero-Drop Frame Queue"]
        EgressQueue --> FFmpeg["FFmpeg Pipe<br/>(zerolatency libx264)"]
        FFmpeg --> SRT["SRT / UDP / RTSP Stream"]
        EgressQueue --> GUI["Local Window Preview"]
    end
```

---

## 📊 Real-World Benchmark Results

Measured directly on an **NVIDIA Jetson Orin Nano (8GB, Ampere SM87 architecture)** running JetPack 6.x and TensorRT 10:

| Metric | Out-of-the-Box (15W Default, Host Compiled) | Optimized Pipeline (MAXN Mode + Native TRT10 FP16) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **GPU Clock Frequency** | 624 MHz | **1020 MHz** (Locked via `jetson_clocks`) | **+63.4% clock headroom** |
| **TRT Kernel Latency** | 53.60 ms | **15.98 ms** | **3.35x faster compute** |
| **Engine Throughput (trtexec)** | 18.7 FPS | **62.2 FPS** | **+232% throughput** |
| **Python Inference (1024x1024)** | 18.2 FPS (54.9 ms) | **34.1 FPS (29.35 ms)** | **+87% FPS gain** |
| **End-to-End Pipeline Latency** | ~68 ms | **~38 ms** | **-44% latency reduction** |
| **RTSP Buffer Backlog** | 1500–3000 ms lag | **< 60 ms glass-to-glass** | **Instantaneous real-time** |

> **Key takeaway:** Cross-compiling TensorRT engines on external x86 servers often misses board-specific SM87 kernel autotuning. Compiling directly on the Orin Nano with local timing caches and locking GPU clocks via MAXN mode unlocks the full 62+ FPS potential of the silicon.

---

## 🚀 Quickstart

### 1. Zero-Dependency Synthetic Demo
Test the pipeline immediately on any workstation or Jetson board without cameras or external models:

```bash
# Clone the repository
git clone https://github.com/your-username/jetson-orin-tensorrt-pipeline.git
cd jetson-orin-tensorrt-pipeline

# Install minimal requirements
pip install -r requirements.txt

# Run turnkey synthetic benchmark
python demo.py
```

### 2. Live Camera Stream with Local Display
To ingest a live RTSP stream and preview annotated detections in a local window:

```bash
python demo.py \
    --source "rtsp://username:password@192.168.1.100:554/stream1" \
    --engine "models/yolov8s.engine" \
    --display
```

### 3. Run Benchmark Suite
Run an automated latency percentile evaluation (P50, P95, P99) and hardware telemetry audit:

```bash
python scripts/benchmark_pipeline.py --engine models/yolov8s.engine --frames 500
```

---

## ⚙️ Compiling Models to TensorRT 10

To convert your PyTorch (`.pt`) or ONNX models into optimized FP16 TensorRT engines:

```bash
# Compile with 4GB workspace and FP16 half precision
python scripts/export_engine.py \
    --weights models/yolov8s.pt \
    --imgsz 640 \
    --workspace 4
```

Alternatively, use the native NVIDIA `trtexec` builder:

```bash
trtexec \
    --onnx=models/yolov8s.onnx \
    --saveEngine=models/yolov8s.engine \
    --fp16 \
    --timingCacheFile=timing.cache
```

---

## 🔧 Jetson Performance Tuning

To achieve maximum performance and ensure clocks remain locked across system reboots, execute the automated clock-locking script:

```bash
sudo chmod +x scripts/setup_jetson_clocks.sh
sudo ./scripts/setup_jetson_clocks.sh
```

**What this script configures:**
1. Sets power mode to **MAXN** (`nvpmodel -m 0`), enabling maximum GPU and CPU clock frequencies.
2. Runs `jetson_clocks` to disable dynamic thermal throttling down-scaling.
3. Installs a persistent `systemd` unit (`jetson-clocks-lock.service`) that re-applies clock locks on every boot.
4. Tunes Linux network socket receive buffers (`net.core.rmem_max = 26214400`) to eliminate packet loss on multi-camera RTSP ingestion.

---

## 🐳 Docker Deployment

A standalone production Dockerfile is included based on NVIDIA's official Jetson L4T TensorRT container.

```bash
# Build the Docker image
docker build -t orin-vision-pipeline -f docker/Dockerfile.jetson .

# Run with GPU runtime access
docker run --runtime nvidia --rm -it \
    --network host \
    orin-vision-pipeline
```

---

## 📁 Repository Structure

```
jetson-orin-tensorrt-pipeline/
├── config.example.yaml        # Comprehensive YAML pipeline configuration
├── requirements.txt           # Minimal dependencies (JetPack 6 & CUDA compatible)
├── demo.py                    # Turnkey CLI entry point
├── src/
│   ├── engine.py              # TensorRT 10 engine execution wrapper
│   ├── pipeline.py            # Main pipeline coordinator
│   ├── utils/
│   │   ├── config_parser.py   # YAML config validation and loader
│   │   ├── metrics.py         # Rolling FPS & Tegrastats telemetry monitor
│   │   └── visualizer.py      # Tactical OSD, HUD, and motion trails
│   └── video/
│       ├── grabber.py         # Threaded GStreamer NVDEC / synthetic grabber
│       └── streamer.py        # Asynchronous zero-latency FFmpeg egress (SRT/RTSP)
├── scripts/
│   ├── export_engine.py       # PyTorch/ONNX to TensorRT 10 FP16 compiler
│   ├── benchmark_pipeline.py  # Latency percentile profiling utility
│   └── setup_jetson_clocks.sh # MAXN mode and persistent clock lock script
└── docker/
    └── Dockerfile.jetson      # Production NVIDIA L4T container configuration
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
