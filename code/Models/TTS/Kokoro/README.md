# Text-to-Speech (TTS)

This directory contains the setup for **Kokoro**, a high-quality, lightweight (82M param) TTS model that runs in real-time.

## 1. GPU Setup (Recommended)

Requires NVIDIA GPU.

### Build

```bash
docker build -f kokoro-gpu.dockerfile -t tts-gpu .
```

### Run

```bash
docker run --gpus all -p 8880:8880 tts-gpu
```

## 2. CPU Setup

Works on standard CPUs using ONNX/CPU-optimized runtime.

### Build

```bash
docker build -f kokoro-cpu.dockerfile -t tts-cpu .
```

### Run

```bash
docker run -p 8880:8880 tts-cpu
```
