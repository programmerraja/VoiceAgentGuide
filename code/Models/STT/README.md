# Speech-to-Text (STT)

This directory contains the setup for **Faster-Whisper**, a highly optimized implementation of OpenAI's Whisper model.

## 1. GPU Setup (Recommended)

Fastest inference using NVIDIA GPUs.

### Build

```bash
docker build -f docker-gpu.dockerfile -t stt-gpu .
```

### Run

```bash
docker run --gpus all -p 8000:8000 stt-gpu
```

## 2. CPU Setup

Quantized inference for standard CPUs.

### Build

```bash
docker build -f docker-cpu.dockerfile -t stt-cpu .
```

### Run

```bash
docker run -p 8000:8000 stt-cpu
```
