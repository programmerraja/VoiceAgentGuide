# LLM Serving

This directory contains configurations to run LLMs locally using either GPU (SGLang) or CPU (Ollama).

## 1. GPU Setup (SGLang)

Best for NVIDIA GPU users. Uses `vLLM` and `SGLang` for high-throughput serving.

### Build

```bash
docker build -f llama-gpu.dockerfile -t llm-gpu .
```

### Run

_Requires `HF_TOKEN` environment variable for accessing Meta Llama models._

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=your_token_here" \
  --ipc=host \
  llm-gpu \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

## 2. CPU Setup (Ollama)

Best for Mac (M-series) or standard CPU inference.

### Build

```bash
docker build -f llama-cpu.dockerfile -t llm-cpu .
```

### Run

```bash
# Start the server
docker run -d -v ollama:/root/.ollama -p 11434:11434 llm-cpu

# Pull and run a model (e.g., Llama 3)
docker exec -it <container_id> ollama run llama3:8b
```
