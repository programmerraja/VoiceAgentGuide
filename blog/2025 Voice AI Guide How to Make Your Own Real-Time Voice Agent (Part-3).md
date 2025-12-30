---
title: "2025 Voice AI Guide: How to Make Your Own Real-Time Voice Agent (Part-3)"
date: 2025-12-21T10:36:47.4747+05:30
draft: true
tags:
---

Welcome back! The waiting is over. In Part 3, we are going to see how to run the components of our voice agent locally, even on a CPU. Finally, you will have homework where you need to integrate all these into generic code to work it locally.

## The Hardware Reality: CPU vs GPU

Before we run anything, we need to address the elephant in the room: **Computation**.

### Why do models crave GPUs?

AI models are essentially giant math problems involving billions of matrix multiplications.

- **CPUs** are like a Ferrari: insanely fast at doing one or two complex things at a time (Sequential Processing).
- **GPUs** are like a bus service: slower at individual tasks, but can transport thousands of people (numbers) at once (Parallel Processing).

Since neural networks need to calculate billions of numbers simultaneously, GPUs are exponentially faster.

**"But I only have a CPU!"**

Don't worry. We can still run these models using a technique called **Quantization**.

Standard models use 16-bit floating-point numbers (e.g., `3.14159...`). Quantization rounds these down to 4-bit or 8-bit integers (e.g., `3`). This drastically reduces the size of the model and makes the math simple enough for a CPU to handle reasonably well, though it will practically always be slower than a GPU.

## Speech-to-Text (STT)

First, we are going to see how to run the STT component. As mentioned in Part 1, we are using Whisper from OpenAI. If you search online, you will find a lot of varieties. We are going to use `faster-whisper` because it is a reimplementation of OpenAI's Whisper using **CTranslate2**. It is up to 4x faster and uses less memory than the original, making it perfect for real-time agents.

From `faster-whisper` itself, we have used `Systran/faster-distil-whisper-medium.en` from Hugging Face, but feel free to explore others:

| Model name     | Params (approx.) | Type         | Typical use case                                  |
| :------------- | :--------------- | :----------- | :------------------------------------------------ |
| **tiny**       | 39M              | Multilingual | Very fast, rough drafts, low-end CPU.             |
| **tiny.en**    | 39M              | English-only | Fast English-only STT with small footprint.       |
| **base**       | 74M              | Multilingual | Better than tiny, still lightweight.              |
| **base.en**    | 74M              | English-only | Accurate English with low compute.                |
| **small**      | 244M             | Multilingual | Good balance of speed and quality.                |
| **small.en**   | 244M             | English-only | Higher‑quality English on moderate hardware.      |
| **medium**     | 769M             | Multilingual | High accuracy, slower; needs stronger machine.    |
| **medium.en**  | 769M             | English-only | Very accurate English, heavier compute.           |
| **large / v2** | 1.55B            | Multilingual | Best quality older large models, GPU recommended. |
| **large-v3**   | 1.55B            | Multilingual | Latest, improved multilingual accuracy.           |

### How to run STT

To serve this, we need a server or inference engine. While `faster-whisper` has a library, we need a server-like architecture (similar to Deepgram) where we connect to a WebSocket server, send audio, and receive text. I have written a simple WebSocket server that runs the model on either CPU or GPU.

I have dockerized everything to make our life easier :)

All the code for this component is located in `code/Models/STT`. Let's look at what's inside:

- `server.py`: The heart of the STT. It starts a **WebSocket server** that receives audio chunks, runs them through the Whisper model, and streams back text.
- `download_model.py`: A helper script to download the specific `faster-whisper` model weights from HuggingFace.
- `docker-gpu.dockerfile`: The environment setup for NVIDIA GPU users (installs CUDA drivers).
- `docker-cpu.dockerfile`: The environment for CPU users (lighter setup).

### Architecture Flow

1.  **WebSocket Connection**: We use WebSockets instead of REST API because we need a persistent connection to stream audio continuously.
2.  **Audio Chunking**: The client (your browser/mic) records audio and chops it into small "chunks" (bytes).
3.  **Streaming**: These chunks are sent over the WebSocket instantly.
4.  **Processing**: The server receives these raw bytes (usually Int16 format), converts them to floating-point numbers (Float32), and feeds them into the Whisper model.
5.  **Voice Activity Detection (VAD)**: The server listens to your audio stream. When it detects silence (you stopped speaking), it commits the transcription and sends it out.

**Example Scenario**:
Imagine you say **"Hello Agent"**.

1.  Your microphone captures 1 second of audio.
2.  The browser slices this into 20 tiny audio packets and shoots them to the server one by one.
3.  The Server processes them in real-time. It hears "He...", then "Hello...", then "Hello A...".
4.  You stop talking. The VAD logic sees 500ms of silence.
5.  It shouts "STOP!" and sends the final text `"Hello Agent"` to the next step.

### How to Run

**On GPU (Recommended):**

```bash
docker build -f docker-gpu.dockerfile -t stt-gpu .
docker run --gpus all -p 8000:8000 stt-gpu
```

**On CPU:**

```bash
docker build -f docker-cpu.dockerfile -t stt-cpu .
docker run -p 8000:8000 stt-cpu
```

## Large Language Model (LLM)

Next, we need a brain. To run a model locally, we need an **Inference Engine**. If you search Google, you will find many options. Here are a few popular ones:

| Engine           | Primary Use                            | Hardware                       | Quantization Support       | Best For                             |
| :--------------- | :------------------------------------- | :----------------------------- | :------------------------- | :----------------------------------- |
| **Ollama**       | Local single-machine LLM serving       | CPU, GPU (NVIDIA, Apple Metal) | GGUF (Q4, Q5, Q8)          | Local dev, prototypes, low traffic   |
| **llama.cpp**    | CPU-optimized inference                | CPU (x86, ARM), GPU            | GGUF (Q2-Q8, AWQ, IQ2-IQ4) | Resource-constrained, edge devices   |
| **vLLM**         | High-throughput production LLM serving | NVIDIA GPU, AMD, Intel         | INT8, FP8, FP16, AWQ, GPTQ | Production APIs, high concurrency    |
| **TensorRT-LLM** | Maximum NVIDIA performance             | NVIDIA GPU only (CC >= 7.0)    | INT8, FP16, FP8 (H100+)    | Ultra-low latency, NVIDIA-only       |
| **SGLang**       | High-throughput production LLM serving | NVIDIA GPU, AMD, Intel         | FP16, INT8                 | Research, RadixAttention, multi-turn |

From this list, we are going to use **SGLang** to run our model on GPU, and for CPU, we can go with **Ollama**, which is very simple and easy to setup.

We are using **Llama 3.1 8B**, which is the current state-of-the-art for small open-source models.

### Folder Structure

Code location: `code/Models/LLM`

- `llama-gpu.dockerfile`: Setup for vLLM or SGLang (GPU).
- `llama-cpu.dockerfile`: Setup for Ollama (CPU).

### Architecture Flow

The LLM server isn't just a text-in/text-out box. It handles queuing and batching to keep up.

1.  **Request Queue**: Your prompt enters a waiting line.
2.  **Batching**: The server groups your request with others (if any).
3.  **Prefill**: It processes your input text (Prompt) to understand the context.
4.  **Decoding (Token by Token)**: It generates one word-part (token) at a time.
5.  **Streaming**: As soon as a token is generated, it is sent back. It doesn't wait for the full sentence.

**Example Scenario**:
Input: **"What is 2+2?"**

1.  **Tokenizer**: Converts text to numbers `[123, 84, 99]`.
2.  **Inference**: The model calculates the most likely next number.
3.  **Token 1**: Generates `"It"`. Sends it immediately.
4.  **Token 2**: Generates `"is"`. Sends it.
5.  **Token 3**: Generates `"4"`. Sends it.
6.  **End**: Sends `<EOS>` (End of Sequence).

### How to Run

**1. On GPU (using SGLang/vLLM):**

```bash
docker build -f llama-gpu.dockerfile -t llm-gpu .
docker run --gpus all -p 30000:30000 llm-gpu
```

_Note: This exposes an OpenAI-compatible endpoint at port 30000._

**2. On CPU (using Ollama):**

```bash
# Easy method: Just install Ollama from ollama.com
ollama run llama3.1
```

_Or using our dockerfile:_

```bash
docker build -f llama-cpu.dockerfile -t llm-cpu .
docker run -p 11434:11434 llm-cpu
```

## 4. Text-to-Speech (TTS)

Finally, for the **Mouth**, we use **[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)**.

Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

We’ll also use the **Kokoro library** and build a **server to expose it as a service**.

### Folder Structure

Code location: `code/Models/TTS/Kokoro`

- `server.py`: Takes text input and streams out audio bytes.
- `download_model.py`: Fetches the model weights (`v0_19` weights).
- `kokoro-gpu.dockerfile`: GPU setup (Requires NVIDIA container toolkit).
- `kokoro-cpu.dockerfile`: CPU setup (Works on standard laptops).

### Architecture Flow

The TTS server receives a stream of text tokens from the LLM. It immediately starts converting them to Phonemes (sound units) and generating audio. It streams this audio back to the user _before_ the LLM has even finished the sentence. This **Streaming Pipeline** is crucial for low latency.

**Example Scenario**:
Input Stream: **"It"** -> **"is"** -> **"4"**.

1.  **Phonemizer**: Receives "It". Converts to phonetic sounds `/ɪt/`.
2.  **Model**: Generates audio waves for that fraction of a second.
3.  **Output**: You hear "It" from your speakers.
4.  _Meanwhile_, the LLM sends "is". The TTS repeats the process.
5.  **Result**: You hear a continuous sentence even though it's being built piece by piece.

### How to Run

**1. On GPU:**

```bash
docker build -f kokoro-gpu.dockerfile -t tts-gpu .
docker run --gpus all -p 8880:8880 tts-gpu
```

**2. On CPU:**

```bash
docker build -f kokoro-cpu.dockerfile -t tts-cpu .
docker run -p 8880:8880 tts-cpu
```

So now that all three components are up and running, it’s your turn to think through how we can integrate them with **Pipecat** and get a fully local **“Hello World”** working end to end.

If you’d like to share your implementation, feel free to raise a PR on our GitHub repo here.

In next part we will be discussing about how to make a real-time voice agent using these components.
