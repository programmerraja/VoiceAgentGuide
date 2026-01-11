---
title: "2025 Voice AI Guide: How to Make Your Own Real-Time Voice Agent (Part-3)"
date: 2025-12-21T10:36:47.4747+05:30
draft: true
tags:
---

Welcome back! The waiting is over. In Part 3, we are going to see how to run the components of our voice agent locally, even on a CPU. Finally, you will have homework where you need to integrate all these into generic code to work it locally.

## The Performance Reality: Setting Expectations with Latency Budgets

Before we dive into running components, **you need to understand what "fast" actually means in voice AI**. Industry benchmarks show that users perceive natural conversation when end-to-end latency (time from user finishing speaking to hearing the agent's response) is **under 800ms**, with the gold standard being **under 500ms**.

Let's break down where those milliseconds go:

### Latency Budget Breakdown

| Component                          | Target Latency | Upper Limit | Notes                                               |
| ---------------------------------- | -------------- | ----------- | --------------------------------------------------- |
| **Speech-to-Text (STT)**           | 200-350ms      | 500ms       | Measured from silence detection to final transcript |
| **LLM Time-to-First-Token (TTFT)** | 100-200ms      | 400ms       | First token generation (not full response)          |
| **Text-to-Speech TTFB**            | 75-150ms       | 250ms       | Time to first byte of audio                         |
| **Network & Orchestration**        | 50-100ms       | 150ms       | WebSocket hops, service-to-service handoff          |
| **Total Mouth-to-Ear Gap**         | 500-800ms      | 1100ms      | Complete turn latency                               |

**Why this matters**: If your STT alone takes 500ms, you've already exhausted most of your latency budget. This is why model choice and orchestration matter a lot.

If you want more depth about latency and other thing you can check articel from pipecat [Conversational Voice AI in 2025](https://voiceaiandvoiceagents.com/) where they cover indepth.

For **local inference on CPU/modest GPU**:

- Expect 1.2-1.5s latency for the first response
- Subsequent turns may hit 800-1000ms as models warm up
- This is acceptable for local development; production requires better hardware or cloud providers

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

### Minimum System Requirements for Local Voice Agents

Here's what you actually need to get started:

#### For Development (CPU-Only)

| Component   | Minimum                                        | Recommended               |
| ----------- | ---------------------------------------------- | ------------------------- |
| **CPU**     | 4-core modern processor (Intel i5/AMD Ryzen 5) | 8-core or better          |
| **RAM**     | 16GB                                           | 32GB                      |
| **Storage** | 50GB SSD                                       | 100GB NVMe SSD            |
| **GPU**     | None required                                  | NVIDIA GTX 1070 or better |
| **Latency** | 1.5-2.5s per turn                              | 800-1200ms per turn       |

#### For Production (GPU-Accelerated)

| Component          | Entry                     | Mid-Range       | High-Performance              |
| ------------------ | ------------------------- | --------------- | ----------------------------- |
| **GPU**            | NVIDIA RTX 3060 (12GB)    | RTX 3080 (10GB) | RTX 4090 (24GB) or Tesla A100 |
| **VRAM**           | 8-12GB                    | 16GB            | 24GB+                         |
| **System RAM**     | 32GB                      | 64GB            | 128GB                         |
| **CPU**            | 8-core (Intel i7/Ryzen 7) | 16-core         | 32-core workstation           |
| **Latency Target** | 800-1000ms                | 500-700ms       | <500ms                        |

**The 2x VRAM Rule**: Your system RAM should be **at least double your total GPU VRAM**. If you have a single RTX 3080 (10GB), you need at least 20GB of system RAM; 32GB+ is better.

## Speech-to-Text (STT)

First, we are going to see how to run the STT component. As mentioned in Part 1, we are using Whisper from OpenAI. But before we blindly pick a model, we need to know what to look for.

### The Blueprints of Hearing: STT Selection Criteria

When selecting a Speech-to-Text model for production, "it works" isn't enough. You need to verify specific metrics to ensure it won't break your conversational flow.

#### 1. Word Error Rate (WER)

This is the cornerstone accuracy metric. It calculates the percentage of incorrect words.
**Formula**: `WER = (Substitutions + Deletions + Insertions) / Total Words`

- **Goal**: Pro systems aim for **5-10% WER** (90-95% accuracy).
- **Reality Check**: For casual voice chats, anything under **15-20%** is often acceptable.
- **Context Matters**: A "digit recognition" task might have 0.3% WER, while "broadcast news" might have 15%. Don't blindly trust paper benchmarks test on _your_ audio.

#### 2. Latency & Real-Time Factor (RTF)

Speed is more than just feeling fast; it's about physics.

- **Time to First Byte (TTFB)**: Time from "speech start" to "partial transcript". Target **<300ms**.
- **Real-Time Factor (RTF)**: `Processing Time / Audio Duration`.
  - If RTF > 1.0, the system is slower than real-time (impossible for live agents).
  - **Target**: You want an RTF of **0.5 or lower** (processing 10s of audio in 5s) to handle overheads.
- **The "Flush Trick"**: Advanced pipelines don't wait. When VAD detects silence, they "flush" the buffer immediately, cutting latency from ~500ms to ~125ms.

#### 3. Noise Robustness & SNR

Lab audio is clean; user audio is messy. Performance drops sharply when **Signal-to-Noise Ratio (SNR)** falls below 3dB.

- **"Talking" Noise**: Background chatter usually doesn't break modern models like Whisper.
- **"Crowded" Noise**: Train stations or cafes are the hardest tests. If your users are mobile, prioritize noise-robust models (like `distil-whisper`) over pure accuracy models.

#### 4. Critical Features for Agents

- **Speaker Diarization**: "Who spoke when?" Essential if you want your agent to talk to multiple people, though it adds latency.
- **Punctuation & Capitalization**: Raw STT is lowercase streams (`hello world`). Good models add punctuation (`Hello, world.`) which is **critical** for the LLM to understand semantics and mood.

### Model Selection for Real-Time Performance

From `faster-whisper` itself, we have used `Systran/faster-distil-whisper-medium.en` from Hugging Face, but feel free to explore others:

| Model name        | Params | Type         | Real-Time Factor (RTF)\* | Typical use case                                                  |
| ----------------- | ------ | ------------ | ------------------------ | ----------------------------------------------------------------- |
| **tiny**          | 39M    | Multilingual | 0.05 (20x real-time)     | Very fast, rough drafts, low-end CPU                              |
| **tiny.en**       | 39M    | English-only | 0.08 (12x real-time)     | Fast English-only STT with small footprint                        |
| **base**          | 74M    | Multilingual | 0.15 (6.5x real-time)    | Better than tiny, still lightweight                               |
| **base.en**       | 74M    | English-only | 0.20 (5x real-time)      | Accurate English with low compute                                 |
| **small**         | 244M   | Multilingual | 0.35 (2.8x real-time)    | Good balance of speed and quality                                 |
| **small.en**      | 244M   | English-only | 0.40 (2.5x real-time)    | Higher-quality English on moderate hardware                       |
| **distil-medium** | 140M   | Multilingual | 0.25 (4x real-time)      | **Best local balance**: 49% smaller, within 1% WER of full medium |
| **medium**        | 769M   | Multilingual | 0.80 (1.25x real-time)   | High accuracy, slower; needs stronger machine                     |
| **medium.en**     | 769M   | English-only | 0.85 (1.17x real-time)   | Very accurate English, heavier compute                            |
| **large / v2**    | 1.55B  | Multilingual | 2.5 (0.4x real-time)     | Best quality older large models, GPU required                     |
| **large-v3**      | 1.55B  | Multilingual | 3.2 (0.3x real-time)     | Latest, improved multilingual, GPU strongly recommended           |

_RTF (Real-Time Factor) = Time to process audio / Length of audio. 0.05 = 50x faster than real-time._

**Recommendation for local voice agents**:

- **CPU-only**: `distil-medium` or `small.en` (aim for <300ms latency)
- **GPU with 8GB VRAM**: `medium.en` (aim for 200-250ms latency)
- **GPU with 16GB+ VRAM**: `large-v3` (aim for 150-200ms latency)

### The Interruptibility Problem: Barge-In and VAD

Here's something rarely discussed openly: **VAD isn't just for silence detection it's a critical component for interruption handling (barge-in).**

When a user speaks while your agent is talking, three things must happen instantly:

1. **Echo Cancellation (AEC)**: Remove your agent's voice from the audio stream so the STT doesn't get confused hearing itself
2. **Voice Activity Detection (VAD)**: Detect the user speaking (probability-based, not just volume threshold)
3. **Immediate TTS Cancellation**: Stop the agent from continuing mid-sentence

Typical barge-in detection requires:

- **VAD Latency**: 85-100ms (using algorithms like Silero VAD, which is Bayesian/probability-based rather than energy-based)
- **Barge-in Stop Latency**: <200ms (system must stop speaking within 200ms of user interruption for natural feel)
- **Accuracy**: 95%+ (must not false-trigger on background noise)

Without proper barge-in handling, your voice agent sounds robotic because users can't interrupt they must wait for the full response.

**What's better: simple energy-based VAD that misses some speech, or Silero VAD that uses neural networks?**

Use **Silero VAD** which has builtin support in pipecat so we don't want to worry about much they handle for both CPU and GPU automatically. It trains models to understand "speech probability" rather than just volume, so it handles:

- Whispers and soft speech
- Background noise (doesn't trigger on dog barks)
- Different accents and speech patterns
- Real-time streaming (10-20ms window processing)

### How to run STT

To serve this, we need a server or inference engine. While `faster-whisper` has a library, we need a server like architecture (similar to Deepgram) where we connect to a WebSocket server, send audio, and receive text. I have written a simple WebSocket server that runs the model on either CPU or GPU.

I have dockerized everything to make our life easier

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

Next, we need a brain. But before we just pick "Llama 3", we need to understand the physics of running a brain on your computer.

### The Blueprints of Thinking: LLM Selection Criteria

Choosing an LLM for voice isn't about choosing the smartest one; it's about choosing the one that fits.

#### 1. The VRAM Formula

Will it fit? Don't guess. Use the math.
**Formula**: `VRAM (GB) ≈ Params (Billions) × Precision (Bytes) × 1.2 (Overhead)`

- **Precision Refresher**:
  - **FP16 (16-bit)**: 2 Bytes/param. (The standard).
  - **INT8 (8-bit)**: 1 Byte/param. (75% smaller than standard).
  - **INT4 (4-bit)**: 0.5 Bytes/param. (The sweet spot for locals).

**Example Calculation (Llama 3 8B)**:

- @ FP16: `8 × 2 × 1.2` = **19.2 GB** (Needs A100/3090/4090)
- @ INT4: `8 × 0.5 × 1.2` = **4.8 GB** (Runs on almost any modern GPU/Laptop!)

_Note: Context window (KV Cache) adds variable memory. 8K context is usually +1GB._

#### 2. Throughput vs. Latency

- **Tokens Per Second (TPS)**: How fast it reads/generates.
  - Humans read/listen at ~4 TPS.
  - > 8 TPS is diminishing returns for voice.
- **Time To First Token (TTFT)**: **This is the King metric**.
  - Sub-200ms = Instant.
  - > 2s = "Is it broken?"
  - **Goal**: Optimize for TTFT, not max throughput.

#### 3. Benchmarks That Actually Matter

Don't just look at the leaderboard. Look at the right columns.

- **MMLU**: General knowledge. Good baseline, but vague.
- **IFEval (Instruction Following)**: **Crucial for Agents**. Can it follow your system prompt instructions? Current small models (~2B) are getting good at this (80%+).
- **GSM8K**: Logic/Math. Good proxy for "reasoning" capability.

For a local voice agent, a **high IFEval** score is often more valuable than a high MMLU score because if the agent ignores your "Keep responses short" instruction, the user experience fails.

### Inference Engines

To run a model locally, we need an **Inference Engine**. If you search Google, you will find many options. Here are a few popular ones:

| Engine           | Primary Use                            | Hardware                       | Quantization Support       | Best For                             |
| :--------------- | :------------------------------------- | :----------------------------- | :------------------------- | :----------------------------------- |
| **Ollama**       | Local single-machine LLM serving       | CPU, GPU (NVIDIA, Apple Metal) | GGUF (Q4, Q5, Q8)          | Local dev, prototypes, low traffic   |
| **llama.cpp**    | CPU-optimized inference                | CPU (x86, ARM), GPU            | GGUF (Q2-Q8, AWQ, IQ2-IQ4) | Resource-constrained, edge devices   |
| **vLLM**         | High-throughput production LLM serving | NVIDIA GPU, AMD, Intel         | INT8, FP8, FP16, AWQ, GPTQ | Production APIs, high concurrency    |
| **TensorRT-LLM** | Maximum NVIDIA performance             | NVIDIA GPU only (CC >= 7.0)    | INT8, FP16, FP8 (H100+)    | Ultra-low latency, NVIDIA-only       |
| **SGLang**       | High-throughput production LLM serving | NVIDIA GPU, AMD, Intel         | FP16, INT8                 | Research, RadixAttention, multi-turn |

From this list, we are going to use **SGLang** to run our model on GPU, and for CPU, we can go with **Ollama**, which is very simple and easy to setup.

We are using **Llama 3.1 8B**, which is the current state-of-the-art for small open-source models.

### Why TTFT (Time-to-First-Token) Is What Matters

When users wait for a response, what they perceive is **how long until they hear the first word**. Here's why:

- **Prefill Phase**: Model processes your entire prompt (100-500ms for 8B models)
- **Decoding Phase**: Model generates one token at a time, streams it immediately to TTS
- **Key Insight**: TTS can start speaking as soon as token #1 arrives

So if your TTFT is 150ms, users hear the first word in 150ms + TTS latency (75-150ms) = **225-300ms total**. The full response might take 5 seconds to complete, but the user hears audio within 300ms.

This is why token-generation-speed-per-second (throughput) matters less than TTFT in conversational AI.

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

Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient.

### The Blueprints of Speaking: TTS Selection Criteria

Evaluating a "Mouth" is tricky because it's both objective (speed) and subjective (beauty).

#### 1. Latency & Real-Time Factor

- **TTFB (Time To First Byte)**: How fast does the first sound play?
  - **<100ms**: The Gold Standard.
  - **<300ms**: Acceptable.
  - **>500ms**: Breaks immersion.
- **Real-Time Factor (RTF)**:
  - Anything **< 0.1** (generating 10s audio in 1s) is amazing.
  - Production systems target **< 0.5**.

#### 2. Human Quality Metrics (MOS)

There isn't a "perfect" score, but we use **Mean Opinion Score (MOS)** (rated 1-5 by humans).

- **4.0 - 5.0**: Near Human. (Modern models like Kokoro/ElevenLabs).
- **2.5**: "Robot Voice". (Old school accessibility TTS).

#### 3. Naturalness & Prosody

"Prosody" is the rhythm and intonation.

- **Context Awareness**: Does it raise its pitch at a question mark? Does it pause for a period?
- **SSML Support**: Can you control it? (e.g. `<break time="500ms"/>` or `<emphasis>`).
- **Voice Cloning**:
  - **Zero-Shot**: 3s audio clip -> new voice. (Good for dynamic users).
  - **Fine-Tuned**: 3-5 hours of audio training. (Necessary for branded, professional voices).

### The Critical: TTS Context Window & Streaming

Here's a nuance many developers miss: **TTS models like Kokoro need context windows to avoid sounding robotic when receiving partial text**.

**The Problem Without Context Awareness:**

```
LLM sends: "It"     → Kokoro generates audio for just "It" → sounds like grunt
LLM sends: "is"     → Kokoro generates audio for just "is" → new voice, disconnected
LLM sends: "4"      → Kokoro generates audio for just "4" → jumpy prosody
```

**The Solution: Context Window in Streaming TTS:**

```
LLM sends: "It"     → Kokoro waits (buffering)
LLM sends: "is"     → Kokoro now has "It is" → generates better prosody
LLM sends: "4"      → Kokoro has "It is 4" → natural cadence
OR, Kokoro predicts: "wait for punctuation before speaking"
```

Kokoro uses a **250-word context window** internally. This means:

1. It buffers incoming tokens until it reaches punctuation (`.`, `!`, `?`, or a configurable threshold)
2. Once it has enough context, it generates audio with proper intonation
3. As more text arrives, it streams the audio bytes back without waiting for the full response

This is why **Kokoro excels at streaming** it doesn't try to speak partial fragments; it waits just enough to sound natural.

**Example**:

```
LLM stream: "Let me think... " (no punctuation yet)
  └─ Kokoro buffers silently

LLM stream: "Let me think... 2+2 equals 4." (full sentence)
  └─ Kokoro now has context → generates natural speech with correct stress
  └─ Streams audio back in chunks (50-100ms windows)
```

We'll also use the **Kokoro library** and build a **server to expose it as a service**.

### Folder Structure

Code location: `code/Models/TTS/Kokoro`

- `server.py`: Takes text input and streams out audio bytes.
- `download_model.py`: Fetches the model weights (`v0_19` weights).
- `kokoro-gpu.dockerfile`: GPU setup (Requires NVIDIA container toolkit).
- `kokoro-cpu.dockerfile`: CPU setup (Works on standard laptops).

If you like A minimal Kokoro-FastAPI server impelementation you can check out [here](https://github.com/programmerraja/Kokoro-FastAPI)

### Architecture Flow

The TTS server receives a stream of text tokens from the LLM. It immediately starts converting them to Phonemes (sound units) and generating audio. It streams this audio back to the user _before_ the LLM has even finished the sentence. This **Streaming Pipeline** is crucial for low latency and natural feel.

**How it works**:

1. **Token Buffering**: TTS receives token #1 from LLM. Checks if it's punctuation.

   - If no punctuation: buffer and wait for more tokens.
   - If punctuation or buffer size > 64 tokens: proceed.

2. **Phonemization**: Convert buffered text to phonetic units (e.g., "Hello" → `/həˈloʊ/`).

3. **Model Inference**: Kokoro generates audio features (mel-spectrogram) from phonemes.

4. **Waveform Generation**: iSTFTNet vocoder converts mel-spec to raw audio bytes.

5. **Streaming**: Audio chunks (50-100ms windows) stream back immediately over WebSocket.

6. **Repeat**: As LLM sends token #2, buffer grows, phonemization updates, new audio generates.

**Example Scenario**:
Input Stream: **"It"** → **"is"** → **"4"** → **"."** (with timestamps)

```
T=0ms:   LLM sends "It"
         Kokoro: "No punctuation, buffering..."

T=150ms: LLM sends " is"
         Kokoro: "Still buffering: 'It is'"

T=300ms: LLM sends " for"
         Kokoro: "Still buffering: 'It is for'"

T=400ms: LLM sends "."
         Kokoro: "Got punctuation! Phonemize: 'ɪt ɪz for'"
         → Infer mel-spec (100ms)
         → Vocoder (50ms)
         → Stream chunk #1 (40ms audio) at T=550ms ✓ User hears "It"

T=550ms: More tokens arrive, regenerate from updated context "It is for."
         → Refined mel-spec (includes proper prosody now)
         → Stream chunk #2 at T=650ms ✓ User hears "is"
         → Stream chunk #3 at T=750ms ✓ User hears "for"

Total latency: ~550ms to first audio, streaming continues until EOS token.
```

### Performance Benchmarks

| Setup                         | Model Size  | TTFB      | Throughput | Real-Time Factor |
| ----------------------------- | ----------- | --------- | ---------- | ---------------- |
| **CPU (Intel i7, 32GB RAM)**  | Kokoro 82M  | 500-800ms | 3-11x RT   | Suitable for dev |
| **GPU (RTX 3060, 12GB VRAM)** | Kokoro 82M  | 97-150ms  | 100x RT    | Production-ready |
| **GPU (RTX 4090, 24GB VRAM)** | Kokoro 82M  | 40-60ms   | 210x RT    | Excellent        |
| **Quantized (4-bit)**         | Kokoro INT4 | 200-300ms | 8-15x RT   | Good balance     |

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

## Putting It Together: End-to-End Latency

Now that we understand each component, here's what your full local pipeline looks like:

### Realistic Local Performance (8B LLM + Kokoro + Whisper on RTX 3060)

```
User speaks: "What is 2+2?"
    ↓
STT (faster-distil-whisper-medium)     : 200ms ✓
LLM (Llama 3.1 8B, TTFT)               : 120ms ✓
    └─ Token 1 "It" available at 120ms
    ↓
TTS (Kokoro buffering for punctuation) : 400ms ✓
    └─ Buffering tokens until "4." (takes ~300ms for full sentence)
    └─ Phonemization + inference: 100ms
    ↓
Streaming audio starts back to user    : 120 + 400 = 520ms ✓
User hears first word "It"

Subsequent tokens stream in background:
    Token 2 "is" available at 180ms    → Audio generated in parallel
    Token 3 "4" available at 250ms     → User hears full "It is 4" by 650ms
    Token EOS at 300ms                 → Stop TTS

TOTAL MOUTH-TO-EAR: ~650ms (acceptable for local, within production <800ms)
```

Compare to production APIs:

- **Deepgram STT + GPT-4 + ElevenLabs TTS (cloud)**: 200-300ms (optimized, lower variance)
- **Your local setup**: 650-800ms (good for dev, acceptable for many use cases)

---

## Homework: Integrate With Pipecat

So now that all three components are up and running, it's your turn to think through how we can integrate them with **Pipecat** and get a fully local **"Hello World"** working end to end.

**Challenge**:

1. Run all three Docker containers (STT, LLM, TTS) locally
2. Create a Pipecat pipeline that:
   - Accepts WebSocket audio from client
   - Sends to STT server (port 8000)
   - Streams STT output to LLM server (port 30000)
   - Streams LLM tokens to TTS server (port 8880)
   - Streams TTS audio back to client
3. Implement **barge-in handling**: If user speaks while TTS is playing, cancel TTS and process new input
4. Measure latency at each step

**Tips**:

- Use `asyncio` and `WebSocket` for non-blocking streaming
- Implement a simple latency meter to log timestamps
- Test with quiet and noisy audio to validate VAD
- Start with synchronous (blocking) for simplicity, then optimize

If you'd like to share your implementation, feel free to raise a PR on our GitHub repo here:
**[https://github.com/programmerraja/VoiceAgentGuide](https://github.com/programmerraja/VoiceAgentGuide)**
