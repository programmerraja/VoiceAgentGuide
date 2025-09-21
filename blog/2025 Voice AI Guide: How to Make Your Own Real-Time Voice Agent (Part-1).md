Over the past few months I’ve been building a fully open-source voice agent, exploring the stack end-to-end and learning a ton along the way. Now I’m ready to share everything I discovered.  

The best part? In 2025 you actually **can** build one yourself. With today’s open-source models and frameworks you can piece together a real-time voice agent that listens, reasons, and talks back almost like a human  without relying on closed platforms.  

Let’s walk through the building blocks, step by step.

## The Core Pipeline

At a high level, a modern voice agent looks like this:

![](../../Images/overview.png)

Pretty simple on paper but each step has its own challenges. Let’s dig deeper.

##  Speech-to-Text (STT)

Speech is a **continuous audio wave** it doesn’t naturally have clear sentence boundaries or pauses. That’s where **Voice Activity Detection (VAD)** comes in:

- **VAD (Voice Activity Detection):** Detects when the user starts and stops talking. Without it, your bot either cuts you off too soon or stares at you blankly.

Once the boundaries are clear, the audio is passed into an **STT model** for transcription.
#### Popular VAD

| Factor                | **Silero VAD**                                                                                                                             | **WebRTC VAD**                                                                                         | **TEN VAD**                                                                                                                               | **Yamnet VAD**                           | **Cobra (Picovoice)**                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| **Accuracy**          | State-of-the-art, >95% in multi-noise[](https://www.qed42.com/insights/voice-activity-detection-in-text-to-speech-how-real-time-vad-works) | Good for silence/non-silence; lower speech/noise discrimination                                        | High, lower false positives than WebRTC/Silero[](https://www.communeify.com/en/blog/ten-vad-webrtc-killer-opensource-ai-voice-detection/) | Good, multi-class capable                | Top-tier (see Picovoice benchmarks)[](https://picovoice.ai/docs/benchmark/vad/) |
| **Latency**           | <1ms per 30+ms chunk (CPU/GPU/ONNX)[](https://github.com/snakers4/silero-vad)                                                              | 10-30ms frame decision, ultra low-lag[](https://github.com/wiseman/py-webrtcvad/issues/68)             | 2-5ms (real-time capable)                                                                                                                 | 5–10ms/classify                          | 5–10ms/classify                                                                 |
| **Chunk Size**        | 30, 60, 100ms selector                                                                                                                     | 10–30ms                                                                                                | 20ms, 40ms custom                                                                                                                         | 30-50ms                                  | 30-50ms                                                                         |
| **Noise Robustness**  | Excellent, trained on 100+ noises[](https://github.com/snakers4/silero-vad)                                                                | Poor for some background noise/overlapping speech[](https://github.com/wiseman/py-webrtcvad/issues/68) | Excellent                                                                                                                                 | Moderate                                 | Excellent                                                                       |
| **Language Support**  | 6000+ languages/no domain restriction[](https://github.com/snakers4/silero-vad)                                                            | Language-agnostic, good for basic speech/silence                                                       | Language-agnostic                                                                                                                         | Multi-language possible                  | Language-agnostic                                                               |
| **Footprint**         | ~2MB JIT, <1MB ONNX, minimal CPU/edge[](https://github.com/snakers4/silero-vad)                                                            | ~158KB binary, extremely light                                                                         | ~400KB[](https://www.reddit.com/r/selfhosted/comments/1lvdfaq/found_a_really_wellmade_opensource_vad_great/)                              | ~2MB (.tflite format)                    | Small, edge-ready                                                               |
| **Streaming Support** | Yes, supports real-time pipelines                                                                                                          | Yes, designed for telecom/audio streams                                                                | Yes, real-time                                                                                                                            | Yes                                      | Yes                                                                             |
| **Integration**       | Python, ONNX, PyTorch, Pipecat, edge/IoT data[](https://github.com/snakers4/silero-vad)                                                    | C/C++/Python, embedded/web/mobile                                                                      | Python, C++, web                                                                                                                          | TensorFlow Lite APIs                     | Python, C, web, WASM                                                            |
| **Licensing**         | MIT (commercial/edge/distribution OK)[](https://github.com/snakers4/silero-vad)                                                            | BSD (very permissive)                                                                                  | Apache 2.0, open                                                                                                                          | Apache 2.0                               | Apache 2.0                                                                      |

 [Silero VAD](https://github.com/snakers4/silero-vad) is the gold standard and pipecat has builtin support so I have choosen that :
- Sub-1ms per chunk on CPU
- Just 2MB in size
- Handles 6000+ languages
- Works with 8kHz & 16kHz audio
- MIT license (unrestricted use)

#### Popular STT Options

What are thing we need focus on choosing STT for voice agent

- **Accuracy**:
	- **Word Error Rate (WER):** Measures transcription mistakes (lower is better).
		- Example: WER 5% means 5 mistakes per 100 words.
	- **Sentence-level correctness:** Some models may get individual words right but fail on sentence structure.
- **Multilingual support:** If your users speak multiple languages, check language coverage.
- **Noise tolerance:** Can it handle background noise, music, or multiple speakers?
- **Accent/voice variation handling:** Works across accents, genders, and speech speeds.
- **Voice Activity Detection (VAD) integration:** Detects when speech starts and ends.
- **Streaming:** Most STT models work in batch mode (great for YouTube captions, bad for live conversations). For real-time agents, we need streaming output words should appear _while you’re still speaking_.
- **Low Latency:** Even 300 500ms delays feel unnatural. Target **sub-second responses**.

Whisper often comes first to mind for most people when discussing speech-to-text because it has a large community, numerous variants, and is backed by OpenAI. 

**OpenAI Whisper Family**

- [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) — State-of-the-art accuracy with multilingual support
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** — Optimized implementation using CTranslate2
- **[Distil-Whisper](https://github.com/huggingface/distil-whisper)** — Lightweight for resource-constrained environments
- **[WhisperX](https://github.com/m-bain/whisperX)** — Enhanced timestamps and speaker diarization

 **NVIDIA** also offers some interesting STT models, though I haven’t tried them yet since Whisper works well for my use case. I’m just listing them here for you to explore:
 
 - [Canary Qwen 2.5B](https://huggingface.co/nvidia/canary-1b) — Leading performance, 5.63% WER
 - [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b) — Ultra-fast inference (3,386 RTFx)

Here the comparsion table

| Model             | WER (EN, Public Bench.)                                    | Multilingual       | Noise/Accent/Voice                 | Sentence Accuracy | VAD Integration | Streaming | Latency                       |
| ----------------- | ---------------------------------------------------------- | ------------------ | ---------------------------------- | ----------------- | --------------- | --------- | ----------------------------- |
| Whisper Large V3  | 2–5%                                                       | 99+                | Excellent                          | Excellent         | Yes (Silero)    | Batch†    | ~700ms†                       |
| Faster-Whisper    | 2–5%                                                       | 99+                | Excellent                          | Excellent         | Yes (Silero)    | Yes       | ~300ms‡                       |
| Canary 1B         | 3.06% (MLS EN) [](https://huggingface.co/nvidia/canary-1b) | 4 (EN, DE, ES, FR) | Top-tier, fair on voice/gender/age | Excellent         | Yes             | Yes       | ~500ms–<1s                    |
| Parakeet TDT 0.6B | 5–7%                                                       | 3 (EN, DE, FR)     | Good                               | Very Good         | Yes             | Yes       | Ultra Low (~3,400x Real-time) |

#### Why I Chose **FastWhisper**

After testing, my pick is **FastWhisper**, an optimized inference engine for Whisper.

**Key Advantages:**
- **12.5× faster** than original Whisper
- **3× faster** than Faster-Whisper with batching
- **Sub-200ms latency** possible with proper tuning
- **Same accuracy** as Whisper
- Runs on **CPU & GPU** with automatic fallback

It’s built in **C++ + CTranslate2**, supports batching, and integrates neatly with **VAD**.

For more you can check [Speech to Text AI Model & Provider Leaderboard](https://artificialanalysis.ai/speech-to-text) 

### Large Language Model (LLM)

Once speech is transcribed, the text goes into an **LLM  the “brain” of your agent**.

What we want in an LLM for voice agents:

- Understands prompts, history, and context
- Generates responses quickly
- Supports **tool calls** (search, RAG, memory, APIs)

#### Leading Open-Source LLMs

**Meta Llama Family**

- [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) — Open-source leader
- Llama 3.2 (1B, 3B, 11B) — Scaled for different deployments
- **128K context window** — remembers long conversations
- **Tool calling support** — built-in function execution

**Others**
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) / **Mixtral 8x7B** — Efficient and competitive
- [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) — Strong multilingual support
- [Google Gemma](https://huggingface.co/google/gemma-2-27b-it) — Lightweight but solid

#### My Choice: Llama 3.3 70B Versatile

Why?
- **Large context window** → keeps conversations coherent
- **Tool use** built-in
- Widely supported in the open-source community

##  Text-to-Speech (TTS)

Now the agent needs to **speak back**  and this is where quality can make or break the experience.

A poor TTS voice instantly ruins immersion. The key requirements are:
- **Low latency**  avoid awkward pauses
- **Natural speech**  no robotic tone
- **Streaming output**  start speaking mid-sentence

#### Open-Source TTS Models I’ve Tried

There are plenty of open-source TTS models available. Here’s a snapshot of the ones I experimented with:

- [**Kokoro-82M**](https://huggingface.co/hexgrad/Kokoro-82M) — Lightweight, #1 on HuggingFace TTS Arena, blazing fast
- [**Chatterbox**](https://huggingface.co/ResembleAI/chatterbox) — Built on Llama, fast inference, rising adoption
- [**XTTS-v2**](https://github.com/coqui-ai/TTS) — Zero-shot voice cloning, 17 languages, streaming support
- [**FishSpeech**](https://github.com/fishaudio/fish-speech) — Natural dialogue flow
- [**Orpheus**](https://huggingface.co/CanopyLabs/Orpheus-3B) — Scales from 150M–3B
- [Dia](https://github.com/nari-labs/dia?tab=readme-ov-file) — A TTS model capable of generating ultra-realistic dialogue in one pass.

| Factor                          | Kokoro-82M                                                                                                                  | Chatterbox                                     | XTTS-v2                                              | FishSpeech                             | Orpheus                                    |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| **Voice Naturalness**           | Human-like, top-rated in community[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models)           | Very natural, quickly improving                | High, especially with good samples                   | Natural, especially for dialogue       | Good, scales with model size               |
| **Expressiveness / Emotion**    | Moderate, some emotional range                                                                                              | Good, improving                                | High, can mimic sample emotion                       | Moderate, aims for conversational flow | Moderate-High, model-dependent             |
| **Accent / Language Coverage**  | 8+ languages (EN, JP, ZH, FR, more)[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models)          | EN-focused, expanding                          | 17+ languages, strong global support                 | Several; focus varies                  | Varies by checkpoint (3B supports many)    |
| **Latency / Inference**         | <300ms for any length, streaming-first[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models)       | Fast inference, suitable for real-time         | ~500ms (depends on hardware), good streaming support | ~400ms, streaming variants             | 3B: ~1s+ (large), 150M: fast (CPU/no-GPU)  |
| **Streaming Support**           | Yes, natural dialogue with chunked streaming[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models) | Yes                                            | Yes, early output                                    | Yes                                    | Yes (3B may be slower)                     |
| **Resource Usage**              | Extremely light (<300MB), great for CPU/edge[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models) | Moderate (500M params), GPU preferred          | Moderate-high, 500M+ params, GPU preferred           | Moderate, CPU/GPU                      | 150M–3B options (higher = more GPU/memory) |
| **Quantization / Optimization** | 8-bit available, runs on most hardware                                                                                      | Some support                                   | Yes, 8-bit/4-bit                                     | Yes                                    | Yes                                        |
| **Voice Cloning / Custom**      | Not by default, needs training                                                                                              | Via fine-tuning                                | Zero-shot (few seconds of target voice)              | Beta, improving cloning                | Fine-tuning supported for custom voices    |
| **Documentation / Community**   | Active, rich demos, open source, growing[](https://github.com/hexgrad/kokoro)                                               | Good docs, quickly growing                     | Very large (Coqui), strong docs                      | Medium but positive community          | Medium, active research group              |
| **License**                     | Apache 2.0 (commercial OK)                                                                                                  | Commercial/Proprietary use may require license | LGPL-3.0, open (see repo)                            | See repo, mostly permissive            | Apache 2.0                                 |
| **Pretrained Voices / Demos**   | Yes (multiple voices, demos available)[](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models)       | Yes, continually adding more                   | Yes, huge library, instant demo                      | Yes                                    | Yes (many public models on Hugging Face)   |
#### Why I Chose **Kokoro-82M**

**Key Advantages:**

- **5–15× smaller** than competing models while maintaining high quality
- Runs under **300MB** — edge-device friendly
- **Sub-300ms latency**
- High-fidelity **24kHz audio**
- **Streaming-first design** — natural conversation flow

**Limitations:**

- No zero-shot voice cloning (uses a fixed voice library)
- Less expressive than XTTS-v2
- Relatively new model with a smaller community

You can also check out my minimal **[Kokoro-FastAPI server](https://github.com/programmerraja/Kokoro-FastAPI)** to experiment with it:  

## Speech-to-Speech Models

Speech-to-Speech (S2S) models represent an exciting advancement in AI, combining **speech recognition, language understanding, and text-to-speech synthesis** into a single, end-to-end pipeline. These models allow **natural, real-time conversations** by converting speech input directly into speech output, reducing latency and minimizing intermediate processing steps.

Some notable models in this space include:

- [**Moshi**](https://github.com/kyutai-labs/moshi): Developed by Kyutai-Labs, Moshi is a state-of-the-art speech-text foundation model designed for **real-time full-duplex dialogue**. Unlike traditional voice agents that process ASR, LLM, and TTS separately, Moshi handles the entire flow end-to-end.

- [CSM](https://github.com/SesameAILabs/csm) (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com/) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

- **[VALL-E & VALL-E X (Microsoft)](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-x/)**: These models support **zero-shot voice conversion** and speech-to-speech synthesis from limited voice samples.

- **[AudioLM (Google Research)](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)**: Leverages **language modeling on audio tokens** to generate high-quality speech continuation and synthesis.


Among these, I’ve primarily worked with **Moshi**. I’ve implemented it on a **FastAPI server with streaming support**, which allows you to test and interact with it in real-time. You can explore the FastAPI implementation here: [FastAPI + Moshi GitHub](https://github.com/programmerraja/FastAPI_Moshi).

## Framework (The Glue)

Finally, you need something to tie all the pieces together: **streaming audio, message passing, and orchestration**.

**Open-Source Frameworks**

**[Pipecat](https://github.com/pipecat-ai/pipecat)**
- Purpose-built for **voice-first agents**
- **Streaming-first** (ultra-low latency)
- Modular design — swap models easily
- Active community

**[Vocode](https://github.com/vocodedev/vocode-python)**
- Developer-friendly, good docs
- Direct telephony integration
- Smaller community, less active

**[LiveKit Agents](https://github.com/livekit/agents)**
- Based on WebRTC
- Supports voice, video, text
- Self-hosting options

**Traditional Orchestration**
- **LangChain** — great for docs, weak at streaming
- **LlamaIndex** — RAG-focused, not optimized for voice
- **Custom builds** — total control, but high overhead

#### Why I Recommend Pipecat

**Voice-Centric Features**

- Streaming-first, frame-based pipeline (TTS can start before text is done)
- Smart Turn Detection v2 (intonation-aware)
- Built-in interruption handling

**Production Ready**
- Sub-500ms latency achievable
- Efficient for long-running agents
- Excellent docs + examples
- Strong, growing community

**Real-World Performance**
- ~500ms voice-to-voice latency in production
- Works with Twilio + phone systems
- Supports multi-agent orchestration
- Scales to thousands of concurrent users

| Feature             | Pipecat     | Vocode      | LiveKit | LangChain |
| ------------------- | ----------- | ----------- | ------- | --------- |
| Voice-First Design  | ✅           | ✅           | ⚠️      | ❌         |
| Real-Time Streaming | ✅           | ✅           | ✅       | ❌         |
| Vendor Neutral      | ✅           | ✅           | ✅       | ⚠️        |
| Turn Detection      | ✅ Smart V2  | ⚠️ Basic    | ✅       | ❌         |
| Community Activity  | ✅ High      | ⚠️ Moderate | ✅ High  | ✅ High    |
| Learning Curve      | ⚠️ Moderate | ⚠️ Moderate | ❌ Steep | ✅ Easy    |

## Lead to Next Part

In this first part, we’ve covered the **core tech stack and models** needed to build a real-time voice agent.

In the next part of the series, we’ll dive into **integration with Pipecat**, explore our **voice architecture**, and walk through **deployment strategies**. Later, we’ll show how to enhance your agent with **RAG (Retrieval-Augmented Generation)**, **memory features**, and other advanced capabilities to make your voice assistant truly intelligent.

Stay tuned the next guide will turn all these building blocks into a working, real time voice agent you can actually deploy.

I’ve created a **GitHub repository** **[VoiceAgentGuide](https://github.com/programmerraja/VoiceAgentGuide)** for this series, where we can store our notes and related resources. Don’t forget to **check it out** and share your **feedback**. Feel free to **contribute or add missing content** by submitting a pull request (PR).


## Resources
- [Voice AI & Voice Agents An Illustrated Primer](https://voiceaiandvoiceagents.com/)
- 







