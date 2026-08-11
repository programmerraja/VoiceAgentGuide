# Providers we use / tested (candidate notes)

Status: **pending your review**. Not in `graph.json` yet.

Sources: Slack `#t-02-voice-chat-ai` + `#standup` (after 2026-06-11).
Sanitized + pruned: dropped company-specific ops details and one-off measured numbers; kept durable, generally-applicable learnings.

Mark each item: `[ ] keep` / `[ ] edit` / `[ ] drop`.

---

## Provider map

| Role | Providers / tools |
|------|-------------------|
| **LLM host** | Cerebras, Fireworks, MorphLLM, Wafer.ai, Groq, Baseten, Featherless, GCP serverless / Vertex AI, Azure OpenAI / AI Foundry, LiveKit Inference, OpenAI (baseline) |
| **Gateway / discovery** | OpenRouter; Together, Venice (as Gemma hosts) |
| **LLM model** | Gemma 4, GPT-OSS, GPT-5.6 Luna, GPT-5.1, DeepSeek-V4-Flash/Pro, Minimax, Nemotron 3, Muse Glimmer, Gemini Flash Lite, GLM 5.1/5.2 |
| **TTS** | ElevenLabs, Inworld, Cartesia, Speechify, Piper, Kokoro, Fish Audio, Deepgram Flux TTS (+ Hume, Sarvam, Rime, Chirp 3 HD, Voxtral watched) |
| **STT** | Deepgram (Nova-3, Flux), AssemblyAI, Soniox, Cartesia Ink |
| **Turn detection** | Pipecat Smart Turn, LiveKit turn detector |
| **Transport / realtime** | LiveKit (Cloud + self-hosted), SIP trunks (Twilio, Plivo) |
| **Observability** | Langfuse (via OTEL) |

---

# LLM hosts

### Cerebras — ultra-low-latency OSS / Gemma host
- [ ] Among the **fastest hosts** for open models (Gemma-class, OSS).
- [ ] Best fit for **latency-sensitive supervisor / realtime** paths.
- [ ] Ultra-fast hosts usually cost **more per token** than serverless clouds for the same model — pick by role, not price alone.

### Fireworks — multi-model serverless host
- [ ] Multi-model serverless can be **markedly slower** than specialized fast hosts — verify latency before using for realtime/supervisor.
- [ ] **Serverless catalogs are unstable** (models get deprecated → 404s). Build fail-soft failover.

### MorphLLM — dedicated / fast OSS inference
- [ ] **Benchmark TTFT vs a baseline** before wiring into the voice runtime.
- [ ] **Measure from colocated infra** — TTFT from a laptop/mobile network is misleading.
- [ ] Vendor latency claims can miss real TTFT; trust your own **P50/P95**.
- [ ] Self-hosting GPUs rarely beats managed cost until **very high monthly token volume + high utilization**.

### Wafer.ai — latency-oriented OSS serverless
- [ ] Positioned as **OSS-only, fast**, with zero data retention.

### Groq — low-latency host
- [ ] Among the **fastest** for open models; re-benchmark periodically (rankings drift).

### Baseten — latency reference host
- [ ] Referenced as a benchmark host for GLM-class models.

### Featherless — cheap host for offline evals / sims
- [ ] Good for **sims + post-call evals** where voice-grade TTFT isn’t required.
- [ ] **Low-tier/fixed plans throttle hard on concurrency** — batch, don’t burst.
- [ ] Treat as a **test sandbox**, not a primary realtime host.

### GCP serverless / Vertex AI — cheaper model path, quota-sensitive
- [ ] Serverless clouds can be **much cheaper** than fast hosts for the same open model.
- [ ] **Quotas can bite mid-session** — verify tier/TPM before prod.
- [ ] Prefer a **host abstraction** so heavier ops (endpoint, service account, secrets) aren’t a one-off code path.
- [ ] Distinguish **Vertex vs the Gemini API** — same models, different auth/quota.

### Azure OpenAI / AI Foundry — regionized enterprise host
- [ ] **Not every model is in every region**; TPM/RPM vary by model + region — deploy region first.
- [ ] **Provisioned throughput** reserves capacity, but **burst spillover is billed**.
- [ ] Different **model families may need separate regional deployments**.
- [ ] Some params are **required, not optional** (e.g. `reasoning_effort` on certain paths) — send an explicit value; it’s billable/behavioral, not a free no-op.

### LiveKit Inference — colocated model host for voice
- [ ] Delivers **very low in-call TTFT** for realtime.
- [ ] May be **Cloud-only** — plan an **exit / self-host** path behind a provider interface.
- [ ] Architecture: **colocate transport + LLM + TTS** on one node to cut hops.

### OpenAI (direct) — baseline comparator
- [ ] Use as a **TTFT / accuracy baseline** when qualifying new hosts.

### OpenRouter / Together / Venice — discovery & routing
- [ ] OpenRouter shows **which hosts serve a given model** before native integration.
- [ ] Together / Venice appeared as **Gemma hosts** (candidates, not deeply validated).

---

# LLM models

### Gemma 4 — strong realtime voice candidate
- [ ] Strong **instruction following, tool use, sub-agent transfers**.
- [ ] Better accuracy than OSS models, at **higher cost per token**.
- [ ] Speed depends heavily on host — specialized fast hosts and colocated inference win.
- [ ] Smaller variants trade **cost/latency vs quality** — pick size after measuring.

### GPT-OSS — cheap workhorse
- [ ] Cheap on fast hosts, but **accuracy trails** tuned models like Gemma.

### GPT-5.6 Luna — cheaper GPT-family option
- [ ] **Much cheaper tokens** than the flagship GPT tier.
- [ ] Acceptable quality/transfers, but **higher TTFT** than colocated open models.

### GPT-5.1 — expensive baseline
- [ ] Use as a latency/cost **upper bound**.

### DeepSeek-V4 (Flash / Pro)
- [ ] Available via gateways and enterprise clouds.
- [ ] To A/B a model as an **eval judge**, you need a **logged ground-truth dataset**, not just prod traces.
- [ ] Treat voice latency/quality as **TBD until measured**.

### Minimax — weak tool / sub-agent reliability
- [ ] Unstable tool-calling / transfer loops in sims.
- [ ] Lesson: **TTFT ≠ production readiness** — test tool calling + multi-agent transfers explicitly.

### Nemotron 3 — often too slow for the main agent
- [ ] Larger variants can be **too slow for primary voice**; possibly OK for supervisors.
- [ ] Same model can **time out in one environment but work in another** — env/proxy matters.

### Muse Glimmer — watchlist
- [ ] Claims to beat Gemma on published benchmarks; **no in-call validation** yet.

### Gemini Flash Lite — bulk / lighter workloads
- [ ] Cheap + fast enough for batch-ish work; watch **rate limits**; confirm Vertex vs Gemini API.

### GLM 5.1 / 5.2 — industry latency picks
- [ ] Frequently cited for fast voice stacks.

---

## Cross-cutting comparisons

| Comparison | Generic takeaway |
|------------|------------------|
| Specialized fast host vs multi-model serverless | Fast host wins latency; a slow host can approach flagship latency → poor supervisor fit |
| Fast host vs serverless cloud (same model) | Fast host wins latency; serverless cloud is often several× cheaper — choose by role |
| Tuned model vs OSS | Tuned (Gemma) = better accuracy, higher $; OSS = cheaper, lower accuracy |
| Reliable model vs unstable on tools | A fast model that fails tool/transfer flows is not production-ready |
| Cheaper GPT tier vs flagship | Much cheaper tokens; keep flagship only if the quality gap justifies it |
| Managed vs self-host GPU | Self-host rarely wins until very high monthly token volume + utilization |

### Cross-cutting practices
- [ ] Report **P50 / P95 TTFT**, measured from **colocated** clients (not laptops).
- [ ] Split roles: **realtime / supervisor → lowest-TTFT hosts**; **sims / evals → cheapest host with enough concurrency**.
- [ ] Periodically **revisit whether a separate supervisor model** is still worth it as base models improve.
- [ ] Instrument **per-provider token + cost** metrics early.
- [ ] Treat provider params (e.g. `reasoning_effort`) as **per-provider optional/required**, never global.
- [ ] Expect **serverless deprecations and quota cliffs** — multi-host **failover is a requirement**, not a nice-to-have.

---

# TTS

### ElevenLabs — cloud production TTS
- [ ] Strong quality + voice catalog (accent/gender).
- [ ] **Cost/credits** become the binding constraint at scale — plan failover early.
- [ ] Keep a **hot secondary** and exercise fallback (providers do have outages).

### Inworld — cheaper / fallback TTS
- [ ] Cheaper primary or failover; one provider per call, different providers across deployments.

### Cartesia — cloud TTS (+ fast endpointing)
- [ ] Solid voices; **Ink** noted as fast for STT/endpointing.
- [ ] Plan for **model deprecations** before hard cutovers.

### Speechify — cloud TTS
- [ ] Good quality but **thin accent catalog** — multi-provider fills gaps.

### Piper — local / CPU TTS
- [ ] Small CPU model; good for **flow/logic testing**, not necessarily prod quality.

### Kokoro — local TTS
- [ ] Small local model; useful size/quality comparison vs Piper.

### Fish Audio — cloud TTS (evaluating)
- [ ] Often cited **best voice quality** with **reliability/latency tradeoffs**.
- [ ] Evaluate **emotion/prosody controls** as a quality axis beyond single-line MOS.

### Deepgram Flux TTS — conversation-native TTS
- [ ] Reads the **whole conversation** (not isolated lines) for tone/pacing continuity.

### General TTS
- [ ] Once latency/accuracy are solved, **voice quality is the next differentiator**.
- [ ] Hume, Sarvam, Rime Arcana, Google Chirp 3 HD, Mistral Voxtral tracked as catalog/quality options.

### Smart TTS / pre-recorded cache
- [ ] **Exact-hash caching → ~0% hit rate**; use **similarity matching**.
- [ ] Streaming LLM→TTS **conflicts with caching** (you need the full utterance to decide) → hurts TTFA on misses.
- [ ] Static lines/greetings can be **pre-recorded**; stream from same-region storage first, pre-download only if too slow.
- [ ] Measure **% repeated utterances** before building a cache.

---

# STT

### Deepgram — cloud ASR
- [ ] Keep a **secondary ASR** for outages.
- [ ] Community signal: **Flux > Nova-3** for voice agents.

### AssemblyAI — failover ASR
- [ ] Practical secondary; keep it provisioned.

### Soniox / Cartesia Ink — fast endpointing
- [ ] Cited as fast endpointing options that cut turn-taking latency.

### General STT
- [ ] Separate **diagnostic transcript metrics** from **billable audio STT**.
- [ ] Background noise keeps endpointing open → inflates TTFA.
- [ ] Build sims for **ASR errors, dialects, noise, interruptions**.

---

# Transport / Orchestration

### LiveKit — realtime media + agents runtime
- [ ] Cloud + self-hosted; **Inference may be Cloud-only** — plan LLM hosting separately.
- [ ] **Egress/recording capacity is a hard limit**; don’t make recording-start failure fatal to the call unless required.
- [ ] Failover: parallel workers, health-check self-hosted, route **new** sessions to Cloud when saturated (in-flight may drop).
- [ ] Filler-phrase support **changes across SDK versions** — prefer **intent-based fillers**.

### SIP / telephony (Twilio, Plivo)
- [ ] Inbound via SIP URI; outbound termination configured at the carrier.
- [ ] **Carrier CPS limits** throttle outbound fan-out — don’t burst.
- [ ] Failures often sit at the **carrier or media layer** — document trunks + health checks.

### Queues (outbound + retry)
- [ ] Separate **main** vs **retry/callback** queues, each with its own concurrency cap.

### Pipecat — orchestration / turn model
- [ ] Smart Turn for end-of-turn / barge-in with slow speakers; compare vs LiveKit’s turn detector.

---

# Observability

### Langfuse — tracing, cost, evals, datasets
- [ ] Features: tracing, sessions, cost/usage, latency, scores/evals, prompt management, **datasets** from eval I/O, OTEL ingestion.
- [ ] Tag **provider** as a property to group latency/cost by backend.
- [ ] Trace **sims + evals**, not only live calls, to estimate cost before scaling.
- [ ] Collapse overlapping post-call evals that contradict each other.

### OpenTelemetry → Langfuse gotchas
- [ ] Shared OTEL exporter + framework auto-instrumentation → **orphan/duplicate traces**; isolate/filter exporters.
- [ ] Langfuse **drops float `usage_details`** → send integers (e.g. `audio_duration_ms`).
- [ ] Async-drained spans look **“instant”** → backdate with real `duration_ms` / TTFT.
- [ ] OTEL misconfig can destabilize workers — treat exporter health as production-critical.

---

# Turn-taking / Latency

- [ ] Track **TTFT** (LLM) separately from **TTFA** (first audio) — fast LLM ≠ low TTFA.
- [ ] Even “fast” end-to-end stacks can sit around **~1–2s median** — budget UX accordingly.
- [ ] Compare **LiveKit turn detector vs Pipecat Smart Turn** for slow speakers / false barge-ins.
- [ ] Background noise fools turn detectors → delayed replies / high TTFA.
- [ ] Tool-call races: user barges in during a fetch → cover with sims.
- [ ] Prefer **intent-based fillers** over SDK-default fillers.

---

## Suggested atlas nodes from providers (checklist)

- [ ] `provider` nodes: Cerebras, Fireworks, MorphLLM, Wafer, Groq, Baseten, Featherless, Vertex/GCP, Azure OpenAI, OpenRouter
- [ ] `model` nodes: Gemma 4, GPT-OSS, GPT-5.6 Luna, DeepSeek-V4-Flash, Minimax, Nemotron 3, GLM 5.x, Gemini Flash Lite
- [ ] TTS: Inworld, Speechify, Fish Audio, Deepgram Flux TTS (+ existing ElevenLabs/Cartesia/Piper/Kokoro)
- [ ] STT: Deepgram (+ Flux), AssemblyAI, Soniox, Cartesia Ink
- [ ] Tools: LiveKit (transport + inference), Pipecat Smart Turn, Langfuse, OpenRouter
- [ ] Metric `ttfa`; patterns `provider-failover`, `host-abstraction`, `colocate-stack`
