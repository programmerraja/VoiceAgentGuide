# Voice and multimodal notes — candidates

Status: **merged into `data/graph.json`** (2026-08-12). Keep as review trail.

Source: personal notes under `MyBlog/.../Voice-Multimodal`.

I merged repeated material and dropped cloud price tables, vendor marketing, loose benchmark
claims, and long lists of links. The writing below is intentionally short and direct. It should
read like working notes, not a generated textbook.

Mark each section `[ ] keep`, `[ ] edit`, or `[ ] drop`.

---

## Audio foundations

### Waveform and frequency-domain audio — new concept

- [ ] A waveform is the microphone signal in the time domain: amplitude changing over time.
- [ ] Working directly with it preserves the original signal, but produces a long input sequence.
- [ ] Most speech systems first turn the waveform into a time-frequency representation. That
  throws away some detail, but gives the model a much smaller and more useful input.

Suggested links: `audio-representation part_of voice-agent`, `stt uses audio-representation`.

### STFT and spectrograms — new concept

- [ ] A Fourier transform tells us which frequencies exist in a signal, but not when they occur.
  Speech changes constantly, so we apply it to short overlapping windows. That is the Short-Time
  Fourier Transform.
- [ ] Stack those windows over time and you get a spectrogram: time on one axis, frequency on the
  other, and energy represented by intensity.
- [ ] This is why convolutions work well on audio. A spectrogram is a grid with local patterns,
  much like an image.

Suggested links: `stft produces spectrogram`, `spectrogram used_by stt`.

### Mel spectrogram — new concept

- [ ] A mel spectrogram remaps frequency to roughly match human hearing. We notice small changes
  at low frequencies more than equally sized changes at high frequencies.
- [ ] It is a practical default for speech recognition and synthesis: compact enough to process,
  while retaining the parts of the signal that matter most for speech.
- [ ] Whisper is a useful mental model: decode audio to PCM, window it, run an FFT, apply mel
  filters and log compression, then feed the log-mel frames to a transformer.

Suggested links: `mel-spectrogram part_of audio-processing`, `whisper uses mel-spectrogram`.

### MFCC — new concept

- [ ] MFCCs compress a mel spectrogram into a small hand-designed feature vector. They were a
  standard input to traditional speech recognizers and are still useful on constrained devices.
- [ ] Modern models usually learn richer representations themselves, so MFCCs are no longer the
  automatic choice.

Suggested links: `mfcc alternative_to mel-spectrogram`.

### Phonemes, graphemes and bytes — new concepts

- [ ] A grapheme is what is written; a phoneme is a sound that can change a word's meaning.
  TTS has to solve pronunciation, not merely read characters.
- [ ] Grapheme-to-phoneme conversion makes that pronunciation step explicit. It is especially
  useful for names, abbreviations and languages where spelling does not map cleanly to sound.
- [ ] Byte-based models take another route: they consume the UTF-8 bytes and learn the mapping
  without a separate text tokenizer or pronunciation dictionary.

Suggested links: `grapheme-to-phoneme part_of tts`, `phoneme related_to pronunciation`.

---

## Front end of a voice agent

### Voice activity detection — enrich existing `vad`

- [ ] VAD works on short audio frames and decides whether each frame contains speech. Adjacent
  speech frames are grouped into a segment before being sent downstream.
- [ ] A simple detector can use short-time energy and zero-crossing rate. Neural detectors handle
  messy rooms better, but cost more and can still fail on music, crosstalk and breathing.
- [ ] VAD quality is not just classification accuracy. In an agent, the visible failures are
  clipped first words, long waits, and the agent speaking over the caller.

### Denoising — new concept

- [ ] Denoising is useful only when it improves the signal that reaches VAD and STT. An aggressive
  filter can remove consonants or distort speaker cues, so cleaner-sounding audio is not always
  better recognition input.
- [ ] Real-time factor matters here because denoising sits on every incoming frame. A model that
  is accurate but consumes the latency budget is the wrong front-end model.

Suggested links: `denoising precedes vad`, `denoising measured_by rtf`.

### Real-time factor — enrich existing `rtf`

- [ ] RTF is processing time divided by audio duration. Below 1 means faster than real time.
- [ ] For a live agent, merely staying below 1 is not enough. The front end needs headroom for
  concurrent calls, jitter and the rest of the pipeline.

---

## Audio tokens and speech language models

### Neural audio codec — new concept

- [ ] A neural codec turns a waveform into a compact stream of learned audio tokens and can decode
  those tokens back into sound. Unlike a normal text tokenizer, it must carry acoustic detail such
  as speaker identity, pitch and timing.
- [ ] Codecs make speech generation look more like language modelling: predict token IDs, then
  decode them into audio.

Suggested links: `neural-audio-codec part_of speech-to-speech`.

### Residual vector quantization — new concept

- [ ] RVQ approximates a vector in stages. The first codebook picks a coarse vector; each later
  codebook quantizes what the earlier stages missed.
- [ ] More codebooks preserve more detail but increase bitrate and generation work. That is the
  central quality-versus-speed trade-off in codec-based speech models.
- [ ] EnCodec and Mimi are good concrete examples.

Suggested links: `rvq used_by neural-audio-codec`.

### Speech tokenizer objectives — new concept

- [ ] Speech tokens are not all trying to preserve the same thing.
- [ ] Semantic tokens keep the words and meaning, which is useful for ASR and understanding.
  Acoustic tokens keep timbre, prosody and fine sound detail, which is useful for generation.
- [ ] Speech-to-speech systems usually need a mixed representation. If the tokenizer drops
  speaker or timing information, the language model cannot recover it later.

Suggested links: `speech-tokenizer part_of speech-language-model`.

### Speech-augmented language model — new concept

- [ ] A speech-augmented LLM does not need to rebuild the language model. A speech encoder turns
  audio into embeddings, an adapter projects them to the LLM's embedding size, and those vectors
  are inserted where an audio placeholder appears in the prompt.
- [ ] The useful mental model is: reshape speech until the LLM can attend to it like text. The
  speech encoder handles perception; the LLM handles the task.
- [ ] Temporal subsampling matters because raw audio produces far too many frames for an LLM
  context window.

Suggested links: `speech-augmented-llm uses speech-encoder`,
`speech-augmented-llm uses modality-adapter`.

### Cascaded vs end-to-end speech — enrich existing `speech-to-speech`

- [ ] A cascaded agent gives us a transcript, interchangeable providers and control over each
  stage. That makes it easier to inspect and repair.
- [ ] An end-to-end speech model can preserve tone and overlap while removing network hops, but it
  is harder to constrain, evaluate and debug.
- [ ] This is not simply old versus new. The right choice depends on whether control or natural
  conversational behaviour matters more for the product.

### Full duplex — new concept

- [ ] Full duplex means listening and speaking at the same time. It is more than streaming in both
  directions: the model also has to decide when to yield, continue, interrupt or handle overlap.
- [ ] Half duplex is easier to operate because each turn has a clear boundary. Full duplex feels
  better only when interruption and timing are reliable.

Suggested links: `full-duplex part_of speech-to-speech`, `full-duplex related_to turn-taking`.

### Inner monologue — new pattern

- [ ] Some speech models predict a text-like stream before producing acoustic tokens. This hidden
  linguistic path gives generation a place to form a coherent answer before committing to sound.
- [ ] It also gives developers something easier to inspect than audio tokens, though it does not
  provide the clean observability of a fully cascaded pipeline.

Suggested links: `inner-monologue used_by moshi`.

---

## TTS architecture

### Acoustic model and vocoder — new concepts

- [ ] A common TTS split is text to acoustic representation, then acoustic representation to
  waveform. The second half is the vocoder.
- [ ] Autoregressive waveform models such as WaveNet showed how natural neural speech could sound,
  but generating one sample at a time was too slow. Later GAN, flow and codec decoders made
  waveform generation practical for real-time use.
- [ ] HiFi-GAN is the useful baseline to remember: fast GAN-based waveform synthesis used behind
  many acoustic models.

Suggested links: `acoustic-model part_of tts`, `vocoder part_of tts`.

### Voice cloning — new concept

- [ ] There are three common levels of cloning: condition on a short reference clip, adapt the
  model to one speaker, or continue from recent speech context.
- [ ] Reference conditioning is convenient; adaptation usually improves fidelity but needs more
  clean data and a separate model or adapter.
- [ ] Evaluate intelligibility and speaker similarity separately. A sample can sound natural while
  sounding like the wrong person.

Suggested links: `voice-cloning part_of tts`, `voice-cloning measured_by speaker-similarity`.

### Kokoro — enrich existing `kokoro-82m` if present

- [ ] Kokoro is interesting because it is small enough to change the deployment calculation. It
  can be a practical local or colocated TTS option where a large codec language model would be
  wasteful.
- [ ] Treat its small size as an operational advantage, not proof of voice quality. Accent,
  pronunciation and streaming behaviour still need to be tested on the actual script.

### Orpheus — candidate model

- [ ] Orpheus treats TTS as audio-token generation with a Llama-based model, then uses the SNAC
  codec to turn hierarchical tokens into a waveform.
- [ ] It is a clear example of TTS moving from spectrogram prediction toward language modelling
  over discrete audio.

Suggested links: `orpheus uses snac`, `orpheus part_of tts`.

### Fish Speech — candidate model

- [ ] Fish Speech separates slower linguistic planning from faster acoustic detail generation.
  That division is useful: meaning and waveform detail do not have to run at the same rate.
- [ ] Its Firefly-GAN decoder reconstructs the waveform from quantized features.

Suggested links: `fish-speech part_of tts`, `fish-speech uses firefly-gan`.

### Chatterbox — candidate model

- [ ] Chatterbox combines a text model, reference-audio tokenizer, speaker embedding and a
  flow-matching acoustic generator. The reference clip conditions both who is speaking and how
  they speak.
- [ ] It is a useful example of why cloning quality depends on more than one speaker vector:
  reference speech can also carry rhythm and expression.

Suggested links: `chatterbox part_of tts`, `chatterbox related_to voice-cloning`.

---

## Speech models

### Moshi — candidate model

- [ ] Moshi uses Mimi to encode and decode audio tokens, a language model for linguistic
  generation, and parallel streams for the user and assistant.
- [ ] The interesting part is not just low latency. It models listening and speaking as concurrent
  streams, so overlap and interruption are part of generation rather than external pipeline logic.
- [ ] Its inner text stream improves linguistic structure before audio-token generation.

Suggested links: `moshi part_of speech-to-speech`, `moshi uses mimi`,
`moshi uses inner-monologue`, `moshi uses full-duplex`.

### Canary — candidate model

- [ ] Canary is an encoder-decoder model for transcription and speech translation built around a
  FastConformer encoder.
- [ ] Its aggressive temporal downsampling is the important architectural idea: reduce the number
  of audio frames while retaining enough local and global context for recognition.
- [ ] Task tokens select transcription, translation, punctuation and capitalization behaviour.

Suggested links: `canary part_of stt`, `canary related_to speech-translation`.

---

## Inference and deployment


### Quantization — new concept

- [ ] Quantization reduces precision to shrink memory use and speed up inference on hardware that
  supports the chosen format.
- [ ] The actual win depends on the runtime and GPU. A smaller checkpoint does not automatically
  mean lower latency.
- [ ] Test quality per component. STT may lose rare words, an LLM may lose tool reliability, and
  TTS may introduce audible artefacts even when aggregate benchmarks barely move.

Suggested links: `quantization part_of inference-optimization`.

### Cache the stable work — new pattern

- [ ] Speaker embeddings are a good cache target: compute them once for a known voice instead of
  re-encoding the reference clip for every utterance.
- [ ] LLM KV caching avoids recomputing the conversation prefix during token generation.
- [ ] Whole-response caching is less useful for an agent because context changes, but fixed prompts
  and repeated system audio can still be cached safely.

Suggested links: `speaker-embedding-cache part_of inference-optimization`.

---

## Evaluation

### Evaluate the layers separately — enrich evaluation

- [ ] Audio quality, conversation quality and business outcome are different evaluations.
- [ ] For TTS: intelligibility, naturalness, speaker similarity and first-audio latency.
- [ ] For STT: WER by accent, language, noise and domain vocabulary—not one overall number.
- [ ] For the agent: task completion, tool correctness, interruption handling, escalation and
  end-to-end latency.
- [ ] A polished demo can hide failures because the room, script and caller are controlled. Keep a
  small set of ugly real-world calls and rerun them on every meaningful change.

### MOS and automated audio scores — enrich existing `mos`

- [ ] MOS is still the clearest answer to “would a person accept this voice?”, but it is slow and
  rater-dependent.
- [ ] Automated scores such as PESQ are useful for comparing signal degradation, not for declaring
  that synthetic speech sounds human.

Suggested links: `tts measured_by mos`, `tts measured_by pesq`.

---

## Intentionally not proposed

- Point-in-time GPU and cloud prices. They will age quickly and several source tables disagree.
- Claimed latency, quality and speed-up percentages without a linked primary benchmark.
- Long provider/tool directories. They are bookmarks, not knowledge.
- Generic “benefits / features / use cases” prose that adds no decision-making value.
- Hard universal targets such as one acceptable latency or resolution rate. These depend on the
  call type, transport, language and user expectation.
