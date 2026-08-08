# Live / Ambient Mode — Design Notes and High-Level Plan

Recorded 2026-08-08 from discussion with the user. **This is post-v1 direction,
not current work.** It exists so that v1 decisions keep the seams this mode
needs, and so the intent is written down next to the plan rather than living in
one conversation. Nothing here changes `plan.md`'s settled decisions.

## Intent

The terrarium has two lives:

1. **Precomputed mode (v1, unchanged).** Offline analysis of a known track,
   timeline file, deterministic-when-pinned playback. The standard
   pre-designed-visuals use case for shows and installations. This mode does
   not need to become more "real-time" — it is a different product and stays
   as-is.
2. **Live / ambient mode (this document).** The piece behaves as an organism
   that *perceives its environment*: an art installation or live-performance
   visual that takes line-in or microphone input and reacts meaningfully — to
   music, but also to talking, laughter, yelling, crowd energy, emotional tone,
   and silence. Target setting: a festival, where anything can happen around it.

## The load-bearing insight: the driver bank is the seam

Revision 4 made the sim's entire musical input a bank of ~16 named driver
signals with per-driver gains, plus a sparse impulse-event lane. The sim does
not know those drivers come from MuQ + All-In-One + demucs. Live mode is **a
second perception stack producing the same driver-bank-shaped interface** — the
modulation layer, impulse lane, species mechanics, soil, and renderer carry
over untouched. This is not a fork; it is a second front end.

Contract implications for v1 (cheap, keep them true):

- Drivers stay **named and source-agnostic**. Nothing downstream may assume a
  driver is "PCA component 3 of MuQ".
- The timeline sampler stays behind an interface that a streaming source
  (WebSocket) could implement. File-playback is one implementation.
- The event lane stays `{ t, kind, strength }` with kinds as open strings —
  `laugh` and `shout` must be as legal as `kick`.

## Perception stack (ambient)

Music-specialized MuQ is the wrong center of gravity for ambient input —
speech, laughter, and crowd noise are out-of-distribution for it. General-audio
models take over, and they are conveniently smaller and more streamable:

| Lane | Source | Latency | Feeds |
|---|---|---|---|
| Fast DSP | loudness, spectral flux, band-split onsets (causal DSP) | ~ms | impulse events, fast drivers |
| Event tagging | AudioSet tagger (YAMNet 3.7M / PANNs CNN14 / BEATs) | ~1 s | presence drivers: speech, laughter, cheering, applause, music, singing; onsets of these → events |
| Character | **CLAP** sliding window (10 s window, ~1 s hop) + text-anchor pairs | ~1–2 s | named semantic axes ("tense crowd"−"calm ambience", "laughter"−"silence", …) |
| Speech tone | emotion2vec / wav2vec2-SER, gated by Silero VAD | ~1–2 s | valence/arousal drivers when speech is present |
| Music sub-stack | streaming chunked MuQ, online beat tracker (BeatNet-class), live stem-activity estimate | ~0.5–2 s | the v1 music drivers, gated by a music-presence driver |

Latency stratification is the same trick as v1's stream split: impulses ride
the millisecond DSP lane; character rides slow-slewed seconds-scale lanes where
1–2 s of latency is imperceptible. Live music and ambient are **one mode** —
the music sub-stack gates in when the tagger says music dominates, it is not a
separate build.

CLAP anchor prompts are **workbench data**: editing the text pairs live is art
direction, exactly like editing depths and gains.

### Streaming MuQ, for the music sub-stack

MuQ (310M, 12 Conformer layers, 25 Hz, 1024-dim) has no track-level state — the
offline pipeline already runs it in 30 s chunks. The barrier is bidirectional
attention *within* the window, so it cannot be made causal without retraining.
The workaround is a trailing window re-run every ~0.5–1 s, keeping the newest
frames: bounded latency, slight right-edge quality loss (newest frames lack
future context). Comfortably real-time on the dev GPU (a full track analyzes in
seconds). Do not attempt in-browser inference (1.2 GB fp32; fp16 NaNs).

### Later: distilled causal student

The "distilled NN" seam in plan.md has a streaming twin: train a small causal
model (causal TCN / GRU / state-space, 5–30M params) on mel input to predict
the offline pipeline's driver outputs directly. Training data is free
(run the offline pipeline over any corpus), the target is soft (drivers that
wiggle the same way, not benchmark fidelity), and bidirectional→streaming
distillation is well-precedented in speech. This could eventually collapse the
whole perception stack into one model small enough for in-browser ONNX. It is
the last phase, not the first.

## Two design commitments

**Species keyed to sound sources, not stems.** The v1 mechanism "an instrument
entering is a population blooming" translates directly: in ambient mode,
species key to sound categories — music/bass energy, voices, laughter/crowd,
ambient texture. A conversation starting near the installation is a population
blooming. Perception expressed by the mechanism the sim already has.

**Habituation via running normalization.** Per-track z-scoring has no "whole
track" in live mode. Replace it with running mean/variance at a long time
constant: the installation *habituates* — a loud room slowly becomes baseline,
and the sudden laugh stands out against it. This is not a workaround for
missing statistics; it is the perceptual mechanism, and it is thematically
central (continuity and memory). Layered time constants (minutes for gain,
hours via soil) give the piece a memory of its day.

## Architecture

```
mic / line-in
   │
perception process (Python, local GPU box at the venue)
   ring buffer → [DSP lane | tagger | CLAP | SER | music sub-stack]
   running-stats normalization → driver frames + events
   │  WebSocket (driver-bank frames @ 10 Hz + sparse events)
   ▼
browser sim (unchanged substrate)
   streaming driver source implements the sampler interface
   → modulation → physarum → render
```

Festival realities to design for: mic gain staging and wind, clipping
robustness, hours of unattended runtime (physarum-can't-die earns its keep),
and long silences — which should read as the world going quiet and dormant,
arguably the most important reaction of all.

## Phasing (high level)

Ordered so each phase yields something runnable, and so a demo-able reactive
installation exists early.

- **L0 — feasibility bench.** No product code. On the dev GPU: measure
  per-window latency of chunked MuQ and sliding CLAP; measure how far
  right-edge streaming embeddings diverge from offline ones for the same audio.
  Go/no-go numbers for the music sub-stack.
- **L1 — the streaming seam.** In the web app, put the timeline sampler behind
  an interface; add a WebSocket driver source. Prove it with a *fake* live
  server that replays an existing precomputed timeline over the socket —
  identical visuals through the streaming path. No perception yet. This is the
  phase that touches v1 code, and it is small.
- **L2 — fast DSP lane.** Python perception process: audio capture, ring
  buffer, loudness/flux/band-onset drivers, impulse events, running
  normalization. Wire to the sim. **This already yields a reactive ambient
  installation** — character comes later, but it moves with the room.
- **L3 — general-audio perception.** CLAP sliding window + text-anchor axes as
  named drivers, AudioSet tagger presence drivers and events, VAD + speech
  tone. Species re-keyed to sound categories (config, not code). Habituation
  time constants tuned in the workbench; anchor prompts editable as data.
- **L4 — music sub-stack.** Music-presence gating, streaming chunked MuQ,
  online beat tracking, live stem-activity estimate (cheap estimator first,
  not full separation). Live-performance use becomes first-class.
- **L5 — venue hardening.** Hours-long soak runs, silence/dormancy behavior,
  clipping/wind robustness, auto-recovery (process supervision, reconnect),
  operator workbench view for on-site tuning.
- **Later — distilled causal student**, as above; possible in-browser
  deployment; possible unification with the offline pipeline.

## Open questions (decide when the work starts, not now)

- Capture hardware and platform for the venue box (Windows/WSL2 vs a Linux
  box; audio capture API in Python).
- Whether ambient character needs a corpus-global PCA/z-score basis for CLAP
  drivers or purely running stats.
- How dormancy should look (alive-fraction floor? soil-only mode?) — an art
  direction question.
- Whether L2's band-split onsets on the raw mix are legible enough at a loud
  venue, or whether live stem separation gets promoted.
