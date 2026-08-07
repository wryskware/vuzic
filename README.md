# Latent Music Terrarium

> **Status: placeholder.** This repo exists, has its handoff brief, and is
> waiting to be scaffolded. Nothing is implemented yet.

A browser-based audiovisual experience that turns music into an evolving
simulated world.

Rather than reacting to volume, bass, or tempo, it uses a music-analysis model
to capture higher-level structure — recurring motifs, mood changes, structural
transitions, similarity between sections — and drives a living WebGPU
environment from that. A returning chorus should bring back a recognizable
world, evolved by whatever changed in the music.

The goal is not literal visualization. It is the feeling that the music is being
interpreted by an artificial ecosystem with memory.

Full brief: [`docs/handoff.md`](docs/handoff.md).

## Two halves

**Music analysis** — offline. A song is processed by an audio/music model and
reduced to a compact timeline describing how the musical state changes.

**Visual simulation** — real time. A WebGPU renderer consumes that timeline and
maintains its own internal state and history, so the world develops
continuously rather than being recomputed frame to frame.

## First milestone

Prove the central concept, nothing more:

- one analyzed song
- one visual simulation
- one deterministic seed
- browser playback
- tight sync between audio and its latent timeline

The question to answer: do model-derived musical representations produce visuals
that feel more coherent, expressive, and memorable than a conventional
audio-reactive visualizer? If the answer is no, that is a finding, and this repo
is cheap to abandon.

## Context

Part of the [Wryskware](https://wryskware.dev) ecosystem — an independent repo
with its own stack and deployment, presented through the Wryskware portfolio.
Likely surface: `dreams.wryskware.dev`.

## Getting started

Not yet. See `docs/scaffolding-notes.md` for the open decisions that need
answering before the first line of code.
