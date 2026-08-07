# Scaffolding Notes — Open Decisions

Written at repo-creation time so the next session starts from questions rather
than a blank page. None of these are decided. Each has a leaning, stated
honestly as a leaning.

> **Superseded in part (2026-08-06).** `docs/plan.md` answers most of these
> against the survey in `docs/research/`. The leanings below held up: hybrid
> analysis, hard dimensionality reduction, Web Audio clock, direct WebGPU. The
> one that did not is §4's lean toward the Lenia family — its interesting
> parameter region is a thin fractal shell, which breaks parameter
> interpolation. Read this file for the *reasoning*, `plan.md` for the
> conclusions.

## 1. Where does the latent representation come from?

The whole premise rests on this. Options, roughly in order of effort:

- **Open audio embedding models** (e.g. CLAP-family text-audio embeddings,
  MERT / music-understanding encoders, or an MFCC→learned-projection baseline).
  Run offline in Python, emit a timeline.
- **Self-supervised structure analysis** — self-similarity matrices, novelty
  curves, segment boundaries. Not "latent" in the model sense but directly
  encodes recurrence and structure, which is what the brief actually asks for.
- **Hybrid** — embeddings for *what a moment feels like*, self-similarity for
  *when moments return*.

Leaning: hybrid, because "a returning chorus brings back a recognizable world"
is a recurrence problem more than an embedding problem, and self-similarity is
cheap, interpretable, and debuggable. Embeddings then supply the character of
each region.

**Decide this first.** It determines the timeline format, and the timeline
format determines the renderer's input contract.

## 2. What is the timeline format?

Needs to be compact, seekable, and stable enough to version. Sketch:

```jsonc
{
  "version": 1,
  "track": { "id": "...", "duration": 214.7, "sampleRate": 48000 },
  "hopSeconds": 0.1,          // fixed grid; renderer lerps between frames
  "dims": 16,                 // reduced latent dimensionality
  "frames": [ /* Float32, dims per frame, likely a side-car .bin */ ],
  "segments": [ { "start": 0, "end": 31.2, "label": "A", "similarTo": [] } ],
  "events": [ { "t": 31.2, "kind": "boundary", "strength": 0.8 } ]
}
```

Open: raw dimensionality vs. PCA/UMAP reduction; JSON+binary side-car vs. a
single binary; whether segment labels are clustered offline or derived live.

Leaning: reduce hard (8–24 dims). The renderer wants a few smooth, meaningful
axes it can bind to simulation parameters, not a 768-dim vector.

## 3. Analysis pipeline packaging

Python offline (librosa / torch) is the pragmatic choice — the model ecosystem
is there. That means this repo is **polyglot**: a `analysis/` Python tool and a
`web/` TypeScript app, with the timeline format as the contract between them.

Open: keep both in this repo (leaning yes — they co-evolve and the contract is
private) or split later. Whether analysis runs locally only, or eventually as a
job somewhere.

## 4. Renderer structure

WebGPU compute + render. Core question: **what carries the memory?**

The simulation state is the memory. GPU buffers holding agent/particle/field
state persist across frames; the latent timeline modulates the *rules*, not the
state directly. Recurrence in the music then produces recurrence in the world
because the same rules re-applied to an evolved state give a related-but-changed
result. That is the mechanism the brief is describing.

Open: what the simulation actually *is* for v1 — particle swarm with learned
force fields, reaction-diffusion, continuous cellular automata (Lenia-like),
agent ecosystem. Leaning: something in the Lenia / continuous-CA family, because
it has genuine emergent morphology, is stateful by construction, and is a pure
compute-shader workload.

Open: TypeScript directly on the WebGPU API vs. a thin wrapper. Leaning: direct.
The abstraction surface here is small and the control matters.

## 5. Audio playback and sync

Latent frames must land on the right audio time under seek, pause, and drift.

Open: `<audio>` + `currentTime` (simple, coarse) vs. Web Audio API with a scheduled
source and sample-accurate clock (more work, exact). Leaning: Web Audio, driving
a clock the renderer samples — sync quality is load-bearing for the whole
experience, and a visualizer that lags is immediately dead.

## 6. Determinism

Seeded PRNG in shaders, fixed simulation timestep decoupled from render rate.
Decide the timestep policy early; retrofitting determinism is miserable.

## 7. Stack choices to make at scaffold time

- Bundler: Vite (default, no real competition here).
- TypeScript strict, Node 22+, npm, own lockfile.
- Shader authoring: raw WGSL files with a Vite import plugin. Avoid a shader DSL.
- Testing: whatever is enough for the analysis pipeline and timeline parsing.
  The visual layer is validated by looking at it.

## First session plan

1. Answer §1 and §2 with the user.
2. Build the analysis tool against one chosen song; inspect the timeline
   visually (a plot of the self-similarity matrix and latent axes) *before*
   building any renderer. If the timeline does not look structured, nothing
   downstream will save it.
3. Scaffold the web app; render the latent axes as a debug overlay.
4. Only then, the terrarium.
