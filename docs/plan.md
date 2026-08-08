# Implementation Plan — v1

Resolves the open decisions in `docs/scaffolding-notes.md` against the evidence
in `docs/research/`. Revised 2026-08-06 after review with the user.

**Priority revision, recorded up front:** the "non-negotiables" in the original
handoff brief overstated the user's actual intent. The deciding factor for every
choice below is **that the output looks good** — not that it is reproducible,
and not that it mirrors musical structure with analytical fidelity. Constantly
changing output is fine. What must be true is that obvious musical events
produce obvious visual effects: a new instrument entering, a drop, a section
change. Recurrence ("chorus A ≈ chorus A′ visually") is a nice-to-have, not the
centerpiece. Determinism survives, but as a tuning tool, not a product feature.

## The shape of the thing

```
  song.wav
     │
     ├── analysis/  (Python, offline, run once per track)
     │      beat grid ─── structure ─── stems ─── character
     │                                               │
     │                    timeline.json + timeline.bin
     │                                               │
     └── web/  (TypeScript + WebGPU, real time) ─────┘
              audio clock → timeline sampler → mapping layer → simulation
                                                    │              │
                                              workbench    persistent GPU state
```

The contract between the two halves is the timeline format, and nothing else.
That seam is what lets the simulation be swapped later without touching
analysis.

## Decision 1 — where the latent representation comes from

**Hybrid: four primary streams plus one optional.** Each from the tool that is
actually best at it:

| Stream | Source | Rate | Purpose |
|---|---|---|---|
| Beat grid | **Beat This!** (MIT) | events | master clock; everything else is beat-synchronous |
| Structure | **All-In-One** (MIT) | events + 100 fps activations | boundaries, functional labels, graded "chorus-ness" |
| **Stems** | **demucs** (already inside All-In-One) | 10 Hz | per-stem activity: bass / drums / vocals / other — **the "instrument comes in" signal** |
| Character | **MuQ** layer ~6 | 10 Hz | what this moment feels like |
| Recurrence *(optional)* | custom, librosa + libfmp | per-beat | "you are re-living time t₀" — kept because it's ~200 lines, no longer load-bearing |

Notes on each:

- **Stems are promoted to a first-class stream.** For the effects the project
  actually wants — an instrument entering, texture thickening, a drop — four
  per-stem activity curves do more visible work than any other signal in the
  file. All-In-One already runs demucs internally; computing smoothed per-stem
  RMS/onset-density curves from those stems is nearly free. These are also the
  default drivers for species populations (Decision 3).
- **Beat This!** beats override All-In-One's where they disagree — measurably
  more robust (GTZAN beat F1 0.890) and actively maintained.
- **All-In-One's 100 fps activation curves** matter more than its labels:
  continuous graded confidence is what a simulation wants; a hard label is a
  step function and steps look like glitches.
- **MuQ layer selection is not a detail.** Published per-layer probing says
  layer ~6 for genre/emotion. Do *not* assume the last layer is best; it isn't,
  for anything.
- **Recurrence is demoted, not deleted.** The 2026 SSM study (arXiv 2603.27218)
  still justifies computing it from chroma rather than embeddings if computed at
  all. It ships as optional extra channels; nothing downstream depends on it.

**License: settled.** MuQ weights are CC-BY-NC-4.0, accepted — this is an art
project. Escape hatch if that changes: **MusicFM (MIT)**, a config swap provided
extraction code doesn't hard-code embedding dimensions.

## Decision 2 — timeline format

JSON manifest plus a Float32 binary sidecar: JSON stays diffable, dense curves
stay compact.

```jsonc
// timeline.json
{
  "version": 2,
  "track":   { "id": "…", "duration": 214.7, "sampleRate": 48000 },

  "grid":    { "hopSeconds": 0.1, "frames": 2147 },

  // stratum 1 — sparse rhythm
  "beats":     [0.482, 0.961, …],
  "downbeats": [0.482, 2.398, …],
  "tempo":     125.3,

  // stratum 2 — sparse structure
  "segments": [
    { "start": 0.0,  "end": 31.2, "label": "intro",  "confidence": 0.83 },
    { "start": 31.2, "end": 62.4, "label": "chorus", "confidence": 0.91 }
  ],
  "segmentSimilarity": [[1.0, 0.12, …], …],   // optional; kept ungraded if present

  // stratum 3 — dense curves, layout of the .bin
  "channels": [
    { "name": "stems",      "dims": 4,  "offset": 0  },  // bass, drums, vocals, other
    { "name": "latent",     "dims": 64, "offset": 4  },  // MuQ L6, standardized + PCA-64
    { "name": "novelty4",   "dims": 1,  "offset": 68 },
    { "name": "novelty16",  "dims": 1,  "offset": 69 },
    { "name": "actChorus",  "dims": 1,  "offset": 70 },  // All-In-One activations…
    { "name": "recurTime",  "dims": 1,  "offset": 71 },  // optional recurrence pair
    { "name": "recurStr",   "dims": 1,  "offset": 72 }
  ]
}
```

`timeline.bin` is a flat `Float32Array` of `frames × totalDims`, row-major —
one contiguous upload straight into a GPU buffer.

Commitments, revised:

1. **Do not compress the latent to fit the mapping layer.** The mapping layer
   decides how much input it wants; the timeline's job is to not destroy
   information. Store 64 PCA dims (a 4-minute track at 10 Hz is ~600 KB — size
   is a non-issue), and optionally emit the raw pooled 1024-dim layer-6
   embedding as a second sidecar (`embedding.bin`, ~10 MB) for future learned
   mappings. PCA's role shrinks to a compact default input and a debug
   visualization, **not** the contract.
2. **Stems are the highest-value channel in the file** for the revised goal.
   Smooth them (EMA, ~beat scale) in analysis so the runtime never has to.
3. `segmentSimilarity` / `recurTime` / `recurStr` are optional. Emit them when
   cheap; design nothing around them.

## Decision 3 — the simulation

**Settled: multi-species physarum with per-species trail fields, a
cross-species interaction matrix, and a slow soil layer.**

The argument for physarum as the substrate is unchanged: this project sweeps
parameters continuously for four minutes without supervision, and physarum
essentially cannot die and cannot explode. It also carries memory at two
timescales for free (trail EMA, network topology hysteresis). The user has
built both physarum and particle-life simulations before, which removes most
of the implementation risk from this phase.

The structural upgrade from the review: **K species, not one**, borrowing
Particle Life's best idea — the interaction matrix — and applying it to trail
fields rather than particle pairs.

```
K species (config, default 4, keyed to stems; NOT bound to any texture format)

per species k:
  trail[k]: independent r32f texture       (own decay rate per species)
  agents:   pos, heading, species id       (single pool, fixed max size)
  params:   sensor dist/angle, rotation, step, deposit, color, aliveFraction
            (sensor/rotation/step as intensity-adaptive curves p1 + p2·x^p3)

M: K×K sense-weight matrix
  species i senses  Σⱼ M[i][j] · trail[j]  at each of its 3 sample points
  M[i][j] > 0 attract, < 0 avoid, 0 ignore — any matrix produces some
  behaviour, none produce NaN (Particle Life's robustness property, inherited)

soil: r32f texture, decay ~0.999 (track-scale memory), biases deposit/sense gain

per tick:
  sense (weighted sum over K trails) → rotate → step
  deposit (i32 fixed-point atomicAdd into per-species buffer)
  resolve i32 → r32f textures; diffuse 3×3 + per-species decay
```

Design points:

- **Independent 1-channel textures, blended only at render time.** Species
  count is a config value, not a consequence of RGBA packing. Each species has
  its own color, intensity, and blend contribution in the compositor; adding a
  species is a loop bound, not a rearchitecture. (Deposit atomics live in i32
  storage buffers per species — WGSL has no texture atomics — with a cheap
  resolve pass to r32f so sensing keeps free bilinear filtering.)
- **Species ↔ stems is the default keying.** Bass species: large sensor
  distance, wide diffusion, slow decay → broad, slow, large-scale structure.
  Drums: short-range, high deposit, fast decay → percussive shimmer. Vocals:
  distinct color, and the matrix pulls other species toward its trail during
  vocal sections. Stem activity drives each species' population and deposit
  strength — **an instrument entering is a population blooming**, an effect
  produced by mechanism rather than parameter twiddling.
- **Population via alive-fraction on a fixed pool.** One max-size agent buffer;
  each species owns a slice; agents beyond the alive fraction park as dormant.
  Particle count and effective species count (population → 0) are runtime
  controls with no reallocation.
- **Soil layer** — trail gives bar-scale memory, soil gives track-scale memory.
  Cheap, and it is what makes minute three look like it remembers minute one.
- **Advected per-agent parameter field** (Flow-Lenia's idea) — still good, now
  a phase-6 enhancement rather than a core commitment. Per-species parameters
  plus the interaction matrix already provide spatial heterogeneity; add the
  advected field if the world still feels globally uniform.

Flow-Lenia remains on the table as a second interpretation of the same
timeline (v2, not v1). Keeping the mapping layer separate from the substrate is
what preserves that option.

## Decision 4 — the mapping layer and the workbench

This is where projects like this usually fail. Two named modules.

### Mapping: preset simplex, understood as a tiny NN you author by hand

k-means the track's own latent timeline into M ≈ 6–10 anchors; tune one full
parameter preset per anchor; at runtime `z → w = softmax(−‖z − c_m‖² / T)`,
`θ = Σ w_m θ_m`. This guarantees parameters stay inside the convex hull of
known-good values, and it is structurally an RBF network whose training data is
the presets — the human supplies the "looks good" signal directly, which is the
signal no off-the-shelf loss function can provide.

The preset vector θ now includes the **interaction matrix M**, per-species
brightness, deposit strengths, and alive-fractions — not just kernel
parameters. Colors are NOT in θ (Revision 2): they are a static per-species
palette outside the mapping.
Stem channels bypass the simplex and drive their species directly (population,
deposit); the simplex handles character.

**The path to a learned mapping runs through this, not around it.** Every
workbench tuning session produces (latent, θ, kept/discarded) triples. Once a
few hundred exist, distilling them into an MLP with wide input — raw 1024-dim
embedding + stems + novelty, no PCA bottleneck — is a small job, and the wide
input is why the timeline stores more than the simplex needs (Decision 2).
Listed under "Later"; do not build first.

Mechanical rules, unchanged from the research:

- **Slew-limit per parameter, by timescale.** Fast params (deposit, sensor
  angle, color) track 10 Hz; slow ones (decay, alive-fractions, matrix M)
  track section scale. Parameter motion faster than the sim's relaxation time
  produces mush.
- **Boundaries are steps, not ramps** — section changes step slow params,
  inject matter, re-seed.
- **Deterministic re-seeding keyed on (seed, section_id, agent_index)** — so a
  returning section injects at the same places and the difference is
  accumulated state. Costs a hash function; keep it.
- **Recurrence pointer blending** — only if the optional channels exist, and
  only after phase 6 works without it.

### Workbench: the art-direction instrument (explicit deliverable)

A huge part of this project is human artistic direction. The workbench is where
it happens, and it is dev tooling, not a viewer feature — the brief's "no
interactive controls" applies to the audience, not the author.

- Live panel over every mapping knob: per-species params, the K×K matrix,
  slew rates, simplex temperature, soil decay. (tweakpane or similar.)
- **Timeline scrub + deterministic replay**: jump to 1:32 where the bass drops,
  tweak, replay the same bars, compare. This loop is the reason determinism
  survives the priority revision.
- Presets and anchors saved/loaded as JSON. The whole mapping is **data, not
  code** — art direction is editing that file live.
- A/B snapshots (run twice from the same tick with two θ sets), per-section
  overrides.
- Every kept/discarded tune is logged — the future NN's training set.

The phase-4 debug sliders grow into this instead of being thrown away.

## Decision 5 — determinism and seeds

Determinism is retained **as tooling**, not as a product promise. Its job is to
make the tuning loop possible (scrub, tweak, replay the identical segment) and
to make bugs reproducible.

- **Seed is a run input, random by default.** Every run draws a fresh seed;
  the workbench displays it and can pin it. Pinned seed + same track + same
  device ⇒ identical run. Unpinned, every run is a new world — which is a
  feature, not a defect.
- Cross-GPU bit-exactness is not achievable (WGSL transcendental ULP bounds,
  FMA contraction, f32 non-associativity) and is not promised. Same-device
  determinism is, and is cheap:
  - Fixed simulation timestep decoupled from rAF; accumulate time, run N
    integer steps.
  - **Index the timeline by sim-tick, not wall clock.**
  - **i32 fixed-point atomicAdd** (scale ~1e7) for trail deposit — integer
    addition is associative, so the scatter is order-independent.
  - PCG hash seeded on `(seed, sim_tick, agent_index)`.

## Decision 6 — audio sync

**Web Audio API.** `AudioContext.currentTime` against a scheduled
`AudioBufferSourceNode` gives a sample-accurate clock; that clock drives
sim-tick count. Convert source audio to **WAV before analysis** — MP3 decoder
offsets of 20–40 ms are a known All-In-One quirk.

## Stack

- Vite, TypeScript strict, Node 22+, npm, own lockfile.
- Raw WGSL files via a Vite import plugin. No shader DSL.
- **Direct WebGPU API, no wrapper.** (three.js WebGPURenderer is the fallback
  if rendering, not simulation, becomes the bottleneck.)
- Python analysis in `analysis/`, `uv` or a pinned venv.
- Tests where they earn their place: timeline parsing, the mapping layer's
  convex-hull guarantee. The visual layer is validated by looking at it.

### Windows gotcha — resolved 2026-08-06, easier than budgeted

The NATTEN problem no longer exists. `all-in-one-fix` is dead on PyPI (files
removed); its successor **`all-in-one-infer` (v3.1.0, imports as
`allin1_infer`)** reimplements neighborhood attention in pure PyTorch and uses
`madmom-infer`/`demucs-infer`, so it has no NATTEN dependency and no torch
upper bound. Verified working environment (WSL2 Ubuntu, RTX 5090, all stages
GPU): Python 3.10, torch 2.8.0+cu128, all-in-one-infer 3.1.0, beat-this 1.1.0
(now on PyPI), muq 0.1.0. Full versions in `analysis/README.md`. Analysis runs
in WSL2 on the ext4 disk (`~/lmt/analysis`); the Windows repo stays source of
truth and outputs are copied back to `data/timelines/<track>/`.

One determinism trap, found and fixed: demucs `apply_model(shifts=1)` draws
its time offset from the unseeded stdlib `random`, and All-In-One's activations
inherit that noise. The structure stage seeds everything before `analyze()`;
same-seed runs are now bit-identical.

## Phases

Ordered so each phase produces something inspectable.

**Phase 0 — environment and song.** WSL2 analysis env, pick a track with
obvious structure and clearly separable stems.

**Phase 1 — analysis pipeline, validated by eye.** Build the streams, emit
`timeline.json` + `.bin`. Plot everything before writing any renderer: stem
activity curves against the waveform (do instrument entrances show as clean
steps?), novelty curves, the latent trajectory colored by segment. Stems are
the first thing to check now — if bass entering doesn't produce an obvious
step in the bass channel, fix it here, because nothing downstream can.

**Phase 2 — timeline contract in the browser.** Vite app, loader,
`Float32Array` sampler with interpolation. Debug overlay: scrolling plot of
stems, latent axes, boundaries. No simulation.

**Phase 3 — audio clock and sync.** Web Audio scheduled source, sim-tick
accumulator, timeline indexed by tick. Validate boundary markers against
audible section changes.

**Phase 4 — multi-species physarum core.** Agent pool with species slices,
K independent trail textures, i32 deposit + resolve, per-species diffuse/decay,
the K×K sense matrix, alive-fractions, render compositor with per-species
color. Fixed timestep, PCG, seed pinning. Driven by sliders, not music. The
user has built physarum and particle-life before — this phase is execution,
not research. Target 1M+ agents.

**Phase 5 — mapping + workbench.** Intensity-adaptive curves, k-means anchors,
softmax blending, slew limits; stems wired directly to species populations.
The debug sliders grow into the workbench: preset save/load, scrub + replay,
A/B, tuning log. **This is where the taste goes**, and it will take longer
than phase 4.

**Phase 6 — memory and events.** Soil layer, section-boundary step events,
deterministic re-seeding. Then the test that matters under the revised
priorities: **does an instrument entering read as an event? Does a drop land?**
Recurrence blending and the advected parameter field go here too, if their
optional channels exist and the world wants them.

**Phase 7 — rendering and art direction.** Color grading, accumulation, bloom,
compositing of the K trail layers. Jenson's advantage was substantially the
renderer. Budget real time here.

Phases 1–3 are substrate-agnostic.

## Revision 2 — 2026-08-06, after first run against the real track

Three user decisions from watching the phase 1–5 build run on Free Fall:

1. **Static color palette.** Blending colors per anchor was a mistake: it
   muddies the image and makes species impossible to track. Colors are now a
   single static per-species palette (art-directed once, per track at most),
   removed from the blended θ. **Brightness/exposure per species stays
   modulatable** (fast slew class) — light responds, hue does not.
2. **Transient event stream — beat-to-beat reactivity without runtime FFT.**
   The timeline gains a sparse `events` array: `{ t, kind, strength }`, kinds
   `kick | snare | hat | bass | vocal`, detected offline from the demucs stems
   (band-split onset detection on drums: <120 Hz kick, mid snare/clap, >4 kHz
   hats; note onsets on bass; onsets on vocals). Sample-accurate, deterministic,
   source-labeled — strictly better than FFT buckets, which remain rejected.
   The sim consumes events as impulse envelopes: deposit bursts, brightness
   flashes, radial splashes, keyed deterministically on (seed, eventIndex).
   The event→response mapping is workbench data, per kind.
3. **Reactivity depth is a tuning target.** The first build read as subtle
   apart from color; defaults should bias toward legible modulation, and the
   remaining phases (events, soil, rendering) are expected to carry most of
   the visible reactivity.

## Revision 3 — 2026-08-07: no scenes. Continuous modulation.

The preset-simplex / anchor model (Decision 4) is **rejected by the user** after
hands-on use. Verbatim intent: "I do not care about scenes. I want real time
reactivity and morphing parameters… we should basically be taking the whole
embedding and using that somehow to drive species params in real time." Tuning
per-anchor presets is exactly the workload the user does not want.

Replacement: **seeded random projection modulation.**

- Each modulatable parameter p has a random unit direction w_p in embedding
  space, keyed on (seed, paramIndex). Per tick:
  `p = clamp(base_p + halfRange_p · tanh(depth · (w_p · ẑ)), lo, hi)`.
  Bounded by construction; continuously morphing at timeline rate. Safe because
  physarum tolerates arbitrary in-range sweeps (the substrate was chosen for
  exactly this).
- **base_p is seeded too** — jittered within safe bounds around curated
  defaults. A new seed is a new personality AND new wiring; the workflow is
  reroll-and-pin, not tune-and-capture. (Direct user ask: "species parameters
  should be randomized too.")
- Input ẑ: the raw pooled 1024-dim MuQ embedding (embedding.bin), per-dim
  z-scored per track at load; fallback to the PCA-64 latent channel when the
  raw file is absent. The PCA intermediate is no longer load-bearing.
- Slew stays (per-class response speeds), depths default LEGIBLE. The impulse
  lane, stems→population drive, soil, static palette, and render chain are
  unchanged and compose after modulation as before.
- Anchors, k-means, simplex, solo/capture are deleted. Boundary partial
  reseeds survive as an optional event (they are events, not scenes). The
  mapping file becomes a ModulationConfig (depths, speeds, palette, render);
  v2 files load with anchors discarded and a warning.
- The distilled-NN idea in "Later" survives — random projection is the
  zero-training baseline of exactly that mapping, and reroll choices are
  themselves preference data.

## Revision 4 — 2026-08-07: the driver bank

Refines Revision 3 after use. Raw-1024 random projections react to everything
and isolate nothing; and nobody can tune 1024 weights.

- **Driver bank, ~16 named signals, each with a live meter in the workbench:**
  novelty4, novelty16, actChorus (structure channels, previously unused), plus
  the top ~13 PCA components of the latent — variance-reordered at load
  (post-PCA smoothing broke variance ordering; fix it at load), z-scored.
  Modulation directions are seeded random vectors in *driver* space.
- **The tuning surface is per-driver gain sliders** (mute the unimportant,
  boost the important) + the existing group depths + reroll. Not weights.
- **Brightness is out of embedding modulation** (it flashed constantly and
  looked bad — user call). Per-species brightness = base × stem-follow
  (its own stem's smoothed activity, with a floor and depth/curve controls,
  so an instrument cutting out visibly dims its species) × impulse flashes.
- CLAP text-anchor axes (Later) slot in as additional named drivers when
  built — the bank is the seam for them.

## Later, and deliberately not now

Additive to the timeline: SongFormer labels, CLAP text-anchor axes,
DEAM valence/arousal probe, chord/key tracking, richer recurrence machinery.

Requires new work but the seams exist: **distilled NN mapping trained on
workbench logs (wide input: raw embedding + stems, no PCA bottleneck)**,
multiple simulations reading the same timeline, multiple songs, video export,
sharing links.

Out of scope: in-browser analysis, user uploads, audience-facing interactive
controls.

## What is genuinely novel here

Nothing found in the survey drives a **real-time stateful simulation from
music-understanding embeddings**. Existing work splits into FFT visualisers,
latent interpolation of generative output (Anadol), and codec-latent terrain
browsers. The one audio-reactive physarum found maps amplitude to sensor
distance. The stem-driven multi-species version with a cross-species
interaction matrix appears to have no prior art at all. The pieces are all
individually well-attested; the assembly is not — which also means there is no
reference implementation to check against when it looks wrong.
