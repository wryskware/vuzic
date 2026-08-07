# Latent Music Terrarium — Agent Orientation

**State: placeholder repo.** No code, no build, no dependencies. The first real
task in this repo is choosing a stack and scaffolding it. Read
`docs/handoff.md` first — it is the source of intent and should not be
contradicted silently.

## What this is

Music → model-derived latent timeline → GPU-driven evolving world, in the
browser. Offline analysis, real-time simulation. Continuity and memory matter
more than reactivity.

## Non-negotiables from the brief

- Visuals must reflect **musical structure**, not instantaneous audio levels.
  Anything that would work identically driven by an FFT magnitude has missed the
  point.
- Repeated musical ideas must produce **recognizably related** visual states.
  This implies the simulation carries state across time, and that similar latent
  vectors map to nearby visual configurations. Design for that from the start.
- The simulation is **stateful and continuous** — it evolves, it is not a pure
  function of the current frame's features.
- Deterministic given (track, seed). Reproducibility is a feature: it makes
  dreams shareable and exportable.
- WebGPU, large and fluid. Compute-shader-driven, not CPU particle loops.

## Deliberately out of scope for v1

Multiple songs, multiple simulations, in-browser analysis, user-uploaded music,
interactive controls, video export, sharing links. All are plausible later; none
are v1. Keep the seams clean enough that they remain possible, but do not build
for them.

## Boundaries

This is an independent repo in the Wryskware ecosystem. It is **not** part of a
monorepo. It may depend on `@wryskware/brand` for identity in its chrome, but
the simulation itself should look like itself, not like the portfolio.

Do not add a portfolio page here — that lives in `wryskware-site`.

## Before writing code

`docs/scaffolding-notes.md` lists the open decisions (analysis model, timeline
format, renderer structure). Resolve them with the user rather than picking
silently; they determine most of the architecture.
