# Latent Music Terrarium — Agent Orientation

**State: placeholder repo.** No code, no build, no dependencies. The first real
task in this repo is choosing a stack and scaffolding it.

`docs/handoff.md` is historical context, not authority — its "non-negotiables"
overstated the user's intent and were revised 2026-08-06. Where handoff.md and
this file or `docs/plan.md` disagree, this file and plan.md win.

## What this is

Music → model-derived latent timeline → GPU-driven evolving world, in the
browser. Offline analysis, real-time simulation. Continuity and memory matter
more than reactivity.

## Priorities (revised 2026-08-06)

- **The deciding factor is that the output looks good.** Not reproducibility,
  not analytical fidelity to musical structure. Constantly changing output is
  fine.
- **Obvious musical events must produce obvious visual effects** — a new
  instrument entering, a drop, a section change. This is why the analysis uses
  music-understanding models and stem separation rather than an FFT, and it is
  the standard to judge the mapping by.
- The simulation is **stateful and continuous** — it evolves, it is not a pure
  function of the current frame's features.
- Recurrence ("chorus A ≈ chorus A′ visually") is a nice-to-have, not the
  centerpiece. Don't design around it; don't delete the cheap machinery that
  enables it.
- **Deterministic given (track, seed, device) — as tooling, not product.**
  Seeds are random per run by default and pinnable in the workbench; determinism
  exists so the tuning loop (scrub, tweak, replay the same bars) works.
- Human artistic direction is a core part of the project. The workbench (live
  parameter panel, scrub + replay, preset editing as data) is a first-class
  deliverable, not a debug leftover.
- WebGPU, large and fluid. Compute-shader-driven, not CPU particle loops.

## Deliberately out of scope for v1

In-browser analysis, audience-facing interactive controls, video export, sharing
links. All are plausible later; none are v1. Keep the seams clean enough that
they remain possible, but do not build for them. (The author-facing workbench is
*not* in this list — it is in scope and load-bearing. Multiple simulations left
this list in plan.md Revision 5: physarum and particle life both ship, selected
by `?sim=`.)

**Multiple songs and user-uploaded music left this list on 2026-08-09**, at the
user's direction. There is a track catalog and a picker in the play tab, a
second real track (`pink-loop`), and `terrarium-server` — the offline pipeline
behind a local HTTP API, so a wav dropped into the panel comes back as a track.
The distinction the original line was protecting still holds: analysis is still
offline and still runs the same pipeline, it just no longer requires a terminal.
The server is optional, the browser probes for it once, and everything works
without it.

## Boundaries

This is an independent repo in the Wryskware ecosystem. It is **not** part of a
monorepo. It may depend on `@wryskware/brand` for identity in its chrome, but
the simulation itself should look like itself, not like the portfolio.

Do not add a portfolio page here — that lives in `wryskware-site`.

## Before writing code

`docs/plan.md` is the current implementation plan: analysis stack, timeline
format, simulation substrate, phasing. It is backed by the survey in
`docs/research/` (audio analysis, simulation candidates). `docs/scaffolding-notes.md`
records the original open questions and the reasoning behind them.

The decisions in `plan.md` are settled. Do not silently reverse one — if the
evidence turns out to contradict a decision, say so and let the user choose.

## Indexed retrieval

Route by what kind of question it is:

- **Structure, once a symbol is known** (callers, callees, dependencies,
  impact, execution paths) → `codebase-memory` graph tools. These MCP tools
  are deferred: batch-load the ones you need in a single ToolSearch call at
  first need.
- **Everything else** — exact strings, known paths, docs and plan questions,
  generated timeline fields, shader bindings, query parameters, dynamic
  references → direct Grep/Glob/Read.
- **Project history** ("what did we decide", "why is it this way") → git log
  and the auto-memory index.

Read the exact source before changing it; source is authoritative when the
index is stale or incomplete. Avoid duplicate search layers and stop when the
evidence is sufficient. The index is local; never send repository source to
the remote `claude.ai Era Context` server.
