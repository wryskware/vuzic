# Roadmap — to the friends-and-family demo

Settled 2026-08-13 in a grilling session with the user. This is the current
sequencing authority for feature work. The milestone that ends this roadmap:
**the terrarium live at a wryskware.dev subdomain as a friends-and-family
demo / portfolio piece.** No wider-audience polish beyond that; after the
milestone, re-plan.

Particle life is the product. Physarum was a good test but is less
impressive; vizfx is low priority. Both stay reachable but get no roadmap
work.

## Phase 1 — Core feel

The core-quality gap: the morphing is great but on a slow track; the sim
does not yet *feel reactive* to impulses, and velocity dynamics are flat.

1. **Velocity dynamics bug.** All particles appear to move at the same
   speed all the time — no dynamic change in velocity. Diagnose first
   (suspect: speeds pinned at max because force sits far above the
   friction equilibrium; the `agility` macro adjusting max speed and
   friction together is the design smell). Regardless of root cause,
   split `agility` into two independent macros: **speed** and **drag**.
2. **Impulse wiggle** (the critical feature). Per-stem impulse events
   perturb the **primary species'** (0–3) interaction-matrix rows —
   the vuzic pattern (per-species row vector), adapted to timeline
   impulses. Design decisions, settled:
   - Signal: existing per-stem timeline impulses. No sub-band analysis
     work unless per-stem visibly fails.
   - Target: primaries only (accents already carry per-stem population
     response). Routed through the modulation system so retargeting or
     adding targets (speed, radius, friction) later is a config edit,
     not a feature.
   - Direction vectors: seeded random, with a workbench reroll button.
   - Depth: a new macro knob, 0 (off) → large.
   - Cost is trivial: the perturbed 8×8 matrix re-upload is 256 bytes
     per frame.
3. **Brightness dynamic range.** The canvas is HDR now, but a species is
   basically always one uniform brightness — there is little mechanism
   for dynamic range among the particles. Starter direction from the
   user: dimmer baseline, higher peak brightness on impulses. Existing
   machinery (impulse `flash`, per-species `brightFollow`) is
   species-uniform and clearly not enough. Candidate mechanisms, to be
   chosen during implementation, not settled:
   - Treat HDR headroom as a budget: baseline sits low (~SDR range),
     impulse flash peaks into the headroom so hits read as *light*.
   - Per-particle luminance variation from state already on hand:
     speed (pairs with the velocity fix), local density, or a
     per-particle random gain so a species shimmers instead of being a
     flat sheet.
   - Transient-shaped flash envelopes (fast attack, slower decay,
     per event kind).
4. **Persistence bug class.** Arc-mode settings for accents do not
   survive a refresh. This is the latest instance of a recurring bug
   shape: a new setting gets added but not wired into the
   serialize/restore round-trip, and it surfaces later as "my tweak
   didn't save." Fix the accent-arc instance, but also attack the
   class: audit the persist paths (serialize/apply pairs likely keep
   manual field lists) and add a round-trip test that constructs a
   fully non-default config, serializes, restores, and asserts
   equality — so any future field added to config but missed in
   persistence fails CI instead of costing a bug report.
5. **Automatic block registration — highest priority bug fix in this
   phase.** Item 4's first two rungs are closed: fields round trip
   because the reader walks the destination's own keys rather than a
   written list, and widgets save because a panel can only be handed a
   `PersistedContainer` and the compiler rejects anything else. The rung
   above them is still open and is the same defect one level up: adding
   a whole new **config block** persists only if somebody remembers to
   name it in `PlifeSim.extrasBlocks()` (and its defaults in
   `defaultExtrasBlocks`, and its clamps in `extrasRules`). Nothing
   fails if they do not — the block simply never saves, and the report
   arrives weeks later as "my tweak didn't save".

   **We cannot keep relying on an agent, or a person, to register a
   block by hand.** Declaring a block must *be* registering it:
   construction enrols it in serialize / restore / clamp, and the only
   way out is an explicit opt-out passed at the declaration site (for
   the genuinely session-only state — `effectiveBudget`, the impulse
   `hold` — which must then say so where it is declared rather than by
   being quietly absent from a list). Shape it however proves cleanest —
   a block registry the config declares itself into, a decorator on the
   defaults function, whatever — but the acceptance test is fixed: add a
   new block, save nothing else, and it round trips; opt it out, and CI
   says the opt-out is deliberate. Until that lands, every new block is
   another instance of a bug we have now paid for four times (accent
   arcs, the whole impulse lane, the palette and render folders' optional
   `onChange`, and this).

## Phase 2 — Tuning tools

These ship in the published app (same build), so they finish before the
cut.

3. **Explorer rebuild.** The current explorer is essentially useless: it
   simulates a different parameter set, looks nothing like main mode,
   and ignores modulation. Contract: explore **from the current
   main-mode state** — identical parameters, identical look, modulation
   live — browsing seeded variations, with two scopes (**everything** /
   **matrix-only**), then adopt-or-cancel back into main mode. Known to
   be real lifting; if it balloons it is the first item allowed to slip
   past the cut, but the intent is to finish it before publishing.
4. **Palette pass.** Recover the palette the initial version used (git
   history); curate ~6–8 more adapted from strong perceptual
   collections, tuned in arc mode. Add per-species color preview boxes
   when adjusting the palette in arc mode.
5. **Seed favorites v0.** Like/dislike buttons in the workbench that
   persist seed + context. Deliberately sequenced after items 1–3 so the
   parameter space is mostly settled and the collected data stays
   meaningful. The eventual "learn what good interaction matrices look
   like" model is a far-future consumer of this data — collect now,
   schedule never (until it earns itself).

## Phase 3 — The cut → publish

6. **Keybindings.** `s` toggles the settings panel, `t` toggles the
   timeline + top menu, double-click toggles fullscreen.
7. **Track publish allowlist.** Catalog entries are default-private; the
   demo build ships only tracks explicitly marked publishable. Allowlist,
   not blacklist — a track under test can never leak by omission. (All
   current music is rights-cleared; the gate is belt-and-suspenders.)
8. **Static deploy.** S3 bucket (AWS account exists, CLI authenticated);
   DNS pointing to AWS handled when we get there — not a blocker to
   build against. Demo shape: static only, no terrarium-server, no
   uploads, no export UI. Plife front and center; physarum/vizfx still
   reachable via `?sim=` but unadvertised; workbench included.

## Post-cut backlog (acknowledged, not scheduled)

- Autopilot (slow drift through preset space) — wanted eventually,
  noncritical.
- Section-change events — low priority; depends on the favorites pool
  (most seeds are bad, so section changes need known-good seeds to draw
  from).
- Learned interaction-matrix model from favorites data.
- Export range selector; hero clip for the portfolio card.
- Physarum / vizfx improvements.
- Everything already listed as deferred in `docs/plan.md` and the
  handoffs (live mode, hosted rendering, preset short-links, richer
  analysis, …).
