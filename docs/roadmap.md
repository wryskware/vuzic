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

1. **Velocity dynamics bug.** ✅ **Done** — the `agility` split into
   independent **speed** and **drag** macros landed (`f1b3f70`), and the
   user confirmed 2026-08-15 that velocity dynamics are reasonably
   solved.
2. **Impulse wiggle** ✅ **Done** (`e792be4`). Per-stem impulse events
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
3. **Brightness dynamic range — in progress 2026-08-15, in a worktree so
   main and the branch can be A/B compared.** Design direction settled
   with the user 2026-08-15:
   - The brightness curve is **perceptually non-linear** (power-curve
     shaped), and the exponent is plausibly a runtime knob.
   - The feel: modest variation around the SDR range for the bulk of
     the population, with **peak velocity spiking into HDR headroom**.
     On a typical 1000-nit HDR screen, most particles live in a
     standard brightness band and peak levels are reserved for accents.
   - On an SDR screen everything maps much closer together — still some
     peak, but it may have to work differently (push lightness toward
     white, or simply read less). Expect iteration here.
   - Possibly an explicit canvas SDR/HDR toggle, if auto-detection
     isn't enough to tune both renditions.

   Original framing: the canvas is HDR now, but a species is
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
4. **Persistence bug class.** ✅ **Done** for its first two rungs
   (`e792be4` — see item 5, which is the remaining rung and is in
   progress as of 2026-08-15). Original framing: arc-mode settings for
   accents do not
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

6. **Save profiles v0 — do this one first.** Priority raised 2026-08-14 at
   the user's direction: there was no save location that survived another
   tab. Everything tunable lands in one autosave key per sim, written by a
   debounce *and* on `pagehide`/`visibilitychange`, so any second tab
   holding stale config overwrote a live tuning session by being closed.
   Two changes, both landed:
   - The hide flush is gone (`web/src/ui/autosave.ts`). Losing it costs at
     most the last 400 ms before a reload, and `maxWaitMs` bounds a dirty
     run at 2 s anyway; keeping it cost other tabs' work.
   - A named profile library (`web/src/ui/profiles.ts`, workbench ▸ data),
     written **only** from its buttons. One localStorage key per profile —
     no index blob, because an index is a read-modify-write and that is the
     same clobber one level up.

   A profile is a whole `ExportRecipe`, not a `modulation.json`: the plife
   matrix comes from the **seed** and the authored θ centre is in no mapping
   file, so a mapping-only profile restores your colours around a different
   world. Loading stages the recipe and reloads, which reuses `main.ts`'s
   one construction order rather than growing a second one.

   Deliberately a subset of `docs/handoffs/preset-strings-v1.md`, which
   still owns compact strings, `#p=` links and the trimmed `PresetV1`
   container. Storage is localStorage, so it is per origin and dies with
   "clear site data" — hence export/import to file. A durable backend (the
   File System Access API, or `terrarium-server`) fits behind the same four
   functions later.

## Phase 2 — Tuning tools

These ship in the published app (same build), so they finish before the
cut. Reordered 2026-08-15 at the user's direction: the explorer rebuild
dropped from first to last — it is lower priority for the user than the
palette pass, seed favorites, and the README, and can run as a
background thread.

3. **Palette pass.** Recover the palette the initial version used (git
   history); curate ~6–8 more adapted from strong perceptual
   collections, tuned in arc mode. Add per-species color preview boxes
   when adjusting the palette in arc mode. Sequenced after the
   brightness work — curating palettes against a moving brightness
   model is wasted work.
4. **Seed favorites v0.** Like/dislike buttons in the workbench that
   persist seed + context. Deliberately sequenced late so the parameter
   space is mostly settled and the collected data stays meaningful. The
   eventual "learn what good interaction matrices look like" model is a
   far-future consumer of this data — collect now, schedule never
   (until it earns itself).
5. **README + dev instructions** (added 2026-08-15). Polished,
   human-first, no thesis: a stranger clones the repo, gets the web app
   running, processes their own song locally through the analysis
   pipeline / `terrarium-server`, and can find their way around well
   enough to fork or contribute a feature. Docs-only, so it can run in
   parallel with anything; slotted near the cut so it documents reality
   rather than chasing it.
6. **Explorer rebuild.** The current explorer is essentially useless: it
   simulates a different parameter set, looks nothing like main mode,
   and ignores modulation. Contract: explore **from the current
   main-mode state** — identical parameters, identical look, modulation
   live — browsing seeded variations, with two scopes (**everything** /
   **matrix-only**), then adopt-or-cancel back into main mode. Known to
   be real lifting; it is the item allowed to slip past the cut.
   Handoff: `docs/handoffs/explorer-rebuild.md`.

## Phase 3 — The cut → publish

7. **Keybindings.** Pulled forward 2026-08-15 into its own short
   mini-thread (worktree, can run in parallel with Phase 1 work).
   `s` toggles the settings panel, `t` toggles the timeline + top menu,
   double-click toggles fullscreen. Handoff:
   `docs/handoffs/keybindings.md`.
8. **Preset strings v1** (moved into the deploy phase 2026-08-15). The
   compact-string codec, `#p=` fragment links, copy string / copy link.
   Its gates (palette v2, recipe v5) are merged and save profiles v0
   landed the apply path it reuses. Shareable links are arguably *the*
   friends-and-family feature. Brief: `docs/handoffs/preset-strings-v1.md`.
9. **Track publish allowlist.** Catalog entries are default-private; the
   demo build ships only tracks explicitly marked publishable. Allowlist,
   not blacklist — a track under test can never leak by omission. (All
   current music is rights-cleared; the gate is belt-and-suspenders.)
10. **Static deploy.** S3 bucket (AWS account exists, CLI authenticated);
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
