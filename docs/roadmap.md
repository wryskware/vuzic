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
3. **Brightness dynamic range.** ✅ **Done** — A/B'd in a worktree,
   approved by the user and merged 2026-08-15 (`81cd072`..`70fb3a3`):
   per-particle luminance from `|v|/maxSpeed` (`plife/luma.ts`), a
   power curve anchored so gain at `mid` is exactly 1 (redistributes
   light, so auto-exposure can't cancel it), HDR headroom spent as a
   `log2(H)`-stop budget, `whitePeak` desaturation for the SDR
   rendition, a session-only SDR/HDR preview toggle, and a `luma`
   config block riding the registry. Carried recipe **v6** (the block
   lands in recipes; `liftV5toV6` fills it, and a latent
   version-stamping bug in `liftV4toV5` was fixed en route; the server
   gate accepts 3–6). Design direction that was settled beforehand:
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
   phase.** ✅ **Done** (`4da3de5`, 2026-08-15): `BlockTable<Config>` is
   exhaustive over the config's own object-valued keys, so declaring a
   block *is* registering it and forgetting is a compile error; opt-outs
   are explicit at the declaration site and CI-asserted. Physarum and
   vizfx migrated too. One manual list remains — `PLIFE_KEYS` in
   `runtime/recipe.ts` — but it fails loudly (recipe validation throws),
   a different bug class. Original framing: item 4's first two rungs are
   closed: fields round trip
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
4. **Seed favorites v0.** ✅ **Done** (`bbb22c7`..`d3e5206`).
   `web/src/ui/favorites.ts` is a library like `profiles.ts`, not a config
   block: one localStorage key per verdict (`lmt.fav.v1.<id>`, id sorts
   chronologically), never an index blob. 👍/👎 sit in the *world seed*
   folder, one row under the reroll button, because the loop is reroll →
   watch → judge; the pool's management (count, likes picker,
   return-to-world, JSONL export/import) is on the data tab.

   The one design decision worth carrying forward: **a like stores a whole
   `ExportRecipe` and a dislike stores none.** A like has to be
   *returnable*, and only whole state returns you to a world (the profiles
   argument, verbatim) — it goes back through `requestProfileApply`, so
   there is still exactly one apply path in the app. A dislike is never
   returned to and is the *common* verdict by construction, and the
   measured sizes are 14.4 kB against 367 B in a 5 MB origin budget shared
   with the profiles and the track cache. Both verdicts carry identical
   *model* context — seed, `matrixGen`, `speciesCount`, track, time —
   because `seedMatrixBase` is a pure function of those, so the matrix is
   re-derivable and storing it would be `explore/log.ts`'s rejected trade
   (8× the bytes for 0× the information). A like that will not fit drops
   its recipe and keeps the verdict.

   Original framing: like/dislike buttons in the workbench that
   persist seed + context. Deliberately sequenced late so the parameter
   space is mostly settled and the collected data stays meaningful. The
   eventual "learn what good interaction matrices look like" model is a
   far-future consumer of this data — collect now, schedule never
   (until it earns itself).
5. **README + dev instructions** ✅ **Done** (`822c0fd`, 2026-08-15).
   The root README is the human front door (run it, process your own
   song, repo tour, dev commands); `analysis/README.md` already carried
   the pipeline depth and the root now delegates to it, so no third doc
   was needed. Flagged open by it: no LICENSE is chosen ("all rights
   reserved" placeholder), and the default sim is still `physarum`
   while plife is the product — both are publish-cut decisions.
   Original framing: polished,
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
9. **Track publish allowlist.** ✅ **Done** (`796e8cb`, 2026-08-19).
   `data/publish.json` names the five tracks the demo ships, read only by
   `sync-data --publish`; the fallback (`synthetic`) always ships because
   `main.ts` falls back to it. Chosen by the user: bug-fix-rush,
   drifting-slow-remastered, free-fall, lost-in-space-1-3, pink-loop.

   The same commit closes a **second, unwritten leak of the same shape**,
   one level down. `sync-data` copied every file in a timeline directory
   except a *skip list* naming `embedding.*` and `plots/` — a blacklist,
   which is exactly what item 9 forbids for tracks and nobody had applied
   to files. `bug-fix-rush` was therefore shipping 116 MB of demucs stems
   (`demix/`) and 20 MB of spectrograms (`spec/`). It is now a wanted-list
   of the files the browser actually reads.

   And a **third gate nobody had written down at all: audio format.** A
   track's `audio.wav` is 30–52 MB, which is unremarkable over localhost
   and indefensible over CloudFront. Bundled audio is transcoded to 160k
   AAC in MP4 (what `decodeAudioData` supports without qualification;
   Opus would save more bytes and cost Safari). `terrarium-server` still
   serves its own `audio.wav`, so the name rides `TrackEntry.audioFile`
   rather than being a constant in four places.

   Net: `web/dist` 519 MB → 27 MB.
10. **Static deploy.** 🟡 **Infrastructure live, demo shape outstanding.**
   Standing up (2026-08-19): private S3 bucket `wryskware-terrarium-site`
   behind CloudFront distribution `E2ISYNUAG82OYT` with origin access
   control — no public bucket, no website endpoint — served at
   `https://dwjp9zs5pyebf.cloudfront.net/`. `tools/deploy.sh` builds the
   publish cut and pushes it in three cache classes, content-first and
   document-last, then invalidates. `DEFAULT_SIM` is now `plife`.

   Deliberately **no custom domain yet**: the subdomain is undecided
   (`dreams.wryskware.dev` versus a `vuzic.app` one), and a distribution
   serves its own name meanwhile, so the choice was never a blocker.
   Worth knowing when deciding: `vuzic.app` already has a Route 53 hosted
   zone and needs no registrar work, while `wryskware.dev` is registered
   at Spaceship on their parking nameservers and needs an NS change there.
   Attaching either is a cert in us-east-1, an alias, and an alias record.

   **Still open — the demo shape.** "Static only, no terrarium-server, no
   uploads, no export UI" is not built: there is no build-mode flag at
   all, only `import.meta.env.DEV`. The published console logs
   `export capability probe failed` (and on HTTPS that probe to
   `http://localhost:8765` is blocked as mixed content), and the panel
   still offers a dead "Download rendered video". Sequenced after preset
   strings v1 because both edit `workbench.ts`. Plife front and center;
   physarum/vizfx still reachable via `?sim=` but unadvertised; workbench
   included.

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
