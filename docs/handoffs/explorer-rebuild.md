# Handoff — Explorer rebuild

Scoped 2026-08-15. Roadmap: Phase 2 item 6 — deliberately **last in
Phase 2** and the one item allowed to slip past the publish cut. Suited
to a background thread; it must not block the palette pass, seed
favorites, or the README.

Routing note for the orchestrator: implementation worker = Opus,
reasoning effort **high** — this is real lifting with cross-system
coupling (sim construction, modulation, persistence, UI). Test spec at
the bottom is part of the brief; a worker's own green tests are not a
substitute.

## Why the current explorer fails

The explorer exists (`web/src/explore/rig.ts` — `ExplorerRig`, nine
tiles; `web/src/explore/search.ts` — `ExplorerSearch`, subspaces,
candidate generation; `web/src/ui/explore-panel.ts`; wiring in
`web/src/main.ts` around lines 1170–1500) but is essentially useless in
practice, for three reasons that are the contract's negation:

1. It simulates a **different parameter set** than main mode.
2. It therefore **looks nothing like** main mode.
3. It **ignores modulation** — tiles run dry, but the whole point of a
   candidate is how it behaves under the music.

## The contract (settled — do not renegotiate silently)

Explore **from the current main-mode state**:

- Tiles run with **identical parameters and identical look** to main
  mode at the moment of entry — same config, same palette, same
  particle budget discipline (nine sims at once must degrade particle
  count, not fidelity of the parameterization).
- **Modulation live** in the tiles: the same timeline drivers and
  impulse lane feed them. Music transport keys stay live (they already
  do — see the keydown handler's explorer-mode comment in `main.ts`).
- Variations are **seeded**: each tile is the current state perturbed
  by a seed, rerollable (`r` is already claimed for this while the mode
  is active).
- **Two scopes**: *everything* and *matrix-only*. Matrix-only varies
  just the interaction matrix (which comes from the seed —
  `web/src/sim/plife/genmatrix.ts`), keeping colors/physics/mapping
  fixed. The existing `EXPLORER_SUBSPACES` machinery may inform this
  but the two-scope contract replaces it as the UI surface.
- **Adopt-or-cancel**: pick a tile → that state becomes main mode;
  cancel → main mode exactly as you left it, nothing saved.

## Machinery worth reusing (don't rebuild these)

- **The state container is `ExportRecipe`** (`web/src/runtime/recipe.ts`).
  Save profiles v0 (`web/src/ui/profiles.ts`) established the pattern:
  a whole recipe captures everything, including seed-derived state, and
  **apply = stage the recipe in the pending slot, then reload**, going
  through `main.ts`'s single construction order. "Adopt" is plausibly
  exactly that path with the tile's recipe. Reuse it rather than
  inventing a second in-place apply; if in-place adoption without a
  reload turns out to be needed for feel, say so and justify it.
- Persistence is by construction now (`web/src/mapping/read-into.ts`,
  branded `PersistedContainer` in the panel layer). Any new config
  block must be registered per whatever mechanism is current when this
  runs — check the roadmap's Phase 1 item 5 status (automatic block
  registration may have landed by then, changing the rules in your
  favor).
- The explorer log (`web/src/explore/log.ts`, jsonl download wired near
  `main.ts:864`) is the seed-favorites data channel. Keep it working —
  logged candidate/choice data becomes the training pool later.

## Constraints

- Explorer state (which tiles, seeds, scope) is session-only. Nothing
  the explorer does may write autosave or profiles until adopt.
- Cancel must be airtight: entering + leaving the explorer with no
  adoption is a no-op on every persisted surface.
- GPU budget: nine concurrent sims must not OOM or thrash — reuse the
  particle-budget governor's discipline; degrade tile particle counts,
  and document the chosen budget split.
- If the design balloons, checkpoint and report rather than shipping a
  half-contract: the roadmap explicitly allows this item to slip.

## Verification / test spec

- Unit: recipe round-trip for a tile — capture main state, perturb
  matrix-only, assert everything outside the matrix is byte-identical
  in the resulting recipe; perturb everything-scope, assert seed and
  parameters differ but schema validates.
- Unit: cancel path — enter/exit leaves autosave and profile keys
  untouched (assert on a mock storage).
- Manual (a human or the requesting session, not a throttled automation
  tab — automated tabs render rAF at ~4–8 fps and cannot judge motion):
  tiles visibly match main-mode look; modulation visibly moves tiles;
  adopt lands the exact tile; cancel restores exactly.
- Full suite (`npm test` in `web/`) green before handing back.

## Return format

Report the design actually built (tile rendering strategy, budget
split, how modulation is fanned out, how adopt is implemented), the
evidence for each contract point, test results, and any contract point
you could not satisfy — with reasoning.
