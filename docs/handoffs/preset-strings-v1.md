# Handoff: Preset strings v1 — save / copy / link

**Status: approved 2026-08-13. This package lands SECOND — it depends on
Palette v2 (`hsluv-palette-v2.md`): modulation v5 and export recipe v4 must be
merged first, so presets never serialize the v1 palette shape.**

Routing note for the orchestrator: implementation worker = Opus, reasoning
effort **medium**. Test spec below is part of the brief (separate authoring
concern; do not substitute your own green tests for it).

## Objective

Let the user save everything they had set as a named local preset, copy it as a
compact string, share it as a link, and get it all back — serverless. Format:
`lmt1.` + base64url( deflate-raw( canonical JSON ) ).

## Why / context

- Immediate need: the user wants to save their tuned looks now. Server-side
  preset storage (user system, public scoping) is explicitly future scope —
  design nothing for it beyond what falls out for free.
- The free part: `serializeExportRecipe` already produces canonical,
  stable-key-order JSON, so a preset string is content-hashable. The future
  server story is "POST string, get short id" — that is ALL the
  future-proofing allowed.
- Music never travels: presets carry a track id as an advisory hint only.
  Copyright constraint, user-stated.

## Format: PresetV1

The export recipe minus export-only concerns. Fields:

```jsonc
{
  "version": 1,               // container version, independent of inner versions
  "sim": "plife",             // concrete ModTarget simId
  "seed": 123456,             // ALWAYS present
  "seedPinned": false,        // pin state at save time
  "track": "pink-loop",       // OPTIONAL advisory hint; loaders ignore if absent locally
  "simulation": { ... },      // SimulationBaseConfig, as in recipe v4
  "modulation": { ... },      // RecipeModulationConfig (v5, includes extras + palette v2)
  "modulationBase": [ ... ],
  "impulses": { ... },
  "render": { ... }
}
```

Dropped vs. the recipe: `rendererBuild`, `output`, `presentation`,
`particleBudget` (derivable from the sim config; recompute at apply time).

**Reuse, don't duplicate**: `validateSimulation`, `validateModulation`,
`validateRender`, `validateImpulse`, `jsonValue`, and `canonical` in
`web/src/runtime/recipe.ts` are the validators. Export them (or factor them
into a shared module both recipe and preset import) and build the preset
parser/serializer on top, with the same fail-loudly style. Capture reuses the
`captureExportRecipe` pattern (`web/src/runtime/capture-recipe.ts`) — same
structuredClone hygiene, same modulationBase manual-vs-modulated rule (already
implemented there; do not re-derive it).

## Encoding

- JSON → `CompressionStream('deflate-raw')` → base64url (`-_`, no padding),
  prefixed `lmt1.`. Decode is the exact inverse with strict failure on: unknown
  prefix, invalid base64url, inflate error, JSON error, validator error. Every
  failure surfaces a readable message in the UI, never a silent fallback.
- The container prefix versions the *envelope*; inner blocks carry their own
  versions (modulation v5 etc.), so future schema bumps ride the existing
  migration machinery.
- Enforce a decoded-size cap before parsing (reuse the recipe's
  `MAX_RECIPE_JSON_CHARS` bound) so a hostile string can't balloon.

## Seed semantics (user-decided 2026-08-13, do not change)

- The current seed is ALWAYS saved into the preset and ALWAYS replayed on
  load, so a loaded preset reproduces exactly what was saved. Pin state is
  restored from the preset.
- The **pin seed** checkbox keeps its current live meaning (unpinned = fresh
  seed per run/reload). Add a tooltip/hint to the pin control in BOTH panels
  (`web/src/ui/plife-panel.ts:229` and the physarum equivalent in
  `web/src/ui/panel.ts`) saying, in substance: "the current seed is always
  saved into presets and replayed when one loads; pinning additionally keeps
  live runs on this seed." Tweakpane has no native tooltip — use whatever
  mechanism matches existing panel conventions (a title attribute on the
  binding's element, or a small readonly hint row); keep it unobtrusive.

## Apply path: boot-time only (decided — no live in-place apply in v1)

- `#p=lmt1....` URL fragment (fragment, NOT query — never reaches a server,
  works on static hosting). Parsed at boot before sim construction; the
  preset's `sim` and `seed` win over `?sim=` / `?seed=` params if both are
  present. After a successful apply, remove the fragment via
  `history.replaceState` so a later reload uses the autosave, not a stale
  re-apply over session tweaks.
- In-app **Load**: write the encoded string to a pending slot
  (`localStorage['lmt.pendingPreset']`) and `location.reload()`. At boot the
  pending slot takes precedence over the fragment and is deleted immediately
  after being read (consumed once), including when applying it fails.
- Applying a preset sets live state, which flows into the normal autosave
  path — no extra sync step.
- Boot precedence: pending slot > `#p=` fragment > existing autosave behavior.
- On any apply failure: fall back to the normal boot path and show the error;
  never leave a half-applied state.
- Track hint: if `track` names a locally available track, select it; otherwise
  keep the current/default track and surface a one-line notice. Driver-gain
  length mismatch across tracks is already handled by the Modulator's resize on
  `setConfig` — verify, don't reimplement.

## UI

A "presets" section in the workbench tab (`web/src/ui/workbench.ts`), following
existing folder conventions:

- name field + **save** button → library entry.
- library dropdown + **load** / **delete**.
- **copy string** and **copy link** (`location.origin + pathname + '#p=' + s`)
  via `navigator.clipboard.writeText`.
- **import**: paste a string (with or without a full URL around it — accept
  both, extract the `lmt1.` token) → apply via the pending-slot path, and offer
  it a library slot (auto-save it under an "imported" name is fine).
- Library storage: `localStorage['lmt.presets']` = JSON array of
  `{ name, savedAt, data }` where `data` is the encoded string (one codec path
  everywhere). Quota failures degrade gracefully like the existing autosave
  (`saveModulationLocal` pattern).
- Keep the existing `modulation.json` file download/upload untouched — it is a
  different, diffable surface and still wanted.
- A one-line size caveat in the UI copy near "copy link" is fine at large K
  (K=64 plife links reach tens of KB; the string always works, some chat apps
  truncate long URLs). Do not engineer around it.

## Files

New: `web/src/runtime/preset.ts` (format + codec + validators reuse),
plus boot wiring in `web/src/main.ts` and UI in `web/src/ui/workbench.ts`.
Touched: `web/src/runtime/recipe.ts` (export/factor validators — no behavior
change; existing recipe tests must stay green), `web/src/ui/panel.ts`,
`web/src/ui/plife-panel.ts` (pin tooltip), `web/src/sim/seed.ts` (whatever the
apply path needs to set a seed at boot — `setPinnedSeed`/`syncUrlSeed` exist).

## Decision authority

Yours: module layout, exact UI copy, how the boot hook slots into `main.ts`'s
existing startup order, whether to round numbers pre-compression.
Escalate (stop and report): if boot-time apply collides with existing `?sim=` /
`?seed=` / vizfx presentation-mode URL handling in a way that needs
restructuring; if the validator refactor forces recipe behavior changes;
anything that would touch the export worker.

## Exclusions

No server storage, no short links, no user system, no live in-place apply, no
preset thumbnails/previews, no migration UI for old presets (there are none —
this is v1 of the container).

## Test spec

- Round-trip: capture → encode → decode → deep-equal, for physarum, plife, and
  one vizfx visual fixture (reuse/extend recipe test fixtures).
- Canonical stability: same state twice → identical string.
- Rejection: bad prefix, truncated base64url, corrupt deflate, valid JSON
  failing block validators, oversize decoded payload — each with a readable
  error.
- Seed: preset stores live seed; decode surfaces `seed`/`seedPinned` for the
  boot path to replay (unit-test the apply decision function, extracted pure).
- Boot precedence and consumption: pending slot beats fragment, slot deleted
  after read even on failure, fragment cleared after successful apply
  (pure-function tests; do not try to integration-test `location.reload`).
- URL extraction: import accepts a bare token and a full URL containing one.
- Node's `CompressionStream` is available in the vitest environment (Node ≥18);
  if the test env lacks it, polyfill in test setup rather than abstracting the
  codec.
- Recipe validator refactor: existing `web/tests/export-recipe.test.ts` passes
  unmodified (or with import-path-only edits).

## Verification & hygiene

- `npm test` + typecheck in `web/`; paste results.
- One real browser pass: save a preset, tweak something, load the preset back,
  confirm restoration; open a `#p=` link in a fresh tab. Stills only —
  automation tabs run rAF at ~4–8 fps; never judge motion or fps.
- **Persisted-state contract**: your browser session writes to
  `lmt.mapping*`, `lmt.presets`, and `lmt.pendingPreset`. Snapshot the user's
  `lmt.mapping*` keys before interacting and restore them before finishing;
  delete any test presets you created. Report that you did.

## Return format

Evidence report: format + codec decisions with reasons; the boot-order change
in `main.ts` (before/after description); how `?seed=`/`?sim=` interplay was
resolved; test output; the browser pass narrative with screenshots; anything
escalated or left open.
