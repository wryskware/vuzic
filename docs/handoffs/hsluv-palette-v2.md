# Handoff: Palette v2 — HSLuv color system

**Status: approved 2026-08-13. This package lands FIRST, before the preset-strings
package (`preset-strings-v1.md`), which depends on the schema bumps made here.**

Routing note for the orchestrator: implementation worker = Opus, reasoning effort
**high** (new UI construction + cross-sim coupling). Test authoring is a separate
pass per global rules; the test spec below was written before implementation and
is part of the deliverable.

## Objective

Replace the flat hex-list palette with a Palette v2 built on HSLuv:

1. **Arc mode** (the vuzic scheme): K species hues equally spread across a
   selectable hue range `[hueStart, hueStart + hueRange)` at a shared HSLuv
   saturation and lightness.
2. **Custom mode**: per-species hex pickers (full `#XXXXXX`), plus a one-click
   **harmonize** action that snaps every color to a shared HSLuv S/L while
   keeping each hue.
3. **Global hue shift + slow autonomous cycle**: a `hueShiftDeg` offset applied
   in HSLuv hue space in *both* modes, and a `hueRateDegPerSec` that advances it
   as a slow loop (0 = static). Not music-driven.
4. **Curated palette catalog**: named palettes loadable from a dropdown, each
   either arc params or an explicit color list.
5. Schema migrations: modulation v4 → v5, export recipe v3 → v4.

## Why (context you must not re-derive differently)

- The deciding factor for this project is that the output **looks good**
  (CLAUDE.md priorities). Colors are currently good; this is an upgrade pass.
- plan.md Revision 2 says hue is never **music-modulated** (modulating hue
  muddied the image and made species impossible to follow). The user has
  confirmed 2026-08-13 that a slow *autonomous* global cycle — all hues rotating
  together, driven by nothing — is consistent with the reason behind that
  decision. Do NOT wire hue to θ, stems, drivers, or impulses.
- HSLuv over OKLCh (decided, do not reopen): HSLuv guarantees every (H, S, L)
  is inside sRGB gamut (S=100 rides the gamut boundary) and equal-L hues have
  equal perceived lightness, which is exactly what "spread K hues at shared
  S/L" needs. The prior art is vuzic, which used HSLuv and looked right.
- **Behavioral ground truth**: the user's earlier vuzic sims at
  `~/Documents/vuzic` (Windows: `C:\Users\perag\Documents\vuzic`). Consult its
  hue-arc / global-hue-adjust implementation before designing the arc controls.
  If the directory is missing, say so in your report; do not guess at its
  behavior.

## Target schema

```jsonc
// replaces Palette in web/src/sim/palette.ts
{
  "mode": "arc" | "custom",
  "arc": { "hueStartDeg": 0, "hueRangeDeg": 360, "sat": 90, "light": 60 }, // HSLuv terms
  "colors": ["#..."],          // authoritative in custom mode; derived in arc mode
  "hueShiftDeg": 0,            // global rotate, HSLuv hue space, both modes
  "hueRateDegPerSec": 0,       // 0 = static; slow autonomous cycle otherwise
  "saturation": 1,             // EXISTING linear-space trim, unchanged semantics
  "brightness": 1              // EXISTING linear-space trim, unchanged semantics
}
```

Naming caution: the existing `saturation` field is a linear-space lerp against
luminance (see `paletteLinear`), NOT HSLuv saturation. Keep the arc's HSLuv
fields visibly distinct (`sat`/`light` under `arc`, or better names of your
choosing) so the two never read as the same knob.

Exact field names and whether `arc` is always present or optional-in-custom-mode
are yours to decide; the serialized shape must be stable and strictly parsed
like everything else in `mapping/persist.ts`.

## Behavior spec

- **Arc mode**: species k gets hue `hueStartDeg + hueShift_eff + k * hueRangeDeg / K`
  (negative `hueRangeDeg` allowed for direction, or provide a direction control —
  match vuzic's feel), at shared `sat`/`light`. Colors are recomputed, not
  hand-stored; the derived hexes still flow into the serialized `colors` array
  OR are recomputed at parse time — your call, but recipes must be
  self-describing (a headless renderer must produce identical colors).
- **Custom mode**: authored hexes are converted to HSLuv, rotated by
  `hueShift_eff`, converted back. Rotation preserves each color's HSLuv S/L.
- **Hue cycle determinism**: `hueShift_eff = hueShiftDeg + hueRateDegPerSec *
  simTimeSeconds`, where sim time comes from the simulation's step clock (see
  `web/src/sim/step-cadence.ts` and how each sim tracks steps), NEVER wall
  clock. Pausing the sim pauses the cycle. A headless export must replay the
  identical color trajectory. `captureExportRecipe` captures `hueShiftDeg`
  as-authored (the base), not the current effective shift, because the export
  replays sim time from zero — think this through against how the export worker
  seeds sim time, and state in your report what you did.
- **Per-frame cost**: recompute ≤64 colors CPU-side and re-upload when the
  effective shift moves. Physarum has `refreshPalette`/`paletteDirty`
  (`web/src/sim/physarum/physarum.ts:219`), plife has `invalidatePalette`;
  vizfx visuals consume the shared palette too. All three substrates must pick
  up the cycle.
- **VizFX caution**: plasma/nebula/kaleido have their own per-layer "hue
  rotation / step (rad)" shader params (`web/src/sim/vizfx/`). Those are
  intra-visual machinery and are OUT of scope — the palette-level shift
  composes upstream of them. Do not touch them.
- **Harmonize button** (custom mode): snaps all colors to the mean (or a
  chosen) HSLuv S/L, keeping hues. One click, undoable by editing pickers.
- **Curated catalog**: new module (e.g. `web/src/sim/palettes.ts`) exporting
  named entries `{ name, palette-v2 fragment }`. Ship ~6–10 starters: at least
  two arc definitions and several non-arc shapes (complementary pair with
  spread, analogous + accent, warm/cool split, mono + accent). Loading one
  overwrites the palette block. Quality bar: they must look good in the actual
  sims, not just as swatches — spend real time here, this is art direction.
- **Physarum default hue walk**: `defaultPaletteColor`
  (`web/src/sim/physarum/config.ts:328`) uses HSL `rotateHue` (41° steps).
  Migrate it to HSLuv rotation ONLY if you verify the shipped default look is
  visually unchanged (side-by-side stills); otherwise freeze the current
  output as literal hexes and note it. **The shipped default appearance must
  not change.**

## Migration

- `MODULATION_VERSION` 4 → 5 (`web/src/mapping/types.ts`,
  `web/src/mapping/persist.ts`). A v4 palette `{colors, saturation,
  brightness}` lifts losslessly to
  `{mode: 'custom', colors, saturation, brightness, hueShiftDeg: 0,
  hueRateDegPerSec: 0, ...}`. Rendered output must be pixel-identical after
  migration — the user has tuned looks in their autosaves and existing
  `modulation.json` files. Follow the existing one-warning-per-session
  migration pattern in persist.ts.
- `EXPORT_RECIPE_VERSION` 3 → 4 (`web/src/runtime/recipe.ts`). Keep parsing v3
  recipes with the same lossless lift (this differs from the profile-rename
  rejection precedent because the lift preserves meaning exactly — say so in a
  comment). v4 validates the palette v2 shape strictly (`validatePalette` and
  the `palettesEqual` sim/modulation agreement check both update).
- `web/src/runtime/capture-recipe.ts` and `web/src/runtime/sim-bundle.ts`
  follow the schema.

## Files

Core: `web/src/sim/palette.ts` (schema + conversions + rotation),
`web/src/sim/physarum/config.ts` (defaults/hue walk), new
`web/src/sim/palettes.ts` (catalog).
Mapping: `web/src/mapping/persist.ts`, `web/src/mapping/types.ts`,
`web/src/mapping/target.ts` (palette in `ModTargetConfig`).
Recipe: `web/src/runtime/recipe.ts`, `capture-recipe.ts`, `sim-bundle.ts`.
Sims: `web/src/sim/physarum/physarum.ts`, `web/src/sim/plife/plife.ts`,
vizfx visuals under `web/src/sim/vizfx/`.
UI: `web/src/ui/panel.ts` (physarum), `web/src/ui/plife-panel.ts` (palette
bindings near line 670), `web/src/ui/vizfx-panel.ts`, `web/src/ui/workbench.ts`.
HSLuv: add the `hsluv` npm package (tiny, MIT); vendoring the conversion
(~200 lines) is acceptable if the package fights strict TS.

## UI expectations

Follow the existing tweakpane folder conventions in each panel. Mode selector;
arc controls (start, range, S, L, cycle rate with a range that makes "slow"
easy — think degrees/sec 0..10 with fine steps); custom pickers + harmonize;
catalog dropdown. The cycle rate control should read as "0 = fixed set of
hues", matching how the user described using vuzic.

## Decision authority

Yours: internal representation, exact field names, UI layout within panel
conventions, package-vs-vendor for HSLuv, how derived arc colors serialize.
Escalate (stop and report, don't guess): if the palette-shared-by-reference
invariant (sim config and modulation config share one Palette object — see
persist.ts comments and recipe's `palettesEqual`) makes the v2 schema awkward;
if pixel-identical migration turns out impossible; anything touching the
export worker's sim-time seeding.

## Exclusions

No music-driven hue. No OKLCh. No preset system (separate package). No
per-layer vizfx shader changes. No gradient/blend between species colors.

## Test spec (write these; green tests from your own understanding are not the bar)

- HSLuv round-trip: hex → HSLuv → hex identity within 1/255 per channel.
- Arc generation: known (K, start, range, S, L) → exact expected hexes
  (snapshot a handful, including range < 360 and wrap-around).
- Rotation: custom-mode rotation preserves HSLuv S/L; 360° shift is identity.
- Determinism: same sim-time sequence → identical color sequence; paused time
  → no drift.
- Migration: a real v4 modulation JSON → v5 → `paletteLinear` outputs
  bit-identical to the v4 path; recipe v3 fixture parses and lifts.
- Recipe v4: strict validation accepts the new shape, rejects a v4 recipe with
  a v1 palette block, sim/modulation palette agreement still enforced.
- Existing suites (`web/tests/export-recipe.test.ts`, sdr/hdr profile tests)
  updated, not deleted.

## Verification & hygiene

- `npm test` and typecheck in `web/` must pass; run them, paste results.
- Visual check: stills only, at fixed hue phases (defaults unchanged; arc mode
  at two different starts; one catalog palette in each sim). Automation tabs
  run rAF at ~4–8 fps — never judge motion or fps, only equilibrium stills.
- **Persisted-state contract**: if you open the app in a browser, do not let
  your session pollute the user's autosave. Before interacting, snapshot
  `localStorage` keys `lmt.mapping*`; restore them before you finish. Report
  that you did.

## Return format

Evidence report: what changed per file (summary, not a diff dump); the
migration story with the fixture you used; how sim-time seeding interacts with
export capture (explicitly); screenshots taken and what they show; test run
output; decisions made with reasons; anything escalated or left open.
