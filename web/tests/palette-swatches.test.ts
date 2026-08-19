/**
 * The palette folder's per-species preview swatches.
 *
 * The row itself is DOM and is not tested here; what is worth pinning is the
 * only thing a swatch can get *wrong* — showing a colour the substrate is not
 * about to draw. So every case below compares `paletteSwatchHexes` against the
 * value the render path actually uploads (`paletteLinear`, exactly as
 * `plife.ts` and `physarum.ts` call it), rather than against another arc
 * formula written here.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  customPalette,
  hexToLinear,
  paletteHuePhase,
  paletteLinear,
  syncPaletteColors,
  type Palette,
} from '../src/sim/palette.ts';
import { PALETTE_CATALOG, applyPaletteEntry } from '../src/sim/palettes.ts';
import { paletteSwatchHexes } from '../src/ui/palette-folder.ts';
import { defaultPlifePalette, defaultPlifePaletteColor, PRIMARY_COUNT } from '../src/sim/plife/config.ts';

/** plife's own upload loop, as `refreshPalette` runs it. */
function rendered(palette: Palette, k: number, groupSize: number): [number, number, number][] {
  const phase = paletteHuePhase(palette, 0);
  return Array.from({ length: k }, (_, i) =>
    paletteLinear(palette, i, defaultPlifePaletteColor(i), phase, k, groupSize),
  );
}

/**
 * A swatch is an 8-bit hex and the render path is float, so "the same colour"
 * is "within one quantisation step per channel" — in *linear* space, where the
 * comparison is being made, one 8-bit sRGB step near white is about 0.01.
 */
function assertMatches(palette: Palette, k: number, groupSize: number, what: string): void {
  const swatches = paletteSwatchHexes(palette, k, groupSize);
  assert.equal(swatches.length, k, what);
  const want = rendered(palette, k, groupSize);
  swatches.forEach((hex, i) => {
    assert.match(hex, /^#[0-9a-f]{6}$/, `${what}: species ${i} is ${hex}`);
    const got = hexToLinear(hex);
    for (let c = 0; c < 3; c++) {
      assert.ok(
        Math.abs((got[c] as number) - (want[i] as number[])[c]!) < 0.012,
        `${what}: species ${i} channel ${c}: swatch ${got[c]} vs rendered ${(want[i] as number[])[c]}`,
      );
    }
  });
}

test('every catalog entry previews the colours plife is about to draw', () => {
  for (const entry of PALETTE_CATALOG) {
    const palette = defaultPlifePalette(8);
    applyPaletteEntry(palette, entry, 8, PRIMARY_COUNT);
    assertMatches(palette, 8, PRIMARY_COUNT, `${entry.name} at K=8`);
  }
});

test('a catalog load edits the arcs in place, so the sliders stay connected', () => {
  // The panel binds its four arc sliders to `palette.arc` itself
  // (`addArcControls(folder, p.arc)`), which is why `applyPaletteEntry` may not
  // hand the palette a fresh object. It used to, and the failure was invisible:
  // the sliders went on editing a detached copy, so after loading any arc entry
  // the arc knobs did nothing at all.
  for (const entry of PALETTE_CATALOG) {
    const palette = defaultPlifePalette(8);
    const arc = palette.arc;
    const accentArc = palette.accentArc;
    applyPaletteEntry(palette, entry, 8, PRIMARY_COUNT);
    assert.equal(palette.arc, arc, `${entry.name} replaced the primaries' arc`);
    assert.equal(palette.accentArc, accentArc, `${entry.name} replaced the accents' arc`);
    if (entry.arc) assert.deepEqual({ ...palette.arc }, { ...entry.arc }, entry.name);
  }

  // And the point of it: a slider write reaches the render path.
  const palette = defaultPlifePalette(8);
  const bound = palette.arc;
  applyPaletteEntry(palette, PALETTE_CATALOG.find((e) => e.mode === 'arc')!, 8, PRIMARY_COUNT);
  bound.hueStartDeg = 200;
  assertMatches(palette, 8, PRIMARY_COUNT, 'after a slider write following a load');
  assert.notDeepEqual(
    rendered(palette, 8, PRIMARY_COUNT)[0],
    rendered({ ...palette, arc: { ...bound, hueStartDeg: 0 } }, 8, PRIMARY_COUNT)[0],
  );
});

test('the preview follows the arc knobs, in every colour space', () => {
  for (const space of ['hsl', 'hsluv', 'oklch'] as const) {
    const palette: Palette = {
      ...customPalette([]),
      mode: 'arc',
      space,
      arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
      accentArc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 80 },
    };
    const before = paletteSwatchHexes(palette, 8, 4);
    assertMatches(palette, 8, 4, `${space} before the knob moves`);

    // The knob a preview exists for: move the start and every primary must move.
    palette.arc.hueStartDeg = 137;
    const after = paletteSwatchHexes(palette, 8, 4);
    assertMatches(palette, 8, 4, `${space} after the knob moves`);
    for (let i = 0; i < 4; i++) {
      assert.notEqual(after[i], before[i], `${space}: primary ${i} did not follow the arc`);
    }
    // ...and the accents must not, because they are a separate arc.
    assert.deepEqual(after.slice(4), before.slice(4), `${space}: accents followed the wrong arc`);
  }
});

test('the preview carries the authored hue shift but not the running cycle', () => {
  const palette: Palette = {
    ...customPalette([]),
    mode: 'arc',
    arc: { hueStartDeg: 10, hueRangeDeg: 200, sat: 90, light: 64 },
    accentArc: { hueStartDeg: 10, hueRangeDeg: 200, sat: 90, light: 82 },
  };
  const base = paletteSwatchHexes(palette, 4, 4);
  palette.hueShiftDeg = 90;
  const shifted = paletteSwatchHexes(palette, 4, 4);
  assert.notDeepEqual(shifted, base, 'the hue-shift slider must move the swatches');
  assertMatches(palette, 4, 4, 'a shifted arc');

  // A running cycle is a function of the sim clock, and this folder polls
  // nothing — so the swatches are pinned to the authored base and must not
  // silently depend on wall time.
  palette.hueRateDegPerSec = 4;
  assert.deepEqual(paletteSwatchHexes(palette, 4, 4), shifted);
});

test('the preview ignores the linear-space trims, deliberately', () => {
  // They are a global exposure and a global desaturation applied to the whole
  // image and then re-exposed by the grade. Folding them in would move every
  // swatch by the same amount and tell you nothing about species separation.
  const palette: Palette = {
    ...customPalette([]),
    mode: 'arc',
    arc: { hueStartDeg: 30, hueRangeDeg: 300, sat: 100, light: 60 },
    accentArc: { hueStartDeg: 30, hueRangeDeg: 300, sat: 100, light: 78 },
  };
  const neutral = paletteSwatchHexes(palette, 4, 4);
  palette.saturation = 0.2;
  palette.brightness = 1.7;
  assert.deepEqual(paletteSwatchHexes(palette, 4, 4), neutral);
});

test('a substrate with no accent group gets one uninterrupted row', () => {
  // Physarum and vizfx pass no groupSize, so K species are all primaries and a
  // poisoned accent arc must not reach a swatch — the same guarantee the render
  // path has, checked at the preview.
  const palette: Palette = {
    ...customPalette([]),
    mode: 'arc',
    arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
    accentArc: { hueStartDeg: 123, hueRangeDeg: 7, sat: 3, light: 4 },
  };
  assertMatches(palette, 6, 6, 'six primaries');
  const clean: Palette = { ...palette, accentArc: { ...palette.arc } };
  assert.deepEqual(paletteSwatchHexes(palette, 6, 6), paletteSwatchHexes(clean, 6, 6));
});

test('in custom mode the swatches are the authored hexes themselves', () => {
  // The folder only shows the row in arc mode, but the function is total: a
  // custom palette at zero shift must return exactly what was typed, not a
  // round trip through a colour space.
  const palette = customPalette(['#ff8800', '#3366cc', '#12ef7a', '#884422']);
  syncPaletteColors(palette, 4);
  assert.deepEqual(paletteSwatchHexes(palette, 4, 4), [
    '#ff8800',
    '#3366cc',
    '#12ef7a',
    '#884422',
  ]);
});

test('a short palette falls back rather than rendering an empty swatch', () => {
  const palette = customPalette(['#ff8800']);
  const swatches = paletteSwatchHexes(palette, 3, 3);
  assert.equal(swatches.length, 3);
  for (const hex of swatches) assert.match(hex, /^#[0-9a-f]{6}$/);
});
