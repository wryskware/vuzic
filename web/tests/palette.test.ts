/**
 * Palette v2 (HSLuv). Written against the handoff's test spec
 * (`docs/handoffs/hsluv-palette-v2.md`) rather than against the implementation,
 * so the expected arc hexes below are snapshots produced once and pasted in, and
 * the migration test compares against an *independently written* copy of the v1
 * colour maths rather than against the function under test.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  arcHue,
  customPalette,
  harmonizePalette,
  hexToHsluv,
  hexToLinear,
  hsluvToHex,
  paletteHex,
  paletteHuePhase,
  paletteLinear,
  paletteSrgb,
  spaceToHex,
  spaceToRgb,
  hexToSpace,
  PALETTE_SPACES,
  rotateHueIn,
  syncPaletteColors,
  wrapHue,
  type Palette,
  type PaletteArc,
} from '../src/sim/palette.ts';
import { PALETTE_CATALOG, applyPaletteEntry } from '../src/sim/palettes.ts';
import { defaultPalette, defaultPaletteColor } from '../src/sim/physarum/config.ts';
import { defaultModulationConfig, parseModulation, serializeModulation } from '../src/mapping/persist.ts';
import { defaultRenderConfig } from '../src/sim/render/config.ts';

// ── conversions ──────────────────────────────────────────────────────────────

test('the vendored HSLuv matches the reference implementation at its fixed points', () => {
  // Published reference values for the HSLuv 1.0 algorithm. If these drift, the
  // transcription in sim/hsluv.ts is wrong, not merely different.
  const [h, s, l] = hexToHsluv('#ff0000');
  assert.ok(Math.abs(h - 12.17705) < 1e-4, `red hue ${h}`);
  assert.ok(Math.abs(s - 100) < 1e-6, `red saturation ${s}`);
  assert.ok(Math.abs(l - 53.23711) < 1e-4, `red lightness ${l}`);
  const green = hexToHsluv('#00ff00');
  assert.ok(Math.abs(green[0] - 127.71501) < 1e-4);
  assert.ok(Math.abs(green[2] - 87.73552) < 1e-4);
  // Neutrals have no hue and no saturation, at either end and in between.
  assert.deepEqual(hexToHsluv('#000000'), [0, 0, 0]);
  assert.deepEqual(hexToHsluv('#ffffff'), [0, 0, 100]);
  assert.equal(hexToHsluv('#808080')[1], 0);
});

test('hex → HSLuv → hex is identity to within one 8-bit step per channel', () => {
  // A deterministic sweep rather than random samples: 512 colours spread over the
  // cube, including the primaries, the greys and the near-black corner where the
  // L* transfer function switches branches.
  const samples: string[] = ['#000000', '#ffffff', '#ff0000', '#00ff00', '#0000ff', '#010203'];
  for (let r = 0; r < 256; r += 51) {
    for (let g = 0; g < 256; g += 51) {
      for (let b = 0; b < 256; b += 51) {
        samples.push(
          `#${[r, g, b].map((c) => c.toString(16).padStart(2, '0')).join('')}`,
        );
      }
    }
  }
  for (const hex of samples) {
    const [h, s, l] = hexToHsluv(hex);
    const back = hsluvToHex(h, s, l);
    for (let i = 0; i < 3; i++) {
      const a = parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16);
      const b = parseInt(back.slice(1 + i * 2, 3 + i * 2), 16);
      assert.ok(Math.abs(a - b) <= 1, `${hex} → ${back} channel ${i}`);
    }
  }
});

/** Shortest angular distance between two hues, in degrees — always 0..180. */
function hueDelta(a: number, b: number): number {
  const d = wrapHue(a - b);
  return Math.min(d, 360 - d);
}

test('wrapHue folds any angle into [0, 360)', () => {
  assert.equal(wrapHue(0), 0);
  assert.equal(wrapHue(360), 0);
  assert.equal(wrapHue(-90), 270);
  assert.equal(wrapHue(725), 5);
});

// ── arc generation ───────────────────────────────────────────────────────────

/** Snapshots, generated once from the arc formula and pinned here deliberately. */
const ARC_CASES: { name: string; arc: PaletteArc; k: number; expect: string[] }[] = [
  {
    name: 'the full wheel divided across K',
    arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
    k: 4,
    expect: ['#ff5a87', '#959d00', '#00a89c', '#9784ff'],
  },
  {
    name: 'a narrow arc stays inside one family',
    arc: { hueStartDeg: 8, hueRangeDeg: 92, sat: 100, light: 68 },
    k: 4,
    expect: ['#ff7e89', '#ff8518', '#d79b00', '#b8a700'],
  },
  {
    name: 'an arc that wraps past 360 keeps going round',
    arc: { hueStartDeg: 330, hueRangeDeg: 120, sat: 95, light: 58 },
    k: 3,
    expect: ['#fa2bc2', '#fb4956', '#b9811b'],
  },
  {
    name: 'a negative range runs the arc backwards',
    arc: { hueStartDeg: 175, hueRangeDeg: -85, sat: 95, light: 63 },
    k: 4,
    expect: ['#22ab99', '#21ad80', '#1faf48', '#72a81f'],
  },
];

for (const { name, arc, k, expect } of ARC_CASES) {
  test(`arc generation: ${name}`, () => {
    const palette: Palette = { ...customPalette([]), mode: 'arc', arc };
    const got = Array.from({ length: k }, (_, i) => paletteHex(palette, i, '#808080', 0, k));
    assert.deepEqual(got, expect);
  });
}

test('the arc divides by K, so a full wheel has no duplicate at the seam', () => {
  const arc: PaletteArc = { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 };
  assert.equal(arcHue(arc, 0, 4), 0);
  assert.equal(arcHue(arc, 1, 4), 90);
  assert.equal(arcHue(arc, 4, 4), 360, 'species K would land back on species 0');
  // Which is exactly why K species use indices 0..K-1 and never collide.
  const palette: Palette = { ...customPalette([]), mode: 'arc', arc };
  const hexes = Array.from({ length: 6 }, (_, i) => paletteHex(palette, i, '#808080', 0, 6));
  assert.equal(new Set(hexes).size, 6);
});

test('an arc palette shares one HSLuv lightness across every species', () => {
  const arc: PaletteArc = { hueStartDeg: 17, hueRangeDeg: 300, sat: 90, light: 55 };
  const palette: Palette = { ...customPalette([]), mode: 'arc', arc };
  for (let i = 0; i < 8; i++) {
    const [, s, l] = hexToHsluv(paletteHex(palette, i, '#808080', 0, 8));
    assert.ok(Math.abs(l - 55) < 0.6, `species ${i} lightness ${l}`);
    assert.ok(Math.abs(s - 90) < 1.5, `species ${i} saturation ${s}`);
  }
});

test('syncPaletteColors rewrites the cache in arc mode and only pads in custom mode', () => {
  const arcPalette: Palette = {
    ...customPalette(['#000000', '#000000', '#000000', '#000000']),
    mode: 'arc',
    arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
  };
  const before = arcPalette.colors;
  syncPaletteColors(arcPalette, 4);
  assert.deepEqual(arcPalette.colors, ['#ff5a87', '#959d00', '#00a89c', '#9784ff']);
  assert.equal(arcPalette.colors, before, 'the array is edited in place, never replaced');

  const custom = customPalette(['#112233']);
  syncPaletteColors(custom, 3, (i) => `#00000${i}`);
  assert.deepEqual(custom.colors, ['#112233', '#000001', '#000002']);
});

// ── rotation ─────────────────────────────────────────────────────────────────

test('custom-mode rotation preserves each colour\'s HSLuv saturation and lightness', () => {
  for (const hex of ['#ff8800', '#3366cc', '#12ef7a', '#884422']) {
    const [, s0, l0] = hexToHsluv(hex);
    for (const deg of [17, 90, 180, 271, -45]) {
      const [h, s, l] = hexToHsluv(rotateHueIn('hsluv', hex, deg));
      assert.ok(Math.abs(s - s0) < 1.5, `${hex} +${deg}° saturation ${s} vs ${s0}`);
      assert.ok(Math.abs(l - l0) < 0.6, `${hex} +${deg}° lightness ${l} vs ${l0}`);
      assert.ok(hueDelta(h, hexToHsluv(hex)[0] + deg) < 1, `${hex} +${deg}° hue ${h}`);
    }
  }
});

test('a 360° shift is the identity, and a 0° shift does not even convert', () => {
  const palette = customPalette(['#ff8800', '#3366cc']);
  // Zero shift short-circuits to the authored string — this is the property the
  // whole "shipped look does not change" guarantee rests on.
  assert.equal(paletteHex(palette, 0, '#808080', 0), '#ff8800');
  assert.equal(paletteHex(palette, 1, '#808080', 720), '#3366cc');
  assert.equal(paletteHex(palette, 0, '#808080', -360), '#ff8800');
  // A full turn expressed as a rotation is identity to within a quantisation step.
  const round = rotateHueIn('hsluv', '#ff8800', 360);
  for (let i = 0; i < 3; i++) {
    const a = parseInt('#ff8800'.slice(1 + i * 2, 3 + i * 2), 16);
    const b = parseInt(round.slice(1 + i * 2, 3 + i * 2), 16);
    assert.ok(Math.abs(a - b) <= 1);
  }
});

test('the shift applies in arc mode too, and equals moving the arc start', () => {
  const arc: PaletteArc = { hueStartDeg: 40, hueRangeDeg: 200, sat: 90, light: 60 };
  const shifted: Palette = { ...customPalette([]), mode: 'arc', arc };
  const moved: Palette = {
    ...customPalette([]),
    mode: 'arc',
    arc: { ...arc, hueStartDeg: 70 },
  };
  for (let i = 0; i < 5; i++) {
    assert.equal(
      paletteHex(shifted, i, '#808080', 30, 5),
      paletteHex(moved, i, '#808080', 0, 5),
    );
  }
});

test('harmonize keeps every hue and gives every colour the mean S and L', () => {
  const palette = customPalette(['#ff0000', '#004400', '#8899ff', '#cccc00']);
  const hues = palette.colors.map((c) => hexToHsluv(c)[0]);
  const meanS = palette.colors.reduce((a, c) => a + hexToHsluv(c)[1], 0) / 4;
  const meanL = palette.colors.reduce((a, c) => a + hexToHsluv(c)[2], 0) / 4;
  harmonizePalette(palette, 4);
  palette.colors.forEach((c, i) => {
    const [h, s, l] = hexToHsluv(c);
    assert.ok(hueDelta(h, hues[i] as number) < 1.5, `hue ${i}: ${h}`);
    assert.ok(Math.abs(s - meanS) < 1.5, `saturation ${i}: ${s}`);
    assert.ok(Math.abs(l - meanL) < 0.6, `lightness ${i}: ${l}`);
  });
});

// ── the autonomous cycle ─────────────────────────────────────────────────────

test('the hue phase is a pure function of the palette base and sim time', () => {
  const palette = customPalette(['#ff8800']);
  palette.hueShiftDeg = 25;
  palette.hueRateDegPerSec = 3;
  assert.equal(paletteHuePhase(palette, 0), 25);
  assert.equal(paletteHuePhase(palette, 10), 55);
  // Replaying the same clock gives the same colours, in the same order, always.
  // Includes single-step gaps (1/60 s) on purpose: at 3°/s those are 0.05° and
  // may quantise to the same hex, which is fine and is not what determinism
  // means. Determinism is that the *same* clock gives the *same* sequence.
  const clock = [0, 1 / 60, 2 / 60, 0.5, 1, 7.25, 60];
  const first = clock.map((t) => paletteHex(palette, 0, '#808080', paletteHuePhase(palette, t)));
  const second = clock.map((t) => paletteHex(palette, 0, '#808080', paletteHuePhase(palette, t)));
  assert.deepEqual(first, second);
  // And that a running cycle is actually running: over seconds, it moves.
  const coarse = [0, 5, 10, 20, 40].map((t) =>
    paletteHex(palette, 0, '#808080', paletteHuePhase(palette, t)),
  );
  assert.equal(new Set(coarse).size, coarse.length, 'a running cycle actually moves');
});

test('a paused clock does not drift: the same sim time is the same colour', () => {
  const palette = customPalette(['#3366cc', '#ff8800']);
  palette.hueRateDegPerSec = 2.5;
  // 600 renders at one frozen sim time — a paused sim still renders every frame.
  const frozen = 12.5;
  const first = paletteHex(palette, 0, '#808080', paletteHuePhase(palette, frozen));
  for (let i = 0; i < 600; i++) {
    assert.equal(paletteHex(palette, 0, '#808080', paletteHuePhase(palette, frozen)), first);
  }
  // And a zero rate is inert no matter how far the clock runs.
  palette.hueRateDegPerSec = 0;
  assert.equal(paletteHex(palette, 1, '#808080', paletteHuePhase(palette, 1e6)), '#ff8800');
});

test('a whole lap of the cycle returns to where it started', () => {
  const palette = customPalette(['#12ef7a']);
  palette.hueRateDegPerSec = 4;
  const lap = 360 / 4;
  const a = paletteHex(palette, 0, '#808080', paletteHuePhase(palette, 0));
  const b = paletteHex(palette, 0, '#808080', paletteHuePhase(palette, lap));
  for (let i = 0; i < 3; i++) {
    const x = parseInt(a.slice(1 + i * 2, 3 + i * 2), 16);
    const y = parseInt(b.slice(1 + i * 2, 3 + i * 2), 16);
    assert.ok(Math.abs(x - y) <= 1, `${a} vs ${b}`);
  }
});

// ── the catalog ──────────────────────────────────────────────────────────────

test('every catalog entry produces K in-gamut colours for any K', () => {
  assert.ok(PALETTE_CATALOG.length >= 6, 'the catalog is a menu, not an example');
  assert.ok(PALETTE_CATALOG.filter((e) => e.mode === 'arc').length >= 2);
  assert.ok(PALETTE_CATALOG.filter((e) => e.mode === 'custom').length >= 3);
  assert.equal(new Set(PALETTE_CATALOG.map((e) => e.name)).size, PALETTE_CATALOG.length);

  for (const entry of PALETTE_CATALOG) {
    for (const k of [1, 4, 8, 16]) {
      const palette = defaultPalette(k);
      palette.hueShiftDeg = 99;
      applyPaletteEntry(palette, entry, k);
      assert.equal(palette.colors.length, k, `${entry.name} at K=${k}`);
      assert.equal(palette.hueShiftDeg, 0, `${entry.name} shows itself unshifted`);
      for (const hex of palette.colors) {
        assert.match(hex, /^#[0-9a-f]{6}$/, `${entry.name}: ${hex}`);
      }
      // The catalog does not touch the author's linear-space trims.
      assert.equal(palette.saturation, 1);
      assert.equal(palette.brightness, 1);
    }
  }
});

test('catalog entries are legible: no two species collide at K = 4', () => {
  for (const entry of PALETTE_CATALOG) {
    const palette = defaultPalette(4);
    applyPaletteEntry(palette, entry, 4);
    assert.equal(new Set(palette.colors).size, 4, `${entry.name} has a duplicate species colour`);
  }
});

// ── migration ────────────────────────────────────────────────────────────────

/**
 * The v1 colour path, rewritten from the pre-palette-v2 source rather than
 * called: a migration test that used the new code as its own oracle would prove
 * only that the new code agrees with itself.
 */
function v1Linear(hex: string, saturation: number, brightness: number): [number, number, number] {
  const [r, g, b] = hexToLinear(hex);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return [
    Math.max(lum + (r - lum) * saturation, 0) * brightness,
    Math.max(lum + (g - lum) * saturation, 0) * brightness,
    Math.max(lum + (b - lum) * saturation, 0) * brightness,
  ];
}

/** A file as the previous build wrote it: v4, with a v1 palette block. */
function v4File(): string {
  const K = 8;
  const base = {
    speciesCount: K,
    palette: defaultPalette(K),
    render: defaultRenderConfig(),
  };
  // A tuned look, the kind that is actually in the user's autosaves.
  base.palette.colors[0] = '#7ad1ff';
  base.palette.colors[3] = '#ff3d6e';
  base.palette.saturation = 0.83;
  base.palette.brightness = 1.14;
  const cfg = defaultModulationConfig(base) as unknown as Record<string, unknown>;
  const text = JSON.parse(serializeModulation(cfg as never)) as Record<string, unknown>;
  text['version'] = 4;
  const palette = text['palette'] as Record<string, unknown>;
  delete palette['mode'];
  delete palette['arc'];
  delete palette['hueShiftDeg'];
  delete palette['hueRateDegPerSec'];
  return JSON.stringify(text);
}

test('a v4 modulation file lifts to v5 and renders bit-identically', () => {
  const text = v4File();
  const old = JSON.parse(text) as { palette: { colors: string[]; saturation: number; brightness: number } };
  const back = parseModulation(text);

  assert.equal(back.version, 5);
  assert.equal(back.palette.mode, 'custom');
  assert.equal(back.palette.hueShiftDeg, 0);
  assert.equal(back.palette.hueRateDegPerSec, 0);
  assert.deepEqual(back.palette.colors, old.palette.colors);
  assert.equal(back.palette.saturation, old.palette.saturation);
  assert.equal(back.palette.brightness, old.palette.brightness);

  // The bar is not "close": every channel of every species must be the identical
  // double, because this is the picture the user already tuned.
  for (let k = 0; k < back.speciesCount; k++) {
    const expected = v1Linear(
      old.palette.colors[k] as string,
      old.palette.saturation,
      old.palette.brightness,
    );
    const phase = paletteHuePhase(back.palette, 0);
    const got = paletteLinear(back.palette, k, defaultPaletteColor(k), phase, back.speciesCount);
    assert.deepEqual(got, expected, `species ${k}`);
  }
});

test('a lifted v4 palette stays identical however far the sim clock runs', () => {
  const back = parseModulation(v4File());
  const at0 = paletteLinear(back.palette, 0, defaultPaletteColor(0), paletteHuePhase(back.palette, 0));
  for (const t of [1, 60, 3600, 1e6]) {
    assert.deepEqual(
      paletteLinear(back.palette, 0, defaultPaletteColor(0), paletteHuePhase(back.palette, t)),
      at0,
      'a migrated file has no cycle, so time cannot change its colours',
    );
  }
});

test('the shipped physarum default is untouched by palette v2', () => {
  const palette = defaultPalette(12);
  assert.equal(palette.mode, 'custom');
  assert.equal(palette.hueShiftDeg, 0);
  assert.equal(palette.hueRateDegPerSec, 0);
  for (let k = 0; k < 12; k++) {
    // Still the HSL 41° walk, not an HSLuv one — migrating it would have changed
    // the shipped look, which the handoff forbids.
    assert.equal(palette.colors[k], defaultPaletteColor(k));
    assert.deepEqual(
      paletteLinear(palette, k, defaultPaletteColor(k), 0, 12),
      v1Linear(defaultPaletteColor(k), 1, 1),
    );
  }
});

// ── two arcs: primaries and accents ──────────────────────────────────────────
// plife's species come in two stem-keyed groups — four primaries and their four
// accents (`PALETTE_HEX` / `speciesStemMap`). `groupSize` is the substrate's
// primary count and is deliberately NOT serialised: it belongs to the species
// layout, not the art direction, so the sim that replays a recipe supplies the
// same constant the sim that captured it did.

const TWO_ARC: Palette = {
  ...customPalette([]),
  mode: 'arc',
  arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
  accentArc: { hueStartDeg: 40, hueRangeDeg: 200, sat: 100, light: 82 },
};

test('each group is spread across its own arc, not across K', () => {
  // Four primaries over the whole wheel, then four accents over their own 200°.
  const hues = Array.from({ length: 8 }, (_, i) =>
    Math.round(hexToHsluv(paletteHex(TWO_ARC, i, '#808080', 0, 8, 4))[0]),
  );
  // Primaries: 0, 90, 180, 270 — the full wheel divided by the *group*, not by 8.
  assert.deepEqual(hues.slice(0, 4), [0, 90, 180, 270]);
  // Accents: 40, 90, 140, 190 — their own start and their own narrower spread.
  assert.deepEqual(hues.slice(4), [40, 90, 140, 190]);
});

test('the accents carry their own lightness, independent of the primaries', () => {
  for (let i = 0; i < 8; i++) {
    const [, , l] = hexToHsluv(paletteHex(TWO_ARC, i, '#808080', 0, 8, 4));
    assert.ok(Math.abs(l - (i < 4 ? 62 : 82)) < 0.6, `species ${i} lightness ${l}`);
  }
});

test('an accent is no longer forced onto its primary\'s complement', () => {
  // The regression this whole split exists for. One arc spread across K=8 put
  // species 0 on hue 0 and species 4 on hue 180 — bass and bass·accent as exact
  // complements, so a "same instrument, sparkling" pair read as two voices.
  const oneArcAcrossK = Math.round(
    hexToHsluv(paletteHex(TWO_ARC, 4, '#808080', 0, 8, 8))[0],
  );
  assert.equal(oneArcAcrossK, 180, 'the old behaviour, kept here as the thing being fixed');

  // With the arcs aligned, an accent instead sits on its own primary's hue.
  const aligned: Palette = { ...TWO_ARC, accentArc: { ...TWO_ARC.arc, light: 80 } };
  for (let i = 0; i < 4; i++) {
    const primary = hexToHsluv(paletteHex(aligned, i, '#808080', 0, 8, 4))[0];
    const accent = hexToHsluv(paletteHex(aligned, i + 4, '#808080', 0, 8, 4))[0];
    assert.ok(Math.abs(primary - accent) < 0.6, `species ${i} pair: ${primary} vs ${accent}`);
  }
});

test('a substrate with no accent species never reads the accent arc', () => {
  // Physarum and vizfx pass no groupSize, so it defaults to K and every species
  // is a primary. A poisoned accent arc must not reach a single colour.
  const poisoned: Palette = {
    ...TWO_ARC,
    accentArc: { hueStartDeg: 123, hueRangeDeg: 7, sat: 3, light: 4 },
  };
  const withAccents = Array.from({ length: 4 }, (_, i) => paletteHex(poisoned, i, '#808080', 0, 4));
  const clean: Palette = { ...poisoned, accentArc: { ...poisoned.arc } };
  const withoutAccents = Array.from({ length: 4 }, (_, i) => paletteHex(clean, i, '#808080', 0, 4));
  assert.deepEqual(withAccents, withoutAccents);
});

test('a third group lifts lightness rather than repeating the accents', () => {
  // K = 12 at groupSize 4 is not a shape any substrate ships, but it must not
  // render group 2 identically to group 1.
  const g1 = hexToHsluv(paletteHex(TWO_ARC, 4, '#808080', 0, 12, 4));
  const g2 = hexToHsluv(paletteHex(TWO_ARC, 8, '#808080', 0, 12, 4));
  // 2° rather than the sub-degree tolerance used elsewhere: at L 92 the colour
  // is close enough to white that 8-bit quantisation moves the recovered hue by
  // about a degree. That is the round trip's precision, not a drift in the arc.
  assert.ok(Math.abs(g1[0] - g2[0]) < 2, `same hue as its accent: ${g1[0]} vs ${g2[0]}`);
  // The lift is 18 but the ceiling is 92, so from an accent at L 82 the real
  // separation is 10. Asserting the ceiling holds matters more than the nominal
  // step: without it a high accent lightness would silently collapse the groups.
  assert.ok(g2[2] > g1[2] + 5, `group 2 lightness ${g2[2]} should clear group 1's ${g1[2]}`);
  assert.ok(g2[2] <= 92.6, `group 2 lightness ${g2[2]} must stay under the ceiling`);
});

test('the hue cycle rotates both arcs together', () => {
  const shifted: Palette = { ...TWO_ARC, hueShiftDeg: 30 };
  for (let i = 0; i < 8; i++) {
    const base = hexToHsluv(paletteHex(TWO_ARC, i, '#808080', 0, 8, 4))[0];
    const now = hexToHsluv(paletteHex(shifted, i, '#808080', paletteHuePhase(shifted, 0), 8, 4))[0];
    assert.ok(Math.abs(wrapHue(now - base) - 30) < 0.6, `species ${i}: ${base} -> ${now}`);
  }
});

test('syncPaletteColors writes the two-arc colours it will actually render', () => {
  const p: Palette = { ...TWO_ARC, colors: [] };
  syncPaletteColors(p, 8, undefined, 4);
  assert.equal(p.colors.length, 8);
  for (let i = 0; i < 8; i++) {
    assert.equal(p.colors[i], paletteHex(p, i, '#808080', 0, 8, 4));
  }
});

// ── selectable colour space ──────────────────────────────────────────────────
// HSL, HSLuv and OKLCh are three different looks driven by the same two numbers
// (user call, 2026-08-13). The slider means "0 is grey, 100 is as colourful as
// this hue gets" in all three; what differs is how lightness behaves across hues.

test('every space round-trips a colour through its own coordinates', () => {
  for (const space of PALETTE_SPACES) {
    for (const hex of ['#ff8800', '#3366cc', '#12ef7a', '#884422', '#808080']) {
      const [h, s, l] = hexToSpace(space, hex);
      const back = spaceToHex(space, h, s, l);
      for (let i = 0; i < 3; i++) {
        const a = parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16);
        const b = parseInt(back.slice(1 + i * 2, 3 + i * 2), 16);
        assert.ok(Math.abs(a - b) <= 2, `${space}: ${hex} -> ${back} channel ${i}`);
      }
    }
  }
});

test('every space produces in-gamut colours across the whole wheel', () => {
  // The point of gamut mapping: OKLCh at a fixed high chroma asks for colours
  // sRGB cannot show, and must come back with ones it can rather than clipping
  // to a different hue.
  for (const space of PALETTE_SPACES) {
    for (let hue = 0; hue < 360; hue += 7) {
      const [r, g, b] = spaceToRgb(space, hue, 100, 62);
      for (const c of [r, g, b]) {
        assert.ok(c >= -1e-6 && c <= 1 + 1e-6, `${space} at hue ${hue}: ${r},${g},${b}`);
        assert.ok(Number.isFinite(c), `${space} at hue ${hue} produced ${c}`);
      }
    }
  }
});

test('the space is what makes lightness even or uneven across hues', () => {
  // This is the whole reason the choice is exposed. HSLuv equalises perceived
  // lightness by construction; HSL does not, and that spread IS its vividness.
  // Clamped before the power: gamut mapping can land a hair below zero, and
  // `Math.pow(-1e-9, 2.2)` is NaN rather than ~0.
  const lin = (c: number): number => Math.pow(Math.min(Math.max(c, 0), 1), 2.2);
  const relLum = (rgb: [number, number, number]): number =>
    0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2]);
  const spread = (space: 'hsl' | 'hsluv' | 'oklch'): number => {
    const lums = Array.from({ length: 12 }, (_, i) => relLum(spaceToRgb(space, i * 30, 100, 62)));
    return Math.max(...lums) / Math.max(Math.min(...lums), 1e-6);
  };
  const hsluv = spread('hsluv');
  const hsl = spread('hsl');
  assert.ok(hsluv < 1.6, `hsluv should be near-flat in luminance, got ${hsluv}x`);
  assert.ok(hsl > 4, `hsl should vary a lot in luminance, got ${hsl}x`);
});

test('an arc skips the 8-bit hex round trip on the render path', () => {
  // paletteRgb is a Float32Array bound to an rgba16float target, so the hex in
  // the middle was pure quantisation. Arc mode must not be rounded to 1/255.
  const p: Palette = {
    ...customPalette([]),
    mode: 'arc',
    space: 'oklch',
    arc: { hueStartDeg: 23, hueRangeDeg: 300, sat: 71, light: 57 },
  };
  let sawFinerThanHex = false;
  for (let i = 0; i < 6; i++) {
    const direct = paletteSrgb(p, i, '#808080', 0, 6);
    const viaHex = hexToLinear(paletteHex(p, i, '#808080', 0, 6));
    const directLinear = direct.map((c) => Math.pow(c, 2.2));
    for (let c = 0; c < 3; c++) {
      // Same colour to within a quantisation step...
      assert.ok(Math.abs((directLinear[c] as number) - (viaHex[c] as number)) < 0.02);
      // ...but not bit-identical, which is the quantisation we just removed.
      if (directLinear[c] !== viaHex[c]) sawFinerThanHex = true;
    }
  }
  assert.ok(sawFinerThanHex, 'the direct path should carry more precision than the hex');
});

test('custom mode at zero shift is still exactly the authored hex', () => {
  // The pixel-identical guarantee, restated against the float path: an authored
  // colour is ground truth, not something to re-derive.
  for (const space of PALETTE_SPACES) {
    const p: Palette = { ...customPalette(['#ff8800', '#3366cc']), space };
    assert.equal(paletteHex(p, 0, '#808080', 0), '#ff8800');
    assert.deepEqual(paletteLinear(p, 0, '#808080', 0), hexToLinear('#ff8800'));
  }
});
