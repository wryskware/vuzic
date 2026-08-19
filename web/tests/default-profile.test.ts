/**
 * The shipped default look, checked against the schema that has to accept it.
 *
 * `runtime/default-profile.ts` parses these same bytes at module load, so
 * without this the first thing to discover a default gone stale against a recipe
 * bump would be a visitor's blank page. The recipe format has moved four times
 * already (v3 through v6, each with a lift), so "it parsed when it was exported"
 * is not a property that stays true on its own.
 *
 * Read from disk rather than imported, because the app imports it `?raw` — a
 * Vite suffix bare `node --test` knows nothing about. Same bytes either way,
 * which is the point.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { EXPORT_RECIPE_VERSION, parseExportRecipe } from '../src/runtime/recipe.ts';
import { PALETTE_CATALOG } from '../src/sim/palettes.ts';

const json = readFileSync(
  fileURLToPath(new URL('../src/runtime/default-profile.json', import.meta.url)),
  'utf8',
);

test('the shipped default profile parses as a current export recipe', () => {
  const recipe = parseExportRecipe(json);
  assert.equal(recipe.version, EXPORT_RECIPE_VERSION);
  assert.equal(recipe.sim, 'plife');
});

test('the default profile pins its seed, so every first visit sees the same world', () => {
  // Not incidental. Most seeds are bad — that is the premise of the whole
  // favorites pool — so a demo that rerolled on arrival would show a stranger a
  // world nobody vetted. If this ever flips to false it should be a decision,
  // not a diff nobody noticed.
  const recipe = parseExportRecipe(json);
  assert.equal(recipe.seedPinned, true);
  assert.equal(typeof recipe.seed, 'number');
});

test('the default profile names a track the publish allowlist ships', () => {
  // `main.ts` starts the default look on its own track, and `pick` would quietly
  // fall through to the fallback if that track were not in the build.
  const recipe = parseExportRecipe(json);
  const allowlist = JSON.parse(
    readFileSync(fileURLToPath(new URL('../../data/publish.json', import.meta.url)), 'utf8'),
  ) as { tracks: string[] };
  assert.ok(
    allowlist.tracks.includes(recipe.track.id),
    `default profile plays "${recipe.track.id}", which data/publish.json does not ship`,
  );
});

test('the wrysk palette entry reproduces the default profile, shift included', () => {
  // A palette entry resets `hueShiftDeg` to 0 by contract, and `colors` in a
  // recipe are the base hues *before* the shift. So the entry's arc must equal
  // the profile's arc rotated by the profile's own shift, or picking "wrysk"
  // from the dropdown shows different colours than arriving at the default does.
  const recipe = parseExportRecipe(json);
  const palette = (recipe.simulation as { palette: Record<string, never> }).palette as unknown as {
    arc: { hueStartDeg: number; hueRangeDeg: number; sat: number; light: number };
    accentArc: { hueStartDeg: number; hueRangeDeg: number; sat: number; light: number };
    hueShiftDeg: number;
  };
  const wrysk = PALETTE_CATALOG.find((entry) => entry.name === 'wrysk');
  assert.ok(wrysk, 'the wrysk entry is missing from the catalog');
  assert.ok(wrysk.arc && wrysk.accentArc, 'wrysk should carry both arcs');

  const shifted = (start: number): number => (start + palette.hueShiftDeg) % 360;
  assert.equal(wrysk.arc.hueStartDeg, shifted(palette.arc.hueStartDeg));
  assert.equal(wrysk.arc.hueRangeDeg, palette.arc.hueRangeDeg);
  assert.equal(wrysk.arc.sat, palette.arc.sat);
  assert.equal(wrysk.arc.light, palette.arc.light);
  assert.equal(wrysk.accentArc.hueStartDeg, shifted(palette.accentArc.hueStartDeg));
  assert.equal(wrysk.accentArc.hueRangeDeg, palette.accentArc.hueRangeDeg);
  assert.equal(wrysk.accentArc.light, palette.accentArc.light);
});
