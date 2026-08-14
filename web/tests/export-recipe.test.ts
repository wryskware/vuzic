import assert from 'node:assert/strict';
import { test } from 'node:test';

import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import { requiredEncoder } from '../src/export/profiles.ts';
import {
  EXPORT_PROFILES,
  MAX_RECIPE_PARTICLE_BUDGET,
  parseExportRecipe,
  serializeExportRecipe,
  validateExportRecipe,
  type ExportRecipe,
} from '../src/runtime/recipe.ts';

function recipe(): ExportRecipe {
  const { render, ...simulation } = defaultPlifeConfig();
  // `render` and `impulses` are both live objects the recipe encodes once, at
  // its top level, so the modulation block carries neither. See
  // `RecipeModulationConfig`.
  const {
    render: _sharedRender,
    impulses: _sharedImpulses,
    ...modulation
  } = defaultModulationConfig({ ...simulation, render }, 'plife');
  return {
    version: 5,
    rendererBuild: 'test-build',
    track: { id: 'pink-loop', contentVersion: 'sha256-deadbeef' },
    sim: 'plife',
    seed: 0xffff_ffff,
    seedPinned: true,
    simulation,
    modulation,
    modulationBase: [1, 2, 3],
    impulses: defaultImpulseConfig(),
    render,
    particleBudget: simulation.budget.cap,
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: 'av1-sdr-debug-1080p120',
      encoder: 'av1_nvenc',
      paperWhiteNits: 203,
      masteringPeakNits: 1000,
    },
  };
}

test('recipe canonical serialization round-trips the complete concrete state', () => {
  const value = recipe();
  const text = serializeExportRecipe(value);
  const parsed = parseExportRecipe(text);
  assert.deepEqual(parsed, value);
  assert.equal(serializeExportRecipe(parsed), text);
  assert.ok(text.endsWith('\n'));
});

test('unsupported recipe versions and mutable sequence behavior are rejected', () => {
  for (const version of [1, 2]) {
    const unsupported = { ...recipe(), version };
    assert.throws(
      () => validateExportRecipe(unsupported),
      new RegExp(`unsupported version ${version}`),
    );
  }

  const sequence = recipe() as unknown as Record<string, unknown>;
  sequence['presentation'] = { mode: 'single', autoAdvance: true };
  assert.throws(() => validateExportRecipe(sequence), /autoAdvance.*must be false/);
});

test('v3 requires a finite explicit authored modulation base', () => {
  const missing = recipe() as unknown as Record<string, unknown>;
  delete missing['modulationBase'];
  assert.throws(() => validateExportRecipe(missing), /modulationBase.*required/);

  const empty = recipe();
  empty.modulationBase = [];
  assert.throws(() => validateExportRecipe(empty), /modulationBase.*non-empty/);

  const nonFinite = recipe();
  nonFinite.modulationBase[1] = Number.NaN;
  assert.throws(() => validateExportRecipe(nonFinite), /modulationBase\[1\].*finite/);
});

test('identity, concrete config, and modulation discriminator must agree', () => {
  const mismatch = recipe();
  mismatch.modulation.sim = 'physarum';
  assert.throws(() => validateExportRecipe(mismatch), /modulation\.sim.*must match/);

  const missing = recipe() as unknown as Record<string, unknown>;
  missing['simulation'] = {};
  assert.throws(() => validateExportRecipe(missing), /simulation\.speciesCount/);

  const partial = recipe() as unknown as { modulation: Record<string, unknown> };
  delete partial.modulation['driverGains'];
  assert.throws(() => validateExportRecipe(partial), /modulation\.driverGains.*required/);

  const unknown = recipe() as unknown as Record<string, unknown>;
  unknown['sim'] = 'not-a-real-visual';
  assert.throws(() => validateExportRecipe(unknown), /sim.*unsupported/);

  const paletteMismatch = recipe();
  paletteMismatch.modulation.palette = structuredClone(paletteMismatch.modulation.palette);
  paletteMismatch.modulation.palette.colors[0] = '#123456';
  assert.throws(
    () => validateExportRecipe(paletteMismatch),
    /modulation\.palette.*must match.*simulation\.palette/,
  );
});

test('malformed and unbounded primitives are rejected instead of coerced', () => {
  const infinite = serializeExportRecipe(recipe()).replace('"seed": 4294967295', '"seed": 1e400');
  assert.throws(() => parseExportRecipe(infinite), /seed.*finite number/);

  const oversized = recipe();
  oversized.particleBudget = MAX_RECIPE_PARTICLE_BUDGET + 1;
  assert.throws(() => validateExportRecipe(oversized), /particleBudget.*finite number/);

  const dropped = recipe() as unknown as { simulation: Record<string, unknown> };
  dropped.simulation['notSerializable'] = undefined;
  assert.throws(() => validateExportRecipe(dropped), /notSerializable.*not a JSON value/);
});

test('output choices are bounded enums, not an encoder argument surface', () => {
  const value = recipe() as unknown as { output: Record<string, unknown> };
  value.output['encoder'] = '- arbitrary ffmpeg args';
  assert.throws(() => validateExportRecipe(value), /output\.encoder.*unsupported/);
});

test('only profiles whose id matches their real transport are accepted', () => {
  for (const profile of ['av1-sdr-debug-1080p120', 'av1-sdr-debug-2160p120']) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = profile;
    assert.doesNotThrow(() => validateExportRecipe(value));
  }

  // The HDR10 profiles are real, and they are the only ones allowed to say so.
  for (const profile of ['av1-hdr10-1080p120', 'av1-hdr10-2160p120']) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = profile;
    value.output['encoder'] = 'av1_nvenc';
    assert.doesNotThrow(() => validateExportRecipe(value));
  }

  // The last two are the retired HEVC ids. They are rejected rather than
  // aliased: a recipe is a reproduction contract, and quietly re-pointing an old
  // id at a different codec would make it lie about what it produced.
  for (const invented of [
    'hdr10-1080p120',
    'hdr10-2160p120',
    'hevc-hdr10-1080p120',
    'hevc-hdr10-2160p120',
  ]) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = invented;
    assert.throws(() => validateExportRecipe(value), /output\.profile.*unsupported/);
  }
});

test('a profile cannot be paired with an encoder that cannot produce it', () => {
  // Every shipping profile is AV1, so no *in-enum* mismatch is constructible
  // today; the pairing rule stays because it is what keeps the encoder a
  // property of the profile rather than an independent browser choice.
  for (const profile of EXPORT_PROFILES) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = profile;
    value.output['encoder'] = requiredEncoder(profile);
    assert.doesNotThrow(() => validateExportRecipe(value));
  }

  const retiredEncoder = recipe() as unknown as { output: Record<string, unknown> };
  retiredEncoder.output['encoder'] = 'hevc_nvenc';
  assert.throws(() => validateExportRecipe(retiredEncoder), /output\.encoder.*unsupported/);
});

test('HDR luminance policy is explicit, bounded, and internally ordered', () => {
  const belowWhite = recipe();
  belowWhite.output.masteringPeakNits = belowWhite.output.paperWhiteNits - 1;
  assert.throws(() => validateExportRecipe(belowWhite), /masteringPeakNits.*at least/);

  const implausibleWhite = recipe();
  implausibleWhite.output.paperWhiteNits = 10;
  assert.throws(() => validateExportRecipe(implausibleWhite), /paperWhiteNits.*80\.\.1000/);
});

// ── palette v2 / recipe v4+ ──────────────────────────────────────────────────

test('the recipe validates the palette v2 shape strictly, field by field', () => {
  // The whole shape round-trips as-is.
  assert.doesNotThrow(() => validateExportRecipe(recipe()));

  const arc = recipe();
  arc.simulation.palette.mode = 'arc';
  arc.modulation.palette.mode = 'arc';
  assert.doesNotThrow(() => validateExportRecipe(arc), 'arc mode is a legal palette');

  const badMode = recipe() as unknown as { simulation: { palette: Record<string, unknown> } };
  badMode.simulation.palette['mode'] = 'gradient';
  assert.throws(() => validateExportRecipe(badMode), /palette\.mode.*must be "arc" or "custom"/);

  const badLight = recipe();
  badLight.simulation.palette.arc.light = 140;
  assert.throws(() => validateExportRecipe(badLight), /palette\.arc\.light.*0\.\.100/);

  const missingArc = recipe() as unknown as { simulation: { palette: Record<string, unknown> } };
  delete missingArc.simulation.palette['arc'];
  assert.throws(() => validateExportRecipe(missingArc), /palette\.arc.*required/);

  const wildRate = recipe();
  wildRate.simulation.palette.hueRateDegPerSec = 1e6;
  assert.throws(() => validateExportRecipe(wildRate), /palette\.hueRateDegPerSec/);
});

test('a current recipe carrying a v1 palette block is rejected, not defaulted', () => {
  const legacyBlock = recipe() as unknown as { simulation: { palette: unknown } };
  legacyBlock.simulation.palette = { colors: ['#ffffff'], saturation: 1, brightness: 1 };
  assert.throws(
    () => validateExportRecipe(legacyBlock),
    /simulation\.palette\.mode.*required/,
    'the migration seam is the version lift, not the validator',
  );
});

test('the sim/modulation palette agreement covers every v2 field', () => {
  for (const mutate of [
    (p: ExportRecipe['modulation']['palette']): void => {
      p.mode = 'arc';
    },
    (p: ExportRecipe['modulation']['palette']): void => {
      p.arc.hueStartDeg = 137;
    },
    (p: ExportRecipe['modulation']['palette']): void => {
      p.hueShiftDeg = 12;
    },
    (p: ExportRecipe['modulation']['palette']): void => {
      p.hueRateDegPerSec = 0.5;
    },
  ]) {
    const value = recipe();
    value.modulation.palette = structuredClone(value.modulation.palette);
    mutate(value.modulation.palette);
    assert.throws(
      () => validateExportRecipe(value),
      /modulation\.palette.*must match.*simulation\.palette/,
    );
  }
});

/** Strip the fields a v4 sidecar could not have carried: the wiggle depths. */
function asV4(value: Record<string, unknown>): Record<string, unknown> {
  const impulses = value['impulses'] as { responses: Record<string, Record<string, unknown>> };
  for (const response of Object.values(impulses.responses)) delete response['wiggle'];
  value['version'] = 4;
  return value;
}

test('a v3 recipe still parses: the palette lift preserves its meaning exactly', () => {
  // A real v3 recipe is this one with the palette blocks written the old way, no
  // wiggle depths, and the embedded modulation config still at v4 — the exact
  // bytes the previous build wrote into a sidecar.
  const value = asV4(recipe() as unknown as Record<string, unknown>);
  value['version'] = 3;
  const v1Palette = {
    colors: (value['simulation'] as { palette: { colors: string[] } }).palette.colors.slice(),
    saturation: 0.83,
    brightness: 1.14,
  };
  (value['simulation'] as Record<string, unknown>)['palette'] = structuredClone(v1Palette);
  const modulation = value['modulation'] as Record<string, unknown>;
  modulation['palette'] = structuredClone(v1Palette);
  modulation['version'] = 4;

  const parsed = parseExportRecipe(JSON.stringify(value));
  assert.equal(parsed.version, 5, 'a v3 sidecar lifts all the way through v4');
  assert.equal(parsed.modulation.version, 5);
  for (const palette of [parsed.simulation.palette, parsed.modulation.palette]) {
    assert.equal(palette.mode, 'custom', 'a v1 palette was always a custom one');
    assert.equal(palette.hueShiftDeg, 0);
    assert.equal(palette.hueRateDegPerSec, 0, 'an old recipe cannot have acquired a cycle');
    assert.deepEqual(palette.colors, v1Palette.colors);
    assert.equal(palette.saturation, 0.83);
    assert.equal(palette.brightness, 1.14);
  }
  // The agreement check still holds after the lift, in both directions.
  assert.doesNotThrow(() => validateExportRecipe(parsed));
});

test('the v3 lift touches nothing but the palette', () => {
  // The expectation is the current recipe with every wiggle depth at 0: that is
  // what the v4 → v5 hop fills in, and it is the *only* thing it fills in.
  const expected = recipe();
  for (const response of Object.values(expected.impulses.responses)) response.wiggle = 0;
  const v3 = asV4(structuredClone(expected) as unknown as Record<string, unknown>);
  v3['version'] = 3;
  const strip = (block: Record<string, unknown>): void => {
    const p = block['palette'] as Record<string, unknown>;
    block['palette'] = {
      colors: p['colors'],
      saturation: p['saturation'],
      brightness: p['brightness'],
    };
  };
  strip(v3['simulation'] as Record<string, unknown>);
  const mod = v3['modulation'] as Record<string, unknown>;
  strip(mod);
  mod['version'] = 4;

  const parsed = parseExportRecipe(JSON.stringify(v3));
  assert.deepEqual(parsed, expected, 'a v3 recipe lifts to exactly the recipe it would have been');
});

test('the v4 lift fills the wiggle lane with 0, not with the shipped default', () => {
  // The reproduction-contract rule, pinned: a recipe written before the lane
  // existed describes a render that had no wiggle in it, so replaying it must
  // not acquire one. `defaultImpulseConfig()` ships non-zero depths, which is
  // what makes this assertion able to fail.
  const authored = recipe();
  assert.ok(
    Object.values(authored.impulses.responses).some((r) => r.wiggle > 0),
    'the shipped responses must have a non-zero wiggle for this test to mean anything',
  );

  const v4 = asV4(structuredClone(authored) as unknown as Record<string, unknown>);
  const parsed = parseExportRecipe(JSON.stringify(v4));
  assert.equal(parsed.version, 5);
  for (const [kind, response] of Object.entries(parsed.impulses.responses)) {
    assert.equal(response.wiggle, 0, `${kind} acquired a wiggle it was never rendered with`);
  }
  // And nothing else moved on the way through.
  const stripped = structuredClone(parsed) as unknown as Record<string, unknown>;
  assert.deepEqual(asV4(stripped), { ...v4, version: 4 });
});
