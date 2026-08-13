import assert from 'node:assert/strict';
import { test } from 'node:test';

import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import {
  MAX_RECIPE_PARTICLE_BUDGET,
  parseExportRecipe,
  serializeExportRecipe,
  validateExportRecipe,
  type ExportRecipe,
} from '../src/runtime/recipe.ts';

function recipe(): ExportRecipe {
  const { render, ...simulation } = defaultPlifeConfig();
  const { render: _sharedRender, ...modulation } = defaultModulationConfig(
    { ...simulation, render },
    'plife',
  );
  return {
    version: 3,
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
  for (const profile of ['hevc-hdr10-1080p120', 'hevc-hdr10-2160p120']) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = profile;
    value.output['encoder'] = 'hevc_nvenc';
    assert.doesNotThrow(() => validateExportRecipe(value));
  }

  for (const invented of ['hdr10-1080p120', 'hdr10-2160p120', 'av1-hdr10-1080p120']) {
    const value = recipe() as unknown as { output: Record<string, unknown> };
    value.output['profile'] = invented;
    assert.throws(() => validateExportRecipe(value), /output\.profile.*unsupported/);
  }
});

test('a profile cannot be paired with an encoder that cannot produce it', () => {
  const sdrWithHevc = recipe() as unknown as { output: Record<string, unknown> };
  sdrWithHevc.output['encoder'] = 'hevc_nvenc';
  assert.throws(() => validateExportRecipe(sdrWithHevc), /output\.encoder.*does not match/);

  const hdrWithAv1 = recipe() as unknown as { output: Record<string, unknown> };
  hdrWithAv1.output['profile'] = 'hevc-hdr10-2160p120';
  assert.throws(() => validateExportRecipe(hdrWithAv1), /output\.encoder.*does not match/);
});

test('HDR luminance policy is explicit, bounded, and internally ordered', () => {
  const belowWhite = recipe();
  belowWhite.output.masteringPeakNits = belowWhite.output.paperWhiteNits - 1;
  assert.throws(() => validateExportRecipe(belowWhite), /masteringPeakNits.*at least/);

  const implausibleWhite = recipe();
  implausibleWhite.output.paperWhiteNits = 10;
  assert.throws(() => validateExportRecipe(implausibleWhite), /paperWhiteNits.*80\.\.1000/);
});
