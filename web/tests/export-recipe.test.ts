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
    version: 1,
    rendererBuild: 'test-build',
    track: { id: 'pink-loop', contentVersion: 'sha256-deadbeef' },
    sim: 'plife',
    seed: 0xffff_ffff,
    seedPinned: true,
    simulation,
    modulation,
    impulses: defaultImpulseConfig(),
    render,
    particleBudget: simulation.budget.cap,
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: 'hdr10-1080p120',
      encoder: 'hevc_nvenc',
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
  const unsupported = { ...recipe(), version: 2 };
  assert.throws(() => validateExportRecipe(unsupported), /unsupported version 2/);

  const sequence = recipe() as unknown as Record<string, unknown>;
  sequence['presentation'] = { mode: 'single', autoAdvance: true };
  assert.throws(() => validateExportRecipe(sequence), /autoAdvance.*must be false/);
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

test('HDR luminance policy is explicit, bounded, and internally ordered', () => {
  const belowWhite = recipe();
  belowWhite.output.masteringPeakNits = belowWhite.output.paperWhiteNits - 1;
  assert.throws(() => validateExportRecipe(belowWhite), /masteringPeakNits.*at least/);

  const implausibleWhite = recipe();
  implausibleWhite.output.paperWhiteNits = 10;
  assert.throws(() => validateExportRecipe(implausibleWhite), /paperWhiteNits.*80\.\.1000/);
});
