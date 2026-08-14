import assert from 'node:assert/strict';
import { test } from 'node:test';

import { captureBrowserExportRecipe } from '../src/export/browser-recipe.ts';
import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import {
  applyVector,
  presetFromConfig,
  presetToVector,
} from '../src/sim/plife/preset.ts';

const output = {
  profile: 'av1-sdr-debug-2160p120',
  encoder: 'av1_nvenc',
  paperWhiteNits: 203,
  masteringPeakNits: 1000,
} as const;

function fixture(
  version = 'sha256-track',
  mode: 'manual' | 'modulated' = 'modulated',
): {
  live: ReturnType<typeof defaultPlifeConfig>;
  modulation: ReturnType<typeof defaultModulationConfig>;
  impulses: ReturnType<typeof defaultImpulseConfig>;
  base: Float64Array;
  applyTheta(theta: ArrayLike<number>): void;
  capture(): ReturnType<typeof captureBrowserExportRecipe>;
} {
  const live = defaultPlifeConfig();
  live.forceGain = 1.75;
  live.render.grade.exposureEv = 0.625;
  const modulation = defaultModulationConfig(live, 'plife');
  modulation.enabled = mode === 'modulated';
  modulation.extras = { stale: true };
  const impulses = defaultImpulseConfig();
  impulses.gain = 2.5;
  const base = presetToVector(presetFromConfig(live), live.speciesCount);
  let currentSeed = 9001;
  const source = {
    sim: {
      simId: 'plife',
      get currentSeed(): number {
        return currentSeed;
      },
      config: live,
      serializeExtras: (): Record<string, unknown> => ({
        macros: structuredClone(live.macros),
        notAutosavedYet: true,
      }),
    },
    modulator: {
      config: modulation,
      mode,
      baseValues: (): Float64Array => base.slice(),
      currentTheta: (): number[] => Array.from(
        presetToVector(presetFromConfig(live), live.speciesCount),
      ),
    },
    impulses: { config: impulses },
  };
  return {
    live,
    modulation,
    impulses,
    base,
    applyTheta: (theta) => applyVector(live, theta),
    capture: () => {
      currentSeed = 42;
      return captureBrowserExportRecipe({
        rendererBuild: 'browser-build-7',
        track: { id: 'pink-loop', version },
        source,
        output,
        currentPinState: (seed) => seed === 42,
      });
    },
  };
}

test('manual capture preserves current authored theta instead of the stale seeded base', () => {
  const f = fixture('sha256-track', 'manual');
  const authored = f.base.slice();
  authored[2] = authored[2] === 0.25 ? 0.75 : 0.25;
  f.applyTheta(authored);

  const recipe = f.capture();
  assert.deepEqual(recipe.modulationBase, Array.from(authored));
  assert.notEqual(recipe.modulationBase[2], f.base[2]);

  const changedAfterCapture = authored.slice();
  changedAfterCapture[2] = authored[2] === 0.5 ? 0.6 : 0.5;
  f.applyTheta(changedAfterCapture);
  assert.equal(recipe.modulationBase[2], authored[2], 'captured theta must not alias live state');
  assert.notEqual(f.base[2], authored[2], 'capture must not mutate the seeded base');
});

test('modulated capture preserves authored base instead of the current music excursion', () => {
  const f = fixture('sha256-track', 'modulated');
  const excursion = f.base.slice();
  excursion[2] = excursion[2] === 0.25 ? 0.75 : 0.25;
  f.applyTheta(excursion);

  const recipe = f.capture();
  assert.deepEqual(recipe.modulationBase, Array.from(f.base));
  assert.notEqual(recipe.modulationBase[2], excursion[2]);
});

test('browser capture reads current live identity, authored base, extras, and output policy', () => {
  const f = fixture();
  const recipe = f.capture();

  assert.equal(recipe.version, 4);
  assert.deepEqual(recipe.track, { id: 'pink-loop', contentVersion: 'sha256-track' });
  assert.equal(recipe.sim, 'plife');
  assert.equal(recipe.seed, 42);
  assert.equal(recipe.seedPinned, true);
  assert.equal(recipe.particleBudget, f.live.budget.cap);
  assert.deepEqual(recipe.modulationBase, Array.from(f.base));
  assert.deepEqual(recipe.modulation.extras, {
    macros: f.live.macros,
    notAutosavedYet: true,
  });
  assert.equal(recipe.simulation.forceGain, 1.75);
  assert.equal(recipe.impulses.gain, 2.5);
  assert.equal(recipe.render.grade.exposureEv, 0.625);
  assert.deepEqual(recipe.output, output);
  assert.equal(Object.hasOwn(recipe.simulation, 'render'), false);
  assert.equal(Object.hasOwn(recipe.modulation, 'render'), false);
  assert.deepEqual(f.modulation.extras, { stale: true }, 'capture must not refresh live autosave state');
});

test('captured state neither aliases nor mutates the live session', () => {
  const f = fixture();
  const recipe = f.capture();
  const captured = structuredClone(recipe);

  f.live.species[0]!.size = 0.123;
  f.live.palette.colors[0] = '#ffffff';
  f.live.render.grade.exposureEv = -3;
  f.modulation.driverGains[0] = 0;
  f.impulses.responses.kick.deposit = 9;
  f.base[0] = -999;
  assert.deepEqual(recipe, captured, 'live edits after capture must not change the recipe');

  recipe.simulation.species[0]!.brightness = 99;
  recipe.modulation.driverGains[0] = 0.25;
  recipe.impulses.responses.kick.flash = 99;
  recipe.render.grade.exposureEv = 4;
  recipe.modulationBase[0] = 123;
  assert.notEqual(f.live.species[0]!.brightness, 99);
  assert.notEqual(f.modulation.driverGains[0], 0.25);
  assert.notEqual(f.impulses.responses.kick.flash, 99);
  assert.equal(f.live.render.grade.exposureEv, -3);
  assert.equal(f.base[0], -999);
});

test('capture refuses a track without an exact content identity', () => {
  assert.throws(() => fixture('').capture(), /no content version/);
});
