/** Shared runtime construction, exercised without a GPU. */
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && !/\.(ts|js|mjs|wgsl)(\?|$)/.test(specifier)) {
      try {
        return next(`${specifier}.ts`, context);
      } catch {
        // Not a TypeScript module; preserve the real resolution error below.
      }
    }
    return next(specifier, context);
  },
  load(url, context, next) {
    if (!url.includes('.wgsl')) return next(url, context);
    const text = readFileSync(fileURLToPath((url.split('?')[0] ?? url)), 'utf8');
    return { format: 'module', shortCircuit: true, source: `export default ${JSON.stringify(text)};` };
  },
});

const [{ buildSimBundle, buildSimBundleFromRecipe }, { defaultModulationConfig }, { defaultImpulseConfig }, { defaultPlifeConfig }, { PlifeSim }, { TimelineSampler }] =
  await Promise.all([
    import('../src/runtime/sim-bundle.ts'),
    import('../src/mapping/persist.ts'),
    import('../src/sim/impulses.ts'),
    import('../src/sim/plife/config.ts'),
    import('../src/sim/plife/plife.ts'),
    import('../src/timeline/sampler.ts'),
  ]);
import type { ExportRecipe } from '../src/runtime/recipe.ts';
import type { Timeline } from '../src/timeline/types.ts';

function sampler(): InstanceType<typeof TimelineSampler> {
  const channels = [
    { name: 'stems', dims: 4, offset: 0 },
    { name: 'novelty16', dims: 1, offset: 4 },
    { name: 'actChorus', dims: 1, offset: 5 },
  ];
  const timeline = {
    manifest: {
      version: 2,
      track: { id: 'test', duration: 1, sampleRate: 48_000 },
      grid: { hopSeconds: 0.1, frames: 2 },
      beats: [],
      downbeats: [],
      tempo: 120,
      segments: [],
      channels,
      events: [],
    },
    data: new Float32Array(12),
    stride: 6,
    channels: new Map(channels.map((channel) => [channel.name, channel])),
    events: [],
  } satisfies Timeline;
  return new TimelineSampler(timeline, 1 / 120);
}

function plifeRecipe(): ExportRecipe {
  const full = defaultPlifeConfig();
  full.forceGain = 1.73;
  full.macros.force = 1.41;
  full.palette.colors[0] = '#123456';
  full.render.grade.exposureEv = 0.75;
  const configured = defaultModulationConfig(full, 'plife');
  configured.enabled = false;
  configured.extras = { macros: structuredClone(full.macros) };
  const impulses = defaultImpulseConfig();
  impulses.gain = 2.25;
  const authored = buildSimBundle({
    id: 'plife',
    seed: 1234,
    sampler: sampler(),
    drivers: null,
    secondsPerTick: 1 / 120,
    simulationConfig: full,
    impulseConfig: impulses,
    resolveModulationConfig: () => configured,
  });
  const modulationBase = Array.from(authored.modulator.baseValues());
  const registry = authored.sim.registry();
  const editedIndex = registry.mask.findIndex((value, index) => value === 1 && registry.slots[index]);
  assert.ok(editedIndex >= 0, 'plife fixture has no modulated slot');
  const spec = registry.slots[editedIndex];
  assert.ok(spec);
  modulationBase[editedIndex] = spec.lo + (spec.hi - spec.lo) * 0.37;
  authored.modulator.restoreBase(modulationBase);
  // Capture while the displayed theta is elsewhere. The recipe's explicit base
  // must win over this transient excursion when the export begins at time zero.
  const excursion = authored.sim.currentVector();
  excursion[editedIndex] = spec.hi;
  authored.sim.applyTheta(excursion, registry.mask);

  const { render, ...simulation } = full;
  const modulationWithShared = structuredClone(authored.modulator.config);
  const { render: _render, impulses: _impulses, ...modulation } = modulationWithShared;
  return {
    version: 5,
    rendererBuild: 'test',
    track: { id: 'test', contentVersion: 'test-v1' },
    sim: 'plife',
    seed: 1234,
    seedPinned: true,
    simulation,
    modulation,
    modulationBase,
    impulses,
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

test('recipe construction applies complete authored state and reconnects shared references', () => {
  const recipe = plifeRecipe();
  const original = structuredClone(recipe);
  const bundle = buildSimBundleFromRecipe({
    recipe,
    sampler: sampler(),
    drivers: null,
    secondsPerTick: 1 / 120,
  });

  assert.equal(bundle.sim.simId, 'plife');
  assert.equal(bundle.sim.config.forceGain, 1.73);
  assert.equal(bundle.sim.config.macros.force, 1.41);
  assert.equal(bundle.impulses.config.gain, 2.25);
  assert.equal(bundle.modulator.config.enabled, false);
  assert.equal(bundle.modulator.config.palette, bundle.sim.config.palette);
  assert.equal(bundle.modulator.config.render, bundle.sim.config.render);
  assert.deepEqual(Array.from(bundle.modulator.baseValues()), recipe.modulationBase);
  const registry = bundle.sim.registry();
  const capturedConfig = {
    ...structuredClone(recipe.simulation),
    render: structuredClone(recipe.render),
  };
  const capturedTheta = new PlifeSim(recipe.seed, capturedConfig).currentVector();
  assert.ok(
    capturedTheta.some(
      (value, index) => registry.mask[index] === 1 && value !== recipe.modulationBase[index],
    ),
    'fixture did not capture a live excursion distinct from the authored base',
  );
  const live = bundle.sim.currentVector();
  for (let i = 0; i < registry.length; i++) {
    if (registry.mask[i] === 1) {
      assert.equal(live[i], recipe.modulationBase[i], `slot ${i} did not start on authored base`);
    }
  }
  assert.equal(bundle.sim.config.palette.colors[0], '#123456');
  assert.equal(bundle.sim.config.render.grade.exposureEv, 0.75);
  assert.deepEqual(recipe, original, 'runtime construction mutated its immutable recipe');
  assert.notEqual(bundle.sim.config, recipe.simulation);
  assert.notEqual(bundle.impulses.config, recipe.impulses);
});

test('generic construction rejects a concrete config for a different substrate', () => {
  const config = defaultPlifeConfig();
  assert.throws(
    () => buildSimBundle({
      id: 'physarum',
      seed: 1,
      sampler: sampler(),
      drivers: null,
      secondsPerTick: 1 / 120,
      simulationConfig: config,
      resolveModulationConfig: (sim) => defaultModulationConfig(sim.config, sim.simId),
    }),
    /physarum.*not a physarum config/,
  );
});

test('recipe construction rejects a base vector that does not fit the selected registry', () => {
  const recipe = plifeRecipe();
  recipe.modulationBase.pop();
  assert.throws(
    () => buildSimBundleFromRecipe({
      recipe,
      sampler: sampler(),
      drivers: null,
      secondsPerTick: 1 / 120,
    }),
    /modulation base has .* expected/,
  );
});

test('browser-compatible construction still omits configs and receives defaults', () => {
  const bundle = buildSimBundle({
    id: 'plife',
    seed: 1,
    sampler: sampler(),
    drivers: null,
    secondsPerTick: 1 / 120,
    resolveModulationConfig: (sim) => defaultModulationConfig(sim.config, sim.simId),
  });
  assert.equal(bundle.sim.config.forceGain, defaultPlifeConfig().forceGain);
  assert.equal(bundle.impulses.config.gain, defaultImpulseConfig().gain);
});
