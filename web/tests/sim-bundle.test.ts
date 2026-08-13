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

const [{ buildSimBundle, buildSimBundleFromRecipe }, { defaultModulationConfig }, { defaultImpulseConfig }, { defaultPlifeConfig }, { TimelineSampler }] =
  await Promise.all([
    import('../src/runtime/sim-bundle.ts'),
    import('../src/mapping/persist.ts'),
    import('../src/sim/impulses.ts'),
    import('../src/sim/plife/config.ts'),
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
  const { render, ...simulation } = full;
  const configured = defaultModulationConfig(full, 'plife');
  configured.enabled = false;
  configured.extras = { macros: structuredClone(full.macros) };
  const { render: _render, ...modulation } = configured;
  const impulses = defaultImpulseConfig();
  impulses.gain = 2.25;
  return {
    version: 1,
    rendererBuild: 'test',
    track: { id: 'test', contentVersion: 'test-v1' },
    sim: 'plife',
    seed: 1234,
    seedPinned: true,
    simulation,
    modulation,
    impulses,
    render,
    particleBudget: simulation.budget.cap,
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: 'hdr10-1080p120',
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
