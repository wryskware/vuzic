/**
 * The opaque per-sim `extras` channel, and the wiring readout it shares a round
 * with.
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads modules straight
 * out of src/, so every import carries an explicit `.ts` extension, and nothing
 * here touches the DOM or WebGPU.
 *
 * What these pin:
 *
 * - **`extras` survives a round trip and nothing else does.** The mapping layer
 *   promises exactly one thing about the block — that it is a plain object — and
 *   the point of the tests below is that it promises nothing *more*: the parser
 *   must not inspect, coerce or reshape the contents, because the moment it does,
 *   `ModulationConfig` has learned a substrate's schema and the seam is gone.
 * - **Absent stays absent.** `extras?: …` with `exactOptionalPropertyTypes`, so a
 *   file from a sim that stores nothing must not grow a null key.
 * - **`groupDriverWeights` reports the wiring it is given.** The Modulator is
 *   driven here through a stub `ModTarget` and a stub sampler — the class needs
 *   no GPU, only the two seams — so the shares are checked against a registry
 *   whose groups are known by construction.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { parseModulation, serializeModulation, defaultModulationConfig } from '../src/mapping/persist.ts';
import { DriverBank, Modulator } from '../src/mapping/modulation.ts';
import { MOD_GROUPS, type ModGroup, type ModSpec } from '../src/mapping/modspec.ts';
import type { ModTarget, ThetaRegistry } from '../src/mapping/target.ts';
import type { TimelineSampler } from '../src/timeline/sampler.ts';
import { defaultRenderConfig } from '../src/sim/render/config.ts';
import type { ModulationConfig } from '../src/mapping/types.ts';

// ── a v4 file, as text, with whatever extras the caller wants ────────────────

function fileWith(extras?: unknown): string {
  const base: Record<string, unknown> = {
    version: 4,
    sim: 'plife',
    speciesCount: 8,
    enabled: true,
    depth: 1,
    driverGains: [1, 0.5],
  };
  if (extras !== undefined) base['extras'] = extras;
  return JSON.stringify(base);
}

test('parseModulation round-trips an extras object untouched', () => {
  const extras = {
    macros: { density: 1.4, force: 0.6, trails: 0.2 },
    matrixGen: { sigma: 0.7, rMin: { lo: 0.003, hi: 0.005 } },
    // A key no sim has ever heard of: the channel is opaque, so it survives too.
    somethingElse: ['a', 2, true],
  };
  const cfg = parseModulation(fileWith(extras));
  assert.deepEqual(cfg.extras, extras);

  // …and back out again through the serialiser, which is the path the workbench
  // download button and the autosave both take.
  const again = parseModulation(serializeModulation(cfg));
  assert.deepEqual(again.extras, extras);
});

test('parseModulation drops a non-object extras', () => {
  for (const bad of [[1, 2, 3], 'nope', 42, null, true]) {
    const cfg = parseModulation(fileWith(bad));
    assert.equal(cfg.extras, undefined, `extras: ${JSON.stringify(bad)} should be dropped`);
    assert.equal('extras' in cfg, false, 'the key must be absent, not undefined');
  }
});

test('an absent extras block stays absent', () => {
  const cfg = parseModulation(fileWith());
  assert.equal('extras' in cfg, false);
  // A file written from it must not carry a null key either.
  assert.equal(JSON.parse(serializeModulation(cfg))['extras'], undefined);
});

test('a legacy file cannot smuggle extras through the v1/v2 lift', () => {
  const text = JSON.stringify({ version: 2, speciesCount: 4, extras: { macros: { density: 2 } } });
  const cfg = parseModulation(text);
  assert.equal('extras' in cfg, false);
});

// ── groupDriverWeights ───────────────────────────────────────────────────────

/**
 * Four slots, one per group, all modulated. That makes each group's aggregate
 * exactly one row of `w`, so the expected shares are computable by hand from the
 * seeded direction — which is what makes this a test of the *aggregation* rather
 * than a restatement of `unitDirection`.
 */
function stubRegistry(): ThetaRegistry {
  const spec = (group: ModGroup): ModSpec => ({
    group,
    lo: 0,
    hi: 2,
    half: 0.5,
    jitter: 0.1,
    mult: false,
  });
  return {
    length: MOD_GROUPS.length,
    slots: MOD_GROUPS.map(spec),
    mask: Uint8Array.from(MOD_GROUPS.map(() => 1)),
    classes: new Uint8Array(MOD_GROUPS.length),
    names: MOD_GROUPS.map((g) => `slot.${g}`),
  };
}

function stubTarget(): ModTarget {
  const registry = stubRegistry();
  return {
    simId: 'stub',
    config: {
      speciesCount: 2,
      palette: { colors: ['#ffffff', '#000000'], saturation: 1, brightness: 1 },
      render: defaultRenderConfig(),
      species: [
        { name: 'a', brightness: 1 },
        { name: 'b', brightness: 1 },
      ],
    },
    currentSeed: 1,
    onSeedChange: null,
    registry: () => registry,
    currentVector: () => new Float64Array(registry.length).fill(1),
    applyTheta: () => {},
    invalidatePalette: () => {},
    setBrightFollow: () => {},
    setImpulses: () => {},
    stemMap: () => Int32Array.from([-1, -1]),
    partialReseed: () => {},
    reseed: () => {},
    snapshot: () => false,
    restoreSnapshot: () => false,
    clearSnapshot: () => {},
    hasSnapshot: false,
  };
}

/** The only two members the Modulator touches on the sampler. */
const stubSampler = {
  getChannel: () => undefined,
  segmentIndexAt: () => -1,
} as unknown as TimelineSampler;

/** Three named drivers over a handful of frames; the values only have to vary. */
function stubBank(): DriverBank {
  const frames = 8;
  const columns = ['novelty·16bar', 'chorus-ness', 'pc-1'].map((name, d) => ({
    name,
    source: -1,
    read: (f: number) => Math.sin((f + 1) * (d + 1)),
  }));
  return new DriverBank(columns, frames, 0.1, 'stub');
}

function makeModulator(bank: DriverBank | null): Modulator {
  const target = stubTarget();
  const cfg: ModulationConfig = defaultModulationConfig(target.config, 'stub');
  return new Modulator(target, stubSampler, bank, cfg, 12345);
}

test('groupDriverWeights returns one normalised row per group', () => {
  const mod = makeModulator(stubBank());
  const rows = mod.groupDriverWeights();
  assert.equal(rows.length, MOD_GROUPS.length);
  assert.deepEqual(rows.map((r) => r.group), [...MOD_GROUPS]);
  for (const row of rows) {
    assert.equal(row.top.length, 3, 'three drivers, so topN=3 fills the line');
    const sum = row.top.reduce((a, t) => a + t.share, 0);
    // Every driver is in the top-3 here, so the shares must account for all of it.
    assert.ok(Math.abs(sum - 1) < 1e-9, `shares should sum to 1, got ${sum}`);
    // Descending by share is what makes "the first name is the one that matters".
    for (let i = 1; i < row.top.length; i++) {
      assert.ok((row.top[i - 1] as { share: number }).share >= (row.top[i] as { share: number }).share);
    }
    for (const t of row.top) assert.ok(['novelty·16bar', 'chorus-ness', 'pc-1'].includes(t.name));
  }
});

test('groupDriverWeights honours topN and the driver gains', () => {
  const mod = makeModulator(stubBank());
  assert.equal((mod.groupDriverWeights(1)[0] as { top: unknown[] }).top.length, 1);

  // Mute two of the three: the survivor must own 100% of every group, which is
  // the property that makes the readout useful for debugging a mute.
  mod.setDriverGain(0, 0);
  mod.setDriverGain(1, 0);
  for (const row of mod.groupDriverWeights()) {
    const top = row.top[0] as { name: string; share: number };
    assert.equal(top.name, 'pc-1');
    assert.ok(Math.abs(top.share - 1) < 1e-9);
  }

  // …and with everything muted there is nothing to report rather than a NaN.
  mod.setAllDriverGains(0);
  for (const row of mod.groupDriverWeights()) assert.equal(row.top.length, 0);
});

test('groupDriverWeights is empty when there are no drivers at all', () => {
  assert.deepEqual(makeModulator(null).groupDriverWeights(), []);
});
