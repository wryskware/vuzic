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
import {
  defaultConfig,
  defaultPhysarumMacros,
  PHYSARUM_MACRO_LABELS,
  PHYSARUM_MACRO_RANGE,
  type PhysarumMacros,
} from '../src/sim/physarum/config.ts';
import {
  defaultPlifeConfig,
  defaultPlifeMacros,
  migrateLegacyMacros,
  MACRO_LABELS,
  MACRO_RANGE,
  MAX_EFFECTIVE_FRICTION,
  MAX_FRICTION,
  type PlifeMacros,
} from '../src/sim/plife/config.ts';
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

// ── the physarum end of the channel ──────────────────────────────────────────

/**
 * Physarum's macro rig, pinned at the only level this harness can reach.
 *
 * `PhysarumSim` itself is not importable here — `node --test` resolves the
 * specifiers in src/ literally, and the sim's imports are extensionless and
 * include `?raw` WGSL — which is why no test in this directory constructs a Sim.
 * What is testable, and what `serializeExtras` / `applyExtras` actually lean on,
 * is that the three macro tables agree: `applyExtras` walks the keys of
 * `defaultPhysarumMacros()` and indexes `PHYSARUM_MACRO_RANGE` for each, and the
 * panel walks `PHYSARUM_MACRO_LABELS` and indexes the same range table. A key
 * present in one table and missing from another is an undefined range — a
 * silently unclamped load, or a slider with no bounds — and that is the failure
 * this catches.
 */
test('the physarum macro tables are one source of truth, and neutral by default', () => {
  const def = defaultPhysarumMacros();
  const keys = Object.keys(def) as (keyof PhysarumMacros)[];

  // Every macro is neutral at 1.0 — the shipped config must be bit-identical to
  // a build with no macro rig at all, which is the whole premise of the layer.
  for (const key of keys) assert.equal(def[key], 1, `${key} must ship neutral`);
  assert.deepEqual(defaultConfig(4).macros, def);

  // Range table: exactly the same key set, and 1.0 inside every range, or "reset
  // to 1" would land outside the slider that is supposed to show it.
  assert.deepEqual(Object.keys(PHYSARUM_MACRO_RANGE).sort(), [...keys].sort());
  for (const key of keys) {
    const r = PHYSARUM_MACRO_RANGE[key];
    assert.ok(r.min < r.max, `${key}: empty range`);
    assert.ok(r.min <= 1 && 1 <= r.max, `${key}: the neutral value is outside its own slider`);
  }

  // Label table: same key set again, no duplicates, and every label states its
  // wiring — the labels are the only place the panel says what a macro touches.
  const labelled = PHYSARUM_MACRO_LABELS.map((l) => l.key);
  assert.equal(new Set(labelled).size, labelled.length, 'a macro is listed twice');
  assert.deepEqual([...labelled].sort(), [...keys].sort());
  for (const { label } of PHYSARUM_MACRO_LABELS) assert.match(label, /\(×/);
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

// ── plife's macro rig, and the agility split ─────────────────────────────────

/**
 * The plife side of the same three-table invariant physarum is pinned on above,
 * plus the migration that carries pre-2026-08-13 files across the `agility`
 * split.
 *
 * Why the split exists is measured, not stylistic: a particle readback at
 * shipped settings found 36–45% of live particles sitting at exactly the
 * `maxSpeed` ceiling, with the force/friction equilibrium ~4× the ceiling at the
 * median and ~40× at p99. The clamp renormalises velocity isotropically, so it
 * preserves heading and discards speed — every particle ends up moving at the
 * same rate. `agility` could not cure that at any setting, because it raised the
 * ceiling and lowered friction, and lowering friction raises the equilibrium by
 * the same factor. `speed` and `drag` move opposite sides of that inequality,
 * which is the whole point of separating them.
 */
test('the plife macro tables are one source of truth, and neutral by default', () => {
  const def = defaultPlifeMacros();
  const keys = Object.keys(def) as (keyof PlifeMacros)[];

  for (const key of keys) assert.equal(def[key], 1, `${key} must ship neutral`);
  assert.deepEqual(defaultPlifeConfig(8).macros, def);

  // The split is the point: two independent keys, and no survivor of the one
  // they replaced. A lingering `agility` would be a macro the panel shows and
  // nothing reads.
  assert.ok(keys.includes('speed'), 'speed macro missing');
  assert.ok(keys.includes('drag'), 'drag macro missing');
  assert.ok(!keys.includes('agility' as keyof PlifeMacros), 'agility must be gone');

  assert.deepEqual(Object.keys(MACRO_RANGE).sort(), [...keys].sort());
  for (const key of keys) {
    const r = MACRO_RANGE[key];
    assert.ok(r.min < r.max, `${key}: empty range`);
    assert.ok(r.min <= 1 && 1 <= r.max, `${key}: the neutral value is outside its own slider`);
  }

  // Both new knobs must reach past neutral by enough to matter. A `speed` that
  // stopped at 2 could not lift the ceiling clear of a 4× equilibrium, and a
  // `drag` that stopped at 2 could not pull the equilibrium down to the ceiling
  // — either bound would ship the knob and keep the bug.
  assert.ok(MACRO_RANGE.speed.max >= 4, 'speed cannot reach the measured equilibrium');
  assert.ok(MACRO_RANGE.drag.max >= 4, 'drag cannot reach the measured equilibrium');

  const labelled = MACRO_LABELS.map((l) => l.key);
  assert.equal(new Set(labelled).size, labelled.length, 'a macro is listed twice');
  assert.deepEqual([...labelled].sort(), [...keys].sort());
  for (const { label } of MACRO_LABELS) assert.match(label, /\(×/);
});

test('migrateLegacyMacros carries an agility file across the split exactly', () => {
  // The old wiring was "× maxSpeed, ÷ friction", so the inverse pair reproduces
  // both halves and the file replays at the look it was saved at.
  const out = migrateLegacyMacros({ density: 0.5, agility: 2 });
  assert.equal(out['speed'], 2);
  assert.equal(out['drag'], 0.5);
  assert.equal(out['density'], 0.5, 'unrelated macros must pass through untouched');

  // Pure: the caller's blob is not ours to write into.
  const src = { agility: 2 };
  migrateLegacyMacros(src);
  assert.deepEqual(src, { agility: 2 });

  // The current schema wins over a stale sibling key, in both directions.
  const both = migrateLegacyMacros({ agility: 2, speed: 1.5, drag: 3 });
  assert.equal(both['speed'], 1.5);
  assert.equal(both['drag'], 3);
  assert.equal(migrateLegacyMacros({ agility: 2, speed: 1.5 })['drag'], 0.5);

  // Nothing to migrate, nothing invented — these fall through to the defaults
  // that `applyExtras` supplies.
  for (const bad of [{}, { agility: 0 }, { agility: -1 }, { agility: 'fast' }, { agility: NaN }]) {
    const o = migrateLegacyMacros(bad as Record<string, unknown>);
    assert.equal(o['speed'], undefined, `${JSON.stringify(bad)} invented a speed`);
    assert.equal(o['drag'], undefined, `${JSON.stringify(bad)} invented a drag`);
  }
});

/**
 * The `drag` slider must reach its own top without the post-macro clamp eating
 * the end of it.
 *
 * This is the failure the split's first cut still had, and it is subtle in a way
 * that matters: at `drag` 6 against the old shared ceiling of 16, three of eight
 * species clipped — and they clipped to *exactly* 16, so the damper species all
 * landed on one identical friction. A clamp that compresses the per-species
 * spread is re-creating, one layer up, the very uniformity the split was made to
 * cure. `MAX_EFFECTIVE_FRICTION` is therefore a separate constant from the θ
 * slider's `MAX_FRICTION`: one bounds what the music may author, the other is
 * the rail behind the macro, and a rail must sit above the whole useful range.
 */
test('drag reaches the top of its slider without clipping a shipped species', () => {
  const cfg = defaultPlifeConfig(8);
  const top = MACRO_RANGE.drag.max;
  const worst = Math.max(...cfg.species.slice(0, cfg.speciesCount).map((s) => s.friction));

  assert.ok(worst > 0, 'no shipped species has friction to scale');
  assert.ok(
    worst * top <= MAX_EFFECTIVE_FRICTION,
    `drag ${top} clips the stiffest shipped species (${worst} × ${top} = ${worst * top} > ${MAX_EFFECTIVE_FRICTION})`,
  );

  // The two ceilings are deliberately different numbers doing different jobs.
  // Collapsing them back together is precisely the regression above.
  assert.ok(
    MAX_EFFECTIVE_FRICTION > MAX_FRICTION,
    'the post-macro rail must sit above the authored ceiling, not on it',
  );

  // …and the authored ceiling must still be reachable without the rail cutting
  // in first, or a θ value the slider offers could never actually be applied.
  assert.ok(MAX_FRICTION <= MAX_EFFECTIVE_FRICTION, 'an authored friction is unreachable');
});
