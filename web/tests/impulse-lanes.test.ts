/**
 * The two lanes roadmap phase 1 item 2 added to the impulse engine:
 *
 *   **wiggle** — an event on a stem displaces that species' row of the
 *   interaction matrix, and the row springs back as the envelope decays.
 *
 * Plus **hold**, the tuning surface's isolation control, which pins one kind's
 * envelope open so a lane can be looked at as a state rather than as a 200 ms
 * transient shared with four other lanes.
 *
 * Three things are worth pinning, and they are all pure — no DOM, no WebGPU:
 *
 *   1. the direction vectors are a function of (seed, roll, K) and of nothing
 *      else, so a pinned seed reproduces a run and the reroll button is the only
 *      thing that changes what a hit does;
 *   2. the **partition survives the lane**. This is the load-bearing one. The
 *      primary/secondary split is only a split as long as the uncoupled cells
 *      are exactly 0 (see `coupled()` and the header of plife/config.ts), and a
 *      lane that fires on every kick is the most effective way imaginable to
 *      erode it;
 *   3. the depth knob means the same thing on every seed, which is what the
 *      `max|A|` normalisation buys and what a test can actually check.
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads modules straight
 * out of src/, so every import carries an explicit `.ts` extension.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { seedWiggleDirs, wiggleAttraction, WIGGLE_LIMIT } from '../src/sim/plife/genmatrix.ts';
import { coupled } from '../src/sim/plife/preset.ts';
import {
  defaultMatrixGen,
  defaultPlifeConfig,
  defaultPlifeMacros,
  MACRO_RANGE,
  PLIFE_BLOCKS,
  PRIMARY_COUNT,
} from '../src/sim/plife/config.ts';
import { blockRules, persistedBlockDecls } from '../src/mapping/blocks.ts';
import {
  defaultImpulseConfig,
  ImpulseEngine,
  applyImpulseConfig,
  MAX_WIGGLE,
  type ImpulseConfig,
} from '../src/sim/impulses.ts';
import { EVENT_KINDS, type TimelineEvent } from '../src/timeline/types.ts';
import { SECONDS_PER_TICK } from '../src/timing.ts';

const K = 8;

/** A matrix whose cells are all distinct and non-zero on the coupled set. */
function testMatrix(k = K, scale = 1): number[] {
  const a = new Array<number>(k * k).fill(0);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      if (coupled(i, j, k)) a[i * k + j] = scale * (0.1 + 0.01 * (i * k + j));
    }
  }
  return a;
}

/** Envelope with `value` on species `species` and 0 everywhere else. */
function envelopeOn(species: number, value: number, k = K): Float32Array {
  const env = new Float32Array(k);
  env[species] = value;
  return env;
}

// ── the direction vectors ────────────────────────────────────────────────────

test('wiggle directions are a pure function of (seed, roll, K)', () => {
  const a = seedWiggleDirs(1234, 0, K);
  assert.deepEqual(Array.from(seedWiggleDirs(1234, 0, K)), Array.from(a), 'not reproducible');

  const rerolled = seedWiggleDirs(1234, 1, K);
  assert.notDeepEqual(Array.from(rerolled), Array.from(a), 'the reroll changed nothing');

  const reseeded = seedWiggleDirs(1235, 0, K);
  assert.notDeepEqual(Array.from(reseeded), Array.from(a), 'a new seed kept the old directions');

  // And a reroll is reversible: the counter is a key, not an accumulator.
  assert.deepEqual(Array.from(seedWiggleDirs(1234, 0, K)), Array.from(a));
});

test('directions are drawn from the matrix stream, not correlated with it', () => {
  // The domain separator earning its keep: if the wiggle reused KEY_MATRIX the
  // direction of a row would track the row's own drawn values, and every hit
  // would read as a volume knob on the matrix rather than as new rules. A
  // correlation test would be flaky; what is checkable is the weaker, sufficient
  // property that the two are not the same numbers.
  const dirs = seedWiggleDirs(7, 0, K);
  const cfg = defaultPlifeConfig(K);
  let identical = 0;
  for (let cell = 0; cell < K * K; cell++) {
    if ((dirs[cell] as number) !== 0 && dirs[cell] === cfg.attraction[cell]) identical++;
  }
  assert.equal(identical, 0);
});

test('only primary rows move, and only on their own coupled cells', () => {
  for (const seed of [0, 1, 42, 0xffff_ffff]) {
    const dirs = seedWiggleDirs(seed, 3, K);
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const d = dirs[i * K + j] as number;
        if (i >= PRIMARY_COUNT) {
          assert.equal(d, 0, `secondary row ${i} has a direction at column ${j}`);
        } else if (!coupled(i, j, K)) {
          assert.equal(d, 0, `uncoupled cell (${i},${j}) is in the wiggle's support`);
        }
      }
    }
  }
});

test('every primary row is a unit vector, so depth means one thing', () => {
  for (const k of [4, 6, 8]) {
    const dirs = seedWiggleDirs(99, 0, k);
    for (let i = 0; i < Math.min(PRIMARY_COUNT, k); i++) {
      let sumSq = 0;
      for (let j = 0; j < k; j++) sumSq += (dirs[i * k + j] as number) ** 2;
      assert.ok(
        Math.abs(Math.sqrt(sumSq) - 1) < 1e-6,
        `K=${k} row ${i} has norm ${Math.sqrt(sumSq)}`,
      );
    }
  }
});

// ── the perturbation ─────────────────────────────────────────────────────────

test('an idle lane writes θ through byte for byte', () => {
  const attraction = testMatrix();
  const out = new Float32Array(K * K);
  const dirs = seedWiggleDirs(5, 0, K);

  for (const [depth, env] of [
    [1, new Float32Array(K)],
    [0, envelopeOn(0, 1)],
  ] as const) {
    out.fill(Number.NaN);
    const moved = wiggleAttraction(out, attraction, dirs, env, depth, K);
    assert.equal(moved, false, `depth ${depth} reported a displacement it did not make`);
    for (let cell = 0; cell < K * K; cell++) {
      assert.equal(out[cell], Math.fround(attraction[cell] as number), `cell ${cell}`);
    }
  }

  // Same for the two null lanes — no directions drawn yet, no engine attached.
  assert.equal(wiggleAttraction(out, attraction, null, envelopeOn(0, 1), 1, K), false);
  assert.equal(wiggleAttraction(out, attraction, dirs, null, 1, K), false);
});

test('a hit displaces exactly one row, and leaves every zero cell at zero', () => {
  const attraction = testMatrix();
  const out = new Float32Array(K * K);
  const dirs = seedWiggleDirs(11, 0, K);
  const hit = 2;

  const moved = wiggleAttraction(out, attraction, dirs, envelopeOn(hit, 0.8), 1, K);
  assert.equal(moved, true);

  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const cell = i * K + j;
      const base = Math.fround(attraction[cell] as number);
      if (i !== hit) {
        assert.equal(out[cell], base, `row ${i} moved; only row ${hit} should have`);
        continue;
      }
      if (!coupled(i, j, K)) {
        assert.equal(out[cell], 0, `uncoupled cell (${i},${j}) left zero under a hit`);
        continue;
      }
      assert.notEqual(out[cell], base, `coupled cell (${i},${j}) did not move`);
    }
  }
});

test('a response pointed at a secondary is a no-op, not a partition leak', () => {
  const attraction = testMatrix();
  const out = new Float32Array(K * K);
  const dirs = seedWiggleDirs(11, 0, K);

  const moved = wiggleAttraction(out, attraction, dirs, envelopeOn(6, 1), 2, K);
  assert.equal(moved, false, 'an accent species reported a displacement');
  for (let cell = 0; cell < K * K; cell++) {
    assert.equal(out[cell], Math.fround(attraction[cell] as number), `cell ${cell}`);
  }
});

test('displacement is proportional to depth, envelope, and the matrix scale', () => {
  const dirs = seedWiggleDirs(3, 0, K);
  const out = new Float32Array(K * K);
  const row = 1;

  const shift = (attraction: number[], depth: number, env: number): number => {
    wiggleAttraction(out, attraction, dirs, envelopeOn(row, env), depth, K);
    // The first coupled cell of the row with a non-zero direction.
    for (let j = 0; j < K; j++) {
      const cell = row * K + j;
      if ((dirs[cell] as number) !== 0) {
        return (out[cell] as number) - Math.fround(attraction[cell] as number);
      }
    }
    throw new Error('the row has no direction at all');
  };

  const base = testMatrix();
  const unit = shift(base, 1, 1);
  assert.ok(Math.abs(unit) > 1e-4, 'depth 1 at full envelope did nothing measurable');
  assert.ok(Math.abs(shift(base, 2, 1) - 2 * unit) < 1e-5, 'depth is not linear');
  assert.ok(Math.abs(shift(base, 1, 0.5) - 0.5 * unit) < 1e-5, 'the envelope is not linear');

  // The seed-invariance property: a matrix drawn twice as large gets twice the
  // displacement, so "wiggle 1" reads the same however `matrixGen.sigma` is set.
  assert.ok(
    Math.abs(shift(testMatrix(K, 2), 1, 1) - 2 * unit) < 1e-5,
    'the max|A| normalisation is not carrying the matrix scale',
  );
});

test('the lane cannot walk a cell outside the attraction slot bound', () => {
  const attraction = testMatrix();
  const out = new Float32Array(K * K);
  const dirs = seedWiggleDirs(3, 0, K);

  // The deepest the two knobs compose to, against the lane's OWN rail — which is
  // not the attraction slot's, deliberately: see `WIGGLE_LIMIT` for the
  // measurement showing that a transient clamped at the authored bound makes the
  // depth knob inert.
  wiggleAttraction(out, attraction, dirs, envelopeOn(0, 1), MACRO_RANGE.wiggle.max * MAX_WIGGLE, K);
  let past = 0;
  for (let cell = 0; cell < K * K; cell++) {
    const v = out[cell] as number;
    assert.ok(
      Number.isFinite(v) && Math.abs(v) <= WIGGLE_LIMIT + 1e-6,
      `cell ${cell} left the rail at ${v}`,
    );
    if (Math.abs(v) > 2) past++;
  }
  assert.ok(past > 0, 'the lane never reached past the authored bound, so its rail does nothing');
});

// ── the engine's lane ────────────────────────────────────────────────────────

function engine(over?: (cfg: ImpulseConfig) => void, events: TimelineEvent[] = []): ImpulseEngine {
  const cfg = defaultImpulseConfig();
  over?.(cfg);
  return new ImpulseEngine(1, K, events, SECONDS_PER_TICK, cfg);
}

test('the wiggle envelope idles at 0, rises on a hit and decays back', () => {
  const e = engine();
  assert.deepEqual(Array.from(e.state.wiggle), new Array<number>(K).fill(0));

  // kick → species 0, wiggle 0.6 at full strength.
  e.testFire('kick', 1);
  const peak = e.state.wiggle[0] as number;
  // Float32 storage, so the comparison is against the rounded depth.
  assert.equal(peak, Math.fround(defaultImpulseConfig().responses.kick.wiggle));
  assert.deepEqual(Array.from(e.state.wiggle).slice(1), new Array<number>(K - 1).fill(0));

  // Decay is the same envelope every other lane rides, so a couple of τ later it
  // is a fraction of the peak, and it reaches exactly 0 rather than crawling.
  for (let t = 0; t < 60; t++) e.update(t, SECONDS_PER_TICK);
  assert.ok((e.state.wiggle[0] as number) < peak * 0.5, 'the envelope did not decay');
  for (let t = 60; t < 400; t++) e.update(t, SECONDS_PER_TICK);
  assert.equal(e.state.wiggle[0], 0, 'the lane never returned to rest');
});

test('the wiggle lane obeys gain, the per-kind depth and the global switch', () => {
  const half = engine((cfg) => {
    cfg.gain = 0.5;
  });
  half.testFire('kick', 1);
  assert.ok(
    Math.abs((half.state.wiggle[0] as number) - 0.5 * defaultImpulseConfig().responses.kick.wiggle) <
      1e-6,
  );

  const muted = engine((cfg) => {
    for (const kind of EVENT_KINDS) cfg.responses[kind].wiggle = 0;
  });
  muted.testFire('kick', 1);
  assert.equal(muted.state.wiggle[0], 0, 'a zero per-kind depth still moved the row');

  const off = engine((cfg) => {
    cfg.enabled = false;
  });
  off.testFire('kick', 1);
  assert.deepEqual(Array.from(off.state.wiggle), new Array<number>(K).fill(0));
});

test('two kinds landing on one species sum, and a retrigger takes the max', () => {
  const e = engine((cfg) => {
    // Both onto species 0, which is where kick and bass already point.
    cfg.responses.bass.wiggle = 0.4;
    cfg.responses.kick.wiggle = 0.6;
  });
  e.testFire('kick', 1);
  e.testFire('bass', 1);
  assert.ok(Math.abs((e.state.wiggle[0] as number) - 1.0) < 1e-6, 'the kinds did not sum');

  const retrigger = engine();
  retrigger.testFire('kick', 1);
  retrigger.testFire('kick', 1);
  assert.ok(
    Math.abs((retrigger.state.wiggle[0] as number) - defaultImpulseConfig().responses.kick.wiggle) <
      1e-6,
    'a retrigger stacked instead of taking the max',
  );
});

// ── hold ─────────────────────────────────────────────────────────────────────

test('a held kind keeps every one of its lanes open until released', () => {
  const e = engine();
  e.setHold('kick');
  const kick = defaultImpulseConfig().responses.kick;

  // Ten seconds of ticks, which is ~40 time constants — a decaying envelope is
  // long gone; a held one is exactly where it started.
  for (let t = 0; t < 1200; t++) e.update(t, SECONDS_PER_TICK);
  assert.equal(e.levelOf('kick'), 1);
  assert.equal(e.state.wiggle[0], Math.fround(kick.wiggle));
  assert.equal(e.state.brightMul[0], Math.fround(1 + kick.flash));
  assert.equal(e.heldKind, 'kick');

  e.setHold(null);
  for (let t = 1200; t < 2400; t++) e.update(t, SECONDS_PER_TICK);
  assert.equal(e.heldKind, null);
  assert.equal(e.state.wiggle[0], 0);
});

test('holding does not spawn splashes, which would fill the disc buffer', () => {
  const e = engine();
  e.setHold('snare'); // the kind with splashCount 2
  for (let t = 0; t < 600; t++) e.update(t, SECONDS_PER_TICK);
  assert.equal(e.activeSplashes, 0);
  // …but the rest of the snare's response is unmistakably live.
  assert.ok((e.state.wiggle[1] as number) > 0);
});

test('a hold obeys the global gain and the enable switch', () => {
  const quiet = engine((cfg) => {
    cfg.gain = 0.25;
  });
  quiet.setHold('kick');
  quiet.update(1, SECONDS_PER_TICK);
  assert.ok(Math.abs(quiet.levelOf('kick') - 0.25) < 1e-6);

  const off = engine((cfg) => {
    cfg.enabled = false;
  });
  off.setHold('kick');
  off.update(1, SECONDS_PER_TICK);
  assert.deepEqual(Array.from(off.state.wiggle), new Array<number>(K).fill(0));
});

// ── persistence ──────────────────────────────────────────────────────────────

test('the wiggle depth and the direction roll are saved settings', () => {
  // Both ride machinery with generic round-trip coverage (persist-roundtrip.ts),
  // so this pins the two things a user would report: the macro they dialled and
  // the direction family they rerolled into.
  const live = defaultImpulseConfig();
  applyImpulseConfig(live, { responses: { snare: { wiggle: 1.75 } } });
  assert.equal(live.responses.snare.wiggle, 1.75);
  applyImpulseConfig(live, { responses: { snare: { wiggle: 99 } } });
  assert.equal(
    live.responses.snare.wiggle,
    MAX_WIGGLE,
    'an out-of-range depth is clamped to the slider',
  );

  assert.ok('wiggle' in defaultPlifeMacros(), 'the macro is not in the defaults, so it cannot save');
  assert.equal(defaultMatrixGen().wiggleRoll, 0);
  const matrixGen = persistedBlockDecls(PLIFE_BLOCKS).find(([name]) => name === 'matrixGen');
  assert.ok(matrixGen, 'matrixGen is not declared as a persisted block, so it cannot save');
  assert.ok(
    blockRules(matrixGen[1], defaultPlifeConfig())['wiggleRoll'],
    'the roll has no clamp rule, so a corrupt file reaches the hash',
  );
});
