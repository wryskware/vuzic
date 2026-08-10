/**
 * Headless tests for particle life's θ registry (`sim/plife/preset.ts`).
 *
 * Same house rules as `modulation.test.ts`: `node --test` loads the modules
 * straight out of src/, which is why every import carries an explicit `.ts`
 * extension, and nothing here touches the DOM or WebGPU.
 *
 * What these pin, in the order the runtime depends on them: the vector layout
 * (persistence indexes by slot), the exclusions (light belongs to the stem lane,
 * stretch belongs to the sliders, minR belongs to the seeded draw), the
 * **primary/secondary partition** (the one property that makes the accents
 * accents rather than four more primaries — and it only holds if the uncoupled
 * cells stay both zero *and* unmodulated), the round trip, the seeded
 * personality's bounds, and the seeded matrix generator's rules.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { baseVector } from '../src/mapping/modulation.ts';
import { MOD_GROUPS, type ModSpec } from '../src/mapping/modspec.ts';
import {
  applyVector,
  clampVector,
  coupled,
  fieldClasses,
  fieldNames,
  globalsBase,
  matrixBase,
  maxRBase,
  minRBase,
  modulationMask,
  modulationSlots,
  presetFromConfig,
  presetToVector,
  vectorLength,
} from '../src/sim/plife/preset.ts';
import { seedMatrixBase } from '../src/sim/plife/genmatrix.ts';
import {
  defaultAttraction,
  defaultPlifeConfig,
  defaultRadii,
  type PlifeConfig,
  MAX_BRIGHTNESS,
  MAX_FRICTION,
  MAX_MIN_R,
  MAX_RADIUS_SCALE,
  MAX_REACH_BRUTE,
  MAX_SIZE,
  MAX_STRETCH,
  MIN_R_FLOOR,
  PRIMARY_COUNT,
  R_CAP,
} from '../src/sim/plife/config.ts';

const K = 8;
const LEN = vectorLength(K);
const SLOTS = modulationSlots(K);
const MASK = modulationMask(K);
const NAMES = fieldNames(K);

const M_BASE = matrixBase(K);
const X_BASE = maxRBase(K);
const N_BASE = minRBase(K);
const G_BASE = globalsBase(K);

/** The shipped θ, as the modulator snapshots it at start-up. */
function defaultTheta(): Float64Array {
  return presetToVector(presetFromConfig(defaultPlifeConfig(K)), K);
}

/** Per-species slot index → its name, without re-parsing the whole table. */
function speciesSlot(species: number, name: string): number {
  const want = `species${species}.${name}`;
  const i = NAMES.indexOf(want);
  assert.ok(i >= 0, `no slot named ${want}`);
  return i;
}

/**
 * The hard [min, max] per slot, recovered from `clampVector` itself — the slot
 * table's bounds are module-private, and probing the clamp is the contract the
 * rest of the app actually sees.
 */
function hardBounds(): { min: Float64Array; max: Float64Array } {
  return {
    min: clampVector(new Float64Array(LEN).fill(-1e6), K),
    max: clampVector(new Float64Array(LEN).fill(1e6), K),
  };
}

// ── layout ───────────────────────────────────────────────────────────────────

test('the vector layout is the one persistence and the runtime index by', () => {
  // 9 slots per species, then three K² blocks, then 4 globals = 268 at K = 8.
  assert.equal(LEN, 9 * K + 3 * K * K + 4);
  assert.equal(LEN, 268);
  assert.equal(NAMES.length, LEN);
  assert.equal(NAMES[0], 'species0.brightness', 'species 0 opens the vector');
  assert.equal(NAMES[LEN - 1], 'gamma', 'the last global closes it');
  assert.equal(fieldClasses(K).length, LEN);
  // and the block bases really do partition the vector in the documented order
  assert.equal(M_BASE, 9 * K);
  assert.equal(X_BASE, M_BASE + K * K);
  assert.equal(N_BASE, X_BASE + K * K);
  assert.equal(G_BASE, N_BASE + K * K);
  assert.equal(LEN - G_BASE, 4);
});

// ── exclusions ───────────────────────────────────────────────────────────────

test('light is out of θ entirely: brightness and intensity are the stem lane (Revision 4)', () => {
  // Two separate lookups in the runtime — a null spec and a zero mask bit — so
  // both are asserted, exactly as physarum's registry is.
  for (let s = 0; s < K; s++) {
    for (const field of ['brightness', 'intensity']) {
      const i = speciesSlot(s, field);
      assert.equal(SLOTS[i], null, `${NAMES[i]} still has a ModSpec`);
      assert.equal(MASK[i], 0, `${NAMES[i]} is still in the modulation mask`);
    }
  }
});

test('stretch is a look knob the music may not touch; the six motion slots are grouped as documented', () => {
  // The group assignment is what the workbench's depth sliders mean, so it is
  // part of the contract and not an implementation detail.
  const expected: Record<string, string | null> = {
    brightness: null,
    intensity: null,
    stretch: null,
    // size joined the look-knob exclusions 2026-08-08 — same argument as
    // stretch, and its wide mod range read as giant orbs among dots.
    size: null,
    aliveFraction: 'population',
    radiusScale: 'structure',
    forceScale: 'structure',
    friction: 'decay',
    wander: 'structure',
  };
  for (let s = 0; s < K; s++) {
    for (const [field, group] of Object.entries(expected)) {
      const i = speciesSlot(s, field);
      const spec = SLOTS[i] as ModSpec | null;
      if (group === null) {
        assert.equal(spec, null, `${NAMES[i]} should be excluded`);
        assert.equal(MASK[i], 0, `${NAMES[i]} should be out of the mask`);
        continue;
      }
      assert.ok(spec, `${NAMES[i]} should be modulated`);
      assert.equal(spec.group, group, `${NAMES[i]} is in the wrong lane`);
      assert.equal(MASK[i], 1, `${NAMES[i]} should be in the mask`);
      assert.ok((MOD_GROUPS as readonly string[]).includes(spec.group), `unknown group ${spec.group}`);
    }
  }
});

// ── the partition ────────────────────────────────────────────────────────────

test('the coupled set is exactly the primary block plus each accent’s three edges', () => {
  // Stated independently of `coupled()`'s implementation: the full P×P block,
  // secondary P+n → {n, (n+1) mod P, itself}, primary n → its own accent P+n.
  const p = PRIMARY_COUNT;
  const want = new Set<string>();
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) want.add(`${i},${j}`);
  for (let n = 0; n < p; n++) {
    const i = p + n;
    want.add(`${i},${n}`);
    want.add(`${i},${(n + 1) % p}`);
    want.add(`${i},${i}`);
    want.add(`${n},${i}`);
  }
  assert.equal(want.size, 16 + 4 * 3 + 4, 'the expected set is 32 cells');

  const got = new Set<string>();
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) if (coupled(i, j, K)) got.add(`${i},${j}`);
  }
  assert.equal(got.size, 32, `coupled() marks ${got.size} of ${K * K} cells`);
  assert.deepEqual([...got].sort(), [...want].sort());
});

test('every uncoupled attraction cell is out of the music’s reach, in both lookups', () => {
  // This is the partition's load-bearing half: a modulated zero drifts, and a
  // minute later the secondaries are just four more primaries.
  let live = 0;
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const slot = M_BASE + i * K + j;
      const spec = SLOTS[slot] as ModSpec | null;
      if (coupled(i, j, K)) {
        assert.ok(spec, `${NAMES[slot]} should be modulated`);
        assert.equal(spec.group, 'matrix', `${NAMES[slot]} is in the wrong lane`);
        assert.equal(MASK[slot], 1, `${NAMES[slot]} is missing from the mask`);
        live++;
      } else {
        assert.equal(spec, null, `${NAMES[slot]} has a ModSpec and will drift off zero`);
        assert.equal(MASK[slot], 0, `${NAMES[slot]} is in the modulation mask`);
      }
    }
  }
  assert.equal(live, 32);
});

test('maxR follows the same partition, multiplicatively; minR is in the mask but immobile', () => {
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const cell = i * K + j;
      const xSpec = SLOTS[X_BASE + cell] as ModSpec | null;
      if (coupled(i, j, K)) {
        assert.ok(xSpec, `${NAMES[X_BASE + cell]} should be modulated`);
        assert.equal(xSpec.group, 'structure', `${NAMES[X_BASE + cell]} is in the wrong lane`);
        assert.equal(xSpec.mult, true, `${NAMES[X_BASE + cell]} should move multiplicatively`);
        assert.equal(MASK[X_BASE + cell], 1);
      } else {
        assert.equal(xSpec, null, `${NAMES[X_BASE + cell]} has a ModSpec`);
        assert.equal(MASK[X_BASE + cell], 0);
      }
      // The hard-core radius is a stability parameter, never a look knob — but
      // it is drawn from the seed, and a drawn value only reaches the config
      // through the mask. So the exclusion is expressed in the spec instead:
      // half 0 means the music moves it by exactly nothing (computeTarget builds
      // base · exp(0 · e) = base), jitter 0 means the generic personality leaves
      // it alone, and the generator owns the slot outright.
      const nSpec = SLOTS[N_BASE + cell] as ModSpec | null;
      assert.ok(nSpec, `${NAMES[N_BASE + cell]} lost its ModSpec — the seeded draw cannot land`);
      assert.equal(MASK[N_BASE + cell], 1, `${NAMES[N_BASE + cell]} is out of the mask`);
      assert.equal(nSpec.half, 0, `${NAMES[N_BASE + cell]} is music-mobile`);
      assert.equal(nSpec.jitter, 0, `${NAMES[N_BASE + cell]} is personality-mobile`);
      assert.equal(nSpec.group, 'structure', `${NAMES[N_BASE + cell]} is in the wrong lane`);
      assert.equal(nSpec.lo, MIN_R_FLOOR, `${NAMES[N_BASE + cell]} floor`);
      // MAX_MIN_R (0.05), not the grid cell size: the brute pair-search lane
      // reaches to 0.5, and a hard core has to stay a fraction of a reach.
      assert.equal(nSpec.hi, MAX_MIN_R, `${NAMES[N_BASE + cell]} ceiling`);
    }
  }
});

test('the shipped defaults obey the partition and the radii stay legal', () => {
  const a = defaultAttraction(K);
  assert.equal(a.length, K * K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const v = a[i * K + j] as number;
      if (!coupled(i, j, K)) assert.equal(v, 0, `A[${i}][${j}] is non-zero outside the partition`);
    }
  }
  // at minimum every primary is self-cohesive, which is what makes it clump
  for (let i = 0; i < PRIMARY_COUNT; i++) {
    assert.notEqual(a[i * K + i], 0, `A[${i}][${i}] is zero — species ${i} will not clump`);
  }

  const { minR, maxR } = defaultRadii(K);
  for (let o = 0; o < K * K; o++) {
    const lo = minR[o] as number;
    const hi = maxR[o] as number;
    // a band narrower than 4 mm of world is a spike, not a tent
    assert.ok(hi >= lo + 0.004 - 1e-9, `pair ${o}: band ${lo}..${hi} is too narrow`);
    assert.ok(lo >= MIN_R_FLOOR, `pair ${o}: minR ${lo} is under the floor`);
    assert.ok(hi <= R_CAP, `pair ${o}: maxR ${hi} exceeds the grid cell size`);
  }
});

// ── preset ⇄ config ⇄ vector ─────────────────────────────────────────────────

test('preset ↔ config ↔ vector round-trips exactly, slot for slot', () => {
  const v0 = presetToVector(presetFromConfig(defaultPlifeConfig(K)), K);
  const cfg = defaultPlifeConfig(K);
  applyVector(cfg, new Float64Array(v0));
  const v1 = presetToVector(presetFromConfig(cfg), K);
  assert.equal(v0.length, LEN);
  for (let i = 0; i < LEN; i++) assert.equal(v1[i], v0[i], NAMES[i]);
});

test('a masked applyVector protects the exclusions and never touches the population block', () => {
  const cfg = defaultPlifeConfig(K);
  // Sentinels on three things the mask must defend: light, the look knob, and an
  // uncoupled matrix cell (0,5 — a primary against someone else's accent).
  const uncoupled = 0 * K + 5;
  assert.ok(!coupled(0, 5, K), 'the chosen probe cell must be outside the partition');
  cfg.species[0]!.brightness = 0.42;
  cfg.species[0]!.intensity = 1.75;
  cfg.species[0]!.stretch = 6.5;
  cfg.attraction[uncoupled] = 0;
  cfg.exposure = 0.77;
  cfg.gamma = 2.4;
  // The population package lives outside θ; a modulated write must not see it.
  cfg.population.floor = 0.123;

  applyVector(cfg, new Float64Array(LEN).fill(9), MASK);

  assert.equal(cfg.species[0]!.brightness, 0.42, 'brightness was written through the mask');
  assert.equal(cfg.species[0]!.intensity, 1.75, 'intensity was written through the mask');
  assert.equal(cfg.species[0]!.stretch, 6.5, 'stretch was written through the mask');
  assert.equal(cfg.attraction[uncoupled], 0, 'an uncoupled cell drifted off zero');
  // minR is deliberately NOT defended by the mask any more: the seeded draw
  // reaches the config through exactly this path. Its immobility is the spec's
  // job (half 0, jitter 0), asserted in the partition test above.
  assert.equal(cfg.minR[uncoupled], 9, 'minR should now flow through the mask');
  assert.equal(cfg.exposure, 0.77);
  assert.equal(cfg.gamma, 2.4);
  assert.equal(cfg.population.floor, 0.123, 'applyVector reached outside θ');

  // …and every masked-in slot took the written value verbatim: applyVector is
  // the hot path and does not clamp — bounding is the modulator's job.
  const after = presetToVector(presetFromConfig(cfg), K);
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 1) assert.equal(after[i], 9, `${NAMES[i]} did not take the write`);
  }
});

// ── bounds ───────────────────────────────────────────────────────────────────

test('clampVector confines every slot to its authored hard range', () => {
  const { min, max } = hardBounds();
  for (let i = 0; i < LEN; i++) {
    const lo = min[i] as number;
    const hi = max[i] as number;
    assert.ok(Number.isFinite(lo) && Number.isFinite(hi), `${NAMES[i]} has a non-finite bound`);
    assert.ok(hi > lo, `${NAMES[i]} has an empty range [${lo}, ${hi}]`);
  }

  // The hard bounds are the ones the rest of the app publishes as constants.
  assert.equal(max[speciesSlot(0, 'brightness')], MAX_BRIGHTNESS);
  assert.equal(max[speciesSlot(3, 'size')], MAX_SIZE);
  assert.equal(max[speciesSlot(5, 'stretch')], MAX_STRETCH);
  assert.equal(max[speciesSlot(7, 'friction')], MAX_FRICTION);
  assert.equal(max[speciesSlot(1, 'radiusScale')], MAX_RADIUS_SCALE);
  for (let o = 0; o < K * K; o++) {
    assert.equal(min[X_BASE + o], MIN_R_FLOOR, `${NAMES[X_BASE + o]} floor`);
    // The outer radius runs to MAX_REACH_BRUTE (half the torus) — ONE authored
    // ceiling for both pair-search modes. The runtime cap is the shader's, and
    // it is the mode's: stencil × cell for the grid walk, half the world for
    // brute. A file written in brute mode therefore stays legal in grid mode and
    // comes back unchanged, rather than being truncated on the way through.
    assert.equal(max[X_BASE + o], MAX_REACH_BRUTE, `${NAMES[X_BASE + o]} may exceed the reach cap`);
    assert.equal(min[N_BASE + o], MIN_R_FLOOR, `${NAMES[N_BASE + o]} floor`);
    assert.ok((max[N_BASE + o] as number) <= MAX_MIN_R, `${NAMES[N_BASE + o]} ceiling`);
  }

  // Every modulation range sits inside its slot's hard range — "where the music
  // may wander" must be a sub-interval of "what a file may contain", or the
  // modulator can author a θ that a reload would then clamp to something else.
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec) continue;
    assert.ok(spec.lo >= (min[i] as number) - 1e-12, `${NAMES[i]} mod lo ${spec.lo} < hard ${min[i]}`);
    assert.ok(spec.hi <= (max[i] as number) + 1e-12, `${NAMES[i]} mod hi ${spec.hi} > hard ${max[i]}`);
  }

  // …and an adversarial vector really does land inside those bounds.
  for (const fill of [1e6, -1e6, Number.MAX_VALUE, -Number.MAX_VALUE]) {
    const v = clampVector(new Float64Array(LEN).fill(fill), K);
    for (let i = 0; i < LEN; i++) {
      const x = v[i] as number;
      assert.ok(
        x >= (min[i] as number) && x <= (max[i] as number),
        `${NAMES[i]} = ${x} outside [${min[i]}, ${max[i]}] for fill ${fill}`,
      );
    }
  }
});

// ── seeded personality ───────────────────────────────────────────────────────

test('the seeded personality stays inside every plife bound and leaves the exclusions alone', () => {
  const defaults = defaultTheta();
  const out = new Float64Array(LEN);
  for (const seed of [1, 7, 99, 4242, 0xdead_beef]) {
    baseVector(seed, defaults, SLOTS, out);
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i] as ModSpec | null;
      if (!spec) {
        assert.equal(out[i], defaults[i], `${NAMES[i]} moved at seed ${seed} despite being excluded`);
        continue;
      }
      const x = out[i] as number;
      assert.ok(
        Number.isFinite(x) && x >= spec.lo - 1e-9 && x <= spec.hi + 1e-9,
        `${NAMES[i]} = ${x} outside [${spec.lo}, ${spec.hi}] at seed ${seed}`,
      );
    }
  }
});

// ── the seeded matrix draw ───────────────────────────────────────────────────

test('seedMatrixBase is deterministic, seed-sensitive, and obeys every generation rule', () => {
  const cfg = defaultPlifeConfig(K);
  const draw = (seed: number, c = cfg): Float64Array => {
    const v = defaultTheta();
    seedMatrixBase(seed, c, v);
    return v;
  };

  // 1. same (seed, cfg) twice ⇒ bit-identical. This is the property the whole
  //    "pin a seed and keep tuning" loop rests on.
  assert.deepEqual(Array.from(draw(1234)), Array.from(draw(1234)));

  // 2. different seeds ⇒ genuinely different worlds, not a jitter of one.
  const a = draw(1);
  const b = draw(2);
  let moved = 0;
  for (let o = 0; o < K * K; o++) if (a[M_BASE + o] !== b[M_BASE + o]) moved++;
  assert.ok(moved >= 16, `only ${moved} attraction cells differ between two seeds`);

  // 3. the partition survives the draw: every uncoupled cell is exactly 0.
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      if (coupled(i, j, K)) continue;
      assert.equal(a[M_BASE + i * K + j], 0, `A[${i}][${j}] is non-zero outside the partition`);
    }
  }

  // 4. with sigma 0 the draw collapses to its structure: every coupled
  //    off-diagonal is exactly 0 and every diagonal is exactly its role's bias
  //    (selfBias for primaries, selfBiasAccent for secondaries — the split is
  //    the singularity fix: a self-attracting accent collapses its small
  //    population into one bright point). This isolates the bias terms from the
  //    noise term.
  for (const bias of [0.4, -0.4]) {
    const accentBias = -bias / 2;
    const flat: PlifeConfig = {
      ...cfg,
      matrixGen: {
        ...cfg.matrixGen,
        sigma: 0,
        selfBias: bias,
        selfBiasAccent: accentBias,
        symmetry: 0,
      },
    };
    const v = draw(77, flat);
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        if (!coupled(i, j, K)) continue;
        const x = v[M_BASE + i * K + j] as number;
        const want = i < 4 ? bias : accentBias;
        if (i === j) assert.ok(Math.abs(x - want) < 1e-12, `diag ${i} = ${x}, expected ${want}`);
        // `===` rather than assert.equal: a negative Gaussian times sigma 0 is
        // -0, which is numerically zero and is not what this test is about.
        else assert.ok(x === 0, `A[${i}][${j}] = ${x} with sigma 0`);
      }
    }
  }

  // 5. symmetry 1 forces a[i][j] === a[j][i] wherever BOTH cells are coupled —
  //    the primary block is the region where that is true of every pair.
  const sym: PlifeConfig = { ...cfg, matrixGen: { ...cfg.matrixGen, symmetry: 1 } };
  const s = draw(9, sym);
  for (let i = 0; i < PRIMARY_COUNT; i++) {
    for (let j = 0; j < PRIMARY_COUNT; j++) {
      const x = s[M_BASE + i * K + j] as number;
      const y = s[M_BASE + j * K + i] as number;
      assert.ok(Math.abs(x - y) < 1e-12, `A[${i}][${j}]=${x} vs A[${j}][${i}]=${y} at symmetry 1`);
    }
  }

  // 6. every drawn attraction cell sits inside the attraction slot's own range,
  //    so a drawn matrix is always somewhere the music could also have gone.
  const big: PlifeConfig = { ...cfg, matrixGen: { ...cfg.matrixGen, sigma: 5, selfBias: 3 } };
  const wild = draw(11, big);
  for (let o = 0; o < K * K; o++) {
    const spec = SLOTS[M_BASE + o] as ModSpec | null;
    if (!spec) continue;
    const x = wild[M_BASE + o] as number;
    assert.ok(x >= spec.lo && x <= spec.hi, `${NAMES[M_BASE + o]} = ${x} escaped [${spec.lo}, ${spec.hi}]`);
  }

  // 7. radii: drawn for EVERY pair (the hard core applies to pairs that ignore
  //    each other too), inside their bands, legal as a tent, under the cap.
  const g = cfg.matrixGen;
  for (let o = 0; o < K * K; o++) {
    const lo = a[N_BASE + o] as number;
    const hi = a[X_BASE + o] as number;
    assert.ok(lo >= MIN_R_FLOOR - 1e-12, `pair ${o}: minR ${lo} is under the floor`);
    assert.ok(lo >= Math.min(g.rMin.lo, MIN_R_FLOOR) - 1e-12 && lo <= g.rMin.hi + 1e-12, `pair ${o}: minR ${lo} outside its band`);
    assert.ok(hi >= lo + 0.003 - 1e-9, `pair ${o}: band ${lo}..${hi} is too narrow`);
    assert.ok(hi <= R_CAP + 1e-12, `pair ${o}: maxR ${hi} exceeds the grid cell size`);
  }

  // 8. nothing outside the three K² blocks is touched — the species block and
  //    the globals belong to the generic personality pass.
  const before = defaultTheta();
  const after = draw(4242);
  for (let i = 0; i < LEN; i++) {
    if (i >= M_BASE && i < G_BASE) continue;
    assert.equal(after[i], before[i], `${NAMES[i]} was written by the matrix generator`);
  }
});

// ── determinism ──────────────────────────────────────────────────────────────

test('the registry is a stable pure function — PlifeSim may cache it safely', () => {
  assert.deepEqual(modulationSlots(K), modulationSlots(K));
  assert.deepEqual(modulationMask(K), modulationMask(K));
  assert.deepEqual(fieldNames(K), fieldNames(K));
  assert.deepEqual(fieldClasses(K), fieldClasses(K));
  // the mask really is the slot table's shadow, not a second hand-written list
  const slots = modulationSlots(K);
  const mask = modulationMask(K);
  assert.equal(mask.length, LEN);
  for (let i = 0; i < LEN; i++) assert.equal(mask[i], slots[i] ? 1 : 0, NAMES[i]);
});
