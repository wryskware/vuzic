/**
 * Headless tests for the pure half of the modulation layer (plan.md Revision 4).
 *
 * Run with `npm test` in web/ — Node strips the types and loads the modules
 * straight out of src/, which is why mapping/*.ts imports carry explicit .ts
 * extensions. Nothing here touches the DOM or WebGPU.
 *
 * The properties that matter, in the order the runtime depends on them:
 *   the driver bank really is variance-ordered and standardised, a gain of 0
 *   really does silence a driver everywhere, the wiring really is a function of
 *   the seed alone, the output really cannot leave its range, the depths really
 *   do isolate — and brightness really is out of all of it.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  baseVector,
  buildDriverBank,
  DriverBank,
  makeRng,
  PC_DRIVER_COUNT,
  STRUCTURE_DRIVERS,
  unitDirection,
  Modulator,
  varianceOrder,
  Z_CLAMP,
} from '../src/mapping/modulation.ts';
import type { ModTarget, ThetaRegistry } from '../src/mapping/target.ts';
import type { TimelineSampler } from '../src/timeline/sampler.ts';
import type { ModulationConfig } from '../src/mapping/types.ts';
import { SlewLimiter } from '../src/mapping/slew.ts';
import { defaultStemFollow, followMultiplier, StemFollow } from '../src/mapping/stemfollow.ts';
import {
  applyVector,
  CLASS_FAST,
  CLASS_SLOW,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  MOD_GROUPS,
  presetFromConfig,
  presetToVector,
  vectorLength,
  vectorToPreset,
  type ModGroup,
  type ModSpec,
} from '../src/mapping/preset.ts';
import {
  defaultModulationConfig,
  modulationFits,
  parseModulation,
  serializeModulation,
} from '../src/mapping/persist.ts';
import { defaultConfig } from '../src/sim/physarum/config.ts';
import { defaultRenderConfig, mergeRenderConfig } from '../src/sim/render/config.ts';
import type { Timeline } from '../src/timeline/types.ts';

const K = 4;
const LEN = vectorLength(K);
const SLOTS = modulationSlots(K);
const MASK = modulationMask(K);

/**
 * The modulator's arithmetic, reproduced from the registry. The class under test
 * needs a PhysarumSim and a TimelineSampler; these tests deliberately exercise the
 * same formula without a GPU, and `targetFor` on the real class is the identical
 * expression (one code path, verified by eye and by the bounds test below).
 *
 * `gains` is Revision 4's addition and multiplies the driver *before* the dot
 * product — which is the only reason a gain of 0 can be exactly, rather than
 * approximately, silent in every wiring.
 */
function targetFor(
  seed: number,
  z: ArrayLike<number>,
  dims: number,
  defaults: Float64Array,
  depth: number,
  groupDepth: Record<ModGroup, number>,
  gains?: ArrayLike<number>,
): Float64Array {
  const base = baseVector(seed, defaults, SLOTS, new Float64Array(LEN));
  const out = new Float64Array(LEN);
  const w = new Float32Array(dims);
  const gz = new Float32Array(dims);
  for (let d = 0; d < dims; d++) gz[d] = ((gains?.[d] as number) ?? 1) * ((z[d] as number) ?? 0);
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec) {
      out[i] = base[i] as number;
      continue;
    }
    unitDirection(seed, i, dims, w);
    let raw = 0;
    for (let d = 0; d < dims; d++) raw += (w[d] as number) * (gz[d] as number);
    const e = Math.tanh(depth * groupDepth[spec.group] * raw);
    const b = base[i] as number;
    const v = spec.mult ? b * Math.exp(spec.half * e) : b + spec.half * e;
    out[i] = v < spec.lo ? spec.lo : v > spec.hi ? spec.hi : v;
  }
  return out;
}

/** A DriverBank straight from a frames x dims array, for the arithmetic tests. */
function bankOf(raw: Float32Array, frames: number, dims: number, names?: string[]): DriverBank {
  return new DriverBank(
    Array.from({ length: dims }, (_, d) => ({
      name: names?.[d] ?? `d${d}`,
      source: d,
      read: (f: number) => raw[f * dims + d] as number,
    })),
    frames,
    0.1,
    'test',
  );
}

/**
 * A minimal Timeline-shaped object: the loader's product, without the loader.
 * `channels` is what `buildDriverBank` walks, so the layout has to be real.
 */
function fakeTimeline(
  frames: number,
  channels: { name: string; dims: number; offset: number }[],
  fill: (frame: number, offset: number) => number,
): Timeline {
  const stride = channels.reduce((a, c) => a + c.dims, 0);
  const data = new Float32Array(frames * stride);
  for (let f = 0; f < frames; f++) {
    for (let o = 0; o < stride; o++) data[f * stride + o] = fill(f, o);
  }
  return {
    manifest: { grid: { frames, hopSeconds: 0.1 } },
    data,
    stride,
    channels: new Map(channels.map((c) => [c.name, c])),
    events: [],
  } as unknown as Timeline;
}

function unitDepths(value = 1): Record<ModGroup, number> {
  const out = {} as Record<ModGroup, number>;
  for (const g of MOD_GROUPS) out[g] = value;
  return out;
}

function defaultTheta(): Float64Array {
  return presetToVector(presetFromConfig(defaultConfig(K)), K);
}

// ── the registry ─────────────────────────────────────────────────────────────

test('the slot table still has the layout the runtime indexes by', () => {
  // 19 per species since `scale` was appended (2026-08-09). Appended, not
  // inserted: the assertions below and the offset constants in preset.ts both
  // depend on brightness still being slot 0 of a species block.
  assert.equal(LEN, 19 * K + K * K + 4);
  const names = fieldNames(K);
  assert.equal(names.length, LEN);
  assert.equal(names[0], 'species0.brightness');
  assert.equal(names[18], 'species0.scale', 'scale is the last slot of a species block');
  assert.equal(names[19], 'species1.brightness', 'the block really is 19 wide');
  assert.ok(!names.some((n) => n.includes('color')), 'no colour slot survives in θ');
  assert.equal(names[names.length - 1], 'stemGain');
  assert.equal(fieldClasses(K).length, LEN);
  assert.equal(fieldClasses(K)[0], CLASS_FAST);
});

test('the exclusions are exactly the documented ones', () => {
  const names = fieldNames(K);
  const excluded = names.filter((_, i) => SLOTS[i] === null);
  // every p3 exponent, brightness + intensity per species (Revision 4), plus
  // all four globals — senseGain joined the excluded set 2026-08-08 (whole-sim
  // scalars have no musical referent; the user's macros own global character)
  assert.equal(excluded.length, 6 * K + 4);
  for (const n of excluded) {
    assert.ok(
      n.endsWith('.p3') ||
        n.endsWith('.brightness') ||
        n.endsWith('.intensity') ||
        n === 'senseGain' ||
        n === 'exposure' ||
        n === 'gamma' ||
        n === 'stemGain',
      `unexpected exclusion ${n}`,
    );
  }
  assert.equal(MASK.length, LEN);
  assert.equal(
    MASK.reduce((a, b) => a + b, 0),
    LEN - excluded.length,
  );
});

test('per-species scale is a modulated structure slot with real authority', () => {
  // The slot exists to fix a specific complaint: sensorDist and moveDist are six
  // independently-wired slots, so the network's apparent scale never moves over a
  // track. A cautious `half` would reproduce that failure with more machinery, so
  // the size of the excursion is part of the contract, not a taste setting.
  const names = fieldNames(K);
  for (let s = 0; s < K; s++) {
    const i = names.indexOf(`species${s}.scale`);
    assert.ok(i >= 0, `species${s}.scale is missing from the registry`);
    const spec = SLOTS[i] as ModSpec | null;
    assert.ok(spec, 'scale must be modulated, not slider-only');
    assert.equal(MASK[i], 1);
    assert.equal(spec.group, 'structure');
    assert.ok(spec.mult, 'scale is a ratio, so it moves in ln space');
    // full-tanh reach, i.e. what the band on the slider draws at base 1
    assert.ok(Math.exp(-spec.half) < 0.7, `×${Math.exp(-spec.half)} is too timid to read`);
    assert.ok(Math.exp(spec.half) > 1.5, `×${Math.exp(spec.half)} is too timid to read`);
    // the envelope is wider than the tanh band, so a rebased hand slider still
    // leaves the music somewhere to go in both directions
    assert.ok(spec.lo < Math.exp(-spec.half) && spec.hi > Math.exp(spec.half));
    // …and it stays inside the hard θ bound the panel slider and the shader clamp
    assert.ok(spec.lo >= 0.25 && spec.hi <= 4);
  }
  // the shipped default is neutral: scale = 1 is the bit-identical A/B baseline
  const defaults = defaultTheta();
  for (let s = 0; s < K; s++) {
    assert.equal(defaults[names.indexOf(`species${s}.scale`)], 1);
  }
});

test('the matrix diagonal is kept positive, so networks always form', () => {
  const names = fieldNames(K);
  for (let i = 0; i < LEN; i++) {
    const m = /^M\[(\d+)]\[(\d+)]$/.exec(names[i] as string);
    if (!m) continue;
    const spec = SLOTS[i] as ModSpec;
    if (m[1] === m[2]) assert.ok(spec.lo > 0, `${names[i]} may go non-positive`);
    else assert.ok(spec.lo < 0, `${names[i]} should be able to repel`);
  }
});

test('a masked applyVector leaves the excluded slots alone', () => {
  const cfg = defaultConfig(K);
  const before = presetToVector(presetFromConfig(cfg), K);
  const v = new Float64Array(LEN).fill(0.5);
  applyVector(cfg, v, MASK);
  const after = presetToVector(presetFromConfig(cfg), K);
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 1) assert.equal(after[i], 0.5, fieldNames(K)[i]);
    else assert.equal(after[i], before[i], `${fieldNames(K)[i]} was written through the mask`);
  }
});

test('preset ↔ config ↔ vector round-trips exactly', () => {
  const cfg = defaultConfig(K);
  const v0 = presetToVector(presetFromConfig(cfg), K);
  applyVector(cfg, new Float64Array(v0));
  const v1 = presetToVector(presetFromConfig(cfg), K);
  for (let i = 0; i < v0.length; i++) assert.equal(v1[i], v0[i], fieldNames(K)[i]);
  assert.equal(vectorToPreset(v0, K).species.length, K);
});

test('brightness is out of the modulation registry entirely (Revision 4)', () => {
  const names = fieldNames(K);
  // No slot may claim the retired group, and the two light slots per species
  // must be untouchable by the modulator in both directions: null spec AND a
  // zero mask bit, since those are two separate lookups in the runtime.
  assert.ok(!(MOD_GROUPS as readonly string[]).includes('brightness'), 'the group is gone');
  for (let i = 0; i < LEN; i++) {
    const n = names[i] as string;
    if (!n.endsWith('.brightness') && !n.endsWith('.intensity')) continue;
    assert.equal(SLOTS[i], null, `${n} still has a ModSpec`);
    assert.equal(MASK[i], 0, `${n} is still in the modulation mask`);
  }
  // …and a full-range masked write leaves the live brightness alone.
  const cfg = defaultConfig(K);
  cfg.species[0]!.brightness = 0.42;
  cfg.species[0]!.intensity = 1.75;
  applyVector(cfg, new Float64Array(LEN).fill(9), MASK);
  assert.equal(cfg.species[0]!.brightness, 0.42);
  assert.equal(cfg.species[0]!.intensity, 1.75);
});

// ── the driver bank ──────────────────────────────────────────────────────────

test('variance reordering: the loudest latent dim becomes pc-1, whatever its index', () => {
  const frames = 400;
  const latentDims = 8;
  const stride = 3 + latentDims;
  // dim 5 gets the biggest swing, dim 2 the second, everything else is quiet —
  // deliberately NOT in stored order, which is the bug Revision 4 fixes.
  const amp = [0.1, 0.05, 0.6, 0.02, 0.03, 1.0, 0.01, 0.04];
  const tl = fakeTimeline(
    frames,
    [
      { name: 'novelty4', dims: 1, offset: 0 },
      { name: 'novelty16', dims: 1, offset: 1 },
      { name: 'actChorus', dims: 1, offset: 2 },
      { name: 'latent', dims: latentDims, offset: 3 },
    ],
    (f, o) => (o < 3 ? Math.sin(f * (0.1 + o * 0.03)) : (amp[o - 3] as number) * Math.sin(f * 0.07 + o)),
  );

  const order = varianceOrder(tl.data, frames, stride, 3, latentDims);
  assert.equal(order[0], 5, 'dim 5 has the most variance');
  assert.equal(order[1], 2);

  const bank = buildDriverBank(tl);
  assert.ok(bank);
  assert.equal(bank.dims, STRUCTURE_DRIVERS.length + Math.min(PC_DRIVER_COUNT, latentDims));
  assert.deepEqual(bank.names.slice(0, 3), ['novelty·4bar', 'novelty·16bar', 'chorus-ness']);
  assert.equal(bank.names[3], 'pc-1');
  assert.equal(bank.sources[3], 5, 'pc-1 is latent dim 5');
  assert.equal(bank.sources[4], 2, 'pc-2 is latent dim 2');
  // strictly descending raw variance across the pc block
  for (let d = 4; d < bank.dims; d++) {
    assert.ok(
      (bank.variance[d] as number) <= (bank.variance[d - 1] as number) + 1e-12,
      `pc-${d - 2} has more variance than pc-${d - 3}`,
    );
  }
});

test('a missing structure channel keeps its slot as a silent driver', () => {
  // Driver indices key the seeded wiring and the saved gains, so they must not
  // shift because an optional channel is absent from one track.
  const frames = 50;
  const tl = fakeTimeline(
    frames,
    [
      { name: 'novelty4', dims: 1, offset: 0 },
      { name: 'latent', dims: 4, offset: 1 },
    ],
    (f, o) => Math.sin(f * 0.1 + o),
  );
  const bank = buildDriverBank(tl);
  assert.ok(bank);
  assert.equal(bank.dims, 3 + 4);
  assert.ok((bank.names[1] as string).includes('absent'));
  const out = new Float32Array(bank.dims);
  bank.sample(1.0, out);
  assert.equal(out[1], 0, 'an absent driver contributes exactly nothing');
  assert.equal(out[2], 0);
  assert.ok(Math.abs(out[0] as number) > 0);

  // no latent channel at all → no bank rather than a fake one
  const bare = fakeTimeline(frames, [{ name: 'stems', dims: 4, offset: 0 }], () => 1);
  assert.equal(buildDriverBank(bare), null);
});

// ── z-scoring ────────────────────────────────────────────────────────────────

test('z-scoring: each dim comes out mean 0, sd 1, and the stored stats say so', () => {
  const rng = makeRng(1234);
  const frames = 500;
  const dims = 8;
  const raw = new Float32Array(frames * dims);
  // dim d has mean 10*d and sd (d+1) — wildly different scales on purpose
  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < dims; d++) {
      const g = (rng() + rng() + rng() + rng() - 2) * 1.2; // roughly normal, |z| < 3
      raw[f * dims + d] = 10 * d + (d + 1) * g;
    }
  }
  const sig = bankOf(raw, frames, dims);
  const out = new Float32Array(dims);
  const sum = new Float64Array(dims);
  const sq = new Float64Array(dims);
  for (let f = 0; f < frames; f++) {
    sig.sample(f * 0.1, out);
    for (let d = 0; d < dims; d++) {
      sum[d] = (sum[d] as number) + (out[d] as number);
      sq[d] = (sq[d] as number) + (out[d] as number) ** 2;
    }
  }
  for (let d = 0; d < dims; d++) {
    assert.ok(Math.abs((sum[d] as number) / frames) < 1e-3, `dim ${d} mean`);
    assert.ok(Math.abs(Math.sqrt((sq[d] as number) / frames) - 1) < 1e-2, `dim ${d} sd`);
    assert.ok(Math.abs((sig.mean[d] as number) - 10 * d) < 0.5, `dim ${d} recorded mean`);
  }
});

test('z-scoring clamps outliers and survives a constant dimension', () => {
  const frames = 200;
  const dims = 2;
  const raw = new Float32Array(frames * dims);
  for (let f = 0; f < frames; f++) {
    raw[f * dims] = 0.5; // constant: sd 0
    raw[f * dims + 1] = f === 0 ? 1e6 : 0; // one enormous outlier
  }
  const sig = bankOf(raw, frames, dims);
  const out = new Float32Array(dims);
  for (let f = 0; f < frames; f++) {
    sig.sample(f * 0.1, out);
    assert.equal(out[0], 0, 'a constant dim contributes exactly nothing');
    assert.ok(Number.isFinite(out[1] as number));
    assert.ok(Math.abs(out[1] as number) <= Z_CLAMP + 1e-6, `|z| ${out[1]} exceeds the clamp`);
  }
  sig.sample(0, out);
  assert.equal(out[1], Z_CLAMP, 'the outlier saturates rather than dominating');
});

test('the bank caps the pc block at PC_DRIVER_COUNT however wide the latent is', () => {
  const frames = 60;
  const tl = fakeTimeline(
    frames,
    [
      { name: 'novelty4', dims: 1, offset: 0 },
      { name: 'novelty16', dims: 1, offset: 1 },
      { name: 'actChorus', dims: 1, offset: 2 },
      { name: 'latent', dims: 64, offset: 3 },
    ],
    (f, o) => Math.sin(f * 0.05 * (o + 1)) * (1 + (o % 7)),
  );
  const bank = buildDriverBank(tl);
  assert.ok(bank);
  assert.equal(bank.dims, STRUCTURE_DRIVERS.length + PC_DRIVER_COUNT);
  assert.equal(bank.label, `drivers-${bank.dims}`);
  assert.equal(bank.names[bank.dims - 1], `pc-${PC_DRIVER_COUNT}`);
  // every driver is standardised, structure channels included — that is what
  // makes one gain slider mean the same thing on all sixteen of them
  const out = new Float32Array(bank.dims);
  const sum = new Float64Array(bank.dims);
  const sq = new Float64Array(bank.dims);
  for (let f = 0; f < frames; f++) {
    bank.sample(f * 0.1, out);
    for (let d = 0; d < bank.dims; d++) {
      sum[d] = (sum[d] as number) + (out[d] as number);
      sq[d] = (sq[d] as number) + (out[d] as number) ** 2;
    }
  }
  for (let d = 0; d < bank.dims; d++) {
    assert.ok(Math.abs((sum[d] as number) / frames) < 1e-3, `${bank.names[d]} mean`);
    assert.ok(
      Math.abs(Math.sqrt((sq[d] as number) / frames) - 1) < 1e-2,
      `${bank.names[d]} sd`,
    );
  }
});

// ── wiring ───────────────────────────────────────────────────────────────────

test('projection directions are unit vectors and a pure function of (seed, index)', () => {
  const dims = 129; // odd, to exercise the Box–Muller tail
  const a = new Float32Array(dims);
  const b = new Float32Array(dims);
  for (const index of [0, 7, 91]) {
    unitDirection(0xc0ffee, index, dims, a);
    unitDirection(0xc0ffee, index, dims, b);
    let norm = 0;
    for (let d = 0; d < dims; d++) {
      assert.equal(a[d], b[d], 'same seed and index give the identical direction');
      norm += (a[d] as number) ** 2;
    }
    assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-5, `‖w‖ = ${Math.sqrt(norm)}`);
  }
  // different index, different direction — near-orthogonal in high dimensions
  unitDirection(0xc0ffee, 0, dims, a);
  unitDirection(0xc0ffee, 1, dims, b);
  let dot = 0;
  for (let d = 0; d < dims; d++) dot += (a[d] as number) * (b[d] as number);
  assert.ok(Math.abs(dot) < 0.4, `slots 0 and 1 are nearly parallel (dot ${dot})`);

  // different seed, different wiring
  unitDirection(1, 3, dims, a);
  unitDirection(2, 3, dims, b);
  let same = 0;
  for (let d = 0; d < dims; d++) if (a[d] === b[d]) same++;
  assert.ok(same < dims / 4, 'a different seed rewires');
});

test('w·ẑ is roughly N(0,1) for a standardised input — the reason depths transfer', () => {
  const dims = 256;
  const trials = 400;
  const w = new Float32Array(dims);
  const rng = makeRng(99);
  let sum = 0;
  let sq = 0;
  for (let t = 0; t < trials; t++) {
    unitDirection(0xabc, t, dims, w);
    let raw = 0;
    for (let d = 0; d < dims; d++) {
      // unit-variance input
      const g = (rng() + rng() + rng() + rng() - 2) * Math.sqrt(3);
      raw += (w[d] as number) * g;
    }
    sum += raw;
    sq += raw * raw;
  }
  const mean = sum / trials;
  const sd = Math.sqrt(sq / trials - mean * mean);
  assert.ok(Math.abs(mean) < 0.2, `mean ${mean}`);
  assert.ok(Math.abs(sd - 1) < 0.15, `sd ${sd}`);
});

// ── seeded personality ───────────────────────────────────────────────────────

test('the same seed reproduces an identical personality; a different one does not', () => {
  const defaults = defaultTheta();
  const a = baseVector(0x1234_5678, defaults, SLOTS, new Float64Array(LEN));
  const again = baseVector(0x1234_5678, defaults, SLOTS, new Float64Array(LEN));
  const other = baseVector(0x8765_4321, defaults, SLOTS, new Float64Array(LEN));
  for (let i = 0; i < LEN; i++) assert.equal(a[i], again[i], fieldNames(K)[i]);

  let differing = 0;
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 1 && Math.abs((a[i] as number) - (other[i] as number)) > 1e-9) differing++;
  }
  const modulated = MASK.reduce((x, y) => x + y, 0);
  assert.ok(
    differing > modulated * 0.9,
    `only ${differing} of ${modulated} modulatable slots differ between seeds`,
  );
});

test('personality differences are big enough to see, not third-decimal', () => {
  // "differences between seeds should be BIGGER" — measured as the mean
  // displacement from the shipped defaults, relative to each slot's excursion.
  const defaults = defaultTheta();
  let total = 0;
  let n = 0;
  for (const seed of [1, 2, 3, 11, 4242, 0xdead_beef]) {
    const b = baseVector(seed, defaults, SLOTS, new Float64Array(LEN));
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i] as ModSpec | null;
      if (!spec || spec.half === 0) continue;
      const d = spec.mult
        ? Math.abs(Math.log(Math.max(b[i] as number, 1e-9) / Math.max(defaults[i] as number, 1e-9)))
        : Math.abs((b[i] as number) - (defaults[i] as number));
      total += d / spec.half;
      n++;
    }
  }
  const mean = total / n;
  assert.ok(mean > 0.2, `mean personality displacement is only ${mean.toFixed(3)}× the half-range`);
});

test('every seeded base lands inside its own modulation range', () => {
  const defaults = defaultTheta();
  const out = new Float64Array(LEN);
  for (let seed = 0; seed < 400; seed++) {
    baseVector(seed * 2654435761, defaults, SLOTS, out);
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i] as ModSpec | null;
      if (!spec) continue;
      assert.ok(
        (out[i] as number) >= spec.lo - 1e-9 && (out[i] as number) <= spec.hi + 1e-9,
        `${fieldNames(K)[i]} = ${out[i]} outside [${spec.lo}, ${spec.hi}] at seed ${seed}`,
      );
    }
  }
});

// ── bounds, depth and group isolation ────────────────────────────────────────

test('bounds: 10k random ẑ never put a parameter outside [lo, hi]', () => {
  const dims = 64;
  const defaults = defaultTheta();
  const rng = makeRng(0xb0_11d5);
  const z = new Float32Array(dims);
  const depths = unitDepths(3); // over-driven on purpose
  for (let trial = 0; trial < 10_000; trial++) {
    // Adversarial as well as typical: full-scale clamped z, mixed signs.
    for (let d = 0; d < dims; d++) {
      z[d] = trial % 3 === 0 ? (rng() < 0.5 ? -Z_CLAMP : Z_CLAMP) : (rng() * 2 - 1) * Z_CLAMP;
    }
    const seed = (trial * 2246822519) >>> 0;
    const v = targetFor(seed, z, dims, defaults, 4, depths);
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i] as ModSpec | null;
      if (!spec) continue;
      const x = v[i] as number;
      assert.ok(
        Number.isFinite(x) && x >= spec.lo - 1e-9 && x <= spec.hi + 1e-9,
        `${fieldNames(K)[i]} = ${x} outside [${spec.lo}, ${spec.hi}] (trial ${trial})`,
      );
    }
  }
});

test('depth 0 is exactly the base — the personality with the music switched off', () => {
  const dims = 32;
  const defaults = defaultTheta();
  const z = new Float32Array(dims).map((_, d) => Math.sin(d) * 3);
  const base = baseVector(777, defaults, SLOTS, new Float64Array(LEN));
  const v = targetFor(777, z, dims, defaults, 0, unitDepths(1));
  for (let i = 0; i < LEN; i++) assert.equal(v[i], base[i], fieldNames(K)[i]);

  // and a zero *group* depth pins only that group
  const oneOff = unitDepths(1);
  oneOff.matrix = 0;
  const partial = targetFor(777, z, dims, defaults, 1, oneOff);
  let moved = 0;
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec) continue;
    if (spec.group === 'matrix') assert.equal(partial[i], base[i], fieldNames(K)[i]);
    else if (Math.abs((partial[i] as number) - (base[i] as number)) > 1e-9) moved++;
  }
  assert.ok(moved > 20, `only ${moved} non-matrix slots moved`);
});

// ── driver gains ─────────────────────────────────────────────────────────────

test('gain 0 silences a driver through every wiring, exactly', () => {
  const dims = 16;
  const defaults = defaultTheta();
  const depths = unitDepths(1);
  const z = new Float32Array(dims).map((_, d) => Math.sin(d * 1.7) * 2.5);

  // One driver at a time: put all the signal on driver `mute`, then mute it.
  // Every slot must land exactly on its base — not "close to", because the gain
  // multiplies the driver before the projection rather than trimming afterwards.
  for (const mute of [0, 3, 15]) {
    const only = new Float32Array(dims);
    only[mute] = 2.5;
    const gains = new Float32Array(dims).fill(1);
    gains[mute] = 0;
    const base = baseVector(4242, defaults, SLOTS, new Float64Array(LEN));
    const v = targetFor(4242, only, dims, defaults, 1, depths, gains);
    for (let i = 0; i < LEN; i++) assert.equal(v[i], base[i], `${fieldNames(K)[i]} (driver ${mute})`);

    // and with the gain back at 1 the very same input does move things
    const live = targetFor(4242, only, dims, defaults, 1, depths);
    let moved = 0;
    for (let i = 0; i < LEN; i++) {
      if (Math.abs((live[i] as number) - (base[i] as number)) > 1e-9) moved++;
    }
    assert.ok(moved > 20, `driver ${mute} at gain 1 moved only ${moved} slots`);
  }

  // all gains 0 → the whole bank is silent, whatever the music is doing
  const silent = targetFor(9, z, dims, defaults, 3, unitDepths(2), new Float32Array(dims));
  const base9 = baseVector(9, defaults, SLOTS, new Float64Array(LEN));
  for (let i = 0; i < LEN; i++) assert.equal(silent[i], base9[i], fieldNames(K)[i]);
});

test('gains scale monotonically: 2x the gain is a bigger excursion, same sign', () => {
  const dims = 16;
  const defaults = defaultTheta();
  const z = new Float32Array(dims).map((_, d) => Math.cos(d * 0.9));
  const base = baseVector(11, defaults, SLOTS, new Float64Array(LEN));
  const half = targetFor(11, z, dims, defaults, 1, unitDepths(1), new Float32Array(dims).fill(0.5));
  const full = targetFor(11, z, dims, defaults, 1, unitDepths(1), new Float32Array(dims).fill(1));
  let checked = 0;
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec || spec.mult) continue;
    const a = (half[i] as number) - (base[i] as number);
    const b = (full[i] as number) - (base[i] as number);
    if (Math.abs(b) < 1e-6) continue;
    assert.ok(a * b >= 0, `${fieldNames(K)[i]} flipped sign with gain`);
    assert.ok(Math.abs(a) <= Math.abs(b) + 1e-9, `${fieldNames(K)[i]} shrank with more gain`);
    checked++;
  }
  assert.ok(checked > 20, `only ${checked} additive slots checked`);
});

test('group isolation: zeroing one group leaves the others exactly where they were', () => {
  const dims = 48;
  const defaults = defaultTheta();
  const z = new Float32Array(dims).map((_, d) => Math.cos(d * 0.7) * 2);
  const all = targetFor(4242, z, dims, defaults, 1, unitDepths(1));
  for (const off of MOD_GROUPS) {
    const depths = unitDepths(1);
    depths[off] = 0;
    const v = targetFor(4242, z, dims, defaults, 1, depths);
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i] as ModSpec | null;
      if (!spec || spec.group === off) continue;
      assert.equal(v[i], all[i], `${fieldNames(K)[i]} moved when ${off} was zeroed`);
    }
  }
});

test('excursions at the default depth are legible, not a whisper', () => {
  // The user's complaint about the first build was that it read as subtle. With
  // ẑ ~ N(0,1) and unit projections, |tanh(1 · w·ẑ)| should average well clear of
  // zero — target ~0.5, i.e. half of each parameter's authored half-range.
  const dims = 128;
  const rng = makeRng(31337);
  const w = new Float32Array(dims);
  let sum = 0;
  const trials = 2000;
  for (let t = 0; t < trials; t++) {
    unitDirection(0x5151, t, dims, w);
    let raw = 0;
    for (let d = 0; d < dims; d++) {
      raw += (w[d] as number) * (rng() + rng() + rng() + rng() - 2) * Math.sqrt(3);
    }
    sum += Math.abs(Math.tanh(raw));
  }
  const mean = sum / trials;
  assert.ok(mean > 0.4, `mean |excursion| is ${mean.toFixed(3)} — too subtle`);
  assert.ok(mean < 0.85, `mean |excursion| is ${mean.toFixed(3)} — pinned to the rails`);
});

// ── slew ─────────────────────────────────────────────────────────────────────

test('slew step response: one time constant reaches 1 - 1/e of the step', () => {
  const classes = new Uint8Array([CLASS_FAST]);
  const slew = new SlewLimiter(classes, { fast: 1, medium: 1, slow: 1 });
  slew.reset([0]);
  const dt = 1 / 600;
  for (let i = 0; i < 600; i++) slew.step([1], dt);
  assert.ok(Math.abs((slew.value[0] as number) - (1 - Math.exp(-1))) < 1e-3);
  for (let i = 0; i < 6000; i++) {
    slew.step([1], dt);
    assert.ok((slew.value[0] as number) <= 1 + 1e-12, 'never overshoots');
  }
  assert.ok((slew.value[0] as number) > 0.999);
});

test('slew classes move at different speeds and τ=0 is a passthrough', () => {
  const classes = new Uint8Array([CLASS_FAST, CLASS_SLOW]);
  const slew = new SlewLimiter(classes, { fast: 0.1, medium: 1, slow: 10 });
  slew.reset([0, 0]);
  for (let i = 0; i < 60; i++) slew.step([1, 1], 1 / 60);
  assert.ok((slew.value[0] as number) > 0.99, 'fast has arrived');
  assert.ok((slew.value[1] as number) < 0.15, 'slow has barely moved');

  const instant = new SlewLimiter(new Uint8Array([CLASS_FAST]), { fast: 0, medium: 1, slow: 1 });
  instant.reset([0]);
  instant.step([0.7], 1 / 60);
  assert.equal(instant.value[0], 0.7);
});

test('response speed is a clock multiplier: 2× dt is 2× as fast', () => {
  const classes = new Uint8Array([CLASS_FAST]);
  const slow = new SlewLimiter(classes, { fast: 1, medium: 1, slow: 1 });
  const fast = new SlewLimiter(classes, { fast: 1, medium: 1, slow: 1 });
  slow.reset([0]);
  fast.reset([0]);
  for (let i = 0; i < 120; i++) {
    slow.step([1], 1 / 120);
    fast.step([1], (1 / 120) * 2);
  }
  // after the same wall time, the 2× lane has consumed 2 time constants
  assert.ok(Math.abs((slow.value[0] as number) - (1 - Math.exp(-1))) < 1e-2);
  assert.ok(Math.abs((fast.value[0] as number) - (1 - Math.exp(-2))) < 1e-2);
});

test('boundary snap moves only its own class, by exactly the given fraction', () => {
  const classes = new Uint8Array([CLASS_FAST, CLASS_SLOW]);
  const slew = new SlewLimiter(classes, { fast: 0.1, medium: 1, slow: 10 });
  slew.reset([0, 0]);
  slew.snapClass(CLASS_SLOW, [1, 1], 0.6);
  assert.equal(slew.value[0], 0);
  assert.ok(Math.abs((slew.value[1] as number) - 0.6) < 1e-12);
  slew.snapClass(CLASS_SLOW, [1, 1], 0);
  assert.ok(Math.abs((slew.value[1] as number) - 0.6) < 1e-12, 'fraction 0 is a no-op');
});

// ── persistence ──────────────────────────────────────────────────────────────

test('a v4 config survives serialize → parse, palette and grade and all', () => {
  const base = defaultConfig(K);
  base.palette.colors[0] = '#123456';
  base.palette.saturation = 0.7;
  base.render.grade.tonemap = 'aces';
  base.render.bloom.threshold = 0.91;
  const cfg = defaultModulationConfig(base);
  cfg.depth = 1.75;
  cfg.groupDepth.structure = 2.25;
  cfg.responseSpeed = 3;
  cfg.slew.fast = 0.03;
  cfg.enabled = false;
  cfg.driverGains = [0, 1, 2, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
  cfg.stemFollow.floor = 0.12;
  cfg.stemFollow.curve = 1.8;
  cfg.stemFollow.smoothingMs = 420;
  // shared by reference with the live config, exactly like the anchor era
  assert.equal(cfg.palette, base.palette);
  assert.equal(cfg.render, base.render);

  const back = parseModulation(serializeModulation(cfg));
  assert.equal(back.version, 4);
  assert.deepEqual(back.driverGains, cfg.driverGains);
  assert.deepEqual(back.stemFollow, cfg.stemFollow);
  assert.deepEqual(back.palette, cfg.palette);
  assert.deepEqual(back.render, cfg.render);
  assert.equal(back.depth, 1.75);
  assert.equal(back.groupDepth.structure, 2.25);
  assert.equal(back.responseSpeed, 3);
  assert.equal(back.slew.fast, 0.03);
  assert.equal(back.enabled, false);
  assert.deepEqual(back.boundary, cfg.boundary);
  assert.ok(!('seed' in back), 'the seed is a run input, never a config field');
});

test('a v2 file loads its palette and render block and discards the anchors', () => {
  const base = defaultConfig(K);
  base.palette.colors[1] = '#abcdef';
  const v2 = {
    version: 2,
    speciesCount: K,
    palette: { colors: base.palette.colors, saturation: 0.4, brightness: 1.1 },
    render: { ...defaultRenderConfig(), grade: { ...defaultRenderConfig().grade, vignette: 0.42 } },
    latentDims: 64,
    kmeans: { k: 8, seed: 1, iterations: 60 },
    temperature: 0.35,
    distanceScale: 12.5,
    slew: { fast: 0.12, medium: 1.2, slow: 8 },
    boundary: { enabled: false, snapFraction: 0.3, respawnFraction: 0.05 },
    anchors: [{ id: 'a0', name: 'anchor 0', center: [1, 2, 3], preset: { species: [] } }],
  };
  const back = parseModulation(JSON.stringify(v2));
  assert.equal(back.version, 4);
  assert.equal(back.palette.colors[1], '#abcdef');
  assert.equal(back.palette.saturation, 0.4);
  assert.equal(back.render.grade.vignette, 0.42);
  assert.ok(!('anchors' in back), 'anchors are gone');
  assert.ok(!('kmeans' in back) && !('temperature' in back), 'the simplex is gone');
  // the scene era's slow slew rates are NOT carried over — Revision 3 wants faster
  assert.equal(back.slew.fast, 0.05);
  assert.equal(back.depth, 1);
  // boundaries survive: they are an event, not scene machinery
  assert.equal(back.boundary.enabled, false);
  assert.equal(back.boundary.snapFraction, 0.3);
});

test('a v3 file migrates to v4 losslessly: nothing dropped, gains default to 1', () => {
  const v3 = {
    version: 3,
    speciesCount: K,
    palette: { colors: ['#111111', '#222222', '#333333', '#444444'], saturation: 0.8, brightness: 1.2 },
    render: { ...defaultRenderConfig(), grade: { ...defaultRenderConfig().grade, vignette: 0.31 } },
    enabled: false,
    depth: 2.5,
    // brightness is a group v3 had and v4 does not; it must not fail the parse
    groupDepth: { structure: 1.4, matrix: 0.2, population: 0.9, brightness: 1.7, decay: 0.3 },
    responseSpeed: 2.75,
    slew: { fast: 0.07, medium: 0.9, slow: 12 },
    boundary: { enabled: false, snapFraction: 0.25, respawnFraction: 0.4 },
  };
  const back = parseModulation(JSON.stringify(v3));
  assert.equal(back.version, 4);
  // everything v3 carried survives verbatim
  assert.equal(back.depth, 2.5);
  assert.equal(back.enabled, false);
  assert.equal(back.responseSpeed, 2.75);
  assert.equal(back.groupDepth.structure, 1.4);
  assert.equal(back.groupDepth.matrix, 0.2);
  assert.equal(back.groupDepth.population, 0.9);
  assert.equal(back.groupDepth.decay, 0.3);
  assert.equal(back.slew.slow, 12);
  assert.deepEqual(back.boundary, v3.boundary);
  assert.equal(back.palette.saturation, 0.8);
  assert.equal(back.render.grade.vignette, 0.31);
  // the retired group is dropped, not carried as dead weight
  assert.ok(!('brightness' in back.groupDepth), 'the brightness group depth is gone');
  // and the two new blocks arrive at their defaults
  assert.deepEqual(back.driverGains, [], 'no gains authored → the Modulator fills them with 1');
  assert.deepEqual(back.stemFollow, defaultStemFollow());
});

test('driver gains are clamped, not rejected, and junk entries default to 1', () => {
  const file = JSON.stringify({
    version: 4,
    speciesCount: K,
    driverGains: [-5, 0, 1, 9, 'x', null, 1.5],
  });
  assert.deepEqual(parseModulation(file).driverGains, [0, 0, 1, 2, 1, 1, 1.5]);
});

test('an unknown version is rejected rather than half-applied', () => {
  assert.throws(() => parseModulation(JSON.stringify({ version: 99, speciesCount: 4 })), /version 99/);
});

test('a hostile speciesCount is rejected before it can size anything', () => {
  const file = (speciesCount: unknown): string => JSON.stringify({ version: 3, speciesCount });
  assert.throws(() => parseModulation(file(1e9)), /speciesCount is not an integer in 1\.\.64/);
  for (const bad of [0, -4, 2.5, Number.NaN, '4']) {
    assert.throws(() => parseModulation(file(bad)), /speciesCount/);
  }
});

test('a config authored for another K is refused rather than half-applied', () => {
  const cfg = defaultModulationConfig(defaultConfig(4));
  assert.ok(modulationFits(cfg, 4));
  assert.ok(!modulationFits(cfg, 6));
});

test('a loaded render block is merged into the live object, not swapped for it', () => {
  const live = defaultRenderConfig();
  const loaded = defaultRenderConfig();
  loaded.grade.vignette = 0.9;
  loaded.bloom.intensity = 2;
  mergeRenderConfig(live, loaded);
  assert.equal(live.grade.vignette, 0.9);
  assert.equal(live.bloom.intensity, 2);
  assert.notEqual(live.grade, loaded.grade);
});

// ── regressions from the 2026-08-07 review ───────────────────────────────────

test('a non-finite sample is contained: it never poisons its dim or any frame', () => {
  const frames = 8;
  const dims = 4;
  const raw = new Float32Array(frames * dims).map((_, i) => Math.sin(i) * 3);
  raw[5] = NaN; // frame 1, dim 1
  const sig = bankOf(raw, frames, dims);
  assert.ok(Number.isFinite(sig.mean[1] as number), 'mean of the affected dim');
  assert.ok(Number.isFinite(sig.std[1] as number), 'sd of the affected dim');
  const out = new Float32Array(dims);
  for (let f = 0; f < frames; f++) {
    sig.sample(f * 0.1, out);
    for (let d = 0; d < dims; d++)
      assert.ok(Number.isFinite(out[d] as number), `frame ${f} dim ${d} is ${out[d]}`);
  }
  // interpolation across the bad frame stays finite too
  sig.sample(0.05, out);
  for (let d = 0; d < dims; d++) assert.ok(Number.isFinite(out[d] as number));
  sig.sample(0.1, out);
  assert.equal(out[1], 0, 'the bad sample itself reads as exactly no contribution');
});

test('the seeded personality puts no mass on a bound, even where the default hugs one', () => {
  const defaults = defaultTheta();
  // species0.diffuseCentre: default 0.12, jitter 0.2, lo 0.111 — the worst case.
  const slot = 5;
  const spec = SLOTS[slot] as ModSpec;
  assert.ok(spec && !spec.mult && Math.abs((defaults[slot] as number) - spec.lo) < spec.jitter / 2);
  const seen: number[] = [];
  const out = new Float64Array(LEN);
  for (let s = 1; s <= 600; s++) {
    baseVector(s, defaults, SLOTS, out);
    const v = out[slot] as number;
    assert.ok(v >= spec.lo && v <= spec.hi, `seed ${s} left the range: ${v}`);
    seen.push(v);
  }
  const onBound = seen.filter((v) => v === spec.lo || v === spec.hi).length;
  assert.equal(onBound, 0, `${onBound}/600 seeds collected on a bound`);
  const mean = seen.reduce((a, b) => a + b, 0) / seen.length;
  const sd = Math.sqrt(seen.reduce((a, b) => a + (b - mean) ** 2, 0) / seen.length);
  // The `onBound` assertion above is what catches a clamp. This one catches the
  // other tempting fix — shrinking the draw to symmetric headroom, which for this
  // slot is ±0.009 and would collapse the spread to sd ~0.003.
  assert.ok(sd > 0.05, `personality spread collapsed to sd ${sd}`);
});

// ── stem-follow: the brightness lane (plan.md Revision 4) ────────────────────

test('stem-follow: silence lands on the floor, full activity is 1, monotone between', () => {
  const cfg = { ...defaultStemFollow(), floor: 0.25, curve: 1 };
  assert.equal(followMultiplier(0, cfg), 0.25, 'a silent instrument keeps the floor, not zero');
  assert.equal(followMultiplier(1, cfg), 1, 'a loud instrument is untouched');
  assert.ok(Math.abs(followMultiplier(0.5, cfg) - 0.625) < 1e-12);
  let prev = -1;
  for (let a = 0; a <= 1.0001; a += 0.05) {
    const v = followMultiplier(a, cfg);
    assert.ok(v >= prev, `not monotone at ${a}`);
    assert.ok(v >= 0.25 - 1e-12 && v <= 1 + 1e-12, `${v} outside [floor, 1]`);
    prev = v;
  }
  // out-of-range activity cannot push the multiplier out of range either
  assert.equal(followMultiplier(-3, cfg), 0.25);
  assert.equal(followMultiplier(7, cfg), 1);
  assert.equal(followMultiplier(Number.NaN, cfg), 0.25);
});

test('stem-follow: floor 0 is a blackout, floor 1 is a no-op, curve biases toward loud', () => {
  const base = defaultStemFollow();
  assert.equal(followMultiplier(0, { ...base, floor: 0 }), 0);
  assert.equal(followMultiplier(0, { ...base, floor: 1 }), 1);
  assert.equal(followMultiplier(0.3, { ...base, floor: 1 }), 1);
  // curve > 1 dims the middle harder; curve < 1 lifts it. Endpoints never move.
  const mid = 0.5;
  const flat = followMultiplier(mid, { ...base, floor: 0, curve: 1 });
  const steep = followMultiplier(mid, { ...base, floor: 0, curve: 2.5 });
  const soft = followMultiplier(mid, { ...base, floor: 0, curve: 0.4 });
  assert.ok(steep < flat && flat < soft, `${steep} < ${flat} < ${soft}`);
  for (const curve of [0.4, 1, 2.5]) {
    assert.equal(followMultiplier(0, { ...base, floor: 0, curve }), 0);
    assert.equal(followMultiplier(1, { ...base, floor: 0, curve }), 1);
  }
  // disabled is exactly 1 whatever else is set
  assert.equal(followMultiplier(0, { ...base, enabled: false, floor: 0 }), 1);
});

test('stem-follow smoothing: a cut fades over its time constant rather than snapping', () => {
  const cfg = { ...defaultStemFollow(), floor: 0, curve: 1, smoothingMs: 300 };
  const follow = new StemFollow(4, 4);
  const dt = 1 / 60;
  const loud = new Float32Array([1, 1, 1, 1]);
  const cut = new Float32Array([1, 1, 0, 1]); // vocals (species 2) drop out
  for (let i = 0; i < 600; i++) follow.update(loud, dt, cfg);
  assert.ok(Math.abs((follow.multiplier[2] as number) - 1) < 1e-3, 'settled at full');

  // one time constant after the cut it should be ~1/e of the way down, not at 0
  for (let i = 0; i < Math.round(0.3 / dt); i++) follow.update(cut, dt, cfg);
  const afterTau = follow.multiplier[2] as number;
  assert.ok(Math.abs(afterTau - Math.exp(-1)) < 0.02, `after τ: ${afterTau}`);
  assert.ok(
    Math.abs((follow.multiplier[0] as number) - 1) < 1e-5,
    'the other species did not move',
  );

  // …and it does get all the way there
  for (let i = 0; i < 600; i++) follow.update(cut, dt, cfg);
  assert.ok((follow.multiplier[2] as number) < 1e-3, 'the cut species is fully dimmed');

  // with a floor it stops at the floor instead of at black
  const floored = new StemFollow(4, 4);
  const withFloor = { ...cfg, floor: 0.25 };
  for (let i = 0; i < 600; i++) floored.update(cut, dt, withFloor);
  assert.ok(Math.abs((floored.multiplier[2] as number) - 0.25) < 1e-3, 'ghost floor');
});

test('stem-follow: a seek snaps the EMA instead of dragging the old moment across', () => {
  // The A/B property: restoring a snapshot must show the brightness that moment
  // had, not a 300 ms fade out of wherever the transport was standing.
  const cfg = { ...defaultStemFollow(), floor: 0, curve: 1, smoothingMs: 300 };
  const follow = new StemFollow(4, 4);
  const dt = 1 / 60;
  const loud = new Float32Array([1, 1, 1, 1]);
  const quiet = new Float32Array([0, 0, 0, 0]);
  for (let i = 0; i < 600; i++) follow.update(loud, dt, cfg);

  // eased: one tick into a quiet passage is still essentially fully bright
  const eased = new StemFollow(4, 4);
  for (let i = 0; i < 600; i++) eased.update(loud, dt, cfg);
  eased.update(quiet, dt, cfg);
  assert.ok((eased.multiplier[0] as number) > 0.9, 'the ease carries the old level');

  // snapped: the same tick, flagged as a discontinuity, lands on the new level
  follow.update(quiet, dt, cfg, true);
  assert.ok((follow.multiplier[0] as number) < 1e-6, 'the snap adopts the new level');
  assert.equal(follow.activity[0], 0);
});

test('stem-follow leaves species with no stem, and tracks with no stems, alone', () => {
  const cfg = defaultStemFollow();
  // K = 6 against a 4-dim stems channel: species 4 and 5 have no instrument
  const follow = new StemFollow(6, 4);
  assert.equal(follow.stemOf(3), 3);
  assert.equal(follow.stemOf(4), -1);
  for (let i = 0; i < 100; i++) follow.update(new Float32Array([0, 0, 0, 0, 0, 0]), 1 / 60, cfg);
  assert.ok((follow.multiplier[0] as number) < 0.3, 'a keyed species dims');
  assert.equal(follow.multiplier[4], 1, 'an unkeyed species is untouched, not blacked out');

  // no stems channel at all: the lane is a no-op rather than a global dimming
  const none = new StemFollow(4, 0);
  for (let i = 0; i < 100; i++) none.update(null, 1 / 60, cfg);
  for (let k = 0; k < 4; k++) assert.equal(none.multiplier[k], 1);
});

test('stem-follow: the shipped default leaves a visible ghost, not a blackout', () => {
  // The user asked for "visibly diminish", not "disappear" — so the default
  // floor has to be clearly above black and clearly below full.
  const d = defaultStemFollow();
  assert.ok(d.enabled, 'the lane is on by default');
  assert.ok(d.floor >= 0.15 && d.floor <= 0.4, `floor ${d.floor} is not a ghost`);
  assert.ok(d.smoothingMs >= 200 && d.smoothingMs <= 400, `smoothing ${d.smoothingMs} ms`);
});

// ── rebase-on-edit: the base is the user's, not the seed's (VST semantics) ────

/**
 * The rebase lane is driven through a real `Modulator` — the class under test —
 * against a **synthetic** registry rather than physarum's.
 *
 * Two reasons, and the second is the load-bearing one. First, the properties
 * here are about exactly which slot moved, so the slot table has to be small
 * enough to state them over by index: one ordinary masked additive slot, one
 * more in a different group, an excluded slot with no spec at all, a
 * multiplicative slot, and — the case no shipped registry offers — a slot that
 * *has* a `ModSpec` but is **not** in the mask, which is the only way to test the
 * two halves of the "never rebase what we never write" gate apart. Second, a
 * registry assembled here cannot shift underneath the run when someone edits a
 * sim's slot table.
 */
const RIG_SLOTS: (ModSpec | null)[] = [
  { group: 'structure', lo: 0, hi: 10, half: 1, jitter: 0.1, mult: false },
  { group: 'matrix', lo: -2, hi: 2, half: 0.5, jitter: 0.1, mult: false },
  null,
  { group: 'decay', lo: 0.1, hi: 10, half: 0.5, jitter: 0.1, mult: true },
  { group: 'population', lo: 0, hi: 1, half: 0.25, jitter: 0.1, mult: false },
];
/** Slot 4 carries a spec and is deliberately out of the mask; slot 2 has neither. */
const RIG_MASK = Uint8Array.from([1, 1, 0, 1, 0]);
/** The shipped θ this rig starts from — every value inside its slot's range. */
const RIG_THETA = [5, 0, 7.5, 1, 0.5];
const RIG_LEN = RIG_SLOTS.length;
/** The masked slots, i.e. the only ones the modulator ever writes or rebases. */
const RIG_MASKED = [0, 1, 3];

interface Rig {
  mod: Modulator;
  /** θ as the "sim" holds it — the panel writes straight into this. */
  theta: Float64Array;
  /** what a panel widget does: write the config directly and tell nobody */
  edit: (index: number, value: number) => void;
  /** one modulator tick */
  tick: (dt?: number) => void;
  /** θ as the modulator last left it */
  read: (index: number) => number;
  /** the θ the constructor captured as `defaults`, for seeded-base expectations */
  defaults: Float64Array;
  cfg: ModulationConfig;
}

/**
 * @param tau slew time constant. 0 makes the limiter a passthrough, so a readback
 *   is the *target* and the assertions are about `base`. A huge value freezes it,
 *   so a readback is the limiter's internal state and the assertions are about
 *   where the slew was left — which is how "the drag does not snap back" is
 *   observable at all without reaching into a private field.
 */
function makeRig(tau = 0): Rig {
  const registry: ThetaRegistry = {
    length: RIG_LEN,
    slots: RIG_SLOTS,
    mask: RIG_MASK,
    classes: new Uint8Array(RIG_LEN).fill(CLASS_FAST),
    names: RIG_SLOTS.map((_, i) => `rig${i}`),
  };
  const theta = Float64Array.from(RIG_THETA);
  const target: ModTarget = {
    simId: 'rig',
    config: {
      speciesCount: 1,
      palette: { colors: ['#ffffff'], saturation: 1, brightness: 1 },
      render: defaultRenderConfig(),
      species: [{ name: 'a', brightness: 1 }],
    },
    currentSeed: 1,
    onSeedChange: null,
    registry: () => registry,
    // A fresh copy every call, exactly like a real target's
    // presetToVector(presetFromConfig(...)): the modulator must not be able to
    // alias the live vector and accidentally diff it against itself.
    currentVector: () => Float64Array.from(theta),
    // Masked write only, which is what every real `applyVector` does — and what
    // makes an edit to an unmasked slot stay put instead of being stamped over.
    applyTheta: (v, mask) => {
      for (let i = 0; i < RIG_LEN; i++) {
        if (!mask || mask[i] === 1) theta[i] = v[i] as number;
      }
    },
    invalidatePalette: () => {},
    setBrightFollow: () => {},
    setImpulses: () => {},
    stemMap: () => Int32Array.from([-1]),
    partialReseed: () => {},
    reseed: () => {},
    snapshot: () => false,
    restoreSnapshot: () => false,
    clearSnapshot: () => {},
    hasSnapshot: false,
  };
  const sampler = {
    getChannel: () => undefined,
    segmentIndexAt: () => -1,
  } as unknown as TimelineSampler;

  // Every driver constant, so each one z-scores to exactly 0 and the modulated
  // target of a slot is exactly its base. That makes "the base moved" an exact
  // equality rather than a tolerance, and leaves `targetFor` as the way to probe
  // a non-zero z deliberately.
  const dims = 4;
  const frames = 16;
  const bank = bankOf(new Float32Array(frames * dims).fill(1), frames, dims);

  const defaults = Float64Array.from(theta);
  const cfg = defaultModulationConfig(target.config, 'rig');
  cfg.depth = 1;
  for (const g of MOD_GROUPS) cfg.groupDepth[g] = 1;
  cfg.responseSpeed = 1;
  cfg.slew = { fast: tau, medium: tau, slow: tau };
  const mod = new Modulator(target, sampler, bank, cfg, 0xa5a5);

  let t = 0;
  return {
    mod,
    theta,
    defaults,
    cfg,
    edit: (index, value) => {
      theta[index] = value;
    },
    tick: (dt = 1 / 60) => {
      t++;
      mod.update({ tick: t, time: t * dt, values: new Float32Array(4) }, dt);
    },
    read: (index) => theta[index] as number,
  };
}

/** The rig's registry is the shape the tests below are stated over. */
test('the rebase rig registry has the mask/spec combinations the gate needs', () => {
  assert.equal(RIG_MASK.length, RIG_LEN);
  assert.equal(RIG_SLOTS[2], null, 'slot 2 must be excluded outright');
  assert.equal(RIG_MASK[2], 0);
  assert.ok(RIG_SLOTS[4], 'slot 4 must carry a spec');
  assert.equal(RIG_MASK[4], 0, 'slot 4 must be out of the mask — the isolating case');
  for (const i of RIG_MASKED) assert.ok(RIG_SLOTS[i] && RIG_MASK[i] === 1, `slot ${i}`);
});

test('a modulated tick with every driver constant leaves theta exactly on the base', () => {
  // The premise the exactness of every assertion below rests on: z = 0, so the
  // target is the base itself and nothing drifts tick to tick.
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  const before = rig.mod.baseValues();
  for (let i = 0; i < 10; i++) rig.tick();
  for (const i of RIG_MASKED) {
    assert.equal(rig.read(i), before[i], `slot ${i} drifted with a silent bank`);
    assert.equal(rig.mod.baseValue(i), before[i], `slot ${i} base drifted`);
  }
  assert.equal(rig.mod.rebaseCount, 0, 'a silent bank produced phantom edits');
});

// ── 1. markApplied ───────────────────────────────────────────────────────────

test('markApplied makes a write read as ours: the quantisation path is not an edit', () => {
  // Tweakpane's refresh pushes every bound number through its widget's `step` and
  // writes the rounded result back. Paired with markApplied that must be inert;
  // unpaired it is indistinguishable from a drag, which is the bug the pairing
  // exists to prevent — and the contrast below is the whole contract.
  const quantise = (v: number): number => Math.round(v * 100) / 100 + 0.005;

  const paired = makeRig(0);
  paired.mod.setMode('modulated');
  paired.tick();
  const base0 = paired.mod.baseValue(0);
  const count0 = paired.mod.rebaseCount;
  for (const i of RIG_MASKED) paired.edit(i, quantise(paired.read(i)));
  paired.mod.markApplied();
  paired.tick();
  assert.equal(paired.mod.rebaseCount, count0, 'a marked write was read as a human edit');
  assert.equal(paired.mod.baseValue(0), base0, 'the base moved on a marked write');

  // …and without the pairing the identical write is an edit, on every masked slot
  // at once — the "random-walk by the excursion twice a second" failure.
  const unpaired = makeRig(0);
  unpaired.mod.setMode('modulated');
  unpaired.tick();
  const was = unpaired.mod.baseValue(0);
  const quantised = quantise(unpaired.read(0));
  for (const i of RIG_MASKED) unpaired.edit(i, quantise(unpaired.read(i)));
  unpaired.tick();
  assert.equal(unpaired.mod.rebaseCount, RIG_MASKED.length, 'the unmarked write was not detected');
  assert.equal(unpaired.mod.baseValue(0), quantised, 'the base did not follow the write');
  assert.notEqual(unpaired.mod.baseValue(0), was);
});

test('markApplied with no update in between still counts as ours on the next tick', () => {
  // The ordering the contract actually promises: edit -> markApplied -> update.
  // There is no tick between the write and the mark, which is exactly the
  // situation a refresh creates.
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.tick();
  const base = rig.mod.baseValues();
  rig.edit(0, 9.25);
  rig.edit(3, 4.5);
  rig.mod.markApplied();
  rig.tick();
  rig.tick();
  assert.equal(rig.mod.rebaseCount, 0);
  for (const i of RIG_MASKED) assert.equal(rig.mod.baseValue(i), base[i], `slot ${i} base moved`);
});

// ── 2. rebase on edit ────────────────────────────────────────────────────────

test('an edit during a modulated tick becomes the slot’s new base', () => {
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.tick();
  const others = rig.mod.baseValues();

  rig.edit(0, 7.25);
  rig.tick();
  assert.equal(rig.mod.baseValue(0), 7.25, 'the base did not follow the drag');
  assert.equal(rig.mod.rebaseCount, 1, 'one edited slot is one rebase');
  // …and only that slot moved
  for (const i of [1, 3]) assert.equal(rig.mod.baseValue(i), others[i], `slot ${i} moved too`);

  // A second edit in the same drag keeps re-anchoring, one rebase per slot write.
  rig.edit(0, 7.5);
  rig.tick();
  assert.equal(rig.mod.baseValue(0), 7.5);
  assert.equal(rig.mod.rebaseCount, 2);

  // Two slots in one tick is two rebases, not one.
  rig.edit(0, 6);
  rig.edit(1, 1.25);
  rig.tick();
  assert.equal(rig.mod.rebaseCount, 4, 'rebaseCount counts slot writes, not ticks');
  assert.equal(rig.mod.baseValue(1), 1.25);
});

test('after a rebase the excursion band recentres on the edited value', () => {
  // The point of the feature: the music now breathes around where the human left
  // the slider, so the *pure* target path has to agree with the new base.
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.tick();

  const spec0 = RIG_SLOTS[0] as ModSpec;
  const spec3 = RIG_SLOTS[3] as ModSpec;
  const zero = new Float32Array(4);
  const before = rig.mod.targetFor(zero)[0] as number;
  assert.equal(before, rig.mod.baseValue(0), 'z = 0 must land exactly on the base');

  rig.edit(0, 7.25); // additive
  rig.edit(3, 2); // multiplicative
  rig.tick();

  assert.equal(rig.mod.targetFor(zero)[0], 7.25, 'z = 0 no longer lands on the edited base');
  assert.equal(rig.mod.targetFor(zero)[3], 2);
  assert.notEqual(rig.mod.targetFor(zero)[0], before);

  // …and the band around it is base ± half (× exp(±half) for a mult slot),
  // clamped into the slot's own range, for every z the drivers can produce.
  const rng = makeRng(0x1eaf);
  const z = new Float32Array(4);
  let sawHigh = false;
  let sawLow = false;
  for (let trial = 0; trial < 2000; trial++) {
    for (let d = 0; d < 4; d++) z[d] = (rng() * 2 - 1) * Z_CLAMP;
    const out = rig.mod.targetFor(z);
    const a = out[0] as number;
    const m = out[3] as number;
    assert.ok(
      a >= Math.max(7.25 - spec0.half, spec0.lo) - 1e-12 && a <= Math.min(7.25 + spec0.half, spec0.hi) + 1e-12,
      `additive target ${a} outside 7.25 +/- ${spec0.half}`,
    );
    assert.ok(
      m >= Math.max(2 * Math.exp(-spec3.half), spec3.lo) - 1e-12 &&
        m <= Math.min(2 * Math.exp(spec3.half), spec3.hi) + 1e-12,
      `multiplicative target ${m} outside 2 * exp(+/-${spec3.half})`,
    );
    if (a > 7.25 + spec0.half * 0.8) sawHigh = true;
    if (a < 7.25 - spec0.half * 0.8) sawLow = true;
  }
  // The band is a band, not a pin: a clamp bug that collapsed it would still
  // satisfy the bounds above.
  assert.ok(sawHigh && sawLow, 'the recentred band never reached either end');
});

test('the slew continues from under the pointer: a drag does not snap back', () => {
  // With the limiter effectively frozen, a readback is its internal state. If the
  // rebase had moved only `base`, the next step would drag the value back toward
  // where the parameter was and the drag would still fight the music.
  const rig = makeRig(1e6);
  rig.mod.setMode('modulated');
  rig.tick();
  const was = rig.read(0);
  assert.ok(Math.abs(was - 9) > 3, 'the fixture needs the edit to be far from the old value');

  rig.edit(0, 9);
  rig.tick();
  assert.ok(Math.abs(rig.read(0) - 9) < 1e-3, `the value snapped back to ${rig.read(0)}`);
  for (let i = 0; i < 30; i++) rig.tick();
  assert.ok(Math.abs(rig.read(0) - 9) < 1e-2, 'the value crept away from the pointer');
});

test('manual mode never rebases, however much the panel writes', () => {
  const rig = makeRig(0);
  assert.equal(rig.mod.mode, 'manual', 'the modulator starts manual');
  const base = rig.mod.baseValues();
  for (let round = 0; round < 5; round++) {
    for (const i of RIG_MASKED) rig.edit(i, round + 1);
    rig.tick();
  }
  assert.equal(rig.mod.rebaseCount, 0, 'manual mode rebased');
  for (const i of RIG_MASKED) assert.equal(rig.mod.baseValue(i), base[i], `slot ${i} base moved`);
  // …and manual mode wrote nothing either: the edits are still standing.
  for (const i of RIG_MASKED) assert.equal(rig.read(i), 5);
});

test('an unmasked slot is never rebased — the modulator does not write it', () => {
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.tick();
  const base = rig.mod.baseValues();

  // Slot 2 has no spec; slot 4 has one but is out of the mask. Both are edits by
  // any other measure, and neither may move a base or the counter.
  rig.edit(2, 99);
  rig.edit(4, 0.75);
  rig.tick();
  rig.tick();
  assert.equal(rig.mod.rebaseCount, 0, 'an unmasked edit was rebased');
  assert.equal(rig.mod.baseValue(2), base[2]);
  assert.equal(rig.mod.baseValue(4), base[4]);
  // …and the modulator left the values alone, rather than stamping over them
  assert.equal(rig.read(2), 99, 'the modulator wrote an unmasked slot');
  assert.equal(rig.read(4), 0.75, 'the modulator wrote an unmasked slot');

  // The very same tick shape on a masked slot is one rebase, so the test above
  // is not passing because nothing was detected at all.
  rig.edit(0, 3.5);
  rig.tick();
  assert.equal(rig.mod.rebaseCount, 1);
});

test('an edit outside the modulation range clamps the base but not the pointer', () => {
  // A panel slider is bounded by the slot's *hard* theta bound, which is wider
  // than the ModSpec range. Base has to clamp or the target sits pinned at the
  // bound for every z; the slew must not, or the value jumps off the pointer.
  const spec = RIG_SLOTS[1] as ModSpec;
  for (const [label, edited, want] of [
    ['above hi', 5, spec.hi],
    ['below lo', -5, spec.lo],
  ] as [string, number, number][]) {
    const rig = makeRig(1e6);
    rig.mod.setMode('modulated');
    rig.tick();
    rig.edit(1, edited);
    rig.tick();
    assert.equal(rig.mod.baseValue(1), want, `${label}: the base was not clamped`);
    assert.ok(Math.abs(rig.read(1) - edited) < 1e-3, `${label}: the value jumped to ${rig.read(1)}`);

    // …and with the limiter live it settles onto the clamped base, not past it.
    const settled = makeRig(0);
    settled.mod.setMode('modulated');
    settled.tick();
    settled.edit(1, edited);
    settled.tick();
    settled.tick();
    assert.equal(settled.read(1), want, `${label}: the settled value left the range`);
  }
});

test('a non-finite write is refused rather than latched into the base', () => {
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.tick();
  const base = rig.mod.baseValues();
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    rig.edit(0, bad);
    rig.tick();
    assert.ok(Number.isFinite(rig.mod.baseValue(0)), `${bad} reached the base`);
    assert.equal(rig.mod.baseValue(0), base[0], `${bad} moved the base`);
    assert.ok(Number.isFinite(rig.read(0)), `${bad} survived into theta`);
  }
  assert.equal(rig.mod.rebaseCount, 0, 'a non-finite write was counted as a rebase');
});

// ── 3. adoptBase and the mode flip ───────────────────────────────────────────

test('setMode("modulated") adopts the sliders rather than gliding back to the seed', () => {
  const rig = makeRig(0);
  const seeded = rig.mod.baseValues();
  // A manual session: the human moves things, and nothing rebases while they do.
  rig.edit(0, 8.5);
  rig.edit(1, -1.5);
  rig.edit(3, 3);
  rig.tick();
  assert.equal(rig.mod.rebaseCount, 0);
  assert.equal(rig.mod.baseValue(0), seeded[0], 'manual mode moved the base');

  rig.mod.setMode('modulated');
  assert.equal(rig.mod.baseValue(0), 8.5, 'the flip did not adopt the slider');
  assert.equal(rig.mod.baseValue(1), -1.5);
  assert.equal(rig.mod.baseValue(3), 3);
  // …and it is an adoption, not a rebase: the counter is a human-edit diagnostic.
  assert.equal(rig.mod.rebaseCount, 0, 'adoptBase inflated the rebase counter');
  // …and the first modulated tick holds there instead of stepping.
  rig.tick();
  assert.equal(rig.read(0), 8.5, 'the flip stepped the value');
});

test('adoptBase clamps into the modulation range and leaves unmasked slots alone', () => {
  const rig = makeRig(0);
  const base = rig.mod.baseValues();
  rig.edit(0, 999); // hi is 10
  rig.edit(1, -99); // lo is -2
  rig.edit(2, 42); // excluded
  rig.edit(4, 9); // spec, but unmasked
  rig.mod.adoptBase();
  assert.equal(rig.mod.baseValue(0), (RIG_SLOTS[0] as ModSpec).hi);
  assert.equal(rig.mod.baseValue(1), (RIG_SLOTS[1] as ModSpec).lo);
  assert.equal(rig.mod.baseValue(2), base[2], 'an excluded slot was adopted');
  assert.equal(rig.mod.baseValue(4), base[4], 'an unmasked slot was adopted');

  // A non-finite live value is refused here too, not clamped into the base.
  rig.edit(3, Number.NaN);
  const b3 = rig.mod.baseValue(3);
  rig.mod.adoptBase();
  assert.equal(rig.mod.baseValue(3), b3, 'NaN was adopted');
});

test('setMode("manual") leaves the base exactly where it was', () => {
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.edit(0, 2.5);
  rig.tick();
  const base = rig.mod.baseValues();
  rig.mod.setMode('manual');
  assert.deepEqual(Array.from(rig.mod.baseValues()), Array.from(base), 'going manual disturbed the base');
  // …and flipping back is an adopt of wherever the sliders now are, which is a
  // no-op when nothing moved in between.
  rig.mod.setMode('modulated');
  assert.deepEqual(Array.from(rig.mod.baseValues()), Array.from(base));
});

test('setSeed still throws the edits away — a reroll is how you get a new creature', () => {
  const rig = makeRig(0);
  rig.mod.setMode('modulated');
  rig.edit(0, 9.75);
  rig.edit(1, 1.75);
  rig.tick();
  assert.equal(rig.mod.baseValue(0), 9.75);

  const seed = 0xbeef_1234;
  rig.mod.setSeed(seed);
  assert.equal(rig.mod.currentSeed, seed >>> 0);

  // The expectation is the seeded personality computed independently from the
  // theta the constructor captured as `defaults` — not from what the edits left.
  const want = baseVector(seed, rig.defaults, RIG_SLOTS, new Float64Array(RIG_LEN));
  for (let i = 0; i < RIG_LEN; i++) {
    assert.equal(rig.mod.baseValue(i), want[i], `slot ${i} did not come back from the seed`);
  }
  assert.notEqual(rig.mod.baseValue(0), 9.75, 'the edit survived a reroll');
  // …and the reroll is visible immediately, including through the mask.
  for (const i of RIG_MASKED) assert.equal(rig.read(i), want[i], `slot ${i} was not applied`);
  // …and it is not itself read as an edit on the next tick.
  const count = rig.mod.rebaseCount;
  rig.tick();
  assert.equal(rig.mod.rebaseCount, count, 'a reroll was read as a human edit');
});

// ── 4. accessors ─────────────────────────────────────────────────────────────

test('the diagnostic accessors report the registry and the base, and copy what they must', () => {
  const rig = makeRig(0);
  // spec(i) is a copy of the registry's spec, never the shared object itself —
  // the registry is aliased into the slew limiter and every projection, so a
  // caller mutating what it was handed must not rewrite the app's ranges.
  for (let i = 0; i < RIG_LEN; i++) {
    const s = rig.mod.spec(i);
    const expected = RIG_SLOTS[i] ?? null;
    if (expected === null) {
      assert.equal(s, null);
      continue;
    }
    assert.notEqual(s, expected, `spec(${i}) must not alias the registry`);
    assert.deepEqual(s, expected, `spec(${i}) must match the registry's values`);
  }
  assert.equal(rig.mod.spec(2), null, 'an excluded slot has no spec');
  assert.equal(rig.mod.spec(RIG_LEN), null, 'out of range is null, not undefined');
  assert.equal(rig.mod.spec(-1), null);

  // baseValue(i) agrees with baseValues(), and is 0 rather than undefined off
  // the end.
  const all = rig.mod.baseValues();
  for (let i = 0; i < RIG_LEN; i++) assert.equal(rig.mod.baseValue(i), all[i], `slot ${i}`);
  assert.equal(rig.mod.baseValue(RIG_LEN), 0);
  assert.equal(rig.mod.baseValue(-1), 0);

  // baseValues() is a copy: the panel refreshes off it every frame.
  all.fill(-1234);
  assert.notEqual(rig.mod.baseValue(0), -1234, 'baseValues() aliased the live base');

  // rebaseCount starts at zero and only ever climbs.
  assert.equal(rig.mod.rebaseCount, 0);
  rig.mod.setMode('modulated');
  rig.tick();
  let last = rig.mod.rebaseCount;
  for (let round = 0; round < 4; round++) {
    rig.edit(0, 1 + round);
    rig.tick();
    assert.ok(rig.mod.rebaseCount >= last, 'the counter went backwards');
    last = rig.mod.rebaseCount;
  }
  assert.equal(last, 4);
});
