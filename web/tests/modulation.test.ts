/**
 * Headless tests for the pure half of the modulation layer (plan.md Revision 3).
 *
 * Run with `npm test` in web/ — Node strips the types and loads the modules
 * straight out of src/, which is why mapping/*.ts imports carry explicit .ts
 * extensions. Nothing here touches the DOM or WebGPU.
 *
 * The properties that matter, in the order the runtime depends on them:
 *   the input really is standardised, the wiring really is a function of the seed
 *   alone, the output really cannot leave its range, and the depths really do
 *   isolate.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  baseVector,
  chooseSignal,
  makeRng,
  ModulationSignal,
  unitDirection,
  Z_CLAMP,
} from '../src/mapping/modulation.ts';
import { SlewLimiter } from '../src/mapping/slew.ts';
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
 */
function targetFor(
  seed: number,
  z: ArrayLike<number>,
  dims: number,
  defaults: Float64Array,
  depth: number,
  groupDepth: Record<ModGroup, number>,
): Float64Array {
  const base = baseVector(seed, defaults, SLOTS, new Float64Array(LEN));
  const out = new Float64Array(LEN);
  const w = new Float32Array(dims);
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec) {
      out[i] = base[i] as number;
      continue;
    }
    unitDirection(seed, i, dims, w);
    let raw = 0;
    for (let d = 0; d < dims; d++) raw += (w[d] as number) * ((z[d] as number) ?? 0);
    const e = Math.tanh(depth * groupDepth[spec.group] * raw);
    const b = base[i] as number;
    const v = spec.mult ? b * Math.exp(spec.half * e) : b + spec.half * e;
    out[i] = v < spec.lo ? spec.lo : v > spec.hi ? spec.hi : v;
  }
  return out;
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
  assert.equal(LEN, 18 * K + K * K + 4);
  const names = fieldNames(K);
  assert.equal(names.length, LEN);
  assert.equal(names[0], 'species0.brightness');
  assert.ok(!names.some((n) => n.includes('color')), 'no colour slot survives in θ');
  assert.equal(names[names.length - 1], 'stemGain');
  assert.equal(fieldClasses(K).length, LEN);
  assert.equal(fieldClasses(K)[0], CLASS_FAST);
});

test('the exclusions are exactly the documented ones', () => {
  const names = fieldNames(K);
  const excluded = names.filter((_, i) => SLOTS[i] === null);
  // every p3 exponent, plus exposure / gamma / stemGain
  assert.equal(excluded.length, 4 * K + 3);
  for (const n of excluded) {
    assert.ok(
      n.endsWith('.p3') || n === 'exposure' || n === 'gamma' || n === 'stemGain',
      `unexpected exclusion ${n}`,
    );
  }
  assert.equal(MASK.length, LEN);
  assert.equal(
    MASK.reduce((a, b) => a + b, 0),
    LEN - excluded.length,
  );
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
  const sig = new ModulationSignal('embedding', raw, frames, dims, 0.1, 'test');
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
  const sig = new ModulationSignal('latent', raw, frames, dims, 0.1, 'test');
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

test('chooseSignal prefers the 1024-dim sidecar and falls back to the latent channel', () => {
  const frames = 40;
  const stride = 6;
  const timeline = {
    manifest: { grid: { frames, hopSeconds: 0.1 } },
    data: new Float32Array(frames * stride).map((_, i) => Math.sin(i)),
    stride,
    channels: new Map([{ name: 'latent', dims: 4, offset: 2 }].map((c) => [c.name, c])),
    events: [],
    embedding: null,
  } as unknown as Timeline;

  const fallback = chooseSignal(timeline);
  assert.ok(fallback);
  assert.equal(fallback.kind, 'latent');
  assert.equal(fallback.dims, 4);
  assert.equal(fallback.label, 'latent-4');

  const wide = chooseSignal({
    ...timeline,
    embedding: {
      frames,
      dims: 16,
      hopSeconds: 0.1,
      data: new Float32Array(frames * 16).map((_, i) => Math.cos(i)),
      source: 'muq',
    },
  } as unknown as Timeline);
  assert.ok(wide);
  assert.equal(wide.kind, 'embedding');
  assert.equal(wide.dims, 16);

  // no sidecar and no channel at all → no modulation rather than a fake signal
  const bare = { ...timeline, channels: new Map() } as unknown as Timeline;
  assert.equal(chooseSignal(bare), null);
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
  oneOff.brightness = 0;
  const partial = targetFor(777, z, dims, defaults, 1, oneOff);
  let moved = 0;
  for (let i = 0; i < LEN; i++) {
    const spec = SLOTS[i] as ModSpec | null;
    if (!spec) continue;
    if (spec.group === 'brightness') assert.equal(partial[i], base[i], fieldNames(K)[i]);
    else if (Math.abs((partial[i] as number) - (base[i] as number)) > 1e-9) moved++;
  }
  assert.ok(moved > 20, `only ${moved} non-brightness slots moved`);
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

test('a v3 config survives serialize → parse, palette and grade and all', () => {
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
  // shared by reference with the live config, exactly like the anchor era
  assert.equal(cfg.palette, base.palette);
  assert.equal(cfg.render, base.render);

  const back = parseModulation(serializeModulation(cfg));
  assert.equal(back.version, 3);
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
  assert.equal(back.version, 3);
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
  const sig = new ModulationSignal('embedding', raw, frames, dims, 0.1, 'test');
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
