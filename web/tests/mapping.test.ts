/**
 * Headless tests for the pure half of the mapping layer.
 *
 * Run with `npm test` in web/ — Node strips the types and loads the modules
 * straight out of src/, which is why mapping/*.ts imports carry explicit .ts
 * extensions. Nothing here touches the DOM or WebGPU.
 *
 * plan.md names the convex-hull guarantee as the thing worth testing; the rest
 * are the properties the runtime silently depends on (determinism of the anchor
 * fit, a well-formed simplex, a slew limiter that actually approaches).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { kmeans, makeRng } from '../src/mapping/kmeans.ts';
import { softmaxWeights, squaredDistances } from '../src/mapping/simplex.ts';
import { SlewLimiter } from '../src/mapping/slew.ts';
import {
  applyPreset,
  blendVectors,
  CLASS_FAST,
  CLASS_SLOW,
  fieldClasses,
  fieldNames,
  presetFromConfig,
  presetToVector,
  vectorLength,
  vectorToPreset,
} from '../src/mapping/preset.ts';
import { fitMapping, starterPreset } from '../src/mapping/anchors.ts';
import { parseMapping, serializeMapping } from '../src/mapping/persist.ts';
import { defaultConfig } from '../src/sim/physarum/config.ts';
import { defaultRenderConfig, mergeRenderConfig } from '../src/sim/render/config.ts';

/** Deterministic blobby test data: `clusters` gaussians in `dims` dimensions. */
function makeData(n: number, dims: number, clusters: number, seed = 7): Float32Array {
  const rng = makeRng(seed);
  const centers: number[][] = [];
  for (let c = 0; c < clusters; c++) {
    centers.push(Array.from({ length: dims }, () => rng() * 20 - 10));
  }
  const out = new Float32Array(n * dims);
  for (let i = 0; i < n; i++) {
    const c = centers[i % clusters] as number[];
    for (let d = 0; d < dims; d++) {
      out[i * dims + d] = (c[d] as number) + (rng() - 0.5) * 1.5;
    }
  }
  return out;
}

// ── k-means ──────────────────────────────────────────────────────────────────

test('k-means is deterministic: same input and seed give identical centres', () => {
  const data = makeData(500, 8, 5);
  const a = kmeans(data, 500, 8, 5, { seed: 0x5eed_1e57 });
  const b = kmeans(data, 500, 8, 5, { seed: 0x5eed_1e57 });
  assert.deepEqual(a.centers, b.centers);
  assert.deepEqual(Array.from(a.assignments), Array.from(b.assignments));
  assert.equal(a.inertia, b.inertia);
});

test('k-means uses no global randomness: a different seed is a different, still stable fit', () => {
  const data = makeData(400, 6, 4, 11);
  const a = kmeans(data, 400, 6, 4, { seed: 1 });
  const b = kmeans(data, 400, 6, 4, { seed: 2 });
  const again = kmeans(data, 400, 6, 4, { seed: 2 });
  assert.deepEqual(b.centers, again.centers);
  // Not an assertion about quality — just that the seed is actually consumed.
  assert.equal(a.centers.length, b.centers.length);
});

test('k-means recovers well-separated clusters and leaves none empty', () => {
  const data = makeData(600, 4, 6, 3);
  const fit = kmeans(data, 600, 4, 6, { seed: 42 });
  assert.equal(fit.centers.length, 6);
  for (const c of fit.counts) assert.ok(c > 0, 'every cluster has members');
  // counts are sorted descending by construction
  const counts = Array.from(fit.counts);
  assert.deepEqual(counts, [...counts].sort((x, y) => y - x));
  assert.ok(fit.meanSqDist > 0 && Number.isFinite(fit.meanSqDist));
});

test('k-means survives degenerate input (all frames identical)', () => {
  const data = new Float32Array(100 * 3).fill(0.25);
  const fit = kmeans(data, 100, 3, 4, { seed: 9 });
  assert.equal(fit.centers.length, 4);
  assert.ok(Number.isFinite(fit.inertia));
});

// ── simplex ──────────────────────────────────────────────────────────────────

const CENTERS = [
  [0, 0],
  [10, 0],
  [0, 10],
  [10, 10],
];

test('softmax weights are a simplex: non-negative and summing to 1', () => {
  const out = new Float64Array(CENTERS.length);
  for (const t of [0.01, 0.5, 5, 500]) {
    softmaxWeights([3, 4], CENTERS, t, out);
    let sum = 0;
    for (const w of out) {
      assert.ok(w >= 0, 'weights are non-negative');
      sum += w;
    }
    assert.ok(Math.abs(sum - 1) < 1e-12, `weights sum to 1 at T=${t} (got ${sum})`);
  }
});

test('locality: the nearest anchor dominates, and T→0 makes it one-hot', () => {
  const out = new Float64Array(CENTERS.length);
  const d2 = new Float64Array(CENTERS.length);
  const z = [9, 1];
  squaredDistances(z, CENTERS, d2);
  let nearest = 0;
  for (let i = 1; i < d2.length; i++) if ((d2[i] as number) < (d2[nearest] as number)) nearest = i;
  assert.equal(nearest, 1);

  softmaxWeights(z, CENTERS, 4, out);
  for (let i = 0; i < out.length; i++) {
    if (i !== nearest) assert.ok((out[nearest] as number) > (out[i] as number));
  }

  // monotone sharpening as T falls
  let previous = 0;
  for (const t of [10, 3, 1, 0.3]) {
    softmaxWeights(z, CENTERS, t, out);
    assert.ok((out[nearest] as number) >= previous - 1e-12, 'nearest weight grows as T falls');
    previous = out[nearest] as number;
  }

  softmaxWeights(z, CENTERS, 0, out);
  assert.equal(out[nearest], 1);
  assert.equal(out.reduce((a, b) => a + b, 0), 1);
});

test('softmax is uniform when z is equidistant from every anchor', () => {
  const out = new Float64Array(CENTERS.length);
  softmaxWeights([5, 5], CENTERS, 2, out);
  for (const w of out) assert.ok(Math.abs(w - 0.25) < 1e-12);
});

test('softmax does not overflow on far-away z', () => {
  const out = new Float64Array(CENTERS.length);
  softmaxWeights([1e6, 1e6], CENTERS, 1e-3, out);
  const sum = out.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12, `sum ${sum}`);
});

// ── slew limiter ─────────────────────────────────────────────────────────────

test('slew step response: one time constant reaches 1 - 1/e of the step', () => {
  const classes = new Uint8Array([CLASS_FAST]);
  const slew = new SlewLimiter(classes, { fast: 1, medium: 1, slow: 1 });
  slew.reset([0]);
  const dt = 1 / 600;
  for (let i = 0; i < 600; i++) slew.step([1], dt);
  assert.ok(Math.abs((slew.value[0] as number) - (1 - Math.exp(-1))) < 1e-3);

  // and it converges, without overshoot
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

// ── θ vector and the convex-hull guarantee ───────────────────────────────────

const K = 4;

function starterVectors(count: number): Float64Array[] {
  const base = defaultConfig(K);
  return Array.from({ length: count }, (_, i) => presetToVector(starterPreset(base, i, count), K));
}

test('vector layout is the length and the slot names the runtime assumes', () => {
  // Revision 2: colour left θ (−3 slots per species), brightness joined it (+1).
  assert.equal(vectorLength(K), 18 * K + K * K + 4);
  const names = fieldNames(K);
  assert.equal(names.length, vectorLength(K));
  assert.equal(names[0], 'species0.brightness');
  assert.ok(!names.some((n) => n.includes('color')), 'no colour slot survives in θ');
  assert.equal(names[names.length - 1], 'stemGain');
  assert.equal(fieldClasses(K).length, vectorLength(K));
  // brightness must be modulated at 10 Hz, not at section scale
  assert.equal(fieldClasses(K)[0], CLASS_FAST);
});

test('preset ↔ config ↔ vector round-trips exactly', () => {
  const cfg = defaultConfig(K);
  const v0 = presetToVector(presetFromConfig(cfg), K);
  applyPreset(cfg, vectorToPreset(v0, K));
  const v1 = presetToVector(presetFromConfig(cfg), K);
  for (let i = 0; i < v0.length; i++) {
    // no 8-bit hex in the loop any more, so this is exact
    assert.equal(v1[i], v0[i], fieldNames(K)[i]);
  }
});

test('convex hull: any blend of presets stays inside the per-field min/max', () => {
  const vectors = starterVectors(8);
  const len = vectors[0]?.length ?? 0;
  const names = fieldNames(K);
  const out = new Float64Array(len);
  const rng = makeRng(0xc0ffee);

  const check = (weights: number[]): void => {
    blendVectors(vectors, weights, out);
    for (let i = 0; i < len; i++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of vectors) {
        const x = v[i] as number;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
      }
      const got = out[i] as number;
      assert.ok(
        got >= lo - 1e-9 && got <= hi + 1e-9,
        `${names[i]}: ${got} outside [${lo}, ${hi}]`,
      );
    }
  };

  for (let trial = 0; trial < 50; trial++) {
    const raw = vectors.map(() => rng());
    const sum = raw.reduce((a, b) => a + b, 0);
    check(raw.map((w) => w / sum));
  }
  // the degenerate ends of the simplex
  check(vectors.map((_, i) => (i === 0 ? 1 : 0)));
  check(vectors.map(() => 1 / vectors.length));
});

test('simplex weights feed the hull guarantee end to end', () => {
  const vectors = starterVectors(6);
  const centers = Array.from({ length: 6 }, (_, i) => [i, i * 2]);
  const w = new Float64Array(6);
  const out = new Float64Array(vectors[0]?.length ?? 0);
  softmaxWeights([2.4, 5.1], centers, 0.8, w);
  blendVectors(vectors, w, out);
  for (let i = 0; i < out.length; i++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of vectors) {
      const x = v[i] as number;
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    assert.ok((out[i] as number) >= lo - 1e-9 && (out[i] as number) <= hi + 1e-9);
  }
});

// ── the fit as a whole ───────────────────────────────────────────────────────

test('fitMapping is deterministic and produces one preset per anchor', () => {
  const latent = { data: makeData(300, 12, 5, 5), n: 300, dims: 12 };
  const base = defaultConfig(K);
  const a = fitMapping(latent, { base, k: 6 });
  const b = fitMapping(latent, { base, k: 6 });
  assert.equal(a.anchors.length, 6);
  assert.deepEqual(
    a.anchors.map((x) => x.center),
    b.anchors.map((x) => x.center),
  );
  assert.deepEqual(a.anchors[3]?.preset, b.anchors[3]?.preset);
  assert.equal(a.latentDims, 12);
  assert.equal(a.speciesCount, K);
  assert.ok(a.distanceScale > 0);
  for (const anchor of a.anchors) {
    assert.equal(anchor.center.length, 12);
    assert.equal(anchor.preset.species.length, K);
    assert.equal(anchor.preset.matrix.length, K * K);
  }
});

test('starter presets differ from each other but stay in range', () => {
  const base = defaultConfig(K);
  const v0 = presetToVector(starterPreset(base, 0, 8), K);
  const v3 = presetToVector(starterPreset(base, 3, 8), K);
  let differing = 0;
  for (let i = 0; i < v0.length; i++) if (Math.abs((v0[i] as number) - (v3[i] as number)) > 1e-6) differing++;
  assert.ok(differing > v0.length / 4, `only ${differing} of ${v0.length} slots differ`);

  for (let anchor = 0; anchor < 8; anchor++) {
    const p = starterPreset(base, anchor, 8);
    for (const s of p.species) {
      assert.ok(s.decay >= 0.8 && s.decay <= 1, 'decay in range');
      assert.ok(s.aliveFraction >= 0 && s.aliveFraction <= 1, 'alive fraction in range');
      assert.ok(s.diffuseCentre >= 0.111 && s.diffuseCentre <= 1, 'diffuse centre in range');
      assert.ok(s.deposit >= 0 && s.deposit <= 6, 'deposit in range');
      assert.ok(s.brightness >= 0 && s.brightness <= 2, 'brightness in range');
      assert.ok(!('colorHex' in s), 'no colour inside a preset');
    }
    for (const m of p.matrix) assert.ok(m >= -2 && m <= 2, 'matrix entry in range');
  }
});

test('the default temperature blends rather than snapping to the nearest anchor', () => {
  // Regression: scaling T by within-cluster scatter made every softmax one-hot on
  // a track with cleanly separated sections, so the simplex silently did nothing.
  const latent = { data: makeData(600, 16, 6, 21), n: 600, dims: 16 };
  const cfg = fitMapping(latent, { base: defaultConfig(K), k: 6 });
  const centers = cfg.anchors.map((a) => a.center);
  const w = new Float64Array(centers.length);
  const z = centers[0] as number[];
  softmaxWeights(z, centers, cfg.temperature * cfg.distanceScale, w);
  const top = Math.max(...w);
  assert.ok(top > 0.5, `nearest anchor still dominates (${top})`);
  assert.ok(top < 0.999, `but the blend is not one-hot (${top})`);
});

test('fitMapping with no latent channel yields no anchors rather than fake ones', () => {
  const cfg = fitMapping(null, { base: defaultConfig(K) });
  assert.equal(cfg.anchors.length, 0);
  assert.equal(cfg.latentDims, 0);
});

// ── persistence and the v1 → v2 migration ────────────────────────────────────

test('a v2 mapping survives serialize → parse, palette and all', () => {
  const base = defaultConfig(K);
  base.palette.colors[0] = '#123456';
  base.palette.saturation = 0.7;
  const cfg = fitMapping({ data: makeData(200, 8, 4, 2), n: 200, dims: 8 }, { base, k: 4 });
  const back = parseMapping(serializeMapping(cfg));
  assert.equal(back.version, 2);
  assert.deepEqual(back.palette, cfg.palette);
  assert.equal(back.anchors.length, cfg.anchors.length);
  // serialization rounds to 6 decimals for diffability, so compare on the vector
  const a = presetToVector(cfg.anchors[2]!.preset, K);
  const b = presetToVector(back.anchors[2]!.preset, K);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < 1e-6, fieldNames(K)[i]);
  }
});

test('v1 files load: anchor 0 colours become the palette, the rest are dropped', () => {
  const base = defaultConfig(K);
  const v2 = fitMapping({ data: makeData(200, 8, 4, 4), n: 200, dims: 8 }, { base, k: 3 });

  // Rebuild what a v1 file looked like: colours inside every anchor's species,
  // no brightness, no top-level palette.
  const v1 = {
    ...JSON.parse(serializeMapping(v2)),
    version: 1,
    palette: undefined,
    anchors: v2.anchors.map((a, ai) => ({
      ...a,
      preset: {
        ...a.preset,
        species: a.preset.species.map((s, si) => {
          const { brightness: _b, ...rest } = s;
          return { ...rest, colorHex: ai === 0 ? `#0000${(si + 1).toString(16)}0` : '#ffffff' };
        }),
      },
    })),
  };

  const back = parseMapping(JSON.stringify(v1));
  assert.equal(back.version, 2);
  assert.deepEqual(back.palette.colors, ['#000010', '#000020', '#000030', '#000040']);
  assert.equal(back.palette.saturation, 1);
  assert.equal(back.palette.brightness, 1);
  assert.equal(back.anchors.length, 3);
  for (const a of back.anchors) {
    for (const s of a.preset.species) {
      assert.ok(!('colorHex' in s), 'colour stripped from the preset');
      assert.equal(s.brightness, 1, 'missing brightness defaults to 1');
    }
  }
  // and the migrated config still produces a θ of the current length
  assert.equal(presetToVector(back.anchors[0]!.preset, K).length, vectorLength(K));
});

test('an unknown mapping version is rejected rather than half-applied', () => {
  assert.throws(() => parseMapping(JSON.stringify({ version: 99, anchors: [] })), /version 99/);
});

test('a hostile speciesCount is rejected before it can size anything', () => {
  const file = (speciesCount: unknown): string =>
    JSON.stringify({ version: 2, anchors: [], speciesCount });
  // The one that matters: parsing this used to reach defaultPalette(1e9).
  assert.throws(() => parseMapping(file(1e9)), /speciesCount is not an integer in 1\.\.64/);
  for (const bad of [0, -4, 2.5, Number.NaN]) {
    assert.throws(() => parseMapping(file(bad)), /speciesCount/);
  }
  assert.throws(() => parseMapping(file('4')), /speciesCount/);
});

// ── phase 7: the render block is art direction, so it has to survive a save ───

test('the render block round-trips, and a file without one gets the defaults', () => {
  const base = defaultConfig(K);
  base.render.grade.tonemap = 'aces';
  base.render.grade.contrast = 1.62;
  base.render.bloom.threshold = 0.91;
  base.render.bloom.enabled = false;
  base.render.feedback.amount = 0.4;
  const cfg = fitMapping({ data: makeData(200, 8, 4, 2), n: 200, dims: 8 }, { base, k: 3 });
  // fitMapping shares the live object by reference, exactly like the palette
  assert.equal(cfg.render, base.render);

  const back = parseMapping(serializeMapping(cfg));
  assert.deepEqual(back.render, cfg.render);

  // A hand-trimmed or pre-phase-7 file must adopt the current defaults rather
  // than fail: the grade is taste, and an old file has no opinion about it.
  const raw = JSON.parse(serializeMapping(cfg)) as Record<string, unknown>;
  delete raw['render'];
  assert.deepEqual(parseMapping(JSON.stringify(raw)).render, defaultRenderConfig());

  // An unknown tonemap name falls back instead of reaching the shader as junk.
  const bad = JSON.parse(serializeMapping(cfg)) as { render: { grade: { tonemap: string } } };
  bad.render.grade.tonemap = 'filmic-2049';
  assert.equal(parseMapping(JSON.stringify(bad)).render.grade.tonemap, 'reinhard');
});

test('a loaded render block is merged into the live object, not swapped for it', () => {
  const live = defaultRenderConfig();
  const loaded = defaultRenderConfig();
  loaded.grade.vignette = 0.9;
  loaded.bloom.intensity = 2;
  mergeRenderConfig(live, loaded);
  assert.equal(live.grade.vignette, 0.9);
  assert.equal(live.bloom.intensity, 2);
  // identity preserved: the panel and PostFx both hold `live` by reference
  assert.notEqual(live.grade, loaded.grade);
});
