/**
 * The per-particle luminance lane, as arithmetic.
 *
 * Everything the GPU does per particle is composed on the CPU by
 * `lumaUniforms`, so the interesting behaviour — the identity at depth 0, the
 * anchoring, what the headroom budget buys, how the SDR rendition differs — is
 * testable without a device. What is NOT testable here is the *feel*: whether a
 * spike reads as a hit is a judgement about motion on a real display, and these
 * tests deliberately claim nothing about it.
 *
 * The last two tests pin the CPU copy against the WGSL copy, because the whole
 * design depends on the two agreeing and nothing else checks that.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import {
  applyLuma,
  defaultPlifeLuma,
  LUMA_RANGE,
  LUMA_WEIGHTS,
  lumaRules,
  lumaUniforms,
  particleGain,
  particleWhiteMix,
  type PlifeLumaConfig,
} from '../src/sim/plife/luma.ts';
import { defaultPlifeConfig, PLIFE_BLOCKS } from '../src/sim/plife/config.ts';
import { blockRules, persistedBlockDecls } from '../src/mapping/blocks.ts';

const src = (relative: string): string =>
  fileURLToPath(new URL(`../src/${relative}`, import.meta.url));

/** A colour with real chroma, so the desaturation tests have something to move. */
const ORANGE: readonly [number, number, number] = [1, 0.34, 0.02];

function cfg(over: Partial<PlifeLumaConfig> = {}): PlifeLumaConfig {
  return { ...defaultPlifeLuma(), ...over };
}

// ── depth 0 is the A/B baseline, exactly ─────────────────────────────────────

test('depth 0 is an exact identity on every display and at every speed', () => {
  for (const headroom of [1, 2, 4, 16]) {
    const u = lumaUniforms(cfg({ depth: 0 }), headroom);
    for (const speed of [0, 0.001, 0.25, 0.5, 0.99, 1]) {
      // Not "close to 1" — the pre-lane shader multiplied by nothing, so the
      // lane's off state has to be a true multiplicative identity or the A/B
      // against main is comparing two different looks.
      assert.equal(particleGain(u, speed), 1, `gain at speed ${speed}, headroom ${headroom}`);
      assert.equal(particleWhiteMix(u, speed), 0, `white at speed ${speed}`);
      assert.deepEqual(applyLuma(u, ORANGE, speed), [...ORANGE]);
    }
    // Including the jitter: a per-particle draw that survived depth 0 would
    // make the baseline speckled, which is a different image.
    assert.equal(particleGain(u, 1, 0), 1);
    assert.equal(particleGain(u, 1, 0.999), 1);
  }
});

test('depth 0 stays an identity however the other five knobs are set', () => {
  const hostile = cfg({
    depth: 0,
    curve: LUMA_RANGE.curve.max,
    mid: LUMA_RANGE.mid.min,
    hdrBudget: LUMA_RANGE.hdrBudget.max,
    whitePeak: 1,
    jitter: LUMA_RANGE.jitter.max,
  });
  const u = lumaUniforms(hostile, 8);
  assert.equal(particleGain(u, 1, 0), 1);
  assert.equal(particleGain(u, 0, 1), 1);
  assert.equal(particleWhiteMix(u, 1), 0);
});

test('the white push and the jitter ramp out below depth 1 rather than stepping', () => {
  // Without the ramp, depth 0.0001 would be "gain ~1 but fully bleached and
  // fully speckled", i.e. a visible discontinuity right next to the off state.
  const tiny = lumaUniforms(cfg({ depth: 0.01 }), 1);
  const one = lumaUniforms(cfg({ depth: 1 }), 1);
  assert.ok(tiny.white > 0);
  assert.ok(tiny.white < one.white / 50, 'white should scale down with depth under the ref');
  assert.ok(tiny.jitterStops < one.jitterStops / 100);
});

// ── the anchor ───────────────────────────────────────────────────────────────

test('a particle at `mid` renders at exactly today’s brightness', () => {
  for (const mid of [0.1, 0.4, 0.75, 1]) {
    for (const curve of [0.5, 1, 2.5, 6]) {
      const u = lumaUniforms(cfg({ mid, curve, jitter: 0 }), 4);
      // The whole reason the lane redistributes light instead of adding it: if
      // this drifted, auto-exposure would divide the net gain back out and the
      // knob would read as a wash rather than as a spread.
      assert.ok(
        Math.abs(particleGain(u, mid) - 1) < 1e-9,
        `gain at mid=${mid}, curve=${curve} was ${particleGain(u, mid)}`,
      );
    }
  }
});

test('gain rises monotonically with speed, and straddles 1 about the anchor', () => {
  const u = lumaUniforms(cfg({ jitter: 0 }), 4);
  let previous = -Infinity;
  for (let i = 0; i <= 40; i++) {
    const g = particleGain(u, i / 40);
    assert.ok(g > previous, `gain must increase with speed (broke at u=${i / 40})`);
    previous = g;
  }
  const mid = defaultPlifeLuma().mid;
  assert.ok(particleGain(u, mid * 0.5) < 1, 'slower than mid must be dimmer than before');
  assert.ok(particleGain(u, 1) > 1, 'the ceiling must be brighter than before');
});

test('the exponent tightens the bulk band and hands the span to the peak', () => {
  // The "most particles in a modest band, peaks spike" requirement, stated as
  // the two properties that make it true. Written after the first draft of this
  // test asserted the opposite of what the curve does: raising the exponent
  // pulls `anchor` DOWN as well as flattening the low end, so slow particles
  // move *toward* today's brightness rather than away from it. The band gets
  // tighter, not darker — and that is the better behaviour, so the arithmetic
  // stayed and the expectation moved.
  const flat = lumaUniforms(cfg({ curve: 1, mid: 0.4, jitter: 0 }), 4);
  const steep = lumaUniforms(cfg({ curve: 5, mid: 0.4, jitter: 0 }), 4);
  assert.equal(flat.stops, steep.stops, 'the span is the exponent-independent part');
  assert.ok(steep.anchor < flat.anchor, 'the anchor drops with the exponent');

  // (1) the slow half of the population is compressed
  const band = (u: ReturnType<typeof lumaUniforms>): number =>
    particleGain(u, 0.4) / particleGain(u, 0);
  assert.ok(band(steep) < band(flat) / 10, 'the bulk band must tighten');

  // (2) …and what it gives up goes to the top
  const contrast = (u: ReturnType<typeof lumaUniforms>): number =>
    particleGain(u, 1) / particleGain(u, 0.25);
  assert.ok(contrast(steep) > contrast(flat), 'the peak-to-bulk contrast must grow');
});

// ── SDR vs HDR ───────────────────────────────────────────────────────────────

test('the span is exactly `depth` on SDR and grows with the display’s stops', () => {
  const c = cfg({ depth: 3, hdrBudget: 1 });
  assert.equal(lumaUniforms(c, 1).stops, 3, 'H = 1 must degenerate to the authored depth');
  // log2(4) = 2 stops of headroom, spent in full.
  assert.equal(lumaUniforms(c, 4).stops, 9);
  assert.equal(lumaUniforms(c, 2).stops, 6);
  // hdrBudget 0 pins both renditions together, which is how one is tuned to
  // match the other.
  assert.equal(lumaUniforms(cfg({ depth: 3, hdrBudget: 0 }), 8).stops, 3);
  // A host that reports a nonsense headroom gets the SDR rendition, not a NaN.
  for (const bogus of [0, -1, Number.NaN]) {
    assert.equal(lumaUniforms(c, bogus).stops, 3, `headroom ${bogus}`);
  }
});

test('an HDR peak is many stops above an SDR peak at identical settings', () => {
  const c = cfg({ jitter: 0 });
  const sdr = lumaUniforms(c, 1);
  const hdr = lumaUniforms(c, 4);
  const ratio = particleGain(hdr, 1) / particleGain(sdr, 1);
  // The brief's "peak velocity spikes into HDR headroom": at the shipped
  // defaults the HDR ceiling sits ~5.4 stops above the SDR one, which is the
  // headroom being spent rather than a louder version of the same image.
  assert.ok(Math.log2(ratio) > 5, `HDR peak was only ${Math.log2(ratio).toFixed(2)} stops higher`);
  // And the bulk barely moves: this must not read as "the HDR canvas is
  // brighter", it must read as "the HDR canvas has further to go".
  const bulk = particleGain(hdr, c.mid * 0.6) / particleGain(sdr, c.mid * 0.6);
  assert.ok(Math.abs(Math.log2(bulk)) < 1.2, `the bulk moved ${Math.log2(bulk).toFixed(2)} stops`);
});

test('the white push is the SDR substitute for headroom and fades as headroom arrives', () => {
  const c = cfg({ whitePeak: 0.6 });
  const sdr = lumaUniforms(c, 1);
  const hdr = lumaUniforms(c, 4);
  assert.ok(Math.abs(particleWhiteMix(sdr, 1) - 0.6) < 1e-12, 'SDR gets the authored amount');
  assert.ok(particleWhiteMix(hdr, 1) < particleWhiteMix(sdr, 1) / 2);
  // At rest nothing is bleached on either — hue is the species label and only
  // the peak is allowed to trade it away.
  assert.ok(particleWhiteMix(sdr, 0) < 1e-6);
});

test('the white push raises the weak channels while holding luminance', () => {
  const u = lumaUniforms(cfg({ depth: 3, whitePeak: 1, jitter: 0 }), 1);
  const before = ORANGE;
  const luminance = (c: readonly number[]): number =>
    (c[0] as number) * LUMA_WEIGHTS[0] +
    (c[1] as number) * LUMA_WEIGHTS[1] +
    (c[2] as number) * LUMA_WEIGHTS[2];
  // Isolate the desaturation from the gain by dividing the gain back out.
  const after = applyLuma(u, before, 1).map((v) => v / particleGain(u, 1));
  assert.ok((after[2] as number) > (before[2] as number) * 5, 'the weakest channel must lift');
  assert.ok((after[0] as number) < (before[0] as number), 'the strongest channel must give way');
  assert.ok(Math.abs(luminance(after) - luminance(before)) < 1e-6, 'luminance is preserved');
});

// ── the spread ───────────────────────────────────────────────────────────────

test('a population of mixed speeds resolves into a real brightness spread', () => {
  // The plumbing claim the screenshots can only corroborate: at depth 0 every
  // member of a mixed-speed population renders at one value, and at depth > 0
  // they do not. The distribution below is a stand-in, not a measurement.
  const speeds = Array.from({ length: 64 }, (_, i) => (i / 63) ** 2);
  const spread = (u: ReturnType<typeof lumaUniforms>): number => {
    const gains = speeds.map((s, i) => particleGain(u, s, (i * 0.618) % 1));
    return Math.max(...gains) / Math.min(...gains);
  };
  assert.equal(spread(lumaUniforms(cfg({ depth: 0 }), 4)), 1);
  assert.ok(spread(lumaUniforms(cfg(), 1)) > 4, 'SDR should still separate the population');
  assert.ok(spread(lumaUniforms(cfg(), 4)) > 100, 'HDR should separate it far more');
});

test('jitter is a fraction of the span, symmetric about no change', () => {
  const u = lumaUniforms(cfg({ jitter: 0.25 }), 1);
  assert.ok(Math.abs(u.jitterStops - 0.25 * u.stops) < 1e-12);
  // 0.5 is the neutral draw, and the two extremes are equal and opposite in
  // stops — so jitter adds texture without moving the population's mean level.
  assert.equal(particleGain(u, 0.4, 0.5), particleGain(u, 0.4));
  const lo = Math.log2(particleGain(u, 0.4, 0));
  const hi = Math.log2(particleGain(u, 0.4, 1));
  const mid = Math.log2(particleGain(u, 0.4, 0.5));
  assert.ok(Math.abs((hi - mid) - (mid - lo)) < 1e-9);
});

// ── persistence wiring ───────────────────────────────────────────────────────

test('the luma block is registered for persistence, with a rule per knob', () => {
  const decl = new Map(persistedBlockDecls(PLIFE_BLOCKS)).get('luma');
  assert.ok(decl, 'PLIFE_BLOCKS must declare the luma block as persisted');
  const config = defaultPlifeConfig();
  assert.deepEqual(Object.keys(decl.defaults(config)).sort(), Object.keys(LUMA_RANGE).sort());
  // Defaults enrol a field in persistence and rules only clamp it — but an
  // unclamped knob here means a hand-edited file can hand the shader an
  // exponent of 0, so this block wants both, keyed identically.
  assert.deepEqual(
    Object.keys(blockRules(decl, config)).sort(),
    Object.keys(LUMA_RANGE).sort(),
  );
  assert.deepEqual(Object.keys(lumaRules()).sort(), Object.keys(LUMA_RANGE).sort());
});

test('every shipped default sits inside the slider that shows it', () => {
  const d = defaultPlifeLuma();
  for (const key of Object.keys(LUMA_RANGE) as (keyof PlifeLumaConfig)[]) {
    const r = LUMA_RANGE[key];
    assert.ok(d[key] >= r.min && d[key] <= r.max, `${key} = ${d[key]} is outside ${r.min}..${r.max}`);
  }
});

// ── the two copies of the arithmetic ─────────────────────────────────────────

test('the Globals struct is exactly as wide as the TS writer thinks', async () => {
  // `writeGlobals` indexes this block positionally. The struct's own comment
  // used to say "nothing checks the two agree" — this is that check, added
  // when the luminance lane grew the block from 28 words to 32, which is
  // precisely the moment a positional writer starts writing into a neighbour.
  const common = await readFile(src('sim/plife/shaders/common.wgsl'), 'utf8');
  const fields = /struct Globals \{([\s\S]*?)\n\}/.exec(common)?.[1] ?? '';
  const scalars = (fields.match(/:\s*(f32|u32)\s*,/g) ?? []).length;
  const plife = await readFile(src('sim/plife/plife.ts'), 'utf8');
  const declared = /const GLOBALS_WORDS = (\d+);/.exec(plife)?.[1];
  assert.equal(scalars, 32, 'the Globals struct is not 32 scalar words');
  assert.equal(Number(declared), scalars);
});

test('the particle shader reads the luminance uniforms this module writes', async () => {
  const render = await readFile(src('sim/plife/shaders/render.wgsl'), 'utf8');
  for (const field of ['lumaStops', 'lumaExponent', 'lumaAnchor', 'lumaWhite', 'lumaJitter']) {
    assert.ok(render.includes(`g.${field}`), `render.wgsl never reads g.${field}`);
  }
  // The exact shape of the composition, so a refactor that drops the anchor or
  // the exp2 fails here rather than on someone's display.
  assert.ok(
    /exp2\(g\.lumaStops \* \(shaped - g\.lumaAnchor\) \+ jitter\)/.test(render),
    'the gain is no longer exp2(stops · (shaped − anchor) + jitter)',
  );
  assert.ok(/pow\(u01, g\.lumaExponent\)/.test(render), 'the shaped speed is no longer u^exponent');
  // The luminance weights exist twice; this is what stops them drifting.
  const weights = /const LUMA_WEIGHTS: vec3f = vec3f\(([^)]*)\)/.exec(render)?.[1] ?? '';
  assert.deepEqual(weights.split(',').map((n) => Number(n.trim())), [...LUMA_WEIGHTS]);
});
