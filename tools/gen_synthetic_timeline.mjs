import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SEED = 0x5eed1234;
const HOP = 0.1;
const FRAMES = 2100;
const DURATION = 210.0;
const BPM = 120;
const BEAT = 60 / BPM;

const CHANNELS = [
  { name: 'stems', dims: 4, offset: 0 },
  { name: 'latent', dims: 64, offset: 4 },
  { name: 'novelty4', dims: 1, offset: 68 },
  { name: 'novelty16', dims: 1, offset: 69 },
  { name: 'actChorus', dims: 1, offset: 70 },
  { name: 'recurTime', dims: 1, offset: 71 },
  { name: 'recurStr', dims: 1, offset: 72 },
];
const TOTAL_DIMS = 73;

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
const gauss = () => {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
};

const SEGMENTS = [
  ['intro', 0, 16],
  ['verse', 16, 56],
  ['chorus', 56, 88],
  ['verse', 88, 128],
  ['chorus', 128, 160],
  ['bridge', 160, 184],
  ['chorus', 184, 206],
  ['outro', 206, 210],
].map(([label, start, end]) => ({
  start,
  end,
  label,
  confidence: Math.round((0.7 + rand() * 0.25) * 1000) / 1000,
}));

const beats = [];
for (let t = 0; t < DURATION - 1e-9; t += BEAT) beats.push(Math.round(t * 1000) / 1000);
const downbeats = beats.filter((_, i) => i % 4 === 0);

const LABELS = [...new Set(SEGMENTS.map((s) => s.label))];
const centers = new Map(
  LABELS.map((l) => [l, Array.from({ length: 64 }, () => gauss() * 0.6)]),
);

const labelAt = (t) => (SEGMENTS.find((s) => t >= s.start && t < s.end) ?? SEGMENTS.at(-1)).label;

// step smoothed over ~0.5 s: logistic with width chosen so 10-90% spans ~0.5 s
const step = (t, at) => 1 / (1 + Math.exp(-(t - at) / 0.11));
const window01 = (t, a, b) => step(t, a) * (1 - step(t, b));

function stemsAt(t) {
  const label = labelAt(t);
  const chorus = label === 'chorus';
  const verse = label === 'verse';

  let drums = step(t, 16) * (1 - step(t, 206));
  // drums drop out during the bridge, back in hard at 184
  drums *= 1 - 0.75 * window01(t, 160, 184);
  drums *= chorus ? 1.0 : 0.72;

  let bass = step(t, 24) * (1 - step(t, 206));
  bass *= 1 - 0.55 * window01(t, 160, 184);
  bass *= chorus ? 0.95 : 0.7;

  let vocals = 0;
  for (const s of SEGMENTS) {
    if (s.label !== 'verse' && s.label !== 'chorus') continue;
    vocals += window01(t, s.start, s.end) * (s.label === 'chorus' ? 0.95 : 0.55);
  }

  let other = 0.85 * window01(t, -1, 16) + 0.8 * window01(t, 160, 184);
  other += 0.3 * step(t, 16) * (1 - step(t, 206));
  other += 0.5 * window01(t, 206, 212);

  const pulse = 0.06 * Math.sin((2 * Math.PI * t) / BEAT);
  const clamp = (x) => Math.min(1, Math.max(0, x));
  return [clamp(bass + pulse * 0.5), clamp(drums + pulse), clamp(vocals), clamp(other)];
}

const bump = (t, c, w) => Math.exp(-((t - c) ** 2) / (2 * (w / 2.355) ** 2));
const boundaries = SEGMENTS.map((s) => s.start).filter((t) => t > 0);

const data = new Float32Array(FRAMES * TOTAL_DIMS);
const latent = new Array(64).fill(0);
const EMA = 0.06;

for (let f = 0; f < FRAMES; f++) {
  const t = f * HOP;
  const row = f * TOTAL_DIMS;

  const st = stemsAt(t);
  for (let i = 0; i < 4; i++) data[row + i] = st[i];

  const c = centers.get(labelAt(t));
  for (let i = 0; i < 64; i++) {
    latent[i] += EMA * (c[i] - latent[i]) + 0.004 * gauss();
    data[row + 4 + i] = latent[i];
  }

  let n4 = 0;
  let n16 = 0;
  for (const b of boundaries) {
    n4 = Math.max(n4, bump(t, b, 4));
    n16 = Math.max(n16, bump(t, b, 16));
  }
  data[row + 68] = n4;
  data[row + 69] = n16;

  let act = 0;
  for (const s of SEGMENTS) {
    if (s.label === 'chorus') act = Math.max(act, window01(t, s.start, s.end));
  }
  data[row + 70] = act;
  data[row + 71] = 0;
  data[row + 72] = 0;
}

const n = SEGMENTS.length;
const segmentSimilarity = Array.from({ length: n }, () => new Array(n).fill(0));
for (let i = 0; i < n; i++) {
  segmentSimilarity[i][i] = 1;
  for (let j = i + 1; j < n; j++) {
    const same = SEGMENTS[i].label === SEGMENTS[j].label;
    const v = same ? 0.86 + rand() * 0.08 : 0.1 + rand() * 0.2;
    const r = Math.round(v * 1000) / 1000;
    segmentSimilarity[i][j] = r;
    segmentSimilarity[j][i] = r;
  }
}

// ── transient events (plan.md Revision 2, item 2) ────────────────────────────
// Derived from the same deterministic structure as everything else: the beat grid
// decides *where*, the stem curves decide *whether* and *how hard*. Its own RNG
// stream, seeded off SEED, so adding this left every pre-existing channel and the
// segment table bit-identical.
const erand = mulberry32(SEED ^ 0x7a11e5);
const jitter = (spread) => (erand() - 0.5) * spread;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const events = [];
const pushEvent = (t, kind, strength) => {
  if (t < 0 || t >= DURATION) return;
  const s = clamp01(strength);
  if (s <= 0.02) return;
  events.push({
    t: Math.round(t * 1000) / 1000,
    kind,
    strength: Math.round(s * 1000) / 1000,
  });
};

for (let i = 0; i < beats.length; i++) {
  const t = beats[i];
  const [bassA, drumsA, vocalsA] = stemsAt(t);
  const chorus = labelAt(t) === 'chorus';
  const inBar = i % 4;

  // kick on 1 and 3, only while the kit is actually in (so the bridge drops out)
  if (drumsA > 0.12 && (inBar === 0 || inBar === 2)) {
    pushEvent(t, 'kick', 0.55 + 0.45 * drumsA + (inBar === 0 ? 0.1 : 0) + jitter(0.08));
  }
  // syncopated extra kick on the "and of 3" in choruses
  if (chorus && drumsA > 0.4 && inBar === 2) {
    pushEvent(t + BEAT / 2, 'kick', 0.45 + 0.3 * drumsA + jitter(0.08));
  }
  // snare on the backbeat
  if (drumsA > 0.12 && (inBar === 1 || inBar === 3)) {
    pushEvent(t, 'snare', 0.6 + 0.4 * drumsA + jitter(0.08));
  }
  // hats: 8ths through choruses, quarters elsewhere
  if (drumsA > 0.25) {
    const div = chorus ? 2 : 1;
    for (let h = 0; h < div; h++) {
      pushEvent(t + (h * BEAT) / div, 'hat', (h === 0 ? 0.5 : 0.32) + 0.3 * drumsA + jitter(0.1));
    }
  }
  // bass note onsets: root on the downbeat, a weaker one on 3, a chorus pickup
  if (bassA > 0.12) {
    if (inBar === 0) pushEvent(t, 'bass', 0.6 + 0.4 * bassA + jitter(0.06));
    if (inBar === 2) pushEvent(t, 'bass', 0.4 + 0.35 * bassA + jitter(0.06));
    if (chorus && inBar === 3) pushEvent(t + BEAT / 2, 'bass', 0.35 + 0.3 * bassA + jitter(0.06));
  }
  // vocals: sparse phrase starts, every two bars while the vocal is present
  if (vocalsA > 0.25 && i % 8 === 0) {
    pushEvent(t + jitter(0.04), 'vocal', 0.5 + 0.5 * vocalsA + jitter(0.1));
  }
}
events.sort((a, b) => a.t - b.t);

const manifest = {
  version: 2,
  track: { id: 'synthetic', duration: DURATION, sampleRate: 48000 },
  grid: { hopSeconds: HOP, frames: FRAMES },
  beats,
  downbeats,
  tempo: BPM,
  segments: SEGMENTS,
  segmentSimilarity,
  channels: CHANNELS,
  events,
};

const outDir = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'data',
  'timelines',
  'synthetic',
);
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'timeline.json'), JSON.stringify(manifest, null, 2) + '\n');
// Node is little-endian on all supported platforms; Float32Array buffer is written raw.
writeFileSync(join(outDir, 'timeline.bin'), Buffer.from(data.buffer));

const expectedBytes = FRAMES * TOTAL_DIMS * 4;
const actualBytes = Buffer.from(data.buffer).byteLength;
const offsetsOk = CHANNELS.reduce((acc, c) => (acc === c.offset ? acc + c.dims : NaN), 0) === TOTAL_DIMS;
const jsonOk = JSON.parse(JSON.stringify(manifest)).grid.frames === FRAMES;
let finite = true;
for (const v of data) if (!Number.isFinite(v)) finite = false;

const chanRange = (off, dims) => {
  let lo = Infinity;
  let hi = -Infinity;
  for (let f = 0; f < FRAMES; f++)
    for (let d = 0; d < dims; d++) {
      const v = data[f * TOTAL_DIMS + off + d];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  return `${lo.toFixed(3)}..${hi.toFixed(3)}`;
};

console.log(`out: ${outDir}`);
console.log(`frames=${FRAMES} totalDims=${TOTAL_DIMS} bytes=${actualBytes} (expected ${expectedBytes}) ${actualBytes === expectedBytes ? 'OK' : 'MISMATCH'}`);
console.log(`offsets contiguous: ${offsetsOk} | json round-trip: ${jsonOk} | all finite: ${finite}`);
console.log(`beats=${beats.length} downbeats=${downbeats.length} segments=${SEGMENTS.length}`);
const eventsSorted = events.every((e, i) => i === 0 || events[i - 1].t <= e.t);
const eventsInRange = events.every((e) => e.strength >= 0 && e.strength <= 1 && e.t >= 0 && e.t < DURATION);
const byKind = events.reduce((acc, e) => ((acc[e.kind] = (acc[e.kind] ?? 0) + 1), acc), {});
console.log(`events=${events.length} sorted: ${eventsSorted} | strength/t in range: ${eventsInRange}`);
console.log(`  ${Object.entries(byKind).map(([k, v]) => `${k}=${v}`).join(' ')}`);
for (const c of CHANNELS) console.log(`  ${c.name.padEnd(10)} [${c.offset}..${c.offset + c.dims - 1}] range ${chanRange(c.offset, c.dims)}`);
