/**
 * Headless tests for the impulse lane (plan.md Revision 2, item 2).
 *
 * Run with `npm test` in web/. Nothing here touches the DOM or WebGPU: the three
 * properties worth pinning are all pure — where an event lands on the tick grid
 * (including after a seek), what the envelope is worth N ticks later, and that
 * hotspot placement really is a function of (seed, eventIndex) and nothing else.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { EventCursor } from '../src/timeline/sampler.ts';
import { normalizeEvents } from '../src/timeline/loader.ts';
import { EVENT_KINDS, type EventKind, type TimelineEvent } from '../src/timeline/types.ts';
import {
  defaultImpulseConfig,
  hotspotAt,
  ImpulseEngine,
  MAX_SPLASHES,
} from '../src/sim/impulses.ts';
import { SECONDS_PER_TICK } from '../src/timing.ts';

const SPT = SECONDS_PER_TICK;

function ev(t: number, kind: EventKind, strength = 1): TimelineEvent {
  return { t, kind, strength };
}

/** Every index a cursor emits between ticks `from` and `to`, inclusive. */
function fired(cursor: EventCursor, from: number, to: number): number[] {
  const out: number[] = [];
  for (let tick = from; tick <= to; tick++) {
    const { start, end } = cursor.advance(tick);
    for (let i = start; i < end; i++) out.push(i);
  }
  return out;
}

// ── event → tick mapping ─────────────────────────────────────────────────────

test('an event maps to the nearest tick and fires on exactly that tick', () => {
  // 1.0s -> tick 120; 1.004s rounds to 120 too; 1.02s -> tick 122.
  const events = [ev(1.0, 'kick'), ev(1.004, 'snare'), ev(1.02, 'hat')];
  const c = new EventCursor(events, SPT);
  assert.equal(c.tickOf(0), 120);
  assert.equal(c.tickOf(1), 120);
  assert.equal(c.tickOf(2), 122);

  assert.deepEqual(fired(c, 118, 119), []);
  assert.deepEqual(fired(c, 120, 120), [0, 1]);
  assert.deepEqual(fired(c, 121, 122), [2]);
});

test('re-advancing to the same tick fires nothing (a paused transport does not retrigger)', () => {
  const c = new EventCursor([ev(0.5, 'kick')], SPT);
  assert.deepEqual(fired(c, 0, 60), [0]);
  for (let i = 0; i < 10; i++) assert.deepEqual(fired(c, 60, 60), []);
});

test('a whole run fires every event exactly once, in order', () => {
  const events = Array.from({ length: 200 }, (_, i) =>
    ev(i * 0.137, EVENT_KINDS[i % EVENT_KINDS.length] as EventKind),
  );
  const c = new EventCursor(events, SPT);
  const seen = fired(c, 0, Math.ceil((events[199] as TimelineEvent).t / SPT) + 5);
  assert.deepEqual(
    seen,
    events.map((_, i) => i),
  );
});

// ── seeks ────────────────────────────────────────────────────────────────────

test('seeking backwards relocates the pointer and replays from there', () => {
  const events = [ev(1, 'kick'), ev(2, 'snare'), ev(3, 'hat'), ev(4, 'bass')];
  const c = new EventCursor(events, SPT);
  assert.deepEqual(fired(c, 0, 500), [0, 1, 2, 3]);
  // jump back to tick 300 (2.5s): events 0 and 1 are behind the new playhead and
  // stay behind it; 2 (3s) and 3 (4s) play again as the walk reaches them.
  assert.deepEqual(fired(c, 300, 500), [2, 3]);
});

test('seeking forward past events skips them instead of dumping a backlog', () => {
  const events = [ev(1, 'kick'), ev(2, 'snare'), ev(3, 'hat'), ev(50, 'bass')];
  const c = new EventCursor(events, SPT);
  // land at tick 480 (4s) directly: the first three are behind us and are dropped
  assert.deepEqual(fired(c, 480, 480), []);
  assert.deepEqual(fired(c, 481, 6020), [3]);
});

test('the app default forward seek (+2s = exactly 240 ticks) skips rather than dumping', () => {
  // Eight hats a quarter-second apart from 1.0s, plus one event past the landing
  // point. ArrowRight seeks +2s, which at 1/120s is exactly 240 ticks — the walk
  // would fire all seven skipped hats on the landing tick.
  const events = [...Array.from({ length: 8 }, (_, i) => ev(1 + i * 0.25, 'hat')), ev(4, 'bass')];
  const c = new EventCursor(events, SPT);
  assert.deepEqual(fired(c, 0, 120), [0]); // 1.0s = tick 120
  assert.deepEqual(fired(c, 360, 360), []); // 3.0s, reached by one 240-tick jump
  assert.deepEqual(fired(c, 361, 520), [8]); // and the cursor is left at the playhead
});

test('a seek that lands exactly on an event still fires it', () => {
  const c = new EventCursor([ev(1, 'kick'), ev(9, 'snare')], SPT);
  fired(c, 0, 200);
  assert.deepEqual(fired(c, 1080, 1080), [1]); // 9s = tick 1080, reached by a jump
});

test('relocation and the plain walk agree', () => {
  const events = Array.from({ length: 120 }, (_, i) => ev(i * 0.25, 'hat'));
  const walked = new EventCursor(events, SPT);
  const jumped = new EventCursor(events, SPT);
  // 1200 ticks of walking, vs. one jump straight to tick 1200 then walking on
  const a = fired(walked, 0, 1800).filter((i) => (events[i] as TimelineEvent).t >= 10);
  fired(jumped, 1200, 1200);
  const b = fired(jumped, 1201, 1800);
  // tick 1200 = 10.0s = event 40 exactly, which the jump fires and the filter keeps
  assert.equal(a[0], 40);
  assert.deepEqual(a.slice(1), b);
});

// ── envelope math ────────────────────────────────────────────────────────────

/** Advance an engine `ticks` sim ticks with no events in the stream. */
function run(engine: ImpulseEngine, ticks: number, startTick = 1): number {
  for (let i = 0; i < ticks; i++) engine.update(startTick + i, SPT);
  return startTick + ticks;
}

test('envelope: instant attack, exponential decay with the configured tau', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  const tau = engine.config.responses.kick.decayMs / 1000;
  engine.testFire('kick', 1);
  assert.equal(engine.levelOf('kick'), 1);

  // exp(-elapsed/tau) exactly; `elapsed` is a whole number of 120 Hz ticks, which
  // is why this is not compared against exp(-1) at tighter than ~1%.
  const ticks = Math.round(tau / SPT);
  run(engine, ticks);
  assert.ok(
    Math.abs(engine.levelOf('kick') - Math.exp((-ticks * SPT) / tau)) < 1e-6,
    String(engine.levelOf('kick')),
  );
  assert.ok(Math.abs(engine.levelOf('kick') - Math.exp(-1)) < 0.02);

  run(engine, ticks * 2);
  assert.ok(Math.abs(engine.levelOf('kick') - Math.exp((-3 * ticks * SPT) / tau)) < 1e-6);
  assert.ok(engine.levelOf('kick') < 0.06 && engine.levelOf('kick') > 0.04);
});

test('envelope: decay is per sim tick, so dt and not wall clock sets the shape', () => {
  const a = new ImpulseEngine(1, 4, [], SPT);
  const b = new ImpulseEngine(1, 4, [], SPT);
  a.testFire('snare', 1);
  b.testFire('snare', 1);
  run(a, 12);
  // same elapsed sim time in half as many double-length ticks
  for (let i = 0; i < 6; i++) b.update(1 + i, SPT * 2);
  assert.ok(Math.abs(a.levelOf('snare') - b.levelOf('snare')) < 1e-6);
});

test('envelope: hats decay fast, vocals slowly', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  engine.testFire('hat', 1);
  engine.testFire('vocal', 1);
  run(engine, 24); // 200 ms
  assert.ok(engine.levelOf('hat') < 0.15, `hat ${engine.levelOf('hat')}`);
  assert.ok(engine.levelOf('vocal') > 0.5, `vocal ${engine.levelOf('vocal')}`);
});

test('envelope: a retrigger takes the max, so a roll sustains and never ramps away', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  for (let i = 0; i < 40; i++) {
    engine.testFire('hat', 0.6);
    run(engine, 8, 1 + i * 8);
  }
  assert.ok(engine.levelOf('hat') <= 0.6 + 1e-6, String(engine.levelOf('hat')));
  assert.ok(engine.levelOf('hat') > 0.2);
});

test('envelopes and splashes reach exactly zero, so an idle engine is a no-op', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  engine.testFire('snare', 1);
  assert.ok(engine.activeSplashes > 0);
  run(engine, 1200); // 10 s
  assert.equal(engine.levelOf('snare'), 0);
  assert.equal(engine.activeSplashes, 0);
  for (const m of [engine.state.depositMul, engine.state.brightMul, engine.state.sensorMul]) {
    assert.deepEqual(Array.from(m), [1, 1, 1, 1]);
  }
});

// ── responses ────────────────────────────────────────────────────────────────

test('responses land on the configured species and nowhere else', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  engine.testFire('kick', 1);
  const d = engine.state.depositMul;
  const cfg = engine.config.responses.kick;
  assert.ok(Math.abs((d[0] as number) - (1 + cfg.deposit)) < 1e-6);
  assert.deepEqual(Array.from(d.subarray(1)), [1, 1, 1]);
  assert.ok(Math.abs((engine.state.brightMul[0] as number) - (1 + cfg.flash)) < 1e-6);
});

test('the bass response is a sensor pop; the kick response is not', () => {
  const engine = new ImpulseEngine(1, 4, [], SPT);
  engine.testFire('bass', 1);
  assert.ok((engine.state.sensorMul[0] as number) > 2);
  engine.reset();
  engine.testFire('kick', 1);
  assert.equal(engine.state.sensorMul[0], 1);
});

test('a response authored for species 2 is clamped when K is smaller', () => {
  const engine = new ImpulseEngine(1, 2, [], SPT);
  engine.testFire('vocal', 1);
  assert.equal(engine.state.depositMul.length, 2);
  assert.ok((engine.state.depositMul[1] as number) > 1);
});

test('species -1 hits every species', () => {
  const cfg = defaultImpulseConfig();
  cfg.responses.kick.species = -1;
  const engine = new ImpulseEngine(1, 4, [], SPT, cfg);
  engine.testFire('kick', 1);
  for (const v of engine.state.depositMul) assert.ok(v > 1);
});

test('a disabled kind, and a disabled engine, do nothing', () => {
  const cfg = defaultImpulseConfig();
  cfg.responses.kick.enabled = false;
  const engine = new ImpulseEngine(1, 4, [], SPT, cfg);
  engine.testFire('kick', 1);
  assert.deepEqual(Array.from(engine.state.depositMul), [1, 1, 1, 1]);

  cfg.responses.kick.enabled = true;
  cfg.enabled = false;
  engine.testFire('kick', 1);
  assert.deepEqual(Array.from(engine.state.depositMul), [1, 1, 1, 1]);
});

test('global gain scales every response', () => {
  const cfg = defaultImpulseConfig();
  cfg.gain = 0.5;
  const engine = new ImpulseEngine(1, 4, [], SPT, cfg);
  engine.testFire('kick', 1);
  assert.ok(
    Math.abs((engine.state.depositMul[0] as number) - (1 + 0.5 * cfg.responses.kick.deposit)) < 1e-6,
  );
});

// ── determinism ──────────────────────────────────────────────────────────────

test('hotspots are a pure function of (seed, eventIndex, slot)', () => {
  for (let i = 0; i < 50; i++) {
    const a = hotspotAt(0xc0ffee, i, 0);
    const b = hotspotAt(0xc0ffee, i, 0);
    assert.deepEqual(a, b);
    assert.ok(a.x >= 0 && a.x < 1 && a.y >= 0 && a.y < 1);
  }
});

test('hotspots move with the seed, with the event index and with the slot', () => {
  const base = hotspotAt(1, 10, 0);
  assert.notDeepEqual(base, hotspotAt(2, 10, 0));
  assert.notDeepEqual(base, hotspotAt(1, 11, 0));
  assert.notDeepEqual(base, hotspotAt(1, 10, 1));
});

test('hotspots are spread over the field, not clustered in a corner', () => {
  let sx = 0;
  let sy = 0;
  const n = 2000;
  for (let i = 0; i < n; i++) {
    const h = hotspotAt(0x5eed, i, 0);
    sx += h.x;
    sy += h.y;
  }
  assert.ok(Math.abs(sx / n - 0.5) < 0.03, `mean x ${sx / n}`);
  assert.ok(Math.abs(sy / n - 0.5) < 0.03, `mean y ${sy / n}`);
});

test('two runs of the same seed over the same events place identical splashes', () => {
  const events = Array.from({ length: 60 }, (_, i) => ev(i * 0.5, 'snare', 0.5 + (i % 3) * 0.1));
  const shot = (seed: number): string => {
    const engine = new ImpulseEngine(seed, 4, events, SPT);
    const out: string[] = [];
    for (let tick = 0; tick < 400; tick++) {
      engine.update(tick, SPT);
      for (const s of engine.state.splashes) {
        out.push(`${tick}:${s.x.toFixed(6)},${s.y.toFixed(6)},${s.strength.toFixed(6)}`);
      }
    }
    return out.join('|');
  };
  assert.equal(shot(4242), shot(4242));
  assert.notEqual(shot(4242), shot(4243));
});

test('splashes are capped, and the newest always survives', () => {
  const cfg = defaultImpulseConfig();
  cfg.responses.snare.splashCount = 8;
  const engine = new ImpulseEngine(9, 4, [], SPT, cfg);
  for (let i = 0; i < 20; i++) engine.testFire('snare', 1);
  assert.ok(engine.activeSplashes <= MAX_SPLASHES);
  const last = hotspotAt(9, 0x4000_0000 + 19, 7);
  assert.ok(
    engine.state.splashes.some((s) => Math.abs(s.x - last.x) < 1e-9 && Math.abs(s.y - last.y) < 1e-9),
  );
});

// ── absent / malformed events ────────────────────────────────────────────────

test('a timeline with no events array is a no-op engine', () => {
  const engine = new ImpulseEngine(1, 4, normalizeEvents(undefined, 100), SPT);
  assert.equal(engine.eventCount, 0);
  for (let tick = 0; tick < 500; tick++) engine.update(tick, SPT);
  assert.deepEqual(Array.from(engine.state.depositMul), [1, 1, 1, 1]);
  assert.equal(engine.activeSplashes, 0);
});

test('normalizeEvents drops junk, clamps strength, and sorts by t', () => {
  const raw = [
    { t: 5, kind: 'snare', strength: 2 },
    { t: 1, kind: 'kick', strength: -1 },
    { t: 3, kind: 'tuba', strength: 0.5 },
    { t: 'x', kind: 'kick', strength: 0.5 },
    { t: 2, kind: 'hat' },
    null,
  ];
  const out = normalizeEvents(raw, 100);
  assert.deepEqual(out, [
    { t: 1, kind: 'kick', strength: 0 },
    { t: 2, kind: 'hat', strength: 1 },
    { t: 5, kind: 'snare', strength: 1 },
  ]);
});

test('normalizeEvents clamps t into the track and tolerates a non-array', () => {
  assert.deepEqual(normalizeEvents({ nope: true }, 10), []);
  assert.deepEqual(normalizeEvents([{ t: 99, kind: 'kick', strength: 1 }], 10), [
    { t: 10, kind: 'kick', strength: 1 },
  ]);
});

test('the shipped synthetic fixture satisfies the events contract', async () => {
  const { readFile } = await import('node:fs/promises');
  const url = new URL('../../data/timelines/synthetic/timeline.json', import.meta.url);
  const manifest = JSON.parse(await readFile(url, 'utf8')) as {
    version: number;
    track: { duration: number };
    events?: TimelineEvent[];
  };
  assert.equal(manifest.version, 2);
  const events = manifest.events ?? [];
  assert.ok(events.length > 100, `only ${events.length} events`);
  const kinds = new Set(events.map((e) => e.kind));
  for (const k of EVENT_KINDS) assert.ok(kinds.has(k), `no ${k} events`);
  for (let i = 1; i < events.length; i++) {
    assert.ok((events[i - 1] as TimelineEvent).t <= (events[i] as TimelineEvent).t, 'not sorted');
  }
  // normalize must be a no-op on a conforming file
  assert.deepEqual(normalizeEvents(events, manifest.track.duration), events);
});

test('the deposit budget leaves the impulse burst its full depth and the atomic its headroom', async () => {
  const { MAX_DEPOSIT, MAX_EFFECTIVE_DEPOSIT, defaultConfig } = await import(
    '../src/sim/physarum/config.ts'
  );
  // Panel caps: impulses-panel.ts response.deposit <= 6 (multiplier 1 + 6·env),
  // panel.ts soil.depositBias <= 3 (fertility 1 + 3). A saturated base firing a
  // full-strength event must not clip — that is the Revision 2 §3 legibility bug.
  assert.ok(
    MAX_DEPOSIT >= MAX_EFFECTIVE_DEPOSIT * 7,
    `impulse burst clips at ${MAX_DEPOSIT / MAX_EFFECTIVE_DEPOSIT}x`,
  );
  // i32 atomicAdd headroom, measured against the true post-soil, post-impulse
  // bound rather than the base clamp. Below this the trail wraps negative.
  const agentsPerCell = 2 ** 31 / (MAX_DEPOSIT * defaultConfig().depositScale);
  assert.ok(agentsPerCell > 3500, `only ${Math.round(agentsPerCell)} agents/cell/tick`);
});
