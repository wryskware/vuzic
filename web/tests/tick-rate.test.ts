/**
 * The live loop's timing contract, after the move off the fixed 120 Hz tick grid.
 *
 * What these pin, and why each is worth pinning:
 *
 * - **One advance per rendered frame, driven by measured dt.** The failure this
 *   replaced was structural: being behind the audio clock bought *more* sim work
 *   on the next frame, which put you further behind. Nothing here may reintroduce
 *   a quantity that grows when a frame is late.
 * - **The clamp is the whole stall policy.** A tab restore, a GC pause and a
 *   machine that simply cannot hold 30 fps are all the same case, and the answer
 *   to all three is that the world runs slow.
 * - **`AudioClock` is a lookup, not an accumulator.** It answers "where is the
 *   audio" every frame from `ctx.currentTime`. A long frame lands on a later
 *   sample; it never owes a backlog, and skipped samples are never simulated.
 * - **Envelope decay is rate-independent.** It was rate-independent across two
 *   FIXED rates before; it now has to hold across an arbitrary mix of step
 *   lengths, which is a strictly stronger property and the one that keeps a
 *   kick's tail the same length on every machine.
 *
 * `TICK_HZ` survives all of this as the rate the analysis was *sampled* at. It
 * is a property of the timeline data and of the offline export, and no assertion
 * here treats it as a simulation clock.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { AudioClock } from '../src/audio/clock.ts';
import { ImpulseEngine } from '../src/sim/impulses.ts';
import { EventCursor } from '../src/timeline/sampler.ts';
import { dtScale } from '../src/sim/frame-timing.ts';
import {
  clampFrameDt,
  LEGACY_MODEL_DT,
  MAX_CONTINUOUS_TICK_GAP,
  MAX_FRAME_DT,
  SECONDS_PER_TICK,
  TICK_HZ,
} from '../src/timing.ts';

// ── the dt clamp ─────────────────────────────────────────────────────────────

test('a frame advances at most 1/60 s of world, however long it really took', () => {
  // 1/60 and not lower: at 1/30 a ceiling-speed particle crossed several
  // hard-core radii per step and close pairs tunnelled (observed on real
  // hardware, 2026-08-20). See MAX_FRAME_DT in timing.ts.
  assert.equal(MAX_FRAME_DT, 1 / 60);
  // Ordinary frames pass through untouched — the clamp is a ceiling, not a quantiser.
  assert.equal(clampFrameDt(1 / 60), 1 / 60);
  assert.equal(clampFrameDt(1 / 144), 1 / 144);
  assert.equal(clampFrameDt(1 / 60), 1 / 60);
  // A backgrounded tab returning after two seconds, and a 10 fps machine, are
  // the same case and get the same answer.
  assert.equal(clampFrameDt(2), MAX_FRAME_DT);
  assert.equal(clampFrameDt(0.1), MAX_FRAME_DT);
  // Non-positive and non-finite deltas advance nothing rather than going backwards.
  assert.equal(clampFrameDt(0), 0);
  assert.equal(clampFrameDt(-0.01), 0);
  assert.equal(clampFrameDt(Number.NaN), 0);
});

test('work per wall second never rises as the frame rate falls', () => {
  // One step per frame is the contract, so "work per second" is just the frame
  // rate — but the WORLD advanced per second is fps x clamped dt, and that is
  // the quantity the old pump inflated by catching up.
  const worldPerSecond = (fps: number): number => fps * clampFrameDt(1 / fps);
  assert.equal(worldPerSecond(144), 1, 'a fast display: real time, exactly');
  assert.equal(worldPerSecond(60), 1, 'the clamp is reached but not exceeded');
  // Below the clamp the world falls behind wall time, deliberately and forever:
  // it does not accrue a debt that a later frame pays off.
  assert.ok(worldPerSecond(30) < 1);
  assert.equal(worldPerSecond(30), 0.5);
  assert.equal(worldPerSecond(15), 0.25);
  assert.equal(worldPerSecond(6), 0.1);
});

// ── the per-step scale legacy substrates run on ──────────────────────────────

test('dtScale is a pure function of the frame, with no carried state', () => {
  assert.equal(LEGACY_MODEL_DT, 1 / 60);
  // 60 Hz is an exact float identity, which is what makes the shipped physarum
  // and vizfx looks bit-identical on the display they were tuned on.
  assert.equal(dtScale(1 / 60, LEGACY_MODEL_DT), 1);
  assert.equal(dtScale(1 / 120, LEGACY_MODEL_DT), 0.5);
  assert.equal(dtScale(1 / 30, LEGACY_MODEL_DT), 2);
  // A frozen world, and a `speed` of 0, take no step at all.
  assert.equal(dtScale(0, LEGACY_MODEL_DT), 0);
  assert.equal(dtScale(-1, LEGACY_MODEL_DT), 0);

  // The point of the whole refactor, expressed as a property: calling it a
  // hundred times in any order gives the same answers as calling it once. There
  // is nowhere for a backlog to live.
  const pattern = [1 / 60, 1 / 240, 1 / 31, 1 / 60, 1 / 90];
  for (let i = 0; i < 100; i++) {
    const dt = pattern[i % pattern.length] as number;
    assert.equal(dtScale(dt, LEGACY_MODEL_DT), dt * 60);
  }
});

test('a run of frames advances exactly the world time its deltas add up to', () => {
  // Any mix of frame lengths, in any order: no rounding to a step grid, no
  // leftover, no burst. World time is the sum of the clamped deltas.
  // All under the 1/60 clamp on purpose: over-clamp frames are the previous
  // test's subject; this one is about the absence of grid rounding.
  const deltas = [1 / 60, 1 / 144, 1 / 90, 1 / 60, 1 / 61, 1 / 240];
  let world = 0;
  for (const d of deltas) world += clampFrameDt(d);
  const expected = deltas.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(world - expected) < 1e-12, `${world} vs ${expected}`);
});

// ── the transport ────────────────────────────────────────────────────────────

function clockAt(atSeconds: number, playing = true): AudioClock {
  const clock = new AudioClock({ duration: 10, beats: [], downbeats: [] });
  Object.assign(clock as unknown as Record<string, unknown>, {
    playing,
    ctx: { currentTime: atSeconds },
    node: { stop: () => {}, disconnect: () => {} },
    originTime: 0,
  });
  return clock;
}

test('the transport is sampled, not drained: no catch-up and no backlog', () => {
  assert.equal(SECONDS_PER_TICK, 1 / TICK_HZ);

  // A frame that lands 100 ms into the track reads the sample at 100 ms. It does
  // not run the twelve it "missed" — those are analysis samples the display was
  // never going to show, and the old pump ran every one of them as a full sim step.
  const short = clockAt(12 * SECONDS_PER_TICK);
  assert.equal(short.sampleTick(), 12);

  // A two-second stall reads the sample at two seconds, in one call, at the same
  // cost. This is the case that used to spiral.
  const stalled = clockAt(2);
  assert.equal(stalled.sampleTick(), 240);

  // Sampling is idempotent: asking twice without the audio moving is the same
  // answer, because nothing is consumed.
  assert.equal(stalled.sampleTick(), 240);
});

test('an idle transport freezes the timeline position, so no event re-fires', () => {
  const idle = clockAt(3, false);
  Object.assign(idle as unknown as Record<string, unknown>, { pausedAt: 1.5 });
  const first = idle.sampleTick();
  assert.equal(first, Math.floor(1.5 / SECONDS_PER_TICK));
  // Frames keep arriving while paused — that is what keeps envelopes decaying
  // and the substrate stepping — but the position they read does not move.
  assert.equal(idle.sampleTick(), first);
  assert.equal(idle.sampleTick(), first);
});

test('reaching the end of the track stops the transport', () => {
  const done = clockAt(10.5);
  done.sampleTick();
  assert.equal(done.isPlaying, false);
  assert.equal(done.simTick, Math.floor(10 / SECONDS_PER_TICK), 'parked at the duration');
});

// ── envelopes ────────────────────────────────────────────────────────────────

/**
 * Decay one kick over one wall-clock second, spending that second as `steps`
 * equal frames. The invariant this protects is unchanged from the fixed-rate
 * era — a kick's tail is a wall-clock duration, not a step count — but it now
 * has to survive step lengths chosen by whatever the display did.
 */
function levelAfterOneSecond(steps: number): number {
  const dt = 1 / steps;
  const engine = new ImpulseEngine(1, 4, [], SECONDS_PER_TICK);
  engine.testFire('kick', 1);
  for (let i = 1; i <= steps; i++) engine.update(0, dt);
  return engine.levelOf('kick');
}

test('envelope decay over one wall second is independent of how the frames fell', () => {
  const at240 = levelAfterOneSecond(240);
  for (const hz of [144, 120, 60, 30]) {
    const at = levelAfterOneSecond(hz);
    // 1e-6, the same tolerance the fixed-rate version of this test used: what
    // separates rate-independent decay from a per-step alpha is orders of
    // magnitude, and what is left here is float summation of the dts.
    assert.ok(Math.abs(at - at240) < 1e-6, `${hz} fps: ${at} vs ${at240}`);
  }
});

test('envelope decay over one wall second survives a ragged frame rate', () => {
  // The realistic case: a second of frames of wildly different lengths, summing
  // to one second. Exponential-in-dt decay makes this exactly equal to the
  // steady case; a per-step alpha would not.
  const engine = new ImpulseEngine(1, 4, [], SECONDS_PER_TICK);
  engine.testFire('kick', 1);
  let spent = 0;
  const ragged = [1 / 60, 1 / 200, 1 / 31, 1 / 60, 1 / 90, 1 / 45];
  for (let i = 0; spent < 1; i++) {
    const dt = Math.min(ragged[i % ragged.length] as number, 1 - spent);
    engine.update(0, dt);
    spent += dt;
  }
  const steady = levelAfterOneSecond(60);
  assert.ok(
    Math.abs(engine.levelOf('kick') - steady) < 1e-6,
    `${engine.levelOf('kick')} vs ${steady}`,
  );
});

// ── transport continuity ─────────────────────────────────────────────────────

/**
 * The threshold three lanes share (`EventCursor`, the Modulator's stem-follow
 * snap, main's section-boundary detector) to tell playback from a seek.
 *
 * This is the contract most easily broken by the move to per-frame advance and
 * the one with the quietest failure: at the old value of 1 tick, EVERY ordinary
 * frame on a 60 Hz display looks like a seek, so the cursor relocates instead of
 * walking and no impulse ever fires again. Nothing throws; the visuals simply
 * stop responding to the music.
 */
test('one frame of ordinary playback is continuous, at every plausible frame rate', () => {
  const gapAt = (fps: number): number => Math.ceil(1 / fps / SECONDS_PER_TICK);
  for (const fps of [144, 120, 90, 60, 45, 30]) {
    assert.ok(
      gapAt(fps) <= MAX_CONTINUOUS_TICK_GAP,
      `${fps} fps crosses ${gapAt(fps)} ticks, past the ${MAX_CONTINUOUS_TICK_GAP}-tick line`,
    );
  }
  // …and it is still tight enough that a seek is a seek: a quarter of a second
  // of timeline is 30 ticks and must not be walked.
  assert.ok(MAX_CONTINUOUS_TICK_GAP < 0.25 / SECONDS_PER_TICK);
});

test('the event cursor walks a frame-sized gap and relocates past a seek', () => {
  const events = Array.from({ length: 6 }, (_, i) => ({
    t: (i + 1) * SECONDS_PER_TICK,
    kind: 'kick' as const,
    strength: 1,
  }));

  // A 60 Hz display crossing two ticks per frame fires everything in between —
  // the whole point of the walk, and what the old 1-tick threshold would have
  // silently thrown away.
  const walking = new EventCursor(events, SECONDS_PER_TICK);
  walking.advance(0);
  let fired = 0;
  for (let tick = 2; tick <= 6; tick += 2) {
    const { start, end } = walking.advance(tick);
    fired += end - start;
  }
  assert.equal(fired, 6, 'every event in the walked span fires exactly once');

  // A scrub does not: it discards what it skipped rather than dumping it.
  const seeking = new EventCursor(events, SECONDS_PER_TICK);
  seeking.advance(0);
  const jump = seeking.advance(MAX_CONTINUOUS_TICK_GAP + 1);
  assert.equal(jump.end - jump.start, 0, 'a seek relocates without firing');
});
