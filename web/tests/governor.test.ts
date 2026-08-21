/**
 * The budget governor's cross-machine contract: the fps-driven verdict law
 * (`governor.ts`) and the display-rate estimate it is held against
 * (`DisplayRateEstimator`). These walk the panels the app must be playable on —
 * the whole reason the law replaced a fixed ideal tuned on the author's
 * hardware.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { governorVerdict } from '../src/sim/plife/governor.ts';
import { DisplayRateEstimator } from '../src/sim/frame-timing.ts';

// Shipped defaults everywhere below unless a case is about the sliders.
const FLOOR = 60;
const IDEAL = 120;

test('a healthy 60 Hz panel grows — the one-way ratchet regression', () => {
  // A 60 Hz display measures 59.x at its very best; with a floor of exactly 60
  // a strict test shed 15%/s forever on a perfectly fine machine, and since the
  // old grow target (ideal 120) was unreachable there, the budget only ever
  // moved down. Meeting the panel's own measured rate must read as "grow".
  assert.equal(governorVerdict(59.2, FLOOR, IDEAL, 60), 'grow');
  // And 59.x must never read as "shrink" even before the display estimate
  // exists — an unreachable target merely holds.
  assert.equal(governorVerdict(59.2, FLOOR, IDEAL, null), 'hold');
});

test('a weak GPU sheds until it fits, wherever the panel is', () => {
  assert.equal(governorVerdict(42, FLOOR, IDEAL, 60), 'shrink');
  assert.equal(governorVerdict(50, FLOOR, IDEAL, 240), 'shrink');
});

test('the deadband holds', () => {
  // Above the floor tolerance (56.4) but under the grow threshold (58.2).
  assert.equal(governorVerdict(57, FLOOR, IDEAL, 60), 'hold');
  // Comfortably fine but not meeting the target on a fast panel.
  assert.equal(governorVerdict(110, FLOOR, IDEAL, 240), 'hold');
});

test('idealFps is a ceiling: fast panels stop asking past it', () => {
  // 240 Hz panel, target min(120, 240) = 120: meeting it within tolerance
  // grows, whether the loop free-runs at 238 or sags to 118.
  assert.equal(governorVerdict(238, FLOOR, IDEAL, 240), 'grow');
  assert.equal(governorVerdict(118, FLOOR, IDEAL, 240), 'grow');
  // 165 and 144 Hz panels behave identically — no divisor special-cases.
  assert.equal(governorVerdict(160, FLOOR, IDEAL, 165), 'grow');
  assert.equal(governorVerdict(140, FLOOR, IDEAL, 144), 'grow');
  assert.equal(governorVerdict(100, FLOOR, IDEAL, 144), 'hold');
});

test('crossed sliders degenerate into shed-below-the-floor, grow-at-the-floor', () => {
  // floor 90 above ideal 60: ideal reads as max(ideal, floor) = 90.
  assert.equal(governorVerdict(89, 90, 60, 240), 'grow');
  assert.equal(governorVerdict(80, 90, 60, 240), 'shrink');
});

test('a throttled display estimate cannot invite growth from inside the shed band', () => {
  // A machine so slow no frame ever fit one slot estimates a multiple of the
  // true period (here 30 for a 60 Hz panel). The grow threshold is floored at
  // floorFps, so near-floor fps holds rather than growing into a machine that
  // is already failing the floor.
  assert.equal(governorVerdict(58, FLOOR, IDEAL, 30), 'hold');
  assert.equal(governorVerdict(30, FLOOR, IDEAL, 30), 'shrink');
});

/** Feed a run of inter-frame deltas (ms), starting at t = 0. */
function feed(est: DisplayRateEstimator, deltas: number[], startMs = 0): number {
  let t = startMs;
  est.note(t);
  for (const d of deltas) {
    t += d;
    est.note(t);
  }
  return t;
}

test('the estimator is null until it has a usable delta', () => {
  const est = new DisplayRateEstimator();
  assert.equal(est.hz, null);
  est.note(0);
  assert.equal(est.hz, null, 'one timestamp is no delta');
  est.note(16.67);
  assert.ok(est.hz !== null);
});

test('a steady cadence reads as the panel rate', () => {
  const est = new DisplayRateEstimator();
  feed(est, Array(120).fill(16.67));
  assert.ok(Math.abs((est.hz as number) - 60) < 0.5, `wanted ~60, got ${est.hz}`);
});

test('a loaded 240 Hz panel is identified from its rare single-slot frames', () => {
  // The app mostly takes 2-3 slots per frame; one frame that fit a single
  // 4.17 ms slot inside the window is enough to name the display.
  const est = new DisplayRateEstimator();
  const deltas = Array.from({ length: 240 }, (_, i) => (i === 100 ? 4.17 : i % 2 ? 8.33 : 12.5));
  feed(est, deltas);
  assert.ok(Math.abs((est.hz as number) - 240) < 3, `wanted ~240, got ${est.hz}`);
});

test('hitches and background gaps never enter the estimate', () => {
  const est = new DisplayRateEstimator();
  const t = feed(est, Array(60).fill(16.67));
  feed(est, [2000, ...Array(60).fill(16.67)], t);
  assert.ok(Math.abs((est.hz as number) - 60) < 0.5, `wanted ~60, got ${est.hz}`);
});

test('coarsened or duplicated timestamps never invent a fast panel', () => {
  const est = new DisplayRateEstimator();
  feed(est, [0.5, 1, 0.1, 1.9]);
  assert.equal(est.hz, null, 'sub-2 ms deltas are not vsync slots');
});

test('moving the window to a slower monitor is adopted within the rolling window', () => {
  const est = new DisplayRateEstimator();
  // Three seconds on a 240 Hz panel...
  const t = feed(est, Array(720).fill(4.17));
  assert.ok((est.hz as number) > 200);
  // ...then twelve seconds on a 60 Hz one: the fast buckets age out.
  feed(est, Array(720).fill(16.67), t);
  assert.ok(Math.abs((est.hz as number) - 60) < 0.5, `wanted ~60, got ${est.hz}`);
});
