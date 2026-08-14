/**
 * The feedback lane's exponent. These guard the flicker fix: with the world
 * frozen the render-domain echo settles at `S/(1-a)` where `a = pow(amount,
 * dtFrames)`, so any jitter in `dtFrames` is amplified by `1/(1-a)^2` — 69x at
 * the shipped 0.88 — and arrives on screen as a whole-image brightness wobble.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { DT_FRAMES_TAU_S, smoothDtFrames } from '../src/sim/step-cadence.ts';

/** The shipped plife feedback amount; the echo's steady-state gain at `a`. */
const AMOUNT = 0.88;
const echoGain = (dtFrames: number): number => 1 / (1 - Math.pow(AMOUNT, dtFrames));

/** rAF deltas alternating +-15% around a nominal rate, the worst ordinary case. */
function jittered(nominalHz: number, frames: number): number[] {
  const dt = 1 / nominalHz;
  return Array.from({ length: frames }, (_, i) => dt * (i % 2 === 0 ? 1.15 : 0.85));
}

test('the first frame snaps, so a fixed-cadence export is exact from frame one', () => {
  // 30 fps export: 2 sixty-hertz frames per rendered frame, no easing out of the
  // 60 Hz seed.
  assert.equal(smoothDtFrames(1, 1 / 30, true), 2);
  // An unprimed previous value snaps too, whatever the caller passed.
  assert.equal(smoothDtFrames(0, 1 / 30), 2);
});

test('a steady cadence is a fixed point at any refresh rate', () => {
  for (const hz of [60, 120, 144, 240]) {
    const want = 60 / hz;
    let v = smoothDtFrames(0, 1 / hz, true);
    for (let i = 0; i < 200; i++) v = smoothDtFrames(v, 1 / hz);
    assert.ok(
      Math.abs(v - want) < 1e-9,
      `${hz} Hz should hold at ${want} frames, got ${v}`,
    );
  }
});

test('+-15% delta jitter moves the frozen echo by well under 1%', () => {
  const deltas = jittered(60, 200);
  let smoothed = smoothDtFrames(0, deltas[0] as number, true);
  let lo = Infinity;
  let hi = -Infinity;
  let rawLo = Infinity;
  let rawHi = -Infinity;
  // The first frame snaps by design, so the run starts on one arbitrary sample
  // and eases to the mean. That is a settling transient, not flicker — the claim
  // under test is about the steady state, so measure it there.
  const settle = 100;
  for (const [i, dt] of deltas.entries()) {
    smoothed = smoothDtFrames(smoothed, dt);
    if (i < settle) continue;
    // What the previous code did: the raw measurement, straight into the power.
    const raw = dt * 60;
    lo = Math.min(lo, echoGain(smoothed));
    hi = Math.max(hi, echoGain(smoothed));
    rawLo = Math.min(rawLo, echoGain(raw));
    rawHi = Math.max(rawHi, echoGain(raw));
  }
  const swing = (hi - lo) / lo;
  const rawSwing = (rawHi - rawLo) / rawLo;
  assert.ok(rawSwing > 0.15, `the unsmoothed lane really does swing (${rawSwing})`);
  assert.ok(swing < 0.01, `smoothed brightness swing should be under 1%, got ${swing}`);
});

test('a genuine rate change is absorbed, not ignored', () => {
  // 60 Hz settled, then the display becomes 144 Hz: within ~3 tau the exponent
  // must describe the new rate, or the echo would be wrong for as long as it
  // lasted.
  let v = smoothDtFrames(0, 1 / 60, true);
  const frames = Math.ceil(5 * DT_FRAMES_TAU_S * 144);
  for (let i = 0; i < frames; i++) v = smoothDtFrames(v, 1 / 144);
  assert.ok(Math.abs(v - 60 / 144) < 0.01, `should track 144 Hz, got ${v}`);
});

test('a backgrounded tab is clamped, so the echo is never cleared outright', () => {
  // Two seconds away: 0.88^120 is zero to any float, i.e. a black flash. The
  // clamp caps one sample at 0.25 s, and the smoother caps the step further.
  const v = smoothDtFrames(1, 2, true);
  assert.equal(v, 15, '0.25 s is the ceiling, in 60 Hz frames');
  assert.ok(Math.pow(AMOUNT, v) > 0.1, 'the echo survives the frame back');
});

test('a coarsened timestamp cannot switch a faded echo fully on', () => {
  // pow(x, 0) is 1 for every x, which would be a disabled feedback lane turning
  // on for one frame. The floor makes the exponent unreachable.
  assert.ok(smoothDtFrames(0, 0, true) > 0);
  assert.ok(smoothDtFrames(1, 0) > 0);
});
