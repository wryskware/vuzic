/** Regression guards for the app-wide 120 Hz clock transition. */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { AudioClock } from '../src/audio/clock.ts';
import { ImpulseEngine } from '../src/sim/impulses.ts';
import { PLIFE_SUBSTEP_DT } from '../src/sim/plife/timing.ts';
import { advanceStepCadence } from '../src/sim/step-cadence.ts';
import { LEGACY_TICK_DIVISOR, SECONDS_PER_TICK, TICK_HZ } from '../src/timing.ts';

function stepsOver(appTicks: number, tickDivisor: number): number {
  let accumulator = 0;
  let total = 0;
  for (let tick = 1; tick <= appTicks; tick++) {
    const next = advanceStepCadence(accumulator, 1, 8, tick, tickDivisor);
    accumulator = next.accumulator;
    total += next.steps;
  }
  return total;
}

test('10 seconds at 120 Hz gives plife 1200 states and legacy substrates 600', () => {
  const ticks = 10 * TICK_HZ;
  assert.equal(LEGACY_TICK_DIVISOR, 2, 'the 120 Hz app clock is exactly 2x the legacy model rate');
  assert.equal(stepsOver(ticks, 1), 1200, 'plife');
  assert.equal(stepsOver(ticks, LEGACY_TICK_DIVISOR), 600, 'physarum');
  assert.equal(stepsOver(ticks, LEGACY_TICK_DIVISOR), 600, 'vizfx');
});

function playingClock(atSeconds: number): AudioClock {
  const clock = new AudioClock({ duration: 10, beats: [], downbeats: [] });
  Object.assign(clock as unknown as Record<string, unknown>, {
    playing: true,
    ctx: { currentTime: atSeconds },
    node: {},
    originTime: 0,
  });
  return clock;
}

test('AudioClock absorbs 100 ms at 120 Hz and resyncs after a long stall', () => {
  const short = playingClock(12 * SECONDS_PER_TICK);
  const ran: number[] = [];
  assert.equal(short.pump((tick) => ran.push(tick)), 12);
  assert.deepEqual(ran, Array.from({ length: 12 }, (_, i) => i + 1));
  assert.equal(short.simTick, 12, '100 ms is fully drained without discarding time');

  const stalled = playingClock(2);
  let calls = 0;
  assert.equal(stalled.pump(() => calls++), 16, 'long stalls still obey the per-frame work cap');
  assert.equal(calls, 16);
  assert.equal(stalled.simTick, 240, 'the clock resyncs to audio after capped catch-up');
});

function impulseAfterOneSecond(hz: number): number {
  const dt = 1 / hz;
  const engine = new ImpulseEngine(1, 4, [], dt);
  engine.testFire('kick', 1);
  for (let tick = 1; tick <= hz; tick++) engine.update(tick, dt);
  return engine.levelOf('kick');
}

test('impulse decay is unchanged over one wall-clock second at 120 Hz', () => {
  const at60 = impulseAfterOneSecond(60);
  const at120 = impulseAfterOneSecond(120);
  assert.ok(Math.abs(at120 - at60) < 1e-6, `${at60} vs ${at120}`);
});

test('Particle Life integration dt is the app tick dt', () => {
  assert.equal(PLIFE_SUBSTEP_DT, SECONDS_PER_TICK);
  assert.equal(PLIFE_SUBSTEP_DT, 1 / 120);
});
