import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  EXPORT_RENDER_DELTA_SECONDS,
  ExportClock,
  exportFrameCount,
  targetSimulationTick,
} from '../src/export/export-clock.ts';

test('frame count is ceil(duration * 120), including exactly 7,200 frames per minute', () => {
  assert.equal(exportFrameCount(0), 0);
  assert.equal(exportFrameCount(1 / 120), 1);
  assert.equal(exportFrameCount(1 / 120 + Number.EPSILON), 2);
  assert.equal(exportFrameCount(60), 7200);
});

test('target app tick is floor(frame * 120 / 120) with integer counters', () => {
  assert.deepEqual(
    Array.from({ length: 8 }, (_, frame) => targetSimulationTick(frame)),
    [0, 1, 2, 3, 4, 5, 6, 7],
  );
  assert.equal(targetSimulationTick(7199), 7199);
  assert.equal(targetSimulationTick(Number.MAX_SAFE_INTEGER), Number.MAX_SAFE_INTEGER);
});

test('each frame advances every missing tick before rendering', () => {
  const clock = new ExportClock(5 / 120);
  const order: string[] = [];
  const frames = [];
  while (!clock.done) {
    clock.advanceThenRender(
      (tick) => order.push(`advance ${tick}`),
      (frame) => {
        order.push(`render ${frame.frameIndex}`);
        frames.push(frame);
      },
    );
  }

  assert.deepEqual(order, [
    'render 0',
    'advance 1',
    'render 1',
    'advance 2',
    'render 2',
    'advance 3',
    'render 3',
    'advance 4',
    'render 4',
  ]);
  assert.equal(frames[4]?.timeSeconds, 4 / 120);
  assert.equal(frames[4]?.deltaSeconds, EXPORT_RENDER_DELTA_SECONDS);
  assert.equal(frames[4]?.targetSimulationTick, 4);
  assert.equal(clock.advanceThenRender(() => assert.fail(), () => assert.fail()), null);
});

test('invalid durations and counters fail before scheduling', () => {
  assert.throws(() => exportFrameCount(-1), /finite non-negative/);
  assert.throws(() => exportFrameCount(Number.POSITIVE_INFINITY), /finite non-negative/);
  assert.throws(() => targetSimulationTick(0.5), /safe integer/);
});
