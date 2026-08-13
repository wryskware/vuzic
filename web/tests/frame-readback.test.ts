import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  alignTo,
  frameReadbackLayout,
  stripReadbackRowPadding,
} from '../src/export/frame-readback.ts';

test('readback layout aligns WebGPU copy rows to 256 bytes', () => {
  const aligned = frameReadbackLayout(3840, 2160, 4);
  assert.equal(aligned.unpaddedBytesPerRow, 15_360);
  assert.equal(aligned.paddedBytesPerRow, 15_360);
  assert.equal(aligned.bytesPerFrame, 33_177_600);
  assert.equal(aligned.stagingBytesPerFrame, 33_177_600);

  const padded = frameReadbackLayout(1919, 2, 4);
  assert.equal(padded.unpaddedBytesPerRow, 7676);
  assert.equal(padded.paddedBytesPerRow, 7680);
  assert.equal(padded.bytesPerFrame, 15_352);
  assert.equal(padded.stagingBytesPerFrame, 15_360);
  assert.equal(alignTo(1, 256), 256);
});

test('row padding is stripped without leaking padding bytes', () => {
  const layout = frameReadbackLayout(3, 2, 1);
  const staging = new Uint8Array(layout.stagingBytesPerFrame).fill(255);
  staging.set([1, 2, 3], 0);
  staging.set([4, 5, 6], layout.paddedBytesPerRow);

  assert.deepEqual(stripReadbackRowPadding(staging, layout), new Uint8Array([1, 2, 3, 4, 5, 6]));
  assert.throws(
    () => stripReadbackRowPadding(staging.subarray(0, staging.length - 1), layout),
    /expected/,
  );
});

test('readback layout rejects invalid or unsafe sizes', () => {
  assert.throws(() => frameReadbackLayout(0, 1, 4), /width/);
  assert.throws(() => frameReadbackLayout(1, -1, 4), /height/);
  assert.throws(() => frameReadbackLayout(1, 1, 0), /bytesPerPixel/);
  assert.throws(() => frameReadbackLayout(Number.MAX_SAFE_INTEGER, 2, 4), /safe integer/);
});
