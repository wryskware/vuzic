import assert from 'node:assert/strict';
import { test } from 'node:test';

import { EXPORT_PROFILES } from '../src/runtime/recipe.ts';
import { sdrDebugProfileDimensions } from '../src/export/sdr-debug-profile.ts';

test('honest SDR debug profiles map to exact native CFR120 dimensions', () => {
  assert.deepEqual(sdrDebugProfileDimensions('av1-sdr-debug-2160p120'), {
    width: 3840,
    height: 2160,
    fps: 120,
  });
  assert.deepEqual(sdrDebugProfileDimensions('av1-sdr-debug-1080p120'), {
    width: 1920,
    height: 1080,
    fps: 120,
  });
  assert.deepEqual(EXPORT_PROFILES, [
    'av1-sdr-debug-2160p120',
    'av1-sdr-debug-1080p120',
  ]);
});
