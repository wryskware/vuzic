import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import { configureCanvas, displayHeadroom, hdrOverride } from '../src/gpu/hdr-canvas.ts';

/**
 * The extended-range swapchain is the one piece of the HDR path that cannot be
 * verified by looking at the image: an 8-bit swapchain and a float one that the
 * compositor silently clamped produce *identical* pixels on an SDR display, and
 * the only difference on an HDR one is highlights the author may well believe
 * are just the tone curve. So the failure modes are pinned here instead.
 */

async function source(relative: string): Promise<string> {
  return readFile(fileURLToPath(new URL(`../src/${relative}`, import.meta.url)), 'utf8');
}

/** A `GPUCanvasContext` stub that records what it was configured with. */
function fakeContext(configuredMode: string | null, throwOnFloat = false) {
  const calls: GPUCanvasConfiguration[] = [];
  return {
    calls,
    configure(config: GPUCanvasConfiguration) {
      if (throwOnFloat && config.format === 'rgba16float') {
        throw new TypeError('unsupported format');
      }
      calls.push(config);
    },
    getConfiguration() {
      const last = calls[calls.length - 1];
      if (!last || last.format !== 'rgba16float') return { toneMapping: { mode: 'standard' } };
      return configuredMode === null ? null : { toneMapping: { mode: configuredMode } };
    },
  };
}

const device = {} as GPUDevice;

function withGlobals<T>(patch: Record<string, unknown>, body: () => T): T {
  const scope = globalThis as Record<string, unknown>;
  const saved = new Map(Object.keys(patch).map((key) => [key, scope[key]]));
  for (const [key, value] of Object.entries(patch)) {
    Object.defineProperty(scope, key, { value, configurable: true, writable: true });
  }
  try {
    return body();
  } finally {
    for (const [key, value] of saved) {
      Object.defineProperty(scope, key, { value, configurable: true, writable: true });
    }
  }
}

const preferredFormat = {
  gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' as GPUTextureFormat },
};

test('an accepted extended configuration is the one that ships', () => {
  const ctx = fakeContext('extended');
  const state = withGlobals({ navigator: preferredFormat }, () =>
    configureCanvas(ctx as unknown as GPUCanvasContext, device, true),
  );
  assert.equal(state.format, 'rgba16float');
  assert.equal(state.extended, true);
  assert.equal(ctx.calls.length, 1);
  assert.deepEqual(ctx.calls[0]?.toneMapping, { mode: 'extended' });
});

test('a browser that ignores toneMapping falls back rather than shipping a clamped float surface', () => {
  // The dangerous case: configure() succeeds, so nothing throws, but the
  // compositor still clamps at 1.0. Without the read-back this would burn a
  // 64-bit swapchain for an image identical to the 32-bit one.
  const ctx = fakeContext('standard');
  const state = withGlobals({ navigator: preferredFormat }, () =>
    configureCanvas(ctx as unknown as GPUCanvasContext, device, true),
  );
  assert.equal(state.extended, false);
  assert.equal(state.format, 'bgra8unorm');
  assert.equal(ctx.calls.length, 2, 'the swapchain must be reconfigured, not left float');
});

test('getConfiguration missing entirely is a fallback, not a crash', () => {
  const ctx = fakeContext(null);
  const state = withGlobals({ navigator: preferredFormat }, () =>
    configureCanvas(ctx as unknown as GPUCanvasContext, device, true),
  );
  assert.equal(state.extended, false);
  assert.equal(state.format, 'bgra8unorm');
});

test('a rejected float format falls back with the reason attached', () => {
  const ctx = fakeContext('extended', true);
  const state = withGlobals({ navigator: preferredFormat }, () =>
    configureCanvas(ctx as unknown as GPUCanvasContext, device, true),
  );
  assert.equal(state.extended, false);
  assert.match(state.detail, /unsupported format/);
});

test('?hdr=off configures the 8-bit swapchain and never asks for float', () => {
  const ctx = fakeContext('extended');
  const state = withGlobals({ navigator: preferredFormat }, () =>
    configureCanvas(ctx as unknown as GPUCanvasContext, device, false),
  );
  assert.equal(state.extended, false);
  assert.equal(ctx.calls.length, 1);
  assert.equal(ctx.calls[0]?.format, 'bgra8unorm');
});

test('the hdr query parameter reads as off, a pinned headroom, or nothing', () => {
  assert.equal(hdrOverride(''), null);
  assert.equal(hdrOverride('?sim=plife'), null);
  assert.equal(hdrOverride('?hdr=off'), 'off');
  assert.equal(hdrOverride('?hdr=0'), 'off');
  assert.equal(hdrOverride('?hdr=3.5'), 3.5);
  // Clamped, so a typo cannot pin the grade somewhere no display can follow.
  assert.equal(hdrOverride('?hdr=1000'), 16);
  // Below diffuse white is meaningless; ignore it rather than darken the image.
  assert.equal(hdrOverride('?hdr=0.5'), null);
  assert.equal(hdrOverride('?hdr=yes'), null);
});

test('reported headroom is clamped, and its absence means SDR unless the display says otherwise', () => {
  const sdrMedia = { matchMedia: () => ({ matches: false }) };
  assert.equal(withGlobals({ screen: { highDynamicRangeHeadroom: 5.5 } }, displayHeadroom), 5.5);
  // A panel reporting less than diffuse white would darken the whole grade.
  assert.equal(withGlobals({ screen: { highDynamicRangeHeadroom: 0.2 } }, displayHeadroom), 1);
  assert.equal(withGlobals({ screen: { highDynamicRangeHeadroom: 1e6 } }, displayHeadroom), 16);
  assert.equal(
    withGlobals({ screen: {}, window: sdrMedia }, displayHeadroom),
    1,
    'no API and an SDR display must collapse the grade to the SDR curve',
  );
});

test('the SDR grade stretches to the headroom and clamps there, not at diffuse white', async () => {
  const grade = await source('sim/render/shaders/grade.wgsl');
  assert.match(
    grade,
    /headroom \* tonemapSelect\(c \/ headroom, p\.tonemap\)/,
    'the display map must be the headroom-stretched curve, which is the identity at 1',
  );
  assert.match(
    grade,
    /clamp\(c, vec3f\(0\.0\), vec3f\(headroom\)\)/,
    'clamping at 1.0 would throw away every value the float swapchain exists to carry',
  );
});

test('the browser preview feeds its measured headroom to the grade', async () => {
  const postfx = await source('sim/render/postfx.ts');
  assert.match(postfx, /this\.ctx\?\.displayHeadroom \?\? 1/);
  // The export path owns its headroom from the mastering policy; the two must
  // not be crossed, or an export would inherit whatever panel is attached.
  assert.match(postfx, /hdr\s*\n?\s*\? Math\.max\(hdr\.masteringPeakNits \/ hdr\.paperWhiteNits, 1\)/);
});
