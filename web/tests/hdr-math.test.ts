import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  BT2020_KB,
  BT2020_KG,
  BT2020_KR,
  DEFAULT_MASTERING_PEAK_NITS,
  DEFAULT_PAPER_WHITE_NITS,
  bt2020NclYCbCr,
  bt709ToBt2020,
  hdrHeadroom,
  masteringDisplayMetadata,
  p010Layout,
  p010Sample,
  pqDecodeNits,
  pqEncodeNits,
  quantizeChroma10,
  quantizeLuma10,
  validateHdrOutputSettings,
} from '../src/export/hdr.ts';

/**
 * External reference points, not values read back out of this implementation.
 * These are the ST 2084 code values quoted for the standard's own anchors.
 */
test('PQ matches the published ST 2084 anchors and both ends are exact', () => {
  assert.equal(pqEncodeNits(0), 0);
  assert.equal(pqEncodeNits(-5), 0);
  assert.equal(pqEncodeNits(10_000), 1);
  // Above the standard's peak is clamped, never extrapolated.
  assert.equal(pqEncodeNits(50_000), 1);

  assert.ok(Math.abs(pqEncodeNits(100) - 0.5081) < 1e-4, `${pqEncodeNits(100)}`);
  assert.ok(Math.abs(pqEncodeNits(1000) - 0.7518) < 1e-4, `${pqEncodeNits(1000)}`);
  assert.ok(Math.abs(pqEncodeNits(203) - 0.5807) < 1e-4, `${pqEncodeNits(203)}`);
  assert.ok(Math.abs(pqEncodeNits(1) - 0.1499) < 1e-4, `${pqEncodeNits(1)}`);
});

test('PQ is strictly increasing and round-trips through its own EOTF', () => {
  let previous = -1;
  for (const nits of [0.001, 0.1, 1, 10, 100, 203, 500, 1000, 4000, 10_000]) {
    const code = pqEncodeNits(nits);
    assert.ok(code > previous, `PQ must increase with luminance at ${nits} nits`);
    previous = code;
    const back = pqDecodeNits(code);
    assert.ok(
      Math.abs(back - nits) <= Math.max(nits * 1e-6, 1e-9),
      `round trip of ${nits} nits produced ${back}`,
    );
  }
  assert.equal(pqDecodeNits(0), 0);
  assert.ok(Math.abs(pqDecodeNits(1) - 10_000) < 1e-6);
});

test('BT.709 to BT.2020 preserves the achromatic axis and narrows saturation', () => {
  const white = bt709ToBt2020([1, 1, 1]);
  for (const channel of white) assert.ok(Math.abs(channel - 1) < 1e-6, `white drifted: ${channel}`);
  assert.deepEqual(bt709ToBt2020([0, 0, 0]), [0, 0, 0]);

  // A BT.709 primary is inside the wider BT.2020 gamut, so it must stop being a
  // pure primary: the other two channels lift off zero and the main one drops.
  const red = bt709ToBt2020([1, 0, 0]);
  assert.ok(red[0] < 1 && red[0] > 0.6);
  assert.ok(red[1] > 0 && red[2] > 0);
});

test('BT.2020 non-constant luminance matrix is self-consistent', () => {
  assert.ok(Math.abs(BT2020_KR + BT2020_KG + BT2020_KB - 1) < 1e-12);

  const grey = bt2020NclYCbCr([0.5, 0.5, 0.5]);
  assert.ok(Math.abs(grey.y - 0.5) < 1e-12);
  assert.ok(Math.abs(grey.cb) < 1e-12);
  assert.ok(Math.abs(grey.cr) < 1e-12);

  const white = bt2020NclYCbCr([1, 1, 1]);
  assert.ok(Math.abs(white.y - 1) < 1e-12);

  // Blue drives Cb positive to its rail, red drives Cr positive to its rail.
  const blue = bt2020NclYCbCr([0, 0, 1]);
  assert.ok(Math.abs(blue.cb - 0.5) < 1e-12);
  const red = bt2020NclYCbCr([1, 0, 0]);
  assert.ok(Math.abs(red.cr - 0.5) < 1e-12);
});

test('limited-range 10-bit quantisation lands on the BT.2100 code points', () => {
  assert.equal(quantizeLuma10(0), 64);
  assert.equal(quantizeLuma10(1), 940);
  assert.equal(quantizeLuma10(-1), 64);
  assert.equal(quantizeLuma10(2), 940);
  assert.equal(quantizeLuma10(0.5), 502);

  assert.equal(quantizeChroma10(0), 512);
  assert.equal(quantizeChroma10(0.5), 960);
  assert.equal(quantizeChroma10(-0.5), 64);
  assert.equal(quantizeChroma10(-2), 64);
});

test('P010 containers keep the ten bits in the high end of the word', () => {
  assert.equal(p010Sample(0), 0);
  assert.equal(p010Sample(64), 64 << 6);
  assert.equal(p010Sample(1023), 0xffc0);
  // Nothing may spill into the neighbouring sample of a packed 32-bit word.
  assert.ok(p010Sample(1023) <= 0xffff);
});

test('the P010 frame layout is exactly three bytes per pixel, in two planes', () => {
  const layout = p010Layout(3840, 2160);
  assert.equal(layout.lumaWordsPerRow, 1920);
  assert.equal(layout.lumaWords, 1920 * 2160);
  assert.equal(layout.chromaWordsPerRow, 1920);
  assert.equal(layout.chromaWords, 1920 * 1080);
  assert.equal(layout.chromaWordOffset, layout.lumaWords);
  assert.equal(layout.byteLength, 3840 * 2160 * 3);
  // The transport claim in the proposal, in bytes: 24.9 MB, not 66.4 MB.
  assert.ok(Math.abs(layout.byteLength / 1e6 - 24.88) < 0.1);

  const hd = p010Layout(1920, 1080);
  assert.equal(hd.byteLength, 1920 * 1080 * 3);

  assert.throws(() => p010Layout(1921, 1080), /even integer/);
  assert.throws(() => p010Layout(1920, 1081), /even integer/);
});

test('headroom is the peak-to-paper-white ratio and never inverts the curve', () => {
  assert.ok(Math.abs(hdrHeadroom(203, 1000) - 1000 / 203) < 1e-12);
  assert.equal(hdrHeadroom(203, 203), 1);
  // A peak below paper white would flip the roll-off; it is floored instead.
  assert.equal(hdrHeadroom(1000, 100), 1);
});

test('the HDR luminance policy is bounded and ordered before it reaches a shader', () => {
  assert.deepEqual(
    validateHdrOutputSettings({
      paperWhiteNits: DEFAULT_PAPER_WHITE_NITS,
      masteringPeakNits: DEFAULT_MASTERING_PEAK_NITS,
    }),
    { paperWhiteNits: 203, masteringPeakNits: 1000 },
  );
  assert.throws(
    () => validateHdrOutputSettings({ paperWhiteNits: 10, masteringPeakNits: 1000 }),
    /paperWhiteNits/,
  );
  assert.throws(
    () => validateHdrOutputSettings({ paperWhiteNits: 203, masteringPeakNits: 50_000 }),
    /masteringPeakNits/,
  );
  assert.throws(
    () => validateHdrOutputSettings({ paperWhiteNits: 600, masteringPeakNits: 500 }),
    /at least/,
  );
});

test('mastering-display metadata is BT.2020/D65 and claims only the chosen peak', () => {
  const m = masteringDisplayMetadata(1000);
  assert.deepEqual(
    [m.redX, m.redY, m.greenX, m.greenY, m.blueX, m.blueY],
    [35_400, 14_600, 8_500, 39_850, 6_550, 2_300],
  );
  assert.deepEqual([m.whiteX, m.whiteY], [15_635, 16_450]);
  assert.equal(m.maxLuminance, 10_000_000);
  assert.equal(m.minLuminance, 1);
  assert.equal(masteringDisplayMetadata(4000).maxLuminance, 40_000_000);

  // Chromaticities are stored in 0.00002 units; check they decode to BT.2020.
  assert.ok(Math.abs(m.redX / 50_000 - 0.708) < 1e-6);
  assert.ok(Math.abs(m.greenY / 50_000 - 0.797) < 1e-6);
  assert.ok(Math.abs(m.whiteX / 50_000 - 0.3127) < 1e-6);
  assert.ok(Math.abs(m.whiteY / 50_000 - 0.329) < 1e-6);
});
