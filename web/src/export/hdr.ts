/**
 * The colour standards the HDR10 export path is built on.
 *
 * Everything here is pure arithmetic with no GPU or FFmpeg dependency, so the
 * numbers that end up in a shader, in an encoder argument, and in a container
 * metadata box are all derived from one tested source. `shaders/hdr-pack.wgsl`
 * mirrors the transfer/matrix constants; `hdr-shader-parity` in the test suite
 * is what keeps the two copies honest.
 *
 * Conventions used throughout:
 *
 * - *scene linear* values are unbounded and have no display meaning;
 * - after the HDR grade, `1.0` means diffuse white (paper white) and `headroom`
 *   means the mastering peak;
 * - `nits` are absolute cd/m²;
 * - PQ code values are normalised to 0..1, where 1.0 is 10,000 nits.
 */

/** ITU-R BT.2100 reference diffuse white for graphics over HDR video. */
export const DEFAULT_PAPER_WHITE_NITS = 203;
/** Mastering-display peak the highlight roll-off and ST 2086 metadata assume. */
export const DEFAULT_MASTERING_PEAK_NITS = 1000;

export const MIN_PAPER_WHITE_NITS = 80;
export const MAX_PAPER_WHITE_NITS = 1000;
export const MIN_MASTERING_PEAK_NITS = 100;
export const MAX_MASTERING_PEAK_NITS = 10_000;

/** SMPTE ST 2084 peak. A PQ code value of 1.0 is this many nits. */
export const PQ_MAX_NITS = 10_000;

// SMPTE ST 2084 constants, in their exact rational form.
export const PQ_M1 = 2610 / 16384;
export const PQ_M2 = (2523 / 4096) * 128;
export const PQ_C1 = 3424 / 4096;
export const PQ_C2 = (2413 / 4096) * 32;
export const PQ_C3 = (2392 / 4096) * 32;

/**
 * ST 2084 inverse EOTF: absolute luminance in nits to a 0..1 PQ code value.
 *
 * Input above 10,000 nits is clamped rather than extrapolated; no display
 * consumes a PQ code above 1.0 and the encoder cannot carry one.
 */
export function pqEncodeNits(nits: number): number {
  if (!Number.isFinite(nits) || nits <= 0) return 0;
  const y = Math.min(nits / PQ_MAX_NITS, 1);
  const ym = Math.pow(y, PQ_M1);
  return Math.pow((PQ_C1 + PQ_C2 * ym) / (1 + PQ_C3 * ym), PQ_M2);
}

/** ST 2084 EOTF: a 0..1 PQ code value back to absolute nits. */
export function pqDecodeNits(code: number): number {
  if (!Number.isFinite(code) || code <= 0) return 0;
  const n = Math.pow(Math.min(code, 1), 1 / PQ_M2);
  const numerator = Math.max(n - PQ_C1, 0);
  const denominator = PQ_C2 - PQ_C3 * n;
  if (denominator <= 0) return PQ_MAX_NITS;
  return Math.pow(numerator / denominator, 1 / PQ_M1) * PQ_MAX_NITS;
}

/**
 * Linear BT.709 to linear BT.2020, the ITU-R BT.2087 normative matrix.
 *
 * Row-major here; the WGSL copy is transposed into column-major because that is
 * how `mat3x3f` is constructed.
 */
export const BT709_TO_BT2020: readonly (readonly [number, number, number])[] = [
  [0.6274039, 0.3292830, 0.0433131],
  [0.0690973, 0.9195404, 0.0113623],
  [0.0163914, 0.0880132, 0.8955953],
];

export function bt709ToBt2020(rgb: readonly [number, number, number]): [number, number, number] {
  const out = BT709_TO_BT2020.map((row) => row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]);
  return [out[0] as number, out[1] as number, out[2] as number];
}

// BT.2020 non-constant-luminance luma coefficients.
export const BT2020_KR = 0.2627;
export const BT2020_KB = 0.0593;
export const BT2020_KG = 1 - BT2020_KR - BT2020_KB;

export interface YCbCr {
  /** 0..1 before quantisation. */
  readonly y: number;
  /** -0.5..0.5 before quantisation. */
  readonly cb: number;
  readonly cr: number;
}

/**
 * BT.2020 non-constant luminance RGB'→Y'CbCr.
 *
 * The input is *already* transfer-encoded (PQ here). Non-constant luminance
 * means the matrix is applied to the non-linear signal, which is what BT.2100
 * specifies for PQ Y'CbCr and what every HDR10 decoder expects.
 */
export function bt2020NclYCbCr(rgbPrime: readonly [number, number, number]): YCbCr {
  const y = BT2020_KR * rgbPrime[0] + BT2020_KG * rgbPrime[1] + BT2020_KB * rgbPrime[2];
  return {
    y,
    cb: (rgbPrime[2] - y) / (2 * (1 - BT2020_KB)),
    cr: (rgbPrime[0] - y) / (2 * (1 - BT2020_KR)),
  };
}

// Limited ("television") range 10-bit quantisation, BT.2100 Table 9.
export const LIMITED_LUMA_OFFSET_10 = 64;
export const LIMITED_LUMA_SCALE_10 = 876;
export const LIMITED_CHROMA_OFFSET_10 = 512;
export const LIMITED_CHROMA_SCALE_10 = 896;
export const LIMITED_LUMA_MIN_10 = 64;
export const LIMITED_LUMA_MAX_10 = 940;
export const LIMITED_CHROMA_MIN_10 = 64;
export const LIMITED_CHROMA_MAX_10 = 960;

export function quantizeLuma10(y: number): number {
  const code = Math.round(y * LIMITED_LUMA_SCALE_10 + LIMITED_LUMA_OFFSET_10);
  return Math.min(Math.max(code, LIMITED_LUMA_MIN_10), LIMITED_LUMA_MAX_10);
}

export function quantizeChroma10(c: number): number {
  const code = Math.round(c * LIMITED_CHROMA_SCALE_10 + LIMITED_CHROMA_OFFSET_10);
  return Math.min(Math.max(code, LIMITED_CHROMA_MIN_10), LIMITED_CHROMA_MAX_10);
}

/**
 * Highlight headroom above diffuse white, in stops-agnostic linear ratio.
 *
 * The HDR display transform is `headroom * tonemap(x / headroom)`: identical to
 * the SDR grade when the headroom is 1, unity-sloped at the bottom so shadows
 * and mid-tones land exactly where the author put them, and asymptotic to the
 * mastering peak instead of to diffuse white.
 */
export function hdrHeadroom(paperWhiteNits: number, masteringPeakNits: number): number {
  return Math.max(masteringPeakNits / paperWhiteNits, 1);
}

export interface HdrOutputSettings {
  readonly paperWhiteNits: number;
  readonly masteringPeakNits: number;
}

export function validateHdrOutputSettings(settings: HdrOutputSettings): HdrOutputSettings {
  const { paperWhiteNits, masteringPeakNits } = settings;
  if (
    !Number.isFinite(paperWhiteNits) ||
    paperWhiteNits < MIN_PAPER_WHITE_NITS ||
    paperWhiteNits > MAX_PAPER_WHITE_NITS
  ) {
    throw new RangeError(
      `paperWhiteNits must be in [${MIN_PAPER_WHITE_NITS}, ${MAX_PAPER_WHITE_NITS}]`,
    );
  }
  if (
    !Number.isFinite(masteringPeakNits) ||
    masteringPeakNits < MIN_MASTERING_PEAK_NITS ||
    masteringPeakNits > MAX_MASTERING_PEAK_NITS
  ) {
    throw new RangeError(
      `masteringPeakNits must be in [${MIN_MASTERING_PEAK_NITS}, ${MAX_MASTERING_PEAK_NITS}]`,
    );
  }
  if (masteringPeakNits < paperWhiteNits) {
    throw new RangeError('masteringPeakNits must be at least paperWhiteNits');
  }
  return { paperWhiteNits, masteringPeakNits };
}

/**
 * SMPTE ST 2086 static mastering-display colour volume.
 *
 * Chromaticities are BT.2020 primaries with a D65 white point in the 0.00002
 * units the metadata box uses; luminance is in 0.0001 cd/m². The values describe
 * the display this render was *graded for*, which is exactly the policy the
 * highlight roll-off was parameterised with — nothing here is measured from
 * content, and no MaxCLL/MaxFALL is claimed anywhere in the pipeline.
 */
export interface MasteringDisplayMetadata {
  readonly redX: number;
  readonly redY: number;
  readonly greenX: number;
  readonly greenY: number;
  readonly blueX: number;
  readonly blueY: number;
  readonly whiteX: number;
  readonly whiteY: number;
  /** 0.0001 cd/m² units. */
  readonly maxLuminance: number;
  readonly minLuminance: number;
}

export const CHROMATICITY_DENOMINATOR = 50_000;
export const LUMINANCE_DENOMINATOR = 10_000;
/** 0.0001 cd/m²: the smallest representable non-zero black, i.e. "reference OLED". */
export const MASTERING_MIN_LUMINANCE_UNITS = 1;

export function masteringDisplayMetadata(masteringPeakNits: number): MasteringDisplayMetadata {
  if (
    !Number.isFinite(masteringPeakNits) ||
    masteringPeakNits < MIN_MASTERING_PEAK_NITS ||
    masteringPeakNits > MAX_MASTERING_PEAK_NITS
  ) {
    throw new RangeError('masteringPeakNits is outside the supported range');
  }
  return {
    redX: 35_400,
    redY: 14_600,
    greenX: 8_500,
    greenY: 39_850,
    blueX: 6_550,
    blueY: 2_300,
    whiteX: 15_635,
    whiteY: 16_450,
    maxLuminance: Math.round(masteringPeakNits * LUMINANCE_DENOMINATOR),
    minLuminance: MASTERING_MIN_LUMINANCE_UNITS,
  };
}

export interface P010Layout {
  readonly width: number;
  readonly height: number;
  /** 32-bit words per luma row; each word holds two 16-bit samples. */
  readonly lumaWordsPerRow: number;
  readonly lumaWords: number;
  /** 32-bit words per chroma row; each word holds one interleaved Cb/Cr pair. */
  readonly chromaWordsPerRow: number;
  readonly chromaWords: number;
  /** Word index where the interleaved CbCr plane begins. */
  readonly chromaWordOffset: number;
  readonly totalWords: number;
  readonly byteLength: number;
}

/**
 * Byte layout of one tightly packed P010LE frame.
 *
 * P010 is 4:2:0 with 16-bit containers holding 10-bit samples in the *most*
 * significant bits: a full-height luma plane followed by a half-height plane of
 * interleaved Cb/Cr pairs. FFmpeg's `p010le` raw pixel format expects exactly
 * this, and NVENC consumes it without a conversion filter.
 */
export function p010Layout(width: number, height: number): P010Layout {
  if (!Number.isSafeInteger(width) || width < 2 || width % 2 !== 0) {
    throw new RangeError('P010 width must be an even integer of at least 2');
  }
  if (!Number.isSafeInteger(height) || height < 2 || height % 2 !== 0) {
    throw new RangeError('P010 height must be an even integer of at least 2');
  }
  const lumaWordsPerRow = width / 2;
  const lumaWords = lumaWordsPerRow * height;
  const chromaWordsPerRow = width / 2;
  const chromaWords = chromaWordsPerRow * (height / 2);
  const totalWords = lumaWords + chromaWords;
  return {
    width,
    height,
    lumaWordsPerRow,
    lumaWords,
    chromaWordsPerRow,
    chromaWords,
    chromaWordOffset: lumaWords,
    totalWords,
    byteLength: totalWords * 4,
  };
}

/** 10-bit sample to its P010 16-bit container: data in the high bits, zero pad. */
export function p010Sample(code10: number): number {
  return (code10 & 0x3ff) << 6;
}
