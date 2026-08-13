// Scene-referred HDR frame → encoder-ready P010LE, on the GPU.
//
// This is the transport decision the proposal asks for. Reading an RGBA16F 4K
// frame back is 66.4 MB/frame; reading back the P010 the encoder actually wants
// is 24.9 MB. The conversion is per-pixel and embarrassingly parallel, so it
// belongs on the device that already holds the pixels.
//
// One invocation owns one 2x2 chroma block: it encodes four luma samples and the
// single co-sited Cb/Cr pair that covers them. Chroma is averaged *after* the PQ
// transfer and the matrix, which is what non-constant-luminance 4:2:0 means.
//
// Every constant below is mirrored by export/hdr.ts and checked against it by
// the shader-parity test.

struct PackParams {
  width: u32,
  height: u32,
  lumaWordsPerRow: u32,
  chromaWordsPerRow: u32,
  chromaWordOffset: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> packed: array<u32>;

// SMPTE ST 2084 (PQ).
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

// BT.2020 non-constant luminance.
const BT2020_KR: f32 = 0.2627;
const BT2020_KG: f32 = 0.6780;
const BT2020_KB: f32 = 0.0593;

// Limited-range 10-bit quantisation, BT.2100.
const LUMA_OFFSET_10: f32 = 64.0;
const LUMA_SCALE_10: f32 = 876.0;
const CHROMA_OFFSET_10: f32 = 512.0;
const CHROMA_SCALE_10: f32 = 896.0;
const LUMA_MIN_10: f32 = 64.0;
const LUMA_MAX_10: f32 = 940.0;
const CHROMA_MIN_10: f32 = 64.0;
const CHROMA_MAX_10: f32 = 960.0;

/** ST 2084 inverse EOTF. Input is luminance normalised so 1.0 is 10,000 nits. */
fn pqEncode(value: vec3f) -> vec3f {
  let y = clamp(value, vec3f(0.0), vec3f(1.0));
  let ym = pow(y, vec3f(PQ_M1));
  return pow((vec3f(PQ_C1) + PQ_C2 * ym) / (vec3f(1.0) + PQ_C3 * ym), vec3f(PQ_M2));
}

fn loadPq(x: u32, y: u32) -> vec3f {
  let texel = textureLoad(source, vec2u(min(x, params.width - 1u), min(y, params.height - 1u)), 0);
  return pqEncode(texel.rgb);
}

fn lumaOf(rgbPrime: vec3f) -> f32 {
  return BT2020_KR * rgbPrime.r + BT2020_KG * rgbPrime.g + BT2020_KB * rgbPrime.b;
}

fn chromaOf(rgbPrime: vec3f, y: f32) -> vec2f {
  return vec2f(
    (rgbPrime.b - y) / (2.0 * (1.0 - BT2020_KB)),
    (rgbPrime.r - y) / (2.0 * (1.0 - BT2020_KR)),
  );
}

fn quantizeLuma(y: f32) -> u32 {
  let code = round(y * LUMA_SCALE_10 + LUMA_OFFSET_10);
  return u32(clamp(code, LUMA_MIN_10, LUMA_MAX_10));
}

fn quantizeChroma(c: f32) -> u32 {
  let code = round(c * CHROMA_SCALE_10 + CHROMA_OFFSET_10);
  return u32(clamp(code, CHROMA_MIN_10, CHROMA_MAX_10));
}

/** 10-bit sample into its P010 16-bit container: most significant bits, zero pad. */
fn container(code: u32) -> u32 {
  return (code & 0x3ffu) << 6u;
}

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) id: vec3u) {
  let blocksX = params.width / 2u;
  let blocksY = params.height / 2u;
  if (id.x >= blocksX || id.y >= blocksY) {
    return;
  }

  let x0 = id.x * 2u;
  let y0 = id.y * 2u;

  var lumas: array<f32, 4>;
  var chromaSum = vec2f(0.0);
  var index = 0u;
  for (var dy = 0u; dy < 2u; dy = dy + 1u) {
    for (var dx = 0u; dx < 2u; dx = dx + 1u) {
      let rgbPrime = loadPq(x0 + dx, y0 + dy);
      let y = lumaOf(rgbPrime);
      lumas[index] = y;
      chromaSum = chromaSum + chromaOf(rgbPrime, y);
      index = index + 1u;
    }
  }

  let row0 = y0 * params.lumaWordsPerRow + id.x;
  let row1 = (y0 + 1u) * params.lumaWordsPerRow + id.x;
  packed[row0] = container(quantizeLuma(lumas[0])) | (container(quantizeLuma(lumas[1])) << 16u);
  packed[row1] = container(quantizeLuma(lumas[2])) | (container(quantizeLuma(lumas[3])) << 16u);

  let chroma = chromaSum * 0.25;
  let chromaWord = params.chromaWordOffset + id.y * params.chromaWordsPerRow + id.x;
  packed[chromaWord] =
    container(quantizeChroma(chroma.x)) | (container(quantizeChroma(chroma.y)) << 16u);
}
