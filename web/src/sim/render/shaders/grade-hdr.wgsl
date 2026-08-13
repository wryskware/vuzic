// The HDR10 sibling of grade.wgsl: the only other pass in the app that produces
// display-referred pixels, and the one the export path uses instead of — never
// after — the 8-bit SDR grade.
//
// It reads the same unclamped scene-linear HDR surface and the same bloom
// pyramid, applies the same authored creative chain, and then diverges at the
// display transform:
//
//   grade.wgsl      tonemap to 0..1, clamp, gamma-encode for an 8-bit swapchain
//   grade-hdr.wgsl  headroom-scaled tonemap, no clamp to diffuse white, output
//                   linear BT.2020 luminance normalised to the ST 2084 domain
//
// The display transform is `H * tonemap(x / H)` where `H = peak / paperWhite`.
// That family has three properties worth stating: it is unit-sloped at the
// origin, so shadows and mid-tones land exactly where the author put them; it
// asymptotes to the mastering peak instead of to diffuse white, so highlights
// use the headroom the display actually has; and at H = 1 it is bit-for-bit the
// same curve grade.wgsl applies. The creative trims after it run in BT.2020
// display space with BT.2020 luma weights, in the same order as the SDR grade.
//
// Output is *linear* BT.2020 in units of nits / 10000 — the ST 2084 input
// domain. The PQ encode, the RGB→Y'CbCr matrix, the 4:2:0 decimation and the
// 10-bit packing all happen in the export packing compute pass, which is where
// the standards constants live and where they are tested.

@group(0) @binding(0) var<uniform> p: Post;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var hdrTex: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;

// ITU-R BT.2087 linear BT.709 → BT.2020. mat3x3f takes columns, so this is the
// transpose of the row-major matrix in export/hdr.ts.
const BT709_TO_BT2020: mat3x3f = mat3x3f(
  vec3f(0.6274039, 0.0690973, 0.0163914),
  vec3f(0.3292830, 0.9195404, 0.0880132),
  vec3f(0.0433131, 0.0113623, 0.8955953),
);

const PQ_MAX_NITS: f32 = 10000.0;

// BT.2020 non-constant-luminance luma weights, for the trims that need one.
fn luma2020(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2627, 0.6780, 0.0593));
}

// Mirrors post-common.wgsl's tonemap family, with BT.2020 luma. Kept separate
// rather than parameterised so the SDR path stays byte-identical to what it was.
fn tonemapReinhardLum2020(x: vec3f) -> vec3f {
  return x / (1.0 + luma2020(x));
}

fn tonemapReinhardJodie2020(x: vec3f) -> vec3f {
  let l = luma2020(x);
  let tv = x / (1.0 + x);
  return mix(x / (1.0 + l), tv, tv);
}

fn tonemapSelect2020(x: vec3f, which: f32) -> vec3f {
  let c = max(x, vec3f(0.0));
  if (which < 0.5) {
    return tonemapReinhardLum2020(c);
  }
  if (which < 1.5) {
    return tonemapReinhardJodie2020(c);
  }
  if (which < 2.5) {
    return tonemapAces(c);
  }
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

/** Authored display curve, stretched to end at the mastering peak. */
fn hdrDisplayMap(x: vec3f, which: f32, headroom: f32) -> vec3f {
  let h = max(headroom, 1.0);
  return h * tonemapSelect2020(max(x, vec3f(0.0)) / h, which);
}

@fragment
fn fsMain(in: PostVSOut) -> @location(0) vec4f {
  var c = textureSampleLevel(hdrTex, samp, in.uv, 0.0).rgb;

  // Bloom joins before the display transform for the same reason it does in
  // SDR: a flash should read as a splash of light, not as a plate of peak white.
  c = c + textureSampleLevel(bloomTex, samp, in.uv, 0.0).rgb * p.bloomIntensity;
  c = max(c, vec3f(0.0));

  // Scene-linear source primaries (BT.709, matching the palette decode) to
  // BT.2020, still linear and still unbounded.
  c = BT709_TO_BT2020 * c;

  // 1.0 now means diffuse white; p.hdrHeadroom means the mastering peak.
  c = hdrDisplayMap(c, p.tonemap, p.hdrHeadroom);

  let black = clamp(p.blackPoint, 0.0, 0.9);
  c = max(c - vec3f(black), vec3f(0.0)) / (1.0 - black);
  c = max((c - vec3f(p.pivot)) * p.contrast + vec3f(p.pivot), vec3f(0.0));

  let l = luma2020(c);
  c = max(mix(vec3f(l), c, p.saturation), vec3f(0.0));

  let aspect = max(p.viewport.x, 1.0) / max(p.viewport.y, 1.0);
  let d = length((in.uv - vec2f(0.5)) * vec2f(aspect, 1.0));
  c = c * mix(1.0, smoothstep(1.15, 0.30, d), clamp(p.vignette, 0.0, 1.0));

  // No display gamma here: the transfer function is PQ and it is applied by the
  // packing pass. Absolute luminance, clamped to the mastering peak the roll-off
  // and the ST 2086 metadata both assume, normalised to the ST 2084 domain.
  let nits = clamp(c * p.hdrPaperWhite, vec3f(0.0), vec3f(p.hdrPeak));
  return vec4f(nits / PQ_MAX_NITS, 1.0);
}
