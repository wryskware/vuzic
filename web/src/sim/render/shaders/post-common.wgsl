// Prepended to every post-processing module. Declares no bindings; each module
// owns its own @group layout, exactly like the sim's common.wgsl.

/** Mirrors PostFx's params buffer. 24 f32 = 96 bytes, a multiple of 16. */
struct Post {
  viewport: vec2f,
  threshold: f32,
  knee: f32,

  bloomIntensity: f32,
  exposureEv: f32,
  tonemap: f32,
  blackPoint: f32,

  contrast: f32,
  pivot: f32,
  saturation: f32,
  vignette: f32,

  gamma: f32,
  autoEnabled: f32,
  autoTarget: f32,
  autoTau: f32,

  autoMin: f32,
  autoMax: f32,
  dt: f32,
  pad0: f32,

  // Zero in every browser preview; only the HDR10 export grade reads these.
  hdrPaperWhite: f32,
  hdrPeak: f32,
  hdrHeadroom: f32,
  pad1: f32,
}

struct PostVSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

// Same fullscreen triangle and same uv convention as the sim's compositor, so a
// pass can sample its own input at `in.uv` and get the identity mapping.
@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> PostVSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let q = corners[vi];
  var out: PostVSOut;
  out.pos = vec4f(q, 0.0, 1.0);
  out.uv = vec2f((q.x + 1.0) * 0.5, (1.0 - q.y) * 0.5);
  return out;
}

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Luminance-only Reinhard: divide by 1 + L, leave the channel ratios alone.
// Highlights keep their hue instead of bleaching, which is what a static
// per-species palette needs — a saturated species must still be identifiable
// inside the brightest part of its own filament. A single channel may land above
// 1 and clip there; that reads as a hot saturated core, not as white.
fn tonemapReinhardLum(x: vec3f) -> vec3f {
  return x / (1.0 + luma(x));
}

// Reinhard–Jodie: the same curve blended toward the per-channel form as the
// pixel gets brighter, so nothing can exceed 1 at all. Cleaner highlights, but
// every hot core converges on white.
fn tonemapReinhardJodie(x: vec3f) -> vec3f {
  let l = luma(x);
  let tv = x / (1.0 + x);
  return mix(x / (1.0 + l), tv, tv);
}

// Narkowicz ACES fit — the phase-4 curve, kept so the two can be compared.
fn tonemapAces(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

// Order must match TONEMAPS in sim/render/config.ts.
fn tonemapSelect(x: vec3f, which: f32) -> vec3f {
  let c = max(x, vec3f(0.0));
  if (which < 0.5) {
    return tonemapReinhardLum(c);
  }
  if (which < 1.5) {
    return tonemapReinhardJodie(c);
  }
  if (which < 2.5) {
    return tonemapAces(c);
  }
  return clamp(c, vec3f(0.0), vec3f(1.0));
}
