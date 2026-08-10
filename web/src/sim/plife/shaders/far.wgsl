// The far-field density pyramid: a per-species density splat and the two
// Gaussian scales the force pass differences.
//
// ## Why a field at all
//
// The near lane is a pairwise sum over a (2s+1)² cell window, so its reach is
// bounded by what you are willing to pay per particle. At a quarter-screen
// radius that window holds thousands of neighbours and the sum over them is —
// this is the whole idea — a Monte-Carlo estimate of a smoothed density
// integral. Estimating an integral by sampling thousands of points, once per
// particle, when the integral itself can be computed once for the whole world on
// a 256×144 grid, is the expensive way round. So the far lane computes it
// directly: splat, blur, sample the gradient.
//
// ## The two scales, and why they are a difference
//
// `G_σ1 * ρ − G_σ2 * ρ` — a difference of Gaussians — is the smooth analogue of
// the near lane's tent. It is positive where species j is denser than its own
// broad background at scale σ2, ~0 where the field is featureless at both
// scales, and it decays smoothly rather than stopping at a cutoff. Following its
// *gradient* therefore means "drift toward (or away from) concentrations of j
// that are bigger than σ1 and smaller than σ2", with no ring and no step
// anywhere.
//
// σ1 is not a knob: it is the near lane's own cutoff (`nearStencil × cell`), so
// the two lanes meet at exactly one seam and neither the band between them nor
// an overlap has to be tuned by hand.
//
// ## Bindings
//
// One module, two pipelines, disjoint binding numbers — the same arrangement
// render.wgsl and step.wgsl already use. A layout names only what its own entry
// points statically touch.
//
//   splat  0 globals, 1 particles
//   blur   2 params, 3 sampler, 4/5 source pair, 6/7 the σ1 pair (DoG only)

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

// One blur pass. Written by `writeBlurParams` in plife.ts, one buffer per pass,
// re-written only when a scale actually moves.
//
//   dir       the axis step in UV, i.e. (1/width, 0) or (0, 1/height)
//   sigma     in texels of THIS axis; the two axes are sized separately so a
//             non-square texel (farW is rounded) stays isotropic in world units
//   stride    texels between taps. 1 until the kernel would need more than
//             MAX_HALF_TAPS taps, then it grows — see the note in plife.ts.
//   halfTaps  taps each side of centre
//   dog       1 = emit (σ1 field − this blur) instead of the blur. The last pass
//             of the chain does the subtraction so the force pass has one pair
//             of textures to sample instead of two.
struct BlurParams {
  dir: vec2f,
  sigma: f32,
  stride: f32,
  halfTaps: u32,
  dog: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(2) var<uniform> b: BlurParams;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var src0: texture_2d<f32>;
@group(0) @binding(5) var src1: texture_2d<f32>;
@group(0) @binding(6) var near0: texture_2d<f32>;
@group(0) @binding(7) var near1: texture_2d<f32>;

// ── splat ─────────────────────────────────────────────────────────────────────
//
// One point per particle, additively blended, energy-weighted. K = 8 species
// pack into two RGBA16F targets; a species index past 8 would have nowhere to go
// and is dropped rather than aliased onto another species' channel (K is 8 by
// design — see the partition note in config.ts — so this is a guard, not a case).
//
// Energy rather than 1.0 for the same reason the near lane weights the pair sum
// by `q.energy`: a particle on its way in should push proportionally to how
// present it is, and the two lanes must agree about what "present" means or a
// population swell would arrive in the two fields at different times.

struct SplatOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) sp: u32,
  @location(1) @interpolate(flat) energy: f32,
}

struct SplatTargets {
  @location(0) a: vec4f,
  @location(1) b: vec4f,
}

@vertex
fn vsSplat(@builtin(vertex_index) i: u32) -> SplatOut {
  var out: SplatOut;
  out.sp = 0u;
  out.energy = 0.0;
  // Off-screen in clip space (x > w), so the clipper drops it before any
  // fragment work. Same trick the particle pass uses for a dormant instance.
  out.pos = vec4f(2.0, 2.0, 0.0, 1.0);
  if (i >= g.maxParticles) {
    return out;
  }
  let p = particles[i];
  if (p.energy <= 0.0) {
    return out;
  }
  let world = vec2f(g.worldW, g.worldH);
  let uv = wrapWorld(p.pos, world) / world;
  out.pos = vec4f(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 0.0, 1.0);
  out.sp = speciesOf(i, g.segSize, g.speciesCount);
  out.energy = min(p.energy, 1.0);
  return out;
}

@fragment
fn fsSplat(in: SplatOut) -> SplatTargets {
  var out: SplatTargets;
  out.a = vec4f(0.0);
  out.b = vec4f(0.0);
  var v = vec4f(0.0);
  v[in.sp % 4u] = in.energy;
  if (in.sp < 4u) {
    out.a = v;
  } else if (in.sp < 8u) {
    out.b = v;
  }
  return out;
}

// ── blur ──────────────────────────────────────────────────────────────────────
//
// A plain separable Gaussian at full splat resolution, both texture pairs in one
// pass. Deliberately NOT a mip pyramid: a coarse level sampled bilinearly has a
// piecewise-constant gradient, and a piecewise-constant gradient on a field that
// is being *followed* draws exactly the rectilinear tile artefact this lane
// exists to avoid. Full resolution keeps the gradient continuous everywhere, and
// at 256×144 × 2 textures the whole chain is four passes over 37 k texels.
//
// The kernel is renormalised by its own weight sum, so truncating the tails
// preserves the field's mean — which matters, because the force pass divides by
// that mean.

struct BlurOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsBlur(@builtin(vertex_index) vi: u32) -> BlurOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let q = corners[vi];
  var out: BlurOut;
  out.pos = vec4f(q, 0.0, 1.0);
  out.uv = vec2f((q.x + 1.0) * 0.5, (1.0 - q.y) * 0.5);
  return out;
}

@fragment
fn fsBlur(in: BlurOut) -> SplatTargets {
  var a = vec4f(0.0);
  var c = vec4f(0.0);
  var wsum = 0.0;
  let inv2s2 = 1.0 / (2.0 * max(b.sigma * b.sigma, 1e-6));
  let n = i32(b.halfTaps);
  for (var k = -n; k <= n; k = k + 1) {
    let x = f32(k) * b.stride;
    let w = exp(-x * x * inv2s2);
    // The sampler is `repeat` on both axes: the world is a torus, so the field
    // has to wrap. Without it the blur would flatten toward the edges and every
    // structure crossing the seam would be torn in the far lane while the near
    // lane (which uses `wrapDelta`) saw it whole.
    let uv = in.uv + b.dir * x;
    a = a + textureSampleLevel(src0, samp, uv, 0.0) * w;
    c = c + textureSampleLevel(src1, samp, uv, 0.0) * w;
    wsum = wsum + w;
  }
  let inv = 1.0 / max(wsum, 1e-9);
  var out: SplatTargets;
  out.a = a * inv;
  out.b = c * inv;
  if (b.dog != 0u) {
    out.a = textureSampleLevel(near0, samp, in.uv, 0.0) - out.a;
    out.b = textureSampleLevel(near1, samp, in.uv, 0.0) - out.b;
  }
  return out;
}
