// Prepended to every vizfx module, ahead of the visual's generated θ prelude.
//
// Unlike physarum's and plife's `common.wgsl` — which declare no bindings, so
// each module owns its own @group layout — this one declares the bindings and
// every pass shares them. That is a consequence of the substrate rather than a
// style difference: the four passes (warp, draw, inject, composite) all read the
// same five resources and differ only in what they compute from them, so four
// layouts would be four copies of one list. WebGPU validates a pipeline layout
// against the bindings an entry point *statically uses*, so a pass that never
// touches `splashes` costs nothing for having it in scope.
//
// The hash is the same PCG the other two substrates use and that `hash3` in
// sim/impulses.ts mirrors on the CPU. That is not incidental: emitter placement
// is drawn on the CPU from the same stream, and "seeded" has to mean one thing.

const TAU: f32 = 6.28318530718;

// ── Globals ───────────────────────────────────────────────────────────────────
//
// 16 words = 64 bytes. The TS writer is `writeGlobals` in vizfx.ts and indexes
// this block positionally, so the word offsets are documented and nothing but
// this comment checks that the two agree.
//
//   0 res.x      4 time        8  layerCount    12 exposure
//   1 res.y      5 dt          9  emitPerLayer  13 injectAmount
//   2 invRes.x   6 tick        10 splashCount   14 injectKey
//   3 invRes.y   7 seed        11 pulse         15 pad0
struct Globals {
  /** field size in texels; the visual's coordinates are derived from its aspect */
  res: vec2f,
  invRes: vec2f,

  /** seconds of SIM time (model step / 60), never app-tick or wall-clock time */
  time: f32,
  /** seconds one warp+draw step advances the field. Fixed 1/60. */
  dt: f32,
  tick: u32,
  seed: u32,

  layerCount: u32,
  emitPerLayer: u32,
  /** live entries in the splash buffer; 0 skips the ring loop entirely */
  splashCount: u32,
  /**
   * The event lane's whole-field transient, 0 at rest: max over layers of
   * (depositMul - 1). The warp reads it, which is what makes a kick move the
   * geometry and not merely the brightness.
   */
  pulse: f32,

  /** scene exposure, applied in the COMPOSITE pass — see the note there */
  exposure: f32,
  /** section-boundary injection: how much seeded light to stamp, 0 = idle */
  injectAmount: f32,
  /** key that makes the injection deterministic in (seed, segment index) */
  injectKey: u32,
  pad0: u32,
}

/**
 * One layer's *composed* runtime state — everything that is not θ.
 *
 *   color.rgb  premultiplied linear rgb: palette x brightness x stem-follow
 *              x impulse flash. The θ gains stay in `theta` and are applied in
 *              the shader, because the macro rig has already been folded into
 *              those and folding them in twice would square it.
 *   color.a    presence 0..1 from the energy lane (config.energy)
 *   mods.x     impulse depositMul — how hard this layer is being struck
 *   mods.y     impulse sensorMul  — how far it is reaching, i.e. emitter swell
 */
struct Layer {
  color: vec4f,
  mods: vec4f,
}

/**
 * One live shockwave ring.
 *
 *   posRadius  centre x, centre y (0..1 screen), base radius, envelope 0..1
 *   tint       linear rgb of the struck layer; .a unused
 *   params     expansion rate (from ResponseConfig.splashPush), ring wobble
 *              (from splashSwirl), two spare
 *
 * The tint is composed on the CPU rather than looked up from `layers` here,
 * because a splash may target *all* species (`ResponseConfig.species < 0`), and
 * "the mean of the palette" is a CPU-side answer to a CPU-side question.
 */
struct Splash {
  posRadius: vec4f,
  tint: vec4f,
  params: vec4f,
}

@group(0) @binding(0) var<uniform> g: Globals;
/** θ, macro-applied and hard-clamped by the CPU. Addressed via the generated `TH_*`. */
@group(0) @binding(1) var<storage, read> theta: array<f32>;
@group(0) @binding(2) var<storage, read> layers: array<Layer>;
/** K x emitPerLayer: (phase, radius x, speed x, size x). Redrawn only on reseed. */
@group(0) @binding(3) var<storage, read> emitters: array<vec4f>;
@group(0) @binding(4) var<storage, read> splashes: array<Splash>;
@group(0) @binding(5) var fieldSamp: sampler;
/** the PREVIOUS field for warp/inject, the CURRENT one for composite — see `vizfx.ts` */
@group(0) @binding(6) var field: texture_2d<f32>;
/** [gain, mean, initialised, pad] from the post chain's measure pass, one frame old */
@group(0) @binding(7) var<storage, read> autoState: array<f32>;

fn pcg(input: u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
  return pcg(a ^ pcg(b ^ pcg(c)));
}

fn rand01(h: u32) -> f32 {
  return f32(h) * 2.3283064365386963e-10;
}

// ── the visual's coordinate system ────────────────────────────────────────────
//
// Centred and aspect-corrected: y runs -0.5 .. 0.5 whatever the window is, and x
// runs the same distance per pixel. Every radius in θ is therefore a fraction of
// the screen HEIGHT and means the same thing on any canvas and in a 1/3-size
// explorer tile — the same property plife's normalised world space buys, for the
// same reason.

fn aspect() -> vec2f {
  return vec2f(max(g.res.x, 1.0) / max(g.res.y, 1.0), 1.0);
}

fn toCentre(uv: vec2f) -> vec2f {
  return (uv - vec2f(0.5)) * aspect();
}

fn toUv(p: vec2f) -> vec2f {
  return p / aspect() + vec2f(0.5);
}

// ── value noise ───────────────────────────────────────────────────────────────
//
// A hashed lattice with smoothstep interpolation, and two octaves of it. Same
// construction as plife's wander noise and for the same reason: it has to be
// *spatially coherent*, so that the filament term breaks an emitter into strands
// rather than into per-pixel grain (which the bloom would then turn into a haze).

fn latticeValue(ix: i32, iy: i32, seed: u32) -> f32 {
  return rand01(hash3(bitcast<u32>(ix), bitcast<u32>(iy), seed));
}

fn noise2(p: vec2f, seed: u32) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let ix = i32(i.x);
  let iy = i32(i.y);
  let a = latticeValue(ix, iy, seed);
  let b = latticeValue(ix + 1, iy, seed);
  let c = latticeValue(ix, iy + 1, seed);
  let d = latticeValue(ix + 1, iy + 1, seed);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/** Two octaves, normalised to 0..1. Three was measured as no more structure per ms. */
fn fbm2(p: vec2f, seed: u32) -> f32 {
  return (noise2(p, seed) * 0.667 + noise2(p * 2.17 + vec2f(11.3, -7.1), seed ^ 0x9e3779b9u) * 0.333);
}

// ── colour ────────────────────────────────────────────────────────────────────

/**
 * Rotate a linear rgb triple about the (1,1,1) axis by `a` radians.
 *
 * A rotation rather than an HSV round trip: it is six multiplies, it preserves
 * luminance exactly (the axis IS the grey line), and — the property the chroma
 * lane depends on — it composes, so applying a small angle every tick for a
 * minute is one large rotation of the accumulated field rather than sixty
 * quantised re-quantisations of it.
 */
fn hueRotate(c: vec3f, a: f32) -> vec3f {
  let k = 0.57735027; // 1/sqrt(3)
  let cs = cos(a);
  let sn = sin(a);
  return c * cs + cross(vec3f(k, k, k), c) * sn + vec3f(k, k, k) * dot(vec3f(k, k, k), c) * (1.0 - cs);
}

// ── the fullscreen triangle ───────────────────────────────────────────────────
//
// One triangle, not two: no diagonal seam, and the rasteriser skips the half of
// it that is off-screen. Same construction as the post chain's.

struct FsIn {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> FsIn {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let q = corners[vi];
  var out: FsIn;
  out.pos = vec4f(q, 0.0, 1.0);
  out.uv = vec2f((q.x + 1.0) * 0.5, (1.0 - q.y) * 0.5);
  return out;
}
