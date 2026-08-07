// Prepended to every physarum shader module. Declares no bindings — helpers that
// need a texture take it as a parameter so each module owns its own @group layout.

const TAU: f32 = 6.28318530718;

struct Globals {
  gridW: u32,
  gridH: u32,
  speciesCount: u32,
  agentsPerSpecies: u32,

  seed: u32,
  simTick: u32,
  depositScale: f32,
  senseGain: f32,

  viewportW: f32,
  viewportH: f32,
  exposure: f32,
  gamma: f32,

  // section-boundary respawn: fraction of agents to re-scatter, and the key that
  // makes the choice deterministic in (seed, segment index, agent index)
  respawnFraction: f32,
  respawnKey: u32,
  // live radial splashes this tick; the splash pass is skipped entirely at 0
  splashCount: u32,
  pad0: u32,

  // soil — track-scale memory. One shared layer at trail resolution.
  soilDecay: f32,
  soilAccum: f32,
  soilDeposit: f32,
  soilSense: f32,

  // >0.5 renders the soil field instead of the trails
  soilView: f32,
  // phase 7 — compositor-only, linear rgb of the soil ember underlay
  soilTintR: f32,
  soilTintG: f32,
  soilTintB: f32,

  // strength of that underlay in output units, and the render-domain feedback lane
  soilTint: f32,
  feedbackAmount: f32,
  feedbackZoom: f32,
  // hard ceiling on one agent's deposit into one cell in one tick, after soil
  // fertility. Mirrors MAX_DEPOSIT in config.ts; the i32 atomic headroom in
  // PhysarumConfig.depositScale is derived from it.
  maxDeposit: f32,
}

// One hotspot disc from the impulse engine. Positions arrive already scaled into
// grid cells, so the shader does no unit conversion.
struct Splash {
  // x, y (cells), radius (cells), envelope 0..1
  posRadius: vec4f,
  // target species (< 0 = all), outward push in cells, swirl radians, unused
  params: vec4f,
}

struct Species {
  sensorDist: vec4f,   // p1, p2, p3, unused
  sensorAngle: vec4f,  // p1, p2, p3, unused
  rotate: vec4f,       // p1, p2, p3, unused
  moveDist: vec4f,     // p1, p2, p3, unused
  color: vec4f,        // linear rgb, intensity
  misc: vec4f,         // deposit, decay, aliveFraction, diffuseCenterWeight
}

struct Agent {
  pos: vec2f,
  heading: f32,
  species: f32,
}

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

// Jenson's intensity-adaptive form: p1 + p2 * x^p3.
fn adaptive(p: vec4f, x: f32) -> f32 {
  return p.x + p.y * pow(max(x, 1e-5), p.z);
}

fn wrapCoord(c: vec2i, g: vec2i) -> vec2i {
  return vec2i(((c.x % g.x) + g.x) % g.x, ((c.y % g.y) + g.y) % g.y);
}

fn wrapPos(p: vec2f, g: vec2f) -> vec2f {
  return p - floor(p / g) * g;
}

fn trailTexel(t: texture_2d_array<f32>, c: vec2i, g: vec2i, layer: i32) -> f32 {
  return textureLoad(t, wrapCoord(c, g), layer, 0).r;
}

// r32float is NOT filterable in core WebGPU, so no sampler is ever created and
// bilinear interpolation is done by hand. Portable and deterministic; the
// float32-filterable feature is irrelevant to this sim.
fn trailBilinear(t: texture_2d_array<f32>, p: vec2f, g: vec2i, layer: i32) -> f32 {
  let q = p - vec2f(0.5);
  let b = floor(q);
  let f = q - b;
  let i0 = vec2i(b);
  let v00 = trailTexel(t, i0, g, layer);
  let v10 = trailTexel(t, i0 + vec2i(1, 0), g, layer);
  let v01 = trailTexel(t, i0 + vec2i(0, 1), g, layer);
  let v11 = trailTexel(t, i0 + vec2i(1, 1), g, layer);
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

// Soil is a single non-array layer on the same grid as the trails, so agent and
// fragment positions index it with no rescaling. Same hand-rolled bilinear as the
// trails, for the same reason: r32float is unfilterable.
fn soilTexel(t: texture_2d<f32>, c: vec2i, g: vec2i) -> f32 {
  return textureLoad(t, wrapCoord(c, g), 0).r;
}

fn soilBilinear(t: texture_2d<f32>, p: vec2f, g: vec2i) -> f32 {
  let q = p - vec2f(0.5);
  let b = floor(q);
  let f = q - b;
  let i0 = vec2i(b);
  let v00 = soilTexel(t, i0, g);
  let v10 = soilTexel(t, i0 + vec2i(1, 0), g);
  let v01 = soilTexel(t, i0 + vec2i(0, 1), g);
  let v11 = soilTexel(t, i0 + vec2i(1, 1), g);
  return clamp(mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y), 0.0, 1.0);
}

// Tone mapping moved to the post chain in phase 7 (sim/render/shaders/grade.wgsl).
// The compositor now writes unbounded linear light into rgba16float and nothing
// in this directory clamps anything.
