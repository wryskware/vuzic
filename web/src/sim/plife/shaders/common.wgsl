// Prepended to every plife shader module. Declares no bindings — helpers that
// need a resource take it as a parameter, so each module owns its own @group
// layout.
//
// This file is a deliberate near-copy of physarum's common.wgsl rather than a
// shared include. The two sims agree on the *hash* (they must: the CPU-side
// `hash3` in sim/impulses.ts mirrors it and both sims key their determinism off
// it) and on nothing else — the struct layouts, the wrap helpers and the units
// are all different. Sharing the file would mean every plife change had to be
// checked against physarum's compile.

const TAU: f32 = 6.28318530718;

// ── Globals ───────────────────────────────────────────────────────────────────
//
// 28 words = 112 bytes = 7 × 16. The three `pad` words are explicit rather than
// implicit: the uniform address space would round 25 words up to 112 bytes
// anyway, and an unnamed hole is exactly the kind of thing the TS writer drifts
// into. Word offsets are documented because that writer (`writeGlobals` in
// plife.ts, GLOBALS_WORDS = 28) indexes this block positionally and nothing
// checks the two agree.
//
//   0 worldW         5 cellH          10 tick             15 respawnKey
//   1 worldH         6 speciesCount   11 dt               16 maxSpeed
//   2 gridW          7 segSize        12 forceGain        17 activeTotal
//   3 gridH          8 maxParticles   13 exposure         18 feedbackAmount
//   4 cellW          9 seed           14 respawnFraction  19 feedbackZoom
//
//  20 riseTau       21 fallTau        22 splashCount      23 nearStencil
//  24 farOn         25 pad0           26 pad1             27 pad2
struct Globals {
  // World space is toroidal, h = 1 and w = aspect, fixed for the sim's life.
  worldW: f32,
  worldH: f32,
  // Neighbour grid. cellW/cellH are world/grid on each axis and are guaranteed
  // >= R_CAP; the (2s+1)^2 search in step.wgsl then finds every pair within
  // s * min(cellW, cellH), which is exactly where reach is clamped.
  gridW: u32,
  gridH: u32,

  cellW: f32,
  cellH: f32,
  speciesCount: u32,
  // Particles per species segment. Species is *derived* from the index, never
  // stored: species(i) = min(i / segSize, speciesCount - 1).
  segSize: u32,

  maxParticles: u32,
  seed: u32,
  tick: u32,
  dt: f32,

  forceGain: f32,
  exposure: f32,
  // section-boundary respawn: fraction of the pool to re-scatter, and the key
  // that makes the choice deterministic in (seed, segment index, particle index)
  respawnFraction: f32,
  respawnKey: u32,

  maxSpeed: f32,
  // sum of the per-species population targets; written for the readout, not read
  // by any pass
  activeTotal: u32,
  // Render-domain trail persistence — the fade pass is the only reader, and it
  // applies both exactly once per rendered frame. These arrive ALREADY corrected
  // for this display's refresh rate: `writeGlobals` raises the authored
  // per-60 Hz-frame values to the (frame length x 60) power, so the shader can
  // stay a plain multiply and a plain zoom and still look the same at 60 and at
  // 240 Hz. Do not re-derive either from a dt here; there isn't one.
  feedbackAmount: f32,
  feedbackZoom: f32,

  // Seconds for a particle's energy to travel the whole 0→1 (rise) or 1→0
  // (fall) range. These are what turn "alive-by-index" into a *target* the
  // population eases toward instead of snapping to — see stepParticles.
  riseTau: f32,
  fallTau: f32,
  // live entries in the splash buffer; 0 skips the splash pass entirely
  splashCount: u32,
  // Cells searched in each direction by the force pass: the loop runs
  // dx,dy in [-s, +s], so the reach cap is s * min(cellW, cellH). Written by the
  // CPU already clamped to 1..MAX_NEAR_STENCIL *and* to (grid - 1) / 2, so the
  // wrapped search window can never visit the same cell twice and double-count a
  // pair on a small grid.
  nearStencil: u32,

  // 0 = the far lane is off: the whole splat/blur chain is skipped on the CPU
  // and the force pass does not sample the (then stale) field. A gate rather
  // than a zero coefficient, because zero coefficients would still cost eight
  // texture fetches per particle per substep for a guaranteed zero.
  farOn: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

// One impulse hotspot disc, in WORLD units (physarum's equivalent struct is in
// grid cells — same layout, different unit, because the two sims measure space
// differently). Written by `uploadSplashes` in plife.ts.
//
//   posRadius  centre x, centre y, radius, envelope 0..1
//   params     target species (< 0 = all), outward push, swirl, unused
struct Splash {
  posRadius: vec4f,
  params: vec4f,
}

// 16 floats per species; the TS writer is `uploadSpecies` in plife.ts.
//   color.rgb  premultiplied linear rgb (palette x intensity x light x flash)
//   color.a    the population TARGET, as an f32 holding an integer count of
//              particles per segment. Not the live count: what is actually
//              present is the set with energy > 0, which eases toward this.
//   geom       size, stretch, forceScale, friction
//   motion     wander, radiusScale, reserved (written 1), unused
//   pad        unused; keeps the struct at 4 vec4f so the stride is obvious
struct Species {
  color: vec4f,
  geom: vec4f,
  motion: vec4f,
  pad: vec4f,
}

// 32 bytes. vec2f has align 8, so pos@0 vel@8 energy@16 and three f32 of tail
// padding put the stride at exactly 32 with no implicit gaps — worth stating,
// because the TS side allocates `maxParticles * 32` and nothing checks it.
//
// `energy` is the particle's *presence*, 0..1: 0 is genuinely absent (not in the
// grid, not drawn, not integrated), 1 is fully here, and everything between is a
// particle on its way in or out. It is the only piece of population state that
// lives on the GPU, and because it is in this buffer the workbench's
// snapshot/restore carries it for free.
struct Particle {
  pos: vec2f,
  vel: vec2f,
  energy: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
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

/** Species of particle `i`. Derived, never stored — see the Globals note. */
fn speciesOf(i: u32, segSize: u32, speciesCount: u32) -> u32 {
  return min(i / max(segSize, 1u), speciesCount - 1u);
}

/** Population target of a species, as stored in `Species.color.a`. */
fn aliveOf(s: Species) -> u32 {
  return u32(max(s.color.w, 0.0));
}

/** Wrap a position into [0, world) on both axes. */
fn wrapWorld(p: vec2f, world: vec2f) -> vec2f {
  return p - floor(p / world) * world;
}

/**
 * Shortest toroidal offset. Every effective radius is <= MAX_REACH (0.06), which
 * is far under half the world on both axes, so a single shift per axis is exact.
 *
 * This is the fix for the classic seam bug where *positions* wrap but *forces*
 * do not: without it, two particles either side of the wrap read as a world
 * apart and the field develops a visible scar along the seam.
 */
fn wrapDelta(d: vec2f, world: vec2f) -> vec2f {
  return d - world * select(vec2f(0.0), sign(d), abs(d) > 0.5 * world);
}

// ── value noise, for the wander force ─────────────────────────────────────────
//
// A hashed lattice with smoothstep bilinear interpolation. Cheap, seeded from
// the same PCG as everything else, and — the property that matters — *spatially
// coherent*: neighbouring particles get nearly the same wander direction, so the
// noise reads as a drifting current rather than as per-particle jitter. White
// noise per particle would be indistinguishable from raising the temperature.

fn latticeValue(ix: i32, iy: i32, seed: u32) -> f32 {
  // bitcast rather than u32(): the lattice coordinate may be negative and
  // bitcast is the one integer reinterpretation WGSL defines unconditionally.
  return rand01(hash3(bitcast<u32>(ix), bitcast<u32>(iy), seed));
}

fn noise2(p: vec2f, seed: u32) -> f32 {
  let base = floor(p);
  let t = p - base;
  let i0 = vec2i(base);
  let u = t * t * (3.0 - 2.0 * t);
  let v00 = latticeValue(i0.x, i0.y, seed);
  let v10 = latticeValue(i0.x + 1, i0.y, seed);
  let v01 = latticeValue(i0.x, i0.y + 1, seed);
  let v11 = latticeValue(i0.x + 1, i0.y + 1, seed);
  return mix(mix(v00, v10, u.x), mix(v01, v11, u.x), u.y);
}
