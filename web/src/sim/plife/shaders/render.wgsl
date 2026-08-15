// Two render passes into the HDR surface: the feedback fade, then the particles.
//
// They live in one module but use *disjoint* binding numbers, and each pipeline
// gets a bind group layout describing only what its own entry points touch.
// WebGPU validates the layout against the bindings an entry point statically
// uses, so a declared-but-unused binding costs nothing; one module keeps the
// `Globals` accessor (`g`) shared, which is the reason for doing it this way.
//
//   fade      0 uniform, 1 sampler, 2 previous HDR texture
//   particles 0 uniform, 3 species, 4 particles, 5 auto-exposure state

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var prevSamp: sampler;
@group(0) @binding(2) var prevFrame: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> species: array<Species>;
@group(0) @binding(4) var<storage, read> particles: array<Particle>;
// [gain, mean, initialised, pad] from the post chain's measure pass, one frame
// old. Applied here rather than downstream for the same reason physarum applies
// it in its compositor: auto-exposure measures the HDR surface, so anything
// applied after the measurement is invisible to the controller and its target
// mean stops meaning what it says.
@group(0) @binding(5) var<storage, read> autoState: array<f32>;

// ── fade ──────────────────────────────────────────────────────────────────────
//
// Physarum accumulates memory in its trail field; a particle sim has none, so
// the render-domain echo *is* the trail. It runs first, into a cleared target,
// so the particle pass can then blend additively on top of an image that already
// contains a decayed copy of the last frame.

struct FadeOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsFade(@builtin(vertex_index) vi: u32) -> FadeOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let q = corners[vi];
  var out: FadeOut;
  out.pos = vec4f(q, 0.0, 1.0);
  out.uv = vec2f((q.x + 1.0) * 0.5, (1.0 - q.y) * 0.5);
  return out;
}

@fragment
fn fsFade(in: FadeOut) -> @location(0) vec4f {
  // zoom > 1 shrinks the source window toward the centre, which makes the echo
  // expand outward each frame. It never samples outside [0,1], so the sampler's
  // clamp-to-edge mode is never exercised.
  //
  // Note what this displacement *is*: (uv - 0.5) * (1 - 1/z), i.e. radial, away
  // from screen centre, with a magnitude proportional to the distance from that
  // centre and nothing whatever to do with any particle's velocity. Accumulated
  // over many frames it draws a radial streak on every bright pixel. That is a
  // deliberate effect (default zoom is now 1.0, which disables it) and it is
  // emphatically NOT the velocity stretch — that one lives in vsParticles.
  let z = max(g.feedbackZoom, 1e-3);
  let uv = vec2f(0.5) + (in.uv - vec2f(0.5)) / z;
  let c = textureSampleLevel(prevFrame, prevSamp, uv, 0.0).rgb * max(g.feedbackAmount, 0.0);
  return vec4f(c, 1.0);
}

// ── particles ─────────────────────────────────────────────────────────────────
//
// One instanced quad per particle, 4 vertices as a triangle strip. Every
// particle in the pool is drawn every frame — dormant ones emit a degenerate
// (zero-area) quad, which the rasteriser discards before any fragment work, so
// culling them on the CPU would buy nothing and would cost a readback.

// Rec. 709 luminance weights, for the peak white push. The one place this sim
// needs a luminance from a linear RGB triple; `LUMA_WEIGHTS` in plife/luma.ts is
// the same vector on the CPU side and `plife-luma.test.ts` pins them together.
const LUMA_WEIGHTS: vec3f = vec3f(0.2126, 0.7152, 0.0722);

struct ParticleOut {
  @builtin(position) pos: vec4f,
  // quad-local coordinates in [-1, 1]^2; length() of this is the splat radius
  @location(0) quad: vec2f,
  // premultiplied linear rgb, already scaled by the energy shaping
  @location(1) tint: vec3f,
}

/** World space -> clip. y is negated because world y grows downward on screen. */
fn toClip(p: vec2f) -> vec4f {
  let ndc = p / vec2f(g.worldW, g.worldH) * 2.0 - vec2f(1.0);
  return vec4f(ndc.x, -ndc.y, 0.0, 1.0);
}

@vertex
fn vsParticles(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> ParticleOut {
  var out: ParticleOut;
  let sp = speciesOf(ii, g.segSize, g.speciesCount);
  let s = species[sp];
  let p = particles[ii];
  let world = vec2f(g.worldW, g.worldH);
  let centre = wrapWorld(p.pos, world);

  // triangle-strip corner order: 0 (-1,-1), 1 (1,-1), 2 (-1,1), 3 (1,1)
  let qx = select(-1.0, 1.0, (vi & 1u) == 1u);
  let qy = select(-1.0, 1.0, (vi & 2u) == 2u);

  // Energy alone decides visibility now. The index test that used to sit here
  // (`localIdx >= aliveOf(s)`) would hide a particle the instant the population
  // target moved past it, which is precisely the pop-out the energy ramp exists
  // to remove — the sim would fade it out over fallTau while the renderer had
  // already dropped it on frame one.
  if (p.energy <= 0.0) {
    out.pos = toClip(centre);
    out.quad = vec2f(0.0);
    out.tint = vec3f(0.0);
    return out;
  }

  // Velocity stretch: the quad's long axis follows the direction of travel and
  // grows with speed, up to (1 + stretch) times the sprite radius. This is what
  // turns a field of dots into a field of motion — the same information a still
  // frame otherwise cannot carry.
  let speed = length(p.vel);
  var dir = vec2f(1.0, 0.0);
  if (speed > 1e-5) {
    dir = p.vel / speed;
  }
  let perp = vec2f(-dir.y, dir.x);
  let size = max(s.geom.x, 1e-6);
  // Normalised speed, hoisted because BOTH velocity cues read it: the stretch
  // below and the luminance lane further down. One quantity rather than two
  // similar expressions, so a change to what "fast" means moves them together.
  // Floored away from zero because it is a `pow` base a few lines later.
  let u01 = max(min(speed / max(g.maxSpeed, 1e-6), 1.0), 1e-6);
  let halfLen = size * (1.0 + max(s.geom.y, 0.0) * u01);

  out.pos = toClip(centre + dir * (halfLen * qx) + perp * (size * qy));
  out.quad = vec2f(qx, qy);

  // ── per-particle luminance ──────────────────────────────────────────────────
  //
  // The only brightness mechanism in this sim that varies WITHIN a species —
  // `brightness` × stem-follow, the impulse flash and `intensity` all move a
  // whole colony together, which is why a species used to render as a flat
  // sheet however loud the music got. Composed (not stacked): those three
  // decide how bright the colony is, this decides how that brightness is
  // distributed across its members.
  //
  // The whole policy — the stops budget, how much of the display's HDR headroom
  // to spend, how far to bleach on SDR — is resolved on the CPU in
  // plife/luma.ts. What is left here is a shaped speed, an exp2, and a lerp.
  //
  // At `lumaStops = 0` (depth 0) every term below is exactly the identity:
  // exp2(0 · x − 0 + 0) = 1 and mix(rgb, _, 0) = rgb. That is the A/B baseline,
  // and it holds by arithmetic rather than by a branch.
  let shaped = pow(u01, g.lumaExponent);
  // Static, drawn once per particle index from the world seed: spatial grain, so
  // a species reads as a population rather than as a sheet. Deliberately NOT
  // per-tick — animated per-particle noise is indistinguishable from raising the
  // temperature, which is the same argument the wander force's value noise makes.
  let jitter = (rand01(hash3(ii, g.seed, 0x1u)) - 0.5) * 2.0 * g.lumaJitter;
  let gain = exp2(g.lumaStops * (shaped - g.lumaAnchor) + jitter);

  // The SDR peak cue. Reinhard preserves channel ratios, so a saturated core
  // cannot pass white however hard it is driven — its strongest channel simply
  // pins. Desaturating toward the colour's own luminance lifts the two weak
  // channels instead, so the peak reads as *light*. `lumaWhite` already carries
  // the headroom trade: it fades out as real headroom arrives, because on a
  // display that can go brighter, bleaching only costs the hue that labels the
  // species.
  var rgb = s.color.rgb;
  rgb = mix(rgb, vec3f(dot(rgb, LUMA_WEIGHTS)), g.lumaWhite * shaped);

  // Energy shaping: pow(e, 1.5) so a particle on its way in or out fades faster
  // than linearly. This is the hook the population lane writes to — a species
  // whose target is falling dims *ahead* of the linear ramp, which reads as
  // "leaving" rather than as "being turned down".
  out.tint = rgb * (pow(max(p.energy, 0.0), 1.5) * gain);
  return out;
}

@fragment
fn fsParticles(in: ParticleOut) -> @location(0) vec4f {
  // Soft gaussian splat. exp(-4 d^2) - exp(-4) puts the edge at exactly 0 (a
  // raw gaussian leaves a visible square seam where the quad ends) and the
  // division renormalises the peak back to 1.
  let d = length(in.quad);
  let edge = exp(-4.0);
  let a = max((exp(-4.0 * d * d) - edge) / (1.0 - edge), 0.0);

  // Scene exposure and the adapted gain, both applied before the surface the
  // controller measures. See the binding-5 comment.
  let gain = max(autoState[0], 1e-4);
  return vec4f(in.tint * (a * g.exposure * gain), a);
}
