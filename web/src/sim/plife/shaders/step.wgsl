// The force integration, the two re-scatter passes, and the impulse splash.
//
// Ping-pong: `stepParticles`, `initParticles` and `respawnParticles` read `src`
// and write `dst`, one thread per particle index, and the CPU flips parity
// afterwards. That is what lets `respawnParticles` be an ordinary pass instead
// of an in-place mutation with a read/write hazard, and it means `dst` is fully
// defined after every dispatch — absent particles are copied through rather than
// skipped.
//
// `splashParticles` is the exception and is deliberately outside that scheme: it
// mutates the *source* buffer in place, before the grid passes of the same
// substep, so the ordinary step then propagates the kick. It therefore needs a
// read_write view of the buffer the others read, which is why it has its own
// binding numbers (8, 9) and its own bind group layout. Same trick as
// render.wgsl: one module, disjoint binding numbers, a layout per pipeline that
// names only what that entry point statically touches.
//
//   step / init / respawn   0 globals, 1 species, 2 interaction, 3 src,
//                           4 dst, 5 cellCount, 6 cellStart, 7 sortedIdx
//   splash                  0 globals, 8 splashes, 9 particles (read_write)

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> species: array<Species>;
// 3 K^2 floats: [attraction | maxR | minR], in that order, matching the theta
// vector's block order (see preset.ts). One buffer, one writeBuffer per apply.
@group(0) @binding(2) var<storage, read> interaction: array<f32>;
@group(0) @binding(3) var<storage, read> src: array<Particle>;
@group(0) @binding(4) var<storage, read_write> dst: array<Particle>;
@group(0) @binding(5) var<storage, read> cellCount: array<u32>;
@group(0) @binding(6) var<storage, read> cellStart: array<u32>;
@group(0) @binding(7) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(8) var<storage, read> splashes: array<Splash>;
@group(0) @binding(9) var<storage, read_write> particles: array<Particle>;

// Hard-core repulsion slope. Universal and sign-blind: it applies to every pair
// regardless of what `attraction` says, because without it two species with a
// mutual attraction collapse to a point in a few frames and the sim is over.
// 3.0 is ~6x the largest shipped attraction magnitude, which is enough that the
// core wins decisively inside rmin and contributes nothing outside it.
const REPULSE: f32 = 3.0;
// Mirrors MIN_R_FLOOR in config.ts. (There is deliberately no mirrored R_CAP
// here any more: reach is clamped against the live cell size from Globals in
// the force loop, so the cap cannot go stale when the config value moves.)
const MIN_R_FLOOR: f32 = 0.002;

// Wander sampling. `WANDER_FREQ` sets how many independent currents fit across
// the world (3 => cells about a third of the height) and `WANDER_DRIFT` is how
// fast the field itself moves, in lattice units per tick. Both are tunable and
// both are visible: raise the frequency for turbulence, raise the drift for a
// field that never settles.
const WANDER_FREQ: f32 = 3.0;
const WANDER_DRIFT: f32 = 0.0015;

// How far a newly grown particle lands from the conspecific it grew out of, in
// world units (world height is 1, so this is 1% of the screen height). Small on
// purpose: the point of donor-relative spawning is that growth *blooms out of
// existing structure*. Make it much larger and it degenerates back into rain.
const SPAWN_JITTER: f32 = 0.01;

// A donor this faint is on its way out itself, and cloning a dying particle
// would seed the new colony into a hole. Below the threshold the spawn falls
// back to a hashed position anywhere in the world.
const DONOR_MIN_ENERGY: f32 = 0.5;

// ── splash tuning ─────────────────────────────────────────────────────────────
//
// `splashPush` in ResponseConfig is authored in PHYSARUM's units — "grid cells
// per tick", roughly 2 (a soft vocal pulse) to 9 (a snare). Particle life has no
// grid cells in that sense, so the number has to be converted rather than
// reused, and one impulse config has to keep driving both sims without being
// re-tuned per substrate. 0.02 world units per authored cell puts a snare at
// 9 x 0.02 = 0.18 world/s of added velocity, which against the shipped maxSpeed
// of 0.3 is about 0.6 of terminal speed: a shove that is unmistakably visible
// and still cannot fling a particle across the frame in one substep.
//
// `splashSwirl` is authored in radians of heading twist (0.3..0.7). There is no
// heading here — the tangential component is a velocity like any other — so the
// number is scaled up until the curl reads at the same strength as the push it
// accompanies. Both of these are taste, not physics: they are the first two
// knobs to reach for if splashes feel weak or feel like explosions.
const PUSH_SCALE: f32 = 0.02;
const PUSH_SWIRL_SCALE: f32 = 3.0;

/**
 * ## Population, as of the dynamic-population lane
 *
 * `aliveOf(species)` is no longer "how many of this segment exist". It is the
 * TARGET, and `energy` is the actual presence:
 *
 *   localIdx < target   the particle is wanted; energy ramps up over riseTau
 *   localIdx >= target   it is not; energy ramps down over fallTau
 *   energy == 0          it is genuinely absent — frozen in place, out of the
 *                        grid, degenerate in the renderer
 *
 * Everything with energy > 0 runs the full force path, in both directions.
 * Outgoing influence is already weighted by `q.energy` in the pair sum below, so
 * a particle on its way in pushes proportionally to how present it is and a
 * particle on its way out lets go gradually. That is the whole reason the ramp
 * exists: instant pop-in and pop-out looked like a rendering fault.
 *
 * Determinism is unaffected. Every input to the target (theta, the smoothed stem
 * levels, the accent activity) is a deterministic function of the tick, and the
 * spawn draw is keyed on (seed, tick, i) like every other draw in this sim, so
 * a given (track, seed, device) still replays exactly.
 */
@compute @workgroup_size(64)
fn stepParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles) {
    return;
  }
  let world = vec2f(g.worldW, g.worldH);
  let si = speciesOf(i, g.segSize, g.speciesCount);
  let localIdx = i - si * g.segSize;
  let me = species[si];
  var p = src[i];

  // `target` is a reserved keyword in WGSL, hence `goal`.
  let goal = aliveOf(me);
  let wanted = localIdx < goal;
  var energy = clamp(p.energy, 0.0, 1.0);

  if (wanted) {
    // Colony spawn. A particle arriving from nothing is placed on top of a
    // living conspecific rather than anywhere in the world, so a swelling
    // population grows out of the structure that is already there instead of
    // raining in uniformly over it. The donor is read from `src`, which this
    // pass never writes, so there is no hazard and no ordering requirement
    // between the two threads.
    if (energy <= 0.0) {
      let hd = hash3(g.seed ^ 0xc0107eeu, g.tick, i);
      let donor = src[si * g.segSize + (hd % max(goal, 1u))];
      let h0 = pcg(hd);
      let h1 = pcg(h0);
      if (goal == 0u || donor.energy < DONOR_MIN_ENERGY) {
        // No usable donor: scatter. Happens on the first frame a species comes
        // back from a target of zero, and only then.
        p.pos = vec2f(rand01(h0) * g.worldW, rand01(h1) * g.worldH);
        p.vel = vec2f(rand01(pcg(h1)) - 0.5, rand01(pcg(pcg(h1))) - 0.5) * (0.1 * g.maxSpeed);
      } else {
        // Box-Muller, polar form — the same draw initParticles uses for its
        // cluster radius, at a far smaller sigma.
        let u1 = max(rand01(h0), 1e-6);
        let radius = sqrt(-2.0 * log(u1)) * SPAWN_JITTER;
        let angle = TAU * rand01(h1);
        p.pos = wrapWorld(donor.pos + vec2f(cos(angle), sin(angle)) * radius, world);
        // Half the donor's velocity: enough to inherit the flock's direction,
        // little enough that a burst of new matter does not add net momentum.
        p.vel = donor.vel * 0.5;
      }
    }
    energy = min(energy + g.dt / max(g.riseTau, 1e-3), 1.0);
  } else {
    energy = max(energy - g.dt / max(g.fallTau, 1e-3), 0.0);
    if (energy <= 0.0) {
      // Absent: frozen where it died, so a later re-spawn of this index is a
      // clean placement rather than a resurrection mid-flight. `dst` is still
      // written, so it remains a complete state after every dispatch.
      var gone = p;
      gone.vel = vec2f(0.0);
      gone.energy = 0.0;
      dst[i] = gone;
      return;
    }
  }

  let k = g.speciesCount;
  let kk = k * k;
  let radiusScale = max(me.motion.y, 1e-3);

  let here = wrapWorld(p.pos, world);
  let gw = i32(g.gridW);
  let gh = i32(g.gridH);
  let cx = i32(clamp(floor(here.x / max(g.cellW, 1e-6)), 0.0, f32(g.gridW - 1u)));
  let cy = i32(clamp(floor(here.y / max(g.cellH, 1e-6)), 0.0, f32(g.gridH - 1u)));

  var force = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      // Toroidal cell wrap. The double modulo is the portable way to get a
      // non-negative remainder out of a signed index.
      let nx = ((cx + dx) % gw + gw) % gw;
      let ny = ((cy + dy) % gh + gh) % gh;
      let c = u32(ny * gw + nx);
      let start = cellStart[c];
      let n = cellCount[c];

      for (var s = 0u; s < n; s = s + 1u) {
        let j = sortedIdx[start + s];
        if (j == i) {
          continue;
        }
        let q = src[j];
        let sj = speciesOf(j, g.segSize, k);
        let pair = si * k + sj;

        // The receiver's radiusScale scales the whole row, so "this species
        // reaches further" is one number rather than K edits. Capped at R_CAP
        // because the grid search cannot see past one cell.
        // Reach is clamped to the live grid cell size, not to a mirrored
        // constant: the 3×3 search only sees one cell width in each direction,
        // so any reach beyond min(cellW, cellH) would silently truncate at the
        // search window. (A mirrored R_CAP constant went stale here once when
        // the config value moved — the cell size in Globals cannot.)
        let rcap = min(g.cellW, g.cellH);
        let rmax = min(interaction[kk + pair] * radiusScale, rcap);
        let d0 = q.pos - p.pos;
        let d = wrapDelta(d0, world);
        let r2 = dot(d, d);
        if (r2 > rmax * rmax) {
          continue;
        }
        // hi is written as max(..., MIN_R_FLOOR) so clamp() never sees lo > hi,
        // which is undefined in WGSL. A pair whose rmax collapses below the
        // floor therefore degenerates to pure repulsion rather than to garbage.
        let rmin = clamp(
          interaction[2u * kk + pair] * radiusScale,
          MIN_R_FLOOR,
          max(rmax - 1e-4, MIN_R_FLOOR),
        );

        var r = sqrt(r2);
        var dir = vec2f(1.0, 0.0);
        if (r < 1e-6) {
          // Exactly coincident particles have no direction to separate along.
          // A hashed one keeps the step deterministic in (i, j, tick) — which is
          // the whole determinism claim — where normalising a zero vector would
          // produce NaN and poison the buffer for the rest of the run.
          let a = rand01(hash3(i, j, g.tick)) * TAU;
          dir = vec2f(cos(a), sin(a));
          r = 1e-6;
        } else {
          dir = d / r;
        }

        var f = 0.0;
        if (r < rmin) {
          // Divergent hard core: 0 at rmin (continuous with the tent), and
          // -REPULSE·rmin/eps ≈ -20·REPULSE as r → 0. The first build used a
          // *linear* core capped at -REPULSE, and dense clusters collapsed into
          // point singularities: the summed tent attraction of a hundred
          // neighbours just outside rmin dwarfed a bounded -3 from the few
          // inside it. A core that steepens toward contact is how the original
          // vuzic sim (R0_FORCE = 100, ~1/r) kept its clusters open, and the
          // eps floor keeps it finite so the integrator stays stable at dt=1/60
          // (the maxSpeed clamp downstream bounds the worst case regardless).
          f = REPULSE * (r - rmin) / (r + 0.05 * rmin);
        } else {
          // Tent: peaks at the band's midpoint, zero at both ends. Zero at rmax
          // is what makes the force continuous as a pair enters and leaves the
          // neighbourhood — a force with a step at the cutoff pumps energy in
          // and the whole field slowly heats up.
          let span = max(rmax - rmin, 1e-6);
          f = interaction[pair] * (1.0 - abs(2.0 * r - (rmin + rmax)) / span);
        }
        force = force + dir * (f * q.energy);
      }
    }
  }

  // Wander: a spatially coherent direction field, sampled at the particle's own
  // position and drifting with the tick. Applied as a unit push scaled by the
  // species' amplitude, so it is a current the particle swims in rather than
  // per-particle noise (which would just be temperature).
  let phi = TAU * noise2(p.pos * WANDER_FREQ + vec2f(f32(g.tick) * WANDER_DRIFT), g.seed);
  force = force + vec2f(cos(phi), sin(phi)) * max(me.motion.x, 0.0);

  var v = p.vel + force * (g.forceGain * max(me.geom.z, 0.0) * g.dt);
  // Exponential damping rather than v *= (1 - friction*dt): the exponential is
  // unconditionally stable, so a modulated friction can never flip the sign of
  // the velocity the way the linear form does past friction*dt = 1.
  v = v * exp(-max(me.geom.w, 0.0) * g.dt);
  let speed = length(v);
  if (speed > g.maxSpeed) {
    v = v * (g.maxSpeed / max(speed, 1e-9));
  }

  // fract() is x - floor(x), so this wraps negatives correctly on both axes.
  var pos = fract((p.pos + v * g.dt) / world) * world;

  // Belt and braces. Nothing above should be able to produce a non-finite value
  // (the coincident-pair case is the one that could, and it is handled), but the
  // particle buffer is a feedback term: one NaN never leaves, it spreads through
  // the force sum to every neighbour and the frame goes black permanently. The
  // dot-product tests catch NaN (fails >= 0) and Inf (fails < 1e12) together.
  let vv = dot(v, v);
  let pp = dot(pos, pos);
  if (!(vv >= 0.0 && vv < 1e12 && pp >= 0.0 && pp < 1e12)) {
    let h = hash3(g.seed ^ 0xbadfeedu, i, g.tick);
    pos = vec2f(rand01(h) * g.worldW, rand01(pcg(h)) * g.worldH);
    v = vec2f(0.0);
  }

  var out = p;
  out.pos = pos;
  out.vel = v;
  out.energy = energy;
  dst[i] = out;
}

/**
 * A fresh world: each species scattered around its own seeded cluster centre,
 * with a gaussian radius. Clusters rather than a uniform sprinkle because a
 * uniform initial condition is the one state particle life takes longest to
 * leave — every force cancels by symmetry and the first ten seconds are a slow
 * fade into structure. Starting clumped means the first frame already has
 * something to look at.
 */
@compute @workgroup_size(64)
fn initParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles) {
    return;
  }
  let sp = speciesOf(i, g.segSize, g.speciesCount);
  let localIdx = i - sp * g.segSize;
  let hc = hash3(g.seed, 0xc1a55eedu, sp);
  let centre = vec2f(rand01(hc) * g.worldW, rand01(pcg(hc)) * g.worldH);

  let h0 = hash3(g.seed ^ 0x9e3779b9u, i, 0x5ca77e5u);
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  // Box-Muller, polar form: radius from the log, angle uniform.
  let u1 = max(rand01(h0), 1e-6);
  let sigma = 0.15 * min(g.worldW, g.worldH);
  let radius = sqrt(-2.0 * log(u1)) * sigma;
  let angle = TAU * rand01(h1);

  var out: Particle;
  out.pos = wrapWorld(centre + vec2f(cos(angle), sin(angle)) * radius, vec2f(g.worldW, g.worldH));
  out.vel = vec2f(rand01(h2) - 0.5, rand01(h3) - 0.5) * (0.1 * g.maxSpeed);
  // A fresh world starts at its target exactly — no ramp-in from black. Anything
  // outside the target must be EXACTLY 0, not merely small: `energy > 0` is the
  // grid's alive filter and the spawn trigger, so a leftover epsilon would bin a
  // particle that is supposed to be absent and would suppress its colony spawn.
  out.energy = select(0.0, 1.0, localIdx < aliveOf(species[sp]));
  out.pad0 = 0.0;
  out.pad1 = 0.0;
  out.pad2 = 0.0;
  dst[i] = out;
}

/**
 * Section-boundary re-scatter: `respawnFraction` of the pool, chosen and placed
 * by hash(seed, segment index, particle index). Uniform rather than clustered,
 * which is the difference from initParticles and the point of the pass — new
 * matter arriving into an old world, not a new world.
 */
@compute @workgroup_size(64)
fn respawnParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles) {
    return;
  }
  var out = src[i];
  let key = hash3(g.seed ^ 0x5bd1e995u, g.respawnKey, i);
  if (rand01(key) < g.respawnFraction) {
    let h1 = pcg(key);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let h4 = pcg(h3);
    out.pos = vec2f(rand01(h1) * g.worldW, rand01(h2) * g.worldH);
    out.vel = vec2f(rand01(h3) - 0.5, rand01(h4) - 0.5) * (0.1 * g.maxSpeed);
    // Energy is deliberately NOT reset here. This pass moves matter; the
    // population lane owns how much matter there is. Forcing 1.0 would light up
    // every absent particle it happened to select and then immediately fade them
    // out again — a section boundary would flash the whole dormant pool.
  }
  dst[i] = out;
}

/**
 * Radial splash — the impulse engine's one mechanical (as opposed to cosmetic)
 * response, ported from physarum. Particles inside a hotspot disc get a velocity
 * kick outward plus a tangential twist, so a snare reads as something physically
 * landing in the field rather than as a brightness change.
 *
 * Two things it does NOT do, both on purpose:
 *
 * - It does not move anyone. Physarum displaces its agents because an agent's
 *   position is its whole state; a particle has momentum, and adding velocity
 *   lets the disc's disturbance travel and decay through the ordinary force
 *   solver instead of being a rip in the field that the next step has to heal.
 * - It does not touch absent particles. Kicking something with energy 0 would
 *   move a corpse that is not in the grid and is not drawn, and would then have
 *   it re-appear somewhere it was never seen leaving.
 *
 * It runs BEFORE the grid passes of the same substep, so the neighbour search
 * and the force sum both see the kicked velocities and the splash propagates
 * into the structure rather than sitting on top of the frame. It is also the one
 * pass that mutates the source buffer in place — see the header note.
 */
@compute @workgroup_size(64)
fn splashParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles || g.splashCount == 0u) {
    return;
  }
  let p = particles[i];
  if (p.energy <= 0.0) {
    return;
  }
  let sp = speciesOf(i, g.segSize, g.speciesCount);
  let world = vec2f(g.worldW, g.worldH);

  var v = p.vel;
  var hit = false;
  for (var n = 0u; n < g.splashCount; n = n + 1u) {
    let sl = splashes[n];
    // `target` is a reserved keyword in WGSL; `want` is not.
    let want = sl.params.x;
    if (want >= 0.0 && u32(want + 0.5) != sp) {
      continue;
    }
    // Toroidal, like every other distance in this sim: a particle near the wrap
    // seam is still inside a disc on the other side.
    let d = wrapDelta(p.pos - sl.posRadius.xy, world);
    let rad = max(sl.posRadius.z, 1e-5);
    let r = length(d);
    if (r > rad) {
      continue;
    }
    // Linear cone. Physarum squares this because its falloff drives a heading
    // blend, where a hard edge is visible as a circle; a velocity impulse with a
    // linear edge just looks like a shock front, which is what it is.
    let falloff = 1.0 - r / rad;

    var dir = vec2f(1.0, 0.0);
    if (r > 1e-5) {
      dir = d / r;
    } else {
      // Dead centre has no outward direction. Hashed on (seed, i, tick) so the
      // choice is deterministic, exactly like the coincident-pair case above.
      let a = rand01(hash3(g.seed ^ 0x5b1a54u, i, g.tick)) * TAU;
      dir = vec2f(cos(a), sin(a));
    }
    let perp = vec2f(-dir.y, dir.x);
    let amp = sl.posRadius.w * falloff * PUSH_SCALE;
    v = v + (dir * sl.params.y + perp * (sl.params.z * PUSH_SWIRL_SCALE)) * amp;
    hit = true;
  }

  if (hit) {
    // Clamped here as well as in stepParticles: the step's clamp happens after
    // the force sum, and an unbounded velocity entering the neighbour search is
    // a particle that crosses several cells in one substep.
    let speed = length(v);
    if (speed > g.maxSpeed) {
      v = v * (g.maxSpeed / max(speed, 1e-9));
    }
    particles[i].vel = v;
  }
}
