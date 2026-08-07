@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> species: array<Species>;
@group(0) @binding(2) var<storage, read> senseM: array<f32>;
@group(0) @binding(3) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(4) var<storage, read_write> deposit: array<atomic<i32>>;
@group(0) @binding(5) var trail: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> splashes: array<Splash>;
@group(0) @binding(7) var soil: texture_2d<f32>;

// Species i senses sum over j of M[i][j] * trail_j. One dispatch covers every
// species; the row of M is selected by the agent's index range.
//
// Soil enters here as a *per-sample-point* gain, which is the only place it can do
// any work: a uniform gain on all three samples would cancel out of the three-way
// comparison below. Weighting each sample by its own soil makes ground that has
// been built on before read louder than fresh ground, so an agent steers toward old
// scars — and a species that avoids a field there avoids it harder. Soil amplifies
// contrast; it does not impose an attractor of its own.
fn sensed(p: vec2f, sp: u32, gi: vec2i) -> f32 {
  var acc = 0.0;
  for (var j = 0u; j < g.speciesCount; j = j + 1u) {
    let w = senseM[sp * g.speciesCount + j];
    if (abs(w) > 1e-4) {
      acc = acc + w * trailBilinear(trail, p, gi, i32(j));
    }
  }
  return acc * (1.0 + g.soilSense * soilBilinear(soil, p, gi));
}

@compute @workgroup_size(64)
fn initAgents(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let total = g.speciesCount * g.agentsPerSpecies;
  if (i >= total) {
    return;
  }
  let sp = i / g.agentsPerSpecies;
  let h0 = hash3(g.seed, 0x9e3779b9u, i);
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  agents[i].pos = vec2f(rand01(h0) * f32(g.gridW), rand01(h1) * f32(g.gridH));
  agents[i].heading = rand01(h2) * TAU;
  agents[i].species = f32(sp);
}

// Partial, section-keyed re-seed. Same shape as initAgents but it only touches a
// fraction of the pool and leaves the trail field alone, so a boundary injects
// new matter into a world that still remembers the last section.
//
// The key is (seed, segment index, agent index) exactly as plan.md specifies: a
// returning section re-scatters the same agents to the same places, and the only
// difference between the two visits is the accumulated state they land in.
@compute @workgroup_size(64)
fn respawnAgents(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let total = g.speciesCount * g.agentsPerSpecies;
  if (i >= total) {
    return;
  }
  let key = hash3(g.seed ^ 0x5bd1e995u, g.respawnKey, i);
  if (rand01(key) >= g.respawnFraction) {
    return;
  }
  let h1 = pcg(key);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  agents[i].pos = vec2f(rand01(h1) * f32(g.gridW), rand01(h2) * f32(g.gridH));
  agents[i].heading = rand01(h3) * TAU;
}

// Radial splash — the impulse engine's one mechanical (as opposed to cosmetic)
// response. Agents inside a hotspot disc are turned outward and thrown outward,
// so a snare reads as something physically landing in the field rather than as a
// brightness change. It runs *before* stepAgents each tick, so the agent then
// takes its normal sensed step from the displaced position and the disturbance
// propagates into the trail network instead of sitting on top of it.
//
// Toroidal, like every other distance in this sim: an agent near the wrap seam is
// still inside a disc on the other side.
@compute @workgroup_size(64)
fn splashAgents(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let total = g.speciesCount * g.agentsPerSpecies;
  if (i >= total || g.splashCount == 0u) {
    return;
  }
  let sp = i / g.agentsPerSpecies;
  let slot = i - sp * g.agentsPerSpecies;
  let s = species[sp];
  let alive = u32(clamp(s.misc.z, 0.0, 1.0) * f32(g.agentsPerSpecies));
  if (slot >= alive) {
    return;
  }

  let gf = vec2f(f32(g.gridW), f32(g.gridH));
  var pos = agents[i].pos;
  var heading = agents[i].heading;
  var hit = false;

  for (var n = 0u; n < g.splashCount; n = n + 1u) {
    let sl = splashes[n];
    // `target` is a reserved keyword in WGSL; `want` is not.
    let want = sl.params.x;
    if (want >= 0.0 && u32(want + 0.5) != sp) {
      continue;
    }
    // shortest toroidal offset from the hotspot centre
    var d = pos - sl.posRadius.xy;
    d = d - gf * round(d / gf);
    let rad = max(sl.posRadius.z, 1e-3);
    let r = length(d);
    if (r > rad) {
      continue;
    }
    // Smooth falloff: a linear cone leaves a hard visible circle edge.
    let f = 1.0 - r / rad;
    let strength = clamp(sl.posRadius.w * f * f, 0.0, 1.0);
    if (strength <= 0.0) {
      continue;
    }

    // Outward direction, twisted by swirl. At the exact centre there is no
    // outward, so the agent keeps its own heading and is only pushed along it.
    var out = vec2f(cos(heading), sin(heading));
    if (r > 1e-4) {
      let a = atan2(d.y, d.x) + sl.params.z * strength;
      out = vec2f(cos(a), sin(a));
    }

    let hv = vec2f(cos(heading), sin(heading));
    let blended = mix(hv, out, strength);
    let len = length(blended);
    let nv = select(out, blended / max(len, 1e-4), len > 1e-4);
    heading = atan2(nv.y, nv.x);
    pos = pos + out * (sl.params.y * strength);
    hit = true;
  }

  if (hit) {
    agents[i].pos = wrapPos(pos, gf);
    agents[i].heading = heading % TAU;
  }
}

@compute @workgroup_size(64)
fn stepAgents(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let total = g.speciesCount * g.agentsPerSpecies;
  if (i >= total) {
    return;
  }
  let sp = i / g.agentsPerSpecies;
  let slot = i - sp * g.agentsPerSpecies;
  let s = species[sp];

  // Dormant agents write nothing and skip all sensing work.
  let alive = u32(clamp(s.misc.z, 0.0, 1.0) * f32(g.agentsPerSpecies));
  if (slot >= alive) {
    return;
  }

  let gi = vec2i(i32(g.gridW), i32(g.gridH));
  let gf = vec2f(f32(g.gridW), f32(g.gridH));
  let a = agents[i];

  // x is the agent's own-species trail intensity, squashed to [0,1) so pow() in
  // adaptive() stays well behaved regardless of the deposit/decay equilibrium.
  let own = max(trailBilinear(trail, a.pos, gi, i32(sp)), 0.0);
  let x = 1.0 - exp(-own * g.senseGain);

  let sd = max(adaptive(s.sensorDist, x), 0.5);
  let sa = adaptive(s.sensorAngle, x);
  let ra = adaptive(s.rotate, x);
  let md = max(adaptive(s.moveDist, x), 0.0);

  let h = a.heading;
  let pF = a.pos + vec2f(cos(h), sin(h)) * sd;
  let pL = a.pos + vec2f(cos(h - sa), sin(h - sa)) * sd;
  let pR = a.pos + vec2f(cos(h + sa), sin(h + sa)) * sd;

  let vF = sensed(pF, sp, gi);
  let vL = sensed(pL, sp, gi);
  let vR = sensed(pR, sp, gi);

  var turn = 0.0;
  if (vF >= vL && vF >= vR) {
    turn = 0.0;
  } else if (vF < vL && vF < vR) {
    let r = rand01(hash3(g.seed, g.simTick, i));
    turn = select(-ra, ra, r > 0.5);
  } else if (vL > vR) {
    turn = -ra;
  } else {
    turn = ra;
  }

  let heading = (h + turn) % TAU;
  let pos = wrapPos(a.pos + vec2f(cos(heading), sin(heading)) * md, gf);

  agents[i].pos = pos;
  agents[i].heading = heading;

  let c = wrapCoord(vec2i(i32(floor(pos.x)), i32(floor(pos.y))), gi);
  let cells = g.gridW * g.gridH;
  let idx = sp * cells + u32(c.y) * g.gridW + u32(c.x);
  // Fertile ground takes more. Fertility used to multiply *after* the CPU clamp,
  // which quietly cost the i32 atomic 4x of the headroom PhysarumConfig
  // .depositScale documents. The final value is clamped here instead, so
  // g.maxDeposit is the true per-agent-per-cell-per-tick bound.
  let fertility = 1.0 + g.soilDeposit * soilBilinear(soil, pos, gi);
  let amount = min(max(s.misc.x, 0.0) * fertility, g.maxDeposit);
  atomicAdd(&deposit[idx], i32(amount * g.depositScale));
}
