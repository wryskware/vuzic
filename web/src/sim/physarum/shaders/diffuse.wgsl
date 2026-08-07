@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> species: array<Species>;
@group(0) @binding(2) var trailSrc: texture_2d_array<f32>;
@group(0) @binding(3) var trailDst: texture_storage_2d_array<r32float, write>;
// read_write is core WebGPU for r32float/r32sint/r32uint, which is exactly why the
// soil is r32float: it lets the accumulate-and-decay be an in-place update with no
// ping-pong texture and no extra dispatch.
@group(0) @binding(4) var soil: texture_storage_2d<r32float, read_write>;

// 3x3 blur with a per-species centre weight (low centre = wide/soft, high centre
// = sharp) followed by per-species multiplicative decay. All K layers in one
// dispatch via the z dimension.
@compute @workgroup_size(8, 8, 1)
fn diffuseDecay(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= g.gridW || id.y >= g.gridH || id.z >= g.speciesCount) {
    return;
  }
  let gi = vec2i(i32(g.gridW), i32(g.gridH));
  let layer = i32(id.z);
  let s = species[id.z];
  let centre = clamp(s.misc.w, 0.0, 1.0);
  let ring = (1.0 - centre) / 8.0;
  let c = vec2i(i32(id.x), i32(id.y));

  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let w = select(ring, centre, dx == 0 && dy == 0);
      sum = sum + w * trailTexel(trailSrc, c + vec2i(dx, dy), gi, layer);
    }
  }

  let v = sum * clamp(s.misc.y, 0.0, 1.0);
  textureStore(trailDst, c, layer, vec4f(v, 0.0, 0.0, 1.0));

  // Soil rides along in the layer-0 invocation: one cell, one writer, no extra
  // dispatch and no new pass. It reads trailSrc, which this pass only ever reads,
  // so there is no read/write hazard against the trail it is folded into.
  if (id.z == 0u) {
    updateSoil(c, gi);
  }
}

// Presence, not intensity: max over species rather than a sum, squashed by the
// same senseGain the agents use. A cell where *anything* built a network counts
// once, so K does not change how fast soil grows.
fn updateSoil(c: vec2i, gi: vec2i) {
  var present = 0.0;
  for (var k = 0u; k < g.speciesCount; k = k + 1u) {
    present = max(present, trailTexel(trailSrc, c, gi, i32(k)));
  }
  let p = 1.0 - exp(-max(present, 0.0) * g.senseGain);
  let prev = textureLoad(soil, c).r;
  let v = clamp(prev * clamp(g.soilDecay, 0.0, 1.0) + g.soilAccum * p, 0.0, 1.0);
  textureStore(soil, c, vec4f(v, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn clearTrail(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= g.gridW || id.y >= g.gridH || id.z >= g.speciesCount) {
    return;
  }
  textureStore(trailDst, vec2i(i32(id.x), i32(id.y)), i32(id.z), vec4f(0.0));
}

// Wiping soil is a separate entry point on purpose: a full reseed calls both, a
// partial (section-boundary) reseed calls neither. Soil surviving the boundary is
// the entire point of the layer.
@compute @workgroup_size(8, 8, 1)
fn clearSoil(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= g.gridW || id.y >= g.gridH || id.z >= 1u) {
    return;
  }
  textureStore(soil, vec2i(i32(id.x), i32(id.y)), vec4f(0.0));
}
