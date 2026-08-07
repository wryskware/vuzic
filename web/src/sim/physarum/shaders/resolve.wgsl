@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read_write> deposit: array<atomic<i32>>;
@group(0) @binding(2) var trailSrc: texture_2d_array<f32>;
@group(0) @binding(3) var trailDst: texture_storage_2d_array<r32float, write>;

// i32 fixed-point -> r32float. Each invocation touches only its own cell, so the
// read-and-clear can be a single atomicExchange with no cross-thread hazard.
@compute @workgroup_size(8, 8, 1)
fn resolveDeposit(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= g.gridW || id.y >= g.gridH || id.z >= g.speciesCount) {
    return;
  }
  let cells = g.gridW * g.gridH;
  let idx = id.z * cells + id.y * g.gridW + id.x;
  let raw = atomicExchange(&deposit[idx], 0);
  let c = vec2i(i32(id.x), i32(id.y));
  let prev = textureLoad(trailSrc, c, i32(id.z), 0).r;
  textureStore(trailDst, c, i32(id.z), vec4f(prev + f32(raw) / g.depositScale, 0.0, 0.0, 1.0));
}
