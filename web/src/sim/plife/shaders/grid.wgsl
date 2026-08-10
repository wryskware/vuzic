// Uniform-grid neighbour binning: count, prefix-sum, scatter. Three entry points
// sharing one bind group, run once per substep before the force pass.
//
// Why a grid at all: the force pass is O(alive x neighbours) and the only thing
// that keeps "neighbours" from meaning "every other particle" is a spatial
// index. The grid's cell size is >= R_CAP on both axes (see plife.ts's
// chooseGrid) and the force pass searches `nearStencil` cells in each direction,
// which together are exactly the condition that makes the search find every pair
// within any legal interaction radius (reach is clamped to stencil x cell).
//
// The buffers are cleared with encoder.clearBuffer rather than a clear pipeline —
// two zero-fills of a few kilobytes each, with no shader, no bind group and no
// dispatch to get wrong.

@group(0) @binding(0) var<uniform> g: Globals;
// Declared but no longer read: the alive filter moved from the species block's
// population target to the particle's own energy (see `isAlive`). The binding
// stays because the bind group layout is shared with nothing else and a
// declared-but-unused binding costs nothing — dropping it would only mean this
// module's layout stopped matching the group plife.ts builds.
@group(0) @binding(1) var<storage, read> species: array<Species>;
@group(0) @binding(2) var<storage, read> particles: array<Particle>;
// cellCount is atomic here because count/scatter race on it; scanCells reads it
// with atomicLoad rather than declaring a second, non-atomic view of the same
// buffer, which WGSL would not allow inside one module.
@group(0) @binding(3) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellFill: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(6) var<storage, read_write> sortedIdx: array<u32>;

fn cellOf(p: vec2f) -> u32 {
  let cx = u32(clamp(floor(p.x / max(g.cellW, 1e-6)), 0.0, f32(g.gridW - 1u)));
  let cy = u32(clamp(floor(p.y / max(g.cellH, 1e-6)), 0.0, f32(g.gridH - 1u)));
  return cy * g.gridW + cx;
}

/**
 * Absent particles are excluded from the grid entirely, not just from the force
 * sum. That matters: with aliveFraction at 0.12 the accents contribute 12% of
 * their segment, and binning the other 88% would make every cell list eight
 * times longer for nothing.
 *
 * The predicate is `energy > 0`, NOT `localIdx < target`. Those used to be the
 * same thing; since the population lane landed they are not. A particle whose
 * index has fallen outside the target is *fading*, and a fading particle has to
 * keep binning and keep interacting — if it dropped out of the grid the instant
 * the target moved, a dissolve would read as a teleport: the visible sprite
 * would go on ramping down while the structure it belonged to had already been
 * torn out from underneath it.
 */
fn isAlive(i: u32) -> bool {
  return particles[i].energy > 0.0;
}

@compute @workgroup_size(256)
fn countParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles) {
    return;
  }
  if (!isAlive(i)) {
    return;
  }
  let c = cellOf(wrapWorld(particles[i].pos, vec2f(g.worldW, g.worldH)));
  atomicAdd(&cellCount[c], 1u);
}

// Exclusive prefix sum over cellCount into cellStart.
//
// ONE workgroup, deliberately. WebGPU offers no cross-workgroup synchronisation
// inside a dispatch, so any multi-workgroup scan needs either a second dispatch
// or a spin on device memory; both are more machinery than this pass deserves.
// numCells is capped at 16384 at init (plife.ts asserts it), so this is 64
// chunks of 256 at the very worst, running once per substep against a force pass
// that touches hundreds of millions of pairs. It is not the bottleneck and it is
// not close.
var<workgroup> scanBuf: array<u32, 256>;
var<workgroup> carry: u32;

@compute @workgroup_size(256)
fn scanCells(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let numCells = g.gridW * g.gridH;

  if (tid == 0u) {
    carry = 0u;
  }
  workgroupBarrier();

  // Every barrier below sits in uniform control flow: `base`, `numCells` and
  // `offset` are identical across the workgroup, so both loops iterate the same
  // number of times for every invocation.
  var base = 0u;
  loop {
    if (base >= numCells) {
      break;
    }
    let idx = base + tid;
    var own = 0u;
    if (idx < numCells) {
      own = atomicLoad(&cellCount[idx]);
    }
    scanBuf[tid] = own;
    workgroupBarrier();

    // Hillis-Steele inclusive scan. O(n log n) work instead of O(n), which for
    // 256 elements is irrelevant and buys a shape with no bank-conflict tricks
    // and no downsweep to get wrong.
    var offset = 1u;
    loop {
      if (offset >= 256u) {
        break;
      }
      var addend = 0u;
      if (tid >= offset) {
        addend = scanBuf[tid - offset];
      }
      workgroupBarrier();
      if (tid >= offset) {
        scanBuf[tid] = scanBuf[tid] + addend;
      }
      workgroupBarrier();
      offset = offset * 2u;
    }

    // inclusive - own = exclusive, plus everything the earlier chunks held.
    // Threads past numCells contributed 0, so chunkTotal is correct even when
    // the last chunk is partial.
    let inclusive = scanBuf[tid];
    let chunkTotal = scanBuf[255];
    if (idx < numCells) {
      cellStart[idx] = carry + inclusive - own;
    }
    workgroupBarrier();
    if (tid == 0u) {
      carry = carry + chunkTotal;
    }
    workgroupBarrier();
    base = base + 256u;
  }
}

@compute @workgroup_size(256)
fn scatterParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= g.maxParticles) {
    return;
  }
  if (!isAlive(i)) {
    return;
  }
  let c = cellOf(wrapWorld(particles[i].pos, vec2f(g.worldW, g.worldH)));
  let slot = cellStart[c] + atomicAdd(&cellFill[c], 1u);
  // Belt and braces: cellStart is a prefix sum of counts taken in this same
  // substep from this same predicate, so slot < activeTotal <= maxParticles
  // always. The bound check costs nothing and turns an impossible race into a
  // dropped particle rather than a corrupted buffer.
  if (slot < g.maxParticles) {
    sortedIdx[slot] = i;
  }
}
