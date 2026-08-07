// Auto-exposure. One workgroup, 4096 strided samples of the HDR frame, reduced
// in shared memory, closed into a first-order loop whose state lives in a 16-byte
// storage buffer the compositor reads back on the *next* frame.
//
// Measuring the already-gained image makes this a multiplicative controller:
// gain *= (target / measured)^alpha. Stable for any alpha in (0,1], and it needs
// no readback to the CPU, so nothing ever stalls the queue.

struct AutoState {
  gain: f32,
  mean: f32,
  init: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> p: Post;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> st: AutoState;

const WG: u32 = 256u;
const GRID: u32 = 64u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn measure(@builtin(local_invocation_index) li: u32) {
  let dims = vec2i(textureDimensions(src));
  let hi = dims - vec2i(1);
  let total = GRID * GRID;

  var sum = 0.0;
  var i = li;
  loop {
    if (i >= total) { break; }
    let gx = (f32(i % GRID) + 0.5) / f32(GRID);
    let gy = (f32(i / GRID) + 0.5) / f32(GRID);
    let c = clamp(vec2i(vec2f(gx, gy) * vec2f(dims)), vec2i(0), hi);
    sum = sum + luma(textureLoad(src, c, 0).rgb);
    i = i + WG;
  }
  partial[li] = sum;
  workgroupBarrier();

  var s = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      partial[li] = partial[li] + partial[li + s];
    }
    workgroupBarrier();
    s = s / 2u;
  }

  if (li != 0u) { return; }

  let mean = partial[0] / f32(total);

  // Switched off, or nothing on screen yet: hold at unity and stay "uninitialised"
  // so the first frame that *does* have content snaps instead of creeping.
  if (p.autoEnabled <= 0.5 || mean <= 1e-6) {
    st.gain = select(1.0, st.gain, p.autoEnabled > 0.5 && st.init > 0.5);
    st.mean = mean;
    st.init = select(0.0, st.init, p.autoEnabled > 0.5);
    return;
  }

  let gain = select(1.0, st.gain, st.init > 0.5);
  let ratio = p.autoTarget / mean;
  // First measurement snaps, so the opening frame is not a fade-in from the
  // wrong exposure; after that it creeps with the configured time constant.
  let alpha = select(1.0, 1.0 - exp(-p.dt / max(p.autoTau, 0.01)), st.init > 0.5);

  st.gain = clamp(gain * pow(ratio, alpha), p.autoMin, p.autoMax);
  st.mean = mean;
  st.init = 1.0;
}
