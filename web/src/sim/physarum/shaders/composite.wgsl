// Phase 7: this pass is now HDR-only. It sums the K trail layers into an
// rgba16float target in linear light and stops there — no tone map, no gamma,
// no clamp. Everything display-referred happens in the grade pass at the end of
// the post chain, which is why a flash can be 20x over white here without the
// image turning into a plate.

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> species: array<Species>;
@group(0) @binding(2) var trail: texture_2d_array<f32>;
@group(0) @binding(3) var soil: texture_2d<f32>;
// [gain, mean, initialised, pad] — written by the post chain's measure pass one
// frame ago. Read-only here; auto-exposure is applied *before* the bloom
// threshold so a loud section does not simply push the whole screen over it.
@group(0) @binding(4) var<storage, read> autoState: array<f32>;
@group(0) @binding(5) var prevSamp: sampler;
@group(0) @binding(6) var prevFrame: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let q = corners[vi];
  var out: VSOut;
  out.pos = vec4f(q, 0.0, 1.0);
  out.uv = vec2f((q.x + 1.0) * 0.5, (1.0 - q.y) * 0.5);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let gw = f32(g.gridW);
  let gh = f32(g.gridH);
  let gridAspect = gw / gh;
  let viewAspect = max(g.viewportW, 1.0) / max(g.viewportH, 1.0);

  // cover: the grid always fills the canvas, cropping on the long axis
  var scale = vec2f(1.0);
  if (viewAspect > gridAspect) {
    scale.y = gridAspect / viewAspect;
  } else {
    scale.x = viewAspect / gridAspect;
  }
  let uv = vec2f(0.5) + (in.uv - vec2f(0.5)) * scale;
  let p = uv * vec2f(gw, gh);
  let gi = vec2i(i32(g.gridW), i32(g.gridH));

  var acc = vec3f(0.0);
  for (var k = 0u; k < g.speciesCount; k = k + 1u) {
    let s = species[k];
    let v = max(trailBilinear(trail, p, gi, i32(k)), 0.0);
    acc = acc + s.color.rgb * s.color.a * v;
  }

  var col = acc * g.exposure * max(autoState[0], 1e-4);

  // Soil as a dim ember underlay, added *after* exposure so the ground holds
  // still while the light moves. sqrt because soil spends most of its life near
  // the bottom of its range. This is the one place structure from track-scale
  // memory reaches the image without the debug view on.
  let sv = soilBilinear(soil, p, gi);
  if (g.soilTint > 0.0) {
    col = col + vec3f(g.soilTintR, g.soilTintG, g.soilTintB) * (g.soilTint * sqrt(sv));
  }

  // Render-domain feedback: last frame's *graded-input* HDR, decayed. Purely a
  // look; the trail field is untouched, so this cannot change how the world grows.
  //
  // Both constants are applied once per RENDERED frame, so they are per-frame
  // quantities and their unit is this display's refresh rate. The CPU hands them
  // over already corrected: `writeGlobals` raises the authored "per 60 Hz frame"
  // numbers to the frame's length in 60 Hz frames, so a 240 Hz monitor receives
  // the fourth root and the accumulated effect over a second is the same
  // everywhere. Nothing here has to know the frame rate.
  if (g.feedbackAmount > 0.0) {
    let z = max(g.feedbackZoom, 1e-3);
    let fuv = vec2f(0.5) + (in.uv - vec2f(0.5)) / z;
    col = col + textureSampleLevel(prevFrame, prevSamp, fuv, 0.0).rgb * g.feedbackAmount;
  }

  // Debug view: soil as an ochre field with the live trails kept as a faint ghost,
  // so you can see whether the networks are sitting on their own history or
  // wandering off it.
  if (g.soilView > 0.5) {
    col = vec3f(0.98, 0.66, 0.28) * sqrt(sv) + col * 0.25;
  }
  return vec4f(col, 1.0);
}
