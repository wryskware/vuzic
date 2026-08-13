// The two passes that belong to the chassis rather than to a visual: the
// composite into the shared HDR chain, and the section-boundary injection.
//
// Neither reads a `TH_*` constant, which is the test of whether a pass belongs
// here — anything that names a parameter is the visual's.

// ── composite ─────────────────────────────────────────────────────────────────
//
// Field -> the post chain's HDR surface, in linear light, with scene exposure
// and the adapted gain applied and nothing else. No tone map, no gamma, no
// clamp: everything display-referred happens in the grade pass at the end of the
// post chain, exactly as in physarum's compositor.
//
// Exposure is applied HERE and deliberately not in the draw pass. The field is
// an accumulation with a memory of `1/(1 - decay)` steps, so exposure baked into
// the draw would take that long to take effect — dragging the slider would feel
// like a lag rather than a gain. Applying it on the way out makes it immediate,
// and it also puts it upstream of the surface auto-exposure measures, which is
// what keeps `autoTarget` meaning what it says.

@fragment
fn fsComposite(in: FsIn) -> @location(0) vec4f {
  let c = textureSampleLevel(field, fieldSamp, in.uv, 0.0).rgb;
  let gain = max(autoState[0], 1e-4);
  return vec4f(max(c, vec3f(0.0)) * g.exposure * gain, 1.0);
}

// ── inject ────────────────────────────────────────────────────────────────────
//
// `ModTarget.partialReseed`, which the mapping layer fires on a section
// boundary. Physarum re-scatters agents and plife respawns a fraction of the
// pool; neither gesture exists here, because a feedback visual has no discrete
// matter to re-scatter — the field IS the world and it is a picture.
//
// So the honest analogue is done to the picture: fade what is there in
// proportion to the boundary fraction, and stamp seeded blotches of new light
// into it. A chorus arriving therefore looks like the nebula being partly
// cleared and re-seeded rather than like nothing happening, and it is
// deterministic in (seed, segment index) like every other seeded thing here.
//
// The fade coefficient is 0.55 rather than 1.0: `respawnFraction` ships at 0.12,
// which under a 1.0 coefficient would be a 12% dip nobody can see, and under a
// coefficient large enough to be visible at 0.12 a hand-set 0.5 would black the
// screen. 0.55 puts the shipped value at a ~7% dip that the blotches land on top
// of, and the top of the slider at a hard but survivable half.

const INJECT_BLOTCHES: u32 = 7u;

@fragment
fn fsInject(in: FsIn) -> @location(0) vec4f {
  let a = clamp(g.injectAmount, 0.0, 1.0);
  var c = textureSampleLevel(field, fieldSamp, in.uv, 0.0).rgb * (1.0 - 0.55 * a);

  let p = toCentre(in.uv);
  for (var i = 0u; i < INJECT_BLOTCHES; i = i + 1u) {
    let h = hash3(g.seed, g.injectKey, i + 1u);
    // Placed inside the middle of the frame rather than anywhere: a blotch half
    // off the edge is a bright rim the warp then smears along the border, which
    // reads as a bug rather than as a section change.
    let cx = (rand01(h) - 0.5) * 0.72 * aspect().x;
    let cy = (rand01(pcg(h)) - 0.5) * 0.72;
    let radius = 0.045 + 0.075 * rand01(pcg(pcg(h)));
    let layer = layers[min(pcg(h ^ 0x51ed2701u) % max(g.layerCount, 1u), g.layerCount - 1u)];
    let d = length(p - vec2f(cx, cy)) / radius;
    // Soft-shouldered rather than gaussian: the blotch is going to be warped and
    // smeared for the next several seconds, and a hard core survives that as a
    // dot while a soft one survives it as a cloud.
    let k = exp(-d * d * 1.6) * 0.55 + exp(-d * 1.4) * 0.45;
    c = c + max(layer.color.rgb, vec3f(1e-3)) * k * a * 1.6;
  }
  return vec4f(c, 1.0);
}
