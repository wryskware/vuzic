// nebula — the warp pass.
//
// One fullscreen resample of the previous field: for each output pixel, work out
// where that pixel's light *came from* one step ago, fetch it, soften it, dim it
// and rotate its hue. Everything the visual has that looks like structure —
// spiral arms, tunnels, the sense that the image is being pulled somewhere — is
// made here, out of an emitter field that is only twelve moving blobs.
//
// The transform is written as an inverse map (destination -> source) because
// that is the only direction a fragment shader can express: forward-mapping
// would need a scatter. It reads backwards from the effect, so: `zoom > 1`
// shrinks the source window toward the centre and therefore makes the image
// EXPAND outward, and `rotate` is subtracted from the apparent spin.
//
// Composed in polar coordinates about the frame centre, in the order
// swirl -> rotate -> zoom -> ripple -> drift, because each later term should act
// on the result of the earlier ones:
//
//   swirl    angle offset that FALLS OFF with radius, so the middle of the
//            frame twists faster than the edge. This is the whole difference
//            between a nebula and a pinwheel: a rigid rotation moves every arm
//            together and reads as the camera turning, while a differential one
//            shears the arms into each other and reads as the material flowing.
//   rotate   rigid angle offset on top, so the whole field precesses slowly.
//   zoom     radial scale. Just above 1 by default: a slow, endless outward
//            drift is what turns a stationary emitter into an arm.
//   ripple   a radial standing wave, so the outward flow is not perfectly
//            smooth and the arms develop knots.
//   drift    a constant translation. Small, and mostly there so a seed can put
//            the centre of the flow somewhere other than the middle of the frame.
//
// Every one of those is a θ slot, so the music moves all of them.

/**
 * Spatial frequency of the radial ripple, in cycles per screen height.
 *
 * Not a θ slot, and that is a judgement rather than an oversight: the *depth* of
 * the ripple is what changes how the image feels, and the frequency mostly
 * changes how many knots there are — which at any value above about 10 is a
 * texture rather than a structure. Fixed here it costs nothing and keeps the
 * slot table under 40.
 */
const RIPPLE_FREQ: f32 = 16.0;

/** Temporal frequency of the same wave, radians per second of sim time. */
const RIPPLE_SPEED: f32 = 1.9;

/** The window height the `blurRadius` slot's texel count is quoted at. */
const BLUR_REFERENCE_HEIGHT: f32 = 1080.0;

/**
 * A cheap symmetric blur at `radius` texels: centre plus four diagonals, equal
 * weights.
 *
 * Diagonals rather than the axis-aligned cross, because with linear filtering
 * four diagonal taps at r cover the neighbourhood more evenly than four axial
 * ones do — an axial cross leaves the corners entirely unsampled and the
 * accumulated field grows a faint plus-shaped anisotropy over a few hundred
 * steps, which is exactly the timescale this substrate runs on.
 *
 * The offset is scaled by the window height against a 1080-line reference, so
 * `blurRadius` describes a fraction of the *image* rather than a count of
 * device pixels. Without it this is the one θ slot whose meaning changes with the
 * window: on the author's 4K display the shipped value was doing a quarter of the
 * softening it does at 1080p, so the same mapping file produced a visibly
 * crisper, grainier nebula on the bigger screen — and, worse, a 1/3-size explorer
 * tile would have been judging a blur three times stronger than the candidate
 * would give you full-screen.
 */
fn blurAt(uv: vec2f, radius: f32) -> vec3f {
  let scale = max(g.res.y / BLUR_REFERENCE_HEIGHT, 0.25);
  let o = g.invRes * max(radius, 0.0) * scale;
  var s = textureSampleLevel(field, fieldSamp, uv, 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(o.x, o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(-o.x, o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(o.x, -o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(-o.x, -o.y), 0.0).rgb;
  return s * 0.2;
}

@fragment
fn fsWarp(in: FsIn) -> @location(0) vec4f {
  let p = toCentre(in.uv);
  let r = length(p);
  let ang = atan2(p.y, p.x);

  // The event lane, in the geometry. `pulse` is 0 at rest and rises with the
  // deposit envelope of whatever just fired, so a kick shoves the whole field
  // outward and adds a twist for the length of its decay. This is the cheapest
  // legible thing a screen-space visual can do with a transient, and it is why
  // the warp reads the impulse state at all.
  let pulse = clamp(g.pulse, 0.0, 4.0);

  // Both are stored as per-step *rates* (see the notes in nebula.ts), so the
  // factor is built here rather than held in θ. Floored well above 0 because it
  // divides a radius.
  let zoom = max(1.0 + th(TH_zoomRate) + th(TH_pulseZoom) * pulse, 0.25);
  let swirl = th(TH_swirl) * exp(-r * r * max(th(TH_swirlFalloff), 0.0)) * (1.0 + 0.6 * pulse);
  let ripple = th(TH_ripple) * sin(r * RIPPLE_FREQ - g.time * RIPPLE_SPEED);

  let a2 = ang + th(TH_rotate) + swirl;
  let r2 = r / zoom + ripple;
  let q = vec2f(cos(a2), sin(a2)) * r2 + vec2f(th(TH_driftX), th(TH_driftY));
  let src = toUv(q);

  // The sampler is mirror-repeat, so a source coordinate pushed outside the
  // frame by drift or ripple folds back instead of smearing the edge texel
  // across the border — which is what clamp-to-edge does, and over a few hundred
  // accumulation steps it builds a bright picture-frame that never decays.
  var c = textureSampleLevel(field, fieldSamp, src, 0.0).rgb;
  let mixAmt = clamp(th(TH_blurMix), 0.0, 1.0);
  if (mixAmt > 0.002) {
    c = mix(c, blurAt(src, th(TH_blurRadius)), mixAmt);
  }

  // ── the resample has to conserve light ──────────────────────────────────────
  //
  // This is a *gather*: each destination pixel copies the value it finds at
  // `src`. When `zoom > 1` the source window is smaller than the frame, so every
  // source texel is read by `zoom²` destination texels and the total light in the
  // field is multiplied by `zoom²` on every single step. The resample was
  // therefore not a transport, it was an amplifier.
  //
  // Compounded into the feedback loop that is catastrophic, because the loop gain
  // stops being `1 - fade`:
  //
  //     loop gain = (1 - fade) · zoom²   =   0.99 × 1.0035²  =  0.99693
  //
  // i.e. 326 steps of memory where `fade` promises 100, and at the ends of the
  // slots' own modulation ranges (fade 0.003, zoomRate 0.012) it is 1.021 — a
  // loop that gains 2% per step and diverges without bound. Measured on the
  // shipped defaults, 40 s from a cleared field: mean field luminance 719 with
  // this term, 56 with `zoomRate` at 0. A 12.9× difference, and the radial
  // profile came out *inverted* — 314 at the centre climbing monotonically to 849
  // at the rim — because light was being manufactured in flight and then piled up
  // against a boundary it cannot cross. That plate, tone-mapped, is the
  // "overexposed, everything goes white" the substrate shipped with: the frame's
  // 10th percentile sat at 403 units, so there was no black left in the image for
  // any amount of exposure to recover.
  //
  // Dividing by the map's area Jacobian is the fix, and it is one line because the
  // radial scale is the only area-changing term — the swirl is an angle offset
  // that depends only on radius (area-preserving), the rotation is rigid and the
  // drift is a translation. Now an expanding shell dims as it grows, which is both
  // what conserving light means and what expanding gas does.
  //
  // Floored at 1 so the term can only ever dim. `zoomRate` may legally go
  // negative — the documented inward-flow variant — and there the honest Jacobian
  // is greater than 1, which would put the loop gain back above 1 for exactly the
  // reason above with the sign flipped. A drain that concentrates light and loses
  // a little of it is a drain; a drain that manufactures it is the bug this
  // comment is about. Clamping here makes `1 - fade` a true ceiling on the loop
  // gain for every θ the table can produce.
  let jacobian = 1.0 / max(zoom * zoom, 1.0);

  // Decay last, so the blur mixes light that has not yet been dimmed twice.
  // Clamped strictly under 1: at a fade of exactly 0 the field is a perfect
  // integrator and any emitter left running saturates it into a white plate
  // within a minute, with no path back down. The θ bound already stops that; this
  // stops a macro or an arithmetic slip from getting round it.
  c = c * (clamp(1.0 - th(TH_fade), 0.0, 0.9995) * jacobian);

  // The chroma lane. Applied to the FEEDBACK rather than to the draw, which is
  // what makes it accumulate: light stamped now is the layer's own colour, and
  // it walks around the hue circle over the following seconds as it is warped
  // outward. That is the gradient from core to arm that the palette alone cannot
  // produce, and it is why hue rotation lives in the `matrix` group — it is the
  // one lane here that says how the voices relate rather than where they are.
  return vec4f(hueRotate(c, th(TH_chromaShift)), 1.0);
}
