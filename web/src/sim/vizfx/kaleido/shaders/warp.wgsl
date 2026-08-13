// kaleido — the warp pass.
//
// One fullscreen resample of the previous field: for each output pixel, work out
// where that pixel's light *came from* one step ago, fetch it, soften it, dim it
// and rotate its hue. In nebula that map is a flow. Here it is a flow composed
// with a **fold**, and the fold is the visual.
//
// ── the fold ─────────────────────────────────────────────────────────────────
//
// Every destination pixel's angle about the frame centre is reduced into one
// wedge and reflected, so the frame becomes 2n mirrored copies of a single
// sliver. The construction, derived rather than copied:
//
//   w  = 2π/n                              one rotational period
//   m  = a − w·floor(a/w + 0.5)            a mod w, centred:  m ∈ [−w/2, w/2)
//   af = |m|                               reflected:         af ∈ [0, w/2] = [0, π/n]
//
// Three properties, each of which is load-bearing:
//
//   * **It is continuous.** `m` is a sawtooth and `|m|` is therefore a triangle
//     wave: at the period boundary m jumps +w/2 → −w/2 and the absolute value is
//     w/2 on both sides. This is the entire difference between mirroring and
//     rotating. A rotation-only fold (`a mod w`, no reflection) is discontinuous
//     at every wedge boundary, so the image is a pinwheel with n visible cuts in
//     it; the mirror fold has no seams anywhere, which is why a real kaleidoscope
//     uses mirrors and not a rotating mask.
//   * **|d af / d a| = 1 almost everywhere.** The triangle wave has unit slope, so
//     the fold is a piecewise isometry — a modular translation composed with a
//     reflection, both of which preserve area exactly. It contributes NOTHING to
//     the Jacobian. Only the radial terms below do.
//   * **It is exactly 2n-to-1 onto [0, π/n].** Which is what makes the gather
//     conserve light rather than multiply it — see the long note further down.
//
// ── what is NOT here, and why ────────────────────────────────────────────────
//
// There is no rotation term and no angular twist. That is not an omission; it is
// forced. See the header of kaleido.ts for the argument in full, but in one line:
// at a fixed radius the only continuous measure-preserving map of the fundamental
// wedge onto itself is the identity, so *any* angular transport either breaks the
// symmetry or breaks the energy balance. The angular life of this visual lives in
// the draw pass, where the emitters advance in the pre-fold angle and the fold
// turns that into a sweep that bounces off the mirror lines — deposition rather
// than transport, and therefore free.
//
// So the flow here is purely radial, and it is the composition of the only two
// radial maps that are exactly area-preserving up to the same single factor:
//
//   zoom       a uniform radial scale. Just above 1 by default, so a petal
//              stamped in its lane is dragged outward into an arm over the
//              field's memory. The impulse lane adds to it, which is what makes a
//              kick a pump of all 2n arms at once.
//   hub drain  a shift in r², which pulls hardest at the centre and dies away as
//              1/r. Against the zoom's outward `z·r` it sets a stagnation radius
//              `√(C/2z)`: inside it material drains into the hub, outside it
//              streams to the rim. It gives the mandala a filled centre, and it
//              is FREE — see the Jacobian note, where C cancels exactly.
//
// ## What used to be here, and why it is not
//
// Revision 1 had a third term: a travelling radial ring wave, a per-radius
// modulation of the zoom. Its Jacobian is genuinely non-unit, so the conservation
// clamp turned every one of its contracting flanks into extra dimming — mean
// multiplier `1 − k/π` with `k = r·ripple·freq`, which measured 0.6%/step at the
// defaults and 2.7–5.5%/step at the modulation extremes against a `fade` of 1%.
// The field's memory therefore ran at 61 steps instead of 100 at rest and 15–27
// under modulation, the loss grew with radius (so the frame went half black at the
// edges) and both its slots were `structure` (so it got worse in a loud passage
// than a quiet one). Measured headless: a dense chorus came out darker, with twice
// the dead frame, than a quiet intro.
//
// The general result, which is why nothing like it came back: **there is no
// area-preserving radial ring wave.** `J = f·f'/r ≡ 1` integrates to `f² = r² + C`
// — a one-parameter family — so an oscillating radial map necessarily has |J| ≠ 1
// somewhere and necessarily pays for it under the clamp. `hubDrain` below IS that
// family, which is exactly why it costs nothing.

/** The window height the `blurRadius` slot's texel count is quoted at. */
const BLUR_REFERENCE_HEIGHT: f32 = 1080.0;

/**
 * The symmetry order, snapped and bounded.
 *
 * `round`, because a non-integer order does not tile the circle: the last partial
 * wedge meets the first at a hard discontinuity, which is a crack across the
 * rosette and not an in-between symmetry. The clamp is belt and braces — the CPU
 * has already hard-clamped θ to [2, 16] — and its floor is 2 rather than 1
 * because at n = 1 the "fold" is a single mirror line and the image is a
 * reflected photograph rather than a rosette.
 *
 * **This function must stay identical in draw.wgsl.** The two passes are separate
 * shader modules that share only `common.wgsl` and the generated θ prelude, so
 * the duplication is unavoidable; what is avoidable is the two disagreeing, and
 * they must not, because the draw deposits into a field the warp is about to
 * fold at the same order.
 */
fn foldOrder() -> f32 {
  return clamp(round(th(TH_foldCount)), 2.0, 16.0);
}

/**
 * `atan2` with the origin made safe. `atan2(0, 0)` is indeterminate in WGSL, and
 * the exact centre pixel is a real pixel; substituting +x there costs one select
 * and makes the fold total.
 */
fn angleOf(p: vec2f) -> f32 {
  let q = select(vec2f(1.0, 0.0), p, dot(p, p) > 1e-18);
  return atan2(q.y, q.x);
}

/** The fold. See the derivation at the top of this file. */
fn foldAngle(a: f32, n: f32) -> f32 {
  let w = TAU / n;
  return abs(a - w * floor(a / w + 0.5));
}

/**
 * A cheap symmetric blur at `radius` texels: centre plus four diagonals, equal
 * weights.
 *
 * Diagonals rather than the axis-aligned cross, because with linear filtering
 * four diagonal taps at r cover the neighbourhood more evenly than four axial
 * ones — an axial cross leaves the corners unsampled and the accumulated field
 * grows a faint plus-shaped anisotropy over a few hundred steps, which is exactly
 * the timescale this substrate runs on.
 *
 * The offset is scaled by the window height against a 1080-line reference, so
 * `blurRadius` describes a fraction of the *image* rather than a count of device
 * pixels — otherwise the same mapping file produces a crisper figure on a bigger
 * screen, and a 1/3-size explorer tile judges a blur three times stronger than
 * the candidate would give you full-screen.
 *
 * Note this cannot break the symmetry even though the kernel is axis-aligned in
 * texel space: it is evaluated at `src`, and every one of the 2n mirror partners
 * of a destination pixel computes the *same* `src`. They therefore take the same
 * five taps and get bit-identical results.
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
  let n = foldOrder();
  // The whole image is generated from this sliver of the previous frame.
  let af = foldAngle(angleOf(p), n);

  // The event lane, in the geometry. `pulse` is 0 at rest and rises with the
  // deposit envelope of whatever just fired, so a kick pumps the figure outward
  // and sharpens the rings for the length of its decay. This is the cheapest
  // legible thing a screen-space visual can do with a transient, and here it is
  // amplified for free: every one of the 2n arms surges at the same instant.
  let pulse = clamp(g.pulse, 0.0, 4.0);

  // Stored as a per-step *rate* (see the note in kaleido.ts), so the factor is
  // built here rather than held in θ. Floored well above 0 because it divides.
  let zoom = max(1.0 + th(TH_zoomRate) + th(TH_pulseZoom) * pulse, 0.25);

  // ── the radial map ──────────────────────────────────────────────────────────
  //
  //   f(r) = sqrt( r²/zoom² + C ),      C = hubDrain, boosted by the impulse lane
  //
  // Composed in r² rather than in r, which is what makes it free: differentiating
  // f² = r²/zoom² + C gives 2·f·f' = 2r/zoom², i.e. f·f' = r/zoom² **for every C**.
  // So the hub term contributes nothing at all to the Jacobian below and can be
  // any amplitude without touching the loop gain. See the note there for the
  // explicit area check at the origin, which is the one place the derivative form
  // could have hidden a singularity and does not.
  //
  // C > 0 means the source is further out than the destination, i.e. material
  // moves inward; the term decays as C·zoom/(2r), so it is a hub control and
  // nothing else. The impulse boost is on the same lane as `pulseZoom` but pulls
  // the other way, so a hit both pumps the rim outward and yanks the hub in — a
  // squeeze rather than a uniform shove, which is much more visible on a figure
  // whose centre is its focal point.
  let drain = max(th(TH_hubDrain), 0.0) * (1.0 + 0.8 * pulse);
  let r2 = sqrt(max(r * r / (zoom * zoom) + drain, 0.0));

  let src = toUv(vec2f(cos(af), sin(af)) * r2);

  // The sampler is mirror-repeat, so a source coordinate pushed outside the frame
  // folds back instead of smearing the edge texel along the border — which is what
  // clamp-to-edge does, and over a few hundred accumulation steps it builds a
  // bright picture-frame that never decays. It matters more here than in nebula:
  // the fundamental wedge is anchored at the +x axis, so the frame's corners at
  // r > 0.89 read from source points beyond the right-hand edge on every step.
  var c = textureSampleLevel(field, fieldSamp, src, 0.0).rgb;
  let mixAmt = clamp(th(TH_blurMix), 0.0, 1.0);
  if (mixAmt > 0.002) {
    c = mix(c, blurAt(src, th(TH_blurRadius)), mixAmt);
  }

  // ── the resample has to conserve light ──────────────────────────────────────
  //
  // This is a *gather*: each destination pixel copies the value it finds at
  // `src`. Total light transforms as
  //
  //     ∫ c_dst d(dst)  =  ∫ c_src(s) / |det ∂src/∂dst| ds
  //
  // so to conserve, the sampled value must be multiplied by the Jacobian
  // J = |det ∂src/∂dst|. Nebula shipped without it and the loop gain was
  // (1 − fade)·zoom² instead of (1 − fade) — 0.997 against a promised 0.99, and
  // at the ends of the modulation ranges above 1, i.e. a feedback loop that
  // manufactures light without bound. Measured there: mean field luminance 719
  // with the term missing against 56 with it, and the radial profile *inverted*.
  //
  // ## The two pieces of J, and why the fold is not one of them
  //
  // In polar form about the centre — src = (f(r), a2(r, a)) with ∂f/∂a = 0 — the
  // area element gives
  //
  //     J = (f(r)/r) · |f'(r)| · |∂a2/∂a|
  //
  // **The angular factor is exactly 1.** a2 = af = foldAngle(a), whose slope is
  // ±1 everywhere except on a measure-zero set of mirror lines: the fold is a
  // modular translation composed with a reflection, and both are isometries of
  // the plane. So the fold is area-preserving piecewise, |det| = 1 exactly, and it
  // contributes nothing here. (This is also why there is no rotation or twist term
  // to account for — see the top of this file for why they cannot exist at all.)
  //
  // **The radial factor is the whole of it, and it is one number.** With
  // f(r) = sqrt(r²/zoom² + C):
  //
  //     f² = r²/zoom² + C
  //     2·f·f' = 2r/zoom²      ⇒   f·f' = r/zoom²
  //     J = f·f'/r = 1/zoom²                        — independent of C
  //
  // which is nebula's term exactly. The hub drain is genuinely free: no clamp of
  // its own, no amplitude limit, no interaction with the loop gain.
  //
  // The derivative form could in principle hide a singularity at the origin, so
  // here is the explicit area check. The destination disc [0, ε] maps to the
  // source annulus [√C, √(ε²/zoom² + C)], whose area is π(ε²/zoom² + C − C) =
  // πε²/zoom². Destination area πε². Ratio 1/zoom² ✓. So the hub is filled by
  // spreading a thin annulus over a disc — total conserved, no void, no
  // concentration blow-up, and no black spot at r = 0 to clamp away.
  //
  // ## The other half: the fold gathers 2n destinations from one source wedge
  //
  // The local Jacobian is not the whole accounting, because this map is *not*
  // injective — every source point in [0, π/n] is read by 2n destination pixels.
  // That looks like a 2n-fold amplifier and is not one, because the field it
  // gathers from is already 2n-fold symmetric:
  //
  //     ∫_frame c(fold(a)) da  =  2n · ∫_wedge c(u) du  =  ∫_frame c(u) du
  //
  // where the last equality is the definition of the field being symmetric. So the
  // fold's contribution to the loop gain is exactly 1, not 2n. And the field IS
  // symmetric, structurally and at every step: the warp's output is a function of
  // `af` alone, so whatever it was handed, what it *writes* is symmetric about the
  // fixed axes — and the draw pass, which is the only other writer, also draws in
  // the folded domain.
  //
  // The one moment the assumption does not hold is the first step after something
  // asymmetric is put in the field: a section-boundary injection (chassis.wgsl
  // stamps blotches anywhere), a snapshot restore, or a change of `foldCount`.
  // There the gain is bounded by 2n × (the fraction of the light that happens to
  // lie in the wedge) and can exceed 1 for exactly one step. It cannot compound,
  // because the output of that step is symmetric by construction — the fold is
  // idempotent on its own range. A one-step bounded flash that then dissolves over
  // the field's memory is precisely what an order change should look like, so this
  // is a feature with a proof attached rather than a tolerated defect.
  //
  // ## The clamp, and the invariant it buys
  //
  // Floored at 1 so the term can only ever dim. Exactly one way J can exceed 1
  // now: `zoomRate` may legally go negative (the documented inward-flow variant,
  // where the honest Jacobian 1/zoom² is above 1). There the honest transform
  // would *amplify*, and amplifying inside a feedback loop is the bug this comment
  // exists about — a drain that concentrates light and loses a little of it is a
  // drain; a drain that manufactures it is not.
  //
  // Note what this clamp costs now, because in revision 1 it cost a great deal.
  // `max(zoom·zoom, 1.0)` binds only when zoom < 1, i.e. only on the small negative
  // part of `zoomRate`'s range, and it is uniform over the frame when it does. The
  // ring wave that used to sit here had a J oscillating about 1 with a
  // radius-growing amplitude, so the clamp bit over half the frame on every step
  // and threw away 0.6–5.5% of the field per step — several times `fade` itself,
  // worse at the rim, and worse the harder the music modulated. That is gone: the
  // clamp is once again the rare guard it is meant to be rather than a silent
  // second decay term.
  //
  // The invariant, for every θ the table can produce:
  //
  //     loop gain = (1 − fade) · J  ≤  1 − fade  <  1
  //
  // at the extremes as well as the defaults. Worst case on the hard bounds:
  // zoomRate ∈ [−0.012, 0.04], pulseZoom ∈ [0, 0.2] and `g.pulse` is capped at 2
  // by the chassis, so zoom ∈ [0.988, 1.44] and J = 1/max(zoom², 1) ∈ (0.48, 1].
  // `hubDrain` does not appear — it cancels exactly, at any amplitude. `fade` is
  // hard-bounded to [0.0005, 0.5] and read as a loss below, and none of `fade`,
  // `zoomRate` or `pulseZoom` carries a macro, so nothing outside θ can move any
  // of this.
  let jacobian = 1.0 / max(zoom * zoom, 1.0);

  // Decay last, so the blur mixes light that has not yet been dimmed twice.
  // Clamped strictly under 1: at a fade of exactly 0 the field is a perfect
  // integrator and any emitter left running saturates it into a white plate within
  // a minute, with no path back down. The θ bound already stops that; this stops a
  // macro or an arithmetic slip from getting round it.
  c = c * (clamp(1.0 - th(TH_fade), 0.0, 0.9995) * jacobian);

  // The chroma lane. Applied to the FEEDBACK rather than to the draw, which is
  // what makes it accumulate: light stamped now is its stem's own colour, and it
  // walks around the hue circle over the following seconds as the radial flow
  // carries it outward. That base-to-tip gradient along every arm is what the
  // palette alone cannot produce, and it is why hue rotation lives in the `matrix`
  // group — it is the one lane that says how the voices relate rather than where
  // they are.
  return vec4f(hueRotate(c, th(TH_chromaShift)), 1.0);
}
