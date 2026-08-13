// tunnel — the warp pass.
//
// One fullscreen resample of the previous field: for each output pixel, work out
// where that pixel's light *came from* one step ago, fetch it, smear it along
// the flow, dim it and rotate its hue. Everything that reads as depth — the
// vanishing point, the sense that the rim is close and the middle is far, the
// beams accelerating as they come at you — is made here, out of a draw pass that
// is sixteen stretched blobs sitting still.
//
// The transform is written as an inverse map (destination → source) because that
// is the only direction a fragment shader can express: forward-mapping would
// need a scatter. It reads backwards from the effect, so a source radius
// *smaller* than the destination radius means the image moves OUTWARD.
//
// Composed in polar coordinates about the **movable centre** — not the frame
// centre, which is nebula's choice and the main structural difference between
// the two warps. The order is vortex → rotate → perspective-zoom, because each
// later term should act on the result of the earlier ones:
//
//   vortex   angle offset falling off as `vc/(r + vc)` — the classic vortex
//            profile, fastest at the core. This is what makes the near end of
//            the shaft twist while the far end streams straight, and it is a
//            different gesture from nebula's swirl: there the differential
//            rotation shears arms *into each other*, here it curls a radial
//            beam into a spiral without changing which sector it belongs to.
//   rotate   rigid angle offset on top, so the whole shaft precesses.
//   zoom     the radial map, and the heart of the visual:
//
//                f(r) = r · (1 − z − q·r)
//
//            `z` (θ's `zoomRate`, plus the breath, plus the impulse lurch) is a
//            uniform rate and `q` (`perspective`) is a rate that grows with
//            radius. Together they give a flow speed of `z·r + q·r²`: nearly
//            zero at the vanishing point, fastest at the rim. That gradient IS
//            the perspective of a shaft seen end-on — a point at depth t along a
//            cylinder projects to screen radius R/t, and moving along the axis
//            by δ maps r → r/(1 − δr/R), whose inverse is r/(1 + δr/R). The
//            form above is that, linearised for the small per-step δ this
//            substrate actually uses, with a uniform term added so the flow
//            never fully stops at the middle.
//
// The angular terms depend only on `r`, so they contribute nothing to the area
// Jacobian (an angle offset that is a function of radius is a shear). The
// translation to and from the movable centre is area-preserving. So the radial
// map is the whole of the energy bookkeeping — see the long note at the bottom.

/**
 * Radians per second of sim time for the autonomous breath.
 *
 * 0.42 rad/s is a ~15 s period. Not a θ slot, and that is a judgement rather
 * than an oversight: the *depth* of the breath is what changes how the visual
 * feels and it is on a slot (`breathe`); the period mostly changes how often it
 * happens, and anything under about 6 s stops reading as breathing and starts
 * reading as a wobble, while anything over about 40 s is indistinguishable from
 * the modulator's own slow drift on `zoomRate`. Fixed here it costs nothing and
 * keeps the slot table under 40.
 */
const BREATH_RATE: f32 = 0.42;

/**
 * Ceiling on the streak kernel's half-length, in screen heights.
 *
 * ## This constant, and the two below it, shipped 4× too large and that was the
 * ## whole reason the first build had no visible structure at all.
 *
 * The reasoning that produced 0.18 was: `streakMix` is a convex combination of
 * field samples, so its gain is exactly 1, so it cannot hurt. That is true about
 * *energy* and irrelevant to *structure* — an averaging kernel applied every
 * step is a diffusion, and diffusion is precisely the operator that destroys
 * detail while conserving mass. At the shipped 0.18 / 0.55 mix / 4.5 length the
 * kernel put taps ±0.03 to ±0.16 screen heights away sixty times a second; over
 * the field's ~30-step residence that is a random walk of ≈0.13 heights along
 * the flow and ≈0.05 across it, against source stamps 0.04 × 0.09. Every
 * feature in the image was averaged out of existence, and the measured result
 * was a smooth blue/cyan fog with faint concentric banding — the *mean* of the
 * structure that should have been there.
 *
 * The reach the smear was supposed to buy is not needed either, and that was the
 * second error in the same argument. Reach comes from the parcel *transit time*,
 * which is long precisely where the flow is slow: with the retuned rates a
 * parcel takes ~100 steps to cross the frame, so an arm is drawn by the flow
 * itself over a full memory. The smear's only remaining job is the honest one —
 * closing the gap between two consecutive stamps so a trail is a line and not a
 * row of beads.
 *
 * So this is now sized to *one step of flow*, not to a fraction of the frame:
 * 0.045 is about twice the largest stamp radius the table can produce at a
 * source ring, which is exactly the coverage "no beading" requires and no more.
 */
const STREAK_CLAMP: f32 = 0.045;

/**
 * How far the streak taps are offset *across* the flow, as a fraction of the
 * kernel's own length.
 *
 * A pure four-tap line along the flow direction is an anisotropic kernel with no
 * width at all, and over a few hundred accumulation steps that grows visible
 * one-texel-wide filaments — the field is an integrator, so any direction the
 * kernel does not touch never gets averaged. Offsetting alternate taps sideways
 * turns the line into a narrow zigzag with a real cross-section.
 *
 * 0.12, down from the 0.30 the first build shipped. Cross-flow spreading is the
 * component that costs the most and buys the least here: the flow is radial, so
 * "across the flow" is "around the tunnel", and blurring around the tunnel is
 * exactly the operation that merges one stem's angular sector into its
 * neighbour's. At 0.30 the accumulated cross-flow spread was ~0.05 heights,
 * which at a 0.2-radius source ring is 14° — half a sector's worth of smearing
 * *per residence*, on a partition whose whole job is to keep four hues apart.
 * 0.12 is still enough to stop the filamenting it exists for.
 *
 * The four (along, across) pairs below are centrally symmetric — (t, s) and
 * (−t, −s) are both present — which matters: an asymmetric kernel is a net
 * translation applied every step, i.e. a second undocumented drift term.
 */
const STREAK_FAN: f32 = 0.12;

/**
 * Four taps along the flow, averaged. `axis` and `fan` arrive in centred
 * (aspect-corrected) units and are converted to uv here, because `toUv` adds the
 * 0.5 origin and these are deltas rather than points.
 */
fn streakAt(uv: vec2f, axis: vec2f, fan: vec2f) -> vec3f {
  let inv = vec2f(1.0) / aspect();
  let a = axis * inv;
  let f = fan * inv;
  var s = textureSampleLevel(field, fieldSamp, uv + a * 0.75 + f, 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + a * 0.28 - f, 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv - a * 0.28 + f, 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv - a * 0.75 - f, 0.0).rgb;
  return s * 0.25;
}

@fragment
fn fsWarp(in: FsIn) -> @location(0) vec4f {
  // Polar about the movable vanishing point, not the frame centre. Everything
  // downstream is in this frame.
  let centre = vec2f(th(TH_centreX), th(TH_centreY));
  let p = toCentre(in.uv);
  let d = p - centre;
  let r = length(d);
  let ang = atan2(d.y, d.x);

  // The event lane, in the geometry. `pulse` is 0 at rest and rises with the
  // deposit envelope of whatever just fired (max over layers, capped at 2 by
  // `VizFxSim.updatePulse`), so a kick shoves the whole shaft and tightens the
  // vortex for the length of its decay. This is why the warp reads the impulse
  // state at all, and it is the drop's lurch.
  let pulse = clamp(g.pulse, 0.0, 4.0);

  // `z` is stored as a per-step RATE, never as a factor near 1 — see the long
  // note on `zoomRate` in tunnel.ts. The breath is added, not multiplied, so it
  // can carry the sum across zero and turn the fountain into a drain; the
  // impulse lurch is added for the same reason and can do it in the other
  // direction on a hard hit.
  let z = th(TH_zoomRate) + th(TH_breathe) * sin(g.time * BREATH_RATE) + th(TH_pulseZoom) * pulse;
  let q = max(th(TH_perspective), 0.0);

  // f(r)/r and f'(r), the two numbers the whole pass is built from.
  //
  // The clamps are unreachable at any θ the table can produce — at the hard
  // ceilings (z ≤ 0.05 + 0.03 + 0.12·2, q ≤ 0.12, r ≤ 1.75 for a corner with the
  // centre pushed to its own hard limit) `deriv` is still ≈ 0.20 and `scale`
  // ≈ 0.41, both comfortably inside. They are here so that an arithmetic slip
  // or a future widened bound cannot produce a folded map (deriv ≤ 0 means the
  // resample turns inside out) rather than because anything is expected to hit
  // them. Note both clamps can only make the Jacobian below *smaller* in the
  // regime that matters, so they cannot smuggle gain in: see the clamp argument
  // at the bottom.
  let scale = clamp(1.0 - z - q * r, 0.15, 2.0);
  let deriv = clamp(1.0 - z - 2.0 * q * r, 0.02, 2.0);

  // The vortex. `vc/(r + vc)` is 1 at the centre, 1/2 at r = vc and falls as 1/r
  // beyond — so the *linear* displacement it produces, `vortex·vc·r/(r + vc)`,
  // is bounded by `vortex·vc` at every radius. That bound is what lets the
  // stamp-continuity floor in draw.wgsl be a finite number rather than something
  // that has to chase the mod range around.
  let vc = max(th(TH_vortexCore), 1e-3);
  let twist = th(TH_vortex) * (vc / (r + vc)) * (1.0 + 0.8 * pulse);
  let ang2 = ang + th(TH_rotate) + twist;

  let srcP = centre + vec2f(cos(ang2), sin(ang2)) * (r * scale);
  let src = toUv(srcP);

  // The sampler is mirror-repeat, so a source coordinate pushed outside the
  // frame folds back inside instead of smearing the edge texel along the border
  // — which is what clamp-to-edge does, and over a few hundred accumulation
  // steps it builds a bright picture-frame that never decays. The `horizon` term
  // below is the other half of that defence: it makes sure the folded region has
  // nothing bright left in it to fold.
  var col = textureSampleLevel(field, fieldSamp, src, 0.0).rgb;

  // ── the streak: per-step motion blur, and NOTHING more ──────────────────────
  //
  // `mix(col, mean-of-four-taps, streakMix)` is a convex combination of field
  // samples: every coefficient is non-negative and they sum to 1, so the
  // operator's gain is exactly 1 in both the sup norm and (up to the mirrored
  // boundary) the L1 norm. It cannot manufacture light, and that is why it is
  // safe to have inside the loop.
  //
  // It is *not* safe to have a lot of. The first build treated this as the
  // visual's reach mechanism and shipped it four times too long, at more than
  // twice the mix; the gain-1 argument says nothing about structure, and an
  // averaging kernel applied sixty times a second is a diffusion. The measured
  // result was a featureless fog. See STREAK_CLAMP above for the full autopsy.
  //
  // What it is for, and all it is for: an arm is a row of stamps, one per step,
  // and where the flow carries a stamp further than the next stamp's own width
  // the trail beads into a dotted line. The stamp-size floor in draw.wgsl is the
  // primary defence and this is the secondary one — it closes the residual gap
  // without asking every source to grow. Sized in units of one step of flow for
  // exactly that reason.
  //
  // `p - srcP` is one step of travel at this pixel, in centred units — the true
  // local flow vector including the vortex and the rotation, not just the radial
  // part, so the smear follows the spiral rather than cutting across it.
  let mixAmt = clamp(th(TH_streakMix), 0.0, 1.0);
  if (mixAmt > 0.002) {
    var axis = (p - srcP) * max(th(TH_streakLength), 0.0);
    let alen = length(axis);
    if (alen > STREAK_CLAMP) {
      axis = axis * (STREAK_CLAMP / alen);
    }
    col = mix(col, streakAt(src, axis, vec2f(-axis.y, axis.x) * STREAK_FAN), mixAmt);
  }

  // ── the resample has to conserve light ──────────────────────────────────────
  //
  // This is a *gather*: each destination pixel copies the value it finds at
  // `src`. Total light therefore transforms as
  //
  //     ∫ c_dst d(dst) = ∫ c_src(s) / |det ∂src/∂dst| ds
  //
  // so conserving it means multiplying the sample by J = |det ∂src/∂dst|. Get
  // this wrong and the resample is not a transport, it is an amplifier: nebula
  // shipped without it and the loop gain became `(1 − fade)·zoom²` instead of
  // `1 − fade`, which at the ends of its own mod ranges was 1.021 — a loop that
  // gains 2% per step and diverges without bound. Measured there: mean field
  // luminance 719 against 56, with the radial profile *inverted*, i.e. a white
  // plate with no black left anywhere for exposure to recover.
  //
  // For a warp written as `src = centre + f(r)·(cos(a + g(r)), sin(a + g(r)))`
  // the determinant is
  //
  //     J = |f(r) · f'(r) / r|
  //
  // The angular term `g(r)` contributes nothing — an angle offset depending only
  // on radius is a shear, and shears are area-preserving — and a rigid rotation
  // and the translation to and from `centre` are both area-preserving too. So
  // with
  //
  //     f(r)  = r · (1 − z − q·r)
  //     f'(r) =      1 − z − 2q·r
  //
  //     J = |(1 − z − q·r) · (1 − z − 2q·r)|  =  scale · deriv
  //
  // which is why the two numbers computed above are the only two this needs.
  // Note `q` is part of `f`, so it is part of `f'`, and the factor of 2 is the
  // whole reason a perspective tunnel dims faster than a uniform zoom of the
  // same speed: `1 − J ≈ 2z + 3q·r`.
  //
  // ## The clamp, and why the loop gain is ≤ 1 − fade for EVERY θ
  //
  // Floored at 1 so the term can only ever dim. `z` may legally go negative —
  // the drain, which is half of what this visual is — and there the honest
  // Jacobian is greater than 1, which would put the loop gain back above 1 for
  // exactly the reason above with the sign flipped. Clamping is not a fudge: it
  // is the statement that a drain may *lose* light and may not *make* it.
  //
  // The clamp does a second job that is specific to this visual. In drain mode
  // material converges on the vanishing point, and with J clamped to 1 the
  // gather copies values inward unchanged rather than scaling them up, so a
  // parcel's value decays at exactly (1 − fade) per step however hard the flow
  // is compressing it. The hottest the field can ever get is therefore
  // `(peak per-step deposit) / fade`, the same bound any accumulator has, and it
  // does not depend on the convergence rate at all. Without the clamp the drain
  // would concentrate as `exp(2|z|/fade)`, which at the extremes of the two mod
  // ranges (z = −0.014, fade = 0.004) is a factor of 1100 on a single pixel.
  //
  // Putting the three loop terms together, for every θ the table can produce:
  //
  //     gain = (streak: convex, ≤ 1) · (clamp(1 − fade − horizon·r², 0, 0.9995)) · (J ≤ 1)
  //          ≤ 1 − fade      … with strict inequality everywhere r > 0
  //
  // At the *extremes*: fade's mod range bottoms at 0.004 (gain ≤ 0.996);
  // `horizon` bottoms at 0.003 and only ever subtracts; `streakMix` at its
  // ceiling of 0.9 is still a convex combination; `perspective` and `zoomRate`
  // at their ceilings only push J further below 1; at their floors (the drain)
  // J is clamped to exactly 1 and the bound is tight. There is no corner of the
  // table where the loop gain exceeds 1 − fade.
  let jacobian = clamp(scale * deriv, 0.0, 1.0);

  // The horizon: extra loss growing as r², i.e. a memory that shortens with
  // distance from the vanishing point. Art (the corners recede into black),
  // motion (material dissolves as it passes the camera rather than reaching the
  // edge at full brightness) and safety (the mirror-repeat fold at the frame
  // edge never has anything bright to fold) — see the slot's comment.
  let horizonLoss = max(th(TH_horizon), 0.0) * r * r;

  // Decay last, so the streak mixes light that has not yet been dimmed twice.
  // Clamped strictly under 1: at a total loss of exactly 0 the field is a
  // perfect integrator and any source left running saturates it into a white
  // plate within a minute, with no path back down. The θ bounds already stop
  // that; this stops a macro or an arithmetic slip from getting round it.
  col = col * (clamp(1.0 - th(TH_fade) - horizonLoss, 0.0, 0.9995) * jacobian);

  // The chroma lane. Applied to the FEEDBACK rather than to the draw, which is
  // what makes it accumulate: light stamped now is its layer's own colour, and
  // it walks around the hue circle over the following seconds as it is carried
  // down the shaft. That is the root-to-tip gradient along a beam that the
  // palette alone cannot produce, and it is why hue rotation lives in the
  // `matrix` group — it is the one lane here that says how the voices relate
  // rather than where they are.
  return vec4f(hueRotate(col, th(TH_chromaShift)), 1.0);
}
