// tunnel — the draw pass.
//
// Additive, on top of the field the warp pass just resampled. Everything here is
// *new* light; nothing here has memory. That split is the whole architecture:
// this pass is a handful of stationary sources, and the warp is what turns them
// into a shaft.
//
// Two things are drawn.
//
//   emitters  K layers × `emitPerLayer` sources, each pinned inside its layer's
//             angular SECTOR of the tunnel and sitting at its layer's own
//             distance from the vanishing point. A layer IS a stem, so "the
//             vocal entered" is one quadrant of the shaft lighting up, and the
//             beams it is already trailing brighten with it.
//   rings     one shockwave per live impulse splash, drawn in the tunnel's own
//             polar space so it reads as a front travelling down the shaft
//             rather than as a flat circle pasted on the frame.
//
// A fullscreen pass rather than instanced quads, deliberately, for nebula's
// reasons: the rings are screen-sized by the time they have expanded, and the
// emitters are sixteen gaussians, which is cheaper per pixel than sixteen draws'
// worth of state changes. It is also the milkdrop idiom the family is named for.
//
// ## Why the sources do not orbit
//
// Nebula's emitters orbit the frame centre; these oscillate inside a wedge. That
// is forced by the sector partition rather than chosen. Layer k owns the wedge
// centred on `(k + ½)·τ/K`, and the warp preserves that partition (radial
// transport does not change an angle, and the two terms that do move every
// sector together). An emitter that *orbited* would leave its own wedge within a
// few seconds, and wrapping it at the boundary would be a visible teleport once
// per lap. An oscillation has no wrap. Each source carries its own phase and
// rate multiplier out of the seeded constellation, so four sources in one wedge
// wag out of phase and read as a bundle of vanes breathing rather than as a
// metronome — and a reroll redistributes those phases and the within-wedge
// depths, so the same table is a different creature.

/**
 * Unit conversion between "θ gains authored as order-1 numbers" and "HDR units
 * in the accumulated field". The one empirical constant in this visual.
 *
 * It has to exist because the two quantities are not in the same units and the
 * ratio between them is not a free choice: the field's brightness is the
 * per-step light integrated over a source's *area* and multiplied by the field's
 * effective memory, so it falls as the square of `coreSize` and rises with
 * `1/(total per-step loss)`. Authoring around it would mean either `emitGain`
 * sliders whose useful values are in the hundreds or a scene exposure two
 * decades away from the other substrates' — both push an arbitrary number
 * somewhere a human has to look at it.
 *
 * ## The arithmetic, calibrated against nebula rather than guessed
 *
 * Nebula's `DRAW_SCALE` is 4.0 and is *measured* — it puts the adapted
 * auto-exposure gain near 1× on a playing track at scene exposure 0.1. So the
 * honest way to pick this one is to hold the product (light per step × effective
 * memory) equal, which cancels most of the modelling error in both estimates.
 *
 * Light deposited per step, in (screen-height)² × gain units. For a stamp
 * `exp(-d²·3) + halo·0.22·exp(-d·1.7)` with `d` measured in units of `size`
 * along one axis and `size·beam` along the other, the integral is
 * `(π/3 + 0.478·halo)·size²·beam` ≈ `1.24·size²·beam` at halo 0.4.
 *
 *   nebula   Σ over 4 layers of 4 emitters × 1.25 × emitGain × size²   (beam = 1)
 *            cores 0.035/0.014/0.020/0.028, gains 1.00/1.35/1.15/0.90,
 *            size ≈ core × 1.14 at typical energy       →  Σ ≈ 0.0133
 *   tunnel   same shape with beam = 2.4 and cores 0.030/0.013/0.020/0.026,
 *            gains 1.00/1.40/1.15/0.85                  →  Σ ≈ 0.0336
 *
 * Effective memory is 1 / (total per-step loss), and the totals are NOT the same
 * because this warp loses light to the expansion as well as to `fade`:
 *
 *   nebula   fade 0.010 + (1 − J) ≈ 0.007 (zoom 1.0035, uniform)   → M ≈ 59
 *   tunnel   fade 0.012 + horizon·⟨r²⟩ ≈ 0.007 + (1 − J) ≈ 0.041,
 *            where 1 − J ≈ 2z + 3q·⟨r⟩ = 0.008 + 3(0.02)(0.55)     → M ≈ 17
 *
 * The tunnel's memory is 3.5× shorter and its per-step light is 2.5× larger, so
 * the two nearly cancel:
 *
 *   DRAW_SCALE = 4.0 × (0.0133 × 59) / (0.0336 × 17) = 4.0 × 0.785 / 0.571 = 5.5
 *
 * Rounded to 6.0 — half a step up rather than down, because the estimate above
 * uses the *mean* radius for the Jacobian loss and the sources all sit inside
 * r = 0.6 where that loss is smaller than the mean, so the true memory near the
 * source rings is a little longer than 17 and the true equilibrium a little
 * higher; but the frame's *mean* is what auto-exposure measures and the mean is
 * dominated by the outer half. 6.0 splits that difference.
 *
 * Chosen as the *geometric* centre of the field's level over a whole track, not
 * as the value that suits one bar — the same call nebula's comment makes and the
 * other half of the `autoMinGain`/`autoMaxGain` argument in config.ts. This
 * visual's dynamic span should be somewhat narrower than nebula's measured 32×,
 * because `fade` is only about a quarter of the total loss here, so modulating
 * it swings the memory far less than it does there.
 *
 * If the core sizes, `beam`, `fade` or `perspective` defaults move by a lot,
 * this is the number that has to follow them. The adapted gain sitting at either
 * rail is the symptom.
 *
 * ## Why it is 9.5 and not the 6.0 the first build shipped
 *
 * 6.0 was **measured correct** — adapted gain 1.96 (dense) and 3.94 (quiet)
 * against rails of 0.01…64, with 0% blown pixels. Nothing about the exposure
 * chain was wrong and nothing about it has been re-decided here. What changed is
 * the amount of light the pass *deposits*, which the structure fix cut on
 * purpose, so this constant follows it to hold the same operating point:
 *
 *   core sizes  .030/.013/.020/.026 → .026/.012/.018/.022   Σ gain·core²
 *                                     2.17e-3 → 1.66e-3      × 0.766
 *   beam        2.4 → 2.0                                    × 0.833
 *   halo        0.40 → 0.26  (stamp integral 1.238 → 1.171)  × 0.946
 *   ⇒ emitter light per step                                 × 0.604
 *
 *   effective memory  1/(fade + horizon·⟨r²⟩ + (1 − J))
 *     was  0.012 + 0.018(0.347) + [2(0.004) + 3(0.020)(0.55)] = 0.0600 → 16.7
 *     now  0.010 + 0.009(0.347) + [2(0.0025) + 3(0.030)(0.55)] = 0.0676 → 14.8
 *                                                              × 0.886
 *
 *   ⇒ emitters:  0.0422×16.7 = 0.705   →   0.0255×14.8 = 0.377
 *   ⇒ vanishing glow (new, and it sits at the stagnation point so its residence
 *     is the full 1/fade ≈ 67 steps rather than the arms' ~15):      + 0.080
 *   ⇒ total                             0.705   →   0.457      × 0.648
 *
 *   DRAW_SCALE = 6.0 / 0.648 = 9.3  →  9.5
 *
 * So the expectation is that the adapted gain lands back near 2 (dense) / 4
 * (quiet), i.e. exactly where it was measured to be right. If the arithmetic is
 * off by the ±50% these estimates are worth, the gain lands somewhere in 1.3…4.4
 * — still mid-rail, still no blown pixels. This is one constant and it is the
 * only thing to move if the re-measurement disagrees.
 */
const DRAW_SCALE: f32 = 9.5;

/**
 * Fixed filament depth and scale. Not θ slots, for the reason nebula fixes its
 * ripple frequency: the slot table is already at 38 of the 40 this family
 * allows, and of the three numbers a filament term has (depth, scale, drift)
 * only the depth changes how the image *feels* — and here even that is a small
 * effect, because a stamp already stretched by `beam` is broken into strands
 * lengthwise whatever the depth is.
 *
 * The noise is sampled in coordinates relative to the **vanishing point**, not
 * the frame, so the texture travels with the centre instead of sliding across
 * the sources whenever the composition moves. `0.3 + 1.4·n` has mean 1.0 for
 * `fbm2`'s mean of 0.5, so the depth does not change the pass's total light
 * output and therefore does not fight the DRAW_SCALE arithmetic above.
 */
const FILAMENT: f32 = 0.45;
const FILAMENT_SCALE: f32 = 6.5;

/**
 * The **far end of the shaft**: a soft glow each layer deposits at the vanishing
 * point, sized as a fraction of that layer's own source radius.
 *
 * ## Why this exists, and why it is not a slot
 *
 * A fountain flow has no inflow to its own centre. Material at radius r is
 * gathered from `f(r) < r`, which was gathered from further in, and so on down
 * to r = 0 where there is nothing — so in a pure outward tunnel **the inner disc
 * empties over 1/fade steps and stays empty**. That is not a tuning error, it is
 * what the map says, and the first build measured it exactly: a black void
 * occupying ~40% of the quiet frame and 15.6% black pixels, because with one
 * stem loud the innermost *active* source was far out and everything inside it
 * had drained away.
 *
 * The honest fix is the thing a real shaft actually shows. Looking down a
 * tunnel, the far end is not a hole — it is a small bright disc where the whole
 * length of the tunnel is compressed into a few degrees of view. So each layer
 * puts a fraction of its light there, and three good things follow:
 *
 *   - The centre can never be empty, in any θ, for any stem configuration. The
 *     fix is structural rather than a default that modulation can undo.
 *   - The image gains a focal point. The first build had none, which is half of
 *     why it read as a gradient rather than as a place.
 *   - It is a per-stem cue in its own right: all four layers deposit at the same
 *     point, so `layerBlend` hands the vanishing point to whichever stem is
 *     loudest. The far end of the tunnel *changes colour with the mix*.
 *
 * Not a θ slot because the slot table is at 38 of the family's 40 and because
 * this is the visual's identity rather than a knob — a tunnel whose far end can
 * be switched off is not a tunnel. Both numbers still ride the existing knobs:
 * the size scales with `sourceRadius` (so `scale` and the modulator move it) and
 * the light scales with `emitGain`, the energy lane and the impulse flash along
 * with everything else in the layer.
 *
 * `VANISH_SIZE` 0.22 of the source radius means a layer pushed far out grows a
 * correspondingly wider far end, which is what makes the fix self-correcting: it
 * covers more of the hole exactly when the hole is bigger. `VANISH_GAIN` 0.03 is
 * small because material at the stagnation point barely moves and therefore
 * accumulates its *full* memory — ~67 steps against ~15 for an arm — so 3% of a
 * source's peak deposit equilibrates at roughly 60% of a source knot's peak
 * brightness. Twice this and the four glows stack into a blown white dot.
 *
 * Soft-shouldered (a gaussian plus a wider exponential) rather than a plain
 * gaussian, for the reason the chassis's inject blotch gives: a hard core
 * survives being warped for several seconds as a dot, a soft one survives it as
 * a cloud.
 */
const VANISH_SIZE: f32 = 0.22;
const VANISH_GAIN: f32 = 0.03;

/**
 * Ring travel per unit of the splash's own push, in units of its base radius.
 *
 * There is no `ringExpand` θ slot here (see the note on `ringGain` in
 * tunnel.ts): the expansion is already authored per event kind in the impulse
 * workbench and arrives as `splashes[i].params.x`. 1.2 reproduces nebula's
 * shipped geometry — the snare (push 9 → params.x 1.5) travels ~0.36 screen
 * heights over its envelope against a base radius of 0.2 — so the two visuals
 * answer the same event at the same physical size.
 */
const RING_TRAVEL: f32 = 1.2;

/** Radians per second of sim time for the autonomous breath. Must match warp.wgsl. */
const BREATH_RATE: f32 = 0.42;

/** Rec.709, the same weights the post chain's `luma()` uses. */
fn lumaOf(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fsDraw(in: FsIn) -> @location(0) vec4f {
  let centre = vec2f(th(TH_centreX), th(TH_centreY));
  let p = toCentre(in.uv);
  let rel = p - centre;

  // ── the flow, recomputed here ───────────────────────────────────────────────
  //
  // warp.wgsl and this pass are separate shader modules, so the four lines below
  // are a deliberate duplicate of the ones there rather than a shared helper.
  // They exist here for two jobs that both need to know how fast the tunnel is
  // running *this step*: the stamp-continuity floor, and which way a shockwave
  // front is travelling. Keeping them identical is a maintenance obligation —
  // the θ slots are the shared source of truth, and any change to the flow law
  // has to land in both files.
  let pulse = clamp(g.pulse, 0.0, 4.0);
  let z = th(TH_zoomRate) + th(TH_breathe) * sin(g.time * BREATH_RATE) + th(TH_pulseZoom) * pulse;
  let q = max(th(TH_perspective), 0.0);
  let vc = max(th(TH_vortexCore), 1e-3);

  let beam = max(th(TH_beam), 1.0);
  let halo = max(th(TH_halo), 0.0);
  let spread = max(th(TH_sectorSpread), 0.0);
  let swayRate = max(th(TH_swaySpeed), 0.0);

  // The filament term. One coherent noise field anchored to the vanishing point
  // and drifting slowly, multiplying every source's contribution — so a stamp is
  // broken into strands *before* the warp stretches it, and the beams come out
  // fibrous instead of smooth. The drift is deliberately unrelated to the sway
  // rate: the two moving at unconnected rates is what stops the strands from
  // looking painted onto the sources.
  let n = fbm2(rel * FILAMENT_SCALE + vec2f(g.time * 0.05, g.time * -0.037), g.seed);
  let filament = mix(1.0, 0.3 + 1.4 * n, FILAMENT);

  // ── how the four layers combine where they overlap ──────────────────────────
  //
  // Light adds. Two layers over one pixel is `c₀w₀ + c₁w₁`, which is correct
  // physics and, with this palette, ruinous art: bass orange is (1.00, 0.20,
  // 0.01) in linear light and vocals cyan is (0.03, 0.68, 1.00), so an orange
  // beam crossing a cyan one sums to (1.03, 0.88, 1.01). That is white — not
  // "white because it is bright" but white in *chromaticity*, before any
  // exposure or tone map has touched it, so nothing downstream can get the hue
  // back.
  //
  // The sector partition makes that rarer here than in nebula (two layers own
  // different wedges, so their material only meets where the vortex has sheared
  // one into another's territory, or where `sectorSpread` is modulated past 1)
  // but it does not remove it — and the places it does happen are exactly the
  // dense, hard-twisted core of the image, which is the part you least want to
  // lose. So the same split nebula uses, unchanged, because it is provably safe:
  //
  //   luminance     Σ wᵢ·luma(cᵢ) — the plain additive sum. Untouched. Two beams
  //                 crossing really are brighter than either, and the exposure
  //                 chain downstream sees exactly the light it saw before.
  //   chromaticity  a weighted average of the layers' unit-luminance hues, with
  //                 the weights raised to `layerBlend` first.
  //
  // At `layerBlend` = 1 the two recombine to the plain sum *identically* — the
  // weights cancel — so 1 is not "the fix off", it is provably the old
  // behaviour, which is what makes this safe to put on a slider. Above 1 the
  // louder layer's hue takes the crossing in proportion to how much louder it
  // is: at 3, a layer twice its neighbour's weight gets eight times the say.
  //
  // What it cannot do: two layers at *equal* weight still average to their
  // midpoint, and the midpoint of complementary hues is grey however it is
  // weighted. That is not a tuning failure, it is what "these two are the same
  // brightness here" means. The exponent buys that exact ties are measure-zero
  // where near-ties were previously being rounded to white.
  //
  // The shockwave rings below stay strictly additive, deliberately: they are the
  // one thing on screen that is *supposed* to be able to reach white, because
  // they are events rather than material.
  let blendSharp = max(th(TH_layerBlend), 1.0);
  /** Σ hueᵢ·wᵢ^blend — unit-luminance hues, so this carries no brightness at all. */
  var hueAcc = vec3f(0.0);
  var hueW = 0.0;
  /** Σ wᵢ·luma(cᵢ). The luminance the result is rescaled back onto. */
  var totalL = 0.0;

  let kmax = max(g.layerCount, 1u);
  /** One layer's share of the circle. K = 4, so a 90° wedge each. */
  let wedge = TAU / f32(kmax);

  for (var l = 0u; l < kmax; l = l + 1u) {
    let lay = layers[l];
    // Presence from the energy lane: 0 is a silent instrument, 1 is a loud one.
    // It scales source SIZE and the layer's overall gain, while the stem-follow
    // lane has already scaled the colour — two different cues off one stem, so a
    // voice arriving both brightens and grows.
    let energy = clamp(lay.color.a, 0.0, 1.5);
    // Both impulse lanes enter under a square root, and that is not timidity.
    // They are authored against physarum, where a multiplier lands on ONE step's
    // deposit; here it lands on a step of a field with tens of steps of memory
    // *and* compounds with the flash the same event is applying to the colour.
    // The shipped kick is deposit 3 (×4) and flash 1.6 (×2.6) — together ×10 on
    // a layer that is already the brightest thing on screen. Under a root the
    // same kick is ×2 here and still ×2.6 on the colour, which reads as a hard
    // flash that does not erase the image it is happening to.
    let gain = thL(l, TH_L_emitGain) * sqrt(max(lay.mods.x, 0.0));
    let ringR = thL(l, TH_L_sourceRadius);
    // `mods.y` is the impulse sensor lane: a bass note makes its sources swell
    // rather than merely flash, so the event moves the shape as well as the
    // light. Rooted for the same reason, and with more force behind it — light
    // goes as the square of this.
    let core = max(thL(l, TH_L_coreSize) * sqrt(max(lay.mods.y, 0.04)), 1e-4);

    // The sector. Centred on `(l + ½)·wedge` rather than on `l·wedge` so that
    // with K = 4 the four stems sit on the diagonals: an axis-aligned partition
    // puts a wedge boundary on the horizontal and vertical centre lines of the
    // frame, which reads as the image being cut in quarters by the window rather
    // than by the tunnel.
    let phi = (f32(l) + 0.5) * wedge;
    // Alternating sway sense, and a rate ladder, by layer index. Structural, not
    // θ: neighbouring layers swaying the same way at the same rate look like one
    // object with a seam in it, while counter-swaying neighbours keep the gap
    // between two wedges opening and closing, which is what makes the partition
    // visible as a *partition*. A slot for it would be a slot whose only useful
    // values are two.
    let dir = select(-1.0, 1.0, (l & 1u) == 0u);
    let rate = swayRate * (0.7 + 0.2 * f32(l));

    var sum = 0.0;
    for (var e = 0u; e < g.emitPerLayer; e = e + 1u) {
      // (phase, radius ×, speed ×, size ×) — drawn from the seed on reseed, so a
      // reroll rearranges the constellation without touching θ. `em.x` is used
      // as a sway PHASE here rather than as an orbital position, which is what
      // spreads four sources across the wedge at any given instant instead of
      // marching them in step.
      let em = emitters[l * g.emitPerLayer + e];
      let s = sin(em.x + dir * g.time * rate * em.z);
      let ang = phi + s * spread * wedge * 0.5;
      // `em.y` is ±22%, which is what stacks the four sources of one layer at
      // slightly different depths down the shaft — so a stem is a small bundle
      // of vanes rather than one wide arc.
      let radius = max(ringR * em.y * (0.82 + 0.36 * energy), 1e-3);
      // The radial unit vector AT the source. The stamp's long axis, the
      // direction the flow will carry it, and the axis the beam points along —
      // all three are this one vector, which is the geometric reason a tunnel
      // source can be a single anisotropic gaussian and still look right.
      let u = vec2f(cos(ang), sin(ang));
      let pos = centre + u * radius;

      // ── the continuity floor ────────────────────────────────────────────────
      //
      // A beam is a row of stamps, one per step, and it reads as a beam only
      // while consecutive stamps overlap. Here the displacement has two
      // components on two different axes, and the stamp is anisotropic, so the
      // floor is applied to each axis against the component that acts on it:
      //
      //   radial      r·|z + q·r|  — along the stamp's LONG axis, so it is
      //               covered by `size · beam` and the floor is divided by beam.
      //   tangential  r·(|rotate| + |vortex|·vc/(r + vc))  — across the stamp,
      //               where `beam` does not help at all.
      //
      // Enforcing it here rather than by keeping the flow slots small is what
      // lets the tunnel actually run: the displacement grows with radius (and
      // with radius *squared* through the perspective term) while `coreSize`
      // does not, so any fixed core is a dotted line at some legal `perspective`
      // and some legal `sourceRadius`. It also degrades in the right direction —
      // a source flung to the fast part of the frame grows a proportionately
      // fatter stamp, which is what something being smeared hard should look
      // like anyway.
      //
      // Note the second consequence, which is why `beam` is worth a slot: a
      // longer beam raises the flow speed this floor tolerates, for free.
      let radDisp = radius * abs(z + q * radius);
      let tanDisp = radius * (abs(th(TH_rotate)) + abs(th(TH_vortex)) * (vc / (radius + vc)));
      var size = core * em.w * (0.65 + 0.7 * energy);
      size = max(size, 1.2 * tanDisp);
      size = max(size, 1.2 * radDisp / beam);
      size = max(size, 1e-4);

      // The anisotropic stamp: stretched by `beam` along `u` (toward and away
      // from the vanishing point) and round across it. A stationary source
      // therefore already reads as a shaft of light pointing down the tunnel,
      // before the warp has drawn anything.
      let dv = p - pos;
      let du = dot(dv, u) / (size * beam);
      let dw = dot(dv, vec2f(-u.y, u.x)) / size;
      let d = sqrt(du * du + dw * dw);
      // Two lobes: a tight gaussian core and a wider exponential halo. The core
      // is what the warp draws into a beam; the halo is what keeps the space
      // between the four vanes from being pure black — and nothing else in this
      // visual fills that space, because there is no outward-spreading gas here,
      // only beams. The halo's rate is 1.7 rather than a gentler 1.15 for
      // nebula's measured reason: a tail still at 3% of peak five radii out gets
      // multiplied by the field's memory and becomes a fog over the structure
      // instead of an atmosphere around it.
      sum = sum + exp(-d * d * 3.0) + halo * 0.22 * exp(-d * 1.7);
    }

    // The far end of the shaft. Round rather than beam-stretched, deliberately:
    // at the vanishing point there is no radial direction to stretch along —
    // that is what makes it the vanishing point — so an elongated stamp here
    // would be a streak pointing in an arbitrary direction. See VANISH_GAIN.
    let vsize = max(ringR * VANISH_SIZE, 1e-3);
    let vd = length(rel) / vsize;
    sum = sum + VANISH_GAIN * (exp(-vd * vd * 1.6) * 0.6 + exp(-vd * 1.5) * 0.4);

    // The floor keeps a silent layer contributing a trace rather than nothing:
    // its beams should fade over the field's own memory, not switch off. 0.3
    // rather than a smaller number — this term multiplies with the stamp *area*
    // (which also grows with energy) and with stem-follow's brightness lane, so
    // a small-looking coefficient here compounds into a swing the auto-exposure
    // cannot follow across a section boundary.
    let weight = max(sum * gain * filament * (0.3 + 0.7 * energy), 0.0);
    // Floored well above zero: this divides the colour, and a layer whose
    // stem-follow has taken it to near-black would otherwise produce a hue
    // vector of ~1e6 and take a crossing on numerical noise alone.
    let cl = max(lumaOf(lay.color.rgb), 1e-3);
    let lw = weight * cl;
    let sh = pow(lw, blendSharp);
    hueAcc = hueAcc + (lay.color.rgb / cl) * sh;
    hueW = hueW + sh;
    totalL = totalL + lw;
  }

  // Unit-luminance hue × the additive luminance. `hueAcc / hueW` has luminance 1
  // by construction (every term does, and the weights normalise), so this is an
  // exact rescale rather than an approximate one — the pass's total light output
  // is independent of `layerBlend`, which is what keeps the knob a colour
  // control and not a second, hidden brightness slider fighting auto-exposure.
  var acc = select(vec3f(0.0), hueAcc * (totalL / max(hueW, 1e-30)), hueW > 0.0);

  // ── shockwave rings, in the tunnel's own coordinates ────────────────────────
  //
  // Nebula draws a ring concentric with the splash. Here that would be a flat
  // circle stuck on the front of a picture with a strong perspective, and it
  // reads as a UI element every time. So the ring is drawn concentric with the
  // **vanishing point** instead, at the tunnel radius the splash landed at, and
  // it is transported along the flow rather than simply expanding:
  //
  //   depth      r₀ = |splash − centre|, i.e. how far down the shaft it struck
  //   direction  the SIGN of the local flow (`z + q·r₀`), so a hit during the
  //              fountain half of the breath rushes at the camera and a hit
  //              during the drain half falls away into the vanishing point. The
  //              same event looks different depending on which way the tunnel is
  //              currently running, which is the whole reason the breath exists.
  //   bearing    the ring is a full annulus but is hottest at the angle the
  //              splash actually struck, falling to a quarter strength on the
  //              far side. A bare annulus has no locatable cause; a hot spot
  //              with an annulus behind it reads as an impact whose shock ran
  //              round the tunnel wall.
  //
  // Width narrows with progress, exactly as nebula's does, which is what makes
  // it a front rather than a growing donut.
  for (var i = 0u; i < g.splashCount; i = i + 1u) {
    let sp = splashes[i];
    let rel0 = toCentre(sp.posRadius.xy) - centre;
    let r0 = max(length(rel0), 1e-3);
    let bearing = atan2(rel0.y, rel0.x);
    let strength = clamp(sp.posRadius.w, 0.0, 2.0);
    let base = max(sp.posRadius.z, 1e-3);
    // Radius grows with the envelope's *decay*, so the front travels as it fades
    // and is gone by the time it would leave the frame.
    let progress = clamp(1.0 - strength, 0.0, 1.0);

    let flowSign = select(-1.0, 1.0, (z + q * r0) >= 0.0);
    let ringR = max(r0 + flowSign * base * sp.params.x * progress * RING_TRAVEL, 0.01);

    let rp = length(rel);
    var da = atan2(rel.y, rel.x) - bearing;
    // Wrapped into (−π, π] so the hot spot is symmetric about the bearing and
    // does not have a seam at the atan2 branch cut.
    da = da - TAU * round(da / TAU);
    // The impulse lane's `splashSwirl` knob, spent on making the front
    // three-lobed and rotating rather than perfectly round — a circle reads as a
    // UI element, a wobbling front reads as something hitting the medium.
    let wobble = 1.0 + 0.12 * sp.params.y * sin(da * 3.0 + progress * 6.0);
    let angW = 0.25 + 0.75 * exp(-da * da * 1.8);
    let width = max(base * 0.16 * (1.0 - 0.55 * progress), 0.004);
    let edge = (rp - ringR * wobble) / width;
    acc = acc + sp.tint.rgb * (exp(-edge * edge) * strength * angW * max(th(TH_ringGain), 0.0));
  }

  return vec4f(max(acc, vec3f(0.0)) * DRAW_SCALE, 1.0);
}
