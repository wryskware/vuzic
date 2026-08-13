// kaleido — the draw pass.
//
// Additive, on top of the field the warp pass just resampled. Everything here is
// *new* light; nothing here has memory. That split is the whole architecture:
// this pass is a handful of sources, and the warp is what turns them into a
// figure.
//
// Everything is evaluated at the **folded** coordinate — the pixel's position
// reduced into the fundamental wedge — and that single decision does three
// things at once.
//
//   1. It makes this step's fresh light symmetric the moment it lands, so the
//      field is never briefly lopsided and the warp's conservation argument (see
//      warp.wgsl) holds on every step rather than on all but the first.
//   2. It replicates every source 2n times for free. Twelve gaussians are
//      evaluated per pixel; a hundred and forty-four petals appear. The symmetry
//      order costs literally nothing at run time.
//   3. It is why the emitters can supply the angular motion the warp is forbidden
//      to. Each cluster oscillates inside its layer's angular lane; unlike an
//      angular term in the warp this is deposition rather than transport, so it
//      moves nothing and there is nothing for it to fail to conserve.
//
// Two things are drawn.
//
//   emitters  K layers × `emitPerLayer` clusters. A layer IS a stem, and its
//             identity is a RADIUS band plus an angular LANE — the two things the
//             fold preserves exactly. See `laneOf` for why one of them was not
//             enough. So "the vocal entered" is one named sector of the figure
//             lighting up, growing outward and starting to sweep.
//   rings     one expanding shockwave per live impulse splash, also folded, which
//             turns each into an n-fold flower. These are the only things on
//             screen that are not continuous, which is exactly why they read as
//             events.
//
// A fullscreen pass rather than instanced quads, deliberately, and more
// obviously right here than in nebula: instancing would need 2n quads per
// emitter and would have to reconstruct the fold per vertex, where the per-pixel
// form gets the replication from one `abs()`.

/**
 * Unit conversion between "θ gains authored as order-1 numbers" and "HDR units in
 * the accumulated field". The one empirical constant in this visual.
 *
 * It exists because the two quantities are not in the same units: the field's
 * brightness is the per-step light integrated over an emitter's *area* and
 * multiplied by the field's memory, so it falls as the square of `petalSize` and
 * rises with `1/fade`. Its job is to put the frame in the right decade so the
 * shared auto-exposure controller has road in both directions — the adapted gain
 * sitting at either rail is the symptom that this number is wrong.
 *
 * ## The arithmetic, against nebula's calibration
 *
 * Nebula's 4.0 is the anchor: it is measured to put the adapted gain near 1× on a
 * playing track at scene exposure 0.1, and the whole chain downstream (exposure,
 * autoTarget 0.1, the 0.01…64 rails) is shared, so the correct thing to match is
 * nebula's **total light per step**. Light from a gaussian goes as its area, i.e.
 * as size², so with comparable gains and lobe shapes:
 *
 *   nebula     16 instances  (4 layers × 4 emitters × 1 copy)
 *              mean size² = (0.035² + 0.014² + 0.020² + 0.028²)/4 = 6.51e-4
 *              16 × 6.51e-4 × 4.0                                 = 0.04166
 *
 *   kaleido   144 instances  (4 layers × 3 emitters × **2n = 12 copies**)
 *              mean size² = (0.022² + 0.009² + 0.014² + 0.019²)/4 = 2.805e-4
 *              144 × 2.805e-4 × DRAW_SCALE                        = 0.04039 × DS
 *
 *   DS = 0.04166 / 0.04039 = 1.031
 *
 * The 2n = 12 in that middle line is the fold's light multiplication and it is the
 * largest single factor in the estimate — an order of magnitude on its own. Most
 * of it is paid back by the smaller petals (0.43×) and the third emitter instead
 * of a fourth (0.75×), which is not a coincidence: those two numbers were chosen
 * *because* 144 nebula-sized petals would cover ~10% of the frame, and coverage is
 * what sets peak-to-mean, and peak-to-mean is what the tone map has to work with.
 * A mandala at nebula's petal sizes is one even plate.
 *
 * That the result lands within 3% of 1.0 is a coincidence and 1.0 is used because
 * the estimate is not precise to better than that anyway.
 *
 * If `fade`'s default or the petal sizes move by a lot, this is the number that
 * has to follow them.
 */
const DRAW_SCALE: f32 = 1.0;

/**
 * The symmetry order this pass's light level is calibrated at, and the reference
 * `foldNorm` normalises against.
 *
 * The fold puts 2n copies of everything on screen, so without a correction the
 * field's level would be proportional to `foldCount` — an order change from 5 to
 * 8 would be a 1.6× brightness step as well as a geometry change, and the
 * auto-exposure controller would spend `autoTau × ln(1.6)` ≈ half a second
 * absorbing it every time. Dividing by `n/6` makes the total light exactly
 * order-invariant, so what an order change changes is only the figure.
 *
 * Note what this trades: at n = 12 each petal is half as bright per pixel as at
 * n = 6 and there are twice as many. That is the right way round — a twelve-fold
 * mandala should be finer and lacier than a six-fold one, not twice as loud.
 */
const FOLD_REFERENCE: f32 = 6.0;

/**
 * The symmetry order, snapped and bounded. **Must stay identical to warp.wgsl's**
 * — the two passes are separate shader modules sharing only `common.wgsl` and the
 * generated θ prelude, so the duplication is unavoidable, but a disagreement
 * would mean depositing into a wedge the warp is not folding at.
 */
fn foldOrder() -> f32 {
  return clamp(round(th(TH_foldCount)), 2.0, 16.0);
}

/** `atan2` with the origin made safe; `atan2(0, 0)` is indeterminate in WGSL. */
fn angleOf(p: vec2f) -> f32 {
  let q = select(vec2f(1.0, 0.0), p, dot(p, p) > 1e-18);
  return atan2(q.y, q.x);
}

/** The fold; the derivation is at the top of warp.wgsl. */
fn foldAngle(a: f32, n: f32) -> f32 {
  let w = TAU / n;
  return abs(a - w * floor(a / w + 0.5));
}

/**
 * Per-layer angular LANE inside the fundamental wedge: (centre, half-width), both
 * as fractions of the wedge. Structural art direction, not θ — see below.
 *
 * ## Why lanes exist at all
 *
 * Revision 1 keyed the four stems on radius alone and let each layer's emitters
 * free-run around the whole circle, relying on the fold to scatter them. Measured
 * headless on a dense passage, that failed: the frame's mean rgb was [12, 57, 93]
 * with bass and drums both at ~0.95 stem level, i.e. a cyan plate. Two causes,
 * and this is the fix for the second (the first is the palette-luminance
 * normalisation of `emitGain` in kaleido.ts).
 *
 * Wherever two layers share pixels, `layerBlend` is winner-take-all by design and
 * the loudest one repaints the other. Nebula survives the same 10× loudness ratio
 * because its four layers occupy different *places* in the frame. Radius alone
 * could not do that here, because the emitter jitter (±22%), the energy swell
 * (±18%) and the modulation of `armRadius` all overlap the bands.
 *
 * ## Why a lane is permanent, which is the whole point
 *
 * **Because the warp has no angular transport** (see the theorem at the top of
 * warp.wgsl), an angular lane is not merely a starting position — it is where that
 * stem's light stays, forever. Light deposited in a lane streams radially outward
 * *along* it and never crosses into another. The constraint that forbade a
 * rotation term is what makes this separator work, which is a pleasing thing for a
 * constraint to do.
 *
 * ## The four lanes
 *
 * Every pair is disjoint in at least one axis: bass/vocals and bass/other by lane,
 * every other pair by radius. Drums is given the widest swing because it is
 * radius-isolated at the hub and can afford to cross everything.
 *
 *   bass    (0.74, 0.20)   r 0.30   lane [0.54, 0.94]
 *   drums   (0.50, 0.34)   r 0.12   lane [0.16, 0.84]   radius-isolated
 *   vocals  (0.28, 0.20)   r 0.21   lane [0.08, 0.48]
 *   other   (0.14, 0.12)   r 0.40   lane [0.02, 0.26]
 *
 * Constants rather than θ slots, for the reason nebula gives for its
 * counter-rotation: this is the stems' *address*. A slider that lets the music
 * move it is a slider that lets the music erase the stem labels — which is exactly
 * what the measurement above caught. What the seed still moves is the lane offset
 * (±LANE_SPREAD/2), the oscillation phase and the three emitter multipliers, so a
 * reroll is a different mandala made of the same four instruments.
 *
 * A `select` chain rather than an indexed array so that nothing depends on how a
 * given backend handles dynamic indexing of a constant array. Four layers; the
 * fall-through is drums.
 */
fn laneOf(l: u32) -> vec2f {
  var t = vec2f(0.50, 0.34);
  t = select(t, vec2f(0.74, 0.20), l == 0u);
  t = select(t, vec2f(0.28, 0.20), l == 2u);
  t = select(t, vec2f(0.14, 0.12), l == 3u);
  return t;
}

/**
 * How far the seed may slide a lane, as a fraction of the wedge.
 *
 * ±0.08 is the largest value that leaves bass/other and bass/vocals disjoint at
 * the worst pair of draws, which is the property the lanes exist for. Bigger
 * would buy more seed-to-seed variety and start reintroducing the bug.
 */
const LANE_SPREAD: f32 = 0.16;

/**
 * A point reduced into the fundamental wedge, radius preserved.
 *
 * Distances computed in this domain are honest plane distances, which is what
 * lets the gaussians below be written as if nothing were folded: the fold is an
 * isometry from each of the 2n wedges onto the fundamental one, so
 * `|foldPoint(p) − foldPoint(c)|` is exactly the distance from `p` to the nearest
 * mirror image of `c`. A petal is therefore a round gaussian in every copy,
 * including the ones straddling a mirror line, where it merges with its own
 * reflection into a single lens on the axis — which is correct, and is what a
 * kaleidoscope shows when a chip sits against a mirror.
 */
fn foldPoint(p: vec2f, n: f32) -> vec2f {
  let a = foldAngle(angleOf(p), n);
  return vec2f(cos(a), sin(a)) * length(p);
}

/** Rec.709, the same weights the post chain's `luma()` uses. */
fn lumaOf(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fsDraw(in: FsIn) -> @location(0) vec4f {
  let n = foldOrder();
  let p = foldPoint(toCentre(in.uv), n);

  // The warp's radial rate, rebuilt here for the continuity floor below. Not the
  // clamped `zoom` the warp uses — what is wanted is how far a stamp laid last
  // step has been carried since, which is the raw rate.
  let pulse = clamp(g.pulse, 0.0, 4.0);
  let zoom = 1.0 + th(TH_zoomRate) + th(TH_pulseZoom) * pulse;

  // The lace term. One coherent noise field, drifting slowly, multiplying every
  // emitter's contribution — so a petal is broken into strands *before* the warp
  // stretches it, and the arms come out fibrous instead of smooth.
  //
  // Sampled at the FOLDED coordinate, which is the whole of why it is safe.
  // Sampled in screen space it would be the one term in either pass that is not
  // symmetric, and it would dissolve the figure it is meant to texture — a
  // mandala with asymmetric mottling on it reads as a broken mandala, not as a
  // textured one. Folded, the noise is itself a mirror pattern and the break-up
  // comes out as lace.
  let nz = fbm2(p * max(th(TH_laceScale), 0.05) + vec2f(g.time * 0.05, g.time * -0.037), g.seed);
  let lace = mix(1.0, 0.25 + 1.5 * nz, clamp(th(TH_lace), 0.0, 1.0));
  let halo = max(th(TH_haloMix), 0.0);

  // ── how the four layers combine where they overlap ──────────────────────────
  //
  // Light adds. Two layers over one pixel is `c₀w₀ + c₁w₁`, which is correct
  // physics and, with this palette, ruinous art: bass orange is (1.00, 0.20, 0.01)
  // in linear light and vocals cyan is (0.03, 0.68, 1.00), so an orange arm
  // crossing a cyan one sums to (1.03, 0.88, 1.01). That is white. Not "white
  // because it is bright" — white in *chromaticity*, before any exposure or tone
  // map has touched it, so no amount of either can get the hue back.
  //
  // This matters more here than anywhere else in the repertoire. In nebula an
  // overlap is an occasional crossing of two arms; the fold maps every layer's
  // arms on top of every other layer's 2n times, so overlap is the normal case
  // rather than the exception, and a plain additive pass would produce a white
  // rosette with coloured fringes.
  //
  // The split below is what lets it be fixed without lying about brightness:
  //
  //   luminance     Σ wᵢ·luma(cᵢ) — the plain additive sum. Untouched. Two arms
  //                 crossing really are brighter than either, and the exposure
  //                 chain downstream still sees exactly the light it saw before.
  //   chromaticity  a weighted average of the layers' unit-luminance hues, with
  //                 the weights raised to `layerBlend` first.
  //
  // At `layerBlend` = 1 the two recombine to the plain sum *identically* — the
  // weights cancel — so 1 is not "the fix off", it is provably the old behaviour,
  // which is what makes this safe to put on a slider. Above 1 the louder layer's
  // hue takes the crossing in proportion to how much louder it is: at 3.5, a layer
  // twice its neighbour's weight gets ~11× the say, so a bass arm crossing a
  // fainter vocal one stays orange and merely brightens.
  //
  // What it cannot do: two layers at *equal* weight still average to their
  // midpoint, and the midpoint of complementary hues is grey however it is
  // weighted. That is what "these two are the same brightness here" means. What
  // the exponent buys is that exact ties are measure-zero while near-ties were
  // previously being rounded to white.
  //
  // The shockwave rings below stay strictly additive, deliberately: they are the
  // one thing on screen that is *supposed* to be able to reach white, because they
  // are events rather than material.
  let blendSharp = max(th(TH_layerBlend), 1.0);
  /** Σ hueᵢ·wᵢ^blend — unit-luminance hues, so this carries no brightness at all. */
  var hueAcc = vec3f(0.0);
  var hueW = 0.0;
  /** Σ wᵢ·luma(cᵢ). The luminance the result is rescaled back onto. */
  var totalL = 0.0;
  let kmax = max(g.layerCount, 1u);
  for (var l = 0u; l < kmax; l = l + 1u) {
    let lay = layers[l];
    // Presence from the energy lane: 0 is a silent instrument, 1 is a loud one. It
    // scales petal size, ring radius and the layer's overall gain, while the
    // stem-follow lane has already scaled the colour — so a voice arriving
    // brightens, grows and moves outward, three cues off one stem.
    let energy = clamp(lay.color.a, 0.0, 1.5);
    // Both impulse lanes enter under a square root, and that is not timidity. They
    // are authored against physarum, where a multiplier lands on ONE step's
    // deposit; here it lands on a step of a field with a hundred steps of memory
    // *and* compounds with the flash the same event applies to the colour. The
    // shipped kick is deposit 3 (×4) and flash 1.6 (×2.6) — together ×10 on a layer
    // that is already the brightest thing on screen. Under a root the same kick is
    // ×2 here and still ×2.6 on the colour: a hard flash that does not erase the
    // image it is happening to.
    let gain = thL(l, TH_L_emitGain) * sqrt(max(lay.mods.x, 0.0));
    let armR = thL(l, TH_L_armRadius);
    let spin = thL(l, TH_L_armSpin);
    // `mods.y` is the impulse sensor lane: a bass note makes its petals swell
    // rather than merely flash. Rooted for the same reason and with more force
    // behind it — light goes as the square of this.
    let core = max(thL(l, TH_L_petalSize) * sqrt(max(lay.mods.y, 0.04)), 1e-4);

    // Alternating sweep sense by layer index. Structural, not θ: two neighbouring
    // rings sweeping the same way lock together and the figure reads as one rigid
    // body, while counter-sweeping rings scissor past each other and the mandala
    // looks alive. A slot for it would be a slot whose only useful values are two.
    let dir = select(-1.0, 1.0, (l & 1u) == 0u);

    // This layer's angular territory. `wedge` is π/n, so a lane expressed as a
    // fraction of it narrows automatically as the symmetry order rises — the
    // figure gets finer rather than merely more numerous.
    let lane = laneOf(l);
    let wedge = TAU * 0.5 / n;
    let halfSwing = max(lane.y * wedge, 1e-3);

    var sum = 0.0;
    for (var e = 0u; e < g.emitPerLayer; e = e + 1u) {
      // (phase, radius ×, speed ×, size ×) — drawn from the seed on reseed, so a
      // reroll rearranges the constellation without touching θ.
      let em = emitters[l * g.emitPerLayer + e];
      // The chassis authors `em.x` as an angle spread evenly round the whole circle
      // and then jittered, precisely so a layer's clusters cannot clump. Read here
      // as a plain 0..1 fraction of a turn (×1/2π) rather than folded by 2π/n, so
      // that even spacing survives intact — at three clusters the seeded values land
      // in [0, 0.22], [0.33, 0.56] and [0.67, 0.89], which is a guarantee rather
      // than a hope. It is then spent on two things at once: sliding this cluster's
      // lane, and de-phasing its oscillation from its siblings', so a layer's
      // clusters scissor across the lane instead of moving as one block.
      //
      // (Revision 1 folded it, which collapsed the spacing and made the scatter
      // depend on the jitter alone. That is why `emittersPerLayer` is 3 — two
      // clusters collapsed to one spoke on about a third of seeds. Three is still
      // right, and now it is comfortable rather than a minimum.)
      let h = fract(em.x * 0.15915494);
      // The sweep, and the visual's only angular motion. Bounded to the lane rather
      // than free-running: `armSpin` is authored as the PEAK tangential rate, so the
      // oscillator frequency is that divided by the lane's half-width, which keeps
      // the slot's units and its shipped numbers meaning what they did before lanes
      // existed. `dir` alternates by layer so neighbouring rings counter-sweep.
      let centreA = clamp(lane.x + (h - 0.5) * LANE_SPREAD, 0.0, 1.0) * wedge;
      let omega = dir * spin * em.z / halfSwing;
      let a = clamp(centreA + halfSwing * sin(g.time * omega + h * TAU), 0.0, wedge);
      let radius = armR * em.y * (0.82 + 0.36 * energy);
      let centre = vec2f(cos(a), sin(a)) * radius;

      // ── the continuity floor ─────────────────────────────────────────────────
      //
      // An arm is a row of stamps, one per step, and it reads as an arm only while
      // consecutive stamps overlap — so a stamp must be at least as wide as the
      // distance separating it from the last one. Three things separate them here
      // and all three are counted:
      //
      //   angRate·radius   the sweep's PEAK tangential travel this step. Peak
      //                    rather than instantaneous, so the floor does not
      //                    breathe with the oscillator.
      //   |zoom−1|·radius  how far the zoom has carried the PREVIOUS stamp outward
      //                    since it was laid.
      //   hubDrain/2r      the hub drain's inward pull, which is the term that
      //                    matters at small radius — at drums' 0.12 it is 0.0013
      //                    screen heights per step against a 0.009 petal.
      //
      // Enforced here rather than respected by keeping the ranges small, which is
      // what lets `armSpin` reach 1.2 rad/s and `armRadius` 0.6: at those extremes
      // the displacement is 0.012 screen heights against a legal petal of 0.005,
      // i.e. two and a half stamp-widths of gap, and the arm would be a dotted
      // line. It also degrades in the right direction — a ring flung wide and swept
      // hard grows a proportionately fatter petal, which is what a thing being
      // dragged that fast should look like anyway.
      let angRate = spin * em.z * g.dt;
      let radialStep = abs(zoom - 1.0) * radius + th(TH_hubDrain) / (2.0 * max(radius, 0.02));
      let size = max(
        core * em.w * (0.6 + 0.8 * energy),
        1.2 * (angRate * radius + radialStep),
      );
      let d = length(p - centre) / size;
      // Two lobes: a tight gaussian core and a wider exponential halo. The core is
      // what the warp draws into an arm; the halo is what keeps the wedge from being
      // black between the arms, and it is on its own θ slot because the ratio
      // between them is most of the difference between "a ring of comets" and "a
      // figure". The halo's rate is 1.7 rather than a gentler 1.15 for the reason
      // nebula records measuring: an exponential tail at 1.15 is still at 3% of peak
      // five radii out, which the field's 100-step memory then multiplies by a
      // hundred — and with 144 tails instead of 16 that is a fog over the whole
      // figure rather than an atmosphere around it.
      sum = sum + exp(-d * d * 3.0) + halo * 0.22 * exp(-d * 1.7);
    }
    // The floor keeps a silent layer contributing a trace rather than nothing: its
    // ring should fade over the field's own memory, not switch off. 0.3 rather than
    // a smaller number because this multiplies with the petal *area* (which also
    // grows with energy) and with stem-follow's brightness lane, so a small-looking
    // exponent here compounds into a swing the auto-exposure cannot follow across a
    // section boundary.
    let weight = max(sum * gain * lace * (0.3 + 0.7 * energy), 0.0);
    // Floored well above zero: this divides the colour, and a layer whose
    // stem-follow has taken it to near-black would otherwise produce a hue vector
    // of ~1e6 and take every crossing on numerical noise alone.
    let cl = max(lumaOf(lay.color.rgb), 1e-3);
    let lw = weight * cl;
    let s = pow(lw, blendSharp);
    hueAcc = hueAcc + (lay.color.rgb / cl) * s;
    hueW = hueW + s;
    totalL = totalL + lw;
  }

  // Unit-luminance hue × the additive luminance. `hueAcc / hueW` has luminance 1
  // by construction (every term does, and the weights normalise), so this is an
  // exact rescale rather than an approximate one — the pass's total light output is
  // independent of `layerBlend`, which is what keeps the knob a colour control and
  // not a second hidden brightness slider fighting the auto-exposure.
  var acc = select(vec3f(0.0), hueAcc * (totalL / max(hueW, 1e-30)), hueW > 0.0);

  // ── shockwave rings ─────────────────────────────────────────────────────────
  //
  // Radius grows with the envelope's *decay* (progress = 1 − strength), so the ring
  // expands as it fades and is gone by the time it would leave the frame. Width
  // narrows with it too, which is what makes it read as a shock front rather than
  // as a growing donut.
  //
  // The centre is folded as well as the pixel, and that is not optional: the
  // impulse engine places splashes anywhere on screen, and a splash whose centre
  // fell outside the fundamental wedge would be a ring the folded pixel coordinate
  // can never reach — i.e. an event that fires and draws nothing. Folded, the ring
  // is the orbit of its arc under the dihedral group: 2n mirrored segments forming
  // a flower, which is the single most kaleidoscope-looking thing this visual does
  // and the reason events read as events here even though everything else on screen
  // is also symmetric.
  for (var i = 0u; i < g.splashCount; i = i + 1u) {
    let sp = splashes[i];
    let centre = foldPoint(toCentre(sp.posRadius.xy), n);
    let strength = clamp(sp.posRadius.w, 0.0, 2.0);
    let base = max(sp.posRadius.z, 1e-3);
    let progress = clamp(1.0 - strength, 0.0, 1.0);
    let radius = base * (0.12 + th(TH_ringExpand) * sp.params.x * progress);
    let d = p - centre;
    // The impulse lane's `splashSwirl` knob, spent on making the front lobed and
    // rotating rather than perfectly round — a circle reads as a UI element, a
    // wobbling front reads as something hitting the medium. Evaluated on the folded
    // offset, so each of the 2n petals of the flower gets the mirrored lobing and
    // the flower stays a flower.
    let wobble = 1.0 + 0.12 * sp.params.y * sin(atan2(d.y, d.x) * 3.0 + progress * 6.0);
    let width = max(base * 0.16 * (1.0 - 0.55 * progress), 0.004);
    let edge = (length(d) - radius * wobble) / width;
    acc = acc + sp.tint.rgb * (exp(-edge * edge) * strength * max(th(TH_ringGain), 0.0));
  }

  // `FOLD_REFERENCE / n` is the order-invariance correction; see its own note. It
  // multiplies the rings as well as the petals, deliberately — an event should keep
  // the same relationship to the figure it is happening to whatever the order is.
  return vec4f(max(acc, vec3f(0.0)) * (DRAW_SCALE * FOLD_REFERENCE / n), 1.0);
}
