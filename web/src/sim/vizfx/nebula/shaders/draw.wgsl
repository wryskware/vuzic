// nebula — the draw pass.
//
// Additive, on top of the field the warp pass just resampled. Everything here is
// *new* light; nothing here has memory. That split is the whole architecture:
// this pass is a handful of moving sources, and the warp is what turns them into
// a nebula.
//
// Two things are drawn.
//
//   emitters  K layers x `emitPerLayer` clusters, each orbiting the frame centre
//             at its layer's radius and angular speed. A layer IS a stem, so
//             "the vocal entered" is four cyan blobs appearing and swelling, and
//             the arms they are already trailing brighten with them.
//   rings     one expanding shockwave per live impulse splash. These are the
//             only things on screen that are not continuous, which is exactly
//             why they read as events.
//
// A fullscreen pass rather than instanced quads, deliberately. The rings are
// screen-sized by the time they have expanded, so a quad for one is a fullscreen
// quad anyway; the emitters are twelve gaussians, which is twelve exp() per
// pixel and cheaper than twelve draws' worth of state changes. It is also the
// milkdrop idiom the family is named for: a per-pixel expression evaluated over
// the whole frame, not a particle system.

/**
 * Unit conversion between "θ gains authored as order-1 numbers" and "HDR units
 * in the accumulated field". The one empirical constant in this visual.
 *
 * It has to exist because the two quantities are not in the same units and the
 * ratio between them is not a free choice: the field's brightness is the
 * per-step light integrated over an emitter's *area* and multiplied by the
 * field's memory, so it falls as the square of `coreSize` and rises with
 * `1/(1-decay)`. With the shipped cores (2–5% of screen height) and decay 0.99
 * that product lands around 0.008 mean field luminance, which is two orders
 * below anything the grade can lift — measured, not estimated: the first build
 * ran with `autoExposure` pinned at its 16× ceiling reporting a mean of 0.0026
 * against a target of 0.3, i.e. an image that was black with the controller out
 * of road.
 *
 * The alternative was to author `emitGain` in the hundreds, or to give the
 * exposure slot a range up to 40. Both push the arbitrary number somewhere a
 * human has to look at it: a gain slider whose useful values are 40–400 tells you
 * nothing, and a scene exposure two decades away from the other two substrates'
 * makes the shared HDR folder lie about what it is showing. The constant here
 * keeps `emitGain` ≈ 1 meaning "normal" and scene exposure ≈ 0.1 meaning what it
 * means in plife, and it is the only place the field's absolute scale is set.
 *
 * If `decay`'s shipped default or the core sizes move by a lot, this is the
 * number that has to follow them — the adapted gain sitting at either rail is the
 * symptom.
 *
 * ## Why it is 13 and not the 200 this shipped with
 *
 * It had to follow the warp's Jacobian fix (see `warp.wgsl`), which is the same
 * clause as the paragraph above with a much larger coefficient: the resample used
 * to multiply the field by `zoom²` every step, so the field was equilibrating 47×
 * higher than the emitters and `fade` alone can explain. Measured on the shipped
 * defaults, mean field luminance went 719 → 15.3 when that was corrected.
 *
 * At 200 the adapted gain therefore sat pinned at `autoMinGain` (0.05) with the
 * measured frame mean at 0.77–3.6 against a target of 0.10 — 8× to 36× over, with
 * the controller out of road. That is the mechanism behind "turning down exposure
 * doesn't help": auto-exposure had already given up, so scene exposure was the
 * only gain left in the chain, and dragging it scaled a uniformly-blown plate
 * down uniformly — dimmer, and no less washed out, because nothing about the
 * *ratio* between the arms and the cores had changed.
 *
 * The value now puts the adapted gain near 1× on a playing track, which is what
 * this constant is for: the controller then has road in both directions, and the
 * exposure slider is once again a thing the controller absorbs rather than the
 * last resort.
 *
 * Chosen as the *geometric* centre of the field's level over a track, not as the
 * value that suits any one bar. Measured at DRAW_SCALE 20 on free-fall, the mean
 * field luminance ran about 1 on a freshly-seeded quiet passage and about 32 at
 * equilibrium in a dense one — a ~32× span, because the field is an integrator
 * whose level is proportional to its memory and `fade` is a modulated slot. The
 * centre of that span is what leaves the controller equal road in both
 * directions; see the matching note on `autoMinGain`/`autoMaxGain` in config.ts,
 * which is the other half of the same argument.
 */
const DRAW_SCALE: f32 = 4.0;

/** Rec.709, the same weights the post chain's `luma()` uses. */
fn lumaOf(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fsDraw(in: FsIn) -> @location(0) vec4f {
  let p = toCentre(in.uv);

  // The filament term. One coherent noise field, drifting slowly, multiplying
  // every emitter's contribution — so a blob is broken into strands *before* the
  // warp stretches it, and the arms come out fibrous instead of smooth. Drift is
  // deliberately not proportional to the orbit speed: the two moving at
  // unrelated rates is what stops the strands from looking painted onto the
  // emitters.
  let n = fbm2(p * max(th(TH_noiseScale), 0.05) + vec2f(g.time * 0.06, g.time * -0.045), g.seed);
  let filament = mix(1.0, 0.25 + 1.5 * n, clamp(th(TH_filament), 0.0, 1.0));
  let halo = max(th(TH_halo), 0.0);

  // ── how the four layers combine where they overlap ──────────────────────────
  //
  // Light adds. Two layers over one pixel is `c₀w₀ + c₁w₁`, and that is what this
  // pass did — which is correct physics and, with this palette, ruinous art: bass
  // orange is (1.00, 0.20, 0.01) in linear light and vocals cyan is (0.03, 0.68,
  // 1.00), so an orange arm crossing a cyan one sums to (1.03, 0.88, 1.01). That
  // is white. Not "white because it is bright" — white in *chromaticity*, before
  // any exposure or tone map has touched it, so no amount of either can get the
  // hue back. It is the second half of the reported symptom, and unlike the first
  // half it survives the warp fix untouched.
  //
  // The split below is what lets it be fixed without lying about brightness:
  //
  //   luminance   Σ wᵢ·luma(cᵢ)   — the plain additive sum. Untouched. Two arms
  //               crossing really are brighter than either, and the exposure
  //               chain downstream still sees exactly the light it saw before.
  //   chromaticity  a weighted average of the layers' unit-luminance hues, with
  //               the weights raised to `layerBlend` first.
  //
  // At `layerBlend` = 1 the two recombine to the plain sum *identically* — the
  // weights cancel — so 1 is not "the fix off", it is provably the old behaviour,
  // which is what makes this safe to put on a slider. Above 1 the louder layer's
  // hue takes the crossing in proportion to how much louder it is: at 3, a layer
  // twice its neighbour's weight gets eight times the say, so an orange arm
  // crossing a fainter cyan one stays orange and merely brightens, and the two
  // stay separable exactly where they used to merge.
  //
  // Note what this cannot do: two layers at *equal* weight still average to their
  // midpoint, and the midpoint of complementary hues is grey however it is
  // weighted. That is not a tuning failure, it is what "these two colours are the
  // same brightness here" means. What the exponent buys is that exact ties are
  // measure-zero while near-ties were previously being rounded to white.
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
  for (var l = 0u; l < kmax; l = l + 1u) {
    let lay = layers[l];
    // Presence from the energy lane: 0 is a silent instrument, 1 is a loud one.
    // It scales emitter SIZE and the layer's overall gain, while the stem-follow
    // lane has already scaled the colour — two different cues off one stem, so a
    // voice arriving both brightens and grows.
    let energy = clamp(lay.color.a, 0.0, 1.5);
    // Both impulse lanes enter under a square root, and that is not timidity.
    // They are authored against physarum, where a multiplier lands on ONE step's
    // deposit; here it lands on a step of a field with a hundred steps of memory
    // *and* compounds with the flash the same event is applying to the colour.
    // The shipped kick is deposit 3 (×4) and flash 1.6 (×2.6) — together ×10 on a
    // layer that is already the brightest thing on screen, which measured as the
    // whole frame going white for the length of the decay. Under a root the same
    // kick is ×2 here and still ×2.6 on the colour, which reads as a hard flash
    // that does not erase the image it is happening to.
    let gain = thL(l, TH_L_emitGain) * sqrt(max(lay.mods.x, 0.0));
    let orbitR = thL(l, TH_L_orbitRadius);
    let orbitS = thL(l, TH_L_orbitSpeed);
    // `mods.y` is the impulse sensor lane: a bass note makes its emitters swell
    // rather than merely flash, so the event moves the shape as well as the light.
    // Rooted for the same reason, and with more force behind it — light goes as
    // the square of this, so the shipped 1.2 sensor bump would otherwise be a
    // 4.8× area.
    let core = max(thL(l, TH_L_coreSize) * sqrt(max(lay.mods.y, 0.04)), 1e-4);

    // Alternating orbit sense by layer index. Structural, not θ: two layers
    // running the same way merge into one rotating mass and the warp shears them
    // into a single spiral, while counter-rotating pairs produce the interleaved
    // arms the whole visual is for. A slot for it would be a slot whose only
    // useful values are two.
    let dir = select(-1.0, 1.0, (l & 1u) == 0u);

    var sum = 0.0;
    for (var e = 0u; e < g.emitPerLayer; e = e + 1u) {
      // (phase, radius x, speed x, size x) — drawn from the seed on reseed, so a
      // reroll rearranges the constellation without touching θ.
      let em = emitters[l * g.emitPerLayer + e];
      let a = em.x + dir * g.time * orbitS * em.z;
      let radius = orbitR * em.y * (0.82 + 0.36 * energy);
      let centre = vec2f(cos(a), sin(a)) * radius;

      // The continuity floor. An arm is a row of stamps, one per step, and it
      // reads as an arm only while consecutive stamps overlap — so a stamp must
      // be at least as wide as the distance the warp is about to carry it before
      // the next one lands. Below that the trail breaks into a dotted line, which
      // is precisely what an emitter looked like whenever its orbit was modulated
      // wide: the displacement grows with radius while `coreSize` does not.
      //
      // Enforcing it here rather than by keeping the angular slots small is what
      // lets the swirl actually be strong. It also degrades in the right
      // direction — an emitter flung to the rim gets a proportionately fatter
      // core, which is what a thing being sheared hard should look like anyway.
      let angRate =
        abs(th(TH_rotate)) + abs(th(TH_swirl)) * exp(-radius * radius * max(th(TH_swirlFalloff), 0.0));
      let size = max(core * em.w * (0.65 + 0.7 * energy), 1.2 * angRate * radius);
      let d = length(p - centre) / size;
      // Two lobes: a tight gaussian core and a wider exponential halo. The core
      // is what the warp draws into an arm; the halo is what keeps the frame from
      // being black between the arms, and it is on its own θ slot because the
      // ratio between them is most of the difference between "a few comets" and
      // "a cloud".
      //
      // The halo's rate is 1.7 rather than the 1.15 the first pass shipped, and
      // the difference is the whole reason that build read as four coloured
      // blobs. An exponential tail at 1.15 is still at 3% of peak five core-radii
      // out — which the field's 100-step memory then multiplies by a hundred, so
      // the tails alone saturated the middle of the frame and buried every arm
      // the warp had drawn. At 1.7 the same point is 0.6% and the halo is an
      // atmosphere around the arms instead of a fog over them.
      sum = sum + exp(-d * d * 3.0) + halo * 0.22 * exp(-d * 1.7);
    }
    // The floor keeps a silent layer contributing a trace rather than nothing:
    // its arms should fade over the field's own memory, not switch off. 0.3
    // rather than the 0.12 first tried — this term multiplies with the emitter
    // *area* (which also grows with energy) and with stem-follow's brightness
    // lane, so a small-looking exponent here compounds into a swing the
    // auto-exposure cannot follow across a section boundary.
    let weight = max(sum * gain * filament * (0.3 + 0.7 * energy), 0.0);
    // Floored well above zero: this divides the colour, and a layer whose
    // stem-follow has taken it to near-black would otherwise produce a hue vector
    // of ~1e6 and take the crossing on numerical noise alone.
    let cl = max(lumaOf(lay.color.rgb), 1e-3);
    let lw = weight * cl;
    let s = pow(lw, blendSharp);
    hueAcc = hueAcc + (lay.color.rgb / cl) * s;
    hueW = hueW + s;
    totalL = totalL + lw;
  }

  // Unit-luminance hue × the additive luminance. `hueAcc / hueW` has luminance 1
  // by construction (every term does, and the weights normalise), so this is an
  // exact rescale rather than an approximate one — the pass's total light output
  // is independent of `layerBlend`, which is what keeps the knob a colour control
  // and not a second, hidden brightness slider fighting the auto-exposure.
  var acc = select(vec3f(0.0), hueAcc * (totalL / max(hueW, 1e-30)), hueW > 0.0);

  // ── shockwave rings ─────────────────────────────────────────────────────────
  //
  // Radius grows with the envelope's *decay* (progress = 1 - strength), so the
  // ring expands as it fades and is gone by the time it would leave the frame.
  // Width narrows with it too, which is what makes it read as a shock front
  // rather than as a growing donut.
  for (var i = 0u; i < g.splashCount; i = i + 1u) {
    let sp = splashes[i];
    let centre = toCentre(sp.posRadius.xy);
    let strength = clamp(sp.posRadius.w, 0.0, 2.0);
    let base = max(sp.posRadius.z, 1e-3);
    let progress = clamp(1.0 - strength, 0.0, 1.0);
    let radius = base * (0.12 + th(TH_ringExpand) * sp.params.x * progress);
    let d = p - centre;
    // The impulse lane's `splashSwirl` knob, spent on making the front
    // three-lobed and rotating rather than perfectly round — a circle reads as a
    // UI element, a wobbling front reads as something hitting the medium.
    let wobble = 1.0 + 0.12 * sp.params.y * sin(atan2(d.y, d.x) * 3.0 + progress * 6.0);
    let width = max(base * 0.16 * (1.0 - 0.55 * progress), 0.004);
    let edge = (length(d) - radius * wobble) / width;
    acc = acc + sp.tint.rgb * (exp(-edge * edge) * strength * max(th(TH_ringGain), 0.0));
  }

  return vec4f(max(acc, vec3f(0.0)) * DRAW_SCALE, 1.0);
}
