// plasma — the draw pass.
//
// Additive, on top of the field the warp pass just resampled. Everything here is
// *new* light; nothing here has memory.
//
// Nebula's draw pass is a constellation of point sources and its picture is what
// the warp makes of them. This one paints no points at all. Each layer evaluates
// a continuous **interference field** over the whole frame:
//
//   radial   `emitPerLayer` travelling waves, sin(k·|p − sourceₑ| − ωt), from
//            slowly-orbiting seeded source points. Where the crests of two
//            sources meet you get a bright fringe; where a crest meets a trough
//            you get a dark lane. That is a two-slit pattern with four slits and
//            it is the oldest picture in wave physics.
//   planar   one travelling plane wave along the layer's own axis, so the frame
//            has an overall direction of march that the ring structure fights.
//
// The two are summed, mixed by `planarMix`, normalised to roughly −1…1, biased,
// rectified, and raised to `bandShape`. Signed-then-rectified is the whole
// difference between bands and blobs: the sign carries the interference (a
// destructive region is genuinely negative, not merely small), and rectifying it
// after biasing is what turns the zero crossings into hard dark lanes.
//
// Two other things are drawn:
//
//   phase shock   `g.pulse` — the deposit envelope of whatever just fired — adds
//                 a phase offset to EVERY fringe in the frame simultaneously, so
//                 a hit slides every band half a width and releases it. This is
//                 the plasma-native answer to "an event happened" and it is a
//                 genuinely different idiom from nebula's expanding rings: the
//                 whole image lurches rather than one spot lighting up.
//   splash fronts each live impulse splash is a travelling *local* phase
//                 disturbance plus a bright front, so the fringes visibly bow
//                 around something hitting the medium.
//
// A fullscreen pass rather than instanced geometry, necessarily: the injected
// field has no support to bound. It is the milkdrop idiom the family is named
// for — a per-pixel expression evaluated over the whole frame.

/**
 * Unit conversion between "θ gains authored as order-1 numbers" and "HDR units in
 * the accumulated field". The one empirical constant in this visual, and the
 * number most likely to be wrong on the first run, so the arithmetic behind it is
 * written out rather than asserted.
 *
 * The field's equilibrium level is
 *
 *     mean field luminance  ≈  P · DRAW_SCALE · memory
 *
 * where P is the spatial mean of this pass's output *before* DRAW_SCALE and
 * memory is 1/fade steps. Both terms differ from nebula's by a large factor and
 * they differ in the same direction, so the constant has to come down twice.
 *
 * ## P for nebula — the thing this is calibrated against
 *
 * Each emitter contributes `exp(−d²·3) + halo·0.22·exp(−d·1.7)` with d = |p−c|/s,
 * whose integral over the plane is s²·(π/3 + 0.478·halo) = 1.248·s² at the
 * shipped halo of 0.42. The effective stamp size is coreSize × (0.65 + 0.7·energy)
 * = coreSize × 1.35 at full presence, so the four layers run at s = 0.047, 0.019,
 * 0.027, 0.038 screen heights. Summing 4 emitters per layer weighted by emitGain:
 *
 *     Σ 4·emitGain_l·1.248·s_l²  =  0.0242 screen-height²
 *
 * against a 16:9 frame of area 1.78 in the same units, so
 *
 *     P_nebula ≈ 0.0136        ← i.e. 1.4% of the frame lit, per step
 *
 * ## P for plasma
 *
 * The interference sum is v = (Σ₄ sin + w·sin) / (4 + w) with w = planarMix·4 =
 * 1.8, so Var(v) = (4·½ + 1.8²·½)/5.8² = 0.108 and σ(v) = 0.33. The band is
 * max(v + bias, 0)^shape with bias −0.05 and shape 2.4, so
 *
 *     δ = E[(v − 0.05)₊^2.4] = σ^2.4 · E[(t − 0.152)₊^2.4] ≈ 0.069 × 0.42 ≈ 0.029
 *
 * — a mean of 0.029 against ridge peaks near 0.4, which is the peak-to-mean of
 * ~12 that `bandShape` exists to produce. Weighted by the four injGains (Σ = 4.3):
 *
 *     P_plasma ≈ 4.30 × 0.029 ≈ 0.125       ← 9.2× nebula's
 *
 * That ratio is the whole point of the brief's warning. It is not the 100× a
 * naive "the field covers the frame" estimate gives, and the reason is exactly
 * `bandShape`: the field *reaches* everywhere but is *bright* over a small
 * fraction of it, which is what stops it being a wash. Flatten the bands and the
 * ratio goes back to 100× and this constant is wrong by an order.
 *
 * ## The two corrections, and the answer
 *
 *     DRAW_SCALE = 4.0 × (P_neb / P_pl) × (mem_neb / mem_pl)
 *                = 4.0 × (0.0136 / 0.125) × (100 / 50)
 *                = 4.0 × 0.109 × 2  =  0.87
 *
 * Shipped at **0.75**, a little under the estimate, deliberately. Two reasons.
 * The estimate is a model, not a measurement — nebula's own constant was measured
 * and the model above under-predicts nebula's observed field level by ~2–3×, so
 * the honest uncertainty band here is roughly ×⅓…×3. And the two directions are
 * not symmetric in consequence: too dim is a frame the auto-exposure lifts (it
 * has 64× of upward road, and nebula measures at an adapted gain of 1.5–2.2, so
 * the shipped repertoire already sits on that side), while too bright is the
 * white plate that config.ts's whole auto-exposure note is about.
 *
 * Chosen as the *geometric centre over a track's dynamics*, not for one bar. The
 * level swings on `bandShape` × `bandBias` (their mod ranges together span ~30× in
 * δ), on `fade` (10× in memory) and on `injGain` (up to 4×), and 0.75 sits at the
 * middle of the product rather than at any end of it.
 *
 * If the adapted gain is measured sitting at a rail, this is the number that has
 * to move — and it should move by the measured ratio, in one step, rather than
 * being crept toward.
 *
 * ## Measured, and therefore now fixed
 *
 * Free-fall, seed 12345, 30 s from a cleared field: adapted gain **0.91** in a
 * dense passage and **3.42** in a quiet one, 0% blown pixels in both. That is
 * inside the band the estimate above predicted and comfortably inside the
 * controller's rails (0.01…64) at both ends, so the model held and this constant
 * is correct. It is not the lever for anything else.
 *
 * In particular it is NOT the lever for the dynamics defect the same measurement
 * exposed (one stem filling the frame, four emptying it). That was a histogram
 * problem, and DRAW_SCALE only moves the level — which auto-exposure then takes
 * straight back out. The fix is `bandGate` below, and it is deliberately built so
 * that a *present* layer's threshold is unchanged, which is what leaves this
 * calibration standing.
 */
const DRAW_SCALE: f32 = 0.75;

/**
 * Radians of fringe phase per unit of `pulseShock × pulse`.
 *
 * 2.6 with the shipped `pulseShock` of 0.6 and the chassis's capped pulse of 2
 * gives ~3.1 rad — almost exactly half a period. Half is the maximum meaningful
 * shift: the field is periodic, so a shift of a whole period is no shift at all,
 * and anything past half starts reading as the bands jumping backwards.
 */
const SHOCK_PHASE: f32 = 2.6;

/**
 * How fast a layer's radial waves travel outward from their sources, in fringe
 * radians per second per unit of `driftSpeed`.
 *
 * 4.0 puts the bass layer (driftSpeed 0.12) at 0.48 rad/s — 6% of a period over
 * the field's memory, i.e. barely moving, which is what a bass slab should do —
 * and drums (0.55) at 2.2 rad/s, or 29% of a period over the memory, which is
 * visible marching. The per-stem contrast in *how the bands move* is as much of
 * the identity as the contrast in how wide they are.
 */
const SOURCE_WAVE_SPEED: f32 = 4.0;

/** The planar wave's own march rate, deliberately unrelated to the radial one. */
const PLANAR_WAVE_SPEED: f32 = 1.7;

/**
 * The planar wave's spatial frequency relative to the layer's `fringe`.
 *
 * Under 1 on purpose. At exactly 1 the planar bands and the radial fringes have
 * the same period and lock into a moiré that reads as a printing artefact; at
 * 0.62 the two beat against each other on a scale of a couple of fringe widths,
 * which is the large-scale cellular structure you actually see.
 */
const PLANAR_FREQ: f32 = 0.62;

/**
 * Floor inside the band's `pow`, so the exponent never sees an exact zero.
 *
 * pow(0, x) is 0 by the spec, but the result then feeds a second `pow` by
 * `layerBlend` and the chain 0 → 0 → a division guarded at 1e-30 is a lot of
 * denormal arithmetic to trust across four drivers. At 1e-5 the floor
 * contributes (1e-5)^2.4 ≈ 1e-12 per layer per step, which is eleven orders below
 * the field level and cannot be seen.
 */
const BAND_FLOOR: f32 = 1e-5;

/**
 * The floor the chassis's energy lane clamps presence at — `defaultEnergy().floor`
 * in `sim/vizfx/config.ts`, "what a fully silent instrument keeps: a ghost, not a
 * hole".
 *
 * Mirrored here because `layers[].color.a` arrives already floored and the shader
 * has no other way to know it. Without the rescale below, a gate keyed on
 * presence sees a live range of 0.4…1.0 rather than 0…1, which has two effects
 * and both were measured:
 *
 *   - a genuinely silent layer sits at 40% of the gate's swing rather than at 0,
 *     so it never fully retreats;
 *   - and, worse, a *fully present* layer sits at 1.0 while `1 − presence` is
 *     still ~0.05, so with `bandGate` running near 0.7 every loud layer was
 *     paying ~0.05 of extra threshold it should not have paid. That is what took
 *     the dense chorus from 37.2% dead frame to 52.8% — the gate was acting as a
 *     near-uniform offset on all four layers instead of as a difference between
 *     them.
 *
 * A mismatch (the lane's floor is user-configurable, 0…1) degrades gracefully
 * rather than breaking: the clamp keeps the rescaled value in 0…1 either way, a
 * higher configured floor merely compresses the gate's swing and a lower one
 * merely leaves the bottom of it unused. It is still a duplicated constant, and
 * the honest fix — the chassis handing the shader its own floor — is a change to
 * `config.ts` and `Globals`, i.e. chassis work rather than visual work.
 */
const ENERGY_FLOOR: f32 = 0.4;

/**
 * Where the crest term starts, as a fraction of full constructive interference.
 *
 * 0.65 lights about 3% of the frame — a thin bright spine running along each
 * current, not a second copy of the band. Narrow is the whole point: the crest
 * exists to raise the field's PEAK without raising its mean, and every unit of
 * width costs mean without buying peak.
 */
const CREST_KNEE: f32 = 0.65;

/**
 * How far a capped impulse floods the gate open, per unit of `pulseShock`.
 *
 * The other half of "presence buys coverage", applied on the event timescale
 * rather than the phrase timescale: a hit briefly *widens* every band in the
 * frame as well as shifting its phase, so the medium swells and relaxes. 0.09 at
 * the shipped `pulseShock` of 0.6 and the chassis's capped pulse of 2 moves the
 * threshold by 0.11 — coverage 40% → 53% — which passes straight through the
 * auto-exposure controller (autoTau 1 s) and therefore reads as the drop it is.
 */
const PULSE_FLOOD: f32 = 0.09;

/**
 * Peak of the shock front's light term, per unit of `shockGain`.
 *
 * 2.0 against the 0.6 this shipped with. See the long note on `shockGain` in
 * plasma.ts: a ring has to out-shout an integrator with fifty steps of memory
 * while itself sweeping past a pixel in about six, so "comparable per-step
 * amplitude" comes out visibly *dimmer* than the sustained field. Measured, the
 * dense frame's maximum was 145/255 with 18 rings live.
 */
const FRONT_LIGHT: f32 = 2.0;

/** Rec.709, the same weights the post chain's `luma()` uses. */
fn lumaOf(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fsDraw(in: FsIn) -> @location(0) vec4f {
  let p = toCentre(in.uv);

  // ── the whole-field phase shock ─────────────────────────────────────────────
  //
  // One number, added to every fringe of every layer. Because the field has
  // memory, what a viewer sees is not the instantaneous shifted pattern but the
  // *smear* between the old phase and the new one — the bands blur outward along
  // their own gradient and re-sharpen as the envelope decays. That is what makes
  // this read as a shock through a medium rather than as an animation.
  let pulse = clamp(g.pulse, 0.0, 4.0);
  let shockAmt = max(th(TH_pulseShock), 0.0);
  let globalPhase = shockAmt * pulse * SHOCK_PHASE;

  // ── splash fronts: local phase disturbance + light ──────────────────────────
  //
  // Radius grows with the envelope's *decay* (progress = 1 − strength), so the
  // front expands as it fades and is gone by the time it would leave the frame;
  // width narrows with it, which is what makes it a shock front rather than a
  // growing donut. Both are nebula's construction, because both are right.
  //
  // What is different is that the front carries a *phase* term as well as a
  // brightness term, and the phase term is the one that matters here: adding
  // ~2.6 rad of phase inside the front's envelope bends every fringe it crosses
  // into a visible arc, so the event deforms the medium instead of being drawn on
  // top of it. The brightness term is what keeps the event legible in a passage
  // where there are no bands bright enough to bend.
  var localPhase = 0.0;
  var frontLight = vec3f(0.0);
  for (var i = 0u; i < g.splashCount; i = i + 1u) {
    let sp = splashes[i];
    let centre = toCentre(sp.posRadius.xy);
    let strength = clamp(sp.posRadius.w, 0.0, 2.0);
    let base = max(sp.posRadius.z, 1e-3);
    let progress = clamp(1.0 - strength, 0.0, 1.0);
    let d = p - centre;
    let radius = base * (0.1 + th(TH_shockExpand) * sp.params.x * progress);
    // The impulse lane's `splashSwirl` knob, spent on making the front three-lobed
    // and rotating rather than perfectly round — a circle reads as a UI element, a
    // wobbling front reads as something hitting the medium.
    let wobble = 1.0 + 0.12 * sp.params.y * sin(atan2(d.y, d.x) * 3.0 + progress * 6.0);
    // Wider than nebula's 0.16 because this front has a second job: a phase kink
    // narrower than a fringe half-period is invisible, since there is no fringe
    // inside it to bend. At 0.35·base the front is 0.017 screen heights at the
    // smallest shipped splash radius, which straddles the finest legal fringe.
    let width = max(base * 0.35 * (1.0 - 0.45 * progress), 0.012);
    let edge = (length(d) - radius * wobble) / width;
    let env = exp(-edge * edge);
    localPhase = localPhase + env * strength * shockAmt * SHOCK_PHASE;
    // The light term is the SQUARE of the phase envelope — the same front, √2
    // narrower. That split is deliberate: the phase kink has to be at least a
    // fringe half-period wide or there is nothing inside it to bend, while the
    // light has to be narrow to be a front rather than a glow, and narrow is
    // also what buys peak amplitude for a fixed amount of deposited energy.
    frontLight =
      frontLight
      + sp.tint.rgb * (env * env * strength * max(th(TH_shockGain), 0.0) * FRONT_LIGHT);
  }
  let phase = globalPhase + localPhase;

  let shape = max(th(TH_bandShape), 0.2);
  let bias = th(TH_bandBias);
  let planar = clamp(th(TH_planarMix), 0.0, 1.0);

  // ── presence buys COVERAGE, not intensity ───────────────────────────────────
  //
  // The one thing this pass has to get right that nebula's does not have to think
  // about. Nebula's layers are small blobs: a silent one contributing 15% of its
  // light is 15% of 0.65% of the frame, i.e. nothing, and scaling amplitude with
  // presence is a perfectly good answer. Here every layer's support is the entire
  // frame, so a silent layer at 15% amplitude is a full-frame haze that lands its
  // ridges in every other layer's dark lanes and fills them. Measured: one stem
  // playing and three at exactly zero produced an edge-to-edge marble with 3.2%
  // black, while four loud stems produced 37.2% — the dynamics ran backwards, and
  // there was no headroom left for a chorus to grow into.
  //
  // So presence moves the *rectification threshold* rather than a gain. A layer
  // that is absent keeps only its strongest constructive ridges (a few thin
  // filaments); a layer that is present opens out into wide bands. Coverage
  // becomes the thing that tracks the music, which is also the only cue
  // auto-exposure cannot undo — the controller normalises the mean, so anything
  // that merely scales the field is cancelled within a second, whereas a
  // threshold changes the field's *histogram* and nothing downstream can put that
  // back.
  //
  // The gate is a *difference between layers*, never an offset on all of them.
  // That distinction is the whole correction from the first attempt: presence is
  // rescaled off the energy lane's floor (see ENERGY_FLOOR) and then passed
  // through a curve that saturates at the top, so a layer that is genuinely
  // playing lands at `gate = bias` — exactly the threshold DRAW_SCALE was
  // calibrated against — and only an *absent* layer pays anything at all.
  //
  // There is deliberately no scene-wide coupling. An earlier pass had one, on the
  // reasoning that a solo intro should not fill as much frame as a chorus; the
  // reference visual measures at 5.3% dead frame in the same quiet window against
  // this one's 7.2%, so the quiet end was never the defect and a term that dimmed
  // it was solving a problem that did not exist while costing the chorus.
  let gateDepth = max(th(TH_bandGate), 0.0);
  let crestGain = max(th(TH_crest), 0.0);
  // A hit floods every layer's gate open for the length of its envelope: the
  // event-timescale version of the same idea, and the reason a drop widens the
  // currents rather than merely brightening them.
  let flood = PULSE_FLOOD * shockAmt * pulse;

  // ── how the four layers combine where they overlap ──────────────────────────
  //
  // Light adds, and for this visual that is ruinous in a way it is not for
  // nebula. Nebula's layers are small blobs that cross occasionally; plasma's are
  // smooth fields with support over the entire frame, so *every* pixel is a
  // crossing. Additively, bass orange (1.00, 0.20, 0.01) plus vocals cyan (0.03,
  // 0.68, 1.00) is (1.03, 0.88, 1.01) — white in **chromaticity**, before any
  // exposure or tone map has touched it, so nothing downstream can recover the
  // hue. Done naively this visual is a uniformly white screen with a texture.
  //
  // The split is nebula's, unchanged, because it is exactly right here:
  //
  //   luminance     Σ wᵢ·luma(cᵢ) — the plain additive sum. Untouched, so the
  //                 exposure chain sees exactly the light it would have seen.
  //   chromaticity  a weighted average of the layers' unit-luminance hues, with
  //                 the weights raised to `layerBlend` first.
  //
  // At layerBlend = 1 the two recombine to the plain sum *identically* (the
  // weights cancel), so 1 is provably the physical answer rather than "the fix
  // off", which is what makes it safe on a slider. Above 1 the locally louder
  // layer's hue takes the pixel in proportion to how much louder it is.
  //
  // The reason this works better here than the geometry might suggest is
  // `bandShape`. Because each layer's contribution is a set of narrow ridges
  // rather than a broad hump, at almost any pixel one layer is well ahead of the
  // others — the exponent has a real ratio to work on, and near-ties (which do
  // average to grey, however weighted) are confined to the thin seams where two
  // layers' ridges cross. Those seams reading as desaturated is correct: it is
  // what "these two are equally bright here" means.
  //
  // The splash fronts stay strictly additive, deliberately: they are the one
  // thing on screen that is *supposed* to be able to reach white, because they are
  // events rather than material.
  let blendSharp = max(th(TH_layerBlend), 1.0);
  /** Σ hueᵢ·wᵢ^blend — unit-luminance hues, so this carries no brightness at all. */
  var hueAcc = vec3f(0.0);
  var hueW = 0.0;
  /** Σ wᵢ·luma(cᵢ). The luminance the result is rescaled back onto. */
  var totalL = 0.0;

  let kmax = max(g.layerCount, 1u);
  let sources = max(g.emitPerLayer, 1u);
  // The planar wave carries slightly less total weight than all the radial
  // sources combined at the shipped mix; scaling by the source count is what
  // keeps `planarMix` meaning the same thing when the constellation is resized.
  let planarW = planar * f32(sources);
  let norm = 1.0 / (f32(sources) + planarW);

  for (var l = 0u; l < kmax; l = l + 1u) {
    let lay = layers[l];
    // Presence from the energy lane: 0 is a silent instrument, 1 is a loud one.
    // It scales the source ring's radius and the layer's overall gain, while the
    // stem-follow lane has already scaled the colour — two cues off one stem.
    let energy = clamp(lay.color.a, 0.0, 1.5);
    // The impulse deposit lane, under a square root for nebula's reason: the
    // multiplier is authored against physarum, where it lands on ONE step's
    // deposit, and here it lands on a step of a field with fifty steps of memory
    // *and* compounds with the flash the same event applies to the colour.
    let gain = thL(l, TH_L_injGain) * sqrt(max(lay.mods.x, 0.0));
    // `fringe` is a spacing, so the wavenumber is its reciprocal. Floored well
    // above zero: at the hard minimum of 0.02 screen heights a fringe is still
    // ~22 device pixels at 1080p, so nothing here goes near Nyquist.
    let kL = TAU / max(thL(l, TH_L_fringe), 0.01);
    // `mods.y` is the impulse sensor lane — how far this layer is *reaching* —
    // and it is spent on the source ring swelling rather than on anything
    // time-integrated. That placement is deliberate: the wave phases below are
    // `g.time × speed`, so a multiplier on `speed` would be a phase jump
    // proportional to the elapsed time, i.e. a whole-field discontinuity that
    // grows through the track. A radius is a geometric quantity with no such
    // memory, and swelling the ring moves the fringes' centre of curvature, which
    // reads as the medium bulging.
    let ringR =
      thL(l, TH_L_srcRadius) * (0.82 + 0.36 * energy) * sqrt(max(lay.mods.y, 0.04));
    let speed = thL(l, TH_L_driftSpeed);
    // Alternating sense by layer index. Structural, not θ, for nebula's reason: a
    // slot whose only useful values are two is not a slider. Counter-rotating
    // neighbours are what make the interleaved bands shear against each other
    // instead of marching in convoy.
    let dir = select(-1.0, 1.0, (l & 1u) == 0u);

    var w = 0.0;
    for (var e = 0u; e < sources; e = e + 1u) {
      // (phase, radius ×, speed ×, size ×) — drawn from the seed on reseed, so a
      // reroll rearranges the interference pattern without touching θ. Here the
      // fourth component is reinterpreted as a per-source *frequency* multiplier:
      // four sources at four slightly different wavelengths beat against each
      // other, and that beat envelope is the large-scale structure. Four sources
      // at one wavelength give a clean symmetric figure instead, which reads as a
      // diagram of an experiment rather than as a medium.
      let em = emitters[l * sources + e];
      let a = em.x + dir * g.time * speed * em.z;
      let c = vec2f(cos(a), sin(a)) * ringR * em.y;
      let d = length(p - c);
      // k·d − ω·t: a wave travelling OUTWARD from the source. The `em.x·3` and
      // `f32(l)·1.7` terms decorrelate the starting phases — the second because
      // the chassis's emitter buffer is strided by its ceiling (6) while the
      // shader indexes by the live count, so neighbouring layers can alias onto
      // the same quadruples, and two layers sharing a fringe phase would read as
      // one layer with a doubled brightness.
      let arg = d * kL * em.w - g.time * speed * SOURCE_WAVE_SPEED + em.x * 3.0 + f32(l) * 1.7;
      w = w + sin(arg + phase);
    }

    // The layer's own axis: four directions 90° apart, structural for the same
    // reason `dir` is. This is the cue that survives being watched from across
    // the room — "the bands changed direction" is legible at a glance in a way
    // that "the bands got slightly narrower" is not.
    let ang = f32(l) * TAU / f32(kmax) + 0.45;
    let axis = vec2f(cos(ang), sin(ang));
    let planarArg = dot(p, axis) * kL * PLANAR_FREQ + dir * g.time * speed * PLANAR_WAVE_SPEED;
    w = w + planarW * sin(planarArg + phase);

    // Signed → biased → rectified → shaped. The rectification is what makes dark
    // lanes rather than a wash: a destructively-interfering region is genuinely
    // negative here, and clipping it to zero leaves a hard boundary that the
    // grade's 1.35 contrast and raised black point then have something to bite on.
    let v = w * norm;
    // The gate, rescaled off the energy lane's floor so it spends its whole swing
    // on the range the lane can actually deliver, then squared-complement so that
    // it saturates: `open` reaches 1 well before presence does. A layer at 95% of
    // full presence is 99.75% open and therefore sits at `gate = bias`, unchanged
    // from the calibrated configuration; a layer at the lane's floor is 0% open
    // and retreats by the whole of `bandGate`, which against the interference
    // sum's σ ≈ 0.33 takes it from ~56% of the frame to ~1%.
    //
    // The saturation is what makes this a difference rather than an offset, and
    // it is the fix for the chorus: without it every loud layer paid ~0.05 of
    // threshold for the 5% of presence it was short, and four layers each paying
    // that is a visibly emptier frame.
    let pres = clamp((energy - ENERGY_FLOOR) / max(1.0 - ENERGY_FLOOR, 0.05), 0.0, 1.0);
    let open = 1.0 - (1.0 - pres) * (1.0 - pres);
    let gate = bias - gateDepth * (1.0 - open) + flood;
    let t = max(v + gate, 0.0);
    let band = pow(t + BAND_FLOOR, shape);
    // ── the crest ───────────────────────────────────────────────────────────────
    //
    // A narrow, bright spine riding the top of each band, and the answer to the
    // one measurement this visual could not explain any other way: the chorus was
    // rendering 52.8% pure black against the reference visual's 1.8%, at the same
    // grade and the same auto-exposure target.
    //
    // The grade makes that arithmetic exact rather than a matter of taste. A pixel
    // renders as true black when `contrast` about `pivot` drives it to zero, i.e.
    // when the tone-mapped value is under `blackPoint + (1−blackPoint)·pivot·
    // (1 − 1/contrast)` = 0.05 + 0.95·0.28·0.2593 = **0.119**. Auto-exposure holds
    // the frame's MEAN at `autoTarget` = 0.10 — below that cutoff — so more than
    // half of any frame is below the black point by construction, and the only
    // thing that can rescue a dark pixel is bloom, which is added *after* the
    // measure pass and therefore rides on top of the normalised mean.
    //
    // Bloom needs something over `bloom.threshold` = 0.9 to work with. The
    // reference visual has emitter cores and shockwave fronts — hard bright edges
    // that clear it easily, and its bloom then fills the gas between its arms.
    // This field had no hard edges at all: with the mean pinned at 0.10 and a
    // measured peak-to-mean of 5.8 its brightest pixel reached ~0.58, so it
    // generated *no bloom whatsoever* and every lane between the currents fell
    // through the black point untouched.
    //
    // No amount of fill light can fix that, and it is worth being precise about
    // why, because "add a soft haze under the bands" is the obvious move and it
    // cannot work: the haze would raise the mean too, and auto-exposure divides it
    // straight back out. A dark pixel is by definition below the mean, the cutoff
    // is *above* the mean, so nothing that scales with the mean can lift it. Only
    // the peak-to-mean ratio matters, and only bloom can act on it.
    //
    // Hence a term that is nearly all peak and nearly no mean. Squared above a
    // high knee, it lights ~3% of the frame; at the shipped gain it roughly
    // triples the injected peak while adding ~40% to the mean, for a net
    // peak-to-mean gain of ~2.5× — which takes the accumulated field from 5.8 to
    // ~14 and puts its crests through the bloom threshold with room to spare.
    //
    // This is the same architectural split the reference visual makes between its
    // gaussian core and its wide exponential halo, run the other way round: it is
    // a soft thing that needed an edge, where that one is an edge that needed
    // something soft around it. And it fixes the two complaints together, since a
    // field with no pixels near white also had no headroom for an impulse to
    // reach white in.
    let crestT = max(t - CREST_KNEE, 0.0) / max(1.0 - CREST_KNEE, 0.05);
    let crest = crestGain * crestT * crestT;
    // The residual amplitude floor, dropped from nebula's 0.3 to 0.15 now that
    // the gate is carrying the presence signal. It still exists for the reason it
    // does there — a quiet layer should fade over the field's own memory rather
    // than switch off — but at 0.3 it was, together with the stem-follow lane's
    // own 0.25 floor, most of what kept three silent stems visible across the
    // whole frame.
    let weight = max((band + crest) * gain * (0.15 + 0.85 * energy), 0.0);
    // Floored well above zero: this divides the colour, and a layer whose
    // stem-follow has taken it to near-black would otherwise produce a hue vector
    // of ~1e6 and take the pixel on numerical noise alone.
    let cl = max(lumaOf(lay.color.rgb), 1e-3);
    let lw = weight * cl;
    let s = pow(lw, blendSharp);
    hueAcc = hueAcc + (lay.color.rgb / cl) * s;
    hueW = hueW + s;
    totalL = totalL + lw;
  }

  // Unit-luminance hue × the additive luminance. `hueAcc / hueW` has luminance 1
  // by construction (every term does, and the weights normalise), so this is an
  // exact rescale: the pass's total light output is independent of `layerBlend`,
  // which is what keeps that knob a colour control and not a second hidden
  // brightness slider fighting auto-exposure — and it is also what lets
  // DRAW_SCALE above be calibrated without knowing where `layerBlend` is set.
  var acc = select(vec3f(0.0), hueAcc * (totalL / max(hueW, 1e-30)), hueW > 0.0);
  acc = acc + frontLight;

  return vec4f(max(acc, vec3f(0.0)) * DRAW_SCALE, 1.0);
}
