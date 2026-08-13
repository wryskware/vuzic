/**
 * **tunnel** — the second vizfx visual: the milkdrop radial vortex.
 *
 * Same shape as `nebula/nebula.ts` and for the same reason: there is no tunnel
 * class. `VizFxSim` owns the ping-pong rig, the ModTarget plumbing and the
 * buffers, and reads the table below to find out what it is drawing. This file
 * plus two WGSL passes is the whole visual.
 *
 * ## What it is, and what makes it a tunnel rather than a nebula
 *
 * A **movable vanishing point** and a flow that is polar about it. Every step
 * the warp resamples the field through a radial map whose speed *grows with
 * radius* — the perspective law of a shaft seen end-on, where the far end
 * (screen centre) barely moves and the near end (the rim) rushes past — plus a
 * vortex twist that is strongest at the core and falls off outward. Light
 * therefore leaves an emitter and is drawn into a long radial beam that
 * accelerates outward, curling as it goes.
 *
 * Nebula's flow is a near-uniform zoom about the frame centre and its structure
 * comes from *differential rotation*: arms shear past each other. This one's
 * structure comes from *differential translation*: a radial velocity gradient,
 * which is what depth looks like. Three things follow from that and none of them
 * is a palette swap:
 *
 * - The frame has a stagnation point (the vanishing point) and a fast rim, so
 *   the image reads as a shaft you are travelling down rather than as an object
 *   you are looking at.
 * - The flow can sit either side of zero. Positive is a **fountain** (flying
 *   forward, everything rushes out past the camera); negative is a **drain**
 *   (flying backward, everything falls into the vanishing point). `breathe`
 *   crosses that boundary on its own every ~15 s and `pulseZoom` shoves it
 *   across on a drop — the lurch.
 * - The four stems are four **angular sectors** of the tunnel, so a stem
 *   entering lights up its own quadrant of the shaft.
 *
 * ## Why sectors and not concentric bands
 *
 * The brief offered both. Sectors, and the argument is about what the flow does
 * to a labelling rather than about taste.
 *
 * Under a radial flow the *angle* is the conserved coordinate and the *radius*
 * is the transported one. A band labelling ("bass owns the annulus at r ≈ 0.6")
 * is destroyed by the warp within a second: the light born in bass's annulus is
 * immediately carried out of it, through every other layer's annulus, so the
 * bands describe only where light is *born* and never where it *is*. Worse, the
 * flow reverses — under a drain the outer band gets the long run and under a
 * fountain the inner one does, so the same table would mean two different
 * compositions depending on the sign of a modulated slot.
 *
 * A sector labelling is preserved: radial transport does not change the angle at
 * all, and the two terms that do (`rotate`, `vortex`) move every sector, so the
 * wedges shear into spiral vanes but never trade places. Per-stem identity is
 * therefore structural under *both* flow directions, which is the property the
 * project's priorities actually ask for.
 *
 * Depth is not given up in exchange: `sourceRadius` is a per-layer slot, so the
 * four stems still sit at four different distances down the shaft. The stems are
 * separated by angle *and* stacked in depth; only the *identity* rides on the
 * angle, because that is the coordinate the warp respects.
 *
 * ## Why the character lives in per-layer defaults
 *
 * As in nebula, the four layers differ in the table below and the differences
 * are the art direction:
 *
 * - **bass** sits at 0.30 with the largest cores — the outer of the two
 *   "streaming" layers, past the vortex core radius, so its arms run nearly
 *   straight out and accelerate as they go. Low end reads as mass moving past
 *   you at speed.
 * - **drums** sits at 0.11, right against the vanishing point, with the
 *   smallest cores. There almost nothing moves radially but the vortex twist is
 *   at its maximum, so a hit resolves as a distinct flash that is immediately
 *   spun rather than as a swell that is flung.
 * - **vocals** sits at 0.19, just inside the vortex core radius, and carries the
 *   most `emitGain` headroom: it is the voice most likely to arrive and leave
 *   mid-track, and both the stem-follow and energy lanes key off it hardest.
 * - **other** sits furthest out at 0.40, so pads and texture are the material
 *   streaming off the frame edges — the background the rest is drawn against.
 *
 * All four are inside 0.5, which is not art direction but a hard geometric
 * constraint: see the note on `sourceRadius`. The first build let the modulator
 * push the outer two past the frame edge and the image lost half its palette.
 *
 * The seeded personality (`ModSpec.jitter`) then displaces all of it, and the
 * seeded emitter placement (`VizFxSim.redrawEmitters`) redistributes the sway
 * phases and the within-sector depths — so a reroll is a different creature.
 *
 * ## The θ table: 38 slots, 32 of them modulated
 *
 * 4 per layer × 4, plus 22 globals. Every `lo`/`hi` is narrower than the hard
 * `min`/`max` beside it: the hard bound is what a file may contain and what a
 * hand slider may reach, the ModSpec range is where the music may wander
 * unsupervised.
 *
 * Excursion sizing follows the same rule physarum states and nebula repeats:
 * with a z-scored input and a unit projection, `w·ẑ` is roughly N(0,1), so the
 * typical |tanh| at depth 1 is about 0.55 and every `half` below is therefore
 * about twice the excursion you should expect to see.
 */
import {
  CLASS_FAST,
  CLASS_MEDIUM,
  CLASS_SLOW,
  type ModGroup,
  type ModSpec,
} from '../../../mapping/modspec.ts';
import type { VizFxVisual, VizSlot } from '../slots.ts';

import warpWgsl from './shaders/warp.wgsl?raw';
import drawWgsl from './shaders/draw.wgsl?raw';

const add = (group: ModGroup, lo: number, hi: number, half: number, jitter: number): ModSpec => ({
  group,
  lo,
  hi,
  half,
  jitter,
  mult: false,
});

const mul = (group: ModGroup, lo: number, hi: number, half: number, jitter: number): ModSpec => ({
  group,
  lo,
  hi,
  half,
  jitter,
  mult: true,
});

/**
 * Per-layer shipped character; see the header. Index-aligned with the stems
 * channel.
 *
 * `sourceRadius` is the number that carries this visual. It is where the layer's
 * emitters sit *down the shaft*, and because the radial flow rate is
 * `zoomRate + perspective·r`, the radius is also what decides how fast that
 * layer's material moves: at the shipped defaults the bass ring (0.30) travels
 * 0.0035 screen heights per step and the drums ring (0.11) travels 0.0006 — a
 * 6× spread that no other slot in the table produces. Two layers placed at the
 * same radius are two layers with the same tempo, and the depth reading is what
 * goes first when they are.
 *
 * ## Why these are half what the first build shipped (0.46/0.20/0.32/0.58)
 *
 * Two measured failures, both of them geometry rather than taste.
 *
 * **The frame is not round.** These radii are in screen *heights* (common.wgsl:
 * y runs −0.5…0.5), and the four sectors sit on the diagonals, so a source at
 * radius r is at (r/√2, r/√2) and leaves the frame at r = 0.707 — sooner near
 * the top and bottom than the numbers suggest. With the old `sourceRadius` mod
 * ceiling of 0.85, and a seeded jitter of 0.4 in ln space on top, bass and
 * `other` were routinely modulated to r ≈ 0.8, i.e. **entirely off-screen**.
 * That is the measured cause of the frame reading as monochrome cyan: mean red
 * was 4/255 in a passage where all four stems were loud, not because bass was
 * dim but because bass was outside the window. Vocals (0.32) and drums (0.20)
 * were the only two reliably in frame, and they are the two blue ones.
 *
 * **A source needs room downstream of it.** The material is what makes the
 * image; a source at 0.58 has 0.1 of frame left to draw an arm in before the
 * flow carries it out. At 0.11…0.40 every layer has most of the frame to stream
 * across, and the four rings are properly nested so each stem owns a band of
 * depth as well as a wedge of angle.
 *
 * The cores are small for the reason nebula's are — 1.2% to 2.6% of screen
 * height. A core wide enough to read as an object on its own does not need the
 * feedback to make it visible, so the feedback stops being what you are looking
 * at. Here there is a second reason: the stamp is stretched radially by `beam`,
 * so a core of 0.026 is already a 0.052-long streak before the warp has touched
 * it. They came down with the radii to hold the *ratio* — four sources spread
 * across a wedge only read as four distinct knots while a knot is small compared
 * with the arc it is spread over, and that arc shrank with the radius.
 */
const LAYER_CHARACTER: readonly {
  emitGain: number;
  sourceRadius: number;
  coreSize: number;
}[] = [
  { emitGain: 1.0, sourceRadius: 0.3, coreSize: 0.026 }, // bass
  { emitGain: 1.4, sourceRadius: 0.11, coreSize: 0.012 }, // drums
  { emitGain: 1.15, sourceRadius: 0.19, coreSize: 0.018 }, // vocals
  { emitGain: 0.85, sourceRadius: 0.4, coreSize: 0.022 }, // other
];

const character = (key: keyof (typeof LAYER_CHARACTER)[number], fallback: number) =>
  (layer: number): number => LAYER_CHARACTER[layer % LAYER_CHARACTER.length]?.[key] ?? fallback;

const LAYER_SLOTS: readonly VizSlot[] = [
  // Excluded from modulation, the same call all three substrates make and for
  // the same reason: light is driven by the stem-follow lane, and modulating the
  // base it scales puts the constant unexplained flashing straight back. The
  // chassis reads this one by name (`params['layer<i>.brightness']` in
  // `uploadLayers`/`syncSpecies`), so the key is part of the contract rather
  // than a choice.
  {
    key: 'brightness',
    label: 'brightness (base × stem-follow)',
    cls: CLASS_FAST,
    min: 0,
    max: 2,
    step: 0.01,
    def: 1,
    mod: null,
  },
  {
    key: 'emitGain',
    label: 'core gain ×',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    def: character('emitGain', 1),
    // Multiplicative: a gain is a ratio, and ±0.5 in ln space is ×0.6…×1.65,
    // which reads the same on the quiet layer and the loud one.
    mod: mul('population', 0.15, 4, 0.5, 0.35),
    macro: 'light',
  },
  {
    key: 'sourceRadius',
    label: 'source ring radius (screen heights from the vanishing point)',
    // SLOW, and more emphatically than nebula's `orbitRadius`: this is not only
    // where the layer lives in the frame, it is how fast the layer's material
    // travels (the flow rate is `zoomRate + perspective·r`). Moving it quickly
    // would change a layer's *tempo* every bar, and the field cannot show that
    // anyway — beams already drawn at the old radius take a memory to clear.
    cls: CLASS_SLOW,
    min: 0.02,
    max: 1.2,
    step: 0.005,
    def: character('sourceRadius', 0.25),
    // **The ceiling is the important number here and it came down from 0.85.**
    // The frame's inscribed circle has radius 0.5 (half the screen height), and
    // the sectors sit on the diagonals where a source at radius r is at
    // (r/√2, r/√2). At 0.85 the music could put a stem at (0.60, 0.60) — off the
    // top of the frame — and it did: measured mean red 4/255 with all four stems
    // loud, because bass and `other` had been modulated out of the window
    // entirely. 0.50 is the largest radius from which no seeded, modulated,
    // macro'd, energy-swollen combination can leave the visible frame.
    //
    // The floor is 0.05 rather than the hard 0.02 for the opposite reason:
    // inside about 0.05 the radial flow rate is under 0.0004/step and the sway
    // arc is smaller than a core, so a layer modulated onto the vanishing point
    // stops being a source with an arm and becomes a stationary dot at the
    // stagnation point. The hand slider may still go there — it is a legitimate
    // thing to want to look at — but the music may not park a stem there.
    mod: mul('structure', 0.05, 0.5, 0.3, 0.4),
    macro: 'scale',
  },
  {
    key: 'coreSize',
    label: 'core size (screen heights, before the beam stretch)',
    cls: CLASS_MEDIUM,
    min: 0.002,
    max: 0.4,
    step: 0.002,
    def: character('coreSize', 0.018),
    // Ceiling down from 0.16 with the radii: 0.16 is a third of the largest
    // legal source radius, so one emitter would fill its own wedge and the four
    // knots that make a sector legible would be one blob.
    mod: mul('structure', 0.006, 0.09, 0.3, 0.4),
    macro: 'scale',
  },
];

/**
 * ## The relationship that decides whether this looks like a tunnel
 *
 * A beam is the trail a source leaves while the field still remembers it, and in
 * this visual the thing that ends a beam is **not** `fade`. It is the expansion
 * itself.
 *
 * The warp conserves light (see the long note in warp.wgsl), so a parcel that
 * travels outward spreads over a larger area and dims by exactly the area ratio.
 * With the radial map `f(r) = r·(1 − z − q·r)` the per-step loss from that alone
 * is `1 − J ≈ 2z + 3q·r`, while the per-step travel is `z·r + q·r²`. The ratio
 * of travel to expansion loss is therefore
 *
 *     (z·r + q·r²) / (2z + 3q·r)   →   r/2  (pure zoom)   …   r/3  (pure perspective)
 *
 * — bounded by the radius and *independent of the rate*. Turning the flow up
 * does not make the beams longer; it makes them faster and equally short. That
 * is a fact about energy-conserving radial expansion in 2D and it is the single
 * most important thing to understand before retuning anything here. (Classic
 * milkdrop tunnels look bright to the rim because they do not conserve energy —
 * that is precisely the divergent loop nebula's Jacobian note is about, and it
 * is not available.)
 *
 * ## The first build got the conclusion from that wrong, and it cost the image
 *
 * The reasoning was: since expansion loss caps travel-to-loss at r/3 whatever
 * the rate, reach cannot come from the flow, so it must come from `streakMix` /
 * `streakLength` — a motion blur inside the loop whose gain is provably 1.
 * Every clause of that is true and the conclusion is still wrong, because gain 1
 * is a statement about *energy* and reach is a statement about *structure*. An
 * averaging kernel applied sixty times a second is a diffusion, and diffusion is
 * exactly the operator that preserves mass while destroying detail. Shipped at
 * mix 0.55 and length 4.5 it erased everything: the measured frame was a smooth
 * blue fog with faint concentric banding — the mean of the structure that should
 * have been there. The streak is now sized to close the gap between two
 * consecutive stamps and nothing more (mix 0.30, length 1.2).
 *
 * ## Where the reach actually comes from: transit time, not smear
 *
 * The r/3 bound is about brightness *per unit radius travelled*. It says nothing
 * about how long a parcel takes to get there, and that is the quantity an arm's
 * visible length is really made of — because the arm is drawn by the flow itself,
 * one stamp per step, for as long as the parcel survives.
 *
 * So the flow is now **slow where the sources are and fast only out at the rim**,
 * which is what the perspective law gives for free once `zoomRate` is small and
 * `perspective` carries the gradient. At the shipped defaults (z = 0.0025,
 * q = 0.030) a parcel born on the vocals ring at r = 0.19 takes ~48 steps to
 * reach r = 0.3 and ~82 to reach r = 0.5, against a memory of 100. It is drawn
 * for its whole life instead of being flung off-frame in twenty steps — and over
 * those 82 steps the vortex has wound it about 1.2 rad, which is the spiral.
 *
 * Numerically the depth gradient is unchanged and still strong: that parcel is
 * ~17× dimmer by r = 0.5, four stops, which in HDR is a clearly visible fall-off
 * rather than a disappearance.
 *
 * Anyone retuning this should move `perspective`, `zoomRate` and `fade`
 * together, and should treat `streakMix` as a hazard rather than a knob:
 * raising the flow shortens every arm twice over, once by transit time and once
 * by expansion, and raising the smear to compensate destroys the arms outright.
 *
 * ## The continuity constraint, and where it is enforced
 *
 * A beam is a row of stamps, one per step, and it reads as a beam only while
 * consecutive stamps overlap. Here the displacement has two components and they
 * do not share an axis:
 *
 *     radial      r·|z + q·r|                       — along the stamp's long axis
 *     tangential  r·(|rotate| + |vortex|·vc/(r+vc)) — across it
 *
 * `draw.wgsl` floors the stamp against both, and against the *right* one in each
 * direction: the radial floor is divided by `beam`, because a stamp stretched
 * radially already covers radial displacement. That is what lets `beam` earn its
 * keep as continuity insurance as well as art direction — a longer beam buys a
 * faster tunnel for free — and it is why the flow slots below can be sized for
 * how the tunnel feels rather than for what the smallest core can survive.
 *
 * Note the vortex profile is *bounded* in linear displacement by construction:
 * `vortex·vc/(r+vc)·r → vortex·vc` as r grows, so the twist can never fling a
 * rim stamp arbitrarily far however hard it is modulated. Nebula's gaussian
 * falloff does not have that property, which is why its swirl range is the one
 * its comments keep apologising for.
 */
const GLOBAL_SLOTS: readonly VizSlot[] = [
  // ── warp · tunnel ──────────────────────────────────────────────────────────
  {
    key: 'centreX',
    label: 'vanishing point x (screen heights)',
    folder: 'warp · tunnel',
    // SLOW, and this is the slot the whole visual is named for. Where the
    // vanishing point sits is a compositional fact — a shaft whose far end
    // wanders every bar is not a shaft, it is a wobble — and it is also the
    // slowest thing the field can follow: the accumulated beams all point at the
    // *old* centre and take a full memory to re-aim.
    //
    // Note the coordinate system (common.wgsl): y runs -0.5…0.5 whatever the
    // window is and x runs the same distance per pixel, so on 16:9 the frame
    // spans x ∈ ±0.889. ±0.7 hard therefore reaches just past the left and right
    // edges — deliberately, because a vanishing point sitting off-frame is a
    // legitimate composition (the shaft runs off the side of the screen) — while
    // the mod range stops at ±0.34, inside the frame, because the music parking
    // the centre off-screen would leave four sectors of which two are invisible.
    cls: CLASS_SLOW,
    min: -0.7,
    max: 0.7,
    step: 0.005,
    def: 0,
    mod: add('structure', -0.3, 0.3, 0.12, 0.1),
  },
  {
    key: 'centreY',
    label: 'vanishing point y (screen heights)',
    folder: 'warp · tunnel',
    cls: CLASS_SLOW,
    // Tighter than x, because the frame is shorter in y: ±0.45 is the frame edge
    // exactly, and the mod range keeps to the middle 40% of the height.
    min: -0.45,
    max: 0.45,
    step: 0.005,
    def: 0,
    // ±0.14 rather than ±0.2, and the number is derived rather than chosen: the
    // largest legal source radius is 0.5, the sectors sit on the diagonals, so a
    // source's vertical offset from the vanishing point is at most 0.5/√2 =
    // 0.354. The frame's half-height is 0.5. 0.14 + 0.354 = 0.494, which is the
    // guarantee that **no combination of a modulated centre and a modulated
    // source radius can put a stem off the top or bottom of the frame** — the
    // failure the first build measured as an all-cyan image. `centreX` gets more
    // room below because the frame is 0.889 wide on 16:9.
    mod: add('structure', -0.14, 0.14, 0.055, 0.05),
  },
  {
    key: 'zoomRate',
    label: 'zoom rate / step  (>0 = fountain outward · <0 = drain inward)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    // Signed, and both signs are the visual rather than one being a degenerate
    // case: positive is flying forward and negative is flying backward. The
    // negative side is bounded tighter because a drain concentrates light toward
    // the vanishing point instead of spreading it, so the same magnitude reaches
    // a hot core much faster than it reaches a bright rim.
    min: -0.03,
    max: 0.05,
    step: 0.0002,
    // Small and *positive*, so the shipped tunnel is a gentle fountain — and
    // small enough that `breathe` below (0.008) carries it across zero twice per
    // breath. The default composition is therefore not "outward"; it is
    // "outward, drifting through still into inward and back", which is the
    // gesture the visual exists for.
    //
    // Down from 0.004, with `perspective` up, which is one change rather than
    // two: the uniform term is the part of the flow that acts *at the vanishing
    // point*, and a parcel's transit time — the thing an arm's length is really
    // made of — is dominated by how slowly it crawls away from the middle. The
    // measured first build ran this term nearly twice as fast and the arms were
    // flung out of the source region before the warp had drawn anything.
    def: 0.0025,
    // ## Why this is a rate rather than the zoom factor itself
    //
    // The same trap nebula documents at length and shipped a bug on: both `half`
    // and `jitter` act in ln space on the *stored number*, so a multiplicative
    // spec on a factor of 1.004 spends almost all of its excursion on the `1` —
    // a "±0.8%" draw becomes a 3× draw on the growth rate, because the growth
    // rate is `x − 1`. Stored as the rate, an additive spec means what it says.
    //
    // No macro, deliberately, and for nebula's reason plus one of this visual's
    // own. Nebula's: zoom trades against frame-filling rather than against
    // liveliness, so folding it into `motion` would mean pushing "motion" always
    // also pushed the image off the edges. This visual's: `motion` also scales
    // `breathe`, and a macro on both would move the centre of the breath and its
    // amplitude together — the breath would stop crossing zero, which is the one
    // thing it is for.
    mod: add('structure', -0.01, 0.014, 0.005, 0.004),
  },
  {
    key: 'perspective',
    label: 'perspective (rim rushes faster than the vanishing point)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    // The depth cue, as a coefficient with units of 1/screen-height: the radial
    // flow rate is `zoomRate + perspective·r`, so this is how much faster the
    // near end of the shaft moves than the far end. At 0 the warp degenerates to
    // nebula's uniform zoom and the image is a flat spiral.
    //
    // Up from 0.02 while `zoomRate` came *down* — the pair is the whole depth
    // reading and it is their ratio that matters. At 0.030 against a uniform
    // 0.0025 the source region (r ≈ 0.2) crawls at 0.0017/step while the rim
    // (r ≈ 0.9) runs at 0.027/step: a 16× velocity gradient across the frame,
    // where the first build's pair gave 5×. That gradient is *the* thing that
    // distinguishes a tunnel from a zoom, and the reason it can be pushed this
    // hard now is that nothing is born out where it is fast — every source ring
    // sits inside 0.5.
    min: 0,
    max: 0.12,
    step: 0.0005,
    def: 0.03,
    // Multiplicative and strictly positive: the perspective gradient always
    // points the same way (near is fast), because a *negative* gradient is a
    // flow that converges on a ring, and a converging ring is the one geometry
    // in this family that piles light up without bound. With q > 0 every
    // stagnation ring the table can produce is a *repelling* one — material
    // leaves it in both directions — which is why the breath can cross zero
    // safely and produce a still glowing ring rather than a growing white one.
    // Ceiling down from 0.08: at 0.08 the rim flow is 0.072/step, which is four
    // stamp-widths per step, and no continuity floor can hold an arm together
    // through that — it was a full-frame smear with the streak on top of it.
    mod: mul('structure', 0.004, 0.06, 0.4, 0.45),
    macro: 'motion',
  },
  {
    key: 'breathe',
    label: 'breath depth (± swing on the zoom rate, ~15 s period)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.03,
    step: 0.0002,
    // Three times `zoomRate`'s default (0.0025), which is the whole point: the
    // sum crosses zero, so the tunnel drifts fountain → still → drain → still →
    // fountain on its own with nothing playing.
    //
    // What "drain" means here is worth stating precisely, because `perspective`
    // is always positive and therefore always outward. The net velocity is
    // `r·(z + q·r)`, so when `z` goes negative there is a **stagnation radius**
    // at `r = |z|/q`: inside it material falls toward the vanishing point,
    // outside it material still streams to the rim. At the trough of the shipped
    // breath (z = −0.0055, q = 0.030) that radius is 0.18 — right on the vocals
    // ring — so the inner half of the shaft inhales while the outer half exhales,
    // and the boundary between them sweeps in and out every 15 s.
    //
    // 0.008 rather than the 0.005 the first build shipped, and the retuned
    // `zoomRate` is why: at z = 0.004 the trough only reached −0.001 and the
    // stagnation radius never got past 0.03, so the breath was a speed change
    // rather than a direction change and the gesture the slot exists for was
    // invisible.
    def: 0.008,
    // Multiplicative: it is an amplitude, so ×2 should mean twice the swing.
    // Note this is the one place `motion` is allowed to touch the zoom lane —
    // scaling the *swing* is unambiguous, where scaling the centre would not be.
    mod: mul('structure', 0.0012, 0.024, 0.35, 0.4),
    macro: 'motion',
  },
  {
    key: 'rotate',
    label: 'rigid rotation / step (rad)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    min: -0.15,
    max: 0.15,
    step: 0.0005,
    // Small, and much smaller than nebula's 0.016. A rigid rotation applied to a
    // *radial* flow turns every arm into an Archimedean spiral, and the arc an
    // arm accumulates is `rotate × transit`, and the transit is now ~82 steps.
    //
    // ## This slot is the *larger* half of the winding budget, which was missed
    //
    // The obvious reading is that a rigid rotation is harmless to the sector
    // partition because it moves every sector by the same angle. That is true of
    // the sectors and false of the picture: the sources do not rotate — they
    // sway about a fixed bearing `(l + ½)·wedge` — so a rigid rotation carries
    // *deposited material* away from its own stationary source, which is exactly
    // the same shear the vortex applies, and it adds to the same footprint.
    //
    // Worse, per unit it is the more potent of the two. The vortex profile falls
    // off as `vc/(r + vc)`, so a parcel travelling outward sees less of it as it
    // goes and accumulates ~37.5 rad per unit `vortex`; a rigid rotation acts
    // undiminished for the whole transit and accumulates ~82 rad per unit. The
    // ceiling this shipped with (base 0.010 + half 0.010 = 0.020 running) was
    // therefore worth **1.6 rad on its own** — more than the vortex's own worst
    // case, and enough to blow the 0.66 rad gap four times over without the
    // vortex contributing anything.
    //
    // That is why the seeds in `vortex`'s note correlate with the vortex value
    // but do not correlate *cleanly*: 20240607 (vortex 0.0318) measured worse on
    // hue than 12345 (vortex 0.0359), and an unmeasured second winding term is
    // the obvious candidate for the inversion.
    //
    //     seeded base    0.003 ± 0.0025 =  0.0005 … 0.0055    (W 0.04 … 0.45 rad)
    //     + excursion    ± 0.004        = −0.0035 … 0.0095    (|W| ≤ 0.78 rad)
    //
    // Typical ~0.25 rad, which leaves the vortex to do the spiralling — correct,
    // because the vortex's radius-dependent profile is what makes the spiral
    // *tighten toward the vanishing point*, and a rigid rotation is the term
    // that reads as the camera turning rather than as the tunnel flowing.
    //
    // Note this is an inference from the geometry rather than from a
    // measurement: the three-seed sweep read the live `vortex` and not this. It
    // is the first thing to re-read if the hue separation is still short.
    def: 0.003,
    mod: add('structure', -0.012, 0.012, 0.004, 0.0025),
    macro: 'motion',
  },
  {
    key: 'vortex',
    label: 'vortex twist (rad/step at the core, falls off outward)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    min: -0.4,
    max: 0.4,
    step: 0.002,
    // Much larger than `rotate` because it is scaled by `vc/(r + vc)`
    // everywhere: at the bass ring (0.30) with the shipped core radius (0.25)
    // the profile is already taking 55% of it. The number that matters is the
    // product, and this is what makes the two comparable.
    //
    // The profile is the classic vortex — angular rate falling as 1/r outside a
    // core — rather than nebula's gaussian, and the reason is continuity rather
    // than realism: `vortex·vc/(r+vc)·r` tends to the constant `vortex·vc`, so
    // the *linear* displacement the twist produces is bounded at every radius
    // and cannot be modulated into a dotted line at the rim. A gaussian falloff
    // has a maximum of the linear displacement somewhere in the middle of the
    // frame and no bound on where the mod range puts it.
    // ## The winding budget, which is what sets this number
    //
    // The arm's spiral is `∫ twist dt` over the parcel's transit, and the budget
    // is **one sector width** (τ/K = 1.57 rad): wind less and there is no
    // spiral, wind more and every stem's material sweeps through its
    // neighbours' wedges and the four hues merge — which is what "per-stem
    // identity is not reading" looks like from the inside.
    //
    // At 0.032 with vc = 0.25, a parcel born at r = 0.19 sees ~0.018 rad/step
    // and takes ~48 steps to reach r = 0.3 and ~82 to reach r = 0.5, for ~1.2
    // rad of winding over the bright part of its life. That is three quarters of
    // a wedge: a clearly curved arm that is still recognisably in the sector it
    // started in.
    //
    // Down from 0.055, which spent ~2.1 rad — more than a full wedge — on top of
    // a rigid `rotate` spending another 0.6. The first build's arms could not
    // have stayed in their sectors even if the blur had left any arms.
    //
    // …and down again from 0.032 to 0.022, on a measurement rather than on the
    // arithmetic — and the measurement is about the *seeded spread*, not about
    // this number on its own.
    //
    // Measured on free-fall, 30 s of accumulation in a dense chorus, three seeds,
    // reading the live `vortex` the modulator actually ran:
    //
    //     seed 777       vortex 0.0147   mean 95   black 18%   meanRGB 33,109,137
    //     seed 20240607  vortex 0.0318   mean 72   black 35%   meanRGB 14, 85,110
    //     seed 12345     vortex 0.0359   mean 47   black 52%   meanRGB 19, 52, 78
    //
    // Winding is monotonically the enemy: the more of it a seed draws, the darker
    // and emptier the frame and the more the four hues collapse into one. That is
    // the budget argument above, confirmed from the outside.
    //
    // What the table can actually control here is the *distribution*, because the
    // running value is the shipped default plus a seeded jitter draw plus the
    // music's excursion — at the old 0.032 the seeded base alone reached 0.052,
    // i.e. ~1.95 rad, over budget before the music said anything. At 0.022 the
    // seeded base over 512 seeds spans 0.002…0.042 with a median of 0.023, and
    // its worst case lands at exactly the 1.57 rad budget rather than past it.
    //
    // ## The budget above was wrong, and the three seeds are what corrected it
    //
    // "One sector width" is too generous, because a sector is not empty before
    // the winding starts. A layer's four sources are already spread across
    // `sectorSpread · wedge` by their sway phases, so the arm's angular footprint
    // is `spread·wedge + W`, and neighbours stay apart only while
    //
    //     sectorSpread·wedge + W  ≤  wedge      ⇒   W ≤ (1 − sectorSpread)·wedge
    //
    // At the `sectorSpread` of 0.72 this shipped with, that gap is 0.44 rad — a
    // quarter of what the budget claimed. The seeds land exactly where that
    // predicts: seed 777 (W ≈ 0.55, ~25% overlap) looks good, seed 12345
    // (W ≈ 1.35, ~75% overlap) collapses. The measurement did not just confirm
    // the argument, it found the arithmetic error in it.
    //
    // ## Two conclusions follow, and only one of them is about this slot
    //
    // First, the gap is a *shared* budget and `rotate` was spending most of it
    // unnoticed. A rigid rotation does not move the sectors relative to each
    // other — but it does move the deposited material relative to its own
    // stationary source, which is the same shear. Per unit it is 82/37.5 = 2.2×
    // as potent as this slot (it acts for the whole transit, where the vortex
    // profile falls off as the parcel travels out), and its old ceiling of 0.020
    // running spent 1.6 rad on its own — more than this slot's worst case. It is
    // tightened alongside; see its own note.
    //
    // Second, `sectorSpread` came down 0.72 → 0.58, which widens the gap from
    // 0.44 to 0.66 rad. That is the cheapest 50% of headroom available: it costs
    // a little separation between the four knots of one layer (still ~4 stamp
    // widths apart at the vocals ring) and buys winding everywhere.
    //
    // ## What this slot is set to, and what it costs
    //
    // Sized so the *running* value — default + seeded jitter + the music's
    // excursion, which is the only quantity that matters — stays where the
    // measurement says the image works:
    //
    //     seeded base    0.016 ± 0.005  =  0.011 … 0.021      (W 0.41 … 0.79 rad)
    //     + excursion    ± 0.007        =  0.004 … 0.028      (W 0.15 … 1.05 rad)
    //
    // ## The check that actually settled these three slots
    //
    // The right quantity is not W and not the gap but their **ratio**, and the
    // measured seeds calibrate it directly: seed 777 looked good at W/gap ≈ 2.0
    // and seed 12345 looked poor at ≈ 3.8. Note what that says — *zero overlap
    // was never the target*. Two arms may cross; what kills the image is one
    // stem's material lying across two neighbours at once.
    //
    // Swept over 512 seeds against the shipped table, taking `W/gap` with the
    // seeded base alone, at an ordinary loud moment (|tanh| ≈ 0.55, the table's
    // own excursion rule) and at a saturated one:
    //
    //                    min   p10   median  p90   max   %over 3.0
    //     base           0.65  0.96  1.30    1.71  2.25   0%
    //     typical loud   1.26  1.67  2.19    2.80  3.36   5%
    //     saturated      1.93  2.47  3.04    3.61  4.00  53%
    //
    // The median ordinary-loud seed now sits at 2.19, i.e. essentially on the
    // seed that measured good, and no seed's *base* reaches the poor band at
    // all — where before a p90 seed got there on the draw alone. The saturated
    // row is a deliberately pessimistic bound: it moves all three slots to their
    // full excursion in the bad direction at once, and they answer to different
    // drivers, so the joint case is far rarer than the marginal one. A drop
    // winding the tunnel up is also the correct thing for a drop to do.
    //
    // The cost is real and it is the music's say over the curl: `half` goes 0.04
    // → 0.007. What it buys back is that the *whole* remaining range is legible
    // — 0.004 to 0.028 is still a 7× swing in winding, from a nearly straight
    // beam to a third of a turn, where before roughly half the reachable range
    // was in the band that merges the hues. A knob whose top half destroys the
    // image is not expression, it is a hazard with a slider on it. And the range
    // is not gone: `motion` at 2 doubles this outright, so the big curl is still
    // available as a deliberate performance choice rather than as a seed lottery.
    //
    // `lo` is 0.004 rather than negative, which is the other half of the fix.
    // Nebula's lesson is that a seed whose swirl came out near zero has no arms
    // at all; here the old symmetric ±0.11 meant a strong negative excursion
    // parked the tunnel at zero curl. A strictly positive floor makes "some
    // curl, always" structural instead of hoped for.
    def: 0.016,
    mod: add('structure', 0.004, 0.034, 0.007, 0.005),
    macro: 'motion',
  },
  {
    key: 'vortexCore',
    label: 'vortex core radius (twist halves here)',
    folder: 'warp · tunnel',
    // SLOW: this sets *where* the tunnel's spin lives, which is a compositional
    // fact of the same kind as the vanishing point's position. It is also the
    // slot that decides which layers are in the swirling part of the shaft and
    // which are in the streaming part, and swapping that around every bar would
    // make the per-stem characters trade places.
    cls: CLASS_SLOW,
    min: 0.02,
    max: 1.5,
    step: 0.005,
    // Outside the vocals ring (0.19) and inside the bass ring (0.30), so the
    // shipped composition is: drums and vocals spun hard near the vanishing
    // point, bass and other streaming out almost straight. That contrast is most
    // of what makes the frame read as having a near and a far.
    def: 0.25,
    mod: mul('structure', 0.06, 0.9, 0.35, 0.5),
    // No macro. `scale` would be the tempting one and it is exactly wrong: a
    // *larger* core radius makes the vortex softer and flatter, so `scale` would
    // mean "bigger" on `sourceRadius`/`coreSize`/`beam` and "gentler" here.
    // Same objection nebula raises against putting `scale` on `noiseScale`.
  },
  {
    key: 'pulseZoom',
    label: 'impulse → tunnel lurch (added to the zoom rate)',
    folder: 'warp · tunnel',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.12,
    step: 0.0005,
    // Small, because it is added to a per-step rate that **compounds**, and the
    // chassis caps `g.pulse` at 2 (see `updatePulse`). At 0.010 and a capped
    // pulse of 2 the flow rate goes from 0.004 to 0.024 — a 6× shove — for the
    // length of the envelope, which over a ~20-step decay is a ~1.5× expansion
    // of the whole field. That is a lurch you feel. At 0.04 the same envelope
    // expanded the frame 8× and the image was simply gone by the time the
    // envelope ended, which is a jump cut rather than a hit.
    //
    // Note the lurch is bigger here than nebula's identical-looking 0.008,
    // because the perspective term multiplies it at the rim: a kick that adds
    // 0.02 to the *uniform* rate adds it at every radius, on top of a rim that
    // was already the fast part. The frame flies apart from the outside in,
    // which is what a drop should look like down a shaft.
    def: 0.01,
    mod: add('structure', 0, 0.04, 0.011, 0.009),
    // No macro, for the same reason nebula gives: `motion` on a compounding
    // per-step kick is a knob whose top end is a discontinuity.
  },

  // ── feedback · memory ──────────────────────────────────────────────────────
  {
    key: 'fade',
    label: 'fade / step  (memory = 1 / this)',
    folder: 'feedback · memory',
    // The substrate's relaxation time, stored as the LOSS rate: memory is
    // `1/fade` steps, so 0.010 is 100 steps or 1.7 s. Nothing may be modulated
    // faster than the substrate's own relaxation time, which makes this MEDIUM.
    //
    // Note this is the *floor* on how long a parcel can live and not the whole
    // story: the field also loses light to the expansion (`1 − J`), which near
    // the rim is several times `fade`. What that means in practice is that
    // `fade` governs the slow-moving material near the vanishing point and the
    // Jacobian governs the fast material at the rim — which is the correct
    // division, because it makes the far end of the shaft the part with the long
    // memory. 100 steps is chosen against the ~80–100 step transit a parcel now
    // takes to cross the frame: shorter and arms are cut off before they are
    // drawn, longer and the vanishing-point glow smears into a permanent haze.
    //
    // Stored as the loss rather than as `decay = 1 − loss` for the reason spelled
    // out on `zoomRate`: a ln-space spec on 0.988 is a spec on the wrong
    // quantity. On the loss rate a multiplicative spec is exactly right —
    // doubling the loss halves the memory — and `half` 0.25 reads as memory
    // ×0.78 to ×1.28.
    cls: CLASS_MEDIUM,
    min: 0.0005,
    max: 0.5,
    step: 0.0005,
    def: 0.01,
    mod: mul('decay', 0.004, 0.05, 0.25, 0.2),
  },
  {
    key: 'horizon',
    label: 'horizon fade (extra loss toward the rim)',
    folder: 'feedback · memory',
    cls: CLASS_MEDIUM,
    // A per-step loss proportional to r², i.e. a memory that shortens with
    // distance from the vanishing point. Three jobs, and the third is why it is
    // not optional:
    //
    //   art       the corners go dark, so the frame reads as a shaft receding
    //             into black rather than as a disc pasted on a rectangle.
    //   motion    material dissolves as it passes the camera instead of
    //             reaching the edge at full brightness and stopping there.
    //   safety    the sampler is mirror-repeat (see `VizFxSim.init`), so light
    //             pushed past the frame edge folds back inside and re-enters
    //             travelling the *wrong way*. At the rim this term is the
    //             dominant loss for every θ the table can produce, so the fold
    //             never has anything bright to fold.
    //
    // The last one is the reason this is not just a vignette in the grade: a
    // grade cannot stop the feedback loop from accumulating, and the accumulation
    // is what would be visible.
    min: 0,
    max: 0.2,
    step: 0.0005,
    // 0.009, halved from the first build's 0.018. That build measured 15.6%
    // black pixels in a quiet passage and a frame whose whole outer third was
    // dead, and this term was one of the two causes (the other, larger one being
    // sources modulated off-screen): the expansion's own Jacobian already dims a
    // parcel ~17× on the way out, so doubling that with an explicit rim loss
    // meant nothing survived to draw the tunnel's walls.
    //
    // 0.009 still gives the rim (r = 1) a memory of ~53 steps against the
    // centre's 100, which is a clear gradient, and it remains the dominant loss
    // beyond r ≈ 1 where the mirror fold lives — so the safety argument is
    // untouched. Below about 0.004 the corners stop going black and the fold
    // becomes visible as a faint counter-flowing haze along the edges.
    def: 0.009,
    mod: mul('decay', 0.002, 0.05, 0.35, 0.4),
  },
  {
    key: 'streakMix',
    label: 'streak mix (per-step motion blur along the flow)',
    folder: 'feedback · memory',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    // **This slot shipped at 0.55 and that was the single largest defect in the
    // first build.** The argument for it was that the smear's gain is provably 1
    // so it could not hurt — true about energy, and irrelevant, because an
    // averaging kernel applied sixty times a second is a diffusion and diffusion
    // destroys detail while conserving mass. Measured: a smooth blue/cyan fog
    // with faint concentric banding and no visible sources, arms or sectors.
    //
    // 0.30 now, and the slot means something narrower: close the residual gap
    // between two consecutive stamps that the stamp-size floor in draw.wgsl
    // leaves open. Ceiling down from 0.9 to 0.65 — past about 0.7 the field is
    // being averaged with itself hard enough that the *sources* smear, and the
    // sources are the only sharp features the image has.
    def: 0.3,
    mod: add('decay', 0, 0.65, 0.18, 0.16),
  },
  {
    key: 'streakLength',
    label: 'streak length (× one step of flow)',
    folder: 'feedback · memory',
    // SLOW: the smear compounds, so the steady state takes a memory to
    // establish and a fast lane on it would be asking for a change the image
    // cannot show.
    cls: CLASS_SLOW,
    min: 0.2,
    max: 8,
    step: 0.05,
    // Quoted as a multiple of *one step's displacement at this pixel*, which is
    // why one number works for a flow that varies 16× across the frame: the
    // smear automatically gets longer where the tunnel is faster, which is the
    // anisotropy real motion blur has.
    //
    // 1.2 — just over one step — because that is what "close the gap between
    // consecutive stamps" literally means, and it is now all this slot is for.
    // The first build ran 4.5 with a hard ceiling of 20 on the theory that this
    // was the visual's reach mechanism; it is not (reach is transit time — see
    // the long note above the table), and at 4.5 it was a diffusion that erased
    // the image. The ceiling came down with it: at 20 against a modulated rim
    // flow the kernel was a fifth of the frame, applied every step.
    def: 1.2,
    mod: mul('decay', 0.4, 4, 0.35, 0.4),
    macro: 'motion',
  },

  // ── colour · chroma ────────────────────────────────────────────────────────
  {
    key: 'chromaShift',
    label: 'hue rotation / step (rad)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    min: -0.12,
    max: 0.12,
    step: 0.0005,
    // Smaller than nebula's 0.004, and it came down from 0.005 with the retuned
    // flow — for the arithmetic reason, not a taste one. Hue travel along an arm
    // is `chromaShift × transit`, and the transit went from ~30 steps to ~90 when
    // the flow slowed. At 0.005 that is 0.45 rad of rotation between the root of
    // an arm and its tip, which is most of the way from orange to pink: enough
    // to move a stem's material *out of its own palette entry*, which on a
    // visual whose per-stem identity is carried by hue is exactly the wrong
    // trade. 0.003 over 90 steps is ~0.27 rad — a visible warm-to-cool drift
    // along an arm that still lands recognisably near where it started.
    def: 0.003,
    // The `matrix` group — "how the voices relate" — because this is what turns
    // four independent palette entries into one continuous gradient they all
    // travel along.
    mod: add('matrix', -0.028, 0.028, 0.011, 0.013),
    macro: 'chroma',
  },
  {
    key: 'layerBlend',
    label: 'layer blend  (1 = additive · higher = loudest hue wins)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    // Floored at 1 rather than 0, and the floor is the whole design: 1 is
    // *provably* the plain additive sum (see the derivation in draw.wgsl), so
    // the bottom of this slider is the physical answer rather than a degenerate
    // one. The shader clamps it too.
    min: 1,
    max: 8,
    step: 0.05,
    // Additive light makes an orange beam crossing a cyan one literally white in
    // linear rgb — (1.03, 0.88, 1.01) off this palette, before exposure or the
    // tone map are involved, so nothing downstream can recover the hue.
    //
    // Sectors reduce how often that happens compared with nebula's concentric
    // orbits — two layers own different wedges, so their material only meets
    // where the vortex has sheared one into another's sector — but they do not
    // remove it, and the places where it *does* happen (a hard-twisted core, a
    // strongly modulated `sectorSpread` past 1) are exactly the places the image
    // is most dense.
    //
    // 2.5 rather than nebula's 3, after the first build measured mean red at
    // 4/255 with all four stems loud. The exponent was **not** the cause — that
    // was tested directly, with the `chroma` macro at 0 forcing plain additive,
    // and the red did not come back — but it is an aggravating factor worth
    // trimming while the real fix (spatial: see `sourceRadius`) lands. The
    // stem-follow lane routinely gives one layer 5× another's composed
    // luminance; at 3 that is 125× the say in the hue, so a quiet stem cannot
    // tint a loud one at all. At 2.5 it is 40× — still decisive where two arms
    // genuinely cross, still leaving a faint layer able to warm or cool a bright
    // one rather than being erased by it.
    def: 2.5,
    mod: mul('matrix', 1, 6, 0.3, 0.3),
    macro: 'chroma',
  },

  // ── emitters · light ───────────────────────────────────────────────────────
  {
    key: 'sectorSpread',
    label: 'sector spread  (0 = four spokes · 1 = wedges just touch)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    // How much of its own 90° wedge a layer's emitters occupy. 0 puts every
    // emitter on its sector's centre line and the tunnel is four hard spokes; 1
    // fills the wedges so they meet and the tunnel is a continuous ring with
    // four coloured quadrants; past 1 the wedges overlap and the stems begin to
    // blend into each other's territory.
    //
    // The hard max is 1.6 — overlap is a legitimate look to reach for by hand,
    // and it is where `layerBlend` starts earning its keep — but the mod range
    // now stops at **0.72**, down from 1.2, and the excursion and jitter came
    // down with it. That is the single largest of the three changes in this
    // pass, and it was found by measurement rather than by reading:
    //
    // Sweeping 512 seeds, the *gap* (see below) turned out to be a bigger
    // lottery than the winding it has to contain. At jitter 0.28 and half 0.30
    // the running spread reached 0.95, i.e. a gap of 0.078 rad — at which point
    // no winding budget exists at all and every seed collapses regardless of
    // what `vortex` is doing. That single tail was responsible for the median
    // `W/gap` at an ordinary loud moment being 3.10 (worse than the seed that
    // measured *poor*) while the median seed's own base sat at a healthy 1.32.
    // Capping the spread at 0.72 guarantees a gap of at least 0.44 rad for every
    // seed at every moment, and dropped that median to 2.19.
    //
    // The lesson generalises past this slot: when two slots multiply into one
    // budget, the *distribution* of the quieter one is as load-bearing as the
    // default of the loud one, and neither shows up by reading either in
    // isolation.
    min: 0,
    max: 1.6,
    step: 0.01,
    // ## This number IS the winding budget, which its name does not suggest
    //
    // A layer's four sources are spread across `spread · wedge` by their sway
    // phases, so an arm's angular footprint is `spread·wedge + W`, where W is
    // however far the warp winds it. Neighbouring stems stay apart only while
    //
    //     spread·wedge + W  ≤  wedge      ⇒     W  ≤  (1 − spread) · wedge
    //
    // Every radian of source spread is therefore a radian of spiral given up. At
    // the 0.72 this shipped with, the gap left for the entire warp was 0.44 rad
    // — and the measured seeds were winding 0.5 to 1.4 rad, which is precisely
    // why the four hues collapsed into the loudest one in a dense chorus.
    //
    // 0.58 leaves 0.66 rad, a 50% larger budget, and this is the cheapest place
    // to find it. The only cost is how far apart one layer's own four knots sit,
    // and at the vocals ring (0.19) with a stamp radius of ~0.02 they are still
    // spread over 0.91 rad ≈ 0.17 of arc — about four stamp widths, so they
    // still resolve as separate sources rather than as one blob on the centre
    // line. The alternative was to spend the same 50% on the vortex, and the
    // vortex is the thing anyone actually looks at.
    //
    // Below about 0.4 the four knots merge and a wedge reads as a single spoke,
    // which loses the sense that a stem is a *bundle* — and it is the bundle
    // that makes "the vocal entered" read as a quadrant lighting up rather than
    // as one line getting brighter.
    def: 0.58,
    // The `matrix` group, and it earns it: this is literally the rule for how
    // much of each other's space the four voices occupy.
    mod: add('matrix', 0.15, 0.72, 0.14, 0.1),
  },
  {
    key: 'swaySpeed',
    label: 'sector sway speed (rad/s of oscillation within the wedge)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 3,
    step: 0.005,
    // The emitters *oscillate* within their wedge rather than orbiting round the
    // centre, and that is forced by the sector partition rather than chosen: an
    // emitter that orbited would leave its layer's wedge, and wrapping it back
    // at the boundary would be a visible teleport every cycle. An oscillation
    // has no wrap. Each emitter carries its own phase and rate multiplier from
    // the seeded constellation, so four emitters in one wedge wag out of phase
    // and read as a bundle of vanes breathing rather than as a metronome.
    //
    // Slow: 0.22 rad/s is a ~29 s round trip across the wedge. Faster than about
    // 1 rad/s and the source ring stops being a place and becomes a shimmer,
    // which throws away the one thing sectors buy. Down a notch from 0.28 with
    // the radii — the same angular rate is a shorter arc at a smaller radius, so
    // holding the *linear* sway speed constant meant slowing the angle.
    def: 0.22,
    mod: mul('structure', 0.03, 1.2, 0.4, 0.45),
    macro: 'motion',
  },
  {
    key: 'halo',
    label: 'halo ×  (wide lobe vs core)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    // 0.26, well under nebula's 0.42, and down from the 0.40 the first build
    // shipped. Two reasons, and the second is the one that was measured.
    //
    // The stamp is already stretched by `beam`, so the halo is elongated too and
    // the same coefficient covers twice the area it would in nebula. And the
    // halo is the term that makes a layer's light *non-local*: `exp(-d·1.7)` is
    // still at 3% of peak five stamp-radii out, which the field's memory then
    // multiplies, so at 0.40 every layer was laying a wide soft wash over the
    // whole frame and the loudest one simply painted over the other three. That
    // is the mechanism behind a frame whose mean red was 4/255 with all four
    // stems loud. Locality is per-stem identity here.
    //
    // Not lower, because the halo is still the only thing filling the space
    // between the four vanes — there is no outward-spreading gas in this visual,
    // only arms — and the vanishing-point glow in draw.wgsl now covers the one
    // place the halo genuinely could not reach.
    def: 0.26,
    mod: mul('population', 0.05, 3, 0.4, 0.4),
    macro: 'light',
  },
  {
    key: 'beam',
    label: 'beam ×  (radial stretch of every stamp)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    // Floored at 1 — a stamp shorter than it is wide would be a *tangential*
    // smear, i.e. an arc, which is the shape this visual is trying not to be.
    min: 1,
    max: 12,
    step: 0.05,
    // The single cheapest thing that makes a source look like it belongs in a
    // tunnel: the stamp is an ellipse whose long axis points at the vanishing
    // point, so even a stationary emitter reads as a shaft of light rather than
    // as a dot. It is also continuity insurance — `draw.wgsl` divides the radial
    // stamp-size floor by this — so raising it lets the flow go faster for free.
    //
    // 2.0, down from 2.4, and the ceiling down from 8 to 5. Past about 3 the
    // sources stop having a locatable *position* — they are already streaks
    // before the warp touches them — and the source ring, which is the thing
    // that makes the sectors legible, dissolves. At the old ceiling of 8 a
    // modulated stamp was 0.15 heights long against a 0.19 source radius, i.e.
    // one emitter reaching from the vanishing point to beyond its own ring.
    def: 2,
    mod: mul('structure', 1, 5, 0.4, 0.4),
    macro: 'scale',
  },

  // ── events · shockwaves ────────────────────────────────────────────────────
  {
    key: 'ringGain',
    label: 'shockwave brightness ×',
    folder: 'events · shockwaves',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    // A little under nebula's 1.4: the rings here are drawn in the tunnel's own
    // polar space (concentric with the vanishing point, hot at the struck
    // bearing — see draw.wgsl), so a ring covers a whole annulus rather than a
    // disc-sized arc and the same gain is more total light.
    //
    // There is no `ringExpand` slot, deliberately: the expansion is already
    // authored per event kind in the impulse workbench (`ResponseConfig
    // .splashPush`, which arrives as `splashes[i].params.x`), and a second
    // global multiplier on top of a per-kind one is two knobs on one number —
    // the objection this table applies to `exposure` and `gamma` as well. The
    // slot budget went to `beam` instead.
    def: 1.3,
    mod: mul('population', 0.2, 4, 0.5, 0.35),
    macro: 'light',
  },

  // ── owned by the shared HDR folder ─────────────────────────────────────────
  // In θ (so they are saved and explored) and excluded from modulation (there is
  // an auto-exposure controller downstream of both; modulating scene exposure
  // makes the controller chase it). No `folder`, so the panel builds no slider —
  // `ui/render-folder.ts` binds them, and two widgets on one number is a bug
  // waiting for a drag.
  {
    key: 'exposure',
    label: 'scene exposure',
    cls: CLASS_MEDIUM,
    min: 0.005,
    max: 2,
    step: 0.001,
    // 0.1, the same decade as nebula and plife, and for the same reason: this
    // only has to put the frame in the right decade because auto-exposure adapts
    // on top of it. The field's absolute scale is set by `DRAW_SCALE` in
    // draw.wgsl and nowhere else — see the arithmetic there, which is calibrated
    // against nebula's so that this number can stay at the family's value.
    def: 0.1,
    mod: null,
  },
  {
    key: 'gamma',
    label: 'display gamma',
    cls: CLASS_MEDIUM,
    min: 1,
    max: 3,
    step: 0.05,
    def: 2.2,
    mod: null,
  },
];

/**
 * Shipped hues. The same four nebula, physarum and plife use for the same four
 * stems — bass is orange in all of them — because that cross-substrate identity
 * is the one piece of the palette worth not re-deciding. What is different here
 * is *where* they are: four fixed quadrants of the shaft rather than four orbits
 * at four radii, so the palette is read as a map rather than as a set of moving
 * objects.
 */
const PALETTE_HEX: readonly string[] = [
  '#ff7a1a', // bass
  '#ff2f6d', // drums
  '#35d6ff', // vocals
  '#a56bff', // other
];

export const TUNNEL: VizFxVisual = {
  id: 'tunnel',
  title: 'terrarium · tunnel',
  // K = 4, and the four species ARE the four stems, in the analysis contract's
  // own order (`STEM_NAMES` in timeline/types.ts). Here the keying is doubly
  // structural: `stemMap()` is the identity, the palette is per stem, and layer
  // k additionally owns angular sector k of the tunnel, which the warp preserves
  // by construction (see the header). "The vocal entered" is one quadrant of the
  // shaft lighting up, and no seed can rewire it.
  speciesCount: 4,
  layerNames: ['bass', 'drums', 'vocals', 'other'],
  paletteHex: PALETTE_HEX,
  // Four per layer, sixteen in all — the same count nebula settled on, but doing
  // a different job. There they scatter around a full circle; here they share a
  // 90° wedge and differ in sway phase and in radius (the constellation's `em.y`
  // is ±22%), so a layer is a small bundle of vanes at slightly different depths
  // rather than one wide arc. Three read as a flat fan and six merged into a
  // solid quadrant, taking the interior structure with them.
  emittersPerLayer: 4,
  layerSlots: LAYER_SLOTS,
  globalSlots: GLOBAL_SLOTS,
  macros: [
    { key: 'light', label: 'light  (× core gain, halo, shockwaves)', min: 0, max: 2 },
    {
      key: 'motion',
      label: 'motion  (× perspective, breath, rotation, vortex, sway, streak length)',
      min: 0,
      max: 2,
    },
    { key: 'scale', label: 'scale  (× source radius, core size, beam length)', min: 0, max: 2 },
    { key: 'chroma', label: 'chroma  (× hue rotation + layer blend)', min: 0, max: 2 },
  ],
  warpWgsl,
  drawWgsl,
};
