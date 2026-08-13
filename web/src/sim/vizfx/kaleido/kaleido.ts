/**
 * **kaleido** — the mirror-rosette vizfx visual.
 *
 * Same shape as `nebula/nebula.ts`: this file plus two WGSL passes, and nothing
 * else. `VizFxSim` owns the ping-pong rig and reads the table below to find out
 * what it is drawing.
 *
 * ## What it is
 *
 * The sampling domain is folded. Every pixel's angle about the frame centre is
 * reduced into one wedge of width π/n and reflected, so the frame is 2n mirrored
 * copies of a single sliver — a milkdrop mandala rather than a nebula. Both
 * passes fold: the warp folds so that the accumulated field is *forced*
 * symmetric every step (which is what makes a section-boundary injection, a
 * snapshot restore or an order change resolve into a rosette instead of staying
 * a mess), and the draw folds so that this step's fresh light is already
 * symmetric when it lands.
 *
 * The consequence worth stating up front: the draw pass evaluates **twelve
 * gaussians** and the viewer sees **a hundred and forty-four petals**. The fold
 * is free replication — the per-pixel cost is the fundamental wedge's cost, and
 * the symmetry order costs nothing at all.
 *
 * ## The theorem this visual is built around
 *
 * A mirror rosette **cannot rotate**. Not "should not" — cannot, in the sense
 * that no rigid rotation of the pattern is compatible with keeping the mirror
 * lines where they are, because rotation and reflection do not commute. And the
 * stronger form, which is what actually decided the design:
 *
 *   *At a fixed radius, the only continuous measure-preserving angular motion of
 *   the fundamental wedge onto itself is the identity.*
 *
 * (A monotone continuous bijection of an interval onto itself that preserves
 * Lebesgue measure is the identity. The only other measure-preserving option is
 * the reflection, which is a flip, not a flow.)
 *
 * So the twist term nebula uses — an angle offset that depends on radius, which
 * is area-preserving there and therefore free — is **not** free here. Its local
 * Jacobian is still 1, but the fold's 2n-to-1 gather stops being exactly
 * balanced the moment the source field's mirror axes and the fold's axes
 * disagree, which is precisely what an angular offset creates. Measured on
 * paper: with the source symmetric about the fold axes and a rigid offset ρ,
 *
 *     gain = 1 + [ ∫_{π/n-ρ}^{π/n} c − ∫_0^{ρ} c ] / ∫_0^{π/n} c
 *
 * which at ρ = 0.1 rad and n = 6 (a wedge of 0.52 rad) can reach 1.19 per step
 * against a fade of 0.01. That diverges. There is no clamp that fixes it without
 * also destroying the flow, because the imbalance is a *global* accounting error
 * and not a local one — no per-pixel factor can see it.
 *
 * The warp is therefore **purely radial**: a zoom and a radial standing wave,
 * both of which are exactly conservative through the ordinary polar Jacobian,
 * plus the fold, which contributes exactly 1. Everything angular happens in the
 * draw pass, where it is *deposition* rather than transport and costs nothing:
 * each layer's emitters advance in the pre-fold angle, so their folded positions
 * triangle-wave across the wedge and bounce off the mirror lines. That bounce is
 * literally what a kaleidoscope does, so the constraint and the look agree.
 *
 * ## Why the character lives in per-layer defaults
 *
 * The fold replicates everything 2n times, so the four stems have to be
 * separable by something the fold *preserves*. It preserves radius exactly and
 * scrambles absolute angle completely. A mandala is read as concentric rings
 * anyway, so radius is the natural separator:
 *
 * - **drums** live at the hub (0.12) with the smallest petals and the fastest
 *   sweep. Near the centre all 2n copies converge, so a hit is a single bright
 *   flash at the middle of the rosette rather than 12 separate dots.
 * - **vocals** at 0.21, the first real ring out. Fast enough to visibly swing
 *   between the mirror lines, so the voice arriving is a ring that starts moving.
 * - **bass** at 0.30 with the largest petals and a slow sweep — the heavy ring,
 *   the one whose arms are wide enough to be the mandala's body.
 * - **other** at 0.40, nearly stationary. Pads become the outer border the rest
 *   of the figure is drawn inside.
 *
 * ## …and why radius alone was not enough (measured, revision 2)
 *
 * Radius is *a* separator and it is not sufficient, which a 30 s headless run on
 * a dense passage showed plainly: the frame's mean rgb came out [12, 57, 93]
 * with bass and drums both at ~0.95 stem level. Two independent causes, both now
 * fixed here rather than by weakening `layerBlend` (weakening it toward 1 is
 * plain additive light, which is how saturated primaries go white in the first
 * place — the answer is separation, not blending).
 *
 * **1. The palette is not luminance-neutral, and `layerBlend` cubes that.** In
 * linear light the four shipped hues have Rec.709 luminances of 0.351 (bass
 * orange), 0.244 (drums pink), 0.560 (vocals cyan) and 0.258 (other violet). The
 * hue weight is `(weight · luma)^layerBlend`, so at the shipped 3.5 the vocals
 * layer carries (0.560/0.244)^3.5 ≈ 18× the hue claim of drums *for identical
 * geometric presence*. The frame was cyan because the palette made it cyan. The
 * `emitGain` defaults below are therefore no longer authored as taste — each is
 * 0.40/luma, so `gain · luma` is 0.40 for every layer and all four have equal
 * light and equal hue claim at equal energy. (Total light is unchanged, so
 * `DRAW_SCALE`'s calibration still holds: the mean of `gain · luma` went 0.396 →
 * 0.400.)
 *
 * **2. Each layer now owns an angular LANE inside the fundamental wedge**, a
 * structural constant in draw.wgsl rather than a θ slot. This is the second
 * separator the fold preserves, and it is the one the radial flow cannot erode —
 * *because* the warp has no angular transport (see the theorem above), a lane is
 * permanent: light deposited in it streams outward along it forever. The lanes
 * are (centre, half-width) as fractions of the wedge: bass (0.74, 0.20), drums
 * (0.50, 0.34), vocals (0.28, 0.20), other (0.14, 0.12). Every pair is disjoint
 * in at least one axis — bass/vocals and bass/other by lane, everything else by
 * radius — so no layer can repaint another's pixels however loud it gets.
 *
 * They are constants and not slots on purpose, and for the reason nebula gives
 * for its counter-rotation: this is the stems' *address*. A slider that lets the
 * music move it is a slider that lets the music erase the stem labels. What the
 * seed still moves is the lane *offset* within ±0.08, the oscillation phase, and
 * the radius/speed/size multipliers — so a reroll is a different mandala and the
 * same four instruments.
 *
 * Layer index also alternates the sweep direction (structural, in draw.wgsl), so
 * neighbouring rings counter-sweep and the figure never reads as one rigid body.
 *
 * ## The θ table: 36 slots, 30 of them modulated
 *
 * 5 per layer × 4, plus 16 globals. Every `lo`/`hi` is strictly inside the hard
 * `min`/`max` beside it — the hard bound is what a file may contain and what a
 * hand slider may reach, the ModSpec range is where the music may wander
 * unsupervised.
 *
 * Excursion sizing follows physarum's rule: with a z-scored input and a unit
 * projection `w·ẑ` is roughly N(0,1), so the typical |tanh| at depth 1 is about
 * 0.55 and every `half` below is about twice the excursion to expect.
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
 * The petals are **smaller than nebula's** — 0.9% to 2.2% of screen height
 * against its 1.4%–3.5% — and that is a consequence of the fold rather than a
 * matter of taste. The fold puts 2n copies of every emitter on screen, so at the
 * shipped order this table paints 144 petals where nebula paints 16. Coverage
 * is what sets the image's peak-to-mean ratio, and peak-to-mean is what the
 * grade has to work with: at nebula's sizes this visual covers ~10% of the frame
 * before the warp has stretched anything, which the tone map flattens into one
 * even plate. At these sizes it covers ~2.4%, four times nebula's, which is as
 * full as a mandala should be and still leaves the cores something to be brighter
 * *than*.
 *
 * The radii are the per-stem keying and are documented in the header. Note they
 * are deliberately spaced with gaps: at ±22% emitter radius jitter the four
 * bands are [0.094,0.146], [0.164,0.256], [0.234,0.366], [0.312,0.488], so the
 * only overlaps are between neighbours and `layerBlend` handles those.
 */
const LAYER_CHARACTER: readonly {
  emitGain: number;
  armRadius: number;
  armSpin: number;
  petalSize: number;
}[] = [
  // `emitGain` is 0.40 / (the palette hue's linear Rec.709 luminance), not a
  // judgement — see cause 1 in the header. luma: 0.351, 0.244, 0.560, 0.258.
  { emitGain: 1.14, armRadius: 0.3, armSpin: 0.13, petalSize: 0.022 }, // bass
  { emitGain: 1.64, armRadius: 0.12, armSpin: 0.42, petalSize: 0.009 }, // drums
  { emitGain: 0.71, armRadius: 0.21, armSpin: 0.24, petalSize: 0.014 }, // vocals
  { emitGain: 1.55, armRadius: 0.4, armSpin: 0.11, petalSize: 0.019 }, // other
];

const character = (key: keyof (typeof LAYER_CHARACTER)[number], fallback: number) =>
  (layer: number): number => LAYER_CHARACTER[layer % LAYER_CHARACTER.length]?.[key] ?? fallback;

const LAYER_SLOTS: readonly VizSlot[] = [
  // Excluded from modulation, the same call all three substrates make and for the
  // same reason: light is driven by the stem-follow lane, and modulating the base
  // it scales puts the constant unexplained flashing straight back.
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
    label: 'petal gain ×',
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
    key: 'armRadius',
    label: 'ring radius (screen heights)',
    // SLOW, and more emphatically so than nebula's orbit radius: this is the ONLY
    // fold-invariant per-stem cue. Absolute angle is scrambled by the fold, so if
    // the four rings swapped places every bar there would be no way left to tell
    // the stems apart except colour, and the whole per-stem keying would be
    // carried by the palette alone.
    cls: CLASS_SLOW,
    min: 0,
    max: 0.9,
    step: 0.005,
    def: character('armRadius', 0.25),
    mod: mul('structure', 0.05, 0.6, 0.3, 0.4),
    macro: 'scale',
  },
  {
    key: 'armSpin',
    label: 'lane sweep (peak rad/s)',
    // The angular life of the whole visual, and the reason it is a *draw* slot
    // rather than a warp slot is the theorem in the header: transport cannot move
    // material angularly without breaking the fold's energy balance, but moving
    // the SOURCE is free, because deposition is not transport.
    //
    // Since revision 2 the petal oscillates inside its layer's angular lane rather
    // than free-running around the circle, so this is the sweep's **peak tangential
    // rate** — draw.wgsl divides it by the lane's half-width to get the oscillator
    // frequency, which keeps the units and the shipped numbers meaning exactly what
    // they did before the lanes existed. One rad/s is a traverse of a 6-fold wedge
    // (π/6 = 0.52 rad) in about half a second, which is why the ceiling is 1.2.
    //
    // Bass 0.13 and `other` 0.11 rather than the 0.10 / 0.07 revision 1 shipped.
    // Those are the two large-petal layers, and a near-stationary large petal
    // integrates 100 steps of deposit into one spot: at 0.07 rad/s and r = 0.40 a
    // petal moves 2.5 of its own widths over the field's memory, which is a hot
    // core rather than an arm and is the likeliest source of the 10% of the quiet
    // frame that measured clipped. At 0.11 it moves 4 widths.
    cls: CLASS_MEDIUM,
    min: 0,
    max: 4,
    step: 0.005,
    def: character('armSpin', 0.2),
    mod: mul('structure', 0.02, 1.2, 0.45, 0.5),
    macro: 'motion',
  },
  {
    key: 'petalSize',
    label: 'petal size (screen heights)',
    cls: CLASS_MEDIUM,
    min: 0.002,
    max: 0.4,
    step: 0.001,
    def: character('petalSize', 0.016),
    // `lo` sits under the smallest shipped default (drums, 0.009) rather than on
    // it, so the seeded draw has room on both sides. It is also well under the
    // continuity floor draw.wgsl applies, which is the slot that actually protects
    // the small end — see the note on the floor there.
    mod: mul('structure', 0.005, 0.14, 0.3, 0.4),
    macro: 'scale',
  },
];

/**
 * ## The two relationships that decide whether this looks like a mandala
 *
 * **1. Sweep against memory.** A petal's arm is the trail it leaves while the
 * field still remembers it, and here the trail is *angular*: the emitter sweeps
 * across the wedge while the radial flow drags the deposit outward. So the arm's
 * angular extent is
 *
 *     arc ≈ armSpin × memory × dt,     memory ≈ 1 / fade  steps
 *
 * At the shipped fade (0.01, i.e. 100 steps = 1.7 s) and the vocals' 0.24 rad/s
 * that is 0.4 rad of sweep — most of its 0.20-wide lane in a 6-fold wedge, so the
 * vocals' ring reads as a fan rather than as a dot. Bass at 0.13 rad/s draws
 * 0.22 rad, a fat lobe. `other` at 0.11 is the near-static border. Those numbers
 * *are* the composition; moving `fade` without moving them re-composes the frame.
 *
 * **2. The fold count against everything.** `foldCount` divides the wedge, and the
 * lanes are expressed as *fractions* of it, so raising n narrows every lane
 * proportionally and the whole figure gets finer rather than merely more numerous.
 * That is why the lanes are fractional and not absolute angles, and why `armSpin`
 * and `foldCount` are the pair to move together.
 *
 * ## The stamp-continuity constraint, and why it is enforced rather than respected
 *
 * An arm is a row of per-step stamps and reads as an arm only while consecutive
 * stamps overlap. Three things separate consecutive stamps here — the emitter's own
 * peak tangential travel `armSpin·dt·radius`, the radial travel the zoom gives the
 * previous stamp, `|zoom−1|·radius`, and near the hub the drain's `hubDrain/2r`.
 * At the modulation extremes (armSpin 1.2, armRadius 0.6) the first alone is 0.012
 * screen heights against a legal petal of 0.005, i.e. two and a half stamp-widths
 * of gap, and the arm becomes a dotted line. `draw.wgsl` therefore floors every
 * stamp at 1.2× the displacement it is about to undergo, which is what lets the
 * ranges above be chosen for how the sweep looks rather than for what the smallest
 * petal can survive.
 */
const GLOBAL_SLOTS: readonly VizSlot[] = [
  // ── kaleido · fold ─────────────────────────────────────────────────────────
  {
    key: 'foldCount',
    label: 'symmetry order n  (2n mirrored copies)',
    folder: 'kaleido · fold',
    /**
     * ## The symmetry order is modulated, snapped, and SLOW — and here is why the
     * other two answers lose
     *
     * The order has to be an integer: at a non-integer n the wedges do not tile
     * the circle and the last partial wedge meets the first at a hard seam, which
     * is a crack across the rosette rather than an in-between symmetry. So the
     * shader rounds. Three ways to live with that were on the table.
     *
     * **(a) Blend the two bracketing integers with a second fetch, crossfaded by
     * the fractional part.** Rejected, and not merely because of ghosting. The
     * fold lives in the *gather*: at orders n and n+1 the two source reads are at
     * unrelated angles, so a 50% crossfade is two rosettes superimposed whose
     * common symmetry is gcd(n, n+1) = 1 — i.e. an image with no symmetry at all,
     * which is the one thing this visual must never be. It also breaks the energy
     * argument outright: the fold conserves exactly *because* the source field is
     * symmetric under the same order being folded, and a blend cannot have the
     * source be symmetric under both. It would cost a second five-tap fetch as
     * well, doubling the warp's bandwidth for a worse picture.
     *
     * **(c) Excluded from modulation, hand slider only.** Safe, and it throws away
     * the best event the visual has. The field has ~100 steps of memory, so an
     * order change does not cut — the old rosette decays over 1.7 s while the new
     * one accumulates through it. That *is* the mandala re-forming, and it is
     * exactly the kind of section-scale gesture the mapping layer exists to fire.
     *
     * **(b) Snap in the shader, put it on the SLOW lane.** Chosen. The cost, stated
     * honestly, is that any hard quantiser dithers when its input parks on a
     * boundary: a value sitting at 6.5 alternates 6/7 and the field time-averages
     * the two into the unsymmetric superposition (a) was rejected for. Three
     * things bound it. The lane is SLOW, so the input is rate-limited and crossings
     * are traversals rather than jitter. The mod range spans only four reachable
     * orders (5…8) and is centred on 6, three half-integers away from either end.
     * And a single-step flip contributes ~1% of the field's content, which is
     * invisible; only a *sustained* park is a problem.
     *
     * Energy under a dither is bounded too, and that is worth writing down because
     * it is the part that could have been fatal rather than ugly. Re-folding a
     * D_n-symmetric field at order m gains 2m × (light in a π/m window). For a
     * field whose arms are narrow the window holds at most one lobe, so the gain
     * n→n+1 is ≤ (n+1)/n and the gain back is n/(n+1): the round trip is exactly 1.
     * The transient is bounded by the order ratio, and it cannot compound, because
     * the warp's output is symmetric under whatever order it just used.
     */
    cls: CLASS_SLOW,
    // Hard floor 2 rather than 1: at n = 1 the fold is a single mirror line and the
    // image is a reflected photograph, not a rosette. The shader clamps to the same
    // pair so a macro or an arithmetic slip cannot reach a degenerate order.
    min: 2,
    max: 16,
    step: 1,
    // 6 is the default because it is the order at which a wedge (30°) is wide
    // enough for a sweeping arm to be legible inside it and narrow enough that the
    // eye reads the frame as one figure rather than as a few objects. It is also
    // the reference the draw pass normalises light against — see FOLD_REFERENCE.
    def: 6,
    // Additive, because this is a *count*: "two more arms" is the meaningful unit,
    // not "×1.4 as many". The range reaches 5…8 after rounding, and both ends sit
    // 0.4 inside an integer so the extremes of a modulation swing are stable orders
    // rather than boundaries.
    mod: add('structure', 4.6, 8.4, 0.9, 0.7),
    // No macro. Its hard range does not straddle 0 and its spec is additive, so a
    // plain multiplier on it is the explosion `VizSlot.macro` warns about — ×1.5 on
    // 6 is 9, which is not "more scale", it is a different visual.
  },

  // ── warp · flow ────────────────────────────────────────────────────────────
  {
    key: 'zoomRate',
    label: 'radial flow / step  (>0 = outward)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    // Negative is legal and is a *different mandala*: light drains toward the hub
    // instead of streaming out of it. Bounded much tighter on that side because an
    // inward flow concentrates every one of the 2n copies onto the same small disc
    // at the centre, so it saturates far faster than the same magnitude outward.
    min: -0.012,
    max: 0.04,
    step: 0.0002,
    // ~1.35× expansion over the field's memory. Enough that a petal becomes a
    // radial arm; below about 0.002 the arms stop and the rosette is a ring of
    // static dots, above about 0.008 they reach the frame edge while still bright
    // and the mandala becomes a tunnel.
    def: 0.003,
    // ## Why this is a rate rather than the zoom factor itself
    //
    // Stored as the growth rate with an ADDITIVE spec, following the bug nebula
    // records shipping: a multiplicative slot holding 1.003 has a ln-space spec on
    // the wrong quantity, because the growth rate is `x − 1` and almost all of the
    // stored number is the 1. A "±0.8%" draw there is a 3.3× draw on the flow
    // speed. `fade` below is the same fix for the same reason.
    //
    // No macro, deliberately, and the same argument as nebula's: `motion` on a
    // *rate* would be defensible, but this is the one flow term that trades against
    // frame-filling rather than against liveliness, so folding it into the motion
    // knob would mean pushing "more motion" always also pushed the figure off the
    // edges.
    mod: add('structure', -0.004, 0.012, 0.0025, 0.0025),
  },
  {
    key: 'hubDrain',
    label: 'hub drain / step  (r² pull toward the centre)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.005,
    step: 0.00002,
    /**
     * ## What replaced the ring wave, and why the ring wave had to go
     *
     * Revision 1 shipped a radial standing wave here — a per-radius modulation of
     * the zoom, with its (genuinely non-unit) Jacobian included and clamped at 1.
     * Measured headless over 30 s, that clamp was the dominant defect in the whole
     * visual. The clamp can only dim, and the wave's J oscillates about 1, so the
     * *mean* per-step multiplier is `1 − k/π` with `k = r · ripple · freq`:
     *
     *     defaults, r = 0.5      k = 0.020   0.64%/step extra   memory 100 → 61
     *     mod extremes, r = 0.5  k = 0.086   2.73%/step extra   memory 100 → 27
     *     mod extremes, r = 1.0  k = 0.171   5.46%/step extra   memory 100 → 15
     *
     * against a `fade` of 1%/step. Three things made that fatal rather than merely
     * lossy. It is **radius-growing**, so it ate the periphery and the frame went
     * half black. Both slots were `structure`, so it got **worse with modulation
     * depth** — i.e. worse in a chorus, which is how a dense passage ended up
     * reading darker than a quiet one. And it is invisible in the θ table: nothing
     * about "ripple 0.004" says "this triples the substrate's decay rate".
     *
     * The general result, worth stating because it is the reason no ring wave
     * came back: **there is no area-preserving radial ring wave.** `J = f·f'/r ≡ 1`
     * integrates to `f² = r² + C`, a one-parameter family, so any oscillating
     * radial map necessarily has |J| ≠ 1 somewhere and necessarily pays for it
     * under the clamp. A warp-side ring wave can be visible or it can be cheap,
     * not both.
     *
     * ## …so this slot is that one-parameter family, which is exactly free
     *
     *     r2 = sqrt(r²/zoom² + C)      ⇒  f·f' = r/zoom²  ⇒  J = 1/zoom²
     *
     * The Jacobian is **independent of C**: this term costs the loop nothing at
     * all, at any amplitude, with no clamp of its own. (Checked against an explicit
     * area rather than only the derivative: the destination disc [0, ε] maps to the
     * source annulus [√C, √(ε²/zoom² + C)], whose area is exactly πε²/zoom². So
     * there is no singularity at the origin, no void and no concentration blow-up
     * — the hub is filled by spreading a thin annulus over a disc, and the total is
     * conserved.)
     *
     * What it *does*: the C term contributes `C·zoom/(2r)` of inward pull, which
     * decays as 1/r, so it is a hub control and nothing else. Against `zoomRate`'s
     * outward `z·r` it sets a stagnation radius `r* = √(C/2z)` — 0.22 at these
     * defaults — inside which material drains toward the centre and outside which
     * it streams to the rim. That gives the mandala a hub that fills instead of a
     * hole, which is the other half of the `%black` fix.
     *
     * No macro, for `zoomRate`'s reason: this trades against where the figure sits
     * in the frame, not against how lively it is.
     */
    def: 0.0003,
    mod: add('structure', 0, 0.0012, 0.0004, 0.0003),
  },
  {
    key: 'pulseZoom',
    label: 'impulse → radial pump',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.2,
    step: 0.001,
    // The whole-field transient, in the geometry. `g.pulse` is capped at 2 by the
    // chassis, so at 0.01 a kick expands the rosette by ~1.35× over its envelope —
    // a visible shove outward that the fold turns into all 2n arms surging at once,
    // which is a much larger gesture than the same number does in nebula. Slightly
    // above nebula's 0.008 for that reason and no higher: this compounds per step.
    def: 0.01,
    mod: add('structure', 0, 0.03, 0.008, 0.008),
  },

  // ── feedback · memory ──────────────────────────────────────────────────────
  {
    key: 'fade',
    label: 'fade / step  (memory = 1 / this)',
    folder: 'feedback · memory',
    // The substrate's relaxation time, stored as the LOSS rate: memory is `1/fade`
    // steps, so 0.01 is 100 steps or 1.7 s. By the rule every registry in this
    // project follows, nothing may be modulated faster than the substrate's own
    // relaxation time, which makes this MEDIUM at its fastest.
    //
    // Stored as the loss rather than as `decay = 1 − loss` for the reason spelled
    // out on `zoomRate`. On the loss rate a multiplicative spec is exactly right —
    // doubling the loss halves the memory — and `half` 0.25 reads as memory ×0.78
    // to ×1.28.
    //
    // This is also the slot that sets how an order change *reads*: the old rosette
    // survives 1/fade steps into the new one, so a fade of 0.07 (14 steps) makes an
    // order change a cut and 0.003 (330 steps) makes it a five-second dissolve.
    cls: CLASS_MEDIUM,
    min: 0.0005,
    max: 0.5,
    step: 0.0005,
    def: 0.01,
    mod: mul('decay', 0.003, 0.07, 0.25, 0.2),
  },
  {
    key: 'blurMix',
    label: 'blur mix (softens the echo)',
    folder: 'feedback · memory',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    // The blur is sampled at the SOURCE point, which every one of the 2n mirror
    // partners shares exactly — so it softens the figure without ever breaking its
    // symmetry, even though the kernel itself is axis-aligned in texel space. That
    // is worth knowing before anyone tries to make it anisotropic.
    def: 0.3,
    mod: add('decay', 0, 0.8, 0.2, 0.2),
  },
  {
    key: 'blurRadius',
    label: 'blur radius (texels)',
    folder: 'feedback · memory',
    // SLOW: measured in TEXELS, so it is the one slot whose meaning depends on the
    // window size (the shader rescales against a 1080-line reference, which fixes
    // the scale but not the granularity), and a fast lane would make the figure's
    // grain change identity with the music.
    cls: CLASS_SLOW,
    min: 0.1,
    max: 8,
    step: 0.05,
    def: 1.6,
    mod: mul('decay', 0.6, 4, 0.3, 0.45),
  },

  // ── colour · chroma ────────────────────────────────────────────────────────
  {
    key: 'chromaShift',
    label: 'hue rotation / step (rad)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    min: -0.15,
    max: 0.15,
    step: 0.0005,
    // Slightly above nebula's 0.004. Applied to the feedback, so it accumulates
    // along the radial arm: a petal is its stem's own hue at the ring and has
    // walked ~0.5 rad round the hue circle by the time the flow has carried it to
    // the rim. That base-to-tip gradient is most of what makes the arms read as
    // one continuous medium rather than as four stacked colour layers, and it
    // matters more here than in nebula because the fold puts every layer's arms on
    // top of every other layer's 2n times.
    def: 0.005,
    // The `matrix` group, and it earns it: `matrix` is "how the voices relate",
    // and this is what turns four independent palette entries into one gradient
    // they all travel along.
    mod: add('matrix', -0.03, 0.03, 0.012, 0.016),
    macro: 'chroma',
  },
  {
    key: 'layerBlend',
    label: 'layer blend  (1 = additive · higher = loudest hue wins)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    // Floored at 1 rather than 0, and the floor is the whole design: 1 is
    // *provably* the plain additive sum (see the derivation in draw.wgsl), so the
    // bottom of this slider is the physical answer rather than a degenerate one.
    min: 1,
    max: 8,
    step: 0.05,
    // Higher than nebula's 3, and the fold is the reason. Additive light makes an
    // orange arm crossing a cyan one literally white in linear rgb —
    // (1.03, 0.88, 1.01) off the shipped palette, before exposure or the tone map,
    // so nothing downstream can recover the hue. In nebula an overlap is an
    // occasional crossing. Here the fold maps every layer's arms on top of each
    // other 2n times and overlap is the normal case, so the exponent has to be
    // strong enough that the *typical* pixel still belongs to a stem. At 3.5 a
    // layer twice its neighbour's weight carries ~11× the hue. Much past 4 and a
    // faint layer stops tinting a bright one at all, and the four stems cease to
    // look like one medium.
    def: 3.5,
    mod: mul('matrix', 1, 6, 0.3, 0.3),
    macro: 'chroma',
  },

  // ── emitters · light ───────────────────────────────────────────────────────
  {
    key: 'haloMix',
    label: 'halo ×  (wide lobe vs petal core)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    // Lower than nebula's 0.42 for the coverage reason in LAYER_CHARACTER: 144
    // exponential tails add up where 16 do not, and a halo that is an atmosphere
    // around a nebula's arms is a fog over a mandala's.
    def: 0.3,
    mod: mul('population', 0.05, 3, 0.4, 0.4),
    macro: 'light',
  },
  {
    key: 'lace',
    label: 'lace (noise break-up, folded)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    // Evaluated at the FOLDED coordinate, which is the whole trick: noise sampled
    // in screen space would be the one term in either pass that is not symmetric,
    // and it would dissolve the figure it is meant to texture. Folded, the noise is
    // itself a mirror pattern, so breaking the petals into strands produces lace
    // rather than grain — the fibrous detail a real kaleidoscope gets from the
    // chips in the tube.
    def: 0.45,
    mod: add('structure', 0, 0.9, 0.3, 0.3),
  },
  {
    key: 'laceScale',
    label: 'lace scale',
    folder: 'emitters · light',
    cls: CLASS_SLOW,
    min: 0.5,
    max: 40,
    step: 0.1,
    // 9 cycles across the screen height, i.e. a noise cell about 0.11 high — five
    // to twelve petal radii, so the term varies *across* a petal and breaks it into
    // strands instead of merely dimming it. Above about 25 the cells are smaller
    // than a petal and the result is grain that the bloom then turns into haze.
    def: 9,
    mod: mul('structure', 3, 22, 0.3, 0.55),
    // No macro: see `rippleFreq`. A bigger scale number makes the lace finer.
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
    // Slightly under nebula's 1.4 because the ring is drawn in the folded domain
    // and therefore lands 2n times: an event's *total* light is an order of
    // magnitude larger here even at equal peak. It is still the one thing on screen
    // allowed to reach white — see the note on the ring loop in draw.wgsl.
    def: 1.2,
    mod: mul('population', 0.2, 4, 0.5, 0.35),
    macro: 'light',
  },
  {
    key: 'ringExpand',
    label: 'shockwave expansion ×',
    folder: 'events · shockwaves',
    cls: CLASS_MEDIUM,
    min: 0.05,
    max: 6,
    step: 0.01,
    // 1.0 rather than nebula's 1.1: a folded ring is clipped to the wedge and
    // replicated, so a large one becomes a rosette of arcs whose own order competes
    // with the figure's. Kept just under so the flower stays inside one ring of
    // petals for most of its envelope.
    def: 1,
    mod: mul('structure', 0.4, 3, 0.4, 0.4),
    macro: 'motion',
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
    // The same 0.1 as nebula and plife, and it is the same 0.1 on purpose: the
    // field's absolute scale is set by `DRAW_SCALE` in draw.wgsl, which is
    // calibrated *against* nebula's so that this slider means the same thing in
    // both visuals. A scene exposure that differed here would make the shared HDR
    // folder lie about what it is showing when the visual changes.
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
 * Shipped hues. The same four every substrate in this project uses for the same
 * four stems — bass is orange in all of them — because that cross-substrate
 * identity is the one piece of the palette worth not re-deciding.
 */
const PALETTE_HEX: readonly string[] = [
  '#ff7a1a', // bass
  '#ff2f6d', // drums
  '#35d6ff', // vocals
  '#a56bff', // other
];

export const KALEIDO: VizFxVisual = {
  id: 'kaleido',
  title: 'terrarium · kaleido',
  // K = 4, and the four species ARE the four stems, in the analysis contract's
  // own order (`STEM_NAMES` in timeline/types.ts). `stemMap()` is the identity,
  // the palette is per stem, stem-follow drives each layer's light and the energy
  // lane drives each ring's radius and petal size. "The vocal entered" is then not
  // something the mapping has to be lucky enough to express — it is a named ring
  // of the figure lighting up and growing.
  speciesCount: 4,
  layerNames: ['bass', 'drums', 'vocals', 'other'],
  paletteHex: PALETTE_HEX,
  // Three per layer, twelve in all — and 144 on screen at the shipped order, since
  // the fold replicates each one 2n times for free. Two was measured as too few in
  // revision 1, which folded the chassis's emitter phase by 2π/n and so collapsed
  // the even spacing the chassis had gone to trouble to create: a layer came out as
  // one spoke on roughly a third of seeds. Since revision 2 the phase is read as a
  // plain fraction of a turn and the spacing survives, so three is now comfortable
  // rather than a minimum — it fills a lane without crowding it. The ceiling is 6,
  // and past about four a lane is solid and the clusters stop being separable.
  emittersPerLayer: 3,
  layerSlots: LAYER_SLOTS,
  globalSlots: GLOBAL_SLOTS,
  macros: [
    { key: 'light', label: 'light  (× petal gain, halo, shockwaves)', min: 0, max: 2 },
    // `motion` scales only the two slots that are genuinely about liveliness. The
    // two radial flow rates (`zoomRate`, `hubDrain`) are deliberately outside it:
    // both trade against where the figure sits in the frame rather than against how
    // alive it is, so folding them in would mean "more motion" always also pushed
    // the mandala off its own centre.
    { key: 'motion', label: 'motion  (× lane sweep + shockwave speed)', min: 0, max: 2 },
    { key: 'scale', label: 'scale  (× ring radius + petal size)', min: 0, max: 2 },
    { key: 'chroma', label: 'chroma  (× hue rotation + layer blend)', min: 0, max: 2 },
  ],
  warpWgsl,
  drawWgsl,
};
