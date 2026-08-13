/**
 * **plasma** — an interference / flow visual on the vizfx chassis.
 *
 * Same shape as `nebula/nebula.ts`: this file plus two WGSL passes, and no
 * class. `VizFxSim` owns the ping-pong rig, the ModTarget plumbing and the
 * buffers, and reads the table below to find out what it is drawing.
 *
 * ## What it is, and how it differs from nebula
 *
 * Nebula's draw pass is twelve to sixteen *point* sources — small gaussians —
 * and the picture is what the warp makes of them over a hundred steps. Plasma's
 * draw pass paints no points at all. Each layer evaluates a **continuous
 * interference field** over the whole frame: a handful of radial travelling
 * waves emitted from slowly-orbiting seeded source points, plus one planar
 * travelling wave along the layer's own axis, summed, rectified and raised to a
 * sharpening exponent. What lands in the field every step is therefore a set of
 * smooth bright fringes with dark lanes between them, covering the entire frame
 * rather than 0.65% of it.
 *
 * The warp then advects that whole field along a slow, large-scale,
 * **divergence-free** flow (see the stream-function derivation in warp.wgsl),
 * so the fringes are dragged one to two fringe-widths over the field's own
 * memory and fold into themselves. That is the "flow" half of the name and it
 * is the reason this is not nebula with a different palette: nebula's structure
 * is *made* by the warp out of nothing, plasma's structure is *deformed* by the
 * warp having already been drawn.
 *
 * A viewer sees: four interleaved bands of coloured light, each stem at its own
 * spatial frequency and travelling along its own axis, everywhere at once,
 * continuously shearing and folding. A hit does not light a spot — it shifts
 * the phase of every fringe in the frame simultaneously and spikes the flow, so
 * the whole image lurches. A splash rings the medium locally: an expanding
 * front that the fringes visibly bend around.
 *
 * ## Why the character lives in per-layer defaults
 *
 * A stem entering has to be an obvious change to the frame, and here it cannot
 * be "a new object appears" because there are no objects. So it is three
 * simultaneous, structural changes:
 *
 * - **a new hue** — the palette, shared with the other two substrates.
 * - **a new spatial frequency** — `fringe` is the layer's band spacing, and the
 *   four ship an octave and a half apart (0.42 / 0.13 / 0.24 / 0.32 screen
 *   heights). Bass is broad slabs; drums is fine corduroy. Even at equal
 *   brightness the two are not confusable.
 * - **a new direction of travel** — the planar wave's axis is `l·2π/K + 0.45`,
 *   structural rather than θ for the reason nebula's orbit-sense alternation is:
 *   a slot whose only useful values are "four, evenly spread" is not a slider.
 *
 * Then `srcRadius` puts each layer's wave sources at its own distance from the
 * centre (drums tight at 0.12, `other` out at 0.42, so pads live at the rim and
 * become the background), and `driftSpeed` gives each its own current rate.
 *
 * ## The θ table: 39 slots, 33 of them modulated
 *
 * 5 per layer × 4, plus 19 globals. Every `lo`/`hi` is strictly inside the hard
 * `min`/`max` beside it and strictly narrower — the hard bound is what a file
 * may contain and how far a hand slider travels, the ModSpec range is where the
 * music may wander unsupervised.
 *
 * Excursion sizing follows physarum's rule and nebula's restatement of it: with
 * a z-scored input and a unit projection `w·ẑ` is roughly N(0,1), the typical
 * |tanh| at depth 1 is about 0.55, and every `half` below is therefore about
 * twice the excursion you should expect to see.
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
 * The single most important column is `fringe`, and it is important for the
 * opposite reason `coreSize` is important in nebula. There, the cores had to be
 * *small* so that the warp had something to do. Here the fringes have to be
 * **far apart relative to one step's flow displacement**, or the field never
 * accumulates coherently: the pattern painted at step n and the pattern painted
 * at step n+1 have to land on top of each other well enough to reinforce, and
 * they only do that while the per-step advection (0.008 screen heights at the
 * shipped `flow`) is a small fraction of a fringe half-period. The finest layer
 * here, drums at 0.13, has a half-period of 0.065 — eight times the per-step
 * displacement. The first pass tried drums at 0.045 and the result was not fine
 * corduroy, it was grey: consecutive steps landed a bright band where the
 * previous step had put a dark lane, and the accumulation averaged to a flat
 * wash with no visible structure at all.
 *
 * That relationship is the analogue of nebula's stamp-continuity constraint and
 * it bounds the *bottom* of `fringe`'s mod range (0.06, i.e. a half-period of
 * 0.03 against a modulated flow ceiling of 0.014) rather than the top.
 */
const LAYER_CHARACTER: readonly {
  injGain: number;
  fringe: number;
  srcRadius: number;
  driftSpeed: number;
}[] = [
  { injGain: 1.0, fringe: 0.42, srcRadius: 0.34, driftSpeed: 0.12 }, // bass
  { injGain: 1.25, fringe: 0.13, srcRadius: 0.12, driftSpeed: 0.55 }, // drums
  { injGain: 1.15, fringe: 0.24, srcRadius: 0.22, driftSpeed: 0.28 }, // vocals
  { injGain: 0.9, fringe: 0.32, srcRadius: 0.42, driftSpeed: 0.09 }, // other
];

const character = (key: keyof (typeof LAYER_CHARACTER)[number], fallback: number) =>
  (layer: number): number => LAYER_CHARACTER[layer % LAYER_CHARACTER.length]?.[key] ?? fallback;

const LAYER_SLOTS: readonly VizSlot[] = [
  // Excluded from modulation, the same call all three substrates make and for
  // the same reason: light is driven by the stem-follow lane, and modulating the
  // base it scales puts the constant unexplained flashing straight back. The
  // chassis reads this one by name (`params['layer<i>.brightness']` in
  // `uploadLayers` / `syncSpecies`), so the key is part of the contract.
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
    key: 'injGain',
    label: 'field gain ×',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    def: character('injGain', 1),
    // Multiplicative: a gain is a ratio, and ±0.5 in ln space is ×0.6…×1.65,
    // which reads the same on the quiet layer and the loud one.
    mod: mul('population', 0.15, 4, 0.5, 0.35),
    macro: 'light',
  },
  {
    key: 'fringe',
    label: 'fringe spacing (screen heights)',
    // SLOW, for two reasons that both point the same way. A layer's spatial
    // frequency *is* its identity here — the thing that lets you tell bass from
    // drums with the colour turned off — and an identity that changes every bar
    // is not an identity. And the field carries the old spacing for `1/fade`
    // steps, so a fast sweep would only ever paint one frequency on top of
    // another and read as grain.
    cls: CLASS_SLOW,
    // Stored as a *spacing* rather than a frequency, deliberately, and it is
    // what makes the `scale` macro honest: bigger number, bigger features. Held
    // as cycles-per-screen-height it would have been the `noiseScale` problem
    // from nebula, where `scale` would mean "finer" on one slider and "coarser"
    // on every other one. The shader takes the reciprocal (`k = TAU / fringe`),
    // which costs one divide per layer per pixel and buys a knob that composes.
    min: 0.02,
    max: 1.2,
    step: 0.005,
    def: character('fringe', 0.25),
    // The floor of the mod range is set by the coherence argument on
    // LAYER_CHARACTER above, not by taste: at 0.06 the half-period is 0.03,
    // against a modulated `flow` ceiling of 0.014 per step. Below about 0.05 the
    // accumulation stops reinforcing and the layer turns into a flat wash.
    mod: mul('structure', 0.06, 0.8, 0.35, 0.4),
    macro: 'scale',
  },
  {
    key: 'srcRadius',
    label: 'source ring radius (screen heights)',
    // SLOW for nebula's `orbitRadius` reason: where a layer's wave sources sit
    // is a compositional fact. It matters slightly less here — the waves reach
    // the whole frame whatever the sources' radius, so this moves the *centre of
    // curvature* of the fringes rather than moving the light — which is why the
    // range is allowed to be wide.
    cls: CLASS_SLOW,
    min: 0,
    max: 1,
    step: 0.005,
    def: character('srcRadius', 0.26),
    mod: mul('structure', 0.04, 0.6, 0.3, 0.45),
    macro: 'scale',
  },
  {
    key: 'driftSpeed',
    label: 'current speed (rad/s — sources orbit + waves travel)',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 4,
    step: 0.005,
    def: character('driftSpeed', 0.2),
    mod: mul('structure', 0.02, 1.4, 0.45, 0.5),
    macro: 'motion',
  },
];

/**
 * ## The relationship that decides whether this looks like anything
 *
 * Nebula's is `arc ≈ (rotate + swirl) × memory`. Plasma's is the *smear length*:
 *
 *     smear ≈ flow × memory   screen heights,   memory ≈ 1 / fade  steps
 *
 * At the shipped defaults that is 0.008 × 50 = **0.4 screen heights** — the
 * field is dragged nearly half the frame before it fades, which is one to three
 * fringe widths depending on the layer. That is the number to think in.
 *
 * Both ends of it are failures and both were seen:
 *
 * - Smear much *under* one fringe width and nothing moves: every step paints the
 *   same fringes in the same place, the accumulation is a 50× brighter copy of
 *   one instant, and the image is a still photograph of a nice pattern. The
 *   first pass had `flow` at 0.002 and this is exactly what it looked like.
 * - Smear over about two screen heights and the fringes are wiped into streaks
 *   before they can accumulate at all: the field's memory is spent averaging
 *   over a long path, and averaging a sinusoid over more than a period gives
 *   zero. You get a grey wash with a texture-free gradient, which is the same
 *   failure as a fringe spacing that is too fine, arrived at from the other end.
 *
 * `fade` is therefore not free to move alone; halving the memory halves the
 * smear. Anyone retuning this should move `flow` and `fade` together, exactly as
 * nebula's note says to move `decay` and its two angular slots together.
 *
 * ## The other relationship: memory vs deposit
 *
 * This visual paints a *large fraction of the frame* every step where nebula
 * paints well under one percent, so its equilibrium level is set by a much
 * bigger per-step deposit against a deliberately shorter memory. `fade` ships at
 * 0.02 (50 steps, 0.83 s) rather than nebula's 0.01, and `DRAW_SCALE` in
 * draw.wgsl is 0.75 rather than 4.0. The arithmetic behind both is written out
 * on `DRAW_SCALE`; the short version is that the two changes together are a
 * ~11× reduction in equilibrium field level relative to nebula's construction,
 * which is what puts the adapted auto-exposure gain back near 1× at scene
 * exposure 0.1.
 */
const GLOBAL_SLOTS: readonly VizSlot[] = [
  // ── warp · flow ────────────────────────────────────────────────────────────
  {
    key: 'flow',
    label: 'flow displacement / step (screen heights)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    // Held as the displacement magnitude itself, in the same screen-height units
    // every other length in this table uses, which is only possible because the
    // stream function in warp.wgsl is normalised so that |∇⊥ψ̂| ≲ 1. Without that
    // normalisation this number would have had to absorb the flow's spatial
    // frequency, and the slot would have meant a different distance at every
    // setting of `flowSize` — i.e. two knobs fighting over one quantity.
    min: 0.0002,
    max: 0.04,
    step: 0.0002,
    // 0.4 screen heights of smear over the shipped memory; see the note above.
    def: 0.008,
    // The ceiling is not chosen for looks. It is chosen so that the flow's own
    // area Jacobian stays a small correction rather than a large one: the second
    // -order term is `flow² · k² · det(Hess ψ̂)`, so it grows as the *square* of
    // this slot, and at the corner (`flow` 0.014, `flowSize` 0.7, a capped pulse
    // and `pulseShock` at its own ceiling) it reaches 14% per step. Clamped to
    // ≤ 1 it can only ever dim, so the loop is safe at any value — but past this
    // ceiling the clamp stops being a correction and starts being the dominant
    // loss term, and `fade` would no longer be what sets the memory. The full
    // derivation is in warp.wgsl.
    mod: mul('structure', 0.002, 0.014, 0.35, 0.3),
    macro: 'motion',
  },
  {
    key: 'flowSize',
    label: 'flow cell size (screen heights)',
    folder: 'warp · flow',
    // SLOW: this is the scale of the *composition* the fringes are being dragged
    // through, and it is also the slot the Jacobian is most sensitive to (as
    // 1/size²), so a fast lane on it would make the field's brightness breathe
    // with the music for a reason nobody could see.
    cls: CLASS_SLOW,
    // A size, not a wavenumber, for the same reason `fringe` is a spacing: the
    // `scale` macro has to mean one thing across the whole table.
    min: 0.25,
    max: 5,
    step: 0.01,
    // 1.4 screen heights — comfortably larger than the frame's short dimension,
    // so the flow reads as one big slow current with a couple of eddies in it
    // rather than as turbulence. At 0.4 the same amplitude produced a fine
    // shearing texture that ate the fringes instead of carrying them.
    def: 1.4,
    mod: mul('structure', 0.7, 2.6, 0.3, 0.35),
    macro: 'scale',
  },
  {
    key: 'flowSpeed',
    label: 'flow evolution (rad/s)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0.005,
    max: 3,
    step: 0.005,
    // Slow on purpose. This is how fast the three stream-function waves slide
    // past each other, i.e. how fast the *shape* of the current changes as
    // opposed to how fast it carries. At 0.22 rad/s the eddy pattern takes about
    // half a minute to rearrange itself, which is what keeps the plasma folding
    // into itself instead of settling into a steady state — the whole point of
    // making the flow time-dependent at all. Above about 1 rad/s the current
    // changes faster than the field's memory and the smear stops having a
    // direction.
    def: 0.22,
    mod: mul('structure', 0.03, 1.2, 0.4, 0.45),
    macro: 'motion',
  },
  {
    key: 'zoomRate',
    label: 'zoom rate / step  (>0 = expand outward)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    // Negative is legal and is a different visual: the fringes converge on the
    // centre instead of spreading, and the frame becomes a drain. Bounded much
    // tighter on that side because an inward map concentrates light, so it
    // saturates faster for the same magnitude — nebula's reasoning, unchanged.
    min: -0.008,
    max: 0.03,
    step: 0.0002,
    // Deliberately an order gentler than the flow: over the 50-step memory
    // 0.0015 is a 1.08× expansion, i.e. a slow breath under the current rather
    // than a second, competing motion. A radial scale is the one term here that
    // is not divergence-free, so every unit of it costs honest Jacobian dimming
    // (1/zoom² per step) — it is priced, and it should be spent sparingly.
    def: 0.0015,
    // Additive and signed, because this is a per-step compounding *rate* and the
    // stored number is the growth, not `1 + growth`. That is the trap nebula's
    // table records shipping with: a ln-space spec on 1.0035 is a spec on the
    // wrong quantity, since almost all of the stored number is the 1.
    //
    // No macro, for nebula's reason: `motion` on a zoom would mean that pushing
    // "more motion" also pushed the image off the frame edges, which is a
    // different trade from liveliness.
    mod: add('structure', -0.003, 0.009, 0.002, 0.0018),
  },
  {
    key: 'pulseShock',
    label: 'impulse → phase shock ×',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 4,
    step: 0.01,
    // The visual's whole transient idiom, and the one thing here that is a
    // genuinely different answer from nebula's expanding rings. `g.pulse` is 0 at
    // rest and rises with the deposit envelope of whatever just fired; this slot
    // turns it into (a) a phase offset added to *every* fringe in the frame at
    // once and (b) a multiplier on the flow amplitude. Both passes read it.
    //
    // 0.6 × the capped pulse of 2 × the 2.6 rad the shader spends per unit is
    // ~3.1 radians — half a fringe period — applied over the length of the
    // envelope and released. Every band in the image slides half a width and
    // comes back. At 2.0 it is more than a full period and the shift is
    // ambiguous (a whole-period shift is no shift), which reads as a stutter
    // rather than a shove; that is why the mod ceiling is there.
    def: 0.6,
    mod: mul('structure', 0.1, 2, 0.4, 0.35),
    macro: 'motion',
  },

  // ── feedback · memory ──────────────────────────────────────────────────────
  {
    key: 'fade',
    label: 'fade / step  (memory = 1 / this)',
    folder: 'feedback · memory',
    // The substrate's relaxation time, stored as the LOSS rate: memory is
    // `1/fade` steps, so 0.02 is 50 steps or 0.83 s. By the rule every registry
    // in this project follows, nothing may be modulated faster than the
    // substrate's own relaxation time, which makes this MEDIUM.
    //
    // Half nebula's memory, and that is a consequence of the deposit rather than
    // a taste: this pass paints a large fraction of the frame every step where
    // nebula paints a fraction of a percent, so the same memory would put the
    // equilibrium level two orders too high and leave the tone map with a white
    // plate. The other half of the correction is `DRAW_SCALE`.
    //
    // Stored as the loss rather than as `decay = 1 - loss` for the reason spelled
    // out on `zoomRate`: on the loss rate a multiplicative spec is exactly right
    // — doubling the loss halves the memory — and `half` 0.25 reads as memory
    // ×0.78 to ×1.28.
    cls: CLASS_MEDIUM,
    min: 0.001,
    max: 0.5,
    step: 0.0005,
    def: 0.02,
    // The ceiling leaves 17 steps of memory, which at the shipped `flow` is a
    // smear of 0.13 screen heights — one fringe width on the bass layer and
    // still visible motion. The floor leaves 167 steps (2.8 s), where the smear
    // is 1.3 screen heights and the image is on the edge of averaging itself
    // flat; that is the top of the useful band, not an arbitrary number.
    mod: mul('decay', 0.006, 0.06, 0.25, 0.2),
  },
  {
    key: 'blurMix',
    label: 'blur mix (softens the echo)',
    folder: 'feedback · memory',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    // Higher than nebula's 0.32, and load-bearing rather than cosmetic. The blur
    // is a normalised five-tap average, so it conserves total light exactly and
    // costs the loop nothing — what it buys is that the field the warp resamples
    // is band-limited at the step scale, which is what makes the advection
    // continuous instead of stepping. It is the plasma analogue of nebula's
    // stamp-size floor: nebula fattens the stamp to cover the displacement, this
    // low-passes the field so the displacement lands inside the sample footprint.
    def: 0.4,
    mod: add('decay', 0, 0.85, 0.2, 0.2),
  },
  {
    key: 'blurRadius',
    label: 'blur radius (texels)',
    folder: 'feedback · memory',
    // SLOW: measured in TEXELS, so it is the one slot whose meaning depends on
    // the window size, and a fast lane on it would make the image's grain change
    // identity with the music.
    cls: CLASS_SLOW,
    min: 0.1,
    max: 8,
    step: 0.05,
    def: 2.2,
    mod: mul('decay', 0.6, 5, 0.3, 0.45),
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
    // Smaller than nebula's 0.004 because the memory is half as long *and* the
    // fringes are wide: 0.003 over 50 steps is 0.15 rad from the moment light is
    // painted to the moment it is gone, which puts a visible gradient across a
    // band's width without letting the trailing edge of a bass fringe arrive at
    // the drums' hue. Above about 0.01 the four stems' colours converge into one
    // travelling rainbow and the per-stem identity — the thing the layer's
    // spacing and direction are also carrying — stops being reinforced by hue.
    def: 0.003,
    // The `matrix` group, and it earns it: `matrix` is "how the voices relate",
    // and this is what turns four independent palette entries into one continuous
    // gradient they all travel along.
    mod: add('matrix', -0.025, 0.025, 0.01, 0.012),
    macro: 'chroma',
  },
  {
    key: 'layerBlend',
    label: 'layer blend  (1 = additive · higher = loudest hue wins)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    // Floored at 1 rather than 0, and the floor is the design: 1 is *provably*
    // the plain additive sum (see the derivation in draw.wgsl), so the bottom of
    // this slider is the physical answer rather than a degenerate one.
    min: 1,
    max: 8,
    step: 0.05,
    // Higher than nebula's 3, and this is the slot where plasma's geometry
    // forces a different number. Nebula's layers are small blobs that overlap
    // occasionally; plasma's are smooth fields that overlap **everywhere**, so
    // the additive-white failure is not an event that happens at a crossing, it
    // is the default state of the entire frame. Additive: bass orange
    // (1.00, 0.20, 0.01) plus vocals cyan (0.03, 0.68, 1.00) is (1.03, 0.88,
    // 1.01) — white in chromaticity, before exposure or the tone map, so nothing
    // downstream can recover the hue. At 3.2 a layer twice its neighbour's local
    // weight carries ~10× the hue, which is what keeps the bands reading as
    // interleaved colours rather than as one grey interference pattern.
    //
    // The band shaping is the other half of the same fix and the reason 3.2 is
    // enough rather than 6: `bandShape` makes each layer's contribution a set of
    // narrow ridges with dark lanes, so at any given pixel one layer is usually
    // well ahead of the others and the exponent has a real ratio to work on.
    // Flatten the bands and this number has to go up to compensate.
    def: 3.2,
    mod: mul('matrix', 1, 6, 0.3, 0.3),
    macro: 'chroma',
  },

  // ── field · bands ──────────────────────────────────────────────────────────
  {
    key: 'bandShape',
    label: 'band sharpness (exponent — carves the dark lanes)',
    folder: 'field · bands',
    cls: CLASS_MEDIUM,
    min: 0.4,
    max: 8,
    step: 0.05,
    // The single knob that decides whether this is a picture or a wash, and the
    // one the brief for a full-frame field warns about hardest. The rectified
    // interference sum is a smooth thing with a standard deviation of about
    // 0.33 and no hard edges anywhere; raised to 1 it fills the frame with a
    // low-contrast haze that the grade's 1.35 contrast cannot rescue, because
    // there is nothing in it to separate. Raised to 2.4 the same field has a
    // mean of ~0.03 against ridge peaks near 0.4 — a peak-to-mean of ~12 — and
    // the dark lanes between the fringes are genuinely dark, which is what the
    // eye reads as structure.
    //
    // It is also, and not incidentally, the slot that sets the field's average
    // level: `DRAW_SCALE` is calibrated against this default, and the geometric
    // centre of this slot's mod range together with `bandBias`'s is where that
    // calibration is aimed. Moving this default without moving `DRAW_SCALE` is
    // the fastest way to ship a white plate.
    def: 2.4,
    mod: mul('population', 1.2, 5, 0.35, 0.3),
  },
  {
    key: 'bandBias',
    label: 'band bias (− deepens the dark lanes, + floods them)',
    folder: 'field · bands',
    cls: CLASS_MEDIUM,
    min: -0.6,
    max: 0.6,
    step: 0.005,
    // Added to the signed interference value *before* rectification, so it moves
    // the zero crossing: negative narrows every bright fringe and widens the
    // lanes, positive does the reverse and eventually floods the lanes shut.
    // Small and slightly negative by default — the exponent is doing most of the
    // carving and this is the fine adjustment on top of it. Paired with
    // `bandShape` in the `population` group because between them they are "how
    // much of the frame is lit", which is what that group means everywhere else.
    def: -0.05,
    mod: add('population', -0.25, 0.2, 0.09, 0.07),
  },
  {
    key: 'bandGate',
    label: 'presence gate (how far a quiet layer retreats)',
    folder: 'field · bands',
    // SLOW. This is the rule by which the visual answers "how much music is
    // there", i.e. a compositional law rather than a level, and a fast lane on it
    // would make layers flicker in and out of existence on a beat.
    cls: CLASS_SLOW,
    min: 0,
    max: 1.2,
    step: 0.005,
    // ## What this slot is for, and what it turned out NOT to be for
    //
    // It was introduced to answer a measurement that has since been retracted: a
    // quiet passage appearing to fill the whole frame while a chorus emptied it.
    // That comparison used a 30 s window which ended in a busy passage, and — the
    // part that actually mattered — it had no like-for-like reference number. Re-
    // measured on a genuinely quiet window (t = 0 + 10 s, seed 12345) against the
    // reference visual on the same protocol: nebula 5.3% dead frame, plasma 7.2%.
    // **The quiet end was never the defect.** A young sparse field drives the
    // shared auto-exposure controller to ~21–23× and at that gain whatever is
    // present is lifted until it fills the frame; that is the HDR chain working,
    // and it happens to the reference visual just as much.
    //
    // What the slot IS for is the thing that survived: a stem that has genuinely
    // dropped out should stop painting. That is worth having on its own terms —
    // it is the geometric half of "an instrument left" — and the mechanism below
    // is sound. It is simply a legibility feature rather than a dynamics fix, and
    // it is sized accordingly (0.35, not the 0.55 an over-read of the retracted
    // measurement first put here).
    //
    // The mechanism is arithmetic, not taste. Without this slot a layer's
    // presence only scaled its *amplitude*: the energy lane floors at 0.4 and the
    // per-layer weight floored at 0.3 + 0.7·energy = 0.58, so a completely silent
    // stem still painted its full-frame band pattern at 0.58 × the stem-follow
    // lane's own 0.25 floor ≈ 15% of a loud one. Fifteen percent of a *full-frame*
    // pattern is not a trace — three of them, each with its own spacing and
    // direction, land their ridges in each other's dark lanes and fill every one
    // of them. There was no black left for a chorus to grow into.
    //
    // So presence has to buy **coverage**, not intensity. This slot is subtracted
    // from the rectification threshold in proportion to how absent a layer is, so
    // a quiet layer keeps only its strongest constructive ridges and a loud one
    // opens out into wide bands. At 0.55, against the interference sum's σ ≈ 0.33,
    // a fully silent layer's coverage falls from ~56% of the frame to ~10% and its
    // total light by ~8×, while a loud layer is left exactly where it was — which
    // is what keeps `DRAW_SCALE`'s calibration (measured correct at adapted gain
    // 0.91 dense / 3.42 quiet) intact through this change.
    //
    // It is a threshold and not another gain for a reason auto-exposure makes
    // unavoidable: the controller normalises the *mean*, so any change that only
    // scales the field is undone within a second and the picture comes back
    // looking the same. A threshold changes the field's histogram — the shape,
    // not the level — and there is nothing downstream that can undo that.
    // 0.35 rather than the 0.55 this shipped at, turned on a measurement after
    // the presence→coverage rescale landed on top of it. The two compose
    // multiplicatively on the same quantity — the rescale already spends most of
    // the 0.4…1.0 presence range on coverage — and together they took the gate
    // past the point where a layer keeps any ridges at all: measured on free-fall
    // at seed 12345, the quiet intro came out **100% black** (adapted gain pinned
    // at the 32× rail with nothing to expose) and the dense chorus 81% black at
    // mean luminance 8.7. That is not a dark picture, it is an empty one.
    //
    // This is the slot's own named fallback for exactly that symptom, and the
    // reason it is the right one is in the paragraph above: the gate moves the
    // field's *histogram*, so it is the only lever here that auto-exposure cannot
    // undo — which cuts both ways, and is why it overshot rather than being
    // absorbed. The rescale is left in place because the ordering it buys (loud
    // fills more frame than quiet) is the thing that was actually wrong.
    def: 0.35,
    // `population` is exactly right here: the group means "how much of the frame
    // is lit", and this is the rule that ties that to how much music is playing.
    mod: add('population', 0, 0.9, 0.25, 0.2),
  },
  {
    key: 'crest',
    label: 'crest gain ×  (bright spine on each current — feeds bloom)',
    folder: 'field · bands',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 24,
    step: 0.05,
    // ## The slot that stops the chorus rendering half-dead
    //
    // Measured against the reference visual under an identical protocol
    // (free-fall, seed 12345, dense chorus at t = 120 s): nebula 1.8% pure black,
    // plasma **52.8%**. Same grade, same auto-exposure target, same chain.
    //
    // The cause is arithmetic in the shared grade and is written out in full on
    // the crest term in draw.wgsl; the short form is that a pixel goes to true
    // black below a tone-mapped 0.119, auto-exposure pins the frame's *mean* at
    // 0.10, and the only thing that can lift a below-mean pixel over an
    // above-mean cutoff is bloom — which is applied after the measure pass. Bloom
    // needs something over 0.9 to work with. The reference visual has emitter
    // cores and shock fronts that clear it trivially; this field, being smooth
    // everywhere, peaked at ~0.58 and generated none at all.
    //
    // So the field needed an *edge*, not more light. This adds one: a narrow
    // bright spine along the top of each current, squared above a high knee so it
    // lights ~3% of the frame. It roughly triples the injected peak for ~40% on
    // the mean — a net peak-to-mean gain of ~2.5×, taking the accumulated field
    // from a measured 5.8 to ~14 and putting its crests through the bloom
    // threshold.
    //
    // It is the same split nebula makes between its tight gaussian core and its
    // wide exponential halo, arrived at from the opposite deficiency, and the
    // ratio between the two is art direction in both cases: at 0 this is the flat
    // field that measured half-dead, and much above 8 the spines stop being the
    // highlight on a current and become the current, which is a different visual
    // (thin neon filaments) and not this one.
    def: 3,
    // `population` with `bandShape`/`bandBias`/`bandGate`: between them they are
    // "how much of the frame is lit", which is what the group means everywhere.
    mod: mul('population', 0.5, 16, 0.5, 0.4),
    macro: 'light',
  },
  {
    key: 'planarMix',
    label: 'planar current vs radial rings',
    folder: 'field · bands',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    // 0 is pure radial interference: concentric rings from each source, moiréing
    // into the hyperbolic fringes of a classic two-slit pattern. 1 is a single
    // planar travelling wave per layer — straight parallel bands marching along
    // the layer's own axis. Neither extreme is as good as the mixture: the pure
    // radial field is beautiful and *centred*, which after thirty seconds reads
    // as a target rather than as a medium, and the pure planar field has no
    // interference in it at all, which is to say it is not a plasma.
    //
    // 0.45 gives the planar term slightly less total weight than the four radial
    // sources combined, so the frame has an overall direction of travel with the
    // ring structure fighting it. The per-layer axes are 90° apart, so the four
    // directions are what makes a stem's arrival legible from across the room.
    def: 0.45,
    mod: add('structure', 0, 0.9, 0.25, 0.2),
  },

  // ── events · shockwaves ────────────────────────────────────────────────────
  {
    key: 'shockGain',
    label: 'shockwave brightness ×',
    folder: 'events · shockwaves',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    // The splash front's own light, on top of the phase disturbance it applies
    // to the fringes. Both are needed and they do different jobs: the phase kink
    // is what makes the event look like something happening *to the medium*
    // (the bands visibly bow around it) and the light is what makes it visible
    // in a dark passage where there are no bands to bow.
    //
    // ## Why 1.6 and not the 1.2 this shipped with
    //
    // Measured: in a dense passage with 18 live rings over the run, the frame's
    // maximum luminance was 145/255 against a mean of 29.6. Nothing in a loud
    // passage came close to white, so the events had no headroom to be events in
    // — which is the brief's standard inverted, since white is supposed to be
    // *reserved* for exactly these.
    //
    // The cause is that a ring competes against an integrator. The sustained field
    // accumulates its per-step injection over ~50 steps of memory, while a shock
    // front sweeps past a given pixel in about six; so to peak above the field a
    // ring needs roughly an order more amplitude per step, not a comparable
    // amount. At 1.2 with the shader's old 0.6 coefficient the ring's per-step
    // peak was 0.72 against a field ridge of ~0.22 — about 3×, which after the
    // 8× difference in dwell came out *below* the field. It is now ~15× per step,
    // and draw.wgsl also squares the front's envelope so the light is a genuinely
    // sharp front rather than the same width as the phase kink.
    def: 1.6,
    mod: mul('population', 0.15, 4, 0.5, 0.35),
    macro: 'light',
  },
  {
    key: 'shockExpand',
    label: 'shockwave expansion ×',
    folder: 'events · shockwaves',
    cls: CLASS_MEDIUM,
    min: 0.05,
    max: 6,
    step: 0.01,
    def: 1.1,
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
    // Deliberately the same 0.1 nebula and plife use. This slot only has to put
    // the frame in the right decade — auto-exposure adapts on top — and keeping
    // it identical across the repertoire is what lets the shared HDR folder mean
    // one thing. The field's absolute scale is set by `DRAW_SCALE` in draw.wgsl,
    // which is where plasma's much larger per-step deposit is paid for.
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
 * Shipped hues. The same four the other substrates use for the same four stems —
 * bass is orange in all of them — because that cross-substrate identity is the
 * one piece of the palette worth not re-deciding.
 */
const PALETTE_HEX: readonly string[] = [
  '#ff7a1a', // bass
  '#ff2f6d', // drums
  '#35d6ff', // vocals
  '#a56bff', // other
];

export const PLASMA: VizFxVisual = {
  id: 'plasma',
  title: 'terrarium · plasma',
  // K = 4, and the four species ARE the four stems, in the analysis contract's
  // own order (`STEM_NAMES` in timeline/types.ts). `stemMap()` is the identity,
  // the palette is per stem, and stem-follow drives each layer's light — so "the
  // vocal entered" is structural rather than something the mapping has to be
  // lucky enough to express.
  speciesCount: 4,
  layerNames: ['bass', 'drums', 'vocals', 'other'],
  paletteHex: PALETTE_HEX,
  // Wave *sources*, not emitters — the chassis's seeded `(phase, radius×,
  // speed×, size×)` quadruple is reinterpreted here as (starting angle, ring
  // radius ×, orbit and wave speed ×, fringe frequency ×). That last one is what
  // makes a reroll a genuinely different interference pattern rather than the
  // same pattern rotated: four sources at four slightly different wavelengths
  // beat against each other, and the beat envelope is the large-scale structure
  // you actually see.
  //
  // Four is the smallest count that produces a pattern you cannot decompose by
  // eye. Two sources give the textbook two-slit hyperbolae, which read as a
  // diagram; three give a recognisable trefoil. Past about five the beats are
  // dense enough that the field averages toward its own mean and the shaping
  // exponent is carving noise rather than structure — which is why the chassis
  // ceiling of 6 is generous rather than restrictive here.
  emittersPerLayer: 4,
  layerSlots: LAYER_SLOTS,
  globalSlots: GLOBAL_SLOTS,
  macros: [
    { key: 'light', label: 'light  (× field gain, crests, shockwaves)', min: 0, max: 2 },
    {
      key: 'motion',
      label: 'motion  (× flow, flow evolution, currents, phase shock, ring speed)',
      min: 0,
      max: 2,
    },
    { key: 'scale', label: 'scale  (× fringe spacing, source rings, flow cells)', min: 0, max: 2 },
    { key: 'chroma', label: 'chroma  (× hue rotation + layer blend)', min: 0, max: 2 },
  ],
  warpWgsl,
  drawWgsl,
};
