/**
 * **nebula** — the first vizfx visual.
 *
 * The whole of it is this file plus two WGSL passes. There is no nebula class:
 * `VizFxSim` owns the ping-pong rig, the ModTarget plumbing and the buffers, and
 * reads the table below to find out what it is drawing. A second visual is
 * expected to be exactly this shape.
 *
 * ## What it is
 *
 * Four layers, one per stem, each with a small constellation of emitters
 * orbiting the frame centre at its own radius and speed. Every step, the warp
 * pass resamples the accumulated field through a zoom + rotation + differential
 * swirl and dims it; the draw pass stamps the emitters on top. Light therefore
 * leaves an emitter, spirals outward over a second or two, shears against the
 * neighbouring layers' counter-rotation, drifts around the hue circle, and
 * fades. Twelve gaussians in, a nebula out.
 *
 * ## Why the character lives in per-layer defaults
 *
 * The four layers differ in the table below and the differences are the art
 * direction, not tuning:
 *
 * - **bass** orbits wide and slow with the largest cores. Low end reads as mass
 *   and mass reads as a slow, big thing near the outside of the frame.
 * - **drums** orbits tight and fast with small cores. A hit has to resolve as a
 *   distinct flash near the middle, not as a swell — and being tight to the
 *   centre puts it exactly where the swirl term is strongest, so every hit is
 *   immediately sheared into the flow.
 * - **vocals** sits between the two and is the layer with the most headroom in
 *   `emitGain`: it is the voice most likely to arrive and leave mid-track, and
 *   the stem-follow and energy lanes both key off it hardest.
 * - **other** orbits widest and slowest of all, which puts pads and texture at
 *   the outer edge of the frame where they become the background the rest is
 *   drawn against.
 *
 * The seeded personality (`ModSpec.jitter`) then displaces all of it, and the
 * seeded emitter placement (`VizFxSim.redrawEmitters`) rearranges the
 * constellation — so a reroll is a different creature, not a different mood.
 *
 * ## The θ table: 40 slots, 34 of them modulated
 *
 * 5 per layer x 4, plus 20 globals. Every `lo`/`hi` is narrower than the hard
 * `min`/`max` beside it, which is the split both older registries use: the hard
 * bound is what a file may contain and what a hand slider may reach, the
 * ModSpec range is where the music may wander unsupervised.
 *
 * Excursion sizing follows physarum's rule: with a z-scored input and a unit
 * projection, `w·ẑ` is roughly N(0,1), so the typical |tanh| at depth 1 is about
 * 0.55 and every `half` below is therefore about twice the excursion you should
 * expect to see.
 */
import { CLASS_FAST, CLASS_MEDIUM, CLASS_SLOW, type ModGroup, type ModSpec } from '../../../mapping/modspec.ts';
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
 * The cores are **small** — 2% to 5% of screen height — and that is the single
 * most important number in this file. The first tuning pass shipped them three
 * times larger, on the reasoning that a bass emitter should be a big soft thing,
 * and the result was four coloured blobs with halos: a core wide enough to read
 * as an object on its own does not need the feedback to make it visible, so the
 * feedback stops being what you are looking at. A core small enough to be almost
 * a point is one the warp has to smear into an arm before there is an image at
 * all, which is the entire premise of the substrate.
 */
const LAYER_CHARACTER: readonly {
  emitGain: number;
  orbitRadius: number;
  orbitSpeed: number;
  coreSize: number;
}[] = [
  { emitGain: 1.0, orbitRadius: 0.26, orbitSpeed: 0.09, coreSize: 0.035 }, // bass
  { emitGain: 1.35, orbitRadius: 0.14, orbitSpeed: 0.34, coreSize: 0.014 }, // drums
  { emitGain: 1.15, orbitRadius: 0.22, orbitSpeed: 0.19, coreSize: 0.02 }, // vocals
  { emitGain: 0.9, orbitRadius: 0.34, orbitSpeed: 0.12, coreSize: 0.028 }, // other
];

const character = (key: keyof (typeof LAYER_CHARACTER)[number], fallback: number) =>
  (layer: number): number => LAYER_CHARACTER[layer % LAYER_CHARACTER.length]?.[key] ?? fallback;

const LAYER_SLOTS: readonly VizSlot[] = [
  // Excluded from modulation, the same call physarum and plife both made and for
  // the same reason: light is driven by the stem-follow lane, and modulating the
  // base it scales puts the constant unexplained flashing straight back.
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
    key: 'orbitRadius',
    label: 'orbit radius (screen heights)',
    // SLOW: where a layer lives in the frame is a compositional fact, and a
    // composition that changes every bar is not a composition. It is also the
    // slot whose motion the field itself takes longest to catch up with — an
    // arm already drawn at the old radius takes 1/(1-decay) steps to fade.
    cls: CLASS_SLOW,
    min: 0,
    max: 0.9,
    step: 0.005,
    def: character('orbitRadius', 0.25),
    mod: mul('structure', 0.05, 0.55, 0.3, 0.45),
    macro: 'scale',
  },
  {
    key: 'orbitSpeed',
    label: 'orbit speed (rad/s)',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 4,
    step: 0.005,
    def: character('orbitSpeed', 0.2),
    mod: mul('structure', 0.02, 1.2, 0.45, 0.5),
    macro: 'motion',
  },
  {
    key: 'coreSize',
    label: 'core size (screen heights)',
    cls: CLASS_MEDIUM,
    min: 0.002,
    max: 0.4,
    step: 0.002,
    def: character('coreSize', 0.07),
    mod: mul('structure', 0.01, 0.18, 0.3, 0.4),
    macro: 'scale',
  },
];

/**
 * ## The one relationship that decides whether this looks like anything
 *
 * An arm is the trail a core leaves while the field still remembers it. So arm
 * length is not set by `rotate` or by `fade` — it is set by their *ratio*:
 *
 *     arc ≈ (rotate + swirl) × memory,   memory ≈ 1 / fade  steps
 *
 * The first tuning pass had a fade of 0.028 (memory 36 steps) and rotate 0.0035,
 * for an arc of 0.13 rad — 7 degrees. Seven degrees of arc is not an arm, it is a
 * slightly oval blob, and that is exactly what it looked like. The defaults below
 * put memory at 100 steps (1.7 s) and the combined angular rate near 0.09
 * rad/step, for several radians of visible trail.
 *
 * `zoomRate` came *down* at the same time and for the same arithmetic: over a
 * 100-step memory, 0.012 per step is a 3.3× expansion, so the trail would leave
 * the frame before it faded and the image would be a permanent explosion. 0.0035
 * is 1.4× over the same window, which reads as an outward current under the
 * rotation rather than instead of it.
 *
 * Anyone retuning this should move `decay` and the two angular slots together,
 * or they will find that raising one undoes the other.
 *
 * ## The constraint that bounds the angular slots, including their mod ranges
 *
 * An arm is a row of stamps, one per step. For it to read as an arm rather than
 * as a string of beads, consecutive stamps have to *overlap*:
 *
 *     (rotate + swirl·e^(-r²·falloff)) × r   <   coreSize
 *
 * at every layer's own radius. This was learned the hard way: with `rotate` 0.024
 * and `swirl` 0.13 the combined rate at r = 0.45 is 0.095 rad/step, i.e. 0.043
 * screen heights of displacement against a 0.035 core — and the frame came out as
 * neat rings of separate dots, which looks like a screensaver from 1997 rather
 * than like gas. Note that `swirl` is the dominant term at every radius the
 * emitters actually occupy, so it is the one to suspect.
 *
 * That constraint is now *enforced* rather than merely respected: `draw.wgsl`
 * floors every stamp's size at 1.2× the displacement it is about to undergo, so
 * an emitter modulated out to the rim grows a proportionately fatter core
 * instead of dissolving into dots. The values below can therefore be chosen for
 * how the flow looks rather than for what the smallest core can survive — but
 * the relationship is still the one to think in when moving them.
 */
const GLOBAL_SLOTS: readonly VizSlot[] = [
  // ── warp · flow ────────────────────────────────────────────────────────────
  {
    key: 'zoomRate',
    label: 'zoom rate / step  (>0 = expand outward)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    // Negative is legal and is a *different visual*: the flow runs inward and the
    // frame becomes a drain rather than a fountain. Bounded much tighter on that
    // side because an inward flow concentrates light at the centre instead of
    // spreading it, so it saturates far faster for the same magnitude.
    min: -0.01,
    max: 0.04,
    step: 0.0002,
    // ~1.4× expansion over the field's memory. Above about this the arms reach
    // the frame edge while still near full brightness and what you get is a zoom
    // tunnel — structure streaming off all four sides — rather than an object
    // sitting in space.
    def: 0.0035,
    // ## Why this is a rate rather than the zoom factor itself
    //
    // It shipped as a multiplicative slot holding the factor (1.0035) and that
    // was a bug, found by reading the live θ off a seed whose image had gone to
    // pieces: the value was 1.0140. Both `jitter` and `half` act in ln space on
    // *the stored number*, so a ±0.8% draw on 1.0035 moves it to 1.012 — which is
    // a 3.3× change in the growth rate, because the growth rate is `x - 1` and
    // almost all of the stored number is the 1. Every seed was therefore drawing
    // a wildly different flow speed from a spec that read as "±0.8%", and roughly
    // a third of them expanded fast enough to throw the arms off-frame before
    // they could form.
    //
    // Stored as the rate, an additive spec means exactly what it says. This is
    // the general trap for any per-step compounding factor in a θ table, and
    // `fade` below is the same fix for the same reason.
    //
    // No macro, deliberately: `motion` multiplying a *rate* would be defensible,
    // but zoom is the one flow term that trades against frame-filling rather than
    // against liveliness, and folding it into the same knob as rotation would
    // mean pushing "motion" always also pushed the image off the edges.
    mod: add('structure', -0.004, 0.012, 0.0025, 0.0025),
  },
  {
    key: 'rotate',
    label: 'rotation / step (rad)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: -0.2,
    max: 0.2,
    step: 0.0005,
    def: 0.016,
    mod: add('structure', -0.04, 0.04, 0.012, 0.006),
    macro: 'motion',
  },
  {
    key: 'swirl',
    label: 'swirl (differential twist)',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: -0.6,
    max: 0.6,
    step: 0.002,
    // Larger than `rotate` because it is scaled down by `exp(-r²·falloff)`
    // everywhere the emitters actually are: at the bass layer's radius the
    // falloff is already taking two thirds of it. The number that matters is the
    // product, and this is what makes it comparable.
    def: 0.09,
    // `jitter` well under `half`: an additive draw of ±0.06 on a 0.09 base can
    // land near zero or flip the sign, and a seed whose swirl came out at 0.02 is
    // a seed with no arms at all — measured, on the seed that prompted the
    // `zoomRate` fix. The music may still swing it that far; the *shipped
    // personality* should not start there.
    mod: add('structure', -0.2, 0.2, 0.06, 0.03),
    macro: 'motion',
  },
  {
    key: 'swirlFalloff',
    label: 'swirl falloff (higher = tighter core)',
    folder: 'warp · flow',
    cls: CLASS_SLOW,
    min: 0.1,
    max: 16,
    step: 0.05,
    def: 3,
    mod: mul('structure', 0.6, 6, 0.3, 0.5),
  },
  {
    key: 'driftX',
    label: 'drift x / step',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: -0.02,
    max: 0.02,
    step: 0.0002,
    def: 0,
    mod: add('structure', -0.004, 0.004, 0.0018, 0.002),
    macro: 'motion',
  },
  {
    key: 'driftY',
    label: 'drift y / step',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: -0.02,
    max: 0.02,
    step: 0.0002,
    def: 0,
    mod: add('structure', -0.004, 0.004, 0.0018, 0.002),
    macro: 'motion',
  },
  {
    key: 'ripple',
    label: 'radial ripple depth',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.05,
    step: 0.0005,
    def: 0.0025,
    mod: add('structure', 0, 0.012, 0.005, 0.004),
    macro: 'motion',
  },
  {
    key: 'pulseZoom',
    label: 'impulse → zoom kick',
    folder: 'warp · flow',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.2,
    step: 0.001,
    // Small, because it is added to a per-step rate that compounds: at 0.008 and
    // a capped pulse of 2 a kick expands the field by ~1.3× over its envelope,
    // which is a visible shove. 0.02 was three times that and read as the camera
    // being yanked.
    def: 0.008,
    mod: add('structure', 0, 0.03, 0.008, 0.008),
  },

  // ── feedback · memory ──────────────────────────────────────────────────────
  {
    key: 'fade',
    label: 'fade / step  (memory = 1 / this)',
    folder: 'feedback · memory',
    // The substrate's relaxation time, stored as the LOSS rate: memory is
    // `1/fade` steps, so 0.01 is 100 steps or 1.7 s. By the rule both other
    // registries follow, nothing may be modulated faster than the substrate's own
    // relaxation time, which makes this MEDIUM — and it is the only slot in the
    // `decay` group's namesake role.
    //
    // Stored as the loss rather than as `decay = 1 - loss` for the reason spelled
    // out on `zoomRate`: a ln-space spec on 0.99 is a spec on the wrong quantity,
    // and a ±1.5% draw there is a ±2.5× draw on the memory. On the loss rate a
    // multiplicative spec is exactly right — doubling the loss halves the memory,
    // which is what "×2" ought to mean — and `half` 0.25 reads as memory ×0.78 to
    // ×1.28.
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
    def: 0.32,
    mod: add('decay', 0, 0.8, 0.2, 0.2),
  },
  {
    key: 'blurRadius',
    label: 'blur radius (texels)',
    folder: 'feedback · memory',
    // SLOW: this is measured in TEXELS, so it is the one slot whose meaning
    // depends on the window size, and a fast lane on it would make the image's
    // grain change identity with the music.
    cls: CLASS_SLOW,
    min: 0.1,
    max: 8,
    step: 0.05,
    def: 1.8,
    mod: mul('decay', 0.6, 4, 0.3, 0.45),
  },

  // ── colour ─────────────────────────────────────────────────────────────────
  {
    key: 'chromaShift',
    label: 'hue rotation / step (rad)',
    folder: 'colour · chroma',
    cls: CLASS_MEDIUM,
    min: -0.15,
    max: 0.15,
    step: 0.0005,
    def: 0.004,
    // The only slot in the `matrix` group, and it earns the group rather than
    // being filed there for want of somewhere else: `matrix` is "how the voices
    // relate", and this is what turns four independent palette entries into one
    // continuous gradient they all travel along.
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
    // Below 1 the exponent would start favouring the *quietest* layer, which is
    // neither physical nor useful; the shader clamps it too.
    min: 1,
    max: 8,
    step: 0.05,
    // Additive light makes an orange arm crossing a cyan one literally white in
    // linear rgb — (1.03, 0.88, 1.01), measured off the shipped palette, before
    // exposure or the tone map are involved. At 3 a layer twice its neighbour's
    // weight carries eight times the hue, which keeps the crossing readable as
    // "the orange one, brighter" instead of as a white blob. Higher than about 4
    // and a faint layer stops tinting a bright one at all, so the four stems
    // cease to look like one medium.
    def: 3,
    // The second slot in the `matrix` group, and it belongs there for the same
    // reason `chromaShift` does: `matrix` is "how the voices relate", and this is
    // literally the rule for what happens when two of them are in the same place.
    mod: mul('matrix', 1, 6, 0.3, 0.3),
    macro: 'chroma',
  },

  // ── emitters · light ───────────────────────────────────────────────────────
  {
    key: 'halo',
    label: 'halo ×  (wide lobe vs core)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    def: 0.42,
    mod: mul('population', 0.05, 3, 0.4, 0.4),
    macro: 'light',
  },
  {
    key: 'filament',
    label: 'filament (noise break-up)',
    folder: 'emitters · light',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 1,
    step: 0.005,
    def: 0.55,
    mod: add('structure', 0, 0.9, 0.3, 0.3),
  },
  {
    key: 'noiseScale',
    label: 'filament scale',
    folder: 'emitters · light',
    cls: CLASS_SLOW,
    min: 0.5,
    max: 40,
    step: 0.1,
    // 7 cycles across the screen height, i.e. a noise cell about 0.14 high —
    // three to six times a core's radius, so the term varies *across* an emitter
    // and breaks it into strands. At the 2.4 the first pass shipped, one noise
    // cell was wider than the whole constellation and the filament slider did
    // nothing except dim the frame slightly.
    def: 7,
    // No macro: `scale` makes things bigger, and a bigger *noise scale* number
    // makes the filaments finer. Wiring it would give one macro two opposite
    // meanings depending on which slider you were watching.
    mod: mul('structure', 2, 18, 0.3, 0.55),
  },

  // ── events ─────────────────────────────────────────────────────────────────
  {
    key: 'ringGain',
    label: 'shockwave brightness ×',
    folder: 'events · shockwaves',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    step: 0.01,
    def: 1.4,
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
    // The field is an accumulation with a memory of ~100 steps, so its raw
    // magnitude is a couple of orders above a per-frame splat's — at the shipped
    // decay of 0.99 a stationary core reaches a hundred times its own per-step
    // contribution. This only has to put the frame in the right decade —
    // auto-exposure adapts on top — and 0.1 parks the adapted gain near 1× on a
    // playing track (and around 6× on a silent one, which is inside the rail),
    // the same reasoning that put physarum at 0.01 and plife at 0.1. The field's
    // absolute scale is set by `DRAW_SCALE` in draw.wgsl, not here.
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
 * Shipped hues. The same four the other two substrates use for the same four
 * stems — bass is orange in all three — because that cross-substrate identity is
 * the one piece of the palette worth not re-deciding.
 */
const PALETTE_HEX: readonly string[] = [
  '#ff7a1a', // bass
  '#ff2f6d', // drums
  '#35d6ff', // vocals
  '#a56bff', // other
];

export const NEBULA: VizFxVisual = {
  id: 'nebula',
  title: 'terrarium · nebula',
  // K = 4, and the four species ARE the four stems, in the analysis contract's
  // own order (`STEM_NAMES` in timeline/types.ts). That is the whole keying:
  // `stemMap()` is the identity, the palette is per stem, and stem-follow drives
  // each layer's light. "The vocal entered" is then not something the mapping has
  // to be lucky enough to express — it is structural.
  speciesCount: 4,
  layerNames: ['bass', 'drums', 'vocals', 'other'],
  paletteHex: PALETTE_HEX,
  // Four per layer, sixteen in all. Three is the smallest count that reads as a
  // *constellation* rather than as a barbell — arms that interleave rather than
  // arms that mirror — and four was measured as the point where the frame stops
  // being a handful of separable objects and starts being a texture. Past that
  // the arms overlap into a single mass and the per-stem identity goes with it,
  // which is why the ceiling is 6 and not 20.
  emittersPerLayer: 4,
  layerSlots: LAYER_SLOTS,
  globalSlots: GLOBAL_SLOTS,
  macros: [
    { key: 'light', label: 'light  (× core, halo, shockwaves)', min: 0, max: 2 },
    { key: 'motion', label: 'motion  (× rotate, swirl, drift, orbit, ripple, ring speed)', min: 0, max: 2 },
    { key: 'scale', label: 'scale  (× orbit radius + core size)', min: 0, max: 2 },
    { key: 'chroma', label: 'chroma  (× hue rotation + layer blend)', min: 0, max: 2 },
  ],
  warpWgsl,
  drawWgsl,
};
