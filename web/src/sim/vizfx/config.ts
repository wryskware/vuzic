/**
 * The chassis's configuration: everything a vizfx visual has that is *not* its
 * θ table.
 *
 * Deliberately thin. θ lives in `params`, keyed by the visual's own slot names
 * (see `slots.ts`), so this file never learns a parameter — which is what keeps
 * "a second visual is one WGSL pair plus one table" true. What is left here is
 * the four blocks every screen-space feedback visual needs and none of which
 * the music should be sweeping: the macro rig, the stems→energy lane, the
 * structural emitter count, and the transport.
 *
 * ## What is outside θ, and why
 *
 * - **macros** — the always-yours performance layer, exactly as in physarum and
 *   plife: plain multipliers applied where the chassis composes its final
 *   numbers (`effectiveTheta`), so modulation, a reroll and a file load can none
 *   of them overwrite them.
 * - **energy** — stems → per-layer emitter *size and presence*. The mapping
 *   layer's own stem-follow lane already drives per-layer *light*
 *   (`setBrightFollow`); this is the geometry half, and it is here rather than
 *   there because it has to keep working with no Modulator at all — which is
 *   exactly the situation an explorer tile is in, and a tile that ignored the
 *   music would be judging a still life.
 * - **emittersPerLayer** — structural. Changing it changes what the visual *is*,
 *   not how loud it is, and it costs a buffer rewrite.
 * - **speed / paused** — transport, same as both other substrates.
 *
 * All four ride the mapping file's opaque `extras` block (see
 * `VizFxSim.serializeExtras`), which is also what carries them to the nine
 * explorer tiles twice a second.
 */
import { customPalette, type Palette } from '../palette.ts';
import { defaultRenderConfig, type RenderConfig } from '../render/config.ts';
import { defaultMacros, defaultParams, type VizFxVisual } from './slots.ts';
import { range, type RuleTree } from '../../mapping/read-into.ts';
import { persisted, persistedElsewhere, type BlockTable } from '../../mapping/blocks.ts';

/**
 * Stems → per-layer presence.
 *
 * The same shape as plife's population lane and stem-follow's brightness lane,
 * for the same reason all three exist separately: they answer to different
 * owners on different timescales. This one is the slowest of the three —
 * emitters swelling and shrinking is a *shape* change, and shape that tracks a
 * beat reads as flicker.
 *
 *   energy_k = floor + (1 - floor) · smoothed(stem_k)^curve
 */
export interface VizFxEnergyConfig {
  /** false = every layer sits at full presence; the lane is a no-op, not a blackout */
  followStems: boolean;
  /** what a fully silent instrument keeps, 0..1 — a ghost, not a hole */
  floor: number;
  /** exponent on the smoothed level; >1 means only loud passages swell */
  curve: number;
  /** EMA time constant in ms */
  smoothingMs: number;
}

export function defaultEnergy(): VizFxEnergyConfig {
  return {
    followStems: true,
    // Higher than stem-follow's 0.25, and higher again after measurement. A
    // layer at zero presence has emitters of zero size, i.e. it stops painting
    // into the field entirely, and the field's memory then erases it over
    // `decay` — "quiet" has to mean quiet, not gone.
    //
    // 0.4 rather than the 0.32 first tried because this lane multiplies with
    // stem-follow's brightness lane and with the impulse flash, and the three
    // together were a ~24× swing between a silent intro and a chorus. Legible is
    // the goal; a white plate for four seconds is past legible.
    floor: 0.4,
    curve: 1,
    // ~2.5× stem-follow's 300 ms. Light may track a phrase; geometry should not.
    smoothingMs: 750,
  };
}

/** Panel range and loader clamp for the energy lane, in one table. */
export const ENERGY_RANGE = {
  floor: { min: 0, max: 1 },
  curve: { min: 0.2, max: 4 },
  smoothingMs: { min: 100, max: 5000 },
} as const;

/** Panel range and loader clamp for every macro. One range for all of them. */
export const MACRO_RANGE = { min: 0, max: 2 } as const;

/** Hard bound on `emittersPerLayer`; the emitter buffer is sized to it. */
export const MAX_EMITTERS_PER_LAYER = 6;

const ENERGY_RULES: RuleTree = {
  floor: range(ENERGY_RANGE.floor.min, ENERGY_RANGE.floor.max),
  curve: range(ENERGY_RANGE.curve.min, ENERGY_RANGE.curve.max),
  smoothingMs: range(ENERGY_RANGE.smoothingMs.min, ENERGY_RANGE.smoothingMs.max),
};

/**
 * **Which blocks of `VizFxConfig` are saved.** Same mechanism and the same
 * guarantee as plife's `PLIFE_BLOCKS` and physarum's `PHYSARUM_BLOCKS`: the table
 * is exhaustive over the config's object-valued keys, so a block added here is a
 * compile error until it is declared, and declaring it is all its registration
 * takes.
 *
 * `macros` is the one block in the repo whose *keys* are not fixed — they are the
 * visual's own macro list — so its defaults and rules are functions of the live
 * config rather than tables. That works for exactly the reason the whole design
 * works: the reader walks the destination, and the destination already has the
 * right keys.
 *
 * `emittersPerLayer` and the transport are scalars on the config root rather than
 * blocks, so they stay in `VizFxSim.serializeExtras` by hand.
 */
export const VIZFX_BLOCKS: BlockTable<VizFxConfig> = {
  macros: persisted(
    (cfg) => Object.fromEntries(Object.keys(cfg.macros).map((key) => [key, 1])),
    {
      rules: (cfg) =>
        Object.fromEntries(
          Object.keys(cfg.macros).map((key) => [key, range(MACRO_RANGE.min, MACRO_RANGE.max)]),
        ),
    },
  ),
  energy: persisted(defaultEnergy, { rules: ENERGY_RULES }),

  render: persistedElsewhere(
    'the mapping file owns the render chain: `ModulationConfig.render` IS this ' +
      'object, shared by reference, and `parseModulation` reads it.',
  ),
  palette: persistedElsewhere(
    'same as render — `ModulationConfig.palette` is this object by reference, and ' +
      'it needs a reader `readInto` cannot be. See `parseModulation`.',
  ),
  params: persistedElsewhere(
    'θ. Reproduced from (seed, preset, modulation) on every load rather than ' +
      'stored — the seed is what persists, and a saved θ would be overwritten by ' +
      'the first modulator tick anyway. The two slots the look tab owns and the ' +
      'modulator excludes are the exception, and they ride the bespoke `look` key ' +
      'in `VizFxSim.serializeExtras`.',
  ),
};

/**
 * What `ModTargetConfig` needs from a species, and nothing more. `brightness` is
 * mirrored out of `params['layer<i>.brightness']` every tick — θ is the source
 * of truth and this is the workbench's readout of it (see `VizFxSim.syncSpecies`).
 */
export interface VizFxLayerConfig {
  name: string;
  brightness: number;
}

export interface VizFxConfig {
  /** the shared HDR chain. Structural (outside θ) but saved with the mapping. */
  render: RenderConfig;
  /** K. The four stems; see the header note in `vizfx.ts`. */
  speciesCount: number;
  species: VizFxLayerConfig[];
  /** static per-layer colour; not part of θ */
  palette: Palette;
  /** θ, keyed by the visual's slot names. The only place a parameter lives. */
  params: Record<string, number>;
  /** the always-yours multipliers, keyed by the visual's macro keys */
  macros: Record<string, number>;
  energy: VizFxEnergyConfig;
  /** orbiting clusters per layer, 1..MAX_EMITTERS_PER_LAYER. Structural. */
  emittersPerLayer: number;
  /** warp+draw steps per 60 Hz model tick; accumulated so fractional values work */
  speed: number;
  paused: boolean;
}

/** Custom mode with the v2 fields neutral — bit-identical to the pre-v2 default. */
export function defaultVizFxPalette(v: VizFxVisual): Palette {
  return customPalette(Array.from({ length: v.speciesCount }, (_, i) => defaultVizFxColor(v, i)));
}

/** Fallback for `paletteLinear` when the palette is shorter than K. */
export function defaultVizFxColor(v: VizFxVisual, index: number): string {
  const hex = v.paletteHex;
  return (hex[index % Math.max(hex.length, 1)] ?? '#808080') as string;
}

export function defaultVizFxConfig(v: VizFxVisual): VizFxConfig {
  const render = defaultRenderConfig();
  // The FIELD is this substrate's feedback. Post's render-domain echo would be a
  // second, redundant one stacked on top of a lane that already advances once
  // per rendered frame, so it would double the memory without adding a control.
  // Off.
  render.feedback = { amount: 0, zoom: 1 };
  // No soil field here, so the ember underlay has nothing to draw from.
  render.grade.soilTint = 0;
  // The field is a wide, soft, luminous accumulation rather than a set of thin
  // filaments, and the grade has to answer that rather than physarum's problem.
  //
  // Contrast up and the black point deepened, because the warp's whole output is
  // *gradients* and a flat grade turns a hundred overlapping soft arms into one
  // even wash. Bloom moved the opposite way from the obvious guess: threshold up
  // from 0.55 and intensity down from 0.85, because bloom applied to something
  // already soft is not glow, it is mush — it was measured burying the arms it
  // was meant to make luminous. Here bloom is for the emitter cores and the
  // shockwave fronts, which are the only hard edges in the image, and for nothing
  // else.
  render.grade.contrast = 1.35;
  render.grade.blackPoint = 0.05;
  render.bloom.threshold = 0.9;
  render.bloom.intensity = 0.45;
  // The tone map is the one place this substrate reverses a decision the shared
  // chain made for good reasons, so it is worth stating why.
  //
  // `reinhard` is the shipped default because it is luminance-only and leaves
  // channel ratios alone: a hot core keeps its species' hue instead of bleaching,
  // and with a static palette hue *is* the species label. The cost — one channel
  // may exceed 1 and clip — is negligible when the bright pixels are physarum's
  // thin filaments.
  //
  // Here the bright pixels are whole arms covering a fifth of the frame, and that
  // same clip is ruinous: the bass orange is (1.0, 0.20, 0.01) in linear light,
  // so every pixel past the knee lands on exactly that hue with no interior
  // gradient, and the image comes out as flat poster-paint strokes. Measured, not
  // predicted — it is what the first three tuning passes all looked like.
  //
  // `reinhardJodie` blends toward per-channel Reinhard with brightness, so
  // nothing can exceed 1 and cores converge on white. That is the thing the
  // shared default rejects, and it is exactly what a nebula wants: white-hot
  // cores, coloured gas around them. Layer identity survives because it is
  // carried by the *arms*, which sit below the knee, rather than by the cores.
  render.grade.tonemap = 'reinhardJodie';
  // The one grade knob that had to move a long way, and the reason is the tone
  // map's shape rather than taste. Reinhard is luminance-only and leaves channel
  // ratios alone, so a pixel far above the knee keeps its hue and loses its
  // *gradient* — the red channel of the bass orange pins at 1 while green and
  // blue stay proportionally low, and the result is a flat poster-paint patch.
  //
  // For physarum that is exactly right: its bright pixels are thin filaments and
  // a saturated hot core is the look. Here the bright regions are whole arms
  // covering a fifth of the frame, so a target of 0.3 put most of the image past
  // the knee and the arms came out as solid strokes of pure hue with no interior
  // structure at all. 0.12 keeps the bulk of the field in the part of the curve
  // that still has slope, and the cores — which are genuinely small — are the
  // only things that reach the top of the curve.
  //
  // 0.10 rather than 0.16 after watching events land: the peak-to-mean ratio of
  // this image is around 20, so a mean of 0.16 puts the peaks at ~3.2, and
  // anything past about 2 is white under any of the tone maps. Simultaneous
  // impulse flashes are what reach it. A dimmer *mean* with the peaks intact is
  // the trade, and the grade's contrast and black point below are what stop that
  // reading as a washed-out image.
  //
  // 0.18, revised up from 0.1 (user call — the shipped frames ran dim). The
  // decisive arithmetic, found while diagnosing plasma's dead choruses: the
  // grade's black cutoff is `blackPoint + (1−blackPoint)·pivot·(1−1/contrast)`
  // ≈ 0.119 at this file's own contrast/blackPoint — *above* a target of 0.1.
  // A controller pinning the frame mean below the black point guarantees that
  // most of every frame renders black and only bloom can rescue the rest,
  // whatever the visuals do upstream. 0.18 puts the mean comfortably over the
  // cutoff while leaving ~2.5 stops to the white shoulder for peaks and rings.
  render.grade.autoTarget = 0.18;
  // 3 s rather than the shared 6. The shipped tau is long so that a transient
  // passes straight through the controller and a drop still reads as a drop —
  // correct, and it stays correct here because this substrate has 1.7 s of
  // memory in the *field itself*, which already smooths a section change into a
  // swell before the controller ever sees it. Left at 6 s the two smoothers
  // stack, and a chorus arriving spent several seconds as a white plate: the
  // per-layer swell (energy lane × stem-follow × impulse flash) is a ~14×
  // excursion, and the controller could not follow it.
  render.grade.autoTau = 1;
  // …and 1 s rather than 3 after watching a chorus arrive with the field finally
  // exposed correctly. The controller is multiplicative, so the time it needs to
  // absorb a level change is `autoTau × ln(ratio)` — it does not depend on the
  // frame rate, only on how far it has to travel.
  //
  // Measured on free-fall: the adapted gain sat at 32 through the quiet intro and
  // wanted 0.78 in the chorus that follows. That is a 41× step, so at 3 s the
  // controller needed `3 × ln(41)` ≈ 11 seconds to get there — and every one of
  // those seconds is a frame that is up to 20× over target, i.e. white. The
  // substrate was spending the first third of every loud section blown out, which
  // is indistinguishable from being permanently blown out if the sections are
  // 30 s long.
  //
  // At 1 s the same step takes ~3.7 s. A drop still reads as a drop — the
  // overshoot is what a drop *is* here — but it now decays over a few seconds
  // instead of most of the section. The stacked-smoother argument that took this
  // from 6 s to 3 s is the same argument one step further: the field itself has
  // ~100 steps of memory, so the image is already smoothed before the controller
  // sees it, and a long tau here is a second smoother on an already-smooth
  // signal. What it is not is a licence to go lower — under about half a second
  // the controller starts tracking individual bars and the image visibly breathes.
  // ── the controller needs more road here than the other two substrates ───────
  //
  // The shared rails span 0.05…16 — about 9 stops — and that was measured on
  // physarum, whose trail field is a fixed-length integrator: its brightness
  // tracks how much the agents deposit and little else.
  //
  // This field's level is proportional to its *memory*, `1/fade`, and `fade` is a
  // modulated slot ranging 0.003…0.07. Memory therefore swings 23× on that slot
  // alone, before `emitGain`, `coreSize` (which enters as an area, i.e. squared)
  // and the energy lane have said anything. Measured on free-fall the mean field
  // luminance ran ~32× between a freshly-seeded quiet passage and a dense one at
  // equilibrium — so a controller centred for one end spends the other end
  // pinned, which is precisely the state the substrate shipped in: `autoMinGain`
  // 0.05 against a frame that wanted 0.0065, with the measured mean 8–36× over
  // target and no way for the exposure slider to help, because the controller had
  // already run out of road before the slider was touched.
  //
  // 0.01…64 is ~12.6 stops. That is not a licence for the scene to be
  // mis-scaled — `DRAW_SCALE` is still centred on the geometric middle of the
  // span — it is headroom so that the ends of the span are absorbed rather than
  // clipped. `autoTau` above is unchanged and is still what protects a drop from
  // being flattened: widening the rails changes where the controller *can* go,
  // not how fast it goes there.
  render.grade.autoMinGain = 0.01;
  render.grade.autoMaxGain = 64;
  return {
    render,
    speciesCount: v.speciesCount,
    species: Array.from({ length: v.speciesCount }, (_, i) => ({
      name: (v.layerNames[i] ?? `layer ${i}`) as string,
      brightness: 1,
    })),
    palette: defaultVizFxPalette(v),
    params: defaultParams(v),
    macros: defaultMacros(v),
    energy: defaultEnergy(),
    emittersPerLayer: Math.min(Math.max(v.emittersPerLayer, 1), MAX_EMITTERS_PER_LAYER),
    speed: 1,
    paused: false,
  };
}
