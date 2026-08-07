/**
 * Phase 7 — the render chain's configuration.
 *
 * The chain is:
 *
 *   K trail layers ─┐
 *   soil (tint) ────┼─► composite ──► HDR rgba16float ─┬─► auto-exposure (compute)
 *   previous HDR ───┘   (scene exposure ×               │
 *                        adapted gain)                  ├─► bloom: threshold + 3-level
 *                                                       │    separable Gaussian chain
 *                                                       └─► grade ──► swapchain (8-bit)
 *
 * Everything the *sim* does stays linear and unbounded; nothing is clamped until
 * the very last pass. That is the fix for "trails clip to white at high stem
 * drive": there is no longer an 8-bit surface anywhere upstream of the tone map,
 * and the tone map chosen has no clipping point at all.
 *
 * These knobs are art direction, not θ. They are NOT blended between anchors
 * (the whole point of Revision 2's static palette is that the image's identity
 * holds still while the music moves brightness and structure), but they *are*
 * serialised with the mapping so a tuning session survives a reload.
 */

/**
 * Highlight rolloff. Deliberate choice, per the phase brief:
 *
 * - `reinhard`      — luminance-only Reinhard, x / (1 + L). **Default.** It is
 *   asymptotic, so no input maps to a flat plate, and it leaves channel ratios
 *   alone, so a hot core stays the species' own colour instead of bleaching.
 *   That matters more here than anywhere else: with a static palette, hue *is*
 *   the species label, and a tone curve that whitens highlights deletes the
 *   label exactly where the image is most interesting. One channel may exceed 1
 *   and clip, which reads as a saturated hot core.
 * - `reinhardJodie` — the same curve blended toward per-channel Reinhard with
 *   brightness. Nothing can exceed 1, cleanest highlights, but every core
 *   converges on white. Tried first here, and rejected on the grounds above.
 * - `aces`          — the Narkowicz fit that shipped in phase 4. Kept for
 *   comparison. Its slope at black is ~0.21, which is where the old image got
 *   its blacks from, and its shoulder reaches exactly 1.0 at finite input,
 *   which is where the clipping came from. Both of those jobs now belong to
 *   controls that can be tuned separately.
 * - `linear`        — clamp only. Debug: shows where the raw values actually are.
 *
 * Deep blacks are produced by `blackPoint` + `contrast` instead of by the tone
 * curve, so the two can be dialled independently.
 */
export const TONEMAPS = ['reinhard', 'reinhardJodie', 'aces', 'linear'] as const;
export type ToneMap = (typeof TONEMAPS)[number];

export function tonemapIndex(name: ToneMap): number {
  const i = TONEMAPS.indexOf(name);
  return i < 0 ? 0 : i;
}

/** Bloom mip levels the chain allocates. 3 is plenty; more is memory for no look. */
export const MAX_BLOOM_LEVELS = 3;

export interface GradeConfig {
  /** extra stops on top of the scene exposure that lives in θ */
  exposureEv: number;
  /** slow adaptation toward `autoTarget` mean luminance */
  autoExposure: boolean;
  /** mean linear luminance of the HDR frame the loop drives toward */
  autoTarget: number;
  /** adaptation time constant in seconds. Long on purpose — see the comment on defaults. */
  autoTau: number;
  autoMinGain: number;
  autoMaxGain: number;
  tonemap: ToneMap;
  /** subtracted after the tone map and rescaled; this is where black actually lands */
  blackPoint: number;
  /** gain about `pivot`; > 1 crushes the dim wash and leaves the filaments */
  contrast: number;
  pivot: number;
  /** 1 = as tone-mapped; > 1 pushes away from luminance */
  saturation: number;
  /** 0 = off, 1 = corners to black */
  vignette: number;
  /** dim ember underlay drawn from the soil field, in output units. 0 = off */
  soilTint: number;
  /** static sRGB hex for that underlay — it is terrain, so it does not change hue */
  soilColor: string;
}

export interface BloomConfig {
  enabled: boolean;
  /** HDR luminance a pixel must exceed to bloom at all */
  threshold: number;
  /** soft-knee width around the threshold; avoids a hard on/off edge */
  knee: number;
  /** how much of the blurred chain is added back */
  intensity: number;
  /** 1..MAX_BLOOM_LEVELS. Each level doubles the glow radius. */
  levels: number;
}

export interface FeedbackConfig {
  /**
   * Render-domain trail persistence: last frame's HDR added back into this one.
   * NOT sim state — the trail field is untouched. 0 = off.
   */
  amount: number;
  /** > 1 makes the echo expand outward each frame; 1 = pure decay in place */
  zoom: number;
}

export interface RenderConfig {
  grade: GradeConfig;
  bloom: BloomConfig;
  feedback: FeedbackConfig;
}

/**
 * Defaults chosen to look striking rather than safe (plan.md Revision 2 §3):
 * deep blacks from an explicit black point, filaments that glow rather than
 * merely being bright, and enough bloom headroom that a kick flash splashes.
 *
 * Notes on the numbers that are not obvious:
 *
 * - `autoTau` is 6 s. Short enough to stop a dense chorus from pinning the image
 *   at white, long enough that a drop still *reads* as a drop — a transient two
 *   orders of magnitude faster than the loop simply passes through it.
 * - `autoMinGain`/`autoMaxGain` span ~9 stops on purpose. Measured on Free Fall,
 *   the mean HDR luminance of the composited frame moves by more than 30x between
 *   a sparse intro and a fully grown chorus at the same θ exposure; a narrow rail
 *   (the first thing tried here, 0.5–2) simply pinned at its minimum and the
 *   image stayed a wash. The rail is there to stop a runaway, not to set the
 *   level — `autoTau` is what protects the dynamics.
 * - `threshold` 0.55 sits below the tone map's midpoint, so ordinary filaments
 *   contribute a little glow and impulse flashes contribute a lot. Raising it
 *   makes bloom an event-only effect.
 * - `soilTint` reads as almost nothing, and that is correct: `blackPoint` plus
 *   `contrast` about a 0.28 pivot crush everything below ~0.09 to true black, so
 *   0.25 of ember survives as roughly 0.15 output on fully grown soil and as
 *   nothing at all on fresh ground. It is terrain, not a feature.
 * - `feedback.amount` defaults to 0. Implemented, one slider away, and it does
 *   look good at ~0.35 with zoom 1.004 — but it fattens every filament and the
 *   brief says legibility wins. Left as an opt-in.
 */
export function defaultRenderConfig(): RenderConfig {
  return {
    grade: {
      exposureEv: 0,
      autoExposure: true,
      autoTarget: 0.3,
      autoTau: 6,
      autoMinGain: 0.05,
      autoMaxGain: 16,
      tonemap: 'reinhard',
      blackPoint: 0.035,
      contrast: 1.25,
      pivot: 0.28,
      saturation: 1.12,
      vignette: 0.35,
      soilTint: 0.25,
      soilColor: '#4a2a12',
    },
    bloom: {
      enabled: true,
      threshold: 0.55,
      knee: 0.35,
      intensity: 0.85,
      levels: MAX_BLOOM_LEVELS,
    },
    feedback: {
      amount: 0,
      zoom: 1,
    },
  };
}

/**
 * Copy a loaded render block *into* the live object, the same trick the palette
 * uses: the panel's widgets are bound to the live object by reference, so a
 * mapping load must mutate it rather than replace it.
 */
export function mergeRenderConfig(live: RenderConfig, loaded: RenderConfig): void {
  Object.assign(live.grade, loaded.grade);
  Object.assign(live.bloom, loaded.bloom);
  Object.assign(live.feedback, loaded.feedback);
}
