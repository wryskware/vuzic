/**
 * Per-particle luminance — the brightness dynamic range lane.
 *
 * ## The problem this exists to fix
 *
 * Every particle of a species used to render at exactly one brightness. The
 * three brightness mechanisms that already existed are all **species-uniform**:
 * `brightness` × stem-follow (`brightFollow`), the impulse `flash`
 * (`brightMul`), and the `intensity` weight. They move a whole colony together,
 * so a species is a flat sheet of identical dots that gets brighter and dimmer
 * as one object. The canvas has been HDR since phase 8 and nothing in the image
 * was ever using the headroom, because *everything* sat at the same level and
 * auto-exposure then parked that level at `autoTarget`.
 *
 * This lane is the fourth mechanism and it is deliberately the only one that
 * varies **within** a species. It composes multiplicatively on top of the other
 * three, in the vertex shader, so it cannot fight them: they decide how bright
 * the colony is, this decides how the colony's own brightness is *distributed*
 * across its members.
 *
 * ## What drives it
 *
 * Per-particle speed, normalised by the velocity ceiling — the exact quantity
 * the sprite's velocity stretch already uses (`min(|v| / maxSpeed, 1)`), so the
 * two motion cues agree by construction rather than by coincidence. That choice
 * is only viable because the `speed`/`drag` split landed: before it, 36-45% of
 * live particles sat pinned at exactly `maxSpeed` and this lane would have been
 * a two-valued switch.
 *
 * ## The curve
 *
 *     gain(u) = 2 ^ ( stops · (u^curve − anchor)  +  jitter )
 *     anchor  = mid^curve
 *
 * Three properties, each load-bearing:
 *
 * - **It is exponential in a shaped speed.** Working in stops rather than in
 *   linear gain is what makes the knob perceptual: one unit of `depth` is one
 *   doubling wherever you are on the curve, which is roughly how brightness is
 *   read. `curve` then shapes *where along the speed axis* the doublings are
 *   spent, and it does so from both ends at once, which is worth stating
 *   because it is not what you would guess: raising it pushes `anchor` down as
 *   well as flattening the low end, so the slow population is *compressed
 *   toward* today's brightness (a tighter band, not a darker one) while the
 *   peak-to-bulk contrast grows. That is exactly the "most particles in a
 *   modest band, peaks spike" brief, and it is one knob rather than two.
 * - **It is anchored, not offset.** `mid` is the normalised speed whose gain is
 *   exactly 1, i.e. the speed that renders at today's brightness. Particles
 *   slower than `mid` get dimmer and faster ones brighter, so the lane
 *   redistributes light instead of adding it — which matters because
 *   auto-exposure would otherwise simply divide any net gain back out and the
 *   knob would read as "no change" at the bottom and "washed out" at the top.
 * - **depth = 0 is bit-identical to the old look.** Not approximately: at depth
 *   0 `lumaUniforms` returns a span of 0 stops, no jitter and no white push, so
 *   the shader computes `exp2(0) = 1` and `mix(rgb, …, 0) = rgb`. That is the
 *   A/B baseline and the escape hatch, and it is a property of the arithmetic
 *   rather than a branch that could rot.
 *
 * ## Headroom as a budget
 *
 *     stops = depth · (1 + hdrBudget · log2(H))
 *
 * `H` is the display's measured headroom over diffuse white (1 on every SDR
 * host). The final grade maps `H · tonemap(c / H)`, so the *display* range
 * available above diffuse white is exactly `log2(H)` stops — this spends them,
 * scaled by `hdrBudget`, and spends nothing when there are none. Multiplicative
 * rather than additive so `depth = 0` stays off on every display; at H = 1 it
 * degenerates to `stops = depth` exactly, which is what makes `depth` mean "the
 * SDR span" and the HDR rendition a documented extension of it rather than a
 * separate tuning.
 *
 * Because `anchor` sits low on the curve (`mid^curve` ≈ 0.13 at the shipped
 * numbers) roughly seven eighths of the span lands *above* the anchor. That is
 * the asymmetry the brief asks for and it falls out of the anchor rather than
 * needing its own knob: the bulk loses a fraction of a stop, the tail gains
 * most of the budget.
 *
 * ## The SDR rendition
 *
 * On an SDR canvas `H = 1`, the span collapses to `depth`, and — worse — the
 * tone curve asymptotes at 1.0, so a *saturated* hot core cannot get brighter
 * once its strongest channel is there. It reads as more saturated, not as more
 * light. `whitePeak` is the substitute: the fastest particles are desaturated
 * toward their own luminance, which lifts the two weak channels and lets the
 * core actually reach white.
 *
 * It is deliberately traded off against real headroom — `white` is divided by
 * `1 + hdrBudget·log2(H)` — because on a display that *has* headroom, bleaching
 * is a cost rather than a cue: `render/config.ts` argues at length that hue is
 * the species label and a tone curve that whitens highlights deletes the label
 * exactly where the image is most interesting. Reinhard was chosen over
 * reinhardJodie for that reason. So: bleach only as far as the display forces
 * you to, and fade it out as headroom arrives.
 *
 * ## Why this is linear RGB and not HSLuv
 *
 * The palette is authored in HSLuv (v2) and that is where hue and perceptual
 * lightness are chosen — but it reaches the GPU as premultiplied *linear* RGB,
 * once per species per frame, and a per-particle HSLuv round trip in the vertex
 * shader would be a transcendental stack run millions of times a frame for a
 * result that is, at this point in the pipeline, a two-axis move: lightness up
 * and saturation down. The stops-domain gain is the lightness axis (a log scale
 * is the perceptual part) and the luminance-preserving desaturation is the
 * saturation axis. Authoring stays in HSLuv; the per-particle deviation does
 * not need it.
 */

import { range, type RuleTree } from '../../mapping/read-into.ts';

export interface PlifeLumaConfig {
  /**
   * Span of the lane, in **stops**, on an SDR display. 0 turns it off outright
   * and reproduces the pre-lane look exactly.
   *
   * Ranged to 8 because that is what the headroom argument costs: to put a peak
   * particle at the top of a 4× display under `H·reinhard(c/H)` the scene value
   * has to sit ~9H above the anchor, which is a bit over 6 stops before the
   * `hdrBudget` multiplier contributes any of it. A 0..2 knob could not reach
   * the effect it names.
   */
  depth: number;
  /**
   * Exponent on the normalised speed. 1 is a straight ramp; higher tightens the
   * slow population into a band around today's brightness and hands the freed
   * span to the fast tail, so spikes read as events rather than as a general
   * brightening.
   */
  curve: number;
  /**
   * The normalised speed that keeps today's brightness — the gain-1 anchor. Low
   * values push more of the population into the "brighter than before" half.
   */
  mid: number;
  /**
   * How much of the display's measured headroom, in stops, the lane spends on
   * peaks. 0 = ignore the display and use `depth` everywhere, which is the knob
   * to reach for when tuning one rendition to match the other.
   */
  hdrBudget: number;
  /**
   * How far the fastest particles are pushed toward white, 0..1. The SDR peak
   * cue; automatically scaled down as real headroom arrives.
   */
  whitePeak: number;
  /**
   * Static per-particle random gain, as a **fraction of the span**. This is the
   * "a species should shimmer, not be a flat sheet" term: it is drawn once per
   * particle index from the world seed, so it is spatial texture rather than
   * temporal noise, and it scales with `depth` so it vanishes with the lane.
   */
  jitter: number;
}

/**
 * Depth at which the white push and the jitter reach their authored strength.
 *
 * Below it they ramp linearly to zero, which is the one thing that makes
 * `depth = 0` an exact identity rather than "the gain is 1 but everything is
 * still bleached and speckled". A ramp rather than a branch so there is no step
 * anywhere on the slider.
 */
export const LUMA_DEPTH_REF = 1;

/** Rec. 709 luminance weights — the desaturation axis. Mirrored in render.wgsl. */
export const LUMA_WEIGHTS: readonly [number, number, number] = [0.2126, 0.7152, 0.0722];

/** Panel range and loader clamp, one table so a saved value fits its slider. */
export const LUMA_RANGE: Readonly<Record<keyof PlifeLumaConfig, { min: number; max: number }>> = {
  depth: { min: 0, max: 8 },
  curve: { min: 0.25, max: 6 },
  mid: { min: 0.02, max: 1 },
  hdrBudget: { min: 0, max: 2 },
  whitePeak: { min: 0, max: 1 },
  jitter: { min: 0, max: 0.5 },
};

/**
 * The shipped starting point, and every number in it is a judgement rather than
 * a measurement — the distribution of `|v| / maxSpeed` depends on the seed, the
 * macros and the track, so there is no "correct" anchor to derive. What the
 * defaults encode:
 *
 * - `depth` 3 on SDR, ×3 on a ~1000-nit panel (headroom ≈ 4 ⇒ 2 stops ⇒ 9
 *   stops total). At 9 stops the tail lands within a stop of the display peak
 *   and the bulk loses about one stop, which is the brief's shape.
 * - `curve` 2.5 and `mid` 0.4 put the anchor at 0.4^2.5 ≈ 0.10, so ~90% of the
 *   span is above the anchor and a particle has to be genuinely fast before it
 *   collects much of it.
 * - `whitePeak` 0.5 is half-way to white at the ceiling, on SDR, and about a
 *   sixth of the way there at headroom 4.
 * - `jitter` 0.12 is ~1 stop peak-to-peak at the shipped span. Enough that a
 *   still frame has grain; far too little to be mistaken for the speed signal.
 */
export function defaultPlifeLuma(): PlifeLumaConfig {
  return {
    depth: 3,
    curve: 2.5,
    mid: 0.4,
    hdrBudget: 1,
    whitePeak: 0.5,
    jitter: 0.12,
  };
}

export function lumaRules(): RuleTree {
  return Object.fromEntries(
    (Object.keys(LUMA_RANGE) as (keyof PlifeLumaConfig)[]).map((key) => {
      const r = LUMA_RANGE[key];
      return [key, range(r.min, r.max)];
    }),
  );
}

/**
 * What reaches the GPU: five floats in the Globals block, composed here so the
 * shader is a fused multiply-add and an `exp2` rather than a policy.
 *
 * Everything display-dependent is resolved on this side. The shader knows
 * nothing about headroom, which is what lets the workbench's headroom override
 * change the whole rendition without touching a pipeline.
 */
export interface LumaUniforms {
  /** total span across `u ∈ [0,1]`, in stops. 0 = the lane is off. */
  stops: number;
  /** the power-curve exponent, floored away from 0 so `pow` stays defined */
  exponent: number;
  /** `mid^exponent` — the shaped speed whose gain is exactly 1 */
  anchor: number;
  /** peak desaturation toward luminance, 0..1 */
  white: number;
  /** half-width of the per-particle static gain, in stops */
  jitterStops: number;
}

/** The off state, and the reason `depth = 0` is an identity rather than a branch. */
const LUMA_OFF: LumaUniforms = { stops: 0, exponent: 1, anchor: 0, white: 0, jitterStops: 0 };

/**
 * Compose the lane against the display it is being watched on.
 *
 * `headroom` is the same number the grade uses (`GpuRuntimeContext.displayHeadroom`,
 * or the export host's mastering ratio) — 1 on every SDR host. Anything that is
 * not a finite number at or above 1 falls back to 1: a display cannot have
 * negative headroom, and a host that reports nonsense should get the SDR
 * rendition rather than propagate a NaN into a uniform, where it would take out
 * every particle's position as well as its colour.
 */
export function lumaUniforms(cfg: PlifeLumaConfig, headroom: number): LumaUniforms {
  const depth = Math.max(cfg.depth, 0);
  if (!(depth > 0)) return LUMA_OFF;
  // `Math.max(NaN, 1)` is NaN, so the guard has to be the comparison rather than
  // the clamp. Same shape everywhere below: `!(x > y)` catches NaN, `x <= y`
  // does not.
  const usable = Number.isFinite(headroom) && headroom > 1 ? headroom : 1;
  const budget = Math.max(cfg.hdrBudget, 0) * Math.log2(usable);
  const stops = depth * (1 + budget);
  const exponent = Math.max(cfg.curve, 0.05);
  const anchor = Math.pow(Math.min(Math.max(cfg.mid, 1e-3), 1), exponent);
  // The two decorations ramp in over the first stop of depth — see LUMA_DEPTH_REF.
  const on = Math.min(depth / LUMA_DEPTH_REF, 1);
  return {
    stops,
    exponent,
    anchor,
    white: (on * Math.min(Math.max(cfg.whitePeak, 0), 1)) / (1 + budget),
    jitterStops: on * Math.max(cfg.jitter, 0) * stops,
  };
}

/** `u^exponent`, the shaped speed both the gain and the white push read. */
export function shapedSpeed(u: LumaUniforms, speed01: number): number {
  return Math.pow(Math.max(Math.min(speed01, 1), 1e-6), u.exponent);
}

/**
 * The multiplier applied to a particle's premultiplied colour.
 *
 * `jitter01` is that particle's static draw in [0, 1) — 0.5 is the neutral
 * centre, which is what a caller with no draw should pass and what makes this
 * function testable without reimplementing the PCG hash.
 *
 * Mirrors `vsParticles` in render.wgsl exactly; `plife-luma.test.ts` pins the
 * two together via the WGSL source.
 */
export function particleGain(u: LumaUniforms, speed01: number, jitter01 = 0.5): number {
  const ev =
    u.stops * (shapedSpeed(u, speed01) - u.anchor) + (jitter01 - 0.5) * 2 * u.jitterStops;
  return Math.pow(2, ev);
}

/** How far toward its own luminance a particle's colour is pushed, 0..1. */
export function particleWhiteMix(u: LumaUniforms, speed01: number): number {
  return u.white * shapedSpeed(u, speed01);
}

/**
 * Apply the whole lane to one premultiplied linear colour. Not used by the
 * renderer — the GPU does this — but it is what the tests measure a *spread*
 * with, and what a future CPU preview would call.
 */
export function applyLuma(
  u: LumaUniforms,
  rgb: readonly [number, number, number],
  speed01: number,
  jitter01 = 0.5,
): [number, number, number] {
  const w = particleWhiteMix(u, speed01);
  const l =
    rgb[0] * LUMA_WEIGHTS[0] + rgb[1] * LUMA_WEIGHTS[1] + rgb[2] * LUMA_WEIGHTS[2];
  const gain = particleGain(u, speed01, jitter01);
  return [
    (rgb[0] + (l - rgb[0]) * w) * gain,
    (rgb[1] + (l - rgb[1]) * w) * gain,
    (rgb[2] + (l - rgb[2]) * w) * gain,
  ];
}
