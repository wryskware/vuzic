/**
 * The substrate-agnostic half of the modulation registry's vocabulary.
 *
 * `preset.ts` (physarum) and `sim/plife/preset.ts` (particle life) each define
 * *which* slots exist and what their bounds are; the types here are what the
 * Modulator, the slew limiter and the workbench share without caring whose
 * registry they are driving. Moving them out of preset.ts is what lets a second
 * simulation bring its own θ without importing physarum's config.
 */

/** Per-parameter timescale class. Motion faster than the sim's relaxation time is mush. */
export type SlewClass = 'fast' | 'medium' | 'slow';

export const CLASS_FAST = 0;
export const CLASS_MEDIUM = 1;
export const CLASS_SLOW = 2;
export const CLASS_NAMES: readonly SlewClass[] = ['fast', 'medium', 'slow'];

/**
 * Which lane of the workbench a slot answers to. One depth slider per group, so a
 * human can say "more shape, less light" without touching 92 numbers.
 *
 * The four groups are shared across simulations on purpose: "structure",
 * "matrix", "population" and "decay" name *roles*, not physarum fields, and a
 * particle-life registry maps its own slots into the same four so the depth
 * sliders keep meaning the same thing when the substrate changes.
 */
export type ModGroup = 'structure' | 'matrix' | 'population' | 'decay';

export const MOD_GROUPS: readonly ModGroup[] = ['structure', 'matrix', 'population', 'decay'];

/**
 * How the music is allowed to move one slot.
 *
 *   additive:        p = clamp(base + half · tanh(depth · w·ẑ), lo, hi)
 *   multiplicative:  p = clamp(base · exp(half · tanh(depth · w·ẑ)), lo, hi)
 *
 * `lo`/`hi` may be *narrower* than the slot's hard θ bound: the hard bound is
 * "what a file may contain", this is "where the music may wander unsupervised".
 * `jitter` is the seeded personality spread applied to the shipped default —
 * additive units, or ln units when `mult`.
 */
export interface ModSpec {
  group: ModGroup;
  lo: number;
  hi: number;
  half: number;
  jitter: number;
  mult: boolean;
}
