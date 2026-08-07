import type { Palette } from '../sim/physarum/config.ts';
import type { RenderConfig } from '../sim/render/config.ts';
import type { Preset } from './preset.ts';
import type { SlewRates } from './slew.ts';

/** One vertex of the simplex: a point in latent space and the θ that belongs there. */
export interface Anchor {
  /** stable across refits only if the anchor was hand-named; generated ids are positional */
  id: string;
  name: string;
  /** k-means centre in the timeline's latent space, `latentDims` long */
  center: number[];
  preset: Preset;
}

export interface BoundaryOptions {
  enabled: boolean;
  /** how far the slow parameters jump towards target on a section change (0..1) */
  snapFraction: number;
  /** fraction of agents respawned at a section change (0..1) */
  respawnFraction: number;
}

export interface KMeansSettings {
  k: number;
  /** fixed literal by default; changing it is a different (still reproducible) anchor set */
  seed: number;
  iterations: number;
}

/**
 * The whole mapping, as data. This file IS the art direction — nothing in the
 * runtime holds tuning state that is not serialised here.
 */
export interface MappingConfig {
  /** 1 = colours inside every anchor preset; 2 = one static top-level palette */
  version: 2;
  /** K the presets were authored for; a mismatch with the live sim forces a refit */
  speciesCount: number;
  /**
   * Static per-species colour, art-directed once. Top level, NOT per anchor —
   * blending hue between anchors was the mistake Revision 2 undoes. The runtime
   * shares this object with `PhysarumConfig.palette`, so a palette edit in the
   * workbench is saved without any syncing step.
   */
  palette: Palette;
  /**
   * Phase 7's exposure/tone/bloom settings. Like the palette, a single top-level
   * object shared by reference with the live `PhysarumConfig.render` and never
   * blended per anchor — the image's grade is the track's identity, not one
   * section's mood. Older files without it get the defaults.
   */
  render: RenderConfig;
  /** dims of the anchors' centres, i.e. the timeline's latent channel width */
  latentDims: number;
  kmeans: KMeansSettings;
  /**
   * Simplex temperature as a multiple of `distanceScale`, so the same number
   * means the same sharpness on any track. Effective T = temperature * distanceScale.
   * ~0.1 is near-nearest-neighbour; ~1 blends a handful of anchors visibly.
   */
  temperature: number;
  /** typical squared distance between neighbouring anchors, from the fit */
  distanceScale: number;
  slew: SlewRates;
  boundary: BoundaryOptions;
  anchors: Anchor[];
}

export const MAPPING_VERSION = 2;
