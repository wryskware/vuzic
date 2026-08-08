import type { Palette } from '../sim/physarum/config.ts';
import type { RenderConfig } from '../sim/render/config.ts';
import type { ModGroup } from './preset.ts';
import type { SlewRates } from './slew.ts';

export interface BoundaryOptions {
  /**
   * Optional, and no longer part of any scene machinery (plan.md Revision 3):
   * a section boundary is an *event*, so it steps the slow lane and injects
   * matter. Nothing else in the runtime knows what a section is.
   */
  enabled: boolean;
  /** how far the slow parameters jump towards target on a section change (0..1) */
  snapFraction: number;
  /** fraction of agents respawned at a section change (0..1) */
  respawnFraction: number;
}

/**
 * The whole mapping, as data. Version 3 is the Revision 3 rewrite: no anchors, no
 * k-means, no simplex, no per-scene presets — a handful of depths and speeds, plus
 * the two static art-direction blocks that survived unchanged.
 *
 * **The seed is deliberately not in here.** The seed is a run input (Decision 5)
 * and lives in the URL or the pin, so a config file describes *how much* the music
 * moves things, not *which* world you happened to be looking at. Sharing a file
 * and sharing a world are two different acts.
 */
export interface ModulationConfig {
  /** 1/2 = anchor era; 3 = seeded random-projection modulation */
  version: 3;
  /** K this was authored for; a mismatch with the live sim cannot be applied */
  speciesCount: number;
  /**
   * Static per-species colour, art-directed once. The runtime shares this object
   * with `PhysarumConfig.palette`, so a palette edit in the workbench is saved
   * without any syncing step.
   */
  palette: Palette;
  /** Phase 7's exposure/tone/bloom settings, shared by reference like the palette. */
  render: RenderConfig;
  /** false = manual: the sliders are absolute and nothing modulates */
  enabled: boolean;
  /** global excursion depth; 1 ≈ typical |tanh| of 0.55 on a unit projection */
  depth: number;
  /** per-group trim on top of `depth` */
  groupDepth: Record<ModGroup, number>;
  /** multiplier on the slew clock; 2 = everything reacts twice as fast */
  responseSpeed: number;
  slew: SlewRates;
  boundary: BoundaryOptions;
}

export const MODULATION_VERSION = 3;
