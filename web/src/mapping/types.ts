import type { Palette } from '../sim/palette.ts';
import type { RenderConfig } from '../sim/render/config.ts';
import type { ModGroup } from './modspec.ts';
import type { SlewRates } from './slew.ts';
import type { StemFollowConfig } from './stemfollow.ts';

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
  /** 1/2 = anchor era; 3 = raw-embedding projections; 4 = the named driver bank */
  version: 4;
  /**
   * Which simulation this mapping drives (`ModTarget.simId`). θ is a different
   * vector for every substrate, so a file authored against one is meaningless
   * applied to another even at the same K — which is why this is checked
   * alongside `speciesCount` rather than trusted. Absent in files written before
   * the second sim existed, and those are all physarum.
   */
  sim: string;
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
  /**
   * Per-driver gain, 0..2, index-aligned with the bank (Revision 4). Applied to
   * the z-scored driver *before* the projection, so 0 mutes that driver in every
   * wiring at once. This is the tuning surface: mute the drivers that are not
   * doing anything you like, boost the ones that are. Length is the track's
   * driver count; a file authored against a different one adopts what fits and
   * defaults the rest to 1.
   */
  driverGains: number[];
  /**
   * The brightness lane (Revision 4). Not modulation and not part of θ — a
   * direct wire from each species' own stem to its light, so it works in manual
   * mode and with every driver muted.
   */
  stemFollow: StemFollowConfig;
  /** multiplier on the slew clock; 2 = everything reacts twice as fast */
  responseSpeed: number;
  slew: SlewRates;
  boundary: BoundaryOptions;
  /**
   * Opaque per-sim block; the sim that owns the mapping serialises and validates
   * its own schema here (plife: macros + matrixGen). The mapping layer only
   * carries it — nothing between the file and `ModTarget.applyExtras` looks
   * inside, which is what keeps `ModulationConfig` from learning a second
   * substrate's fields. Absent is simply "defaults", so no version bump.
   */
  extras?: Record<string, unknown>;
}

export const MODULATION_VERSION = 4;
