/**
 * The seam between the mapping layer and whatever it is driving.
 *
 * Until now the Modulator was constructor-typed against `PhysarumSim` and
 * reached into `mapping/preset.ts` — physarum's θ registry — directly. That is
 * exactly one simulation's worth of coupling, and it is what a second substrate
 * (particle life) would have to fight. So the dependency is inverted: a sim
 * publishes its own registry and its own vector↔config conversion, and the
 * mapping layer only knows `ModTarget`.
 *
 * What deliberately did **not** move behind this interface:
 *
 * - **The driver bank, the projections, the slew classes and the groups.** They
 *   are the same machinery whatever they drive; a second sim brings a different
 *   slot table, not a different modulator.
 * - **The palette and the render block.** Both are already sim-agnostic (see
 *   `sim/palette.ts` and `sim/render/config.ts`) and both are shared by
 *   reference with `ModulationConfig`, which is what makes a workbench edit an
 *   edit to the thing that gets serialised.
 *
 * `ModTargetConfig` is structural on purpose: `PhysarumConfig` satisfies it
 * without declaring that it does, so nothing in the sim's own config has to bend
 * toward the mapping layer's shape.
 */
import type { ModSpec } from './modspec.ts';
import type { Palette } from '../sim/palette.ts';
import type { RenderConfig } from '../sim/render/config.ts';
import type { ImpulseState } from '../sim/impulses.ts';

/**
 * One sim's θ registry, precomputed once.
 *
 * Five parallel views of the same slot order, because the runtime indexes by
 * slot in five different hot places (mask test, spec lookup, class lookup, name
 * lookup, length) and rebuilding any of them per tick would be waste. The sim
 * builds this once and hands out the same object.
 */
export interface ThetaRegistry {
  readonly length: number;
  readonly slots: readonly (ModSpec | null)[];
  readonly mask: Uint8Array;
  readonly classes: Uint8Array;
  readonly names: readonly string[];
  /**
   * Slots this sim draws wholesale from the seed, applied AFTER the generic
   * personality jitter; the generic path handles slots it does not touch.
   *
   * Optional because it is a property of the substrate, not of the mapping
   * layer: particle life draws its whole interaction matrix per seed (a jittered
   * constant matrix makes every seed the same creature), physarum does not and
   * leaves this undefined.
   */
  readonly seedBase?: (seed: number, base: Float64Array) => void;
}

/**
 * The config surface every modulatable sim exposes (structural).
 *
 * `species` is the narrowest thing the mapping layer and the workbench actually
 * read off a species — its name, for labels, and its brightness, for the
 * stem-follow readout. A sim's own `SpeciesConfig` is free to carry anything
 * else it likes; it satisfies this by having those two fields.
 */
export interface ModTargetConfig {
  speciesCount: number;
  palette: Palette;
  render: RenderConfig;
  species: { name: string; brightness: number }[];
}

/**
 * What the mapping layer and workbench need from a sim. `PhysarumSim`
 * implements this.
 *
 * Everything here already existed on PhysarumSim — this is a description of the
 * coupling that was there, not a new contract invented for it. The one genuinely
 * new member is `simId`, which exists because saved mappings are per sim: a
 * physarum `modulation.json` describes physarum's θ and must not be applied to
 * a different substrate that happens to have the same species count.
 */
export interface ModTarget {
  /** persistence discriminator: 'physarum' | 'plife' | … */
  readonly simId: string;
  readonly config: ModTargetConfig;
  readonly currentSeed: number;
  onSeedChange: ((seed: number) => void) | null;
  registry(): ThetaRegistry;
  /** θ as the live config currently has it. */
  currentVector(): Float64Array;
  /** Write θ (masked) into the live config and push any GPU state derived from it. */
  applyTheta(v: ArrayLike<number>, mask?: Uint8Array): void;
  invalidatePalette(): void;
  setBrightFollow(values: Float32Array | null): void;
  setImpulses(state: ImpulseState | null): void;
  /** species index → stem index for the stem-follow lane, -1 = not keyed */
  stemMap(): Int32Array;
  /**
   * The sim's own out-of-θ state, as an opaque block for `ModulationConfig.extras`.
   *
   * The mapping layer carries this and never inspects it: plife puts its macro
   * rig and its matrix-generation settings here, physarum its macro rig. That
   * is the whole point of the seam — `ModulationConfig` never learns any
   * substrate's schema, and a sim with nothing to save leaves both undefined.
   */
  serializeExtras?(): Record<string, unknown>;
  /**
   * The inverse, run on load. **Must tolerate `undefined` and garbage** — the
   * block is opaque, so nothing upstream has validated it — by clamping what it
   * can and falling back to defaults for the rest. It must never throw: a broken
   * extras block costs its own values, not the whole file.
   */
  applyExtras?(raw: Record<string, unknown> | undefined): void;
  partialReseed(key: number, fraction: number): void;
  reseed(seed: number): void;
  snapshot(): boolean;
  restoreSnapshot(): boolean;
  clearSnapshot(): void;
  readonly hasSnapshot: boolean;
}
