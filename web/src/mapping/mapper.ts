/**
 * The mapping layer, at runtime.
 *
 *   z (latent, 10 Hz)  ──►  softmax over anchors  ──►  θ = Σ wₘ θₘ
 *                                                        │
 *                                     per-class slew limiter (fast/medium/slow)
 *                                                        │
 *                                             PhysarumConfig, every sim tick
 *
 * Stems do not pass through here: they keep their direct path into deposit
 * (config.stemDrive), because the simplex handles character and stems handle
 * "an instrument just came in".
 *
 * Modes, and why:
 *
 *   manual  — the mapper computes weights (so the workbench can show them) but
 *             writes nothing. The sliders rule. This is phase-4 behaviour.
 *   mapped  — the mapper writes θ into the live config every tick. The sliders
 *             still show the live values because they are bound to that same
 *             object; edits survive until the next tick, i.e. they are read-only
 *             in practice. Tuning is done through *solo* instead.
 *   solo    — one anchor is applied exactly (temperature → 0) and the mapper
 *             then *holds*: no further writes, so the sliders are live again and
 *             edit that anchor's look directly. "capture" stores them back into
 *             the anchor. Leaving solo resumes blending from wherever the config
 *             now sits, so there is no jump.
 */
import type { PhysarumSim } from '../sim/physarum/physarum.ts';
import { mergeRenderConfig } from '../sim/render/config.ts';
import type { FeaturesFrame, TimelineSampler } from '../timeline/sampler.ts';
import {
  applyPreset,
  applyVector,
  blendVectors,
  CLASS_SLOW,
  fieldClasses,
  presetFromConfig,
  presetToVector,
  vectorLength,
  type Preset,
} from './preset.ts';
import { softmaxWeights } from './simplex.ts';
import { SlewLimiter } from './slew.ts';
import type { MappingConfig } from './types.ts';

export type MappingMode = 'manual' | 'mapped';

export interface MapperEvents {
  onSectionChange?: (segmentIndex: number, respawned: number) => void;
}

export class Mapper {
  mode: MappingMode = 'manual';

  private readonly sim: PhysarumSim;
  private readonly sampler: TimelineSampler;
  private readonly events: MapperEvents;

  private cfg!: MappingConfig;
  private vectors: Float64Array[] = [];
  private centers: number[][] = [];
  private target!: Float64Array;
  private slew!: SlewLimiter;
  private weightBuf = new Float64Array(0);
  private distBuf = new Float64Array(0);

  private soloIndex: number | null = null;
  private lastSegment = -2;
  private zView: Float32Array | null = null;

  constructor(
    sim: PhysarumSim,
    sampler: TimelineSampler,
    config: MappingConfig,
    events: MapperEvents = {},
  ) {
    this.sim = sim;
    this.sampler = sampler;
    this.events = events;
    this.setConfig(config);
  }

  get config(): MappingConfig {
    return this.cfg;
  }

  /** Weights of the last update, one per anchor. Read-only view for the workbench. */
  get weights(): Float64Array {
    return this.weightBuf;
  }

  /** ‖z − cₘ‖² of the last update. */
  get distances(): Float64Array {
    return this.distBuf;
  }

  get solo(): number | null {
    return this.soloIndex;
  }

  /**
   * False when the timeline has no latent channel or the config has no anchors —
   * mapped mode is then unavailable rather than silently blending nothing.
   */
  get available(): boolean {
    const c = this.sampler.getChannel('latent');
    return this.cfg.anchors.length > 0 && c !== undefined && c.dims > 0;
  }

  get unavailableReason(): string {
    if (!this.sampler.getChannel('latent')) return 'timeline has no "latent" channel';
    if (this.cfg.anchors.length === 0) return 'mapping has no anchors';
    return '';
  }

  setConfig(config: MappingConfig): void {
    this.cfg = config;
    const k = this.sim.config.speciesCount;
    // The palette is one object shared with the live config: copy the loaded
    // values *into* it (so the panel's colour pickers keep their bindings) and
    // then re-share, so later edits land in the thing that gets serialised.
    const live = this.sim.config.palette;
    live.colors = live.colors.map((c, i) => config.palette.colors[i] ?? c);
    live.saturation = config.palette.saturation;
    live.brightness = config.palette.brightness;
    config.palette = live;
    // Same trick for the phase-7 render block: the panel's widgets and PostFx
    // both hold the live object by reference, so a load mutates it in place.
    mergeRenderConfig(this.sim.config.render, config.render);
    config.render = this.sim.config.render;
    // Both of those just rewrote colour strings the sim caches in linear form.
    this.sim.invalidatePalette();
    const len = vectorLength(k);
    this.vectors = config.anchors.map((a) => presetToVector(a.preset, k));
    this.centers = config.anchors.map((a) => a.center);
    this.target = new Float64Array(len);
    this.weightBuf = new Float64Array(config.anchors.length);
    this.distBuf = new Float64Array(config.anchors.length);
    this.slew = new SlewLimiter(fieldClasses(k), config.slew);
    this.slew.reset(presetToVector(presetFromConfig(this.sim.config), k));
    this.soloIndex = null;
    this.lastSegment = -2;
  }

  /** Re-read slew rates after the panel edits them. */
  syncRates(): void {
    this.slew.setRates(this.cfg.slew);
  }

  setMode(mode: MappingMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    // Start from where the config actually is, so switching modes never steps.
    this.slew.reset(presetToVector(presetFromConfig(this.sim.config), this.sim.config.speciesCount));
    this.lastSegment = -2;
  }

  /** Load an anchor's preset into the live config exactly and hold there. */
  setSolo(index: number | null): void {
    if (index === null) {
      this.soloIndex = null;
      this.slew.reset(
        presetToVector(presetFromConfig(this.sim.config), this.sim.config.speciesCount),
      );
      return;
    }
    const anchor = this.cfg.anchors[index];
    if (!anchor) return;
    this.soloIndex = index;
    this.mode = 'mapped';
    this.applyAnchor(index);
  }

  applyAnchor(index: number): void {
    const anchor = this.cfg.anchors[index];
    const vec = this.vectors[index];
    if (!anchor || !vec) return;
    applyPreset(this.sim.config, anchor.preset);
    this.sim.uploadMatrix();
    this.slew.reset(vec);
  }

  /** "capture current params into this anchor's preset". */
  captureInto(index: number): Preset | null {
    const anchor = this.cfg.anchors[index];
    if (!anchor) return null;
    anchor.preset = presetFromConfig(this.sim.config);
    this.vectors[index] = presetToVector(anchor.preset, this.sim.config.speciesCount);
    return anchor.preset;
  }

  /** θ as the sim currently has it — what the tuning log records. */
  currentTheta(): number[] {
    return Array.from(presetToVector(presetFromConfig(this.sim.config), this.sim.config.speciesCount));
  }

  /** The latent vector of the last update, copied. */
  currentZ(): number[] {
    return this.zView ? Array.from(this.zView) : [];
  }

  /**
   * One tick. `dt` is the fixed sim timestep, so slew behaviour does not depend
   * on frame rate or on how many ticks a frame drains.
   */
  update(frame: FeaturesFrame, dt: number): void {
    const channel = this.sampler.getChannel('latent');
    if (!channel || this.cfg.anchors.length === 0) return;

    const z = frame.values.subarray(channel.offset, channel.offset + channel.dims);
    this.zView = z;

    if (this.soloIndex !== null) {
      this.weightBuf.fill(0);
      this.weightBuf[this.soloIndex] = 1;
    } else {
      const t = Math.max(this.cfg.temperature, 0) * Math.max(this.cfg.distanceScale, 1e-9);
      softmaxWeights(z, this.centers, t, this.weightBuf, this.distBuf);
    }

    if (this.mode !== 'mapped') return;
    // Solo holds: the sliders own the config until solo is cleared.
    if (this.soloIndex !== null) return;

    blendVectors(this.vectors, this.weightBuf, this.target);

    if (this.cfg.boundary.enabled) {
      const seg = this.sampler.segmentIndexAt(frame.time);
      if (seg >= 0 && seg !== this.lastSegment) {
        if (this.lastSegment !== -2) {
          this.slew.snapClass(CLASS_SLOW, this.target, this.cfg.boundary.snapFraction);
          this.sim.partialReseed(seg, this.cfg.boundary.respawnFraction);
          this.events.onSectionChange?.(seg, this.cfg.boundary.respawnFraction);
        }
        this.lastSegment = seg;
      }
    }

    const v = this.slew.step(this.target, dt);
    applyVector(this.sim.config, v);
    this.sim.uploadMatrix();
  }

  /** ‖z − cₘ‖² for one anchor, from the last update. */
  distanceTo(index: number): number {
    return this.distBuf[index] ?? 0;
  }
}
