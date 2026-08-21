import type { GpuRuntimeContext } from '../../gpu/runtime-context';
import type { ModTarget, ThetaRegistry } from '../../mapping/target';
import {
  applyVector,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  presetFromConfig,
  presetToVector,
  vectorLength,
} from '../../mapping/preset';
import type { FeaturesFrame } from '../../timeline/sampler';
import { LEGACY_MODEL_DT } from '../../timing';
import { MAX_SPLASHES, type ImpulseState } from '../impulses';
import { hexToLinear, paletteHuePhase, paletteLinear } from '../palette';
import { HDR_FORMAT, PostFx } from '../render/postfx';
import type { RenderFrame, Sim } from '../types';
import { dtScale, smoothDtFrames } from '../frame-timing';
import {
  defaultConfig,
  defaultPaletteColor,
  MAX_BRIGHTNESS,
  MAX_DEPOSIT,
  MAX_EFFECTIVE_DEPOSIT,
  MAX_SPECIES_SCALE,
  MIN_SPECIES_SCALE,
  PHYSARUM_BLOCKS,
  type PhysarumConfig,
  type SpeciesConfig,
} from './config';
import { applyBlocks, serializeBlocks } from '../../mapping/blocks';

import commonWgsl from './shaders/common.wgsl?raw';
import agentsWgsl from './shaders/agents.wgsl?raw';
import resolveWgsl from './shaders/resolve.wgsl?raw';
import diffuseWgsl from './shaders/diffuse.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';

const AGENT_WORKGROUP = 64;
const GRID_WORKGROUP = 8;
const FLOATS_PER_SPECIES = 24;
/** must match the Splash struct in common.wgsl: two vec4f */
const FLOATS_PER_SPLASH = 8;
/** must match the Globals struct in common.wgsl, padding included */
const GLOBALS_WORDS = 28;
const STEM_DIMS = 4;

/**
 * A complete copy of the sim's persistent state on spare GPU resources.
 * trailB is deliberately absent: resolve rewrites every texel of it from
 * deposit + trailA before diffuse reads it, so it carries nothing across a step.
 */
interface SimSnapshot {
  agents: GPUBuffer;
  deposit: GPUBuffer;
  trail: GPUTexture;
  soil: GPUTexture;
  seed: number;
  /** the world's own elapsed seconds — the palette's hue cycle rides this */
  worldSeconds: number;
  lastPcgTick: number;
  /** wall-clock label for the workbench; not part of sim state */
  takenAt: number;
}

export interface PhysarumStats {
  gridW: number;
  gridH: number;
  totalAgents: number;
  aliveAgents: number;
  stepsThisFrame: number;
  /** render passes in the post chain last frame; the compositor is the +1 */
  renderPasses: number;
}

export class PhysarumSim implements Sim, ModTarget {
  readonly name = 'physarum';
  /**
   * The persistence discriminator (`ModTarget.simId`). Saved mappings are per
   * sim, so this is what keeps a physarum `modulation.json` from being applied
   * to a different substrate that happens to share its species count.
   */
  readonly simId = 'physarum';
  readonly config: PhysarumConfig;

  /**
   * Notified whenever the seed changes — a reseed, and equally a snapshot restore
   * putting an older seed back. Anything that keys its own hashing off the sim's
   * seed (the impulse engine's hotspots) has to follow it, or an A/B restore runs
   * the restored world against the newer seed's splash positions.
   */
  onSeedChange: ((seed: number) => void) | null = null;

  private ctx: GpuRuntimeContext | null = null;
  private ready = false;
  private seed: number;
  /** null = no usable stems channel; stem drive stays off and the stems read as 0 */
  private stemOffset: number | null = null;
  private readonly stems: Float32Array;

  private gridW = 0;
  private gridH = 0;
  private agentsPerSpecies = 0;
  private totalAgents = 0;

  /**
   * Live impulse lane, owned by the ImpulseEngine and mutated in place. It is read,
   * never written, and it is applied *after* whatever set the base parameters — the
   * modulator's slew limiter included — so transients are never smoothed away.
   */
  private impulses: ImpulseState | null = null;
  private splashCount = 0;

  /**
   * Per-species brightness multiplier from the stem-follow lane (Revision 4),
   * held by reference and mutated by its owner every tick, like `impulses`.
   * Null means the lane is not wired at all and brightness is absolute.
   */
  private brightFollow: Float32Array | null = null;

  private globalsBuf!: GPUBuffer;
  private speciesBuf!: GPUBuffer;
  private splashBuf!: GPUBuffer;
  private matrixBuf!: GPUBuffer;
  private agentBuf!: GPUBuffer;
  private depositBuf!: GPUBuffer;
  private trailA!: GPUTexture;
  private trailB!: GPUTexture;
  private trailAView!: GPUTextureView;
  private trailBView!: GPUTextureView;
  /** shared track-scale memory; one layer, trail resolution, read-write in place */
  private soil!: GPUTexture;
  private soilView!: GPUTextureView;

  private initPipeline!: GPUComputePipeline;
  private respawnPipeline!: GPUComputePipeline;
  private splashPipeline!: GPUComputePipeline;
  private stepPipeline!: GPUComputePipeline;
  private resolvePipeline!: GPUComputePipeline;
  private diffusePipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
  private clearSoilPipeline!: GPUComputePipeline;
  private compositePipeline!: GPURenderPipeline;

  private simBind!: GPUBindGroup;
  private resolveBind!: GPUBindGroup;
  private diffuseBind!: GPUBindGroup;
  /**
   * One per ping-pong parity: the compositor reads the *other* HDR surface as
   * the feedback source, so which bind group is correct alternates with the
   * frame. Rebuilt whenever PostFx reallocates (canvas resize).
   */
  private compositeBinds: GPUBindGroup[] = [];
  private compositeLayout!: GPUBindGroupLayout;
  private compositeSizeVersion = -1;

  /** The phase-7 render chain. Owns the HDR surfaces, bloom, grading, auto-exposure. */
  readonly post: PostFx;

  private readonly globalsBytes = new ArrayBuffer(GLOBALS_WORDS * 4);
  private readonly globalsU32 = new Uint32Array(this.globalsBytes);
  private readonly globalsF32 = new Float32Array(this.globalsBytes);
  /**
   * Linearised palette (3 floats per species) and soil tint. Both are static art
   * direction whose source of truth is a set of CSS colour strings, and both are
   * read from uploadSpecies()/writeGlobals() — which run on every sim substep and
   * again on every render. Re-parsing the strings there was pure waste, so they
   * are derived once and re-derived only when something says they changed.
   */
  private readonly paletteRgb: Float32Array;
  private readonly soilTint = new Float32Array(3);
  private paletteDirty = true;
  private speciesData!: Float32Array<ArrayBuffer>;
  private matrixData!: Float32Array<ArrayBuffer>;
  private readonly splashData = new Float32Array(MAX_SPLASHES * FLOATS_PER_SPLASH);

  /**
   * Render-frame length expressed in 60 Hz frames for the feedback lane,
   * smoothed — see `smoothDtFrames`. Seeded at the 60 Hz value and snapped to
   * the first real measurement by `dtFramesPrimed`.
   */
  private renderDtFrames = 1;
  private dtFramesPrimed = false;

  /**
   * Seconds of world time since the world began — this substrate's clock.
   *
   * The palette's autonomous hue cycle needs an absolute time that is *the
   * simulation's*, not the wall's. Pausing advances it by nothing, so the cycle
   * stops with the picture; a snapshot carries it, so restoring a world restores
   * its colour phase; a headless export starts it at zero and replays the
   * identical trajectory.
   */
  private worldSeconds = 0;
  /**
   * This frame's step length in units of the model's own 1/60 s reference step.
   *
   * Physarum is a per-step model: `moveDist` is cells per step, `rotate` is
   * radians per step, `decay` is a factor per step, deposit is an amount per
   * step. With one variable-length step per rendered frame, every one of those
   * has to be re-expressed for the length of the step actually being taken, or
   * the world would run at the refresh rate — 2.4× on a 144 Hz panel — and every
   * authored preset would mean something different per machine.
   *
   * 1.0 on a 60 Hz display, where it is an exact float identity and the uploaded
   * uniforms are bit-identical to the ones the fixed 60 Hz cadence produced.
   * Held at its last live value while paused rather than dropping to 0, because
   * `uploadSpecies` also runs from `render`.
   */
  private stepScale = 1;
  private pendingSingleStep = false;
  private lastPcgTick = 0;
  private stepsThisFrame = 0;
  private respawnFraction = 0;
  private respawnKey = 0;
  private snap: SimSnapshot | null = null;

  /** θ registry and stem keying, both pure functions of K — built once, never rebuilt. */
  private theta: ThetaRegistry | null = null;
  private stems2stem: Int32Array | null = null;

  constructor(seed: number, config: PhysarumConfig = defaultConfig()) {
    this.seed = seed >>> 0;
    this.config = config;
    this.stems = new Float32Array(config.speciesCount);
    this.paletteRgb = new Float32Array(config.speciesCount * 3);
    this.post = new PostFx(config.render);
  }

  /**
   * Tell the sim its colours changed. Everything that edits `config.palette` or
   * `render.grade.soilColor` — the panel's pickers, a mapping load — has to call
   * this; nothing polls, because the palette is static by design and a poll would
   * put the string parsing back in the hot path it was taken out of.
   */
  invalidatePalette(): void {
    this.paletteDirty = true;
  }

  /** Sim time in seconds — the argument the palette's hue cycle is a function of. */
  get simSeconds(): number {
    return this.worldSeconds;
  }

  private refreshPalette(): void {
    this.paletteDirty = false;
    const phase = paletteHuePhase(this.config.palette, this.simSeconds);
    for (let k = 0; k < this.config.speciesCount; k++) {
      // The fallback is passed explicitly: `paletteLinear` is sim-agnostic now
      // and cannot know physarum's authored hue walk, which is what a palette
      // shorter than K should fall back to.
      const [r, g, b] = paletteLinear(
        this.config.palette,
        k,
        defaultPaletteColor(k),
        phase,
        this.config.speciesCount,
      );
      this.paletteRgb[k * 3 + 0] = r;
      this.paletteRgb[k * 3 + 1] = g;
      this.paletteRgb[k * 3 + 2] = b;
    }
    this.soilTint.set(hexToLinear(this.config.render.grade.soilColor));
  }

  get currentSeed(): number {
    return this.seed;
  }

  /** The only place `seed` is assigned after construction, so listeners cannot be missed. */
  private setSeed(seed: number): void {
    this.seed = seed >>> 0;
    this.onSeedChange?.(this.seed);
  }

  /**
   * Bind the timeline's stems channel. A missing or too-narrow channel disables stem
   * drive rather than reading whatever floats sit at another channel's offset.
   */
  setStemChannel(channel: { offset: number; dims: number } | undefined): void {
    if (!channel || channel.dims < STEM_DIMS) {
      this.stemOffset = null;
      this.stems.fill(0);
      this.config.stemDrive = false;
      console.warn(
        `physarum: no usable stems channel (${
          channel ? `${channel.dims} dims, need ${STEM_DIMS}` : 'absent'
        }); stem drive disabled`,
      );
      return;
    }
    this.stemOffset = channel.offset;
  }

  /**
   * Hand the sim the impulse engine's live state object. Passing null (or never
   * calling this) leaves every multiplier at 1 and skips the splash pass entirely,
   * which is exactly what a timeline with no `events` array should do.
   */
  setImpulses(state: ImpulseState | null): void {
    this.impulses = state;
  }

  /**
   * Hand the sim the stem-follow lane's live multiplier array (length K). Held
   * by reference: the owner writes it every tick and this never copies.
   */
  setBrightFollow(values: Float32Array | null): void {
    this.brightFollow = values;
  }

  // ── ModTarget: θ, as the mapping layer sees it ─────────────────────────────
  //
  // This block is the whole of physarum's side of the seam. It is thin on
  // purpose: the registry and the vector↔config conversion already lived in
  // `mapping/preset.ts`, and all that changed is *who asks whom*. The modulator
  // used to import preset.ts and reach into `sim.config`; now the sim publishes
  // its own registry and the modulator holds a `ModTarget`.

  /**
   * The slot table, precomputed. Every view is a pure function of K, which is
   * fixed for the life of the sim, so this is built on first ask and then handed
   * out unchanged — the modulator reads `mask` and `slots` in its per-tick loop.
   */
  registry(): ThetaRegistry {
    if (this.theta) return this.theta;
    const k = this.config.speciesCount;
    this.theta = {
      length: vectorLength(k),
      slots: modulationSlots(k),
      mask: modulationMask(k),
      classes: fieldClasses(k),
      names: fieldNames(k),
    };
    return this.theta;
  }

  currentVector(): Float64Array {
    return presetToVector(presetFromConfig(this.config), this.config.speciesCount);
  }

  /**
   * θ → the live config, plus the one piece of GPU state derived from θ rather
   * than re-read from it every step. Everything else in the vector is picked up
   * by `uploadSpecies` on the next substep; M lives in its own buffer and has to
   * be pushed, which is why the modulator used to have to remember to call
   * `uploadMatrix` after every write. It no longer does — that pairing is this
   * method's job, and a sim whose θ needs different GPU work does it here too.
   */
  applyTheta(v: ArrayLike<number>, mask?: Uint8Array): void {
    applyVector(this.config, v, mask);
    this.uploadMatrix();
  }

  /**
   * Species k is keyed to stem k for the first four species and to nothing after
   * that — the same keying as stem drive in `tick`, because the stems channel is
   * 4-dim by contract (STEM_DIMS) and species 4+ have no instrument of their own.
   *
   * Returned as data rather than baked into StemFollow because live-mode-notes
   * wants "which species follows which sound source" to become configuration; a
   * sim that keys differently (or a future UI that re-keys by hand) writes a
   * different array here and nothing else changes.
   */
  stemMap(): Int32Array {
    if (this.stems2stem) return this.stems2stem;
    const k = this.config.speciesCount;
    const map = new Int32Array(k);
    for (let i = 0; i < k; i++) map[i] = i < STEM_DIMS ? i : -1;
    this.stems2stem = map;
    return map;
  }

  // ── ModTarget: the opaque per-sim extras block ─────────────────────────────
  //
  // Three parts of physarum's config are outside θ *and* outside everything the
  // mapping layer knows how to carry: the macro rig, the soil block, and the
  // direct stems→deposit path. None of them belongs in `ModulationConfig`'s
  // schema — that file describes a mapping, not a substrate — so they travel in
  // the opaque `extras` channel and this pair is the only code that understands
  // their shape.
  //
  // Unlike plife there is no `matrixGen` block here: physarum's M is authored,
  // not drawn from the seed.
  //
  // The channel has two consumers, and the second is why `soil` and `stemDrive`
  // are here at all. The first is persistence (autosave / `modulation.json`).
  // The second is explorer mode: `ExplorerRig.syncStyle` fans the serialised
  // block out to all nine tiles twice a second, so a panel edit to something
  // outside both θ and extras is an edit the tiles never see — which is exactly
  // how the soil sliders came to do nothing while the grid was open.

  /**
   * A plain snapshot of everything physarum wants saved outside θ.
   *
   * The blocks come from `PHYSARUM_BLOCKS`, which is exhaustive over
   * `PhysarumConfig`'s own object-valued keys — so a block saves because it was
   * *declared*, not because it was remembered here (roadmap phase 1 item 5).
   * `soil.debugView` is absent because the declaration says it is session-only,
   * which is also where the reason is written. What is left below is the state
   * that is not a block: two scalars on the config root, and the look pair.
   */
  serializeExtras(): Record<string, unknown> {
    return {
      ...serializeBlocks(this.config, PHYSARUM_BLOCKS),
      stemDrive: this.config.stemDrive,
      stemGain: this.config.stemGain,
      // θ, but the two slots excluded from modulation and owned by the look tab.
      // Saved here because nothing else persists θ, and a reload that reset the
      // exposure you just dialled in is indistinguishable from a bug.
      look: { exposure: this.config.exposure, gamma: this.config.gamma },
    };
  }

  /**
   * The inverse, and deliberately paranoid: `extras` is opaque to every layer
   * between the file and here, so nothing upstream has validated it. Every field
   * is clamped into the range its slider shows, anything missing or non-finite
   * falls back to the shipped default, and this never throws.
   *
   * Runs on load (`Modulator.setConfig`) and, nine times over, on every explorer
   * style sync. Both callers depend on it writing **in place**: the panel's
   * tweakpane bindings hold `config.soil` by reference, so replacing the object
   * would leave every soil slider bound to an orphan. `applyBlocks` guarantees
   * that, and it walks each block's own live keys, so a field added to
   * `SoilConfig` or `PhysarumMacros` round trips the moment its defaults function
   * knows about it — no reader to update, and no way to half-update one.
   *
   * `soil.debugView` is lifted over both walks (the reset and the read) by its
   * `sessionOnlyFields` declaration, so whatever the panel currently has stays.
   */
  applyExtras(raw: Record<string, unknown> | undefined): void {
    const o = (raw ?? {}) as Record<string, unknown>;
    applyBlocks(this.config, PHYSARUM_BLOCKS, o);

    // Defaults stated as literals rather than pulled from `defaultConfig()`:
    // this runs nine times a sync and `defaultConfig` allocates a whole K²
    // matrix and a species array to answer two scalars.
    this.config.stemDrive = readBool(o['stemDrive'], false);
    this.config.stemGain = clampNum(o['stemGain'], 1.5, 0, 6);

    // Scene exposure / gamma. Bounds are the panel's own (ui/panel.ts's
    // `exposureRange` and the gamma slider); absent keeps what the preset gave
    // us, so a file from before this block loads unchanged.
    const lk = plainObject(o['look']);
    if (lk['exposure'] !== undefined) {
      this.config.exposure = clampNum(lk['exposure'], this.config.exposure, 0.005, 1.5);
    }
    if (lk['gamma'] !== undefined) {
      this.config.gamma = clampNum(lk['gamma'], this.config.gamma, 1, 3);
    }
  }

  /**
   * The alive fraction actually in force for one species: θ's base scaled by the
   * `density` macro and clamped back into 0..1. Two callers — `uploadSpecies`,
   * which writes it to the GPU, and `stats`, which recomputes the agent count
   * from the config — and they must agree, or the readout describes a world that
   * is not running.
   */
  private effectiveAlive(s: SpeciesConfig): number {
    return Math.min(
      Math.max(s.aliveFraction, 0) * Math.max(this.config.macros.density, 0),
      1,
    );
  }

  stats(): PhysarumStats {
    let alive = 0;
    for (const s of this.config.species) {
      alive += Math.floor(this.effectiveAlive(s) * this.agentsPerSpecies);
    }
    return {
      gridW: this.gridW,
      gridH: this.gridH,
      totalAgents: this.totalAgents,
      aliveAgents: alive,
      stepsThisFrame: this.stepsThisFrame,
      renderPasses: this.post.passCount + 1,
    };
  }

  /**
   * The sim-specific middle of the app's status line (`Sim.status`). This is
   * exactly the string main.ts used to interpolate by hand; it moved here when a
   * second substrate arrived with a different set of numbers to report.
   */
  status(): string {
    const st = this.stats();
    return (
      `${st.gridW}×${st.gridH}×${this.config.speciesCount} · ` +
      `${st.aliveAgents.toLocaleString()} agents · seed ${this.seed}`
    );
  }

  requestSingleStep(): void {
    this.pendingSingleStep = true;
  }

  async init(ctx: GpuRuntimeContext): Promise<void> {
    this.ctx = ctx;
    const { device } = ctx;
    const k = this.config.speciesCount;

    if (k > device.limits.maxTextureArrayLayers) {
      throw new Error(
        `physarum: speciesCount ${k} exceeds maxTextureArrayLayers ${device.limits.maxTextureArrayLayers}`,
      );
    }

    const grid = this.chooseGrid(ctx);
    this.gridW = grid.w;
    this.gridH = grid.h;
    this.agentsPerSpecies = Math.max(AGENT_WORKGROUP, Math.floor(this.config.maxAgents / k));
    this.totalAgents = this.agentsPerSpecies * k;

    this.speciesData = new Float32Array(k * FLOATS_PER_SPECIES);
    this.matrixData = new Float32Array(k * k);

    this.globalsBuf = device.createBuffer({
      label: 'physarum.globals',
      size: GLOBALS_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.speciesBuf = device.createBuffer({
      label: 'physarum.species',
      size: this.speciesData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.matrixBuf = device.createBuffer({
      label: 'physarum.matrix',
      size: Math.max(this.matrixData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.splashBuf = device.createBuffer({
      label: 'physarum.splashes',
      size: this.splashData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // COPY_SRC/COPY_DST on the state-carrying resources exist for the workbench's
    // snapshot/restore; they cost nothing when unused.
    this.agentBuf = device.createBuffer({
      label: 'physarum.agents',
      size: this.totalAgents * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.depositBuf = device.createBuffer({
      label: 'physarum.deposit',
      size: this.gridW * this.gridH * k * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const trailDesc: GPUTextureDescriptor = {
      size: { width: this.gridW, height: this.gridH, depthOrArrayLayers: k },
      format: 'r32float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    };
    this.trailA = device.createTexture({ ...trailDesc, label: 'physarum.trailA' });
    this.trailB = device.createTexture({ ...trailDesc, label: 'physarum.trailB' });
    this.trailAView = this.trailA.createView({ dimension: '2d-array' });
    this.trailBView = this.trailB.createView({ dimension: '2d-array' });

    // One shared layer at trail resolution — see SoilConfig for why shared rather
    // than per-species. STORAGE_BINDING is used read_write by the diffuse pass and
    // TEXTURE_BINDING read-only by agents and the compositor.
    this.soil = device.createTexture({
      label: 'physarum.soil',
      size: { width: this.gridW, height: this.gridH },
      format: 'r32float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    this.soilView = this.soil.createView();

    const src = (body: string): GPUShaderModule =>
      device.createShaderModule({ code: `${commonWgsl}\n${body}` });

    // r32float is unfilterable in core WebGPU; every read is a textureLoad, so no
    // sampler exists anywhere in this sim and float32-filterable is never required.
    const trailTex: GPUTextureBindingLayout = {
      sampleType: 'unfilterable-float',
      viewDimension: '2d-array',
    };
    const trailStore: GPUStorageTextureBindingLayout = {
      access: 'write-only',
      format: 'r32float',
      viewDimension: '2d-array',
    };
    const soilTex: GPUTextureBindingLayout = {
      sampleType: 'unfilterable-float',
      viewDimension: '2d',
    };
    const soilStore: GPUStorageTextureBindingLayout = {
      access: 'read-write',
      format: 'r32float',
      viewDimension: '2d',
    };
    const C = GPUShaderStage.COMPUTE;

    const simLayout = device.createBindGroupLayout({
      label: 'physarum.simLayout',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: C, buffer: { type: 'storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, texture: trailTex },
        { binding: 6, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: C, texture: soilTex },
      ],
    });
    const resolveLayout = device.createBindGroupLayout({
      label: 'physarum.resolveLayout',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
        { binding: 2, visibility: C, texture: trailTex },
        { binding: 3, visibility: C, storageTexture: trailStore },
      ],
    });
    const diffuseLayout = device.createBindGroupLayout({
      label: 'physarum.diffuseLayout',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: C, texture: trailTex },
        { binding: 3, visibility: C, storageTexture: trailStore },
        { binding: 4, visibility: C, storageTexture: soilStore },
      ],
    });
    const F = GPUShaderStage.FRAGMENT;
    this.compositeLayout = device.createBindGroupLayout({
      label: 'physarum.compositeLayout',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: F, texture: trailTex },
        { binding: 3, visibility: F, texture: soilTex },
        // auto-exposure gain, produced by the post chain one frame ago
        { binding: 4, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: F, sampler: { type: 'filtering' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });

    const agentsModule = src(agentsWgsl);
    const diffuseModule = src(diffuseWgsl);
    const simPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
    const diffusePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [diffuseLayout] });

    this.initPipeline = device.createComputePipeline({
      label: 'physarum.init',
      layout: simPipelineLayout,
      compute: { module: agentsModule, entryPoint: 'initAgents' },
    });
    this.respawnPipeline = device.createComputePipeline({
      label: 'physarum.respawn',
      layout: simPipelineLayout,
      compute: { module: agentsModule, entryPoint: 'respawnAgents' },
    });
    this.splashPipeline = device.createComputePipeline({
      label: 'physarum.splash',
      layout: simPipelineLayout,
      compute: { module: agentsModule, entryPoint: 'splashAgents' },
    });
    this.stepPipeline = device.createComputePipeline({
      label: 'physarum.step',
      layout: simPipelineLayout,
      compute: { module: agentsModule, entryPoint: 'stepAgents' },
    });
    this.resolvePipeline = device.createComputePipeline({
      label: 'physarum.resolve',
      layout: device.createPipelineLayout({ bindGroupLayouts: [resolveLayout] }),
      compute: { module: src(resolveWgsl), entryPoint: 'resolveDeposit' },
    });
    this.diffusePipeline = device.createComputePipeline({
      label: 'physarum.diffuse',
      layout: diffusePipelineLayout,
      compute: { module: diffuseModule, entryPoint: 'diffuseDecay' },
    });
    this.clearPipeline = device.createComputePipeline({
      label: 'physarum.clear',
      layout: diffusePipelineLayout,
      compute: { module: diffuseModule, entryPoint: 'clearTrail' },
    });
    this.clearSoilPipeline = device.createComputePipeline({
      label: 'physarum.clearSoil',
      layout: diffusePipelineLayout,
      compute: { module: diffuseModule, entryPoint: 'clearSoil' },
    });

    // The compositor's target is the HDR surface, not the swapchain: phase 7
    // moved tone mapping and the display transfer function into the post chain.
    const compositeModule = src(compositeWgsl);
    this.compositePipeline = device.createRenderPipeline({
      label: 'physarum.composite',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }),
      vertex: { module: compositeModule, entryPoint: 'vsMain' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fsMain',
        targets: [{ format: HDR_FORMAT }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.post.init(ctx);
    this.post.ensureSize(ctx.width, ctx.height);

    // trailA is the stable state read by agents and the compositor; trailB is
    // scratch, fully rewritten by resolve every tick. Roles never swap, so there
    // is no ping-pong bookkeeping and no read/write aliasing within a dispatch.
    this.simBind = device.createBindGroup({
      layout: simLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuf } },
        { binding: 1, resource: { buffer: this.speciesBuf } },
        { binding: 2, resource: { buffer: this.matrixBuf } },
        { binding: 3, resource: { buffer: this.agentBuf } },
        { binding: 4, resource: { buffer: this.depositBuf } },
        { binding: 5, resource: this.trailAView },
        { binding: 6, resource: { buffer: this.splashBuf } },
        { binding: 7, resource: this.soilView },
      ],
    });
    this.resolveBind = device.createBindGroup({
      layout: resolveLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuf } },
        { binding: 1, resource: { buffer: this.depositBuf } },
        { binding: 2, resource: this.trailAView },
        { binding: 3, resource: this.trailBView },
      ],
    });
    this.diffuseBind = device.createBindGroup({
      layout: diffuseLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuf } },
        { binding: 1, resource: { buffer: this.speciesBuf } },
        { binding: 2, resource: this.trailBView },
        { binding: 3, resource: this.trailAView },
        { binding: 4, resource: this.soilView },
      ],
    });
    this.rebuildCompositeBinds();

    this.ready = true;
    this.uploadMatrix();
    this.reseed(this.seed);
  }

  /**
   * Fresh seed, empty trails, empty soil, agents re-scattered — unless
   * `keepWorld`, in which case only the *rules* change.
   *
   * ## keepWorld: new physics, same matter
   *
   * `setSeed` fires `onSeedChange`, which is the actual content of a reroll: the
   * impulse hotspots re-key and the modulator's seeded rewire restamps the
   * personality and the projection wiring. All of that still happens. What
   * `keepWorld` skips is everything that would destroy the accumulated world —
   * the agent scatter, the trail clear and the soil clear.
   *
   * The trail field is the point here, more so than in plife. Physarum's look
   * *is* its accumulated network, minutes of deposit in a field that a full
   * reseed wipes; keeping it means a reroll is a new personality moving into an
   * existing city rather than a new city. Soil (track-scale memory) stays for the
   * same reason the section-boundary `partialReseed` leaves it alone.
   *
   * Auto-exposure is not reset either: nothing went black, so there is nothing to
   * fade up from.
   *
   * ## Determinism
   *
   * A pinned seed still reproduces a run *from load* — a fresh load clears and
   * scatters, so (track, seed, device) determines the world it starts in.
   * Rerolling in place does not, and is not meant to: it is a live-performance
   * act taken against whatever the world happened to be at that instant, the
   * same doctrine as idle free-running (see the transport note in main.ts).
   */
  reseed(seed: number, opts?: { keepWorld?: boolean }): void {
    if (!this.ready || !this.ctx) return;
    const { device } = this.ctx;
    this.setSeed(seed);
    if (opts?.keepWorld === true) {
      // The seeded rewire has already landed (via `onSeedChange`) but it wrote
      // the *config*; these two push the parts of it the GPU caches — the
      // sensor/deposit matrix and the species block — without touching a single
      // agent, texel of trail or texel of soil. The pcg tick is kept rather than
      // rewound to 0: the noise stream belongs to the world being kept.
      this.uploadMatrix();
      this.uploadSpecies();
      this.writeGlobals(this.lastPcgTick);
      return;
    }
    // A new world starts its colour phase where a fresh load would, so a reseed
    // and a reload of the same recipe agree. `keepWorld` returned above and
    // therefore keeps its clock, which is the same rule the pcg tick follows.
    this.worldSeconds = 0;
    this.writeGlobals(0);
    this.uploadSpecies();
    // A new world starts from black; letting the adapted gain carry over would
    // make the first second of it a slow fade up from the old exposure.
    this.post.resetAutoExposure();

    const encoder = device.createCommandEncoder({ label: 'physarum.reseed' });
    encoder.clearBuffer(this.depositBuf);

    const clearPass = encoder.beginComputePass({ label: 'clearTrail' });
    clearPass.setPipeline(this.clearPipeline);
    clearPass.setBindGroup(0, this.diffuseBind);
    clearPass.dispatchWorkgroups(...this.gridDispatch());
    // A *full* reseed is "start a new world", so track-scale memory goes too. The
    // section-boundary partialReseed deliberately does not do this.
    clearPass.setPipeline(this.clearSoilPipeline);
    clearPass.dispatchWorkgroups(...this.soilDispatch());
    clearPass.end();

    const initPass = encoder.beginComputePass({ label: 'initAgents' });
    initPass.setPipeline(this.initPipeline);
    initPass.setBindGroup(0, this.simBind);
    initPass.dispatchWorkgroups(Math.ceil(this.totalAgents / AGENT_WORKGROUP));
    initPass.end();

    device.queue.submit([encoder.finish()]);
  }

  /**
   * Section-boundary re-seed: re-scatter `fraction` of the agent pool, chosen and
   * placed by hash(seed, segmentIndex, agentIndex). Trails and soil are untouched —
   * that is the whole point, new matter arriving into an old world.
   */
  partialReseed(segmentIndex: number, fraction: number): void {
    if (!this.ready || !this.ctx) return;
    const f = Math.min(Math.max(fraction, 0), 1);
    if (f <= 0) return;
    const { device } = this.ctx;

    this.respawnFraction = f;
    this.respawnKey = segmentIndex >>> 0;
    this.writeGlobals(this.lastPcgTick);

    const encoder = device.createCommandEncoder({ label: 'physarum.partialReseed' });
    const pass = encoder.beginComputePass({ label: 'respawnAgents' });
    pass.setPipeline(this.respawnPipeline);
    pass.setBindGroup(0, this.simBind);
    pass.dispatchWorkgroups(Math.ceil(this.totalAgents / AGENT_WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Leave the uniform clean: every other pass shares this Globals block.
    this.respawnFraction = 0;
    this.writeGlobals(this.lastPcgTick);
  }

  /** Wipe track-scale memory without disturbing anything else (workbench button). */
  clearSoil(): void {
    if (!this.ready || !this.ctx) return;
    const { device } = this.ctx;
    const encoder = device.createCommandEncoder({ label: 'physarum.clearSoil' });
    const pass = encoder.beginComputePass({ label: 'clearSoil' });
    pass.setPipeline(this.clearSoilPipeline);
    pass.setBindGroup(0, this.diffuseBind);
    pass.dispatchWorkgroups(...this.soilDispatch());
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  get hasSnapshot(): boolean {
    return this.snap !== null;
  }

  get snapshotAge(): number {
    return this.snap ? performance.now() - this.snap.takenAt : 0;
  }

  /**
   * Copy the whole persistent state onto spare GPU resources. Allocated on first
   * use (~2x the sim's own footprint) and reused after that.
   */
  snapshot(): boolean {
    if (!this.ready || !this.ctx) return false;
    const { device } = this.ctx;
    const k = this.config.speciesCount;

    if (!this.snap) {
      this.snap = {
        agents: device.createBuffer({
          label: 'physarum.snap.agents',
          size: this.totalAgents * 16,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        deposit: device.createBuffer({
          label: 'physarum.snap.deposit',
          size: this.gridW * this.gridH * k * 4,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        trail: device.createTexture({
          label: 'physarum.snap.trail',
          size: { width: this.gridW, height: this.gridH, depthOrArrayLayers: k },
          format: 'r32float',
          usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        }),
        soil: device.createTexture({
          label: 'physarum.snap.soil',
          size: { width: this.gridW, height: this.gridH },
          format: 'r32float',
          usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        }),
        seed: this.seed,
        worldSeconds: this.worldSeconds,
        lastPcgTick: this.lastPcgTick,
        takenAt: performance.now(),
      };
    }
    const snap = this.snap;
    snap.seed = this.seed;
    snap.worldSeconds = this.worldSeconds;
    snap.lastPcgTick = this.lastPcgTick;
    snap.takenAt = performance.now();

    const encoder = device.createCommandEncoder({ label: 'physarum.snapshot' });
    encoder.copyBufferToBuffer(this.agentBuf, 0, snap.agents, 0, this.agentBuf.size);
    encoder.copyBufferToBuffer(this.depositBuf, 0, snap.deposit, 0, this.depositBuf.size);
    encoder.copyTextureToTexture(
      { texture: this.trailA },
      { texture: snap.trail },
      { width: this.gridW, height: this.gridH, depthOrArrayLayers: k },
    );
    encoder.copyTextureToTexture(
      { texture: this.soil },
      { texture: snap.soil },
      { width: this.gridW, height: this.gridH, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Restore the last snapshot exactly. Parameters are *not* restored — that is the A/B. */
  restoreSnapshot(): boolean {
    if (!this.ready || !this.ctx || !this.snap) return false;
    const { device } = this.ctx;
    const snap = this.snap;
    const k = this.config.speciesCount;

    const encoder = device.createCommandEncoder({ label: 'physarum.restore' });
    encoder.copyBufferToBuffer(snap.agents, 0, this.agentBuf, 0, this.agentBuf.size);
    encoder.copyBufferToBuffer(snap.deposit, 0, this.depositBuf, 0, this.depositBuf.size);
    encoder.copyTextureToTexture(
      { texture: snap.trail },
      { texture: this.trailA },
      { width: this.gridW, height: this.gridH, depthOrArrayLayers: k },
    );
    encoder.copyTextureToTexture(
      { texture: snap.soil },
      { texture: this.soil },
      { width: this.gridW, height: this.gridH, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);

    this.setSeed(snap.seed);
    this.worldSeconds = snap.worldSeconds;
    // The hue phase is a function of the clock that was just rewound, so the
    // cached linearisation is now stale by however far the cycle had run.
    this.paletteDirty = true;
    this.lastPcgTick = snap.lastPcgTick;
    return true;
  }

  clearSnapshot(): void {
    if (!this.snap) return;
    this.snap.agents.destroy();
    this.snap.deposit.destroy();
    this.snap.trail.destroy();
    this.snap.soil.destroy();
    this.snap = null;
  }

  /** Re-upload M after the panel edits it. */
  uploadMatrix(): void {
    if (!this.ready || !this.ctx) return;
    const k = this.config.speciesCount;
    for (let i = 0; i < k * k; i++) this.matrixData[i] = this.config.matrix[i] ?? 0;
    this.ctx.device.queue.writeBuffer(this.matrixBuf, 0, this.matrixData);
  }

  /**
   * One rendered frame's worth of world: exactly one model step, stretched or
   * squeezed to cover `dt` seconds.
   *
   * There is no substep loop and no accumulator — `dt` arrives already clamped
   * by `MAX_FRAME_DT`, `speed` scales it into world time, and `stepScale`
   * carries the whole of it into the per-step quantities (see the field's note).
   * A slow machine therefore takes one longer, coarser step rather than several
   * short ones, and being late buys no extra GPU work.
   */
  tick(frame: FeaturesFrame, _stepIndex: number, dt: number): void {
    if (!this.ready || !this.ctx) return;
    void _stepIndex;

    // The stems channel is 4-dim by contract; species beyond 4 get no stem drive.
    const stemOffset = this.stemOffset;
    if (stemOffset !== null) {
      for (let k = 0; k < this.stems.length; k++) {
        this.stems[k] = k < STEM_DIMS ? (frame.values[stemOffset + k] ?? 0) : 0;
      }
    }

    let scale = 0;
    if (this.config.paused) {
      // A manual single step is worth one 60 Hz frame of world, so the button
      // means the same thing whatever the display is doing.
      if (this.pendingSingleStep) {
        this.pendingSingleStep = false;
        scale = 1;
      }
    } else {
      scale = dtScale(dt * Math.max(this.config.speed, 0), LEGACY_MODEL_DT);
    }

    this.stepsThisFrame = scale > 0 ? 1 : 0;
    if (scale <= 0) return;

    this.stepScale = scale;
    this.worldSeconds += scale * LEGACY_MODEL_DT;
    // Only when a cycle is actually running: a static palette must not be
    // re-linearised every frame, which is what the dirty flag exists to avoid.
    if (this.config.palette.hueRateDegPerSec !== 0) this.paletteDirty = true;
    this.uploadSplashes();
    // The pcg key is a plain per-step counter now rather than a function of a
    // transport tick. It never rewinds, which is all the agents' tie-break and
    // respawn hashes need; reproducing a given frame across runs is not a
    // property this loop has any more.
    this.runStep(this.lastPcgTick + 1);
  }

  /**
   * Impulse hotspots, normalised 0..1 → grid cells. Radius is a fraction of the
   * *short* axis so a disc stays a disc on a non-square grid.
   */
  private uploadSplashes(): void {
    const list = this.impulses?.splashes ?? [];
    const n = Math.min(list.length, MAX_SPLASHES);
    // One dead frame still has to be written, so the shader does not re-read a
    // stale disc; after that, an empty list costs nothing.
    if (n === 0 && this.splashCount === 0) return;

    const shortAxis = Math.min(this.gridW, this.gridH);
    for (let i = 0; i < n; i++) {
      const s = list[i];
      const o = i * FLOATS_PER_SPLASH;
      if (!s) continue;
      this.splashData[o + 0] = s.x * this.gridW;
      this.splashData[o + 1] = s.y * this.gridH;
      this.splashData[o + 2] = Math.max(s.radius * shortAxis, 1);
      this.splashData[o + 3] = s.strength;
      this.splashData[o + 4] = s.species;
      // Push (cells) and swirl (radians) are both applied once per step for the
      // length of the envelope's decay, so both are rates and both scale with
      // the step. Without this a splash would shove harder on a fast display.
      this.splashData[o + 5] = s.push * this.stepScale;
      this.splashData[o + 6] = s.swirl * this.stepScale;
      this.splashData[o + 7] = 0;
    }
    this.splashCount = n;
    if (n > 0) {
      (this.ctx as GpuRuntimeContext).device.queue.writeBuffer(
        this.splashBuf,
        0,
        this.splashData,
        0,
        n * FLOATS_PER_SPLASH,
      );
    }
  }

  private runStep(pcgTick: number): void {
    const { device } = this.ctx as GpuRuntimeContext;
    this.lastPcgTick = pcgTick;
    this.writeGlobals(pcgTick);
    this.uploadSpecies();

    const encoder = device.createCommandEncoder({ label: 'physarum.step' });
    const [gx, gy, gz] = this.gridDispatch();

    // Impulse lane, ahead of sensing: agents are displaced and re-aimed first, then
    // take their normal step from there, so a splash propagates through the sim
    // rather than sitting on top of the frame.
    if (this.splashCount > 0) {
      const splashPass = encoder.beginComputePass({ label: 'impulse splash' });
      splashPass.setPipeline(this.splashPipeline);
      splashPass.setBindGroup(0, this.simBind);
      splashPass.dispatchWorkgroups(Math.ceil(this.totalAgents / AGENT_WORKGROUP));
      splashPass.end();
    }

    // Separate passes rather than one pass with three dispatches: WebGPU's
    // per-dispatch usage scopes would allow the latter, but pass boundaries make
    // the trailA-read -> trailB-write -> trailA-write ordering unambiguous.
    const agentPass = encoder.beginComputePass({ label: 'sense/rotate/step/deposit' });
    agentPass.setPipeline(this.stepPipeline);
    agentPass.setBindGroup(0, this.simBind);
    agentPass.dispatchWorkgroups(Math.ceil(this.totalAgents / AGENT_WORKGROUP));
    agentPass.end();

    const resolvePass = encoder.beginComputePass({ label: 'resolve i32 -> r32float' });
    resolvePass.setPipeline(this.resolvePipeline);
    resolvePass.setBindGroup(0, this.resolveBind);
    resolvePass.dispatchWorkgroups(gx, gy, gz);
    resolvePass.end();

    const diffusePass = encoder.beginComputePass({ label: 'diffuse + decay' });
    diffusePass.setPipeline(this.diffusePipeline);
    diffusePass.setBindGroup(0, this.diffuseBind);
    diffusePass.dispatchWorkgroups(gx, gy, gz);
    diffusePass.end();

    device.queue.submit([encoder.finish()]);
  }

  render(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    frame: RenderFrame,
  ): void {
    if (!this.ready || !this.ctx) return;

    // Post surfaces follow the canvas, not the sim grid: a resize re-allocates
    // them (and invalidates the compositor's feedback binding) but never touches
    // accumulated trail state.
    this.post.ensureSize(this.ctx.width, this.ctx.height);
    if (this.post.sizeVersion !== this.compositeSizeVersion) this.rebuildCompositeBinds();
    this.post.gamma = this.config.gamma;

    // How long this frame is, in 60 Hz frames. The feedback lane is applied once
    // per *rendered* frame by the compositor, so both of its constants are
    // per-frame quantities, and both are silently wrong on any display that is
    // not 60 Hz: at 240 Hz the echo would decay four times too fast and the
    // radial zoom would accumulate four times as much. Raising both to the
    // dtFrames power makes the authored numbers mean "per 60 Hz frame" on every
    // monitor. Physarum ships feedback at 0, so this is a latent bug being
    // closed rather than one anybody has seen.
    //
    // The 0.25 s ceiling is for a tab that was backgrounded: a two-second gap
    // would otherwise raise the amount to the 120th power and clear the echo
    // outright, which is a black flash on the first frame back.
    //
    // The 1e-3 s floor is not cosmetic. Browser timestamps can be coarsened, so
    // two renders can report the same timestamp; that would make
    // the exponent 0, and `Math.pow(0, 0)` is **1**, which with physarum's
    // shipped `amount = 0` would switch the feedback lane fully on for one
    // frame. A frame is never zero seconds long, so the floor costs nothing and
    // keeps `pow(0, dtFrames)` at 0 for every reachable input.
    //
    // Both clamps now live in `smoothDtFrames`, which also smooths the
    // measurement — plife's flicker-on-pause was the unsmoothed version of this
    // exact line, and the two substrates share one feedback model.
    this.renderDtFrames = smoothDtFrames(
      this.renderDtFrames,
      frame.deltaSeconds,
      !this.dtFramesPrimed,
    );
    this.dtFramesPrimed = true;

    // This is the LAST Globals write before the composite pass, which is what
    // lets the substep path publish an uncorrected `renderDtFrames` harmlessly —
    // the only reader of words 25/26 is the compositor, and it runs after this.
    this.writeGlobals(this.lastPcgTick);
    this.uploadSpecies();

    const pass = encoder.beginRenderPass({
      label: 'physarum.composite',
      colorAttachments: [
        {
          view: this.post.hdrTargetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, this.compositeBinds[this.post.currentParity] as GPUBindGroup);
    pass.draw(3);
    pass.end();

    this.post.run(encoder, targetView, frame);
  }

  /**
   * Two bind groups, one per ping-pong parity, differing only in which HDR
   * surface is bound as the feedback source.
   */
  private rebuildCompositeBinds(): void {
    const { device } = this.ctx as GpuRuntimeContext;
    this.compositeBinds = [0, 1].map((parity) =>
      device.createBindGroup({
        label: `physarum.composite.parity${parity}`,
        layout: this.compositeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: { buffer: this.speciesBuf } },
          { binding: 2, resource: this.trailAView },
          { binding: 3, resource: this.soilView },
          { binding: 4, resource: { buffer: this.post.autoBuffer } },
          { binding: 5, resource: this.post.linearSampler },
          { binding: 6, resource: this.post.hdrViewAt(1 - parity) },
        ],
      }),
    );
    this.compositeSizeVersion = this.post.sizeVersion;
  }

  dispose(): void {
    if (!this.ready) return;
    this.ready = false;
    this.post.dispose();
    this.clearSnapshot();
    this.trailA.destroy();
    this.trailB.destroy();
    this.soil.destroy();
    this.globalsBuf.destroy();
    this.speciesBuf.destroy();
    this.matrixBuf.destroy();
    this.splashBuf.destroy();
    this.agentBuf.destroy();
    this.depositBuf.destroy();
  }

  /** Soil is a single layer on the same grid, so only z differs from gridDispatch(). */
  private soilDispatch(): [number, number, number] {
    return [
      Math.ceil(this.gridW / GRID_WORKGROUP),
      Math.ceil(this.gridH / GRID_WORKGROUP),
      1,
    ];
  }

  private gridDispatch(): [number, number, number] {
    return [
      Math.ceil(this.gridW / GRID_WORKGROUP),
      Math.ceil(this.gridH / GRID_WORKGROUP),
      this.config.speciesCount,
    ];
  }

  private writeGlobals(pcgTick: number): void {
    const ctx = this.ctx as GpuRuntimeContext;
    if (this.paletteDirty) this.refreshPalette();
    const u = this.globalsU32;
    const f = this.globalsF32;
    u[0] = this.gridW;
    u[1] = this.gridH;
    u[2] = this.config.speciesCount;
    u[3] = this.agentsPerSpecies;
    u[4] = this.seed >>> 0;
    u[5] = pcgTick >>> 0;
    f[6] = this.config.depositScale;
    f[7] = this.config.senseGain;
    f[8] = ctx.width;
    f[9] = ctx.height;
    // Scene exposure (θ) and the grade's manual trim are one multiply, applied
    // here in the compositor. That is deliberate: auto-exposure measures the HDR
    // surface, so anything applied *after* the measurement would be invisible to
    // the controller and the "target mean" readout would not mean what it says.
    f[10] = this.config.exposure * Math.pow(2, this.config.render.grade.exposureEv);
    f[11] = this.config.gamma;
    f[12] = this.respawnFraction;
    u[13] = this.respawnKey >>> 0;
    u[14] = this.splashCount >>> 0;
    // The step's length in model steps. Only the diffusion kernel reads it —
    // every other per-step quantity is scaled on this side, where it is one
    // multiply on a uniform instead of one per invocation.
    f[15] = this.stepScale;
    const soil = this.config.soil;
    // Soil is an exponential accumulator on the same per-step clock as the
    // trails: the decay is a factor (so it exponentiates) and the accumulation
    // is a rate (so it multiplies). Both are exact identities at scale 1.
    f[16] = Math.pow(Math.min(Math.max(soil.decay, 0), 1), this.stepScale);
    f[17] = Math.max(soil.accum, 0) * this.stepScale;
    f[18] = Math.max(soil.depositBias, 0);
    f[19] = Math.max(soil.senseBias, 0);
    f[20] = soil.debugView ? 1 : 0;
    const grade = this.config.render.grade;
    f[21] = this.soilTint[0] as number;
    f[22] = this.soilTint[1] as number;
    f[23] = this.soilTint[2] as number;
    f[24] = Math.max(grade.soilTint, 0);
    // The feedback lane, converted from "per 60 Hz frame" (what the slider and
    // the shipped default mean) to "per frame at this display's refresh rate"
    // (what the compositor actually applies). Both are geometric per frame, so
    // the conversion is a power, and `renderDtFrames` is 1 on a 60 Hz display —
    // the numbers are unchanged there, which is the point.
    //
    // Amount is clamped below 1 first: the lane is geometric, so >= 1 never
    // decays, and exponentiating a value >= 1 would not fix that.
    //
    // Macro `trails` multiplies the authored amount BEFORE the clamp and the
    // power, because the per-60-Hz-frame semantics belong to the *effective*
    // value: applying the macro after the exponentiation would make it mean
    // something different on every refresh rate.
    const fbAmount = Math.min(
      Math.max(this.config.render.feedback.amount, 0) *
        Math.max(this.config.macros.trails, 0),
      0.95,
    );
    const fbZoom = Math.max(this.config.render.feedback.zoom, 1e-3);
    f[25] = Math.pow(fbAmount, this.renderDtFrames);
    f[26] = Math.pow(fbZoom, this.renderDtFrames);
    f[27] = MAX_DEPOSIT;
    ctx.device.queue.writeBuffer(this.globalsBuf, 0, this.globalsBytes);
  }

  private uploadSpecies(): void {
    const ctx = this.ctx as GpuRuntimeContext;
    if (this.paletteDirty) this.refreshPalette();
    const d = this.speciesData;
    const list = this.config.species;
    const imp = this.impulses;
    const macros = this.config.macros;
    for (let k = 0; k < this.config.speciesCount; k++) {
      const s = list[k];
      const o = k * FLOATS_PER_SPECIES;
      if (!s) {
        d.fill(0, o, o + FLOATS_PER_SPECIES);
        continue;
      }
      // The impulse lane, applied here and nowhere else: multiplicative on top of
      // whatever wrote the base value this tick (sliders in manual mode, the
      // slew-limited blend in mapped mode), and always 1.0 when nothing is firing.
      const depositMul = imp ? (imp.depositMul[k] ?? 1) : 1;
      const brightMul = imp ? (imp.brightMul[k] ?? 1) : 1;
      const sensorMul = imp ? (imp.sensorMul[k] ?? 1) : 1;

      // Per-species structural scale — the θ slot, so the modulator and the
      // sliders both reach it. It multiplies BOTH distance lanes below (and
      // nothing else), which is the entire point: sensor reach and step length
      // move together, so the network gets coarser or finer instead of merely
      // differently proportioned.
      //
      // Composed here, alongside the macros rather than inside them, so the three
      // layers stay separable and stay in this order:
      //
      //   θ curve  ×  scale (θ, per-species)  ×  macro (outside θ, global)
      //                                          ×  impulse sensorMul (transient)
      //
      // Multiplication is associative and exactly 1.0 is an exact float identity,
      // so a config at scale = 1 produces bit-identical uniforms to the code
      // before this slot existed — which is what makes the shipped defaults a
      // usable A/B baseline.
      //
      // Clamped to the same hard bounds the panel slider and the θ registry use,
      // because this is the point where the value becomes geometry: a file or a
      // rebased base carrying something wilder must not reach the shader.
      const scale = Math.min(
        Math.max(s.scale, MIN_SPECIES_SCALE),
        MAX_SPECIES_SCALE,
      );

      // Sensor pop scales the whole curve p1 + p2·x^p3, not just its base, so the
      // reach grows at every trail intensity instead of only in empty space.
      // Macro `reach` rides the same lane and multiplies alongside it: both are
      // outside θ, and both mean "how far this species looks".
      const reach = Math.max(macros.reach, 0) * scale;
      d[o + 0] = s.sensorDist.p1 * reach * sensorMul;
      d[o + 1] = s.sensorDist.p2 * reach * sensorMul;
      d[o + 2] = s.sensorDist.p3;
      d[o + 3] = 0;
      d[o + 4] = s.sensorAngle.p1;
      d[o + 5] = s.sensorAngle.p2;
      d[o + 6] = s.sensorAngle.p3;
      d[o + 7] = 0;
      // Rotation is an angular *rate* — radians per model step — so the whole
      // adaptive curve p1 + p2·x^p3 scales with the step's length. `adaptive()`
      // is linear in p1 and p2, so scaling those two scales its output at every
      // trail intensity and the exponent stays alone.
      d[o + 8] = s.rotate.p1 * this.stepScale;
      d[o + 9] = s.rotate.p2 * this.stepScale;
      d[o + 10] = s.rotate.p3;
      d[o + 11] = 0;
      // Macro `agility`, the whole move curve — same treatment as `reach`, one
      // lane down: how far an agent travels per step at any trail intensity.
      // `scale` rides here too; that pairing is what makes it a *scale* rather
      // than a second reach knob.
      //
      // `stepScale` rides here as well, and only here among the two distance
      // lanes: move distance is a *velocity* (cells per step) and must cover the
      // step actually being taken, while `sensorDist` is a reach — a length, not
      // a rate — and a shorter step does not make an agent short-sighted.
      const agility = Math.max(macros.agility, 0) * scale * this.stepScale;
      d[o + 12] = s.moveDist.p1 * agility;
      d[o + 13] = s.moveDist.p2 * agility;
      d[o + 14] = s.moveDist.p3;
      d[o + 15] = 0;
      // Hue is static art direction (palette); brightness is the reactive half.
      // It multiplies the compositor weight only — never deposit — so light can
      // react hard without the trail field changing shape underneath it.
      //
      // Composition order, Revision 4, and it is the order the whole brightness
      // rework depends on:
      //
      //   base (slider / θ)  ×  stem-follow  →  clamp to MAX_BRIGHTNESS
      //                                          ×  impulse flash
      //
      // The stem lane is the slow, legible part ("the vocal dropped out, so the
      // cyan species faded") and is clamped with the base. The flash is the fast
      // transient part and multiplies *outside* that clamp, so a kick still reads
      // on a species the stem lane is currently holding at its floor — the same
      // argument as MAX_EFFECTIVE_DEPOSIT vs MAX_DEPOSIT for the deposit lane.
      const follow = this.brightFollow ? (this.brightFollow[k] ?? 1) : 1;
      const light = Math.min(
        Math.max(s.brightness, 0) * Math.max(follow, 0),
        MAX_BRIGHTNESS,
      );
      d[o + 16] = this.paletteRgb[k * 3 + 0] as number;
      d[o + 17] = this.paletteRgb[k * 3 + 1] as number;
      d[o + 18] = this.paletteRgb[k * 3 + 2] as number;
      d[o + 19] = s.intensity * light * brightMul;
      // The one music hookup in phase 4: stem k scales species k's deposit.
      // Two clamps, not one. The stem-driven base saturates at
      // MAX_EFFECTIVE_DEPOSIT; the impulse burst then multiplies *that* and only
      // has to fit under MAX_DEPOSIT. Folding the burst into the base clamp made
      // it vanish exactly when the music was densest — a full kick on a species
      // whose stem was already active asked for 12 and got 6, so the burst
      // collapsed from 4x to <=2x and the workbench slider went dead above ~1.5.
      // The i32 deposit atomic wraps to a large negative trail (and blacks out
      // the species) past 2^31 / (deposit * depositScale) agents in one cell;
      // MAX_DEPOSIT is what that headroom is sized against.
      //
      // Macro `deposit` joins the *base* lane, inside MAX_EFFECTIVE_DEPOSIT
      // rather than outside it: it is a re-statement of the authored rate, not a
      // transient, so it saturates with the slider it scales. The impulse burst
      // still multiplies outside that clamp, exactly as before.
      const stem = this.config.stemDrive ? Math.max(this.stems[k] ?? 0, 0) : 0;
      const base = Math.min(
        Math.max(s.deposit, 0) *
          Math.max(macros.deposit, 0) *
          (1 + Math.max(this.config.stemGain, 0) * stem),
        MAX_EFFECTIVE_DEPOSIT,
      );
      // Deposit is an amount per step, i.e. a rate; MAX_DEPOSIT is the i32
      // atomic's per-step headroom and therefore clamps the scaled value, not
      // the authored one.
      d[o + 20] = Math.min(base * Math.max(depositMul, 0) * this.stepScale, MAX_DEPOSIT);
      // Decay is a geometric factor per step, so it exponentiates rather than
      // multiplying — same conversion the render-domain feedback lane does with
      // `renderDtFrames`, and an exact identity at scale 1.
      d[o + 21] = Math.pow(s.decay, this.stepScale);
      // Macro `density`, clamped back into 0..1. `stats()` recomputes the same
      // number from the config — hence the shared helper, so the readout and the
      // GPU can never disagree.
      d[o + 22] = this.effectiveAlive(s);
      d[o + 23] = s.diffuseCentre;
    }
    ctx.device.queue.writeBuffer(this.speciesBuf, 0, d);
  }

  private chooseGrid(ctx: GpuRuntimeContext): { w: number; h: number } {
    const k = this.config.speciesCount;
    const cfg = this.config;
    let w = Math.max(1, ctx.width) * cfg.gridScale;
    let h = Math.max(1, ctx.height) * cfg.gridScale;

    const dimCap = Math.min(cfg.maxGridDim, ctx.device.limits.maxTextureDimension2D);
    const shrink = Math.min(1, dimCap / Math.max(w, h));
    w *= shrink;
    h *= shrink;

    // deposit buffer is gridW * gridH * K * 4 bytes and must fit one binding
    const budget = Math.min(
      ctx.device.limits.maxStorageBufferBindingSize,
      ctx.device.limits.maxBufferSize,
    );
    const maxCells = Math.floor(budget / (4 * k));
    if (w * h > maxCells) {
      const f = Math.sqrt(maxCells / (w * h));
      w *= f;
      h *= f;
    }

    // Grid is fixed for the life of the sim: resizing it would throw away all
    // accumulated trail state. Canvas resizes only change composite scaling.
    const snap = (v: number): number =>
      Math.max(256, Math.floor(v / GRID_WORKGROUP) * GRID_WORKGROUP);
    return { w: snap(w), h: snap(h) };
  }
}

// ── extras validation helpers ────────────────────────────────────────────────
//
// Deliberately total functions: `applyExtras` is handed whatever was in the
// file, and the contract is that it never throws — a broken block loses its
// values to the defaults, not the whole load.
//
// Duplicated from plife.ts rather than shared. They are four lines each, and the
// alternative is one sim importing from the other (a dependency neither wants)
// or a new module whose whole content is two clamps.

function plainObject(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
}

function clampNum(v: unknown, fallback: number, lo: number, hi: number): number {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  return Math.min(Math.max(n, lo), hi);
}

/** Strict: only a real boolean counts, so `"false"` and `0` fall back rather than coerce. */
function readBool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback;
}
