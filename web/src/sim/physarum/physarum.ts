import type { GpuContext } from '../../gpu/context';
import type { FeaturesFrame } from '../../timeline/sampler';
import { MAX_SPLASHES, type ImpulseState } from '../impulses';
import { HDR_FORMAT, PostFx } from '../render/postfx';
import type { Sim } from '../types';
import {
  defaultConfig,
  hexToLinear,
  MAX_DEPOSIT,
  MAX_EFFECTIVE_DEPOSIT,
  paletteLinear,
  type PhysarumConfig,
} from './config';

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
/** sim substeps a single clock tick may expand into; folded into the PCG tick key */
const MAX_SUBSTEPS = 8;
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
  stepAccumulator: number;
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

export class PhysarumSim implements Sim {
  readonly name = 'physarum';
  readonly config: PhysarumConfig;

  /**
   * Notified whenever the seed changes — a reseed, and equally a snapshot restore
   * putting an older seed back. Anything that keys its own hashing off the sim's
   * seed (the impulse engine's hotspots) has to follow it, or an A/B restore runs
   * the restored world against the newer seed's splash positions.
   */
  onSeedChange: ((seed: number) => void) | null = null;

  private ctx: GpuContext | null = null;
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

  private stepAccumulator = 0;
  private pendingSingleStep = false;
  private lastPcgTick = 0;
  private stepsThisFrame = 0;
  private respawnFraction = 0;
  private respawnKey = 0;
  private snap: SimSnapshot | null = null;

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

  private refreshPalette(): void {
    this.paletteDirty = false;
    for (let k = 0; k < this.config.speciesCount; k++) {
      const [r, g, b] = paletteLinear(this.config.palette, k);
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

  stats(): PhysarumStats {
    let alive = 0;
    for (const s of this.config.species) {
      alive += Math.floor(Math.min(Math.max(s.aliveFraction, 0), 1) * this.agentsPerSpecies);
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

  requestSingleStep(): void {
    this.pendingSingleStep = true;
  }

  async init(ctx: GpuContext): Promise<void> {
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

  /** Fresh seed, empty trails, empty soil, agents re-scattered. */
  reseed(seed: number): void {
    if (!this.ready || !this.ctx) return;
    const { device } = this.ctx;
    this.setSeed(seed);
    this.stepAccumulator = 0;
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
        stepAccumulator: this.stepAccumulator,
        lastPcgTick: this.lastPcgTick,
        takenAt: performance.now(),
      };
    }
    const snap = this.snap;
    snap.seed = this.seed;
    snap.stepAccumulator = this.stepAccumulator;
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
    this.stepAccumulator = snap.stepAccumulator;
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

  tick(frame: FeaturesFrame, simTick: number): void {
    if (!this.ready || !this.ctx) return;

    // The stems channel is 4-dim by contract; species beyond 4 get no stem drive.
    const stemOffset = this.stemOffset;
    if (stemOffset !== null) {
      for (let k = 0; k < this.stems.length; k++) {
        this.stems[k] = k < STEM_DIMS ? (frame.values[stemOffset + k] ?? 0) : 0;
      }
    }

    let steps = 0;
    if (this.config.paused) {
      if (this.pendingSingleStep) {
        this.pendingSingleStep = false;
        steps = 1;
      }
    } else {
      this.stepAccumulator += Math.max(this.config.speed, 0);
      while (this.stepAccumulator >= 1 && steps < MAX_SUBSTEPS) {
        this.stepAccumulator -= 1;
        steps++;
      }
      if (this.stepAccumulator > MAX_SUBSTEPS) this.stepAccumulator = 0;
    }

    this.stepsThisFrame = steps;
    if (steps > 0) this.uploadSplashes();
    for (let s = 0; s < steps; s++) {
      this.runStep(simTick * MAX_SUBSTEPS + s);
    }
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
      this.splashData[o + 5] = s.push;
      this.splashData[o + 6] = s.swirl;
      this.splashData[o + 7] = 0;
    }
    this.splashCount = n;
    if (n > 0) {
      (this.ctx as GpuContext).device.queue.writeBuffer(
        this.splashBuf,
        0,
        this.splashData,
        0,
        n * FLOATS_PER_SPLASH,
      );
    }
  }

  private runStep(pcgTick: number): void {
    const { device } = this.ctx as GpuContext;
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

  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.ready || !this.ctx) return;

    // Post surfaces follow the canvas, not the sim grid: a resize re-allocates
    // them (and invalidates the compositor's feedback binding) but never touches
    // accumulated trail state.
    this.post.ensureSize(this.ctx.width, this.ctx.height);
    if (this.post.sizeVersion !== this.compositeSizeVersion) this.rebuildCompositeBinds();
    this.post.gamma = this.config.gamma;

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

    this.post.run(encoder, targetView);
  }

  /**
   * Two bind groups, one per ping-pong parity, differing only in which HDR
   * surface is bound as the feedback source.
   */
  private rebuildCompositeBinds(): void {
    const { device } = this.ctx as GpuContext;
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
    const ctx = this.ctx as GpuContext;
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
    u[15] = 0;
    const soil = this.config.soil;
    f[16] = Math.min(Math.max(soil.decay, 0), 1);
    f[17] = Math.max(soil.accum, 0);
    f[18] = Math.max(soil.depositBias, 0);
    f[19] = Math.max(soil.senseBias, 0);
    f[20] = soil.debugView ? 1 : 0;
    const grade = this.config.render.grade;
    f[21] = this.soilTint[0] as number;
    f[22] = this.soilTint[1] as number;
    f[23] = this.soilTint[2] as number;
    f[24] = Math.max(grade.soilTint, 0);
    // Clamped below 1: the feedback lane is geometric, so >= 1 never decays.
    f[25] = Math.min(Math.max(this.config.render.feedback.amount, 0), 0.95);
    f[26] = Math.max(this.config.render.feedback.zoom, 1e-3);
    f[27] = MAX_DEPOSIT;
    ctx.device.queue.writeBuffer(this.globalsBuf, 0, this.globalsBytes);
  }

  private uploadSpecies(): void {
    const ctx = this.ctx as GpuContext;
    if (this.paletteDirty) this.refreshPalette();
    const d = this.speciesData;
    const list = this.config.species;
    const imp = this.impulses;
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

      // Sensor pop scales the whole curve p1 + p2·x^p3, not just its base, so the
      // reach grows at every trail intensity instead of only in empty space.
      d[o + 0] = s.sensorDist.p1 * sensorMul;
      d[o + 1] = s.sensorDist.p2 * sensorMul;
      d[o + 2] = s.sensorDist.p3;
      d[o + 3] = 0;
      d[o + 4] = s.sensorAngle.p1;
      d[o + 5] = s.sensorAngle.p2;
      d[o + 6] = s.sensorAngle.p3;
      d[o + 7] = 0;
      d[o + 8] = s.rotate.p1;
      d[o + 9] = s.rotate.p2;
      d[o + 10] = s.rotate.p3;
      d[o + 11] = 0;
      d[o + 12] = s.moveDist.p1;
      d[o + 13] = s.moveDist.p2;
      d[o + 14] = s.moveDist.p3;
      d[o + 15] = 0;
      // Hue is static art direction (palette); brightness is the modulated half.
      // It multiplies the compositor weight only — never deposit — so light can
      // react hard without the trail field changing shape underneath it.
      d[o + 16] = this.paletteRgb[k * 3 + 0] as number;
      d[o + 17] = this.paletteRgb[k * 3 + 1] as number;
      d[o + 18] = this.paletteRgb[k * 3 + 2] as number;
      d[o + 19] = s.intensity * Math.max(s.brightness, 0) * brightMul;
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
      const stem = this.config.stemDrive ? Math.max(this.stems[k] ?? 0, 0) : 0;
      const base = Math.min(
        Math.max(s.deposit, 0) * (1 + Math.max(this.config.stemGain, 0) * stem),
        MAX_EFFECTIVE_DEPOSIT,
      );
      d[o + 20] = Math.min(base * Math.max(depositMul, 0), MAX_DEPOSIT);
      d[o + 21] = s.decay;
      d[o + 22] = s.aliveFraction;
      d[o + 23] = s.diffuseCentre;
    }
    ctx.device.queue.writeBuffer(this.speciesBuf, 0, d);
  }

  private chooseGrid(ctx: GpuContext): { w: number; h: number } {
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
