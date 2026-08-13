/**
 * **vizfx** — the third substrate family: milkdrop/projectM-idiom screen-space
 * feedback visuals.
 *
 * Physarum and particle life are agent systems: they own matter (agents,
 * particles), step it, and render a picture *of* it. A vizfx visual owns no
 * matter at all. Its world is a pair of HDR textures and the picture **is** the
 * state — light is written into the field, and every step the field is
 * resampled through a warp (zoom, rotation, swirl, drift, decay) and written
 * back. Structure is not simulated, it is *accumulated*: twelve moving blobs
 * become a nebula because a hundred warped copies of them are on screen at once.
 *
 * Per model step (fixed 1/60 s; Particle Life alone now integrates at 1/120):
 *
 *   warp   full-screen, field[p] resampled + softened + dimmed + hue-rotated
 *          into field[1-p]                                     [loadOp clear]
 *   draw   full-screen, additive, the visual's emitters and event rings on top
 *                                                              [loadOp load]
 *   flip p
 *
 * Per frame:
 *
 *   composite  field[p] × exposure × adapted gain → the post chain's HDR surface
 *   post.run   measure → bloom → grade → swapchain
 *
 * ## Chassis and visual
 *
 * This class is the chassis: the ping-pong rig, the ModTarget contract, the
 * buffers, the seeded emitter placement, the stems lane, snapshot/restore. It
 * names no parameter. A **visual** (`slots.ts`'s `VizFxVisual`) supplies a θ
 * table and a WGSL pair, and everything else — the registry, the panel, the
 * uniform layout, the shader's `TH_*` constants — is generated from the table.
 * `nebula/nebula.ts` is the first one; the intended cost of a second is one WGSL
 * pair plus one table, with no edit here.
 *
 * `simId` is the *visual's* id, not 'vizfx', and that is the settled call: θ is
 * a different vector per visual, so two visuals must never share an autosave
 * slot for the same reason physarum and plife must not. The family is code
 * reuse; it is not a persistence key.
 *
 * ## The four "species" are the four stems
 *
 * K = 4 and layer k *is* stem k, in the analysis contract's own order. The
 * palette is per stem, `stemMap()` is the identity, `setBrightFollow` scales
 * each layer's light and the energy lane (`config.energy`) scales each layer's
 * emitter size. So "the vocals came in" is two simultaneous cues on one
 * identifiable colour, and no seed can rewire it into something else — the
 * legibility the project's priorities demand is structural here rather than
 * fortunate.
 *
 * Note the two lanes are deliberately not one. Stem-follow belongs to the
 * Modulator and is absent from an explorer tile; the energy lane belongs to the
 * sim and is not, which is what keeps nine tiles responding to the music while
 * they are being judged.
 *
 * ## Performance, stated up front
 *
 * Three full-screen passes per step plus the post chain, at canvas resolution.
 * At 1080p and speed 1 that is ~6 M fragment invocations per step, each running
 * a dozen `exp()` — under a millisecond on anything that runs the other two
 * substrates at all. There is deliberately **no** adaptive machinery here: plife
 * has a governor because its cost is quadratic in a number the music moves, and
 * this substrate's cost is a constant times the pixel count. If it is ever slow,
 * the window is too big, and that is a fact about the window.
 */
import type { GpuContext } from '../../gpu/context.ts';
import type { ModTarget, ThetaRegistry } from '../../mapping/target.ts';
import type { FeaturesFrame } from '../../timeline/sampler.ts';
import { LEGACY_SUBSTRATE_HZ, LEGACY_TICK_DIVISOR } from '../../timing.ts';
import { hash3, MAX_SPLASHES, pcg, type ImpulseState } from '../impulses.ts';
import { paletteLinear } from '../palette.ts';
import { mergeRenderConfig } from '../render/config.ts';
import { HDR_FORMAT, PostFx } from '../render/postfx.ts';
import type { Sim } from '../types.ts';
import { advanceStepCadence } from '../step-cadence.ts';
import {
  defaultEnergy,
  defaultVizFxColor,
  defaultVizFxConfig,
  ENERGY_RANGE,
  MACRO_RANGE,
  MAX_EMITTERS_PER_LAYER,
  MAX_SUBSTEPS,
  type VizFxConfig,
} from './config.ts';
import {
  applyVector,
  effectiveTheta,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  paramsToVector,
  vectorLength,
  wgslPrelude,
  type VizFxVisual,
} from './slots.ts';

import commonWgsl from './shaders/common.wgsl?raw';
import chassisWgsl from './shaders/chassis.wgsl?raw';

/** f32 slots in the Globals uniform; must match `struct Globals` in common.wgsl. */
const GLOBALS_WORDS = 16;
/** must match `struct Layer` in common.wgsl: two vec4f */
const FLOATS_PER_LAYER = 8;
/** must match `struct Splash` in common.wgsl: three vec4f */
const FLOATS_PER_SPLASH = 12;
/** one vec4f per emitter: (phase, radius ×, speed ×, size ×) */
const FLOATS_PER_EMITTER = 4;
/** the stems channel's width, by analysis contract */
const STEM_DIMS = 4;

/**
 * Seconds one model step advances the field. The app clock is 120 Hz, but this
 * substrate remains a 60-step-per-second per-step model: one model tick is
 * consumed for every two app ticks. `speed` still means steps per MODEL tick,
 * and — the part that matters here — every per-step
 * quantity in θ (`zoom`, `decay`, `rotate`) is a *per-step* factor whose meaning
 * would otherwise change with the refresh rate.
 *
 * This is the one place vizfx differs structurally from the render-domain
 * feedback plife uses: plife's echo runs once per rendered frame and has to be
 * exponentiated by the frame length to look the same at 60 and 240 Hz. The field
 * here runs on the fixed clock, so no correction exists and none is needed.
 */
const STEP_DT = 1 / LEGACY_SUBSTRATE_HZ;

/**
 * Ceiling on `brightness × stem-follow`, matching both other substrates and for
 * the same reason: impulse flashes multiply on top and are deliberately *not*
 * bounded by it, so a kick stays legible on a layer the stem lane is currently
 * holding at its floor.
 */
const MAX_BRIGHTNESS = 2;

/**
 * Divisor turning `ResponseConfig.splashPush` — authored in *physarum grid
 * cells per tick*, which is not a unit this substrate has — into a ring
 * expansion multiplier.
 *
 * 6 is chosen so the shipped snare (push 9) expands to about a third of the
 * screen height over its decay and the shipped vocal (push 2) to about a tenth.
 * A pass-through would put the snare's front two screen heights out, i.e.
 * off-frame while it still had most of its envelope left, and the event would
 * read as a flash with no shockwave at all.
 */
const PUSH_TO_EXPAND = 6;

/**
 * The world, on a spare texture. One rgba16float copy of the live field, plus
 * the CPU-side lane state that is not in it.
 *
 * The field is genuinely the whole of the GPU state — there is no agent buffer,
 * no grid, nothing rebuilt per step — so unlike plife's snapshot this one is
 * exact by construction rather than by an argument about which buffers carry
 * across.
 */
interface VizFxSnapshot {
  texture: GPUTexture;
  width: number;
  height: number;
  /** which ping-pong slot the copy came from; restore puts it back in the same one */
  parity: number;
  seed: number;
  steps: number;
  stepAccumulator: number;
  /**
   * The energy lane's smoothed state. Same reasoning as plife's population
   * snapshot: without it a restore would reproduce the picture while the lane
   * kept driving it from a posture belonging to some later moment of the song.
   */
  stemLevel: Float32Array;
  energy: Float32Array;
  /** wall-clock label for the workbench; not part of sim state */
  takenAt: number;
}

export interface VizFxStats {
  fieldW: number;
  fieldH: number;
  /** render passes in the post chain last frame; the composite is the +1 */
  renderPasses: number;
  stepsThisTick: number;
  activeRings: number;
}

export class VizFxSim implements Sim, ModTarget {
  readonly name: string;
  /** The persistence discriminator — the VISUAL's id. See the header. */
  readonly simId: string;
  readonly config: VizFxConfig;
  /** The visual this chassis is running. Read by `main.ts` to build explorer tiles. */
  readonly visual: VizFxVisual;

  /** The shared render chain. Owns the HDR surfaces, bloom, grading, auto-exposure. */
  readonly post: PostFx;

  /** Notified on every seed change — a reseed, and equally a snapshot restore. */
  onSeedChange: ((seed: number) => void) | null = null;

  private ctx: GpuContext | null = null;
  private ready = false;
  private seed: number;

  // ── the field: this substrate's entire world ───────────────────────────────
  private fieldW = 0;
  private fieldH = 0;
  private fieldTex: GPUTexture[] = [];
  private fieldViews: GPUTextureView[] = [];
  /** index of the CURRENT field; the warp reads it and writes `1 - parity` */
  private parity = 0;

  private sampler!: GPUSampler;
  private globalsBuf!: GPUBuffer;
  private thetaBuf!: GPUBuffer;
  private layerBuf!: GPUBuffer;
  private emitterBuf!: GPUBuffer;
  private splashBuf!: GPUBuffer;

  private bindLayout!: GPUBindGroupLayout;
  private warpPipeline!: GPURenderPipeline;
  private drawPipeline!: GPURenderPipeline;
  private injectPipeline!: GPURenderPipeline;
  private compositePipeline!: GPURenderPipeline;
  /** indexed by which field texture the group binds; `binds[p]` reads `fieldTex[p]` */
  private binds: GPUBindGroup[] = [];

  private readonly globalsBytes = new ArrayBuffer(GLOBALS_WORDS * 4);
  private readonly globalsF32 = new Float32Array(this.globalsBytes);
  private readonly globalsU32 = new Uint32Array(this.globalsBytes);
  // `Float32Array<ArrayBuffer>` rather than plain `Float32Array`: WebGPU's
  // `writeBuffer` refuses a possibly-shared backing store, and the default type
  // parameter is `ArrayBufferLike`. Same annotation, for the same reason, as
  // plife's `speciesData`.
  /** θ with the macro rig folded in — what the shader actually reads */
  private readonly effTheta: Float32Array<ArrayBuffer>;
  private readonly layerData: Float32Array<ArrayBuffer>;
  private readonly emitterData: Float32Array<ArrayBuffer>;
  private readonly splashData = new Float32Array(MAX_SPLASHES * FLOATS_PER_SPLASH);
  private splashCount = 0;

  /**
   * Linearised palette (3 floats per layer). Static art direction whose source of
   * truth is a set of CSS strings, re-derived only when something says they
   * changed — `uploadLayers` runs every tick and re-parsing there was pure waste.
   */
  private readonly paletteRgb: Float32Array;
  private paletteDirty = true;

  // ── the stems → presence lane (CPU-owned; see `updateEnergy`) ──────────────
  /** null = no usable stems channel; the lane is off and every layer sits at 1 */
  private stemOffset: number | null = null;
  private readonly stemLevel: Float32Array;
  private readonly energy: Float32Array;
  /**
   * False until the first tick, which snaps the EMA to the music instead of
   * easing into it from zero. Without it the world opens with every layer at its
   * floor and spends the smoothing constant discovering that the track is playing
   * — a slow unexplained swell as the first thing anyone sees.
   */
  private energyPrimed = false;

  /** Live impulse lane, owned by the ImpulseEngine and mutated in place. */
  private impulses: ImpulseState | null = null;
  /** Per-layer brightness multiplier from the stem-follow lane, held by reference. */
  private brightFollow: Float32Array | null = null;
  /** max over layers of (depositMul - 1), 0 at rest. The warp's transient input. */
  private pulse = 0;

  private stepAccumulator = 0;
  private pendingSingleStep = false;
  private stepsThisTick = 0;
  /** steps since the world was last cleared. The shader's clock, and the PCG key. */
  private steps = 0;
  private snap: VizFxSnapshot | null = null;

  /** θ registry and stem keying, both pure functions of the table — built once. */
  private theta: ThetaRegistry | null = null;
  private species2stem: Int32Array | null = null;

  constructor(visual: VizFxVisual, seed: number, config: VizFxConfig = defaultVizFxConfig(visual)) {
    this.visual = visual;
    this.name = visual.id;
    this.simId = visual.id;
    this.seed = seed >>> 0;
    this.config = config;
    const k = Math.max(1, config.speciesCount);
    this.paletteRgb = new Float32Array(k * 3);
    this.stemLevel = new Float32Array(k);
    this.energy = new Float32Array(k).fill(1);
    this.effTheta = new Float32Array(vectorLength(visual));
    this.layerData = new Float32Array(k * FLOATS_PER_LAYER);
    this.emitterData = new Float32Array(k * MAX_EMITTERS_PER_LAYER * FLOATS_PER_EMITTER);
    this.post = new PostFx(config.render);
  }

  get currentSeed(): number {
    return this.seed;
  }

  get hasSnapshot(): boolean {
    return this.snap !== null;
  }

  get snapshotAge(): number {
    return this.snap ? performance.now() - this.snap.takenAt : 0;
  }

  /** Emitters per layer, sanitised. The buffer is sized to the ceiling either way. */
  private get emitPerLayer(): number {
    const n = Math.round(this.config.emittersPerLayer);
    return Math.min(Math.max(Number.isFinite(n) ? n : 1, 1), MAX_EMITTERS_PER_LAYER);
  }

  /** The only place `seed` is assigned after construction, so listeners cannot be missed. */
  private setSeed(seed: number): void {
    this.seed = seed >>> 0;
    this.onSeedChange?.(this.seed);
  }

  invalidatePalette(): void {
    this.paletteDirty = true;
  }

  setImpulses(state: ImpulseState | null): void {
    this.impulses = state;
  }

  setBrightFollow(values: Float32Array | null): void {
    this.brightFollow = values;
  }

  requestSingleStep(): void {
    this.pendingSingleStep = true;
  }

  /**
   * Bind the timeline's stems channel for the energy lane. Mirrors both other
   * substrates' setters exactly, including the failure mode: a missing or
   * too-narrow channel disables the lane rather than reading whatever floats sit
   * at another channel's offset, and says so once.
   */
  setStemChannel(channel: { offset: number; dims: number } | undefined): void {
    if (!channel || channel.dims < STEM_DIMS) {
      this.stemOffset = null;
      this.stemLevel.fill(0);
      this.energy.fill(1);
      this.config.energy.followStems = false;
      console.warn(
        `${this.simId}: no usable stems channel (${
          channel ? `${channel.dims} dims, need ${STEM_DIMS}` : 'absent'
        }); per-layer energy disabled`,
      );
      return;
    }
    this.stemOffset = channel.offset;
  }

  /**
   * The energy lane's live state, by reference and read-only by convention — the
   * arrays are reused every tick rather than copied, because the panel wants to
   * *watch* them and a per-frame allocation for a readout is exactly the waste
   * the render loop cannot afford.
   */
  energyState(): { level: Float32Array; energy: Float32Array } {
    return { level: this.stemLevel, energy: this.energy };
  }

  // ── ModTarget: θ, as the mapping layer sees it ─────────────────────────────

  /**
   * The slot table, precomputed. Every view is a pure function of the visual's
   * table, which is frozen for the life of the sim, so this is built on first ask
   * and then handed out **by identity** — `Modulator.setConfig` passes `.classes`
   * straight into a SlewLimiter that holds it by reference, so returning a fresh
   * object here would quietly detach the limiter from the registry.
   *
   * No `seedBase`: unlike plife, nothing in this θ is drawn wholesale from the
   * seed. The generic personality jitter is the whole of what a reroll does to
   * the *parameters*; what it also does — rearrange the emitter constellation —
   * is outside θ entirely and lives in `redrawEmitters`.
   */
  registry(): ThetaRegistry {
    if (this.theta) return this.theta;
    this.theta = {
      length: vectorLength(this.visual),
      slots: modulationSlots(this.visual),
      mask: modulationMask(this.visual),
      classes: fieldClasses(this.visual),
      names: fieldNames(this.visual),
    };
    return this.theta;
  }

  currentVector(): Float64Array {
    return paramsToVector(this.visual, this.config.params);
  }

  /**
   * θ → the live parameter store. Nothing GPU-side is derived from θ that is not
   * re-uploaded every tick, so unlike plife's this pairing has no buffer to push
   * — the one thing that *is* derived is the workbench's per-species brightness
   * readout, which `syncSpecies` mirrors out of θ.
   */
  applyTheta(v: ArrayLike<number>, mask?: Uint8Array): void {
    applyVector(this.visual, this.config.params, v, mask);
    this.syncSpecies();
  }

  /**
   * Layer k → stem k. The identity, and structurally so: the layers *are* the
   * stems in this substrate (see the header), so there is nothing here to key.
   */
  stemMap(): Int32Array {
    if (this.species2stem) return this.species2stem;
    const k = this.config.speciesCount;
    const map = new Int32Array(k);
    for (let i = 0; i < k; i++) map[i] = i < STEM_DIMS ? i : -1;
    this.species2stem = map;
    return map;
  }

  /**
   * `ModTargetConfig.species[k].brightness` is the workbench's stem-follow
   * readout, and the value it wants is θ's. θ is the source of truth (the slider
   * binds `config.params`), so this mirrors one into the other rather than
   * keeping two writable copies — which is what a second bound object would be.
   */
  private syncSpecies(): void {
    for (let k = 0; k < this.config.speciesCount; k++) {
      const dst = this.config.species[k];
      if (!dst) continue;
      const v = this.config.params[`layer${k}.brightness`];
      dst.brightness = typeof v === 'number' && Number.isFinite(v) ? v : 1;
    }
  }

  // ── ModTarget: the opaque per-sim extras block ─────────────────────────────
  //
  // Three blocks are outside θ *and* outside anything the mapping layer knows how
  // to carry: the macro rig, the energy lane and the structural emitter count.
  // None belongs in `ModulationConfig`'s schema — that file describes a mapping,
  // not a substrate — so they travel in the opaque `extras` channel and this pair
  // is the only code that understands their shape.
  //
  // The channel has two consumers and the second is easy to forget: persistence,
  // and `ExplorerRig.syncStyle`, which fans the serialised block into all nine
  // tiles twice a second. Anything outside both θ and extras is a panel edit the
  // tiles never see.

  serializeExtras(): Record<string, unknown> {
    return {
      macros: { ...this.config.macros },
      energy: { ...this.config.energy },
      emittersPerLayer: this.emitPerLayer,
      // θ slots, but the two that are excluded from modulation and owned by the
      // look tab — so they are art direction in practice, and nothing else saves
      // θ. Without them here, every reload came back at the registry's `def`.
      look: {
        exposure: this.config.params['exposure'],
        gamma: this.config.params['gamma'],
      },
    };
  }

  /**
   * The inverse, and deliberately paranoid: `extras` is opaque to every layer
   * between the file and here, so nothing upstream has validated it. Every field
   * is clamped into the range its slider shows, anything missing or non-finite
   * falls back to the shipped default, and this never throws.
   *
   * Every write is **in place** into the existing sub-objects, which is load
   * bearing rather than incidental: the panel's tweakpane bindings hold
   * `config.macros` and `config.energy` by reference, so replacing either object
   * would leave the panel editing something nothing runs.
   */
  applyExtras(raw: Record<string, unknown> | undefined): void {
    const o = (raw ?? {}) as Record<string, unknown>;

    const src = plainObject(o['macros']);
    for (const m of this.visual.macros) {
      this.config.macros[m.key] = clampNum(src[m.key], 1, MACRO_RANGE.min, MACRO_RANGE.max);
    }

    const e = this.config.energy;
    const es = plainObject(o['energy']);
    const defE = defaultEnergy();
    e.followStems = readBool(es['followStems'], defE.followStems);
    e.floor = clampNum(es['floor'], defE.floor, ENERGY_RANGE.floor.min, ENERGY_RANGE.floor.max);
    e.curve = clampNum(es['curve'], defE.curve, ENERGY_RANGE.curve.min, ENERGY_RANGE.curve.max);
    e.smoothingMs = clampNum(
      es['smoothingMs'],
      defE.smoothingMs,
      ENERGY_RANGE.smoothingMs.min,
      ENERGY_RANGE.smoothingMs.max,
    );

    // Rounded because it is a *count* the shader indexes with, and a slider that
    // snapped visually to 3 while holding 3.4 would be a constellation that
    // disagreed with its own label. A change here redraws the emitters, because
    // the extra ones have never been placed — and `redrawEmitters` is seeded, so
    // doing it on every style sync is idempotent rather than a flicker.
    const want = Math.round(
      clampNum(o['emittersPerLayer'], this.visual.emittersPerLayer, 1, MAX_EMITTERS_PER_LAYER),
    );
    // Clamped to the slot's own θ bound, so a hand-edited file cannot put the
    // slider off its rail. Absent means "keep what the preset gave us".
    const look = plainObject(o['look']);
    for (const key of ['exposure', 'gamma'] as const) {
      const slot = this.visual.globalSlots.find((s) => s.key === key);
      if (!slot || look[key] === undefined) continue;
      // Fallback is the live value, not `slot.def` — that is per-layer-callable
      // on some slots, and these two already hold whatever the preset set.
      const current = this.config.params[key] ?? 0;
      this.config.params[key] = clampNum(look[key], current, slot.min, slot.max);
    }

    if (want !== this.config.emittersPerLayer) {
      this.config.emittersPerLayer = want;
      this.redrawEmitters();
    }
  }

  // ── stats / status ─────────────────────────────────────────────────────────

  stats(): VizFxStats {
    return {
      fieldW: this.fieldW,
      fieldH: this.fieldH,
      renderPasses: this.post.passCount + 1,
      stepsThisTick: this.stepsThisTick,
      activeRings: this.splashCount,
    };
  }

  status(): string {
    return `${this.fieldW}×${this.fieldH} field · ${this.config.speciesCount} layers · seed ${this.seed}`;
  }

  // ── init ───────────────────────────────────────────────────────────────────

  async init(ctx: GpuContext): Promise<void> {
    this.ctx = ctx;
    const { device } = ctx;
    const k = this.config.speciesCount;

    // mirror-repeat, not clamp-to-edge: the warp's drift and ripple terms push
    // source coordinates a little outside the frame, and clamping smears the edge
    // texel along the border where — since the field is an accumulator — it
    // builds a bright picture-frame that never decays. Mirroring folds the read
    // back inside instead, which is invisible.
    this.sampler = device.createSampler({
      label: `${this.simId}.field`,
      addressModeU: 'mirror-repeat',
      addressModeV: 'mirror-repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.globalsBuf = device.createBuffer({
      label: `${this.simId}.globals`,
      size: GLOBALS_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.thetaBuf = device.createBuffer({
      label: `${this.simId}.theta`,
      size: Math.max(this.effTheta.byteLength, 16),
      usage: storage,
    });
    this.layerBuf = device.createBuffer({
      label: `${this.simId}.layers`,
      size: Math.max(k * FLOATS_PER_LAYER * 4, 16),
      usage: storage,
    });
    this.emitterBuf = device.createBuffer({
      label: `${this.simId}.emitters`,
      size: this.emitterData.byteLength,
      usage: storage,
    });
    // Fixed size and always allocated: 32 × 48 B is 1.5 KiB, and a buffer that
    // may or may not exist would make the bind group conditional.
    this.splashBuf = device.createBuffer({
      label: `${this.simId}.splashes`,
      size: MAX_SPLASHES * FLOATS_PER_SPLASH * 4,
      usage: storage,
    });

    const F = GPUShaderStage.FRAGMENT;
    const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
    // One layout for all four passes. See the note at the top of common.wgsl:
    // WebGPU validates a pipeline layout against the bindings an entry point
    // *statically uses*, so the composite pass costs nothing for having the
    // splash buffer in scope, and four near-identical layouts would be four
    // copies of one list to keep in step.
    this.bindLayout = device.createBindGroupLayout({
      label: `${this.simId}.layout`,
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, buffer: ro },
        { binding: 2, visibility: F, buffer: ro },
        { binding: 3, visibility: F, buffer: ro },
        { binding: 4, visibility: F, buffer: ro },
        { binding: 5, visibility: F, sampler: { type: 'filtering' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: F, buffer: ro },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindLayout],
    });

    // common.wgsl (bindings and helpers), then the GENERATED θ prelude, then the
    // pass body. Nothing forward-references anything in that order — see
    // `wgslPrelude`.
    const prelude = wgslPrelude(this.visual);
    const src = (label: string, body: string): GPUShaderModule =>
      device.createShaderModule({ label, code: `${commonWgsl}\n${prelude}\n${body}` });

    const pass = (
      label: string,
      module: GPUShaderModule,
      entryPoint: string,
      blend?: GPUBlendState,
    ): GPURenderPipeline =>
      device.createRenderPipeline({
        label,
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vsMain' },
        fragment: {
          module,
          entryPoint,
          targets: [blend ? { format: HDR_FORMAT, blend } : { format: HDR_FORMAT }],
        },
        primitive: { topology: 'triangle-list' },
      });

    const warpModule = src(`${this.simId}.warp`, this.visual.warpWgsl);
    const drawModule = src(`${this.simId}.draw`, this.visual.drawWgsl);
    const chassisModule = src(`${this.simId}.chassis`, chassisWgsl);

    this.warpPipeline = pass(`${this.simId}.warp`, warpModule, 'fsWarp');
    // Additive, and unbounded: the field is rgba16float with no clipping point,
    // so overlapping emitters sum honestly and the tone map at the end of the
    // post chain is the only thing that ever compresses.
    this.drawPipeline = pass(`${this.simId}.draw`, drawModule, 'fsDraw', {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    });
    this.injectPipeline = pass(`${this.simId}.inject`, chassisModule, 'fsInject');
    this.compositePipeline = pass(`${this.simId}.composite`, chassisModule, 'fsComposite');

    this.post.init(ctx);
    this.post.ensureSize(ctx.width, ctx.height);

    this.ensureField(ctx.width, ctx.height);
    this.ready = true;
    this.reseed(this.seed);
  }

  /**
   * Allocate (or reallocate) the ping-pong field for the current canvas size.
   *
   * **A resize clears the world**, and that is a decision rather than an
   * oversight. The alternative is a best-effort blit of the old field into the
   * new one, which sounds kinder and is not: the field's contents are in
   * *aspect-corrected* coordinates that every θ radius is expressed in, so a blit
   * either stretches the accumulated arms (changing what every radius meant) or
   * letterboxes them (leaving a hard rectangle of old light that the warp then
   * smears for the next thirty seconds). Both look like a bug; clearing looks
   * like a restart, which is what it is.
   *
   * The other two substrates have the same property for the same reason and say
   * so — physarum fixes its grid at init and plife its world aspect — they simply
   * do not have to re-allocate to express it.
   */
  private ensureField(width: number, height: number): boolean {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.fieldW && h === this.fieldH && this.fieldTex.length === 2) return false;
    const ctx = this.ctx;
    if (!ctx) return false;
    const { device } = ctx;

    this.releaseField();
    this.fieldW = w;
    this.fieldH = h;
    // COPY_SRC/COPY_DST are for the workbench's snapshot/restore; they cost
    // nothing when unused.
    this.fieldTex = [0, 1].map((i) =>
      device.createTexture({
        label: `${this.simId}.field${i}`,
        size: { width: w, height: h },
        format: HDR_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      }),
    );
    this.fieldViews = this.fieldTex.map((t) => t.createView());
    this.binds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `${this.simId}.bind${p}`,
        layout: this.bindLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: { buffer: this.thetaBuf } },
          { binding: 2, resource: { buffer: this.layerBuf } },
          { binding: 3, resource: { buffer: this.emitterBuf } },
          { binding: 4, resource: { buffer: this.splashBuf } },
          { binding: 5, resource: this.sampler },
          { binding: 6, resource: this.fieldViews[p] as GPUTextureView },
          { binding: 7, resource: { buffer: this.post.autoBuffer } },
        ],
      }),
    );
    this.clearField();
    // A snapshot of a differently-sized field cannot be restored into this one
    // (`copyTextureToTexture` requires matching extents) and resampling it would
    // hit exactly the objection above. Dropped, loudly enough for the workbench's
    // button to grey out on the next refresh.
    if (this.snap && (this.snap.width !== w || this.snap.height !== h)) this.clearSnapshot();
    return true;
  }

  private releaseField(): void {
    for (const t of this.fieldTex) t.destroy();
    this.fieldTex = [];
    this.fieldViews = [];
    this.binds = [];
  }

  /** Both halves of the ping-pong to black. Two empty render passes, no shader. */
  private clearField(): void {
    if (!this.ctx || this.fieldViews.length !== 2) return;
    const encoder = this.ctx.device.createCommandEncoder({ label: `${this.simId}.clear` });
    for (const view of this.fieldViews) {
      encoder
        .beginRenderPass({
          colorAttachments: [
            { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
          ],
        })
        .end();
    }
    this.ctx.device.queue.submit([encoder.finish()]);
  }

  // ── world lifecycle ────────────────────────────────────────────────────────

  /**
   * Fresh seed: a new emitter constellation, and — unless `keepWorld` — an empty
   * field.
   *
   * `setSeed` fires `onSeedChange`, and that callback is most of what a reroll
   * *is*: it re-keys the impulse hotspots and re-runs the modulator's seeded
   * rewire, which restamps the whole personality. All of that still happens under
   * `keepWorld`. What `keepWorld` skips is the one thing that is not the seed's
   * doing — throwing the accumulated field away.
   *
   * The result is the gesture the option exists for, and it is more visible here
   * than in either other substrate: the nebula you are looking at stays exactly
   * where it is, and a different flow takes hold of it. You watch the arms you
   * already have be re-warped rather than watching them be replaced. Auto-exposure
   * is not reset either — there is no fade up from black, because nothing went
   * black — and the step clock carries on, because it is the world's clock and the
   * world did not restart.
   */
  reseed(seed: number, opts?: { keepWorld?: boolean }): void {
    this.setSeed(seed);
    this.redrawEmitters();
    if (!this.ready || !this.ctx) return;
    if (opts?.keepWorld === true) return;
    this.steps = 0;
    this.stepAccumulator = 0;
    this.energyPrimed = false;
    this.clearField();
    // A new world starts from black; letting the adapted gain carry over would
    // make the first second a slow fade up from the old exposure.
    this.post.resetAutoExposure();
  }

  /**
   * Adopt another vizfx sim's live *look* — the render/grade block, the two
   * CPU-owned exposure slots, and the adapted auto-exposure gain.
   *
   * This exists for one caller: the live substrate swap, when both sides are
   * vizfx visuals. Within the family the look is one preference, not one per
   * visual — every visual ships the identical chassis grade, so what this
   * carries across is never a visual's art direction, it is whatever the human
   * has set the chain to *right now*. Without it, milkdrop mode's
   * section-boundary auto-advance reset exposure, target and τ to the shipped
   * defaults on every transition, which from the workbench reads as "the
   * sliders are stuck at the defaults" — every edit lived at most one section.
   *
   * Deliberately wins over the incoming visual's own autosaved render block
   * (merged one step earlier by `Modulator.setConfig`): during a cycling
   * session the thing on screen is the thing being tuned, and coming back to a
   * preset with the exposure you had ten seconds ago beats coming back to the
   * exposure you had last week. The merge is into the *same* `config.render`
   * object the modulator re-shared, so the carried values are also what the
   * next autosave serialises.
   *
   * Seeding the adapted gain is the same decision at the controller's
   * timescale: the incoming field starts empty, and a controller re-hunting
   * from 1× against black spends the first seconds racing to a rail (measured:
   * 1× → 32× over ~3 s after a swap). The outgoing gain is 1–2 frames stale —
   * the readback mirror — which is close enough for a starting point the
   * controller immediately refines.
   */
  adoptLook(other: VizFxSim): void {
    mergeRenderConfig(this.config.render, other.config.render);
    const p = this.config.params;
    const q = other.config.params;
    for (const key of ['exposure', 'gamma'] as const) {
      const v = q[key];
      if (typeof v === 'number' && Number.isFinite(v)) p[key] = v;
    }
    this.post.seedAutoExposure(other.post.autoExposureState.gain);
  }

  /**
   * The seeded emitter constellation: `emitPerLayer` clusters per layer, each
   * with its own starting phase and its own multipliers on the layer's radius,
   * speed and size.
   *
   * This is the part of a reroll that is *not* θ, and it is deliberately outside
   * it: an emitter's phase has no musical referent and no sensible range for a
   * slider, but scattering them differently is most of what makes one seed's
   * nebula a different creature from another's. It is drawn from the same PCG the
   * shaders use, so "seeded" means one thing across the whole project.
   *
   * The multiplier bands are narrow on purpose. At ±50% on radius the three
   * clusters of a layer stop reading as one constellation and start reading as
   * three unrelated objects, which loses the per-stem identity the whole keying
   * is for.
   */
  private redrawEmitters(): void {
    const k = this.config.speciesCount;
    for (let l = 0; l < k; l++) {
      for (let e = 0; e < MAX_EMITTERS_PER_LAYER; e++) {
        const h = hash3(this.seed ^ 0x7e0b_1a5f, l + 1, e + 1);
        const o = (l * MAX_EMITTERS_PER_LAYER + e) * FLOATS_PER_EMITTER;
        // Evenly spaced round the circle, then jittered: a purely random phase
        // clumps two of three clusters together about a third of the time, and a
        // clumped constellation warps into one arm instead of three.
        const spread = (e / Math.max(this.emitPerLayer, 1)) * Math.PI * 2;
        this.emitterData[o + 0] = spread + rand01(h) * 1.4;
        this.emitterData[o + 1] = 0.78 + rand01(pcg(h)) * 0.44;
        this.emitterData[o + 2] = 0.72 + rand01(pcg(pcg(h))) * 0.56;
        this.emitterData[o + 3] = 0.7 + rand01(pcg(pcg(pcg(h)))) * 0.6;
      }
    }
    // Pushed here rather than by every caller: `reseed` and `applyExtras` both
    // redraw, and an extras sync that changed the count without uploading would
    // leave the shader reading clusters that had never been placed.
    if (this.ready && this.ctx) this.uploadEmitters();
  }

  /**
   * Section-boundary re-seed: fade the field in proportion to `fraction` and
   * stamp seeded blotches of new light into it.
   *
   * The mapping layer calls this on every segment crossing, and the other two
   * substrates answer it by re-scattering matter. There is no matter here, so the
   * gesture is done to the picture instead — see the long note on `fsInject` in
   * chassis.wgsl for why that is the honest analogue rather than a stand-in.
   */
  partialReseed(key: number, fraction: number): void {
    if (!this.ready || !this.ctx || this.binds.length !== 2) return;
    const f = Math.min(Math.max(fraction, 0), 1);
    if (f <= 0) return;

    this.uploadTheta();
    this.uploadLayers();
    this.writeGlobals(f, key >>> 0);

    const { device } = this.ctx;
    const encoder = device.createCommandEncoder({ label: `${this.simId}.inject` });
    const p = encoder.beginRenderPass({
      label: 'inject',
      colorAttachments: [
        {
          view: this.fieldViews[1 - this.parity] as GPUTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    p.setPipeline(this.injectPipeline);
    p.setBindGroup(0, this.binds[this.parity] as GPUBindGroup);
    p.draw(3);
    p.end();
    device.queue.submit([encoder.finish()]);
    this.parity = 1 - this.parity;
    // Leave the uniform clean: every other pass shares this Globals block, and an
    // `injectAmount` left set would make the next composite read a stale one.
    this.writeGlobals(0, 0);
  }

  snapshot(): boolean {
    if (!this.ready || !this.ctx || this.fieldTex.length !== 2) return false;
    const { device } = this.ctx;
    if (!this.snap || this.snap.width !== this.fieldW || this.snap.height !== this.fieldH) {
      this.snap?.texture.destroy();
      this.snap = {
        texture: device.createTexture({
          label: `${this.simId}.snap`,
          size: { width: this.fieldW, height: this.fieldH },
          format: HDR_FORMAT,
          usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        }),
        width: this.fieldW,
        height: this.fieldH,
        parity: this.parity,
        seed: this.seed,
        steps: this.steps,
        stepAccumulator: this.stepAccumulator,
        stemLevel: new Float32Array(this.stemLevel.length),
        energy: new Float32Array(this.energy.length),
        takenAt: performance.now(),
      };
    }
    const snap = this.snap;
    snap.parity = this.parity;
    snap.seed = this.seed;
    snap.steps = this.steps;
    snap.stepAccumulator = this.stepAccumulator;
    snap.stemLevel.set(this.stemLevel);
    snap.energy.set(this.energy);
    snap.takenAt = performance.now();

    const encoder = device.createCommandEncoder({ label: `${this.simId}.snapshot` });
    encoder.copyTextureToTexture(
      { texture: this.fieldTex[this.parity] as GPUTexture },
      { texture: snap.texture },
      { width: this.fieldW, height: this.fieldH },
    );
    device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Restore the last snapshot exactly. Parameters are *not* restored — that is the A/B. */
  restoreSnapshot(): boolean {
    if (!this.ready || !this.ctx || !this.snap) return false;
    const snap = this.snap;
    if (snap.width !== this.fieldW || snap.height !== this.fieldH) return false;
    const { device } = this.ctx;

    const encoder = device.createCommandEncoder({ label: `${this.simId}.restore` });
    encoder.copyTextureToTexture(
      { texture: snap.texture },
      { texture: this.fieldTex[snap.parity] as GPUTexture },
      { width: snap.width, height: snap.height },
    );
    device.queue.submit([encoder.finish()]);

    this.parity = snap.parity;
    this.setSeed(snap.seed);
    this.steps = snap.steps;
    this.stepAccumulator = snap.stepAccumulator;
    // The energy posture comes back with the picture. Without this the lane would
    // immediately start pulling the restored field toward the *current* moment's
    // levels, and an A/B would be comparing two different arrangements as well as
    // two different parameter sets.
    this.stemLevel.set(snap.stemLevel);
    this.energy.set(snap.energy);
    this.energyPrimed = true;
    return true;
  }

  clearSnapshot(): void {
    if (!this.snap) return;
    this.snap.texture.destroy();
    this.snap = null;
  }

  // ── per-tick ───────────────────────────────────────────────────────────────

  /**
   * One app tick: on every second tick, advance the energy lane and run `speed`
   * warp+draw steps. Odd ticks deliberately leave this 60 Hz model untouched.
   *
   * The steps happen here rather than in `render` — which is where the other
   * feedback in this project (plife's render-domain echo) lives — and that is the
   * decision the whole substrate's determinism rests on. The field is the world,
   * so it must advance on the fixed clock: at 240 Hz it would otherwise decay and
   * zoom four times as fast as at 60, every per-step θ factor would mean a
   * different thing per machine, and a pinned seed would not reproduce a run.
   * plife can afford the other choice because its echo is decoration over a world
   * that is already stepping on the clock.
   *
   * One encoder and one submit per step, because `writeGlobals` moves between
   * them: the step clock is in the uniform, and two passes recorded before a
   * single submit would both read whichever value was written last.
   */
  tick(frame: FeaturesFrame, simTick: number): void {
    if (!this.ready || !this.ctx || this.binds.length !== 2) return;

    let steps = 0;
    if (this.config.paused) {
      if (this.pendingSingleStep) {
        this.pendingSingleStep = false;
        steps = 1;
      }
    } else {
      const next = advanceStepCadence(
        this.stepAccumulator,
        this.config.speed,
        MAX_SUBSTEPS,
        simTick,
        LEGACY_TICK_DIVISOR,
      );
      this.stepAccumulator = next.accumulator;
      steps = next.steps;
    }
    this.stepsThisTick = steps;

    // Normal odd app ticks intentionally do no model work at all. Keeping these
    // CPU smoothers on the same even-tick 60 Hz schedule as the field preserves
    // their sampled inputs and wall-clock response; a requested manual step is
    // allowed through immediately even if it lands on an odd app tick.
    const modelTick = simTick % LEGACY_TICK_DIVISOR === 0;
    if (!modelTick && steps === 0) return;

    // Ahead of the step loop and outside the paused branch: the energy lane is a
    // *smoother*, and freezing its EMA while the transport runs would mean
    // un-pausing snapped every layer to whatever the music had become.
    this.updateEnergy(frame, STEP_DT, !this.energyPrimed);
    this.energyPrimed = true;
    this.updatePulse();
    if (steps === 0) return;

    // Once per tick, not once per step: all three describe the state of this
    // tick's music, and every step of this tick paints under the same one — which
    // is what makes a splash a sustained ring for the length of its decay rather
    // than a single step's flicker.
    this.uploadTheta();
    this.uploadLayers();
    this.uploadSplashes();

    for (let s = 0; s < steps; s++) this.runStep();
  }

  private runStep(): void {
    const { device } = this.ctx as GpuContext;
    this.steps++;
    this.writeGlobals(0, 0);

    const dstView = this.fieldViews[1 - this.parity] as GPUTextureView;
    const bind = this.binds[this.parity] as GPUBindGroup;
    const encoder = device.createCommandEncoder({ label: `${this.simId}.step` });

    const warp = encoder.beginRenderPass({
      label: 'warp',
      colorAttachments: [
        { view: dstView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    warp.setPipeline(this.warpPipeline);
    warp.setBindGroup(0, bind);
    warp.draw(3);
    warp.end();

    // `load`, on the surface the warp just wrote, with additive blending: the
    // draw pass is new light *on top of* the resampled field, which is the whole
    // shape of the substrate. Note the bind group is still the SOURCE parity's —
    // the draw pass never samples the field, so binding the texture it is not
    // writing costs nothing and saves a second pair of bind groups.
    const draw = encoder.beginRenderPass({
      label: 'draw',
      colorAttachments: [
        { view: dstView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'load', storeOp: 'store' },
      ],
    });
    draw.setPipeline(this.drawPipeline);
    draw.setBindGroup(0, bind);
    draw.draw(3);
    draw.end();

    device.queue.submit([encoder.finish()]);
    this.parity = 1 - this.parity;
  }

  /**
   * Stems → per-layer presence. The geometry half of "an instrument arrived";
   * `StemFollow` in the mapping layer owns the light half.
   *
   * `snap` adopts the sampled level outright instead of easing into it, for the
   * two moments where the EMA's history belongs to a different part of the song:
   * the first tick, and a snapshot restore.
   */
  private updateEnergy(frame: FeaturesFrame, dt: number, snap: boolean): void {
    const cfg = this.config.energy;
    const k = this.config.speciesCount;
    if (this.stemOffset === null || !cfg.followStems) {
      this.energy.fill(1);
      return;
    }
    const tau = Math.max(cfg.smoothingMs, 1) / 1000;
    const alpha = snap ? 1 : dt > 0 ? 1 - Math.exp(-dt / tau) : 0;
    const floor = Math.min(Math.max(cfg.floor, 0), 1);
    const curve = Math.max(cfg.curve, 0.01);
    for (let i = 0; i < k; i++) {
      const raw = frame.values[this.stemOffset + i];
      const x = typeof raw === 'number' && Number.isFinite(raw) ? Math.min(Math.max(raw, 0), 1) : 0;
      const prev = this.stemLevel[i] as number;
      const next = prev + (x - prev) * alpha;
      this.stemLevel[i] = next;
      this.energy[i] = floor + (1 - floor) * Math.pow(next, curve);
    }
  }

  /**
   * The whole-field transient the warp reads: how hard the loudest thing is
   * currently being struck.
   *
   * A max rather than a sum, because it drives a *geometric* effect: two
   * simultaneous hits should shove the field once, hard, not twice as hard as
   * either. Sourced from `depositMul` — the lane the event responses call
   * "deposit", i.e. how much light this hit is worth — because that is the
   * quantity that already means "something struck the medium".
   */
  private updatePulse(): void {
    const imp = this.impulses;
    if (!imp) {
      this.pulse = 0;
      return;
    }
    let p = 0;
    for (let i = 0; i < this.config.speciesCount; i++) {
      p = Math.max(p, (imp.depositMul[i] ?? 1) - 1);
    }
    // Capped at 2 rather than at the 4 the shipped kick actually reaches. This
    // drives the warp's zoom kick, which compounds per step: at pulse 3 and the
    // shipped `pulseZoom` the field expanded ~3× over one envelope, which is not
    // a kick, it is a jump cut.
    this.pulse = Math.min(p, 2);
  }

  // ── uploads ────────────────────────────────────────────────────────────────

  private refreshPalette(): void {
    this.paletteDirty = false;
    for (let k = 0; k < this.config.speciesCount; k++) {
      const [r, g, b] = paletteLinear(this.config.palette, k, defaultVizFxColor(this.visual, k));
      this.paletteRgb[k * 3 + 0] = r;
      this.paletteRgb[k * 3 + 1] = g;
      this.paletteRgb[k * 3 + 2] = b;
    }
  }

  private uploadTheta(): void {
    effectiveTheta(this.visual, this.config.params, this.config.macros, this.effTheta);
    (this.ctx as GpuContext).device.queue.writeBuffer(this.thetaBuf, 0, this.effTheta);
  }

  /**
   * Per layer: the composed colour and the two impulse multipliers.
   *
   * The composition order is the one every substrate in this repo uses, and it is
   * what keeps the lanes from fighting: θ owns the base (`brightness`), the
   * stem-follow lane scales it, the pair is clamped, and the impulse flash
   * multiplies on top of the clamp — so a kick is still visible on a layer the
   * stem lane is holding at its floor.
   */
  private uploadLayers(): void {
    if (this.paletteDirty) this.refreshPalette();
    const imp = this.impulses;
    const k = this.config.speciesCount;
    for (let i = 0; i < k; i++) {
      const base = this.config.params[`layer${i}.brightness`] ?? 1;
      const follow = this.brightFollow?.[i] ?? 1;
      const lit = Math.min(Math.max(base * follow, 0), MAX_BRIGHTNESS) * (imp?.brightMul[i] ?? 1);
      const o = i * FLOATS_PER_LAYER;
      this.layerData[o + 0] = (this.paletteRgb[i * 3 + 0] as number) * lit;
      this.layerData[o + 1] = (this.paletteRgb[i * 3 + 1] as number) * lit;
      this.layerData[o + 2] = (this.paletteRgb[i * 3 + 2] as number) * lit;
      this.layerData[o + 3] = this.energy[i] as number;
      this.layerData[o + 4] = imp?.depositMul[i] ?? 1;
      this.layerData[o + 5] = imp?.sensorMul[i] ?? 1;
      this.layerData[o + 6] = 0;
      this.layerData[o + 7] = 0;
    }
    (this.ctx as GpuContext).device.queue.writeBuffer(this.layerBuf, 0, this.layerData);
  }

  private uploadEmitters(): void {
    (this.ctx as GpuContext).device.queue.writeBuffer(this.emitterBuf, 0, this.emitterData);
  }

  /**
   * The live shockwave list. The tint is resolved here rather than in the shader
   * because a response may target *all* species (`ResponseConfig.species < 0`),
   * and "the mean of the four layer colours" is a CPU-side answer.
   */
  private uploadSplashes(): void {
    if (this.paletteDirty) this.refreshPalette();
    const live = this.impulses?.splashes ?? [];
    const k = this.config.speciesCount;
    const n = Math.min(live.length, MAX_SPLASHES);
    for (let i = 0; i < n; i++) {
      const s = live[i];
      if (!s) continue;
      const o = i * FLOATS_PER_SPLASH;
      this.splashData[o + 0] = s.x;
      this.splashData[o + 1] = s.y;
      this.splashData[o + 2] = s.radius;
      this.splashData[o + 3] = s.strength;
      let r = 0;
      let g = 0;
      let b = 0;
      if (s.species >= 0 && s.species < k) {
        r = this.paletteRgb[s.species * 3 + 0] as number;
        g = this.paletteRgb[s.species * 3 + 1] as number;
        b = this.paletteRgb[s.species * 3 + 2] as number;
      } else {
        for (let c = 0; c < k; c++) {
          r += (this.paletteRgb[c * 3 + 0] as number) / k;
          g += (this.paletteRgb[c * 3 + 1] as number) / k;
          b += (this.paletteRgb[c * 3 + 2] as number) / k;
        }
      }
      this.splashData[o + 4] = r;
      this.splashData[o + 5] = g;
      this.splashData[o + 6] = b;
      this.splashData[o + 7] = 0;
      this.splashData[o + 8] = Math.min(Math.max(s.push / PUSH_TO_EXPAND, 0.15), 3);
      this.splashData[o + 9] = Math.min(Math.max(s.swirl, 0), 2);
      this.splashData[o + 10] = 0;
      this.splashData[o + 11] = 0;
    }
    this.splashCount = n;
    (this.ctx as GpuContext).device.queue.writeBuffer(
      this.splashBuf,
      0,
      this.splashData,
      0,
      Math.max(n, 1) * FLOATS_PER_SPLASH,
    );
  }

  private writeGlobals(injectAmount: number, injectKey: number): void {
    const f = this.globalsF32;
    const u = this.globalsU32;
    f[0] = this.fieldW;
    f[1] = this.fieldH;
    f[2] = 1 / Math.max(this.fieldW, 1);
    f[3] = 1 / Math.max(this.fieldH, 1);
    // Sim time, not wall clock: the emitters' orbits are a function of the step
    // count, so a paused transport freezes them and a pinned seed reproduces
    // exactly where every cluster was at every step.
    f[4] = this.steps * STEP_DT;
    f[5] = STEP_DT;
    u[6] = this.steps >>> 0;
    u[7] = this.seed >>> 0;
    u[8] = this.config.speciesCount >>> 0;
    u[9] = this.emitPerLayer >>> 0;
    u[10] = this.splashCount >>> 0;
    f[11] = this.pulse;
    f[12] = Math.max(this.config.params['exposure'] ?? 0.1, 0);
    f[13] = injectAmount;
    u[14] = injectKey >>> 0;
    u[15] = 0;
    (this.ctx as GpuContext).device.queue.writeBuffer(this.globalsBuf, 0, this.globalsBytes);
  }

  // ── per-frame ──────────────────────────────────────────────────────────────

  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    const ctx = this.ctx;
    if (!this.ready || !ctx) return;

    this.post.ensureSize(ctx.width, ctx.height);
    // After `post.ensureSize`, because the bind groups this rebuilds name
    // `post.autoBuffer` — which is allocated in `post.init` and stable across a
    // resize, so the order is belt and braces rather than load-bearing. The field
    // reallocation itself is not: a bind group naming a destroyed texture is a
    // validation error on the very next pass.
    this.ensureField(ctx.width, ctx.height);
    if (this.binds.length !== 2) return;

    // θ owns the display transfer function; PostFx only applies it.
    this.post.gamma = Math.min(Math.max(this.config.params['gamma'] ?? 2.2, 1), 3);
    // Exposure rides the Globals block and the composite is the only reader, so
    // this is what makes the exposure slider immediate rather than waiting for the
    // next tick's upload — which matters because a paused transport still renders.
    this.writeGlobals(0, 0);

    const pass = encoder.beginRenderPass({
      label: `${this.simId}.composite`,
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
    pass.setBindGroup(0, this.binds[this.parity] as GPUBindGroup);
    pass.draw(3);
    pass.end();

    this.post.run(encoder, targetView);
  }

  dispose(): void {
    this.releaseField();
    this.clearSnapshot();
    this.globalsBuf?.destroy();
    this.thetaBuf?.destroy();
    this.layerBuf?.destroy();
    this.emitterBuf?.destroy();
    this.splashBuf?.destroy();
    this.post.dispose();
    this.ready = false;
    this.ctx = null;
  }
}

// ── small helpers, shared with the extras loader ──────────────────────────────

function rand01(h: number): number {
  return (h >>> 0) * 2.3283064365386963e-10;
}

function plainObject(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
}

function clampNum(v: unknown, fallback: number, min: number, max: number): number {
  const x = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  return x < min ? min : x > max ? max : x;
}

function readBool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback;
}
