/**
 * Particle life on the GPU: uniform-grid neighbour search, pairwise tent forces,
 * HDR sprite splats with a render-domain trail.
 *
 * Per substep:
 *
 *   clearBuffer(cellCount, cellFill)          — no shader, no dispatch
 *   splashParticles   one thread / particle   → src velocities  [only if discs]
 *   countParticles    one thread / particle   → cellCount
 *   scanCells         one workgroup           → cellStart (exclusive prefix sum)
 *   scatterParticles  one thread / particle   → sortedIdx
 *   stepParticles     one thread / particle   → dst particles  [flip parity]
 *
 * Per frame:
 *
 *   fade      full-screen, previous HDR × feedback, into a cleared HDR target
 *   particles instanced quads, additive, on top
 *   post.run  measure → bloom → grade → swapchain
 *
 * ## Performance, stated up front
 *
 * The force pass is the whole cost and it scales as
 * `alive × (particles per (2s+1)² neighbourhood)`, where `s` is
 * `field.nearStencil`. With the shipped defaults —
 * 2²⁰ pool, primaries at aliveFraction 0.5, accents at 0.12 — that is roughly
 * 325 k live particles in a grid of ~700 cells, so ~460 per cell and ~4 200
 * candidates tested per particle per substep: about 1.4 × 10⁹ candidate tests
 * per substep, 8 × 10¹⁰ per second at 60 fps. **That will not hold 60 fps on
 * most hardware.** The levers, in the order worth pulling:
 *
 *   1. `maxParticles` — quadratic in effect (fewer particles *and* shorter cell
 *      lists). 2¹⁸ is a 16× reduction in pair work.
 *   2. `aliveFraction` per species — same lever, per voice, and it is in θ. Note
 *      that the population lane multiplies *below* 1 most of the time (a stem at
 *      its floor keeps 0.35 of its colony), so the figure above is the worst
 *      case, reached only when every instrument is at full.
 *   3. `field.nearStencil` — the search window is (2s+1)² cells, so 3 costs 5.4×
 *      what 1 does and 2 costs 2.8×. This is the lever that used to be `R_CAP`,
 *      and it is now a slider rather than a grid rebuild, because the cell size
 *      (occupancy) and the reach (stencil × cell) are separate numbers.
 *
 * None of them are changed here: the brief specifies these defaults, they are
 * the right *shape*, and the number that matters can only be found by running
 * it. This comment exists so the first person to see 8 fps knows exactly which
 * three numbers to touch.
 *
 * ## The other lane: `field.pairSearch = 'brute'`
 *
 * All of the above describes the GRID lane, whose whole reason to exist is to
 * make the neighbour set smaller than "everyone". The brute lane gives that up
 * on purpose: every live particle is tested against every other, cost
 * `alive²`, and in exchange the reach cap goes from `stencil × cell` (≤ 0.06)
 * to half the torus (0.5). That is not a tuning difference. Below 0.06 a
 * cluster is an emergent agreement between many filigree strands; at 0.25 it is
 * something the force law itself knows about, and continent-scale flows become
 * available at all.
 *
 * The cost is honest and quadratic: at ~33 k alive it is ~1.1 × 10⁹ pair tests
 * per substep, which is the same order as the *grid* lane's worst case at 325 k
 * and roughly 100× its cost at 33 k. `maxParticles` is the lever, and it is now
 * a much sharper one — halving the pool quarters the work rather than halving
 * it.
 */
import type { GpuContext } from '../../gpu/context';
import type { ModTarget, ThetaRegistry } from '../../mapping/target';
import type { FeaturesFrame } from '../../timeline/sampler';
import { MAX_SPLASHES, type ImpulseState } from '../impulses';
import { paletteLinear } from '../palette';
import { HDR_FORMAT, PostFx } from '../render/postfx';
import type { Sim } from '../types';
import {
  defaultMatrixGen,
  defaultPlifeConfig,
  defaultPlifeField,
  defaultPlifeMacros,
  defaultPlifePaletteColor,
  defaultPlifePopulation,
  FAR_GAIN_RANGE,
  FAR_SCALE_MIN_RATIO,
  FAR_SCALE_RANGE,
  FAR_SCALE_REF,
  MACRO_RANGE,
  MAX_BRIGHTNESS,
  MAX_FRICTION,
  MAX_MIN_R,
  MAX_NEAR_STENCIL,
  MAX_RADIUS_SCALE,
  MAX_REACH_BRUTE,
  MAX_SIZE,
  MAX_STRETCH,
  MAX_SUBSTEPS,
  MIN_R_FLOOR,
  PAIR_SEARCH_MODES,
  PRIMARY_COUNT,
  R_CAP,
  type PairSearch,
  type PlifeConfig,
  type PlifeMacros,
} from './config';
import { seedMatrixBase } from './genmatrix';
import {
  applyVector,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  presetFromConfig,
  presetToVector,
  vectorLength,
} from './preset';

import commonWgsl from './shaders/common.wgsl?raw';
import gridWgsl from './shaders/grid.wgsl?raw';
import stepWgsl from './shaders/step.wgsl?raw';
import renderWgsl from './shaders/render.wgsl?raw';
import farWgsl from './shaders/far.wgsl?raw';

/** One thread per particle for the force pass; 64 keeps occupancy up on the long inner loop. */
const PARTICLE_WORKGROUP = 64;
/** The grid passes are memory-bound and trivially parallel, so they go wide. */
const GRID_WORKGROUP = 256;
/** must match the Species struct in common.wgsl */
const FLOATS_PER_SPECIES = 16;
/** must match the Splash struct in common.wgsl: two vec4f */
const FLOATS_PER_SPLASH = 8;
/** must match the Globals struct in common.wgsl, padding included */
const GLOBALS_WORDS = 28;
/**
 * Height of the far lane's per-species density target, in texels; the width
 * follows the world's aspect so the texels stay square.
 *
 * 144 is a deliberate *under*-sampling of the screen and that is the point: this
 * field is only ever read through Gaussians of σ ≥ 0.02 world (≈ 3 texels), so
 * anything finer would be blurred away in the first pass having cost bandwidth
 * to splat. It is also fine enough that the smallest σ2 the panel allows (0.05
 * world ≈ 7 texels) still resolves as a Gaussian rather than as a box.
 *
 * The whole chain is 8 textures at this size — at 16:9, 256×144×8 B × 8 = 2.4 MB,
 * which fits in L2 on anything modern and is why a four-pass full-resolution
 * blur is cheaper here than the pyramid it replaces.
 */
const FAR_HEIGHT = 144;
/**
 * Gaussian taps each side of centre, per blur pass. Beyond this the tap *stride*
 * grows instead of the tap count, so the cost of the σ2 pass is bounded no
 * matter how far the far-scale knob is dragged.
 *
 * Striding is safe here because the σ2 pass runs on the output of the σ1 pass:
 * its source is already band-limited to σ1 ≥ ~3 texels, so a stride under about
 * that is not sampling anything the source still contains. At the very top of
 * the range (σ2 = 0.5) the stride reaches ~3.4 texels, which is marginal at
 * stencil 1 and comfortable at 2 or 3 — and the field being estimated there is a
 * half-screen blur, where a percent of ringing is not a thing anyone can see.
 */
const FAR_MAX_HALF_TAPS = 64;
/** blur passes in the chain: σ1 X, σ1 Y, σ2-residual X, σ2-residual Y + DoG */
const FAR_BLUR_PASSES = 4;
/**
 * Converts a normalised density gradient into a force in the near lane's units.
 *
 * This is the one empirical constant in the far lane, and it is empirical
 * because the quantity it scales — how steep a *relative* density gradient the
 * sim's own clustering produces — is a property of the world, not of the
 * arithmetic. Everything else in `uploadFar` is a normalisation with a
 * derivation.
 *
 * ## How it was sized, and why not the way it was planned
 *
 * The plan was to set `farGain` 1 at a terminal drift of 0.02–0.05 world/s
 * (terminal speed is `|F| · forceGain · forceScale / friction`). That was
 * measured directly — splat and blur the live particle buffer on the CPU, apply
 * the same coefficients, take the gradient — and at 5e-4 it landed exactly in
 * that band: median drift 0.008, p90 0.024, p99 0.057 world/s.
 *
 * And it was invisible. The reason is a number the budget did not account for:
 * the near lane's own motion. Particles in a clumped world run at a median
 * speed of 0.13–0.26 world/s, so their random walk has a diffusivity that
 * flattens a 0.03 world/s current before it can move anything by σ2. Measured
 * on a fixed snapshot, 6000 substeps, as the coarse-grained per-species density
 * contrast (σ = 0.2, std/mean — it rises when a species occupies domains rather
 * than the whole world):
 *
 *   farGain (at 5e-4)   ×1     ×3     ×6     ×10    ×20    ×30
 *   contrast            0.034  0.039  0.059  0.064  0.178  0.256
 *   speed-clamp frac    0.475  0.469  0.488  0.485  0.571  0.615
 *
 * Baseline contrast with the lane off is 0.034, and the near lane alone already
 * pins ~48% of particles at `maxSpeed`, so "does the clamp saturate" cannot be
 * the test here — the clamp is already saturated by the thing the far lane is
 * meant to sit under.
 *
 * 4e-3 puts `farGain` 1 at the ×8 column: contrast roughly doubled, no measured
 * change in clamp saturation, and a median far drift of ~0.06 world/s against a
 * near-lane median speed of ~0.26. That preserves the *intent* of the budget —
 * the far field is a current the near field's structures are carried on, never
 * the thing shaping them — while being a current you can see. The panel's
 * ceiling of 3 reaches the ×24 column, which is the strongly-segregating,
 * clamp-pushing regime a maximum ought to be.
 */
const FAR_FORCE_SCALE = 4e-3;
/** the stems channel's width, by analysis contract — bass, drums, vocals, other */
const STEM_DIMS = 4;
/** must match `struct Particle` in common.wgsl: two vec2f + four f32 */
const BYTES_PER_PARTICLE = 32;
/**
 * Ceiling on the neighbour grid. `scanCells` is a single-workgroup scan of 256
 * elements per iteration, so this is 64 iterations at worst — cheap, but the
 * bound has to exist because the pass has no way to grow past one workgroup.
 */
const MAX_CELLS = 16384;
/**
 * Seconds a single substep advances the world. Constant, not derived from the
 * frame time: `config.speed` means "substeps per clock tick", so raising it
 * makes the world run faster rather than making each step coarser, and the
 * integrator's stability margin never depends on frame rate.
 */
const SUBSTEP_DT = 1 / 60;
/**
 * Seconds one *clock* tick covers, for the population lane's EMAs. `Sim.tick`
 * is handed a frame and a tick index but no dt — the whole app runs on a fixed
 * 1/60 s timestep (main.ts's `SECONDS_PER_TICK`) and the sim's own substep dt is
 * the same number — so this is stated as a constant rather than threaded through
 * the interface for one lane. If the app's timestep ever becomes configurable,
 * this is the line that has to follow it.
 */
const TICK_DT = 1 / 60;

/**
 * The sim's whole persistent state on a spare buffer. Just the particles: the
 * grid is rebuilt from scratch every substep and carries nothing across, and the
 * HDR echo is render-domain — deliberately *not* part of sim state, so an A/B
 * restore reproduces the world rather than the screen.
 */
interface PlifeSnapshot {
  particles: GPUBuffer;
  seed: number;
  stepAccumulator: number;
  lastPcgTick: number;
  /** which ping-pong slot the copy came from; restore puts it back in the same one */
  parity: number;
  /**
   * The population lane's CPU state — the smoothed stem levels and the two
   * multiplier arrays derived from them, plus the accent EMA. Particle *energy*
   * rides along inside the particle buffer for free; these do not, and leaving
   * them out would mean a restore reproduced the world's matter while the lane
   * kept driving it from a posture that belongs to some later moment of the
   * song. Same reasoning as stem-follow's snap-on-seek.
   */
  stemLevel: Float32Array;
  popMul: Float32Array;
  accentMul: Float32Array;
  accentActivity: number;
  /** wall-clock label for the workbench; not part of sim state */
  takenAt: number;
}

export interface PlifeStats {
  gridW: number;
  gridH: number;
  totalParticles: number;
  aliveParticles: number;
  stepsThisFrame: number;
  /** render passes in the post chain last frame; fade + particles are the +2 */
  renderPasses: number;
}

export class PlifeSim implements Sim, ModTarget {
  readonly name = 'plife';
  /**
   * The persistence discriminator (`ModTarget.simId`). θ here is a different
   * vector from physarum's, so the two must never share an autosave slot.
   */
  readonly simId = 'plife';
  readonly config: PlifeConfig;

  /** Notified on every seed change — a reseed, and equally a snapshot restore. */
  onSeedChange: ((seed: number) => void) | null = null;

  /**
   * Force the grid pair search on this instance whatever the config says.
   * **Explorer tiles only** — `main.ts` sets it in `createExplorerTile`.
   *
   * Measured (2026-08-09, 13 k alive): nine brute tiles run correctly and look
   * right, at ~212 ms/frame — 4.7 fps, which is not a search tool, it is a
   * slideshow. Nine times an O(N²) pass is nine times a cost that is already the
   * whole frame, and the explorer's job is to answer "which of these nine" in
   * seconds.
   *
   * Why a flag rather than a value on the tile's cloned config: `syncStyle` fans
   * the live sim's serialised extras into all nine tiles twice a second, so
   * anything set on the clone at construction is overwritten within 500 ms. This
   * sits *outside* the extras channel, which is the only place it cannot be
   * undone by a sync.
   *
   * The honest cost of the choice: a tile in grid mode shows the same θ through
   * a 0.06 reach cap, so at large radii the nine tiles under-represent what the
   * full-size sim will do with the candidate you pick. Stated in the explorer
   * folder's own label rather than left as a surprise.
   */
  forceGridSearch = false;

  /** The pair search actually in force — `forceGridSearch` outranks the config. */
  private pairSearch(): PairSearch {
    return this.forceGridSearch ? 'grid' : this.config.field.pairSearch;
  }

  /** The phase-7 render chain. Owns the HDR surfaces, bloom, grading, auto-exposure. */
  readonly post: PostFx;

  private ctx: GpuContext | null = null;
  private ready = false;
  private seed: number;

  /** World space: h = 1, w = aspect. Fixed for the sim's life — see `init`. */
  private worldW = 1;
  private worldH = 1;
  private gridW = 0;
  private gridH = 0;
  private cellW = R_CAP;
  private cellH = R_CAP;
  private segSize = 0;
  private totalParticles = 0;
  private activeTotal = 0;

  /**
   * `config.field.nearStencil`, clamped to what this grid can actually search.
   *
   * The second clamp is the one that is not obvious: the force pass wraps its
   * cell indices toroidally, so a stencil wide enough to reach past the grid
   * would visit the same cell twice in one sweep and count every pair in it
   * twice. `(min(gridW, gridH) - 1) / 2` is the widest window that cannot. It
   * never binds at any sane aspect (a 16:9 world is ~88×50 cells) but it is a
   * correctness bound, so it is enforced rather than assumed.
   */
  private stencil(): number {
    const want = Math.round(this.config.field.nearStencil);
    const room = Math.floor((Math.min(this.gridW, this.gridH) - 1) / 2);
    return Math.max(1, Math.min(want, MAX_NEAR_STENCIL, Math.max(room, 1)));
  }

  /**
   * The reach cap the shader will apply, in the mode currently selected:
   * `stencil × cell` for the grid walk, half the short world axis for brute.
   * Public so the panel can state it honestly in a label — it is derived from
   * config knobs and a grid the panel does not own, and re-deriving it there
   * would be the third place this arithmetic lives.
   */
  get nearReach(): number {
    if (this.pairSearch() === 'brute') {
      return Math.min(MAX_REACH_BRUTE, 0.5 * Math.min(this.worldW, this.worldH));
    }
    return this.stencil() * Math.min(this.cellW, this.cellH);
  }

  /**
   * The largest effective outer radius any pair actually holds right now,
   * `max_ij(maxR[i][j] · radiusScale_i · reach)`, clamped to the mode's cap.
   *
   * Distinct from `nearReach`, which is the *permission*: with default radii of
   * ~0.01 the brute lane's cap is 0.5 and this is 0.03, and reporting 0.5 as
   * "the reach" would be a lie in the one place the workbench needs the truth —
   * the far lane's seam (below) and the panel's label.
   */
  get liveReach(): number {
    const cap = this.nearReach;
    const k = this.config.speciesCount;
    const reachMacro = Math.max(this.config.macros.reach, 0);
    let widest = 0;
    for (let i = 0; i < k; i++) {
      const scale = Math.min(
        Math.max(this.config.species[i]?.radiusScale ?? 1, 0.05) * reachMacro,
        MAX_RADIUS_SCALE,
      );
      for (let j = 0; j < k; j++) {
        widest = Math.max(widest, (this.config.maxR[i * k + j] ?? 0) * scale);
      }
    }
    return Math.min(widest, cap);
  }

  /**
   * σ1, the far lane's inner scale: where the near lane stops and the far lane
   * is allowed to start.
   *
   * In **grid** mode this is exactly what it always was — `stencil × cell`, the
   * lane's cap — and deliberately not re-derived from the live radii, because
   * grid mode is the regression baseline and its far lane may not shift under a
   * change that is about brute mode.
   *
   * In **brute** mode the cap is a permission (0.5, half the torus) rather than
   * a description, so the seam follows the *live* widest radius instead: put at
   * the cap it would sit far outside anything the near lane is actually doing
   * and the far lane would be describing a world that is not there.
   *
   * The ceiling is the part worth stating. σ2 is floored at
   * `FAR_SCALE_MIN_RATIO × σ1`, so an unbounded σ1 would push σ2 outside the
   * range its own slider covers, and at σ1 near half the world both Gaussians
   * are flat and the DoG is a difference of two nothings. Capping σ1 at
   * `FAR_SCALE_RANGE.max / FAR_SCALE_MIN_RATIO` keeps the band inside the
   * slider's own interval at every reach. Past that point the honest statement
   * is that the near lane *is* the far lane, and the DoG narrowing toward zero
   * as reach grows is that statement made arithmetically.
   */
  private get farSigma1(): number {
    if (this.pairSearch() !== 'brute') {
      return this.stencil() * Math.min(this.cellW, this.cellH);
    }
    return Math.min(this.liveReach, FAR_SCALE_RANGE.max / FAR_SCALE_MIN_RATIO);
  }

  /**
   * σ2 as the shader will actually use it: the knob, floored at
   * `FAR_SCALE_MIN_RATIO × σ1`.
   *
   * The floor is not a taste clamp. The far term is `G_σ1*ρ − G_σ2*ρ`, and at
   * σ2 ≤ σ1 that difference is zero or *inverted* — the lane would silently
   * reverse its sign rather than fade out. Public so the panel can show the
   * effective value when the two knobs disagree.
   */
  get farSigma(): number {
    return Math.max(this.config.field.farScale, FAR_SCALE_MIN_RATIO * this.farSigma1);
  }

  /** Is the far lane doing anything? Gates the whole chain and the shader's sampling. */
  private farActive(): boolean {
    return this.ready && this.config.field.farGain > 0;
  }

  /**
   * Live impulse lane, owned by the ImpulseEngine and mutated in place. Read,
   * never written, and applied *after* whatever set the base parameters — the
   * modulator's slew limiter included — so transients are never smoothed away.
   *
   * All four lanes are wired: the three multipliers in `uploadSpecies`, and the
   * splash discs through `uploadSplashes` into the `splashParticles` pass.
   */
  private impulses: ImpulseState | null = null;
  /** Per-species brightness multiplier from the stem-follow lane, held by reference. */
  private brightFollow: Float32Array | null = null;

  // ── population lane state (CPU-owned; see `updatePopulation`) ───────────────
  //
  // This is sim-owned art direction, not a mapping-layer lane. It duplicates
  // StemFollow's arithmetic on purpose and does NOT reuse the class: that
  // instance belongs to the Modulator, its knobs live in ModulationConfig, and
  // it drives *brightness*. Population has to keep working with the Modulator
  // absent (manual mode, a track with no drivers, the future plife panel), it
  // wants its own much slower time constant, and it must survive a snapshot
  // restore alongside the particles rather than alongside the mapping. Sharing
  // one object across those two owners would couple the sim's population to
  // whether anything is modulating it, which is the wrong dependency.

  /** null = no usable stems channel; the lane is off and every popMul stays 1 */
  private stemOffset: number | null = null;
  /** null = channel absent; contributes 0 to the accent activity */
  private noveltyOffset: number | null = null;
  private chorusOffset: number | null = null;
  /** length K, the smoothed stem level behind each popMul */
  private readonly stemLevel: Float32Array;
  /** length K, 1.0 = untouched — stems → colony size */
  private readonly popMul: Float32Array;
  /** length K, 1.0 = untouched — novelty → accent colony size, secondaries only */
  private readonly accentMul: Float32Array;
  /** scalar EMA of max(novelty16, actChorus); the accents share one activity */
  private accentActivity = 0;
  /**
   * False until the first tick, which snaps the EMAs to the music instead of
   * easing into them from zero. Without it the world opens at θ's full base
   * population (that is what `init` seeds), then spends the smoothing constant
   * discovering that the track is quiet and sheds two thirds of itself — a slow
   * unexplained collapse as the first thing anyone sees.
   */
  private popPrimed = false;
  /** length K, the integer population target `uploadSpecies` last wrote */
  private readonly targetAlive: Uint32Array;

  // ── far field (the density pyramid) ────────────────────────────────────────
  //
  // Sized from the WORLD's aspect, not the canvas's, and therefore fixed for the
  // sim's life exactly like the neighbour grid: a canvas resize re-allocates the
  // post surfaces and leaves these alone, because the field is a property of the
  // world being simulated rather than of the window it is being watched through.
  // Each sim instance owns its own set, which is what makes the explorer's nine
  // tiles work with no extra plumbing — a tile is a PlifeSim with a smaller ctx.
  private farW = 0;
  private farH = 0;
  /** the four stages, two RGBA16F textures each (K = 8 species, 4 per texture) */
  private farSplat: GPUTexture[] = [];
  private farTmp: GPUTexture[] = [];
  private farNear: GPUTexture[] = [];
  private farDog: GPUTexture[] = [];
  private farSampler!: GPUSampler;
  private splatPipeline!: GPURenderPipeline;
  private blurPipeline!: GPURenderPipeline;
  private splatBinds: GPUBindGroup[] = [];
  private blurBinds: GPUBindGroup[] = [];
  private blurParamBufs: GPUBuffer[] = [];
  /** per pass: the two views it renders into, in attachment order */
  private blurTargets: GPUTextureView[][] = [];
  private farSplatViews: GPUTextureView[] = [];
  private readonly blurParams = new ArrayBuffer(32);
  private readonly blurParamsF32 = new Float32Array(this.blurParams);
  private readonly blurParamsU32 = new Uint32Array(this.blurParams);

  private globalsBuf!: GPUBuffer;
  private speciesBuf!: GPUBuffer;
  private interactionBuf!: GPUBuffer;
  /** ping-pong; `parity` names the source, `1 - parity` the destination */
  private particleBuf: GPUBuffer[] = [];
  private cellCountBuf!: GPUBuffer;
  private cellFillBuf!: GPUBuffer;
  private cellStartBuf!: GPUBuffer;
  private sortedIdxBuf!: GPUBuffer;
  private splashBuf!: GPUBuffer;

  private countPipeline!: GPUComputePipeline;
  private scanPipeline!: GPUComputePipeline;
  private scatterPipeline!: GPUComputePipeline;
  private stepPipeline!: GPUComputePipeline;
  /** The same module and the same layout, specialised with `BRUTE = 1`. */
  private stepBrutePipeline!: GPUComputePipeline;
  private initPipeline!: GPUComputePipeline;
  private respawnPipeline!: GPUComputePipeline;
  private splashPipeline!: GPUComputePipeline;
  private fadePipeline!: GPURenderPipeline;
  private particlePipeline!: GPURenderPipeline;

  private gridBinds: GPUBindGroup[] = [];
  private stepBinds: GPUBindGroup[] = [];
  /**
   * The one bind group that binds a particle buffer read_write at *source*
   * parity. Indexed by `parity` like the others, so it always names the buffer
   * the grid and force passes are about to read.
   */
  private splashBinds: GPUBindGroup[] = [];
  private particleBinds: GPUBindGroup[] = [];
  /**
   * One per PostFx ping-pong parity: the fade pass reads the *other* HDR surface,
   * so which bind group is correct alternates with the frame. Rebuilt whenever
   * PostFx reallocates (canvas resize).
   */
  private fadeBinds: GPUBindGroup[] = [];
  private fadeLayout!: GPUBindGroupLayout;
  private fadeSizeVersion = -1;

  private readonly globalsBytes = new ArrayBuffer(GLOBALS_WORDS * 4);
  private readonly globalsU32 = new Uint32Array(this.globalsBytes);
  private readonly globalsF32 = new Float32Array(this.globalsBytes);
  /**
   * Linearised palette (3 floats per species). Static art direction whose source
   * of truth is a set of CSS strings, re-derived only when something says they
   * changed — `uploadSpecies` runs on every substep and re-parsing there was
   * pure waste.
   */
  private readonly paletteRgb: Float32Array;
  private paletteDirty = true;
  private speciesData!: Float32Array<ArrayBuffer>;
  private interactionData!: Float32Array<ArrayBuffer>;
  private readonly splashData = new Float32Array(MAX_SPLASHES * FLOATS_PER_SPLASH);
  private splashCount = 0;

  /**
   * `performance.now()` at the last `render`, and the frame length that follows
   * from it expressed in 60 Hz frames. The feedback lane is the only thing that
   * reads them — see the note in `render` — and they are fields rather than
   * locals because `writeGlobals` is called from the substep path too and has to
   * publish the same corrected numbers there.
   */
  private lastRenderAt = 0;
  private renderDtFrames = 1;

  private parity = 0;
  private stepAccumulator = 0;
  private pendingSingleStep = false;
  private lastPcgTick = 0;
  private stepsThisFrame = 0;
  private respawnFraction = 0;
  private respawnKey = 0;
  private snap: PlifeSnapshot | null = null;

  /** θ registry and stem keying, both pure functions of K — built once, never rebuilt. */
  private theta: ThetaRegistry | null = null;
  private species2stem: Int32Array | null = null;

  constructor(seed: number, config: PlifeConfig = defaultPlifeConfig()) {
    this.seed = seed >>> 0;
    this.config = config;
    this.paletteRgb = new Float32Array(config.speciesCount * 3);
    const k = Math.max(1, config.speciesCount);
    this.stemLevel = new Float32Array(k);
    this.popMul = new Float32Array(k).fill(1);
    this.accentMul = new Float32Array(k).fill(1);
    this.targetAlive = new Uint32Array(k);
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

  /** The only place `seed` is assigned after construction, so listeners cannot be missed. */
  private setSeed(seed: number): void {
    this.seed = seed >>> 0;
    this.onSeedChange?.(this.seed);
  }

  /**
   * Tell the sim its colours changed. Everything that edits `config.palette` has
   * to call this; nothing polls, because the palette is static by design and a
   * poll would put the string parsing back in the hot path.
   */
  invalidatePalette(): void {
    this.paletteDirty = true;
  }

  private refreshPalette(): void {
    this.paletteDirty = false;
    for (let k = 0; k < this.config.speciesCount; k++) {
      const [r, g, b] = paletteLinear(this.config.palette, k, defaultPlifePaletteColor(k));
      this.paletteRgb[k * 3 + 0] = r;
      this.paletteRgb[k * 3 + 1] = g;
      this.paletteRgb[k * 3 + 2] = b;
    }
  }

  /** Hand the sim the impulse engine's live state object; null leaves every multiplier at 1. */
  setImpulses(state: ImpulseState | null): void {
    this.impulses = state;
  }

  /**
   * Bind the timeline's stems channel for the population lane. Mirrors
   * physarum's setter exactly, including the failure mode: a missing or
   * too-narrow channel disables the lane rather than reading whatever floats sit
   * at another channel's offset, and says so once.
   *
   * Note that plife now reads the stems channel *twice*, for two unrelated
   * things: here for population, and inside the Modulator's StemFollow for
   * brightness. That is intended — see the ownership note on the lane's fields.
   */
  setStemChannel(channel: { offset: number; dims: number } | undefined): void {
    if (!channel || channel.dims < STEM_DIMS) {
      this.stemOffset = null;
      this.stemLevel.fill(0);
      this.popMul.fill(1);
      this.config.population.followStems = false;
      console.warn(
        `plife: no usable stems channel (${
          channel ? `${channel.dims} dims, need ${STEM_DIMS}` : 'absent'
        }); population stem-follow disabled`,
      );
      return;
    }
    this.stemOffset = channel.offset;
  }

  /**
   * Bind the two structure channels the accent population reads. Either may be
   * absent, in which case it contributes 0 and the other still drives the lane;
   * with both absent the accents sit at their θ base forever, which is the
   * phase-before-this behaviour.
   *
   * Why these two and why direct: novelty is "something changed" and actChorus
   * is "this is the big part", and their max is as close as the timeline gets to
   * "the section just arrived and it matters". The seeded projections can give
   * the accents character but cannot promise that any particular seed spends it
   * on the chorus; this wire can, and does, on every seed.
   */
  setAccentChannels(
    novelty16: { offset: number; dims: number } | undefined,
    actChorus: { offset: number; dims: number } | undefined,
  ): void {
    this.noveltyOffset = novelty16 && novelty16.dims >= 1 ? novelty16.offset : null;
    this.chorusOffset = actChorus && actChorus.dims >= 1 ? actChorus.offset : null;
    if (this.noveltyOffset === null && this.chorusOffset === null) {
      this.accentActivity = 0;
      this.accentMul.fill(1);
      console.warn('plife: neither novelty16 nor actChorus is present; accent population idle');
    }
  }

  /**
   * The population lane's live state, by reference and read-only by convention —
   * the arrays are reused every tick, not copied, because the future parameter
   * panel wants to *watch* them and a per-frame allocation for a readout is
   * exactly the kind of waste the render loop cannot afford.
   */
  popState(): { target: Uint32Array; popMul: Float32Array; accentMul: Float32Array } {
    return { target: this.targetAlive, popMul: this.popMul, accentMul: this.accentMul };
  }

  /** Hand the sim the stem-follow lane's live multiplier array (length K), by reference. */
  setBrightFollow(values: Float32Array | null): void {
    this.brightFollow = values;
  }

  requestSingleStep(): void {
    this.pendingSingleStep = true;
  }

  // ── ModTarget: θ, as the mapping layer sees it ─────────────────────────────

  /**
   * The slot table, precomputed. Every view is a pure function of K, which is
   * fixed for the life of the sim, so this is built on first ask and then handed
   * out **by identity** — `Modulator.setConfig` passes `.classes` straight into a
   * SlewLimiter that holds it by reference, so returning a fresh object here
   * would quietly detach the limiter from the registry.
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
      // The three K² blocks are not jittered defaults, they are drawn from the
      // seed — see genmatrix.ts. Bound to `this.config` rather than to a copy so
      // that a live edit of `matrixGen` is what the next draw uses.
      seedBase: (seed, base) => seedMatrixBase(seed, this.config, base),
    };
    return this.theta;
  }

  currentVector(): Float64Array {
    return presetToVector(presetFromConfig(this.config), this.config.speciesCount);
  }

  /**
   * θ → the live config, plus the GPU state derived from θ rather than re-read
   * from it every step. The species block is picked up by `uploadSpecies` on the
   * next substep; the three K² interaction blocks live in their own buffer and
   * have to be pushed, which is what this pairing owns so no caller has to
   * remember it.
   */
  applyTheta(v: ArrayLike<number>, mask?: Uint8Array): void {
    applyVector(this.config, v, mask);
    this.uploadInteractions();
  }

  /**
   * Species k → stem k for the primaries, and species 4+n → stem n for the
   * accents: an accent is *the same instrument*, so it must brighten with it.
   * That is the one place plife's keying differs from physarum's (which keys
   * only its first four species and leaves the rest at -1).
   */
  stemMap(): Int32Array {
    if (this.species2stem) return this.species2stem;
    const k = this.config.speciesCount;
    const map = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      map[i] = i < 2 * PRIMARY_COUNT ? i % PRIMARY_COUNT : -1;
    }
    this.species2stem = map;
    return map;
  }

  // ── ModTarget: the opaque per-sim extras block ─────────────────────────────
  //
  // Three blocks of plife's config are outside θ *and* outside everything the
  // mapping layer knows how to carry: the macro rig, the matrix generation
  // settings, and the population lane. None belongs in `ModulationConfig`'s
  // schema — that file describes a mapping, not a substrate — so they travel in
  // the opaque `extras` channel and this pair is the only code that understands
  // their shape.
  //
  // The channel has two consumers, and the second is why `population` is here at
  // all. The first is persistence (autosave / `modulation.json`). The second is
  // explorer mode: `ExplorerRig.syncStyle` fans the serialised block out to all
  // nine tiles twice a second, so anything outside both θ and extras is a panel
  // edit the tiles never see — which is exactly how the population sliders came
  // to do nothing while the grid was open.

  /** A plain snapshot of everything plife wants saved outside θ. */
  serializeExtras(): Record<string, unknown> {
    const m = this.config.macros;
    const g = this.config.matrixGen;
    const p = this.config.population;
    return {
      macros: { ...m },
      matrixGen: { ...g, rMin: { ...g.rMin }, rMax: { ...g.rMax } },
      population: { ...p, accent: { ...p.accent } },
      field: { ...this.config.field },
      // Length K, index-aligned with `config.species`. A flat boolean array
      // rather than a list of names, because the species *are* their indices
      // everywhere else in this sim (the matrix, the palette, the stem map), and
      // a name list would invite the question of what happens when K changes —
      // which `applyExtras` answers by length instead.
      speciesEnabled: this.config.species
        .slice(0, this.config.speciesCount)
        .map((s) => s.enabled !== false),
    };
  }

  /**
   * The inverse, and deliberately paranoid: `extras` is opaque to every layer
   * between the file and here, so nothing upstream has validated it. Every field
   * is clamped into the range its slider shows, anything missing or non-finite
   * falls back to the shipped default, and this never throws.
   *
   * Runs on load (`Modulator.setConfig`) and, nine times over, on every explorer
   * style sync. It restores the matrix *generation* settings but does not
   * re-draw — the drawn matrix is already in the saved θ, so a loaded file
   * reproduces its look, and the redraw button is how you ask for a new one
   * under the restored numbers.
   *
   * Every write is **in place** into the existing sub-objects, and that is load
   * bearing rather than incidental: the panel's tweakpane bindings and the
   * per-species population readout hold `config.population` (and
   * `config.population.accent`) by reference, so replacing either object would
   * leave the panel editing something nothing runs.
   */
  applyExtras(raw: Record<string, unknown> | undefined): void {
    const o = (raw ?? {}) as Record<string, unknown>;

    const m = this.config.macros;
    const src = plainObject(o['macros']);
    const defM = defaultPlifeMacros();
    for (const key of Object.keys(defM) as (keyof PlifeMacros)[]) {
      const r = MACRO_RANGE[key];
      m[key] = clampNum(src[key], defM[key], r.min, r.max);
    }

    const g = this.config.matrixGen;
    const gs = plainObject(o['matrixGen']);
    const defG = defaultMatrixGen();
    g.sigma = clampNum(gs['sigma'], defG.sigma, 0, 4);
    g.symmetry = clampNum(gs['symmetry'], defG.symmetry, 0, 1);
    g.selfBias = clampNum(gs['selfBias'], defG.selfBias, -1, 1);
    g.selfBiasAccent = clampNum(gs['selfBiasAccent'], defG.selfBiasAccent, -1, 1);
    g.accentGain = clampNum(gs['accentGain'], defG.accentGain, 0, 2);
    // `lo <= hi` is enforced rather than assumed, because the generator draws
    // uniformly between them and an inverted band would draw nonsense. The two
    // ceilings differ: an outer radius may be drawn up to `MAX_REACH_BRUTE`
    // (half the torus, the widest any mode can realise), while the hard core
    // stays under `MAX_MIN_R` — it is a contact distance, not a reach.
    readBand(gs['rMin'], g.rMin, defG.rMin, MAX_MIN_R);
    readBand(gs['rMax'], g.rMax, defG.rMax, MAX_REACH_BRUTE);

    // The reach block. `nearStencil` is rounded because it is a *cell count* —
    // the shader indexes with it — and a slider that snapped visually to 2 while
    // holding 2.4 would be a search window that disagreed with its own label.
    const fl = this.config.field;
    const fs = plainObject(o['field']);
    const defF = defaultPlifeField();
    // An unknown / absent / mistyped mode falls back to the default rather than
    // throwing: this block is opaque to everything between the file and here, and
    // the one thing a bad value must not do is put an O(N²) pass on screen by
    // accident. Grid is the safe answer in both directions.
    const mode = fs['pairSearch'];
    fl.pairSearch = PAIR_SEARCH_MODES.includes(mode as PairSearch)
      ? (mode as PairSearch)
      : defF.pairSearch;
    fl.nearStencil = Math.round(
      clampNum(fs['nearStencil'], defF.nearStencil, 1, MAX_NEAR_STENCIL),
    );
    fl.farGain = clampNum(fs['farGain'], defF.farGain, FAR_GAIN_RANGE.min, FAR_GAIN_RANGE.max);
    fl.farScale = clampNum(
      fs['farScale'],
      defF.farScale,
      FAR_SCALE_RANGE.min,
      FAR_SCALE_RANGE.max,
    );

    // The population lane. Ranges are the panel's own (ui/plife-panel.ts, the
    // "population · stems → colonies" folder), so a loaded value can never sit
    // outside the slider meant to show it. Note the two floors that are not 0:
    // `curve` is an exponent the lane raises a 0..1 level to, and `riseTau` /
    // `fallTau` are divisors in the shader — both have a degenerate value at
    // zero that the sliders already refuse to produce.
    const p = this.config.population;
    const ps = plainObject(o['population']);
    const defP = defaultPlifePopulation();
    p.followStems = readBool(ps['followStems'], defP.followStems);
    p.floor = clampNum(ps['floor'], defP.floor, 0, 1);
    p.curve = clampNum(ps['curve'], defP.curve, 0.2, 4);
    p.smoothingMs = clampNum(ps['smoothingMs'], defP.smoothingMs, 100, 5000);
    p.riseTau = clampNum(ps['riseTau'], defP.riseTau, 0.05, 3);
    p.fallTau = clampNum(ps['fallTau'], defP.fallTau, 0.1, 8);

    const a = p.accent;
    const as = plainObject(ps['accent']);
    const defA = defP.accent;
    a.enabled = readBool(as['enabled'], defA.enabled);
    a.floor = clampNum(as['floor'], defA.floor, 0, 1);
    a.boost = clampNum(as['boost'], defA.boost, 0, 3);
    a.smoothingMs = clampNum(as['smoothingMs'], defA.smoothingMs, 100, 5000);

    // Per-species on/off. Written IN PLACE into the existing species objects for
    // the same reason the blocks above are — the panel's tweakpane bindings hold
    // each `config.species[i]` by reference — and driven by the *live* K rather
    // than by the array's length, so a file written at a different K is padded
    // with `true` (a species the file never heard of is present, which is the
    // safe direction) and any surplus entries are ignored. Anything that is not
    // a real boolean falls back to `true` via `readBool`: a corrupt entry should
    // cost you a switch, not a voice.
    const flags = Array.isArray(o['speciesEnabled']) ? (o['speciesEnabled'] as unknown[]) : [];
    for (let i = 0; i < this.config.speciesCount; i++) {
      const sp = this.config.species[i];
      if (sp) sp.enabled = readBool(flags[i], true);
    }
  }

  // ── stats / status ─────────────────────────────────────────────────────────

  stats(): PlifeStats {
    return {
      gridW: this.gridW,
      gridH: this.gridH,
      totalParticles: this.totalParticles,
      aliveParticles: this.aliveCount(),
      stepsThisFrame: this.stepsThisFrame,
      renderPasses: this.post.passCount + 2,
    };
  }

  /** The sim-specific middle of the app's status line. */
  status(): string {
    return `${this.gridW}×${this.gridH} cells · ${this.aliveCount().toLocaleString()} particles · seed ${this.seed}`;
  }

  /**
   * The summed population *target*. Not a live count: what is actually present
   * includes whatever is still fading out, and that number only exists on the
   * GPU. Reading it back would cost a fence per frame to make a status line
   * marginally more honest.
   */
  private aliveCount(): number {
    return this.activeTotal;
  }

  // ── init ───────────────────────────────────────────────────────────────────

  async init(ctx: GpuContext): Promise<void> {
    this.ctx = ctx;
    const { device } = ctx;
    const k = this.config.speciesCount;

    // World space is fixed for the sim's life, exactly like physarum's grid:
    // rescaling it mid-run would move every particle relative to every radius and
    // throw the accumulated structure away. A canvas resize only re-allocates the
    // post surfaces; the world keeps the aspect it started with.
    this.worldH = 1;
    this.worldW = Math.max(Math.max(ctx.width, 1) / Math.max(ctx.height, 1), 0.2);

    this.chooseGrid();

    // Segment size is a whole number of workgroups so that "species boundary" and
    // "workgroup boundary" coincide, and totalParticles is exactly segSize·K —
    // no leftover tail whose derived species would alias the last segment.
    const perSpecies = Math.floor(this.config.maxParticles / k);
    this.segSize = Math.max(
      PARTICLE_WORKGROUP,
      Math.floor(perSpecies / PARTICLE_WORKGROUP) * PARTICLE_WORKGROUP,
    );
    this.totalParticles = this.segSize * k;

    this.speciesData = new Float32Array(k * FLOATS_PER_SPECIES);
    // Four K² blocks now: θ's attraction / maxR / minR, then the far lane's
    // per-pair coefficient, which is not θ and is rebuilt every frame.
    this.interactionData = new Float32Array(4 * k * k);

    // Square texels: the width follows the world's aspect. Clamped because a
    // degenerate aspect (a very tall explorer tile) would otherwise ask for a
    // one-texel-wide field, and the blur's `repeat` wrap on a 1-texel axis is a
    // constant.
    this.farH = FAR_HEIGHT;
    this.farW = Math.min(
      Math.max(Math.round((FAR_HEIGHT * this.worldW) / this.worldH), 32),
      1024,
    );

    const cells = this.gridW * this.gridH;

    this.globalsBuf = device.createBuffer({
      label: 'plife.globals',
      size: GLOBALS_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.speciesBuf = device.createBuffer({
      label: 'plife.species',
      size: this.speciesData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.interactionBuf = device.createBuffer({
      label: 'plife.interaction',
      size: Math.max(this.interactionData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // COPY_SRC/COPY_DST exist for the workbench's snapshot/restore; they cost
    // nothing when unused.
    this.particleBuf = [0, 1].map((i) =>
      device.createBuffer({
        label: `plife.particles${i}`,
        size: this.totalParticles * BYTES_PER_PARTICLE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    );
    // COPY_DST on the two atomic buffers is what makes encoder.clearBuffer legal —
    // that is the whole reason there is no clear pipeline in this sim.
    this.cellCountBuf = device.createBuffer({
      label: 'plife.cellCount',
      size: cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellFillBuf = device.createBuffer({
      label: 'plife.cellFill',
      size: cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellStartBuf = device.createBuffer({
      label: 'plife.cellStart',
      size: cells * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.sortedIdxBuf = device.createBuffer({
      label: 'plife.sortedIdx',
      size: this.totalParticles * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    // Fixed-size and always allocated: MAX_SPLASHES × 32 bytes is 1 KiB, and a
    // buffer that may or may not exist would make the bind group conditional.
    this.splashBuf = device.createBuffer({
      label: 'plife.splashes',
      size: MAX_SPLASHES * FLOATS_PER_SPLASH * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // The far chain's four stages. Allocated unconditionally even when the lane
    // is off: they are 2.4 MB in total and making them conditional would make
    // three bind groups conditional with them, which is a lot of branching to
    // avoid an allocation that a slider can undo at any moment.
    const farStage = (name: string): GPUTexture[] =>
      [0, 1].map((i) =>
        device.createTexture({
          label: `plife.far.${name}${i}`,
          size: { width: this.farW, height: this.farH },
          format: HDR_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        }),
      );
    this.farSplat = farStage('splat');
    this.farTmp = farStage('tmp');
    this.farNear = farStage('near');
    this.farDog = farStage('dog');
    this.farSplatViews = this.farSplat.map((t) => t.createView());
    // `repeat` on both axes, because the world is a torus: the blur has to wrap
    // or the field flattens toward the edges and anything crossing the seam is
    // torn in the far lane while the near lane (`wrapDelta`) still sees it whole.
    this.farSampler = device.createSampler({
      label: 'plife.far.sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.blurParamBufs = Array.from({ length: FAR_BLUR_PASSES }, (_, i) =>
      device.createBuffer({
        label: `plife.far.blurParams${i}`,
        size: this.blurParams.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    );

    const src = (body: string): GPUShaderModule =>
      device.createShaderModule({ code: `${commonWgsl}\n${body}` });

    const C = GPUShaderStage.COMPUTE;
    const uniform: GPUBufferBindingLayout = { type: 'uniform' };
    const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
    const rw: GPUBufferBindingLayout = { type: 'storage' };

    const gridLayout = device.createBindGroupLayout({
      label: 'plife.gridLayout',
      entries: [
        { binding: 0, visibility: C, buffer: uniform },
        { binding: 1, visibility: C, buffer: ro },
        { binding: 2, visibility: C, buffer: ro },
        { binding: 3, visibility: C, buffer: rw },
        { binding: 4, visibility: C, buffer: rw },
        { binding: 5, visibility: C, buffer: rw },
        { binding: 6, visibility: C, buffer: rw },
      ],
    });
    // Bindings 10–12 are the far lane's field. `initParticles` and
    // `respawnParticles` share this layout and never touch them, which is legal:
    // a layout may describe more than an entry point statically uses.
    const stepLayout = device.createBindGroupLayout({
      label: 'plife.stepLayout',
      entries: [
        { binding: 0, visibility: C, buffer: uniform },
        { binding: 1, visibility: C, buffer: ro },
        { binding: 2, visibility: C, buffer: ro },
        { binding: 3, visibility: C, buffer: ro },
        { binding: 4, visibility: C, buffer: rw },
        { binding: 5, visibility: C, buffer: ro },
        { binding: 6, visibility: C, buffer: ro },
        { binding: 7, visibility: C, buffer: ro },
        { binding: 10, visibility: C, sampler: { type: 'filtering' } },
        { binding: 11, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 12, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });
    // The splash pass's own layout. It shares step.wgsl's module but names only
    // bindings 0, 8 and 9 — which is legal and is the point: binding 9 is a
    // read_write view of the same buffer bindings 3 names read-only for the
    // force pass, and one layout cannot describe both. WebGPU validates a
    // pipeline layout against the bindings its entry point *statically uses*, so
    // `splashParticles` never has to see `src`, `dst` or the grid.
    const splashLayout = device.createBindGroupLayout({
      label: 'plife.splashLayout',
      entries: [
        { binding: 0, visibility: C, buffer: uniform },
        { binding: 8, visibility: C, buffer: ro },
        { binding: 9, visibility: C, buffer: rw },
      ],
    });

    const gridModule = src(gridWgsl);
    const stepModule = src(stepWgsl);
    const gridPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [gridLayout] });
    const stepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [stepLayout] });
    const splashPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [splashLayout],
    });

    this.countPipeline = device.createComputePipeline({
      label: 'plife.count',
      layout: gridPipelineLayout,
      compute: { module: gridModule, entryPoint: 'countParticles' },
    });
    this.scanPipeline = device.createComputePipeline({
      label: 'plife.scan',
      layout: gridPipelineLayout,
      compute: { module: gridModule, entryPoint: 'scanCells' },
    });
    this.scatterPipeline = device.createComputePipeline({
      label: 'plife.scatter',
      layout: gridPipelineLayout,
      compute: { module: gridModule, entryPoint: 'scatterParticles' },
    });
    this.stepPipeline = device.createComputePipeline({
      label: 'plife.step',
      layout: stepPipelineLayout,
      compute: { module: stepModule, entryPoint: 'stepParticles', constants: { BRUTE: 0 } },
    });
    // The brute lane. One more pipeline off the same module and the same layout,
    // specialised through step.wgsl's `override BRUTE` — so the two force kernels
    // share every line except the loop that chooses which `j`s to visit, and
    // neither pays for the other's register pressure. `runStep` picks between
    // them per substep, which is what makes the panel's mode dropdown live: no
    // rebuild, no reallocation, no state to migrate.
    this.stepBrutePipeline = device.createComputePipeline({
      label: 'plife.step.brute',
      layout: stepPipelineLayout,
      compute: { module: stepModule, entryPoint: 'stepParticles', constants: { BRUTE: 1 } },
    });
    this.initPipeline = device.createComputePipeline({
      label: 'plife.init',
      layout: stepPipelineLayout,
      compute: { module: stepModule, entryPoint: 'initParticles' },
    });
    this.respawnPipeline = device.createComputePipeline({
      label: 'plife.respawn',
      layout: stepPipelineLayout,
      compute: { module: stepModule, entryPoint: 'respawnParticles' },
    });
    this.splashPipeline = device.createComputePipeline({
      label: 'plife.splash',
      layout: splashPipelineLayout,
      compute: { module: stepModule, entryPoint: 'splashParticles' },
    });

    const V = GPUShaderStage.VERTEX;
    const F = GPUShaderStage.FRAGMENT;
    this.fadeLayout = device.createBindGroupLayout({
      label: 'plife.fadeLayout',
      entries: [
        { binding: 0, visibility: F, buffer: uniform },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });
    // The particle pass reads two storage buffers in the *vertex* stage. That is
    // core WebGPU (read-only storage is permitted in vertex; writable storage is
    // not), and it is what lets one instanced draw cover the whole pool with no
    // vertex buffer and no per-frame upload.
    const particleLayout = device.createBindGroupLayout({
      label: 'plife.particleLayout',
      entries: [
        { binding: 0, visibility: V | F, buffer: uniform },
        { binding: 3, visibility: V, buffer: ro },
        { binding: 4, visibility: V, buffer: ro },
        { binding: 5, visibility: F, buffer: ro },
      ],
    });

    const renderModule = src(renderWgsl);
    this.fadePipeline = device.createRenderPipeline({
      label: 'plife.fade',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.fadeLayout] }),
      vertex: { module: renderModule, entryPoint: 'vsFade' },
      fragment: { module: renderModule, entryPoint: 'fsFade', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    this.particlePipeline = device.createRenderPipeline({
      label: 'plife.particles',
      layout: device.createPipelineLayout({ bindGroupLayouts: [particleLayout] }),
      vertex: { module: renderModule, entryPoint: 'vsParticles' },
      fragment: {
        module: renderModule,
        entryPoint: 'fsParticles',
        targets: [
          {
            format: HDR_FORMAT,
            // Additive, and unbounded: the HDR surface has no clipping point, so
            // a thousand overlapping splats sum honestly and the tone map at the
            // end of the post chain is the only thing that ever compresses.
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-strip' },
    });

    // ── far field pipelines ──────────────────────────────────────────────────
    //
    // One module, two pipelines, disjoint binding numbers — the same arrangement
    // render.wgsl uses. The splat reads particles in the *vertex* stage (core
    // WebGPU allows read-only storage there) so one draw covers the whole pool
    // with no vertex buffer.
    const farModule = src(farWgsl);
    const splatLayout = device.createBindGroupLayout({
      label: 'plife.far.splatLayout',
      entries: [
        { binding: 0, visibility: V, buffer: uniform },
        { binding: 1, visibility: V, buffer: ro },
      ],
    });
    const blurLayout = device.createBindGroupLayout({
      label: 'plife.far.blurLayout',
      entries: [
        { binding: 2, visibility: F, buffer: uniform },
        { binding: 3, visibility: F, sampler: { type: 'filtering' } },
        { binding: 4, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 5, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });
    // Additive and unbounded, exactly like the particle pass: a density splat is
    // a sum, and rgba16float has the range to hold one honestly.
    const additive: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };
    this.splatPipeline = device.createRenderPipeline({
      label: 'plife.far.splat',
      layout: device.createPipelineLayout({ bindGroupLayouts: [splatLayout] }),
      vertex: { module: farModule, entryPoint: 'vsSplat' },
      fragment: {
        module: farModule,
        entryPoint: 'fsSplat',
        targets: [
          { format: HDR_FORMAT, blend: additive },
          { format: HDR_FORMAT, blend: additive },
        ],
      },
      primitive: { topology: 'point-list' },
    });
    this.blurPipeline = device.createRenderPipeline({
      label: 'plife.far.blur',
      layout: device.createPipelineLayout({ bindGroupLayouts: [blurLayout] }),
      vertex: { module: farModule, entryPoint: 'vsBlur' },
      fragment: {
        module: farModule,
        entryPoint: 'fsBlur',
        targets: [{ format: HDR_FORMAT }, { format: HDR_FORMAT }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.splatBinds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `plife.far.splat.parity${p}`,
        layout: splatLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: { buffer: this.particleBuf[p] as GPUBuffer } },
        ],
      }),
    );
    // The chain, stage by stage:
    //
    //   0  splat --σ1 X--> tmp
    //   1  tmp   --σ1 Y--> near        (near = G_σ1 * ρ)
    //   2  near  --σr X--> tmp         (σr² = σ2² − σ1²; blurs compose)
    //   3  tmp   --σr Y--> dog = near − that   (= G_σ1*ρ − G_σ2*ρ)
    //
    // Pass 3 is the only one that reads `near`, but every bind group has to name
    // bindings 6/7 because the shader statically uses them. The other three bind
    // `dog` there instead of `near` — deliberately, not arbitrarily: pass 1
    // *writes* `near`, and a texture cannot be a render attachment and a sampled
    // resource in the same pass.
    const chain: { srcTex: GPUTexture[]; nearTex: GPUTexture[]; dst: GPUTexture[] }[] = [
      { srcTex: this.farSplat, nearTex: this.farDog, dst: this.farTmp },
      { srcTex: this.farTmp, nearTex: this.farDog, dst: this.farNear },
      { srcTex: this.farNear, nearTex: this.farDog, dst: this.farTmp },
      { srcTex: this.farTmp, nearTex: this.farNear, dst: this.farDog },
    ];
    this.blurBinds = chain.map((stage, i) =>
      device.createBindGroup({
        label: `plife.far.blur${i}`,
        layout: blurLayout,
        entries: [
          { binding: 2, resource: { buffer: this.blurParamBufs[i] as GPUBuffer } },
          { binding: 3, resource: this.farSampler },
          { binding: 4, resource: (stage.srcTex[0] as GPUTexture).createView() },
          { binding: 5, resource: (stage.srcTex[1] as GPUTexture).createView() },
          { binding: 6, resource: (stage.nearTex[0] as GPUTexture).createView() },
          { binding: 7, resource: (stage.nearTex[1] as GPUTexture).createView() },
        ],
      }),
    );
    this.blurTargets = chain.map((stage) => stage.dst.map((t) => t.createView()));

    this.post.init(ctx);
    this.post.ensureSize(ctx.width, ctx.height);

    this.gridBinds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `plife.grid.parity${p}`,
        layout: gridLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: { buffer: this.speciesBuf } },
          { binding: 2, resource: { buffer: this.particleBuf[p] as GPUBuffer } },
          { binding: 3, resource: { buffer: this.cellCountBuf } },
          { binding: 4, resource: { buffer: this.cellFillBuf } },
          { binding: 5, resource: { buffer: this.cellStartBuf } },
          { binding: 6, resource: { buffer: this.sortedIdxBuf } },
        ],
      }),
    );
    this.stepBinds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `plife.step.parity${p}`,
        layout: stepLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: { buffer: this.speciesBuf } },
          { binding: 2, resource: { buffer: this.interactionBuf } },
          { binding: 3, resource: { buffer: this.particleBuf[p] as GPUBuffer } },
          { binding: 4, resource: { buffer: this.particleBuf[1 - p] as GPUBuffer } },
          { binding: 5, resource: { buffer: this.cellCountBuf } },
          { binding: 6, resource: { buffer: this.cellStartBuf } },
          { binding: 7, resource: { buffer: this.sortedIdxBuf } },
          { binding: 10, resource: this.farSampler },
          { binding: 11, resource: (this.farDog[0] as GPUTexture).createView() },
          { binding: 12, resource: (this.farDog[1] as GPUTexture).createView() },
        ],
      }),
    );
    this.splashBinds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `plife.splash.parity${p}`,
        layout: splashLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 8, resource: { buffer: this.splashBuf } },
          { binding: 9, resource: { buffer: this.particleBuf[p] as GPUBuffer } },
        ],
      }),
    );
    this.particleBinds = [0, 1].map((p) =>
      device.createBindGroup({
        label: `plife.render.parity${p}`,
        layout: particleLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 3, resource: { buffer: this.speciesBuf } },
          { binding: 4, resource: { buffer: this.particleBuf[p] as GPUBuffer } },
          { binding: 5, resource: { buffer: this.post.autoBuffer } },
        ],
      }),
    );
    this.rebuildFadeBinds();

    this.ready = true;
    this.uploadInteractions();
    this.reseed(this.seed);
  }

  /**
   * Grid dims and cell sizes. The invariant: `cellW >= R_CAP && cellH >= R_CAP`.
   * R_CAP is now purely an occupancy choice — how many particles a cell list
   * holds — because reach is capped at `stencil × cell` rather than at one cell.
   *
   * `floor`, not `ceil`: with `ceil` the grid would cover slightly *more* than
   * the world, the last cell along each axis would be partly outside it, and the
   * wrap seam would put two particles well inside R_CAP two cells apart — a
   * neighbour silently missed along one strip of the torus. `floor` makes the
   * grid tile the world exactly, at the price of cells a hair larger than R_CAP.
   */
  private chooseGrid(): void {
    let gw = Math.max(1, Math.floor(this.worldW / R_CAP));
    let gh = Math.max(1, Math.floor(this.worldH / R_CAP));
    // Cannot happen at any sane aspect ratio (a 16:9 world is ~700 cells), but
    // the scan pass depends on the bound, so it is enforced rather than assumed.
    while (gw * gh > MAX_CELLS && (gw > 1 || gh > 1)) {
      gw = Math.max(1, Math.floor(gw / 2));
      gh = Math.max(1, Math.floor(gh / 2));
    }
    this.gridW = gw;
    this.gridH = gh;
    this.cellW = this.worldW / gw;
    this.cellH = this.worldH / gh;
  }

  // ── world lifecycle ────────────────────────────────────────────────────────

  /** Fresh seed, particles re-scattered into seeded per-species clusters. */
  reseed(seed: number): void {
    if (!this.ready || !this.ctx) return;
    const { device } = this.ctx;
    this.setSeed(seed);
    this.stepAccumulator = 0;
    this.lastPcgTick = 0;
    this.uploadSpecies();
    this.writeGlobals(0);
    // A new world starts from black; letting the adapted gain carry over would
    // make the first second a slow fade up from the old exposure.
    this.post.resetAutoExposure();

    const encoder = device.createCommandEncoder({ label: 'plife.reseed' });
    const pass = encoder.beginComputePass({ label: 'initParticles' });
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.stepBinds[this.parity] as GPUBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.totalParticles / PARTICLE_WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
    // init writes the *destination* half of the ping-pong, like every other pass
    // in step.wgsl, so the flip is what makes it the live state. The other buffer
    // is then stale, which is fine: `stepParticles` rewrites every element of it.
    this.parity = 1 - this.parity;
  }

  /**
   * Section-boundary re-seed: re-scatter `fraction` of the pool, chosen and
   * placed by hash(seed, segmentIndex, particleIndex). Uniformly, not in
   * clusters — new matter arriving into an old world.
   */
  partialReseed(segmentIndex: number, fraction: number): void {
    if (!this.ready || !this.ctx) return;
    const f = Math.min(Math.max(fraction, 0), 1);
    if (f <= 0) return;
    const { device } = this.ctx;

    this.respawnFraction = f;
    this.respawnKey = segmentIndex >>> 0;
    this.writeGlobals(this.lastPcgTick);

    const encoder = device.createCommandEncoder({ label: 'plife.partialReseed' });
    const pass = encoder.beginComputePass({ label: 'respawnParticles' });
    pass.setPipeline(this.respawnPipeline);
    pass.setBindGroup(0, this.stepBinds[this.parity] as GPUBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.totalParticles / PARTICLE_WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
    this.parity = 1 - this.parity;

    // Leave the uniform clean: every other pass shares this Globals block.
    this.respawnFraction = 0;
    this.writeGlobals(this.lastPcgTick);
  }

  snapshot(): boolean {
    if (!this.ready || !this.ctx) return false;
    const { device } = this.ctx;
    if (!this.snap) {
      this.snap = {
        particles: device.createBuffer({
          label: 'plife.snap.particles',
          size: this.totalParticles * BYTES_PER_PARTICLE,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        seed: this.seed,
        stepAccumulator: this.stepAccumulator,
        lastPcgTick: this.lastPcgTick,
        parity: this.parity,
        stemLevel: new Float32Array(this.stemLevel.length),
        popMul: new Float32Array(this.popMul.length),
        accentMul: new Float32Array(this.accentMul.length),
        accentActivity: 0,
        takenAt: performance.now(),
      };
    }
    const snap = this.snap;
    snap.seed = this.seed;
    snap.stepAccumulator = this.stepAccumulator;
    snap.lastPcgTick = this.lastPcgTick;
    snap.parity = this.parity;
    snap.stemLevel.set(this.stemLevel);
    snap.popMul.set(this.popMul);
    snap.accentMul.set(this.accentMul);
    snap.accentActivity = this.accentActivity;
    snap.takenAt = performance.now();

    const source = this.particleBuf[this.parity] as GPUBuffer;
    const encoder = device.createCommandEncoder({ label: 'plife.snapshot' });
    encoder.copyBufferToBuffer(source, 0, snap.particles, 0, source.size);
    device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Restore the last snapshot exactly. Parameters are *not* restored — that is the A/B. */
  restoreSnapshot(): boolean {
    if (!this.ready || !this.ctx || !this.snap) return false;
    const { device } = this.ctx;
    const snap = this.snap;
    const target = this.particleBuf[snap.parity] as GPUBuffer;

    const encoder = device.createCommandEncoder({ label: 'plife.restore' });
    encoder.copyBufferToBuffer(snap.particles, 0, target, 0, target.size);
    device.queue.submit([encoder.finish()]);

    this.parity = snap.parity;
    this.setSeed(snap.seed);
    this.stepAccumulator = snap.stepAccumulator;
    this.lastPcgTick = snap.lastPcgTick;
    // Population posture comes back with the particles. Without this the ramps
    // would immediately start pulling the restored world toward the *current*
    // moment's targets, and an A/B would be comparing two different populations
    // as well as two different parameter sets.
    this.stemLevel.set(snap.stemLevel);
    this.popMul.set(snap.popMul);
    this.accentMul.set(snap.accentMul);
    this.accentActivity = snap.accentActivity;
    return true;
  }

  clearSnapshot(): void {
    if (!this.snap) return;
    this.snap.particles.destroy();
    this.snap = null;
  }

  /**
   * Take another plife sim's *world* — its matter, not its picture.
   *
   * This exists for one caller: leaving explorer mode. Picking a tile has always
   * handed the live sim the tile's θ, so the parameters were right and the world
   * was somebody else's — you chose a colony arrangement and got back a different
   * one built from the same numbers. Copying the particle buffer is what makes
   * "exit" mean "keep looking at the thing I picked, full screen".
   *
   * ## What moves
   *
   * The particle buffer, GPU→GPU, and nothing else on the GPU. `struct Particle`
   * (common.wgsl) is `pos`, `vel`, `energy` and three pad words — 32 bytes — and
   * that is the *whole* of a particle: species is derived from the index
   * (`speciesOf`), never stored, so the segmentation is a pure function of
   * `segSize` and K and is identical in both sims by construction. The grid
   * buffers (`cellCount`, `cellStart`, `sortedIdx`) are cleared and rebuilt from
   * scratch at the top of every substep and carry nothing across; there is
   * nothing in them to copy.
   *
   * The population lane's CPU posture rides along, for the same reason
   * `restoreSnapshot` carries it: `energy` in the buffer is the *current*
   * presence of each particle and the lane's smoothed levels are the target it is
   * easing toward. Arriving with the tile's matter and the live sim's stale
   * targets would make the first second after exit a visible re-shaping of the
   * colony sizes you just chose.
   *
   * ## What deliberately does not move
   *
   * - **The render domain** — the HDR feedback echo. Tile and canvas are
   *   different resolutions, so there is nothing size-compatible to copy, and the
   *   trail re-forms from the particles in about half a second anyway.
   * - **Auto-exposure.** The controller re-adapts on its own time constant; it is
   *   a property of the screen, not of the world. Not reset either — a reset
   *   would open the full-screen view with a fade up from black.
   * - **The seed.** It is only the phase of the wander noise and the key future
   *   partial reseeds hash against, and adopting it would fire `onSeedChange` →
   *   the modulator's seeded rewire → a redrawn interaction matrix, i.e. it would
   *   destroy the very look this method exists to preserve.
   *
   * ## The size assertion
   *
   * Callers build tiles from a `structuredClone` of the live config, so
   * `maxParticles` and K match and the buffers are the same length. That is a
   * property of one call site, not of the type, so it is checked rather than
   * assumed: a mismatch skips the promotion and says so, because a partial copy
   * would leave the live world half one sim and half another.
   *
   * (One thing is genuinely approximate: world *width* is the aspect of the
   * surface each sim was `init`ed against, so a tile's world is a couple of
   * percent wider than the canvas's. Positions past the live `worldW` are wrapped
   * by every shader that reads them, so the cost is one thin vertical strip
   * folding to the left edge on the first step, on a torus where that is a
   * translation rather than a tear. Correcting it exactly would need a compute
   * pass instead of a copy, which is not worth it for 2%.)
   *
   * @returns false if the promotion was skipped; the live world is untouched.
   */
  adoptParticleState(from: PlifeSim): boolean {
    if (!this.ready || !this.ctx || !from.ready) return false;
    const src = from.particleBuf[from.parity];
    const dst = this.particleBuf[this.parity];
    if (
      !src ||
      !dst ||
      src.size !== dst.size ||
      from.segSize !== this.segSize ||
      from.config.speciesCount !== this.config.speciesCount
    ) {
      console.warn(
        `plife: cannot adopt particle state (${src?.size ?? 0}B/${from.segSize} vs ` +
          `${dst?.size ?? 0}B/${this.segSize}); keeping the live world`,
      );
      return false;
    }

    // `parity` names the buffer the next grid/force/render pass reads, on both
    // sides — so this reads the tile's *current* state and writes the one the
    // live sim is about to pick up, with no flip needed.
    const { device } = this.ctx;
    const encoder = device.createCommandEncoder({ label: 'plife.adopt' });
    encoder.copyBufferToBuffer(src, 0, dst, 0, dst.size);
    device.queue.submit([encoder.finish()]);

    this.stemLevel.set(from.stemLevel);
    this.popMul.set(from.popMul);
    this.accentMul.set(from.accentMul);
    this.accentActivity = from.accentActivity;
    // The adopted posture is already the music's; priming again would snap the
    // EMAs on the next tick and undo exactly what was just copied.
    this.popPrimed = true;
    return true;
  }

  // ── per-tick ───────────────────────────────────────────────────────────────

  tick(frame: FeaturesFrame, simTick: number): void {
    if (!this.ready || !this.ctx) return;

    // Ahead of the substep loop and outside the paused branch: the population
    // lane is a *smoother*, and freezing its EMAs while the transport runs would
    // mean un-pausing snapped the whole field to whatever the music had become.
    this.updatePopulation(frame, TICK_DT, !this.popPrimed);
    this.popPrimed = true;

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
    // Once per tick, not once per substep: the disc list is a property of the
    // tick's envelopes, and every substep of this tick applies the same discs
    // (which is what makes a splash a sustained shove for the length of its
    // decay rather than a single frame's nudge).
    if (steps > 0) this.uploadSplashes();
    // Once per clock tick, ahead of the substeps and outside the loop: the far
    // field is a *smoothed* quantity at σ ≥ 0.04 world, and nothing in it moves
    // measurably in the 1/60 s a substep covers. Re-splatting it per substep
    // would be four times the cost at the ceiling for a field that had barely
    // changed, and the near lane is the one that has to resolve fast motion.
    if (steps > 0 && this.farActive()) this.runFarField();
    for (let s = 0; s < steps; s++) {
      this.runStep(simTick * MAX_SUBSTEPS + s);
    }
  }

  /**
   * The far lane, once per frame: splat every live particle into a per-species
   * density target, then four separable-Gaussian passes that leave
   * `G_σ1*ρ − G_σ2*ρ` in `farDog`. `stepParticles` samples that field's gradient
   * every substep.
   *
   * Its own encoder and submit, ahead of the substeps' — the force pass reads
   * what this writes, and a submit boundary is the least ambiguous way to say so.
   */
  private runFarField(): void {
    const { device } = this.ctx as GpuContext;
    // Order matters: `uploadSpecies` is what recomputes `targetAlive`, and the
    // per-species mean density the coefficients divide by comes from it.
    this.uploadSpecies();
    this.writeGlobals(this.lastPcgTick);
    this.uploadFar();
    this.writeBlurParams();

    const encoder = device.createCommandEncoder({ label: 'plife.far' });
    const splat = encoder.beginRenderPass({
      label: 'plife.far.splat',
      colorAttachments: this.farSplatViews.map((view) => ({
        view,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear' as const,
        storeOp: 'store' as const,
      })),
    });
    splat.setPipeline(this.splatPipeline);
    splat.setBindGroup(0, this.splatBinds[this.parity] as GPUBindGroup);
    // One point per pool slot. Dormant slots emit an off-screen position and are
    // clipped before any fragment work, exactly as the sprite pass does with a
    // degenerate quad — cheaper than a CPU-side compaction and a readback.
    splat.draw(this.totalParticles);
    splat.end();

    for (let i = 0; i < FAR_BLUR_PASSES; i++) {
      const targets = this.blurTargets[i] as GPUTextureView[];
      const pass = encoder.beginRenderPass({
        label: `plife.far.blur${i}`,
        colorAttachments: targets.map((view) => ({
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear' as const,
          storeOp: 'store' as const,
        })),
      });
      pass.setPipeline(this.blurPipeline);
      pass.setBindGroup(0, this.blurBinds[i] as GPUBindGroup);
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
  }

  /**
   * The four blur passes' parameters. Rewritten every frame rather than on
   * change: it is 128 bytes total, and "on change" would mean tracking two
   * knobs, the stencil and the grid, any of which moving invalidates all four.
   *
   * σ1 is the near lane's cutoff (see `farSigma1`), σ2 the knob (floored), and
   * the second pair of passes blurs by the *residual* σr with
   * σr² = σ2² − σ1², because Gaussians compose in quadrature: blurring the
   * σ1 field by σr yields the σ2 field for free instead of a second full-width
   * blur of the raw splat.
   */
  private writeBlurParams(): void {
    const { device } = this.ctx as GpuContext;
    const s1 = this.farSigma1;
    const s2 = this.farSigma;
    const sr = Math.sqrt(Math.max(s2 * s2 - s1 * s1, 1e-12));
    const passes: { sigma: number; axis: 0 | 1; dog: 0 | 1 }[] = [
      { sigma: s1, axis: 0, dog: 0 },
      { sigma: s1, axis: 1, dog: 0 },
      { sigma: sr, axis: 0, dog: 0 },
      { sigma: sr, axis: 1, dog: 1 },
    ];
    const f = this.blurParamsF32;
    const u = this.blurParamsU32;
    for (let i = 0; i < passes.length; i++) {
      const p = passes[i] as { sigma: number; axis: 0 | 1; dog: 0 | 1 };
      // Per-axis, because `farW` is rounded and its texels are not exactly the
      // same world size as `farH`'s. Sizing each axis in its own texels keeps
      // the kernel isotropic in the units that matter.
      const texels =
        p.axis === 0 ? (p.sigma * this.farW) / this.worldW : (p.sigma * this.farH) / this.worldH;
      const s = Math.max(texels, 1e-3);
      const half = Math.min(Math.ceil(3 * s), FAR_MAX_HALF_TAPS);
      f[0] = p.axis === 0 ? 1 / this.farW : 0;
      f[1] = p.axis === 0 ? 0 : 1 / this.farH;
      f[2] = s;
      // Tap spacing. 1 texel until the ±3σ support needs more taps than the
      // budget, then it stretches — the kernel keeps its shape and loses
      // resolution instead of losing its tails.
      f[3] = (3 * s) / Math.max(half, 1);
      u[4] = half;
      u[5] = p.dog;
      u[6] = 0;
      u[7] = 0;
      device.queue.writeBuffer(this.blurParamBufs[i] as GPUBuffer, 0, this.blurParams);
    }
  }

  /**
   * The far lane's per-pair coefficient block — the fourth K² block of the
   * interaction buffer, rebuilt every frame because two of its factors (the
   * populations, the matrix) move every tick.
   *
   * As implemented, with everything folded in:
   *
   *   Afar[i][j] = (A[i][j] / max|A|)      structure and sign, reused from θ
   *              × farGain                 the one global strength knob
   *              × FAR_FORCE_SCALE         units: gradient → near-lane force
   *              × (σ2 / FAR_SCALE_REF)    far-scale compensation
   *              × (farW·farH / alive_j)   1 / ρ̄_j, the mean-density normaliser
   *              × (farH / 2·worldH)       central difference → world derivative
   *
   * and the shader computes `Σ_j Afar[i][j] · (D_j(u+e) − D_j(u−e))`.
   *
   * Why each normalisation is not optional:
   *
   * - **max|A|** — the attraction matrix is drawn from the seed and its overall
   *   scale varies with `matrixGen.sigma`. Without dividing by its own maximum,
   *   `farGain` would mean something different on every reroll. Reusing A's
   *   *structure* is what makes the far lane agree with the near one about who
   *   chases whom — and it is why a structurally-zero cell of the partition
   *   contributes exactly 0 out here too, by arithmetic rather than by a rule
   *   that could drift.
   * - **1 / ρ̄_j** — the population lane swings a species' count by 3× over a
   *   track and `aliveFraction` differs 3× between primaries and accents. An
   *   un-normalised density field would make the far force track the population
   *   rather than the *shape* of it, so a loud chorus would be a stronger
   *   current and a species with a small colony would be ignored. Dividing by
   *   the current mean makes the field a relative density and the lane
   *   population-invariant.
   * - **σ2 / FAR_SCALE_REF** — a structure at scale σ2 has a DoG gradient going
   *   as 1/σ2, so without this the far-scale knob would double as a strength
   *   knob and there would be no way to ask for "bigger organisation, same pull".
   *
   * The far term is added into `force` *before* the integrator's
   * `forceGain · forceScale` multiply, so it inherits both — and therefore the
   * `force` macro — deliberately: it is a force among forces, and a performance
   * gesture that pushes the world harder should push all of it harder.
   */
  private uploadFar(): void {
    const k = this.config.speciesCount;
    const kk = k * k;
    const d = this.interactionData;
    const base = 3 * kk;
    const gain = Math.max(this.config.field.farGain, 0);

    let maxAbs = 0;
    for (let i = 0; i < kk; i++) maxAbs = Math.max(maxAbs, Math.abs(this.config.attraction[i] ?? 0));

    if (gain <= 0 || maxAbs <= 0) {
      d.fill(0, base, base + kk);
    } else {
      const scaleComp = this.farSigma / FAR_SCALE_REF;
      const texels = this.farW * this.farH;
      const gradToWorld = this.farH / (2 * this.worldH);
      const common = (gain * FAR_FORCE_SCALE * scaleComp * gradToWorld * texels) / maxAbs;
      for (let j = 0; j < k; j++) {
        // A species with nobody in it has no density to normalise against, and
        // 1/0 would be an infinite coefficient multiplying an all-zero field —
        // NaN, in a buffer that feeds back into itself. Zero is the honest
        // answer: an absent species exerts no far force.
        const alive = this.targetAlive[j] ?? 0;
        const norm = alive > 0 ? common / alive : 0;
        for (let i = 0; i < k; i++) {
          d[base + i * k + j] = (this.config.attraction[i * k + j] ?? 0) * norm;
        }
      }
    }
    (this.ctx as GpuContext).device.queue.writeBuffer(
      this.interactionBuf,
      base * 4,
      d,
      base,
      kk,
    );
  }

  /**
   * Stems → colony size, novelty → accent colony size. Both are EMA-smoothed
   * here and consumed by `uploadSpecies`, which is the only place a population
   * target is ever computed.
   *
   * `snap` adopts the sampled levels outright instead of easing into them, for
   * transport discontinuities — a restore, a seek — where the EMA's history
   * belongs to a different moment of the song and easing across the jump would
   * drag the old section's population more than a second into the new one.
   */
  private updatePopulation(frame: FeaturesFrame, dt: number, snap = false): void {
    const pop = this.config.population;
    const k = this.config.speciesCount;
    const map = this.stemMap();
    const values = frame.values;

    // popMul_k = floor + (1 - floor) · smoothed(stem_k)^curve. Deliberately the
    // same arithmetic as mapping/stemfollow.ts's followMultiplier — the two
    // lanes are the same idea applied to different quantities, and they should
    // respond to a given instrument identically in shape even though population
    // moves four times slower. See the ownership note on the fields above for
    // why this is a copy of the arithmetic rather than a use of the class.
    const stemOn = pop.followStems && this.stemOffset !== null;
    const floor = Math.min(Math.max(pop.floor, 0), 1);
    const curve = Math.max(pop.curve, 0.01);
    const alpha = emaAlpha(dt, pop.smoothingMs, snap);
    for (let i = 0; i < k; i++) {
      const stem = map[i] ?? -1;
      if (!stemOn || stem < 0) {
        this.popMul[i] = 1;
        continue;
      }
      const raw = values[(this.stemOffset as number) + stem] as number;
      const x = Number.isFinite(raw) ? Math.min(Math.max(raw, 0), 1) : 0;
      const prev = this.stemLevel[i] as number;
      const next = prev + (x - prev) * alpha;
      this.stemLevel[i] = next;
      this.popMul[i] = floor + (1 - floor) * Math.pow(next, curve);
    }

    const acc = pop.accent;
    let a = 0;
    if (acc.enabled) {
      const nv = this.noveltyOffset === null ? 0 : (values[this.noveltyOffset] as number);
      const ch = this.chorusOffset === null ? 0 : (values[this.chorusOffset] as number);
      const m = Math.max(Number.isFinite(nv) ? nv : 0, Number.isFinite(ch) ? ch : 0);
      a = Math.min(Math.max(m, 0), 1);
    }
    const alphaAcc = emaAlpha(dt, acc.smoothingMs, snap);
    this.accentActivity += (a - this.accentActivity) * alphaAcc;

    // The accent multiplier has a floor term AND a boost term, which is what
    // makes it different from the stem lane rather than a second copy of it:
    // below the θ base in a plain section, above it in a surprising one. A
    // primary is never touched — the accents are the voice that reacts to
    // structure, and if the primaries did it too nothing would stand out.
    const aFloor = Math.min(Math.max(acc.floor, 0), 1);
    const boost = Math.max(acc.boost, 0);
    const act = this.accentActivity;
    const mul = Math.min(
      Math.max(aFloor + (1 - aFloor) * act + boost * act, 0),
      1 + boost,
    );
    for (let i = 0; i < k; i++) {
      const s = this.config.species[i];
      this.accentMul[i] = acc.enabled && s?.role === 'secondary' ? mul : 1;
    }
  }

  /**
   * Impulse hotspots, normalised 0..1 → world units. Radius is a fraction of the
   * *short* axis so a disc stays a disc on a non-square world — the same
   * conversion physarum does, one unit system down.
   */
  private uploadSplashes(): void {
    const list = this.impulses?.splashes ?? [];
    const n = Math.min(list.length, MAX_SPLASHES);
    // One dead frame still has to be written, so the shader does not re-read a
    // stale disc; after that, an empty list costs nothing.
    if (n === 0 && this.splashCount === 0) return;

    const shortAxis = Math.min(this.worldW, this.worldH);
    for (let i = 0; i < n; i++) {
      const s = list[i];
      if (!s) continue;
      const o = i * FLOATS_PER_SPLASH;
      this.splashData[o + 0] = s.x * this.worldW;
      this.splashData[o + 1] = s.y * this.worldH;
      this.splashData[o + 2] = Math.max(s.radius * shortAxis, 1e-4);
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
    // uploadSpecies first: it is what computes `activeTotal`, which writeGlobals
    // then publishes. (Physarum writes globals first; it has no such dependency.)
    this.uploadSpecies();
    this.writeGlobals(pcgTick);

    const encoder = device.createCommandEncoder({ label: 'plife.step' });
    encoder.clearBuffer(this.cellCountBuf);
    encoder.clearBuffer(this.cellFillBuf);

    const gridBind = this.gridBinds[this.parity] as GPUBindGroup;
    const gridGroups = Math.ceil(this.totalParticles / GRID_WORKGROUP);

    // Impulse lane, ahead of the grid: particles are shoved first, and the count
    // / scatter / force chain then runs on the displaced velocities, so a splash
    // propagates through the sim rather than sitting on top of the frame. Same
    // ordering rationale as physarum's splash-before-sense.
    //
    // This is the one pass that writes the buffer the rest of the substep reads,
    // in place. It is safe because it is a separate compute pass (so the writes
    // are visible to everything encoded after it) and because each thread
    // touches only its own particle — no thread reads a neighbour here.
    if (this.splashCount > 0) {
      const splash = encoder.beginComputePass({ label: 'plife.splash' });
      splash.setPipeline(this.splashPipeline);
      splash.setBindGroup(0, this.splashBinds[this.parity] as GPUBindGroup);
      splash.dispatchWorkgroups(Math.ceil(this.totalParticles / PARTICLE_WORKGROUP));
      splash.end();
    }

    // Four passes rather than one with four dispatches: each stage reads what the
    // previous stage wrote, and pass boundaries make that ordering unambiguous.
    //
    // All three grid passes run in BOTH modes, and that is a deliberate
    // departure from "brute needs no grid". Brute does not need the *binning* —
    // it needs the **compaction**, and count/scan/scatter is exactly a
    // compaction that happens to be sorted by cell. The alternative is a naive
    // `0..maxParticles` inner loop, which at shipped populations wastes ~87% of
    // an already-quadratic pass on absent pool slots; against that, three
    // one-thread-per-particle memory passes are free. See the brute branch in
    // step.wgsl for how it reads the live count back out of the prefix sum.
    const count = encoder.beginComputePass({ label: 'plife.count' });
    count.setPipeline(this.countPipeline);
    count.setBindGroup(0, gridBind);
    count.dispatchWorkgroups(gridGroups);
    count.end();

    const scan = encoder.beginComputePass({ label: 'plife.scan' });
    scan.setPipeline(this.scanPipeline);
    scan.setBindGroup(0, gridBind);
    scan.dispatchWorkgroups(1);
    scan.end();

    const scatter = encoder.beginComputePass({ label: 'plife.scatter' });
    scatter.setPipeline(this.scatterPipeline);
    scatter.setBindGroup(0, gridBind);
    scatter.dispatchWorkgroups(gridGroups);
    scatter.end();

    const brute = this.pairSearch() === 'brute';
    const force = encoder.beginComputePass({ label: 'plife.force' });
    force.setPipeline(brute ? this.stepBrutePipeline : this.stepPipeline);
    force.setBindGroup(0, this.stepBinds[this.parity] as GPUBindGroup);
    force.dispatchWorkgroups(Math.ceil(this.totalParticles / PARTICLE_WORKGROUP));
    force.end();

    device.queue.submit([encoder.finish()]);
    this.parity = 1 - this.parity;
  }

  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.ready || !this.ctx) return;

    // Post surfaces follow the canvas, not the world: a resize re-allocates them
    // (and invalidates the fade pass's feedback binding) but never touches the
    // particle state.
    this.post.ensureSize(this.ctx.width, this.ctx.height);
    if (this.post.sizeVersion !== this.fadeSizeVersion) this.rebuildFadeBinds();
    this.post.gamma = this.config.gamma;

    // How long this frame is, in 60 Hz frames. The feedback lane is applied once
    // per *rendered* frame by the fade pass, so both of its constants are
    // per-frame quantities and both were silently wrong on any display that is
    // not 60 Hz: at 240 Hz the echo decayed four times too fast and — far more
    // visibly — the radial zoom accumulated four times as much, which is what
    // manufactured the fake radial "tracers". Raising both to the dtFrames power
    // makes the authored numbers mean "per 60 Hz frame" on every monitor.
    //
    // The 0.25 s clamp is for a tab that was backgrounded: a two-second gap
    // would otherwise raise 0.88 to the 120th power and clear the echo outright,
    // which is a black flash on the first frame back.
    const now = performance.now();
    // Floored at 1e-3 s, not 0: performance.now() is coarsened and two renders
    // can share a timestamp, making the exponent 0 — and pow(x, 0) is 1 for
    // every x, which would switch a disabled or heavily-faded echo fully on
    // for that frame. The floor is a no-op at any real frame rate.
    this.renderDtFrames =
      this.lastRenderAt === 0
        ? 1
        : Math.min(Math.max((now - this.lastRenderAt) / 1000, 1e-3), 0.25) * 60;
    this.lastRenderAt = now;

    // Both render passes read Globals (exposure, feedback) and the species block
    // (colour, size, stretch), and a frame can be drawn with zero substeps. This
    // is also the LAST Globals write before the fade pass, which is what lets
    // the substep path publish an uncorrected `renderDtFrames` harmlessly — the
    // only reader of words 18/19 is the fade pass, and it runs after this line.
    this.uploadSpecies();
    this.writeGlobals(this.lastPcgTick);

    const fade = encoder.beginRenderPass({
      label: 'plife.fade',
      colorAttachments: [
        {
          view: this.post.hdrTargetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    fade.setPipeline(this.fadePipeline);
    fade.setBindGroup(0, this.fadeBinds[this.post.currentParity] as GPUBindGroup);
    fade.draw(3);
    fade.end();

    const draw = encoder.beginRenderPass({
      label: 'plife.particles',
      colorAttachments: [{ view: this.post.hdrTargetView, loadOp: 'load', storeOp: 'store' }],
    });
    draw.setPipeline(this.particlePipeline);
    draw.setBindGroup(0, this.particleBinds[this.parity] as GPUBindGroup);
    // 4 vertices (one strip quad) × the whole pool. Dormant instances collapse to
    // a degenerate quad in the vertex shader, which the rasteriser drops.
    draw.draw(4, this.totalParticles);
    draw.end();

    this.post.run(encoder, targetView);
  }

  /** Two bind groups, one per PostFx parity, differing only in the feedback source. */
  private rebuildFadeBinds(): void {
    const { device } = this.ctx as GpuContext;
    this.fadeBinds = [0, 1].map((parity) =>
      device.createBindGroup({
        label: `plife.fade.parity${parity}`,
        layout: this.fadeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.globalsBuf } },
          { binding: 1, resource: this.post.linearSampler },
          { binding: 2, resource: this.post.hdrViewAt(1 - parity) },
        ],
      }),
    );
    this.fadeSizeVersion = this.post.sizeVersion;
  }

  dispose(): void {
    if (!this.ready) return;
    this.ready = false;
    this.post.dispose();
    this.clearSnapshot();
    this.globalsBuf.destroy();
    this.speciesBuf.destroy();
    this.interactionBuf.destroy();
    for (const b of this.particleBuf) b.destroy();
    this.cellCountBuf.destroy();
    this.cellFillBuf.destroy();
    this.cellStartBuf.destroy();
    this.sortedIdxBuf.destroy();
    this.splashBuf.destroy();
    for (const t of [...this.farSplat, ...this.farTmp, ...this.farNear, ...this.farDog]) {
      t.destroy();
    }
    for (const b of this.blurParamBufs) b.destroy();
  }

  // ── GPU uploads ────────────────────────────────────────────────────────────

  /**
   * The three K² blocks, in θ's own order: attraction, then maxR, then minR.
   * Called from `applyTheta` (every modulated tick) and once at init.
   *
   * The clamps here are the *last* line of defence and they are not redundant
   * with the registry's bounds: a hand-edited mapping file, or a panel slider
   * that has not been written yet, can put a radius outside the range the grid
   * search is correct for.
   *
   * Note which cap this is. maxR is clamped to `MAX_REACH_BRUTE` — the AUTHORED
   * ceiling, i.e. what a *file* may legally contain — and NOT to the reach of
   * the pair-search mode currently in force. The runtime cap belongs to the
   * shader, which derives it from the live cell size and stencil (grid) or the
   * live world size (brute); doing it here as well would bake the current mode
   * into the uploaded numbers and mean that switching to grid and back had
   * quietly truncated the matrix.
   */
  uploadInteractions(): void {
    if (!this.ready || !this.ctx) return;
    const k = this.config.speciesCount;
    const kk = k * k;
    const d = this.interactionData;
    for (let i = 0; i < kk; i++) {
      d[i] = this.config.attraction[i] ?? 0;
      d[kk + i] = Math.min(Math.max(this.config.maxR[i] ?? 0, MIN_R_FLOOR), MAX_REACH_BRUTE);
      d[2 * kk + i] = Math.min(Math.max(this.config.minR[i] ?? 0, MIN_R_FLOOR), MAX_MIN_R);
    }
    this.ctx.device.queue.writeBuffer(this.interactionBuf, 0, d);
  }

  private writeGlobals(pcgTick: number): void {
    const ctx = this.ctx as GpuContext;
    const u = this.globalsU32;
    const f = this.globalsF32;
    const cfg = this.config;
    f[0] = this.worldW;
    f[1] = this.worldH;
    u[2] = this.gridW;
    u[3] = this.gridH;
    f[4] = this.cellW;
    f[5] = this.cellH;
    u[6] = cfg.speciesCount;
    u[7] = this.segSize;
    u[8] = this.totalParticles;
    u[9] = this.seed >>> 0;
    u[10] = pcgTick >>> 0;
    const macros = cfg.macros;
    f[11] = SUBSTEP_DT;
    // Macro `force`, outside θ: forceGain is modulated, this multiplies whatever
    // the modulator left there — same relationship stem-follow has to brightness.
    f[12] = Math.max(cfg.forceGain, 0) * Math.max(macros.force, 0);
    // Scene exposure (θ) and the grade's manual trim are one multiply, applied in
    // the particle shader. Deliberate: auto-exposure measures the HDR surface, so
    // anything applied after the measurement is invisible to the controller.
    f[13] = Math.max(cfg.exposure, 0) * Math.pow(2, cfg.render.grade.exposureEv);
    f[14] = this.respawnFraction;
    u[15] = this.respawnKey >>> 0;
    // Macro `agility`, half of it: speed up here, damping down in uploadSpecies.
    f[16] = Math.max(Math.max(cfg.maxSpeed, 1e-4) * Math.max(macros.agility, 0), 1e-4);
    u[17] = this.activeTotal >>> 0;
    // The feedback lane, converted from "per 60 Hz frame" (what the sliders and
    // the shipped defaults mean) to "per frame at this display's refresh rate"
    // (what the fade pass actually applies). Both are geometric per frame, so
    // the conversion is a power, and `renderDtFrames` is 1 on a 60 Hz display —
    // the numbers are unchanged there, which is the point.
    //
    // Amount is clamped below 1 first: the lane is geometric, so >= 1 never
    // decays, and exponentiating a value >= 1 would not fix that.
    //
    // Macro `trails` multiplies the authored amount BEFORE the power, because the
    // per-60-Hz-frame semantics belong to the *effective* value: applying the
    // macro after the exponentiation would make it mean something different on
    // every refresh rate.
    const fbAmount = Math.min(
      Math.max(cfg.render.feedback.amount, 0) * Math.max(macros.trails, 0),
      0.97,
    );
    const fbZoom = Math.max(cfg.render.feedback.zoom, 1e-3);
    f[18] = Math.pow(fbAmount, this.renderDtFrames);
    f[19] = Math.pow(fbZoom, this.renderDtFrames);
    // Floored well above zero: the shader divides dt by these, and a tau of 0
    // would be an instant ramp — which is exactly the pop-in the lane exists to
    // remove, arrived at by accident from a slider dragged to the end.
    f[20] = Math.max(cfg.population.riseTau, 1e-3);
    f[21] = Math.max(cfg.population.fallTau, 1e-3);
    u[22] = this.splashCount >>> 0;
    u[23] = this.stencil() >>> 0;
    u[24] = this.farActive() ? 1 : 0;
    u[25] = 0;
    u[26] = 0;
    u[27] = 0;
    ctx.device.queue.writeBuffer(this.globalsBuf, 0, this.globalsBytes);
  }

  private uploadSpecies(): void {
    const ctx = this.ctx as GpuContext;
    if (this.paletteDirty) this.refreshPalette();
    const d = this.speciesData;
    const list = this.config.species;
    const imp = this.impulses;
    const macros = this.config.macros;
    // Macro `agility`, the damping half. Floored at 0.1 rather than 0: a species
    // with no friction at all never settles, it just accumulates speed until the
    // maxSpeed clamp is the only thing shaping the motion.
    const frictionDiv = Math.max(macros.agility, 1e-3);
    let active = 0;

    for (let k = 0; k < this.config.speciesCount; k++) {
      const s = list[k];
      const o = k * FLOATS_PER_SPECIES;
      if (!s) {
        d.fill(0, o, o + FLOATS_PER_SPECIES);
        this.targetAlive[k] = 0;
        continue;
      }
      // The impulse lane, applied here and nowhere else: multiplicative on top of
      // whatever wrote the base value this tick, and always 1.0 when idle.
      const depositMul = imp ? (imp.depositMul[k] ?? 1) : 1;
      const brightMul = imp ? (imp.brightMul[k] ?? 1) : 1;
      const sensorMul = imp ? (imp.sensorMul[k] ?? 1) : 1;

      // Composition order, identical to physarum's and load-bearing for the same
      // reason:
      //
      //   base (slider / θ)  ×  stem-follow  →  clamp to MAX_BRIGHTNESS
      //                                          ×  impulse flash
      //
      // The stem lane is the slow legible part and is clamped with the base; the
      // flash is the fast transient part and multiplies *outside* that clamp, so
      // a kick still reads on a species the stem lane is holding at its floor.
      const follow = this.brightFollow ? (this.brightFollow[k] ?? 1) : 1;
      const light = Math.min(Math.max(s.brightness, 0) * Math.max(follow, 0), MAX_BRIGHTNESS);
      // Macro `accents`, both halves of it: an accent species gets brighter AND
      // more numerous, which is what makes one slider read as "more filigree"
      // rather than as two unrelated changes. Primaries are untouched by design.
      const accentMacro = s.role === 'secondary' ? Math.max(macros.accents, 0) : 1;
      const weight = Math.max(s.intensity, 0) * accentMacro * light * Math.max(brightMul, 0);

      // Population target. θ owns the base fraction; the two lanes multiply on
      // top of it and the product is clamped back into 0..1 before it becomes a
      // count — exactly the layering brightness already uses (base × stem-follow,
      // clamped, × impulse flash), applied to how many rather than how bright.
      //
      // This is a *target*, not a live count. The GPU eases each particle's
      // energy toward it over riseTau / fallTau; nothing here appears or
      // disappears instantly, which is the whole point of the lane.
      //
      // The two macros join the same product, pre-clamp: `density` on every
      // species and `accents` on the secondaries only.
      //
      // `enabled` gates the whole thing, and it is the OUTERMOST factor on
      // purpose. Everything else in this product is a scaling — a lane, a macro,
      // a θ base — and any of them can be argued back up by another; the switch
      // is a decision about whether the voice exists, so it composes last and
      // nothing composes over it. The corollary is worth stating because it is
      // the one interaction that could surprise: this is a product of
      // non-negative terms, so ANY zero wins. `accents` at 0 already silences
      // every secondary, and re-enabling one secondary while the macro sits at 0
      // brings back nothing — that is the macro's zero, not a broken switch.
      //
      // Nothing here is instant. This is the *target*; the GPU eases each
      // particle's energy toward it over riseTau / fallTau, so switching a
      // species off drains it over the fall-τ instead of popping it out.
      const frac = s.enabled
        ? Math.min(Math.max(s.aliveFraction, 0), 1) *
          Math.max(this.popMul[k] ?? 1, 0) *
          Math.max(this.accentMul[k] ?? 1, 0) *
          Math.max(macros.density, 0) *
          accentMacro
        : 0;
      const alive = Math.floor(Math.min(Math.max(frac, 0), 1) * this.segSize);
      this.targetAlive[k] = alive;
      active += alive;

      // Colour arrives at the GPU premultiplied. The particle shader has no
      // per-species work to do beyond the splat, which matters when it runs
      // millions of times a frame.
      d[o + 0] = (this.paletteRgb[k * 3 + 0] as number) * weight;
      d[o + 1] = (this.paletteRgb[k * 3 + 1] as number) * weight;
      d[o + 2] = (this.paletteRgb[k * 3 + 2] as number) * weight;
      d[o + 3] = alive;
      d[o + 4] = Math.min(Math.max(s.size, 1e-6), MAX_SIZE);
      d[o + 5] = Math.min(Math.max(s.stretch, 0), MAX_STRETCH);
      // The impulse engine's `deposit` lane maps to *force* here: physarum's
      // "write harder into the trail" and plife's "push harder on the field" are
      // the same idea — energy into the system — expressed in each substrate's
      // own currency, so one ResponseConfig drives both without re-tuning.
      d[o + 6] = Math.max(s.forceScale, 0) * Math.max(depositMul, 0);
      // Macro `agility` divides here (less damping = longer slides) and the 0.1
      // floor is what keeps "agility at its ceiling" from meaning "no damping".
      d[o + 7] = Math.min(
        Math.max(Math.max(s.friction, 0) / frictionDiv, 0.1),
        MAX_FRICTION,
      );
      // Macro `chaos`.
      d[o + 8] = Math.max(s.wander, 0) * Math.max(macros.chaos, 0);
      // …and the `sensor` lane maps to reach, which is what it already means.
      // Macro `reach` sits between the slider and the impulse lane; the shader's
      // own clamp at the cell size is still the correctness bound.
      d[o + 9] = Math.min(
        Math.max(s.radiusScale, 0.05) * Math.max(macros.reach, 0) * Math.max(sensorMul, 0),
        MAX_RADIUS_SCALE,
      );
      d[o + 10] = 1; // reserved
      d[o + 11] = 0;
      d[o + 12] = 0;
      d[o + 13] = 0;
      d[o + 14] = 0;
      d[o + 15] = 0;
    }

    this.activeTotal = active;
    ctx.device.queue.writeBuffer(this.speciesBuf, 0, d);
  }
}

/**
 * One-pole EMA coefficient for a `smoothingMs` time constant over `dt` seconds.
 * `snap` returns 1, which adopts the new value outright — the transport-jump
 * case, where the filter's history belongs to a different moment of the song.
 */
function emaAlpha(dt: number, smoothingMs: number, snap: boolean): number {
  if (snap) return 1;
  if (!(dt > 0)) return 0;
  return 1 - Math.exp(-dt / (Math.max(smoothingMs, 1) / 1000));
}

// ── extras validation helpers ────────────────────────────────────────────────
//
// Deliberately total functions: `applyExtras` is handed whatever was in the
// file, and the contract is that it never throws — a broken block loses its
// values to the defaults, not the whole load.

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

/** A radius band, clamped into `[MIN_R_FLOOR, cap]` with `lo <= hi` restored. */
function readBand(
  raw: unknown,
  dst: { lo: number; hi: number },
  def: { lo: number; hi: number },
  cap: number,
): void {
  const o = plainObject(raw);
  const lo = clampNum(o['lo'], def.lo, MIN_R_FLOOR, cap);
  const hi = clampNum(o['hi'], def.hi, MIN_R_FLOOR, cap);
  dst.lo = Math.min(lo, hi);
  dst.hi = Math.max(lo, hi);
}
