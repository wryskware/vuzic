/**
 * Seeded random-projection modulation — plan.md Revision 3, and the whole of the
 * mapping layer now that anchors, k-means and the simplex are gone.
 *
 *   ẑ (per-dim z-scored embedding, 10 Hz)
 *        │
 *        ├── w₀·ẑ ──► tanh ──► base₀ + half₀·(…) ──┐
 *        ├── w₁·ẑ ──► tanh ──► base₁ · exp(…)   ───┤  per-class slew  ──► PhysarumConfig
 *        └── …                                     ┘
 *
 * Each modulatable parameter owns a random unit direction wᵢ in signal space,
 * keyed on hash(seed, i). With a z-scored input and a unit projection, wᵢ·ẑ is
 * roughly N(0,1) whatever the track, which is the entire reason the depths mean
 * the same thing everywhere and can be shipped LEGIBLE rather than cautious.
 *
 * `tanh` is the bound: the excursion can never leave ±halfᵢ, so no depth setting
 * can put a parameter outside its authored range and the final `clamp` is belt
 * and braces rather than the thing doing the work. Physarum tolerates arbitrary
 * in-range sweeps — that is why it was chosen (Decision 3) — so "always bounded,
 * always moving" is a complete safety argument.
 *
 * The seed does three jobs at once and they are deliberately the same seed:
 *   wiring    — which embedding directions move which parameter
 *   base      — the *personality*, jittered around the shipped defaults
 *   the world — agent positions and impulse hotspots (the sim's own use)
 * so "reroll" is one act and a pinned seed reproduces the run exactly.
 *
 * What this is NOT: it does not learn, and it does not know what any embedding
 * dimension means. It is the zero-training baseline of the distilled-NN mapping
 * in plan.md's "Later", and the reroll/keep choices it produces are that NN's
 * preference data.
 */
import { hash3 } from '../sim/impulses.ts';
import type { PhysarumSim } from '../sim/physarum/physarum.ts';
import { mergeRenderConfig } from '../sim/render/config.ts';
import type { FeaturesFrame, TimelineSampler } from '../timeline/sampler.ts';
import type { Embedding, Timeline } from '../timeline/types.ts';
import {
  applyVector,
  CLASS_SLOW,
  fieldClasses,
  modulationMask,
  modulationSlots,
  MOD_GROUPS,
  presetFromConfig,
  presetToVector,
  vectorLength,
  type ModGroup,
  type ModSpec,
} from './preset.ts';
import { SlewLimiter } from './slew.ts';
import type { ModulationConfig } from './types.ts';

/** mulberry32 — small, fast, deterministic across engines for a u32 seed. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Domain separators, so wiring and personality never draw the same stream. */
const KEY_WIRING = 0x5eed_0117;
const KEY_BASE = 0x0000_ba5e;

/**
 * z-scores beyond this are clipped. Real embeddings have heavy tails on a few
 * dims (a silent intro is genuinely 8σ from the mean of a loud track) and one
 * such dim would otherwise dominate every projection it appears in. ±4 keeps
 * >99.99% of a normal signal untouched and turns the outliers into a plateau.
 */
export const Z_CLAMP = 4;

export type SignalKind = 'embedding' | 'latent';

/**
 * The modulator's input, z-scored per dimension over the whole track at load.
 *
 * Per-*track* standardisation is the point: it makes "how unusual is this moment
 * for this song" the signal, rather than "where does this song sit among all
 * songs", and it is what lets one set of depths work on any track. Cost is one
 * pass over frames x dims (2725 x 1024 is ~3 M floats — a few ms) and one copy.
 */
export class ModulationSignal {
  readonly kind: SignalKind;
  readonly dims: number;
  readonly frames: number;
  readonly hopSeconds: number;
  readonly source: string;
  readonly mean: Float64Array;
  readonly std: Float64Array;
  /** frames x dims, already z-scored and clipped */
  private readonly z: Float32Array;

  constructor(
    kind: SignalKind,
    raw: Float32Array,
    frames: number,
    dims: number,
    hopSeconds: number,
    source: string,
    stride = dims,
    offset = 0,
  ) {
    this.kind = kind;
    this.dims = dims;
    this.frames = frames;
    this.hopSeconds = hopSeconds;
    this.source = source;
    this.mean = new Float64Array(dims);
    this.std = new Float64Array(dims);
    this.z = new Float32Array(frames * dims);

    // Non-finite samples are excluded here and written as z = 0 below. This is
    // the *only* place the float path is checked, and it has to be: a single NaN
    // in the sidecar would otherwise poison mean[d]/std[d] for that dimension,
    // and NaN fails every `<`/`>` comparison so neither the z clamp nor the
    // target clamp downstream would catch it — it would reach SlewLimiter.step,
    // whose state is a feedback term, and latch there for the rest of the run.
    const counts = new Int32Array(dims);
    for (let f = 0; f < frames; f++) {
      const o = f * stride + offset;
      for (let d = 0; d < dims; d++) {
        const x = raw[o + d] as number;
        if (!Number.isFinite(x)) continue;
        this.mean[d] = (this.mean[d] as number) + x;
        counts[d] = (counts[d] as number) + 1;
      }
    }
    for (let d = 0; d < dims; d++)
      this.mean[d] = (this.mean[d] as number) / Math.max(counts[d] as number, 1);
    for (let f = 0; f < frames; f++) {
      const o = f * stride + offset;
      for (let d = 0; d < dims; d++) {
        const x = raw[o + d] as number;
        if (!Number.isFinite(x)) continue;
        const delta = x - (this.mean[d] as number);
        this.std[d] = (this.std[d] as number) + delta * delta;
      }
    }
    let bad = 0;
    for (let d = 0; d < dims; d++) {
      // Population sd. A constant dimension gets sd 0 and is floored to 1, which
      // makes its z exactly 0 forever — it contributes nothing rather than NaN.
      const v = Math.sqrt((this.std[d] as number) / Math.max(counts[d] as number, 1));
      this.std[d] = v > 1e-12 ? v : 1;
    }
    for (let f = 0; f < frames; f++) {
      const o = f * stride + offset;
      const q = f * dims;
      for (let d = 0; d < dims; d++) {
        const x = raw[o + d] as number;
        if (!Number.isFinite(x)) {
          this.z[q + d] = 0;
          bad++;
          continue;
        }
        const v = (x - (this.mean[d] as number)) / (this.std[d] as number);
        this.z[q + d] = v < -Z_CLAMP ? -Z_CLAMP : v > Z_CLAMP ? Z_CLAMP : v;
      }
    }
    if (bad > 0)
      console.warn(
        `modulation: ${bad} non-finite sample(s) in ${source}; zeroed (they contribute nothing)`,
      );
  }

  get label(): string {
    return `${this.kind === 'embedding' ? 'embedding' : 'latent'}-${this.dims}`;
  }

  /**
   * Linear interpolation between grid frames — the same scheme `TimelineSampler`
   * uses, so the modulator moves in lockstep with everything else the tick feeds.
   */
  sample(time: number, out: Float32Array): Float32Array {
    const pos = Math.min(Math.max(time / this.hopSeconds, 0), this.frames - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, this.frames - 1);
    const t = pos - i0;
    const a = i0 * this.dims;
    const b = i1 * this.dims;
    if (t === 0 || i0 === i1) {
      out.set(this.z.subarray(a, a + this.dims));
      return out;
    }
    for (let d = 0; d < this.dims; d++) {
      const va = this.z[a + d] as number;
      out[d] = va + ((this.z[b + d] as number) - va) * t;
    }
    return out;
  }
}

/**
 * Pick the widest input the track actually shipped. The 1024-dim sidecar is
 * preferred (plan.md Revision 3: "the PCA intermediate is no longer
 * load-bearing"); a fresh clone has it gitignored away, so the 64-dim PCA
 * `latent` channel is a first-class fallback and says so once.
 */
export function embeddingSignal(emb: Embedding): ModulationSignal | null {
  if (emb.dims <= 0 || emb.frames <= 0) return null;
  return new ModulationSignal(
    'embedding',
    emb.data,
    emb.frames,
    emb.dims,
    emb.hopSeconds,
    emb.source,
  );
}

export function chooseSignal(timeline: Timeline, channelName = 'latent'): ModulationSignal | null {
  const emb = timeline.embedding;
  if (emb) {
    const wide = embeddingSignal(emb);
    if (wide) return wide;
  }
  const c = timeline.channels.get(channelName);
  if (!c || c.dims === 0) return null;
  console.info(
    `modulation: driving from the ${c.dims}-dim "${channelName}" channel` +
      ' (the wide sidecar upgrades this in the background if the track ships one)',
  );
  return new ModulationSignal(
    'latent',
    timeline.data,
    timeline.manifest.grid.frames,
    c.dims,
    timeline.manifest.grid.hopSeconds,
    `timeline channel "${channelName}"`,
    timeline.stride,
    c.offset,
  );
}

/**
 * A random unit vector in `dims` dimensions, from a seeded stream. Box–Muller
 * gives isotropy (uniform-per-component would concentrate on the cube's
 * diagonals, which correlates every parameter that shares a sign pattern), and
 * the normalisation is what makes w·ẑ ~ N(0,1) for a z-scored ẑ.
 */
export function unitDirection(seed: number, index: number, dims: number, out: Float32Array): void {
  const rng = makeRng(hash3(seed >>> 0, index >>> 0, KEY_WIRING));
  let norm = 0;
  for (let d = 0; d < dims; d += 2) {
    // rng() can return exactly 0; nudge it off the log's singularity.
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const a = 2 * Math.PI * u2;
    const g0 = r * Math.cos(a);
    out[d] = g0;
    norm += g0 * g0;
    if (d + 1 < dims) {
      const g1 = r * Math.sin(a);
      out[d + 1] = g1;
      norm += g1 * g1;
    }
  }
  const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
  for (let d = 0; d < dims; d++) out[d] = (out[d] as number) * inv;
}

/**
 * The seeded personality: every modulatable slot's base is the shipped default
 * displaced by up to ±jitter (±jitter in ln space when the slot is
 * multiplicative), then clamped into that slot's safe sub-range.
 *
 * This is the "species parameters should be randomized too" ask. It is a
 * *uniform* draw rather than a normal one on purpose — a normal would put most
 * seeds near the defaults and make rerolling feel like it did nothing, and the
 * user asked for differences between seeds to be bigger, not subtler.
 *
 * Out-of-range draws are **reflected** off the bound, not clamped. Clamping
 * looks harmless and is the opposite of what the user asked for: several shipped
 * defaults sit right on their safe bound (bass diffuseCentre 0.12 with lo 0.111,
 * drums sensorDist.p1 4 with lo 0.5, bass aliveFraction 0.85 with hi 1), so a
 * clamped uniform draw collects a fifth to a half of all seeds on exactly that
 * bound — those seeds are identical in that slot, which is precisely the
 * seed-to-seed variety Revision 3 asked to increase. Reflection keeps the full
 * ±jitter spread and puts zero probability mass on any single value; shrinking
 * the draw to symmetric headroom would have been the other obvious fix and is
 * worse, because for exactly those slots the headroom is nearly zero.
 */
function reflect(x: number, lo: number, hi: number): number {
  if (!(hi > lo)) return lo;
  const span = hi - lo;
  let t = (x - lo) % (2 * span);
  if (t < 0) t += 2 * span;
  return lo + (t <= span ? t : 2 * span - t);
}

export function baseVector(
  seed: number,
  defaults: ArrayLike<number>,
  slots: readonly (ModSpec | null)[],
  out: Float64Array,
): Float64Array {
  for (let i = 0; i < slots.length; i++) {
    const spec = slots[i];
    const def = (defaults[i] as number) ?? 0;
    if (!spec) {
      out[i] = def;
      continue;
    }
    const u = makeRng(hash3(seed >>> 0, i >>> 0, KEY_BASE))() * 2 - 1;
    if (spec.mult && def > 0 && spec.lo > 0) {
      // Multiplicative slots reflect in ln space, where the jitter is defined.
      out[i] = Math.exp(
        reflect(Math.log(def) + spec.jitter * u, Math.log(spec.lo), Math.log(spec.hi)),
      );
      continue;
    }
    const v = spec.mult ? def * Math.exp(spec.jitter * u) : def + spec.jitter * u;
    out[i] = reflect(v, spec.lo, spec.hi);
  }
  return out;
}

export type ModulationMode = 'manual' | 'modulated';

export interface ModulatorEvents {
  onSectionChange?: (segmentIndex: number, respawned: number) => void;
}

/** Mean |tanh excursion| per group, for the workbench's live readout. */
export type GroupExcursions = Record<ModGroup, number>;

export class Modulator {
  mode: ModulationMode = 'manual';
  /** hold the current ẑ: parameters stop morphing, slew still settles */
  frozen = false;

  /** Swappable: the wide sidecar arrives after first paint (see `attachSignal`). */
  signal: ModulationSignal | null;

  private readonly sim: PhysarumSim;
  private readonly sampler: TimelineSampler;
  private readonly events: ModulatorEvents;

  private cfg!: ModulationConfig;
  private seed: number;

  private readonly slots: readonly (ModSpec | null)[];
  private readonly mask: Uint8Array;
  private readonly length: number;
  /** length x dims, row-major; row i is parameter i's projection direction */
  private w: Float32Array;
  private readonly base: Float64Array;
  private readonly target: Float64Array;
  private readonly defaults: Float64Array;
  private zbuf: Float32Array;
  private slew!: SlewLimiter;

  private readonly excursionSum: Float64Array = new Float64Array(MOD_GROUPS.length);
  private readonly excursionCount: Int32Array = new Int32Array(MOD_GROUPS.length);
  private lastSegment = -2;

  constructor(
    sim: PhysarumSim,
    sampler: TimelineSampler,
    signal: ModulationSignal | null,
    config: ModulationConfig,
    seed: number,
    events: ModulatorEvents = {},
  ) {
    this.sim = sim;
    this.sampler = sampler;
    this.signal = signal;
    this.events = events;
    this.seed = seed >>> 0;

    const k = sim.config.speciesCount;
    this.length = vectorLength(k);
    this.slots = modulationSlots(k);
    this.mask = modulationMask(k);
    // Defaults are the config as the app started: the shipped art direction,
    // captured before any seed has had a chance to displace it.
    this.defaults = presetToVector(presetFromConfig(sim.config), k);
    this.base = new Float64Array(this.length);
    this.target = new Float64Array(this.length);
    this.zbuf = new Float32Array(signal?.dims ?? 0);
    this.w = new Float32Array(this.length * (signal?.dims ?? 0));

    this.setConfig(config);
    this.rewire();
  }

  get config(): ModulationConfig {
    return this.cfg;
  }

  get currentSeed(): number {
    return this.seed;
  }

  /** Modulation needs an input; without one the app is a slider box, honestly. */
  get available(): boolean {
    return this.signal !== null;
  }

  get unavailableReason(): string {
    return this.signal
      ? ''
      : 'no embedding sidecar and no "latent" channel — nothing to project from';
  }

  get sourceLabel(): string {
    return this.signal ? this.signal.label : 'none';
  }

  /** How many slots the music actually moves, of how many in θ. */
  get modulatedCount(): number {
    let n = 0;
    for (const m of this.mask) n += m;
    return n;
  }

  setConfig(config: ModulationConfig): void {
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
    // Same trick for the phase-7 render block.
    mergeRenderConfig(this.sim.config.render, config.render);
    config.render = this.sim.config.render;
    this.sim.invalidatePalette();

    this.slew = new SlewLimiter(fieldClasses(k), config.slew);
    this.slew.reset(presetToVector(presetFromConfig(this.sim.config), k));
    this.lastSegment = -2;
  }

  /** Re-read slew rates after the panel edits them. */
  syncRates(): void {
    this.slew.setRates(this.cfg.slew);
  }

  setMode(mode: ModulationMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    // Start from where the config actually is, so switching modes never steps.
    this.slew.reset(presetToVector(presetFromConfig(this.sim.config), this.sim.config.speciesCount));
    this.lastSegment = -2;
  }

  /**
   * A new seed is a new world: new wiring, new personality, and (the caller's
   * job) a re-scattered sim. Wired to `sim.onSeedChange` in main, so a reseed, a
   * pinned-seed reload and a snapshot restore all land here without anyone
   * having to remember to call it.
   */
  setSeed(seed: number): void {
    const next = seed >>> 0;
    this.seed = next;
    this.rewire();
    this.applyBase();
  }

  /**
   * Swap in a wider input that landed after startup. The 1024-dim sidecar is
   * ~11 MB and awaiting it before the first frame meant a blank canvas for the
   * whole download on any real network, so the app boots on the 64-dim PCA
   * fallback and upgrades here when the fetch resolves.
   *
   * The seeded personality is unchanged by this (it does not depend on the
   * input), but the *wiring* does — the projection rows are dims-long — so this
   * rewires. Visible effect is a smooth move through the slew limiter to a
   * different point in the same bounded ranges, which is what modulation does
   * every tick anyway.
   */
  attachSignal(signal: ModulationSignal): void {
    this.signal = signal;
    this.zbuf = new Float32Array(signal.dims);
    this.w = new Float32Array(this.length * signal.dims);
    this.rewire();
    console.info(`modulation: upgraded input to ${signal.label} (${signal.source})`);
  }

  /** Rebuild projections and personality for the current seed. */
  private rewire(): void {
    const dims = this.signal?.dims ?? 0;
    if (dims > 0) {
      if (this.w.length !== this.length * dims) this.w = new Float32Array(this.length * dims);
      const row = new Float32Array(dims);
      for (let i = 0; i < this.length; i++) {
        if (this.mask[i] !== 1) continue;
        unitDirection(this.seed, i, dims, row);
        this.w.set(row, i * dims);
      }
    }
    baseVector(this.seed, this.defaults, this.slots, this.base);
  }

  /**
   * Write the personality straight into the live config and reset the slew to it.
   * Called on every seed change so a reroll is instantly visible — including in
   * manual mode, where it is the only thing that writes.
   */
  applyBase(): void {
    applyVector(this.sim.config, this.base, this.mask);
    this.slew.reset(this.base);
    this.sim.uploadMatrix();
  }

  /** Projection direction of one slot, copied. Exported shape for tests. */
  direction(index: number): Float32Array {
    const dims = this.signal?.dims ?? 0;
    return this.w.slice(index * dims, index * dims + dims);
  }

  /** The seeded personality vector, copied. */
  baseValues(): Float64Array {
    return this.base.slice();
  }

  /** θ as the sim currently has it — what the tuning log records. */
  currentTheta(): number[] {
    return Array.from(
      presetToVector(presetFromConfig(this.sim.config), this.sim.config.speciesCount),
    );
  }

  /** Mean |excursion| per group over the last tick, 0…1. */
  excursions(): GroupExcursions {
    const out = {} as GroupExcursions;
    MOD_GROUPS.forEach((g, i) => {
      const n = this.excursionCount[i] as number;
      out[g] = n > 0 ? (this.excursionSum[i] as number) / n : 0;
    });
    return out;
  }

  /**
   * One tick. `dt` is the fixed sim timestep, so response behaviour does not
   * depend on frame rate or on how many ticks a frame drains.
   *
   * Order matters and is fixed in main: modulator → impulses → sim. Impulses
   * multiply on top of whatever this wrote and are never slewed through it.
   */
  update(frame: FeaturesFrame, dt: number): void {
    // Boundary re-seed is an *event*, not scene machinery (Revision 3), so it
    // fires on its own toggle regardless of whether modulation is driving.
    // Segment tracking runs unconditionally and the toggle gates the *response*.
    // Short-circuiting on `enabled` instead would let `lastSegment` go stale
    // while the toggle is off, so re-enabling mid-track would read as a
    // transition and fire a partial reseed at an arbitrary point in the song.
    const crossed = this.trackSegment(frame.time);
    const boundaryHit = crossed && this.cfg.boundary.enabled;
    if (boundaryHit) {
      this.sim.partialReseed(this.lastSegment, this.cfg.boundary.respawnFraction);
      this.events.onSectionChange?.(this.lastSegment, this.cfg.boundary.respawnFraction);
    }

    if (this.mode !== 'modulated' || !this.signal) return;

    if (!this.frozen) this.signal.sample(frame.time, this.zbuf);
    this.computeTarget();

    if (boundaryHit) this.slew.snapClass(CLASS_SLOW, this.target, this.cfg.boundary.snapFraction);

    // responseSpeed scales time, not the coefficients: doubling it makes every
    // class exactly twice as quick and keeps their ratios (the tuned part) intact.
    const v = this.slew.step(this.target, dt * Math.max(this.cfg.responseSpeed, 0));
    applyVector(this.sim.config, v, this.mask);
    this.sim.uploadMatrix();
  }

  /** Advance `lastSegment`; true only for a genuine crossing (never the first sight). */
  private trackSegment(time: number): boolean {
    const seg = this.sampler.segmentIndexAt(time);
    if (seg < 0 || seg === this.lastSegment) return false;
    const first = this.lastSegment === -2;
    this.lastSegment = seg;
    return !first;
  }

  private computeTarget(): void {
    const dims = this.signal?.dims ?? 0;
    const z = this.zbuf;
    const w = this.w;
    const depth = Math.max(this.cfg.depth, 0);
    this.excursionSum.fill(0);
    this.excursionCount.fill(0);

    for (let i = 0; i < this.length; i++) {
      const spec = this.slots[i];
      const base = this.base[i] as number;
      if (!spec) {
        this.target[i] = base;
        continue;
      }
      const gi = MOD_GROUPS.indexOf(spec.group);
      const gd = this.cfg.groupDepth[spec.group] ?? 1;
      let raw = 0;
      const o = i * dims;
      for (let d = 0; d < dims; d++) raw += (w[o + d] as number) * (z[d] as number);
      const e = Math.tanh(depth * Math.max(gd, 0) * raw);
      const v = spec.mult ? base * Math.exp(spec.half * e) : base + spec.half * e;
      // NaN-safe on purpose: `<`/`>` both answer false for NaN, so a plain clamp
      // would pass one through to the slew limiter and latch it forever. The
      // signal is already sanitised at load; this is the second line of defence
      // against a future arithmetic slip in the projection.
      this.target[i] = Number.isFinite(v)
        ? v < spec.lo
          ? spec.lo
          : v > spec.hi
            ? spec.hi
            : v
        : base;
      if (gi >= 0) {
        this.excursionSum[gi] = (this.excursionSum[gi] as number) + Math.abs(e);
        this.excursionCount[gi] = (this.excursionCount[gi] as number) + 1;
      }
    }
  }

  /**
   * The target vector for an arbitrary ẑ, without touching the sim. Pure, and
   * the hook the bounds / determinism / group-isolation tests drive.
   */
  targetFor(z: ArrayLike<number>, out = new Float64Array(this.length)): Float64Array {
    const dims = this.signal?.dims ?? 0;
    for (let d = 0; d < dims; d++) this.zbuf[d] = (z[d] as number) ?? 0;
    this.computeTarget();
    out.set(this.target);
    return out;
  }
}
