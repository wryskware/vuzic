/**
 * Seeded random-projection modulation over a **named driver bank** — plan.md
 * Revision 4, and the whole of the mapping layer now that anchors, k-means and
 * the simplex are gone.
 *
 *   timeline ──► DriverBank (D ≈ 16 named, z-scored channels, 10 Hz)
 *                     │
 *                gains ⊙ ẑ          ← the tuning surface: one slider per driver
 *                     │
 *        ├── w₀·(g⊙ẑ) ──► tanh ──► base₀ + half₀·(…) ──┐
 *        ├── w₁·(g⊙ẑ) ──► tanh ──► base₁ · exp(…)   ───┤ per-class slew ──► config
 *        └── …                                        ┘
 *
 * **Why the bank exists.** Revision 3 projected from the raw 1024-dim MuQ
 * embedding. That reacts to everything and isolates nothing: every parameter is
 * a different random blend of a thousand unnamed directions, so when the image
 * does something you cannot say what in the music caused it, and there is no
 * knob that means anything. Nobody can tune 1024 weights. So the input is now a
 * short bank of channels a human can name, meter and mute:
 *
 *   0..2   novelty·4bar, novelty·16bar, chorus-ness  — the structure channels,
 *          shipped in the timeline since v2 and never used until now
 *   3..D   pc-1 … pc-13, the highest-variance components of the 64-dim latent
 *
 * Each is z-scored per track, so w·ẑ is still ~N(0,1) and the depths still mean
 * the same thing on any song. Each has a gain (0..2, default 1) applied *before*
 * the projection, so muting a driver silences it in every wiring at once — which
 * is the property that makes the bank a tuning surface rather than a readout.
 *
 * **Variance reordering.** The analysis PCA emits components in variance order,
 * but the post-PCA smoothing stage does not preserve that ordering (it is a
 * per-dimension filter with the same coefficients, applied to signals with very
 * different spectra). Measured on Free Fall, the stored order is 0, 2, 4, 5, 8,
 * 11, 6 … — dims 1 and 3 carry less variance than dim 8. So the bank measures
 * variance over the actual track at load and sorts descending. pc-1 is then the
 * component that genuinely moves most, which is what makes the top 13 the right
 * 13 to keep.
 *
 * `tanh` is the bound: the excursion can never leave ±halfᵢ, so no depth or gain
 * setting can put a parameter outside its authored range and the final `clamp`
 * is belt and braces. Physarum tolerates arbitrary in-range sweeps — that is why
 * it was chosen (Decision 3) — so "always bounded, always moving" is a complete
 * safety argument.
 *
 * The seed does three jobs at once and they are deliberately the same seed:
 *   wiring    — which drivers move which parameter
 *   base      — the *personality*, jittered around the shipped defaults
 *   the world — agent positions and impulse hotspots (the sim's own use)
 * so "reroll" is one act and a pinned seed reproduces the run exactly.
 *
 * Brightness is **not** here any more (Revision 4): it left the registry for the
 * stem-follow lane in `stemfollow.ts`, which this class drives every tick in
 * both manual and modulated mode.
 */
import { hash3 } from '../sim/impulses.ts';
import type { PhysarumSim } from '../sim/physarum/physarum.ts';
import { mergeRenderConfig } from '../sim/render/config.ts';
import type { FeaturesFrame, TimelineSampler } from '../timeline/sampler.ts';
import type { Timeline } from '../timeline/types.ts';
import { STEM_NAMES } from '../timeline/types.ts';
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
import { StemFollow } from './stemfollow.ts';
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
 * z-scores beyond this are clipped. Real music channels have heavy tails on a
 * few dims (a silent intro is genuinely 8σ from the mean of a loud track) and
 * one such dim would otherwise dominate every projection it appears in. ±4 keeps
 * >99.99% of a normal signal untouched and turns the outliers into a plateau.
 */
export const Z_CLAMP = 4;

/** Gain range for a driver. 0 mutes it everywhere; 2 doubles its say. */
export const MAX_DRIVER_GAIN = 2;

/**
 * Largest tick step that counts as continuous playback. 1 is a normal advance and
 * 0 is an idle transport re-ticking the same position; anything else is a seek.
 * Same threshold, and the same reasoning, as `EventCursor`'s walk gap.
 */
export const MAX_CONTINUOUS_TICK_GAP = 1;

/**
 * The named structure drivers, in bank order. These are 1-dim timeline channels
 * that have shipped since the v2 format and were never read by the runtime until
 * Revision 4. A track missing one keeps the slot (as a constant, contributing
 * exactly nothing) so that driver indices — and therefore saved gains and the
 * seeded wiring — do not shift between tracks.
 */
export const STRUCTURE_DRIVERS: readonly { channel: string; name: string }[] = [
  { channel: 'novelty4', name: 'novelty·4bar' },
  { channel: 'novelty16', name: 'novelty·16bar' },
  { channel: 'actChorus', name: 'chorus-ness' },
];

/** How many latent components join the bank. 3 + 13 = 16 sliders, which is tunable by a human. */
export const PC_DRIVER_COUNT = 13;

/**
 * The modulator's input: D named channels, z-scored per dimension over the whole
 * track at load.
 *
 * Per-*track* standardisation is the point: it makes "how unusual is this moment
 * for this song" the signal, rather than "where does this song sit among all
 * songs", and it is what lets one set of depths work on any track.
 */
export class DriverBank {
  readonly dims: number;
  readonly frames: number;
  readonly hopSeconds: number;
  readonly source: string;
  /** display names, index-aligned with the bank */
  readonly names: readonly string[];
  /** latent dim each driver came from, or -1 for a structure channel */
  readonly sources: Int32Array;
  /** raw (pre-z) variance of each driver over the track — the reordering evidence */
  readonly variance: Float64Array;
  readonly mean: Float64Array;
  readonly std: Float64Array;
  /** frames x dims, already z-scored and clipped */
  private readonly z: Float32Array;

  constructor(
    columns: readonly { name: string; source: number; read: (frame: number) => number }[],
    frames: number,
    hopSeconds: number,
    source: string,
  ) {
    const dims = columns.length;
    this.dims = dims;
    this.frames = frames;
    this.hopSeconds = hopSeconds;
    this.source = source;
    this.names = columns.map((c) => c.name);
    this.sources = Int32Array.from(columns.map((c) => c.source));
    this.variance = new Float64Array(dims);
    this.mean = new Float64Array(dims);
    this.std = new Float64Array(dims);
    this.z = new Float32Array(frames * dims);

    // Non-finite samples are excluded here and written as z = 0 below. This is
    // the *only* place the float path is checked, and it has to be: a single NaN
    // would otherwise poison mean[d]/std[d] for that driver, and NaN fails every
    // `<`/`>` comparison so neither the z clamp nor the target clamp downstream
    // would catch it — it would reach SlewLimiter.step, whose state is a feedback
    // term, and latch there for the rest of the run.
    let bad = 0;
    for (let d = 0; d < dims; d++) {
      const read = (columns[d] as { read: (f: number) => number }).read;
      let sum = 0;
      let n = 0;
      for (let f = 0; f < frames; f++) {
        const x = read(f);
        if (!Number.isFinite(x)) continue;
        sum += x;
        n++;
      }
      const mean = sum / Math.max(n, 1);
      let acc = 0;
      for (let f = 0; f < frames; f++) {
        const x = read(f);
        if (!Number.isFinite(x)) continue;
        acc += (x - mean) * (x - mean);
      }
      // Population variance. A constant driver gets sd 0, floored to 1 below,
      // which makes its z exactly 0 forever — it contributes nothing rather
      // than NaN.
      const varv = acc / Math.max(n, 1);
      const sd = Math.sqrt(varv);
      this.mean[d] = mean;
      this.variance[d] = varv;
      this.std[d] = sd > 1e-12 ? sd : 1;
      const inv = 1 / (this.std[d] as number);
      for (let f = 0; f < frames; f++) {
        const x = read(f);
        if (!Number.isFinite(x)) {
          this.z[f * dims + d] = 0;
          bad++;
          continue;
        }
        const v = (x - mean) * inv;
        this.z[f * dims + d] = v < -Z_CLAMP ? -Z_CLAMP : v > Z_CLAMP ? Z_CLAMP : v;
      }
    }
    if (bad > 0)
      console.warn(
        `modulation: ${bad} non-finite sample(s) in ${source}; zeroed (they contribute nothing)`,
      );
  }

  get label(): string {
    return `drivers-${this.dims}`;
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
 * Rank latent dimensions by their variance over this track, descending.
 * Exported because "the stored order is not the variance order" is the claim
 * Revision 4 rests on, and it is worth a test rather than a comment.
 */
export function varianceOrder(
  data: Float32Array,
  frames: number,
  stride: number,
  offset: number,
  dims: number,
): Int32Array {
  const varv = new Float64Array(dims);
  for (let d = 0; d < dims; d++) {
    let sum = 0;
    let n = 0;
    for (let f = 0; f < frames; f++) {
      const x = data[f * stride + offset + d] as number;
      if (!Number.isFinite(x)) continue;
      sum += x;
      n++;
    }
    const mean = sum / Math.max(n, 1);
    let acc = 0;
    for (let f = 0; f < frames; f++) {
      const x = data[f * stride + offset + d] as number;
      if (!Number.isFinite(x)) continue;
      acc += (x - mean) * (x - mean);
    }
    varv[d] = acc / Math.max(n, 1);
  }
  const order = Int32Array.from({ length: dims }, (_, i) => i);
  // Ties break on the stored index, so the bank is a pure function of the file.
  return order.sort((a, b) => {
    const dv = (varv[b] as number) - (varv[a] as number);
    return dv !== 0 ? dv : a - b;
  });
}

/**
 * Build the bank from a timeline. The 64-dim `latent` channel is the sole PCA
 * source — Revision 4 deleted the 11 MB raw-embedding sidecar path, and since
 * `latent` is always present in a v2 timeline there is no longer a cold/warm
 * distinction: the input is the same from the first frame to the last.
 */
export function buildDriverBank(timeline: Timeline, channelName = 'latent'): DriverBank | null {
  const { data, stride } = timeline;
  const { frames, hopSeconds } = timeline.manifest.grid;
  const latent = timeline.channels.get(channelName);
  if (!latent || latent.dims === 0) {
    console.warn(`modulation: timeline has no usable "${channelName}" channel; nothing to drive from`);
    return null;
  }

  const columns: { name: string; source: number; read: (frame: number) => number }[] = [];
  const missing: string[] = [];
  for (const spec of STRUCTURE_DRIVERS) {
    const c = timeline.channels.get(spec.channel);
    if (!c || c.dims === 0) {
      missing.push(spec.channel);
      // Keep the slot: driver indices (and therefore saved gains and the seeded
      // wiring) must not shift because one optional channel is absent.
      columns.push({ name: `${spec.name} (absent)`, source: -1, read: () => 0 });
      continue;
    }
    const o = c.offset;
    columns.push({ name: spec.name, source: -1, read: (f) => data[f * stride + o] as number });
  }
  if (missing.length > 0)
    console.warn(`modulation: structure channel(s) ${missing.join(', ')} absent; those drivers read 0`);

  const order = varianceOrder(data, frames, stride, latent.offset, latent.dims);
  const take = Math.min(PC_DRIVER_COUNT, latent.dims);
  for (let i = 0; i < take; i++) {
    const dim = order[i] as number;
    const o = latent.offset + dim;
    columns.push({ name: `pc-${i + 1}`, source: dim, read: (f) => data[f * stride + o] as number });
  }

  const bank = new DriverBank(columns, frames, hopSeconds, `timeline "${channelName}" + structure`);
  console.info(
    `modulation: ${bank.dims} drivers — ${bank.names.join(', ')}` +
      ` (pc order from latent dims ${Array.from(bank.sources.slice(STRUCTURE_DRIVERS.length)).join(',')})`,
  );
  return bank;
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

  /** The named input bank. Null only when the timeline has no latent channel at all. */
  readonly drivers: DriverBank | null;

  /**
   * The brightness lane (Revision 4). Runs in *both* modes — it is not modulation,
   * it is a direct wire from a stem to its species' light — so muting every driver
   * gain still leaves an image that breathes with the arrangement.
   */
  readonly stemFollow: StemFollow;

  private readonly sim: PhysarumSim;
  private readonly sampler: TimelineSampler;
  private readonly events: ModulatorEvents;

  private cfg!: ModulationConfig;
  private seed: number;

  private readonly slots: readonly (ModSpec | null)[];
  private readonly mask: Uint8Array;
  private readonly length: number;
  /** length x dims, row-major; row i is parameter i's projection direction */
  private readonly w: Float32Array;
  private readonly base: Float64Array;
  private readonly target: Float64Array;
  private readonly defaults: Float64Array;
  /** live z per driver — also what the workbench meters read */
  private readonly zbuf: Float32Array;
  /** gains ⊙ z, rebuilt once per tick and shared by every projection */
  private readonly gz: Float32Array;
  /** scratch ẑ for `targetFor`, so the scripting hook never writes live state */
  private readonly probe: Float32Array;
  /** per-driver gain, mirrored into cfg.driverGains so it persists */
  private readonly gains: Float32Array;
  /** stems row for the follow lane; length K, allocation-free */
  private readonly stemRow: Float32Array;
  private readonly stemChannel: { offset: number; dims: number } | null;
  private slew!: SlewLimiter;

  private readonly excursionSum: Float64Array = new Float64Array(MOD_GROUPS.length);
  private readonly excursionCount: Int32Array = new Int32Array(MOD_GROUPS.length);
  private lastSegment = -2;
  /** NaN until the first tick, so the first update snaps rather than fades in. */
  private lastTick = Number.NaN;

  constructor(
    sim: PhysarumSim,
    sampler: TimelineSampler,
    drivers: DriverBank | null,
    config: ModulationConfig,
    seed: number,
    events: ModulatorEvents = {},
  ) {
    this.sim = sim;
    this.sampler = sampler;
    this.drivers = drivers;
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
    const dims = drivers?.dims ?? 0;
    this.zbuf = new Float32Array(dims);
    this.gz = new Float32Array(dims);
    this.probe = new Float32Array(dims);
    this.gains = new Float32Array(dims).fill(1);
    this.w = new Float32Array(this.length * dims);

    const stems = sampler.getChannel('stems');
    this.stemChannel = stems && stems.dims > 0 ? { offset: stems.offset, dims: stems.dims } : null;
    this.stemRow = new Float32Array(k);
    this.stemFollow = new StemFollow(k, Math.min(this.stemChannel?.dims ?? 0, STEM_NAMES.length));
    sim.setBrightFollow(this.stemFollow.multiplier);

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
    return this.drivers !== null;
  }

  get unavailableReason(): string {
    return this.drivers ? '' : 'the timeline has no "latent" channel — nothing to build drivers from';
  }

  get sourceLabel(): string {
    return this.drivers ? this.drivers.label : 'none';
  }

  /** How many slots the music actually moves, of how many in θ. */
  get modulatedCount(): number {
    let n = 0;
    for (const m of this.mask) n += m;
    return n;
  }

  // ── the driver bank, as the workbench sees it ──────────────────────────────

  get driverCount(): number {
    return this.drivers?.dims ?? 0;
  }

  driverName(index: number): string {
    return this.drivers?.names[index] ?? `driver ${index}`;
  }

  /** Live z of driver `index` — the meter value. Updated in both modes. */
  driverValue(index: number): number {
    return this.zbuf[index] ?? 0;
  }

  driverGain(index: number): number {
    return this.gains[index] ?? 1;
  }

  /** Gains live in the config so they persist; the Float32Array is the hot copy. */
  setDriverGain(index: number, value: number): void {
    if (index < 0 || index >= this.gains.length) return;
    const g = Math.min(Math.max(Number.isFinite(value) ? value : 1, 0), MAX_DRIVER_GAIN);
    this.gains[index] = g;
    this.cfg.driverGains[index] = g;
  }

  setAllDriverGains(value: number): void {
    for (let d = 0; d < this.gains.length; d++) this.setDriverGain(d, value);
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

    // Gains are authored against a driver count the file cannot know (it depends
    // on the track's latent width), so adopt what fits and default the rest to 1,
    // then write the normalised array back — the config is what gets saved.
    //
    // Only the prefix is overwritten. Anything past this track's driver count is
    // carried through untouched, because the config object is what the workbench
    // autosaves and serialises: truncating here would mean loading a file on a
    // track with a narrower latent (or none at all, where the bank is null and
    // this length is 0) silently destroys the rest of the user's saved gains.
    const gains = config.driverGains.slice();
    for (let d = 0; d < this.gains.length; d++) {
      const v = gains[d];
      const g = typeof v === 'number' && Number.isFinite(v) ? Math.min(Math.max(v, 0), MAX_DRIVER_GAIN) : 1;
      this.gains[d] = g;
      gains[d] = g;
    }
    config.driverGains = gains;

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

  /** Rebuild projections and personality for the current seed. */
  private rewire(): void {
    const dims = this.drivers?.dims ?? 0;
    if (dims > 0) {
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
    const dims = this.drivers?.dims ?? 0;
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
    // Transport discontinuity detection, same policy as `EventCursor`: playback
    // advances one tick, an idle transport repeats the same tick (gap 0), and
    // anything else — restart, arrow-key seek, A/B restore, overlay scrub — is a
    // jump. The stem-follow EMA has to snap across one, because its state belongs
    // to the position we just left; fading over ~300 ms into the new position is
    // precisely what makes an A/B restore fail to reproduce the snapshot moment's
    // brightness. Nothing calls a reset for this: every seek path in the app ends
    // up here, so detecting it is the only way to cover all of them.
    const gap = frame.tick - this.lastTick;
    const jumped = !(gap >= 0 && gap <= MAX_CONTINUOUS_TICK_GAP);
    this.lastTick = frame.tick;

    // The brightness lane, ahead of everything and outside the mode check: it is
    // a direct wire, not modulation, so it works in manual mode and it survives
    // every driver gain being muted.
    this.updateStemFollow(frame, dt, jumped);

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

    // Sampled in both modes so the workbench meters are live even with modulation
    // off — the bank is an instrument panel first and an input second.
    if (this.drivers && !this.frozen) this.drivers.sample(frame.time, this.zbuf);

    if (this.mode !== 'modulated' || !this.drivers) return;

    this.computeTarget(this.zbuf);

    if (boundaryHit) this.slew.snapClass(CLASS_SLOW, this.target, this.cfg.boundary.snapFraction);

    // responseSpeed scales time, not the coefficients: doubling it makes every
    // class exactly twice as quick and keeps their ratios (the tuned part) intact.
    const v = this.slew.step(this.target, dt * Math.max(this.cfg.responseSpeed, 0));
    applyVector(this.sim.config, v, this.mask);
    this.sim.uploadMatrix();
  }

  private updateStemFollow(frame: FeaturesFrame, dt: number, snap: boolean): void {
    const c = this.stemChannel;
    if (!c) {
      this.stemFollow.update(null, dt, this.cfg.stemFollow, snap);
      return;
    }
    for (let k = 0; k < this.stemRow.length; k++) {
      this.stemRow[k] = k < c.dims ? (frame.values[c.offset + k] ?? 0) : 0;
    }
    this.stemFollow.update(this.stemRow, dt, this.cfg.stemFollow, snap);
  }

  /** Advance `lastSegment`; true only for a genuine crossing (never the first sight). */
  private trackSegment(time: number): boolean {
    const seg = this.sampler.segmentIndexAt(time);
    if (seg < 0 || seg === this.lastSegment) return false;
    const first = this.lastSegment === -2;
    this.lastSegment = seg;
    return !first;
  }

  private computeTarget(z: ArrayLike<number>): void {
    const dims = this.drivers?.dims ?? 0;
    const gz = this.gz;
    // One multiply per driver per tick, then every projection reads the same
    // vector: a gain of 0 is therefore *exactly* zero in every wiring, not
    // approximately zero in most of them.
    for (let d = 0; d < dims; d++) gz[d] = (this.gains[d] as number) * ((z[d] as number) ?? 0);

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
      for (let d = 0; d < dims; d++) raw += (w[o + d] as number) * (gz[d] as number);
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
   * the hook the bounds / determinism / gain-gating tests drive.
   *
   * It writes into its own scratch buffer rather than `zbuf`, which since
   * Revision 4 is live state: the workbench meters read it every refresh, and in
   * `frozen` mode `update` deliberately stops resampling it, so stamping a
   * synthetic vector there would show up on the meters and — frozen — hold as the
   * ẑ the parameters slew toward. A scripting hook must not be able to do that.
   */
  targetFor(z: ArrayLike<number>, out = new Float64Array(this.length)): Float64Array {
    const dims = this.drivers?.dims ?? 0;
    const scratch = this.probe;
    for (let d = 0; d < dims; d++) scratch[d] = (z[d] as number) ?? 0;
    this.computeTarget(scratch);
    out.set(this.target);
    return out;
  }
}
