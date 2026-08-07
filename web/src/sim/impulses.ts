/**
 * The impulse engine — beat-to-beat reactivity (plan.md Revision 2, item 2).
 *
 *   timeline.events ──► EventCursor (pointer walk, tick-indexed)
 *                              │
 *                       per-kind envelope: instant attack, exponential decay
 *                              │
 *              ┌───────────────┼──────────────┬────────────────┐
 *          deposit          brightness      sensor          radial
 *           burst            flash           pop            splash
 *              └───────────────┴──────────────┴────────────────┘
 *                              │
 *                    ImpulseState, read by the sim every tick
 *
 * Why this is a separate lane and not another θ field:
 *
 * - The mapper slew-limits everything it writes, by design — parameter motion
 *   faster than the sim's relaxation time produces mush. A transient is exactly
 *   the thing that must *not* be slewed, so impulses are applied downstream of
 *   the slew limiter, multiplicatively on top of whatever set the base value.
 *   That makes them work identically in manual and mapped mode, and it means
 *   neither lane can fight the other: the mapper still owns the base, impulses
 *   own the deviation, and the deviation always decays back to 1.0.
 *
 * - Envelopes advance per *sim tick*, not per rAF, so the shape of a hit does
 *   not depend on frame rate, and a paused transport freezes the song's events
 *   while the workbench's test-fire button still decays normally.
 *
 * Determinism: hotspot positions are hashed on (seed, eventIndex, slot) with the
 * same PCG the shaders use. No wall clock, no Math.random, so the same seed and
 * the same track put the same splashes in the same places on every run.
 */
import { EVENT_KINDS, type EventKind, type TimelineEvent } from '../timeline/types.ts';
import { EventCursor } from '../timeline/sampler.ts';

/** GPU-side cap. Overflow drops the weakest active splash, never the newest. */
export const MAX_SPLASHES = 32;

/** Below this the envelope contributes nothing visible; the splash is retired. */
const SPLASH_CUTOFF = 0.01;

/**
 * What one event kind does. Every field is data — the whole point is that the
 * event→response mapping is tuned in the workbench, not compiled in.
 */
export interface ResponseConfig {
  enabled: boolean;
  /** species this kind drives; negative means all of them */
  species: number;
  /** envelope time constant in ms; ~80 (hat) to ~400 (vocal) */
  decayMs: number;
  /** deposit *= 1 + deposit·env */
  deposit: number;
  /** brightness *= 1 + flash·env */
  flash: number;
  /** sensor distance *= 1 + sensor·env */
  sensor: number;
  /** hotspot discs spawned per event; 0 disables the splash entirely */
  splashCount: number;
  /** disc radius as a fraction of min(gridW, gridH) */
  splashRadius: number;
  /** outward push in grid cells per tick at full envelope */
  splashPush: number;
  /** tangential twist in radians at full envelope; makes the ring curl */
  splashSwirl: number;
}

export interface ImpulseConfig {
  enabled: boolean;
  /** global depth multiplier on every response */
  gain: number;
  responses: Record<EventKind, ResponseConfig>;
}

/** One live hotspot disc, in normalised grid coordinates. */
export interface ImpulseSplash {
  /** 0..1 across the grid */
  x: number;
  y: number;
  /** fraction of min(gridW, gridH) */
  radius: number;
  /** current envelope, 0..1-ish */
  strength: number;
  push: number;
  swirl: number;
  /** target species, or negative for all */
  species: number;
}

/**
 * The sim's read-only view. Mutated in place every tick, so a consumer holds the
 * object once and never re-reads a reference.
 */
export interface ImpulseState {
  /** length K, 1.0 = untouched */
  depositMul: Float32Array;
  brightMul: Float32Array;
  sensorMul: Float32Array;
  splashes: readonly ImpulseSplash[];
}

interface LiveSplash extends ImpulseSplash {
  kind: number;
  /** peak envelope, so the readout and the retire test are stable */
  born: number;
}

function response(over: Partial<ResponseConfig> = {}): ResponseConfig {
  return {
    enabled: true,
    species: 0,
    decayMs: 200,
    deposit: 0,
    flash: 0,
    sensor: 0,
    splashCount: 0,
    splashRadius: 0.18,
    splashPush: 4,
    splashSwirl: 0.3,
    ...over,
  };
}

/**
 * Defaults biased toward LEGIBLE, not subtle — the first build read as too subtle
 * and plan.md Revision 2 item 3 makes reactivity depth an explicit tuning target.
 * Species indices follow the stem keying: 0 bass, 1 drums, 2 vocals, 3 other.
 */
export function defaultImpulseConfig(): ImpulseConfig {
  return {
    enabled: true,
    gain: 1,
    responses: {
      // kick → the bass species blooms and lights up. No splash: the low end
      // should feel like the whole field swelling, not like something hitting it.
      kick: response({ species: 0, decayMs: 240, deposit: 3, flash: 1.6 }),
      // snare → the visible one. Two discs of drum agents thrown outward.
      snare: response({
        species: 1,
        decayMs: 200,
        deposit: 1,
        flash: 1,
        splashCount: 2,
        splashRadius: 0.2,
        // 9 cells/tick at full envelope ≈ 30 cells of outward travel over the
        // 200 ms decay — several times an agent's own step, so the disc visibly
        // empties and refills instead of being lost in the churn.
        splashPush: 9,
        splashSwirl: 0.35,
      }),
      // hat → fast small flashes on the same fine, fast-decaying species.
      hat: response({ species: 1, decayMs: 90, deposit: 0.5, flash: 1 }),
      // bass note → sensor pop: the species reaches further for a moment, so its
      // network visibly re-organises rather than just getting brighter.
      bass: response({ species: 0, decayMs: 320, deposit: 0.5, flash: 0.3, sensor: 1.2 }),
      // vocal → a slow wide pulse; one soft disc, low push, long decay.
      vocal: response({
        species: 2,
        decayMs: 380,
        deposit: 1.4,
        flash: 1.8,
        sensor: 0.3,
        splashCount: 1,
        splashRadius: 0.34,
        splashPush: 2,
        splashSwirl: 0.7,
      }),
    },
  };
}

// ── deterministic hashing, mirroring common.wgsl ──────────────────────────────

export function pcg(input: number): number {
  let state = (Math.imul(input >>> 0, 747796405) + 2891336453) >>> 0;
  state = state >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

export function hash3(a: number, b: number, c: number): number {
  return pcg(((a >>> 0) ^ pcg(((b >>> 0) ^ pcg(c >>> 0)) >>> 0)) >>> 0);
}

function rand01(h: number): number {
  return (h >>> 0) * 2.3283064365386963e-10;
}

/**
 * Where splash `slot` of event `eventIndex` lands, in 0..1 grid coordinates.
 * Pure and exported because "the same seed puts the splash in the same place" is
 * the determinism claim worth testing.
 */
export function hotspotAt(
  seed: number,
  eventIndex: number,
  slot: number,
): { x: number; y: number } {
  const h = hash3(seed ^ 0x9e3779b9, (eventIndex >>> 0) * 0x2545f491 + slot, slot + 1);
  return { x: rand01(h), y: rand01(pcg(h)) };
}

/** Index of `kind` in EVENT_KINDS; -1 for anything unknown. */
export function kindIndex(kind: EventKind): number {
  return EVENT_KINDS.indexOf(kind);
}

export class ImpulseEngine {
  readonly config: ImpulseConfig;
  private seed: number;
  private readonly speciesCount: number;
  private cursor: EventCursor;
  private events: readonly TimelineEvent[];

  /** current envelope per kind, index-aligned with EVENT_KINDS */
  private readonly levels = new Float32Array(EVENT_KINDS.length);
  private live: LiveSplash[] = [];
  /** walks so repeated test-fires of one kind are not identical splashes */
  private testCounter = 0;

  readonly state: ImpulseState;

  constructor(
    seed: number,
    speciesCount: number,
    events: readonly TimelineEvent[],
    secondsPerTick: number,
    config: ImpulseConfig = defaultImpulseConfig(),
  ) {
    this.seed = seed >>> 0;
    this.speciesCount = Math.max(1, speciesCount);
    this.events = events;
    this.cursor = new EventCursor(events, secondsPerTick);
    this.config = config;
    this.state = {
      depositMul: new Float32Array(this.speciesCount).fill(1),
      brightMul: new Float32Array(this.speciesCount).fill(1),
      sensorMul: new Float32Array(this.speciesCount).fill(1),
      splashes: this.live,
    };
  }

  get eventCount(): number {
    return this.events.length;
  }

  /** Envelope of one kind right now, for the workbench readout. */
  levelOf(kind: EventKind): number {
    return this.levels[kindIndex(kind)] ?? 0;
  }

  get activeSplashes(): number {
    return this.live.length;
  }

  /** A reseed re-keys every hotspot, so the world's splashes move with it. */
  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  /** Drop all envelopes and re-locate the cursor on the next advance. */
  reset(): void {
    this.levels.fill(0);
    this.live.length = 0;
    this.cursor.reset();
    this.writeState();
  }

  /**
   * One sim tick. `tick` selects which events fire (never wall clock); `dt` is the
   * fixed timestep and drives every envelope, so a paused transport still decays.
   */
  update(tick: number, dt: number): void {
    this.decay(dt);

    if (this.config.enabled) {
      const { start, end } = this.cursor.advance(tick);
      for (let i = start; i < end; i++) {
        const ev = this.events[i];
        if (ev) this.fire(ev.kind, ev.strength, i);
      }
    } else {
      // Keep the cursor at the playhead so re-enabling does not dump a backlog.
      this.cursor.advance(tick);
    }

    this.writeState();
  }

  /**
   * Fire a synthetic event immediately — the workbench's per-kind test button.
   * Keyed on a private counter so it is reproducible within a run and can never
   * collide with a real event index.
   */
  testFire(kind: EventKind, strength = 1): void {
    this.fire(kind, strength, 0x4000_0000 + this.testCounter++);
    this.writeState();
  }

  private decay(dt: number): void {
    const resp = this.config.responses;
    for (let k = 0; k < EVENT_KINDS.length; k++) {
      const level = this.levels[k] as number;
      if (level <= 0) continue;
      const kind = EVENT_KINDS[k] as EventKind;
      const next = level * Math.exp(-dt / tau(resp[kind].decayMs));
      this.levels[k] = next < SPLASH_CUTOFF ? 0 : next;
    }

    let w = 0;
    for (const s of this.live) {
      const kind = EVENT_KINDS[s.kind] as EventKind;
      s.strength *= Math.exp(-dt / tau(resp[kind].decayMs));
      if (s.strength >= SPLASH_CUTOFF) this.live[w++] = s;
    }
    this.live.length = w;
  }

  private fire(kind: EventKind, strength: number, eventIndex: number): void {
    const r = this.config.responses[kind];
    if (!r.enabled) return;
    const amp = Math.min(Math.max(strength, 0), 1) * Math.max(this.config.gain, 0);
    if (amp <= 0) return;

    const k = kindIndex(kind);
    // Instant attack. A retrigger takes the max rather than summing: a 16th-note
    // hat roll should read as a sustained shimmer, not as a runaway ramp.
    if (amp > (this.levels[k] as number)) this.levels[k] = amp;

    const count = Math.min(Math.round(r.splashCount), MAX_SPLASHES);
    for (let s = 0; s < count; s++) {
      const { x, y } = hotspotAt(this.seed, eventIndex, s);
      this.push({
        kind: k,
        x,
        y,
        radius: Math.max(r.splashRadius, 1e-3),
        strength: amp,
        born: amp,
        push: r.splashPush,
        swirl: r.splashSwirl,
        species: this.resolveSpecies(r.species),
      });
    }
  }

  /** Newest always survives; the weakest active splash is what gets evicted. */
  private push(s: LiveSplash): void {
    if (this.live.length >= MAX_SPLASHES) {
      let weakest = 0;
      for (let i = 1; i < this.live.length; i++) {
        if ((this.live[i] as LiveSplash).strength < (this.live[weakest] as LiveSplash).strength) {
          weakest = i;
        }
      }
      this.live[weakest] = s;
      return;
    }
    this.live.push(s);
  }

  /** A response authored for 4 species must still do something sane at K=2. */
  private resolveSpecies(index: number): number {
    if (index < 0) return -1;
    return Math.min(Math.floor(index), this.speciesCount - 1);
  }

  private writeState(): void {
    const { depositMul, brightMul, sensorMul } = this.state;
    depositMul.fill(1);
    brightMul.fill(1);
    sensorMul.fill(1);
    if (!this.config.enabled) return;

    for (let k = 0; k < EVENT_KINDS.length; k++) {
      const env = this.levels[k] as number;
      if (env <= 0) continue;
      const kind = EVENT_KINDS[k] as EventKind;
      const r = this.config.responses[kind];
      if (!r.enabled) continue;
      const target = this.resolveSpecies(r.species);
      const lo = target < 0 ? 0 : target;
      const hi = target < 0 ? this.speciesCount - 1 : target;
      for (let sp = lo; sp <= hi; sp++) {
        depositMul[sp] = (depositMul[sp] as number) + r.deposit * env;
        brightMul[sp] = (brightMul[sp] as number) + r.flash * env;
        sensorMul[sp] = (sensorMul[sp] as number) + r.sensor * env;
      }
    }
  }
}

/**
 * ms of "decay time" → the exponential time constant, in seconds. Interpreting the
 * knob as τ means the envelope is at 37% after `decayMs` and effectively gone at
 * ~3× that, which is the intuition the numbers in ResponseConfig are written to.
 */
function tau(decayMs: number): number {
  return Math.max(decayMs, 1) / 1000;
}
