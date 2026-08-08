/**
 * Per-parameter-class slew limiting, applied between the modulator's output and
 * the sim. Plan.md's rule: "parameter motion faster than the sim's relaxation time
 * produces mush", so each slot approaches its target with a time constant chosen
 * by its class rather than snapping to the projected value.
 *
 * Exponential approach (a first-order low pass), not a constant-rate ramp:
 * α = 1 − exp(−dt/τ) is frame-rate independent, never overshoots, and is
 * exactly the same operator the trail field already applies to itself.
 *
 * Section boundaries are the exception — see `snapClass`.
 */
import { CLASS_FAST, CLASS_MEDIUM, CLASS_SLOW } from './preset.ts';

export interface SlewRates {
  /** brightness, deposit, sensor angle — tracks the 10 Hz signal */
  fast: number;
  /** geometry the trail field has to catch up with */
  medium: number;
  /** decay, alive-fractions, the sense matrix — section scale */
  slow: number;
}

/**
 * Revision 3 defaults, materially faster than the scene era's {0.12, 1.2, 8}.
 * The old numbers were sized to make a jump *between anchors* look like a
 * dissolve; there are no anchors now and the target is already continuous, so
 * heavy smoothing only buys latency. τ=0.05 s tracks the 10 Hz input almost
 * exactly (the input's own hop is 0.1 s), 0.5 s is the "shape is changing"
 * timescale, and 8 s keeps decay/alive-fraction/M at section scale where the
 * trail field can follow them without turning to mush.
 */
export const DEFAULT_SLEW: SlewRates = { fast: 0.05, medium: 0.5, slow: 8 };

export class SlewLimiter {
  readonly value: Float64Array;
  private readonly classes: Uint8Array;
  private readonly tau: Float64Array;
  /** step() runs every tick; the per-class α is scratch, not state. */
  private readonly alpha = new Float64Array(3);

  constructor(classes: Uint8Array, rates: SlewRates = DEFAULT_SLEW) {
    this.classes = classes;
    this.value = new Float64Array(classes.length);
    this.tau = new Float64Array(3);
    this.setRates(rates);
  }

  setRates(rates: SlewRates): void {
    this.tau[CLASS_FAST] = Math.max(rates.fast, 0);
    this.tau[CLASS_MEDIUM] = Math.max(rates.medium, 0);
    this.tau[CLASS_SLOW] = Math.max(rates.slow, 0);
  }

  /** Jump the whole state to `v` — used on load, on a reroll, and after a mode switch. */
  reset(v: ArrayLike<number>): void {
    for (let i = 0; i < this.value.length; i++) this.value[i] = (v[i] as number) ?? 0;
  }

  /** One step of exponential approach towards `target`. Returns the internal state. */
  step(target: ArrayLike<number>, dt: number): Float64Array {
    if (!(dt > 0)) return this.value;
    const alpha = this.alpha;
    alpha[CLASS_FAST] = alphaFor(this.tau[CLASS_FAST] as number, dt);
    alpha[CLASS_MEDIUM] = alphaFor(this.tau[CLASS_MEDIUM] as number, dt);
    alpha[CLASS_SLOW] = alphaFor(this.tau[CLASS_SLOW] as number, dt);
    const v = this.value;
    for (let i = 0; i < v.length; i++) {
      const a = alpha[this.classes[i] as number] as number;
      const t = (target[i] as number) ?? 0;
      v[i] = (v[i] as number) + (t - (v[i] as number)) * a;
    }
    return v;
  }

  /**
   * Section boundaries are steps, not ramps: move one class a fixed fraction of
   * the way to its target in a single call, bypassing its time constant.
   * fraction 1 = a hard cut, 0 = nothing.
   */
  snapClass(cls: number, target: ArrayLike<number>, fraction: number): void {
    const f = Math.min(Math.max(fraction, 0), 1);
    if (f === 0) return;
    const v = this.value;
    for (let i = 0; i < v.length; i++) {
      if (this.classes[i] !== cls) continue;
      const t = (target[i] as number) ?? 0;
      v[i] = (v[i] as number) + (t - (v[i] as number)) * f;
    }
  }
}

function alphaFor(tau: number, dt: number): number {
  if (tau <= 0) return 1;
  return 1 - Math.exp(-dt / tau);
}
