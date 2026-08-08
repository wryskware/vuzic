/**
 * Stem-follow — per-species brightness, plan.md Revision 4.
 *
 * Revision 3 let the embedding modulate brightness. Verdict after use: it
 * flashed constantly, for reasons nobody watching could name, and it looked
 * bad. So brightness leaves the modulation registry entirely and gets one job
 * instead: **a species is as bright as its own instrument is loud.**
 *
 *   multiplier_k = floor + (1 - floor) · smoothed(stem_k)^curve
 *   effective    = clamp(base_brightness_k · multiplier_k, 0, MAX) · impulseFlash
 *
 * Three properties this buys that a projection could not:
 *
 * - **Legible.** Vocals cut out at 2:22 on Free Fall, the cyan species fades.
 *   No interpretation required, and no seed can rewire it into something else.
 * - **A floor, not a gate.** `floor` (default 0.25) is what a silent instrument
 *   is left with — a ghost of the network it built, not a black hole. The user
 *   asked for "visibly diminish", not "disappear".
 * - **Smoothed, not switched.** The stem curves are already EMA'd in analysis at
 *   beat scale, but a hard section cut is still a step; τ ≈ 300 ms turns it into
 *   a fade. Smoothing is applied whether or not the lane is enabled, so toggling
 *   it on mid-track does not step.
 *
 * The lane is deliberately *outside* θ: it is not slewed (its own smoothing is
 * the slew), not seeded, and not part of any saved parameter vector. It composes
 * multiplicatively under the impulse flashes, which still multiply on top —
 * a kick still flashes a species that is currently dimmed, just from lower down.
 */

export interface StemFollowConfig {
  /** false = brightness sliders are absolute again (the phase-4 behaviour) */
  enabled: boolean;
  /** what a fully silent instrument keeps, 0..1. 0 is a blackout, 1 is no effect. */
  floor: number;
  /** exponent on the smoothed activity; >1 makes only loud passages bright */
  curve: number;
  /** EMA time constant in ms — how fast a cut fades rather than snaps */
  smoothingMs: number;
}

export const DEFAULT_STEM_FOLLOW: StemFollowConfig = {
  enabled: true,
  floor: 0.25,
  curve: 1,
  smoothingMs: 300,
};

export function defaultStemFollow(): StemFollowConfig {
  return { ...DEFAULT_STEM_FOLLOW };
}

/**
 * The pure arithmetic, exported so the tests can pin it without a sim.
 * `activity` is the already-smoothed stem level; the clamp on it matters because
 * the analysis curves are normalised per track and a hand-edited timeline is not
 * guaranteed to be.
 */
export function followMultiplier(activity: number, cfg: StemFollowConfig): number {
  if (!cfg.enabled) return 1;
  const a = Math.min(Math.max(Number.isFinite(activity) ? activity : 0, 0), 1);
  const floor = Math.min(Math.max(cfg.floor, 0), 1);
  const curve = Math.max(cfg.curve, 0.01);
  return floor + (1 - floor) * Math.pow(a, curve);
}

/**
 * Per-species brightness multipliers, mutated in place every tick. The sim holds
 * the `multiplier` array by reference (exactly like `ImpulseState`), so nothing
 * is copied per frame and nothing has to be re-fetched.
 *
 * **The keying is data, not code.** Which species follows which stem arrives as
 * a map from the sim (`ModTarget.stemMap`) rather than being the identity baked
 * in here. That is the seam live-mode-notes asks for — "species keyed to sound
 * sources should be config, not code" — and it is what lets a second substrate
 * key three particle types to four stems, or a future UI re-key by hand, without
 * this class knowing anything about it. Omitting the map reproduces the old
 * behaviour exactly: species k follows stem k while k < stemDims.
 */
export class StemFollow {
  /** length K; 1.0 = untouched. Read by PhysarumSim.uploadSpecies. */
  readonly multiplier: Float32Array;
  /** length K; the smoothed stem level behind each multiplier, for the readout */
  readonly activity: Float32Array;
  /** length K; species k's stem index, or -1 where it follows nothing */
  private readonly stem: Int32Array;

  constructor(speciesCount: number, stemDims: number, map?: ArrayLike<number>) {
    const k = Math.max(1, speciesCount);
    this.multiplier = new Float32Array(k).fill(1);
    this.activity = new Float32Array(k);
    this.stem = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      // Sanitised, not trusted: an out-of-range entry would read past the stems
      // row (or before it) every tick, and `undefined` from a short map would
      // propagate as NaN through the EMA and latch there.
      if (map === undefined) {
        this.stem[i] = i < stemDims ? i : -1;
        continue;
      }
      const s = map[i];
      this.stem[i] = typeof s === 'number' && s >= 0 && s < stemDims ? Math.floor(s) : -1;
    }
  }

  get speciesCount(): number {
    return this.multiplier.length;
  }

  /** Species k's stem index, or -1 when it has none. */
  stemOf(k: number): number {
    return this.stem[k] ?? -1;
  }

  /**
   * One sim tick. `stems` is the timeline's stems row (or null when the track has
   * no usable stems channel, in which case every multiplier stays at 1 and the
   * lane is a no-op rather than a blackout).
   *
   * `snap` adopts the sampled level outright instead of easing into it. It is for
   * transport discontinuities — restart, arrow-key seek, A/B restore — where the
   * EMA's history belongs to a different moment of the song. Easing across a jump
   * would carry the old position's brightness ~300 ms into the new one, which is
   * exactly the thing an A/B restore exists to reproduce faithfully.
   */
  update(stems: ArrayLike<number> | null, dt: number, cfg: StemFollowConfig, snap = false): void {
    const tau = Math.max(cfg.smoothingMs, 1) / 1000;
    const alpha = snap ? 1 : dt > 0 ? 1 - Math.exp(-dt / tau) : 0;
    for (let k = 0; k < this.multiplier.length; k++) {
      const s = this.stem[k] as number;
      if (s < 0 || !stems) {
        this.multiplier[k] = 1;
        continue;
      }
      const raw = stems[s] as number;
      const x = Number.isFinite(raw) ? Math.min(Math.max(raw, 0), 1) : 0;
      const prev = this.activity[k] as number;
      const next = prev + (x - prev) * alpha;
      this.activity[k] = next;
      this.multiplier[k] = followMultiplier(next, cfg);
    }
  }
}
