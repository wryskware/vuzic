/**
 * Time constant of the render-frame-length smoother, in seconds. A quarter
 * second is ~15 frames at 60 Hz: long enough to erase per-frame jitter, short
 * enough that plugging in a 144 Hz monitor or dropping to a governed 45 fps is
 * fully absorbed before the next bar of music.
 */
export const DT_FRAMES_TAU_S = 0.25;

/**
 * The render frame's length in 60 Hz frames, smoothed.
 *
 * Callers raise the feedback lane's per-60-Hz-frame constants to this power, and
 * the *unsmoothed* measurement is what made a paused sim flicker. With the world
 * frozen the echo `HDR_n = a·HDR_{n-1} + particles` settles at `particles/(1-a)`
 * — at the shipped amount of 0.88 that is an 8.3x loop gain whose sensitivity to
 * `a` is `1/(1-a)^2 ≈ 69`. Ordinary rAF delta jitter of a few percent therefore
 * arrived on screen as a whole-image brightness wobble that read exactly like an
 * unstable auto-exposure. Running, the same wobble lands on transient pixels and
 * is invisible; frozen, it is the only thing moving.
 *
 * Smoothing rather than pinning to 1: the point of the exponent is that a
 * display's real frame length decides the decay, and pinning would put a
 * brightness step at the moment of pausing on any panel that is not 60 Hz.
 *
 * `snap` adopts the measurement outright, for the first frame a sim renders —
 * an export at a fixed 30 fps cadence must be exact from frame one rather than
 * easing out of a 60 Hz seed.
 *
 * The 0.25 s clamp on the input is the backgrounded-tab guard the substrates
 * already applied: a two-second gap would otherwise raise 0.88 to the 120th
 * power and clear the echo outright, which is a black flash on the first frame
 * back. The 1e-3 floor is for coarsened timestamps — two renders can share one,
 * and `pow(x, 0)` is 1 for every x, which would switch a faded echo fully on.
 *
 * This is the RENDER domain's clock and is deliberately separate from the sim
 * domain's `planSteps` below: the echo is a property of how often the screen is
 * repainted, the world is a property of how much time passed.
 */
export function smoothDtFrames(prev: number, deltaSeconds: number, snap = false): number {
  const dt = Math.min(Math.max(deltaSeconds, 1e-3), 0.25);
  const measured = dt * 60;
  if (snap || !(prev > 0)) return measured;
  const alpha = 1 - Math.exp(-dt / DT_FRAMES_TAU_S);
  return prev + (measured - prev) * alpha;
}

/**
 * A frame's world time expressed in units of a substrate's *reference* step —
 * the dt its integrator and its authored per-step constants were written
 * against (1/60 for physarum and vizfx, 1/120 for particle life).
 *
 * There is exactly **one** integration step per rendered frame, so this is the
 * whole of the sim domain's cadence logic: no step count to choose, no
 * accumulator, no catch-up. A frame that took twice as long takes one step twice
 * as long, and a frame that arrives late buys no extra work — which is the whole
 * difference from the fixed-grid cadence this replaced, where being behind meant
 * running more steps and therefore being further behind.
 *
 * Substrates whose model is written in per-step units multiply their rate-like
 * quantities by this; substrates that integrate a true dt just use `dtWorld`.
 */
export function dtScale(dtWorld: number, modelDt: number): number {
  if (!(dtWorld > 0) || !(modelDt > 0)) return 0;
  return dtWorld / modelDt;
}

/**
 * Deltas outside this band are not vsync slots and never enter the estimate:
 * under 2 ms is a coarsened or duplicated timestamp (no shipping panel runs
 * past ~500 Hz), over 100 ms is a hitch, a modal, or a backgrounded tab.
 */
const SLOT_MIN_MS = 2;
const SLOT_MAX_MS = 100;
/** The rolling window: ten one-second buckets, each keeping its smallest delta. */
const SLOT_BUCKET_MS = 1000;
const SLOT_BUCKETS = 10;

/**
 * What refresh rate is this window's display actually running? — estimated from
 * nothing but the timestamps of rendered frames.
 *
 * The principle: `requestAnimationFrame` is driven by the compositor's vsync,
 * so two callbacks can never be closer together than one display period —
 * every inter-frame delta is (jitter aside) a whole multiple of the slot. The
 * *smallest* delta seen recently is therefore the display period itself, and
 * that stays true while the app is slow, as long as at least one frame in the
 * window fit a single slot. On a machine that NEVER fits one slot the estimate
 * reads as a multiple of the true period, which errs in the safe direction:
 * the budget governor (its only consumer) targets a rate the machine has
 * actually demonstrated, rather than demanding one it has not.
 *
 * A rolling window rather than a session min, because the display is not a
 * constant: dragging the window from a 240 Hz panel to a 60 Hz one, or a
 * laptop dropping its refresh on battery, must be adopted — the stale fast
 * buckets age out within ~`SLOT_BUCKETS` seconds. The current partial bucket
 * is included in the estimate so the first reading arrives within a couple of
 * frames of startup, not a second later.
 *
 * Fed with caller-supplied timestamps (no `performance.now()` here) so the
 * governor's synthetic-clock dev handle exercises this too.
 */
export class DisplayRateEstimator {
  private lastNow = -1;
  private bucketStart = -1;
  private bucketMin = Number.POSITIVE_INFINITY;
  private mins: number[] = [];

  /** Feed one frame timestamp, in ms, once per rendered frame. */
  note(nowMs: number): void {
    if (this.lastNow >= 0) {
      const d = nowMs - this.lastNow;
      if (d >= SLOT_MIN_MS && d <= SLOT_MAX_MS) {
        this.bucketMin = Math.min(this.bucketMin, d);
      }
    }
    this.lastNow = nowMs;
    if (this.bucketStart < 0) {
      this.bucketStart = nowMs;
    } else if (nowMs - this.bucketStart >= SLOT_BUCKET_MS) {
      if (Number.isFinite(this.bucketMin)) {
        this.mins.push(this.bucketMin);
        if (this.mins.length > SLOT_BUCKETS) this.mins.shift();
      }
      this.bucketMin = Number.POSITIVE_INFINITY;
      this.bucketStart = nowMs;
    }
  }

  /** The display rate in Hz, or null until a usable delta has been seen. */
  get hz(): number | null {
    const min = Math.min(this.bucketMin, ...this.mins);
    return Number.isFinite(min) ? 1000 / min : null;
  }
}
