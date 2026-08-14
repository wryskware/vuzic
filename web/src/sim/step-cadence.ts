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
 */
export function smoothDtFrames(prev: number, deltaSeconds: number, snap = false): number {
  const dt = Math.min(Math.max(deltaSeconds, 1e-3), 0.25);
  const measured = dt * 60;
  if (snap || !(prev > 0)) return measured;
  const alpha = 1 - Math.exp(-dt / DT_FRAMES_TAU_S);
  return prev + (measured - prev) * alpha;
}

export interface StepCadenceResult {
  accumulator: number;
  steps: number;
}

/**
 * Expand a saved `speed` into whole substrate steps on its scheduled app ticks.
 *
 * `tickDivisor = 1` is Particle Life: every 120 Hz app tick is a model tick.
 * `tickDivisor = 2` is Physarum/VizFx: odd app ticks intentionally do no model
 * work and even ticks consume the unchanged per-model-tick `speed`. Gating the
 * accumulation, instead of multiplying saved speed by 0.5, preserves the old
 * grouping for fractional and multi-step speeds as well as the average rate.
 */
export function advanceStepCadence(
  accumulator: number,
  speed: number,
  maxSteps: number,
  appTick: number,
  tickDivisor: number,
): StepCadenceResult {
  const divisor = Math.max(1, Math.floor(tickDivisor));
  if (Math.abs(Math.trunc(appTick)) % divisor !== 0) return { accumulator, steps: 0 };

  const limit = Math.max(0, Math.floor(maxSteps));
  let next = accumulator + Math.max(speed, 0);
  let steps = 0;
  while (next >= 1 && steps < limit) {
    next -= 1;
    steps++;
  }
  // Preserve the substrates' existing spiral guard: discard a backlog larger
  // than one whole per-tick step budget rather than carrying it forever.
  if (next > limit) next = 0;
  return { accumulator: next, steps };
}

