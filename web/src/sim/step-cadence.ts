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

