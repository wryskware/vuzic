/** App-global fixed simulation clock. Never derive this from rAF or wall time. */
export const TICK_HZ = 120;
export const SECONDS_PER_TICK = 1 / TICK_HZ;

/**
 * Physarum and VizFx are authored as 60-step-per-second models. The app clock is
 * now twice that rate, so they consume one model tick for every two app ticks.
 */
export const LEGACY_SUBSTRATE_HZ = 60;
export const LEGACY_TICK_DIVISOR = TICK_HZ / LEGACY_SUBSTRATE_HZ;

