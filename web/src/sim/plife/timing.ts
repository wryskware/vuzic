import { SECONDS_PER_TICK } from '../../timing.ts';

/** Particle Life intentionally takes one true dt integration per 120 Hz app tick. */
export const PLIFE_SUBSTEP_DT = SECONDS_PER_TICK;

