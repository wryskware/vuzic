/**
 * The rate the offline analysis timeline is **sampled** at. This is a data rate,
 * not a simulation clock: `TimelineSampler` is indexed by tick, `EventCursor`
 * walks the event list on it, and the offline export renders on it.
 *
 * The live loop does NOT step on this grid. It advances the world once per
 * rendered frame by the measured wall delta (see `main.ts`), and only reads this
 * to work out *where in the timeline* the current audio position is.
 */
export const TICK_HZ = 120;
export const SECONDS_PER_TICK = 1 / TICK_HZ;

/**
 * Physarum and VizFx are authored as 60-step-per-second models: every per-step
 * quantity in their configs and θ tables (move distance, rotation, trail decay,
 * zoom rate, fade) means "per 1/60 s". That reference rate survives the move to
 * variable dt — `planSteps` in `sim/frame-timing.ts` expresses a frame's world
 * time in units of it.
 */
export const LEGACY_SUBSTRATE_HZ = 60;
export const LEGACY_MODEL_DT = 1 / LEGACY_SUBSTRATE_HZ;

/**
 * Hard ceiling on the world time one rendered frame may advance, in seconds.
 *
 * A frame that took longer than this advances the world by this much and no
 * more, so the world runs *slower* than wall time on a machine that cannot keep
 * up rather than the loop trying to catch back up — which is the death spiral
 * the old fixed-grid pump could get into, where being behind bought more work
 * per frame and therefore more lateness. It is also the tab-restore guard: a
 * backgrounded tab returns with a multi-second delta, and without the clamp the
 * first frame back would integrate minutes of world in one step.
 *
 * 1/60, not 1/30, after the first real-hardware pass (2026-08-20): at 1/30 a
 * particle at the speed ceiling crosses several hard-core radii in one step, so
 * close pairs tunnel through their repulsion instead of being turned by it and
 * tight clusters break. 1/60 keeps the worst-case step at the scale the shipped
 * forces were tuned for (it is also the legacy substrates' authored model dt).
 * The price is that below 60 fps the world slows proportionally sooner.
 */
export const MAX_FRAME_DT = 1 / 60;

/** Measured wall delta → the world time a frame is allowed to advance. */
export function clampFrameDt(seconds: number): number {
  return seconds > 0 ? Math.min(seconds, MAX_FRAME_DT) : 0;
}

/**
 * Largest jump in timeline position, in ticks, that still counts as continuous
 * playback rather than as somebody having moved the playhead.
 *
 * Three lanes share it — `EventCursor`'s walk, the Modulator's stem-follow snap,
 * and main's section-boundary detector — and they must agree, because they are
 * three readings of one question: did the music arrive here, or was it put here?
 *
 * Under the fixed tick grid this was **1**. The pump ran every tick in order, so
 * a normal advance was exactly one tick and anything else was a seek. One
 * advance per rendered FRAME breaks that outright: a 60 Hz display crosses two
 * ticks of a 120 Hz timeline every frame and a 30 fps one crosses four, so a
 * threshold of 1 would classify ordinary playback as a continuous seek — every
 * impulse relocated away without firing, the stem lane snapping every frame.
 * The threshold therefore has to be a *duration*.
 *
 * The duration is the longest FRAME still treated as ordinary playback, and it
 * is deliberately NOT derived from `MAX_FRAME_DT`: audio time is unclamped, so
 * a slow frame crosses its full wall duration of timeline no matter how little
 * world the sim was allowed to advance — tying the two together would mean
 * tightening the sim clamp silently reclassified slow-but-playing frames as
 * seeks and killed every impulse on struggling machines. 15 fps (~67 ms, 8
 * ticks) is the floor of "still playing"; past it, a scrub, an arrow key, an
 * A/B restore or a restart relocates without firing, which is what stops a drag
 * across a chorus dumping every kick in it into one frame.
 */
const SLOWEST_CONTINUOUS_FPS = 15;
export const MAX_CONTINUOUS_TICK_GAP = Math.ceil(TICK_HZ / SLOWEST_CONTINUOUS_FPS);
