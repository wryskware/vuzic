/**
 * The budget governor's decision law — the pure part, split from `plife.ts` so
 * the cross-machine cases can be tested without a GPU (`tests/governor.test.ts`
 * walks the panels: 240/165/144/120/60 Hz, weak GPUs, crossed sliders).
 *
 * ## The sensor is the presented frame rate, on purpose (2026-08-20)
 *
 * This reverses the same-day GPU-meter law, at the user's direction, and the
 * reasoning is worth keeping: the deliverable is *playable on other people's
 * machines from the browser*, and the meter law was display-blind — it sized
 * every machine's world against a fixed `idealFps` budget, so a 60 Hz panel ran
 * a half-empty frame forever, and no reading of GPU microseconds can say
 * whether frames are actually reaching the screen. "How many frames does the
 * browser show per second" is the one number that means the same thing on every
 * machine, so it is the whole sensor.
 *
 * The cost, accepted knowingly: fps conflates causes. Another tab's jank sheds
 * particles the particles did not earn. On a stranger's machine that trade is
 * right — shedding buys back playability whoever caused the lag — and the
 * shipped hysteresis (slow grow, 1 s cadence) keeps a passing hitch from
 * mattering.
 *
 * ## The law
 *
 *   fps <  floorFps × FLOOR_TOLERANCE   → shrink
 *   fps >= target  × GROW_TOLERANCE     → grow,   target = min(idealFps, display)
 *   otherwise                           → hold
 *
 * `idealFps` is a CEILING on ambition, not a demand: the live grow target is
 * the display's own measured rate capped by it. That is what generalizes the
 * old fixed law — a 60 Hz panel grows toward 60, a 144 Hz one toward 120, and
 * the author's 240 Hz one stops asking past 120 — without anyone touching a
 * slider. With no display estimate yet (the first frames of a session) the
 * target falls back to `idealFps`, which merely delays growth, never sheds.
 *
 * A machine that cannot hold the floor even at `BUDGET_MIN` simply stays shed;
 * below ~60 fps the world then runs slower than wall time under `MAX_FRAME_DT`
 * while the audio plays on — slow, not weird, which is the failure mode the
 * whole mechanism is for.
 */

export type GovernorVerdict = 'shrink' | 'grow' | 'hold';

/**
 * How close to the grow target counts as meeting it — the vsync guard. A
 * display presents at its refresh rate and no faster, so a panel measures
 * 0.x fps *under* its nominal rate at its very best; a strict `fps >= target`
 * could never be met by the healthiest machine in the world. ×0.97 of a
 * 120-target is 116.4, which one frame of jitter in the ~25-frame EMA does not
 * fall below, while a genuinely struggling 100 fps still reads as "hold".
 */
export const GOVERNOR_GROW_TOLERANCE = 0.97;

/**
 * The same vsync guard, floor side — and it is the fix for a one-way ratchet.
 * The shipped `floorFps` is 60 and a 60 Hz panel measures 59.x forever, so a
 * strict `fps < floorFps` shed 15% every interval on a perfectly healthy
 * machine, and with the grow target equally unreachable mid-shed the budget
 * only ever moved down. Wider than the grow side's 0.97 because the two errors
 * are not symmetric: tolerating 56 fps on a 60 Hz panel is invisible,
 * collapsing the population for measurement noise is not.
 */
export const GOVERNOR_FLOOR_TOLERANCE = 0.94;

/**
 * One decision, from one fps sample against the two sliders and the measured
 * display rate (`DisplayRateEstimator.hz`, null until it has data).
 *
 * `idealFps` is read as `max(idealFps, floorFps)` so crossed sliders degenerate
 * into "shed below the floor, grow at the floor" rather than a band both
 * branches claim; the grow threshold is additionally floored at `floorFps` so a
 * throttled display estimate (a machine so slow no frame ever fit one slot)
 * cannot invite growth from inside the shed band.
 */
export function governorVerdict(
  fps: number,
  floorFps: number,
  idealFps: number,
  displayHz: number | null,
): GovernorVerdict {
  const ideal = Math.max(idealFps, floorFps);
  if (fps < floorFps * GOVERNOR_FLOOR_TOLERANCE) return 'shrink';
  const target = displayHz === null ? ideal : Math.min(ideal, displayHz);
  if (fps >= Math.max(target, floorFps) * GOVERNOR_GROW_TOLERANCE) return 'grow';
  return 'hold';
}
