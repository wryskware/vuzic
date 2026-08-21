/**
 * Particle life — configuration and shipped art direction.
 *
 * The substrate is the second one in the repo (physarum is the first) and it
 * exists because the two fail differently: physarum makes *networks* — slow,
 * accreting, structural — and particle life makes *swarms* — fast, granular,
 * legible on the beat. Both satisfy `ModTarget`, so they share the whole
 * modulation / impulse / render stack and differ only in what θ means.
 *
 * ## The species partition (this is the design, not an implementation detail)
 *
 * K = 8, in two halves:
 *
 *   0..3  **primaries**  — keyed to the four stems (bass, drums, vocals, other).
 *                          They interact densely: the full 4×4 attraction block
 *                          is live and every cell of it is modulated.
 *   4..7  **secondaries** — sparse-coupled accent species, keyed to the *same*
 *                          stems (species 4 ↔ stem 0, … species 7 ↔ stem 3).
 *                          Each couples to exactly three things: its own
 *                          primary (chases it), one neighbour primary (avoids
 *                          it) and itself (disperses, so it reads as filigree
 *                          rather than as another blob).
 *
 * Every other cell of the 8×8 matrix is exactly 0 **and excluded from
 * modulation** — see `coupled()` in `preset.ts`. That is what makes the
 * partition hold *under* modulation: if the uncoupled cells were modulated they
 * would drift off zero and within a minute the secondaries would just be four
 * more primaries, which is the whole difference gone. Secondaries also run small
 * populations (aliveFraction ~0.12) because an accent that is as dense as the
 * thing it accents is not an accent.
 *
 * ## Units
 *
 * World space is `h = 1`, `w = aspect`, toroidal. Every radius, size and speed
 * below is in those units, so a "0.02" radius is 2% of the screen height and
 * means the same thing on any canvas. The grid the neighbour search uses has
 * cells of at least `R_CAP`; how far the search reaches is `nearStencil` cells,
 * so the two numbers are no longer the same thing — see `R_CAP` and
 * `PlifeFieldConfig` below.
 */
import { defaultRenderConfig, type RenderConfig } from '../render/config.ts';
import { customPalette, type Palette } from '../palette.ts';
import { oneOf, range, type RuleTree } from '../../mapping/read-into.ts';
import { persisted, persistedElsewhere, type BlockTable } from '../../mapping/blocks.ts';
import { defaultPlifeLuma, lumaRules, type PlifeLumaConfig } from './luma.ts';

/**
 * The neighbour grid's **cell size** — and only that, since the near stencil
 * landed. It used to be the interaction cap as well; the two have been split.
 *
 * What each number is now:
 *
 *   `R_CAP`         the grid's cell size. An *occupancy* choice: it sets how
 *                   many particles land in one cell list, which is what the
 *                   force pass's inner loop iterates. Smaller cells mean shorter
 *                   lists and more of them.
 *   `nearStencil`   how many cells in each direction the force pass searches
 *                   (`PlifeFieldConfig`). The reach cap is `stencil · cell`.
 *
 * The correctness invariant is unchanged in substance: a pair further apart than
 * the search window is silently missed, so *some* cap must equal the window. It
 * is now `stencil × cell` rather than `1 × cell`, clamped in the shader against
 * the live cell size from Globals so it can never go stale.
 *
 * Why keep the cell at 0.02 rather than growing it with reach: cost. The force
 * pass tests every candidate in the (2s+1)² neighbourhood, so per-particle work
 * goes as (reach)² · density either way — but a *larger cell* wastes a fixed
 * fraction of that work on candidates well outside the reach (the corners of the
 * search window), while a *smaller cell with a wider stencil* keeps the searched
 * area a tighter fit around the actual radius. 0.02 was measured as the right
 * occupancy for this particle count; the stencil is what buys reach on top of it.
 */
export const R_CAP = 0.02;

/**
 * Widest near stencil the panel and the loader allow. 3 cells ≈ 0.06 world, i.e.
 * 6% of the screen height, at ~5.4× the pair work of a 1-cell search (49 cells
 * vs 9). Higher is not forbidden by anything structural — it is a frame-time
 * judgement, and it is also the point past which the far-field lane (phase 2) is
 * simply a better way to buy scale than more brute-force pairs.
 */
export const MAX_NEAR_STENCIL = 3;

/**
 * Hard ceiling on any effective interaction radius, `maxR[i][j] · radiusScale[i]`
 * — the widest stencil's reach. This is the bound a *file* may contain and the
 * bound the panel sliders expose; the shader still clamps to the reach of the
 * stencil actually in force, so dialling the stencil down never silently starts
 * missing pairs.
 */
export const MAX_REACH = R_CAP * MAX_NEAR_STENCIL;

/**
 * The **brute** lane's reach cap, and — since the two modes share one authored
 * ceiling — the widest outer radius a file, a slider or a seeded draw may hold.
 *
 * 0.5 is half the unit torus, and it is a *geometric* bound rather than a taste
 * one: `wrapDelta` takes the minimum image, so at separations past half the
 * world the "nearer" copy of a particle is the one through the seam and a radius
 * beyond 0.5 stops meaning "further" — it means every pair is inside every
 * radius. The author's prior sims cap their radius slider at the same place for
 * the same reason.
 *
 * In **grid** mode the shader still clamps to `nearStencil × cell` (≤ 0.06), so
 * an authored radius above that simply saturates. That is deliberate and is the
 * same relationship the modulation band on `Rmax` already documents: one
 * authored ceiling, two runtime caps, and the mode you are in decides which one
 * binds. See `PlifeFieldConfig.pairSearch`.
 */
export const MAX_REACH_BRUTE = 0.5;

/** Floor on the hard-core radius. Below this the repulsion slope explodes. */
export const MIN_R_FLOOR = 0.002;

/**
 * Ceiling on the hard-core radius, `minR[i][j] · radiusScale[i]`.
 *
 * It used to be `R_CAP` (0.02), on the argument that a contact distance wider
 * than a grid cell is a different kind of sim. That argument was about the
 * *grid*, and the brute lane has no cells — what it has is regimes where the
 * outer radius is 0.3, and a 0.02 core inside a 0.3 tent is a pinhole: the tent
 * is then 93% ramp and every cluster collapses to a point before the core is
 * felt. 0.05 lets a large-radius world have a proportionate core.
 *
 * Judgement, and stated as one: 0.05 is 2.5× the old ceiling and a tenth of the
 * outer cap, which keeps a "core is a quarter of the reach" shape available at
 * reach 0.2 without letting the core become the whole force law. In grid mode a
 * core above the reach cap simply degenerates to pure repulsion — the shader
 * clamps `rmin` under `rmax`, so it stays safe rather than becoming garbage.
 */
export const MAX_MIN_R = 0.05;

/**
 * Ceiling on `brightness × stem-follow`, matching physarum's `MAX_BRIGHTNESS`
 * and for the same reason: impulse flashes multiply on top of this and are
 * deliberately *not* bounded by it, so a kick stays legible on a species the
 * stem lane is currently holding at its floor.
 */
export const MAX_BRIGHTNESS = 2;

/** Sprite clamps, mirrored by nothing in the shader — the CPU is the only guard. */
export const MAX_SIZE = 0.02;
export const MAX_STRETCH = 8;
export const MAX_FRICTION = 16;
export const MAX_RADIUS_SCALE = 3;

/**
 * The ceiling on `friction × drag` — the value that actually reaches the GPU,
 * after the macro has composed on top of θ.
 *
 * Deliberately a *different* number from `MAX_FRICTION`, because the two bound
 * different things. `MAX_FRICTION` is what the music and the sliders may author;
 * this is the hard rail behind the macro, and a rail has to sit above the whole
 * useful range or it stops being a rail and becomes a governor — the exact
 * failure the `speed`/`drag` split was made to cure. At the old shared value of
 * 16, `drag` 6 already clipped three of eight species, and worse, it clipped
 * them all to *exactly* 16: the per-species friction spread (the thing that
 * makes different species move differently) collapsed at the top of the slider.
 *
 * 64 is where the useful range ends rather than where stability does. The
 * damping is `exp(-friction · dt)`, which is unconditionally stable at any
 * value, so nothing here is a correctness bound; what 64 buys is that the
 * shipped species (friction 1.8 … 3.5) clear the top of the `drag` slider
 * without clipping — 3.5 × 16 = 56 — while a file that has *also* pushed θ
 * friction toward its own ceiling still gets caught. Past ~550 the exponential
 * kills the whole velocity every substep and more range would buy nothing.
 */
export const MAX_EFFECTIVE_FRICTION = 64;

export interface PlifeSpeciesConfig {
  name: string;
  role: 'primary' | 'secondary';
  /**
   * false = this voice is silent: its population target is 0 and it drains
   * through the normal fall-τ. K stays 8 — the pool segment, the matrix row and
   * the grid are all untouched — so re-enabling is instant and costs no rebuild.
   *
   * It lives here, on the config, rather than in θ, and that is the whole point:
   * `PlifeSpeciesPreset` omits it, so the modulator, the personality jitter, the
   * explorer's mutations and rebase all structurally cannot see it. "Four
   * primaries and one accent" is an authored decision about *which voices exist*,
   * and a decision the music can undo on the next chorus is not a decision.
   *
   * Deliberately not wired to brightness or intensity: zero population is
   * already zero light, and dimming as well would make a re-enable fade in
   * twice.
   */
  enabled: boolean;
  /** 0..2, the stem-follow *base* — same semantics as physarum's `brightness` */
  brightness: number;
  /** HDR weight multiplier, 0..4 */
  intensity: number;
  /** fraction of this species' pool segment that is alive */
  aliveFraction: number;
  /** multiplies this species' interaction radii as the RECEIVER (row scaling) */
  radiusScale: number;
  /** multiplies the total force applied to this species */
  forceScale: number;
  /** per-second exponential velocity damping: v *= exp(-friction·dt) */
  friction: number;
  /** noise-force amplitude, world units / s² */
  wander: number;
  /** sprite radius in world units */
  size: number;
  /** velocity stretch factor for the sprite quad */
  stretch: number;
}

/**
 * The population lane's art direction. Outside θ, exactly like physarum's `soil`
 * block: `presetFromConfig` / `applyVector` never see these fields, so a
 * modulated run and a hand-tuned run share them unchanged.
 *
 * The layering is the same one stem-follow already uses for brightness:
 *
 *   target_k = enabled_k × clamp(aliveFraction_k (θ) × popMul_k × accentMul_k, 0, 1) × segSize
 *
 * θ owns the base — how big this colony is *as a matter of parameter* — and the
 * two multipliers own the deviation from it. Neither is allowed to redefine the
 * base, so the modulator and the population lane cannot fight, and turning the
 * lane off restores θ's number exactly.
 */
export interface PlifePopulationConfig {
  /** false = aliveFraction is absolute again; every popMul stays at 1 */
  followStems: boolean;
  /** what a fully silent instrument keeps, 0..1 — a third of its colony, not none */
  floor: number;
  /** exponent on the smoothed stem level; >1 means only loud passages are crowded */
  curve: number;
  /**
   * EMA time constant in ms. Deliberately ~4× stem-follow's 300: brightness may
   * track a beat, but a colony that gains and sheds a third of its members every
   * bar reads as flicker, not as breathing. Population is the slow lane.
   */
  smoothingMs: number;
  /** seconds for a spawning particle to reach full presence */
  riseTau: number;
  /** seconds for a dismissed particle to reach zero. Longer than riseTau: things
   * should arrive decisively and leave reluctantly. */
  fallTau: number;
  accent: PlifeAccentConfig;
}

/**
 * Novelty → the accent species' population. Secondaries (4..7) only.
 *
 * The seeded random projections already give the accents *character* — which
 * ones cluster, how far they reach, what they chase. What they cannot give is
 * legibility: on one seed the chorus makes the accents bloom, on the next it
 * makes them wander sideways, and "the chorus arrived" stops being something a
 * viewer can see. This wire is the fix, and it is direct on purpose: novelty
 * goes to accent population, on every seed, with no projection in between.
 */
export interface PlifeAccentConfig {
  enabled: boolean;
  /** what the accents keep in a plain, unsurprising section, 0..1 */
  floor: number;
  /** headroom above the θ base at full novelty; 1.0 = the accents can double */
  boost: number;
  /** EMA time constant in ms; faster than the stem lane, because a section
   * boundary is an event and should not take two bars to land */
  smoothingMs: number;
}

export function defaultPlifePopulation(): PlifePopulationConfig {
  return {
    followStems: true,
    floor: 0.35,
    curve: 1,
    smoothingMs: 1200,
    riseTau: 0.4,
    fallTau: 1.5,
    accent: {
      enabled: true,
      floor: 0.15,
      boost: 1.0,
      smoothingMs: 800,
    },
  };
}

/**
 * The performance layer: seven always-yours multipliers, composing OUTSIDE θ
 * exactly like stem-follow and the impulse lanes do — which is why modulation
 * can never overwrite them. 1.0 everywhere = neutral.
 *
 * This is the rung that was missing between "16 driver gains" (which change what
 * the music *says*, indirectly, and differently on every seed) and 268 fine
 * sliders (which the modulator writes over on the next tick). A macro is neither:
 * it is a multiplier applied where the sim composes its final numbers, so it
 * survives every reroll, every load and every tick, and it means the same thing
 * on every seed.
 */
export interface PlifeMacros {
  /** × every species' population target (pre-clamp) */
  density: number;
  /** × the effective forceGain */
  force: number;
  /** × every species' radiusScale (the shader still clamps at the cell size) */
  reach: number;
  /**
   * × maxSpeed — the velocity **ceiling**, and nothing else.
   *
   * Split out of the old `agility` macro on 2026-08-13, after measuring that the
   * ceiling was not a safety rail but the sim's primary velocity governor: at
   * shipped defaults the force/friction equilibrium sits ~4× the ceiling at the
   * median and ~40× at p99, so 36-45% of live particles were pinned at exactly
   * `maxSpeed` and the rest sat just under it. The clamp renormalises v
   * isotropically, so it keeps heading and deletes speed — which is precisely
   * the "everything moves at the same speed" symptom.
   *
   * `agility` could not fix that at any setting, and that is why it had to go:
   * it multiplied the ceiling AND divided friction, and dividing friction raises
   * the equilibrium by the same factor it raises the ceiling. Both halves moved
   * the same side of the saturation inequality, so no value of one knob could
   * ever unpin the sim. These two can, in opposite directions.
   */
  speed: number;
  /**
   * × every species' friction — the exponential damping rate, and nothing else.
   *
   * This is the knob that lowers the equilibrium *under* the ceiling: terminal
   * speed goes as force/friction, so `drag` divides it. Raise this to recover
   * velocity spread without making the world faster; raise `speed` instead to
   * keep the spread and let the whole thing move quicker.
   */
  drag: number;
  /** × every species' wander */
  chaos: number;
  /**
   * × the impulse lane's **wiggle** depth — how far an event displaces its
   * species' row of the interaction matrix (`genmatrix.ts`'s
   * `wiggleAttraction`). 0 turns the lane off outright; the per-kind depths in
   * the events folder are what it multiplies, and they already carry the scale.
   *
   * Ranged 0..4 rather than 0..2 like the trims because it is a feature knob,
   * and because the lane's own rail (`WIGGLE_LIMIT`) sits four times above θ's,
   * so there is real range above the shipped depths to push into.
   */
  wiggle: number;
  /** secondaries only: × population target and × intensity */
  accents: number;
  /** × render.feedback.amount (clamped ≤ 0.97) */
  trails: number;
}

export function defaultPlifeMacros(): PlifeMacros {
  return {
    density: 1,
    force: 1,
    reach: 1,
    speed: 1,
    drag: 1,
    chaos: 1,
    wiggle: 1,
    accents: 1,
    trails: 1,
  };
}

/**
 * Panel range per macro, and the same table `PlifeSim.applyExtras` clamps a
 * loaded file into. One source so a saved value can never sit outside the slider
 * that is supposed to show it.
 */
export const MACRO_RANGE: Readonly<Record<keyof PlifeMacros, { min: number; max: number }>> = {
  density: { min: 0, max: 2 },
  force: { min: 0, max: 2 },
  reach: { min: 0, max: 2 },
  // Wider than the 0..2 the rest of the rig uses, and measured rather than
  // guessed: the equilibrium sits ~4× the ceiling at the median, so a 0..2
  // `speed` could not lift the ceiling clear of it and a 0..2 `drag` could not
  // pull the equilibrium down to it. These two need the range to actually reach
  // the unsaturated regime; the others are trims and 0..2 is right for them.
  // (The working point the user settled on is `speed` 1.5, well inside this.)
  speed: { min: 0, max: 4 },
  // Raised 6 → 16 on user report (2026-08-13): tuning landed at `speed` 1.5 with
  // `drag` at the old ceiling and still wanting more. Measurement agreed — at
  // drag 6 three of eight species were already pinned on the post-macro clamp —
  // so the clamp moved to `MAX_EFFECTIVE_FRICTION` and this followed it up.
  drag: { min: 0, max: 16 },
  chaos: { min: 0, max: 2 },
  wiggle: { min: 0, max: 4 },
  accents: { min: 0, max: 2 },
  trails: { min: 0, max: 1.5 },
};

/**
 * Files written before 2026-08-13 carry a single `agility` macro, which meant
 * "× maxSpeed, ÷ friction". Both halves survive the split, so the migration is
 * exact rather than approximate — `speed = agility` reproduces the old ceiling
 * and `drag = 1/agility` reproduces the old damping — and a loaded file replays
 * at the look it was saved at rather than snapping to neutral.
 *
 * Pure, and returns a copy: `PlifeSim.applyExtras` runs this over the untrusted
 * `extras.macros` blob before its clamping walk, and the caller's object is not
 * this function's to write into. A file that already carries `speed`/`drag`
 * wins outright — those are the current schema, and a stale `agility` sitting
 * beside them must not override what the newer keys say.
 */
export function migrateLegacyMacros(
  src: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const out = { ...src };
  const legacy = out['agility'];
  // `> 0` and not merely finite: the old wiring divided by this, so 0 (and any
  // negative, which the old range could not produce but a hand-edited file can)
  // has no inverse to migrate and falls through to the shipped default.
  if (typeof legacy === 'number' && Number.isFinite(legacy) && legacy > 0) {
    if (out['speed'] === undefined) out['speed'] = legacy;
    if (out['drag'] === undefined) out['drag'] = 1 / legacy;
  }
  return out;
}

/** Order the macro folder lists them in, with the wiring stated in the label. */
export const MACRO_LABELS: readonly { key: keyof PlifeMacros; label: string }[] = [
  { key: 'density', label: 'density  (× all populations)' },
  { key: 'force', label: 'force  (× forceGain·forceScale)' },
  { key: 'reach', label: 'reach  (× radius scales)' },
  { key: 'speed', label: 'speed  (× velocity ceiling)' },
  { key: 'drag', label: 'drag  (× friction)' },
  { key: 'chaos', label: 'chaos  (× wander)' },
  { key: 'wiggle', label: 'wiggle  (× impulse → matrix rows)' },
  { key: 'accents', label: 'accents  (× accent pop + glow)' },
  { key: 'trails', label: 'trails  (× echo persistence)' },
];

/**
 * How far the two force lanes reach. Structural — outside θ, like the macro rig
 * and the population lane, and for the same reason: this is a decision about
 * what *kind* of world this is (how much of it one particle can feel), not a
 * quantity the music should be sweeping bar by bar.
 *
 * Not to be confused with the interaction *matrix*, which is θ and is drawn from
 * the seed. This block is the geometry the matrix acts through.
 *
 * seam: a θ/modulation lane for the far knobs is deliberately future work. If it
 * ever lands, `farGain` and `farScale` move into `preset.ts`'s slot table and out
 * of here; `nearStencil` and `pairSearch` do not, because they change the
 * *search window* and a modulated search window is a modulated correctness
 * bound.
 */
export type PairSearch = 'grid' | 'brute';

/** The legal `pairSearch` values, in panel order. One list, so the dropdown and
 * the loader's validation cannot drift apart. */
export const PAIR_SEARCH_MODES: readonly PairSearch[] = ['grid', 'brute'];
export interface PlifeFieldConfig {
  /**
   * How the near lane finds its pairs. This is the one structural knob that
   * changes the *shape* of the cost, not its size.
   *
   *   `'grid'`   the uniform grid: each particle searches the (2s+1)² cells
   *              around it. Cost is `alive × neighbours-in-window`, i.e. linear
   *              in the population at fixed density — but the reach is capped at
   *              `nearStencil × R_CAP` ≤ 0.06, because a pair outside the search
   *              window is silently missed.
   *   `'brute'`  every particle against every live particle. Cost is O(N²) and
   *              there is no reach cap but the torus's own: radii up to
   *              `MAX_REACH_BRUTE` (0.5) are exact.
   *
   * Grid is the default and stays the regression baseline. Brute exists because
   * the 0.06 ceiling is not a tuning limit, it is a *kind* limit: at 6% of screen
   * height a cluster is made of many strands agreeing, and the large-scale flows
   * and continent-sized clusters a particle-life sim is worth watching for
   * simply cannot form. The author's prior sims were brute-force and ran 32 k
   * particles with radii to 0.5; this is that lane, restored.
   *
   * Not modulated and not in θ, exactly like `nearStencil`: it changes what the
   * force pass *searches*, and a modulated search is a modulated correctness
   * bound.
   */
  pairSearch: PairSearch;
  /**
   * Cells searched in each direction by the force pass, 1..`MAX_NEAR_STENCIL`.
   * Effective near reach cap is `nearStencil × R_CAP` — 0.02 / 0.04 / 0.06.
   * **Grid mode only**: brute mode has no cells and ignores this entirely.
   */
  nearStencil: number;
  /**
   * The far lane's global strength. 0 turns the whole density pyramid off — no
   * splat, no blur, no sampling — so it is a real bypass, not a zero multiply.
   *
   * **Grid mode only**, like `nearStencil`: the lane exists to hand the
   * truncated cell search a long-range term, and brute's pair sum already
   * reaches the half-torus cap. In brute mode the whole chain is bypassed
   * regardless of this value (`farActive` in plife.ts), and the BRUTE pipeline
   * compiles the sampling out.
   *
   * Sized so 1.0 is a *current*, not a force: measured, it is a median terminal
   * drift around 0.06 world/s against a near-lane median speed of ~0.26, which
   * roughly doubles the coarse-grained density contrast without changing how
   * often the speed clamp is hit. 3 is the strongly-segregating end. See
   * `FAR_FORCE_SCALE` in plife.ts for the sweep those numbers come from.
   */
  farGain: number;
  /**
   * σ2, the far lane's outer smoothing scale, in world units. This is the size
   * of the organisation the lane produces: 0.05 is a coarse grain, 0.5 is
   * hemisphere-scale banding.
   *
   * σ1 has no knob — it is `nearStencil × R_CAP`, the near lane's own cutoff, so
   * the two lanes meet at one seam by construction. σ2 is clamped up to
   * `FAR_SCALE_MIN_RATIO × σ1` if it is dialled below it, because a
   * difference-of-Gaussians with σ2 ≤ σ1 has no band left to be a band.
   */
  farScale: number;
}

/**
 * Panel range and loader clamp for the far knobs, in one table for the same
 * reason `MACRO_RANGE` is: a saved value can never sit outside the slider that
 * is meant to show it.
 */
export const FAR_GAIN_RANGE = { min: 0, max: 3 } as const;
export const FAR_SCALE_RANGE = { min: 0.05, max: 0.5 } as const;

/** σ2 must exceed σ1 by at least this factor or the DoG band collapses. */
export const FAR_SCALE_MIN_RATIO = 1.5;

/**
 * The reference far scale the lane's strength is normalised to. The sampled
 * gradient is multiplied by `farScale / FAR_SCALE_REF`, so moving the far scale
 * changes the SIZE of the organisation and not how hard it pulls: a structure at
 * scale σ2 has a DoG gradient going as 1/σ2, and this cancels it.
 */
export const FAR_SCALE_REF = 0.2;

export function defaultPlifeField(): PlifeFieldConfig {
  return {
    pairSearch: 'grid',
    // Back to 1 (user call, 2026-08-09). The 2 that shipped here bought reach
    // 0.04 instead of 0.02 at 2.8× the pair work, and the trade did not survive
    // contact: past ~32 k alive it was a felt frame-time regression, and the
    // reach it bought is a rounding error next to what brute mode makes
    // available. Grid mode is back to its old cost envelope and the stencil is
    // what you raise when you want 0.04 / 0.06 *within* that lane.
    nearStencil: 1,
    farGain: 1,
    farScale: FAR_SCALE_REF,
  };
}

/**
 * The particle budget and its adaptive governor.
 *
 * Structural, outside θ, and outside the macro rig too — it is not art direction
 * at all. Every other population lever in this sim answers "how big should this
 * colony be"; this one answers "how much of this machine is there", which is a
 * property of the hardware the world is being watched on and not of the world.
 *
 * ## Where it composes
 *
 * `uploadSpecies` builds each species' target from θ's `aliveFraction` through
 * the population lane, the accent lane, the `density` and `accents` macros and
 * the `enabled` switch. The budget clamp runs *after* all of that, on the sum: if
 * Σtargets exceeds `effectiveBudget` every target is scaled by the same factor,
 * so the species **ratios** — which are the art direction — survive untouched and
 * only the absolute count moves. Fewer is fine; more than the budget never
 * happens.
 *
 * That ordering is the whole design. `density` stays a scaler you can push (it
 * simply stops buying particles once the budget binds), and the budget stays a
 * ceiling rather than a competing opinion about the mix.
 *
 * ## The governor
 *
 * `effectiveBudget` is the governor's state, not a setting: it lives between a
 * hard floor (`BUDGET_MIN`) and `cap`, and it moves in response to the measured
 * frame rate — down fast when the frame rate falls under `floorFps`, up slowly
 * when it is meeting what this machine's display can show (its measured refresh
 * rate, capped by `idealFps` — see `governor.ts`), and nowhere at all in
 * between. It is deliberately
 * **session-only**: what your machine could sustain last night, in another tab
 * configuration, at another window size, is not a fact about this run. Only the
 * four settings below are persisted.
 *
 * Nothing here despawns anything directly. The governor moves *targets*, and the
 * population lane's rise/fall τ is what turns a moved target into a fade — which
 * is why a budget cut reads as the world thinning out rather than as a third of
 * it blinking off.
 */
export interface PlifeBudgetConfig {
  /**
   * Hard ceiling on Σ(per-species targets). Defaults to `maxParticles`, i.e. no
   * clamp at all out of the box: the pool is already the bound, and a budget that
   * shipped below it would silently change what the defaults look like.
   */
  cap: number;
  /** false = `effectiveBudget` is pinned at `cap` and the frame rate is ignored */
  adaptive: boolean;
  /** below this measured fps the governor sheds particles */
  floorFps: number;
  /** at (or within tolerance of) this measured fps the governor grows back */
  idealFps: number;
}

/**
 * The lowest the governor may take `effectiveBudget`, and the bottom of the cap
 * slider. Below about this the sim stops being a particle-life sim and becomes a
 * few hundred dots per species — if 2 048 particles will not hold the floor fps,
 * the answer is a different `pairSearch` or a smaller window, not fewer
 * particles, and pretending otherwise would let a bad frame time chase the world
 * down to nothing.
 *
 * 2^11 rather than a round 2 000, and that is not decoration: it is the bottom of
 * the cap slider, whose step is `BUDGET_STEP` and whose top is `maxParticles`
 * (2^18 by default). Tweakpane quantises a stepped slider onto a grid anchored at
 * one of its ends, and with a min, max and step that are not commensurate the
 * widget snaps to a grid neither end sits on — which was observed writing 15 144
 * back into a cap that had been set to 15 000. Making all three powers of two
 * puts both ends on the grid whichever end it is anchored to.
 */
export const BUDGET_MIN = 2_048;

/**
 * Cap-slider granularity. 2^10, so `BUDGET_MIN`, the step and the default pool
 * (2^18) are all commensurate — see the note above. Fine enough to land within
 * 0.4% of any budget worth asking for, coarse enough that a drag produces a
 * number you can read back out.
 */
export const BUDGET_STEP = 1_024;

/** Panel range and loader clamp for the two frame-rate knobs. */
export const BUDGET_FPS_RANGE = { min: 15, max: 240 } as const;

export function defaultPlifeBudget(cap: number): PlifeBudgetConfig {
  return {
    cap: Math.max(Math.round(cap), BUDGET_MIN),
    adaptive: true,
    // 60 / 120 rather than, say, 30 / 60: this sim's whole legibility argument is
    // that a hit resolves *now*, and a 30 fps particle world reads as a slideshow
    // of clusters rather than as motion. `idealFps` is a CEILING on the grow
    // target, not a demand: the governor targets min(idealFps, the display's
    // measured refresh rate), so a 60 Hz panel grows toward 60 and a 240 Hz one
    // stops asking past 120 — see `governor.ts` for the law and the tolerances
    // that make both ends reachable on real panels.
    floorFps: 60,
    idealFps: 120,
  };
}

/**
 * How a seed *draws* the interaction matrix and the radii.
 *
 * These are **generation** settings, not parameters: they act once, at reroll
 * time (`seedMatrixBase` in `genmatrix.ts`), and never per tick. Nothing here is
 * in θ, nothing here is modulated, and nothing here is read by the shader — what
 * reaches the GPU is the matrix these numbers produced.
 *
 * Why they exist at all: before this the matrix and the radii shipped as
 * constants and the seed only jittered them, so every seed looked like the same
 * world wearing a slightly different hat. Drawing the whole matrix from the seed
 * — the way a hand-rolled particle-life sim does — is what makes a reroll a new
 * *creature* rather than a new mood.
 *
 * **Persistence:** these ride the mapping file's opaque `extras` block (see
 * `PlifeSim.serializeExtras` / `applyExtras`). A load restores the settings, but
 * it does *not* re-draw: the drawn θ is already in the saved vector, so a loaded
 * file reproduces its look regardless, and pressing "redraw from settings" is
 * how you ask for a new matrix under the restored numbers.
 */
export interface MatrixGenConfig {
  /** stddev of the Gaussian each coupled attraction cell is drawn from */
  sigma: number;
  /** 0 = fully asymmetric draw (chasing dynamics), 1 = forced symmetric (the classic rule) */
  symmetry: number;
  /** added to every PRIMARY diagonal after the draw; <0 = self-avoiding (filigree), >0 = cohesive blobs */
  selfBias: number;
  /**
   * Added to every SECONDARY diagonal instead. Defaults negative: an accent
   * that self-attracts collapses its small population into one bright point —
   * self-dispersal is what makes accents read as filigree threaded through the
   * primaries rather than as four more blobs.
   */
  selfBiasAccent: number;
  /** multiplier on the drawn value in every secondary-coupled cell (the accents' edges) */
  accentGain: number;
  /**
   * Which family of **impulse-wiggle direction vectors** this world uses —
   * the reroll counter, not a strength.
   *
   * The directions are drawn from `(seed, wiggleRoll)` in `seedWiggleDirs`, so a
   * pinned seed reproduces them and the workbench's reroll button (which just
   * increments this) asks for a different set of things that hits *do* to the
   * matrix while leaving the matrix itself, the personality and the world alone.
   *
   * It lives in this block rather than in a block of its own because this block
   * is exactly "what the seed draws, and how" — the wiggle directions are a
   * seeded draw over the same matrix — and because a new top-level config block
   * is a schema event for the export recipe (`PLIFE_KEYS`) while a new field in
   * an existing one is not.
   */
  wiggleRoll: number;
  /** uniform band the per-pair inner (hard-core) radius is drawn from, world units */
  rMin: { lo: number; hi: number };
  /** uniform band the per-pair outer radius is drawn from, world units */
  rMax: { lo: number; hi: number };
}

/**
 * The shipped generation settings.
 *
 * `symmetry` 0.25 is mostly-asymmetric on purpose: a symmetric matrix settles
 * into static blobs, and the chase/orbit behaviour that makes particle life
 * worth watching *is* the asymmetry. `selfBias` +0.15 keeps every species
 * cohesive enough to clump at all, since a zero-mean draw would leave half the
 * diagonals self-repulsive and those species would never form anything.
 */
/**
 * Where `matrixGen.wiggleRoll` wraps. Nothing is special about 2^20 beyond being
 * far past any number of times a person will press a button and small enough
 * that it stays an exact integer everywhere it is mixed.
 */
export const MAX_WIGGLE_ROLL = 1 << 20;

export function defaultMatrixGen(): MatrixGenConfig {
  return {
    sigma: 0.45,
    symmetry: 0.25,
    selfBias: 0.15,
    selfBiasAccent: -0.2,
    accentGain: 1.0,
    wiggleRoll: 0,
    rMin: { lo: 0.003, hi: 0.005 },
    rMax: { lo: 0.008, hi: 0.014 },
  };
}

/**
 * Ranges are the panel's own (`ui/plife-panel.ts`), so a loaded value can never
 * sit outside the slider meant to show it.
 *
 * The rule tables only *bound* values. A field with no entry still persists; it
 * is simply unclamped, which is right for the ones whose legal range is "any real
 * number". Nothing is enrolled in persistence by being named here — enrolment is
 * the defaults function's job (see `readInto`) one level down and the block
 * declaration's job (see `PLIFE_BLOCKS`) one level up, which is what makes these
 * tables safe to leave incomplete.
 *
 * The non-obvious floors: `population.curve` is an exponent a 0..1 level is
 * raised to, and `riseTau` / `fallTau` are divisors in the shader — all three
 * have a degenerate value at zero that the sliders already refuse to produce.
 * `nearStencil` is an integer because the shader indexes with it, and a slider
 * that snapped visually to 2 while holding 2.4 would be a search window
 * disagreeing with its own label. The radius bands share `MIN_R_FLOOR` as a
 * floor but not a ceiling: an outer radius may reach `MAX_REACH_BRUTE` (half the
 * torus), while the hard core stays under `MAX_MIN_R` — it is a contact
 * distance, not a reach.
 */
const MACRO_RULES: RuleTree = Object.fromEntries(
  (Object.keys(MACRO_RANGE) as (keyof PlifeMacros)[]).map((key) => {
    const r = MACRO_RANGE[key];
    return [key, range(r.min, r.max)];
  }),
);

const band = (cap: number): RuleTree => ({
  lo: range(MIN_R_FLOOR, cap),
  hi: range(MIN_R_FLOOR, cap),
});

const MATRIX_GEN_RULES: RuleTree = {
  sigma: range(0, 4),
  symmetry: range(0, 1),
  selfBias: range(-1, 1),
  selfBiasAccent: range(-1, 1),
  accentGain: range(0, 2),
  // A counter, not a quantity: any non-negative integer is a legal family of
  // wiggle directions, and the ceiling is only here so a corrupt file cannot hand
  // `Math.imul` something absurd. Wrapped rather than clamped by the panel — see
  // the reroll button.
  wiggleRoll: range(0, MAX_WIGGLE_ROLL, { int: true }),
  rMin: band(MAX_MIN_R),
  rMax: band(MAX_REACH_BRUTE),
};

const POPULATION_RULES: RuleTree = {
  floor: range(0, 1),
  curve: range(0.2, 4),
  smoothingMs: range(100, 5000),
  riseTau: range(0.05, 3),
  fallTau: range(0.1, 8),
  accent: {
    floor: range(0, 1),
    boost: range(0, 3),
    smoothingMs: range(100, 5000),
  },
};

const FIELD_RULES: RuleTree = {
  // An unknown / absent / mistyped mode falls back to the default rather than
  // throwing: this block is opaque to everything between the file and the sim,
  // and the one thing a bad value must not do is put an O(N²) pass on screen by
  // accident. Grid is the safe answer in both directions.
  pairSearch: oneOf(PAIR_SEARCH_MODES),
  nearStencil: range(1, MAX_NEAR_STENCIL, { int: true }),
  farGain: range(FAR_GAIN_RANGE.min, FAR_GAIN_RANGE.max),
  farScale: range(FAR_SCALE_RANGE.min, FAR_SCALE_RANGE.max),
};

/**
 * `cap` is bounded by the *live* pool rather than by whatever the file thought it
 * was: a cap above the pool is not a bigger world, it is a clamp that never
 * binds, and stating it as `maxParticles` keeps the slider honest. Hence a
 * function of the live config rather than a flat table.
 */
const budgetRules = (cfg: PlifeConfig): RuleTree => ({
  cap: range(BUDGET_MIN, Math.max(cfg.maxParticles, BUDGET_MIN), { int: true }),
  floorFps: range(BUDGET_FPS_RANGE.min, BUDGET_FPS_RANGE.max),
  idealFps: range(BUDGET_FPS_RANGE.min, BUDGET_FPS_RANGE.max),
});

/**
 * **Which blocks of `PlifeConfig` are saved — and declaring one here is all it
 * takes to save it.**
 *
 * This is the roadmap's phase 1 item 5, and the shape is `mapping/blocks.ts`'s:
 * the table is exhaustive over the *config's own* object-valued keys, so adding
 * `foo: FooConfig` to `PlifeConfig` is a compile error until `foo` is declared
 * here, and one `persisted(defaultFoo)` enrols it in serialize, restore **and**
 * clamp at once. Before this, a new block had to be remembered in three separate
 * places (`PlifeSim.extrasBlocks()`, a defaults table and a rules table) and
 * nothing failed if it was remembered in none of them — the block simply never
 * saved, and the report arrived weeks later as "my tweak didn't save".
 *
 * Order is the serialized key order, and is the order that shipped before this
 * table existed, so an autosave written by an older build still diffs cleanly
 * against a newer one.
 *
 * Two blocks are declared *not* saved here, which is the other half of the
 * design: an opt-out is a sentence somebody wrote and somebody reviewed, and
 * `unpersistedBlocks()` puts every one of them under an exhaustive CI assertion.
 */
export const PLIFE_BLOCKS: BlockTable<PlifeConfig> = {
  macros: persisted(defaultPlifeMacros, {
    rules: MACRO_RULES,
    // Pre-split files carry `agility` instead of `speed`/`drag`. It runs before
    // the walk so the migrated values are bounded by the same table as the rest.
    migrate: migrateLegacyMacros,
  }),
  matrixGen: persisted(defaultMatrixGen, { rules: MATRIX_GEN_RULES }),
  population: persisted(defaultPlifePopulation, { rules: POPULATION_RULES }),
  field: persisted(defaultPlifeField, { rules: FIELD_RULES }),
  // The four SETTINGS of the budget block, and only those. The governor's own
  // state (`PlifeSim.governorBudget`, surfaced as `effectiveBudget`) is
  // deliberately not a field of this block and must never become one: it is a
  // live measurement of this machine in this session, and a saved one would open
  // every future run of this mapping at whatever the frame rate happened to be
  // when it was written. It lives on the sim, outside the config entirely, which
  // is the structural way of saying the same thing — and `plife-extras` pins it.
  budget: persisted((cfg) => defaultPlifeBudget(cfg.maxParticles), { rules: budgetRules }),
  // The per-particle luminance lane. Every field is art direction and persists;
  // the display headroom it is composed against is deliberately NOT a field of
  // this block (it is a measurement of the panel this session happens to be
  // watched on — same reasoning as `effectiveBudget` above). Every knob is
  // bounded, including the ones whose degenerate value is only *nearly*
  // harmful: `curve` is an exponent the shader raises a possibly-zero base to,
  // and `mid` is its own base — the shader floors both again, but a file
  // should not be the reason it has to.
  luma: persisted(defaultPlifeLuma, { rules: lumaRules() }),

  render: persistedElsewhere(
    'the mapping file owns the render chain: `ModulationConfig.render` IS this ' +
      'object, shared by reference, and `parseModulation` reads it. A second copy ' +
      'in `extras` would be two encodings of one block for a loader to disagree ' +
      'about.',
  ),
  palette: persistedElsewhere(
    'same as render — `ModulationConfig.palette` is this object by reference, and ' +
      'the palette needs a reader `readInto` cannot be (per-species colours, ' +
      're-derived from the arcs in arc mode). See `parseModulation`.',
  ),
};

/** Scene exposure / gamma — θ slots the look tab owns, hoisted out of θ's vector. */
export const LOOK_RULES: RuleTree = { exposure: range(0.05, 4), gamma: range(1, 3) };

export interface PlifeConfig {
  /** phase-7 render chain. Structural (outside θ) but saved with the mapping. */
  render: RenderConfig;
  /** the performance layer: seven multipliers the modulator never writes. Outside θ. */
  macros: PlifeMacros;
  /** dynamic population: stems → colony size, novelty → accents. Outside θ. */
  population: PlifePopulationConfig;
  /** how a seed draws the matrix and radii. Generation-time only; outside θ. */
  matrixGen: MatrixGenConfig;
  /** how far the near (grid) lane reaches. Structural; outside θ. */
  field: PlifeFieldConfig;
  /** particle ceiling + the adaptive governor. Structural; outside θ. */
  budget: PlifeBudgetConfig;
  /**
   * Per-particle brightness dynamic range: speed → luminance, with the display's
   * HDR headroom as the budget. Outside θ, like every other lane that composes
   * on top of it — see `luma.ts`. `depth: 0` reproduces the pre-lane look.
   */
  luma: PlifeLumaConfig;
  /** K. 8 by design — see the partition note at the top of this file. */
  speciesCount: number;
  /** pool size; the per-species segment is floor(maxParticles / K) */
  maxParticles: number;
  /** global multiplier on the summed pair force */
  forceGain: number;
  /** world units per second, hard clamp on |v| */
  maxSpeed: number;
  /** scene exposure, multiplied in the particle shader (pre-measure) */
  exposure: number;
  gamma: number;
  /** sim substeps per clock tick, accumulated so fractional values work */
  speed: number;
  paused: boolean;
  /** static per-species colour; not part of θ */
  palette: Palette;
  species: PlifeSpeciesConfig[];
  /** K·K row-major; `[i*K+j]` = force ON species i FROM species j */
  attraction: number[];
  /** K·K, world units — per-pair inner (hard-core) radius. Static; not modulated. */
  minR: number[];
  /** K·K, world units — per-pair outer radius */
  maxR: number[];
}

/**
 * Shipped hues. 0–3 are the primaries and deliberately reuse physarum's stem
 * colours — bass is orange in both sims, which is the one piece of identity
 * worth sharing across substrates. 4–7 are the accents: lighter, higher-key
 * relatives of their primary, so a secondary reads as "the same instrument,
 * sparkling" rather than as a fifth voice.
 */
const PALETTE_HEX: readonly string[] = [
  '#ff7a1a', // bass
  '#ff2f6d', // drums
  '#35d6ff', // vocals
  '#a56bff', // other
  '#ffd24a', // bass·acc   — warm gold off the orange
  '#ff8a5c', // drums·acc  — coral off the magenta
  '#8dffe0', // vocals·acc — mint off the cyan
  '#f0a8ff', // other·acc  — orchid off the violet
];

/** Fallback colour for `paletteLinear` when the palette is shorter than K. */
export function defaultPlifePaletteColor(index: number): string {
  return PALETTE_HEX[index % PALETTE_HEX.length] as string;
}

/** Custom mode with the v2 fields neutral — bit-identical to the pre-v2 default. */
export function defaultPlifePalette(k: number): Palette {
  return customPalette(
    Array.from({ length: Math.max(1, k) }, (_, i) => defaultPlifePaletteColor(i)),
  );
}

/** How many species are primaries. The rest are accents keyed to the same stems. */
export const PRIMARY_COUNT = 4;

interface Template extends Omit<PlifeSpeciesConfig, 'name' | 'brightness' | 'enabled'> {
  name: string;
}

/**
 * Per-species character. The differentiation is *mild* on purpose: this is the
 * shipped starting point and the seeded personality jitter (preset.ts) plus the
 * workbench are what turn it into a look. What each knob buys:
 *
 * - **bass** reaches furthest (`radiusScale` 1.25) and slides longest
 *   (`friction` 1.8). Low end reads as mass through motion and reach, not
 *   through sprite size any more — the size spread across species is capped
 *   near ×1.4 (user call, 2026-08-08: large per-species size differences read
 *   as ugly translucent orbs among dots, not as hierarchy).
 * - **drums** is the opposite in every axis: short reach (0.85), heavy damping
 *   (3.5) and a small sprite, so a hit resolves *now* and does not smear into
 *   the next one. This is the species the impulse lane hits hardest.
 * - **vocals** gets extra wander (0.035), which is the cheapest way to make a
 *   line feel sung rather than sequenced — it never settles completely.
 * - **other** gets the longest streak (`stretch` 3.5), which turns pads and
 *   texture into motion rather than into more dots.
 *
 * Secondaries share one template: sparse (0.1 alive), bright (1.6), small,
 * loose (`forceScale` 1.3, `wander` 0.05) and very streaky (`stretch` 6.5).
 * Together those make an accent that arrives as a spark and leaves a comet.
 *
 * The `stretch` values were raised ~1.6× when the feedback zoom went to 1.0.
 * A radially expanding echo used to manufacture a fake streak on every particle,
 * which drowned the real velocity stretch; with that gone the honest cue has to
 * carry the motion by itself, and at the old numbers it barely read.
 */
const TEMPLATES: readonly Template[] = [
  {
    name: 'bass',
    role: 'primary',
    intensity: 1.2,
    aliveFraction: 0.35,
    radiusScale: 1.25,
    forceScale: 1,
    friction: 1.8,
    wander: 0.02,
    size: 0.0022,
    stretch: 4,
  },
  {
    name: 'drums',
    role: 'primary',
    intensity: 1.2,
    aliveFraction: 0.35,
    radiusScale: 0.85,
    forceScale: 1,
    friction: 3.5,
    wander: 0.02,
    size: 0.0018,
    stretch: 4,
  },
  {
    name: 'vocals',
    role: 'primary',
    intensity: 1.2,
    aliveFraction: 0.35,
    radiusScale: 1,
    forceScale: 1,
    friction: 2.5,
    wander: 0.035,
    size: 0.0022,
    stretch: 4,
  },
  {
    name: 'other',
    role: 'primary',
    intensity: 1.2,
    aliveFraction: 0.35,
    radiusScale: 1,
    forceScale: 1,
    friction: 2.5,
    wander: 0.02,
    size: 0.0022,
    stretch: 5.5,
  },
];

const ACCENT: Omit<Template, 'name'> = {
  role: 'secondary',
  intensity: 1.6,
  aliveFraction: 0.1,
  radiusScale: 1,
  forceScale: 1.3,
  friction: 2,
  wander: 0.05,
  size: 0.0016,
  stretch: 6.5,
};

/**
 * `brightness` and `enabled` are supplied here rather than in the templates
 * above because both are *state*, not character: every shipped species starts
 * lit and present, and the templates describe what a voice is like once it is.
 */
export function defaultPlifeSpecies(index: number, k = 8): PlifeSpeciesConfig {
  const primaries = Math.min(PRIMARY_COUNT, k);
  if (index < primaries) {
    const t = TEMPLATES[index % TEMPLATES.length] as Template;
    return { ...t, brightness: 1, enabled: true };
  }
  const stem = (index - primaries) % TEMPLATES.length;
  const base = TEMPLATES[stem] as Template;
  return { ...ACCENT, name: `${base.name}·acc`, brightness: 1, enabled: true };
}

/**
 * The primary 4×4 attraction block. Row i = force on species i; column j = from
 * species j.
 *
 * These are **starting points for the workbench**, not a tuned look. The shape
 * they encode: every primary is self-cohesive (+0.4…+0.55 on the diagonal, which
 * is what makes a species clump at all), and the off-diagonals are mixed sign
 * with mild asymmetry — bass pulls on drums harder than drums pulls back, vocals
 * pushes 'other' away while 'other' chases it. Asymmetric pairs are the whole
 * reason particle life produces chase/orbit behaviour instead of settling into
 * static blobs, so the asymmetry is the load-bearing part; the exact magnitudes
 * are not.
 */
const PRIMARY_BLOCK: readonly number[] = [
  //  bass  drums vocals other
  0.5, -0.2, 0.15, 0.3, // bass
  -0.25, 0.45, 0.35, -0.15, // drums
  0.2, 0.1, 0.55, -0.3, // vocals
  0.35, -0.1, -0.2, 0.4, // other
];

/**
 * The full K×K attraction matrix, with the secondary partition applied.
 *
 * Secondary `4+n`: chases primary n (+0.5), avoids primary n+1 (−0.35), and
 * disperses from itself (−0.15) so it strings out into filigree instead of
 * balling up. Primary n barely notices its own accent (+0.08) — enough that the
 * accent perturbs the primary's shape, far too little to let the accent drag it.
 * Every other cell is exactly 0 and stays that way (preset.ts excludes them).
 *
 * **This is a placeholder, not what you will see.** The modulator's `seedBase`
 * hook (`genmatrix.ts`) overwrites the whole attraction block on the first
 * rewire, which happens before the first frame — so on screen the matrix is
 * always a seeded draw. This function survives because it is the pre-rewire
 * value (the app is briefly alive without a Modulator, e.g. with no timeline),
 * and because it is the shape the tests pin the partition against.
 */
export function defaultAttraction(k = 8): number[] {
  const a = new Array<number>(k * k).fill(0);
  const p = Math.min(PRIMARY_COUNT, k);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      a[i * k + j] = PRIMARY_BLOCK[i * PRIMARY_COUNT + j] ?? 0;
    }
  }
  for (let n = 0; n + p < k && n < p; n++) {
    const i = p + n;
    a[i * k + n] = 0.5;
    a[i * k + ((n + 1) % p)] = -0.35;
    a[i * k + i] = -0.15;
    a[n * k + i] = 0.08;
  }
  return a;
}

/**
 * Per-pair radii, deterministic rather than random.
 *
 * The variety here is *shipped art direction*: a matrix where every pair has the
 * same band produces a single characteristic length and the image looks like one
 * material. Different bands per pair are what make some pairs form tight shells
 * and others loose halos at the same attraction strength.
 *
 * There is no RNG in this function on purpose — the seeded draw lives in
 * `genmatrix.ts`, which owns both radius blocks outright and overwrites these on
 * the first rewire, exactly as it does the attraction matrix. What is left here
 * is the pre-rewire placeholder and the shape the tests pin the legality rules
 * (band width, floor, cap) against.
 */
export function defaultRadii(k = 8): { minR: number[]; maxR: number[] } {
  const minR = new Array<number>(k * k).fill(0);
  const maxR = new Array<number>(k * k).fill(0);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const o = i * k + j;
      const lo = 0.003 + (0.002 * ((i * 3 + j * 7) % 5)) / 4;
      const hi = 0.008 + (0.006 * ((i * 5 + j * 3) % 7)) / 6;
      minR[o] = lo;
      // A band narrower than 4 mm of world makes the tent a spike, which reads
      // as a hard shell with no gradient — always leave room for the ramp.
      maxR[o] = Math.max(hi, lo + 0.004);
    }
  }
  return { minR, maxR };
}

/**
 * 2^18. See the perf note in plife.ts: this is the single biggest lever on frame
 * time, because the force pass cost is (alive particles) × (candidates per
 * neighbourhood) and both terms scale with it. The first build shipped 2^20; the
 * user's call was to cap around 256k — the fully-swelled worst case (every
 * population multiplier at its ceiling) is bounded by this rather than merely
 * unlikely to be hit.
 *
 * Named rather than inline because the budget's default cap *is* this number:
 * the shipped budget must be a no-op, and two literals that had to agree would
 * eventually not.
 */
export const DEFAULT_MAX_PARTICLES = 262_144;

export function defaultPlifeConfig(speciesCount = 8): PlifeConfig {
  const k = Math.max(1, Math.floor(speciesCount));
  const { minR, maxR } = defaultRadii(k);
  const render = defaultRenderConfig();
  // Feedback IS the trail persistence for plife. Physarum accumulates memory in
  // its own trail field and therefore ships with feedback off; a particle sim
  // has no field at all, so without a render-domain echo the image is a cloud of
  // unconnected dots and every motion cue is lost. 0.88 per 60 Hz frame is
  // roughly a half-second tail — long enough to draw the path, short enough that
  // the frame does not smear to mush.
  //
  // Zoom is 1.0 — no radial drift — and that is a fix, not a taste change. The
  // fade pass expands the echo away from the centre once per *rendered* frame,
  // so at 1.0015 on a 240 Hz display the echo crawled outward ~1.4× per second
  // and every particle grew a streak pointing away from screen centre with a
  // length proportional to its radius. That fake radial smear is what was being
  // read as the velocity stretch. Radial drift is now an opt-in effect: the
  // slider is still there, and the per-frame exponentiation in `writeGlobals`
  // means whatever you dial reads the same on any refresh rate.
  render.feedback = { amount: 0.88, zoom: 1.0 };
  // No soil in this sim, so the ember underlay has nothing to draw from.
  render.grade.soilTint = 0;
  return {
    render,
    macros: defaultPlifeMacros(),
    population: defaultPlifePopulation(),
    matrixGen: defaultMatrixGen(),
    field: defaultPlifeField(),
    budget: defaultPlifeBudget(DEFAULT_MAX_PARTICLES),
    luma: defaultPlifeLuma(),
    speciesCount: k,
    maxParticles: DEFAULT_MAX_PARTICLES,
    forceGain: 1,
    maxSpeed: 0.3,
    // The particle shader multiplies colour by exposure · 2^exposureEv and
    // auto-exposure adapts on top of that, so this only has to put the frame in
    // the right decade. 1.0 was the first guess and measured wrong: at default
    // density the controller sat pinned at its 0.05 minimum-gain rail (scene
    // ~20× too bright before adaptation), which leaves it no room to pull a
    // dense chorus down. 0.1 parks the adapted gain mid-range with headroom in
    // both directions — same reasoning that put physarum at 0.01.
    exposure: 0.1,
    gamma: 2.2,
    speed: 1,
    paused: false,
    palette: defaultPlifePalette(k),
    species: Array.from({ length: k }, (_, i) => defaultPlifeSpecies(i, k)),
    attraction: defaultAttraction(k),
    minR,
    maxR,
  };
}
