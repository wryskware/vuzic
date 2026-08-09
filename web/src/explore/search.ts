/**
 * Explorer-mode search: a (1+8) evolution strategy over θ.
 *
 * The workbench's sliders are the *analytic* way to steer a look: you know which
 * number you want and you go move it. Explorer mode is the other way — nine
 * tiles, one of them the current world and eight of them near-misses of it, and
 * the human answers the only question they are reliably good at: "that one". No
 * gradient exists here (the objective is a person's taste, evaluated by looking),
 * so the search is the crudest thing that works on a black box with a noisy
 * ordinal oracle: sample around the incumbent, keep the winner, repeat. That is
 * a (1+λ) ES with λ = 8, and 8 is set by the 3×3 grid, not by any convergence
 * argument.
 *
 * Three design commitments are worth naming, because each has an obvious-looking
 * alternative that is worse here:
 *
 *   - **Per-slot scale, not a global step.** A perturbation is `σ · spec.half`,
 *     so the mutation borrows the same "how far is a legible move" number the
 *     music itself is scaled by (see `ModSpec.half` in modspec.ts). A single σ in
 *     raw units would move `sensorDist` (half 9) and `decay` (half 0.035)
 *     by the same absolute amount, which means one tile changes nothing visible
 *     and another one detonates.
 *
 *   - **Reflection at the bounds, not clamping.** Same argument as
 *     `baseVector`'s: a clamped draw piles probability mass onto the bound, and
 *     for the several slots whose useful values sit right against a bound that
 *     turns "eight different candidates" into "three different candidates and
 *     five identical ones". Reflection keeps the full spread.
 *
 *   - **Momentum as a *repeat*, not as an average.** When the human picks the
 *     same direction twice, what they are saying is "keep going". So candidate 0
 *     of the next generation is the winning displacement applied again with no
 *     fresh noise — a free line search along whatever the human liked, at zero
 *     cost in tiles, and immediately abandoned the moment they pick something
 *     else. A decayed velocity vector (the usual ES/Adam shape) would keep
 *     nudging in a stale direction after the human has changed their mind, which
 *     reads as the grid ignoring them.
 *
 * Everything is keyed off `hash3` rather than a rolling stream, so a
 * `CandidateSet` is a pure function of (start θ, the sequence of picks/rerolls,
 * subspace, step, runSeed) — the same reason the modulator's wiring is. That is
 * what makes `genSeed` in the verdict log worth recording: it reproduces all
 * nine θs of a judged generation from four numbers instead of storing them.
 *
 * ## The search is not the only thing that moves the centre
 *
 * `start`, `pick`, `reroll` and `back` are the search steering itself. `recenter`
 * is the *outside world* steering it: the workbench's sliders are live during
 * explorer mode, and a slider drag is a move through θ space exactly like a pick
 * is — just one the human aimed rather than chose. So it enters through the same
 * door, pushes history like a pick does, and advances the generation, and the
 * grid regrows around wherever the human put the centre. Without it the panel and
 * the grid are two disconnected searches over the same vector.
 *
 * CONTRACT: the exported names and shapes here are the seam the 9-up rig
 * consumes. Additions are fine; renames are not.
 */
import { makeRng, reflect } from '../mapping/modulation.ts';
import { hash3 } from '../sim/impulses.ts';
import type { ModGroup, ModSpec } from '../mapping/modspec.ts';
import type { ThetaRegistry } from '../mapping/target.ts';

/** 'all' or one of the four shared mod groups — the frozen/free split. */
export type ExplorerSubspace = ModGroup | 'all';

export const EXPLORER_SUBSPACES: readonly ExplorerSubspace[] = [
  'all',
  'structure',
  'matrix',
  'population',
  'decay',
];

/** Candidates per generation — the 8 non-center tiles of the 3×3. */
export const CANDIDATE_COUNT = 8;

export interface CandidateSet {
  /** the current best θ, full registry length */
  center: Float64Array;
  /** CANDIDATE_COUNT perturbed θ vectors, full length, all within slot bounds */
  candidates: readonly Float64Array[];
  /** RNG key of this set; (center, subspace, step, genSeed) reproduce it */
  genSeed: number;
  /** 0 at start(), +1 per pick/reroll */
  generation: number;
}

/**
 * Domain separators, in the same spirit as modulation.ts's `KEY_WIRING` /
 * `KEY_BASE` and genmatrix.ts's `KEY_MATRIX`. Two levels because the hash takes
 * three inputs and we have four (runSeed, generation, candidate, domain): the
 * generation key folds seed+generation, the candidate key folds that with the
 * tile index. Distinct constants so that an explorer stream can never collide
 * with the wiring, personality or matrix streams for the same seed — a collision
 * there would tie the shape of a mutation to the shape of the world it mutates.
 */
const KEY_ES_GEN = 0xe50e_115e;
const KEY_ES_CAND = 0xe5ca_11d0;

/** σ bounds. Below 0.05 every tile looks like the centre; above 1 the grid is noise. */
export const MIN_STEP = 0.05;
export const MAX_STEP = 1;
export const DEFAULT_STEP = 0.35;

/**
 * How far back `back()` can walk. 64 is generous for a session of hand-steering
 * and bounded so a long run cannot grow a `length`-wide Float64Array per pick
 * forever; the oldest centre is dropped, which is the one nobody walks back to.
 */
const MAX_HISTORY = 64;

/**
 * Standard gaussians from a uniform stream — Box–Muller, same idiom (and the
 * same `1e-12` guard against `rng()` returning exactly 0 at the log's
 * singularity) as `unitDirection`. The sine half is kept rather than discarded
 * because a perturbation asks for one gaussian per slot, not per pair, and
 * throwing half the stream away would double the draws for no benefit.
 */
function gaussianSource(rng: () => number): () => number {
  let spare: number | null = null;
  return function next(): number {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const a = 2 * Math.PI * u2;
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

/**
 * True when a slot's `ModSpec` wants its displacement measured in ln units.
 *
 * `mult` alone is not enough. A ratio only means anything about a strictly
 * positive quantity, so a multiplicative slot whose `lo` reaches 0 or below (no
 * shipped registry has one — this is defence against a future slot table) falls
 * back to the additive path rather than taking the log of a non-positive number.
 * `hi > lo` is required for the same reason: a collapsed range has no ln span to
 * reflect inside, and `reflect` already answers `lo` for it.
 */
function isLogScaled(spec: ModSpec): boolean {
  return spec.mult && spec.lo > 0 && spec.hi > spec.lo;
}

/**
 * Move `v` by `delta`, in whichever space the slot lives in, and land inside
 * `[lo, hi]`.
 *
 * The zero short-circuit is not an optimisation. `exp(log(v))` and
 * `lo + (v - lo)` are both off by an ULP or two from `v`, and there are slots a
 * generator owns outright and the search must not touch — plife's `Rmin` carries
 * a `ModSpec` with `half: 0` purely so the mask lets the seeded value through
 * (see `sim/plife/preset.ts`) — where a per-generation round trip through
 * log/exp would let a stability parameter drift. Zero displacement means the
 * value is returned untouched, bit for bit.
 */
function displace(v: number, delta: number, spec: ModSpec): number {
  if (delta === 0) return v;
  if (isLogScaled(spec)) {
    const clamped = Math.min(Math.max(v, spec.lo), spec.hi);
    return Math.exp(reflect(Math.log(clamped) + delta, Math.log(spec.lo), Math.log(spec.hi)));
  }
  return reflect(v + delta, spec.lo, spec.hi);
}

/**
 * The displacement `displace` would have to be given to turn `from` into `to`.
 *
 * The zero short-circuit mirrors `displace`'s and is load-bearing for the same
 * slots: without it, a log-scaled slot whose centre sits outside `[lo, hi]`
 * (legal — the mod range may be narrower than the hard θ bound) measures
 * `log(to) − log(clamp(from))` ≠ 0 for an unmoved value, and a centre at 0
 * measures −Infinity, which `reflect` turns into NaN in the momentum tile.
 */
function displacementOf(from: number, to: number, spec: ModSpec): number {
  if (from === to) return 0;
  if (isLogScaled(spec)) {
    const clamped = Math.min(Math.max(from, spec.lo), spec.hi);
    return Math.log(to) - Math.log(clamped);
  }
  return to - from;
}

function cloneSet(s: CandidateSet): CandidateSet {
  return {
    center: Float64Array.from(s.center),
    candidates: s.candidates.map((c) => Float64Array.from(c)),
    genSeed: s.genSeed,
    generation: s.generation,
  };
}

export class ExplorerSearch {
  private readonly registry: ThetaRegistry;
  /** the run's root key; `genSeed` is derived from this and the generation */
  readonly runSeed: number;

  private sub: ExplorerSubspace = 'all';
  private sigma = DEFAULT_STEP;

  private centre: Float64Array | null = null;
  private gen = 0;

  /** previous centres, oldest first; `back()` pops the tail */
  private readonly history: Float64Array[] = [];

  /**
   * The winning displacement of the last `pick`, per slot, in each slot's own
   * space (ln units where `isLogScaled`), zero everywhere that was frozen. Null
   * means "no direction to repeat" and all eight tiles are fresh draws.
   */
  private momentum: Float64Array | null = null;

  /** the last generated set, kept internally so `current()` and `pick()` never
   * depend on a copy the caller may have mutated */
  private last: CandidateSet | null = null;
  /** the subspace `last` was generated under — see `pick()` */
  private lastSub: ExplorerSubspace = 'all';

  constructor(registry: ThetaRegistry, seed: number) {
    this.registry = registry;
    this.runSeed = seed >>> 0;
  }

  get subspace(): ExplorerSubspace {
    return this.sub;
  }

  get step(): number {
    return this.sigma;
  }

  /** Generations completed since `start()`. */
  get generation(): number {
    return this.gen;
  }

  /** Whether the next generation's tile 0 repeats the last winning step. */
  get hasMomentum(): boolean {
    return this.momentum !== null;
  }

  /** How many `back()` steps remain. */
  get depth(): number {
    return this.history.length;
  }

  /**
   * Takes effect from the next generated set.
   *
   * Momentum is cleared because a held displacement is expressed over the slots
   * that were free when it was won; replaying it under a different frozen/free
   * split would move slots the human just froze, which is the one thing the
   * subspace control promises not to happen.
   */
  setSubspace(s: ExplorerSubspace): void {
    if (s === this.sub) return;
    this.sub = s;
    this.momentum = null;
  }

  /**
   * Mutation scale, 0.05..1 of each slot's half-range; clamped.
   *
   * Deliberately does *not* clear momentum: changing σ is "same direction,
   * bigger stride", which is exactly the line search momentum exists to serve.
   * (The repeated step itself keeps its original size — it is a replay of a
   * displacement, not a fresh draw — so σ takes hold on tiles 1..7 first.)
   */
  setStep(sigma: number): void {
    const s = Number.isFinite(sigma) ? sigma : DEFAULT_STEP;
    this.sigma = Math.min(Math.max(s, MIN_STEP), MAX_STEP);
  }

  /** Begin a run at θ (copied, not retained). Returns generation 0. */
  start(theta: Float64Array): CandidateSet {
    if (theta.length !== this.registry.length) {
      throw new Error(
        `ExplorerSearch.start: θ length ${theta.length} ≠ registry length ${this.registry.length}`,
      );
    }
    this.centre = Float64Array.from(theta);
    this.history.length = 0;
    this.momentum = null;
    this.gen = 0;
    return this.generate();
  }

  /** Candidate `index` wins: it becomes the new center; momentum retained. */
  pick(index: number): CandidateSet {
    if (!Number.isInteger(index) || index < 0 || index >= CANDIDATE_COUNT) {
      throw new Error(`ExplorerSearch.pick: index ${index} outside 0..${CANDIDATE_COUNT - 1}`);
    }
    const last = this.requireLast('pick');
    const previous = last.center;
    const winner = last.candidates[index] as Float64Array;

    // The displacement is measured over the slots that were free when the tile
    // was *drawn*, not the ones free now: those are the only slots that could
    // have moved, and the two differ only if the subspace changed between
    // generating and picking (which already cleared the old momentum).
    this.momentum = this.measure(previous, winner, this.lastSub);

    this.history.push(previous);
    if (this.history.length > MAX_HISTORY) this.history.shift();

    this.centre = Float64Array.from(winner);
    this.gen++;
    return this.generate();
  }

  /**
   * Fresh candidates around the same center (momentum kept).
   *
   * The generation counter still advances, so the new set gets a new `genSeed`
   * and the verdict log can tell "I looked at these eight and wanted eight
   * others" apart from "I looked at these eight" — a reroll is a judgement about
   * the whole neighbourhood, and it is preference data.
   */
  reroll(): CandidateSet {
    this.requireLast('reroll');
    this.gen++;
    return this.generate();
  }

  /**
   * Revert to the previous center; null when already at the root.
   *
   * Momentum goes with it. Undoing a step is the human saying that step was
   * wrong, and repeating it as tile 0 of the very next grid would be the search
   * arguing with them.
   */
  back(): CandidateSet | null {
    this.requireLast('back');
    const previous = this.history.pop();
    if (!previous) return null;
    this.centre = previous;
    this.momentum = null;
    this.gen++;
    return this.generate();
  }

  /**
   * Move the centre to a θ that came from *outside* the search — in practice a
   * workbench slider edited while the nine tiles are on screen — and regrow the
   * neighbourhood around it.
   *
   * Treated as a step of the climb rather than as a restart, and every part of
   * that is deliberate:
   *
   * - **History is pushed**, so `back()` undoes a slider drag exactly as it undoes
   *   a pick. The alternative (silently replacing the centre) would make the one
   *   act with no undo the one most likely to be a mistake.
   * - **Momentum is cleared.** A held displacement is the answer to "you liked
   *   that direction, want more of it?"; the human has just answered a different
   *   question by hand, and replaying the old step as tile 0 would be the search
   *   arguing with them. Same reasoning as `back()`.
   * - **The generation advances**, so the set gets a fresh `genSeed` and the
   *   verdict log can tell an edited neighbourhood apart from a rerolled one.
   *
   * `start()` is deliberately *not* what this reuses: a restart drops history and
   * would make an edit mid-climb throw away every generation that led to it.
   */
  recenter(theta: Float64Array): CandidateSet {
    if (theta.length !== this.registry.length) {
      throw new Error(
        `ExplorerSearch.recenter: θ length ${theta.length} ≠ registry length ${this.registry.length}`,
      );
    }
    this.requireLast('recenter');
    // `this.centre` is never aliased out (`generate` copies it into the set and
    // `cloneSet` copies again), so it can go onto the stack as-is.
    this.history.push(this.centre as Float64Array);
    if (this.history.length > MAX_HISTORY) this.history.shift();

    this.centre = Float64Array.from(theta);
    this.momentum = null;
    this.gen++;
    return this.generate();
  }

  current(): CandidateSet {
    return cloneSet(this.requireLast('current'));
  }

  // ── internals ───────────────────────────────────────────────────────────────

  private requireLast(who: string): CandidateSet {
    if (!this.last) throw new Error(`ExplorerSearch.${who}: called before start()`);
    return this.last;
  }

  /**
   * A slot moves only if the registry says the music may move it *and* it sits
   * in the active subspace. Everything else is copied bit-identically — the
   * frozen half of the frozen/free split has to be exactly frozen, or "I locked
   * the matrix and explored structure" quietly stops being true.
   */
  private isFree(i: number, sub: ExplorerSubspace): ModSpec | null {
    if ((this.registry.mask[i] as number) !== 1) return null;
    const spec = this.registry.slots[i] ?? null;
    if (!spec) return null;
    if (sub !== 'all' && spec.group !== sub) return null;
    return spec;
  }

  private generate(): CandidateSet {
    const centre = this.centre as Float64Array;
    const n = this.registry.length;
    const sub = this.sub;
    const genSeed = hash3(this.runSeed, this.gen >>> 0, KEY_ES_GEN);

    const candidates: Float64Array[] = [];
    for (let k = 0; k < CANDIDATE_COUNT; k++) {
      const out = new Float64Array(n);
      out.set(centre);
      if (k === 0 && this.momentum) {
        this.repeatStep(out, centre, sub, this.momentum);
      } else {
        this.perturb(out, centre, sub, hash3(genSeed, k >>> 0, KEY_ES_CAND));
      }
      candidates.push(out);
    }

    const set: CandidateSet = {
      center: Float64Array.from(centre),
      candidates,
      genSeed,
      generation: this.gen,
    };
    this.last = set;
    this.lastSub = sub;
    // The caller owns what it is handed; internal state is never aliased out.
    return cloneSet(set);
  }

  /** One fresh gaussian draw per free slot, scaled by σ·half. */
  private perturb(
    out: Float64Array,
    centre: Float64Array,
    sub: ExplorerSubspace,
    key: number,
  ): void {
    const gauss = gaussianSource(makeRng(key));
    for (let i = 0; i < out.length; i++) {
      const spec = this.isFree(i, sub);
      if (!spec) continue;
      // The draw is consumed even for a spec that cannot move (half 0), so that
      // one slot's mobility does not shift every later slot's stream.
      const g = gauss();
      out[i] = displace(centre[i] as number, this.sigma * spec.half * g, spec);
    }
  }

  /** Tile 0 under momentum: the winning displacement, again, with no noise. */
  private repeatStep(
    out: Float64Array,
    centre: Float64Array,
    sub: ExplorerSubspace,
    d: Float64Array,
  ): void {
    for (let i = 0; i < out.length; i++) {
      const spec = this.isFree(i, sub);
      if (!spec) continue;
      out[i] = displace(centre[i] as number, d[i] as number, spec);
    }
  }

  private measure(from: Float64Array, to: Float64Array, sub: ExplorerSubspace): Float64Array {
    const d = new Float64Array(from.length);
    for (let i = 0; i < d.length; i++) {
      const spec = this.isFree(i, sub);
      if (!spec) continue;
      d[i] = displacementOf(from[i] as number, to[i] as number, spec);
    }
    return d;
  }
}
