/**
 * The seeded interaction-matrix draw.
 *
 * ## Why this exists
 *
 * Until now the attraction matrix and the per-pair radii shipped as *constants*
 * (`defaultAttraction` / `defaultRadii` in config.ts) and the seed only jittered
 * them through the generic personality pass in `mapping/modulation.ts`. Jitter
 * around a fixed shape is not a new world: the same species chased the same
 * species at slightly different strengths, and every seed looked like the same
 * creature in a different mood. A hand-rolled particle-life sim draws its whole
 * matrix from noise on every reset, and that is what makes a reset interesting.
 *
 * So this module draws the three K² blocks wholesale from the seed, and it runs
 * **after** `baseVector` — the generic jitter goes first over everything, then
 * this overwrites the blocks it owns. Slots it does not touch (the species
 * block, the globals) keep the personality the generic path gave them. That
 * ordering is stated once in `Modulator.rewire` and depended on here.
 *
 * ## The rules, and why each one is there
 *
 * A pure Gaussian matrix is not automatically a good one. Four generation knobs
 * (`MatrixGenConfig`) shape the draw:
 *
 * - **sigma** — the scale of the whole thing. Too small and everything is a gas;
 *   too large and every pair collapses into its hard core.
 * - **symmetry** — a blend toward `(a[i][j] + a[j][i]) / 2`. Fully symmetric is
 *   the classic rule and it is *boring*: symmetric forces conserve enough that
 *   the field settles into static blobs. Asymmetry is what buys chase and orbit.
 *   The blend only fires when both cells are coupled, because the partition
 *   (see `coupled()`) outranks it — symmetrising a live cell against a zero one
 *   would drag the live one halfway to zero and quietly dissolve the partition.
 * - **selfBias** — added to the diagonal after the draw. A zero-mean diagonal
 *   leaves half the species self-repulsive, and a self-repulsive species never
 *   forms anything to look at. Positive is cohesive blobs, negative is filigree.
 * - **accentGain** — scales every secondary-coupled cell, so the accents can be
 *   made more or less violent than the primaries without touching sigma.
 *
 * Radii are drawn for **every** pair, coupled or not. The shader's hard-core
 * repulsion term reads minR/maxR for a pair whose attraction is exactly zero —
 * two particles that ignore each other still cannot occupy the same point — so
 * leaving the uncoupled cells at whatever stale value the base held would make
 * the partition's own pairs collide on a different scale from everyone else.
 */
import { hash3 } from '../impulses.ts';
import { makeRng } from '../../mapping/modulation.ts';
import { MIN_R_FLOOR, PRIMARY_COUNT, R_CAP, type PlifeConfig } from './config.ts';
import { ATTRACTION_MOD, coupled, matrixBase, maxRBase, minRBase } from './preset.ts';

/**
 * Domain separator for the matrix stream, in the same family as the mapping
 * layer's `KEY_WIRING` / `KEY_BASE`. Distinct so that the matrix draw, the
 * projection wiring and the personality jitter never consume the same numbers
 * for the same (seed, index) — three uses of one seed that must not correlate.
 */
const KEY_MATRIX = 0x5eed_a71c;

/**
 * Minimum width of a pair's force tent, world units. A band narrower than this
 * makes the ramp a spike, which reads as a hard shell with no gradient — the
 * same 3 mm floor `defaultRadii` enforces, restated because this function is the
 * one that has to guarantee it after two independent uniform draws.
 */
const MIN_BAND = 0.003;

/** Box–Muller, one draw per call. Two uniforms in, the first normal out. */
function gaussian(rng: () => number): number {
  // rng() can return exactly 0; nudge it off the log's singularity, exactly as
  // `unitDirection` does in the mapping layer.
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Overwrite the attraction, maxR and minR blocks of an already-built base vector
 * with a draw from `seed`.
 *
 * Pure: everything it reads is (seed, cfg) and everything it writes is `base`.
 * Deterministic per *cell* rather than per stream position — the RNG is re-keyed
 * from `hash3(seed, i·K + j, KEY_MATRIX)` for each pair — so the value in a cell
 * does not depend on how many cells were visited before it, and K could change
 * without every pair moving.
 */
export function seedMatrixBase(seed: number, cfg: PlifeConfig, base: Float64Array): void {
  const k = cfg.speciesCount;
  const gen = cfg.matrixGen;
  const p = Math.min(PRIMARY_COUNT, k);
  const mBase = matrixBase(k);
  const xBase = maxRBase(k);
  const nBase = minRBase(k);

  const sigma = Math.max(gen.sigma, 0);
  const accentGain = Math.max(gen.accentGain, 0);
  const symmetry = clamp(gen.symmetry, 0, 1);

  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const cell = i * k + j;
      const rng = makeRng(hash3(seed >>> 0, cell >>> 0, KEY_MATRIX));

      // The Gaussian is drawn unconditionally, even for a cell that will be
      // written as 0. That keeps the stream position identical for every cell,
      // so the radii below land in the same place whatever the partition says —
      // changing `coupled()` then changes which cells attract, not the whole
      // world's geometry.
      const g = gaussian(rng);
      let a = 0;
      if (coupled(i, j, k)) {
        a = g * sigma;
        // A secondary edge is any cell touching an accent, in either direction:
        // the accent's own three couplings and the primary's one back-coupling.
        if (i >= p || j >= p) a *= accentGain;
        // Two self-bias knobs, one per role, because the roles WANT opposite
        // signs: primaries cohere (positive — a species that never clumps never
        // forms anything to look at), accents disperse (negative — an accent
        // that self-attracts collapses its small population into a single
        // bright point, which is the "singularity" failure the first generator
        // shipped by applying the primary bias to every diagonal).
        if (i === j) a += i < p ? gen.selfBias : gen.selfBiasAccent;
      }
      base[mBase + cell] = a;

      // Radii: every pair, coupled or not — the hard core is not optional.
      const minR = Math.max(uniform(rng, gen.rMin.lo, gen.rMin.hi), MIN_R_FLOOR);
      const maxR = Math.min(
        Math.max(uniform(rng, gen.rMax.lo, gen.rMax.hi), minR + MIN_BAND),
        R_CAP,
      );
      base[nBase + cell] = minR;
      base[xBase + cell] = maxR;
    }
  }

  // Symmetry, as a second pass: it reads two cells at once, so it cannot happen
  // inside the draw loop without the second cell being undrawn half the time.
  if (symmetry > 0) {
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        if (!coupled(i, j, k) || !coupled(j, i, k)) continue;
        const ij = mBase + i * k + j;
        const ji = mBase + j * k + i;
        const aij = base[ij] as number;
        const aji = base[ji] as number;
        const m = (aij + aji) * 0.5;
        base[ij] = aij + (m - aij) * symmetry;
        base[ji] = aji + (m - aji) * symmetry;
      }
    }
  }

  // Clamp last, and only the coupled cells: an uncoupled cell is exactly 0 and
  // must stay bit-exactly 0, which a clamp into a range that contains 0 would
  // preserve anyway but which is worth not relying on. The bound is the
  // attraction slot's own ModSpec, so "where a drawn matrix may sit" and "where
  // the music may wander" are the same interval by construction.
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      if (!coupled(i, j, k)) continue;
      const o = mBase + i * k + j;
      base[o] = clamp(base[o] as number, ATTRACTION_MOD.lo, ATTRACTION_MOD.hi);
    }
  }
}

/** Uniform in [lo, hi], tolerant of an inverted band from a hand-dragged slider. */
function uniform(rng: () => number, lo: number, hi: number): number {
  const a = Math.min(lo, hi);
  const b = Math.max(lo, hi);
  return a + (b - a) * rng();
}
