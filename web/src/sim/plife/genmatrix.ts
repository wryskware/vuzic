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
import { MAX_REACH_BRUTE, MIN_R_FLOOR, PRIMARY_COUNT, type PlifeConfig } from './config.ts';
import {
  ATTRACTION_MOD,
  coupled,
  matrixBase,
  maxRBase,
  minRBase,
} from './preset.ts';

/**
 * Domain separator for the matrix stream, in the same family as the mapping
 * layer's `KEY_WIRING` / `KEY_BASE`. Distinct so that the matrix draw, the
 * projection wiring and the personality jitter never consume the same numbers
 * for the same (seed, index) — three uses of one seed that must not correlate.
 */
const KEY_MATRIX = 0x5eed_a71c;

/**
 * The wiggle lane's own domain separator, and the `roll` counter's mixer.
 *
 * Separate from `KEY_MATRIX` for the same reason that one is separate from the
 * mapping layer's keys: the perturbation directions and the matrix they perturb
 * are drawn from one seed and must not correlate, or every hit would push each
 * row along the direction it already points and the wiggle would read as a
 * volume knob on the matrix rather than as a change of rules.
 */
const KEY_WIGGLE = 0x77_19_61_ed;

/**
 * The wiggle lane's own rail on one attraction cell.
 *
 * Deliberately a different number from `ATTRACTION_LIMIT` (2), which bounds what
 * a *slider, a file or the seeded draw* may put in a cell. This bounds what an
 * *event* may transiently reach, and it has to sit well above the authored range
 * or it stops being a rail and becomes a governor — the same relationship, and
 * the same failure, as `MAX_EFFECTIVE_FRICTION` versus `MAX_FRICTION` after the
 * speed/drag split.
 *
 * That failure is exactly what shipped first here. A drawn row already sits near
 * ±1.2, so with the lane clamped at θ's ±2 every cell hit the rail at depth ~1
 * and every larger depth produced the *same* railed row: measured, depth 1 and
 * depth 4 turned the hit species by 9.1° and 8.7°, a four-fold knob worth
 * nothing. The precedent for a transient exceeding θ's bound is already in the
 * codebase and stated in the same terms — `MAX_BRIGHTNESS` bounds what the
 * sliders may author and the impulse *flash* multiplies past it on purpose.
 *
 * Finite for a reason, though: the shader's hard-core repulsion slope is 3.0, so
 * an attraction far past that starts winning against the core and clusters
 * collapse toward a point instead of reorganising.
 */
export const WIGGLE_LIMIT = 8;

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
      // Capped at the AUTHORED ceiling (`MAX_REACH_BRUTE`, half the torus), not
      // at the cell size and not at the grid lane's reach: the drawn outer
      // radius is allowed anywhere a hand-authored one is, and which runtime cap
      // binds is the pair-search mode's business. That cap stays in the shader,
      // where it can follow a mode or stencil change without a redraw.
      const maxR = Math.min(
        Math.max(uniform(rng, gen.rMax.lo, gen.rMax.hi), minR + MIN_BAND),
        MAX_REACH_BRUTE,
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

// ── the impulse wiggle ───────────────────────────────────────────────────────
//
// Roadmap phase 1 item 2. An event on a stem shoves that stem's species' *row*
// of the interaction matrix along a fixed random direction and lets it spring
// back with the envelope — so a hit changes what the species wants for a moment
// instead of only how bright or how many it is.
//
// The pattern is lifted from the author's prior particle-life sim (vuzic), where
// a per-channel audio value displaced each type's row along a column of a random
// orthonormal matrix. Two deliberate departures:
//
//   - **Primaries only, and only on their coupled cells.** vuzic had no
//     partition to protect; this sim's whole primary/secondary distinction is
//     that the uncoupled cells are exactly 0 (see `coupled()` and the note in
//     config.ts). A direction vector with support outside the row's coupled set
//     would make a kick briefly give the bass species opinions about three
//     accents it has no relationship with, which is the partition dissolving on
//     every downbeat.
//   - **No orthonormalisation across rows.** vuzic's Gram–Schmidt kept the
//     channels' displacements independent; here it is not available at all,
//     because subtracting row 0's projection from row 1 would leak row 0's
//     accent column into row 1 and break the support restriction above. Each
//     primary row is drawn and normalised on its own instead — with 4 rows in
//     5-dimensional supports the draws are near-orthogonal in practice, and the
//     property that actually matters (one species' hit does not push another
//     species' rules) holds exactly, because the rows are disjoint *outputs*.

/**
 * Unit direction vectors for the wiggle lane: K·K row-major, `[i·K + j]`.
 *
 * Non-zero only on primary rows (`i < PRIMARY_COUNT`) and only on that row's
 * coupled cells; every other entry is exactly 0, so a consumer can add
 * `depth · dirs[cell]` to any cell of the matrix without consulting the
 * partition itself.
 *
 * Pure, and keyed per *cell* like `seedMatrixBase` — the value in a cell is a
 * function of (seed, roll, cell) and not of how many cells were drawn before it.
 * `roll` is the workbench's reroll counter: same seed and roll, same directions,
 * on every machine and every reload; bump it and the world keeps its matrix but
 * gets a new set of things that hits do to it.
 */
export function seedWiggleDirs(seed: number, roll: number, k: number): Float32Array {
  const p = Math.min(PRIMARY_COUNT, k);
  const dirs = new Float32Array(k * k);
  // The roll is mixed into the domain separator rather than into the seed, so a
  // reroll cannot collide with the *matrix* stream of some other seed.
  const key = (KEY_WIGGLE ^ Math.imul(roll >>> 0, 0x9e37_79b9)) >>> 0;

  for (let i = 0; i < p; i++) {
    let sumSq = 0;
    for (let j = 0; j < k; j++) {
      if (!coupled(i, j, k)) continue;
      const cell = i * k + j;
      const g = gaussian(makeRng(hash3(seed >>> 0, cell >>> 0, key)));
      dirs[cell] = g;
      sumSq += g * g;
    }
    // Unit norm per row, so "depth" means the same displacement whatever K is
    // and whatever the row's support happens to be. A row of exact zeros (which
    // no finite draw produces, but the arithmetic must not divide by) simply
    // stays still.
    const norm = sumSq > 1e-12 ? 1 / Math.sqrt(sumSq) : 0;
    for (let j = 0; j < k; j++) dirs[i * k + j] = (dirs[i * k + j] as number) * norm;
  }
  return dirs;
}

/**
 * Write the attraction block the GPU should see this instant: θ's matrix plus
 * the live wiggle. Returns whether anything was actually displaced, which is
 * what lets the caller stop re-uploading once every envelope has decayed.
 *
 *   A'[i][j] = clamp( A[i][j] + depth · env_i · max|A| · dir[i][j], ±WIGGLE_LIMIT )
 *
 * `max|A|` is the same normaliser the far lane uses (`uploadFar`) and it is
 * there for the same reason: the matrix is drawn from the seed and its overall
 * scale moves with `matrixGen.sigma`, so without dividing the depth knob into
 * the matrix's own units, "wiggle 1" would be a nudge on one seed and a
 * detonation on the next. With it, the knob reads as "displace this row by N
 * times the strongest interaction in the world" — so depth 1 is a perturbation
 * the size of the matrix itself, and the top of the range is several times it.
 *
 * The rail is `WIGGLE_LIMIT`, not θ's `ATTRACTION_LIMIT`, and that is the whole
 * difference between a depth knob that works and the one that shipped first —
 * see the constant.
 */
export function wiggleAttraction(
  out: Float32Array,
  attraction: readonly number[],
  dirs: Float32Array | null,
  envelope: ArrayLike<number> | null,
  depth: number,
  k: number,
): boolean {
  const kk = k * k;
  let scale = 0;
  if (dirs && envelope && depth > 0) {
    for (let cell = 0; cell < kk; cell++) {
      scale = Math.max(scale, Math.abs(attraction[cell] ?? 0));
    }
    scale *= depth;
  }

  let moved = false;
  for (let i = 0; i < k; i++) {
    const env = scale > 0 ? Math.max((envelope as ArrayLike<number>)[i] ?? 0, 0) : 0;
    const gain = env * scale;
    if (gain <= 0) {
      for (let j = 0; j < k; j++) {
        const cell = i * k + j;
        out[cell] = attraction[cell] ?? 0;
      }
      continue;
    }
    // A secondary row (or any row whose direction vector is zero) lands here
    // only if a response was pointed at it; `dirs` is zero there, so the row
    // comes out unchanged and `moved` is honest about it.
    let rowMoved = false;
    for (let j = 0; j < k; j++) {
      const cell = i * k + j;
      const base = attraction[cell] ?? 0;
      const d = (dirs as Float32Array)[cell] as number;
      if (d === 0) {
        out[cell] = base;
        continue;
      }
      out[cell] = clamp(base + gain * d, -WIGGLE_LIMIT, WIGGLE_LIMIT);
      rowMoved = true;
    }
    moved ||= rowMoved;
  }
  return moved;
}

/** Uniform in [lo, hi], tolerant of an inverted band from a hand-dragged slider. */
function uniform(rng: () => number, lo: number, hi: number): number {
  const a = Math.min(lo, hi);
  const b = Math.max(lo, hi);
  return a + (b - a) * rng();
}
