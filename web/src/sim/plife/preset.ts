/**
 * θ for particle life — the parameter vector and the registry of what the music
 * is allowed to move.
 *
 * Structurally this mirrors `mapping/preset.ts` (physarum's registry) and
 * exports the same function family, because everything downstream — the
 * Modulator, the slew limiter, the workbench readout, persistence — indexes θ by
 * slot and does not care whose θ it is. What differs is only *which* slots exist.
 *
 * ## Vector order (load-bearing; persistence and the tests pin it)
 *
 *   [0                      , 9K            )  K species blocks, 9 slots each
 *   [9K                     , 9K +   K²     )  attraction, row-major [i·K+j]
 *   [9K +   K²              , 9K + 2K²      )  maxR,       row-major
 *   [9K + 2K²               , 9K + 3K²      )  minR,       row-major
 *   [9K + 3K²               , 9K + 3K² + 4  )  globals
 *
 * so `vectorLength(K) = 9K + 3K² + 4`, which is 268 at K = 8.
 *
 * Physarum's registry needed offset constants because its species block contains
 * nested `AdaptiveTriple`s; plife's is flat, so the species block is expressed as
 * a table of accessors instead — one row per slot, carrying its own bounds, its
 * `ModSpec` and a get/set pair. Adding a per-species knob is then one row and a
 * bump to `PER_SPECIES`, with no offset arithmetic to keep in sync.
 *
 * ## What is excluded from modulation, and why
 *
 * - **brightness / intensity** — the same call as physarum's Revision 4: light
 *   is driven by the stem-follow lane, not by the projections. Modulating either
 *   half of that product puts the constant flashing straight back.
 * - **stretch** — it is a *look* knob (how much a sprite smears with velocity),
 *   and modulating it makes the image's material change identity every few bars.
 * - **exposure / gamma** — there is an auto-exposure controller downstream of
 *   both. Modulating scene exposure makes the controller chase it.
 * - **every uncoupled attraction / maxR cell** — this is the important one. The
 *   primary/secondary partition (see `config.ts`) is only a partition as long as
 *   the zero cells *stay* zero; a modulated zero drifts, and within a minute the
 *   secondaries are just four more primaries. The panel can still move those
 *   cells by hand — "the sliders own this" is exactly what `mod: null` means —
 *   but the music cannot.
 * - **minR, all of it** — the hard-core radius is a stability parameter, not a
 *   look. Modulating it moves the singular point of the repulsion term while the
 *   integrator is running. Note that minR is excluded *differently* from the
 *   rest: it holds a `ModSpec` with `half: 0` and `jitter: 0` rather than a
 *   `null`, so its mask bit is 1 and the seeded draw in `genmatrix.ts` can reach
 *   the live config through `applyTheta` — while the music and the generic
 *   personality jitter still move it by exactly zero. See `MINR_BOUND`.
 */
import { CLASS_FAST, CLASS_MEDIUM, CLASS_SLOW, type ModGroup, type ModSpec } from '../../mapping/modspec.ts';
import {
  MIN_R_FLOOR,
  PRIMARY_COUNT,
  R_CAP,
  type PlifeConfig,
  type PlifeSpeciesConfig,
} from './config.ts';

/** Vector slots per species. */
export const PER_SPECIES = 9;
/** Globals appended after the three K² blocks. */
export const GLOBAL_SLOTS = 4;

const add = (group: ModGroup, lo: number, hi: number, half: number, jitter: number): ModSpec => ({
  group,
  lo,
  hi,
  half,
  jitter,
  mult: false,
});

const mul = (group: ModGroup, lo: number, hi: number, half: number, jitter: number): ModSpec => ({
  group,
  lo,
  hi,
  half,
  jitter,
  mult: true,
});

/** θ for one species — `PlifeSpeciesConfig` minus its identity. */
export type PlifeSpeciesPreset = Omit<PlifeSpeciesConfig, 'name' | 'role'>;

export interface PlifePreset {
  species: PlifeSpeciesPreset[];
  /** K·K row-major, same convention as PlifeConfig */
  attraction: number[];
  maxR: number[];
  minR: number[];
  forceGain: number;
  maxSpeed: number;
  exposure: number;
  gamma: number;
}

/**
 * One species slot. `min`/`max` are the *hard* bounds — what a loaded file may
 * contain — and are deliberately wider than the `ModSpec`'s `lo`/`hi`, which is
 * where the music is allowed to wander unsupervised.
 */
interface SpeciesSlot {
  name: string;
  cls: number;
  min: number;
  max: number;
  /** null = excluded from modulation; the panel sliders keep it. */
  mod: ModSpec | null;
  get(s: PlifeSpeciesPreset): number;
  set(s: PlifeSpeciesPreset, v: number): void;
}

/**
 * Slot table for one species, in vector order.
 *
 * Class assignment follows the same rule as physarum: motion faster than the
 * substrate's own relaxation time is mush. Here the relaxation time is set by
 * `friction` (τ ≈ 1/friction ≈ 0.3 s), so anything that changes how a particle
 * *moves* is MEDIUM at fastest; population and the matrix are section-scale
 * (SLOW) because a population change has to disperse before it reads; only the
 * pure-appearance slots are FAST.
 *
 * Excursion sizing, as in physarum: with a z-scored input and a unit projection,
 * w·ẑ is roughly N(0,1), so the typical |tanh| at depth 1 is ~0.55. Every `half`
 * below is therefore about twice the excursion you should expect to see.
 */
const SPECIES_SLOTS: readonly SpeciesSlot[] = [
  {
    name: 'brightness',
    cls: CLASS_FAST,
    min: 0,
    max: 2,
    mod: null,
    get: (s) => s.brightness,
    set: (s, v) => {
      s.brightness = v;
    },
  },
  {
    name: 'intensity',
    cls: CLASS_FAST,
    min: 0,
    max: 4,
    mod: null,
    get: (s) => s.intensity,
    set: (s, v) => {
      s.intensity = v;
    },
  },
  // Population is additive: "a tenth of the pool arrives" means the same thing
  // whether the species was at 0.12 or 0.5, whereas a multiplicative move would
  // be invisible on the accents and violent on the primaries.
  {
    name: 'aliveFraction',
    cls: CLASS_SLOW,
    min: 0,
    max: 1,
    mod: add('population', 0.02, 1, 0.3, 0.2),
    get: (s) => s.aliveFraction,
    set: (s, v) => {
      s.aliveFraction = v;
    },
  },
  // Reach, force and friction are all rates, so they move multiplicatively:
  // ±0.5 in ln space is ×0.6…×1.65 and reads the same at either end of the range.
  {
    name: 'radiusScale',
    cls: CLASS_MEDIUM,
    min: 0.25,
    max: 3,
    mod: mul('structure', 0.5, 2, 0.35, 0.3),
    get: (s) => s.radiusScale,
    set: (s, v) => {
      s.radiusScale = v;
    },
  },
  {
    name: 'forceScale',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 6,
    mod: mul('structure', 0.3, 3, 0.5, 0.35),
    get: (s) => s.forceScale,
    set: (s, v) => {
      s.forceScale = v;
    },
  },
  // Friction is this sim's `decay`: it is the memory constant of the velocity
  // field, and it is the only slot in the `decay` group.
  {
    name: 'friction',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 16,
    mod: mul('decay', 0.8, 8, 0.5, 0.4),
    get: (s) => s.friction,
    set: (s, v) => {
      s.friction = v;
    },
  },
  {
    name: 'wander',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 0.5,
    mod: mul('structure', 0.002, 0.15, 0.7, 0.5),
    get: (s) => s.wander,
    set: (s, v) => {
      s.wander = v;
    },
  },
  // Sprite size is FAST because it is pure appearance — nothing in the physics
  // reads it — but its excursion is small (0.25 in ln space, ×0.78…×1.28) since
  // total light on screen goes as size².
  {
    name: 'size',
    cls: CLASS_FAST,
    min: 0.0002,
    max: 0.02,
    mod: mul('structure', 0.0008, 0.006, 0.25, 0.3),
    get: (s) => s.size,
    set: (s, v) => {
      s.size = v;
    },
  },
  {
    name: 'stretch',
    cls: CLASS_FAST,
    min: 0,
    max: 8,
    mod: null,
    get: (s) => s.stretch,
    set: (s, v) => {
      s.stretch = v;
    },
  },
];

if (SPECIES_SLOTS.length !== PER_SPECIES) {
  throw new Error(`plife: species slot table is ${SPECIES_SLOTS.length}, expected ${PER_SPECIES}`);
}

/**
 * Is cell (i, j) of the interaction matrix part of the coupled set?
 *
 * This is the partition, stated once and used everywhere: the modulation mask
 * derives from it, the shipped defaults obey it, and the tests pin it. In words:
 *
 *   - the full primary block interacts (i < P and j < P);
 *   - secondary `P+n` couples to primary n, primary (n+1) mod P, and itself;
 *   - primary n couples to its own accent P+n;
 *   - nothing else.
 *
 * `k` is passed so the shape is a pure function of K rather than of a module
 * constant, which is what lets the tests evaluate it for any K.
 */
export function coupled(i: number, j: number, k: number): boolean {
  const p = Math.min(PRIMARY_COUNT, k);
  if (i < p && j < p) return true;
  if (i >= p) {
    const n = i - p;
    return j === n || j === (n + 1) % p || j === i;
  }
  // i < p here: the only extra coupling a primary has is to its own accent.
  return j === i + p;
}

interface Bound {
  name: string;
  cls: number;
  min: number;
  max: number;
  mod: ModSpec | null;
}

/** Attraction, for a coupled cell. Uncoupled cells share the bounds and get `mod: null`. */
const ATTRACTION_BOUND: Bound = {
  name: 'A',
  cls: CLASS_SLOW,
  min: -2,
  max: 2,
  mod: add('matrix', -1.2, 1.2, 0.45, 0.35),
};

/**
 * The outer radius of a pair's tent. Multiplicative and narrow: the interesting
 * range is under one octave (8 mm to 50 mm of world), and the effective value is
 * capped at R_CAP in the shader anyway because the neighbour grid's cell size is
 * derived from it.
 */
const MAXR_BOUND: Bound = {
  name: 'Rmax',
  cls: CLASS_SLOW,
  min: MIN_R_FLOOR,
  max: R_CAP,
  mod: mul('structure', 0.008, R_CAP, 0.2, 0.25),
};

/**
 * The hard-core radius. **In the registry but music-immobile.**
 *
 * This is the one slot that carries a `ModSpec` purely so the *mask* is 1, and
 * it needs a word of explanation because it looks like a mistake. minR is drawn
 * from the seed by `genmatrix.ts`, and a drawn value only reaches the live
 * config through `applyTheta`, which skips every masked-out slot. So mask 1 is
 * the delivery mechanism, and the spec is then rigged to move nothing:
 *
 *   - `half: 0`   — `computeTarget` builds `v = base · exp(half · e)`, and with
 *                   half 0 that is `base · e⁰ = base` for every possible ẑ. The
 *                   music cannot move it by any depth, gain or projection.
 *   - `jitter: 0` — `baseVector`'s personality draw is `base · exp(jitter · u)`,
 *                   so the generic jitter leaves the drawn value exactly alone.
 *                   The generator owns this slot outright.
 *
 * The physics reason is unchanged from before: the hard-core radius is a
 * stability parameter, not a look. Moving the singular point of the repulsion
 * term while the integrator is running is how you get a field that explodes.
 */
const MINR_BOUND: Bound = {
  name: 'Rmin',
  cls: CLASS_SLOW,
  min: MIN_R_FLOOR,
  max: R_CAP,
  mod: { group: 'structure', lo: MIN_R_FLOOR, hi: R_CAP, half: 0, jitter: 0, mult: true },
};

/**
 * The `ModSpec` every coupled attraction cell shares, exported so the seeded
 * matrix generator can clamp its draw to exactly the range the music is allowed
 * to wander inside rather than restating those numbers a second time. Non-null
 * by construction — `ATTRACTION_BOUND.mod` is a literal above.
 */
export const ATTRACTION_MOD: ModSpec = ATTRACTION_BOUND.mod as ModSpec;

/**
 * Globals. `forceGain` and `maxSpeed` are the two that change the *character* of
 * every species at once — how hard the field pushes and how fast anything is
 * allowed to travel — which makes them the right pair to have moving on the
 * music. Both are multiplicative and both are bounded well inside the range
 * where the explicit-Euler integrator is stable at dt = 1/60.
 */
const GLOBAL_BOUNDS: readonly Bound[] = [
  {
    name: 'forceGain',
    cls: CLASS_MEDIUM,
    min: 0,
    max: 8,
    mod: mul('structure', 0.2, 4, 0.4, 0.3),
  },
  {
    name: 'maxSpeed',
    cls: CLASS_MEDIUM,
    min: 0.005,
    max: 2,
    mod: mul('structure', 0.05, 1, 0.3, 0.2),
  },
  { name: 'exposure', cls: CLASS_MEDIUM, min: 0.05, max: 4, mod: null },
  { name: 'gamma', cls: CLASS_MEDIUM, min: 1, max: 3, mod: null },
];

if (GLOBAL_BOUNDS.length !== GLOBAL_SLOTS) {
  throw new Error(`plife: global slot table is ${GLOBAL_BOUNDS.length}, expected ${GLOBAL_SLOTS}`);
}

// ── layout ────────────────────────────────────────────────────────────────────

export function vectorLength(k: number): number {
  return PER_SPECIES * k + 3 * k * k + GLOBAL_SLOTS;
}

/** Base of the attraction block. Named `matrixBase` to match physarum's registry. */
export function matrixBase(k: number): number {
  return PER_SPECIES * k;
}

export function maxRBase(k: number): number {
  return PER_SPECIES * k + k * k;
}

export function minRBase(k: number): number {
  return PER_SPECIES * k + 2 * k * k;
}

export function globalsBase(k: number): number {
  return PER_SPECIES * k + 3 * k * k;
}

/** Human-readable slot names, in vector order. Used by the workbench and by tests. */
export function fieldNames(k: number): string[] {
  const names: string[] = [];
  for (let s = 0; s < k; s++) {
    for (const b of SPECIES_SLOTS) names.push(`species${s}.${b.name}`);
  }
  for (const label of ['A', 'Rmax', 'Rmin']) {
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) names.push(`${label}[${i}][${j}]`);
  }
  for (const b of GLOBAL_BOUNDS) names.push(b.name);
  return names;
}

/** Slew class per slot, as the small ints the limiter indexes with. */
export function fieldClasses(k: number): Uint8Array {
  const out = new Uint8Array(vectorLength(k));
  let o = 0;
  for (let s = 0; s < k; s++) {
    for (const b of SPECIES_SLOTS) out[o++] = b.cls;
  }
  for (const b of [ATTRACTION_BOUND, MAXR_BOUND, MINR_BOUND]) {
    for (let i = 0; i < k * k; i++) out[o++] = b.cls;
  }
  for (const b of GLOBAL_BOUNDS) out[o++] = b.cls;
  return out;
}

/** Which of the three K² blocks (and which cell) a slot falls in, or null. */
function blockAt(slot: number, k: number): { bound: Bound; i: number; j: number } | null {
  const kk = k * k;
  const mBase = matrixBase(k);
  if (slot < mBase || slot >= globalsBase(k)) return null;
  const off = slot - mBase;
  const which = Math.floor(off / kk);
  const cell = off - which * kk;
  const bound = which === 0 ? ATTRACTION_BOUND : which === 1 ? MAXR_BOUND : MINR_BOUND;
  return { bound, i: Math.floor(cell / k), j: cell % k };
}

function boundAt(slot: number, k: number): { min: number; max: number } {
  const mBase = matrixBase(k);
  if (slot < mBase) return SPECIES_SLOTS[slot % PER_SPECIES] as SpeciesSlot;
  const gBase = globalsBase(k);
  if (slot >= gBase) return GLOBAL_BOUNDS[slot - gBase] as Bound;
  return (blockAt(slot, k) as { bound: Bound }).bound;
}

/**
 * The modulation registry, in vector order: one `ModSpec` per slot the music may
 * move, `null` for every slot it may not. Single source of truth for the
 * modulator, the tests and the workbench readout.
 */
export function modulationSlots(k: number): (ModSpec | null)[] {
  const out: (ModSpec | null)[] = new Array<ModSpec | null>(vectorLength(k)).fill(null);
  const mBase = matrixBase(k);
  for (let i = 0; i < mBase; i++) out[i] = (SPECIES_SLOTS[i % PER_SPECIES] as SpeciesSlot).mod;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const cell = i * k + j;
      const live = coupled(i, j, k);
      out[mBase + cell] = live ? ATTRACTION_BOUND.mod : null;
      out[maxRBase(k) + cell] = live ? MAXR_BOUND.mod : null;
      out[minRBase(k) + cell] = MINR_BOUND.mod;
    }
  }
  const gBase = globalsBase(k);
  for (let i = 0; i < GLOBAL_SLOTS; i++) out[gBase + i] = (GLOBAL_BOUNDS[i] as Bound).mod;
  return out;
}

/** 1 where a slot is modulated, 0 where the sliders own it. */
export function modulationMask(k: number): Uint8Array {
  const slots = modulationSlots(k);
  const out = new Uint8Array(slots.length);
  for (let i = 0; i < slots.length; i++) out[i] = slots[i] ? 1 : 0;
  return out;
}

/** Clamp every slot into its authored hard range. Applied to loaded/generated presets. */
export function clampVector(v: Float64Array, k: number): Float64Array {
  for (let i = 0; i < v.length; i++) {
    const b = boundAt(i, k);
    const x = v[i] as number;
    v[i] = x < b.min ? b.min : x > b.max ? b.max : x;
  }
  return v;
}

// ── Preset ⇄ config ⇄ vector ──────────────────────────────────────────────────

export function cloneSpeciesPreset(s: PlifeSpeciesPreset): PlifeSpeciesPreset {
  return {
    brightness: s.brightness,
    intensity: s.intensity,
    aliveFraction: s.aliveFraction,
    radiusScale: s.radiusScale,
    forceScale: s.forceScale,
    friction: s.friction,
    wander: s.wander,
    size: s.size,
    stretch: s.stretch,
  };
}

export function clonePreset(p: PlifePreset): PlifePreset {
  return {
    species: p.species.map(cloneSpeciesPreset),
    attraction: p.attraction.slice(),
    maxR: p.maxR.slice(),
    minR: p.minR.slice(),
    forceGain: p.forceGain,
    maxSpeed: p.maxSpeed,
    exposure: p.exposure,
    gamma: p.gamma,
  };
}

/** θ ← the live config. The snapshot the modulator takes of the shipped defaults. */
export function presetFromConfig(cfg: PlifeConfig): PlifePreset {
  const kk = cfg.speciesCount * cfg.speciesCount;
  return {
    species: cfg.species.slice(0, cfg.speciesCount).map(cloneSpeciesPreset),
    attraction: cfg.attraction.slice(0, kk),
    maxR: cfg.maxR.slice(0, kk),
    minR: cfg.minR.slice(0, kk),
    forceGain: cfg.forceGain,
    maxSpeed: cfg.maxSpeed,
    exposure: cfg.exposure,
    gamma: cfg.gamma,
  };
}

/** The live config ← θ. Species names, roles and structural fields are untouched. */
export function applyPreset(cfg: PlifeConfig, p: PlifePreset): void {
  const k = cfg.speciesCount;
  for (let s = 0; s < k; s++) {
    const dst = cfg.species[s];
    const src = p.species[s];
    if (!dst || !src) continue;
    for (const slot of SPECIES_SLOTS) slot.set(dst, slot.get(src));
  }
  for (let i = 0; i < k * k; i++) {
    cfg.attraction[i] = p.attraction[i] ?? 0;
    cfg.maxR[i] = p.maxR[i] ?? 0;
    cfg.minR[i] = p.minR[i] ?? 0;
  }
  cfg.forceGain = p.forceGain;
  cfg.maxSpeed = p.maxSpeed;
  cfg.exposure = p.exposure;
  cfg.gamma = p.gamma;
}

export function presetToVector(p: PlifePreset, k: number, out?: Float64Array): Float64Array {
  const v = out ?? new Float64Array(vectorLength(k));
  for (let s = 0; s < k; s++) {
    const sp = p.species[s];
    if (!sp) continue;
    const o = s * PER_SPECIES;
    for (let n = 0; n < PER_SPECIES; n++) v[o + n] = (SPECIES_SLOTS[n] as SpeciesSlot).get(sp);
  }
  const kk = k * k;
  const mBase = matrixBase(k);
  const xBase = maxRBase(k);
  const nBase = minRBase(k);
  for (let i = 0; i < kk; i++) {
    v[mBase + i] = p.attraction[i] ?? 0;
    v[xBase + i] = p.maxR[i] ?? 0;
    v[nBase + i] = p.minR[i] ?? 0;
  }
  const gBase = globalsBase(k);
  v[gBase + 0] = p.forceGain;
  v[gBase + 1] = p.maxSpeed;
  v[gBase + 2] = p.exposure;
  v[gBase + 3] = p.gamma;
  return v;
}

/**
 * Write a vector straight into the live config. The per-tick path: no
 * intermediate Preset object, no allocation.
 *
 * `mask`, when given, restricts the write to the slots it marks — that is how a
 * modulated run leaves the excluded slots (brightness, stretch, exposure, gamma,
 * and every uncoupled matrix cell) under the panel's control instead of stamping
 * a stale copy over every edit. minR is *not* in that list any more: it is in the
 * mask so the seeded draw can land, and immobile by spec instead — see
 * `MINR_BOUND`.
 */
export function applyVector(cfg: PlifeConfig, v: ArrayLike<number>, mask?: Uint8Array): void {
  const k = cfg.speciesCount;
  const on = (i: number): boolean => mask === undefined || mask[i] === 1;
  for (let s = 0; s < k; s++) {
    const dst = cfg.species[s];
    if (!dst) continue;
    const o = s * PER_SPECIES;
    for (let n = 0; n < PER_SPECIES; n++) {
      if (!on(o + n)) continue;
      (SPECIES_SLOTS[n] as SpeciesSlot).set(dst, v[o + n] as number);
    }
  }
  const kk = k * k;
  const mBase = matrixBase(k);
  const xBase = maxRBase(k);
  const nBase = minRBase(k);
  for (let i = 0; i < kk; i++) {
    if (on(mBase + i)) cfg.attraction[i] = v[mBase + i] as number;
    if (on(xBase + i)) cfg.maxR[i] = v[xBase + i] as number;
    if (on(nBase + i)) cfg.minR[i] = v[nBase + i] as number;
  }
  const gBase = globalsBase(k);
  if (on(gBase + 0)) cfg.forceGain = v[gBase + 0] as number;
  if (on(gBase + 1)) cfg.maxSpeed = v[gBase + 1] as number;
  if (on(gBase + 2)) cfg.exposure = v[gBase + 2] as number;
  if (on(gBase + 3)) cfg.gamma = v[gBase + 3] as number;
}

export function vectorToPreset(v: ArrayLike<number>, k: number): PlifePreset {
  const species: PlifeSpeciesPreset[] = [];
  for (let s = 0; s < k; s++) {
    const o = s * PER_SPECIES;
    const sp: PlifeSpeciesPreset = {
      brightness: 1,
      intensity: 1,
      aliveFraction: 0,
      radiusScale: 1,
      forceScale: 1,
      friction: 1,
      wander: 0,
      size: 0.002,
      stretch: 0,
    };
    for (let n = 0; n < PER_SPECIES; n++) {
      (SPECIES_SLOTS[n] as SpeciesSlot).set(sp, v[o + n] as number);
    }
    species.push(sp);
  }
  const kk = k * k;
  const mBase = matrixBase(k);
  const xBase = maxRBase(k);
  const nBase = minRBase(k);
  const attraction: number[] = [];
  const maxR: number[] = [];
  const minR: number[] = [];
  for (let i = 0; i < kk; i++) {
    attraction.push(v[mBase + i] as number);
    maxR.push(v[xBase + i] as number);
    minR.push(v[nBase + i] as number);
  }
  const gBase = globalsBase(k);
  return {
    species,
    attraction,
    maxR,
    minR,
    forceGain: v[gBase + 0] as number,
    maxSpeed: v[gBase + 1] as number,
    exposure: v[gBase + 2] as number,
    gamma: v[gBase + 3] as number,
  };
}
