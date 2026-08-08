/**
 * θ — the parameter vector, and the registry of what the music is allowed to move.
 *
 * A `Preset` is exactly the subset of `PhysarumConfig` that a human tunes: every
 * per-species field except its name, the K×K sense matrix, and the four global
 * knobs. Structural fields (speciesCount, maxAgents, gridScale, depositScale,
 * speed, paused, stemDrive) are *not* part of θ — they describe the machine, not
 * the art direction, and moving them would resize buffers mid-run.
 *
 * θ has two representations and they are interchangeable:
 *
 *   Preset  — JSON-shaped, mirrors PhysarumConfig field for field.
 *   vector  — Float64Array of length 18·K + K² + 4. What gets modulated, slewed
 *             and clamped.
 *
 * Colour is NOT in here (plan.md Revision 2). Modulating hue muddied the image
 * and made species impossible to track, so the palette is static and lives in
 * `PhysarumConfig.palette` / `ModulationConfig.palette`.
 *
 * **Revision 3.** The simplex is gone; the same slot table now feeds a seeded
 * random-projection modulator. Each slot therefore carries a `ModSpec` — group,
 * excursion half-range, personality jitter, additive vs multiplicative — or
 * `null`, meaning "the sliders own this, the music never touches it".
 *
 * **Revision 4.** `brightness` and `intensity` are still θ fields (they are
 * saved, slewed and seeded like everything else) but their `ModSpec` is now
 * `null`: light is driven by the stem-follow lane, not by the projections. The
 * `brightness` mod *group* is therefore gone entirely, along with its depth
 * slider — there is nothing left in it.
 */
import {
  MAX_EFFECTIVE_DEPOSIT,
  type AdaptiveTriple,
  type PhysarumConfig,
  type SpeciesConfig,
} from '../sim/physarum/config.ts';
// The vocabulary — slew classes, mod groups, ModSpec — is substrate-agnostic and
// lives in `modspec.ts` so a second simulation can bring its own registry
// without importing physarum's. It is re-exported here because this file was the
// original home and every importer already reaches for it by this name.
import { CLASS_FAST, CLASS_MEDIUM, CLASS_SLOW, type ModGroup, type ModSpec } from './modspec.ts';

export {
  CLASS_FAST,
  CLASS_MEDIUM,
  CLASS_SLOW,
  CLASS_NAMES,
  MOD_GROUPS,
} from './modspec.ts';
export type { SlewClass, ModGroup, ModSpec } from './modspec.ts';

/** A species' θ — `SpeciesConfig` minus its identity. */
export type SpeciesPreset = Omit<SpeciesConfig, 'name'>;

export interface Preset {
  species: SpeciesPreset[];
  /** K*K row-major, same convention as PhysarumConfig.matrix */
  matrix: number[];
  senseGain: number;
  exposure: number;
  gamma: number;
  stemGain: number;
}

/** Vector slots per species. */
export const PER_SPECIES = 18;
/** Globals appended after the matrix block. */
export const GLOBAL_SLOTS = 4;

const S_BRIGHTNESS = 0;
const S_INTENSITY = 1;
const S_DEPOSIT = 2;
const S_DECAY = 3;
const S_ALIVE = 4;
const S_DIFFUSE = 5;
const S_SENSOR_DIST = 6; // +3
const S_SENSOR_ANGLE = 9; // +3
const S_ROTATE = 12; // +3
const S_MOVE = 15; // +3

const G_SENSE_GAIN = 0;
const G_EXPOSURE = 1;
const G_GAMMA = 2;
const G_STEM_GAIN = 3;

interface Bound {
  min: number;
  max: number;
  cls: number;
  name: string;
  /** null = excluded from modulation; the panel sliders keep it. */
  mod: ModSpec | null;
}

const add = (
  group: ModGroup,
  lo: number,
  hi: number,
  half: number,
  jitter: number,
): ModSpec => ({ group, lo, hi, half, jitter, mult: false });

const mul = (
  group: ModGroup,
  lo: number,
  hi: number,
  half: number,
  jitter: number,
): ModSpec => ({ group, lo, hi, half, jitter, mult: true });

/**
 * p1 (constant term) and p2 (gain on intensity) modulate; **p3, the exponent,
 * does not**. p3 sits inside `p2·xᵖ³` where x ∈ [0,1), so moving it rescales the
 * whole adaptive curve non-linearly and small changes near x→0 are violent — it
 * is a shape-of-response knob, not a value, and plan.md Revision 3 explicitly
 * says keep exponents fixed. It stays on the sliders.
 */
const TRIPLE_BOUNDS = (
  label: string,
  cls: number,
  lo: number,
  hi: number,
  p1: ModSpec | null,
  p2: ModSpec | null,
  expLo = 0.1,
  expHi = 4,
): Bound[] => [
  { name: `${label}.p1`, cls, min: lo, max: hi, mod: p1 },
  { name: `${label}.p2`, cls, min: lo - (hi - lo), max: hi, mod: p2 },
  { name: `${label}.p3`, cls, min: expLo, max: expHi, mod: null },
];

/**
 * Slot table for one species. Classes follow plan.md Revision 3: brightness /
 * deposit / sensor angle track the 10 Hz signal; decay and alive-fraction are
 * section-scale; the rest sit in between because they move geometry the trail
 * field has to catch up with.
 *
 * Excursions are sized to be LEGIBLE (Revision 2 §3, Revision 3): with a
 * z-scored input and a unit projection, w·ẑ is roughly N(0,1), so at depth 1 the
 * typical |tanh| is ~0.55 and a slot spends most of its life at ±0.55·half from
 * its base. Every `half` below is therefore about twice the excursion you should
 * expect to see, not the excursion itself.
 */
const SPECIES_BOUNDS: Bound[] = [
  // Light — **excluded from modulation since Revision 4**. Revision 2 handed
  // brightness to the embedding; in use it flashed constantly, for reasons no
  // viewer could name, and the user's call was to take it back. Brightness is
  // now driven by `mapping/stemfollow.ts` — a species is as bright as its own
  // stem is loud — and `intensity` goes with it, because it is the same lane
  // (it multiplies brightness in the compositor) and modulating one half of a
  // product you have deliberately stopped modulating is just the flashing back
  // under another name. Both stay on the sliders in both modes.
  { name: 'brightness', cls: CLASS_FAST, min: 0, max: 2, mod: null },
  { name: 'intensity', cls: CLASS_FAST, min: 0, max: 4, mod: null },
  // Deposit is multiplicative: it is a rate, and ±0.7 in ln space is ×0.5…×2,
  // which reads the same whether the base landed at 0.4 or at 3.
  {
    name: 'deposit',
    cls: CLASS_FAST,
    min: 0,
    max: MAX_EFFECTIVE_DEPOSIT,
    mod: mul('population', 0.08, MAX_EFFECTIVE_DEPOSIT, 0.7, 0.4),
  },
  // Narrow and slow, on purpose: decay is the trail's memory constant. 0.88 vs
  // 0.95 is already a different world, and anything below ~0.85 erases structure
  // faster than agents can build it.
  { name: 'decay', cls: CLASS_SLOW, min: 0.8, max: 1, mod: add('decay', 0.86, 0.99, 0.035, 0.025) },
  {
    name: 'aliveFraction',
    cls: CLASS_SLOW,
    min: 0,
    max: 1,
    mod: add('population', 0.08, 1, 0.32, 0.22),
  },
  {
    name: 'diffuseCentre',
    cls: CLASS_MEDIUM,
    min: 0.111,
    max: 1,
    mod: add('structure', 0.111, 1, 0.3, 0.2),
  },
  ...TRIPLE_BOUNDS(
    'sensorDist',
    CLASS_MEDIUM,
    0,
    80,
    add('structure', 0.5, 50, 9, 6),
    add('structure', -18, 40, 9, 6),
  ),
  ...TRIPLE_BOUNDS(
    'sensorAngle',
    CLASS_FAST,
    -Math.PI,
    Math.PI,
    add('structure', -1.4, 1.4, 0.4, 0.3),
    add('structure', -1.4, 1.4, 0.35, 0.25),
  ),
  ...TRIPLE_BOUNDS(
    'rotate',
    CLASS_MEDIUM,
    -Math.PI,
    Math.PI,
    add('structure', -1.5, 1.5, 0.35, 0.25),
    add('structure', -1.5, 1.5, 0.5, 0.4),
  ),
  ...TRIPLE_BOUNDS(
    'moveDist',
    CLASS_MEDIUM,
    0,
    8,
    add('structure', 0.15, 5, 0.8, 0.6),
    add('structure', -2, 5, 0.7, 0.5),
  ),
];

/**
 * Off-diagonal M. The diagonal is special-cased in `modulationSlots` — self
 * attraction has to stay positive or the species stops forming a network at all,
 * which is a dead frame rather than a different look.
 */
const MATRIX_BOUND: Bound = {
  name: 'matrix',
  cls: CLASS_SLOW,
  min: -2,
  max: 2,
  mod: add('matrix', -1.7, 1.7, 0.75, 0.6),
};

const MATRIX_DIAGONAL_MOD: ModSpec = add('matrix', 0.3, 2, 0.5, 0.45);

/**
 * Globals, and the three deliberate exclusions:
 *
 * - **exposure / gamma** — phase 7 put an auto-exposure controller downstream of
 *   both. Modulating scene exposure makes the controller chase it and the image
 *   breathes for reasons that have nothing to do with the music; modulating
 *   display gamma is a monitor calibration knob, not art direction. The render
 *   chain's own `exposureEv` trim is the right place to push light, and
 *   per-species brightness already carries the modulated version of it.
 * - **stemGain** — it is the depth of the *other* input lane (stems → deposit).
 *   Letting the embedding modulate how loudly the stems speak means an
 *   instrument entering reads differently depending on where the track sits in
 *   embedding space, which is exactly the legibility Revision 2 §2 bought.
 *
 * `senseGain` is in, multiplicatively: it sets where the adaptive curves
 * saturate, so it changes the *character* of every species at once — a good
 * global to have moving, and safe in any range because it only rescales x∈[0,1).
 */
const GLOBAL_BOUNDS: Bound[] = [
  {
    name: 'senseGain',
    cls: CLASS_MEDIUM,
    min: 0.02,
    max: 4,
    mod: mul('structure', 0.02, 2, 0.7, 0.5),
  },
  { name: 'exposure', cls: CLASS_MEDIUM, min: 0.005, max: 1.5, mod: null },
  { name: 'gamma', cls: CLASS_MEDIUM, min: 1, max: 3, mod: null },
  { name: 'stemGain', cls: CLASS_FAST, min: 0, max: 6, mod: null },
];

if (SPECIES_BOUNDS.length !== PER_SPECIES) {
  throw new Error(`mapping: species slot table is ${SPECIES_BOUNDS.length}, expected ${PER_SPECIES}`);
}

export function vectorLength(k: number): number {
  return PER_SPECIES * k + k * k + GLOBAL_SLOTS;
}

export function matrixBase(k: number): number {
  return PER_SPECIES * k;
}

export function globalsBase(k: number): number {
  return PER_SPECIES * k + k * k;
}

/** Human-readable slot names, in vector order. Used by the workbench and by tests. */
export function fieldNames(k: number): string[] {
  const names: string[] = [];
  for (let s = 0; s < k; s++) {
    for (const b of SPECIES_BOUNDS) names.push(`species${s}.${b.name}`);
  }
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) names.push(`M[${i}][${j}]`);
  for (const b of GLOBAL_BOUNDS) names.push(b.name);
  return names;
}

/** Slew class per slot, as the small ints the limiter indexes with. */
export function fieldClasses(k: number): Uint8Array {
  const out = new Uint8Array(vectorLength(k));
  let o = 0;
  for (let s = 0; s < k; s++) {
    for (const b of SPECIES_BOUNDS) out[o++] = b.cls;
  }
  for (let i = 0; i < k * k; i++) out[o++] = MATRIX_BOUND.cls;
  for (const b of GLOBAL_BOUNDS) out[o++] = b.cls;
  return out;
}

function boundAt(slot: number, k: number): Bound {
  const mBase = matrixBase(k);
  if (slot < mBase) return SPECIES_BOUNDS[slot % PER_SPECIES] as Bound;
  if (slot < globalsBase(k)) return MATRIX_BOUND;
  return GLOBAL_BOUNDS[slot - globalsBase(k)] as Bound;
}

/**
 * The modulation registry, in vector order: one `ModSpec` per slot the music may
 * move, `null` for every slot it may not. This is the single source of truth the
 * modulator, the tests and the workbench readout all index by slot.
 */
export function modulationSlots(k: number): (ModSpec | null)[] {
  const out: (ModSpec | null)[] = new Array<ModSpec | null>(vectorLength(k));
  const mBase = matrixBase(k);
  const gBase = globalsBase(k);
  for (let i = 0; i < mBase; i++) out[i] = (SPECIES_BOUNDS[i % PER_SPECIES] as Bound).mod;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      out[mBase + i * k + j] = i === j ? MATRIX_DIAGONAL_MOD : MATRIX_BOUND.mod;
    }
  }
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

/** Clamp every slot into its authored range. Applied to generated/loaded presets, not per tick. */
export function clampVector(v: Float64Array, k: number): Float64Array {
  for (let i = 0; i < v.length; i++) {
    const b = boundAt(i, k);
    const x = v[i] as number;
    v[i] = x < b.min ? b.min : x > b.max ? b.max : x;
  }
  return v;
}

function triple(t: AdaptiveTriple): AdaptiveTriple {
  return { p1: t.p1, p2: t.p2, p3: t.p3 };
}

export function cloneSpeciesPreset(s: SpeciesPreset): SpeciesPreset {
  return {
    brightness: s.brightness,
    intensity: s.intensity,
    deposit: s.deposit,
    decay: s.decay,
    aliveFraction: s.aliveFraction,
    diffuseCentre: s.diffuseCentre,
    sensorDist: triple(s.sensorDist),
    sensorAngle: triple(s.sensorAngle),
    rotate: triple(s.rotate),
    moveDist: triple(s.moveDist),
  };
}

export function clonePreset(p: Preset): Preset {
  return {
    species: p.species.map(cloneSpeciesPreset),
    matrix: p.matrix.slice(),
    senseGain: p.senseGain,
    exposure: p.exposure,
    gamma: p.gamma,
    stemGain: p.stemGain,
  };
}

/** θ ← the live config. The snapshot the modulator takes of the shipped defaults. */
export function presetFromConfig(cfg: PhysarumConfig): Preset {
  return {
    species: cfg.species.slice(0, cfg.speciesCount).map(cloneSpeciesPreset),
    matrix: cfg.matrix.slice(0, cfg.speciesCount * cfg.speciesCount),
    senseGain: cfg.senseGain,
    exposure: cfg.exposure,
    gamma: cfg.gamma,
    stemGain: cfg.stemGain,
  };
}

/** The live config ← θ. Species names and structural fields are left untouched. */
export function applyPreset(cfg: PhysarumConfig, p: Preset): void {
  const k = cfg.speciesCount;
  for (let s = 0; s < k; s++) {
    const dst = cfg.species[s];
    const src = p.species[s];
    if (!dst || !src) continue;
    dst.brightness = src.brightness;
    dst.intensity = src.intensity;
    dst.deposit = src.deposit;
    dst.decay = src.decay;
    dst.aliveFraction = src.aliveFraction;
    dst.diffuseCentre = src.diffuseCentre;
    copyTriple(dst.sensorDist, src.sensorDist);
    copyTriple(dst.sensorAngle, src.sensorAngle);
    copyTriple(dst.rotate, src.rotate);
    copyTriple(dst.moveDist, src.moveDist);
  }
  for (let i = 0; i < k * k; i++) cfg.matrix[i] = p.matrix[i] ?? 0;
  cfg.senseGain = p.senseGain;
  cfg.exposure = p.exposure;
  cfg.gamma = p.gamma;
  cfg.stemGain = p.stemGain;
}

function copyTriple(dst: AdaptiveTriple, src: AdaptiveTriple): void {
  dst.p1 = src.p1;
  dst.p2 = src.p2;
  dst.p3 = src.p3;
}

export function presetToVector(p: Preset, k: number, out?: Float64Array): Float64Array {
  const v = out ?? new Float64Array(vectorLength(k));
  for (let s = 0; s < k; s++) {
    const sp = p.species[s];
    const o = s * PER_SPECIES;
    if (!sp) continue;
    v[o + S_BRIGHTNESS] = sp.brightness;
    v[o + S_INTENSITY] = sp.intensity;
    v[o + S_DEPOSIT] = sp.deposit;
    v[o + S_DECAY] = sp.decay;
    v[o + S_ALIVE] = sp.aliveFraction;
    v[o + S_DIFFUSE] = sp.diffuseCentre;
    writeTriple(v, o + S_SENSOR_DIST, sp.sensorDist);
    writeTriple(v, o + S_SENSOR_ANGLE, sp.sensorAngle);
    writeTriple(v, o + S_ROTATE, sp.rotate);
    writeTriple(v, o + S_MOVE, sp.moveDist);
  }
  const mBase = matrixBase(k);
  for (let i = 0; i < k * k; i++) v[mBase + i] = p.matrix[i] ?? 0;
  const gBase = globalsBase(k);
  v[gBase + G_SENSE_GAIN] = p.senseGain;
  v[gBase + G_EXPOSURE] = p.exposure;
  v[gBase + G_GAMMA] = p.gamma;
  v[gBase + G_STEM_GAIN] = p.stemGain;
  return v;
}

function writeTriple(v: Float64Array, o: number, t: AdaptiveTriple): void {
  v[o] = t.p1;
  v[o + 1] = t.p2;
  v[o + 2] = t.p3;
}

function readTriple(
  dst: AdaptiveTriple,
  v: ArrayLike<number>,
  o: number,
  mask?: Uint8Array,
): void {
  if (mask === undefined || mask[o] === 1) dst.p1 = (v[o] as number) ?? 0;
  if (mask === undefined || mask[o + 1] === 1) dst.p2 = (v[o + 1] as number) ?? 0;
  if (mask === undefined || mask[o + 2] === 1) dst.p3 = (v[o + 2] as number) ?? 1;
}

/**
 * Write a vector straight into the live config. The per-tick path: no intermediate
 * Preset object, no allocation.
 *
 * `mask`, when given, restricts the write to the slots it marks — that is how a
 * modulated run leaves the excluded slots (p3 exponents, exposure, gamma,
 * stemGain) under the panel's control instead of stamping a stale copy over
 * every edit.
 */
export function applyVector(
  cfg: PhysarumConfig,
  v: ArrayLike<number>,
  mask?: Uint8Array,
): void {
  const k = cfg.speciesCount;
  const on = (i: number): boolean => mask === undefined || mask[i] === 1;
  for (let s = 0; s < k; s++) {
    const dst = cfg.species[s];
    if (!dst) continue;
    const o = s * PER_SPECIES;
    if (on(o + S_BRIGHTNESS)) dst.brightness = v[o + S_BRIGHTNESS] as number;
    if (on(o + S_INTENSITY)) dst.intensity = v[o + S_INTENSITY] as number;
    if (on(o + S_DEPOSIT)) dst.deposit = v[o + S_DEPOSIT] as number;
    if (on(o + S_DECAY)) dst.decay = v[o + S_DECAY] as number;
    if (on(o + S_ALIVE)) dst.aliveFraction = v[o + S_ALIVE] as number;
    if (on(o + S_DIFFUSE)) dst.diffuseCentre = v[o + S_DIFFUSE] as number;
    readTriple(dst.sensorDist, v, o + S_SENSOR_DIST, mask);
    readTriple(dst.sensorAngle, v, o + S_SENSOR_ANGLE, mask);
    readTriple(dst.rotate, v, o + S_ROTATE, mask);
    readTriple(dst.moveDist, v, o + S_MOVE, mask);
  }
  const mBase = matrixBase(k);
  for (let i = 0; i < k * k; i++) {
    if (on(mBase + i)) cfg.matrix[i] = v[mBase + i] as number;
  }
  const gBase = globalsBase(k);
  if (on(gBase + G_SENSE_GAIN)) cfg.senseGain = v[gBase + G_SENSE_GAIN] as number;
  if (on(gBase + G_EXPOSURE)) cfg.exposure = v[gBase + G_EXPOSURE] as number;
  if (on(gBase + G_GAMMA)) cfg.gamma = v[gBase + G_GAMMA] as number;
  if (on(gBase + G_STEM_GAIN)) cfg.stemGain = v[gBase + G_STEM_GAIN] as number;
}

export function vectorToPreset(v: ArrayLike<number>, k: number): Preset {
  const species: SpeciesPreset[] = [];
  for (let s = 0; s < k; s++) {
    const o = s * PER_SPECIES;
    const sp: SpeciesPreset = {
      brightness: v[o + S_BRIGHTNESS] as number,
      intensity: v[o + S_INTENSITY] as number,
      deposit: v[o + S_DEPOSIT] as number,
      decay: v[o + S_DECAY] as number,
      aliveFraction: v[o + S_ALIVE] as number,
      diffuseCentre: v[o + S_DIFFUSE] as number,
      sensorDist: { p1: 0, p2: 0, p3: 1 },
      sensorAngle: { p1: 0, p2: 0, p3: 1 },
      rotate: { p1: 0, p2: 0, p3: 1 },
      moveDist: { p1: 0, p2: 0, p3: 1 },
    };
    readTriple(sp.sensorDist, v, o + S_SENSOR_DIST);
    readTriple(sp.sensorAngle, v, o + S_SENSOR_ANGLE);
    readTriple(sp.rotate, v, o + S_ROTATE);
    readTriple(sp.moveDist, v, o + S_MOVE);
    species.push(sp);
  }
  const mBase = matrixBase(k);
  const matrix: number[] = [];
  for (let i = 0; i < k * k; i++) matrix.push(v[mBase + i] as number);
  const gBase = globalsBase(k);
  return {
    species,
    matrix,
    senseGain: v[gBase + G_SENSE_GAIN] as number,
    exposure: v[gBase + G_EXPOSURE] as number,
    gamma: v[gBase + G_GAMMA] as number,
    stemGain: v[gBase + G_STEM_GAIN] as number,
  };
}
