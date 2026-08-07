/**
 * θ — the art-directable parameter vector.
 *
 * A `Preset` is exactly the subset of `PhysarumConfig` that a human tunes: every
 * per-species field except its name, the K×K sense matrix, and the four global
 * knobs. Structural fields (speciesCount, maxAgents, gridScale, depositScale,
 * speed, paused, stemDrive) are *not* part of θ — they describe the machine, not
 * the art direction, and blending them would resize buffers mid-run.
 *
 * θ has two representations and they are interchangeable:
 *
 *   Preset  — JSON-shaped, mirrors PhysarumConfig field for field. What gets
 *             saved, loaded, hand-edited and captured.
 *   vector  — Float64Array of length 18·K + K² + 4. What gets blended, slewed
 *             and clamped.
 *
 * Colour is NOT in here (plan.md Revision 2). Blending hue between anchors
 * muddied the image and made species impossible to track, so the palette is
 * static and lives in `PhysarumConfig.palette` / `MappingConfig.palette`. What
 * θ carries instead is a per-species `brightness` in the fast slew class: light
 * responds to the music, hue does not.
 *
 * Everything downstream (simplex blend, slew limiter, convex-hull test) works on
 * the vector; nothing downstream needs to know a field's name.
 */
import {
  MAX_EFFECTIVE_DEPOSIT,
  type AdaptiveTriple,
  type PhysarumConfig,
  type SpeciesConfig,
} from '../sim/physarum/config.ts';

/** Per-parameter timescale class. Motion faster than the sim's relaxation time is mush. */
export type SlewClass = 'fast' | 'medium' | 'slow';

export const CLASS_FAST = 0;
export const CLASS_MEDIUM = 1;
export const CLASS_SLOW = 2;
export const CLASS_NAMES: readonly SlewClass[] = ['fast', 'medium', 'slow'];

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
}

const TRIPLE_BOUNDS = (
  label: string,
  cls: number,
  lo: number,
  hi: number,
  expLo = 0.1,
  expHi = 4,
): Bound[] => [
  { name: `${label}.p1`, cls, min: lo, max: hi },
  { name: `${label}.p2`, cls, min: lo - (hi - lo), max: hi },
  { name: `${label}.p3`, cls, min: expLo, max: expHi },
];

/**
 * Slot table for one species. Classes follow plan.md: deposit / sensor angle /
 * brightness track 10 Hz; decay and alive-fraction are section-scale; the rest
 * sit in between because they move geometry the trail field has to catch up with.
 */
const SPECIES_BOUNDS: Bound[] = [
  { name: 'brightness', cls: CLASS_FAST, min: 0, max: 2 },
  { name: 'intensity', cls: CLASS_FAST, min: 0, max: 4 },
  { name: 'deposit', cls: CLASS_FAST, min: 0, max: MAX_EFFECTIVE_DEPOSIT },
  { name: 'decay', cls: CLASS_SLOW, min: 0.8, max: 1 },
  { name: 'aliveFraction', cls: CLASS_SLOW, min: 0, max: 1 },
  { name: 'diffuseCentre', cls: CLASS_MEDIUM, min: 0.111, max: 1 },
  ...TRIPLE_BOUNDS('sensorDist', CLASS_MEDIUM, 0, 80),
  ...TRIPLE_BOUNDS('sensorAngle', CLASS_FAST, -Math.PI, Math.PI),
  ...TRIPLE_BOUNDS('rotate', CLASS_MEDIUM, -Math.PI, Math.PI),
  ...TRIPLE_BOUNDS('moveDist', CLASS_MEDIUM, 0, 8),
];

const MATRIX_BOUND: Bound = { name: 'matrix', cls: CLASS_SLOW, min: -2, max: 2 };

const GLOBAL_BOUNDS: Bound[] = [
  { name: 'senseGain', cls: CLASS_MEDIUM, min: 0.02, max: 4 },
  { name: 'exposure', cls: CLASS_MEDIUM, min: 0.005, max: 1.5 },
  { name: 'gamma', cls: CLASS_MEDIUM, min: 1, max: 3 },
  { name: 'stemGain', cls: CLASS_FAST, min: 0, max: 6 },
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

/** θ ← the live config. This is "capture current params into this anchor's preset". */
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

function readTriple(dst: AdaptiveTriple, v: ArrayLike<number>, o: number): void {
  dst.p1 = (v[o] as number) ?? 0;
  dst.p2 = (v[o + 1] as number) ?? 0;
  dst.p3 = (v[o + 2] as number) ?? 1;
}

/**
 * Write a vector straight into the live config. The per-tick path in mapped mode:
 * no intermediate Preset object, no allocation.
 */
export function applyVector(cfg: PhysarumConfig, v: ArrayLike<number>): void {
  const k = cfg.speciesCount;
  for (let s = 0; s < k; s++) {
    const dst = cfg.species[s];
    if (!dst) continue;
    const o = s * PER_SPECIES;
    dst.brightness = v[o + S_BRIGHTNESS] as number;
    dst.intensity = v[o + S_INTENSITY] as number;
    dst.deposit = v[o + S_DEPOSIT] as number;
    dst.decay = v[o + S_DECAY] as number;
    dst.aliveFraction = v[o + S_ALIVE] as number;
    dst.diffuseCentre = v[o + S_DIFFUSE] as number;
    readTriple(dst.sensorDist, v, o + S_SENSOR_DIST);
    readTriple(dst.sensorAngle, v, o + S_SENSOR_ANGLE);
    readTriple(dst.rotate, v, o + S_ROTATE);
    readTriple(dst.moveDist, v, o + S_MOVE);
  }
  const mBase = matrixBase(k);
  for (let i = 0; i < k * k; i++) cfg.matrix[i] = v[mBase + i] as number;
  const gBase = globalsBase(k);
  cfg.senseGain = v[gBase + G_SENSE_GAIN] as number;
  cfg.exposure = v[gBase + G_EXPOSURE] as number;
  cfg.gamma = v[gBase + G_GAMMA] as number;
  cfg.stemGain = v[gBase + G_STEM_GAIN] as number;
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

/**
 * θ = Σ wₘ θₘ. Weights are assumed non-negative and summing to 1 (the simplex
 * guarantees it), which is exactly what makes the result a convex combination:
 * every slot lands inside the min/max of that slot across the presets.
 */
export function blendVectors(
  vectors: readonly Float64Array[],
  weights: ArrayLike<number>,
  out: Float64Array,
): Float64Array {
  out.fill(0);
  for (let m = 0; m < vectors.length; m++) {
    const w = weights[m] ?? 0;
    if (w === 0) continue;
    const v = vectors[m] as Float64Array;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) + w * (v[i] as number);
  }
  return out;
}
