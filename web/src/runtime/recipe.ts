/**
 * Immutable state handed from the authoring surface to a headless renderer.
 *
 * This module deliberately has no DOM or localStorage dependency. The browser,
 * a Node worker, and eventually the API validator can all consume the same
 * versioned contract.
 */
import {
  MAX_MASTERING_PEAK_NITS,
  MAX_PAPER_WHITE_NITS,
  MIN_MASTERING_PEAK_NITS,
  MIN_PAPER_WHITE_NITS,
} from '../export/hdr.ts';
import { requiredEncoder } from '../export/profiles.ts';
import { MODULATION_VERSION, type ModulationConfig } from '../mapping/types.ts';
import type { PhysarumConfig } from '../sim/physarum/config.ts';
import type { PlifeConfig } from '../sim/plife/config.ts';
import type { ImpulseConfig } from '../sim/impulses.ts';
import {
  MAX_BLOOM_LEVELS,
  TONEMAPS,
  type RenderConfig,
} from '../sim/render/config.ts';
import type { VizFxConfig } from '../sim/vizfx/config.ts';
import { isVizFxId } from '../sim/vizfx/ids.ts';
import { EVENT_KINDS } from '../timeline/types.ts';

/**
 * 6: particle life grew a `luma` block — per-particle brightness dynamic range
 * driven by speed, roadmap phase 1 item 3. See `liftExportRecipe` for why a v5
 * recipe lifts to `depth: 0` rather than to the shipped default.
 *
 * 5: the impulse lane grew a `wiggle` depth per event kind (the matrix-row
 * perturbation, roadmap phase 1 item 2). See `liftExportRecipe` for why a v4
 * recipe lifts to `wiggle: 0` rather than to the shipped default.
 */
export const EXPORT_RECIPE_VERSION = 6 as const;

/**
 * Additive within recipe v3 for the SDR debug profiles: a v3 recipe naming one
 * means exactly what it always did, so no existing capture, request, or sidecar
 * is invalidated.
 *
 * The HDR10 profiles were briefly named `hevc-hdr10-*` and are now `av1-hdr10-*`.
 * That rename is deliberately *not* aliased. The HEVC ids existed for less than a
 * day, described an encoder this pipeline no longer uses, and a recipe is a
 * reproduction contract — silently re-pointing an old id at a different codec
 * would make it lie. An old recipe is rejected by `$.output.profile` instead.
 */
export const EXPORT_PROFILES = [
  'av1-hdr10-2160p120',
  'av1-hdr10-1080p120',
  'av1-sdr-debug-2160p120',
  'av1-sdr-debug-1080p120',
] as const;
export type ExportProfile = (typeof EXPORT_PROFILES)[number];

export const EXPORT_ENCODERS = ['av1_nvenc'] as const;
export type ExportEncoder = (typeof EXPORT_ENCODERS)[number];

/** A conservative request bound; the worker may impose a lower device bound. */
export const MAX_RECIPE_PARTICLE_BUDGET = 16_777_216;
export const MAX_RECIPE_JSON_CHARS = 8 * 1024 * 1024;

/** Render is stored once at recipe.render rather than repeated by reference. */
export type SimulationBaseConfig =
  | Omit<PhysarumConfig, 'render'>
  | Omit<PlifeConfig, 'render'>
  | Omit<VizFxConfig, 'render'>;

/** ModulationConfig also shares render in the live app; the recipe does not. */
/**
 * `impulses` joins `render` in being omitted: both are one live object in the
 * browser but appear once at the recipe's top level, and duplicating either
 * inside the modulation block would create a second copy the validator would
 * then have to prove equal to the first (the problem `palettesEqual` already
 * exists to solve). One encoding, at `$.impulses`.
 */
export type RecipeModulationConfig = Omit<ModulationConfig, 'render' | 'impulses'>;

export interface ExportRecipeV4 {
  version: typeof EXPORT_RECIPE_VERSION;
  rendererBuild: string;
  track: {
    id: string;
    contentVersion: string;
  };
  /** Concrete ModTarget.simId (physarum, plife, or one concrete VizFX visual). */
  sim: string;
  seed: number;
  seedPinned: boolean;
  simulation: SimulationBaseConfig;
  modulation: RecipeModulationConfig;
  /** Author-owned modulation centre, never the current music-excursed theta. */
  modulationBase: number[];
  impulses: ImpulseConfig;
  render: RenderConfig;
  /** Authored fixed export cap. Preview adaptive-quality state is not consulted. */
  particleBudget: number;
  /** A recipe exports one concrete visual and never reads browser auto-advance state. */
  presentation: {
    mode: 'single';
    autoAdvance: false;
  };
  output: {
    profile: ExportProfile;
    encoder: ExportEncoder;
    /** Diffuse white level used by the scene-linear to PQ transform. */
    paperWhiteNits: number;
    /** Peak luminance used by highlight mapping and static mastering metadata. */
    masteringPeakNits: number;
  };
}

export type ExportRecipe = ExportRecipeV4;

/**
 * Exported so `./preset.ts` can derive its own key list by subtraction rather
 * than hand-keeping a second one that would silently drift out of step.
 */
export const TOP_LEVEL_KEYS = [
  'version',
  'rendererBuild',
  'track',
  'sim',
  'seed',
  'seedPinned',
  'simulation',
  'modulation',
  'modulationBase',
  'impulses',
  'render',
  'particleBudget',
  'presentation',
  'output',
] as const;

const MAX_STRING_CHARS = 4096;
const MAX_KEY_CHARS = 128;
export const MAX_ARRAY_LENGTH = 131_072;
const MAX_OBJECT_KEYS = 4096;
const MAX_DEPTH = 32;
const MAX_NODES = 250_000;
export const MAX_ABS_CONFIG_NUMBER = 1e12;
const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

/**
 * Throws a bare `"$.path complaint"`. The document kind it belongs to is added
 * by `withLabel` at each public entry point, so the same block validators can
 * report against a recipe *or* a preset string without either one borrowing the
 * other's name in the message a person actually reads.
 */
export function fail(path: string, message: string): never {
  throw new Error(`${path} ${message}`);
}

const RECIPE_LABEL = 'export recipe: ';

/**
 * Prefix a validation failure with the document kind, once.
 *
 * Idempotent by inspection, because these nest: `serializeExportRecipe` calls
 * `validateExportRecipe`, and a preset's parser calls block validators that
 * throw the shared bare form.
 */
export function withLabel<T>(label: string, run: () => T): T {
  try {
    return run();
  } catch (error) {
    const message = (error as Error).message;
    if (message.startsWith(label)) throw error;
    throw new Error(`${label}${message}`);
  }
}

export function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    fail(path, 'must be an object');
  }
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) fail(path, 'must be a plain object');
  return value as Record<string, unknown>;
}

export function keysExactly(value: Record<string, unknown>, allowed: readonly string[], path: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) fail(`${path}.${key}`, 'is not supported by this schema version');
  }
  for (const key of allowed) {
    if (!Object.hasOwn(value, key)) fail(`${path}.${key}`, 'is required');
  }
}

export function keysRequired(
  value: Record<string, unknown>,
  allowed: readonly string[],
  required: readonly string[],
  path: string,
): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) fail(`${path}.${key}`, 'is not supported by this schema version');
  }
  for (const key of required) {
    if (!Object.hasOwn(value, key)) fail(`${path}.${key}`, 'is required');
  }
}

export function boundedString(value: unknown, path: string, max = MAX_STRING_CHARS): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > max) {
    fail(path, `must be a non-empty string of at most ${max} characters`);
  }
  return value;
}

export function identifier(value: unknown, path: string): string {
  const text = boundedString(value, path, 128);
  if (!ID.test(text)) fail(path, 'contains unsupported characters');
  return text;
}

export function finiteNumber(value: unknown, path: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    fail(path, `must be a finite number in ${min}..${max}`);
  }
  return value;
}

export function integer(value: unknown, path: string, min: number, max: number): number {
  const n = finiteNumber(value, path, min, max);
  if (!Number.isSafeInteger(n)) fail(path, 'must be a safe integer');
  return n;
}

export function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'must be a boolean');
  return value;
}

interface WalkBudget {
  nodes: number;
}

/** Reject values JSON.stringify would silently drop, coerce, or expand without bound. */
function jsonValue(
  value: unknown,
  path: string,
  budget: WalkBudget,
  ancestors: Set<object>,
  depth: number,
): void {
  budget.nodes++;
  if (budget.nodes > MAX_NODES) fail(path, `exceeds the ${MAX_NODES}-value limit`);
  if (depth > MAX_DEPTH) fail(path, `exceeds the maximum nesting depth of ${MAX_DEPTH}`);

  if (value === null || typeof value === 'boolean') return;
  if (typeof value === 'string') {
    if (value.length > MAX_STRING_CHARS) fail(path, `exceeds ${MAX_STRING_CHARS} characters`);
    return;
  }
  if (typeof value === 'number') {
    finiteNumber(value, path, -MAX_ABS_CONFIG_NUMBER, MAX_ABS_CONFIG_NUMBER);
    return;
  }
  if (typeof value !== 'object') fail(path, 'is not a JSON value');
  if (ancestors.has(value)) fail(path, 'contains a cycle');
  ancestors.add(value);

  if (Array.isArray(value)) {
    if (value.length > MAX_ARRAY_LENGTH) fail(path, `exceeds ${MAX_ARRAY_LENGTH} entries`);
    for (let i = 0; i < value.length; i++) {
      if (!Object.hasOwn(value, i)) fail(`${path}[${i}]`, 'is a sparse array entry');
      jsonValue(value[i], `${path}[${i}]`, budget, ancestors, depth + 1);
    }
  } else {
    const record = object(value, path);
    const names = Object.keys(record);
    if (names.length > MAX_OBJECT_KEYS) fail(path, `exceeds ${MAX_OBJECT_KEYS} properties`);
    if (Object.getOwnPropertySymbols(record).length > 0) fail(path, 'contains symbol properties');
    for (const key of names) {
      if (key.length === 0 || key.length > MAX_KEY_CHARS) fail(path, 'contains an invalid property name');
      if (key === '__proto__' || key === 'prototype' || key === 'constructor') {
        fail(`${path}.${key}`, 'is not an allowed property name');
      }
      const descriptor = Object.getOwnPropertyDescriptor(record, key);
      if (!descriptor || !('value' in descriptor)) fail(`${path}.${key}`, 'must be a data property');
      jsonValue(descriptor.value, `${path}.${key}`, budget, ancestors, depth + 1);
    }
  }
  ancestors.delete(value);
}

/** The whole-document JSON walk, with its private budget allocated for you. */
export function assertJsonValue(value: unknown, path = '$'): void {
  jsonValue(value, path, { nodes: 0 }, new Set<object>(), 0);
}

export function validateRender(value: unknown, path: string): void {
  const render = object(value, path);
  keysExactly(render, ['grade', 'bloom', 'feedback'], path);

  const grade = object(render['grade'], `${path}.grade`);
  const gradeKeys = [
    'exposureEv', 'autoExposure', 'autoTarget', 'autoTau', 'autoMinGain', 'autoMaxGain',
    'tonemap', 'blackPoint', 'contrast', 'pivot', 'saturation', 'vignette', 'soilTint',
    'soilColor',
  ] as const;
  keysExactly(grade, gradeKeys, `${path}.grade`);
  finiteNumber(grade['exposureEv'], `${path}.grade.exposureEv`, -32, 32);
  boolean(grade['autoExposure'], `${path}.grade.autoExposure`);
  finiteNumber(grade['autoTarget'], `${path}.grade.autoTarget`, 0, 1000);
  finiteNumber(grade['autoTau'], `${path}.grade.autoTau`, 0.001, 3600);
  const autoMin = finiteNumber(grade['autoMinGain'], `${path}.grade.autoMinGain`, 0, 1e6);
  const autoMax = finiteNumber(grade['autoMaxGain'], `${path}.grade.autoMaxGain`, 0, 1e6);
  if (autoMin > autoMax) fail(`${path}.grade`, 'autoMinGain must not exceed autoMaxGain');
  if (!TONEMAPS.includes(grade['tonemap'] as never)) fail(`${path}.grade.tonemap`, 'is unsupported');
  finiteNumber(grade['blackPoint'], `${path}.grade.blackPoint`, 0, 1);
  finiteNumber(grade['contrast'], `${path}.grade.contrast`, 0, 100);
  finiteNumber(grade['pivot'], `${path}.grade.pivot`, 0, 1);
  finiteNumber(grade['saturation'], `${path}.grade.saturation`, 0, 100);
  finiteNumber(grade['vignette'], `${path}.grade.vignette`, 0, 1);
  finiteNumber(grade['soilTint'], `${path}.grade.soilTint`, 0, 100);
  boundedString(grade['soilColor'], `${path}.grade.soilColor`, 64);

  const bloom = object(render['bloom'], `${path}.bloom`);
  keysExactly(bloom, ['enabled', 'threshold', 'knee', 'intensity', 'levels'], `${path}.bloom`);
  boolean(bloom['enabled'], `${path}.bloom.enabled`);
  finiteNumber(bloom['threshold'], `${path}.bloom.threshold`, 0, 1e6);
  finiteNumber(bloom['knee'], `${path}.bloom.knee`, 0, 1e6);
  finiteNumber(bloom['intensity'], `${path}.bloom.intensity`, 0, 1e4);
  integer(bloom['levels'], `${path}.bloom.levels`, 1, MAX_BLOOM_LEVELS);

  const feedback = object(render['feedback'], `${path}.feedback`);
  keysExactly(feedback, ['amount', 'zoom'], `${path}.feedback`);
  finiteNumber(feedback['amount'], `${path}.feedback.amount`, 0, 2);
  finiteNumber(feedback['zoom'], `${path}.feedback.zoom`, 0.25, 4);
}

export function validateImpulse(value: unknown, path: string): void {
  const impulses = object(value, path);
  keysExactly(impulses, ['enabled', 'gain', 'responses'], path);
  boolean(impulses['enabled'], `${path}.enabled`);
  finiteNumber(impulses['gain'], `${path}.gain`, 0, 100);
  const responses = object(impulses['responses'], `${path}.responses`);
  keysExactly(responses, EVENT_KINDS, `${path}.responses`);
  const responseKeys = [
    'enabled', 'species', 'decayMs', 'deposit', 'flash', 'sensor', 'wiggle', 'splashCount',
    'splashRadius', 'splashPush', 'splashSwirl',
  ] as const;
  for (const kind of EVENT_KINDS) {
    const response = object(responses[kind], `${path}.responses.${kind}`);
    keysExactly(response, responseKeys, `${path}.responses.${kind}`);
    boolean(response['enabled'], `${path}.responses.${kind}.enabled`);
    integer(response['species'], `${path}.responses.${kind}.species`, -1, 63);
    finiteNumber(response['decayMs'], `${path}.responses.${kind}.decayMs`, 1, 60_000);
    finiteNumber(response['deposit'], `${path}.responses.${kind}.deposit`, -1000, 1000);
    finiteNumber(response['flash'], `${path}.responses.${kind}.flash`, -1000, 1000);
    finiteNumber(response['sensor'], `${path}.responses.${kind}.sensor`, -1000, 1000);
    finiteNumber(response['wiggle'], `${path}.responses.${kind}.wiggle`, -1000, 1000);
    integer(response['splashCount'], `${path}.responses.${kind}.splashCount`, 0, 32);
    finiteNumber(response['splashRadius'], `${path}.responses.${kind}.splashRadius`, 0, 1);
    finiteNumber(response['splashPush'], `${path}.responses.${kind}.splashPush`, -1000, 1000);
    finiteNumber(response['splashSwirl'], `${path}.responses.${kind}.splashSwirl`, -1000, 1000);
  }
}

const PALETTE_SPACES = ['hsl', 'hsluv', 'oklch'] as const;

const PALETTE_KEYS = [
  'mode',
  'space',
  'arc',
  'accentArc',
  'colors',
  'hueShiftDeg',
  'hueRateDegPerSec',
  'saturation',
  'brightness',
] as const;

/**
 * Palette v2, validated strictly and exactly — a v1 palette block inside a v4
 * recipe is rejected by `keysExactly`, not quietly defaulted. A recipe is a
 * reproduction contract; the *lift* from an old recipe version is the migration
 * seam (`liftExportRecipe`) and this is not it.
 */
function validatePalette(value: unknown, speciesCount: number, path: string): void {
  const palette = object(value, path);
  keysExactly(palette, PALETTE_KEYS, path);
  if (palette['mode'] !== 'arc' && palette['mode'] !== 'custom') {
    fail(`${path}.mode`, 'must be "arc" or "custom"');
  }
  if (!PALETTE_SPACES.includes(palette['space'] as never)) {
    fail(`${path}.space`, `must be one of ${PALETTE_SPACES.join(', ')}`);
  }
  // Both arcs, validated the same way: the primaries' and the accents'. They are
  // independent colour families, so nothing here relates one to the other.
  for (const key of ['arc', 'accentArc'] as const) {
    const arc = object(palette[key], `${path}.${key}`);
    keysExactly(arc, ['hueStartDeg', 'hueRangeDeg', 'sat', 'light'], `${path}.${key}`);
    // Angles are unbounded within the generic config-number bound: 720 and 0 are
    // the same wheel position, and rejecting one would make a hand-written recipe
    // fail for being arithmetically honest.
    finiteNumber(
      arc['hueStartDeg'],
      `${path}.${key}.hueStartDeg`,
      -MAX_ABS_CONFIG_NUMBER,
      MAX_ABS_CONFIG_NUMBER,
    );
    finiteNumber(
      arc['hueRangeDeg'],
      `${path}.${key}.hueRangeDeg`,
      -MAX_ABS_CONFIG_NUMBER,
      MAX_ABS_CONFIG_NUMBER,
    );
    finiteNumber(arc['sat'], `${path}.${key}.sat`, 0, 100);
    finiteNumber(arc['light'], `${path}.${key}.light`, 0, 100);
  }

  const colors = palette['colors'];
  if (!Array.isArray(colors) || colors.length !== speciesCount) {
    fail(`${path}.colors`, `must contain exactly ${speciesCount} colors`);
  }
  for (let i = 0; i < colors.length; i++) boundedString(colors[i], `${path}.colors[${i}]`, 64);
  finiteNumber(
    palette['hueShiftDeg'],
    `${path}.hueShiftDeg`,
    -MAX_ABS_CONFIG_NUMBER,
    MAX_ABS_CONFIG_NUMBER,
  );
  finiteNumber(palette['hueRateDegPerSec'], `${path}.hueRateDegPerSec`, -3600, 3600);
  finiteNumber(palette['saturation'], `${path}.saturation`, 0, 100);
  finiteNumber(palette['brightness'], `${path}.brightness`, 0, 100);
}

/**
 * The simulation and modulation blocks are one live object in the browser, so a
 * recipe that encodes them differently is a recipe whose two halves disagree
 * about the look. Every v2 field participates — an arc that matched only in its
 * derived hexes would drift apart the moment a headless renderer recomputed it.
 */
function palettesEqual(a: unknown, b: unknown): boolean {
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  for (const key of ['mode', 'space', 'hueShiftDeg', 'hueRateDegPerSec', 'saturation', 'brightness']) {
    if (left[key] !== right[key]) return false;
  }
  for (const which of ['arc', 'accentArc']) {
    const la = left[which] as Record<string, unknown>;
    const ra = right[which] as Record<string, unknown>;
    for (const key of ['hueStartDeg', 'hueRangeDeg', 'sat', 'light']) {
      if (la[key] !== ra[key]) return false;
    }
  }
  const lc = left['colors'] as unknown[];
  const rc = right['colors'] as unknown[];
  return lc.length === rc.length && lc.every((color, index) => color === rc[index]);
}

function validateSpecies(value: unknown, speciesCount: number, path: string): void {
  if (!Array.isArray(value) || value.length !== speciesCount) {
    fail(path, `must contain exactly ${speciesCount} species`);
  }
  for (let i = 0; i < value.length; i++) {
    const species = object(value[i], `${path}[${i}]`);
    boundedString(species['name'], `${path}[${i}].name`, 128);
    finiteNumber(species['brightness'], `${path}[${i}].brightness`, 0, 100);
  }
}

const PHYSARUM_KEYS = [
  'macros', 'speciesCount', 'maxAgents', 'gridScale', 'maxGridDim', 'depositScale',
  'senseGain', 'exposure', 'gamma', 'speed', 'paused', 'stemDrive', 'stemGain', 'soil',
  'palette', 'species', 'matrix',
] as const;

const PLIFE_KEYS = [
  'macros', 'population', 'matrixGen', 'field', 'budget', 'luma', 'speciesCount', 'maxParticles',
  'forceGain', 'maxSpeed', 'exposure', 'gamma', 'speed', 'paused', 'palette', 'species',
  'attraction', 'minR', 'maxR',
] as const;

/**
 * The per-particle luminance lane, validated field by field rather than left to
 * the generic JSON walk.
 *
 * `runtimeStateFromRecipe` clones `$.simulation` straight onto a `PlifeConfig`
 * with no defaulting step, so a missing or non-numeric field here does not
 * become the shipped default — it becomes `undefined`, and `lumaUniforms` turns
 * that into a NaN uniform, which takes out every particle's *position* as well
 * as its colour. A block that reaches a shader unmediated has to be checked
 * where it enters.
 */
const LUMA_KEYS = ['depth', 'curve', 'mid', 'hdrBudget', 'whitePeak', 'jitter'] as const;

const VIZFX_KEYS = [
  'speciesCount', 'species', 'palette', 'params', 'macros', 'energy', 'emittersPerLayer',
  'speed', 'paused',
] as const;

export function numericArray(value: unknown, length: number, path: string): void {
  if (!Array.isArray(value) || value.length !== length) fail(path, `must contain exactly ${length} values`);
  for (let i = 0; i < value.length; i++) {
    finiteNumber(value[i], `${path}[${i}]`, -MAX_ABS_CONFIG_NUMBER, MAX_ABS_CONFIG_NUMBER);
  }
}

/**
 * `particleBudget` is `null` for a document that has no such field to agree
 * with — a preset string derives the budget at apply time instead of carrying
 * it (`./preset.ts`). Everything else about the block is validated identically,
 * which is the point of not writing a second copy of this.
 */
export function validateSimulation(
  value: unknown,
  sim: string,
  particleBudget: number | null,
): number {
  const simulation = object(value, '$.simulation');
  if (Object.hasOwn(simulation, 'render')) fail('$.simulation.render', 'belongs at $.render');
  const speciesCount = integer(simulation['speciesCount'], '$.simulation.speciesCount', 1, 64);

  if (sim === 'physarum') {
    keysExactly(simulation, PHYSARUM_KEYS, '$.simulation');
    const maxAgents = integer(
      simulation['maxAgents'],
      '$.simulation.maxAgents',
      1,
      MAX_RECIPE_PARTICLE_BUDGET,
    );
    if (particleBudget !== null && particleBudget !== maxAgents) fail('$.particleBudget', 'must match $.simulation.maxAgents');
    boolean(simulation['paused'], '$.simulation.paused');
    boolean(simulation['stemDrive'], '$.simulation.stemDrive');
    numericArray(simulation['matrix'], speciesCount * speciesCount, '$.simulation.matrix');
  } else if (sim === 'plife') {
    keysExactly(simulation, PLIFE_KEYS, '$.simulation');
    integer(
      simulation['maxParticles'],
      '$.simulation.maxParticles',
      1,
      MAX_RECIPE_PARTICLE_BUDGET,
    );
    boolean(simulation['paused'], '$.simulation.paused');
    const budget = object(simulation['budget'], '$.simulation.budget');
    keysExactly(budget, ['cap', 'adaptive', 'floorFps', 'idealFps'], '$.simulation.budget');
    const cap = integer(budget['cap'], '$.simulation.budget.cap', 1, MAX_RECIPE_PARTICLE_BUDGET);
    if (particleBudget !== null && particleBudget !== cap) fail('$.particleBudget', 'must match $.simulation.budget.cap');
    boolean(budget['adaptive'], '$.simulation.budget.adaptive');
    const luma = object(simulation['luma'], '$.simulation.luma');
    keysExactly(luma, LUMA_KEYS, '$.simulation.luma');
    // Bounded generously rather than to the panel's ranges: this layer's job is
    // to refuse what a shader cannot survive, and `lumaUniforms` clamps the rest
    // into the sliders. A recipe authored on a build with a wider slider must
    // still render.
    for (const key of LUMA_KEYS) finiteNumber(luma[key], `$.simulation.luma.${key}`, 0, 64);
    numericArray(simulation['attraction'], speciesCount * speciesCount, '$.simulation.attraction');
    numericArray(simulation['minR'], speciesCount * speciesCount, '$.simulation.minR');
    numericArray(simulation['maxR'], speciesCount * speciesCount, '$.simulation.maxR');
  } else if (isVizFxId(sim)) {
    keysExactly(simulation, VIZFX_KEYS, '$.simulation');
    if (particleBudget !== null && particleBudget !== 0) fail('$.particleBudget', 'must be 0 for a non-particle visual');
    boolean(simulation['paused'], '$.simulation.paused');
    integer(simulation['emittersPerLayer'], '$.simulation.emittersPerLayer', 1, 64);
  } else {
    fail('$.sim', 'is unsupported');
  }

  validatePalette(simulation['palette'], speciesCount, '$.simulation.palette');
  validateSpecies(simulation['species'], speciesCount, '$.simulation.species');
  return speciesCount;
}

const MODULATION_KEYS = [
  'version', 'sim', 'speciesCount', 'palette', 'enabled', 'depth', 'groupDepth',
  'driverGains', 'stemFollow', 'responseSpeed', 'slew', 'boundary', 'extras',
] as const;

const MODULATION_REQUIRED_KEYS = MODULATION_KEYS.filter((key) => key !== 'extras');

export function validateModulation(
  value: unknown,
  sim: string,
  speciesCount: number,
  simulationPalette: unknown,
): void {
  const modulation = object(value, '$.modulation');
  keysRequired(modulation, MODULATION_KEYS, MODULATION_REQUIRED_KEYS, '$.modulation');
  if (Object.hasOwn(modulation, 'render')) fail('$.modulation.render', 'belongs at $.render');
  if (modulation['version'] !== MODULATION_VERSION) {
    fail('$.modulation.version', `must be ${MODULATION_VERSION}`);
  }
  if (modulation['sim'] !== sim) fail('$.modulation.sim', 'must match $.sim');
  if (modulation['speciesCount'] !== speciesCount) {
    fail('$.modulation.speciesCount', 'must match $.simulation.speciesCount');
  }
  validatePalette(modulation['palette'], speciesCount, '$.modulation.palette');
  if (!palettesEqual(modulation['palette'], simulationPalette)) {
    fail('$.modulation.palette', 'must match $.simulation.palette');
  }
  boolean(modulation['enabled'], '$.modulation.enabled');
  finiteNumber(modulation['depth'], '$.modulation.depth', 0, 100);

  const groupDepth = object(modulation['groupDepth'], '$.modulation.groupDepth');
  keysExactly(groupDepth, ['structure', 'matrix', 'population', 'decay'], '$.modulation.groupDepth');
  for (const group of ['structure', 'matrix', 'population', 'decay'] as const) {
    finiteNumber(groupDepth[group], `$.modulation.groupDepth.${group}`, 0, 100);
  }

  const gains = modulation['driverGains'];
  if (!Array.isArray(gains) || gains.length > 256) {
    fail('$.modulation.driverGains', 'must be an array of at most 256 gains');
  }
  for (let i = 0; i < gains.length; i++) {
    finiteNumber(gains[i], `$.modulation.driverGains[${i}]`, 0, 2);
  }

  const stemFollow = object(modulation['stemFollow'], '$.modulation.stemFollow');
  keysExactly(stemFollow, ['enabled', 'floor', 'curve', 'smoothingMs'], '$.modulation.stemFollow');
  boolean(stemFollow['enabled'], '$.modulation.stemFollow.enabled');
  finiteNumber(stemFollow['floor'], '$.modulation.stemFollow.floor', 0, 1);
  finiteNumber(stemFollow['curve'], '$.modulation.stemFollow.curve', 0.01, 100);
  finiteNumber(stemFollow['smoothingMs'], '$.modulation.stemFollow.smoothingMs', 1, 60_000);
  finiteNumber(modulation['responseSpeed'], '$.modulation.responseSpeed', 0, 100);

  const slew = object(modulation['slew'], '$.modulation.slew');
  keysExactly(slew, ['fast', 'medium', 'slow'], '$.modulation.slew');
  for (const speed of ['fast', 'medium', 'slow'] as const) {
    finiteNumber(slew[speed], `$.modulation.slew.${speed}`, 0, 3600);
  }

  const boundary = object(modulation['boundary'], '$.modulation.boundary');
  keysExactly(boundary, ['enabled', 'snapFraction', 'respawnFraction'], '$.modulation.boundary');
  boolean(boundary['enabled'], '$.modulation.boundary.enabled');
  finiteNumber(boundary['snapFraction'], '$.modulation.boundary.snapFraction', 0, 1);
  finiteNumber(boundary['respawnFraction'], '$.modulation.boundary.respawnFraction', 0, 1);
}

/** The authored θ centre: bounded, finite, and never empty. */
export function validateModulationBase(value: unknown, path = '$.modulationBase'): void {
  if (!Array.isArray(value) || value.length === 0) fail(path, 'must be a non-empty array');
  if (value.length > MAX_ARRAY_LENGTH) fail(path, `exceeds ${MAX_ARRAY_LENGTH} entries`);
  for (let i = 0; i < value.length; i++) {
    finiteNumber(value[i], `${path}[${i}]`, -MAX_ABS_CONFIG_NUMBER, MAX_ABS_CONFIG_NUMBER);
  }
}

/** Validate without allocating configuration-sized arrays or touching browser state. */
export function validateExportRecipe(value: unknown): asserts value is ExportRecipe {
  withLabel(RECIPE_LABEL, () => validateRecipeBody(value));
}

function validateRecipeBody(value: unknown): void {
  assertJsonValue(value);
  const recipe = object(value, '$');
  keysExactly(recipe, TOP_LEVEL_KEYS, '$');

  if (recipe['version'] !== EXPORT_RECIPE_VERSION) {
    fail('$.version', `has unsupported version ${String(recipe['version'])}`);
  }
  boundedString(recipe['rendererBuild'], '$.rendererBuild', 256);
  const track = object(recipe['track'], '$.track');
  keysExactly(track, ['id', 'contentVersion'], '$.track');
  identifier(track['id'], '$.track.id');
  boundedString(track['contentVersion'], '$.track.contentVersion', 256);
  const sim = identifier(recipe['sim'], '$.sim');
  integer(recipe['seed'], '$.seed', 0, 0xffff_ffff);
  boolean(recipe['seedPinned'], '$.seedPinned');
  const particleBudget = integer(
    recipe['particleBudget'],
    '$.particleBudget',
    0,
    MAX_RECIPE_PARTICLE_BUDGET,
  );
  const speciesCount = validateSimulation(recipe['simulation'], sim, particleBudget);
  const simulation = object(recipe['simulation'], '$.simulation');
  validateModulation(recipe['modulation'], sim, speciesCount, simulation['palette']);
  validateModulationBase(recipe['modulationBase']);

  validateImpulse(recipe['impulses'], '$.impulses');
  validateRender(recipe['render'], '$.render');

  const presentation = object(recipe['presentation'], '$.presentation');
  keysExactly(presentation, ['mode', 'autoAdvance'], '$.presentation');
  if (presentation['mode'] !== 'single') fail('$.presentation.mode', 'must be "single"');
  if (presentation['autoAdvance'] !== false) {
    fail('$.presentation.autoAdvance', 'must be false in the single-visual recipe');
  }

  const output = object(recipe['output'], '$.output');
  keysExactly(
    output,
    ['profile', 'encoder', 'paperWhiteNits', 'masteringPeakNits'],
    '$.output',
  );
  if (!EXPORT_PROFILES.includes(output['profile'] as never)) fail('$.output.profile', 'is unsupported');
  if (!EXPORT_ENCODERS.includes(output['encoder'] as never)) fail('$.output.encoder', 'is unsupported');
  // The codec is a property of the profile, not an independent browser choice:
  // an HDR10 profile is only meaningful through its 10-bit encoder.
  if (output['encoder'] !== requiredEncoder(output['profile'] as ExportProfile)) {
    fail('$.output.encoder', 'does not match the encoder required by $.output.profile');
  }
  const paperWhiteNits = finiteNumber(
    output['paperWhiteNits'],
    '$.output.paperWhiteNits',
    MIN_PAPER_WHITE_NITS,
    MAX_PAPER_WHITE_NITS,
  );
  const masteringPeakNits = finiteNumber(
    output['masteringPeakNits'],
    '$.output.masteringPeakNits',
    MIN_MASTERING_PEAK_NITS,
    MAX_MASTERING_PEAK_NITS,
  );
  if (masteringPeakNits < paperWhiteNits) {
    fail('$.output.masteringPeakNits', 'must be at least $.output.paperWhiteNits');
  }
}

/** Stable key order, so a document is content-hashable and diffs are readable. */
export function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === 'object') {
    const source = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(source).sort()) out[key] = canonical(source[key]);
    return out;
  }
  return value;
}

/** Canonical key ordering makes sidecars stable and recipes content-hashable. */
export function serializeExportRecipe(recipe: ExportRecipe): string {
  validateExportRecipe(recipe);
  return `${JSON.stringify(canonical(recipe), null, 2)}\n`;
}

/** Shape of a v1 palette block, as it appears inside a v3 recipe. */
function liftPaletteBlock(value: unknown): unknown {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return value;
  const old = value as Record<string, unknown>;
  if (typeof old['mode'] === 'string') return value; // already v2
  return {
    mode: 'custom',
    // Pre-v2 palettes are authored hexes at zero shift, which no space touches.
    space: 'hsluv',
    // Not `defaultArc()`/`defaultAccentArc()`: this module must not import sim
    // art direction, and neither arc is reachable in custom mode anyway. These
    // are the same numbers; the "v3 lift touches nothing but the palette" test
    // is what fails loudly if the defaults ever move apart.
    arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
    accentArc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 80 },
    colors: old['colors'],
    hueShiftDeg: 0,
    hueRateDegPerSec: 0,
    saturation: old['saturation'],
    brightness: old['brightness'],
  };
}

/**
 * v3 → v4, and the reason this is a *lift* rather than a rejection.
 *
 * The precedent this looks like is the `hevc-hdr10-*` → `av1-hdr10-*` profile
 * rename above, which is deliberately NOT aliased: those ids named a different
 * codec, so re-pointing them would make an old recipe lie about what it
 * produced. This is the opposite case. A v1 palette block `{colors, saturation,
 * brightness}` and a v2 block in custom mode with a zero hue shift and a zero
 * cycle rate are the *same function of the same inputs* — `paletteHex`
 * short-circuits that case to the authored hex with no conversion — so the lift
 * preserves meaning exactly and an old export replays to the same pixels.
 *
 * Anything that is not the palette is untouched: v3 and v4 differ in nothing
 * else.
 */
function liftV3toV4(recipe: Record<string, unknown>): Record<string, unknown> {
  const simulation = recipe['simulation'];
  const modulation = recipe['modulation'];
  const lifted: Record<string, unknown> = { ...recipe, version: 4 };
  if (simulation !== null && typeof simulation === 'object' && !Array.isArray(simulation)) {
    lifted['simulation'] = {
      ...(simulation as Record<string, unknown>),
      palette: liftPaletteBlock((simulation as Record<string, unknown>)['palette']),
    };
  }
  if (modulation !== null && typeof modulation === 'object' && !Array.isArray(modulation)) {
    const mod = modulation as Record<string, unknown>;
    lifted['modulation'] = {
      ...mod,
      // The modulation block carries its own version, which the validator pins
      // to MODULATION_VERSION. A v3 recipe embeds a v4 modulation config whose
      // only v5 change is this same palette lift, so bumping it here is honest.
      ...(mod['version'] === MODULATION_VERSION - 1 ? { version: MODULATION_VERSION } : {}),
      palette: liftPaletteBlock(mod['palette']),
    };
  }
  return lifted;
}

/**
 * v4 → v5: the impulse lane's new `wiggle` depth, filled in at **0** and not at
 * the shipped default.
 *
 * This is the same rule the palette lift follows, applied to a case where the
 * two candidate answers differ. A recipe is a reproduction contract, and the
 * render a v4 recipe describes was produced by a build with no wiggle lane at
 * all — so the value that preserves its meaning is the one that does nothing.
 * Defaulting it to the new 0.6 would silently re-render an old export with an
 * effect it never had, which is exactly the "makes an old recipe lie" failure
 * the profile rename above refuses to commit.
 *
 * `matrixGen.wiggleRoll` needs no lift for the same reason it is not validated:
 * it is a key for a draw that only matters when the depth is non-zero, and at
 * `wiggle: 0` every family of directions produces the same matrix.
 */
function liftV4toV5(recipe: Record<string, unknown>): Record<string, unknown> {
  // Literal 5, NOT `EXPORT_RECIPE_VERSION`. Each hop must land on the version it
  // is named for or the chain skips the hops above it — when this said
  // `EXPORT_RECIPE_VERSION` and that constant went to 6, a v4 sidecar arrived at
  // "v6" having never met the v6 lift, i.e. missing the block v6 requires.
  // `liftV3toV4` has always used a literal for the same reason.
  const lifted: Record<string, unknown> = { ...recipe, version: 5 };
  const impulses = recipe['impulses'];
  if (impulses === null || typeof impulses !== 'object' || Array.isArray(impulses)) return lifted;
  const src = impulses as Record<string, unknown>;
  const responses = src['responses'];
  if (responses === null || typeof responses !== 'object' || Array.isArray(responses)) {
    return lifted;
  }
  const out: Record<string, unknown> = {};
  for (const [kind, response] of Object.entries(responses as Record<string, unknown>)) {
    out[kind] =
      response !== null && typeof response === 'object' && !Array.isArray(response)
        ? { wiggle: 0, ...(response as Record<string, unknown>) }
        : response;
  }
  lifted['impulses'] = { ...src, responses: out };
  return lifted;
}

/**
 * v5 → v6: particle life's `luma` block, filled in at **depth 0** — the lane's
 * off state — for exactly the reason the wiggle lift fills 0.
 *
 * The remaining five fields are written as literals rather than pulled from
 * `defaultPlifeLuma()`, and that is deliberate: a lift describes what a v5
 * recipe *meant*, which is a fact frozen in 2026, while the defaults function is
 * live art direction that will move. Importing it would make this lift's output
 * drift every time somebody retunes the shipped look — the same class of bug as
 * defaulting `wiggle` to 0.6. They are inert at depth 0 anyway; they exist here
 * only because `runtimeStateFromRecipe` clones the block onto the config with no
 * defaulting step, so a partial block would reach the shader as NaN.
 *
 * A physarum or vizfx recipe passes through untouched: the block is plife's.
 */
function liftV5toV6(recipe: Record<string, unknown>): Record<string, unknown> {
  const lifted: Record<string, unknown> = { ...recipe, version: EXPORT_RECIPE_VERSION };
  if (recipe['sim'] !== 'plife') return lifted;
  const simulation = recipe['simulation'];
  if (simulation === null || typeof simulation !== 'object' || Array.isArray(simulation)) {
    return lifted;
  }
  lifted['simulation'] = {
    luma: { depth: 0, curve: 2.5, mid: 0.4, hdrBudget: 1, whitePeak: 0.5, jitter: 0.12 },
    ...(simulation as Record<string, unknown>),
  };
  return lifted;
}

/**
 * Bring an older recipe up to the current schema, one hop at a time.
 *
 * Chained rather than branched: a v3 sidecar has to arrive at v5 through the
 * same v4 the v4 sidecars go through, or the two paths would eventually
 * disagree about what a v3 recipe means. A recipe at any other version falls
 * through unchanged and is rejected by `validateExportRecipe`.
 */
export function liftExportRecipe(value: unknown): unknown {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return value;
  let recipe = value as Record<string, unknown>;
  if (recipe['version'] !== 3 && recipe['version'] !== 4 && recipe['version'] !== 5) return value;
  if (recipe['version'] === 3) recipe = liftV3toV4(recipe);
  if (recipe['version'] === 4) recipe = liftV4toV5(recipe);
  if (recipe['version'] === 5) recipe = liftV5toV6(recipe);
  return recipe;
}

export function parseExportRecipe(text: string): ExportRecipe {
  return withLabel(RECIPE_LABEL, () => parseRecipeText(text));
}

function parseRecipeText(text: string): ExportRecipe {
  if (text.length > MAX_RECIPE_JSON_CHARS) {
    fail('$', `exceeds ${MAX_RECIPE_JSON_CHARS} serialized characters`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    fail('$', `is not valid JSON (${(error as Error).message})`);
  }
  const lifted = liftExportRecipe(value);
  validateExportRecipe(lifted);
  return lifted;
}
