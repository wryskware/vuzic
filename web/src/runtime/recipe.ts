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

export const EXPORT_RECIPE_VERSION = 3 as const;

/**
 * Additive within recipe v3: a v3 recipe naming an SDR debug profile means
 * exactly what it always did, so no existing capture, request, or sidecar is
 * invalidated by the HDR profiles joining the list.
 */
export const EXPORT_PROFILES = [
  'hevc-hdr10-2160p120',
  'hevc-hdr10-1080p120',
  'av1-sdr-debug-2160p120',
  'av1-sdr-debug-1080p120',
] as const;
export type ExportProfile = (typeof EXPORT_PROFILES)[number];

export const EXPORT_ENCODERS = ['hevc_nvenc', 'av1_nvenc'] as const;
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
export type RecipeModulationConfig = Omit<ModulationConfig, 'render'>;

export interface ExportRecipeV3 {
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
  /** V3 exports one concrete visual and never reads browser auto-advance state. */
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

export type ExportRecipe = ExportRecipeV3;

const TOP_LEVEL_KEYS = [
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
const MAX_ARRAY_LENGTH = 131_072;
const MAX_OBJECT_KEYS = 4096;
const MAX_DEPTH = 32;
const MAX_NODES = 250_000;
const MAX_ABS_CONFIG_NUMBER = 1e12;
const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function fail(path: string, message: string): never {
  throw new Error(`export recipe: ${path} ${message}`);
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    fail(path, 'must be an object');
  }
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) fail(path, 'must be a plain object');
  return value as Record<string, unknown>;
}

function keysExactly(value: Record<string, unknown>, allowed: readonly string[], path: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) fail(`${path}.${key}`, 'is not supported by this schema version');
  }
  for (const key of allowed) {
    if (!Object.hasOwn(value, key)) fail(`${path}.${key}`, 'is required');
  }
}

function keysRequired(
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

function boundedString(value: unknown, path: string, max = MAX_STRING_CHARS): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > max) {
    fail(path, `must be a non-empty string of at most ${max} characters`);
  }
  return value;
}

function identifier(value: unknown, path: string): string {
  const text = boundedString(value, path, 128);
  if (!ID.test(text)) fail(path, 'contains unsupported characters');
  return text;
}

function finiteNumber(value: unknown, path: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    fail(path, `must be a finite number in ${min}..${max}`);
  }
  return value;
}

function integer(value: unknown, path: string, min: number, max: number): number {
  const n = finiteNumber(value, path, min, max);
  if (!Number.isSafeInteger(n)) fail(path, 'must be a safe integer');
  return n;
}

function boolean(value: unknown, path: string): boolean {
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

function validateRender(value: unknown, path: string): void {
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

function validateImpulse(value: unknown, path: string): void {
  const impulses = object(value, path);
  keysExactly(impulses, ['enabled', 'gain', 'responses'], path);
  boolean(impulses['enabled'], `${path}.enabled`);
  finiteNumber(impulses['gain'], `${path}.gain`, 0, 100);
  const responses = object(impulses['responses'], `${path}.responses`);
  keysExactly(responses, EVENT_KINDS, `${path}.responses`);
  const responseKeys = [
    'enabled', 'species', 'decayMs', 'deposit', 'flash', 'sensor', 'splashCount',
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
    integer(response['splashCount'], `${path}.responses.${kind}.splashCount`, 0, 32);
    finiteNumber(response['splashRadius'], `${path}.responses.${kind}.splashRadius`, 0, 1);
    finiteNumber(response['splashPush'], `${path}.responses.${kind}.splashPush`, -1000, 1000);
    finiteNumber(response['splashSwirl'], `${path}.responses.${kind}.splashSwirl`, -1000, 1000);
  }
}

function validatePalette(value: unknown, speciesCount: number, path: string): void {
  const palette = object(value, path);
  keysExactly(palette, ['colors', 'saturation', 'brightness'], path);
  const colors = palette['colors'];
  if (!Array.isArray(colors) || colors.length !== speciesCount) {
    fail(`${path}.colors`, `must contain exactly ${speciesCount} colors`);
  }
  for (let i = 0; i < colors.length; i++) boundedString(colors[i], `${path}.colors[${i}]`, 64);
  finiteNumber(palette['saturation'], `${path}.saturation`, 0, 100);
  finiteNumber(palette['brightness'], `${path}.brightness`, 0, 100);
}

function palettesEqual(a: unknown, b: unknown): boolean {
  const left = a as { colors: unknown[]; saturation: number; brightness: number };
  const right = b as { colors: unknown[]; saturation: number; brightness: number };
  return left.saturation === right.saturation &&
    left.brightness === right.brightness &&
    left.colors.length === right.colors.length &&
    left.colors.every((color, index) => color === right.colors[index]);
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
  'macros', 'population', 'matrixGen', 'field', 'budget', 'speciesCount', 'maxParticles',
  'forceGain', 'maxSpeed', 'exposure', 'gamma', 'speed', 'paused', 'palette', 'species',
  'attraction', 'minR', 'maxR',
] as const;

const VIZFX_KEYS = [
  'speciesCount', 'species', 'palette', 'params', 'macros', 'energy', 'emittersPerLayer',
  'speed', 'paused',
] as const;

function numericArray(value: unknown, length: number, path: string): void {
  if (!Array.isArray(value) || value.length !== length) fail(path, `must contain exactly ${length} values`);
  for (let i = 0; i < value.length; i++) {
    finiteNumber(value[i], `${path}[${i}]`, -MAX_ABS_CONFIG_NUMBER, MAX_ABS_CONFIG_NUMBER);
  }
}

function validateSimulation(
  value: unknown,
  sim: string,
  particleBudget: number,
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
    if (particleBudget !== maxAgents) fail('$.particleBudget', 'must match $.simulation.maxAgents');
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
    if (particleBudget !== cap) fail('$.particleBudget', 'must match $.simulation.budget.cap');
    boolean(budget['adaptive'], '$.simulation.budget.adaptive');
    numericArray(simulation['attraction'], speciesCount * speciesCount, '$.simulation.attraction');
    numericArray(simulation['minR'], speciesCount * speciesCount, '$.simulation.minR');
    numericArray(simulation['maxR'], speciesCount * speciesCount, '$.simulation.maxR');
  } else if (isVizFxId(sim)) {
    keysExactly(simulation, VIZFX_KEYS, '$.simulation');
    if (particleBudget !== 0) fail('$.particleBudget', 'must be 0 for a non-particle visual');
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

function validateModulation(
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

/** Validate without allocating configuration-sized arrays or touching browser state. */
export function validateExportRecipe(value: unknown): asserts value is ExportRecipe {
  jsonValue(value, '$', { nodes: 0 }, new Set<object>(), 0);
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
  const modulationBase = recipe['modulationBase'];
  if (!Array.isArray(modulationBase) || modulationBase.length === 0) {
    fail('$.modulationBase', 'must be a non-empty array');
  }
  if (modulationBase.length > MAX_ARRAY_LENGTH) {
    fail('$.modulationBase', `exceeds ${MAX_ARRAY_LENGTH} entries`);
  }
  for (let i = 0; i < modulationBase.length; i++) {
    finiteNumber(
      modulationBase[i],
      `$.modulationBase[${i}]`,
      -MAX_ABS_CONFIG_NUMBER,
      MAX_ABS_CONFIG_NUMBER,
    );
  }

  validateImpulse(recipe['impulses'], '$.impulses');
  validateRender(recipe['render'], '$.render');

  const presentation = object(recipe['presentation'], '$.presentation');
  keysExactly(presentation, ['mode', 'autoAdvance'], '$.presentation');
  if (presentation['mode'] !== 'single') fail('$.presentation.mode', 'must be "single" in v3');
  if (presentation['autoAdvance'] !== false) {
    fail('$.presentation.autoAdvance', 'must be false in the single-visual v3 recipe');
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
  // an HDR10 profile is only meaningful through its Main10 encoder.
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

function canonical(value: unknown): unknown {
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

export function parseExportRecipe(text: string): ExportRecipe {
  if (text.length > MAX_RECIPE_JSON_CHARS) {
    fail('$', `exceeds ${MAX_RECIPE_JSON_CHARS} serialized characters`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    fail('$', `is not valid JSON (${(error as Error).message})`);
  }
  validateExportRecipe(value);
  return value;
}
