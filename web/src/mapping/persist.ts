/**
 * The mapping is data, so persistence is the whole product surface of the
 * workbench: a MappingConfig JSON file *is* the art direction for a track.
 *
 * Three paths, deliberately: a file you can commit and diff, a localStorage
 * autosave so a reload never costs you a tuning session, and a strict parser so
 * a hand-edited file fails loudly instead of half-applying.
 */
import { MAPPING_VERSION, type Anchor, type MappingConfig } from './types.ts';
import type { Preset, SpeciesPreset } from './preset.ts';
import { defaultPalette, type AdaptiveTriple, type Palette } from '../sim/physarum/config.ts';
import {
  defaultRenderConfig,
  TONEMAPS,
  type RenderConfig,
  type ToneMap,
} from '../sim/render/config.ts';

const STORAGE_KEY = 'lmt.mapping';

/**
 * Sanity bound on K, not a hardware limit — the real ceiling is the device's
 * `maxTextureArrayLayers`, which the sim checks at init. This only has to be low
 * enough that a corrupt file cannot make the parser allocate.
 */
const MAX_SPECIES_COUNT = 64;

/** Six decimals keeps files diffable without visibly changing any parameter. */
function round(_key: string, value: unknown): unknown {
  return typeof value === 'number' && Number.isFinite(value)
    ? Number(value.toFixed(6))
    : value;
}

export function serializeMapping(cfg: MappingConfig): string {
  return JSON.stringify(cfg, round, 2);
}

function fail(what: string): never {
  throw new Error(`mapping: ${what}`);
}

function num(v: unknown, what: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) fail(`${what} is not a finite number`);
  return v;
}

/**
 * A count that sizes arrays, so it is checked *before* anything allocates from
 * it: a hand-broken file with `speciesCount: 1e9` must fail here rather than ask
 * for a billion-element palette, and a fractional one must not produce a
 * half-formed array either.
 */
function count(v: unknown, what: string, max: number): number {
  const n = num(v, what);
  if (!Number.isInteger(n) || n < 1 || n > max) fail(`${what} is not an integer in 1..${max}`);
  return n;
}

function numArray(v: unknown, what: string): number[] {
  if (!Array.isArray(v)) fail(`${what} is not an array`);
  return v.map((x, i) => num(x, `${what}[${i}]`));
}

function triple(v: unknown, what: string): AdaptiveTriple {
  const o = v as Record<string, unknown> | undefined;
  if (!o || typeof o !== 'object') fail(`${what} is missing`);
  return {
    p1: num(o['p1'], `${what}.p1`),
    p2: num(o['p2'], `${what}.p2`),
    p3: num(o['p3'], `${what}.p3`),
  };
}

function speciesPreset(v: unknown, what: string): SpeciesPreset {
  const o = v as Record<string, unknown> | undefined;
  if (!o || typeof o !== 'object') fail(`${what} is missing`);
  return {
    // v1 files have no brightness (and a colorHex we drop on the floor here; the
    // palette is lifted out of anchor 0 in parseMapping instead).
    brightness: o['brightness'] === undefined ? 1 : num(o['brightness'], `${what}.brightness`),
    intensity: num(o['intensity'], `${what}.intensity`),
    deposit: num(o['deposit'], `${what}.deposit`),
    decay: num(o['decay'], `${what}.decay`),
    aliveFraction: num(o['aliveFraction'], `${what}.aliveFraction`),
    diffuseCentre: num(o['diffuseCentre'], `${what}.diffuseCentre`),
    sensorDist: triple(o['sensorDist'], `${what}.sensorDist`),
    sensorAngle: triple(o['sensorAngle'], `${what}.sensorAngle`),
    rotate: triple(o['rotate'], `${what}.rotate`),
    moveDist: triple(o['moveDist'], `${what}.moveDist`),
  };
}

function preset(v: unknown, what: string): Preset {
  const o = v as Record<string, unknown> | undefined;
  if (!o || typeof o !== 'object') fail(`${what} is missing`);
  const species = o['species'];
  if (!Array.isArray(species)) fail(`${what}.species is not an array`);
  return {
    species: species.map((s, i) => speciesPreset(s, `${what}.species[${i}]`)),
    matrix: numArray(o['matrix'], `${what}.matrix`),
    senseGain: num(o['senseGain'], `${what}.senseGain`),
    exposure: num(o['exposure'], `${what}.exposure`),
    gamma: num(o['gamma'], `${what}.gamma`),
    stemGain: num(o['stemGain'], `${what}.stemGain`),
  };
}

/** Warn about the v1 → v2 lift once per session, not once per anchor per load. */
let warnedMigration = false;

function palette(v: unknown, speciesCount: number, what: string): Palette {
  const o = (v ?? {}) as Record<string, unknown>;
  const raw = Array.isArray(o['colors']) ? (o['colors'] as unknown[]) : [];
  const fallback = defaultPalette(speciesCount);
  const colors = fallback.colors.map((c, i) => (typeof raw[i] === 'string' ? (raw[i] as string) : c));
  return {
    colors,
    saturation: o['saturation'] === undefined ? 1 : num(o['saturation'], `${what}.saturation`),
    brightness: o['brightness'] === undefined ? 1 : num(o['brightness'], `${what}.brightness`),
  };
}

/**
 * Phase 7's render block. Every field is optional and falls back to the shipped
 * default — the grade is art direction, so a file that predates it (or that a
 * human trimmed by hand) should adopt the current defaults rather than fail.
 */
function renderBlock(v: unknown, what: string): RenderConfig {
  const out = defaultRenderConfig();
  const o = (v ?? {}) as Record<string, unknown>;
  const pick = <T extends object>(dst: T, raw: unknown, label: string): void => {
    const src = (raw ?? {}) as Record<string, unknown>;
    for (const key of Object.keys(dst) as (keyof T & string)[]) {
      const value = src[key];
      if (value === undefined) continue;
      const current = dst[key];
      if (typeof current === 'number') {
        (dst as Record<string, unknown>)[key] = num(value, `${label}.${key}`);
      } else if (typeof current === 'boolean') {
        (dst as Record<string, unknown>)[key] = value === true;
      } else if (typeof value === 'string') {
        (dst as Record<string, unknown>)[key] = value;
      }
    }
  };
  pick(out.grade, o['grade'], `${what}.grade`);
  pick(out.bloom, o['bloom'], `${what}.bloom`);
  pick(out.feedback, o['feedback'], `${what}.feedback`);
  if (!TONEMAPS.includes(out.grade.tonemap)) {
    out.grade.tonemap = defaultRenderConfig().grade.tonemap as ToneMap;
  }
  return out;
}

/**
 * v1 kept a `colorHex` inside every species of every anchor preset, so a track's
 * colour was blended. Revision 2 makes it static: keep anchor 0's colours as the
 * palette (anchor 0 is the untouched default, the closest thing v1 had to an
 * authored one), and drop the rest. Lossy by design, and said out loud once.
 */
function migratedPalette(anchorsRaw: unknown[], speciesCount: number): Palette {
  const first = (anchorsRaw[0] as Record<string, unknown> | undefined)?.['preset'] as
    | Record<string, unknown>
    | undefined;
  const species = Array.isArray(first?.['species']) ? (first['species'] as unknown[]) : [];
  const colors = species
    .map((s) => (s as Record<string, unknown>)['colorHex'])
    .filter((c): c is string => typeof c === 'string');
  if (!warnedMigration) {
    warnedMigration = true;
    console.warn(
      'mapping: migrating a v1 file — per-anchor colours are gone (plan.md Revision 2). ' +
        "Anchor 0's colours became the static palette; every other anchor's were discarded.",
    );
  }
  return palette({ colors }, speciesCount, 'palette');
}

export function parseMapping(text: string): MappingConfig {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    fail(`not valid JSON (${(err as Error).message})`);
  }
  const o = raw as Record<string, unknown>;
  if (!o || typeof o !== 'object') fail('top level is not an object');
  const version = o['version'];
  if (version !== MAPPING_VERSION && version !== 1) {
    fail(`unsupported version ${String(version)}`);
  }

  const anchorsRaw = o['anchors'];
  if (!Array.isArray(anchorsRaw)) fail('anchors is not an array');
  const speciesCount = count(o['speciesCount'], 'speciesCount', MAX_SPECIES_COUNT);
  const anchors: Anchor[] = anchorsRaw.map((a, i) => {
    const ao = a as Record<string, unknown>;
    return {
      id: typeof ao['id'] === 'string' ? (ao['id'] as string) : `a${i}`,
      name: typeof ao['name'] === 'string' ? (ao['name'] as string) : `anchor ${i}`,
      center: numArray(ao['center'], `anchors[${i}].center`),
      preset: preset(ao['preset'], `anchors[${i}].preset`),
    };
  });

  const km = (o['kmeans'] ?? {}) as Record<string, unknown>;
  const slew = (o['slew'] ?? {}) as Record<string, unknown>;
  const boundary = (o['boundary'] ?? {}) as Record<string, unknown>;

  return {
    version: MAPPING_VERSION,
    speciesCount,
    palette:
      version === 1
        ? migratedPalette(anchorsRaw, speciesCount)
        : palette(o['palette'], speciesCount, 'palette'),
    render: renderBlock(o['render'], 'render'),
    latentDims: num(o['latentDims'], 'latentDims'),
    kmeans: {
      k: num(km['k'], 'kmeans.k'),
      seed: num(km['seed'], 'kmeans.seed'),
      iterations: num(km['iterations'], 'kmeans.iterations'),
    },
    temperature: num(o['temperature'], 'temperature'),
    distanceScale: num(o['distanceScale'], 'distanceScale'),
    slew: {
      fast: num(slew['fast'], 'slew.fast'),
      medium: num(slew['medium'], 'slew.medium'),
      slow: num(slew['slow'], 'slew.slow'),
    },
    boundary: {
      enabled: boundary['enabled'] !== false,
      snapFraction: num(boundary['snapFraction'], 'boundary.snapFraction'),
      respawnFraction: num(boundary['respawnFraction'], 'boundary.respawnFraction'),
    },
    anchors,
  };
}

/**
 * A mapping authored for a different K, or against a different latent width,
 * cannot be applied field-for-field. The caller refits instead of guessing.
 */
export function mappingFits(cfg: MappingConfig, speciesCount: number, latentDims: number): boolean {
  if (cfg.speciesCount !== speciesCount) return false;
  if (cfg.latentDims !== latentDims) return false;
  return cfg.anchors.every(
    (a) => a.center.length === latentDims && a.preset.species.length >= speciesCount,
  );
}

export function saveMappingLocal(cfg: MappingConfig): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, serializeMapping(cfg));
    return true;
  } catch {
    return false; // private mode / quota — autosave is a convenience, not a contract
  }
}

export function loadMappingLocal(): MappingConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return parseMapping(raw);
  } catch (err) {
    console.warn('mapping: ignoring unusable autosave —', err);
    return null;
  }
}

export function clearMappingLocal(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // nothing to do
  }
}

export function downloadText(filename: string, text: string, mime = 'application/json'): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  // The click is synchronous but the fetch of the blob is not; give it a tick.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Opens a file picker and resolves with the file's text, or null if cancelled. */
export function pickTextFile(accept = '.json,application/json'): Promise<string | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      file.text().then(resolve).catch(() => resolve(null));
    });
    // Cancel fires no 'change' in most browsers; the promise simply never settles,
    // which is fine — nothing awaits it except an event handler.
    input.click();
  });
}
