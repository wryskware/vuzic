/**
 * The modulation config is data, so persistence is the workbench's product
 * surface: a `modulation.json` file *is* the art direction for a track — the
 * palette, the grade, and how hard the music is allowed to push.
 *
 * Three paths, deliberately: a file you can commit and diff, a localStorage
 * autosave so a reload never costs you a tuning session, and a strict parser so
 * a hand-edited file fails loudly instead of half-applying.
 *
 * Version 3 (plan.md Revision 3) dropped anchors, k-means centres, per-anchor
 * presets, temperature and distanceScale. A v1/v2 file still loads — its palette
 * and render block are the parts that were never scene-specific — and everything
 * else is discarded with one warning rather than silently reinterpreted.
 */
import { MODULATION_VERSION, type BoundaryOptions, type ModulationConfig } from './types.ts';
import { MOD_GROUPS, type ModGroup } from './preset.ts';
import { defaultPalette, type Palette, type PhysarumConfig } from '../sim/physarum/config.ts';
import {
  defaultRenderConfig,
  TONEMAPS,
  type RenderConfig,
  type ToneMap,
} from '../sim/render/config.ts';
import { DEFAULT_SLEW, type SlewRates } from './slew.ts';

const STORAGE_KEY = 'lmt.mapping';

/**
 * Sanity bound on K, not a hardware limit — the real ceiling is the device's
 * `maxTextureArrayLayers`, which the sim checks at init. This only has to be low
 * enough that a corrupt file cannot make the parser allocate.
 */
const MAX_SPECIES_COUNT = 64;

/**
 * Shipped depths. 1.0 means "the typical excursion is ~0.55 of each parameter's
 * authored half-range", which is the LEGIBLE end plan.md Revision 2 §3 and
 * Revision 3 both ask for — subtle was the complaint, not the goal.
 *
 * `decay` and `matrix` sit below 1 not to be timid but because they are the two
 * groups whose effects arrive on a lag: the trail field integrates decay over
 * seconds and M reshapes a network over tens of seconds, so a full-depth swing
 * there reads as drift rather than as response.
 */
export function defaultGroupDepth(): Record<ModGroup, number> {
  return {
    structure: 1,
    matrix: 0.7,
    population: 1,
    brightness: 1,
    decay: 0.6,
  };
}

export function defaultModulationConfig(base: PhysarumConfig): ModulationConfig {
  return {
    version: MODULATION_VERSION,
    speciesCount: base.speciesCount,
    // Shared by reference with the live config: a palette edit in the workbench
    // is an edit to the thing that gets serialised, with no sync step to forget.
    palette: base.palette,
    render: base.render,
    enabled: true,
    depth: 1,
    groupDepth: defaultGroupDepth(),
    responseSpeed: 1,
    slew: { ...DEFAULT_SLEW },
    boundary: { enabled: true, snapFraction: 0.6, respawnFraction: 0.12 },
  };
}

/** Six decimals keeps files diffable without visibly changing any parameter. */
function round(_key: string, value: unknown): unknown {
  return typeof value === 'number' && Number.isFinite(value) ? Number(value.toFixed(6)) : value;
}

export function serializeModulation(cfg: ModulationConfig): string {
  return JSON.stringify(cfg, round, 2);
}

function fail(what: string): never {
  throw new Error(`modulation: ${what}`);
}

function num(v: unknown, what: string, fallback?: number): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    if (fallback !== undefined) return fallback;
    fail(`${what} is not a finite number`);
  }
  return v as number;
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

/** Warn about the legacy lift once per session, not once per load. */
let warnedMigration = false;

function warnLegacy(version: number): void {
  if (warnedMigration) return;
  warnedMigration = true;
  console.warn(
    `modulation: migrating a v${version} mapping file. Anchors, k-means centres, ` +
      'per-anchor presets, temperature and distanceScale are gone (plan.md Revision 3) ' +
      'and have been discarded. The palette and render blocks were kept; depths, ' +
      'speeds and slew rates are back at the v3 defaults.',
  );
}

function boundary(v: unknown): BoundaryOptions {
  const o = (v ?? {}) as Record<string, unknown>;
  return {
    enabled: o['enabled'] !== false,
    snapFraction: num(o['snapFraction'], 'boundary.snapFraction', 0.6),
    respawnFraction: num(o['respawnFraction'], 'boundary.respawnFraction', 0.12),
  };
}

function slewRates(v: unknown): SlewRates {
  const o = (v ?? {}) as Record<string, unknown>;
  return {
    fast: num(o['fast'], 'slew.fast', DEFAULT_SLEW.fast),
    medium: num(o['medium'], 'slew.medium', DEFAULT_SLEW.medium),
    slow: num(o['slow'], 'slew.slow', DEFAULT_SLEW.slow),
  };
}

function groupDepth(v: unknown): Record<ModGroup, number> {
  const o = (v ?? {}) as Record<string, unknown>;
  const out = defaultGroupDepth();
  for (const g of MOD_GROUPS) {
    if (o[g] !== undefined) out[g] = num(o[g], `groupDepth.${g}`);
  }
  return out;
}

export function parseModulation(text: string): ModulationConfig {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    fail(`not valid JSON (${(err as Error).message})`);
  }
  const o = raw as Record<string, unknown>;
  if (!o || typeof o !== 'object') fail('top level is not an object');
  const version = o['version'];
  if (version !== 3 && version !== 2 && version !== 1) {
    fail(`unsupported version ${String(version)}`);
  }
  const speciesCount = count(o['speciesCount'], 'speciesCount', MAX_SPECIES_COUNT);

  // v1/v2: everything except palette and render described the anchor system.
  // Rebuild the rest from defaults rather than half-translating it.
  if (version !== 3) {
    warnLegacy(version as number);
    return {
      version: MODULATION_VERSION,
      speciesCount,
      palette: palette(o['palette'], speciesCount, 'palette'),
      render: renderBlock(o['render'], 'render'),
      enabled: true,
      depth: 1,
      groupDepth: defaultGroupDepth(),
      responseSpeed: 1,
      slew: { ...DEFAULT_SLEW },
      boundary: boundary(o['boundary']),
    };
  }

  return {
    version: MODULATION_VERSION,
    speciesCount,
    palette: palette(o['palette'], speciesCount, 'palette'),
    render: renderBlock(o['render'], 'render'),
    enabled: o['enabled'] !== false,
    depth: num(o['depth'], 'depth', 1),
    groupDepth: groupDepth(o['groupDepth']),
    responseSpeed: num(o['responseSpeed'], 'responseSpeed', 1),
    slew: slewRates(o['slew']),
    boundary: boundary(o['boundary']),
  };
}

/**
 * A config authored for a different K cannot be applied field-for-field (the
 * palette is per species). The caller falls back to defaults instead of guessing.
 */
export function modulationFits(cfg: ModulationConfig, speciesCount: number): boolean {
  return cfg.speciesCount === speciesCount;
}

export function saveModulationLocal(cfg: ModulationConfig): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, serializeModulation(cfg));
    return true;
  } catch {
    return false; // private mode / quota — autosave is a convenience, not a contract
  }
}

export function loadModulationLocal(): ModulationConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return parseModulation(raw);
  } catch (err) {
    console.warn('modulation: ignoring unusable autosave —', err);
    return null;
  }
}

export function clearModulationLocal(): void {
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
