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
 *
 * Version 4 (Revision 4) adds per-driver gains and the stem-follow block, and
 * retires the `brightness` group depth. A v3 file is a *lossless* upgrade in the
 * other direction: everything it carries still means what it meant, so it is
 * kept whole, the gains default to 1 (nothing muted) and stem-follow adopts the
 * shipped defaults. One warning, once per session.
 *
 * Version 5 (palette v2) is lossless in the same way, and narrower: only the
 * palette block changed shape. A v4 palette `{colors, saturation, brightness}`
 * *is* a v5 palette in custom mode with a zero hue shift and a zero cycle rate,
 * and `paletteHex` short-circuits exactly that case to the authored hex string
 * with no colour-space conversion at all. A file the user tuned before palette
 * v2 therefore renders bit-identically after loading — which is the whole
 * requirement, since the autosaves and the committed `modulation.json` files are
 * where this project's art direction actually lives.
 *
 * ## Reading is driven by the defaults, not by a written list
 *
 * Every scalar and nested block below is read by `readInto`, which walks the
 * *defaults object* it is given and copies whatever the raw blob has for each of
 * its keys. That is deliberate and load bearing: the write side is
 * `JSON.stringify`, which is total, so a reader that enumerated fields by hand
 * would drop any field a future feature added — silently, on the way back in,
 * with nothing failing anywhere. That defect shipped at least once (the accents'
 * arc), so the enumeration is gone.
 *
 * What is still hand-written, and why: `speciesCount` (it sizes allocations and
 * must be checked before anything allocates from it), the version dispatch, and
 * every **array** — `colors` is per species and re-derived in arc mode,
 * `driverGains` is the track's driver count. Arrays have semantics a generic
 * walker cannot know; scalars do not, which is why scalars are where the
 * forgetting happened.
 */
import { defaultBoundary, MODULATION_VERSION, type ModulationConfig } from './types.ts';
import type { ModGroup } from './modspec.ts';
import { oneOf, range, readInto, type RuleTree } from './read-into.ts';
import {
  defaultImpulseConfig,
  IMPULSE_RULES,
  type ImpulseConfig,
} from '../sim/impulses.ts';
// The one constant the loader and the live slider must agree on: raising it in
// one place only would clip loaded files below what the workbench allows.
import { MAX_DRIVER_GAIN } from './modulation.ts';
import { defaultStemFollow } from './stemfollow.ts';
import {
  PALETTE_SPACES,
  syncPaletteColors,
  FALLBACK_COLOR,
  MAX_HUE_RATE_DEG_PER_SEC,
  type Palette,
} from '../sim/palette.ts';
// `defaultPalette` is physarum's shipped art direction and is the only thing
// here that still knows a specific sim. It is the fallback a hand-trimmed file
// falls back *to*, not something the format depends on; a second sim's loader
// would want its own, and that is the next seam to cut if one appears.
import { defaultPalette } from '../sim/physarum/config.ts';
import { defaultRenderConfig, TONEMAPS, type RenderConfig } from '../sim/render/config.ts';
import { DEFAULT_SLEW } from './slew.ts';

/**
 * One autosave slot per sim. Physarum keeps the original unsuffixed key so every
 * existing browser autosave survives the day a second simulation lands; anything
 * else gets its own, because the two files describe different θ and silently
 * sharing a slot would mean whichever sim you opened last ate the other's tuning
 * session.
 */
const storageKey = (sim: string): string => (sim === 'physarum' ? 'lmt.mapping' : `lmt.mapping.${sim}`);

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
    decay: 0.6,
  };
}

/**
 * `base` is structural rather than a `PhysarumConfig`: the three fields below are
 * all this needs, and every modulatable sim has them (`ModTargetConfig`). Call
 * sites still hand it their concrete config.
 */
export function defaultModulationConfig(
  base: { speciesCount: number; palette: Palette; render: RenderConfig },
  sim = 'physarum',
): ModulationConfig {
  return {
    version: MODULATION_VERSION,
    sim,
    speciesCount: base.speciesCount,
    // Shared by reference with the live config: a palette edit in the workbench
    // is an edit to the thing that gets serialised, with no sync step to forget.
    palette: base.palette,
    render: base.render,
    enabled: true,
    depth: 1,
    groupDepth: defaultGroupDepth(),
    // Empty rather than filled: the driver count depends on the track's latent
    // width, which nothing here knows. The Modulator sizes and normalises this
    // on setConfig and writes the real array back.
    driverGains: [],
    stemFollow: defaultStemFollow(),
    // The event lane's tuning. Present here — rather than optional-and-absent
    // like `extras` — because it is sim-agnostic and always exists: the engine
    // is constructed for every substrate. `buildSimBundle` replaces this with
    // the live engine's own object by reference immediately after construction,
    // the same trick the palette and the render block use, so a panel edit is an
    // edit to the thing that gets serialised with no sync step to forget.
    impulses: defaultImpulseConfig(),
    responseSpeed: 1,
    slew: { ...DEFAULT_SLEW },
    boundary: defaultBoundary(),
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

/**
 * Bounds for the palette blocks that have them.
 *
 * Hue angles are absent on purpose — any real angle is a legal angle, and a file
 * saying 720 means the same wheel position as 0. S and L are here because they
 * are coordinates with ends, and a file with `light: 500` should render as white
 * rather than fail to load.
 */
const ARC_RULES: RuleTree = { sat: range(0, 100), light: range(0, 100) };

const PALETTE_RULES: RuleTree = {
  mode: oneOf(['arc', 'custom']),
  // An unknown or absent space falls back to HSLuv, which is what every palette
  // written before the space was selectable meant.
  space: oneOf(PALETTE_SPACES),
  arc: ARC_RULES,
  accentArc: ARC_RULES,
  hueRateDegPerSec: range(-MAX_HUE_RATE_DEG_PER_SEC, MAX_HUE_RATE_DEG_PER_SEC),
};

/** The one render field with a closed vocabulary; the rest are free scalars. */
const RENDER_RULES: RuleTree = { grade: { tonemap: oneOf(TONEMAPS) } };

/**
 * Palette v2, and the v1 lift.
 *
 * A v4-and-earlier palette has no `mode`, so it reads as `custom` — which is
 * precisely what it was — with the two hue fields at zero. Nothing about how its
 * colours reach a pixel changes.
 *
 * Everything except `colors` is `readInto`'s: the arcs, the mode, the space and
 * the four global scalars all round trip because they are in `defaultPalette`'s
 * return value, and a field added to `Palette` tomorrow will too. `colors` stays
 * bespoke because it is per species and, in arc mode, a derived cache.
 */
function palette(v: unknown, speciesCount: number): Palette {
  const o = (v ?? {}) as Record<string, unknown>;
  const fallback = defaultPalette(speciesCount);
  // `defaultPalette` is `customPalette`, i.e. mode custom, space hsluv, the two
  // shipped arcs, and the four v2 scalars at their neutral values — which is
  // field for field what the hand-written reader used to fall back to. So the
  // walk below is not merely equivalent to it, it is the same numbers stated
  // once instead of nine times.
  const out = readInto(defaultPalette(speciesCount), o, PALETTE_RULES);

  const raw = Array.isArray(o['colors']) ? (o['colors'] as unknown[]) : [];
  out.colors = fallback.colors.map((c, i) => (typeof raw[i] === 'string' ? (raw[i] as string) : c));
  // In arc mode the stored hexes are a cache, and a hand-edited or stale one
  // must not survive the load. In custom mode this only pads a short list, which
  // the map above has already done.
  syncPaletteColors(out, speciesCount, (i) => fallback.colors[i] ?? FALLBACK_COLOR);
  return out;
}

/**
 * Phase 7's render block. Every field is optional and falls back to the shipped
 * default — the grade is art direction, so a file that predates it (or that a
 * human trimmed by hand) should adopt the current defaults rather than fail.
 */
function renderBlock(v: unknown): RenderConfig {
  return readInto(defaultRenderConfig(), v, RENDER_RULES);
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
      'speeds and slew rates are back at the current defaults.',
  );
}

/** Same policy, different story: v3 → v4 loses nothing, so it says so. */
let warnedV3 = false;

function warnV3(): void {
  if (warnedV3) return;
  warnedV3 = true;
  console.warn(
    'modulation: migrating a v3 mapping file to v4 (plan.md Revision 4). Everything ' +
      'it carried was kept; driver gains default to 1 (no driver muted) and the ' +
      'stem-follow brightness lane adopts the shipped defaults. The `brightness` ' +
      'group depth no longer exists — brightness left the projections for stem-follow.',
  );
}

/** v4 → v5 is narrower still: the palette block grew fields, nothing changed meaning. */
let warnedV4 = false;

function warnV4(): void {
  if (warnedV4) return;
  warnedV4 = true;
  console.warn(
    'modulation: migrating a v4 mapping file to v5 (palette v2). The palette is now ' +
      'in custom mode with a zero global hue shift and no hue cycle, which is exactly ' +
      'what a v4 palette meant — the rendered colours are unchanged.',
  );
}

/**
 * Gains are clamped, not rejected: this is a slider value, and a file with a
 * silly number in it should still load with that driver simply pinned to the end
 * of its range. Non-numbers fall back to 1, i.e. "not muted".
 */
function driverGains(v: unknown): number[] {
  if (!Array.isArray(v)) return [];
  return v.map((x) =>
    typeof x === 'number' && Number.isFinite(x) ? Math.min(Math.max(x, 0), MAX_DRIVER_GAIN) : 1,
  );
}

const STEM_FOLLOW_RULES: RuleTree = {
  floor: range(0, 1),
  curve: range(0.01, Infinity),
  smoothingMs: range(1, Infinity),
};

/**
 * The impulse lane, persisted with the mapping since 2026-08-13.
 *
 * It was the largest block of settings in the workbench that had **no** save
 * path at all: the events folder bound `ImpulseConfig` directly and nothing ever
 * wrote it anywhere, so every splash radius and decay τ died on refresh. It
 * belongs here rather than in the sim's `extras` because the engine is
 * sim-agnostic — but the autosave slot is per sim, so each substrate still keeps
 * its own event tuning, which is what you want when "deposit burst" means
 * something in physarum and nothing in plife.
 *
 * Optional, like `extras`, so absent is simply "defaults" and no version bump is
 * owed: a v5 file written before this exists loads with the shipped responses.
 */
function impulses(v: unknown): ImpulseConfig {
  return readInto(defaultImpulseConfig(), v, IMPULSE_RULES);
}

/**
 * The opaque per-sim block, carried through unexamined. The only thing checked
 * is that it is a plain object, because that is the whole of the contract the
 * mapping layer offers: anything else is dropped and the sim's `applyExtras`
 * sees `undefined`, which it must already handle. Validating the *contents*
 * here would mean this file knowing plife's schema, which is exactly what the
 * channel exists to avoid.
 */
function extras(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : undefined;
}

/**
 * The whole mapping's rule tree, mirroring `ModulationConfig`'s shape.
 *
 * Only fields that need a *bound* appear here. A field with no entry still round
 * trips — it is read because it is a key of the defaults, not because it is
 * named here — so this table is about clamping corrupt input, never about
 * enrolling a field in persistence. That distinction is the whole point: there
 * is no list you can forget to add a new setting to.
 *
 * `groupDepth` needs no entry at all: `defaultGroupDepth` is keyed by
 * `MOD_GROUPS`, so walking its keys walks the group vocabulary — a group added
 * to `modspec.ts` persists the moment it has a default, and a retired one (v3's
 * `brightness`) is simply not a key here and so is never read back.
 */
const MODULATION_RULES: RuleTree = {
  palette: PALETTE_RULES,
  render: RENDER_RULES,
  stemFollow: STEM_FOLLOW_RULES,
  impulses: IMPULSE_RULES,
  boundary: { snapFraction: range(0, 1), respawnFraction: range(0, 1) },
};

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
  if (version !== 5 && version !== 4 && version !== 3 && version !== 2 && version !== 1) {
    fail(`unsupported version ${String(version)}`);
  }
  const speciesCount = count(o['speciesCount'], 'speciesCount', MAX_SPECIES_COUNT);
  // Every file written before a second simulation existed is a physarum file,
  // so an absent discriminator is not ambiguous — it is an answer.
  const sim = typeof o['sim'] === 'string' ? o['sim'] : 'physarum';

  // The skeleton every branch below reads into. Its palette and render blocks are
  // throwaway defaults that the walk will fill and the two lines after it will
  // then replace outright: both need handling `readInto` cannot do — a per-species
  // colour array, and `syncPaletteColors` re-deriving that array in arc mode —
  // so they are built by their own readers and assigned over the walk's result.
  const skeleton = (): ModulationConfig =>
    defaultModulationConfig(
      { speciesCount, palette: defaultPalette(speciesCount), render: defaultRenderConfig() },
      sim,
    );

  // v1/v2: everything except palette and render described the anchor system.
  // Rebuild the rest from defaults rather than half-translating it. `boundary` is
  // the one block read from the file, because it outlived the anchors.
  if (version === 1 || version === 2) {
    warnLegacy(version);
    const legacy = skeleton();
    readInto(legacy.boundary, o['boundary'], MODULATION_RULES['boundary'] as RuleTree);
    legacy.palette = palette(o['palette'], speciesCount);
    legacy.render = renderBlock(o['render']);
    legacy.impulses = impulses(o['impulses']);
    return legacy;
  }

  // v3 → v4 keeps everything. The only fields it cannot have are the two new
  // ones, which take their defaults; `groupDepth.brightness`, if present, is
  // simply not read, because that group no longer exists.
  if (version === 3) warnV3();
  if (version === 4) warnV4();

  // One recursive walk covers every scalar and every nested block there is:
  // `enabled`, `depth`, `responseSpeed`, `groupDepth`, `slew`, `stemFollow`,
  // `boundary` and `impulses`. Adding a field to `ModulationConfig` — which
  // forces it into `defaultModulationConfig` — is all it takes to persist it.
  const cfg = readInto(skeleton(), o, MODULATION_RULES);

  // The five exceptions, and every one of them is an exception for a stated
  // reason rather than by omission.
  //
  // `version` because the walk copies the *file's* number and this parser's
  // output is always current; `speciesCount` because it sizes allocations and
  // was validated up front; `palette` and `render` per the skeleton note; and
  // `driverGains` because it is an array whose length is the track's driver
  // count, not the config's business.
  cfg.version = MODULATION_VERSION;
  cfg.speciesCount = speciesCount;
  cfg.palette = palette(o['palette'], speciesCount);
  cfg.render = renderBlock(o['render']);
  cfg.driverGains = driverGains(o['driverGains']);

  // Omitted rather than written as `undefined`: `extras` is optional and the
  // absent case has to stay absent, so a sim with nothing to store never adds
  // a null key to every file it saves.
  const extra = extras(o['extras']);
  if (extra === undefined) delete cfg.extras;
  else cfg.extras = extra;
  return cfg;
}

/**
 * A config authored for a different K cannot be applied field-for-field (the
 * palette is per species), and one authored against a different *simulation*
 * cannot be applied at all — θ is a different vector. The caller falls back to
 * defaults instead of guessing.
 */
export function modulationFits(
  cfg: ModulationConfig,
  speciesCount: number,
  sim = 'physarum',
): boolean {
  return cfg.speciesCount === speciesCount && cfg.sim === sim;
}

export function saveModulationLocal(cfg: ModulationConfig, sim = 'physarum'): boolean {
  try {
    localStorage.setItem(storageKey(sim), serializeModulation(cfg));
    return true;
  } catch {
    return false; // private mode / quota — autosave is a convenience, not a contract
  }
}

export function loadModulationLocal(sim = 'physarum'): ModulationConfig | null {
  try {
    const raw = localStorage.getItem(storageKey(sim));
    if (!raw) return null;
    return parseModulation(raw);
  } catch (err) {
    console.warn('modulation: ignoring unusable autosave —', err);
    return null;
  }
}

export function clearModulationLocal(sim = 'physarum'): void {
  try {
    localStorage.removeItem(storageKey(sim));
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
