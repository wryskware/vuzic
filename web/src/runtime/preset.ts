/**
 * Preset strings: a whole tuned look as one pasteable token.
 *
 * `lmt1.` + base64url( deflate-raw( canonical JSON ) ). See
 * `docs/handoffs/preset-strings-v1.md`.
 *
 * ## What a preset is, and is not
 *
 * A `PresetV1` is an `ExportRecipe` minus the four fields that describe an
 * *export* rather than a *look*: `rendererBuild` (a server capability), `output`
 * (resolution and encoder), `presentation` (a constant in the single-visual
 * recipe), and `particleBudget` (already implied by the simulation config, and
 * recomputed by `recipeFromPreset` — carrying it would be a second copy the
 * validator would then have to prove equal to the first).
 *
 * `track` is the one field that changes shape: a recipe carries
 * `{ id, contentVersion }` because it must reproduce a render bit for bit, while
 * a preset carries the bare id as an **advisory hint**. Music never travels in a
 * preset — the string is the look, and the loader honours the hint only if it
 * happens to have that track locally.
 *
 * ## Why the codec, and why deflate
 *
 * `serializeExportRecipe` already emits canonical, stable-key-order JSON, so the
 * only thing between "a look" and "a string a person can paste" is compression
 * and a URL-safe alphabet. Both exist natively: `CompressionStream` and `btoa`.
 * Nothing is invented here, and no dependency is added. Compression is not a
 * nicety — a K=64 particle-life config is three float matrices of 4096 entries
 * each, and the JSON runs to hundreds of kilobytes of highly repetitive digits,
 * which is exactly what deflate is for.
 *
 * Numbers are **not** rounded before compression. It was allowed by the brief
 * and refused: a preset says "the look I saved", and quantising a slider on the
 * way out means a loaded preset is a near miss rather than the thing. The
 * saving would have been a few percent after deflate.
 *
 * ## Why the validators are imported and not restated
 *
 * A preset carries the same blocks as a recipe, so it gets the same strict,
 * fail-loudly reading of them (`./recipe.ts` owns `validateSimulation`,
 * `validateModulation`, `validateRender`, `validateImpulse` and the JSON walk).
 * A second parser for the same bytes is a second set of bugs, and it would drift
 * the first time a block grew a field. The only thing this module validates on
 * its own is the *container*: its key set, its version, and its seed fields.
 */
import { DEFAULT_MASTERING_PEAK_NITS, DEFAULT_PAPER_WHITE_NITS } from '../export/hdr.ts';
import { requiredEncoder } from '../export/profiles.ts';
import type { ImpulseConfig } from '../sim/impulses.ts';
import type { RenderConfig } from '../sim/render/config.ts';
import {
  assertJsonValue,
  boolean,
  canonical,
  EXPORT_RECIPE_VERSION,
  fail,
  identifier,
  integer,
  keysRequired,
  MAX_RECIPE_JSON_CHARS,
  object,
  TOP_LEVEL_KEYS,
  validateImpulse,
  validateModulation,
  validateModulationBase,
  validateRender,
  validateSimulation,
  withLabel,
  type ExportProfile,
  type ExportRecipe,
  type RecipeModulationConfig,
  type SimulationBaseConfig,
} from './recipe.ts';

/** The envelope version. Inner blocks keep their own (modulation v5, …). */
export const PRESET_VERSION = 1 as const;

/**
 * The token prefix, which is also the envelope's version marker. A future
 * `lmt2.` is a different container and is rejected here rather than guessed at;
 * the blocks inside ride their own existing migration machinery.
 */
export const PRESET_PREFIX = 'lmt1.';

const PRESET_LABEL = 'preset: ';

/** Matches a token wherever it sits — bare, or inside a pasted URL. */
const TOKEN_PATTERN = /lmt1\.[A-Za-z0-9_-]+/;

/**
 * A hostile string must not be allowed to cost an inflate at all. Two bounds,
 * because they stop different things: this one caps the work, and
 * `MAX_RECIPE_JSON_CHARS` (reused from the recipe, deliberately) caps what the
 * inflate is allowed to *expand to* — the zip-bomb case, where a tiny token
 * describes gigabytes.
 */
export const MAX_PRESET_TOKEN_CHARS = 2 * 1024 * 1024;

/** What the four dropped recipe fields become when a preset is rehydrated. */
export const PRESET_RENDERER_BUILD = 'preset (not an export capture)';
const PRESET_OUTPUT_PROFILE: ExportProfile = 'av1-sdr-debug-1080p120';
/** A preset with no track hint still has to name one for the recipe's schema. */
export const PRESET_TRACK_UNKNOWN = 'unknown';

export interface PresetV1 {
  version: typeof PRESET_VERSION;
  /** Concrete ModTarget.simId — physarum, plife, or one concrete VizFX visual. */
  sim: string;
  /** Always present, always replayed: half of "the look I saved" is the seed. */
  seed: number;
  seedPinned: boolean;
  /** Advisory only. Absent, or a track id the loader may or may not have. */
  track?: string;
  simulation: SimulationBaseConfig;
  modulation: RecipeModulationConfig;
  modulationBase: number[];
  impulses: ImpulseConfig;
  render: RenderConfig;
}

/** The dropped fields, named once. See the module note for why each goes. */
const DROPPED_RECIPE_KEYS: readonly string[] = [
  'rendererBuild',
  'output',
  'presentation',
  'particleBudget',
];

/**
 * Derived from the recipe's own key list rather than written out again, so a
 * field added to the recipe is a field a preset carries — or a deliberate line
 * in `DROPPED_RECIPE_KEYS` — and never something silently missing from one of
 * the two documents.
 */
export const PRESET_KEYS: readonly string[] = [
  'version',
  ...TOP_LEVEL_KEYS.filter((key) => key !== 'version' && !DROPPED_RECIPE_KEYS.includes(key)),
];

/** Everything but the advisory track hint. */
const PRESET_REQUIRED_KEYS: readonly string[] = PRESET_KEYS.filter((key) => key !== 'track');

// ── format ───────────────────────────────────────────────────────────────────

export function validatePresetV1(value: unknown): asserts value is PresetV1 {
  withLabel(PRESET_LABEL, () => validatePresetBody(value));
}

function validatePresetBody(value: unknown): void {
  assertJsonValue(value);
  const preset = object(value, '$');
  keysRequired(preset, PRESET_KEYS, PRESET_REQUIRED_KEYS, '$');
  if (preset['version'] !== PRESET_VERSION) {
    fail('$.version', `has unsupported version ${String(preset['version'])}`);
  }
  const sim = identifier(preset['sim'], '$.sim');
  integer(preset['seed'], '$.seed', 0, 0xffff_ffff);
  boolean(preset['seedPinned'], '$.seedPinned');
  // Present-but-absent is the whole point of the hint: a preset made on a
  // machine with a private track should still carry its look.
  if (Object.hasOwn(preset, 'track')) identifier(preset['track'], '$.track');

  // `null` budget: a preset has no `$.particleBudget` for the simulation block
  // to agree with, because `recipeFromPreset` derives it. Everything else about
  // the block is read exactly as a recipe's is.
  const speciesCount = validateSimulation(preset['simulation'], sim, null);
  const simulation = object(preset['simulation'], '$.simulation');
  validateModulation(preset['modulation'], sim, speciesCount, simulation['palette']);
  validateModulationBase(preset['modulationBase']);
  validateImpulse(preset['impulses'], '$.impulses');
  validateRender(preset['render'], '$.render');
}

/** Canonical and compact: this one is a transport, not a diffable file. */
export function serializePreset(preset: PresetV1): string {
  validatePresetV1(preset);
  return JSON.stringify(canonical(preset));
}

export function parsePreset(text: string): PresetV1 {
  return withLabel(PRESET_LABEL, () => {
    if (text.length > MAX_RECIPE_JSON_CHARS) {
      fail('$', `exceeds ${MAX_RECIPE_JSON_CHARS} serialized characters`);
    }
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch (error) {
      fail('$', `is not valid JSON (${(error as Error).message})`);
    }
    validatePresetV1(value);
    return value;
  });
}

// ── recipe ⇄ preset ──────────────────────────────────────────────────────────

/**
 * Trim a captured recipe to a preset.
 *
 * Cloned, never aliased, for the reason `captureExportRecipe` clones: the caller
 * hands us live objects the frame loop is still writing to, and a preset that
 * shared one would encode whatever the world looked like when the string was
 * finally serialized rather than when it was asked for.
 */
export function presetFromRecipe(recipe: ExportRecipe): PresetV1 {
  const preset: PresetV1 = {
    version: PRESET_VERSION,
    sim: recipe.sim,
    seed: recipe.seed,
    seedPinned: recipe.seedPinned,
    simulation: structuredClone(recipe.simulation),
    modulation: structuredClone(recipe.modulation),
    modulationBase: Array.from(recipe.modulationBase),
    impulses: structuredClone(recipe.impulses),
    render: structuredClone(recipe.render),
  };
  // Only a real id travels. `PRESET_TRACK_UNKNOWN` is what a *rehydrated*
  // preset without a hint says, and re-encoding one must not turn that stub
  // into a hint pointing at a track called "unknown".
  if (recipe.track.id !== PRESET_TRACK_UNKNOWN) preset.track = recipe.track.id;
  return preset;
}

function derivedParticleBudget(sim: string, simulation: SimulationBaseConfig): number {
  const config = simulation as Record<string, unknown>;
  if (sim === 'physarum') return config['maxAgents'] as number;
  if (sim === 'plife') return (config['budget'] as { cap: number }).cap;
  // A vizfx visual has no particles; the recipe says 0 and the validator agrees.
  return 0;
}

/**
 * Rehydrate a preset into the recipe the whole apply path already speaks.
 *
 * The four dropped fields come back as honest stubs rather than plausible
 * values: `rendererBuild` says what it is instead of borrowing a build id that
 * would then travel into an export, and `output` is a valid profile/encoder pair
 * that nothing on the load path reads. This is the same trade `ui/profiles.ts`
 * documents — it buys the entire existing boot sequence
 * (`buildSimBundleFromRecipe`) rather than growing a second reading of what a
 * saved look means.
 */
export function recipeFromPreset(preset: PresetV1): ExportRecipe {
  validatePresetV1(preset);
  const recipe: ExportRecipe = {
    version: EXPORT_RECIPE_VERSION,
    rendererBuild: PRESET_RENDERER_BUILD,
    track: { id: preset.track ?? PRESET_TRACK_UNKNOWN, contentVersion: PRESET_TRACK_UNKNOWN },
    sim: preset.sim,
    seed: preset.seed,
    seedPinned: preset.seedPinned,
    simulation: structuredClone(preset.simulation),
    modulation: structuredClone(preset.modulation),
    modulationBase: Array.from(preset.modulationBase),
    impulses: structuredClone(preset.impulses),
    render: structuredClone(preset.render),
    particleBudget: derivedParticleBudget(preset.sim, preset.simulation),
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: PRESET_OUTPUT_PROFILE,
      encoder: requiredEncoder(PRESET_OUTPUT_PROFILE),
      paperWhiteNits: DEFAULT_PAPER_WHITE_NITS,
      masteringPeakNits: DEFAULT_MASTERING_PEAK_NITS,
    },
  };
  return recipe;
}

// ── codec ────────────────────────────────────────────────────────────────────

/**
 * `withLabel` for a promise. Not the same function: a `try` around a call that
 * returns a pending promise catches nothing, so an inflate failure would arrive
 * unlabelled — and the label is the difference between "$ is not valid
 * base64url" and a message that says what kind of document was being read.
 */
async function withLabelAsync<T>(label: string, run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    const message = (error as Error).message;
    if (message.startsWith(label)) throw error;
    throw new Error(`${label}${message}`);
  }
}


function source(bytes: Uint8Array<ArrayBuffer>): ReadableStream<BufferSource> {
  // Enqueued up front rather than written through a writer: a writer whose
  // `write` is not awaited before the reader starts deadlocks on backpressure
  // for exactly the large configs this codec exists for.
  // Typed `BufferSource` to match what a compression stream's writable half
  // accepts; the readable half of the pipe is `Uint8Array` either way.
  return new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** Read a stream to one buffer, refusing to be the thing that runs out of memory. */
async function drain(
  stream: ReadableStream<Uint8Array>,
  cap: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > cap) {
      await reader.cancel();
      fail('$', `expands past the ${cap}-byte limit`);
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function toBase64Url(bytes: Uint8Array): string {
  // Chunked: `String.fromCharCode(...bytes)` on a few hundred kilobytes is a
  // call with a few hundred thousand arguments, which overflows the stack.
  let binary = '';
  const STEP = 0x8000;
  for (let i = 0; i < bytes.length; i += STEP) {
    binary += String.fromCharCode(...bytes.subarray(i, i + STEP));
  }
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

function fromBase64Url(text: string): Uint8Array<ArrayBuffer> {
  const padded = text.replaceAll('-', '+').replaceAll('_', '/');
  let binary: string;
  try {
    binary = atob(padded);
  } catch {
    fail('$', 'is not valid base64url');
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function encodePreset(preset: PresetV1): Promise<string> {
  const json = new TextEncoder().encode(serializePreset(preset));
  const deflated = await drain(
    source(json).pipeThrough(new CompressionStream('deflate-raw')),
    MAX_RECIPE_JSON_CHARS,
  );
  return PRESET_PREFIX + toBase64Url(deflated);
}

/**
 * The exact inverse, strict at every step: unknown prefix, invalid base64url,
 * inflate error, oversize payload, JSON error, block validator. Each throws a
 * message a person can read, and none of them falls back to anything.
 */
export async function decodePreset(token: string): Promise<PresetV1> {
  return withLabelAsync(PRESET_LABEL, async () => {
    if (!token.startsWith(PRESET_PREFIX)) {
      fail('$', `does not start with "${PRESET_PREFIX}"`);
    }
    if (token.length > MAX_PRESET_TOKEN_CHARS) {
      fail('$', `exceeds ${MAX_PRESET_TOKEN_CHARS} encoded characters`);
    }
    const body = token.slice(PRESET_PREFIX.length);
    if (body.length === 0 || !/^[A-Za-z0-9_-]+$/.test(body)) {
      fail('$', 'is not valid base64url');
    }
    const bytes = fromBase64Url(body);
    let inflated: Uint8Array<ArrayBuffer>;
    try {
      inflated = await drain(
        source(bytes).pipeThrough(new DecompressionStream('deflate-raw')),
        MAX_RECIPE_JSON_CHARS,
      );
    } catch (error) {
      const message = (error as Error).message;
      // The size failure is this module's own and already says what happened.
      if (message.includes('expands past')) throw error;
      fail('$', `could not be decompressed (${message})`);
    }
    return parsePreset(new TextDecoder().decode(inflated));
  });
}

// ── links and pasted text ────────────────────────────────────────────────────

/**
 * Pull a token out of whatever was pasted — the bare string, or a whole URL
 * with one in its fragment. Accepting both is not politeness: "copy link" and
 * "copy string" both exist, and a person pasting the wrong one into the wrong
 * box should get their look rather than an error about a format they did not
 * know they had chosen.
 */
export function extractPresetToken(input: string): string | null {
  return TOKEN_PATTERN.exec(input.trim())?.[0] ?? null;
}

/** A fragment, never a query: `#p=` never reaches a server and needs no host. */
export function presetLink(base: string, token: string): string {
  return `${base}#p=${token}`;
}

/** The token in a URL's fragment, if it has one. */
export function presetTokenFromHash(hash: string): string | null {
  const match = /^#?p=(lmt1\.[A-Za-z0-9_-]+)$/.exec(hash);
  return match?.[1] ?? null;
}

// ── the boot decision, extracted pure ────────────────────────────────────────

export type BootPresetSource = 'pending' | 'fragment' | 'none';

/**
 * Which of the two boot channels wins.
 *
 * The pending slot beats the fragment because the slot is a deliberate act
 * *inside* this session ("load that one") while the fragment is how the tab was
 * opened, possibly reloads ago. Both are consumed by their reader — the slot
 * unconditionally, the fragment only on success — but that is the caller's job;
 * this function only decides.
 */
export function chooseBootPreset(
  pending: string | null,
  fragment: string | null,
): { token: string; source: Exclude<BootPresetSource, 'none'> } | { token: null; source: 'none' } {
  if (pending !== null) return { token: pending, source: 'pending' };
  if (fragment !== null) return { token: fragment, source: 'fragment' };
  return { token: null, source: 'none' };
}

export interface PresetApplyPlan {
  sim: string;
  seed: number;
  /** What `setPinnedSeed`/`syncUrlSeed` get: the seed, or null to unpin. */
  pinSeed: number | null;
  /** The advisory track, or null when the preset named none. */
  trackHint: string | null;
}

/**
 * What applying a preset means, as data.
 *
 * The seed rule, decided 2026-08-13 and stated here because it is the one thing
 * about presets that is not obvious: the seed is **always** replayed, pinned or
 * not. A particle-life matrix is generated from it, so a preset loaded onto a
 * fresh seed is a different world wearing the saved colours. Pinning is a
 * separate question — "should *live* runs stay on this seed" — and its state
 * travels in the preset so a reload does what the preset said.
 */
export function presetApplyPlan(preset: PresetV1): PresetApplyPlan {
  return {
    sim: preset.sim,
    seed: preset.seed,
    pinSeed: preset.seedPinned ? preset.seed : null,
    trackHint: preset.track ?? null,
  };
}
