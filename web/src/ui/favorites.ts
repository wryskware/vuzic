/**
 * Seed favorites — one keypress that says "this world is good" or "this world is
 * not", stored where it survives the session.
 *
 * Roadmap phase 2 item 4. Most seeds are mediocre and a few are extraordinary,
 * and until now the only record of that was the human's memory of a number that
 * had already been rerolled away. This is the pool.
 *
 * ## Two consumers, and why the record is shaped the way it is
 *
 * The **human** wants to get back to a world they liked. That needs whole state,
 * because a plife world is the seed *and* the generation settings *and* the θ
 * centre *and* the palette — see `./profiles.ts` for the same argument at
 * length. So a like carries a whole `ExportRecipe` and returns through the
 * profile library's stage-then-reload path, which already exists and is already
 * the one construction order `main.ts` performs.
 *
 * The **far-future model** ("learn what good interaction matrices look like")
 * wants (matrix, verdict) pairs, in bulk, as JSONL. It does not want the
 * palette, and it certainly does not want 20 kB per row. What it needs is the
 * matrix, and the matrix is not stored: `seedMatrixBase(seed, cfg)` is a pure
 * function of `seed`, `matrixGen` and `speciesCount`, all three of which *are*
 * stored. That is the same trade `explore/log.ts` makes when it stores `genSeed`
 * instead of nine θ vectors — record what determines the sample, not the sample.
 *
 * ## Why a dislike is smaller than a like, and why that is not an oversight
 *
 * A dislike stores no recipe. Nobody has ever wanted to return to a world they
 * just rejected, and dislikes are the *common* verdict by construction — the
 * whole premise of the feature is that most seeds are bad. Paying a full recipe
 * for each of them would fill a 5 MB origin budget (shared with the profile
 * library and the offline track cache) with states no human and no loader will
 * ever read. Both verdicts carry identical *model* content — seed, matrixGen,
 * speciesCount, track, timestamp — so the training-data view is symmetric; only
 * the human's return path is not, and only where it has no meaning.
 *
 * The same reasoning runs one step further at write time: if a like does not fit,
 * the recipe is dropped and the verdict is written anyway (see `writeFavorite`).
 * Losing the ability to revisit one world is much cheaper than losing the datum.
 *
 * ## Storage layout: one key per record, no index
 *
 * `lmt.fav.v1.<id>`, enumerated by prefix scan, exactly as `./profiles.ts` does
 * and for exactly the same reason: a single `lmt.favorites` array would be a
 * read-modify-write, and two tabs pressing 👍 at once would leave one verdict.
 * That bug has already been paid for once in this codebase and is not coming
 * back. `explore/log.ts` still holds one blob, and it is a running counterexample
 * rather than a precedent — it can afford to be, because explorer mode is one tab
 * by construction.
 *
 * The id sorts chronologically as a string, so a prefix scan comes back in press
 * order without parsing anything, and it is unique per (millisecond, seed,
 * in-page counter, random suffix) so no two writers can land on one key.
 *
 * localStorage dies with "clear site data", so `exportFavoritesJsonl` /
 * `importFavoritesJsonl` are the durable half. JSONL because that is what a
 * training loader streams, and because the ids are carried in the rows: importing
 * the same file twice yields the same set rather than a doubled one.
 *
 * Nothing here is a config block and nothing here rides `PLIFE_BLOCKS`. This is a
 * library of user data, like the profiles next to it, not a lane of the sim.
 */
import {
  parseExportRecipe,
  validateExportRecipe,
  type ExportRecipe,
} from '../runtime/recipe.ts';

/** Namespace for the pool. Versioned in the key, like the two logs. */
const PREFIX = 'lmt.fav.v1.';

/** Record schema version, repeated inside the row so an exported line stands alone. */
export const FAVORITE_VERSION = 1 as const;

export type FavoriteVerdict = 'like' | 'dislike';

export interface SeedFavorite {
  v: typeof FAVORITE_VERSION;
  /** `<compact ISO>-<seed hex>-<counter><random>`; also the storage key's tail. */
  id: string;
  /** ISO timestamp of the press. */
  at: string;
  verdict: FavoriteVerdict;
  /** `ModTarget.simId` — a seed means nothing without the substrate it drew. */
  sim: string;
  seed: number;
  /** Whether the seed was pinned when judged: a pinned world is one being worked on. */
  seedPinned: boolean;
  /** Needed to re-derive the matrix: the draw is K×K and `coupled()` depends on K. */
  speciesCount: number;
  /**
   * The sim's matrix-generation block (plife's `matrixGen`), verbatim, or `null`
   * for a substrate that has none. Held untyped on purpose: this is a data
   * record, it must keep whatever fields the block had *on the day it was
   * pressed*, and narrowing it to today's `MatrixGenConfig` would quietly make
   * old rows lie about the settings the human actually judged.
   */
  gen: Record<string, unknown> | null;
  /** Catalog track id — the same seed reads differently against different music. */
  track: string;
  /** Transport position at the press, seconds. Which part of the song was judged. */
  time: number;
  /** Whole state, likes only. See the module note on why a dislike has none. */
  recipe?: ExportRecipe;
}

/** What a caller supplies; `v`, `id` and `at` are stamped here. */
export type FavoriteInput = Omit<SeedFavorite, 'v' | 'id' | 'at'>;

export interface FavoriteResult {
  /** The id it was stored under, or null if nothing was stored. */
  id: string | null;
  /** Panel status line. */
  message: string;
  /** True when a like had to give up its recipe to fit. */
  degraded: boolean;
}

/**
 * Bumped on every id, so two presses inside one millisecond in one page cannot
 * collide however the random suffix falls. The suffix covers the other tab.
 */
let counter = 0;

function randomSuffix(): string {
  return Math.floor(Math.random() * 0x1_0000)
    .toString(16)
    .padStart(4, '0');
}

/**
 * A key that sorts by time and cannot be produced twice.
 *
 * The timestamp is first because a prefix scan then comes back in press order
 * for free; the seed is in it because a key you read in devtools should say
 * which world it is about; the counter and the random tail are the two
 * independent guards against a same-millisecond collision (same page, other
 * page).
 */
export function newFavoriteId(seed: number, at: string): string {
  const stamp = at.replace(/[^0-9A-Z]/g, '');
  const world = (seed >>> 0).toString(16).padStart(8, '0');
  const seq = (counter++ & 0xfff).toString(16).padStart(3, '0');
  return `${stamp}-${world}-${seq}${randomSuffix()}`;
}

/**
 * Write a fully-formed record under its own id.
 *
 * The quota path is the interesting one. A dislike is a few hundred bytes and
 * effectively always fits; a like is a whole recipe and one day will not. Rather
 * than report a failure and lose the verdict, the recipe is stripped and the row
 * is retried — the human loses the ability to revisit that one world, which is
 * the cheaper half of what the record is for. Only if the bare row also fails is
 * there nothing to be done, and then the message says what to delete.
 */
export function writeFavorite(fav: SeedFavorite): FavoriteResult {
  if (fav.recipe !== undefined) {
    // Refused here rather than at read time: a row that cannot be parsed back is
    // a row that fails at boot, where there is no panel left to say so. Same
    // contract as `saveProfileText`.
    try {
      validateExportRecipe(fav.recipe);
    } catch (err) {
      return { id: null, message: `not a usable state: ${(err as Error).message}`, degraded: false };
    }
  }
  const key = `${PREFIX}${fav.id}`;
  try {
    localStorage.setItem(key, JSON.stringify(fav));
    return { id: fav.id, message: `${fav.verdict === 'like' ? '👍' : '👎'} seed ${fav.seed}`, degraded: false };
  } catch {
    // fall through
  }
  if (fav.recipe !== undefined) {
    const bare: SeedFavorite = { ...fav };
    delete bare.recipe;
    try {
      localStorage.setItem(key, JSON.stringify(bare));
      return { id: fav.id, message: `👍 seed ${fav.seed} (verdict only — no room for the state)`, degraded: true };
    } catch {
      // fall through
    }
  }
  return { id: null, message: 'no room — export and delete some favorites', degraded: false };
}

/** Stamp and store a verdict. The one function a button calls. */
export function recordFavorite(input: FavoriteInput): FavoriteResult {
  const at = new Date().toISOString();
  return writeFavorite({ v: FAVORITE_VERSION, id: newFavoriteId(input.seed, at), at, ...input });
}

/**
 * How many rows are in the pool, without parsing any of them.
 *
 * Separate from `listFavorites` because the panel shows a count on every refresh
 * and a like carries a full recipe: parsing the library to print one number would
 * be the most expensive thing the workbench does.
 */
export function countFavorites(): number {
  let n = 0;
  try {
    for (let i = 0; i < localStorage.length; i++) {
      if (localStorage.key(i)?.startsWith(PREFIX) === true) n++;
    }
  } catch {
    return 0;
  }
  return n;
}

/**
 * Every stored row, oldest first, optionally narrowed to one sim.
 *
 * Rows are shape-checked and a failure drops that row only — the per-key layout
 * is what makes that possible, and it is why a hand-edited entry costs one
 * verdict rather than the pool.
 */
export function listFavorites(sim?: string): SeedFavorite[] {
  const out: SeedFavorite[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key === null || !key.startsWith(PREFIX)) continue;
      const fav = parseFavorite(localStorage.getItem(key));
      if (fav === null) continue;
      if (sim !== undefined && fav.sim !== sim) continue;
      out.push(fav);
    }
  } catch {
    return [];
  }
  return out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

export function readFavorite(id: string): SeedFavorite | null {
  try {
    return parseFavorite(localStorage.getItem(`${PREFIX}${id}`));
  } catch {
    return null;
  }
}

export function deleteFavorite(id: string): void {
  try {
    localStorage.removeItem(`${PREFIX}${id}`);
  } catch {
    // nothing to do
  }
}

/**
 * The state to return to, as recipe text ready for `requestProfileApply`.
 *
 * Routed through `parseExportRecipe` rather than handed over raw so that an old
 * row gets the same version lift a stored profile or an imported sidecar gets —
 * a favorite is a recipe that has been sitting in a browser across schema
 * changes, which is precisely the case `liftExportRecipe` exists for.
 */
export function favoriteRecipeText(fav: SeedFavorite): string | null {
  if (fav.recipe === undefined) return null;
  try {
    return JSON.stringify(parseExportRecipe(JSON.stringify(fav.recipe)));
  } catch {
    return null;
  }
}

/** One row per line, press order. What a training loader streams. */
export function exportFavoritesJsonl(): string {
  return listFavorites()
    .map((fav) => JSON.stringify(fav))
    .join('\n');
}

export interface ImportResult {
  added: number;
  /** Rows that were not readable as favorites, or that would not store. */
  skipped: number;
}

/**
 * Merge a JSONL file into the pool, keyed by each row's own id.
 *
 * Idempotent by construction: re-importing a file rewrites the same keys rather
 * than growing a duplicate set, which is what makes "export, clear site data,
 * import" a lossless round trip and "import the same file twice" a no-op.
 */
export function importFavoritesJsonl(text: string): ImportResult {
  let added = 0;
  let skipped = 0;
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    const fav = parseFavorite(line);
    if (fav === null || writeFavorite(fav).id === null) {
      skipped++;
      continue;
    }
    added++;
  }
  return { added, skipped };
}

/**
 * Shape-checked, not blind-cast, for `explore/log.ts`'s reason: the store is a
 * string a person can hand-edit and a file is a string anyone can write, and a
 * malformed row must not reach the UI or the JSONL a training loader consumes.
 *
 * The embedded recipe is deliberately *not* validated here. It is validated on
 * the way in (`writeFavorite`) and on the way out (`favoriteRecipeText`), and a
 * row whose recipe went stale is still a perfectly good datum — dropping it
 * would throw away the verdict to punish the half nobody was asking for.
 */
function parseFavorite(raw: string | null): SeedFavorite | null {
  if (raw === null) return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  const e = value as Partial<SeedFavorite> | null;
  if (
    !e ||
    typeof e !== 'object' ||
    e.v !== FAVORITE_VERSION ||
    typeof e.id !== 'string' ||
    e.id === '' ||
    typeof e.at !== 'string' ||
    (e.verdict !== 'like' && e.verdict !== 'dislike') ||
    typeof e.sim !== 'string' ||
    typeof e.seed !== 'number' ||
    !Number.isFinite(e.seed) ||
    typeof e.seedPinned !== 'boolean' ||
    typeof e.speciesCount !== 'number' ||
    !Number.isFinite(e.speciesCount) ||
    typeof e.track !== 'string' ||
    typeof e.time !== 'number' ||
    !Number.isFinite(e.time) ||
    !(e.gen === null || (typeof e.gen === 'object' && !Array.isArray(e.gen)))
  ) {
    return null;
  }
  return e as SeedFavorite;
}
