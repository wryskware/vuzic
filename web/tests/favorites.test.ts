/**
 * The seed favorites pool.
 *
 * What these pin, and why each one is a property somebody could otherwise
 * regress without noticing:
 *
 * - **Per-key storage, no index.** Two presses are two independent writes, in
 *   any interleaving. A pool held in one JSON array would be a read-modify-write
 *   and the second tab's verdict would vanish — the exact loss the profile
 *   library was rebuilt to escape, and the reason `explore/log.ts`'s single blob
 *   is not the precedent this module follows.
 * - **A verdict is never lost to a full origin.** A like that will not fit gives
 *   up its recipe and keeps the datum, because the datum is the half that has a
 *   second consumer.
 * - **The export is a lossless, idempotent round trip.** localStorage dies with
 *   "clear site data"; if the file cannot rebuild the pool exactly, the pool is
 *   not durable and the training data is not either.
 * - **A row is checked, not trusted.** The store is hand-editable and a file is
 *   anyone's; one bad row must cost one verdict, never the pool.
 */
import assert from 'node:assert/strict';
import { after, beforeEach, test } from 'node:test';

const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
const values = new Map<string, string>();
/** Set to a key prefix to make writes to it fail the way a full origin does. */
let failWritesTo: string | null = null;
/** Bytes above which a write fails, for the "a like does not fit" case. */
let maxValueChars = Infinity;

Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    get length(): number {
      return values.size;
    },
    key: (i: number): string | null => Array.from(values.keys())[i] ?? null,
    getItem: (key: string): string | null => values.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      if (failWritesTo !== null && key.startsWith(failWritesTo)) {
        throw new DOMException('full', 'QuotaExceededError');
      }
      if (value.length > maxValueChars) throw new DOMException('full', 'QuotaExceededError');
      values.set(key, value);
    },
    removeItem: (key: string): void => void values.delete(key),
  },
});

const { captureBrowserExportRecipe } = await import('../src/export/browser-recipe.ts');
const { defaultModulationConfig } = await import('../src/mapping/persist.ts');
const { defaultImpulseConfig } = await import('../src/sim/impulses.ts');
const { defaultPlifeConfig } = await import('../src/sim/plife/config.ts');
const { presetFromConfig, presetToVector } = await import('../src/sim/plife/preset.ts');
const { parseExportRecipe } = await import('../src/runtime/recipe.ts');
const { PROFILE_RENDERER_BUILD, profileOutput } = await import('../src/ui/profiles.ts');
const {
  countFavorites,
  deleteFavorite,
  exportFavoritesJsonl,
  favoriteRecipeText,
  importFavoritesJsonl,
  listFavorites,
  newFavoriteId,
  readFavorite,
  recordFavorite,
  writeFavorite,
  FAVORITE_VERSION,
} = await import('../src/ui/favorites.ts');
type SeedFavorite = Awaited<ReturnType<typeof listFavorites>>[number];

after(() => {
  if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage);
  else Reflect.deleteProperty(globalThis, 'localStorage');
});

beforeEach(() => {
  values.clear();
  failWritesTo = null;
  maxValueChars = Infinity;
});

/** A live-session capture, the same shape the workbench's `captureProfile` takes. */
function capture(seed = 4242): ReturnType<typeof captureBrowserExportRecipe> {
  const live = defaultPlifeConfig();
  live.matrixGen.sigma = 0.77;
  const base = Array.from(presetToVector(presetFromConfig(live), live.speciesCount));
  return captureBrowserExportRecipe({
    rendererBuild: PROFILE_RENDERER_BUILD,
    track: { id: 'pink-loop', version: 'unversioned' },
    source: {
      sim: { simId: 'plife', currentSeed: seed, config: live },
      modulator: {
        config: defaultModulationConfig(live, 'plife'),
        mode: 'modulated',
        baseValues: () => Float64Array.from(base),
        currentTheta: () => base.slice(),
      },
      impulses: { config: defaultImpulseConfig() },
    },
    output: profileOutput(),
    currentPinState: () => true,
  });
}

/** The input a workbench button builds, for one verdict on one captured world. */
function press(verdict: 'like' | 'dislike', seed = 4242): Parameters<typeof recordFavorite>[0] {
  const recipe = capture(seed);
  return {
    verdict,
    sim: recipe.sim,
    seed: recipe.seed,
    seedPinned: recipe.seedPinned,
    speciesCount: 8,
    gen: JSON.parse(JSON.stringify((recipe.simulation as Record<string, unknown>)['matrixGen'])),
    track: 'pink-loop',
    time: 91.5,
    ...(verdict === 'like' ? { recipe } : {}),
  };
}

test('a like round trips field-exact, recipe included', () => {
  const input = press('like', 4242);
  const result = recordFavorite(input);
  assert.ok(result.id !== null);
  assert.equal(result.degraded, false);

  const listed = listFavorites('plife');
  assert.equal(listed.length, 1);
  const [fav] = listed as [SeedFavorite];
  assert.deepEqual(readFavorite(result.id), fav, 'read by id disagreed with the listing');

  // Field-exact, not "looks about right": every field is either a thing the
  // human needs to return to the world or a thing the model needs to re-derive
  // the matrix, and a silently dropped one is a silently useless row.
  assert.equal(fav.v, FAVORITE_VERSION);
  assert.equal(fav.id, result.id);
  assert.equal(fav.verdict, 'like');
  assert.equal(fav.sim, 'plife');
  assert.equal(fav.seed, 4242);
  assert.equal(fav.seedPinned, true);
  assert.equal(fav.speciesCount, 8);
  assert.equal(fav.track, 'pink-loop');
  assert.equal(fav.time, 91.5);
  assert.equal((fav.gen as Record<string, number>)['sigma'], 0.77, 'the generation block did not survive');
  assert.deepEqual(fav.gen, input.gen);
  assert.deepEqual(fav.recipe, input.recipe);
  assert.match(fav.at, /^\d{4}-\d\d-\d\dT/);
});

test('a like is returnable, a dislike carries the same model context and no state', () => {
  recordFavorite(press('like', 11));
  recordFavorite(press('dislike', 22));
  const [like, dislike] = listFavorites('plife') as [SeedFavorite, SeedFavorite];

  // Both halves of what the eventual model reads are present on both verdicts;
  // the asymmetry is only in the human's return path.
  for (const fav of [like, dislike]) {
    assert.equal(typeof fav.seed, 'number');
    assert.equal(fav.speciesCount, 8);
    assert.equal((fav.gen as Record<string, number>)['sigma'], 0.77);
  }
  assert.equal(dislike.recipe, undefined, 'a dislike stored a whole recipe');
  assert.equal(favoriteRecipeText(dislike), null);

  const text = favoriteRecipeText(like);
  assert.ok(text !== null, 'a like was not returnable');
  // Through the strict parser, exactly as `requestProfileApply` will run it.
  assert.equal(parseExportRecipe(text).seed, 11);
});

test('two writers interleaving cannot lose a verdict', () => {
  // Tab A prepares its record, tab B presses and writes, then A writes. A pool
  // held in one array would have A overwrite B's row with a snapshot taken
  // before it existed; per-key writes make the interleaving irrelevant.
  const at = '2026-08-15T22:10:30.123Z';
  const a: SeedFavorite = { v: FAVORITE_VERSION, id: newFavoriteId(7, at), at, ...press('like', 7) };
  const b: SeedFavorite = { v: FAVORITE_VERSION, id: newFavoriteId(7, at), at, ...press('dislike', 7) };
  assert.notEqual(a.id, b.id, 'two presses in the same millisecond on the same seed collided');

  writeFavorite(b);
  writeFavorite(a);

  const listed = listFavorites();
  assert.equal(listed.length, 2, 'one writer clobbered the other');
  assert.deepEqual(
    listed.map((f) => f.verdict).sort(),
    ['dislike', 'like'],
    'both rows survived but one was overwritten by the other',
  );
  // And no shared row exists to become one: every key this module writes is a
  // record of its own.
  assert.equal(values.size, 2, 'the pool grew a key that is not a record');
  for (const key of values.keys()) assert.match(key, /^lmt\.fav\.v1\./);
});

test('the pool does not disturb, and is not disturbed by, its neighbours', () => {
  values.set('lmt.profile.plife.mine', '{"version":6}');
  values.set('lmt.mapping.plife', '{"version":5}');
  const { id } = recordFavorite(press('like'));
  assert.equal(listFavorites().length, 1, 'a neighbouring key was read as a favorite');
  assert.equal(countFavorites(), 1);

  deleteFavorite(id as string);
  assert.equal(values.get('lmt.mapping.plife'), '{"version":5}');
  assert.equal(values.get('lmt.profile.plife.mine'), '{"version":6}');
});

test('export produces jsonl that re-imports to exactly the same set', () => {
  recordFavorite(press('like', 1));
  recordFavorite(press('dislike', 2));
  recordFavorite(press('like', 3));
  const before = listFavorites();
  const jsonl = exportFavoritesJsonl();

  // One self-describing JSON object per line — what a training loader streams.
  const lines = jsonl.split('\n');
  assert.equal(lines.length, 3);
  for (const line of lines) assert.equal(typeof (JSON.parse(line) as SeedFavorite).verdict, 'string');

  // "Clear site data", then the file.
  values.clear();
  assert.deepEqual(listFavorites(), []);
  assert.deepEqual(importFavoritesJsonl(jsonl), { added: 3, skipped: 0 });
  assert.deepEqual(listFavorites(), before, 'the pool did not survive its own export');
});

test('importing the same file twice is a no-op, not a doubling', () => {
  recordFavorite(press('like', 1));
  const jsonl = exportFavoritesJsonl();
  importFavoritesJsonl(jsonl);
  importFavoritesJsonl(jsonl);
  assert.equal(countFavorites(), 1, 'ids were not the merge key');
});

test('a malformed row costs one verdict, not the pool', () => {
  recordFavorite(press('like', 1));
  values.set('lmt.fav.v1.hand-written-nonsense', '{"v":1,"verdict":"maybe"}');
  values.set('lmt.fav.v1.not-even-json', 'oh dear');
  const listed = listFavorites();
  assert.equal(listed.length, 1);
  assert.equal(listed[0]?.seed, 1);
  // An import of the same junk reports it rather than swallowing it.
  assert.deepEqual(importFavoritesJsonl('{"v":1,"verdict":"maybe"}\nnot json\n'), {
    added: 0,
    skipped: 2,
  });
});

test('a like that will not fit keeps its verdict and drops its state', () => {
  // Small enough to refuse a recipe, large enough to take a bare row.
  maxValueChars = 2000;
  const result = recordFavorite(press('like', 99));
  assert.ok(result.id !== null, 'a full origin lost the verdict entirely');
  assert.equal(result.degraded, true);
  assert.match(result.message, /verdict only/);
  const [fav] = listFavorites() as [SeedFavorite];
  assert.equal(fav.verdict, 'like');
  assert.equal(fav.seed, 99);
  assert.equal(fav.recipe, undefined);
  assert.equal((fav.gen as Record<string, number>)['sigma'], 0.77, 'the model context went with the state');
});

test('storage that refuses everything reports it and stores nothing', () => {
  failWritesTo = 'lmt.fav.';
  const result = recordFavorite(press('dislike'));
  assert.equal(result.id, null);
  assert.match(result.message, /no room/);
  assert.equal(countFavorites(), 0);
});

test('a state that could not be loaded is never stored', () => {
  const input = press('like');
  // The one wart of embedding a recipe: it can go stale. Refused at write time,
  // where a person is looking, rather than at boot, where nobody is.
  (input.recipe as unknown as Record<string, unknown>)['version'] = 99;
  const result = recordFavorite(input);
  assert.equal(result.id, null);
  assert.match(result.message, /not a usable state/);
  assert.equal(countFavorites(), 0);
});

test('ids sort chronologically, so the pool lists in press order', () => {
  const early = newFavoriteId(1, '2026-08-15T22:10:30.123Z');
  const late = newFavoriteId(1, '2026-08-15T22:10:31.000Z');
  assert.ok(early < late, 'a prefix scan would come back out of order');
});
