/**
 * The named profile library.
 *
 * These pin the properties that make a profile a *hard* save location, as
 * distinct from the mapping autosave next to it:
 *
 * - **Per-key storage, no index.** Two saves are two independent writes. A
 *   library held in one JSON array would be a read-modify-write, and two tabs
 *   saving two profiles would leave one of them — which is the exact class of
 *   loss this module exists to escape. `two profiles coexist` and `deleting one
 *   leaves the other` are that property, stated as behaviour.
 * - **Nothing is stored that could not be loaded.** Save parses first, so a
 *   library entry is never a file that fails at boot, when there is no longer a
 *   panel to say so.
 * - **The pending slot is consumed exactly once, including on failure.** A slot
 *   that survived its own parse failure would re-apply on every reload, and the
 *   app could not be started normally again without devtools.
 * - **Names survive verbatim.** They are encoded into the key, so dots — which
 *   also separate the key's own fields — must not truncate or collide.
 */
import assert from 'node:assert/strict';
import { after, beforeEach, test } from 'node:test';

const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
const values = new Map<string, string>();
/** Set to a key prefix to make writes to it fail the way a full origin does. */
let failWritesTo: string | null = null;

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
      values.set(key, value);
    },
    removeItem: (key: string): void => void values.delete(key),
  },
});

const { captureBrowserExportRecipe } = await import('../src/export/browser-recipe.ts');
const { defaultModulationConfig } = await import('../src/mapping/persist.ts');
const { defaultImpulseConfig } = await import('../src/sim/impulses.ts');
const { defaultPlifeConfig } = await import('../src/sim/plife/config.ts');
const { defaultConfig } = await import('../src/sim/physarum/config.ts');
const { presetFromConfig, presetToVector } = await import('../src/sim/plife/preset.ts');
const { parseExportRecipe } = await import('../src/runtime/recipe.ts');
const {
  consumePendingProfile,
  deleteProfile,
  listProfiles,
  normalizeProfileName,
  profileOutput,
  PROFILE_RENDERER_BUILD,
  readProfileText,
  requestProfileApply,
  saveProfile,
  saveProfileText,
} = await import('../src/ui/profiles.ts');

after(() => {
  if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage);
  else Reflect.deleteProperty(globalThis, 'localStorage');
});

beforeEach(() => {
  values.clear();
  failWritesTo = null;
});

/** A live-session capture, the same shape the workbench's `captureProfile` takes. */
function capture(sim: 'plife' | 'physarum' = 'plife', seed = 4242): ReturnType<typeof captureBrowserExportRecipe> {
  const live = sim === 'plife' ? defaultPlifeConfig() : defaultConfig(4);
  live.render.grade.exposureEv = 0.625;
  const modulation = defaultModulationConfig(live, sim);
  const base =
    sim === 'plife'
      ? Array.from(presetToVector(presetFromConfig(live as never), live.speciesCount))
      : [0.5, 0.25];
  return captureBrowserExportRecipe({
    rendererBuild: PROFILE_RENDERER_BUILD,
    track: { id: 'pink-loop', version: 'unversioned' },
    source: {
      sim: { simId: sim, currentSeed: seed, config: live },
      modulator: {
        config: modulation,
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

test('a saved profile comes back byte-identical and parses', () => {
  const recipe = capture();
  const saved = saveProfile('chorus hot', recipe);
  assert.equal(saved.sim, 'plife');

  const text = readProfileText('plife', 'chorus hot');
  assert.ok(text !== null, 'the profile was not readable after saving');
  // The whole point of storing a recipe: it comes back through the strict
  // parser, seed and theta included, not through a lossy panel-shaped subset.
  const back = parseExportRecipe(text);
  assert.deepEqual(back, recipe);
  assert.equal(back.seed, 4242);
  assert.equal(back.seedPinned, true);
});

test('a profile lands on the shelf of its own sim, not the one that is running', () => {
  saveProfile('a physarum look', capture('physarum'));
  assert.deepEqual(listProfiles('physarum'), ['a physarum look']);
  assert.deepEqual(listProfiles('plife'), []);
});

test('two profiles coexist, and deleting one leaves the other', () => {
  saveProfile('one', capture('plife', 1));
  saveProfile('two', capture('plife', 2));
  assert.deepEqual(listProfiles('plife'), ['one', 'two']);

  deleteProfile('plife', 'one');
  assert.deepEqual(listProfiles('plife'), ['two']);
  const survivor = parseExportRecipe(readProfileText('plife', 'two') as string);
  assert.equal(survivor.seed, 2, 'deleting a neighbour rewrote the survivor');
});

test('a name with dots and slashes round trips through list, read and delete', () => {
  // Dots also separate the key's own fields, so this is the case a naive
  // `key.split('.')` gets wrong.
  const name = 'v1.2 · chorus / hot — ü';
  saveProfile(name, capture());
  assert.deepEqual(listProfiles('plife'), [name]);
  assert.ok(readProfileText('plife', name) !== null);
  deleteProfile('plife', name);
  assert.deepEqual(listProfiles('plife'), []);
});

test('text that is not a recipe is refused and stores nothing', () => {
  const bad = saveProfileText('junk', '{"version":1,"nope":true}');
  assert.equal(bad.sim, null);
  assert.match(bad.message, /not a profile/);
  assert.deepEqual(listProfiles('plife'), []);
  assert.equal(values.size, 0, 'a rejected profile still occupied storage');
});

test('a full origin reports something a person can act on, and does not throw', () => {
  failWritesTo = 'lmt.profile.';
  const result = saveProfile('too big', capture());
  assert.match(result.message, /no room/);
  assert.deepEqual(listProfiles('plife'), []);
});

test('the autosave slot and the pending key are not part of the library', () => {
  values.set('lmt.mapping.plife', '{"version":5}');
  saveProfile('mine', capture());
  requestProfileApply(readProfileText('plife', 'mine') as string);

  assert.deepEqual(listProfiles('plife'), ['mine'], 'a non-profile key was listed as a profile');
  deleteProfile('plife', 'mine');
  assert.equal(
    values.get('lmt.mapping.plife'),
    '{"version":5}',
    'deleting a profile disturbed the mapping autosave',
  );
});

test('an apply is validated before it is staged', () => {
  const failure = requestProfileApply('not json at all');
  assert.ok(failure !== null, 'a corrupt profile was staged instead of reported');
  assert.equal(consumePendingProfile(), null, 'a rejected apply left something staged');
});

test('a staged profile is consumed exactly once', () => {
  const recipe = capture();
  assert.equal(requestProfileApply(JSON.stringify(recipe)), null);

  const applied = consumePendingProfile();
  assert.ok(applied !== null);
  assert.equal(applied.seed, recipe.seed);
  assert.deepEqual(applied.modulationBase, recipe.modulationBase);
  assert.equal(consumePendingProfile(), null, 'the pending slot survived being read');
});

test('a pending profile that fails to parse is still consumed', () => {
  // It can only get here by being written by hand or by a later schema change,
  // and if it survived its own failure the app could not be booted normally
  // again.
  values.set('lmt.pendingProfile', '{"version":99}');
  assert.equal(consumePendingProfile(), null);
  assert.equal(values.has('lmt.pendingProfile'), false, 'a bad pending profile re-applies forever');
});

test('names are trimmed, and an empty one is refused', () => {
  assert.equal(normalizeProfileName('  hot  '), 'hot');
  assert.equal(normalizeProfileName('   '), null);
  assert.equal(normalizeProfileName(''), null);
});
