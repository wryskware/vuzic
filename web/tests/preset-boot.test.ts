/**
 * The two boot channels, as behaviour: the one-shot slot and the `#p=` fragment.
 *
 * These are the properties that keep a bad preset from becoming an app that
 * cannot be started. `location.reload` is deliberately not exercised — the apply
 * *is* a reload, and the thing worth testing is what the next boot reads, which
 * is exactly what these call.
 */
import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

const values = new Map<string, string>();
let failWrites = false;

Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    get length(): number {
      return values.size;
    },
    key: (i: number): string | null => Array.from(values.keys())[i] ?? null,
    getItem: (key: string): string | null => values.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      if (failWrites) throw new DOMException('full', 'QuotaExceededError');
      values.set(key, value);
    },
    removeItem: (key: string): void => void values.delete(key),
  },
});

const loc = {
  origin: 'https://terrarium.example',
  pathname: '/app/',
  search: '?sim=plife',
  hash: '',
};
Object.defineProperty(globalThis, 'location', { configurable: true, value: loc });

let replaced: string | null = null;
Object.defineProperty(globalThis, 'history', {
  configurable: true,
  value: {
    state: null,
    replaceState: (_state: unknown, _title: string, url: string): void => {
      replaced = url;
      // A real `replaceState` to a fragment-less URL drops the fragment.
      loc.hash = '';
    },
  },
});

const { defaultModulationConfig } = await import('../src/mapping/persist.ts');
const { defaultImpulseConfig } = await import('../src/sim/impulses.ts');
const { defaultPlifeConfig } = await import('../src/sim/plife/config.ts');
const { encodePreset, presetFromRecipe } = await import('../src/runtime/preset.ts');
const { EXPORT_RECIPE_VERSION, validateExportRecipe } = await import('../src/runtime/recipe.ts');
const {
  clearFragment,
  consumeBootPreset,
  presetLinkFor,
  presetStringFromRecipe,
  requestPresetApply,
  takePendingPresetToken,
} = await import('../src/ui/presets.ts');

const PENDING_KEY = 'lmt.pendingPreset';

function plifeRecipe(seed = 4242): Parameters<typeof presetFromRecipe>[0] {
  const live = defaultPlifeConfig();
  const { render, ...simulation } = live;
  const {
    render: _sharedRender,
    impulses: _sharedImpulses,
    ...modulation
  } = defaultModulationConfig(live, 'plife');
  const built = {
    version: EXPORT_RECIPE_VERSION,
    rendererBuild: 'test-build',
    track: { id: 'pink-loop', contentVersion: 'sha256-deadbeef' },
    sim: 'plife',
    seed,
    seedPinned: false,
    simulation,
    modulation,
    modulationBase: [1, 2, 3],
    impulses: defaultImpulseConfig(),
    render,
    particleBudget: simulation.budget.cap,
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: 'av1-sdr-debug-1080p120',
      encoder: 'av1_nvenc',
      paperWhiteNits: 203,
      masteringPeakNits: 1000,
    },
  } as unknown as Parameters<typeof presetFromRecipe>[0];
  validateExportRecipe(built);
  return built;
}

beforeEach(() => {
  values.clear();
  failWrites = false;
  loc.hash = '';
  loc.search = '?sim=plife';
  replaced = null;
});

test('a staged preset is applied once and the slot is gone afterwards', async () => {
  const token = await encodePreset(presetFromRecipe(plifeRecipe(777)));
  assert.equal(await requestPresetApply(token), null);
  assert.equal(values.get(PENDING_KEY), token, 'the slot was not armed');

  const first = await consumeBootPreset();
  assert.equal(first.error, null);
  assert.equal(first.preset?.source, 'pending');
  assert.equal(first.preset?.plan.seed, 777, 'the preset did not replay its own seed');
  assert.equal(values.has(PENDING_KEY), false, 'the slot survived being read');

  // The next boot is an ordinary one.
  assert.deepEqual(await consumeBootPreset(), { preset: null, error: null });
});

test('the slot is consumed even when what it holds cannot be applied', async () => {
  values.set(PENDING_KEY, 'lmt1.notarealpreset');
  const result = await consumeBootPreset();
  assert.equal(result.preset, null);
  assert.match(String(result.error), /preset: /, 'the failure must be readable');
  assert.equal(
    values.has(PENDING_KEY),
    false,
    'a slot that survives its own failure re-applies on every reload',
  );
});

test('a corrupt token is refused in the panel rather than armed for the next boot', async () => {
  const failure = await requestPresetApply('lmt1.nope');
  assert.match(String(failure), /preset: /);
  assert.equal(values.has(PENDING_KEY), false, 'a preset that cannot load must not be staged');
});

test('a full origin degrades to a message instead of a silent no-op', async () => {
  const token = await encodePreset(presetFromRecipe(plifeRecipe()));
  failWrites = true;
  assert.match(String(await requestPresetApply(token)), /could not stage/);
});

test('a #p= link applies once and then leaves the address bar alone', async () => {
  const token = await encodePreset(presetFromRecipe(plifeRecipe(31337)));
  loc.hash = `#p=${token}`;

  const applied = await consumeBootPreset();
  assert.equal(applied.error, null);
  assert.equal(applied.preset?.source, 'fragment');
  assert.equal(applied.preset?.plan.seed, 31337);
  assert.equal(applied.preset?.plan.trackHint, 'pink-loop', 'the advisory hint travels');
  assert.equal(replaced, '/app/?sim=plife', 'the query survived and the fragment did not');
  assert.equal(loc.hash, '', 'a reload would re-apply a fragment that was left behind');

  const second = await consumeBootPreset();
  assert.deepEqual(second, { preset: null, error: null });
});

test('a #p= link that cannot be decoded is left in the address bar to be read', async () => {
  loc.hash = '#p=lmt1.AAAAAAAA';
  const result = await consumeBootPreset();
  assert.equal(result.preset, null);
  assert.match(String(result.error), /preset: /);
  assert.equal(replaced, null, 'nothing was applied, so nothing should have been cleaned up');
  assert.equal(loc.hash, '#p=lmt1.AAAAAAAA');
});

test('a staged preset outranks a link the tab happens to have been opened with', async () => {
  const staged = await encodePreset(presetFromRecipe(plifeRecipe(111)));
  const linked = await encodePreset(presetFromRecipe(plifeRecipe(222)));
  values.set(PENDING_KEY, staged);
  loc.hash = `#p=${linked}`;

  const result = await consumeBootPreset();
  assert.equal(result.preset?.source, 'pending');
  assert.equal(result.preset?.plan.seed, 111);
  assert.equal(values.has(PENDING_KEY), false);
  // The losing fragment is untouched, so the link still works on the next load.
  assert.equal(loc.hash, `#p=${linked}`);
});

test('a link is built from origin and path only', async () => {
  const token = await presetStringFromRecipe(plifeRecipe());
  assert.equal(presetLinkFor(token), `https://terrarium.example/app/#p=${token}`);
  assert.equal(takePendingPresetToken(), null);
  clearFragment();
  assert.equal(replaced, '/app/?sim=plife');
});
