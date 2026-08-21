/**
 * Preset strings — the format, the codec, and the boot decision.
 *
 * The properties pinned here are the ones a shareable string has to have and a
 * localStorage profile never needed:
 *
 * - **A string is the whole look, and only the look.** Round-tripping one has to
 *   give back the same simulation config, mapping, θ centre, impulses, grade and
 *   *seed* — and the four export-only fields have to come back as stubs rather
 *   than as something a later export could mistake for a real capture.
 * - **The same state twice is the same string.** Canonical key order is what
 *   makes a preset content-hashable, which is the only piece of future-proofing
 *   the brief allowed for eventual server-side storage.
 * - **Every failure is loud and readable.** A preset arrives from outside — a
 *   chat message, a URL, a paste — so bad prefix, bad base64, corrupt deflate,
 *   valid-JSON-wrong-shape and zip-bomb all have to fail with a message rather
 *   than half-apply.
 * - **The boot channels are consumed exactly once.** A pending slot that
 *   survived its own failure would re-apply on every reload; a `#p=` fragment
 *   that survived a success would re-apply over a session of tweaks.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import { defaultConfig as defaultPhysarumConfig } from '../src/sim/physarum/config.ts';
import { defaultVizFxConfig } from '../src/sim/vizfx/config.ts';
import { NEBULA } from './vizfx-modules.ts';
import {
  chooseBootPreset,
  decodePreset,
  encodePreset,
  extractPresetToken,
  MAX_PRESET_TOKEN_CHARS,
  parsePreset,
  presetApplyPlan,
  presetFromRecipe,
  presetLink,
  presetTokenFromHash,
  PRESET_PREFIX,
  PRESET_RENDERER_BUILD,
  PRESET_TRACK_UNKNOWN,
  recipeFromPreset,
  serializePreset,
  validatePresetV1,
  type PresetV1,
} from '../src/runtime/preset.ts';
import {
  EXPORT_RECIPE_VERSION,
  MAX_RECIPE_JSON_CHARS,
  validateExportRecipe,
  type ExportRecipe,
} from '../src/runtime/recipe.ts';

type Fixture = 'plife' | 'physarum' | 'nebula';

/**
 * One recipe per substrate, built the way `tests/export-recipe.test.ts` builds
 * its plife one: from the shipped defaults, with `render` and `impulses` lifted
 * out of the config and the modulation block the way a recipe encodes them.
 */
function recipeFor(sim: Fixture): ExportRecipe {
  const live =
    sim === 'plife'
      ? defaultPlifeConfig()
      : sim === 'physarum'
        ? defaultPhysarumConfig()
        : defaultVizFxConfig(NEBULA);
  const { render, ...simulation } = live;
  const {
    render: _sharedRender,
    impulses: _sharedImpulses,
    ...modulation
  } = defaultModulationConfig({ ...simulation, render } as never, sim);
  const budget =
    sim === 'plife'
      ? (simulation as ReturnType<typeof defaultPlifeConfig>).budget.cap
      : sim === 'physarum'
        ? (simulation as ReturnType<typeof defaultPhysarumConfig>).maxAgents
        : 0;
  const recipe = {
    version: EXPORT_RECIPE_VERSION,
    rendererBuild: 'test-build',
    track: { id: 'pink-loop', contentVersion: 'sha256-deadbeef' },
    sim,
    seed: 0x0bad_f00d,
    seedPinned: true,
    simulation,
    modulation,
    modulationBase: [0.25, -0.5, 1],
    impulses: defaultImpulseConfig(),
    render,
    particleBudget: budget,
    presentation: { mode: 'single', autoAdvance: false },
    // Deliberately not the stub `recipeFromPreset` writes back, so "the output
    // block did not survive the trip" is a real assertion rather than a
    // coincidence of two defaults agreeing.
    output: {
      profile: 'av1-hdr10-2160p120',
      encoder: 'av1_nvenc',
      paperWhiteNits: 250,
      masteringPeakNits: 1200,
    },
  } as unknown as ExportRecipe;
  validateExportRecipe(recipe);
  return recipe;
}

const SIMS: Fixture[] = ['plife', 'physarum', 'nebula'];

// ── round trip ───────────────────────────────────────────────────────────────

test('a preset round-trips the whole look, for every substrate', async () => {
  for (const sim of SIMS) {
    const recipe = recipeFor(sim);
    const preset = presetFromRecipe(recipe);
    const token = await encodePreset(preset);
    assert.ok(token.startsWith(PRESET_PREFIX), `${sim}: token is not an lmt1. string`);

    const decoded = await decodePreset(token);
    assert.deepEqual(decoded, preset, `${sim}: the decoded preset is not the encoded one`);

    // Every block a look is made of survived, unchanged, from the recipe.
    assert.deepEqual(decoded.simulation, recipe.simulation, `${sim}: simulation`);
    assert.deepEqual(decoded.modulation, recipe.modulation, `${sim}: modulation`);
    assert.deepEqual(decoded.modulationBase, recipe.modulationBase, `${sim}: θ centre`);
    assert.deepEqual(decoded.impulses, recipe.impulses, `${sim}: impulses`);
    assert.deepEqual(decoded.render, recipe.render, `${sim}: render`);
    assert.equal(decoded.seed, recipe.seed, `${sim}: seed`);
    assert.equal(decoded.seedPinned, recipe.seedPinned, `${sim}: pin state`);
    assert.equal(decoded.track, recipe.track.id, `${sim}: track hint`);
  }
});

test('rehydrating a preset gives back a valid recipe with honest stubs', () => {
  for (const sim of SIMS) {
    const recipe = recipeFor(sim);
    const back = recipeFromPreset(presetFromRecipe(recipe));
    assert.doesNotThrow(() => validateExportRecipe(back), `${sim}: not a valid recipe`);

    // The budget is derived rather than carried, and derived correctly — this is
    // the one dropped field that has a right answer.
    assert.equal(back.particleBudget, recipe.particleBudget, `${sim}: particleBudget`);
    // The other three say what they are instead of borrowing something an export
    // could later mistake for a real capture.
    assert.equal(back.rendererBuild, PRESET_RENDERER_BUILD);
    assert.deepEqual(back.presentation, { mode: 'single', autoAdvance: false });
    assert.notDeepEqual(back.output, recipe.output, 'the output stub is not the captured one');

    // Everything that is the look is byte-identical.
    const strip = (r: ExportRecipe): unknown => {
      const { rendererBuild, output, presentation, particleBudget, track, ...rest } = r;
      void rendererBuild;
      void output;
      void presentation;
      void particleBudget;
      void track;
      return rest;
    };
    assert.deepEqual(strip(back), strip(recipe), `${sim}: the look did not survive the trip`);
  }
});

test('a preset with no track hint rehydrates, and re-encodes without inventing one', () => {
  const preset = presetFromRecipe(recipeFor('plife'));
  delete preset.track;
  assert.doesNotThrow(() => validatePresetV1(preset), 'the hint is optional');

  const recipe = recipeFromPreset(preset);
  assert.equal(recipe.track.id, PRESET_TRACK_UNKNOWN);
  // …and the stub does not become a hint pointing at a track called "unknown".
  assert.equal(presetFromRecipe(recipe).track, undefined);
  assert.equal(presetApplyPlan(preset).trackHint, null);
});

// ── canonical stability ──────────────────────────────────────────────────────

test('the same state produces the same string, whatever order it was built in', async () => {
  const preset = presetFromRecipe(recipeFor('plife'));
  const once = await encodePreset(preset);
  const twice = await encodePreset(structuredClone(preset));
  assert.equal(once, twice, 'two encodes of the same look disagree');

  // Key order is an accident of how an object was assembled; a content-hashable
  // string must not depend on it.
  const shuffled = Object.fromEntries(
    Object.entries(preset as unknown as Record<string, unknown>).reverse(),
  ) as unknown as PresetV1;
  assert.equal(await encodePreset(shuffled), once, 'key order leaked into the string');
  assert.equal(serializePreset(shuffled), serializePreset(preset));
});

test('compression is doing real work on a real config', async () => {
  const preset = presetFromRecipe(recipeFor('plife'));
  const json = serializePreset(preset);
  const token = await encodePreset(preset);
  assert.ok(
    token.length < json.length,
    `the string (${token.length}) is not smaller than the JSON (${json.length})`,
  );
});

// ── rejection ────────────────────────────────────────────────────────────────

test('a string that is not a preset is refused, with a message that says why', async () => {
  const good = await encodePreset(presetFromRecipe(recipeFor('plife')));

  await assert.rejects(
    () => decodePreset(good.replace(PRESET_PREFIX, 'lmt2.')),
    /preset: .*does not start with "lmt1\."/,
    'an unknown container version must not be guessed at',
  );
  await assert.rejects(
    () => decodePreset('have you seen my look'),
    /preset: .*does not start with "lmt1\."/,
  );
  await assert.rejects(() => decodePreset(PRESET_PREFIX), /preset: .*base64url/);
  await assert.rejects(
    () => decodePreset(`${PRESET_PREFIX}not+valid+base64url`),
    /preset: .*base64url/,
  );
  // Truncated: a real token cut mid-stream. The alphabet is still legal, so this
  // has to fail in the inflate rather than in the decoder.
  await assert.rejects(
    () => decodePreset(good.slice(0, good.length - 40)),
    /preset: .*(decompressed|base64url)/,
  );
  // Corrupt: legal base64url that is not a deflate stream.
  await assert.rejects(
    () => decodePreset(`${PRESET_PREFIX}${'AAAA'.repeat(16)}`),
    /preset: .*could not be decompressed/,
  );
});

test('valid JSON in the wrong shape is refused by the block validators', () => {
  const preset = presetFromRecipe(recipeFor('plife'));

  const brokenRender = structuredClone(preset);
  brokenRender.render.bloom.levels = 999;
  assert.throws(() => serializePreset(brokenRender), /preset: \$\.render\.bloom\.levels/);

  const brokenPalette = structuredClone(preset);
  brokenPalette.modulation.palette.colors[0] = '#c0ffee';
  assert.throws(
    () => serializePreset(brokenPalette),
    /preset: \$\.modulation\.palette.*must match/,
    'the sim/mapping palette agreement is enforced in a preset too',
  );

  const noSeed = structuredClone(preset) as unknown as Record<string, unknown>;
  delete noSeed['seed'];
  assert.throws(() => parsePreset(JSON.stringify(noSeed)), /preset: \$\.seed.*required/);

  const exportOnly = structuredClone(preset) as unknown as Record<string, unknown>;
  exportOnly['output'] = { profile: 'av1-sdr-debug-1080p120' };
  assert.throws(
    () => parsePreset(JSON.stringify(exportOnly)),
    /preset: \$\.output.*not supported/,
    'a preset is the recipe minus the export fields, and says so',
  );

  const wrongVersion = structuredClone(preset) as unknown as Record<string, unknown>;
  wrongVersion['version'] = 2;
  assert.throws(() => parsePreset(JSON.stringify(wrongVersion)), /preset: \$\.version/);

  assert.throws(() => parsePreset('{oh no'), /preset: \$.*not valid JSON/);
});

test('a small string that inflates to a huge one is refused before it is parsed', async () => {
  // The zip-bomb case: ~9 MB of one repeated byte deflates to a few kilobytes,
  // and the token below is far under every input bound. Only the *output* bound
  // can stop it, which is why there are two.
  const huge = new TextEncoder().encode('a'.repeat(MAX_RECIPE_JSON_CHARS + 1_000_000));
  const deflated: Uint8Array[] = [];
  const stream = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(huge);
      controller.close();
    },
  }).pipeThrough(new CompressionStream('deflate-raw'));
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    deflated.push(value);
  }
  const bytes = Buffer.concat(deflated);
  const token =
    PRESET_PREFIX +
    bytes.toString('base64').replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
  assert.ok(token.length < MAX_PRESET_TOKEN_CHARS, 'the bomb must be small enough to be accepted');

  await assert.rejects(() => decodePreset(token), /preset: .*expands past/);
});

test('an oversize token is refused without being inflated at all', async () => {
  const token = PRESET_PREFIX + 'A'.repeat(MAX_PRESET_TOKEN_CHARS + 1);
  await assert.rejects(() => decodePreset(token), /preset: .*exceeds .* encoded characters/);
});

// ── seed semantics ───────────────────────────────────────────────────────────

test('the live seed is always stored, and the plan always replays it', () => {
  const recipe = recipeFor('plife');
  recipe.seed = 123_456;

  recipe.seedPinned = true;
  const pinned = presetApplyPlan(presetFromRecipe(recipe));
  assert.deepEqual(pinned, { sim: 'plife', seed: 123_456, pinSeed: 123_456, trackHint: 'pink-loop' });

  // Unpinned is the interesting half: the seed is still saved and still
  // replayed — what changes is only whether *live* runs stay on it.
  recipe.seedPinned = false;
  const loose = presetApplyPlan(presetFromRecipe(recipe));
  assert.equal(loose.seed, 123_456, 'an unpinned preset still replays its own seed');
  assert.equal(loose.pinSeed, null, 'an unpinned preset must not pin the seed on load');
});

// ── links and pasted text ────────────────────────────────────────────────────

test('a token is found whether it was pasted bare or inside a link', async () => {
  const token = await encodePreset(presetFromRecipe(recipeFor('plife')));
  const link = presetLink('https://terrarium.example/app/', token);

  assert.equal(extractPresetToken(token), token);
  assert.equal(extractPresetToken(`  ${token}\n`), token, 'whitespace from a paste');
  assert.equal(extractPresetToken(link), token);
  assert.equal(extractPresetToken(`look at this ${link} isn't it nice`), token);
  assert.equal(extractPresetToken('https://terrarium.example/app/#p=nope'), null);
  assert.equal(extractPresetToken(''), null);

  assert.ok(link.includes('#p='), 'a fragment, never a query');
  assert.equal(presetTokenFromHash(`#p=${token}`), token);
  assert.equal(presetTokenFromHash('#p=lmt2.abc'), null);
  assert.equal(presetTokenFromHash(''), null);
});

// ── the boot decision ────────────────────────────────────────────────────────

test('the pending slot beats the fragment, and nothing beats neither', () => {
  assert.deepEqual(chooseBootPreset('lmt1.staged', 'lmt1.linked'), {
    token: 'lmt1.staged',
    source: 'pending',
  });
  assert.deepEqual(chooseBootPreset(null, 'lmt1.linked'), {
    token: 'lmt1.linked',
    source: 'fragment',
  });
  assert.deepEqual(chooseBootPreset(null, null), { token: null, source: 'none' });
});
