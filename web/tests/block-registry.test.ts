/**
 * Block registration — roadmap phase 1 item 5, and the last rung of the
 * "my tweak didn't save" class.
 *
 * ## What is being defended
 *
 * The two rungs below this are closed. Fields round trip because `readInto`
 * walks the destination's own keys (`persist-roundtrip.test.ts`), and widgets
 * save because a panel can only be handed a `PersistedContainer`
 * (`persisting.test.ts`). The rung above them was the same defect one level up:
 * a whole new **config block** persisted only if somebody remembered to name it
 * in the sim's `extrasBlocks()`, *and* in a defaults table, *and* in a rules
 * table — and nothing failed if they remembered it in none of the three. The
 * block simply never saved.
 *
 * `mapping/blocks.ts` closes it by deriving the block set from the config
 * itself: `BlockTable<C>` is exhaustive over `C`'s own object-valued keys, so an
 * undeclared block is a **compile error**. These tests are the belt behind that
 * brace, and they are the acceptance test the roadmap fixed:
 *
 *   1. declare a block, register nothing else by hand, and it round trips
 *      through serialize / restore / clamp;
 *   2. opt a block out, and CI says the opt-out was deliberate — an exhaustive
 *      assertion over every declared-but-unpersisted block, with its reason.
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads modules straight
 * out of src/, so every import carries an explicit `.ts` extension. The one
 * exception is `./vizfx-modules.ts`, imported **first** because its two resolver
 * hooks (`?raw` WGSL, extensionless relative specifiers) are what make the
 * concrete sims importable at all — see its header.
 */
import './vizfx-modules.ts';
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  applyBlocks,
  blockRules,
  persisted,
  persistedBlocks,
  persistedElsewhere,
  serializeBlocks,
  sessionOnly,
  sessionOnlyFields,
  undeclaredBlocks,
  unpersistedBlocks,
  type BlockTable,
} from '../src/mapping/blocks.ts';
import { oneOf, range } from '../src/mapping/read-into.ts';
import { defaultPlifeConfig, PLIFE_BLOCKS, type PlifeConfig } from '../src/sim/plife/config.ts';
import { defaultConfig, PHYSARUM_BLOCKS } from '../src/sim/physarum/config.ts';
import { defaultVizFxConfig, VIZFX_BLOCKS } from '../src/sim/vizfx/config.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { NEBULA } from './vizfx-modules.ts';

interface ExtrasSim {
  serializeExtras(): Record<string, unknown>;
  applyExtras(raw: Record<string, unknown> | undefined): void;
}

const { PlifeSim } = (await import('../src/sim/plife/plife.ts')) as {
  PlifeSim: new (seed: number, config: PlifeConfig) => ExtrasSim & { config: PlifeConfig };
};

const { PhysarumSim } = (await import('../src/sim/physarum/physarum.ts')) as {
  PhysarumSim: new (seed: number) => ExtrasSim;
};

const { VizFxSim } = (await import('../src/sim/vizfx/vizfx.ts')) as {
  VizFxSim: new (visual: typeof NEBULA, seed: number) => ExtrasSim;
};

// ── acceptance test 1: one declaration is the whole of registering a block ────

/**
 * A config with a **new block** in it — `lane` — plus one of every other case a
 * real config has. Nothing below names `lane` again: no defaults table, no rules
 * table, no serialize list, no restore list. Its single `persisted(...)`
 * declaration in `DEMO_BLOCKS` is the entire registration, and the tests that
 * follow are what "and that is sufficient" means.
 */
interface DemoConfig {
  /** the new block */
  lane: { gain: number; mode: string; nested: { depth: number } };
  /** a block with no rules at all — rules clamp, defaults enrol */
  transport: { speed: number; paused: boolean };
  /** persisted, except for one transient field */
  probe: { fps: number; debugView: boolean };
  /** opted out entirely */
  governor: { measuredFps: number };
  /** opted out because it is saved by another path */
  theta: { a: number };
  /** not a block: scalars and arrays are not this mechanism's domain */
  name: string;
  weights: number[];
}

function demoConfig(): DemoConfig {
  return {
    lane: { gain: 1, mode: 'grid', nested: { depth: 2 } },
    transport: { speed: 1, paused: false },
    probe: { fps: 60, debugView: false },
    governor: { measuredFps: 0 },
    theta: { a: 0 },
    name: 'demo',
    weights: [1, 2, 3],
  };
}

const DEMO_BLOCKS: BlockTable<DemoConfig> = {
  lane: persisted(() => ({ gain: 1, mode: 'grid', nested: { depth: 2 } }), {
    rules: {
      gain: range(0, 2),
      mode: oneOf(['grid', 'brute']),
      nested: { depth: range(0, 4, { int: true }) },
    },
  }),
  transport: persisted(() => ({ speed: 1, paused: false })),
  probe: persisted(() => ({ fps: 60, debugView: false }), {
    rules: { fps: range(15, 240) },
    sessionOnlyFields: {
      debugView: 'a transient toggle; a restored one is a world in the wrong mode with no cause',
    },
  }),
  governor: sessionOnly('a live measurement of this machine, in this session, at this window size'),
  theta: persistedElsewhere('reproduced from the seed on load; the seed is what persists'),
};

test('declaring a block is the whole of registering it: serialize, restore, clamp', () => {
  const authored = demoConfig();
  authored.lane.gain = 1.75;
  authored.lane.mode = 'brute';
  authored.lane.nested.depth = 3;
  authored.transport.speed = 2.5;
  authored.transport.paused = true;
  authored.probe.fps = 144;

  // The write side, and the read side, with no per-block code anywhere.
  const saved = JSON.parse(JSON.stringify(serializeBlocks(authored, DEMO_BLOCKS))) as unknown;
  const restored = demoConfig();
  applyBlocks(restored, DEMO_BLOCKS, saved);

  assert.deepEqual(restored.lane, authored.lane, 'the new block did not round trip');
  // `transport` has no rules at all and still round trips: rules only clamp,
  // defaults enrol. If this ever fails, someone has made the rule table
  // load-bearing for persistence and the forgetting is back.
  assert.deepEqual(restored.transport, authored.transport);
  assert.equal(restored.probe.fps, 144);
});

test('a declared block is clamped by its declared rules, and never throws', () => {
  const live = demoConfig();
  applyBlocks(live, DEMO_BLOCKS, {
    lane: { gain: 99, mode: 'nonsense', nested: { depth: 2.6 } },
    probe: { fps: -100 },
  });
  assert.equal(live.lane.gain, 2, 'clamped to the declared range');
  assert.equal(live.lane.mode, 'grid', 'an unknown enum value falls back to the default');
  assert.equal(live.lane.nested.depth, 3, 'integers are rounded after clamping');
  assert.equal(live.probe.fps, 15);

  // Hostile input costs the field, not the load. Same contract `readInto` has.
  for (const bad of [undefined, null, 42, 'nope', [1, 2], { lane: 7 }, { lane: null }]) {
    const fresh = demoConfig();
    applyBlocks(fresh, DEMO_BLOCKS, bad);
    assert.deepEqual(fresh.lane, demoConfig().lane, `a ${JSON.stringify(bad)} blob moved a field`);
  }
});

test('restore is a function of the blob, not of history', () => {
  // The explorer fans one serialised block out to nine tiles twice a second, so
  // a partial block must not leave the untouched half at whatever the previous
  // tile put there — that would make a tile's look depend on what it showed last.
  const live = demoConfig();
  applyBlocks(live, DEMO_BLOCKS, { lane: { gain: 1.9, mode: 'brute' } });
  assert.equal(live.lane.gain, 1.9);
  applyBlocks(live, DEMO_BLOCKS, { lane: { mode: 'brute' } });
  assert.equal(live.lane.gain, 1, 'a field absent from the blob must return to its default');
});

test('restore writes in place, so a panel keeps editing what runs', () => {
  const live = demoConfig();
  const held = [live.lane, live.lane.nested, live.transport];
  applyBlocks(live, DEMO_BLOCKS, serializeBlocks(live, DEMO_BLOCKS));
  assert.equal(live.lane, held[0]);
  assert.equal(live.lane.nested, held[1], 'a nested sub-object was replaced, orphaning its bindings');
  assert.equal(live.transport, held[2]);
});

// ── acceptance test 2: an opt-out has to be deliberate ───────────────────────

test('an opted-out block is neither written nor read', () => {
  const authored = demoConfig();
  authored.governor.measuredFps = 143;
  authored.theta.a = 7;

  const saved = serializeBlocks(authored, DEMO_BLOCKS);
  assert.equal('governor' in saved, false, 'session-only state was written to the file');
  assert.equal('theta' in saved, false);

  // …and a file that carries one anyway (hand-edited, or written by an older
  // build) cannot put it back.
  const live = demoConfig();
  applyBlocks(live, DEMO_BLOCKS, { governor: { measuredFps: 143 }, theta: { a: 7 } });
  assert.equal(live.governor.measuredFps, 0);
  assert.equal(live.theta.a, 0);
});

test('a session-only FIELD is lifted over both walks', () => {
  const authored = demoConfig();
  authored.probe.debugView = true;
  authored.probe.fps = 90;

  const saved = serializeBlocks(authored, DEMO_BLOCKS) as Record<string, Record<string, unknown>>;
  assert.deepEqual(saved['probe'], { fps: 90 }, 'the session-only field was written');

  // Neither the reset-to-defaults walk nor the read walk may move it: whatever
  // the panel currently has stays, in both directions.
  const live = demoConfig();
  live.probe.debugView = true;
  applyBlocks(live, DEMO_BLOCKS, { probe: { fps: 90, debugView: false } });
  assert.equal(live.probe.fps, 90);
  assert.equal(live.probe.debugView, true, 'a file overwrote session-only state');
});

test('every opt-out is enumerable, with the reason given at its declaration site', () => {
  assert.deepEqual(
    unpersistedBlocks(DEMO_BLOCKS).map(({ block, kind }) => ({ block, kind })),
    [
      { block: 'governor', kind: 'session-only' },
      { block: 'theta', kind: 'persisted-elsewhere' },
    ],
  );
  assert.deepEqual(sessionOnlyFields(DEMO_BLOCKS).map(({ block, field }) => ({ block, field })), [
    { block: 'probe', field: 'debugView' },
  ]);
});

// ── the detector: an undeclared block is caught, loudly ──────────────────────

test('a block the table never heard of is reported, not silently dropped', () => {
  const grown = demoConfig() as DemoConfig & { brightnessCurve?: object };
  assert.deepEqual(undeclaredBlocks(grown, DEMO_BLOCKS), [], 'the table starts complete');

  // What a future feature does: a new block on the config. In src/ this is a
  // compile error (`BlockTable<C>` is exhaustive over C's block keys); here it is
  // the runtime half of the same statement, because a test can run it.
  grown.brightnessCurve = { exponent: 2 };
  assert.deepEqual(undeclaredBlocks(grown, DEMO_BLOCKS), ['brightnessCurve']);

  // Scalars and arrays are deliberately not in this mechanism's domain — they are
  // fields of no block, which is the rung below's problem.
  const scalars = demoConfig() as DemoConfig & { extra?: unknown };
  scalars.extra = 3;
  assert.deepEqual(undeclaredBlocks(scalars, DEMO_BLOCKS), []);
  scalars.extra = [1, 2];
  assert.deepEqual(undeclaredBlocks(scalars, DEMO_BLOCKS), []);
});

// ── the three real substrates ────────────────────────────────────────────────

test('every block of every live config is declared', () => {
  assert.deepEqual(undeclaredBlocks(defaultPlifeConfig(), PLIFE_BLOCKS), []);
  assert.deepEqual(undeclaredBlocks(defaultConfig(4), PHYSARUM_BLOCKS), []);
  assert.deepEqual(undeclaredBlocks(defaultVizFxConfig(NEBULA), VIZFX_BLOCKS), []);
});

/**
 * The saved key set, and its order, pinned per sim.
 *
 * Order because it is the serialized key order and this change was not allowed to
 * move it: an autosave written by an older build has to keep diffing cleanly
 * against a newer one. The *set* because losing a block from it is exactly the
 * bug this rung exists to make impossible — a block that stops being persisted
 * has to fail here rather than in a bug report six weeks later.
 */
test('the persisted block set is what shipped, per sim', () => {
  assert.deepEqual(persistedBlocks(PLIFE_BLOCKS), [
    'macros',
    'matrixGen',
    'population',
    'field',
    'budget',
    // Joined 2026-08-15 with the brightness-range lane (recipe v6 carries it;
    // `liftV5toV6` fills it for older sidecars).
    'luma',
  ]);
  assert.deepEqual(persistedBlocks(PHYSARUM_BLOCKS), ['macros', 'soil']);
  assert.deepEqual(persistedBlocks(VIZFX_BLOCKS), ['macros', 'energy']);
});

/**
 * The exhaustive opt-out assertion the roadmap asks for, on the real tables.
 *
 * A block that leaves the save channel has to be added here by hand, which is the
 * whole point: "not saved" is then a decision somebody made and somebody else
 * reviewed, rather than a name that was quietly never typed. The reason is
 * required to be a sentence rather than a word, because the reason is the part
 * that survives into the next reading of the code.
 */
test('every block that does not persist says why, and CI knows the list', () => {
  const pairs = (table: Parameters<typeof unpersistedBlocks>[0]): unknown =>
    unpersistedBlocks(table).map(({ block, kind }) => ({ block, kind }));

  assert.deepEqual(pairs(PLIFE_BLOCKS), [
    { block: 'palette', kind: 'persisted-elsewhere' },
    { block: 'render', kind: 'persisted-elsewhere' },
  ]);
  assert.deepEqual(pairs(PHYSARUM_BLOCKS), [
    { block: 'palette', kind: 'persisted-elsewhere' },
    { block: 'render', kind: 'persisted-elsewhere' },
  ]);
  assert.deepEqual(pairs(VIZFX_BLOCKS), [
    { block: 'palette', kind: 'persisted-elsewhere' },
    { block: 'params', kind: 'persisted-elsewhere' },
    { block: 'render', kind: 'persisted-elsewhere' },
  ]);

  for (const table of [PLIFE_BLOCKS, PHYSARUM_BLOCKS, VIZFX_BLOCKS]) {
    for (const { block, reason } of unpersistedBlocks(table)) {
      assert.ok(reason.length > 40, `"${block}" opts out with a reason too short to be one`);
    }
    for (const { block, field, reason } of sessionOnlyFields(table)) {
      assert.ok(reason.length > 40, `"${block}.${field}" opts out with a reason too short to be one`);
    }
  }

  // The only field-level override in the repo. It is deliberately awkward to
  // spell, and this is the assertion that keeps it rare.
  assert.deepEqual(sessionOnlyFields(PLIFE_BLOCKS), []);
  assert.deepEqual(sessionOnlyFields(VIZFX_BLOCKS), []);
  assert.deepEqual(
    sessionOnlyFields(PHYSARUM_BLOCKS).map(({ block, field }) => ({ block, field })),
    [{ block: 'soil', field: 'debugView' }],
  );
});

test('every persisted block has a schema, and it is the block that is live', () => {
  // A defaults function whose keys are not the live block's keys is a block that
  // half-persists: `readInto` walks the live object, so a field the defaults
  // function does not know about is never reset and a key the config does not
  // have is never written.
  for (const [table, cfg] of [
    [PLIFE_BLOCKS, defaultPlifeConfig()],
    [PHYSARUM_BLOCKS, defaultConfig(4)],
    [VIZFX_BLOCKS, defaultVizFxConfig(NEBULA)],
  ] as const) {
    for (const name of persistedBlocks(table as never)) {
      const live = (cfg as unknown as Record<string, object>)[name] as object;
      const decl = (table as unknown as Record<string, { defaults: (c: unknown) => object }>)[name];
      assert.ok(decl, name);
      const defaults = decl.defaults(cfg);
      assert.ok(Object.keys(defaults).length > 0, `${name} has no fields to persist`);
      assert.deepEqual(
        Object.keys(defaults).sort(),
        Object.keys(live).sort(),
        `${name}: the defaults function and the live block disagree about the schema`,
      );
    }
  }
});

/**
 * The serialized shape, pinned per sim, key for key.
 *
 * This rung was allowed to change *how* the blocks are chosen and nothing about
 * *what comes out*: every existing autosave, `modulation.json` and profile has to
 * keep loading, and an export recipe validates `$.simulation` against an exact
 * key list (`PLIFE_KEYS` in `runtime/recipe.ts`). So this is a byte-level
 * regression guard on the migration, not a restatement of the tables above —
 * `soil` in particular has to keep emitting exactly four keys, with `debugView`
 * absent, now that it is a declared block rather than a hand-built literal.
 */
test('the serialized extras shape is exactly what shipped before the registry', () => {
  assert.deepEqual(Object.keys(new PlifeSim(1, defaultPlifeConfig()).serializeExtras()), [
    'macros',
    'matrixGen',
    'population',
    'field',
    'budget',
    'luma',
    'speciesEnabled',
    'look',
  ]);

  const physarum = new PhysarumSim(1).serializeExtras();
  assert.deepEqual(Object.keys(physarum), ['macros', 'soil', 'stemDrive', 'stemGain', 'look']);
  assert.deepEqual(Object.keys(physarum['soil'] as object), [
    'decay',
    'accum',
    'depositBias',
    'senseBias',
  ]);

  assert.deepEqual(Object.keys(new VizFxSim(NEBULA, 1).serializeExtras()), [
    'macros',
    'energy',
    'emittersPerLayer',
    'look',
  ]);
});

// ── the named session-only state the roadmap called out ──────────────────────

test("plife's budget governor never reaches the file", () => {
  const sim = new PlifeSim(1234, defaultPlifeConfig());
  const saved = JSON.stringify(sim.serializeExtras());
  // Searched for by name across the whole blob rather than checked key by key:
  // the failure being prevented is somebody promoting the governor into
  // `PlifeBudgetConfig`, and it would then appear *inside* the budget block.
  assert.equal(
    saved.includes('effectiveBudget') || saved.includes('governorBudget'),
    false,
    'the frame-rate governor was serialised; a load would open at last night’s fps',
  );

  // …and the four settings that ARE the budget block still round trip, so this
  // is an assertion about the governor and not about the block being dropped.
  sim.config.budget.cap = 65_536;
  sim.config.budget.adaptive = false;
  sim.config.budget.floorFps = 45;
  const restored = new PlifeSim(1, defaultPlifeConfig());
  restored.applyExtras(sim.serializeExtras());
  assert.deepEqual(restored.config.budget, sim.config.budget);
});

test("the impulse lane's hold is session-only by not being config at all", () => {
  // `ui/impulses-panel.ts` binds it to an object named `sessionOnly`, which
  // nothing serialises. The structural half of that statement is that it is not
  // a field of `ImpulseConfig` — if it ever becomes one it is persisted by
  // `readInto` immediately, and a reload brings back a world stuck mid-hit.
  const cfg = defaultImpulseConfig() as Record<string, unknown>;
  assert.equal('hold' in cfg, false);
  for (const response of Object.values(cfg['responses'] as Record<string, object>)) {
    assert.equal('hold' in response, false);
  }
});

// ── the property that makes it worth having: adding a block costs one line ───

test('a block added to a real config round trips with nothing else registered', () => {
  // The acceptance test, spelled against `PlifeConfig` rather than a toy. The
  // only thing written for the new block is its one declaration; serialize,
  // restore, clamp and the "reset first" behaviour all follow from it.
  interface Grown extends PlifeConfig {
    brightnessCurve: { exponent: number; hdrPeak: number };
  }
  const GROWN_BLOCKS: BlockTable<Grown> = {
    ...PLIFE_BLOCKS,
    brightnessCurve: persisted(() => ({ exponent: 1, hdrPeak: 1 }), {
      rules: { exponent: range(0.25, 4), hdrPeak: range(1, 8) },
    }),
  };

  const authored = { ...defaultPlifeConfig(), brightnessCurve: { exponent: 2.5, hdrPeak: 4 } };
  assert.deepEqual(undeclaredBlocks(authored, GROWN_BLOCKS), []);
  assert.ok(persistedBlocks(GROWN_BLOCKS).includes('brightnessCurve'));

  const saved = JSON.parse(JSON.stringify(serializeBlocks(authored, GROWN_BLOCKS))) as Record<
    string,
    unknown
  >;
  const restored = { ...defaultPlifeConfig(), brightnessCurve: { exponent: 1, hdrPeak: 1 } };
  applyBlocks(restored, GROWN_BLOCKS, saved);
  assert.deepEqual(restored.brightnessCurve, { exponent: 2.5, hdrPeak: 4 });

  // …clamped by the rules it declared, and by nothing it had to register twice.
  applyBlocks(restored, GROWN_BLOCKS, { brightnessCurve: { exponent: 99, hdrPeak: 0 } });
  assert.deepEqual(restored.brightnessCurve, { exponent: 4, hdrPeak: 1 });

  // The existing blocks are untouched by the extension: the spread is a table,
  // not a schema change.
  const decl = GROWN_BLOCKS.macros;
  assert.ok(decl.kind === 'persisted' && 'density' in blockRules(decl, authored));
});
