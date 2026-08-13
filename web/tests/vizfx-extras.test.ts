/**
 * The vizfx chassis as the mapping layer actually meets it: a `ModTarget` whose
 * θ is a generated table and whose out-of-θ state travels in `ModulationConfig`'s
 * opaque `extras` block.
 *
 * These drive the **real `VizFxSim`**, constructed headlessly — see
 * `./vizfx-modules.ts` for the two-hook shim that makes the module loadable and
 * for why the constructor touches no GPU. `extras-persist.test.ts` notes that no
 * test in this directory constructs a Sim; this is the first, and it is worth the
 * shim for one reason: `mapping/target.ts` says `applyExtras` "**must tolerate
 * `undefined` and garbage** … It must never throw: a broken extras block costs
 * its own values, not the whole file." That is a contract about a real method's
 * behaviour on hostile input, and a reimplementation of it in the test would
 * assert nothing.
 *
 * What these pin:
 *
 * - the registry the Modulator holds **by identity**, and its agreement with the
 *   table the panel and the persistence layer read;
 * - the θ round trip through `currentVector` / `applyTheta`, including the mask
 *   and the species-brightness mirror;
 * - `serializeExtras` → `applyExtras` as a genuine round trip, in place;
 * - `applyExtras` against every shape a JSON file can hand it.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  defaultModulationConfig,
  modulationFits,
  parseModulation,
  serializeModulation,
} from '../src/mapping/persist.ts';
import {
  ENERGY_RANGE,
  MACRO_RANGE,
  MAX_EMITTERS_PER_LAYER,
  defaultEnergy,
} from '../src/sim/vizfx/config.ts';
import {
  defaultParams,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  vectorLength,
} from '../src/sim/vizfx/slots.ts';
import { NEBULA, VizFxSim } from './vizfx-modules.ts';

const LEN = vectorLength(NEBULA);
const NAMES = fieldNames(NEBULA);
const MASK = modulationMask(NEBULA);
const K = NEBULA.speciesCount;

function sim(): InstanceType<typeof VizFxSim> {
  return new VizFxSim(NEBULA, 0x5eed_1234);
}

// ── the ModTarget seam ───────────────────────────────────────────────────────

test('the registry is the table, handed out by identity', () => {
  const s = sim();
  const a = s.registry();
  // `Modulator.setConfig` passes `.classes` straight into a SlewLimiter that
  // holds it by reference, so a fresh object here would quietly detach the
  // limiter from the registry — the sim's own comment, and it is the reason this
  // is an identity assertion rather than a deepEqual.
  assert.equal(s.registry(), a, 'the registry is rebuilt on every ask');
  assert.equal(a.classes, s.registry().classes);

  assert.equal(a.length, LEN);
  assert.deepEqual([...a.names], NAMES);
  assert.deepEqual(a.slots, modulationSlots(NEBULA));
  assert.deepEqual([...a.mask], [...MASK]);
  assert.deepEqual([...a.classes], [...fieldClasses(NEBULA)]);
  // No `seedBase`: unlike plife, nothing in this θ is drawn wholesale from the
  // seed — a reroll's other half (the emitter constellation) is outside θ.
  assert.equal(a.seedBase, undefined);
});

test('the persistence discriminator is the visual, not the family', () => {
  // θ is a different vector per visual, so two visuals must never share an
  // autosave slot — "the family is code reuse; it is not a persistence key".
  const s = sim();
  assert.equal(s.simId, 'nebula');
  assert.equal(s.simId, NEBULA.id);
  const cfg = defaultModulationConfig(s.config, s.simId);
  assert.ok(modulationFits(cfg, K, 'nebula'));
  assert.ok(!modulationFits(cfg, K, 'vizfx'), 'a file must not be shared with the family');
  assert.ok(!modulationFits(cfg, K, 'physarum'));
  assert.ok(!modulationFits(cfg, K + 1, 'nebula'));
});

test('currentVector ↔ applyTheta round-trips, and the mask protects the exclusions', () => {
  const s = sim();
  const v0 = s.currentVector();
  assert.equal(v0.length, LEN);
  s.applyTheta(v0);
  const v1 = s.currentVector();
  for (let i = 0; i < LEN; i++) assert.equal(v1[i], v0[i], NAMES[i]);

  // A full-range masked write leaves brightness, exposure and gamma alone — two
  // separate mechanisms (a null spec and a zero mask bit) and this is the second.
  s.config.params['layer0.brightness'] = 0.42;
  s.config.params['exposure'] = 0.07;
  s.applyTheta(new Float64Array(LEN).fill(9), MASK);
  assert.equal(s.config.params['layer0.brightness'], 0.42);
  assert.equal(s.config.params['exposure'], 0.07);
  // …and every masked slot took the write, clamped to its hard bound.
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 0) continue;
    const x = s.config.params[NAMES[i] as string] as number;
    assert.ok(Number.isFinite(x), `${NAMES[i]} = ${x}`);
    assert.ok(x <= 9, `${NAMES[i]} = ${x} was written past its hard max`);
  }
});

test('the workbench brightness readout is a mirror of θ, not a second copy', () => {
  // `species[k].brightness` is what the stem-follow readout shows; θ is the
  // source of truth and `syncSpecies` mirrors one into the other, so an
  // applyTheta must move both.
  const s = sim();
  const v = s.currentVector();
  for (let k = 0; k < K; k++) v[NAMES.indexOf(`layer${k}.brightness`)] = 0.25 + k * 0.1;
  s.applyTheta(v);
  for (let k = 0; k < K; k++) {
    assert.equal(s.config.species[k]?.brightness, 0.25 + k * 0.1, `species ${k}`);
    assert.equal(s.config.params[`layer${k}.brightness`], 0.25 + k * 0.1);
  }
  assert.deepEqual(
    s.config.species.map((x) => x.name),
    [...NEBULA.layerNames],
  );
});

// ── extras: the round trip ───────────────────────────────────────────────────

test('serializeExtras → applyExtras restores the whole out-of-θ block', () => {
  const a = sim();
  a.config.macros['light'] = 1.7;
  a.config.macros['chroma'] = 0.2;
  a.config.energy.followStems = false;
  a.config.energy.floor = 0.15;
  a.config.energy.curve = 2.4;
  a.config.energy.smoothingMs = 1200;
  a.config.emittersPerLayer = 6;
  const block = a.serializeExtras();

  const b = sim();
  b.applyExtras(block);
  assert.deepEqual(b.serializeExtras(), block);
  assert.deepEqual(b.config.macros, a.config.macros);
  assert.deepEqual(b.config.energy, a.config.energy);
  assert.equal(b.config.emittersPerLayer, 6);

  // The three blocks the sim is expected to carry, and no fourth: anything
  // outside θ *and* outside this block is a panel edit the explorer tiles never
  // see, which is the failure mode the comment on `serializeExtras` warns about.
  // `look` is the one θ-derived member, and deliberate: `exposure` and `gamma`
  // are excluded from modulation and owned by the look tab, and nothing else in
  // the file persists θ — without it, every reload reset the exposure.
  assert.deepEqual(Object.keys(block).sort(), ['emittersPerLayer', 'energy', 'look', 'macros']);
  // …and the block is a snapshot, not a live view of the config.
  a.config.macros['light'] = 0.1;
  assert.equal((block['macros'] as Record<string, number>)['light'], 1.7);
});

test('applyExtras writes in place, because the panel holds those objects by reference', () => {
  const s = sim();
  const macros = s.config.macros;
  const energy = s.config.energy;
  s.applyExtras({ macros: { light: 0.5 }, energy: { floor: 0.9 }, emittersPerLayer: 2 });
  assert.equal(s.config.macros, macros, 'config.macros was replaced, detaching the panel');
  assert.equal(s.config.energy, energy, 'config.energy was replaced, detaching the panel');
  assert.equal(macros['light'], 0.5);
  assert.equal(energy.floor, 0.9);
});

test('an unlisted macro key cannot enter the rig', () => {
  // `applyExtras` walks the *visual's* macro list, so a file from another visual
  // (or a hand-edited one) cannot add a knob the panel never built and the
  // shader never reads.
  const s = sim();
  s.applyExtras({ macros: { light: 1.5, bogus: 9, __proto__: { light: 0 } } });
  assert.deepEqual(Object.keys(s.config.macros).sort(), NEBULA.macros.map((m) => m.key).sort());
  assert.equal(s.config.macros['light'], 1.5);
  assert.equal(s.config.macros['bogus'], undefined);
  assert.equal(({} as Record<string, unknown>)['light'], undefined, 'Object.prototype was polluted');
});

// ── extras: garbage ──────────────────────────────────────────────────────────

/** Everything a JSON file can put where an object belongs, and then some. */
const HOSTILE: unknown[] = [
  undefined,
  {},
  { macros: null, energy: null, emittersPerLayer: null },
  { macros: 'light', energy: 42, emittersPerLayer: 'four' },
  { macros: [1, 2], energy: [], emittersPerLayer: [] },
  { macros: { light: 'x', motion: null, scale: {}, chroma: [] }, energy: { floor: 'low' } },
  { macros: { light: Number.NaN }, energy: { floor: Number.NaN, curve: Number.NaN, smoothingMs: Number.NaN } },
  {
    macros: { light: Number.POSITIVE_INFINITY, motion: Number.NEGATIVE_INFINITY },
    energy: { floor: 1e308, curve: -1e308, smoothingMs: 1e-308 },
    emittersPerLayer: Number.POSITIVE_INFINITY,
  },
  { macros: { light: -5, motion: 99 }, energy: { floor: -1, curve: 0, smoothingMs: 0 }, emittersPerLayer: 0 },
  { energy: { followStems: 'yes' }, emittersPerLayer: -3 },
  { energy: { followStems: 0 }, emittersPerLayer: 1e9 },
  JSON.parse('{"macros":{"light":2},"energy":{"__proto__":{"floor":9}},"emittersPerLayer":3.4}'),
  { macros: { light: 1 }, energy: { floor: 0.5 }, emittersPerLayer: 4, unknownBlock: { a: 1 } },
];

test('applyExtras never throws, whatever the file contains', () => {
  // The block is opaque to every layer between the file and here, so nothing
  // upstream has validated it — `mapping/target.ts` makes tolerating that the
  // method's job. A throw here would cost the whole load, not one field.
  for (const raw of HOSTILE) {
    const s = sim();
    assert.doesNotThrow(
      () => s.applyExtras(raw as Record<string, unknown> | undefined),
      `threw on ${JSON.stringify(raw) ?? 'undefined'}`,
    );
  }
});

test('applyExtras leaves every field inside the range its own slider shows', () => {
  for (const raw of HOSTILE) {
    const s = sim();
    s.applyExtras(raw as Record<string, unknown> | undefined);
    const label = JSON.stringify(raw) ?? 'undefined';

    for (const m of NEBULA.macros) {
      const x = s.config.macros[m.key] as number;
      assert.ok(Number.isFinite(x), `${m.key} = ${x} from ${label}`);
      assert.ok(x >= MACRO_RANGE.min && x <= MACRO_RANGE.max, `${m.key} = ${x} from ${label}`);
    }

    const e = s.config.energy;
    assert.equal(typeof e.followStems, 'boolean', `followStems from ${label}`);
    for (const [key, range] of Object.entries(ENERGY_RANGE)) {
      const x = e[key as 'floor' | 'curve' | 'smoothingMs'];
      assert.ok(Number.isFinite(x), `${key} = ${x} from ${label}`);
      assert.ok(x >= range.min && x <= range.max, `${key} = ${x} from ${label}`);
    }

    const n = s.config.emittersPerLayer;
    assert.ok(Number.isInteger(n), `emittersPerLayer = ${n} from ${label}`);
    assert.ok(n >= 1 && n <= MAX_EMITTERS_PER_LAYER, `emittersPerLayer = ${n} from ${label}`);
  }
});

test('a missing field takes the shipped default, not the previous value or zero', () => {
  const s = sim();
  s.config.energy.floor = 0.05;
  s.config.energy.curve = 3.3;
  s.config.macros['motion'] = 0.3;
  s.applyExtras({});
  assert.deepEqual(s.config.energy, defaultEnergy());
  assert.equal(s.config.macros['motion'], 1, 'an absent macro resets to neutral, not to 0');
  assert.equal(s.config.emittersPerLayer, NEBULA.emittersPerLayer);
});

test('emittersPerLayer is rounded, because the shader indexes with it', () => {
  const s = sim();
  s.applyExtras({ emittersPerLayer: 3.4 });
  assert.equal(s.config.emittersPerLayer, 3);
  s.applyExtras({ emittersPerLayer: 2.5 });
  assert.equal(s.config.emittersPerLayer, 3, 'Math.round, and away from zero on the half');
  s.applyExtras({ emittersPerLayer: 99 });
  assert.equal(s.config.emittersPerLayer, MAX_EMITTERS_PER_LAYER);
  s.applyExtras({ emittersPerLayer: 0.2 });
  assert.equal(s.config.emittersPerLayer, 1, 'a layer with no emitters would paint nothing');
});

test('the emitter constellation is a pure function of the seed', () => {
  // "A reroll is a different creature, not a different mood": the seeded
  // personality moves θ and `redrawEmitters` rearranges the constellation, which
  // is outside θ entirely. Two claims, both from the sim's own comments — the
  // placement is seeded (so a style sync that changes the count is idempotent
  // rather than a flicker) and it does depend on the seed.
  //
  // Reached through the private field because the placement has no other
  // observable without a GPU; the assertion below fails loudly if it is renamed.
  const read = (s: InstanceType<typeof VizFxSim>): number[] => {
    const data = (s as unknown as { emitterData: Float32Array }).emitterData;
    assert.ok(data instanceof Float32Array, 'emitterData is no longer where this test looks');
    return [...data];
  };

  const a = sim();
  a.applyExtras({ emittersPerLayer: 5 });
  const b = sim();
  b.applyExtras({ emittersPerLayer: 5 });
  assert.deepEqual(read(a), read(b), 'the same seed placed a different constellation');
  // …and repeating the sync — which the explorer does twice a second — changes
  // nothing, since the count is unchanged the second time.
  a.applyExtras({ emittersPerLayer: 5 });
  assert.deepEqual(read(a), read(b));

  const c = new VizFxSim(NEBULA, 0x0dd_beef);
  c.applyExtras({ emittersPerLayer: 5 });
  const [x, y] = [read(a), read(c)];
  assert.ok(x.some((v, i) => v !== y[i]), 'a different seed placed the same constellation');
});

// ── extras through the mapping file ──────────────────────────────────────────

test('the whole block survives a real serialize → parse → applyExtras', () => {
  // The two consumers of this channel are persistence and the explorer's style
  // sync; this is the persistence one, end to end, including the six-decimal
  // rounding the serialiser applies to every number it walks.
  const a = sim();
  a.config.macros['scale'] = 1.25;
  a.config.energy.floor = 0.55;
  a.config.energy.smoothingMs = 900;
  a.config.energy.followStems = false;
  a.config.emittersPerLayer = 5;

  const cfg = defaultModulationConfig(a.config, a.simId);
  cfg.extras = a.serializeExtras();
  const back = parseModulation(serializeModulation(cfg));
  assert.equal(back.sim, 'nebula');
  assert.deepEqual(back.extras, cfg.extras, 'the mapping layer reshaped an opaque block');

  const b = sim();
  b.applyExtras(back.extras);
  assert.deepEqual(b.serializeExtras(), a.serializeExtras());
});

test('a mapping file with no extras at all leaves the sim on its defaults', () => {
  const s = sim();
  const cfg = defaultModulationConfig(s.config, s.simId);
  const back = parseModulation(serializeModulation(cfg));
  assert.equal('extras' in back, false, 'a sim that stored nothing must not grow a null key');
  s.applyExtras(back.extras);
  assert.deepEqual(s.config.macros, Object.fromEntries(NEBULA.macros.map((m) => [m.key, 1])));
  assert.deepEqual(s.config.energy, defaultEnergy());
});

test('θ itself never travels in extras — it is the vector, and only the vector', () => {
  // The seam's whole point: `ModulationConfig` carries θ as a vector the mapping
  // layer understands and carries everything else opaquely. A parameter that
  // leaked into `extras` would be saved twice and modulated once.
  // The two exceptions are `exposure` and `gamma`: θ slots with `mod: null` and
  // no `folder`, so they are never written by the modulator and nothing else
  // saves them. Saved twice, modulated once is the failure this guards; saved
  // once, modulated never is what those two are.
  const extras = sim().serializeExtras();
  delete extras['look'];
  const block = JSON.stringify(extras);
  for (const name of Object.keys(defaultParams(NEBULA))) {
    const key = name.includes('.') ? (name.split('.')[1] as string) : name;
    assert.ok(!block.includes(`"${key}"`), `${key} appears in the extras block as well as in θ`);
  }
});
