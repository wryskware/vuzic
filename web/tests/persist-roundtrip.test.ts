/**
 * The round-trip property, for the whole settings surface.
 *
 * ## What this is defending
 *
 * Persistence in this app is asymmetric: the write side is `JSON.stringify` (or
 * a block spread), which is total, and the read side used to be a hand-written
 * allowlist. A field added to a config and not added to the reader was therefore
 * *written* and silently *dropped*, and the only signal was a user eventually
 * noticing that one setting kept resetting. `Palette.accentArc` shipped that way;
 * the whole `ImpulseConfig` was never persisted at all.
 *
 * So these tests do not check a list of fields. A list is the thing that failed.
 * They **derive** the fields from the defaults functions — the same objects
 * `readInto` walks — perturb every leaf to a distinct legal non-default value,
 * round trip, and demand equality. Adding a field to any config below therefore
 * puts it under test automatically, and a reader that drops it fails here rather
 * than in a bug report six weeks later.
 *
 * The exception lists are deliberately explicit and small. Anything named there
 * is a field whose round trip is *intentionally* lossy, and each one carries the
 * reason. A new exception should be hard to add and obvious in review; that is
 * the point of listing them by name rather than by predicate.
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads modules straight
 * out of src/, so every import carries an explicit `.ts` extension, and nothing
 * here touches the DOM or WebGPU.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// `PlifeSim` reaches its shaders and some of its siblings without an explicit
// extension, which the ESM loader will not resolve on its own. Same hooks
// `sim-bundle.test.ts` installs, and for the same reason: the sim under test is
// the real one, not a stub.
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && !/\.(ts|js|mjs|wgsl)(\?|$)/.test(specifier)) {
      try {
        return next(`${specifier}.ts`, context);
      } catch {
        // Not a TypeScript module; preserve the real resolution error below.
      }
    }
    return next(specifier, context);
  },
  load(url, context, next) {
    if (!url.includes('.wgsl')) return next(url, context);
    const text = readFileSync(fileURLToPath(url.split('?')[0] ?? url), 'utf8');
    return { format: 'module', shortCircuit: true, source: `export default ${JSON.stringify(text)};` };
  },
});

const [
  { defaultModulationConfig, parseModulation, serializeModulation },
  { readInto, range, oneOf },
  { defaultBoundary },
  { defaultImpulseConfig, IMPULSE_RULES },
  { defaultRenderConfig },
  { defaultArc, defaultAccentArc, mergePalette },
  { defaultPalette },
  { defaultExtrasBlocks, defaultPlifeConfig, extrasRules, MACRO_RANGE },
  { PlifeSim },
] = await Promise.all([
  import('../src/mapping/persist.ts'),
  import('../src/mapping/read-into.ts'),
  import('../src/mapping/types.ts'),
  import('../src/sim/impulses.ts'),
  import('../src/sim/render/config.ts'),
  import('../src/sim/palette.ts'),
  import('../src/sim/physarum/config.ts'),
  import('../src/sim/plife/config.ts'),
  import('../src/sim/plife/plife.ts'),
]);

// ── the perturber ────────────────────────────────────────────────────────────

interface Leaf {
  min?: number;
  max?: number;
  int?: boolean;
  oneOf?: readonly string[];
}

/** The rule objects are symbol-tagged; reading them back only needs the fields. */
function leafOf(rule: unknown): Leaf | null {
  if (rule === null || typeof rule !== 'object') return null;
  const r = rule as Leaf;
  return r.oneOf !== undefined || r.min !== undefined ? r : null;
}

/** Six decimals is what `serializeModulation` writes; anything finer cannot survive. */
function round6(v: number): number {
  return Number(v.toFixed(6));
}

/**
 * Move every leaf of `obj` to a distinct legal value that is **not** the default.
 *
 * "Not the default" is what makes the assertion meaningful: a reader that ignored
 * a field entirely would still round trip if the perturbed value happened to
 * equal the default it falls back to.
 *
 * Values land at 37.1% of the declared range because it is an ugly fraction —
 * a midpoint would coincide with too many shipped defaults to notice a reader
 * that quietly substituted one.
 */
function perturb<T extends object>(obj: T, rules: Record<string, unknown> = {}): T {
  const target = obj as Record<string, unknown>;
  for (const key of Object.keys(target)) {
    const current = target[key];
    const rule = rules[key];
    const leaf = leafOf(rule);

    if (current !== null && typeof current === 'object' && !Array.isArray(current)) {
      perturb(current, leaf ? {} : ((rule ?? {}) as Record<string, unknown>));
      continue;
    }
    if (typeof current === 'number') {
      let next: number;
      if (leaf && leaf.min !== undefined && Number.isFinite(leaf.max)) {
        const { min, max } = leaf as { min: number; max: number };
        next = min + (max - min) * 0.371;
        if (leaf.int === true) next = Math.round(next);
        // A range whose 37.1% point *is* the default still has to move.
        if (next === current) next = leaf.int === true ? Math.round((min + max) / 2) : (min + max) / 2;
      } else {
        // Unbounded (or half-bounded) field: any distinct finite value will do,
        // and staying positive keeps it legal for the ones that are rates.
        next = Math.abs(current) * 1.7 + 0.317;
      }
      target[key] = round6(next);
      assert.notEqual(target[key], current, `perturb produced the default for "${key}"`);
      continue;
    }
    if (typeof current === 'boolean') {
      target[key] = !current;
      continue;
    }
    if (typeof current === 'string') {
      if (leaf?.oneOf) {
        const other = leaf.oneOf.find((v) => v !== current);
        assert.ok(other, `"${key}" has a one-value vocabulary; nothing to perturb to`);
        target[key] = other;
      } else {
        // The only free-form strings in these configs are sRGB hexes.
        target[key] = current.startsWith('#') ? '#123456' : `${current}-perturbed`;
      }
      continue;
    }
  }
  return obj;
}

/** Every leaf path of an object, so a diff can say *which* field failed to survive. */
function leaves(obj: unknown, prefix = ''): Map<string, unknown> {
  const out = new Map<string, unknown>();
  if (obj === null || typeof obj !== 'object') {
    out.set(prefix, obj);
    return out;
  }
  if (Array.isArray(obj)) {
    obj.forEach((v, i) => {
      for (const [k, val] of leaves(v, `${prefix}[${i}]`)) out.set(k, val);
    });
    return out;
  }
  for (const [key, value] of Object.entries(obj)) {
    for (const [k, val] of leaves(value, prefix ? `${prefix}.${key}` : key)) out.set(k, val);
  }
  return out;
}

function assertSurvives(before: unknown, after: unknown, skip: readonly string[], what: string): void {
  const a = leaves(before);
  const b = leaves(after);
  const lost: string[] = [];
  for (const [path, value] of a) {
    if (skip.some((s) => path === s || path.startsWith(`${s}.`) || path.startsWith(`${s}[`))) {
      continue;
    }
    if (!b.has(path)) {
      lost.push(`${path}: missing after round trip`);
    } else if (!Object.is(b.get(path), value)) {
      lost.push(`${path}: wrote ${JSON.stringify(value)}, read back ${JSON.stringify(b.get(path))}`);
    }
  }
  assert.deepEqual(lost, [], `${what} lost ${lost.length} field(s) in the round trip:\n  ${lost.join('\n  ')}`);
}

// ── the mapping file ─────────────────────────────────────────────────────────

/**
 * Fields whose round trip is intentionally lossy, each for a stated reason.
 *
 * Short by design. Every entry is a place the parser is allowed to disagree with
 * the file, and anything not listed must survive byte for byte.
 */
const MODULATION_EXCEPTIONS = [
  // The parser's output is always the current version, whatever the file said.
  'version',
  // Sizes allocations, so it is validated up front and echoed rather than read.
  'speciesCount',
  // A derived cache in arc mode: `syncPaletteColors` re-derives it from the arcs
  // on load, so a perturbed hex is *supposed* not to survive. Custom mode is
  // covered by its own test below.
  'palette.colors',
  // Length is the track's driver count, which a config file does not determine.
  'driverGains',
];

test('every field of a mapping file survives serialize → parse', () => {
  const base = defaultPlifeConfig();
  const authored = perturb(defaultModulationConfig(base, 'plife'), {
    palette: {
      mode: oneOf(['arc', 'custom']),
      space: oneOf(['hsluv', 'oklch']),
      hueRateDegPerSec: range(-10, 10),
      arc: { sat: range(0, 100), light: range(0, 100) },
      accentArc: { sat: range(0, 100), light: range(0, 100) },
    },
    render: { grade: { tonemap: oneOf(['reinhard', 'aces']) } },
    impulses: IMPULSE_RULES,
    stemFollow: { floor: range(0, 1) },
    boundary: { snapFraction: range(0, 1), respawnFraction: range(0, 1) },
  });
  // Three fields the perturber is not allowed to touch, for reasons outside the
  // round trip: the version has to name a version the parser accepts, K sizes
  // the palette, and the discriminator has to keep naming a real sim or
  // `modulationFits` would reject the result for the wrong reason.
  authored.version = 5;
  authored.speciesCount = base.speciesCount;
  authored.sim = 'plife';

  const parsed = parseModulation(serializeModulation(authored));
  assertSurvives(authored, parsed, MODULATION_EXCEPTIONS, 'the mapping file');
  assert.equal(parsed.version, 5);
  assert.equal(parsed.speciesCount, base.speciesCount);
});

test('a custom-mode palette keeps its authored hexes, arcs and all', () => {
  const base = defaultPlifeConfig();
  const authored = defaultModulationConfig(base, 'plife');
  authored.palette.mode = 'custom';
  authored.palette.colors = authored.palette.colors.map((_, i) => `#0${i}0${i}0${i}`);
  // The accents' arc is the field that shipped broken — carried in the file,
  // dropped by the parser — so it is pinned by name as well as by the sweep.
  authored.palette.accentArc = { hueStartDeg: 137, hueRangeDeg: -211, sat: 63, light: 41 };
  authored.palette.arc = { hueStartDeg: 22, hueRangeDeg: 180, sat: 71, light: 55 };

  const parsed = parseModulation(serializeModulation(authored));
  assert.deepEqual(parsed.palette.colors, authored.palette.colors);
  assert.deepEqual(parsed.palette.accentArc, authored.palette.accentArc);
  assert.deepEqual(parsed.palette.arc, authored.palette.arc);
});

test('arc-mode accent settings survive a reload', () => {
  const base = defaultPlifeConfig();
  const authored = defaultModulationConfig(base, 'plife');
  authored.palette.mode = 'arc';
  authored.palette.accentArc = { hueStartDeg: 300, hueRangeDeg: 90, sat: 44, light: 88 };

  const parsed = parseModulation(serializeModulation(authored));
  assert.equal(parsed.palette.mode, 'arc');
  assert.deepEqual(parsed.palette.accentArc, authored.palette.accentArc);
  // And the derived cache actually followed the arc it was told to follow,
  // rather than surviving as whatever was in the file.
  assert.notDeepEqual(parsed.palette.colors, defaultPalette(base.speciesCount).colors);
});

test('the impulse lane is persisted with the mapping, kind by kind', () => {
  const base = defaultPlifeConfig();
  const authored = defaultModulationConfig(base, 'plife');
  authored.impulses.gain = 2.15;
  authored.impulses.enabled = false;
  const snare = authored.impulses.responses.snare;
  snare.decayMs = 777;
  snare.splashRadius = 0.41;
  snare.splashSwirl = -1.25;
  snare.species = 3;

  const parsed = parseModulation(serializeModulation(authored));
  assert.deepEqual(parsed.impulses, authored.impulses);
});

test('a mapping file written before the impulse lane loads with the shipped responses', () => {
  const base = defaultPlifeConfig();
  const authored = defaultModulationConfig(base, 'plife') as Record<string, unknown>;
  delete authored['impulses'];

  const parsed = parseModulation(JSON.stringify(authored));
  assert.deepEqual(parsed.impulses, defaultImpulseConfig());
});

test('merging a loaded palette into the live one carries every field', () => {
  // The layer the round-trip tests above do NOT reach, and the one that actually
  // broke: a file can be saved correctly and parsed correctly and still lose a
  // field on the way into the live config, because the live palette is a shared
  // object that a load has to *merge* into rather than replace. That merge was a
  // hand-listed sequence of assignments and had gone stale by two fields
  // (`space`, `accentArc`) — which is precisely the reported accent-arc bug, and
  // it survived every test in this file until this one existed.
  const live = defaultPalette(8);
  const loaded = perturb(defaultPalette(8), {
    mode: oneOf(['arc', 'custom']),
    space: oneOf(['hsluv', 'oklch']),
    arc: { sat: range(0, 100), light: range(0, 100) },
    accentArc: { sat: range(0, 100), light: range(0, 100) },
  });

  mergePalette(live, loaded);
  assertSurvives(loaded, live, [], 'the palette merge');
  // And the merge must not have swapped the object the panel is bound to.
  assert.ok(Array.isArray(live.colors));
});

test('the palette merge keeps the live length when the file was authored at another K', () => {
  const live = defaultPalette(8);
  const loaded = defaultPalette(3);
  loaded.colors = ['#111111', '#222222', '#333333'];
  mergePalette(live, loaded);
  assert.equal(live.colors.length, 8, 'a shorter file resized the array the sim indexes');
  assert.deepEqual(live.colors.slice(0, 3), ['#111111', '#222222', '#333333']);
});

// ── the opaque per-sim extras block ──────────────────────────────────────────

/**
 * `look` is not a config block — `exposure` and `gamma` live on the config root —
 * and `speciesEnabled` is an array keyed to the live K, so both are checked
 * separately below rather than by the block sweep.
 */
const EXTRAS_EXCEPTIONS = ['look', 'speciesEnabled'];

test("every field of plife's extras block survives serialize → apply", () => {
  const authored = new PlifeSim(1234, defaultPlifeConfig());
  const rules = extrasRules(authored.config.maxParticles);
  for (const [name, live] of Object.entries(defaultExtrasBlocks(authored.config.maxParticles))) {
    const block = (authored.config as unknown as Record<string, object>)[name];
    assert.ok(block, `extras block "${name}" is not on the config`);
    readInto(block, perturb(live, rules[name] as Record<string, unknown>), rules[name]);
  }
  // The band ordering the loader restores would otherwise be the one legitimate
  // difference between what we authored and what comes back.
  const g = authored.config.matrixGen;
  for (const b of [g.rMin, g.rMax]) {
    const { lo, hi } = b;
    b.lo = Math.min(lo, hi);
    b.hi = Math.max(lo, hi);
  }

  const saved = authored.serializeExtras();
  const restored = new PlifeSim(9999, defaultPlifeConfig());
  restored.applyExtras(saved);

  for (const name of Object.keys(defaultExtrasBlocks(authored.config.maxParticles))) {
    if (EXTRAS_EXCEPTIONS.includes(name)) continue;
    const before = (authored.config as unknown as Record<string, object>)[name];
    const after = (restored.config as unknown as Record<string, object>)[name];
    assertSurvives(before, after, [], `extras block "${name}"`);
  }
});

test('applyExtras writes in place, so the panel keeps editing what runs', () => {
  const sim = new PlifeSim(1, defaultPlifeConfig());
  // Exactly the references a tweakpane binding holds.
  const held = [
    sim.config.macros,
    sim.config.population,
    sim.config.population.accent,
    sim.config.matrixGen,
    sim.config.matrixGen.rMin,
    sim.config.matrixGen.rMax,
    sim.config.field,
    sim.config.budget,
  ];
  sim.applyExtras(sim.serializeExtras());
  assert.equal(sim.config.macros, held[0]);
  assert.equal(sim.config.population, held[1]);
  assert.equal(sim.config.population.accent, held[2]);
  assert.equal(sim.config.matrixGen, held[3]);
  assert.equal(sim.config.matrixGen.rMin, held[4]);
  assert.equal(sim.config.matrixGen.rMax, held[5]);
  assert.equal(sim.config.field, held[6]);
  assert.equal(sim.config.budget, held[7]);
});

test('scene exposure and gamma round trip through the look block', () => {
  const sim = new PlifeSim(1, defaultPlifeConfig());
  sim.config.exposure = 0.63;
  sim.config.gamma = 2.71;
  const restored = new PlifeSim(2, defaultPlifeConfig());
  restored.applyExtras(sim.serializeExtras());
  assert.equal(restored.config.exposure, 0.63);
  assert.equal(restored.config.gamma, 2.71);
});

test('per-species enable flags round trip, and a short list pads with present', () => {
  const sim = new PlifeSim(1, defaultPlifeConfig());
  const wanted = sim.config.species.map((_, i) => i % 3 !== 0);
  sim.config.species.forEach((s, i) => (s.enabled = wanted[i] as boolean));

  const restored = new PlifeSim(2, defaultPlifeConfig());
  restored.applyExtras(sim.serializeExtras());
  assert.deepEqual(
    restored.config.species.map((s) => s.enabled),
    wanted,
  );

  const short = new PlifeSim(3, defaultPlifeConfig());
  short.applyExtras({ speciesEnabled: [false, false] });
  assert.deepEqual(
    short.config.species.map((s) => s.enabled),
    [false, false, ...Array<boolean>(short.config.speciesCount - 2).fill(true)],
  );
});

test('applyExtras is a function of the blob, not of history', () => {
  // The explorer fans one serialised block out to nine tiles twice a second, so
  // a second apply with a *partial* block must not leave the first apply's
  // values behind — that would make a tile's look depend on what it showed last.
  const sim = new PlifeSim(1, defaultPlifeConfig());
  sim.applyExtras({ macros: { density: 1.9, chaos: 0.2 } });
  assert.equal(sim.config.macros.density, 1.9);
  sim.applyExtras({ macros: { chaos: 0.4 } });
  assert.equal(sim.config.macros.chaos, 0.4);
  assert.equal(sim.config.macros.density, 1, 'a field absent from the blob must return to its default');
});

// ── the reader itself ────────────────────────────────────────────────────────

test('readInto never throws and never invents a field', () => {
  const dst = { a: 1, b: true, c: 'x', nested: { d: 2 } };
  const hostile = JSON.parse(
    '{"a":"not a number","b":1,"c":7,"nested":null,"__proto__":{"polluted":true},"extra":9}',
  ) as unknown;
  readInto(dst, hostile);
  assert.deepEqual(dst, { a: 1, b: true, c: 'x', nested: { d: 2 } });
  assert.equal((dst as Record<string, unknown>)['extra'], undefined);
  assert.equal(({} as Record<string, unknown>)['polluted'], undefined);
  // Non-objects are simply "no fields present".
  assert.deepEqual(readInto({ a: 1 }, 42), { a: 1 });
  assert.deepEqual(readInto({ a: 1 }, null), { a: 1 });
  assert.deepEqual(readInto({ a: 1 }, [3]), { a: 1 });
});

test('readInto clamps to the rule, rounds integers, and rejects unknown enum values', () => {
  const dst = { n: 1, i: 1, s: 'grid' };
  readInto(dst, { n: 99, i: 2.6, s: 'nonsense' }, {
    n: range(0, 10),
    i: range(1, 5, { int: true }),
    s: oneOf(['grid', 'brute']),
  });
  assert.equal(dst.n, 10);
  assert.equal(dst.i, 3);
  assert.equal(dst.s, 'grid');
});

test('a field with a default but no rule still round trips', () => {
  // The property that makes the rule tables safe to leave incomplete: rules
  // clamp, defaults enrol. If this ever fails, someone has made the rule table
  // load-bearing for persistence and the forgetting is back.
  const withNoRules = perturb(defaultRenderConfig(), {});
  const copy = readInto(defaultRenderConfig(), JSON.parse(JSON.stringify(withNoRules)));
  assert.deepEqual(copy, withNoRules);
});

test('the blocks with defaults functions are the blocks that persist', () => {
  // A guard on the arrangement rather than on any one field: each of these is a
  // defaults function whose keys are the persisted schema, so an empty or
  // partial one would silently shrink what survives a reload.
  for (const [what, value] of [
    ['boundary', defaultBoundary()],
    ['impulses', defaultImpulseConfig()],
    ['render', defaultRenderConfig()],
    ['arc', defaultArc()],
    ['accentArc', defaultAccentArc()],
    ['macros', defaultExtrasBlocks(1024)['macros']],
  ] as const) {
    assert.ok(Object.keys(value as object).length > 0, `${what} has no fields to persist`);
  }
  // The macro rig's rule table and its defaults must name the same knobs, or a
  // macro is either unclamped or clamped against a knob that does not exist.
  assert.deepEqual(
    Object.keys(defaultExtrasBlocks(1024)['macros'] as object).sort(),
    Object.keys(MACRO_RANGE).sort(),
  );
});
