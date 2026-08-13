/**
 * Headless tests for the vizfx chassis's declarative θ table (`sim/vizfx/slots.ts`)
 * and for nebula's instance of it (`sim/vizfx/nebula/nebula.ts`).
 *
 * Same house rules as `modulation.test.ts`: nothing here touches the DOM or
 * WebGPU, and every import carries an explicit `.ts` extension. The one
 * exception is `NEBULA` itself, which arrives through `./vizfx-modules.ts` —
 * the table lives in a module that imports its shaders with Vite's `?raw`, and
 * that helper is the two-hook shim which makes the *real* module loadable. See
 * its header; no logic is replaced.
 *
 * What these pin, and why each is worth pinning:
 *
 * - **The table is the single source of truth.** Five parallel views (names,
 *   classes, mask, specs, length), a live parameter store, a vector conversion,
 *   the effective-θ fold and a WGSL prelude are all generated from it. A slot
 *   that is well-formed in one view and not another is a class of bug that
 *   cannot be found by reading any single file.
 * - **The generated prelude really does describe the vector.** It is the one
 *   place the shader and the table could silently disagree, and the failure mode
 *   the generation exists to prevent — a shader compiling and reading the wrong
 *   parameter — is invisible in every other test.
 * - **Macros compose outside θ.** `effectiveTheta` is the only multiply; a slot
 *   scaled twice, or a slot scaled that should not be, is a knob that means
 *   something different from what it says.
 * - **Per-step compounding quantities are stored as rates.** This is the one
 *   bug the table's own comments record having shipped (`zoomRate` as a factor
 *   near 1.0), and the encoding is what makes a ln-space spec mean what it says.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { baseVector } from '../src/mapping/modulation.ts';
import {
  CLASS_FAST,
  CLASS_MEDIUM,
  CLASS_SLOW,
  MOD_GROUPS,
  type ModSpec,
} from '../src/mapping/modspec.ts';
import {
  applyVector,
  defaultMacros,
  defaultParams,
  eachSlot,
  effectiveTheta,
  fieldClasses,
  fieldNames,
  modulationMask,
  modulationSlots,
  paramsToVector,
  slotName,
  vectorLength,
  wgslPrelude,
  type VizSlot,
} from '../src/sim/vizfx/slots.ts';
import { MACRO_RANGE } from '../src/sim/vizfx/config.ts';
import { NEBULA } from './vizfx-modules.ts';

const V = NEBULA;
const LEN = vectorLength(V);
const NAMES = fieldNames(V);
const SLOTS = modulationSlots(V);
const MASK = modulationMask(V);
const ALL = eachSlot(V);
const K = V.speciesCount;
const PER_LAYER = V.layerSlots.length;

/** The shipped θ, as the modulator snapshots it at start-up. */
function defaultTheta(): Float64Array {
  return paramsToVector(V, defaultParams(V));
}

function indexOf(name: string): number {
  const i = NAMES.indexOf(name);
  assert.ok(i >= 0, `no slot named ${name}`);
  return i;
}

function globalSlot(key: string): VizSlot {
  const s = V.globalSlots.find((g) => g.key === key);
  assert.ok(s, `no global slot ${key}`);
  return s as VizSlot;
}

function defaultOf(slot: VizSlot, layer: number): number {
  return typeof slot.def === 'function' ? slot.def(layer) : slot.def;
}

/**
 * The three slots that are θ but are nobody's modulation target, per the table's
 * own comments: `brightness` belongs to the stem-follow lane, `exposure` and
 * `gamma` to the shared HDR folder and the auto-exposure controller downstream.
 */
const CPU_OWNED = ['brightness', 'exposure', 'gamma'];

// ── layout ───────────────────────────────────────────────────────────────────

test('the θ table is the shape nebula.ts documents: 40 slots, 34 of them modulated', () => {
  assert.equal(PER_LAYER, 5, 'five slots per layer');
  assert.equal(V.globalSlots.length, 20, 'twenty globals');
  assert.equal(K, 4, 'K = 4, and the four layers ARE the four stems');
  assert.equal(LEN, PER_LAYER * K + V.globalSlots.length);
  assert.equal(LEN, 40);
  assert.equal(
    MASK.reduce((a, b) => a + b, 0),
    34,
    'the modulated count the header states',
  );
  // Layer blocks first, then globals — the layout `thL()` is one multiply and an
  // add because of, and the layout persistence indexes by.
  assert.equal(NAMES[0], 'layer0.brightness');
  assert.equal(NAMES[PER_LAYER - 1], 'layer0.coreSize', 'a layer block is 5 wide');
  assert.equal(NAMES[PER_LAYER], 'layer1.brightness');
  assert.equal(NAMES[PER_LAYER * K], 'zoomRate', 'the globals start after the last layer');
  assert.equal(NAMES[LEN - 1], 'gamma', 'the last global closes the vector');
  // …and the per-layer art direction is index-aligned with the stems channel
  assert.deepEqual([...V.layerNames], ['bass', 'drums', 'vocals', 'other']);
  assert.equal(V.paletteHex.length, K, 'one shipped hue per layer');
});

test('the registry views are five descriptions of one slot order', () => {
  // The runtime indexes by slot in five hot places and rebuilding any of them per
  // tick would be waste — which only works if they agree, slot for slot.
  assert.equal(ALL.length, LEN);
  assert.equal(NAMES.length, LEN);
  assert.equal(SLOTS.length, LEN);
  assert.equal(MASK.length, LEN);
  assert.equal(fieldClasses(V).length, LEN);

  const classes = fieldClasses(V);
  for (let i = 0; i < LEN; i++) {
    const { slot, layer } = ALL[i] as { slot: VizSlot; layer: number };
    assert.equal(NAMES[i], slotName(slot, layer), `name at ${i}`);
    assert.equal(SLOTS[i], slot.mod, `spec at ${i} is the table's own object`);
    assert.equal(MASK[i], slot.mod ? 1 : 0, `${NAMES[i]}: mask disagrees with the spec`);
    assert.equal(classes[i], slot.cls, `${NAMES[i]}: class`);
    assert.ok(
      classes[i] === CLASS_FAST || classes[i] === CLASS_MEDIUM || classes[i] === CLASS_SLOW,
      `${NAMES[i]}: ${classes[i]} is not a slew class`,
    );
    assert.equal(layer, i < PER_LAYER * K ? Math.floor(i / PER_LAYER) : -1, `layer of ${NAMES[i]}`);
  }
  assert.equal(new Set(NAMES).size, LEN, 'a slot name is used twice');
});

test('every slot is well-formed: a usable range, a reachable default, a real step', () => {
  for (const { slot, layer } of ALL) {
    const n = slotName(slot, layer);
    assert.ok(slot.min < slot.max, `${n}: empty hard range`);
    assert.ok(slot.step > 0, `${n}: step ${slot.step}`);
    assert.ok(slot.step <= slot.max - slot.min, `${n}: step coarser than the whole range`);
    assert.ok(slot.label.length > 0, `${n}: no panel label`);
    const def = defaultOf(slot, layer);
    assert.ok(Number.isFinite(def), `${n}: default ${def}`);
    assert.ok(def >= slot.min && def <= slot.max, `${n}: default ${def} outside its own slider`);
  }
  // per-layer defaults really are per layer: the art direction is in the table
  const radius = V.layerSlots.find((s) => s.key === 'orbitRadius') as VizSlot;
  const perLayer = Array.from({ length: K }, (_, l) => defaultOf(radius, l));
  assert.equal(new Set(perLayer).size, K, 'the four layers ship at four different radii');
});

test('every ModSpec sits inside its hard bound and is wide enough to use', () => {
  for (const { slot, layer } of ALL) {
    const spec = slot.mod;
    if (!spec) continue;
    const n = slotName(slot, layer);
    assert.ok((MOD_GROUPS as readonly string[]).includes(spec.group), `${n}: group ${spec.group}`);
    assert.ok(spec.lo < spec.hi, `${n}: lo ${spec.lo} >= hi ${spec.hi}`);
    assert.ok(spec.half > 0, `${n}: half ${spec.half} — a modulated slot that cannot move`);
    assert.ok(spec.jitter > 0, `${n}: jitter ${spec.jitter} — a slot with no personality`);
    // "lo/hi may be narrower than the slot's hard θ bound" (mapping/modspec.ts),
    // and nebula.ts claims every one of its slots takes that option.
    assert.ok(spec.lo >= slot.min, `${n}: mod lo ${spec.lo} below the hard min ${slot.min}`);
    assert.ok(spec.hi <= slot.max, `${n}: mod hi ${spec.hi} above the hard max ${slot.max}`);
    assert.ok(
      spec.hi - spec.lo < slot.max - slot.min,
      `${n}: the mod range is not narrower than the hard range`,
    );
    // The shipped default has to be a legal *base*, or the seeded draw starts by
    // reflecting off a bound it should never have been near.
    const def = defaultOf(slot, layer);
    assert.ok(def >= spec.lo && def <= spec.hi, `${n}: default ${def} outside [${spec.lo}, ${spec.hi}]`);
    if (spec.mult) {
      // exp() scaling of a non-positive base is a no-op at 0 and a sign flip below
      // it; `baseVector` silently drops to the additive path when either is <= 0.
      assert.ok(spec.lo > 0, `${n}: a multiplicative slot with lo ${spec.lo}`);
      assert.ok(def > 0, `${n}: a multiplicative slot with default ${def}`);
    }
    // The full two-sided excursion has to fit in the range, or the tanh band is
    // clipped for a centred base and `half` is not the number it claims to be.
    const span = spec.mult ? Math.log(spec.hi / spec.lo) : spec.hi - spec.lo;
    assert.ok(span >= 2 * spec.half - 1e-12, `${n}: range ${span} narrower than the band ${2 * spec.half}`);
    assert.ok(span >= 2 * spec.jitter - 1e-12, `${n}: range ${span} narrower than the jitter draw`);
  }
});

// ── exclusions ───────────────────────────────────────────────────────────────

test('the exclusions are exactly the three the table documents', () => {
  const excluded = NAMES.filter((_, i) => SLOTS[i] === null);
  // brightness x 4 layers (the stem-follow lane owns light), plus the HDR pair.
  assert.equal(excluded.length, K + 2);
  for (const n of excluded) {
    const key = n.startsWith('layer') ? (n.split('.')[1] as string) : n;
    assert.ok(CPU_OWNED.includes(key), `unexpected exclusion ${n}`);
  }
  // Two separate lookups in the runtime — a null spec and a zero mask bit — so
  // both are asserted, exactly as physarum's and plife's registries are.
  for (let i = 0; i < LEN; i++) {
    const { slot } = ALL[i] as { slot: VizSlot };
    if (!CPU_OWNED.includes(slot.key)) continue;
    assert.equal(SLOTS[i], null, `${NAMES[i]} still has a ModSpec`);
    assert.equal(MASK[i], 0, `${NAMES[i]} is still in the modulation mask`);
  }
});

test('nothing is modulated faster than the substrate can relax', () => {
  // The rule both older registries follow and nebula.ts restates on `fade`: the
  // field's memory is 1/fade steps — 1.7 s at the shipped default — so a slot
  // swept on the fast lane would be asking for motion the image cannot show.
  // FAST survives in the table only on `brightness`, which is not a modulation
  // target at all: it is the stem-follow lane's, and that lane wants a phrase's
  // worth of attack.
  const classes = fieldClasses(V);
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 0) continue;
    assert.notEqual(classes[i], CLASS_FAST, `${NAMES[i]} is modulated on the fast lane`);
  }
  for (const slot of V.layerSlots) {
    if (slot.cls === CLASS_FAST) assert.equal(slot.mod, null, `${slot.key} is fast and modulated`);
  }
});

test('the stamp-size floor that licenses the angular ranges is still in the shader', () => {
  // nebula.ts widened the angular slots on the strength of one claim: "that
  // constraint is now *enforced* rather than merely respected: draw.wgsl floors
  // every stamp's size at 1.2x the displacement it is about to undergo". Delete
  // the floor and the mod ranges below become wrong without anything in the table
  // changing — which is exactly the kind of cross-file coupling worth a test.
  const m = /max\([^;]*?,\s*([\d.]+) \* angRate \* radius\)/.exec(V.drawWgsl) as string[] | null;
  assert.ok(m, 'draw.wgsl no longer floors the stamp size by its own displacement');
  const floor = Number((m as string[])[1]);
  assert.ok(floor >= 1, `the floor is ${floor}x the displacement — below 1 it guarantees nothing`);

  // …and the floor is load-bearing rather than decorative: at the widest legal
  // orbit with the strongest legal swirl and the smallest legal core, one step's
  // displacement is several core-widths, so without it the arm is a dotted line.
  const radius = (V.layerSlots.find((s) => s.key === 'orbitRadius') as VizSlot).mod as ModSpec;
  const core = (V.layerSlots.find((s) => s.key === 'coreSize') as VizSlot).mod as ModSpec;
  const rotate = (globalSlot('rotate').mod as ModSpec).hi;
  const swirl = (globalSlot('swirl').mod as ModSpec).hi;
  const falloff = (globalSlot('swirlFalloff').mod as ModSpec).lo;
  const disp = (rotate + swirl * Math.exp(-radius.hi * radius.hi * falloff)) * radius.hi;
  assert.ok(disp > core.lo * 2, `worst-case displacement ${disp} vs smallest core ${core.lo}`);
});

test('a global with no folder is one of the two the shared HDR folder binds', () => {
  // `VizSlot.folder` is globals-only and omitting it means *no slider is built*.
  // A modulated global with no folder would be a knob nothing can reach.
  for (const slot of V.globalSlots) {
    if (slot.folder !== undefined) {
      assert.ok(slot.folder.length > 0, `${slot.key}: empty folder name`);
      continue;
    }
    assert.ok(
      slot.key === 'exposure' || slot.key === 'gamma',
      `${slot.key} builds no slider and is not one of the HDR pair`,
    );
    assert.equal(slot.mod, null, `${slot.key} has no slider and is still modulated`);
  }
  for (const slot of V.layerSlots) {
    assert.equal(slot.folder, undefined, `${slot.key}: a per-layer slot lands in its layer's folder`);
  }
});

// ── macros ───────────────────────────────────────────────────────────────────

test('every macro is declared, claimed, neutral at 1, and legal on the slot it scales', () => {
  const keys = V.macros.map((m) => m.key);
  assert.equal(new Set(keys).size, keys.length, 'a macro is declared twice');
  for (const m of V.macros) {
    assert.ok(m.min <= 1 && 1 <= m.max, `${m.key}: neutral is outside its own slider`);
    assert.ok(m.min < m.max, `${m.key}: empty range`);
    assert.match(m.label, /×/, 'a macro label states what it multiplies');
    assert.ok(
      m.min >= MACRO_RANGE.min && m.max <= MACRO_RANGE.max,
      `${m.key}: outside the loader's clamp, so a saved value could not round-trip`,
    );
  }
  assert.deepEqual(defaultMacros(V), Object.fromEntries(keys.map((k) => [k, 1])));

  const claimed = new Set<string>();
  for (const { slot, layer } of ALL) {
    if (slot.macro === undefined) continue;
    const n = slotName(slot, layer);
    assert.ok(keys.includes(slot.macro), `${n}: macro '${slot.macro}' is not declared`);
    claimed.add(slot.macro);
    // "only slots whose neutral value is 0 (additive) or whose scale is genuinely
    // proportional (a gain, a radius) may carry one" — VizSlot.macro.
    assert.ok(
      slot.mod?.mult === true || (slot.min <= 0 && 0 <= slot.max),
      `${n}: macro on a slot that is neither proportional nor neutral at 0`,
    );
  }
  for (const k of keys) assert.ok(claimed.has(k), `macro '${k}' scales nothing`);
});

test('each macro scales exactly the slots its design says it does', () => {
  // The wiring rather than the label, because the wiring is what the performer
  // feels. Pinned as a set so that adding a slot to a macro is a deliberate edit
  // here as well as there — a macro that quietly grows a fifth slot is a knob
  // that stops meaning what its label says.
  //
  // NOTE: `motion`'s shipped label reads "(× rotate, swirl, drift, orbit speed)"
  // and omits `ripple` and `ringExpand`, both of which it does scale. The wiring
  // below is the truth; the label is the thing that is behind.
  const wiring: Record<string, string[]> = {
    light: ['emitGain', 'halo', 'ringGain'],
    motion: ['orbitSpeed', 'rotate', 'swirl', 'driftX', 'driftY', 'ripple', 'ringExpand'],
    scale: ['orbitRadius', 'coreSize'],
    // `layerBlend` joined `chroma` with the overlap fix: the macro is "how much
    // colour identity is in play", and how hard two crossing layers keep their
    // own hue is that question as much as how far the field's hue travels.
    chroma: ['chromaShift', 'layerBlend'],
  };
  const actual: Record<string, string[]> = {};
  for (const slot of [...V.layerSlots, ...V.globalSlots]) {
    if (slot.macro === undefined) continue;
    (actual[slot.macro] ??= []).push(slot.key);
  }
  for (const key of Object.keys(wiring)) {
    assert.deepEqual((actual[key] ?? []).sort(), (wiring[key] as string[]).sort(), `macro ${key}`);
  }
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(wiring).sort());
});

test('the two slots documented as macro-free stay macro-free', () => {
  // Both are reasoned decisions in nebula.ts, and both would be silently undone
  // by someone adding `macro:` to look consistent.
  //   zoomRate — `motion` on it would mean "more motion" also pushed the image
  //   off the frame edges, which is a different trade from liveliness.
  //   noiseScale — a bigger number makes filaments *finer*, so `scale` would have
  //   two opposite meanings depending on which slider you watched.
  assert.equal(globalSlot('zoomRate').macro, undefined);
  assert.equal(globalSlot('noiseScale').macro, undefined);
});

// ── params ⇄ vector ──────────────────────────────────────────────────────────

test('paramsToVector → applyVector reproduces the store exactly', () => {
  const params = defaultParams(V);
  assert.equal(Object.keys(params).length, LEN, 'one store key per slot, no more');
  const v = paramsToVector(V, params);
  const round = { ...params };
  for (const k of Object.keys(round)) round[k] = -999;
  applyVector(V, round, v);
  assert.deepEqual(round, params);

  // …and an edited store survives the same round trip, since that is the path a
  // rebase, an explorer candidate and a saved file all take.
  params['swirl'] = 0.123;
  params['layer2.coreSize'] = 0.031;
  const edited = paramsToVector(V, params);
  const back: Record<string, number> = {};
  applyVector(V, back, edited);
  assert.deepEqual(back, params);
});

test('applyVector clamps to the hard bound and never propagates garbage', () => {
  const defaults = defaultTheta();

  const high: Record<string, number> = {};
  applyVector(V, high, new Float64Array(LEN).fill(1e9));
  const low: Record<string, number> = {};
  applyVector(V, low, new Float64Array(LEN).fill(-1e9));
  for (const { slot, layer } of ALL) {
    const n = slotName(slot, layer);
    assert.equal(high[n], slot.max, `${n}: not clamped up`);
    assert.equal(low[n], slot.min, `${n}: not clamped down`);
  }

  // NaN / Infinity / a short vector all fall back to the shipped default rather
  // than writing a value no slider could show — this is the only door into the
  // store, so a poisoned θ here is a poisoned uniform buffer.
  const junk: Record<string, number> = {};
  applyVector(V, junk, [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]);
  for (let i = 0; i < LEN; i++) {
    assert.equal(junk[NAMES[i] as string], defaults[i], `${NAMES[i]} took a junk value`);
  }
  // …and the same on the way out: a store with holes and hostile types reads as
  // the defaults, not as NaN.
  const hostile = { swirl: 'loud', 'layer0.coreSize': Number.NaN } as unknown as Record<string, number>;
  const v = paramsToVector(V, hostile);
  for (let i = 0; i < LEN; i++) assert.equal(v[i], defaults[i], `${NAMES[i]}`);
});

test('a masked applyVector leaves the excluded slots exactly where they were', () => {
  const params = defaultParams(V);
  const before = { ...params };
  applyVector(V, params, new Float64Array(LEN).fill(0.5), MASK);
  for (let i = 0; i < LEN; i++) {
    const n = NAMES[i] as string;
    const slot = (ALL[i] as { slot: VizSlot }).slot;
    if (MASK[i] === 1) {
      const want = 0.5 < slot.min ? slot.min : 0.5 > slot.max ? slot.max : 0.5;
      assert.equal(params[n], want, n);
    } else {
      assert.equal(params[n], before[n], `${n} was written through the mask`);
    }
  }
});

// ── the generated WGSL prelude ───────────────────────────────────────────────

/** `const NAME: u32 = 12u;` → { NAME: 12 }. */
function preludeConstants(text: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of text.matchAll(/^const (TH_[A-Za-z0-9_]+): u32 = (\d+)u;$/gm)) {
    out[m[1] as string] = Number(m[2]);
  }
  return out;
}

test('every TH_ constant in the prelude is the slot\'s own position in the vector', () => {
  const consts = preludeConstants(wgslPrelude(V));
  assert.equal(consts['TH_PER_LAYER'], PER_LAYER);
  assert.equal(consts['TH_COUNT'], LEN);

  // Per-layer constants are *offsets within a layer block*; `thL(l, off)` adds
  // the block base. Checked against the registry for every layer, not just 0,
  // because the offset form is only correct if the blocks really are contiguous.
  for (const slot of V.layerSlots) {
    const off = consts[`TH_L_${slot.key}`];
    assert.ok(off !== undefined, `TH_L_${slot.key} is missing from the prelude`);
    for (let l = 0; l < K; l++) {
      assert.equal(
        l * (consts['TH_PER_LAYER'] as number) + (off as number),
        indexOf(`layer${l}.${slot.key}`),
        `thL(${l}, TH_L_${slot.key}) addresses the wrong slot`,
      );
    }
  }
  // Globals are absolute indices.
  for (const slot of V.globalSlots) {
    assert.equal(consts[`TH_${slot.key}`], indexOf(slot.key), `TH_${slot.key}`);
  }
  // Exactly one constant per slot plus the two counts — a stale constant left
  // behind by a rename is as dangerous as a missing one.
  assert.equal(Object.keys(consts).length, PER_LAYER + V.globalSlots.length + 2);

  const prelude = wgslPrelude(V);
  assert.match(prelude, /fn th\(i: u32\) -> f32/);
  assert.match(prelude, /fn thL\(l: u32, off: u32\) -> f32/);
  assert.match(prelude, /GENERATED/, 'the generated text says so, for whoever finds it in a dump');
});

test('the shaders reference only constants the prelude defines', () => {
  // The failure this exists for: a renamed slot leaves a shader reading a name
  // nothing declares. In the browser that is a WGSL compile error, which is loud
  // — but only once someone runs it, and only for the visual that broke.
  const consts = preludeConstants(wgslPrelude(V));
  const used = new Set<string>();
  for (const src of [V.warpWgsl, V.drawWgsl]) {
    for (const m of src.matchAll(/\bTH_[A-Za-z0-9_]+\b/g)) used.add(m[0]);
  }
  assert.ok(used.size > 10, `only ${used.size} TH_ references found — is the shader text loading?`);
  for (const name of used) {
    assert.ok(consts[name] !== undefined, `${name} is used by a shader and defined nowhere`);
  }
});

test('every θ slot is read by a shader, or is one of the three the CPU owns', () => {
  // A slot in the table that nothing consumes is a slider that does nothing, a
  // saved number that means nothing, and a modulation target that spends depth
  // on nothing. The three exceptions are read on the CPU: brightness feeds the
  // per-layer uniform, exposure and gamma feed the post chain.
  const src = `${V.warpWgsl}\n${V.drawWgsl}`;
  for (const slot of V.layerSlots) {
    if (CPU_OWNED.includes(slot.key)) continue;
    assert.ok(src.includes(`TH_L_${slot.key}`), `layer slot ${slot.key} is read by neither pass`);
  }
  for (const slot of V.globalSlots) {
    if (CPU_OWNED.includes(slot.key)) continue;
    assert.ok(src.includes(`TH_${slot.key}`), `global slot ${slot.key} is read by neither pass`);
  }
});

// ── macro folding ────────────────────────────────────────────────────────────

test('effectiveTheta with every macro at 1 is the identity on θ', () => {
  const params = defaultParams(V);
  const out = effectiveTheta(V, params, defaultMacros(V), new Float32Array(LEN));
  for (let i = 0; i < LEN; i++) {
    assert.equal(out[i], Math.fround(params[NAMES[i] as string] as number), NAMES[i]);
  }
  // A macro rig with missing or hostile entries is also neutral, rather than
  // NaN — the rig arrives from the opaque extras block.
  const junk = { light: Number.NaN, motion: 'x', scale: undefined } as unknown as Record<string, number>;
  const out2 = effectiveTheta(V, params, junk, new Float32Array(LEN));
  for (let i = 0; i < LEN; i++) assert.equal(out2[i], out[i], NAMES[i]);
});

test('a macro at 2 scales exactly the slots that declare it, and nothing else', () => {
  const params = defaultParams(V);
  const neutral = effectiveTheta(V, params, defaultMacros(V), new Float32Array(LEN));
  for (const m of V.macros) {
    const macros = { ...defaultMacros(V), [m.key]: 2 };
    const out = effectiveTheta(V, params, macros, new Float32Array(LEN));
    let scaled = 0;
    for (let i = 0; i < LEN; i++) {
      const { slot, layer } = ALL[i] as { slot: VizSlot; layer: number };
      const n = slotName(slot, layer);
      if (slot.macro === m.key) {
        const want = Math.min((params[n] as number) * 2, slot.max);
        assert.equal(out[i], Math.fround(want), `${n} under macro ${m.key}`);
        scaled++;
      } else {
        assert.equal(out[i], neutral[i], `${n} moved when macro ${m.key} did`);
      }
    }
    assert.ok(scaled > 0, `macro ${m.key} scaled nothing`);
  }
  // The one composition rule worth stating: two macros never touch one slot, so
  // "the macro multiply happens here and nowhere else" is a single multiply.
  const both = effectiveTheta(V, params, { ...defaultMacros(V), light: 2, motion: 2 }, new Float32Array(LEN));
  for (let i = 0; i < LEN; i++) {
    const { slot } = ALL[i] as { slot: VizSlot };
    if (slot.macro === 'light' || slot.macro === 'motion') continue;
    assert.equal(both[i], neutral[i], NAMES[i]);
  }
});

test('a macro saturates at the hard bound rather than handing the GPU an impossible θ', () => {
  const params = defaultParams(V);
  // Park every macro'd slot on its own ceiling, then double it.
  for (const { slot, layer } of ALL) {
    if (slot.macro !== undefined) params[slotName(slot, layer)] = slot.max;
  }
  const out = effectiveTheta(V, params, { light: 2, motion: 2, scale: 2, chroma: 2 }, new Float32Array(LEN));
  for (let i = 0; i < LEN; i++) {
    const { slot, layer } = ALL[i] as { slot: VizSlot; layer: number };
    if (slot.macro === undefined) continue;
    assert.equal(out[i], Math.fround(slot.max), `${slotName(slot, layer)} escaped its hard max`);
  }
  // …and a macro at 0 is a floor, not a sign flip, on the slots that go negative
  const zeroed = effectiveTheta(V, defaultParams(V), { light: 0, motion: 0, scale: 0, chroma: 0 }, new Float32Array(LEN));
  for (let i = 0; i < LEN; i++) {
    const { slot } = ALL[i] as { slot: VizSlot };
    if (slot.macro === undefined) continue;
    assert.equal(zeroed[i], Math.fround(Math.max(0, slot.min)), NAMES[i]);
  }
});

// ── the per-step rate trap ───────────────────────────────────────────────────

test('no slot encodes a per-step compounding quantity as a factor near 1', () => {
  // The shipped bug, from `zoomRate`'s own comment: the slot held the zoom factor
  // (1.0035) with a multiplicative spec, so a "±0.8%" ln-space draw on the stored
  // number was a 3.3x draw on the growth rate — because the growth rate is x - 1
  // and almost all of the stored number is the 1. `fade` is the same fix: the
  // loss rate, not `decay = 1 - loss`.
  //
  // The signature both bugs share, and the one this asserts: a multiplicative
  // slot whose base sits on top of 1.0 with a small ln-space excursion. A genuine
  // proportional quantity (a gain, a radius) does not look like that — its range
  // spans an order of magnitude and its half is a real fraction of it.
  for (const { slot, layer } of ALL) {
    const spec = slot.mod;
    if (!spec?.mult) continue;
    const n = slotName(slot, layer);
    const def = defaultOf(slot, layer);
    assert.ok(
      !(Math.abs(def - 1) <= 0.02 && spec.half < 0.05),
      `${n}: default ${def} with a ln half of ${spec.half} is a factor-near-1 encoding`,
    );
    // Same statement from the other side: a ln-space spec is only meaningful if
    // the quantity is proportional, i.e. halving the number halves the effect.
    // A range that cannot reach half its own base fails that.
    assert.ok(spec.hi / spec.lo >= 1.5, `${n}: ln-space spec on a range of only ${spec.hi / spec.lo}x`);
  }
});

test('zoomRate is a rate: additive, signed, and small', () => {
  const slot = globalSlot('zoomRate');
  const spec = slot.mod as ModSpec;
  assert.equal(spec.mult, false, 'a rate takes an additive spec — that is the whole fix');
  assert.ok(slot.min < 0 && slot.max > 0, 'the hard range straddles 0: a factor encoding could not');
  assert.ok(spec.lo < 0 && spec.hi > 0, 'and so does the mod range — inward flow is a legal visual');
  assert.ok(Math.abs(defaultOf(slot, -1)) < 0.05, 'the stored number is the growth, not 1 + growth');
  // The shader's own reading, pinned so the encoding cannot drift apart from it.
  assert.match(V.warpWgsl, /1\.0 \+ th\(TH_zoomRate\)/);

  // The arithmetic the comment argues from: over the field's memory (1/fade
  // steps) the whole mod range stays inside a legible expansion, where the
  // shipped bug reached 3.3x.
  const fadeDef = defaultOf(globalSlot('fade'), -1);
  const memory = 1 / fadeDef;
  assert.ok(Math.abs(memory - 100) < 1, `memory is ${memory} steps, not the documented 100`);
  assert.ok(
    (1 + spec.hi) ** memory < 4,
    `the top of the mod range expands ${(1 + spec.hi) ** memory}x over one memory`,
  );
  assert.ok((1 + defaultOf(slot, -1)) ** memory < 1.6, 'the shipped default is the documented ~1.4x');
});

test('fade is a loss rate: multiplicative on the loss, and the memory band is legible', () => {
  const slot = globalSlot('fade');
  const spec = slot.mod as ModSpec;
  assert.ok(spec.mult, 'doubling the loss should halve the memory — that is what x2 must mean');
  assert.ok(slot.min > 0 && slot.max < 1, 'a loss rate lives strictly inside (0, 1)');
  assert.ok(spec.lo > 0 && spec.hi < 1);
  assert.ok(defaultOf(slot, -1) < 0.5, 'the stored number is the loss, not `decay` near 1');
  assert.match(V.warpWgsl, /1\.0 - th\(TH_fade\)/, 'the shader reads it as a loss');
  // half 0.25 in ln space is memory x0.78 .. x1.28, per the slot's comment.
  assert.ok(Math.abs(Math.exp(-spec.half) - 0.78) < 0.01, `memory floor x${Math.exp(-spec.half)}`);
  assert.ok(Math.abs(Math.exp(spec.half) - 1.28) < 0.01, `memory ceiling x${Math.exp(spec.half)}`);
  // The whole mod range still leaves the substrate a memory long enough to warp
  // anything into an arm: the arc is (rotate + swirl) x memory.
  assert.ok(1 / spec.hi >= 10, `the fastest legal fade leaves only ${1 / spec.hi} steps of memory`);
});

// ── the seeded personality ───────────────────────────────────────────────────

test('the same seed reproduces an identical personality; a different one does not', () => {
  const defaults = defaultTheta();
  const a = baseVector(0x1234_5678, defaults, SLOTS, new Float64Array(LEN));
  const again = baseVector(0x1234_5678, defaults, SLOTS, new Float64Array(LEN));
  const other = baseVector(0x8765_4321, defaults, SLOTS, new Float64Array(LEN));
  for (let i = 0; i < LEN; i++) assert.equal(a[i], again[i], NAMES[i]);

  let differing = 0;
  for (let i = 0; i < LEN; i++) {
    if (MASK[i] === 0) {
      assert.equal(a[i], defaults[i], `${NAMES[i]} was jittered despite being excluded`);
      continue;
    }
    if (Math.abs((a[i] as number) - (other[i] as number)) > 1e-12) differing++;
  }
  const modulated = MASK.reduce((x, y) => x + y, 0);
  assert.ok(
    differing > modulated * 0.9,
    `only ${differing} of ${modulated} modulatable slots differ between seeds`,
  );
});

test('every seeded base lands inside its own modulation range', () => {
  const defaults = defaultTheta();
  const out = new Float64Array(LEN);
  for (let seed = 0; seed < 400; seed++) {
    baseVector(seed * 2654435761, defaults, SLOTS, out);
    for (let i = 0; i < LEN; i++) {
      const spec = SLOTS[i];
      if (!spec) continue;
      const x = out[i] as number;
      assert.ok(
        Number.isFinite(x) && x >= spec.lo - 1e-12 && x <= spec.hi + 1e-12,
        `${NAMES[i]} = ${x} outside [${spec.lo}, ${spec.hi}] at seed ${seed}`,
      );
    }
  }
});

test('no seed ships a swirl with no arms in it', () => {
  // From the slot's comment: an additive draw of ±0.06 on a 0.09 base can land
  // near zero or flip sign, and a seed whose swirl came out at 0.02 has no arms
  // at all. The fix was `jitter` well under `half` — the music may still swing it
  // that far, the shipped personality may not start there.
  const i = indexOf('swirl');
  const spec = SLOTS[i] as ModSpec;
  assert.ok(spec.jitter < spec.half, `jitter ${spec.jitter} is not under half ${spec.half}`);
  const defaults = defaultTheta();
  const out = new Float64Array(LEN);
  for (let seed = 1; seed <= 512; seed++) {
    baseVector(seed, defaults, SLOTS, out);
    assert.ok((out[i] as number) > 0.05, `seed ${seed} ships swirl ${out[i]}`);
  }
});
