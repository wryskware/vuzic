/**
 * Headless tests for explorer mode's (1+8) evolution strategy (`explore/search.ts`).
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads the modules
 * straight out of src/, which is why every import carries an explicit `.ts`
 * extension, and nothing here touches the DOM or WebGPU.
 *
 * These are written against the *contract*, not against the code: the expected
 * displacement arithmetic below is re-derived from the spec ("σ·half in each
 * slot's own space, reflected into [lo, hi]") rather than imported, so a change
 * to the search that quietly changes the arithmetic fails here instead of
 * agreeing with itself.
 *
 * What they pin, in the order the rig depends on them:
 *
 * - **The frozen half of the frozen/free split is exactly frozen.** Not "close
 *   to" — bit-identical, including plife's `Rmin` (a slot that is *in* the mask
 *   and carries a real `ModSpec`, and is nevertheless immobile because
 *   `half: 0`), and including across the momentum path, which is a second,
 *   separate write of every free slot.
 * - **Reflection, not clamping.** A default sitting on a bound must still
 *   produce eight different tiles.
 * - **Determinism**, because the verdict log stores `genSeed` instead of the
 *   eight losing θs and is worthless if those cannot be regrown.
 * - **Momentum is a repeat**, exactly, and dies exactly when it is supposed to.
 * - **Nothing internal is aliased out**, because the 9-up rig hands these arrays
 *   to a renderer.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  CANDIDATE_COUNT,
  DEFAULT_STEP,
  EXPLORER_SUBSPACES,
  ExplorerSearch,
  MAX_STEP,
  MIN_STEP,
  type CandidateSet,
  type ExplorerSubspace,
} from '../src/explore/search.ts';
import { reflect } from '../src/mapping/modulation.ts';
import { MOD_GROUPS, type ModGroup, type ModSpec } from '../src/mapping/modspec.ts';
import type { ThetaRegistry } from '../src/mapping/target.ts';

import {
  fieldClasses as phyClasses,
  fieldNames as phyNames,
  modulationMask as phyMask,
  modulationSlots as phySlots,
  presetFromConfig as phyPresetFromConfig,
  presetToVector as phyPresetToVector,
  vectorLength as phyLength,
} from '../src/mapping/preset.ts';
import { defaultConfig } from '../src/sim/physarum/config.ts';

import {
  fieldClasses as plClasses,
  fieldNames as plNames,
  minRBase,
  modulationMask as plMask,
  modulationSlots as plSlots,
  presetFromConfig as plPresetFromConfig,
  presetToVector as plPresetToVector,
  vectorLength as plLength,
} from '../src/sim/plife/preset.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';

// ── fixtures ─────────────────────────────────────────────────────────────────

/**
 * A registry whose slots are whatever the test wants.
 *
 * The real registries are ~100–270 slots of authored bounds; a synthetic one is
 * how a property gets stated over an *exactly* known slot table — "slot 1 is the
 * only free one", "slot 3 has half 0", "slot 4's default sits on its own lower
 * bound" — none of which any shipped registry offers on demand.
 *
 * `mask` defaults to the slot table's shadow (the invariant every real registry
 * holds), and is overridable so the mask-vs-spec halves of `isFree` can be
 * tested apart: a slot may have a spec and a 0 mask bit, which must freeze it.
 */
function syntheticRegistry(slots: (ModSpec | null)[], mask?: number[]): ThetaRegistry {
  return {
    length: slots.length,
    slots,
    mask: Uint8Array.from(slots.map((s, i) => (mask ? (mask[i] as number) : s ? 1 : 0))),
    classes: new Uint8Array(slots.length),
    names: slots.map((_, i) => `slot${i}`),
  };
}

/** A ModSpec with sane defaults; every field is overridable. */
function spec(over: Partial<ModSpec> = {}): ModSpec {
  return { group: 'structure', lo: 0, hi: 1, half: 0.1, jitter: 0.1, mult: false, ...over };
}

function physarumRegistry(k: number): ThetaRegistry {
  return {
    length: phyLength(k),
    slots: phySlots(k),
    mask: phyMask(k),
    classes: phyClasses(k),
    names: phyNames(k),
  };
}

function physarumTheta(k: number): Float64Array {
  return phyPresetToVector(phyPresetFromConfig(defaultConfig(k)), k);
}

function plifeRegistry(k: number): ThetaRegistry {
  return {
    length: plLength(k),
    slots: plSlots(k),
    mask: plMask(k),
    classes: plClasses(k),
    names: plNames(k),
  };
}

function plifeTheta(k: number): Float64Array {
  return plPresetToVector(plPresetFromConfig(defaultPlifeConfig(k)), k);
}

// ── the contract's arithmetic, restated ──────────────────────────────────────

/** Spec: a ratio only means something about a strictly positive quantity. */
function logScaled(s: ModSpec): boolean {
  return s.mult && s.lo > 0 && s.hi > s.lo;
}

/**
 * Spec: move `v` by `delta` in the slot's own space and land inside [lo, hi] by
 * *reflection*; a zero displacement returns `v` untouched, bit for bit.
 */
function displaceBySpec(v: number, delta: number, s: ModSpec): number {
  if (delta === 0) return v;
  if (logScaled(s)) return Math.exp(reflect(Math.log(v) + delta, Math.log(s.lo), Math.log(s.hi)));
  return reflect(v + delta, s.lo, s.hi);
}

/** Spec: the displacement that turned `from` into `to`, in the slot's own space. */
function displacementBySpec(from: number, to: number, s: ModSpec): number {
  return logScaled(s) ? Math.log(to) - Math.log(from) : to - from;
}

/** Spec: slot i may move only if the mask, the spec and the subspace all allow it. */
function freeSpec(r: ThetaRegistry, i: number, sub: ExplorerSubspace): ModSpec | null {
  if (r.mask[i] !== 1) return null;
  const s = r.slots[i] ?? null;
  if (!s) return null;
  if (sub !== 'all' && s.group !== sub) return null;
  return s;
}

/** Every slot of every candidate that the subspace says must not have moved. */
function assertFrozenOutside(r: ThetaRegistry, set: CandidateSet, sub: ExplorerSubspace, where: string): void {
  for (const [k, cand] of set.candidates.entries()) {
    assert.equal(cand.length, r.length, `${where}: candidate ${k} is the wrong length`);
    for (let i = 0; i < r.length; i++) {
      if (freeSpec(r, i, sub)) continue;
      assert.equal(cand[i], set.center[i], `${where}: ${r.names[i]} moved in candidate ${k}`);
    }
  }
}

/** Every candidate value of a moving slot is finite and inside its own bounds. */
function assertInBounds(r: ThetaRegistry, set: CandidateSet, where: string): void {
  for (const [k, cand] of set.candidates.entries()) {
    for (let i = 0; i < r.length; i++) {
      const s = r.slots[i];
      if (!s) continue;
      const x = cand[i] as number;
      assert.ok(
        Number.isFinite(x) && x >= s.lo && x <= s.hi,
        `${where}: ${r.names[i]} = ${x} outside [${s.lo}, ${s.hi}] in candidate ${k}`,
      );
    }
  }
}

/**
 * A θ a slider drag could plausibly have produced: every slot that has a spec is
 * moved to a deterministic point strictly inside its own `[lo, hi]`, so a
 * recentred grid can be bounds-checked like any other. Slots with no spec keep
 * whatever `base` had — the panel cannot move those either.
 */
function nudged(r: ThetaRegistry, base: Float64Array, t: number): Float64Array {
  const out = Float64Array.from(base);
  for (let i = 0; i < r.length; i++) {
    const s = r.slots[i];
    if (!s) continue;
    const mix = 0.1 + 0.8 * ((((i * 0.618_033_988_75) % 1) + t) % 1);
    out[i] = s.lo + (s.hi - s.lo) * mix;
  }
  return out;
}

type Act = (s: ExplorerSearch) => CandidateSet | null;

/**
 * A fixed, deliberately varied script of explorer acts, including the two the
 * outside world performs. Interleaving is the point: determinism has to survive
 * the *order*, not just the count, and `recenter` is a centre move that arrives
 * from a different direction than every other one.
 *
 * Returned as a list of closures rather than run inline so that the determinism
 * test and the ownership test drive the exact same sequence — a script that
 * drifted between the two would make the ownership test quietly vacuous.
 */
function scriptFor(r: ThetaRegistry, theta: Float64Array): Act[] {
  const alt1 = nudged(r, theta, 0.13);
  const alt2 = nudged(r, theta, 0.61);
  return [
    (s) => s.start(theta),
    (s) => s.pick(3),
    (s) => s.reroll(),
    (s) => {
      s.setStep(0.7);
      return s.pick(0);
    },
    (s) => s.pick(6),
    (s) => s.recenter(alt1),
    (s) => {
      s.setSubspace('structure');
      return s.reroll();
    },
    (s) => s.pick(1),
    (s) => s.back(),
    (s) => {
      s.setSubspace('all');
      return s.pick(7);
    },
    (s) => s.recenter(alt2),
    (s) => {
      s.setStep(0.12);
      return s.reroll();
    },
    (s) => s.pick(2),
    (s) => s.back(),
  ];
}

function runScript(s: ExplorerSearch, acts: Act[]): CandidateSet[] {
  return acts.map((act, i) => {
    const set = act(s);
    if (!set) throw new Error(`script step ${i} returned null — the script is stale`);
    return set;
  });
}

function sameSet(a: CandidateSet, b: CandidateSet, where: string): void {
  assert.equal(a.genSeed, b.genSeed, `${where}: genSeed`);
  assert.equal(a.generation, b.generation, `${where}: generation`);
  assert.deepEqual(Array.from(a.center), Array.from(b.center), `${where}: center`);
  assert.equal(a.candidates.length, b.candidates.length, `${where}: candidate count`);
  for (const [k, c] of a.candidates.entries()) {
    assert.deepEqual(Array.from(c), Array.from(b.candidates[k] as Float64Array), `${where}: candidate ${k}`);
  }
}

// ── shape ────────────────────────────────────────────────────────────────────

test('the exported constants are the ones the 9-up rig is built around', () => {
  assert.equal(CANDIDATE_COUNT, 8, 'the 3×3 grid is a centre plus eight');
  assert.equal(MIN_STEP, 0.05);
  assert.equal(MAX_STEP, 1);
  assert.ok(DEFAULT_STEP >= MIN_STEP && DEFAULT_STEP <= MAX_STEP);
  assert.deepEqual([...EXPLORER_SUBSPACES], ['all', ...MOD_GROUPS]);
});

test('a fresh search reports the documented defaults and a set of the right shape', () => {
  const r = syntheticRegistry([spec(), spec({ group: 'matrix' })]);
  const s = new ExplorerSearch(r, 0x1234);
  assert.equal(s.step, DEFAULT_STEP);
  assert.equal(s.subspace, 'all');
  assert.equal(s.generation, 0);
  assert.equal(s.depth, 0);
  assert.equal(s.hasMomentum, false);
  assert.equal(s.runSeed, 0x1234);

  const set = s.start(Float64Array.from([0.5, 0.5]));
  assert.equal(set.generation, 0, 'start() is generation 0');
  assert.equal(set.candidates.length, CANDIDATE_COUNT);
  assert.equal(set.center.length, r.length);
  assert.deepEqual(Array.from(set.center), [0.5, 0.5], 'the centre is θ as given');
});

// ── 1. freeness ──────────────────────────────────────────────────────────────

/**
 * The slot table this whole freeness block is stated over. Both halves of the
 * mask/spec test are present, one slot per group so the subspace filter has
 * something to filter, and a `half: 0` slot in the mask — the shape plife's
 * `Rmin` really has.
 */
function freenessRegistry(): ThetaRegistry {
  const slots: (ModSpec | null)[] = [
    spec({ group: 'structure', lo: -5, hi: 5, half: 1 }), // 0 free
    null, // 1 excluded outright
    spec({ group: 'matrix', lo: -5, hi: 5, half: 1 }), // 2 free
    spec({ group: 'structure', lo: -5, hi: 5, half: 0 }), // 3 in the mask, immobile
    spec({ group: 'population', lo: -5, hi: 5, half: 1 }), // 4 free
    spec({ group: 'decay', lo: 0.1, hi: 10, half: 1, mult: true }), // 5 free, log-scaled
    spec({ group: 'structure', lo: -5, hi: 5, half: 1 }), // 6 spec but masked out
    spec({ group: 'decay', lo: 0.002, hi: 0.05, half: 0, mult: true }), // 7 Rmin's shape
  ];
  return syntheticRegistry(slots, [1, 0, 1, 1, 1, 1, 0, 1]);
}

test('only the mask ∧ spec ∧ subspace slots may move; everything else is bit-identical', () => {
  const r = freenessRegistry();
  const theta = Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]);

  for (const sub of EXPLORER_SUBSPACES) {
    const s = new ExplorerSearch(r, 0xbeef);
    s.setSubspace(sub);
    s.setStep(1); // the loudest σ there is: if anything leaks, it leaks here
    let set = s.start(theta);
    // Many generations, and a mix of acts, because the momentum path writes every
    // free slot a second time by a different route than `perturb` does.
    for (let g = 0; g < 24; g++) {
      assertFrozenOutside(r, set, sub, `subspace ${sub} gen ${g}`);
      assertInBounds(r, set, `subspace ${sub} gen ${g}`);
      // slot 3 and slot 7 are free by the mask/spec/subspace rule but immobile
      for (const immobile of [3, 7]) {
        if (!freeSpec(r, immobile, sub)) continue;
        for (const [k, c] of set.candidates.entries()) {
          assert.equal(c[immobile], set.center[immobile], `half-0 slot ${immobile} moved (gen ${g}, tile ${k})`);
        }
      }
      set = g % 4 === 3 ? s.reroll() : s.pick(g % CANDIDATE_COUNT);
    }
  }
});

test('a narrowed subspace really does move only its own group', () => {
  const r = freenessRegistry();
  const theta = Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]);
  // The mobile slot of each group, so "did the subspace do anything at all" is a
  // question with an answer rather than a vacuous pass.
  const mobileOf: Record<ModGroup, number> = { structure: 0, matrix: 2, population: 4, decay: 5 };
  for (const group of MOD_GROUPS) {
    const s = new ExplorerSearch(r, 0xc0ffee);
    s.setSubspace(group);
    s.setStep(1);
    let moved = 0;
    let set = s.start(theta);
    for (let g = 0; g < 8; g++) {
      for (const c of set.candidates) if (c[mobileOf[group]] !== set.center[mobileOf[group]]) moved++;
      set = s.pick(g % CANDIDATE_COUNT);
    }
    assert.ok(moved > 0, `subspace ${group} froze its own group's mobile slot`);
  }
});

test("plife's Rmin is in the mask, carries a spec, and still never moves", () => {
  // The slot the doc comment singles out: `mult: true, half: 0`, so it is free by
  // the mask/spec test and immobile by its half — and the log/exp round trip in
  // the multiplicative path is exactly where an ULP of drift per generation
  // would come from.
  const K = 8;
  const r = plifeRegistry(K);
  const nBase = minRBase(K);
  const theta = plifeTheta(K);
  for (let o = 0; o < K * K; o++) {
    const s = r.slots[nBase + o] as ModSpec;
    assert.ok(s, 'Rmin lost its ModSpec — the fixture is stale');
    assert.equal(s.half, 0);
    assert.equal(s.mult, true);
    assert.equal(r.mask[nBase + o], 1);
  }

  const search = new ExplorerSearch(r, 0x51de);
  search.setStep(MAX_STEP);
  let set = search.start(theta);
  for (let g = 0; g < 40; g++) {
    for (const [k, c] of set.candidates.entries()) {
      for (let o = 0; o < K * K; o++) {
        assert.equal(c[nBase + o], set.center[nBase + o], `Rmin[${o}] drifted (gen ${g}, tile ${k})`);
      }
    }
    // …and the centre itself never drifts off the value it started at
    for (let o = 0; o < K * K; o++) assert.equal(set.center[nBase + o], theta[nBase + o], `Rmin[${o}] centre`);
    set = g % 5 === 4 ? search.reroll() : search.pick(g % CANDIDATE_COUNT);
  }
});

test('an immobile log-scaled slot stays frozen even when the centre is outside its bounds', () => {
  // `ModSpec` documents `lo`/`hi` as possibly *narrower* than a slot's hard θ
  // bound, so a legal θ can hand the search a centre outside a slot's modulation
  // range. For an immobile slot the momentum path measured
  // `log(to) − log(clamp(from))` ≠ 0 for a value that never moved, and drifted it
  // once per generation; a centre at 0 measured −Infinity and produced a NaN
  // tile. Both are freeness/bounds violations, and both are momentum-only — the
  // fresh-draw path was always fine — so the fixture has to keep picking.
  for (const [label, centre, s0] of [
    ['below lo', 0.0001, spec({ mult: true, half: 0, lo: 0.002, hi: 0.05 })],
    ['above hi', 0.5, spec({ mult: true, half: 0, lo: 0.002, hi: 0.05 })],
    ['at zero', 0, spec({ mult: true, half: 0, lo: 0.1, hi: 10 })],
    ['negative', -4, spec({ mult: true, half: 0, lo: 0.1, hi: 10 })],
  ] as [string, number, ModSpec][]) {
    const r = syntheticRegistry([spec({ lo: -10, hi: 10, half: 1 }), s0]);
    const search = new ExplorerSearch(r, 0x0ff);
    search.setStep(MAX_STEP);
    let set = search.start(Float64Array.from([5, centre]));
    for (let g = 0; g < 24; g++) {
      for (const [k, c] of set.candidates.entries()) {
        assert.equal(c[1], centre, `${label}: the immobile slot moved (gen ${g}, tile ${k})`);
        assert.ok(Number.isFinite(c[0] as number), `${label}: tile ${k} went non-finite at gen ${g}`);
      }
      assert.equal(set.center[1], centre, `${label}: the centre drifted at gen ${g}`);
      set = search.pick(g % CANDIDATE_COUNT); // momentum is the path under test
    }
  }
});

test("physarum's excluded slots are frozen across a long run", () => {
  const K = 4;
  const r = physarumRegistry(K);
  const theta = physarumTheta(K);
  const excluded = [...r.names.keys()].filter((i) => !r.slots[i]);
  assert.ok(excluded.length > 0, 'the fixture expects some excluded slots');

  const s = new ExplorerSearch(r, 0xfeed);
  s.setStep(MAX_STEP);
  let set = s.start(theta);
  for (let g = 0; g < 30; g++) {
    assertFrozenOutside(r, set, 'all', `physarum gen ${g}`);
    for (const i of excluded) assert.equal(set.center[i], theta[i], `${r.names[i]} centre drifted`);
    set = g % 4 === 3 ? s.reroll() : s.pick((g * 3) % CANDIDATE_COUNT);
  }
});

// ── 2. bounds ────────────────────────────────────────────────────────────────

test('every candidate lands inside its slot bounds, on both real registries', () => {
  for (const [label, r, theta] of [
    ['physarum', physarumRegistry(4), physarumTheta(4)],
    ['plife', plifeRegistry(8), plifeTheta(8)],
  ] as [string, ThetaRegistry, Float64Array][]) {
    for (const sub of EXPLORER_SUBSPACES) {
      const s = new ExplorerSearch(r, 0xa11 ^ sub.length);
      s.setSubspace(sub);
      s.setStep(MAX_STEP); // over-driven on purpose
      let set = s.start(theta);
      for (let g = 0; g < 12; g++) {
        assertInBounds(r, set, `${label}/${sub} gen ${g}`);
        set = s.pick(g % CANDIDATE_COUNT);
      }
    }
  }
});

test('the degenerate specs behave as documented: lo===hi pins, non-positive mult is additive', () => {
  const r = syntheticRegistry([
    spec({ lo: 0.5, hi: 0.5, half: 1 }), // collapsed range
    spec({ lo: -1, hi: 1, half: 1, mult: true }), // mult with lo <= 0 → additive
    spec({ lo: 2, hi: 2, half: 1, mult: true }), // mult with hi <= lo → additive, then pinned
    spec({ lo: 0.001, hi: 1000, half: 3, mult: true }), // a genuine log-scaled slot
  ]);
  const s = new ExplorerSearch(r, 0xd06);
  s.setStep(MAX_STEP);
  let set = s.start(Float64Array.from([0.5, 0, 2, 0.001]));
  for (let g = 0; g < 30; g++) {
    for (const [k, c] of set.candidates.entries()) {
      assert.equal(c[0], 0.5, `lo===hi slot left its pin (gen ${g}, tile ${k})`);
      assert.equal(c[2], 2, `collapsed mult slot left its pin (gen ${g}, tile ${k})`);
      // A non-positive `lo` taking the ln path would be NaN, not a number in range.
      assert.ok(Number.isFinite(c[1] as number) && (c[1] as number) >= -1 && (c[1] as number) <= 1, `mult lo<=0: ${c[1]}`);
      assert.ok(
        Number.isFinite(c[3] as number) && (c[3] as number) >= 0.001 && (c[3] as number) <= 1000,
        `log-scaled: ${c[3]}`,
      );
    }
    set = s.reroll();
  }
});

test('bounds are reflected, not clamped: a default on a bound still gives eight different tiles', () => {
  // The failure this catches is the tempting one — clamp instead of reflect —
  // which for a slot whose useful value sits against a bound turns the grid into
  // "three different candidates and five identical ones".
  for (const [label, at, s0] of [
    ['lower bound', 0, spec({ lo: 0, hi: 1, half: 1 })],
    ['upper bound', 1, spec({ lo: 0, hi: 1, half: 1 })],
    ['log lower bound', 0.01, spec({ lo: 0.01, hi: 100, half: 2, mult: true })],
  ] as [string, number, ModSpec][]) {
    const r = syntheticRegistry([s0]);
    const search = new ExplorerSearch(r, 0xb0117);
    search.setStep(MAX_STEP);
    let set = search.start(Float64Array.from([at]));
    const seen: number[] = [];
    for (let g = 0; g < 60; g++) {
      for (const c of set.candidates) seen.push(c[0] as number);
      set = search.reroll();
    }
    assert.equal(seen.length, 60 * CANDIDATE_COUNT);
    const onBound = seen.filter((v) => v === s0.lo || v === s0.hi).length;
    assert.equal(onBound, 0, `${label}: ${onBound}/${seen.length} candidates piled up on a bound`);
    // …and the mass is genuinely spread, not merely off the bound by an epsilon.
    // Measured in the slot's own space, which is where the spread was authored.
    const toSpace = (v: number): number => (logScaled(s0) ? Math.log(v) : v);
    const [lo, hi] = [toSpace(s0.lo), toSpace(s0.hi)];
    const inner = seen.filter((v) => toSpace(v) > lo + (hi - lo) * 0.1 && toSpace(v) < hi - (hi - lo) * 0.1);
    assert.ok(inner.length > seen.length * 0.2, `${label}: only ${inner.length}/${seen.length} landed off the rails`);
  }
});

test('the perturbation scale is σ · half, per slot', () => {
  // A range wide enough that reflection never bites, so the spread that comes
  // out is the spread that went in. Fixed seeds, so the tolerance is a
  // measurement band and not a flake.
  const wide = (half: number): ModSpec => spec({ lo: -1e6, hi: 1e6, half });
  const r = syntheticRegistry([wide(1), wide(4)]);
  const sd = (sigma: number, slot: number): number => {
    const s = new ExplorerSearch(r, 0x5ca1e);
    s.setStep(sigma);
    let set = s.start(new Float64Array(2));
    const v: number[] = [];
    for (let g = 0; g < 200; g++) {
      for (const c of set.candidates) v.push((c[slot] as number) - (set.center[slot] as number));
      set = s.reroll();
    }
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
  };
  for (const [sigma, slot, half] of [
    [0.35, 0, 1],
    [1, 0, 1],
    [0.35, 1, 4],
    [0.05, 1, 4],
  ] as [number, number, number][]) {
    const got = sd(sigma, slot);
    const want = sigma * half;
    assert.ok(Math.abs(got / want - 1) < 0.15, `σ=${sigma} half=${half}: sd ${got.toFixed(4)} vs expected ${want}`);
  }
});

// ── 3. determinism ───────────────────────────────────────────────────────────

test('two independently constructed searches replay an interleaved script bit-for-bit', () => {
  for (const [label, r, theta] of [
    ['synthetic', freenessRegistry(), Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075])],
    ['physarum', physarumRegistry(4), physarumTheta(4)],
    ['plife', plifeRegistry(8), plifeTheta(8)],
  ] as [string, ThetaRegistry, Float64Array][]) {
    const a = runScript(new ExplorerSearch(r, 0x9e3779b9), scriptFor(r, Float64Array.from(theta)));
    const b = runScript(new ExplorerSearch(r, 0x9e3779b9), scriptFor(r, Float64Array.from(theta)));
    assert.equal(a.length, b.length);
    for (const [i, set] of a.entries()) sameSet(set, b[i] as CandidateSet, `${label} step ${i}`);
  }
});

test('a different run seed is a different neighbourhood, and genSeed is per generation', () => {
  const r = syntheticRegistry([spec({ lo: -10, hi: 10, half: 1 }), spec({ group: 'matrix', lo: -10, hi: 10, half: 1 })]);
  const theta = Float64Array.from([0, 0]);
  const a = new ExplorerSearch(r, 1).start(theta);
  const b = new ExplorerSearch(r, 2).start(theta);
  assert.notEqual(a.genSeed, b.genSeed, 'two run seeds share a genSeed');
  let differing = 0;
  for (const [k, c] of a.candidates.entries()) {
    if ((c[0] as number) !== (b.candidates[k]?.[0] as number)) differing++;
  }
  assert.equal(differing, CANDIDATE_COUNT, 'a different seed reproduced the same tiles');

  // genSeed is what the verdict log stores in place of eight θs, so a collision
  // between generations would make two different grids indistinguishable.
  const s = new ExplorerSearch(r, 0x1234);
  const seen = new Set<number>();
  let set = s.start(theta);
  for (let g = 0; g < 300; g++) {
    assert.equal(set.generation, g, 'the generation counter skipped');
    assert.equal(seen.has(set.genSeed), false, `genSeed repeated at generation ${g}`);
    seen.add(set.genSeed);
    set = g % 3 === 0 ? s.reroll() : s.pick(g % CANDIDATE_COUNT);
  }
});

test('the same generation drawn twice is the same eight tiles — a reroll is a new generation', () => {
  const r = syntheticRegistry([spec({ lo: -10, hi: 10, half: 1 })]);
  const a = new ExplorerSearch(r, 77);
  const first = a.start(Float64Array.from([0]));
  const again = a.current();
  sameSet(first, again, 'current()');
  const rerolled = a.reroll();
  assert.notEqual(rerolled.genSeed, first.genSeed, 'a reroll reused the generation key');
  assert.equal(rerolled.generation, first.generation + 1);
  assert.deepEqual(Array.from(rerolled.center), Array.from(first.center), 'a reroll moved the centre');
});

// ── 4. stream stability ──────────────────────────────────────────────────────

test('a slot being immobile does not shift the draws of the slots after it', () => {
  // Two registries identical but for slot 1's `half`. If the draw were skipped
  // for an immobile slot, slot 2 would take slot 1's gaussian in one of them and
  // its own in the other. The claim under test is that the stream position of a
  // free slot depends on the slot table, not on whether the slot can move.
  const wide = (half: number, group: ModGroup = 'structure'): ModSpec =>
    spec({ group, lo: -10, hi: 10, half });
  const still = syntheticRegistry([wide(1), wide(0), wide(1), wide(2)]);
  const moving = syntheticRegistry([wide(1), wide(3), wide(1), wide(2)]);
  const theta = new Float64Array(4);

  const a = new ExplorerSearch(still, 0x5713).start(theta);
  const b = new ExplorerSearch(moving, 0x5713).start(theta);
  for (const [k, c] of a.candidates.entries()) {
    const d = b.candidates[k] as Float64Array;
    assert.equal(c[0], d[0], `tile ${k}: slot 0 shifted`);
    assert.equal(c[2], d[2], `tile ${k}: slot 2 shifted by slot 1's mobility`);
    assert.equal(c[3], d[3], `tile ${k}: slot 3 shifted by slot 1's mobility`);
    assert.equal(c[1], theta[1], `tile ${k}: the half-0 slot moved`);
    assert.notEqual(d[1], theta[1], `tile ${k}: the mobile slot did not move`);
  }
});

// ── 5. momentum ──────────────────────────────────────────────────────────────

test('after a pick, tile 0 is the winning displacement applied again, exactly', () => {
  // A centre deliberately near a bound so reflection is live on the repeat: the
  // property is "apply the same displacement again", not "add the same delta".
  const r = syntheticRegistry([
    spec({ group: 'structure', lo: 0, hi: 1, half: 0.5 }),
    spec({ group: 'structure', lo: 0.1, hi: 10, half: 0.5, mult: true }),
    spec({ group: 'matrix', lo: -2, hi: 2, half: 0.4 }),
    spec({ group: 'structure', lo: -2, hi: 2, half: 0 }),
  ]);
  const s = new ExplorerSearch(r, 0x37);
  s.setStep(0.8);
  let prev = s.start(Float64Array.from([0.9, 5, -1.8, 0.25]));

  for (let g = 0; g < 12; g++) {
    const index = (g * 5 + 1) % CANDIDATE_COUNT;
    const winner = prev.candidates[index] as Float64Array;
    const next = s.pick(index);
    assert.equal(s.hasMomentum, true);
    assert.deepEqual(Array.from(next.center), Array.from(winner), `gen ${g}: the winner is the new centre`);

    for (let i = 0; i < r.length; i++) {
      const sp = freeSpec(r, i, 'all') as ModSpec;
      const d = displacementBySpec(prev.center[i] as number, winner[i] as number, sp);
      const want = displaceBySpec(next.center[i] as number, d, sp);
      assert.equal((next.candidates[0] as Float64Array)[i], want, `gen ${g}: tile 0 slot ${i}`);
    }
    prev = next;
  }
});

test('under momentum tiles 1..7 stay stochastic; with none, all eight are fresh draws', () => {
  const r = syntheticRegistry([spec({ lo: -10, hi: 10, half: 1 }), spec({ lo: -10, hi: 10, half: 1 })]);
  const s = new ExplorerSearch(r, 0x5a1a);
  const first = s.start(Float64Array.from([0, 0]));
  assert.equal(s.hasMomentum, false);
  const distinct = (set: CandidateSet): number => new Set(set.candidates.map((c) => `${c[0]},${c[1]}`)).size;
  assert.equal(distinct(first), CANDIDATE_COUNT, 'a momentum-free grid has eight different tiles');

  const second = s.pick(2);
  assert.equal(s.hasMomentum, true);
  assert.equal(distinct(second), CANDIDATE_COUNT, 'the momentum grid collapsed two tiles together');
  // Tile 0 is the repeat; every other tile must be a fresh draw, so none of them
  // may coincide with it and they must differ between two consecutive momentum
  // generations even though tile 0 does not.
  const third = s.reroll();
  assert.deepEqual(Array.from(third.candidates[0] as Float64Array), Array.from(second.candidates[0] as Float64Array),
    'the repeat is not a repeat across a reroll');
  for (let k = 1; k < CANDIDATE_COUNT; k++) {
    assert.notDeepEqual(
      Array.from(third.candidates[k] as Float64Array),
      Array.from(second.candidates[k] as Float64Array),
      `tile ${k} was not redrawn by the reroll`,
    );
  }
});

test('momentum survives a reroll and a step change; back, setSubspace and start clear it', () => {
  const r = syntheticRegistry([spec({ group: 'structure', lo: -10, hi: 10, half: 1 })]);
  const theta = Float64Array.from([0]);

  const s = new ExplorerSearch(r, 0x99);
  s.start(theta);
  s.pick(1);
  assert.equal(s.hasMomentum, true, 'a pick creates momentum');
  s.reroll();
  assert.equal(s.hasMomentum, true, 'a reroll dropped momentum');
  s.setStep(0.9);
  assert.equal(s.hasMomentum, true, 'σ is a stride change, not a direction change');
  s.setSubspace('all');
  assert.equal(s.hasMomentum, true, 'setting the subspace it already has is a no-op');
  s.setSubspace('structure');
  assert.equal(s.hasMomentum, false, 'a real subspace change kept a stale direction');

  s.pick(1);
  assert.equal(s.hasMomentum, true);
  s.back();
  assert.equal(s.hasMomentum, false, 'undoing a step kept repeating it');

  s.pick(1);
  assert.equal(s.hasMomentum, true);
  s.start(theta);
  assert.equal(s.hasMomentum, false, 'a new run inherited the old run´s direction');

  // …and a momentum-free grid really does draw all eight, which is what makes
  // the flags above mean something.
  const set = s.current();
  assert.equal(new Set(set.candidates.map((c) => c[0])).size, CANDIDATE_COUNT);
});

test('momentum is measured over the slots free at the pick and applied over the slots free now', () => {
  // The subspace changes *between* generating a grid and picking from it, which
  // is the one moment those two sets differ. The winner moved a matrix slot; the
  // repeat must not, because the human has since frozen matrix.
  const r = syntheticRegistry([
    spec({ group: 'structure', lo: -10, hi: 10, half: 1 }),
    spec({ group: 'matrix', lo: -10, hi: 10, half: 1 }),
  ]);
  const s = new ExplorerSearch(r, 0xb11);
  const grid = s.start(new Float64Array(2));
  const winner = grid.candidates[4] as Float64Array;
  assert.notEqual(winner[1], grid.center[1], 'the fixture needs the matrix slot to have moved');

  s.setSubspace('structure');
  const next = s.pick(4);
  const structureSpec = r.slots[0] as ModSpec;
  const d = displacementBySpec(grid.center[0] as number, winner[0] as number, structureSpec);
  assert.equal((next.candidates[0] as Float64Array)[0], displaceBySpec(winner[0] as number, d, structureSpec));
  assert.equal((next.candidates[0] as Float64Array)[1], next.center[1], 'the frozen matrix slot was moved by the repeat');
  assertFrozenOutside(r, next, 'structure', 'post-freeze grid');
});

// ── 6. history ───────────────────────────────────────────────────────────────

test('back() restores the previous centre bit-identically, bumps the generation, drops momentum', () => {
  const r = syntheticRegistry([
    spec({ lo: -10, hi: 10, half: 1 }),
    spec({ lo: 0.1, hi: 10, half: 0.5, mult: true }),
  ]);
  const s = new ExplorerSearch(r, 0xbac4);
  const centres: number[][] = [];
  let set = s.start(Float64Array.from([0, 1]));
  for (let i = 0; i < 6; i++) {
    centres.push(Array.from(set.center));
    set = s.pick(i % CANDIDATE_COUNT);
    assert.equal(s.depth, i + 1, 'a pick did not push the outgoing centre');
  }
  const genBefore = s.generation;
  for (let i = 5; i >= 0; i--) {
    const back = s.back() as CandidateSet;
    assert.ok(back, 'back() gave up early');
    assert.deepEqual(Array.from(back.center), centres[i], `back to centre ${i}`);
    assert.equal(s.depth, i, 'back() did not pop');
    assert.equal(s.hasMomentum, false, 'back() kept the momentum it just undid');
    assert.equal(back.generation, genBefore + (6 - i), 'back() did not advance the generation');
  }
  assert.equal(s.back(), null, 'back() at the root should be null');
  assert.equal(s.generation, genBefore + 6, 'a null back() still advanced the generation');
});

test('the history caps at 64 and drops the oldest, not the newest', () => {
  const r = syntheticRegistry([spec({ lo: -10, hi: 10, half: 1 })]);
  const s = new ExplorerSearch(r, 0xca9);
  const centres: number[] = [];
  let set = s.start(new Float64Array(1));
  for (let i = 0; i < 70; i++) {
    centres.push(set.center[0] as number);
    set = s.pick(i % CANDIDATE_COUNT);
    assert.ok(s.depth <= 64, `depth reached ${s.depth}`);
  }
  assert.equal(s.depth, 64, 'the history should be full');

  // 70 picks, 64 remembered ⇒ the oldest 6 are gone and the walk back ends at
  // the centre that was current just before pick #6.
  let steps = 0;
  let last: CandidateSet | null = null;
  for (;;) {
    const b = s.back();
    if (!b) break;
    last = b;
    steps++;
    assert.ok(steps <= 64, 'the history grew past its cap');
  }
  assert.equal(steps, 64);
  assert.equal((last as CandidateSet).center[0], centres[70 - 64], 'the wrong end of the history was dropped');
  assert.equal(s.depth, 0);
});

// ── recenter: the outside world moving the centre ────────────────────────────

test('recenter rejects a wrong-length θ, and needs a run the same way pick does', () => {
  const r = syntheticRegistry([spec(), spec()]);
  const fresh = new ExplorerSearch(r, 1);
  // Argument validation comes first, exactly as `pick` validates its index before
  // it looks for a generation — otherwise the two calls report different faults
  // for the same mistake depending on when it is made.
  assert.throws(() => fresh.recenter(new Float64Array(1)), /length/);
  assert.throws(() => fresh.recenter(new Float64Array(3)), /length/);
  // …and with a legal θ but no run, it fails the way `pick`/`reroll` do.
  assert.throws(() => fresh.recenter(new Float64Array(2)), /before start\(\)/);

  const s = new ExplorerSearch(r, 1);
  s.start(new Float64Array(2));
  assert.throws(() => s.recenter(new Float64Array(1)), /length/);
  assert.throws(() => s.recenter(new Float64Array(0)), /length/);
  assert.doesNotThrow(() => s.recenter(new Float64Array(2)));
});

test('recenter takes the θ verbatim as the new centre and gives it a fresh genSeed', () => {
  const r = freenessRegistry();
  const theta = Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]);
  const target = nudged(r, theta, 0.4);

  const s = new ExplorerSearch(r, 0x4ece7);
  const before = s.start(theta);
  const after = s.recenter(target);

  assert.deepEqual(Array.from(after.center), Array.from(target), 'the centre is not the θ it was given');
  assert.equal(after.generation, before.generation + 1, 'recenter did not advance the generation');
  assert.notEqual(after.genSeed, before.genSeed, 'recenter reused the old generation key');
  assert.equal(s.generation, after.generation);
  // and the grid really did regrow — these are not the tiles from before
  for (let k = 0; k < CANDIDATE_COUNT; k++) {
    assert.notDeepEqual(
      Array.from(after.candidates[k] as Float64Array),
      Array.from(before.candidates[k] as Float64Array),
      `tile ${k} survived the recenter`,
    );
  }
});

test('recenter pushes the outgoing centre, so back() undoes a slider drag like a pick', () => {
  const r = syntheticRegistry([
    spec({ lo: -10, hi: 10, half: 1 }),
    spec({ group: 'matrix', lo: 0.1, hi: 10, half: 0.5, mult: true }),
  ]);
  const theta = Float64Array.from([0, 1]);
  const s = new ExplorerSearch(r, 0xd7a9);
  s.start(theta);
  s.pick(2);
  s.pick(5);
  const preDrag = s.current().center;
  const depth = s.depth;

  const target = nudged(r, theta, 0.27);
  s.recenter(target);
  assert.equal(s.depth, depth + 1, 'a slider drag with no undo is the one most likely to be a mistake');

  const back = s.back() as CandidateSet;
  assert.ok(back, 'back() after a recenter gave up');
  assert.deepEqual(Array.from(back.center), Array.from(preDrag), 'back() did not restore the pre-drag centre');
  assert.equal(s.depth, depth);

  // …and the history under it is intact: the picks that led there still unwind.
  assert.ok(s.back(), 'the recenter ate the generations that led to it');
  assert.ok(s.back(), 'the recenter ate the generations that led to it');
  assert.equal(s.back(), null, 'the root should now be reached');

  // A recenter is also subject to the same 64-deep cap as every other push.
  const t = new ExplorerSearch(r, 0xd7a9);
  t.start(theta);
  for (let i = 0; i < 70; i++) t.recenter(nudged(r, theta, i / 70));
  assert.equal(t.depth, 64);
});

test('recenter clears momentum: tile 0 is a fresh draw and the old displacement does not leak', () => {
  const r = syntheticRegistry([
    spec({ group: 'structure', lo: -10, hi: 10, half: 1 }),
    spec({ group: 'structure', lo: 0.1, hi: 10, half: 0.5, mult: true }),
  ]);
  const theta = Float64Array.from([0, 1]);
  const s = new ExplorerSearch(r, 0x3a5e);

  const grid = s.start(theta);
  const winner = grid.candidates[4] as Float64Array;
  s.pick(4);
  assert.equal(s.hasMomentum, true, 'the fixture needs momentum to exist before the recenter');

  const target = nudged(r, theta, 0.55);
  const after = s.recenter(target);
  assert.equal(s.hasMomentum, false, 'recenter kept a direction the human has just overruled');

  // Every tile is a fresh draw: none coincides with another…
  const distinct = new Set(after.candidates.map((c) => `${c[0]},${c[1]}`));
  assert.equal(distinct.size, CANDIDATE_COUNT, 'two tiles of a recentred grid are identical');
  // …and specifically, tile 0 is NOT the pre-recenter displacement replayed at
  // the new centre, which is what a leak would look like.
  for (let i = 0; i < r.length; i++) {
    const sp = r.slots[i] as ModSpec;
    const d = displacementBySpec(grid.center[i] as number, winner[i] as number, sp);
    assert.notEqual(d, 0, `the fixture needs slot ${i} to have actually moved`);
    const leak = displaceBySpec(after.center[i] as number, d, sp);
    assert.notEqual((after.candidates[0] as Float64Array)[i], leak, `slot ${i} leaked the old displacement`);
  }
});

test('recenter copies the θ it is given rather than retaining it', () => {
  const r = freenessRegistry();
  const theta = Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]);

  const clean = new ExplorerSearch(r, 0x0b1);
  clean.start(theta);
  const expected = [clean.recenter(nudged(r, theta, 0.31)), clean.reroll(), clean.pick(2)];

  const s = new ExplorerSearch(r, 0x0b1);
  s.start(theta);
  const owned = nudged(r, theta, 0.31);
  const got = [s.recenter(owned)];
  owned.fill(-777); // the panel is free to reuse its scratch vector
  got.push(s.reroll(), s.pick(2));

  for (const [i, set] of got.entries()) sameSet(set, expected[i] as CandidateSet, `after mutating θ, step ${i}`);
});

test('a recentred grid obeys freeness and bounds like every other generator', () => {
  for (const [label, r, theta] of [
    ['synthetic', freenessRegistry(), Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075])],
    ['physarum', physarumRegistry(4), physarumTheta(4)],
    ['plife', plifeRegistry(8), plifeTheta(8)],
  ] as [string, ThetaRegistry, Float64Array][]) {
    for (const sub of EXPLORER_SUBSPACES) {
      const s = new ExplorerSearch(r, 0x9e3 ^ sub.length);
      s.setSubspace(sub);
      s.setStep(MAX_STEP);
      s.start(theta);
      for (let g = 0; g < 6; g++) {
        const set = s.recenter(nudged(r, theta, g / 6));
        assertFrozenOutside(r, set, sub, `${label}/${sub} recenter ${g}`);
        assertInBounds(r, set, `${label}/${sub} recenter ${g}`);
        s.pick(g % CANDIDATE_COUNT);
      }
    }
  }
});

// ── 7. ownership ─────────────────────────────────────────────────────────────

test('nothing internal is aliased out: scribbling on a returned set changes nothing', () => {
  const r = freenessRegistry();
  const theta = Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]);

  const SEED = 0x5c81b;
  // The reference: the same script, run by a search nobody scribbles on.
  const clean = runScript(new ExplorerSearch(r, SEED), scriptFor(r, Float64Array.from(theta)));

  // The same script again, except every set handed out — including the extra
  // copies `current()` produces — is filled with garbage the moment it arrives.
  // If any of those arrays were the internal one, the *next* set would show it.
  const s = new ExplorerSearch(r, SEED);
  const dirty: CandidateSet[] = [];
  /** Copy the set out, then fill every array it handed over with garbage. */
  const wreck = (set: CandidateSet): CandidateSet => {
    dirty.push({
      center: Float64Array.from(set.center),
      candidates: set.candidates.map((c) => Float64Array.from(c)),
      genSeed: set.genSeed,
      generation: set.generation,
    });
    set.center.fill(999);
    for (const c of set.candidates) c.fill(-999);
    // `current()` is a second copy of the same internal state — wreck it too.
    const cur = s.current();
    cur.center.fill(-12345);
    for (const c of cur.candidates) c.fill(12345);
    return set;
  };

  for (const act of scriptFor(r, Float64Array.from(theta))) wreck(act(s) as CandidateSet);

  // Same script, same seed: the vandalised run must be indistinguishable from
  // the clean one. Any aliased array would have poisoned the very next centre.
  assert.equal(dirty.length, clean.length);
  for (const [i, set] of dirty.entries()) sameSet(set, clean[i] as CandidateSet, `vandalised step ${i}`);

  // …and the caller's θ is copied, not retained: mutating it after start() is inert.
  const own = Float64Array.from(theta);
  const t = new ExplorerSearch(r, 4);
  const before = t.start(own);
  own.fill(42);
  sameSet(t.current(), before, 'start() retained the caller´s θ');
});

// ── 8. validation ────────────────────────────────────────────────────────────

test('start rejects a θ of the wrong length and pick rejects an index off the grid', () => {
  const r = syntheticRegistry([spec(), spec()]);
  const s = new ExplorerSearch(r, 1);
  assert.throws(() => s.start(new Float64Array(1)), /length/);
  assert.throws(() => s.start(new Float64Array(3)), /length/);
  assert.throws(() => s.start(new Float64Array(0)), /length/);

  s.start(new Float64Array(2));
  for (const bad of [-1, 8, 9, 1.5, Number.NaN, Number.POSITIVE_INFINITY, -0.5, 100]) {
    assert.throws(() => s.pick(bad), /outside 0\.\.7/, `pick(${bad}) was accepted`);
  }
  for (let i = 0; i < CANDIDATE_COUNT; i++) assert.doesNotThrow(() => s.pick(i));
});

test('the acts that need a generation say so instead of reading a null centre', () => {
  const r = syntheticRegistry([spec()]);
  for (const act of [
    (s: ExplorerSearch) => s.pick(0),
    (s: ExplorerSearch) => s.reroll(),
    (s: ExplorerSearch) => s.back(),
    (s: ExplorerSearch) => s.current(),
    (s: ExplorerSearch) => s.recenter(new Float64Array(1)),
  ]) {
    assert.throws(() => act(new ExplorerSearch(r, 1)), /before start\(\)/);
  }
});

test('setStep clamps into [MIN_STEP, MAX_STEP] and survives junk', () => {
  const r = syntheticRegistry([spec()]);
  const s = new ExplorerSearch(r, 1);
  for (const [given, want] of [
    [0, MIN_STEP],
    [-5, MIN_STEP],
    [0.049, MIN_STEP],
    [MIN_STEP, MIN_STEP],
    [0.5, 0.5],
    [MAX_STEP, MAX_STEP],
    [2, MAX_STEP],
    [1e9, MAX_STEP],
  ] as [number, number][]) {
    s.setStep(given);
    assert.equal(s.step, want, `setStep(${given})`);
  }
  // Non-finite is not a step at all; whatever it lands on must still be legal.
  for (const junk of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    s.setStep(junk);
    assert.ok(s.step >= MIN_STEP && s.step <= MAX_STEP, `setStep(${junk}) → ${s.step}`);
  }
});

test('setSubspace reports what it was given, and takes effect from the next set', () => {
  const r = freenessRegistry();
  const s = new ExplorerSearch(r, 2);
  const before = s.start(Float64Array.from([0, 0, 0, 1.25, 0, 1, 3.5, 0.0075]));
  s.setSubspace('matrix');
  assert.equal(s.subspace, 'matrix');
  // the already-generated set is untouched by the change
  sameSet(s.current(), before, 'setSubspace rewrote the live grid');
  assertFrozenOutside(r, s.reroll(), 'matrix', 'after setSubspace');
});
