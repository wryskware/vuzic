/**
 * Headless tests for the explorer verdict log (`explore/log.ts`).
 *
 * House rules are `modulation.test.ts`'s: `node --test` loads the modules
 * straight out of src/, which is why every import carries an explicit `.ts`
 * extension, and nothing here touches the DOM or WebGPU.
 *
 * `localStorage` does not exist in Node, so this file is the one place in the
 * suite that installs a browser global. Every test that does so restores it in a
 * `finally` — `node --test` runs one process per *file*, so a leaked stub would
 * be visible to every later test here, and a leaked `console.warn` stub would
 * silently swallow output for the rest of the run.
 *
 * What these pin:
 *
 * - **The stamps are the log's, not the caller's.** `at` and `simId` are the two
 *   fields a training loader cannot recover from anywhere else, and a mislabelled
 *   `simId` is a θ paired with the wrong registry, which is nonsense.
 * - **The store is treated as hostile.** It is a string a user can hand-edit, a
 *   quota that can fill, and a feature a browser can switch off; none of those
 *   may cost the click.
 * - **The two write failures are discriminated.** A quota failure drops history;
 *   an unusable store must not, because dropping there destroys good data to
 *   appease a sink that will never accept it.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ExplorerLog, type ExplorerAction, type ExplorerEntry } from '../src/explore/log.ts';
import { EXPLORER_SUBSPACES, type ExplorerSubspace } from '../src/explore/search.ts';

const KEY = 'lmt.explorerLog.v1';
const MAX_ENTRIES = 200;

// ── a controllable localStorage ──────────────────────────────────────────────

interface Stub {
  /** the single stored value, or null */
  value: string | null;
  sets: number;
  gets: number;
  removes: number;
  /** every write throws — a disabled / private-mode store */
  failAlways: boolean;
  /** writes longer than this throw a quota error */
  maxLength: number;
  /** reads throw — a store that is present but broken */
  failReads: boolean;
}

function quotaError(): Error {
  const e = new Error('QuotaExceededError: storage is full');
  e.name = 'QuotaExceededError';
  return e;
}

/**
 * Installs a stub `globalThis.localStorage` and returns it plus its undo.
 *
 * The log reaches the store through the bare identifier `localStorage`, which
 * resolves against `globalThis` at call time — so installing it here is enough,
 * and deleting it again really does put the module back to "no storage at all".
 */
function installStorage(initial: string | null = null): { stub: Stub; restore: () => void } {
  const stub: Stub = {
    value: initial,
    sets: 0,
    gets: 0,
    removes: 0,
    failAlways: false,
    maxLength: Number.POSITIVE_INFINITY,
    failReads: false,
  };
  const api = {
    getItem(k: string): string | null {
      stub.gets++;
      if (stub.failReads) throw new Error('storage is disabled');
      return k === KEY ? stub.value : null;
    },
    setItem(k: string, v: string): void {
      stub.sets++;
      if (stub.failAlways) throw new Error('storage is disabled');
      if (v.length > stub.maxLength) throw quotaError();
      if (k === KEY) stub.value = v;
    },
    removeItem(k: string): void {
      stub.removes++;
      if (stub.failAlways) throw new Error('storage is disabled');
      if (k === KEY) stub.value = null;
    },
  };
  const g = globalThis as Record<string, unknown>;
  const had = 'localStorage' in g;
  const previous = g['localStorage'];
  g['localStorage'] = api;
  return {
    stub,
    restore: () => {
      if (had) g['localStorage'] = previous;
      else delete g['localStorage'];
    },
  };
}

/** Silences (and counts) the one warning the unusable-store path emits. */
function muteWarn(): { calls: number; restore: () => void } {
  const state = { calls: 0, restore: () => {} };
  const original = console.warn;
  console.warn = () => {
    state.calls++;
  };
  state.restore = () => {
    console.warn = original;
  };
  return state;
}

type Draft = Omit<ExplorerEntry, 'at' | 'simId'>;

function draft(over: Partial<Draft> = {}): Draft {
  return {
    action: 'pick',
    generation: 0,
    subspace: 'all',
    step: 0.35,
    genSeed: 12345,
    theta: [1, 2, 3],
    ...over,
  };
}

/** A well-formed stored row, for the corruption tests to mutate. */
function storedRow(over: Partial<ExplorerEntry> = {}): ExplorerEntry {
  return {
    at: '2026-08-08T00:00:00.000Z',
    simId: 'plife',
    action: 'pick',
    generation: 3,
    subspace: 'matrix',
    step: 0.5,
    genSeed: 999,
    theta: [0.1, 0.2],
    ...over,
  };
}

const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

// ── stamping ─────────────────────────────────────────────────────────────────

test('append stamps `at` and `simId` itself, and no caller can override either', () => {
  const { stub, restore } = installStorage();
  try {
    const before = Date.now();
    const log = new ExplorerLog('physarum');
    // The type forbids these two keys; the point is that the *runtime* forbids
    // them too, because a mislabelled simId pairs a θ with the wrong registry.
    const smuggled = { ...draft(), at: 'not-a-date', simId: 'plife' } as unknown as Draft;
    const entry = log.append(smuggled);

    assert.equal(entry.simId, 'physarum', 'the caller relabelled the substrate');
    assert.match(entry.at, ISO, `at is not an ISO stamp: ${entry.at}`);
    const t = Date.parse(entry.at);
    assert.ok(t >= before && t <= Date.now() + 1000, `at ${entry.at} is not now`);

    // …and the entry that came back is the entry that was kept.
    assert.equal(log.size, 1);
    assert.deepEqual(log.entries[0], entry);
    assert.deepEqual(JSON.parse(stub.value as string), [entry], 'the store disagrees with memory');
  } finally {
    restore();
  }
});

test('every action and subspace the vocabulary names round-trips through the store', () => {
  const { restore } = installStorage();
  try {
    const actions: ExplorerAction[] = [
      'start',
      'pick',
      'reroll',
      'back',
      'like',
      'adopt',
      'recenter',
      'revert',
    ];
    const log = new ExplorerLog('plife');
    for (const action of actions) {
      for (const subspace of EXPLORER_SUBSPACES as readonly ExplorerSubspace[]) {
        log.append(draft({ action, subspace }));
      }
    }
    const written = actions.length * EXPLORER_SUBSPACES.length;
    assert.equal(log.size, written);
    // A reload must recover all of them: the shape check on read is where a typo
    // in either vocabulary would quietly eat rows.
    const reloaded = new ExplorerLog('plife');
    assert.equal(reloaded.size, written, 'the read-back shape check dropped valid rows');
    assert.deepEqual(reloaded.entries.map((e) => e.action), log.entries.map((e) => e.action));
  } finally {
    restore();
  }
});

// ── theta ────────────────────────────────────────────────────────────────────

test('theta is rounded to five decimals and non-finite values become 0', () => {
  const { restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    const entry = log.append(
      draft({
        theta: [
          1.234_567_89,
          -1.234_567_89,
          0.000_004_9,
          0.000_005_1,
          123.456_789_1,
          Number.NaN,
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          0,
          -3,
        ],
      }),
    );
    assert.equal(entry.theta[0], 1.23457);
    assert.equal(entry.theta[1], -1.23457);
    assert.equal(entry.theta[2], 0, 'below the fifth decimal is 0, not a long float');
    assert.equal(entry.theta[3], 0.00001);
    assert.equal(entry.theta[4], 123.45679, 'five decimals kept');
    assert.equal(entry.theta[5], 0, 'NaN must be 0, not null — JSON turns NaN into null');
    assert.equal(entry.theta[6], 0, 'Infinity must be 0');
    assert.equal(entry.theta[7], 0, '-Infinity must be 0');
    assert.equal(entry.theta[8], 0);
    assert.equal(entry.theta[9], -3);
    for (const v of entry.theta) assert.ok(Number.isFinite(v), `${v} is not a finite float`);

    // Rounding is what halves the JSON, so it has to survive stringification too:
    // no value may come back out with a sixth decimal.
    for (const line of log.exportJsonl().split('\n')) {
      for (const v of (JSON.parse(line) as ExplorerEntry).theta) {
        assert.ok(Math.abs(v * 1e5 - Math.round(v * 1e5)) < 1e-6, `${v} carries more than five decimals`);
      }
    }
  } finally {
    restore();
  }
});

test('append hands back a copy: annotating it does not edit the stored row', () => {
  const { stub, restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    const caller = [1.5, 2.5];
    const entry = log.append(draft({ theta: caller, note: 'as written' }));

    entry.note = 'scribbled on afterwards';
    entry.theta[0] = -999;
    entry.action = 'revert';
    // …and the caller's own θ array is not retained either.
    caller[1] = -888;

    assert.equal(log.entries[0]?.note, 'as written', 'the stored row followed the caller´s edit');
    assert.deepEqual(log.entries[0]?.theta, [1.5, 2.5], 'the stored θ followed the caller´s edit');
    assert.equal(log.entries[0]?.action, 'pick');
    const stored = JSON.parse(stub.value as string) as ExplorerEntry[];
    assert.deepEqual(stored[0]?.theta, [1.5, 2.5]);
    assert.equal((JSON.parse(log.exportJsonl()) as ExplorerEntry).note, 'as written');
  } finally {
    restore();
  }
});

test('an empty theta is legal, and a long one is not truncated', () => {
  const { restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    assert.deepEqual(log.append(draft({ theta: [] })).theta, []);
    const long = Array.from({ length: 268 }, (_, i) => i / 7);
    assert.equal(log.append(draft({ theta: long })).theta.length, 268);
  } finally {
    restore();
  }
});

// ── the cap ──────────────────────────────────────────────────────────────────

test('the log caps at 200 entries and drops the oldest', () => {
  const { stub, restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    for (let i = 0; i < MAX_ENTRIES + 37; i++) {
      log.append(draft({ generation: i }));
      assert.ok(log.size <= MAX_ENTRIES, `size reached ${log.size}`);
    }
    assert.equal(log.size, MAX_ENTRIES);
    assert.equal(log.entries[0]?.generation, 37, 'the wrong end was dropped');
    assert.equal(log.entries[MAX_ENTRIES - 1]?.generation, MAX_ENTRIES + 36);
    // and the stored copy agrees, so a reload sees the same window
    const stored = JSON.parse(stub.value as string) as ExplorerEntry[];
    assert.equal(stored.length, MAX_ENTRIES);
    assert.equal(stored[0]?.generation, 37);
  } finally {
    restore();
  }
});

// ── export ───────────────────────────────────────────────────────────────────

test('exportJsonl is one parseable object per line, in order, with no trailing newline', () => {
  const { restore } = installStorage();
  try {
    const log = new ExplorerLog('physarum');
    assert.equal(log.exportJsonl(), '', 'an empty log exports nothing, not a blank line');

    for (let i = 0; i < 5; i++) log.append(draft({ generation: i, genSeed: 1000 + i, pickedIndex: i % 8 }));
    const text = log.exportJsonl();
    const lines = text.split('\n');
    assert.equal(lines.length, 5);
    assert.equal(text.includes('\n\n'), false);
    assert.equal(text.endsWith('\n'), false, 'a trailing newline is a sixth, empty record');
    for (const [i, line] of lines.entries()) {
      const parsed = JSON.parse(line) as ExplorerEntry;
      assert.equal(parsed.generation, i, 'the export is out of order');
      assert.equal(parsed.genSeed, 1000 + i);
      assert.equal(parsed.pickedIndex, i % 8);
      assert.equal(parsed.simId, 'physarum');
      assert.match(parsed.at, ISO);
    }
  } finally {
    restore();
  }
});

test('clear() empties memory and the store together', () => {
  const { stub, restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    for (let i = 0; i < 5; i++) log.append(draft({ generation: i }));
    assert.equal(log.size, 5);

    log.clear();
    assert.equal(log.size, 0);
    assert.deepEqual([...log.entries], []);
    assert.equal(log.exportJsonl(), '');
    assert.equal(stub.value, null, 'the store still holds the cleared entries');
    assert.equal(stub.removes, 1);
    // A reload after a clear is empty, which is the property a user asking to
    // clear actually cares about.
    assert.equal(new ExplorerLog('plife').size, 0);
    // …and the log keeps working afterwards
    log.append(draft());
    assert.equal(log.size, 1);
  } finally {
    restore();
  }
});

// ── reading a hostile store ──────────────────────────────────────────────────

test('a corrupt stored value starts the log empty instead of throwing', () => {
  for (const bad of ['{', 'not json at all', '', '[1,2,3', ' ', 'undefined']) {
    const { restore } = installStorage(bad);
    try {
      let log!: ExplorerLog;
      assert.doesNotThrow(() => {
        log = new ExplorerLog('plife');
      }, `constructor threw on ${JSON.stringify(bad)}`);
      assert.equal(log.size, 0, `${JSON.stringify(bad)} was read as a log`);
      // …and it is still usable: the log is a bonus, not a precondition
      log.append(draft());
      assert.equal(log.size, 1);
    } finally {
      restore();
    }
  }
});

test('a stored value that is valid JSON but not an array is ignored', () => {
  for (const bad of ['{"a":1}', '"a string"', '42', 'null', 'true']) {
    const { restore } = installStorage(bad);
    try {
      assert.equal(new ExplorerLog('plife').size, 0, `${bad} was accepted as a log`);
    } finally {
      restore();
    }
  }
});

test('malformed rows are filtered out and the good ones survive alongside them', () => {
  const good = [storedRow({ generation: 1 }), storedRow({ generation: 2, action: 'adopt', subspace: 'all' })];
  const bad: unknown[] = [
    null,
    undefined,
    42,
    'a string',
    [],
    {},
    storedRow({ at: 12 as unknown as string }),
    storedRow({ simId: null as unknown as string }),
    storedRow({ action: 'nonsense' as ExplorerAction }),
    storedRow({ action: undefined as unknown as ExplorerAction }),
    storedRow({ subspace: 'brightness' as unknown as ExplorerSubspace }),
    storedRow({ generation: '3' as unknown as number }),
    storedRow({ step: null as unknown as number }),
    storedRow({ genSeed: Number.NaN }),
    storedRow({ genSeed: 'x' as unknown as number }),
    storedRow({ theta: 'not an array' as unknown as number[] }),
    storedRow({ theta: undefined as unknown as number[] }),
  ];
  // Interleaved, so a filter that stops at the first bad row is caught too.
  const mixed: unknown[] = [];
  for (const [i, b] of bad.entries()) {
    mixed.push(b);
    if (i === 4) mixed.push(good[0]);
    if (i === 11) mixed.push(good[1]);
  }
  const { restore } = installStorage(JSON.stringify(mixed));
  try {
    const log = new ExplorerLog('plife');
    assert.equal(log.size, 2, `${log.size} rows survived; expected the 2 well-formed ones`);
    assert.deepEqual(log.entries.map((e) => e.generation), [1, 2]);
    assert.deepEqual([...log.entries], good);
    // Nothing malformed may reach the JSONL a training loader streams.
    for (const line of log.exportJsonl().split('\n')) {
      const e = JSON.parse(line) as ExplorerEntry;
      assert.ok(Number.isFinite(e.genSeed) && Array.isArray(e.theta) && typeof e.simId === 'string');
    }
  } finally {
    restore();
  }
});

test("the panel↔grid merge's two new actions are written, stored and read back", () => {
  // `recenter` and `revert` arrived with the merge. They have to survive the
  // *stored* round trip specifically: the read-side validator is a separate list
  // from the type union, and a union that grew while the set did not would make
  // every new row vanish on the next reload rather than fail loudly.
  const { stub, restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    const recentre = log.append(draft({ action: 'recenter', generation: 4 }));
    const revert = log.append(draft({ action: 'revert', generation: 5 }));
    assert.equal(recentre.action, 'recenter');
    assert.equal(revert.action, 'revert');

    const stored = JSON.parse(stub.value as string) as ExplorerEntry[];
    assert.deepEqual(stored.map((e) => e.action), ['recenter', 'revert']);
    const reloaded = new ExplorerLog('plife');
    assert.equal(reloaded.size, 2, 'a new action was dropped by the read-side check');
    assert.deepEqual(reloaded.entries.map((e) => e.action), ['recenter', 'revert']);
    // …and they reach the JSONL a training loader streams, not just memory
    assert.deepEqual(
      log.exportJsonl().split('\n').map((l) => (JSON.parse(l) as ExplorerEntry).action),
      ['recenter', 'revert'],
    );
  } finally {
    restore();
  }
});

test("a pre-merge 'adopt' row still loads — the store predates the vocabulary change", () => {
  // `adopt` is no longer written, but a browser that ran the old build has rows
  // carrying it. Retiring the value from the *validator* would silently eat that
  // history on the first load of the new build.
  const legacy = [
    storedRow({ action: 'adopt', generation: 1 }),
    storedRow({ action: 'like', generation: 2 }),
    storedRow({ action: 'recenter', generation: 3 }),
  ];
  const { restore } = installStorage(JSON.stringify(legacy));
  try {
    const log = new ExplorerLog('plife');
    assert.equal(log.size, 3, 'a legacy row was evicted');
    assert.deepEqual(log.entries.map((e) => e.action), ['adopt', 'like', 'recenter']);
    // …and a new row appends alongside it rather than replacing the history
    log.append(draft({ action: 'revert', generation: 4 }));
    assert.deepEqual(new ExplorerLog('plife').entries.map((e) => e.action), [
      'adopt',
      'like',
      'recenter',
      'revert',
    ]);
  } finally {
    restore();
  }
});

test('a bogus action is still rejected, including near-misses of the new values', () => {
  // The widened set must not have widened into "anything goes": the actions are
  // a closed vocabulary a loader switches on.
  const bogus = ['recentre', 'Recenter', 'RECENTER', 'reverts', 'revert ', '', 'exit', 'adopted', 'undo'];
  const { restore } = installStorage(
    JSON.stringify([
      ...bogus.map((action) => storedRow({ action: action as ExplorerAction })),
      storedRow({ action: 'revert', generation: 99 }),
    ]),
  );
  try {
    const log = new ExplorerLog('plife');
    assert.equal(log.size, 1, `${log.size - 1} bogus actions were let through`);
    assert.equal(log.entries[0]?.action, 'revert');
    assert.equal(log.entries[0]?.generation, 99);
  } finally {
    restore();
  }
});

test('rows from another sim are kept — simId is the discriminator, one file for the loader', () => {
  const { restore } = installStorage(JSON.stringify([storedRow({ simId: 'physarum', generation: 7 })]));
  try {
    const log = new ExplorerLog('plife');
    assert.equal(log.size, 1, 'a foreign row was evicted on read');
    const appended = log.append(draft({ generation: 8 }));
    assert.equal(appended.simId, 'plife');
    assert.deepEqual(log.entries.map((e) => e.simId), ['physarum', 'plife']);
  } finally {
    restore();
  }
});

test('a store whose reads throw is the same as no store: empty, and still usable', () => {
  const { stub, restore } = installStorage('[]');
  try {
    stub.failReads = true;
    let log!: ExplorerLog;
    assert.doesNotThrow(() => {
      log = new ExplorerLog('plife');
    });
    assert.equal(log.size, 0);
    stub.failReads = false;
    log.append(draft());
    assert.equal(log.size, 1);
  } finally {
    restore();
  }
});

test('no localStorage at all: construct, append, export and clear all survive', () => {
  const g = globalThis as Record<string, unknown>;
  const had = 'localStorage' in g;
  const previous = g['localStorage'];
  delete g['localStorage'];
  const warn = muteWarn();
  try {
    let log!: ExplorerLog;
    assert.doesNotThrow(() => {
      log = new ExplorerLog('plife');
    }, 'the constructor needs a store it may not have');
    assert.equal(log.size, 0);
    assert.doesNotThrow(() => log.append(draft()), 'a missing store cost the click');
    assert.doesNotThrow(() => log.append(draft({ generation: 1 })));
    assert.equal(log.size, 2, 'the entries must survive in memory');
    assert.equal(log.exportJsonl().split('\n').length, 2);
    assert.doesNotThrow(() => log.clear());
    assert.equal(log.size, 0);
  } finally {
    warn.restore();
    if (had) g['localStorage'] = previous;
    else delete g['localStorage'];
  }
});

// ── writing to a hostile store ───────────────────────────────────────────────

test('a quota failure drops the oldest half and retries, rather than losing the click', () => {
  const { stub, restore } = installStorage();
  try {
    const log = new ExplorerLog('plife');
    for (let i = 0; i < 20; i++) log.append(draft({ generation: i }));
    assert.equal(log.size, 20);

    // A budget that fits 20 rows but not 21: the next append's first write must
    // fail on size and the halved retry must succeed.
    stub.maxLength = (stub.value as string).length;
    const setsBefore = stub.sets;
    assert.doesNotThrow(() => log.append(draft({ generation: 20 })));
    assert.equal(stub.sets - setsBefore, 2, 'the quota path should be one failed write and one retry');

    // 21 in hand, half of them (floor) dropped ⇒ 11 kept, newest first survivor.
    assert.equal(log.size, 11, `expected 11 survivors, got ${log.size}`);
    assert.equal(log.entries[0]?.generation, 10, 'the newest, not the oldest, were dropped');
    assert.equal(log.entries[10]?.generation, 20);
    const stored = JSON.parse(stub.value as string) as ExplorerEntry[];
    assert.deepEqual(stored.map((e) => e.generation), log.entries.map((e) => e.generation),
      'memory and the store disagree after the retry');

    // …and the log keeps writing afterwards, i.e. it did not give up on the store
    stub.maxLength = Number.POSITIVE_INFINITY;
    const sets = stub.sets;
    log.append(draft({ generation: 21 }));
    assert.equal(stub.sets - sets, 1, 'a recovered store should take a single write');
    assert.equal(log.size, 12);
  } finally {
    restore();
  }
});

test('an unusable store keeps every entry in memory, never throws, and is not retried', () => {
  const { stub, restore } = installStorage();
  const warn = muteWarn();
  try {
    const log = new ExplorerLog('plife');
    stub.failAlways = true;

    assert.doesNotThrow(() => log.append(draft({ generation: 0 })), 'a broken store cost the click');
    // The discrimination is the retry itself: the full write and the halved
    // write, and only then the conclusion that this is not a size problem.
    assert.equal(stub.sets, 2, 'the halving retry should have been attempted exactly once');
    assert.equal(log.size, 1, 'entries were dropped to appease a store that will never take them');
    assert.equal(warn.calls, 1, 'the warning should fire once, not once per click');

    for (let i = 1; i < 12; i++) log.append(draft({ generation: i }));
    assert.equal(stub.sets, 2, 'the log kept hammering a store it already knows is dead');
    assert.equal(warn.calls, 1, 'the warning fired again');
    assert.equal(log.size, 12, 'the in-memory history was not retained');
    assert.deepEqual(log.entries.map((e) => e.generation), [...Array(12).keys()]);

    // Everything else still works off the in-memory list.
    assert.equal(log.exportJsonl().split('\n').length, 12);
    assert.doesNotThrow(() => log.clear());
    assert.equal(log.size, 0);
  } finally {
    warn.restore();
    restore();
  }
});

test('a store that fails only the halved retry still keeps the full history', () => {
  // The other half of the discrimination: if even half fails, the problem was
  // never size, so nothing may be dropped.
  const { stub, restore } = installStorage();
  const warn = muteWarn();
  try {
    const log = new ExplorerLog('plife');
    for (let i = 0; i < 10; i++) log.append(draft({ generation: i }));
    stub.maxLength = 1; // no payload of any plausible size fits
    assert.doesNotThrow(() => log.append(draft({ generation: 10 })));
    assert.equal(log.size, 11, 'the halved retry failing must not cost the entries');
    assert.deepEqual(log.entries.map((e) => e.generation), [...Array(11).keys()]);
    assert.equal(warn.calls, 1);
  } finally {
    warn.restore();
    restore();
  }
});
