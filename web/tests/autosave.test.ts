/**
 * The autosave clock.
 *
 * These pin the two failures that made "my settings don't save" a recurring bug
 * report rather than a one-off, both of which are properties of the *timer*, not
 * of any config field:
 *
 * - A plain trailing debounce can be **starved**. `pane.refresh()` re-emits
 *   `change` for readonly monitors, so a folder-level handler sees a change
 *   stream at the refresh cadence — ~125 ms on a 240 Hz display against a 400 ms
 *   debounce — and the write is deferred forever. That is the exact mechanism
 *   behind the already-diagnosed "my exposure settings are lost on reload", and
 *   it is why `maxWaitMs` exists.
 * - An unflushed debounce loses the **last** edit. Move a slider, reload within
 *   the window, and precisely that one setting is gone while its neighbours are
 *   fine — which is close to undiagnosable from a bug report and reads as
 *   "settings save at random".
 *
 * Node has no DOM, so the hide-flush listeners are not installed here; `flush()`
 * is exercised directly, which is what those listeners call.
 */
import { test, mock } from 'node:test';
import assert from 'node:assert/strict';

import { createAutosave } from '../src/ui/autosave.ts';

test('a quiet edit is written once, after the debounce', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    let writes = 0;
    const autosave = createAutosave(() => writes++, { delayMs: 400, maxWaitMs: 2000 });
    autosave.schedule();
    mock.timers.tick(399);
    assert.equal(writes, 0, 'wrote before the debounce elapsed');
    mock.timers.tick(2);
    assert.equal(writes, 1);
    // Nothing dirty, so the deadline timer must not produce a second write.
    mock.timers.tick(5000);
    assert.equal(writes, 1);
    autosave.dispose();
  } finally {
    mock.timers.reset();
  }
});

test('a change stream faster than the debounce still gets written', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    let writes = 0;
    const autosave = createAutosave(() => writes++, { delayMs: 400, maxWaitMs: 2000 });
    // A readonly monitor re-emitting at a 240 Hz panel's refresh cadence. Under a
    // plain trailing debounce this writes zero times, forever.
    for (let elapsed = 0; elapsed < 3000; elapsed += 125) {
      autosave.schedule();
      mock.timers.tick(125);
    }
    assert.ok(writes >= 1, 'a starving change stream suppressed the write entirely');
    autosave.dispose();
  } finally {
    mock.timers.reset();
  }
});

test('flush writes a pending edit immediately, and only if there is one', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    let writes = 0;
    const autosave = createAutosave(() => writes++, { delayMs: 400, maxWaitMs: 2000 });
    autosave.flush();
    assert.equal(writes, 0, 'flushed a clean autosave');

    autosave.schedule();
    autosave.flush();
    assert.equal(writes, 1, 'the edit made just before a reload was lost');
    // The pending timers must not fire a duplicate after the flush consumed it.
    mock.timers.tick(5000);
    assert.equal(writes, 1);
    autosave.dispose();
  } finally {
    mock.timers.reset();
  }
});

test('dispose flushes, then stops accepting work', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    let writes = 0;
    const autosave = createAutosave(() => writes++, { delayMs: 400, maxWaitMs: 2000 });
    autosave.schedule();
    // A panel rebuild — a sim swap, a reroll — must not be a way to lose the
    // edit made a moment before it.
    autosave.dispose();
    assert.equal(writes, 1);

    autosave.schedule();
    mock.timers.tick(5000);
    assert.equal(writes, 1, 'a disposed autosave kept writing');
  } finally {
    mock.timers.reset();
  }
});
