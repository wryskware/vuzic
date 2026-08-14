/**
 * The persistence guarantee, tested at the seam that now enforces it.
 *
 * The recurring bug this closes is not any one setting — it is the *shape* of
 * the old wiring. Persistence used to be opt-in three different ways: a folder
 * you remembered to wrap, a `.on('change', persist)` you remembered to attach
 * per binding, and an `onChange?()` hook a folder builder's caller remembered to
 * pass. Two of those are silent when forgotten, and every "I tuned it, refreshed,
 * it was gone" report so far has been one of them: the accent arcs (reader list),
 * the whole impulse lane (no save path at all), the palette and render folders
 * (an optional host callback).
 *
 * So there is now exactly one mechanism — `persisting()` — and a type that makes
 * it non-optional: shared folder builders take a `PersistedContainer`, which only
 * `persisting()` can produce, and `createPanelTabs` wraps every tab page at the
 * single point they are created. A panel physically cannot reach an unpersisted
 * surface. The compiler enforces the *type* half of that; this file pins the
 * *behaviour* half, which no type can state:
 *
 *   - a binding added at any depth, at any time, schedules a save;
 *   - readonly monitors do not (they re-emit on every `pane.refresh()`, and
 *     including them would turn an idle panel into a permanent write loop);
 *   - buttons do not, which is why `saveNow` exists and is the one hand call.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { persisting, saveNow, type Autosave } from '../src/ui/autosave.ts';
import type { PanelContainer } from '../src/ui/panel.ts';

interface Change {
  (): void;
}

/**
 * A tweakpane stand-in that records the change handlers attached to it. Only the
 * four members `PanelContainer` names exist, which is the point: if a builder
 * ever reaches past that surface, this stops compiling and the seam gets to be
 * re-decided deliberately.
 */
function fakeContainer(fired: Change[]): PanelContainer {
  const container: PanelContainer = {
    addFolder: () => fakeContainer(fired),
    addBinding: ((): unknown => ({
      on: (_event: string, handler: Change) => {
        fired.push(handler);
        return {};
      },
    })) as PanelContainer['addBinding'],
    addButton: ((): unknown => ({
      on: () => ({}),
    })) as PanelContainer['addButton'],
    dispose: () => {},
  };
  return container;
}

function counter(): { autosave: Autosave; saves: () => number } {
  let saves = 0;
  const autosave: Autosave = {
    schedule: () => {
      saves += 1;
    },
    flush: () => {},
    dispose: () => {},
  };
  return { autosave, saves: () => saves };
}

test('a binding at any depth schedules a save', () => {
  const handlers: Change[] = [];
  const { autosave, saves } = counter();
  const root = persisting(fakeContainer(handlers), autosave);

  const state = { value: 1 };
  root.addBinding(state, 'value');
  root.addFolder({ title: 'one' }).addBinding(state, 'value');
  root
    .addFolder({ title: 'one' })
    .addFolder({ title: 'two' })
    .addFolder({ title: 'three' })
    .addBinding(state, 'value');

  assert.equal(handlers.length, 3, 'a binding somewhere below the wrapper was not hooked');
  for (const fire of handlers) fire();
  assert.equal(saves(), 3);
});

test('a folder added long after the wrap is still persisted', () => {
  // The failure this rules out is a wrapper that hooks what exists at the time
  // and not what arrives later — which would be an invisible expiry date on the
  // guarantee, and exactly what a panel that builds folders lazily would hit.
  const handlers: Change[] = [];
  const { autosave, saves } = counter();
  const root = persisting(fakeContainer(handlers), autosave);

  const state = { value: 1 };
  root.addBinding(state, 'value');
  handlers.forEach((fire) => fire());
  const before = saves();

  const late = root.addFolder({ title: 'added later' });
  late.addBinding(state, 'value');
  assert.equal(handlers.length, 2);
  (handlers[1] as Change)();
  assert.equal(saves(), before + 1);
});

test('readonly monitors are skipped, so an idle panel never writes', () => {
  const handlers: Change[] = [];
  const { autosave, saves } = counter();
  const root = persisting(fakeContainer(handlers), autosave);

  const readout = { text: '—' };
  root.addBinding(readout, 'text', { readonly: true });
  root.addFolder({ title: 'nested' }).addBinding(readout, 'text', { readonly: true, label: 'x' });

  assert.equal(handlers.length, 0, 'a monitor was hooked; refresh() would write forever');
  assert.equal(saves(), 0);
});

test('buttons are not hooked, and saveNow is how one asks', () => {
  const handlers: Change[] = [];
  const { autosave, saves } = counter();
  const root = persisting(fakeContainer(handlers), autosave);

  root.addButton({ title: 'test-fire snare' });
  assert.equal(handlers.length, 0, 'a button was hooked; test-fire would autosave');
  assert.equal(saves(), 0);

  // A button that edits config says so, in the one call that greps.
  saveNow(root);
  assert.equal(saves(), 1);
});

test('saveNow on a container that is not persisted is a no-op, not a crash', () => {
  // Not reachable through the types, but `saveNow` is exported and a plain
  // container is what a test or a future non-panel caller might hand it.
  const plain = fakeContainer([]) as never;
  assert.doesNotThrow(() => saveNow(plain));
});
