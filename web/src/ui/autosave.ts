/**
 * The workbench's autosave clock — one debouncer, shared by every panel.
 *
 * ## Why this is not just `setTimeout`
 *
 * Each panel used to own a private `let saveTimer` and a plain trailing
 * debounce, and that shape has failed twice in the same way. A trailing debounce
 * writes `delayMs` after the *last* change; if changes keep arriving faster than
 * `delayMs`, the timer is reset forever and the write never happens at all. In a
 * tweakpane panel that is not a hypothetical: `pane.refresh()` re-emits `change`
 * for readonly monitors on every refresh cycle, and a folder-level handler sees
 * those. On a 240 Hz display the refresh cadence is ~125 ms against a 400 ms
 * debounce, and the observed symptom was "my exposure settings are lost on
 * reload" — with the edit itself perfectly correct, saved by nothing.
 *
 * Filtering monitors out at each registration site fixes each *instance*.
 * **`maxWaitMs`** fixes the class: once dirty, the write happens within
 * `maxWaitMs` no matter what the change stream does, so a starving stream costs
 * a slightly stale save rather than a lost one.
 *
 * ## Why there is no flush on hide
 *
 * There was one — `pagehide` and `visibilitychange → hidden` wrote
 * synchronously, to save the edit made inside the last debounce window before a
 * reload. It was removed on 2026-08-14, deliberately, because it made every
 * background tab a writer: the autosave slot is a single key per sim, so a tab
 * holding stale config would overwrite a live tuning session in another tab the
 * moment it was hidden or closed. Automation tabs made that routine.
 *
 * The trade is small and one-sided. Losing a hide flush costs at most the last
 * 400 ms of edits before a reload — and `maxWaitMs` already bounds a dirty run
 * at 2 s regardless. Keeping it cost other tabs' work. Durable state that must
 * survive a hostile neighbour does not belong in the autosave slot at all; it
 * belongs in a named profile (`./profiles.ts`), which is written only when a
 * person presses a button.
 */

import type { FolderApi } from 'tweakpane';

import type { PanelContainer } from './panel.ts';

export interface Autosave {
  /** Mark dirty and start (or extend) the debounce. Safe to call at any rate. */
  schedule(): void;
  /** Write now, if dirty. Used by explicit save buttons and by `dispose`. */
  flush(): void;
  /** Flushes any pending write, then stops. Call when a panel is torn down. */
  dispose(): void;
}

export interface AutosaveOptions {
  /** quiet period after the last change before writing */
  delayMs?: number;
  /**
   * Hard ceiling from the *first* change of a dirty run. This is the anti-
   * starvation bound — see the header. 2 s is chosen to be long enough that a
   * slider drag still coalesces into one or two writes, and short enough that
   * nothing a person could do between an edit and a reload falls through it.
   */
  maxWaitMs?: number;
}

export function createAutosave(write: () => void, options: AutosaveOptions = {}): Autosave {
  const delayMs = options.delayMs ?? 400;
  const maxWaitMs = options.maxWaitMs ?? 2000;
  let quiet: ReturnType<typeof setTimeout> | null = null;
  let deadline: ReturnType<typeof setTimeout> | null = null;
  let dirty = false;
  let disposed = false;

  const clear = (): void => {
    if (quiet !== null) clearTimeout(quiet);
    if (deadline !== null) clearTimeout(deadline);
    quiet = null;
    deadline = null;
  };

  const autosave: Autosave = {
    schedule(): void {
      if (disposed) return;
      dirty = true;
      if (quiet !== null) clearTimeout(quiet);
      quiet = setTimeout(() => autosave.flush(), delayMs);
      // Set once per dirty run and never extended — extending it would make it a
      // second debounce rather than a ceiling, which is the whole bug.
      if (deadline === null) deadline = setTimeout(() => autosave.flush(), maxWaitMs);
    },
    flush(): void {
      clear();
      if (!dirty || disposed) return;
      dirty = false;
      write();
    },
    dispose(): void {
      // Flush before tearing down: a panel rebuild (a sim swap, a reroll) must
      // not be a way to lose the edit you made a moment before it.
      autosave.flush();
      disposed = true;
      clear();
    },
  };

  return autosave;
}

/** How a persisted container carries its autosave, for `saveNow` to find. */
const AUTOSAVE = Symbol('persisting.autosave');

declare const persistedBrand: unique symbol;

/**
 * A container whose bindings — and the bindings of every folder it makes, at any
 * depth, whenever they are added — schedule a save.
 *
 * The brand is unforgeable outside this module, so `persisting()` is the *only*
 * way to obtain one. That is the point: a folder builder that takes a
 * `PersistedContainer` cannot be handed a raw pane or folder, so "this widget
 * was never wired to the save path" stops being something a person can do by
 * omission and becomes a type error at the call site.
 *
 * This replaced two idioms that were both opt-in and both silent when forgotten:
 * `.on('change', persist)` attached per binding (~40 of them), and an optional
 * `onChange?()` on a folder builder's host, which every widget inside had to
 * remember to call. The palette and render folders shipped with the second one,
 * and every setting in them depended on a callback the *caller* could simply not
 * pass — which is exactly how "I tweaked it, refreshed, it was gone" keeps
 * happening.
 */
export type PersistedContainer<T extends PanelContainer = PanelContainer> = Omit<T, 'addFolder'> & {
  /**
   * Typed to return a persisted container, because at runtime it does — the
   * proxy re-wraps every folder it makes. Without this the brand would stop at
   * the first `addFolder` and a nested folder would look unpersisted to the
   * compiler while being persisted in fact, which is the wrong way round for a
   * type whose whole job is to be believed.
   */
  addFolder(params: Parameters<PanelContainer['addFolder']>[0]): PersistedContainer<FolderApi>;
  readonly [persistedBrand]: true;
};

/**
 * Wrap a panel container so **everything added to it persists, by construction**.
 *
 * This is the answer to the recurring "a new setting was added but not wired
 * into the save path". Previously each panel attached `.on('change', persist)`
 * to each of some forty bindings by hand, so persisting a new widget was a line
 * you had to remember, in a file where forty neighbours already had it and none
 * of them would fail if yours were missing. Here you cannot remember it, because
 * there is nothing to remember: hand a folder to `persisting` once and every
 * binding under it — and under every sub-folder it creates, however deep and
 * whenever it is added — schedules a save.
 *
 * Readonly bindings are skipped, from the `readonly` flag the call site already
 * passes rather than from a maintained exclusion list. That matters: monitors
 * re-emit `change` on every `pane.refresh()`, so including them would turn an
 * idle panel into a write every `maxWaitMs` forever. Detecting them from their
 * own declaration is the one way to get this right that cannot drift.
 *
 * Buttons are passed through untouched — a button emits no `change`, so one that
 * mutates config (a "reset to defaults") still calls `schedule()` itself. There
 * are a handful, they are conspicuous, and a wrapper that fired on every button
 * would be saving on "test-fire snare" too.
 *
 * A `Proxy` rather than a hand-built façade because callers legitimately reach
 * past the three methods `PanelContainer` names — `palette-folder.ts` sets
 * `folder.hidden` to show the arc controls only in arc mode, for one — and a
 * façade would swallow those writes silently, which is a worse bug than the one
 * being fixed. The proxy intercepts exactly two members and forwards the rest.
 */
export function persisting<T extends PanelContainer>(
  container: T,
  autosave: Autosave,
): PersistedContainer<T> {
  return new Proxy(container, {
    get(target, prop, receiver) {
      if (prop === AUTOSAVE) return autosave;
      if (prop === 'addFolder') {
        return (params: Parameters<PanelContainer['addFolder']>[0]) =>
          persisting(target.addFolder(params), autosave);
      }
      if (prop === 'addBinding') {
        return (...args: Parameters<PanelContainer['addBinding']>) => {
          const binding = target.addBinding(...args);
          const params = args[2] as { readonly?: boolean } | undefined;
          if (params?.readonly !== true) binding.on('change', () => autosave.schedule());
          return binding;
        };
      }
      const value = Reflect.get(target, prop, receiver);
      // Bound to the real container: tweakpane's APIs are classes with private
      // state, and an unbound method called through the proxy loses `this`.
      return typeof value === 'function' ? (value as () => unknown).bind(target) : value;
    },
    // `folder.hidden = …` and `folder.title = …` are accessors on tweakpane's
    // prototypes that reach into the real controller. Forwarding with the target
    // as the receiver rather than the proxy keeps `this` the genuine object.
    set(target, prop, value) {
      return Reflect.set(target, prop, value, target);
    },
  }) as unknown as PersistedContainer<T>;
}

/**
 * Ask a persisted container's autosave to save, for the one thing its wrapper
 * cannot see: a **button** that edits config.
 *
 * Buttons emit no `change`, so `persisting` deliberately passes them through
 * untouched — wrapping them would save on "test-fire snare" too. A button that
 * mutates state therefore has to say so, and this is how, in one named import
 * that greps: `saveNow(container)`. It is the only hand-written save call left
 * in the panel layer, and there are a handful of them.
 */
export function saveNow(container: PersistedContainer): void {
  const carried = (container as unknown as Record<symbol, Autosave | undefined>)[AUTOSAVE];
  carried?.schedule();
}
