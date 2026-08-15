/**
 * The pure half of the global keydown layer (handoffs/keybindings.md).
 *
 * Nothing here touches the DOM, so it loads straight into `node --test`: the
 * host in `main.ts` supplies the one DOM-shaped fact — `ev.target` — as
 * `TargetInfo`, and dispatches on the returned `KeyPlan`. `planKey` is the
 * single source of truth for which keys the app claims, so adding a binding
 * means adding a branch here plus the matching handler branch, and the
 * coverage in `tests/keys.test.ts` names both.
 *
 * `ev.code` is stable across layouts (the key's physical position), while
 * `ev.key` is not — layout-dependent letters are exactly what a profile-name
 * field may be typed in, so claims are keyed on `code` from day one.
 */

interface PlanBase {
  /** The handler should call `ev.preventDefault()` after acting on this. */
  preventDefault: boolean;
}

export type KeyPlan =
  | (PlanBase & { kind: 'none' })
  | (PlanBase & { kind: 'halt' })
  | (PlanBase & { kind: 'play-toggle' })
  | (PlanBase & { kind: 'seek'; delta: number })
  | (PlanBase & { kind: 'explorer-reroll' })
  | (PlanBase & { kind: 'explorer-back' })
  | (PlanBase & { kind: 'explorer-exit' })
  | (PlanBase & { kind: 'toggle-panel' })
  | (PlanBase & { kind: 'toggle-timeline' });

/**
 * The one DOM fact `planKey` needs. The `tagName` of an HTML event target is
 * always a bare tag name, so `undefined` here means "not one" — a `window`
 * target, where the bare-letter claims are exactly what you want.
 */
export interface TargetInfo {
  tagName: string | undefined;
  isContentEditable: boolean;
}

const none = { kind: 'none' as const, preventDefault: false };

const EDITABLE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * The load-bearing guard, promoted to its own function so the test can name
 * it. Superset of the old `HTMLInputElement` bail: the settings panel and the
 * profile store both put text into the UI, and none of that may toggle it.
 */
export function isEditableTarget(t: TargetInfo): boolean {
  return (
    t.isContentEditable ||
    (typeof t.tagName === 'string' && EDITABLE_TAGS.has(t.tagName.toUpperCase()))
  );
}

/**
 * Map one keydown to what the app intends to do with it.
 *
 * Order matters and is asserted in the test: the editable guard beats
 * everything, then the explorer's claims (live only while it is active; the
 * transport keys stay live in both modes), then the transport and view
 * toggles. `preventDefault` is part of the plan, not a side effect — the
 * handler calls it, so a test can pin that every claimed key suppresses the
 * default while an unclaimed key is left to the browser.
 */
export function planKey(
  code: string | null,
  shiftKey: boolean,
  explorerActive: boolean,
  target: TargetInfo,
): KeyPlan {
  if (code === null || isEditableTarget(target)) return none;

  if (explorerActive) {
    if (code === 'KeyR') return { kind: 'explorer-reroll', preventDefault: true };
    if (code === 'Backspace') return { kind: 'explorer-back', preventDefault: true };
    if (code === 'Escape') return { kind: 'explorer-exit', preventDefault: true };
  }

  switch (code) {
    case 'KeyH':
      return { kind: 'halt', preventDefault: true };
    case 'Space':
      return { kind: 'play-toggle', preventDefault: true };
    case 'ArrowLeft':
      return { kind: 'seek', delta: shiftKey ? -10 : -2, preventDefault: true };
    case 'ArrowRight':
      return { kind: 'seek', delta: shiftKey ? 10 : 2, preventDefault: true };
    default:
      break;
  }

  // The explorer is a locked-in GUI — there is nothing to toggle. The
  // transport keys above stay live (the music still drives the nine tiles),
  // but the view toggles are meaningless over a full-screen grid.
  if (!explorerActive) {
    if (code === 'KeyS') return { kind: 'toggle-panel', preventDefault: true };
    if (code === 'KeyT') return { kind: 'toggle-timeline', preventDefault: true };
  }

  return none;
}
