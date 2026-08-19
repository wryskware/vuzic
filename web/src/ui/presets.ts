/**
 * The browser half of preset strings: the clipboard, the `#p=` fragment, and
 * the one-shot slot that carries a preset across the reload that applies it.
 *
 * `runtime/preset.ts` owns the format and knows nothing about a browser. This
 * module is where a token meets `localStorage`, `location` and the clipboard,
 * and it is deliberately thin: everything that can be decided without a DOM is
 * decided over there, where it can be tested.
 *
 * ## Why the slot, and why it is not the profile library's
 *
 * Applying in place would mean rebuilding the simulation, the impulse engine and
 * the modulator under a running frame loop — the sequence `main.ts` performs
 * exactly once at startup, in an order its own comments call load-bearing.
 * Reloading runs that sequence instead of duplicating it (`ui/profiles.ts` says
 * the same thing at more length; this is the same decision, not a second one).
 *
 * The slot is its own key rather than `lmt.pendingProfile` for one reason: a
 * preset carries an **advisory track hint** and a profile does not. Loading a
 * profile has never switched tracks, and folding presets into that slot would
 * either silently give profiles that behaviour or silently deny it to presets.
 * Two keys, two meanings, and the boot path consumes both unconditionally.
 */
import {
  chooseBootPreset,
  decodePreset,
  encodePreset,
  extractPresetToken,
  presetApplyPlan,
  presetFromRecipe,
  presetLink,
  presetTokenFromHash,
  recipeFromPreset,
  type PresetApplyPlan,
} from '../runtime/preset.ts';
import type { ExportRecipe } from '../runtime/recipe.ts';

/**
 * Armed by "load", read once at boot. Outside the `lmt.profile.` namespace so a
 * profile prefix scan never lists it, and outside `lmt.mapping` so the autosave
 * never collides with it.
 */
const PENDING_KEY = 'lmt.pendingPreset';

/** The live look as a preset string. */
export async function presetStringFromRecipe(recipe: ExportRecipe): Promise<string> {
  return encodePreset(presetFromRecipe(recipe));
}

/** `origin + pathname` and the token — never the query, never a stale fragment. */
export function presetLinkFor(token: string): string {
  return presetLink(`${location.origin}${location.pathname}`, token);
}

/**
 * Arm an apply, to be consumed by the next boot.
 *
 * The token is **decoded before it is armed**, so a corrupt string fails in the
 * panel where a person can read the message rather than one reload later, on a
 * screen that no longer has a panel to say it on.
 */
export async function requestPresetApply(token: string): Promise<string | null> {
  try {
    await decodePreset(token);
  } catch (error) {
    return (error as Error).message;
  }
  try {
    localStorage.setItem(PENDING_KEY, token);
    return null;
  } catch {
    return 'could not stage the preset for loading';
  }
}

/**
 * Read and **delete** the pending token, whatever happens next.
 *
 * Consumed once, unconditionally — including when it goes on to fail. A slot
 * that survived its own failure would re-apply on every subsequent reload,
 * which turns one bad preset into an app that cannot be started normally again
 * without opening devtools.
 */
export function takePendingPresetToken(): string | null {
  try {
    const token = localStorage.getItem(PENDING_KEY);
    localStorage.removeItem(PENDING_KEY);
    return token;
  } catch {
    return null;
  }
}

export function readFragmentToken(): string | null {
  try {
    return presetTokenFromHash(location.hash);
  } catch {
    return null;
  }
}

/**
 * Drop the fragment once it has been applied, so a later reload uses the
 * autosave rather than re-applying the link over a session of tweaks.
 *
 * `replaceState`, not a `location.hash = ''` assignment: the latter leaves a
 * bare `#` and pushes a history entry, so back would walk through applies.
 */
export function clearFragment(): void {
  try {
    history.replaceState(history.state, '', `${location.pathname}${location.search}`);
  } catch {
    // no History API / opaque origin — a stale fragment is cosmetic here
  }
}

export interface BootPreset {
  recipe: ExportRecipe;
  plan: PresetApplyPlan;
  source: 'pending' | 'fragment';
}

/**
 * The staged or linked preset for this boot, if any.
 *
 * Async because inflating is: `DecompressionStream` has no synchronous form.
 * That is why this is called at the very top of `main()`, before anything is
 * constructed — it is one `await` in a function that already has several, and
 * everything downstream is built from what it returns.
 */
export async function consumeBootPreset(): Promise<{
  preset: BootPreset | null;
  error: string | null;
}> {
  // Consumed first and unconditionally: see `takePendingPresetToken`.
  const chosen = chooseBootPreset(takePendingPresetToken(), readFragmentToken());
  if (chosen.token === null) return { preset: null, error: null };
  try {
    const preset = await decodePreset(chosen.token);
    const boot: BootPreset = {
      recipe: recipeFromPreset(preset),
      plan: presetApplyPlan(preset),
      source: chosen.source,
    };
    // Only on success. A fragment that failed stays in the address bar, where
    // its owner can still see and copy the string that did not work.
    if (chosen.source === 'fragment') clearFragment();
    return { preset: boot, error: null };
  } catch (error) {
    return { preset: null, error: (error as Error).message };
  }
}

/** Accepts a bare token or a whole pasted link. */
export { extractPresetToken };
