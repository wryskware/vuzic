/**
 * Named save profiles — the workbench's **hard** save location.
 *
 * ## Why this exists next to the autosave, rather than instead of it
 *
 * The mapping autosave is one key per sim (`lmt.mapping.<sim>`) written by a
 * debounce whenever a widget changes. That is the right shape for "do not lose
 * my session", and the wrong shape for "keep this look": every tab writes to the
 * same key, so a second tab — another window, or a browser-automation session —
 * overwrites a tuning session simply by existing. Removing the flush-on-hide
 * (see `./autosave.ts`) stops the worst of it, but the underlying property is
 * unchanged: an autosave slot is *whatever happened last*, everywhere.
 *
 * A profile is the opposite by construction. **Nothing in this module is ever
 * called except from a button.** No timer, no lifecycle hook, no visibility
 * listener. A background tab cannot reach a profile, so a saved look stays saved
 * until you overwrite or delete it by hand.
 *
 * ## Why a whole export recipe, and not a `modulation.json`
 *
 * A modulation file is depths, palette, grade and event responses — everything
 * *except* what actually determines a particle-life world's shape. The
 * interaction matrix is generated from the **seed**, and the authored θ centre
 * (`modulationBase`) lives in no mapping file at all. Save the mapping alone and
 * reload, and you get your colours back around a different matrix.
 *
 * `ExportRecipe` already carries all of it — seed, pin state, the full
 * simulation config, θ, impulses, render — and already has a strict parser
 * (`parseExportRecipe`) and a canonical rehydrator (`buildSimBundleFromRecipe`).
 * So a profile *is* a recipe, verbatim, with the two export-only fields stubbed:
 * `rendererBuild` (a server capability that a static build has no access to) and
 * `output` (resolution and encoder, meaningless for a look). Both are ignored on
 * load. This costs one small wart and buys the entire format, validator and
 * apply path for free — and it is deliberately a stepping stone: preset strings
 * v1 (`docs/handoffs/preset-strings-v1.md`) trims those fields into `PresetV1`
 * and adds the codec, and a v0 profile lifts into one by deleting four keys.
 *
 * ## Storage layout: one key per profile, and no index
 *
 * `lmt.profile.<simId>.<encoded name>`, enumerated by prefix scan. The obvious
 * alternative — a single `lmt.profiles` array — is a read-modify-write, which is
 * exactly the multi-tab clobber this module exists to escape: two tabs saving
 * two profiles would leave one of them. Per-key writes have no such race, a
 * corrupt entry costs one profile rather than the library, and delete is a
 * `removeItem`. Listing is always scoped to one sim, so the remainder after the
 * prefix is unambiguously the name however many dots it contains.
 */
import { debugExportOutput, DEFAULT_EXPORT_PROFILE } from './export-panel.ts';
import { parseExportRecipe, serializeExportRecipe, type ExportRecipe } from '../runtime/recipe.ts';

/** Namespace for the library. Everything under it is written only by a button. */
const PREFIX = 'lmt.profile.';

/**
 * The hand-off slot for an apply. Written by "load", read once at boot.
 *
 * Deliberately outside `PREFIX` so a prefix scan never lists it as a profile.
 */
const PENDING_KEY = 'lmt.pendingProfile';

/**
 * What `rendererBuild` says in a profile. It is a required string on the recipe
 * and a profile has no renderer to name, so it says what it is rather than
 * borrowing a plausible-looking build id that would then travel into an export.
 */
export const PROFILE_RENDERER_BUILD = 'profile (not an export capture)';

/**
 * A profile's stub `output`. Reused from the export panel rather than restated,
 * so it stays a valid `ExportProfile`/`ExportEncoder` pair as those evolve. On
 * load it is dropped; on an export it would be replaced by the real capability.
 */
export function profileOutput(): ExportRecipe['output'] {
  return debugExportOutput(DEFAULT_EXPORT_PROFILE);
}

function keyFor(sim: string, name: string): string {
  return `${PREFIX}${sim}.${encodeURIComponent(name)}`;
}

/**
 * Trim and reject empty. Nothing else is policed: the name is encoded into the
 * key, so any character survives a round trip, and a person naming a profile
 * "chorus — hot / v2" should get that name back.
 */
export function normalizeProfileName(raw: string): string | null {
  const name = raw.trim();
  return name === '' ? null : name;
}

/** Saved profile names for one sim, alphabetical. Never throws. */
export function listProfiles(sim: string): string[] {
  const prefix = `${PREFIX}${sim}.`;
  const names: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key === null || !key.startsWith(prefix)) continue;
      try {
        names.push(decodeURIComponent(key.slice(prefix.length)));
      } catch {
        // A key we did not write, or one mangled by hand. Skipping it keeps the
        // rest of the library listable, which is the whole point of per-key
        // storage.
      }
    }
  } catch {
    // Storage disabled entirely: an empty library, not a broken panel.
    return [];
  }
  return names.sort((a, b) => a.localeCompare(b));
}

export interface SaveResult {
  /** Which sim's shelf it landed on — the recipe's own, never the live one. */
  sim: string | null;
  /** Panel status line. */
  message: string;
}

/**
 * Write one profile from recipe text.
 *
 * The text is **parsed before it is stored**, and the shelf it lands on is the
 * parsed recipe's own `sim`. Two properties fall out of that, and both are worth
 * the redundant parse on the save-current path: a profile that could not be
 * loaded can never be saved in the first place, and an imported file goes where
 * it will actually work rather than onto whichever sim happened to be running.
 *
 * Returns a message rather than a boolean because the failures a person can act
 * on differ: an invalid file, storage being off, and the origin's ~5 MB budget
 * being full (delete a profile, or a track from the offline cache).
 */
export function saveProfileText(name: string, text: string): SaveResult {
  let recipe: ExportRecipe;
  try {
    recipe = parseExportRecipe(text);
  } catch (err) {
    return { sim: null, message: `not a profile: ${(err as Error).message}` };
  }
  try {
    localStorage.setItem(keyFor(recipe.sim, name), text);
    return { sim: recipe.sim, message: `saved "${name}"` };
  } catch (err) {
    const quota = err instanceof DOMException && err.name.includes('Quota');
    return {
      sim: recipe.sim,
      message: quota ? `no room for "${name}" — delete a profile` : `could not save "${name}"`,
    };
  }
}

/** Save the live session. See `saveProfileText` for why this round trips. */
export function saveProfile(name: string, recipe: ExportRecipe): SaveResult {
  return saveProfileText(name, serializeExportRecipe(recipe));
}

/** The stored text, unparsed — what "export to file" downloads. */
export function readProfileText(sim: string, name: string): string | null {
  try {
    return localStorage.getItem(keyFor(sim, name));
  } catch {
    return null;
  }
}

export function deleteProfile(sim: string, name: string): void {
  try {
    localStorage.removeItem(keyFor(sim, name));
  } catch {
    // nothing to do
  }
}

/**
 * Arm an apply, to be consumed by the next boot.
 *
 * Applying a profile in place would mean rebuilding the simulation, the impulse
 * engine and the modulator under a running frame loop, restoring θ, and putting
 * the seed back — the same sequence `main.ts` already performs exactly once, at
 * startup, in an order its own comments call load-bearing. Reloading runs that
 * sequence instead of duplicating it. It costs a flash of black and it cannot
 * half-apply.
 *
 * The text is validated *before* it is armed, so a corrupt profile fails in the
 * panel where the person can read it, rather than one reload later.
 */
export function requestProfileApply(text: string): string | null {
  try {
    parseExportRecipe(text);
  } catch (err) {
    return (err as Error).message;
  }
  try {
    localStorage.setItem(PENDING_KEY, text);
    return null;
  } catch {
    return 'could not stage the profile for loading';
  }
}

/**
 * Read and **delete** the pending profile, whatever happens next.
 *
 * Consumed once, unconditionally — including when it fails to parse. A slot that
 * survived its own failure would re-apply on every subsequent reload, which
 * turns one bad profile into an app that cannot be started normally again
 * without opening devtools.
 */
export function consumePendingProfile(): ExportRecipe | null {
  let text: string | null = null;
  try {
    text = localStorage.getItem(PENDING_KEY);
    localStorage.removeItem(PENDING_KEY);
  } catch {
    return null;
  }
  if (text === null) return null;
  try {
    return parseExportRecipe(text);
  } catch (err) {
    console.warn('profile: ignoring an unusable pending profile —', err);
    return null;
  }
}
