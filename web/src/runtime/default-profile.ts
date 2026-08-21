/**
 * The look a first-time visitor arrives to.
 *
 * `default-profile.json` is a whole `ExportRecipe`, exported from the workbench
 * by the author — the same artefact the profile shelf saves, not a second
 * format. That is deliberate for the reason save profiles v0 already settled:
 * a plife matrix is generated from the **seed**, and the authored θ centre is in
 * no mapping file, so anything less than a whole recipe restores the colours
 * around a different world.
 *
 * It rides the *existing* boot path rather than adding one. `main.ts` already
 * knows how to start from a recipe — that is what a staged profile and a `#p=`
 * preset both do — so this is simply the lowest-precedence source of the same
 * value, below every form of state a visitor could have of their own.
 *
 * ## It is parsed, not trusted
 *
 * Through `parseExportRecipe`, the same validator every other recipe goes
 * through, including the version lifts. A default that has gone stale against
 * the schema therefore fails loudly and identically to any other bad recipe
 * instead of half-applying — and `tests/default-profile.test.ts` runs the same
 * parse, so it fails in CI rather than in front of a visitor.
 *
 * Imported `?raw` rather than as JSON so those bytes are what gets validated. An
 * object import would hand the validator a value the bundler had already parsed
 * and re-serialised, which is a different thing from the file on disk.
 */
import defaultProfileJson from './default-profile.json?raw';
import { parseExportRecipe, type ExportRecipe } from './recipe';

/**
 * Parsed once, at module load. Eager on purpose: if the shipped default is
 * malformed, the failure belongs at startup next to the code that owns it, not
 * at the first moment a visitor happens to need it.
 */
export const DEFAULT_PROFILE: ExportRecipe = parseExportRecipe(defaultProfileJson);
