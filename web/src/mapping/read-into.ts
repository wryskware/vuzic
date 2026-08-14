/**
 * One generic reader for every "load a saved settings blob into a live object"
 * path in the app.
 *
 * ## Why this exists
 *
 * Persistence here is asymmetric by construction. The **write** side is total —
 * `JSON.stringify(config)` emits every field there is, and `serializeExtras`
 * spreads whole blocks — while the **read** side used to be a hand-written
 * allowlist: `parseModulation` rebuilt `ModulationConfig` field by field, and
 * every sim's `applyExtras` read its blocks field by field. A field added to the
 * config and not added to *both* readers was therefore written to localStorage
 * and silently dropped on the way back in, which surfaces later as "my tweak
 * didn't save" with nothing in the console and nothing failing in CI.
 *
 * That was not one bug, it was a shape: `Palette.accentArc` (the reported
 * instance) is the same defect as the population lane's fields, the budget
 * block's, and every field a future feature adds.
 *
 * The fix is to make the reader walk the **destination** instead of a written
 * list. `readInto` iterates the keys of the defaults object it is handed and
 * copies whatever the raw blob has for each one, so the set of fields that round
 * trip is exactly the set of fields the defaults function declares. TypeScript
 * already forces a defaults function to return every field of its interface, so
 * adding a field to the interface forces it into the defaults, and being in the
 * defaults is *sufficient* to be persisted. There is no third list to forget.
 *
 * ## What it deliberately does not handle
 *
 * Arrays. Every array in this codebase has semantics a generic walker cannot
 * know — `colors` is per species and re-derived in arc mode, `driverGains` is
 * the track's driver count, `speciesEnabled` is padded against the live K, and
 * θ's matrices are K·K. Those keep their bespoke readers; this handles the
 * scalar/nested-object mass, which is where the forgetting happened.
 *
 * ## Contract
 *
 * - **Never throws.** These blobs come from localStorage and hand-edited files,
 *   and a corrupt one must cost you the field, not the session. Anything absent,
 *   mistyped or non-finite leaves the default in place.
 * - **Writes in place.** The panel's tweakpane bindings hold sub-objects like
 *   `config.population` and `config.budget` by reference, so replacing one would
 *   leave the panel editing something nothing runs. Recursion mutates; it never
 *   substitutes.
 * - **Only keys the destination already has** are ever written, so a hostile blob
 *   cannot inject `__proto__`, a new key, or a key of the wrong type.
 */

/** Tag distinguishing a leaf rule from a nested rule tree at runtime. */
const LEAF = Symbol('read-into.leaf');

export interface NumberRule {
  readonly [LEAF]: true;
  readonly min: number;
  readonly max: number;
  /** round after clamping — for fields the consumer indexes or allocates with */
  readonly int?: boolean;
}

export interface StringRule {
  readonly [LEAF]: true;
  readonly oneOf: readonly string[];
}

export type Rule = NumberRule | StringRule | RuleTree;

/** Rules mirror the shape of the object being read, with leaves at the leaves. */
export interface RuleTree {
  readonly [key: string]: Rule | undefined;
}

/**
 * Clamp bound for a numeric field. Call sites pass the *same* table the panel
 * builds its slider from wherever one exists (`MACRO_RANGE`, `FAR_GAIN_RANGE`,
 * …), which is the property worth having: a loaded value can never sit outside
 * the widget that is supposed to show it.
 */
export function range(min: number, max: number, opts?: { int?: boolean }): NumberRule {
  return opts?.int === true ? { [LEAF]: true, min, max, int: true } : { [LEAF]: true, min, max };
}

/** Legal values for a string field; anything else keeps the default. */
export function oneOf(values: readonly string[]): StringRule {
  return { [LEAF]: true, oneOf: values };
}

function isLeaf(rule: Rule | undefined): rule is NumberRule | StringRule {
  return rule !== undefined && LEAF in rule;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function source(raw: unknown): Record<string, unknown> {
  return isPlainObject(raw) ? raw : {};
}

/**
 * Copy `raw` into `dst`, field by field, driven by `dst`'s own keys.
 *
 * `dst` is either a freshly built defaults object (parsing a file into a new
 * config) or the live config itself (a sim's `applyExtras`). Both work, and the
 * second is why this mutates: see the in-place note above.
 */
export function readInto<T extends object>(dst: T, raw: unknown, rules: RuleTree = {}): T {
  const src = source(raw);
  const target = dst as Record<string, unknown>;
  for (const key of Object.keys(target)) {
    const current = target[key];
    const rule = rules[key];

    // Nested block. Recursed unconditionally — an absent sub-object simply leaves
    // every default in place, which is one branch fewer than checking for it.
    if (isPlainObject(current)) {
      readInto(current, src[key], isLeaf(rule) ? {} : (rule ?? {}));
      continue;
    }

    const value = src[key];
    if (value === undefined) continue;

    if (typeof current === 'number') {
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      const r = isLeaf(rule) && 'min' in rule ? rule : null;
      if (!r) {
        target[key] = value;
        continue;
      }
      const clamped = Math.min(Math.max(value, r.min), r.max);
      target[key] = r.int === true ? Math.round(clamped) : clamped;
      continue;
    }

    if (typeof current === 'boolean') {
      // Strictly a boolean, rather than the older `!== false` idiom: a file with
      // a string or a number there is corrupt, and the default is a better answer
      // than coercing whatever it happens to be truthy as.
      if (typeof value === 'boolean') target[key] = value;
      continue;
    }

    if (typeof current === 'string') {
      if (typeof value !== 'string') continue;
      if (isLeaf(rule) && 'oneOf' in rule && !rule.oneOf.includes(value)) continue;
      target[key] = value;
      continue;
    }

    // Arrays, null, functions: not this function's job. See the header.
  }
  return dst;
}
