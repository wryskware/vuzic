/**
 * The block registry — the rung above `readInto`, and the last one of the
 * "my tweak didn't save" class.
 *
 * ## What was still broken
 *
 * `readInto` closed the **field** rung: a field round trips because the reader
 * walks the destination's own keys, so being in a defaults function is
 * *sufficient* to be persisted and there is no second list to forget.
 * `persisting()` closed the **widget** rung: a panel can only be handed a
 * `PersistedContainer`, so a binding that never reached the save path became a
 * type error rather than a bug report.
 *
 * One rung was still open, and it is the same defect one level up. A whole new
 * config *block* — the macro rig, the population lane, the impulse folder's
 * settings, the next one — persisted only if somebody remembered to name it in
 * the sim's `extrasBlocks()`, *and* its defaults in a defaults table, *and* its
 * clamps in a rules table. Nothing failed when they did not; the block simply
 * never saved, and the report arrived weeks later. That bug has been paid for
 * four times (the accents' arc, the whole impulse lane, the palette and render
 * folders' optional `onChange`, and block registration itself).
 *
 * ## The fix: the config *is* the registry
 *
 * The same trick, applied one level up. `readInto` walks the destination's keys;
 * this walks the **config's** keys. Every key of a sim's config whose value is a
 * plain object is a *block*, and a `BlockTable<Config>` is an exhaustive record
 * over exactly those keys:
 *
 * ```ts
 * export const PLIFE_BLOCKS: BlockTable<PlifeConfig> = { … };
 * ```
 *
 * TypeScript already forces a defaults function to return every field of its
 * interface. It equally forces this table to have an entry for every block of
 * its config — so adding `foo: FooConfig` to `PlifeConfig` is a **compile
 * error** until `foo` is declared, and declaring it as `persisted(defaultFoo)`
 * is the whole of the work: serialize, restore and clamp all follow from the one
 * declaration. `undeclaredBlocks` is the runtime belt behind that brace, for the
 * CI check that the table and the live config still agree.
 *
 * ## Opting out
 *
 * A block that genuinely should not ride the save channel says so **where it is
 * declared**, with a reason, and shows up in `unpersistedBlocks()` — which is
 * what a test asserts against. Two kinds, because they are two different
 * statements and lumping them together would make one of them a lie:
 *
 *   `sessionOnly(why)`         — this state is about *this run* and restoring it
 *                                would be wrong (a measured frame-rate budget, a
 *                                debug view left switched on).
 *   `persistedElsewhere(why)`  — this state is saved, just not through this
 *                                channel (the palette and the render chain are
 *                                one live object shared with `ModulationConfig`,
 *                                and a copy in `extras` would be a second
 *                                encoding for the loader to disagree with).
 *
 * The same override exists one level down, per field, for the rare block that is
 * persisted except for one transient switch: `sessionOnlyFields` on the
 * declaration. It is deliberately awkward to spell and it appears in
 * `sessionOnlyFields()` for the same exhaustive assertion.
 *
 * ## What it does not do
 *
 * Non-block state — scalars on the config root, arrays, anything living on the
 * sim rather than on the config — is not in this mechanism's domain, and each
 * sim still handles those itself (plife's `look` and `speciesEnabled`,
 * physarum's `stemDrive`/`stemGain`, vizfx's `emittersPerLayer`). They are a
 * handful, they are named in one place per sim, and a *scalar* added to the
 * config root is a field of no block — which the field rung, not this one, would
 * have to answer. Blocks are where the forgetting happened.
 */
import { readInto, type RuleTree } from './read-into.ts';

/** A rules table, or a function of the live config for the ones with live bounds. */
export type BlockRules<C> = RuleTree | ((cfg: C) => RuleTree);

export interface PersistedBlock<C> {
  readonly kind: 'persisted';
  /** the shipped values; its keys are this block's persisted schema (see `readInto`) */
  readonly defaults: (cfg: C) => object;
  /** bounds only — a field with no rule still round trips */
  readonly rules: BlockRules<C>;
  /** runs over the raw blob before the walk, for a block that changed shape */
  readonly migrate: ((raw: Record<string, unknown>) => Record<string, unknown>) | null;
  /** field → why this one field of an otherwise persisted block is session-only */
  readonly sessionOnlyFields: Readonly<Record<string, string>>;
}

export interface UnpersistedBlock {
  readonly kind: 'session-only' | 'persisted-elsewhere';
  readonly reason: string;
}

export type BlockDecl<C> = PersistedBlock<C> | UnpersistedBlock;

export interface PersistOptions<C> {
  rules?: BlockRules<C>;
  migrate?: (raw: Record<string, unknown>) => Record<string, unknown>;
  sessionOnlyFields?: Readonly<Record<string, string>>;
}

/**
 * Declare a block as saved. One call is the whole registration: `serializeBlocks`
 * snapshots it, `applyBlocks` resets it to `defaults` and reads the blob back
 * into it under `rules`.
 */
export function persisted<C>(
  defaults: (cfg: C) => object,
  opts: PersistOptions<C> = {},
): PersistedBlock<C> {
  return {
    kind: 'persisted',
    defaults,
    rules: opts.rules ?? {},
    migrate: opts.migrate ?? null,
    sessionOnlyFields: opts.sessionOnlyFields ?? {},
  };
}

/** Declare a block as belonging to this run and not to the file. State why. */
export function sessionOnly(reason: string): UnpersistedBlock {
  return { kind: 'session-only', reason };
}

/** Declare a block as saved through another path. State which, and why. */
export function persistedElsewhere(reason: string): UnpersistedBlock {
  return { kind: 'persisted-elsewhere', reason };
}

/**
 * Arrays are excluded because every array in these configs has semantics a
 * generic walker cannot know — the same line `readInto` draws, for the same
 * reason. They are handled by their sim's own reader.
 */
type IsBlock<T> = [T] extends [readonly unknown[]] ? false : [T] extends [object] ? true : false;

/** The keys of `C` that are config blocks. This is the set a table must cover. */
export type BlockKeys<C> = {
  [K in keyof C]-?: IsBlock<NonNullable<C[K]>> extends true ? K : never;
}[keyof C];

/**
 * Exhaustive over `BlockKeys<C>`: a block with no entry is a compile error, and
 * an entry naming a key that is not a block is one too.
 */
export type BlockTable<C> = { readonly [K in BlockKeys<C>]: BlockDecl<C> };

function entries<C>(table: BlockTable<C>): [string, BlockDecl<C>][] {
  return Object.entries(table as Record<string, BlockDecl<C>>);
}

/** Same test the type makes, at runtime: a plain object, and not an array. */
function isBlock(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function resolve<C>(rules: BlockRules<C>, cfg: C): RuleTree {
  return typeof rules === 'function' ? rules(cfg) : rules;
}

/**
 * Snapshot every persisted block, by name, deep-cloned.
 *
 * Cloned rather than handed out by reference because the result is a *file*: the
 * explorer fans one of these out to nine tiles, and a tile that then edited a
 * shared sub-object would be editing the source world.
 */
export function serializeBlocks<C extends object>(
  cfg: C,
  table: BlockTable<C>,
): Record<string, unknown> {
  const src = cfg as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [name, decl] of entries(table)) {
    if (decl.kind !== 'persisted') continue;
    const live = src[name];
    // Only if the live config actually has the block. A table entry for a key the
    // config does not carry is caught by the type; this is the total-function
    // half, because `serializeExtras` runs on the autosave clock.
    if (!isBlock(live)) continue;
    const clone = structuredClone(live) as Record<string, unknown>;
    for (const field of Object.keys(decl.sessionOnlyFields)) delete clone[field];
    out[name] = clone;
  }
  return out;
}

/**
 * The inverse, **in place** and total.
 *
 * In place because the panel's tweakpane bindings hold `config.population` and
 * `config.matrixGen.rMin` by reference, so replacing an object would leave the
 * panel editing something nothing runs. Total because `extras` is opaque to every
 * layer between the file and here: a corrupt blob costs the fields it broke, not
 * the load.
 *
 * Each block is reset to its shipped defaults *first*, so a partial saved block
 * does not leave the untouched half at whatever the previous load — or the
 * explorer's last style sync, nine times a second — happened to put there.
 * Session-only fields are lifted over both walks, so neither the reset nor the
 * read can move them.
 */
export function applyBlocks<C extends object>(cfg: C, table: BlockTable<C>, raw: unknown): void {
  const src = isBlock(raw) ? raw : {};
  const dst = cfg as Record<string, unknown>;
  for (const [name, decl] of entries(table)) {
    if (decl.kind !== 'persisted') continue;
    const live = dst[name];
    if (!isBlock(live)) continue;

    const held = Object.keys(decl.sessionOnlyFields).map((f) => [f, live[f]] as const);
    readInto(live, decl.defaults(cfg));
    const blob = decl.migrate ? decl.migrate(isBlock(src[name]) ? src[name] : {}) : src[name];
    readInto(live, blob, resolve(decl.rules, cfg));
    for (const [field, value] of held) live[field] = value;
  }
}

/** The persisted declarations, in declaration order. */
export function persistedBlockDecls<C>(table: BlockTable<C>): [string, PersistedBlock<C>][] {
  return entries(table).filter((e): e is [string, PersistedBlock<C>] => e[1].kind === 'persisted');
}

/** The blocks that ride the channel, in declaration order — i.e. the saved keys. */
export function persistedBlocks<C>(table: BlockTable<C>): string[] {
  return persistedBlockDecls(table).map(([name]) => name);
}

/**
 * A declaration's rules against a live config — the very table `applyBlocks`
 * clamps with, exposed so a test can perturb a block to legal-but-not-default
 * values without restating any bound.
 */
export function blockRules<C>(decl: PersistedBlock<C>, cfg: C): RuleTree {
  return resolve(decl.rules, cfg);
}

export interface UnpersistedEntry {
  block: string;
  kind: 'session-only' | 'persisted-elsewhere';
  reason: string;
}

/**
 * Every block that opted out, with the reason given at its declaration site.
 *
 * This is what a test asserts against, exhaustively. An opt-out added without a
 * matching change to that assertion fails CI, which is the whole point: "not
 * saved" has to be a decision somebody made and somebody else reviewed, never a
 * name that was quietly never typed.
 */
export function unpersistedBlocks<C>(table: BlockTable<C>): UnpersistedEntry[] {
  return entries(table)
    .filter((e): e is [string, UnpersistedBlock] => e[1].kind !== 'persisted')
    .map(([block, decl]) => ({ block, kind: decl.kind, reason: decl.reason }))
    .sort((a, b) => a.block.localeCompare(b.block));
}

export interface SessionOnlyField {
  block: string;
  field: string;
  reason: string;
}

/** The field-level overrides, for the same exhaustive assertion one level down. */
export function sessionOnlyFields<C>(table: BlockTable<C>): SessionOnlyField[] {
  const out: SessionOnlyField[] = [];
  for (const [block, decl] of entries(table)) {
    if (decl.kind !== 'persisted') continue;
    for (const [field, reason] of Object.entries(decl.sessionOnlyFields)) {
      out.push({ block, field, reason });
    }
  }
  return out.sort((a, b) => a.block.localeCompare(b.block) || a.field.localeCompare(b.field));
}

/**
 * Blocks the live config carries that the table never heard of — the runtime belt
 * behind the compiler's brace.
 *
 * `BlockTable<C>` already makes an undeclared block a type error, which is the
 * real guard. This exists because a test can *run* it: `assert.deepEqual(
 * undeclaredBlocks(defaultPlifeConfig(), PLIFE_BLOCKS), [])` fails loudly if the
 * two ever drift — through a cast, a widened type, or a config assembled
 * somewhere the table never saw — and a loud failure is the one thing this bug
 * class has never had.
 */
export function undeclaredBlocks<C extends object>(cfg: C, table: BlockTable<C>): string[] {
  const declared = new Set(Object.keys(table as Record<string, unknown>));
  const src = cfg as Record<string, unknown>;
  return Object.keys(src).filter((key) => isBlock(src[key]) && !declared.has(key));
}
