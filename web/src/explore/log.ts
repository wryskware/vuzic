/**
 * Explorer verdict log — every deliberate act in explorer mode, as preference
 * data.
 *
 * `mapping/tuninglog.ts` records what a human *kept*. This records what a human
 * *chose*, which is strictly more informative: a pick is one θ preferred over
 * eight named alternatives, and eight rejections are worth more to a ranking
 * loss than one acceptance is to a regression. That is the whole reason
 * explorer mode is logged separately rather than as more `TuningEntry` rows —
 * the entries answer a different question and a training loader should not have
 * to guess which kind a row is.
 *
 * The eight rejected θs are **not** stored, and that is the design, not a
 * shortcut. `genSeed` plus the centre θ plus (subspace, step) reproduces all
 * nine vectors exactly — `ExplorerSearch` is a pure function of them — so
 * storing the losers would be storing 8× the bytes for 0× the information, in a
 * 5 MB budget that has to survive 200 entries × ~100 floats. Same argument that
 * removed `z` from the tuning log: record what determines the sample, not the
 * sample.
 *
 * `theta` *is* stored even though it is technically re-derivable from the chain
 * of picks, because that derivation needs every earlier row to be intact and one
 * evicted row would silently corrupt everything after it. The centre is the
 * expensive thing to be wrong about.
 *
 * Exported as JSONL because that is what a training loader wants to stream.
 *
 * CONTRACT: exported names/shapes are the seam the 9-up rig consumes.
 */
import { EXPLORER_SUBSPACES, type ExplorerSubspace } from './search.ts';

/**
 * Versioned from birth, for the reason the tuning log had to learn the hard way:
 * the entry shape will change, and a mixed-schema export is discovered by the
 * consumer, which is a training run.
 */
const STORAGE_KEY = 'lmt.explorerLog.v1';

/**
 * Lower than the tuning log's 300 because an explorer entry is a fat one — a
 * full θ per row, ~100 floats for physarum and more for plife at higher K — and
 * because explorer acts arrive far faster than tuning acts do (a pick is one
 * click). 200 rows is several sessions of steering and comfortably inside a
 * localStorage origin budget.
 */
const MAX_ENTRIES = 200;

/**
 * Every deliberate act the mode records.
 *
 * `recenter` and `revert` arrived with the panel↔grid merge: the workbench is
 * live during exploration, so "the human moved the centre by hand" and "the human
 * threw the whole run away and went back to where they started" are now things
 * that happen, and both are preference data — a hand-aimed θ is a stated
 * preference, and a revert is a rejection of everything after it.
 *
 * `adopt` is kept but no longer written: exit no longer copies anything anywhere
 * (the centre has been live all along), so there is no adoption to record. It
 * stays in the union because logs from before the merge are still in
 * localStorage and still valid rows, and dropping it would make `isEntry` evict
 * a user's own history on the next read.
 */
export type ExplorerAction =
  | 'start'
  | 'pick'
  | 'reroll'
  | 'back'
  | 'like'
  | 'adopt'
  | 'recenter'
  | 'revert';

export interface ExplorerEntry {
  /** ISO timestamp */
  at: string;
  simId: string;
  action: ExplorerAction;
  generation: number;
  subspace: ExplorerSubspace;
  /** mutation scale in force */
  step: number;
  /** RNG key of the generation this act judged — reproduces all 9 θs */
  genSeed: number;
  /** which tile won, for 'pick' */
  pickedIndex?: number;
  /** the center θ AFTER the action, rounded, full registry length */
  theta: number[];
  note?: string;
}

/**
 * 5 decimals. Every slot in every shipped registry is an O(1)–O(100) quantity
 * with a `half` no finer than 1e-3, so the fifth decimal is already below the
 * resolution any of this is meaningful at — and the rounding roughly halves the
 * JSON, which is the binding constraint. Non-finite becomes 0 rather than
 * `null`: `JSON.stringify` turns NaN into `null`, and a null in a float column
 * breaks a loader in a way a 0 does not.
 */
function round(v: number): number {
  return Number.isFinite(v) ? Number(v.toFixed(5)) : 0;
}

export class ExplorerLog {
  private readonly simId: string;
  private items: ExplorerEntry[];
  /** cleared the first time storage proves unusable, so it warns once, not once per click */
  private persisting = true;

  constructor(simId: string) {
    this.simId = simId;
    this.items = readStored();
  }

  get size(): number {
    return this.items.length;
  }

  get entries(): readonly ExplorerEntry[] {
    return this.items;
  }

  /**
   * `at` and `simId` are stamped here rather than passed in, so no caller can
   * mislabel which substrate a θ belongs to — an explorer θ applied to the wrong
   * registry is nonsense, and the log is the one place that pairing is recorded.
   */
  append(entry: Omit<ExplorerEntry, 'at' | 'simId'>): ExplorerEntry {
    const full: ExplorerEntry = {
      ...entry,
      at: new Date().toISOString(),
      simId: this.simId,
      theta: entry.theta.map(round),
    };
    this.items.push(full);
    if (this.items.length > MAX_ENTRIES) this.items = this.items.slice(-MAX_ENTRIES);
    this.persist();
    // A copy, not the stored row — the same ownership contract as the search's
    // cloned CandidateSets: a caller annotating what it was handed must not be
    // editing the JSONL a training loader will stream later.
    return { ...full, theta: [...full.theta] };
  }

  exportJsonl(): string {
    return this.items.map((e) => JSON.stringify(e)).join('\n');
  }

  clear(): void {
    this.items = [];
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore — the in-memory log is already empty, which is what was asked for
    }
  }

  /**
   * Persistence is best-effort by design. Explorer mode is a live steering loop;
   * a full disk quota or a storage-disabled browser must cost the log, never the
   * click.
   *
   * The two failure modes are treated differently on purpose, because they are
   * not the same problem and the naive shared handler gets one of them badly
   * wrong. A **quota** failure means the history is too big, so half of it is
   * dropped (oldest first — the newest verdicts are the ones the human is
   * producing right now) and retried. **Unusable storage** — disabled, private
   * mode, no `localStorage` at all — means the history is fine and the sink is
   * not, so dropping entries would be destroying good data to appease a store
   * that is never going to accept it. The discrimination is the retry itself: if
   * even half fails to write, it was not about size. In that case nothing is
   * dropped, the entries stay in memory for the session, and we stop trying so
   * one broken store does not throw on every click for the rest of it.
   */
  private persist(): void {
    if (!this.persisting) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
      return;
    } catch {
      // fall through to the halving retry
    }
    const trimmed = this.items.slice(Math.floor(this.items.length / 2));
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
      this.items = trimmed;
    } catch (err) {
      this.persisting = false;
      console.warn('explorer log: localStorage unavailable, keeping entries in memory only', err);
    }
  }
}

const ACTIONS = new Set<string>([
  'start',
  'pick',
  'reroll',
  'back',
  'like',
  'adopt',
  'recenter',
  'revert',
]);
const SUBSPACES = new Set<string>(EXPLORER_SUBSPACES);

/**
 * Shape-checked, not blind-cast. The store is a string a user can hand-edit and
 * a future version can half-migrate; a row that fails the check is dropped
 * rather than allowed into `size`, into the UI, or — worst — into the JSONL a
 * training loader streams.
 */
function isEntry(x: unknown): x is ExplorerEntry {
  const e = x as Partial<ExplorerEntry> | null;
  return (
    !!e &&
    typeof e.at === 'string' &&
    typeof e.simId === 'string' &&
    ACTIONS.has(e.action as string) &&
    SUBSPACES.has(e.subspace as string) &&
    typeof e.generation === 'number' &&
    typeof e.step === 'number' &&
    typeof e.genSeed === 'number' &&
    Number.isFinite(e.genSeed) &&
    Array.isArray(e.theta)
  );
}

/**
 * Rows from *every* sim are kept, not just this instance's. The key is shared on
 * purpose: `simId` already discriminates, and a loader wants one file. Filtering
 * on read would also mean a physarum session silently evicting a plife session's
 * history on its next write.
 */
function readStored(): ExplorerEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry) : [];
  } catch {
    // Corrupt JSON, or no storage at all. Start empty; the log is a bonus.
    return [];
  }
}
