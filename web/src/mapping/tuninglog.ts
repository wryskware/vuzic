/**
 * Append-only tuning log — the future NN's preference data.
 *
 * plan.md's "Later" is a distilled MLP trained on what a human kept. In the
 * anchor era that meant (latent, θ, kept/discarded) triples from hand-tuned
 * presets. Revision 3 deleted the tuning workload, so what a human now expresses
 * a preference over is a **seed**: reroll until it looks right, then pin it. The
 * log therefore records seeds and verdicts, one entry per deliberate act
 * (reroll, pin, save, keep, discard, snapshot, restore), never per frame.
 *
 * `z` is deliberately absent. It used to hold the 64-dim latent at the moment of
 * the entry; the input is 1024-dim now, and 300 entries x 1024 floats does not
 * fit in localStorage. Nothing is lost: (track, seed, time) determines ẑ exactly,
 * and all three are recorded.
 *
 * Exported as JSONL because that is what a training loader wants to stream.
 */

/**
 * Versioned, because the entry shape changed with Revision 3: `z`, `anchor` and
 * `anchorName` are gone, `seed`/`source` are new, and the action union lost
 * 'capture'. A pre-Revision-3 log is a different schema, and reading it under the
 * old key would silently mix both in `size` and in the JSONL export — the
 * consumer of that export is a training loader, which is the last place a
 * `seed: undefined` row should turn up.
 */
const STORAGE_KEY = 'lmt.tuningLog.v3';
const LEGACY_KEYS = ['lmt.tuningLog'];
const MAX_ENTRIES = 300;

export type TuningAction =
  | 'reroll'
  | 'pin'
  | 'unpin'
  | 'save'
  | 'keep'
  | 'discard'
  | 'snapshot'
  | 'restore';

export interface TuningEntry {
  /** ISO timestamp */
  at: string;
  /** sim tick the sample was taken at */
  tick: number;
  /** track seconds */
  time: number;
  action: TuningAction;
  /** the world seed in force — wiring, personality and agent placement, all three */
  seed: number;
  /** which input drove the modulator, e.g. "embedding-1024" */
  source: string;
  /** the full parameter vector, in mapping/preset.ts's slot order */
  theta: number[];
  note?: string;
}

function round(v: number): number {
  return Number.isFinite(v) ? Number(v.toFixed(5)) : 0;
}

export class TuningLog {
  private items: TuningEntry[];

  constructor() {
    this.items = readStored();
  }

  get size(): number {
    return this.items.length;
  }

  get entries(): readonly TuningEntry[] {
    return this.items;
  }

  append(entry: Omit<TuningEntry, 'at'>): TuningEntry {
    const full: TuningEntry = {
      ...entry,
      at: new Date().toISOString(),
      theta: entry.theta.map(round),
    };
    this.items.push(full);
    if (this.items.length > MAX_ENTRIES) this.items = this.items.slice(-MAX_ENTRIES);
    this.persist();
    return full;
  }

  clear(): void {
    this.items = [];
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }

  toJsonl(): string {
    return this.items.map((e) => JSON.stringify(e)).join('\n');
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
    } catch {
      // Quota: drop the oldest half rather than losing the newest work.
      this.items = this.items.slice(Math.floor(this.items.length / 2));
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
      } catch {
        // give up silently; the log is a bonus, not the product
      }
    }
  }
}

const ACTIONS = new Set<string>([
  'reroll',
  'pin',
  'unpin',
  'save',
  'keep',
  'discard',
  'snapshot',
  'restore',
]);

/** Shape-checked, not blind-cast: hand-edited or half-migrated rows drop out. */
function isEntry(x: unknown): x is TuningEntry {
  const e = x as Partial<TuningEntry> | null;
  return (
    !!e &&
    typeof e.at === 'string' &&
    typeof e.seed === 'number' &&
    Number.isFinite(e.seed) &&
    typeof e.source === 'string' &&
    Array.isArray(e.theta) &&
    ACTIONS.has(e.action as string)
  );
}

function readStored(): TuningEntry[] {
  try {
    for (const k of LEGACY_KEYS) localStorage.removeItem(k);
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry) : [];
  } catch {
    return [];
  }
}
