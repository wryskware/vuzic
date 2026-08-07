/**
 * Append-only tuning log — the future NN's training set.
 *
 * plan.md's "Later" is a distilled MLP trained on (latent, θ, kept/discarded)
 * triples produced by exactly this workbench. Nothing here trains anything; it
 * just makes sure the data exists by the time someone wants it. One entry per
 * deliberate act (capture, save, keep, discard), never per frame.
 *
 * Exported as JSONL because that is what a training loader wants to stream.
 */

const STORAGE_KEY = 'lmt.tuningLog';
const MAX_ENTRIES = 300;

export type TuningAction = 'capture' | 'save' | 'keep' | 'discard' | 'snapshot' | 'restore';

export interface TuningEntry {
  /** ISO timestamp */
  at: string;
  /** sim tick the sample was taken at */
  tick: number;
  /** track seconds */
  time: number;
  action: TuningAction;
  anchor: number | null;
  anchorName: string | null;
  /** the latent vector at that moment */
  z: number[];
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
      z: entry.z.map(round),
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

function readStored(): TuningEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as TuningEntry[]) : [];
  } catch {
    return [];
  }
}
