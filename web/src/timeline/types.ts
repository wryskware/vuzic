export interface TrackInfo {
  id: string;
  duration: number;
  sampleRate: number;
}

export interface GridInfo {
  hopSeconds: number;
  frames: number;
}

export type SegmentLabel =
  | 'intro'
  | 'verse'
  | 'chorus'
  | 'bridge'
  | 'outro'
  | (string & {});

export interface Segment {
  start: number;
  end: number;
  label: SegmentLabel;
  confidence: number;
}

export interface ChannelSpec {
  name: string;
  dims: number;
  offset: number;
}

/**
 * Transient event kinds, pinned by the analysis contract (plan.md Revision 2).
 * Detected offline from the demucs stems; the runtime never does FFT.
 */
export const EVENT_KINDS = ['kick', 'snare', 'hat', 'bass', 'vocal'] as const;
export type EventKind = (typeof EVENT_KINDS)[number];

/** Sparse, sorted by `t` ascending. `strength` is 0..1. */
export interface TimelineEvent {
  t: number;
  kind: EventKind;
  strength: number;
}

export function isEventKind(x: unknown): x is EventKind {
  return typeof x === 'string' && (EVENT_KINDS as readonly string[]).includes(x);
}

export interface TimelineManifest {
  version: 2;
  track: TrackInfo;
  grid: GridInfo;
  beats: number[];
  downbeats: number[];
  tempo: number;
  segments: Segment[];
  segmentSimilarity?: number[][];
  channels: ChannelSpec[];
  /**
   * Optional and additive — the field arrived after version 2 shipped, so absent
   * or empty is a normal timeline with no beat-to-beat reactivity, not an error.
   */
  events?: TimelineEvent[];
}

/**
 * The raw pooled MuQ layer-6 embedding, `embedding.json` + `embedding.bin` next
 * to the timeline. plan.md Decision 2 emits it as a second sidecar precisely so a
 * later mapping could use the full width, and Revision 3's modulator is that
 * mapping: 1024 dims is what the random projections read.
 *
 * It is ~11 MB and **gitignored** (`data/​**​/embedding.bin`), so a fresh clone
 * legitimately has no such file. Absent is not an error — the modulator falls
 * back to the timeline's 64-dim PCA `latent` channel.
 */
export interface Embedding {
  frames: number;
  dims: number;
  hopSeconds: number;
  /** frames x dims, row-major, as stored — no scaling applied here */
  data: Float32Array;
  /** free-text provenance from embedding.json, for the workbench readout */
  source: string;
}

export interface Timeline {
  manifest: TimelineManifest;
  /** frames x stride, row-major */
  data: Float32Array;
  /** sum of all channel dims; the row stride of `data` */
  stride: number;
  channels: ReadonlyMap<string, ChannelSpec>;
  /** validated, clamped and sorted; empty when the manifest carries none */
  events: readonly TimelineEvent[];
  /**
   * The wide sidecar. `loadTimeline` always leaves this null — the file is ~11 MB
   * and awaiting it would block first paint — and `main` fetches it separately,
   * handing the result to `Modulator.attachSignal`. The field stays because
   * `chooseSignal` still honours it when a caller (a test, a future preload
   * path) has the embedding in hand at construction time.
   */
  embedding: Embedding | null;
}

/** Channel names the runtime knows about by name. All are optional at load time. */
export const KNOWN_CHANNELS = [
  'stems',
  'latent',
  'novelty4',
  'novelty16',
  'actChorus',
  'recurTime',
  'recurStr',
] as const;

/** stems channel dim order, fixed by the analysis contract */
export const STEM_NAMES = ['bass', 'drums', 'vocals', 'other'] as const;
export type StemName = (typeof STEM_NAMES)[number];
