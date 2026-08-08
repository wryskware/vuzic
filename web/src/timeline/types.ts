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
 * The raw pooled 1024-dim MuQ embedding sidecar (`embedding.json` + `.bin`) is
 * **not loaded by the runtime any more** (plan.md Revision 4). Projecting from
 * 1024 unnamed dimensions reacted to everything and isolated nothing, so the
 * mapping layer now builds a bank of ~16 *named* drivers from the timeline's own
 * 64-dim `latent` channel plus its structure channels. The sidecar stays on disk
 * and in the analysis output — it is the training input for the distilled-NN
 * mapping in plan.md's "Later" — but nothing in `web/` fetches it, which also
 * deletes the 11 MB background download and the cold/warm input swap with it.
 */
export interface Timeline {
  manifest: TimelineManifest;
  /** frames x stride, row-major */
  data: Float32Array;
  /** sum of all channel dims; the row stride of `data` */
  stride: number;
  channels: ReadonlyMap<string, ChannelSpec>;
  /** validated, clamped and sorted; empty when the manifest carries none */
  events: readonly TimelineEvent[];
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
