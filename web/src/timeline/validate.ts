import {
  isEventKind,
  type ChannelSpec,
  type Timeline,
  type TimelineEvent,
  type TimelineManifest,
} from './types.ts';

function record(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`timeline: ${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, path: string, min: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min) {
    throw new Error(`timeline: ${path} must be a finite number >= ${min}`);
  }
  return value;
}

function integer(value: unknown, path: string, min: number): number {
  const number = finite(value, path, min);
  if (!Number.isSafeInteger(number)) {
    throw new Error(`timeline: ${path} must be a safe integer`);
  }
  return number;
}

function finiteArray(value: unknown, path: string, min = Number.NEGATIVE_INFINITY): void {
  if (!Array.isArray(value)) throw new Error(`timeline: ${path} must be an array`);
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`, min);
  }
}

/**
 * Validate the portion of the v2 manifest needed to safely construct its dense
 * binary view. Browser fetches and Node filesystem reads both enter here.
 */
export function validateTimelineManifest(value: unknown): asserts value is TimelineManifest {
  const manifest = record(value, 'manifest');
  if (manifest['version'] !== 2) {
    throw new Error(`timeline: unsupported version ${String(manifest['version'])}`);
  }

  const track = record(manifest['track'], 'track');
  if (typeof track['id'] !== 'string' || track['id'].length === 0) {
    throw new Error('timeline: track.id must be a non-empty string');
  }
  finite(track['duration'], 'track.duration', 0);
  integer(track['sampleRate'], 'track.sampleRate', 1);

  const grid = record(manifest['grid'], 'grid');
  finite(grid['hopSeconds'], 'grid.hopSeconds', Number.MIN_VALUE);
  integer(grid['frames'], 'grid.frames', 1);

  finiteArray(manifest['beats'], 'beats', 0);
  finiteArray(manifest['downbeats'], 'downbeats', 0);
  finite(manifest['tempo'], 'tempo', 0);
  if (!Array.isArray(manifest['segments'])) {
    throw new Error('timeline: segments must be an array');
  }
  for (let index = 0; index < manifest['segments'].length; index++) {
    const segment = record(manifest['segments'][index], `segments[${index}]`);
    const start = finite(segment['start'], `segments[${index}].start`, 0);
    const end = finite(segment['end'], `segments[${index}].end`, 0);
    if (end < start) throw new Error(`timeline: segments[${index}].end must not precede start`);
    if (typeof segment['label'] !== 'string' || segment['label'].length === 0) {
      throw new Error(`timeline: segments[${index}].label must be a non-empty string`);
    }
    const confidence = finite(segment['confidence'], `segments[${index}].confidence`, 0);
    if (confidence > 1) throw new Error(`timeline: segments[${index}].confidence must be <= 1`);
  }
  if (manifest['segmentSimilarity'] !== undefined) {
    if (!Array.isArray(manifest['segmentSimilarity'])) {
      throw new Error('timeline: segmentSimilarity must be an array');
    }
    for (let row = 0; row < manifest['segmentSimilarity'].length; row++) {
      finiteArray(manifest['segmentSimilarity'][row], `segmentSimilarity[${row}]`);
    }
  }

  if (!Array.isArray(manifest['channels']) || manifest['channels'].length === 0) {
    throw new Error('timeline: no channels');
  }

  const channels: ChannelSpec[] = manifest['channels'].map((value, index) => {
    const channel = record(value, `channels[${index}]`);
    if (typeof channel['name'] !== 'string' || channel['name'].length === 0) {
      throw new Error(`timeline: channels[${index}].name must be a non-empty string`);
    }
    return {
      name: channel['name'],
      dims: integer(channel['dims'], `channels[${index}].dims`, 1),
      offset: integer(channel['offset'], `channels[${index}].offset`, 0),
    };
  });

  const names = new Set<string>();
  const sorted = [...channels].sort((a, b) => a.offset - b.offset);
  let cursor = 0;
  for (const channel of sorted) {
    if (names.has(channel.name)) {
      throw new Error(`timeline: duplicate channel "${channel.name}"`);
    }
    names.add(channel.name);
    if (channel.offset !== cursor) {
      throw new Error(
        `timeline: channel "${channel.name}" offset ${channel.offset} leaves a gap/overlap at ${cursor}`,
      );
    }
    cursor += channel.dims;
    if (!Number.isSafeInteger(cursor)) throw new Error('timeline: channel stride is too large');
  }
}

/**
 * Optional events are validated defensively. A malformed event is ignored so
 * older or hand-edited manifests still load without enabling bad impulses.
 */
export function normalizeEvents(raw: unknown, duration: number): TimelineEvent[] {
  if (!Array.isArray(raw)) return [];
  const out: TimelineEvent[] = [];
  let dropped = 0;
  for (const event of raw) {
    const candidate = event as Partial<TimelineEvent> | null;
    if (
      !candidate ||
      typeof candidate.t !== 'number' ||
      !Number.isFinite(candidate.t) ||
      !isEventKind(candidate.kind)
    ) {
      dropped++;
      continue;
    }
    const t = Math.min(Math.max(candidate.t, 0), Math.max(duration, 0));
    const rawStrength = candidate.strength;
    const strength =
      typeof rawStrength === 'number' && Number.isFinite(rawStrength) ? rawStrength : 1;
    out.push({ t, kind: candidate.kind, strength: Math.min(Math.max(strength, 0), 1) });
  }
  if (dropped > 0) console.warn(`timeline: dropped ${dropped} malformed event(s)`);
  out.sort((a, b) => a.t - b.t);
  return out;
}

/** Build the shared runtime Timeline after transport-specific I/O completes. */
export function createTimeline(
  manifestValue: unknown,
  binary: ArrayBuffer | Uint8Array,
): Timeline {
  validateTimelineManifest(manifestValue);
  const manifest = manifestValue;
  const stride = manifest.channels.reduce((sum, channel) => sum + channel.dims, 0);
  const expected = manifest.grid.frames * stride * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(expected)) throw new Error('timeline.bin: expected size is too large');
  if (binary.byteLength !== expected) {
    throw new Error(
      `timeline.bin: expected ${expected} bytes (${manifest.grid.frames} frames x ${stride} dims), got ${binary.byteLength}`,
    );
  }

  let data: Float32Array;
  if (binary instanceof ArrayBuffer) {
    data = new Float32Array(binary);
  } else {
    // Node Buffers may be slices of a pooled, unaligned backing store. Copy only
    // that adapter case so the Float32Array always begins at its own byte zero.
    const owned = new Uint8Array(binary.byteLength);
    owned.set(binary);
    data = new Float32Array(owned.buffer);
  }

  return {
    manifest,
    data,
    stride,
    channels: new Map(manifest.channels.map((channel) => [channel.name, channel])),
    events: normalizeEvents(manifest.events, manifest.track.duration),
  };
}
