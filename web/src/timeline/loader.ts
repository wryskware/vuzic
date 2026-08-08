import {
  isEventKind,
  type ChannelSpec,
  type Embedding,
  type Timeline,
  type TimelineEvent,
  type TimelineManifest,
} from './types.ts';

/**
 * The `events` array is optional and additive, and the analysis half that emits it
 * is developed independently — so it is validated rather than trusted, and a bad
 * entry drops that entry instead of failing the load. Absent, empty, or entirely
 * malformed all resolve to the same thing: a timeline with no impulses.
 *
 * Sorting is defensive. The contract says ascending, and the whole runtime lookup
 * is a pointer walk that assumes it, so it is cheap insurance against a hand-edited
 * or concatenated file. `sort` is stable in every supported engine, so equal `t`
 * keeps the file's order.
 */
export function normalizeEvents(raw: unknown, duration: number): TimelineEvent[] {
  if (!Array.isArray(raw)) return [];
  const out: TimelineEvent[] = [];
  let dropped = 0;
  for (const e of raw) {
    const rec = e as Partial<TimelineEvent> | null;
    if (!rec || typeof rec.t !== 'number' || !Number.isFinite(rec.t) || !isEventKind(rec.kind)) {
      dropped++;
      continue;
    }
    // A slightly-out-of-range t is a rounding artefact, not corruption: clamp it.
    const t = Math.min(Math.max(rec.t, 0), Math.max(duration, 0));
    const s = typeof rec.strength === 'number' && Number.isFinite(rec.strength) ? rec.strength : 1;
    out.push({ t, kind: rec.kind, strength: Math.min(Math.max(s, 0), 1) });
  }
  if (dropped > 0) console.warn(`timeline: dropped ${dropped} malformed event(s)`);
  out.sort((a, b) => a.t - b.t);
  return out;
}

function validate(m: TimelineManifest): number {
  if (m.version !== 2) throw new Error(`timeline: unsupported version ${m.version}`);
  if (!Array.isArray(m.channels) || m.channels.length === 0)
    throw new Error('timeline: no channels');

  const sorted = [...m.channels].sort((a, b) => a.offset - b.offset);
  let cursor = 0;
  for (const c of sorted) {
    if (c.offset !== cursor)
      throw new Error(
        `timeline: channel "${c.name}" offset ${c.offset} leaves a gap/overlap at ${cursor}`,
      );
    cursor += c.dims;
  }
  return cursor;
}

interface EmbeddingManifest {
  version?: number;
  frames?: number;
  dims?: number;
  hopSeconds?: number;
  source?: string;
  dtype?: string;
}

/**
 * The wide sidecar, fetched best-effort. Every failure path — 404 because the
 * file is gitignored, a truncated .bin, a manifest that disagrees with the bytes —
 * resolves to null with one `console.info`, because the modulator has a working
 * 64-dim fallback and a missing optional file must never dead-end a load.
 */
export async function loadEmbedding(baseUrl: string): Promise<Embedding | null> {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  try {
    const [manifestRes, binRes] = await Promise.all([
      fetch(`${base}embedding.json`),
      fetch(`${base}embedding.bin`),
    ]);
    // Vite's dev server answers an unknown path with index.html and a 200, so
    // `ok` alone does not mean "the file exists" — check that we were actually
    // handed JSON before parsing, or a missing sidecar logs a syntax error.
    const isJson = manifestRes.headers.get('content-type')?.includes('json') ?? false;
    if (!manifestRes.ok || !binRes.ok || !isJson) {
      console.info(
        `timeline: no embedding sidecar (embedding.json ${manifestRes.status}${
          manifestRes.ok && !isJson ? ' but not JSON' : ''
        } / embedding.bin ${binRes.status}) — modulation falls back to the latent channel`,
      );
      return null;
    }
    const m = (await manifestRes.json()) as EmbeddingManifest;
    const frames = Number(m.frames);
    const dims = Number(m.dims);
    const hopSeconds = Number(m.hopSeconds);
    if (!Number.isInteger(frames) || !Number.isInteger(dims) || frames < 1 || dims < 1) {
      console.info('timeline: embedding.json has no usable frames/dims; ignoring the sidecar');
      return null;
    }
    if (m.dtype !== undefined && m.dtype !== 'float32-le') {
      console.info(`timeline: embedding dtype "${m.dtype}" is not float32-le; ignoring the sidecar`);
      return null;
    }
    const buf = await binRes.arrayBuffer();
    const expected = frames * dims * 4;
    if (buf.byteLength !== expected) {
      console.info(
        `timeline: embedding.bin is ${buf.byteLength} bytes, expected ${expected} ` +
          `(${frames}x${dims} f32); ignoring the sidecar`,
      );
      return null;
    }
    return {
      frames,
      dims,
      hopSeconds: Number.isFinite(hopSeconds) && hopSeconds > 0 ? hopSeconds : 0.1,
      data: new Float32Array(buf),
      source: typeof m.source === 'string' ? m.source : 'embedding',
    };
  } catch (err) {
    console.info('timeline: embedding sidecar unavailable —', err);
    return null;
  }
}

export async function loadTimeline(baseUrl: string): Promise<Timeline> {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

  const [manifestRes, binRes] = await Promise.all([
    fetch(`${base}timeline.json`),
    fetch(`${base}timeline.bin`),
  ]);
  if (!manifestRes.ok) throw new Error(`timeline.json: HTTP ${manifestRes.status}`);
  if (!binRes.ok) throw new Error(`timeline.bin: HTTP ${binRes.status}`);

  const manifest = (await manifestRes.json()) as TimelineManifest;
  const stride = validate(manifest);

  const buf = await binRes.arrayBuffer();
  const expected = manifest.grid.frames * stride * 4;
  if (buf.byteLength !== expected)
    throw new Error(
      `timeline.bin: expected ${expected} bytes (${manifest.grid.frames} frames x ${stride} dims), got ${buf.byteLength}`,
    );

  const channels = new Map<string, ChannelSpec>(manifest.channels.map((c) => [c.name, c]));
  const events = normalizeEvents(manifest.events, manifest.track.duration);
  // The wide sidecar is deliberately NOT awaited here. It is ~11 MB and this
  // promise gates GPU init, the panel and the first rAF, so awaiting it meant a
  // blank screen for the whole download. `main` fetches it in the background and
  // hands it to `Modulator.attachSignal`; until then modulation runs on the
  // 64-dim PCA `latent` channel, which is what a fresh clone gets anyway.
  return { manifest, data: new Float32Array(buf), stride, channels, events, embedding: null };
}
