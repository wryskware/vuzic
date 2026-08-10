import {
  isEventKind,
  type ChannelSpec,
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

/**
 * `fetcher` exists so a track can come from somewhere other than the bundle
 * without this function knowing where. Bundled timelines pass plain `fetch`;
 * server ones pass the Cache-API read-through in `cache.ts`, which is the whole
 * mechanism behind "a track you have loaded before still loads with the server
 * down". Nothing else about the load differs.
 */
export async function loadTimeline(
  baseUrl: string,
  fetcher: typeof fetch = fetch,
): Promise<Timeline> {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

  const [manifestRes, binRes] = await Promise.all([
    fetcher(`${base}timeline.json`),
    fetcher(`${base}timeline.bin`),
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
  // One fetch pair, and that is the whole load (Revision 4). The 11 MB
  // embedding sidecar the runtime used to chase in the background is gone: the
  // driver bank is built from this file's own `latent` and structure channels,
  // so the modulation input is complete from the first frame.
  return { manifest, data: new Float32Array(buf), stride, channels, events };
}
