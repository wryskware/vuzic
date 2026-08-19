/**
 * What tracks exist, and where each one loads from.
 *
 * Three sources, in one list, deliberately indistinguishable to everything
 * downstream: **bundled** tracks shipped in `public/timelines/` by
 * `scripts/sync-data.mjs`, **server** tracks offered by a running
 * `terrarium-server`, and **cached** server tracks whose payloads are still in
 * the Cache API from a previous session. A `TrackEntry` carries a base URL and a
 * fetcher, and `loadTimeline` and `AudioClock` take both — so nothing else in
 * the app branches on where a track came from.
 *
 * The server is optional in the strongest sense: it is probed once, with a short
 * timeout, and everything works without it. `analysis/README.md` documents how
 * to run one.
 */
import { DEMO_BUILD } from '../runtime/demo.ts';
import { cachingFetch, listCachedTracks } from './cache.ts';

/**
 * Where a `terrarium-server` would be. A constant, not a setting: the server is
 * a local development tool, `--port` defaults to this on the python side, and a
 * configurable endpoint would be a knob with exactly one possible value.
 */
export const SERVER_BASE = 'http://localhost:8765';

/** How long the startup probe waits before deciding there is no server. */
const PROBE_TIMEOUT_MS = 1500;

export type TrackSource = 'bundled' | 'server' | 'cached';

export interface TrackEntry {
  /** URL key and directory name; what `?track=` carries. */
  id: string;
  /** Human name — the analysis manifest's own `track.id` ("Free Fall (Remastered)"). */
  title: string;
  duration: number;
  /** Hash of the exact timeline.json + timeline.bin bytes this entry loads. */
  version: string;
  /** Base URL for `timeline.json` / `timeline.bin` / the audio file, no trailing slash. */
  base: string;
  hasAudio: boolean;
  /**
   * The audio file's name under `base`. Not a constant, because the two sources
   * genuinely disagree: `sync-data` transcodes bundled audio to `audio.m4a` (a
   * 40 MB WAV per track is fine over localhost and indefensible over CloudFront),
   * while `terrarium-server` serves the pipeline's own `audio.wav` untouched.
   *
   * Defaulted rather than required so a stale `index.json` or an older server
   * still resolves to the historic name instead of losing its audio silently.
   */
  audioFile: string;
  source: TrackSource;
}

/** What a track's audio was always called, before bundled builds started transcoding. */
export const DEFAULT_AUDIO_FILE = 'audio.wav';

/**
 * Bundled and server tracks fetch differently; the difference stops here.
 *
 * Bound, never bare `fetch`: consumers store this on an object and call it as a
 * method (`this.fetcher(...)` in AudioClock), and native fetch with a wrong
 * receiver fails its WebIDL brand check. Crucially that failure is a *rejected
 * promise*, not a throw — so an unbound copy here didn't crash anything, it
 * silently rode `preload()`'s catch into the click-track fallback and every
 * bundled track lost its audio. `loadTimeline` masked it by calling the fetcher
 * as a free function, where an undefined receiver defaults to the global.
 */
export function fetcherFor(entry: TrackEntry): typeof fetch {
  return entry.source === 'bundled'
    ? globalThis.fetch.bind(globalThis)
    : (cachingFetch as typeof fetch);
}

interface ServerTrackRow {
  id?: unknown;
  title?: unknown;
  duration?: unknown;
  version?: unknown;
  hasAudio?: unknown;
  audio?: unknown;
}

function bundledBase(id: string): string {
  // No leading slash: BASE_URL already ends in one, and the app may be served
  // from a subpath.
  return `${import.meta.env.BASE_URL}timelines/${id}`;
}

/**
 * The bundled listing, written by `scripts/sync-data.mjs` at dev/build time.
 *
 * A generated file rather than a hardcoded array because the set of bundled
 * tracks is a property of `data/timelines/`, and a list in TypeScript would go
 * stale the first time someone dropped a track in without editing it. Its
 * absence is survivable — `DEFAULT_TRACK` still loads by URL — so a failure here
 * is a warning, not a throw.
 */
export async function loadBundledTracks(): Promise<TrackEntry[]> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}timelines/index.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows = (await res.json()) as ServerTrackRow[];
    return rows.filter(isRow).map((r) => ({
      id: String(r.id),
      title: String(r.title ?? r.id),
      duration: Number(r.duration) || 0,
      version: String(r.version ?? ''),
      base: bundledBase(String(r.id)),
      hasAudio: r.hasAudio === true,
      audioFile: typeof r.audio === 'string' ? r.audio : DEFAULT_AUDIO_FILE,
      source: 'bundled' as const,
    }));
  } catch (err) {
    console.warn('timelines/index.json missing or malformed; run `npm run sync-data`', err);
    return [];
  }
}

function isRow(r: ServerTrackRow): boolean {
  return typeof r.id === 'string' && /^[A-Za-z0-9._-]+$/.test(r.id);
}

/**
 * Ask the server what it has. `null` means "no server", which is not an error.
 *
 * One unavoidable wart: with nothing listening, Chrome writes its own
 * `ERR_CONNECTION_REFUSED` line to the console before `fetch` ever rejects.
 * That comes from the network stack, not from here, and no amount of catching
 * suppresses it — the info line below is what this module contributes.
 */
export async function probeServer(): Promise<TrackEntry[] | null> {
  // The published cut has no server to find, and on HTTPS this request is
  // blocked as mixed content before it is even attempted. "No server" is
  // already an ordinary answer here, so declining to ask needs no other path.
  if (DEMO_BUILD) return null;
  try {
    const res = await fetch(`${SERVER_BASE}/tracks`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = (await res.json()) as { tracks?: ServerTrackRow[] };
    const rows = Array.isArray(body.tracks) ? body.tracks : [];
    return rows.filter(isRow).map(serverEntry);
  } catch {
    return null;
  }
}

export function serverEntry(r: ServerTrackRow): TrackEntry {
  const id = String(r.id);
  return {
    id,
    title: String(r.title ?? id),
    duration: Number(r.duration) || 0,
    version: String(r.version ?? ''),
    base: `${SERVER_BASE}/tracks/${id}`,
    hasAudio: r.hasAudio === true,
    audioFile: typeof r.audio === 'string' ? r.audio : DEFAULT_AUDIO_FILE,
    source: 'server',
  };
}

/**
 * One list, first source wins per id.
 *
 * Order is bundled, then server, then cached, and it is a precedence order:
 * a track that shipped in the build loads from the build (no network, no cache
 * staleness), a live server outranks its own leftovers, and the cache is what is
 * left when the first two have nothing to say. The same id appearing in two
 * sources is the normal case, not a conflict — `sync-data` bundles exactly the
 * directory the server writes into.
 */
export function mergeCatalog(...groups: readonly TrackEntry[][]): TrackEntry[] {
  const byId = new Map<string, TrackEntry>();
  for (const group of groups) {
    for (const entry of group) if (!byId.has(entry.id)) byId.set(entry.id, entry);
  }
  return [...byId.values()].sort((a, b) => a.title.localeCompare(b.title));
}

export interface Catalog {
  tracks: TrackEntry[];
  /** Did a server answer the probe? */
  server: boolean;
  /**
   * How many tracks it offered — reported separately from the merged list
   * because the merge hides them: a server track that is also bundled is listed
   * as bundled, so counting `source === 'server'` after the merge would show
   * zero against a server holding every track the app can see.
   */
  serverCount: number;
}

/** Everything available right now, and whether a server answered. */
export async function buildCatalog(): Promise<Catalog> {
  const [bundled, server, cached] = await Promise.all([
    loadBundledTracks(),
    probeServer(),
    listCachedTracks(),
  ]);
  // Not in the published cut: there, no server is the design rather than a
  // condition worth reporting, and naming a localhost port to a stranger reads
  // as something being broken.
  if (server === null && !DEMO_BUILD) {
    console.info(
      `analysis server not reachable at ${SERVER_BASE}; ` +
        `${bundled.length} bundled and ${cached.length} cached track(s) available`,
    );
  }
  return {
    tracks: mergeCatalog(bundled, server ?? [], cached),
    server: server !== null,
    serverCount: server?.length ?? 0,
  };
}

/** POST an audio file; resolves to the job id the server assigned. */
export async function uploadForAnalysis(file: File): Promise<{ jobId: string; trackId: string }> {
  const body = new FormData();
  body.append('file', file);
  const res = await fetch(`${SERVER_BASE}/analyze`, { method: 'POST', body });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`analyze: HTTP ${res.status} ${detail.slice(0, 200)}`);
  }
  return (await res.json()) as { jobId: string; trackId: string };
}

export interface JobState {
  status: 'queued' | 'running' | 'done' | 'error';
  stage: string;
  message: string;
  error: string;
  progress: number;
  trackId: string;
}

export async function pollJob(jobId: string): Promise<JobState> {
  const res = await fetch(`${SERVER_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`job ${jobId}: HTTP ${res.status}`);
  return (await res.json()) as JobState;
}
