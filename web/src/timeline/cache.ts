/**
 * Offline storage for server-fetched tracks.
 *
 * **Cache API, not localStorage.** A track is `timeline.json` (~200 KB),
 * `timeline.bin` (~600 KB) and `audio.wav` (~40 MB). localStorage holds ~5 MB of
 * *strings*, so the audio alone would not fit even before base64 inflated it by
 * a third. IndexedDB would fit, but storing HTTP payloads there means reading
 * each one fully into an ArrayBuffer, writing it as a blob, and hand-rolling the
 * Response on the way back out — while the Cache API stores Responses natively,
 * streams them, and hands back something `loadTimeline` and `decodeAudioData`
 * already know how to eat. The read-through below is the whole integration.
 *
 * The *index* is in localStorage, and only the index: ids, titles, durations and
 * content versions, a couple of hundred bytes per track. It is there so the
 * picker can list what is available before any async work, and it is treated as
 * a hint rather than a truth — `listCachedTracks` drops any entry whose payload
 * is not actually in the Cache, which is what keeps the two stores from drifting
 * when eviction takes one and not the other.
 *
 * Versioning is per track id + the server's `version` field, which is a content
 * hash of the timeline pair rather than an mtime. A track whose version moved is
 * evicted whole before it is loaded, so a stale `timeline.bin` can never be
 * paired with a fresh `timeline.json`.
 */
import type { TrackEntry } from './catalog.ts';

const CACHE_NAME = 'lmt-tracks-v1';
const INDEX_KEY = 'lmt.trackCache.v1';

/** What the index remembers. `base` is the server URL the payloads are keyed by. */
interface IndexEntry {
  id: string;
  title: string;
  duration: number;
  version: string;
  hasAudio: boolean;
  base: string;
}

type CacheIndex = Record<string, IndexEntry>;

/** Storage is unavailable in some embeddings; every path here degrades to "no cache". */
function openCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return Promise.resolve(null);
  return caches.open(CACHE_NAME).catch(() => null);
}

function readIndex(): CacheIndex {
  try {
    const raw = localStorage.getItem(INDEX_KEY);
    const parsed: unknown = raw === null ? null : JSON.parse(raw);
    return parsed !== null && typeof parsed === 'object' ? (parsed as CacheIndex) : {};
  } catch {
    return {};
  }
}

function writeIndex(index: CacheIndex): void {
  try {
    localStorage.setItem(INDEX_KEY, JSON.stringify(index));
  } catch {
    // A full or disabled localStorage costs the offline *listing*, not the
    // payloads: they are already in the Cache and load fine by URL.
  }
}

/** Every URL a track occupies, in load order. */
function urlsFor(entry: { base: string; hasAudio: boolean }): string[] {
  const files = ['timeline.json', 'timeline.bin', ...(entry.hasAudio ? ['audio.wav'] : [])];
  return files.map((f) => `${entry.base}/${f}`);
}

/**
 * Fetch through the cache: a hit never touches the network, a miss is stored.
 *
 * This is the whole reason a previously loaded server track survives the server
 * going away — the loader and the audio clock call it instead of `fetch` and
 * neither of them has to know which side answered.
 */
export async function cachingFetch(url: string): Promise<Response> {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) return hit;
  const res = await fetch(url);
  // `clone()` before the caller reads the body — a Response body is a one-shot
  // stream, and putting the original would leave the caller with a drained one.
  if (res.ok && cache) await cache.put(url, res.clone()).catch(() => undefined);
  return res;
}

/** Forget a track's payloads and its index row. */
export async function dropCachedTrack(id: string): Promise<void> {
  const index = readIndex();
  const entry = index[id];
  if (!entry) return;
  const cache = await openCache();
  if (cache) await Promise.all(urlsFor(entry).map((u) => cache.delete(u).catch(() => false)));
  delete index[id];
  writeIndex(index);
}

/**
 * Evict `id` if the server now reports different content for it. Call before
 * loading a server track, so the read-through below repopulates from the network.
 */
export async function invalidateIfStale(entry: TrackEntry): Promise<void> {
  const known = readIndex()[entry.id];
  if (known && known.version !== entry.version) {
    console.info(`track cache: "${entry.id}" changed on the server; refetching`);
    await dropCachedTrack(entry.id);
  }
}

/** Record a track as cached. Called after a load, when the payloads are in. */
export function rememberCachedTrack(entry: TrackEntry): void {
  const index = readIndex();
  index[entry.id] = {
    id: entry.id,
    title: entry.title,
    duration: entry.duration,
    version: entry.version,
    hasAudio: entry.hasAudio,
    base: entry.base,
  };
  writeIndex(index);
}

/**
 * What is genuinely available offline, reconciled against the Cache.
 *
 * `timeline.json` and `timeline.bin` are both required — a track missing either
 * cannot load, and offering it in the picker would be offering a dead end. Audio
 * is not required: the clock falls back to the click track built from the beat
 * grid, which is exactly the behaviour a bundled timeline with no audio gets.
 */
export async function listCachedTracks(): Promise<TrackEntry[]> {
  const index = readIndex();
  // Before `openCache`, which would otherwise *create* an empty cache store on
  // every cold start just to find nothing in it.
  if (Object.keys(index).length === 0) return [];
  const cache = await openCache();
  if (!cache) return [];

  const out: TrackEntry[] = [];
  let pruned = false;
  for (const entry of Object.values(index)) {
    const [manifest, bin] = await Promise.all([
      cache.match(`${entry.base}/timeline.json`),
      cache.match(`${entry.base}/timeline.bin`),
    ]);
    if (!manifest || !bin) {
      delete index[entry.id];
      pruned = true;
      continue;
    }
    const audio = entry.hasAudio ? await cache.match(`${entry.base}/audio.wav`) : undefined;
    out.push({
      id: entry.id,
      title: entry.title,
      duration: entry.duration,
      version: entry.version,
      base: entry.base,
      hasAudio: entry.hasAudio && audio !== undefined,
      source: 'cached',
    });
  }
  if (pruned) writeIndex(index);
  return out;
}
