import type { Timeline } from './types.ts';
import { createTimeline } from './validate.ts';

export { normalizeEvents } from './validate.ts';

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

  const manifest: unknown = await manifestRes.json();
  const buf = await binRes.arrayBuffer();
  // One fetch pair, and that is the whole load (Revision 4). The 11 MB
  // embedding sidecar the runtime used to chase in the background is gone: the
  // driver bank is built from this file's own `latent` and structure channels,
  // so the modulation input is complete from the first frame.
  return createTimeline(manifest, buf);
}
