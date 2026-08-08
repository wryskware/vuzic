const STORAGE_KEY = 'lmt.pinnedSeed';

export function randomSeed(): number {
  const buf = new Uint32Array(1);
  if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(buf);
  else buf[0] = (Math.random() * 0x100000000) >>> 0;
  return (buf[0] as number) >>> 0;
}

export function loadPinnedSeed(): number | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null || !/^\d+$/.test(raw)) return null;
    return Number(raw) >>> 0;
  } catch {
    return null;
  }
}

export function setPinnedSeed(seed: number | null): void {
  try {
    if (seed === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, String(seed >>> 0));
  } catch {
    // private mode / storage disabled — pinning is a convenience, not a contract
  }
}

const URL_PARAM = 'seed';

/**
 * Rewrite `?seed=` **only when it is already there**, so the address bar never
 * lies about which world is running. Adding it unasked would be worse than the
 * bug it fixes: an unpinned run is meant to be a new world on every reload
 * (Decision 5), and a stray URL param would silently freeze it.
 */
export function syncUrlSeed(seed: number | null): void {
  try {
    const url = new URL(location.href);
    if (!url.searchParams.has(URL_PARAM)) return;
    if (seed === null) url.searchParams.delete(URL_PARAM);
    else url.searchParams.set(URL_PARAM, String(seed >>> 0));
    history.replaceState(history.state, '', url);
  } catch {
    // no History API / opaque origin — the URL is a convenience, not a contract
  }
}

/**
 * ?seed= wins, then a pinned seed, then a fresh random one.
 *
 * The URL branch reports `pinned: true` because it *is* pinned in every sense
 * the workbench checkbox means — the run is fixed and reproducible across
 * reloads. Reporting false there made the checkbox read "not pinned" for a run
 * that could not be rerolled away from: ticking the box would write localStorage
 * while the URL kept winning on the next load. The workbench now calls
 * `syncUrlSeed` alongside `setPinnedSeed`, so reroll/pin/URL move together and
 * the box tells the truth.
 */
export function resolveSeed(): { seed: number; pinned: boolean } {
  const param = new URLSearchParams(location.search).get(URL_PARAM);
  if (param !== null && /^\d+$/.test(param)) return { seed: Number(param) >>> 0, pinned: true };
  const pinned = loadPinnedSeed();
  if (pinned !== null) return { seed: pinned, pinned: true };
  return { seed: randomSeed(), pinned: false };
}
