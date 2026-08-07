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

/** ?seed= wins, then a pinned seed, then a fresh random one. */
export function resolveSeed(): { seed: number; pinned: boolean } {
  const param = new URLSearchParams(location.search).get('seed');
  if (param !== null && /^\d+$/.test(param)) return { seed: Number(param) >>> 0, pinned: false };
  const pinned = loadPinnedSeed();
  if (pinned !== null) return { seed: pinned, pinned: true };
  return { seed: randomSeed(), pinned: false };
}
