/**
 * Concrete VizFX identities in picker/transition order.
 *
 * Kept separate from the implementations so recipe and API validation can use
 * the whitelist in Node without importing WGSL modules or constructing visuals.
 */
export const VIZFX_IDS = ['nebula', 'tunnel', 'kaleido', 'plasma'] as const;

export type VizFxId = (typeof VIZFX_IDS)[number];

export function isVizFxId(value: string): value is VizFxId {
  return (VIZFX_IDS as readonly string[]).includes(value);
}
