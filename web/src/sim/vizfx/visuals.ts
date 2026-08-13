/**
 * The vizfx implementation registry.
 *
 * `ids.ts` owns identity and order without importing WGSL, so recipes and the
 * API can validate names in Node. This module binds each ID to its actual visual
 * implementation; the typed record and runtime identity check make either half
 * impossible to omit or mis-key silently.
 *
 * Order matters twice: it is the order the preset picker lists, and it is the
 * cycle order when a section-boundary transition advances the repertoire.
 * `id` doubles as the `?sim=` value and the autosave-slot key (`ModTarget.simId`),
 * which is why two visuals must never share one — their θ tables differ, so a
 * shared slot would apply one visual's saved vector to another's registry.
 */
import type { VizFxVisual } from './slots.ts';
import { VIZFX_IDS, type VizFxId } from './ids.ts';
import { NEBULA } from './nebula/nebula.ts';
import { TUNNEL } from './tunnel/tunnel.ts';
import { KALEIDO } from './kaleido/kaleido.ts';
import { PLASMA } from './plasma/plasma.ts';

/**
 * The repertoire, in cycle order.
 *
 * NEBULA stays first: it is the shipped default, the exemplar every other table
 * was written against, and the visual the panel opens on.
 *
 * The rest are ordered for *contrast between neighbours*, because this list is
 * also the order a section-boundary transition advances through — so what
 * matters is that consecutive entries do not look like each other. Nebula's soft
 * spiral gives way to tunnel's rushing depth, tunnel's radial flow to kaleido's
 * locked mirror symmetry, and kaleido's rigid rosette to plasma's unstructured
 * flowing field. Grouping the two radial visuals together instead would make one
 * section change out of three read as "nothing happened".
 */
const VISUALS_BY_ID: Readonly<Record<VizFxId, VizFxVisual>> = {
  nebula: NEBULA,
  tunnel: TUNNEL,
  kaleido: KALEIDO,
  plasma: PLASMA,
};

export const VIZFX_VISUALS: readonly VizFxVisual[] = VIZFX_IDS.map((id) => {
  const visual = VISUALS_BY_ID[id];
  if (visual.id !== id) throw new Error(`VizFX registry mismatch: ${id} resolved to ${visual.id}`);
  return visual;
});
