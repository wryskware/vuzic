/**
 * The vizfx repertoire, in one place.
 *
 * Adding a visual to the app is adding a line here — `main.ts` derives the
 * `?sim=` whitelist, the substrate constructor and the picker's options from
 * this list, so a new warp/draw pair plus a θ table registers itself. Nothing
 * else may restate the set: a second copy would be a second place to forget,
 * and the one bug that pattern produces (a visual that runs but cannot be
 * selected, or vice versa) is invisible until someone reaches for it.
 *
 * Order matters twice: it is the order the preset picker lists, and it is the
 * cycle order when a section-boundary transition advances the repertoire.
 * `id` doubles as the `?sim=` value and the autosave-slot key (`ModTarget.simId`),
 * which is why two visuals must never share one — their θ tables differ, so a
 * shared slot would apply one visual's saved vector to another's registry.
 */
import type { VizFxVisual } from './slots.ts';
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
export const VIZFX_VISUALS: readonly VizFxVisual[] = [NEBULA, TUNNEL, KALEIDO, PLASMA];
