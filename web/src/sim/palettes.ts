/**
 * The curated palette catalog.
 *
 * Art direction, not machinery: a menu of starting points known to look good
 * *in the sims*, chosen by loading them into physarum's trail field and plife's
 * particle splats rather than by admiring swatches. The two substrates stress a
 * palette in opposite directions — physarum accumulates and blooms, so it wants
 * hues that stay separable after they have been added together and pushed
 * through the HDR chain, while plife draws thousands of small discrete dots and
 * wants per-species contrast at a glance — so an entry that only works in one of
 * them did not survive.
 *
 * Every colour is authored in HSLuv, not as a hex literal. That is the point of
 * HSLuv: a list of hues at one shared (S, L) is a list of colours that are
 * equally saturated and equally *bright* to the eye, so no species quietly
 * dominates because its hex happened to be luminous. Yellow at L 62 is not
 * brighter than blue at L 62, which is exactly the thing RGB hex lists get wrong.
 *
 * Entries do not touch `saturation`, `brightness` or `hueRateDegPerSec`: those
 * are trims the author tuned against their grade and their track, and a palette
 * change should not silently re-expose the scene or start a rotation nobody
 * asked for. `hueShiftDeg` *is* reset, so a loaded entry shows the hues it was
 * designed with.
 *
 * ## The second pass (2026-08-15)
 *
 * The nine original entries were invented. The six added later are *adapted*,
 * each from a named collection — cmocean, Okabe & Ito's CUD set, Paul Tol's
 * schemes, MetBrewer, matplotlib — plus `origin`, recovered from the first
 * commit that ran a simulation. Every one of them names its source in a comment,
 * and every one of them was changed on the way in, always in the same two ways
 * and always for the same reason:
 *
 * - **Lightness is flattened.** Scientific maps encode a scalar, so they ramp
 *   lightness by design. Species are not a scalar. A species that is dark
 *   because of its index is a species you cannot find, and against this
 *   project's deep black point "dark" starts around HSLuv L 45.
 * - **Members are dropped.** A qualitative set built for ink on white can afford
 *   two hues 20° apart; plife draws them as dots a few pixels across and cannot.
 *   Where a source offered six or eight, the widest-separated four were taken —
 *   four, because that is plife's primary count and a four-colour entry gets its
 *   accents from the group lift, which is the relationship the shipped palette
 *   already has.
 *
 * Two entries deliberately keep an uneven lightness — `origin`, because that
 * spread *is* what the first build looked like, and `viridis`, because the ramp
 * is the entire content of that map. They are the exceptions and they say so.
 *
 * Two candidates were tried in plife and dropped, and are recorded because a
 * rejection is the more useful half of a curation pass: cmocean's `haline` (see
 * the note where it would have sat) and Paul Tol's `light` pastel scheme. The
 * pastels failed for a reason that generalises — at HSLuv L 72 and S ~60, bloom
 * and the grade carry every species most of the way to white, so a set chosen
 * for being gentle on paper arrives here as cream and pale blue. Nothing in this
 * catalog can be quiet by being pale; the quiet entries are quiet by being
 * *narrow* (`verdant arc`) or *monochrome* (`steel + signal`).
 */
import {
  GROUP_LIGHT_LIFT,
  GROUP_LIGHT_MAX,
  hsluvToHex,
  syncPaletteColors,
  type Palette,
  type PaletteArc,
  type PaletteMode,
  type PaletteSpace,
} from './palette.ts';

/** An HSLuv triple: hue 0..360, saturation 0..100, lightness 0..100. */
export type HsluvColor = readonly [number, number, number];

export interface PaletteCatalogEntry {
  /** dropdown label; also the stable id, so keep it stable */
  name: string;
  /** one line about what it is for, shown as the folder's help text */
  note: string;
  mode: PaletteMode;
  /**
   * The space the entry was art-directed in. Every shipped entry was authored
   * against HSLuv, so loading one sets the palette to HSLuv — reinterpreting an
   * HSLuv triple as OKLCh would silently render a different palette than the one
   * that was tuned in the sims.
   */
  space?: PaletteSpace;
  /** arc entries only: the primaries' arc */
  arc?: PaletteArc;
  /**
   * arc entries only, optional: the accents' arc. Omitted means "the primaries'
   * arc, lighter" — see `applyPaletteEntry`. Name it to make the accents a
   * genuinely separate colour family.
   */
  accentArc?: PaletteArc;
  /** custom entries only; cycled if K exceeds its length */
  colors?: readonly HsluvColor[];
}

/**
 * Ordered loosely by how much of the wheel they use, because that is how you
 * choose one: a full spectrum when K is large and every species must be
 * findable, a narrow arc or a two-pole scheme when the image should read as one
 * mood with accents.
 */
export const PALETTE_CATALOG: readonly PaletteCatalogEntry[] = [
  {
    name: 'spectrum',
    note: 'the whole wheel, evenly divided — vuzic\'s default, best at large K',
    mode: 'arc',
    // S 100 is the sRGB gamut boundary at each (H, L); L 62 sits above the point
    // where the deep black point in this project's grade swallows the midtones.
    arc: { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 },
  },
  {
    name: 'ember arc',
    note: 'red through orange to yellow; hot and legible over a black field',
    mode: 'arc',
    // Deliberately short: 92° keeps every species inside "fire" while still
    // separating them. Lightness is up at 68 because the warm end of the wheel
    // is where bloom eats contrast first.
    arc: { hueStartDeg: 8, hueRangeDeg: 92, sat: 100, light: 68 },
  },
  {
    name: 'abyss arc',
    note: 'cyan through blue and violet to magenta; cold, deep, high contrast',
    mode: 'arc',
    arc: { hueStartDeg: 200, hueRangeDeg: 130, sat: 100, light: 58 },
  },
  {
    name: 'verdant arc',
    note: 'yellow-green through green to teal; the quietest of the arcs',
    mode: 'arc',
    // Reversed (negative range) so species 0 is the teal end. Direction matters
    // in physarum, where species 0 is the one the default matrix favours.
    arc: { hueStartDeg: 175, hueRangeDeg: -85, sat: 95, light: 63 },
  },
  {
    name: 'thermal arc',
    note: 'violet through magenta and red to orange; cmocean thermal, as an arc',
    mode: 'arc',
    // Source: cmocean's `thermal` (Thyng et al. 2016, "True Colors of
    // Oceanography"), a perceptually uniform sequential map that runs
    // blue-violet → red → orange → yellow. Only its *hue path* is borrowed:
    // cmocean ramps lightness along the map because it encodes a scalar, and
    // species are not a scalar — a species that is dark because of its index is
    // a species that disappears. So the path is flattened onto one HSLuv
    // lightness and the range stops short of cmocean's pale yellow tail, which
    // at equal L is indistinguishable from the orange before it.
    arc: { hueStartDeg: 280, hueRangeDeg: 170, sat: 100, light: 66 },
  },
  // cmocean's `haline` was tried here as its cold counterpart (262° → 82°, the
  // salinity map's own sweep) and rejected in plife on 2026-08-15. Four species
  // over a ~180° cold arc puts two of them in the cyan–teal band, where HSLuv
  // hue discriminability is at its worst: species 1 and 2 came out #00aac6 and
  // #00af99 and read as one colour at dot size. Widening the sweep to 215° fixed
  // the collapse and cost the entry its reason to exist — it lands on 'verdant
  // arc' with a blue on the front. Kept as a note so the next person does not
  // re-derive it.
  {
    name: 'teal & coral',
    note: 'complementary pair, two hues of spread each — maximum species contrast',
    mode: 'custom',
    // Interleaved rather than grouped: adjacent species indices land on opposite
    // poles, so neighbouring agents in a physarum trail never share a pole.
    colors: [
      [188, 100, 64],
      [16, 100, 66],
      [204, 100, 58],
      [32, 100, 72],
    ],
  },
  {
    name: 'violet + gold',
    note: 'three analogous violets and one warm accent; the accent reads as the event',
    mode: 'custom',
    colors: [
      [268, 95, 56],
      [288, 95, 60],
      [306, 95, 62],
      [58, 100, 80],
    ],
  },
  {
    name: 'warm / cool split',
    note: 'two warm, two cool, nothing in between — reads as two populations',
    mode: 'custom',
    colors: [
      [34, 100, 72],
      [238, 95, 52],
      [48, 100, 76],
      [258, 95, 56],
    ],
  },
  {
    name: 'steel + signal',
    note: 'monochrome blue at three lightnesses plus one hot accent',
    mode: 'custom',
    // The one entry that deliberately breaks the equal-lightness rule: a mono
    // scheme has only lightness left to separate species with.
    colors: [
      [222, 62, 42],
      [222, 62, 58],
      [222, 62, 74],
      [22, 100, 70],
    ],
  },
  {
    name: 'aurora',
    note: 'green-teal to violet-magenta; the widest custom set, good at K = 4..8',
    mode: 'custom',
    // 138 → 190 rather than 148 → 176: at the tighter spacing physarum's first
    // two species read as one green, because species 0 dominates its field and
    // the trail blur averages the pair together before the grade ever sees them.
    colors: [
      [138, 100, 68],
      [190, 100, 60],
      [272, 100, 56],
      [326, 100, 62],
    ],
  },
  {
    name: 'origin',
    note: 'the four colours the first build shipped with — orange, magenta, cyan, violet',
    mode: 'custom',
    // Recovered from `11fe27d` ("web: WebGPU terrarium app (phases 2-7)"), the
    // first commit with a running sim. The palette was the four `colorHex`
    // fields on physarum's species TEMPLATES, authored as sRGB hex and keyed to
    // the stems: bass `#ff7a1a`, drums `#ff2f6d`, vocals `#35d6ff`, other
    // `#a56bff`. (Past K=4 that build walked the hue by 41° per cycle in *HSL*;
    // physarum still does, via `defaultPaletteColor`. The walk is not
    // reproduced here — the catalog's own convention, hue kept and lightness
    // lifted per group, is what a v2 entry means by "the next four".)
    //
    // Converted to HSLuv and rounded to integers, which moves each colour by at
    // most 2/255 in one channel (#ff7a1a → #ff7a23 is the worst of them). The
    // one entry with no equal-lightness discipline at all, because it predates
    // the idea: the cyan sits at L 80 and the magenta at 56, and that spread —
    // a bright vocal line over a dark drum — is the look being recovered, not a
    // defect to normalise away.
    colors: [
      [27, 100, 66],
      [2, 100, 56],
      [221, 100, 80],
      [277, 100, 58],
    ],
  },
  {
    name: 'okabe–ito',
    note: 'the colourblind-safe qualitative standard; every pair separable to every viewer',
    mode: 'custom',
    // Source: Okabe & Ito, "Color Universal Design" (2008) — the eight-colour
    // qualitative set chosen so no pair collides under protan, deutan or tritan
    // vision. Four of the eight are taken: orange `#E69F00`, sky blue
    // `#56B4E9`, bluish green `#009E73`, reddish purple `#CC79A7`. The other
    // four are black (useless on a black field) and three hues that fall
    // between these — at K=4 the widest-spread four is the whole point.
    //
    // Their authored HSLuv lightnesses are 71 / 70 / 58 / 61: the set was
    // designed for ink on white, so its darker members are the ones that lose
    // to this project's black point. Flattened to a shared L 66 — the level the
    // rest of the catalog's four-colour entries sit at, chosen so the +18 group
    // lift lands the accents at 84 rather than washing them out near the 92 cap.
    colors: [
      [50, 100, 66],
      [235, 80, 66],
      [154, 100, 66],
      [335, 55, 66],
    ],
  },
  {
    name: 'tol vibrant',
    note: 'Paul Tol\'s vibrant scheme — blue, teal, orange, magenta at full chroma',
    mode: 'custom',
    // Source: Paul Tol, "Colour Schemes" (SRON technical note), the `vibrant`
    // qualitative scheme. Four of its six: blue `#0077BB`, teal `#009988`,
    // orange `#EE7733`, magenta `#EE3377` — the cyan and the red are dropped
    // because each sits within ~20° of a neighbour here and plife's dots are
    // small enough that 20° is not a difference.
    //
    // Two warm and two cool but interleaved cool/cool/warm/warm rather than
    // alternating: in plife the first two indices are bass and drums, and
    // giving the rhythm section the cold half is what makes the orange and
    // magenta of vocals/other read as the melodic voices.
    colors: [
      [245, 100, 66],
      [174, 100, 66],
      [27, 88, 66],
      [358, 85, 66],
    ],
  },
  {
    name: 'egypt',
    note: 'vermillion, lapis, malachite, amber — four pigments, maximum mutual distance',
    mode: 'custom',
    // Source: MetBrewer's `Egypt` (Blake Robert Mills), taken from an Egyptian
    // painted panel in the Met's collection. Natively four colours — `#DD5129`,
    // `#0F7BA2`, `#43B284`, `#FAB255` — which is exactly plife's primary count,
    // so nothing is dropped and nothing is invented.
    //
    // Their HSLuv lightnesses run 53 / 48 / 66 / 78; flattened to 64 so the
    // lapis stops being the species you cannot find. The hues (19 / 232 / 150 /
    // 48) are within 4° of the source and are the reason this reads as pigment
    // rather than as a hue wheel: nothing is on a cardinal.
    colors: [
      [19, 88, 64],
      [232, 98, 64],
      [150, 83, 64],
      [48, 89, 64],
    ],
  },
  {
    name: 'lakota',
    note: 'cyan, gold, crimson, deep green — high chroma, four quadrants of the wheel',
    mode: 'custom',
    // Source: MetBrewer's `Lakota` (Blake Robert Mills), from Lakota beadwork.
    // Four of its six: `#04A3BD`, `#F0BE3D`, `#931E18`, `#247D3F`; the orange
    // and the near-black navy are dropped, the first for sitting between the
    // gold and the crimson and the second for being L 17.
    //
    // The crimson is the adaptation that matters: at its authored L 32 it is a
    // hole in a plife field. At the shared L 66 it is no longer crimson so much
    // as a hot red, and the set survives because the *spacing* — 216 / 61 / 13
    // / 135, four hues that are nearly a quadrant apart each — is what was
    // worth taking from beadwork in the first place.
    //
    // The gold is the one member kept off the shared L 66, at 78. HSLuv's yellow
    // runs out of chroma as lightness falls, so a hue-61 gold at 66 is an olive
    // and the source's `#F0BE3D` is simply not in the set any more. 'violet +
    // gold' makes the same exception for the same colour and the same reason.
    colors: [
      [216, 100, 66],
      [61, 95, 78],
      [13, 90, 66],
      [135, 87, 66],
    ],
  },
  {
    name: 'viridis',
    note: 'purple → blue → teal → green, brightening as it goes; a ladder, not a ring',
    mode: 'custom',
    // Source: matplotlib's `viridis` (Smith & van der Walt, 2015), the
    // perceptually uniform default. The second entry after 'steel + signal' to
    // break the equal-lightness rule, and for the opposite reason: viridis *is*
    // its lightness ramp, so flattening it would leave four hues that are
    // merely adjacent. Sampled at four even stops of the map, then the floor
    // lifted (L 27 → 46, where the grade's black point stops swallowing it) and
    // the ceiling pulled down (L 91 → 84, so the accents' +18 lift has somewhere
    // to go before the 92 cap).
    //
    // In plife this reads as depth rather than as four voices: bass sits low and
    // violet, and each species up the stack is lighter and greener. Judge it
    // against 'steel + signal', which does the same trick inside one hue.
    colors: [
      [273, 58, 46],
      [212, 89, 58],
      [135, 81, 70],
      [77, 97, 84],
    ],
  },
];

export function paletteCatalogEntry(name: string): PaletteCatalogEntry | undefined {
  return PALETTE_CATALOG.find((entry) => entry.name === name);
}

/**
 * Overwrite a palette's colour definition with a catalog entry, in place.
 *
 * In place because the palette object is shared by reference with
 * `ModulationConfig` and bound by the panel's colour pickers — replacing it
 * would leave both pointing at the old one.
 */
export function applyPaletteEntry(
  palette: Palette,
  entry: PaletteCatalogEntry,
  speciesCount: number,
  groupSize = speciesCount,
): void {
  palette.mode = entry.mode;
  palette.space = entry.space ?? 'hsluv';
  // `Object.assign` and not `palette.arc = {...}`, for the same reason `colors`
  // is spliced rather than reassigned two blocks down, and it is the same bug
  // one field over: the panel's four arc sliders are bound to *this object* by
  // reference (`addArcControls(folder, p.arc)`), so replacing it left every
  // slider editing a detached copy. The symptom was silent and specific —
  // load any arc entry, then drag the arc knobs and nothing happens, while the
  // widgets still read whatever they read before the load.
  if (entry.arc) Object.assign(palette.arc, entry.arc);
  // An arc entry that does not name an accent arc gets its primaries' arc
  // lifted in lightness, which is the relationship the shipped plife palette
  // already has (+13..+21 HSLuv L at the same hue family). An entry that *does*
  // name one is making a deliberate second-family choice and is left alone.
  if (entry.accentArc) Object.assign(palette.accentArc, entry.accentArc);
  else if (entry.arc) {
    Object.assign(palette.accentArc, entry.arc, {
      light: Math.min(entry.arc.light + GROUP_LIGHT_LIFT, GROUP_LIGHT_MAX),
    });
  }
  if (entry.mode === 'custom' && entry.colors && entry.colors.length > 0) {
    const source = entry.colors;
    const next: string[] = [];
    for (let i = 0; i < Math.max(1, speciesCount); i++) {
      const [h, s, l] = source[i % source.length] as HsluvColor;
      // Past the end of the list, keep the hue and raise the lightness rather
      // than repeating the colour verbatim. That is plife's own convention for
      // its accent species ("lighter, higher-key relative of its primary" —
      // see `PALETTE_HEX` in plife/config.ts), and without it every catalog
      // entry renders plife's eight species as four indistinguishable pairs.
      const cycle = Math.floor(i / source.length);
      next.push(hsluvToHex(h, s, Math.min(l + cycle * 18, 92)));
    }
    palette.colors.length = 0;
    palette.colors.push(...next);
  }
  // The entry is authored at zero shift, so show it that way. The *rate* and the
  // linear-space trims are the author's and are left alone.
  palette.hueShiftDeg = 0;
  syncPaletteColors(palette, speciesCount, undefined, groupSize);
}
