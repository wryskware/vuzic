/**
 * The curated palette catalog.
 *
 * Art direction, not machinery: nine starting points that are known to look good
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
  if (entry.arc) palette.arc = { ...entry.arc };
  // An arc entry that does not name an accent arc gets its primaries' arc
  // lifted in lightness, which is the relationship the shipped plife palette
  // already has (+13..+21 HSLuv L at the same hue family). An entry that *does*
  // name one is making a deliberate second-family choice and is left alone.
  if (entry.accentArc) palette.accentArc = { ...entry.accentArc };
  else if (entry.arc) {
    palette.accentArc = {
      ...entry.arc,
      light: Math.min(entry.arc.light + GROUP_LIGHT_LIFT, GROUP_LIGHT_MAX),
    };
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
