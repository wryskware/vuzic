/**
 * Colour, as every simulation needs it: the palette shape, the sRGB parsing, and
 * the hue machinery that keeps a large K distinguishable.
 *
 * None of this is physarum's. A palette is "K colours plus a global saturation
 * and brightness trim", which is true of any substrate that draws K things, and
 * the mapping layer serialises it (`ModulationConfig.palette`) without knowing
 * whose colours they are. What *is* physarum's — the shipped hexes, the species
 * templates, `defaultPalette` — stays in `physarum/config.ts`, because that is
 * art direction for one sim rather than machinery for all of them.
 *
 * ## Palette v2 (2026-08-13)
 *
 * v1 was a flat hex list. v2 keeps that list — it is still how you author one
 * colour at a time — and adds the vuzic scheme on top of it:
 *
 * - **arc mode**: hues spread evenly across `[hueStartDeg, +hueRangeDeg)` at one
 *   shared saturation and lightness, in the palette's chosen colour space.
 *   `colors` becomes a derived cache and the arc parameters are authoritative.
 * - **custom mode**: `colors` is authoritative, exactly as in v1.
 * - **a global hue shift**, `hueShiftDeg`, applied in the palette's hue space in
 *   *both* modes, plus `hueRateDegPerSec` which advances it against the simulation's
 *   own step clock. Not music-driven — plan.md Revision 2 rules out modulating
 *   hue, and the user confirmed on 2026-08-13 that a slow autonomous rotation of
 *   *all* hues together is consistent with the reason behind that decision
 *   (species stay mutually distinguishable; only the whole wheel turns).
 *
 * `hueShiftDeg` is the *authored base*, never the current effective shift. The
 * cycle is a pure function of it and of sim time, which is what lets a headless
 * export replay the identical colour trajectory from sim time zero.
 *
 * ### Naming, deliberately
 *
 * `saturation`/`brightness` (v1, unchanged) are a **linear-space** trim: a lerp
 * against luminance and a multiply, applied after the colour is chosen.
 * `arc.sat`/`arc.light` are **cylindrical colour-space** coordinates that choose
 * the colour in the first place, in whichever space `Palette.space` names. They
 * are different knobs and the names must not converge.
 */
import { hsluvToRgb, rgbToHsluv } from './hsluv.ts';
import { oklchRelToRgb, relChromaOf, rgbToOklch } from './oklch.ts';

export type PaletteMode = 'arc' | 'custom';

/**
 * Which cylindrical colour space the arcs and the hue rotation work in.
 *
 * Selectable rather than decided, because the three are genuinely different
 * looks and the project's deciding factor is which one looks good (CLAUDE.md):
 *
 * - `hsl` — no perceptual correction at all. Yellow is luminous and blue is
 *   dark at the same L, so a wheel of species has real hierarchy and real pop.
 *   The classic naive choice, and the most vivid of the three.
 * - `hsluv` — equalises perceived lightness across hues and rescales saturation
 *   so 100 is exactly the gamut boundary. Nothing clips and no species
 *   out-shouts another, at the cost of the punch HSL gets from letting yellow
 *   be brighter.
 * - `oklch` — the most perceptually uniform of the three, and the one with the
 *   best hue linearity: CIELUV, which HSLuv is built on, visibly bends blues
 *   toward purple, and Oklab does not. Its chroma is *relative* to what each hue
 *   can reach in sRGB (see `oklchRelToRgb` for why absolute chroma is unusable
 *   as a control), so like HSLuv it never clips.
 *
 * The same two numbers drive all three (`arc.sat`, `arc.light`, both 0..100).
 * They are not the same *quantity* — OKLCh's is chroma, not saturation — but
 * they occupy the same slider on purpose (user call, 2026-08-13): 0 is grey and
 * 100 is as colourful as that hue gets, which is true in every space here.
 */
export type PaletteSpace = 'hsl' | 'hsluv' | 'oklch';

export const PALETTE_SPACES: readonly PaletteSpace[] = ['hsl', 'hsluv', 'oklch'];

/**
 * The vuzic hue arc. `vuzic-app/src/vuzic/particle_life/ui.rs` computes
 * `hue = (base_hue + hue_spread * typ / num_types) * 360` and feeds it to
 * `hsluv_to_rgb(hue, 100, 50)`; `hueRangeDeg` is that `hue_spread` in degrees
 * and `hueStartDeg` is that `base_hue`. A negative range runs the arc backwards,
 * which is how direction is expressed rather than a separate boolean.
 */
export interface PaletteArc {
  /** hue of species 0, degrees; wrapped, so any real value is legal */
  hueStartDeg: number;
  /** total sweep across all K species, degrees; negative reverses direction */
  hueRangeDeg: number;
  /**
   * Saturation 0..100 — chroma in `oklch`. 100 is as colourful as the hue gets:
   * exactly the gamut boundary in HSLuv, gamut-mapped to it in OKLCh, and plain
   * full saturation in HSL. vuzic's value was 100.
   */
  sat: number;
  /** Lightness 0..100, shared across a group so no species reads as brighter. */
  light: number;
}

/**
 * Static per-species colour, art-directed once per track and never blended
 * (plan.md Revision 2). The palette lives *outside* θ, here and in
 * `ModulationConfig.palette`, as a single object shared by both.
 */
export interface Palette {
  /** which of `arc` / `colors` is authoritative */
  mode: PaletteMode;
  /**
   * The cylindrical space both arcs and the global hue rotation work in.
   *
   * Applies to arc generation and to the hue shift in custom mode. It does NOT
   * touch an authored hex at zero shift — those are sRGB strings and stay
   * exactly what was typed, in every space.
   */
  space: PaletteSpace;
  /**
   * The primaries' arc — species 0..groupSize-1.
   *
   * Always present, so the UI and the strict parsers never face an optional
   * block. On a substrate with no accent species this is the only arc there is,
   * because `groupSize` defaults to K and every species is then a primary.
   */
  arc: PaletteArc;
  /**
   * The accents' arc — species groupSize..2·groupSize-1, i.e. plife's four
   * accent voices, which are keyed to the same stems as its four primaries.
   *
   * A separate arc rather than an offset from `arc`: the accents are their own
   * colour family (user call, 2026-08-13). A single arc spread across all K got
   * this actively wrong — at K=8 with a 360° range, bass landed on hue 0 and
   * bass·accent on 180, its exact complement, so every "same instrument,
   * sparkling" pair read as two unrelated voices.
   *
   * Unused, but still serialised, when `groupSize` is K.
   */
  accentArc: PaletteArc;
  /**
   * per species, sRGB hex; index k is species k. Authoritative in custom mode.
   * In arc mode it is a derived cache kept in step by `syncPaletteColors` —
   * nothing on the render path reads it, so a stale entry cannot reach a pixel.
   * It is serialised in both modes so a palette is legible as data.
   *
   * Never carries `hueShiftDeg`: these are the base colours, and the shift is
   * applied on top of them at render time in both modes.
   */
  colors: string[];
  /** authored global rotation in the palette's hue space, degrees, both modes */
  hueShiftDeg: number;
  /** 0 = a fixed set of hues. Otherwise degrees per second of *simulation* time. */
  hueRateDegPerSec: number;
  /** 0 = greyscale, 1 = as authored, >1 pushes away from luminance (linear space) */
  saturation: number;
  /** global exposure trim on the per-species colours, 0..2 (linear space) */
  brightness: number;
}

/**
 * What `paletteLinear` uses when the palette has no entry for the index asked
 * for. A caller that has an authored default for that slot — physarum does,
 * `defaultPaletteColor` — passes it in; the neutral grey is what is left when
 * nobody does, and it is deliberately visible rather than black so a short
 * palette reads as a bug instead of as a missing species.
 */
export const FALLBACK_COLOR = '#808080';

/** Slider bound for the autonomous cycle. 10°/s is a 36-second lap: already fast. */
export const MAX_HUE_RATE_DEG_PER_SEC = 10;

/**
 * The arc a palette starts life with. Only reachable by switching a palette to
 * arc mode or loading a catalog entry — no shipped default palette is in arc
 * mode, because the shipped look must not change.
 *
 * S = 100 is vuzic's value and the reason HSLuv was chosen: it is the most
 * saturated colour that is still *inside* sRGB at that hue and lightness, so
 * nothing clips. L = 62 rather than vuzic's 50 because this project composites
 * into an HDR chain with a deep black point, where 50 reads muddy.
 */
export function defaultArc(): PaletteArc {
  return { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 62 };
}

/**
 * The accents' arc, out of the box: the primaries' wheel, lighter.
 *
 * L 80 against the primaries' 62 is an +18 lift, which is the middle of what the
 * shipped plife palette already does — its four accents sit +13 to +21 HSLuv
 * lightness above their primaries at the same S=100. Starting the two arcs
 * aligned means the default two-arc arc mode reads as "the same four voices,
 * sparkling", and pulling `accentArc.hueStartDeg` away from zero is then a
 * deliberate move toward a second colour family rather than the state you land in.
 */
export function defaultAccentArc(): PaletteArc {
  return { hueStartDeg: 0, hueRangeDeg: 360, sat: 100, light: 80 };
}

/**
 * Lightness added per group beyond the accents, and its ceiling.
 *
 * Two arcs cover the two groups that exist today (primaries, accents). A
 * substrate that one day runs K = 3·groupSize would otherwise render group 2
 * identically to group 1; lifting keeps them apart with the same convention the
 * catalog uses for entries shorter than K.
 */
export const GROUP_LIGHT_LIFT = 18;
export const GROUP_LIGHT_MAX = 92;

/**
 * Which arc species `index` belongs to, and how its lightness is adjusted.
 *
 * `groupSize` is the substrate's primary count — plife's `PRIMARY_COUNT`, or K
 * on a substrate whose species are all primaries. It is *not* serialised: it is
 * a property of the simulation's species layout, not of the art direction, and
 * the sim that reads a recipe back is the same sim that knows the constant. That
 * is what keeps a headless replay identical without adding a field a
 * hand-written recipe could contradict.
 */
export function arcForIndex(
  palette: Palette,
  index: number,
  groupSize: number,
): { arc: PaletteArc; hueIndex: number; light: number } {
  const size = Math.max(1, groupSize);
  const group = Math.floor(index / size);
  const arc = group === 0 ? palette.arc : palette.accentArc;
  const lift = group <= 1 ? 0 : (group - 1) * GROUP_LIGHT_LIFT;
  return {
    arc,
    hueIndex: index % size,
    light: Math.min(arc.light + lift, GROUP_LIGHT_MAX),
  };
}

/**
 * A v2 palette around an authored hex list — the shape every shipped default
 * uses, and the shape a v1 file lifts to losslessly. Everything the v1 renderer
 * did is reproduced exactly: mode `custom`, zero shift, zero rate.
 */
export function customPalette(colors: string[]): Palette {
  return {
    mode: 'custom',
    // HSLuv stays the default so nothing about an existing palette moves; the
    // space only matters in arc mode or under a non-zero hue shift anyway.
    space: 'hsluv',
    arc: defaultArc(),
    accentArc: defaultAccentArc(),
    colors,
    hueShiftDeg: 0,
    hueRateDegPerSec: 0,
    saturation: 1,
    brightness: 1,
  };
}

/** Degrees into [0, 360). */
export function wrapHue(degrees: number): number {
  const h = degrees % 360;
  return h < 0 ? h + 360 : h;
}

/**
 * A hue on one arc, before `hueShiftDeg`.
 *
 * `i * range / size` and not `i * range / (size - 1)`: the arc is a *fraction of
 * the wheel divided across the group*, so a full 360° range gives evenly spaced
 * hues with no duplicate at the seam. That is vuzic's formula and the reason a
 * 360° arc looks right rather than having its first and last species collide.
 *
 * `size` is the group's size, not K — with two arcs each spreads across its own
 * group, so the four primaries use the whole wheel and the four accents use the
 * whole wheel again rather than the eight of them sharing it.
 */
export function arcHue(arc: PaletteArc, index: number, groupSize: number): number {
  const size = Math.max(1, groupSize);
  return arc.hueStartDeg + (index * arc.hueRangeDeg) / size;
}

/**
 * The effective global shift at a given point on the simulation's clock.
 *
 * `simSeconds` is the substrate's own accumulated step time — never wall clock,
 * never `performance.now()`. A paused sim takes no steps, so the cycle stops
 * with the picture, and a headless export replaying from step zero walks the
 * identical sequence of shifts.
 */
export function paletteHuePhase(palette: Palette, simSeconds: number): number {
  return palette.hueShiftDeg + palette.hueRateDegPerSec * simSeconds;
}

function hexOf(rgb: [number, number, number]): string {
  const to = (v: number): string =>
    Math.round(Math.min(Math.max(v, 0), 1) * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${to(rgb[0])}${to(rgb[1])}${to(rgb[2])}`;
}

export function hsluvToHex(h: number, s: number, l: number): string {
  return hexOf(hsluvToRgb(wrapHue(h), s, l));
}

export function hexToHsluv(input: string): [number, number, number] {
  const [r, g, b] = parseSrgb(input);
  return rgbToHsluv(r, g, b);
}

/**
 * `(hue, sat, light)` -> hex, in whichever space the palette works in.
 *
 * `sat` and `light` are always 0..100 so one pair of sliders drives all three
 * spaces; each space scales them into its own units here and nowhere else.
 */
export function spaceToRgb(
  space: PaletteSpace,
  h: number,
  s: number,
  l: number,
): [number, number, number] {
  const hue = wrapHue(h);
  if (space === 'hsluv') return hsluvToRgb(hue, s, l);
  if (space === 'oklch') return oklchRelToRgb(l / 100, s / 100, hue);
  return hslToRgb(hue / 360, s / 100, l / 100);
}

export function spaceToHex(space: PaletteSpace, h: number, s: number, l: number): string {
  return hexOf(spaceToRgb(space, h, s, l));
}

/** The inverse of `spaceToHex`, same 0..100 conventions. */
export function hexToSpace(space: PaletteSpace, input: string): [number, number, number] {
  const [r, g, b] = parseSrgb(input);
  if (space === 'hsluv') return rgbToHsluv(r, g, b);
  if (space === 'oklch') {
    const [h, c, l] = rgbToOklch(r, g, b);
    return [h, relChromaOf(l, c, h) * 100, l * 100];
  }
  const [h, s, l] = rgbToHsl(r, g, b);
  return [h * 360, s * 100, l * 100];
}

/**
 * Rotate a colour's hue in the palette's space, preserving its other two
 * coordinates. This is custom mode's global hue shift, so which space it happens
 * in is the same choice the arcs make — rotating in HSL moves perceived
 * lightness and rotating in HSLuv does not, and the user picks which they want.
 */
export function rotateHueIn(space: PaletteSpace, hex: string, degrees: number): string {
  const [h, s, l] = hexToSpace(space, hex);
  return spaceToHex(space, h + degrees, s, l);
}

/** True when a shift is a whole number of turns, i.e. the identity. */
function isNoShift(degrees: number): boolean {
  return Math.abs(wrapHue(degrees)) < 1e-9 || Math.abs(wrapHue(degrees) - 360) < 1e-9;
}

/**
 * Species `index`'s colour as an sRGB hex, with the arc and the global shift
 * resolved but *not* the linear-space saturation/brightness trim.
 *
 * The zero-shift custom case short-circuits to the authored string with no
 * conversion at all. That is not an optimisation: it is what guarantees a v1
 * palette lifted to v2 renders bit-identically, since an HSLuv round trip is
 * only accurate to within a quantisation step.
 */
export function paletteHex(
  palette: Palette,
  index: number,
  fallback: string = FALLBACK_COLOR,
  phaseDeg = 0,
  speciesCount = palette.colors.length,
  groupSize = speciesCount,
): string {
  // Custom mode at zero shift returns the authored *string*, not a round trip
  // through `paletteSrgb`. `#ff8800` must come back as `#ff8800` and not as
  // whatever re-encoding a float produces, because this is the value the colour
  // pickers bind to and the value a modulation file serialises.
  if (palette.mode === 'custom' && isNoShift(phaseDeg)) {
    return palette.colors[index] ?? fallback;
  }
  return hexOf(paletteSrgb(palette, index, fallback, phaseDeg, speciesCount, groupSize));
}

/**
 * Species `index` as *unquantised* sRGB in 0..1 — the render path's entry point.
 *
 * `paletteHex` is the authoring and serialisation view of the same colour, and
 * it is 8 bits per channel because a hex string is. Nothing downstream needs
 * that: `paletteRgb` is a `Float32Array` uploaded straight to the GPU and the
 * scene target is `rgba16float`, so the palette is float from here to the
 * compositor. Rounding to a hex in the middle was a pure artefact of using the
 * authoring format as the interchange format, and in arc mode — where the colour
 * is *computed*, not typed — there is no reason to pay it.
 *
 * Custom mode still starts from the authored hex, because there the 8-bit value
 * is the ground truth rather than a rounding of one.
 */
export function paletteSrgb(
  palette: Palette,
  index: number,
  fallback: string = FALLBACK_COLOR,
  phaseDeg = 0,
  speciesCount = palette.colors.length,
  groupSize = speciesCount,
): [number, number, number] {
  if (palette.mode === 'arc') {
    const { arc, hueIndex, light } = arcForIndex(palette, index, groupSize);
    const hue = arcHue(arc, hueIndex, groupSize) + phaseDeg;
    return spaceToRgb(palette.space, hue, arc.sat, light);
  }
  const hex = palette.colors[index] ?? fallback;
  if (isNoShift(phaseDeg)) return parseSrgb(hex);
  const [h, s, l] = hexToSpace(palette.space, hex);
  return spaceToRgb(palette.space, h + phaseDeg, s, l);
}

/**
 * sRGB -> linear rgb. Accepts every notation tweakpane's colour picker can write
 * back into a string binding (#rgb, #rrggbb, #rrggbbaa, rgb()/rgba()); alpha is
 * ignored because per-species weight is the separate `intensity` knob.
 */
export function hexToLinear(input: string): [number, number, number] {
  return srgbToLinear(parseSrgb(input));
}

/**
 * sRGB 0..1 -> linear rgb, this project's convention.
 *
 * Plain `pow(c, 2.2)`, not the sRGB piecewise curve. That is what `hexToLinear`
 * has always done and what every authored colour in the repo was tuned against,
 * so the exact-curve version — correct in the abstract — would restyle every
 * shipped palette by a small amount for no gain the eye would call an
 * improvement. The one place the real curve is used is inside `oklch.ts`, where
 * it has to be, because Oklab is defined against it.
 */
export function srgbToLinear(srgb: [number, number, number]): [number, number, number] {
  return srgb.map((c) => Math.pow(Math.min(Math.max(c, 0), 1), 2.2)) as [number, number, number];
}

/**
 * Species `index`'s palette colour as linear rgb, with the palette's global
 * saturation and brightness folded in. Saturation is a lerp against luminance,
 * so 0 is greyscale and values > 1 extrapolate without changing hue.
 *
 * `phaseDeg` is the effective hue shift from `paletteHuePhase`; the substrates
 * pass their own sim clock through it. Defaulting it to 0 is what keeps every
 * v1-shaped call site (and every test) meaning exactly what it meant.
 */
export function paletteLinear(
  palette: Palette,
  index: number,
  fallback: string = FALLBACK_COLOR,
  phaseDeg = 0,
  speciesCount = palette.colors.length,
  groupSize = speciesCount,
): [number, number, number] {
  const [r, g, b] = srgbToLinear(
    paletteSrgb(palette, index, fallback, phaseDeg, speciesCount, groupSize),
  );
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const s = palette.saturation;
  const m = palette.brightness;
  return [
    Math.max(lum + (r - lum) * s, 0) * m,
    Math.max(lum + (g - lum) * s, 0) * m,
    Math.max(lum + (b - lum) * s, 0) * m,
  ];
}

/**
 * Bring the serialised `colors` cache back in step with the arc.
 *
 * Called after anything edits arc parameters, after a catalog load, and at parse
 * time, so a file on disk always shows the colours it will render. In custom
 * mode this only pads a short list, because there `colors` is the authority.
 */
export function syncPaletteColors(
  palette: Palette,
  speciesCount: number,
  fallback: (index: number) => string = () => FALLBACK_COLOR,
  groupSize = speciesCount,
): void {
  const k = Math.max(1, speciesCount);
  const next: string[] = [];
  for (let i = 0; i < k; i++) {
    next.push(
      palette.mode === 'arc'
        ? paletteHex(palette, i, FALLBACK_COLOR, 0, k, groupSize)
        : (palette.colors[i] ?? fallback(i)),
    );
  }
  // In place: the palette object is shared by reference with `ModulationConfig`
  // and bound by the panel's colour pickers, so it must not be replaced.
  palette.colors.length = 0;
  palette.colors.push(...next);
}

/**
 * Snap every authored colour to one shared HSLuv saturation and lightness while
 * keeping each hue — the custom-mode way to reach the arc's evenness without
 * giving up hand-chosen hues.
 *
 * The shared values default to the mean of what is already there, so one click
 * is a nudge toward consistency rather than a jump to somebody else's look. Pure
 * on `colors`; editing any picker afterwards undoes it for that species.
 */
export function harmonizePalette(palette: Palette, speciesCount: number): void {
  const k = Math.min(speciesCount, palette.colors.length);
  if (k < 1) return;
  const hsl = palette.colors.slice(0, k).map((c) => hexToSpace(palette.space, c));
  let sumS = 0;
  let sumL = 0;
  for (const [, s, l] of hsl) {
    sumS += s;
    sumL += l;
  }
  const meanS = sumS / k;
  const meanL = sumL / k;
  for (let i = 0; i < k; i++) {
    const [h] = hsl[i] as [number, number, number];
    palette.colors[i] = spaceToHex(palette.space, h, meanS, meanL);
  }
}

function parseSrgb(input: string): [number, number, number] {
  const s = input.trim();
  const hex = /^#?([0-9a-f]{3,8})$/i.exec(s);
  if (hex) {
    const d = hex[1] as string;
    if (d.length === 3 || d.length === 4) {
      const n = parseInt(d.slice(0, 3), 16);
      return [((n >> 8) & 15) / 15, ((n >> 4) & 15) / 15, (n & 15) / 15];
    }
    if (d.length === 6 || d.length === 8) {
      const n = parseInt(d.slice(0, 6), 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    }
  }
  const fn = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(s);
  if (fn) {
    return [Number(fn[1]) / 255, Number(fn[2]) / 255, Number(fn[3]) / 255];
  }
  return [0.5, 0.5, 0.5];
}

/**
 * HSL hue rotation, kept exactly as it was.
 *
 * This is *not* the palette's hue machinery — that is `rotateHueIn`. This is
 * physarum's shipped default hue walk (`defaultPaletteColor`, 41° steps), and
 * the two are different transforms that produce different hexes. The handoff
 * allowed migrating the walk to HSLuv only if the shipped look was provably
 * unchanged; it is not (HSL rotation moves perceived lightness and HSLuv does
 * not, which is the entire point of HSLuv), so the walk stays on HSL and this
 * function stays. Deleting it would silently restyle every default palette.
 */
export function rotateHue(hex: string, degrees: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1] as string, 16);
  const [h, s, l] = rgbToHsl(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
  const [r, g, b] = hslToRgb((h + degrees / 360) % 1, s, l);
  return hexOf([r, g, b]);
}

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return [0, 0, l];
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t: number): number => {
    let u = t;
    if (u < 0) u += 1;
    if (u > 1) u -= 1;
    if (u < 1 / 6) return p + (q - p) * 6 * u;
    if (u < 1 / 2) return q;
    if (u < 2 / 3) return p + (q - p) * (2 / 3 - u) * 6;
    return p;
  };
  return [f(h + 1 / 3), f(h), f(h - 1 / 3)];
}
