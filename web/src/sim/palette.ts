/**
 * Colour, as every simulation needs it: the palette shape, the sRGB parsing, and
 * the hue walk that keeps a large K distinguishable.
 *
 * None of this is physarum's. A palette is "K colours plus a global saturation
 * and brightness trim", which is true of any substrate that draws K things, and
 * the mapping layer serialises it (`ModulationConfig.palette`) without knowing
 * whose colours they are. What *is* physarum's — the shipped hexes, the species
 * templates, `defaultPalette` — stays in `physarum/config.ts`, because that is
 * art direction for one sim rather than machinery for all of them.
 */

/**
 * Static per-species colour, art-directed once per track and never blended
 * (plan.md Revision 2). Modulating hue muddied the image and made species
 * impossible to follow; the palette therefore lives *outside* θ, here and in
 * `ModulationConfig.palette`, as a single object shared by both.
 */
export interface Palette {
  /** per species, sRGB hex; index k is species k. Shorter than K falls back to the default. */
  colors: string[];
  /** 0 = greyscale, 1 = as authored, >1 pushes away from luminance */
  saturation: number;
  /** global exposure trim on the per-species colours, 0..2 */
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

/**
 * sRGB -> linear rgb. Accepts every notation tweakpane's colour picker can write
 * back into a string binding (#rgb, #rrggbb, #rrggbbaa, rgb()/rgba()); alpha is
 * ignored because per-species weight is the separate `intensity` knob.
 */
export function hexToLinear(input: string): [number, number, number] {
  const srgb = parseSrgb(input);
  return srgb.map((c) => Math.pow(Math.min(Math.max(c, 0), 1), 2.2)) as [number, number, number];
}

/**
 * Species `index`'s palette colour as linear rgb, with the palette's global
 * saturation and brightness folded in. Saturation is a lerp against luminance,
 * so 0 is greyscale and values > 1 extrapolate without changing hue.
 */
export function paletteLinear(
  palette: Palette,
  index: number,
  fallback: string = FALLBACK_COLOR,
): [number, number, number] {
  const hex = palette.colors[index] ?? fallback;
  const [r, g, b] = hexToLinear(hex);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const s = palette.saturation;
  const m = palette.brightness;
  return [
    Math.max(lum + (r - lum) * s, 0) * m,
    Math.max(lum + (g - lum) * s, 0) * m,
    Math.max(lum + (b - lum) * s, 0) * m,
  ];
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

export function rotateHue(hex: string, degrees: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1] as string, 16);
  const [h, s, l] = rgbToHsl(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
  const [r, g, b] = hslToRgb((h + degrees / 360) % 1, s, l);
  const to = (v: number) =>
    Math.round(Math.min(Math.max(v, 0), 1) * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${to(r)}${to(g)}${to(b)}`;
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
