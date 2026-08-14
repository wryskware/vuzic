/**
 * OKLCh — Björn Ottosson's Oklab in cylindrical form, plus the gamut mapping a
 * fixed-chroma arc needs.
 *
 * Why this exists alongside `hsluv.ts`: the two solve the same problem with
 * opposite biases, and which one looks better is a question about the picture
 * rather than about colour science.
 *
 * - **HSLuv** rescales saturation so that 100 is *always* exactly the sRGB gamut
 *   boundary, and equalises perceived lightness across hues. Nothing ever
 *   clips and no species out-shouts another, but the correction deliberately
 *   darkens yellow and lightens blue, which is most of what "vivid" means.
 * - **OKLCh** keeps chroma an absolute quantity. A fixed chroma is genuinely the
 *   same colourfulness at every hue — but the sRGB gamut is not a cylinder, so
 *   some (L, C, h) simply do not exist and have to be mapped back in.
 *
 * The mapping here reduces chroma toward the achromatic axis at constant L and
 * h, which is the standard approach (CSS Color 4 does the same thing) and the
 * only one that preserves the hue the arc asked for. Clamping RGB instead is
 * cheaper and wrong: it shifts hue, and on a wheel of K species that shows up as
 * two neighbouring species converging on the same colour near the gamut corners.
 */

/** Linear-light sRGB component -> encoded sRGB. The real piecewise curve. */
function encodeSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Encoded sRGB component -> linear light. */
function decodeSrgb(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Oklab -> linear sRGB. Ottosson's matrices, unmodified. */
function oklabToLinear(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

/** Linear sRGB -> Oklab. */
function linearToOklab(r: number, g: number, b: number): [number, number, number] {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

/** A hair of slack, so a colour exactly on the boundary is not judged outside. */
const GAMUT_EPS = 1e-4;

function inGamut(rgb: [number, number, number]): boolean {
  return rgb.every((c) => c >= -GAMUT_EPS && c <= 1 + GAMUT_EPS);
}

/**
 * OKLCh -> sRGB in 0..1, gamut-mapped by chroma reduction at constant L and hue.
 *
 * `L` is 0..1, `chroma` is absolute (sRGB tops out near 0.37 and only at a few
 * hues), `hueDeg` is degrees. When the requested chroma is unreachable the
 * result is the most colourful colour that *is* reachable at that lightness and
 * hue — so an arc at a high fixed chroma still spans the wheel evenly, it just
 * rides the gamut hull where it has to, which is what HSLuv does by construction.
 */
export function oklchToRgb(L: number, chroma: number, hueDeg: number): [number, number, number] {
  const lightness = Math.min(Math.max(L, 0), 1);
  const rad = (hueDeg * Math.PI) / 180;
  const ca = Math.cos(rad);
  const sa = Math.sin(rad);
  const at = (c: number): [number, number, number] => oklabToLinear(lightness, c * ca, c * sa);

  let hi = Math.max(chroma, 0);
  let rgb = at(hi);
  if (!inGamut(rgb)) {
    // Chroma 0 is the grey axis and is always inside, so this bisection always
    // has a valid low end and terminates on a colour we can actually show.
    let lo = 0;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      if (inGamut(at(mid))) lo = mid;
      else hi = mid;
    }
    rgb = at(lo);
  }
  return rgb.map((c) => Math.min(Math.max(encodeSrgb(Math.min(Math.max(c, 0), 1)), 0), 1)) as [
    number,
    number,
    number,
  ];
}

/** sRGB 0..1 -> OKLCh as `[hueDeg, chroma, L]`, ordered to match the H/S/L convention. */
export function rgbToOklch(r: number, g: number, b: number): [number, number, number] {
  const [L, a, bb] = linearToOklab(decodeSrgb(r), decodeSrgb(g), decodeSrgb(b));
  const chroma = Math.sqrt(a * a + bb * bb);
  let hue = (Math.atan2(bb, a) * 180) / Math.PI;
  if (hue < 0) hue += 360;
  return [hue, chroma, L];
}

/**
 * The largest chroma sRGB can show at this lightness and hue.
 *
 * Measured, not assumed, because the sRGB solid is nothing like a cylinder: at
 * L 0.62 this runs from 0.105 at hue 200 (cyan-blue) to 0.305 at hue 315
 * (magenta), a 3x spread.
 */
export function maxChromaFor(L: number, hueDeg: number): number {
  const lightness = Math.min(Math.max(L, 0), 1);
  const rad = (hueDeg * Math.PI) / 180;
  const ca = Math.cos(rad);
  const sa = Math.sin(rad);
  let lo = 0;
  let hi = 0.5; // beyond anything sRGB reaches at any hue or lightness
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2;
    if (inGamut(oklabToLinear(lightness, mid * ca, mid * sa))) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * OKLCh with the slider's 0..100 read as a *fraction of what this hue can do*.
 *
 * This is what the arc actually uses, and it is deliberately not absolute
 * chroma. Absolute chroma is the theoretically nicer quantity — the same number
 * really is the same colourfulness at every hue — but it is unusable as a
 * control here: the value that keeps every hue inside sRGB at L 0.62 is 0.105,
 * while magenta alone reaches 0.305, so a slider scaled to the maximum spends
 * most of its travel gamut-mapping and the top quarter of it does nothing at
 * all. Measured on the shipped default: at a nominal 100, *every* hue on the
 * wheel was out of gamut, and 0% of them were inside BT.2020 either.
 *
 * Relative chroma makes 100 mean "as colourful as this hue gets", which is the
 * promise the UI makes and the same one HSLuv's S=100 keeps. What OKLCh still
 * brings over HSLuv is its far better hue linearity — CIELUV, which HSLuv is
 * built on, visibly bends blues toward purple as lightness changes — and a more
 * even lightness response. Those are the reasons to pick it; equal absolute
 * chroma across hues was never reachable in sRGB at a useful level anyway.
 */
export function oklchRelToRgb(L: number, rel: number, hueDeg: number): [number, number, number] {
  const amount = Math.min(Math.max(rel, 0), 1);
  return oklchToRgb(L, amount * maxChromaFor(L, hueDeg), hueDeg);
}

/** Relative chroma of an sRGB colour: the inverse of `oklchRelToRgb`'s scaling. */
export function relChromaOf(L: number, chroma: number, hueDeg: number): number {
  const max = maxChromaFor(L, hueDeg);
  return max > 1e-6 ? Math.min(chroma / max, 1) : 0;
}
