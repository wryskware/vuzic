/**
 * HSLuv ↔ sRGB, vendored.
 *
 * This is the reference HSLuv 1.0 algorithm (Alexei Boronine, MIT), transcribed
 * rather than depended on. Two reasons, both practical: `web/package.json`
 * carries exactly one runtime dependency that is not tweakpane, and the test
 * runner is bare `node --test tests/*.test.ts` with no bundler in the loop, so a
 * package that ships anything other than plain ESM+types is a fight for ~200
 * lines of arithmetic that will never change again. The handoff sanctions this.
 *
 * Why HSLuv at all (decided in `docs/handoffs/hsluv-palette-v2.md`, do not
 * reopen): every (H, S, L) triple is *inside* the sRGB gamut by construction —
 * S = 100 rides the gamut boundary for that hue and lightness rather than
 * clipping — and two hues at equal L have equal perceived lightness. That is
 * exactly the property "K species hues spread across an arc at one shared S/L"
 * needs, and it is what the user's earlier vuzic sims used.
 *
 * Ranges: H in [0, 360), S in [0, 100], L in [0, 100]. RGB is *non-linear* sRGB
 * in 0..1 — `sim/palette.ts` owns the linearisation.
 */

/** linear sRGB → CIE XYZ (D65), row-major. */
const M = [
  [3.240969941904521, -1.537383177570093, -0.498610760293],
  [-0.96924363628087, 1.87596750150772, 0.041555057407175],
  [0.055630079696993, -0.20397695888897, 1.056971514242878],
] as const;

const M_INV = [
  [0.41239079926595, 0.35758433938387, 0.18048078840183],
  [0.21263900587151, 0.71516867876775, 0.072192315360733],
  [0.019330818715591, 0.11919477979462, 0.95053215224966],
] as const;

const REF_U = 0.19783000664283;
const REF_V = 0.46831999493879;
/** CIE standard: κ and ε of the L* transfer function. */
const KAPPA = 903.2962962;
const EPSILON = 0.0088564516;

interface Line {
  slope: number;
  intercept: number;
}

/**
 * The six lines in the (u, v) chromaticity plane where one sRGB channel hits 0
 * or 1 at this lightness. The gamut boundary at L is their inner envelope, which
 * is what makes `maxChromaForLh` exact instead of a search.
 */
function bounds(l: number): Line[] {
  const out: Line[] = [];
  const sub1 = Math.pow(l + 16, 3) / 1560896;
  const sub2 = sub1 > EPSILON ? sub1 : l / KAPPA;
  for (const row of M) {
    const [m1, m2, m3] = row;
    for (let t = 0; t < 2; t++) {
      const top1 = (284517 * m1 - 94839 * m3) * sub2;
      const top2 = (838422 * m3 + 769860 * m2 + 731718 * m1) * l * sub2 - 769860 * t * l;
      const bottom = (632260 * m3 - 126452 * m2) * sub2 + 126452 * t;
      out.push({ slope: top1 / bottom, intercept: top2 / bottom });
    }
  }
  return out;
}

function lengthOfRayUntilIntersect(theta: number, line: Line): number {
  return line.intercept / (Math.sin(theta) - line.slope * Math.cos(theta));
}

/** Largest in-gamut chroma for this lightness and hue. */
function maxChromaForLh(l: number, h: number): number {
  const hrad = (h / 180) * Math.PI;
  let min = Number.POSITIVE_INFINITY;
  for (const line of bounds(l)) {
    const length = lengthOfRayUntilIntersect(hrad, line);
    if (length >= 0 && length < min) min = length;
  }
  return min;
}

function yToL(y: number): number {
  return y <= EPSILON ? y * KAPPA : 116 * Math.cbrt(y) - 16;
}

function lToY(l: number): number {
  return l <= 8 ? l / KAPPA : Math.pow((l + 16) / 116, 3);
}

function fromLinear(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function toLinear(c: number): number {
  return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}

function dot(row: readonly number[], v: readonly number[]): number {
  return (row[0] as number) * (v[0] as number) +
    (row[1] as number) * (v[1] as number) +
    (row[2] as number) * (v[2] as number);
}

function xyzToRgb(xyz: readonly number[]): [number, number, number] {
  return [fromLinear(dot(M[0], xyz)), fromLinear(dot(M[1], xyz)), fromLinear(dot(M[2], xyz))];
}

function rgbToXyz(rgb: readonly number[]): [number, number, number] {
  const lin = [toLinear(rgb[0] as number), toLinear(rgb[1] as number), toLinear(rgb[2] as number)];
  return [dot(M_INV[0], lin), dot(M_INV[1], lin), dot(M_INV[2], lin)];
}

function xyzToLuv(xyz: readonly number[]): [number, number, number] {
  const [x, y, z] = xyz as [number, number, number];
  const divider = x + 15 * y + 3 * z;
  const l = yToL(y);
  if (l === 0 || divider === 0) return [l, 0, 0];
  return [l, 13 * l * ((4 * x) / divider - REF_U), 13 * l * ((9 * y) / divider - REF_V)];
}

function luvToXyz(luv: readonly number[]): [number, number, number] {
  const [l, u, v] = luv as [number, number, number];
  if (l === 0) return [0, 0, 0];
  const varU = u / (13 * l) + REF_U;
  const varV = v / (13 * l) + REF_V;
  const y = lToY(l);
  const x = -(9 * y * varU) / ((varU - 4) * varV - varU * varV);
  const z = (9 * y - 15 * varV * y - varV * x) / (3 * varV);
  return [x, y, z];
}

function luvToLch(luv: readonly number[]): [number, number, number] {
  const [l, u, v] = luv as [number, number, number];
  const c = Math.sqrt(u * u + v * v);
  if (c < 1e-8) return [l, 0, 0];
  let h = (Math.atan2(v, u) * 180) / Math.PI;
  if (h < 0) h += 360;
  return [l, c, h];
}

function lchToLuv(lch: readonly number[]): [number, number, number] {
  const [l, c, h] = lch as [number, number, number];
  const hrad = (h / 180) * Math.PI;
  return [l, Math.cos(hrad) * c, Math.sin(hrad) * c];
}

/** HSLuv (h 0..360, s 0..100, l 0..100) → non-linear sRGB 0..1. */
export function hsluvToRgb(h: number, s: number, l: number): [number, number, number] {
  let c: number;
  if (l > 99.9999999) {
    c = 0;
    l = 100;
  } else if (l < 0.00000001) {
    c = 0;
    l = 0;
  } else {
    c = (maxChromaForLh(l, h) / 100) * s;
  }
  return xyzToRgb(luvToXyz(lchToLuv([l, c, h])));
}

/** Non-linear sRGB 0..1 → HSLuv. */
export function rgbToHsluv(r: number, g: number, b: number): [number, number, number] {
  const [l, c, h] = luvToLch(xyzToLuv(rgbToXyz([r, g, b])));
  if (l > 99.9999999) return [h, 0, 100];
  if (l < 0.00000001) return [h, 0, 0];
  return [h, (c / maxChromaForLh(l, h)) * 100, l];
}
