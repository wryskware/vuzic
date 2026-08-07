import { defaultRenderConfig, type RenderConfig } from '../render/config.ts';

/**
 * Ceiling on the *base* deposit: deposit * (1 + stemGain * stem). Matches the
 * `deposit` slider max, so nothing reachable without stem drive is clipped.
 *
 * Event impulses are deliberately NOT bounded by this — see MAX_DEPOSIT.
 */
export const MAX_EFFECTIVE_DEPOSIT = 6;

/**
 * Hard ceiling on what one agent can write to one cell in one tick, after the
 * impulse burst (CPU, physarum.ts) and after soil fertility (GPU, agents.wgsl).
 * Both sites clamp to this and nothing else may exceed it — the i32 atomic
 * headroom is derived from it alone.
 *
 * 7x MAX_EFFECTIVE_DEPOSIT, which is the whole reachable CPU-side range: the
 * impulse multiplier is 1 + response.deposit * envelope and the panel caps
 * response.deposit at 6, so a fully stem-saturated species firing a full-strength
 * event asks for exactly 7x and gets all of it. Revision 2 §3 wants events
 * legible, so the burst gets real headroom instead of sharing the base clamp.
 *
 * Soil fertility (up to 4x, panel-capped depositBias 3) multiplies on the GPU on
 * top of that and is clamped into the same ceiling, so it is only ever clipped in
 * the corner where burst and stem are both already maxed.
 */
export const MAX_DEPOSIT = 7 * MAX_EFFECTIVE_DEPOSIT;

export interface AdaptiveTriple {
  /** constant term */
  p1: number;
  /** gain on the intensity term */
  p2: number;
  /** exponent on the intensity term */
  p3: number;
}

export interface SpeciesConfig {
  name: string;
  /**
   * Modulatable light, 0..2, default 1. Revision 2: hue is static art direction,
   * *brightness* is what the music moves. Multiplies `intensity` in the
   * compositor only — deliberately not wired into deposit, so it changes how the
   * world looks without changing how it grows.
   */
  brightness: number;
  intensity: number;
  deposit: number;
  /** multiplicative per tick; tau ~= 1/(1-decay) ticks of memory */
  decay: number;
  aliveFraction: number;
  /** 3x3 blur centre weight: 1/9 = box blur (widest), 1 = no blur */
  diffuseCentre: number;
  sensorDist: AdaptiveTriple;
  sensorAngle: AdaptiveTriple;
  rotate: AdaptiveTriple;
  moveDist: AdaptiveTriple;
}

/**
 * Static colour, art-directed once per track and never blended (plan.md
 * Revision 2). Blending hue per anchor muddied the image and made species
 * impossible to follow; the palette therefore lives *outside* θ, here and in
 * `MappingConfig.palette`, as a single object shared by both.
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
 * Track-scale memory (plan.md Decision 3). One shared r32float field at trail
 * resolution, accumulated from trail presence and decayed a hair per tick, that
 * biases where agents go and how hard they deposit.
 *
 * **Shared, not per-species** — deliberately. A per-species soil is arithmetically
 * just a slower trail, and the trail already carries per-species memory via its own
 * decay; duplicating it per species buys a longer τ and nothing else. A *shared*
 * soil is a different object: it is terrain. Whatever species carved a network
 * there leaves ground that every species finds easier, so a returning section
 * re-lights old scars even when a different instrument is playing them. It also
 * costs 1/K the memory and folds into one grid-sized branch instead of K.
 *
 * These knobs are structural, like depositScale and gridScale — they are NOT part
 * of the blended θ, so the mapper never moves them.
 */
export interface SoilConfig {
  /** multiplicative per tick; τ ≈ 1/(1-decay) ticks. 0.999 ≈ 17 s at 60 fps */
  decay: number;
  /** per-tick gain on squashed trail presence; equilibrium ≈ accum/(1-decay) */
  accum: number;
  /** deposit × (1 + depositBias · soil). Bounded ≤ 3 so the i32 atomic keeps headroom. */
  depositBias: number;
  /** sensed field × (1 + senseBias · soil) at each sample point — old ground reads louder */
  senseBias: number;
  /** render the soil field instead of the trails */
  debugView: boolean;
}

export interface PhysarumConfig {
  /**
   * Phase 7: exposure/tone/bloom/feedback. Structural like `soil` — outside θ, so
   * nothing here is blended between anchors, but it *is* saved with the mapping.
   */
  render: RenderConfig;
  /** K. Any value >= 1; bounded only by maxTextureArrayLayers (>= 256 everywhere). */
  speciesCount: number;
  /** pool size before rounding down to a whole number per species */
  maxAgents: number;
  /** sim grid = canvas device pixels * gridScale, clamped to maxGridDim */
  gridScale: number;
  maxGridDim: number;
  /**
   * Fixed-point scale for the i32 deposit atomics. 1.25e4, not the 1e7 the plan
   * suggests: headroom matters more than resolution here. Overflowing i32 wraps
   * the trail negative and blacks the species out, so the bound is sized off the
   * worst case one agent can write to one cell in one tick, which is MAX_DEPOSIT
   * (42) — impulse burst and soil fertility are both clamped into it, on the CPU
   * and on the GPU respectively, and nothing else scales the deposit.
   * 2^31 / (42 * 1.25e4) = ~4000 agents/cell/tick, while the 8e-5 quantum is
   * still ~1e-5 of a typical trail value.
   */
  depositScale: number;
  /** x = 1 - exp(-trail * senseGain); sets where the adaptive curves saturate */
  senseGain: number;
  exposure: number;
  gamma: number;
  /** sim steps per clock tick, accumulated so fractional values work */
  speed: number;
  paused: boolean;
  /** phase-5 preview: stem k scales species k's deposit */
  stemDrive: boolean;
  stemGain: number;
  /** track-scale memory; not part of θ */
  soil: SoilConfig;
  /** static per-species colour; not part of θ */
  palette: Palette;
  species: SpeciesConfig[];
  /** K*K row-major; M[i*K+j] = how much species i is drawn to species j's trail */
  matrix: number[];
}

interface Template extends Omit<SpeciesConfig, 'name' | 'brightness'> {
  name: string;
  /** palette entry, not a species field — see `defaultPalette` */
  colorHex: string;
}

const TEMPLATES: Template[] = [
  {
    name: 'bass',
    colorHex: '#ff7a1a',
    intensity: 1.3,
    deposit: 1.2,
    decay: 0.955,
    aliveFraction: 0.85,
    diffuseCentre: 0.12,
    sensorDist: { p1: 13, p2: 9, p3: 1.0 },
    sensorAngle: { p1: 0.38, p2: 0.22, p3: 1.0 },
    rotate: { p1: 0.12, p2: 0.3, p3: 1.0 },
    moveDist: { p1: 1.0, p2: 0.6, p3: 1.0 },
  },
  {
    name: 'drums',
    colorHex: '#ff2f6d',
    intensity: 1.0,
    deposit: 1.5,
    decay: 0.88,
    aliveFraction: 0.6,
    diffuseCentre: 0.55,
    sensorDist: { p1: 4, p2: 3, p3: 1.5 },
    sensorAngle: { p1: 0.62, p2: 0.5, p3: 2.0 },
    rotate: { p1: 0.45, p2: 0.9, p3: 1.0 },
    moveDist: { p1: 1.7, p2: 1.1, p3: 1.0 },
  },
  {
    name: 'vocals',
    colorHex: '#35d6ff',
    intensity: 1.15,
    deposit: 1.0,
    decay: 0.93,
    aliveFraction: 0.55,
    diffuseCentre: 0.3,
    sensorDist: { p1: 7, p2: 5, p3: 0.7 },
    sensorAngle: { p1: 0.44, p2: 0.26, p3: 1.0 },
    rotate: { p1: 0.24, p2: 0.5, p3: 1.0 },
    moveDist: { p1: 1.1, p2: 0.8, p3: 1.0 },
  },
  {
    name: 'other',
    colorHex: '#a56bff',
    intensity: 1.0,
    deposit: 0.85,
    decay: 0.92,
    aliveFraction: 0.45,
    diffuseCentre: 0.42,
    sensorDist: { p1: 5, p2: 11, p3: 1.6 },
    sensorAngle: { p1: 0.28, p2: 0.55, p3: 1.5 },
    rotate: { p1: 0.32, p2: 0.45, p3: 2.0 },
    moveDist: { p1: 1.25, p2: 0.55, p3: 1.0 },
  },
];

/** Self-attraction plus small asymmetric cross terms. Row i = what species i senses. */
const DEFAULT_MATRIX_4 = [
  1.0, 0.15, -0.1, 0.05,
  0.25, 1.0, 0.1, -0.2,
  -0.15, 0.2, 1.0, 0.3,
  0.1, -0.25, 0.35, 1.0,
];

function cloneTriple(t: AdaptiveTriple): AdaptiveTriple {
  return { p1: t.p1, p2: t.p2, p3: t.p3 };
}

/** Species k's authored colour. The hue walk keeps K > 4 distinguishable. */
export function defaultPaletteColor(index: number): string {
  const t = TEMPLATES[index % TEMPLATES.length] as Template;
  const cycle = Math.floor(index / TEMPLATES.length);
  return cycle === 0 ? t.colorHex : rotateHue(t.colorHex, cycle * 41);
}

export function defaultPalette(k: number): Palette {
  return {
    colors: Array.from({ length: Math.max(1, k) }, (_, i) => defaultPaletteColor(i)),
    saturation: 1,
    brightness: 1,
  };
}

export function defaultSpecies(index: number): SpeciesConfig {
  const t = TEMPLATES[index % TEMPLATES.length] as Template;
  const cycle = Math.floor(index / TEMPLATES.length);
  return {
    name: cycle === 0 ? t.name : `${t.name}${cycle + 1}`,
    brightness: 1,
    intensity: t.intensity,
    deposit: t.deposit,
    decay: t.decay,
    aliveFraction: t.aliveFraction,
    diffuseCentre: t.diffuseCentre,
    sensorDist: cloneTriple(t.sensorDist),
    sensorAngle: cloneTriple(t.sensorAngle),
    rotate: cloneTriple(t.rotate),
    moveDist: cloneTriple(t.moveDist),
  };
}

export function defaultMatrix(k: number): number[] {
  if (k === 4) return DEFAULT_MATRIX_4.slice();
  const m = new Array<number>(k * k).fill(0);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      if (i === j) m[i * k + j] = 1.0;
      else if (j === (i + 1) % k) m[i * k + j] = 0.2;
      else if (j === (i + k - 1) % k) m[i * k + j] = -0.15;
    }
  }
  return m;
}

/**
 * Defaults sized to be *legible*, not subtle (plan.md Revision 2 §3): at 0.999 the
 * field remembers ~17 s of activity, and the biases reach +45% deposit / +70% sense
 * on fully grown soil — enough that minute three visibly re-uses minute one's
 * channels.
 *
 * accum defaults to exactly 1 - decay, which makes soil a true EMA of presence:
 * a permanently busy cell converges on 1.0 and never clips. Measured: pushing accum
 * above that (0.0015 was the first try) drives ~99% of the grid to the clamp inside
 * 30 s and the field turns into a flat wash with no scars in it. Raise it only to
 * make soil *appear* faster, and expect to lose contrast for it.
 */
export function defaultSoil(): SoilConfig {
  return {
    decay: 0.999,
    accum: 0.001,
    depositBias: 0.45,
    senseBias: 0.7,
    debugView: false,
  };
}

export function defaultConfig(speciesCount = 4): PhysarumConfig {
  const k = Math.max(1, Math.floor(speciesCount));
  return {
    speciesCount: k,
    maxAgents: 1048576,
    gridScale: 0.5,
    maxGridDim: 1600,
    depositScale: 1.25e4,
    senseGain: 0.1,
    // Phase 7: scene exposure feeds an HDR surface and auto-exposure adapts on
    // top of it, so this only has to put the frame in the right decade. 0.01
    // lands the adapted gain near 1 on Free Fall, which keeps the controller's
    // rails (0.05 … 16) far away in both directions. The old 0.04 was tuned
    // against ACES writing straight to an 8-bit swapchain.
    exposure: 0.01,
    gamma: 2.2,
    speed: 1,
    paused: false,
    stemDrive: false,
    stemGain: 1.5,
    render: defaultRenderConfig(),
    soil: defaultSoil(),
    palette: defaultPalette(k),
    species: Array.from({ length: k }, (_, i) => defaultSpecies(i)),
    matrix: defaultMatrix(k),
  };
}

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
export function paletteLinear(palette: Palette, index: number): [number, number, number] {
  const hex = palette.colors[index] ?? defaultPaletteColor(index);
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
