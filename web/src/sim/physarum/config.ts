import { defaultRenderConfig, type RenderConfig } from '../render/config.ts';
import { customPalette, rotateHue, type Palette } from '../palette.ts';

// Colour machinery moved to `sim/palette.ts` when the mapping layer stopped
// depending on physarum concretely: it is shared by every sim, whereas the
// hexes below are this one's art direction. Re-exported so the existing
// importers (physarum.ts, mapping/persist.ts) do not have to care where it
// lives — new code should import from '../palette.ts' directly.
export { hexToLinear, paletteLinear } from '../palette.ts';
export type { Palette } from '../palette.ts';

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

/**
 * Ceiling on `brightness × stem-follow`, matching the `brightness` slider max.
 * Both factors are already bounded (slider ≤ 2, follow ≤ 1) so this only bites
 * if a loaded file carries something out of range — but the composition order in
 * `uploadSpecies` is load-bearing, so it clamps where it composes.
 *
 * Impulse flashes multiply on top of this and are deliberately NOT bounded by
 * it, exactly as deposit bursts are not bounded by MAX_EFFECTIVE_DEPOSIT: a
 * flash has to stay legible on a species the stem lane is currently dimming.
 */
export const MAX_BRIGHTNESS = 2;

/**
 * Hard bounds on `SpeciesConfig.scale`. One table, read by the θ slot table
 * (`mapping/preset.ts`), the panel slider and the clamp in `uploadSpecies`, so a
 * loaded file, a seeded personality and a slider can never disagree about how far
 * this may go. ×4 already turns a species into three or four fat channels across
 * the frame; ×0.25 is the point below which the trail is thinner than the blur
 * kernel and the network stops resolving at all.
 */
export const MIN_SPECIES_SCALE = 0.25;
export const MAX_SPECIES_SCALE = 4;

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
   * Per-species light, 0..2, default 1. Multiplies `intensity` in the compositor
   * only — deliberately not wired into deposit, so it changes how the world looks
   * without changing how it grows.
   *
   * **Revision 4:** no longer modulated by the embedding (it flashed constantly).
   * This is now the *base* that the stem-follow lane scales — with stem-follow
   * off it is absolute again, exactly as in phase 4.
   */
  brightness: number;
  intensity: number;
  deposit: number;
  /** multiplicative per tick; tau ~= 1/(1-decay) ticks of memory */
  decay: number;
  aliveFraction: number;
  /** 3x3 blur centre weight: 1/9 = box blur (widest), 1 = no blur */
  diffuseCentre: number;
  /**
   * This species' structural scale, 0.25..4, default 1. One multiplier on BOTH
   * the sensorDist curve and the moveDist curve, so the whole geometry of the
   * network — how far an agent looks and how far it then walks — moves together
   * rather than drifting apart.
   *
   * It exists because sensorDist and moveDist are six independent θ slots and the
   * modulator wires each one to its own random projection: they move, but they
   * move *incoherently*, which cancels out into a network whose apparent scale
   * never changes across a track. Scaling the pair by one number is the only
   * shape of control that reads as "this species got coarser".
   *
   * Not to be confused with the `reach`/`agility` macros: those are global
   * performance trims outside θ and multiply on top of this. This one is
   * per-species character, inside θ, and the music may move it.
   */
  scale: number;
  sensorDist: AdaptiveTriple;
  sensorAngle: AdaptiveTriple;
  rotate: AdaptiveTriple;
  moveDist: AdaptiveTriple;
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
 * in the modulation registry, so the music never moves them.
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

/**
 * The performance layer: five always-yours multipliers that compose OUTSIDE θ,
 * at exactly the points where stem-follow and the impulse lanes already compose.
 * That placement is the whole design — it is what makes it impossible for the
 * modulator to write over them, because the modulator's only reach is θ and
 * these are applied after θ has been read. 1.0 everywhere = neutral.
 *
 * They exist because there was no rung between "driver gains" (which change what
 * the music *says*, indirectly, and differently on every seed) and the hundreds
 * of fine θ sliders (which the modulator overwrites on the next tick). A macro is
 * neither: it survives every reroll, every load and every tick, and it means the
 * same thing on every seed.
 *
 * Physarum has no per-species force term and no primary/accent partition, so
 * plife's `force` and `accents` have no counterpart here. `deposit` is physarum's
 * "energy into the system" knob — the same role plife's `force` plays, in this
 * substrate's currency.
 */
export interface PhysarumMacros {
  /** × every species' aliveFraction (clamped back into 0..1 after) */
  density: number;
  /** × every species' deposit, inside the MAX_EFFECTIVE_DEPOSIT clamp */
  deposit: number;
  /** × the sensorDist curve, p1 and p2 both — the same lane the impulse sensorMul uses */
  reach: number;
  /** × the moveDist curve, p1 and p2 */
  agility: number;
  /** × render.feedback.amount (clamped ≤ 0.95, physarum's ceiling) */
  trails: number;
}

export function defaultPhysarumMacros(): PhysarumMacros {
  return {
    density: 1,
    deposit: 1,
    reach: 1,
    agility: 1,
    trails: 1,
  };
}

/**
 * Panel range per macro, and the same table `PhysarumSim.applyExtras` clamps a
 * loaded file into. One source, so a saved value can never sit outside the
 * slider that is supposed to show it.
 */
export const PHYSARUM_MACRO_RANGE: Readonly<
  Record<keyof PhysarumMacros, { min: number; max: number }>
> = {
  density: { min: 0, max: 2 },
  deposit: { min: 0, max: 2 },
  reach: { min: 0, max: 2 },
  agility: { min: 0, max: 2 },
  trails: { min: 0, max: 1.5 },
};

/** Order the macro folder lists them in, with the wiring stated in the label. */
export const PHYSARUM_MACRO_LABELS: readonly { key: keyof PhysarumMacros; label: string }[] = [
  { key: 'density', label: 'density  (× all alive fractions)' },
  { key: 'deposit', label: 'deposit  (× all deposit rates)' },
  { key: 'reach', label: 'reach  (× sensor distance)' },
  { key: 'agility', label: 'agility  (× move distance)' },
  { key: 'trails', label: 'trails  (× echo persistence)' },
];

export interface PhysarumConfig {
  /**
   * Phase 7: exposure/tone/bloom/feedback. Structural like `soil` — outside θ, so
   * nothing here is modulated, but it *is* saved with the modulation config.
   */
  render: RenderConfig;
  /** the performance layer: five multipliers the modulator never writes. Outside θ. */
  macros: PhysarumMacros;
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
  /** sim steps per 60 Hz model tick, accumulated so fractional values work */
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

// `scale` is omitted alongside `brightness`: both ship at 1 for every template
// (the authored character lives in the curves themselves, and scale is the knob
// that moves them afterwards), so listing it four times would only be four more
// places for a future edit to disagree with itself.
interface Template extends Omit<SpeciesConfig, 'name' | 'brightness' | 'scale'> {
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

/**
 * Custom mode, not an arc, and that is the whole migration story for the shipped
 * default: `customPalette` is exactly the v1 palette with the v2 fields at their
 * neutral values (zero shift, zero rate), and `paletteHex` short-circuits a
 * zero-shift custom palette straight to the authored string. The default look is
 * therefore bit-identical to what it was before palette v2 existed.
 */
export function defaultPalette(k: number): Palette {
  return customPalette(Array.from({ length: Math.max(1, k) }, (_, i) => defaultPaletteColor(i)));
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
    scale: 1,
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
    macros: defaultPhysarumMacros(),
    soil: defaultSoil(),
    palette: defaultPalette(k),
    species: Array.from({ length: k }, (_, i) => defaultSpecies(i)),
    matrix: defaultMatrix(k),
  };
}
