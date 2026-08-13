/**
 * The declarative θ table a vizfx *visual* is made of, and everything the rest
 * of the app derives from it.
 *
 * ## Why a table rather than a config interface
 *
 * Physarum and particle life each hand-write a registry (`mapping/preset.ts`,
 * `sim/plife/preset.ts`): a slot table, a `Preset` interface, a
 * `presetToVector` / `applyVector` pair and a set of offset helpers, all of
 * which have to be kept in step with a `Config` interface and with a panel that
 * names the same fields a third time. That is the right shape when θ *means*
 * something structurally different per slot — plife's three K² blocks are not a
 * flat list of scalars and pretending otherwise would lose the partition.
 *
 * A screen-space feedback visual has no such structure. Its θ is a flat list of
 * scalars: some per *layer*, the rest global. So the table below is the only
 * place a slot is named, and five things are generated from it:
 *
 *   1. `ThetaRegistry` — names, classes, mask, ModSpecs (mapping layer)
 *   2. the live parameter store — a `Record<string, number>` keyed by slot name
 *   3. the vector ⇄ store conversion (`paramsToVector` / `applyVector`)
 *   4. the **effective** θ the GPU is handed, with the macro rig folded in
 *   5. a **WGSL prelude** of `TH_*` index constants, so the shader addresses θ
 *      by name without anyone maintaining a second copy of the layout
 *
 * The intended cost of a second visual is therefore one WGSL pair plus one
 * table. Nothing in `vizfx.ts` names a nebula parameter.
 *
 * ## Vector order (load-bearing; persistence and the mapping layer pin it)
 *
 *   [0        , P·K )   K layer blocks, P slots each
 *   [P·K      , P·K + G) globals
 *
 * so `vectorLength(K) = P·K + G`. The layer-block-first layout is deliberately
 * the same shape plife uses, which is what lets `thL()` below be one multiply
 * and an add.
 */
import type { ModSpec } from '../../mapping/modspec.ts';

/**
 * One θ slot, and the whole of what is said about it anywhere.
 *
 * `min`/`max` are the **hard** bounds — what a loaded file may contain and how
 * far a hand slider travels on an unmodulated slot — and are deliberately wider
 * than the `ModSpec`'s `lo`/`hi`, which is where the music may wander
 * unsupervised. Same split as both older registries.
 */
export interface VizSlot {
  /** slot key. Per-layer slots are addressed as `layer<i>.<key>` in the registry. */
  key: string;
  /** panel label; states the wiring where it is not obvious from the name */
  label: string;
  /**
   * Which panel folder this slot's slider belongs in. **Globals only** — a
   * per-layer slot always lands in its own layer's folder, since grouping four
   * copies of one knob anywhere else would separate a layer from itself.
   *
   * Omitted means *no slider is built for this slot*, which is not the same as
   * "no folder": it says something else already owns the control. `exposure` and
   * `gamma` are the two — both are θ, both are excluded from modulation, and
   * both are bound by the shared HDR folder (`ui/render-folder.ts`), so a second
   * slider here would be two widgets writing one number.
   */
  folder?: string;
  /** slew class, as the small ints `SlewLimiter` indexes with */
  cls: number;
  min: number;
  max: number;
  /** slider granularity. Also the quantisation a `pane.refresh()` writes back. */
  step: number;
  /** shipped default. A function when a layer's character differs from its siblings. */
  def: number | ((layer: number) => number);
  /** null = excluded from modulation; the sliders own it outright */
  mod: ModSpec | null;
  /**
   * Which macro multiplies this slot on its way to the GPU, if any.
   *
   * Macros are plain multipliers applied *outside* θ (see `effectiveTheta`), so
   * only slots whose neutral value is 0 (additive) or whose scale is genuinely
   * proportional (a gain, a radius) may carry one. A slot centred on 1 — `zoom`
   * is the example — must not: ×1.5 on a 1.012 zoom is a 1.5× per-frame
   * explosion, not "more motion".
   */
  macro?: string;
}

/** One always-yours multiplier. Composes outside θ; the modulator never writes it. */
export interface VizMacro {
  key: string;
  label: string;
  min: number;
  max: number;
}

/**
 * A concrete visual: its shaders, its θ table, its art direction.
 *
 * Everything here is data. `VizFxSim` reads it and knows nothing else about
 * what it is drawing — which is the point of the chassis/visual split.
 */
export interface VizFxVisual {
  /** persistence discriminator and `?sim=` value, e.g. 'nebula' */
  id: string;
  /** tweakpane window title */
  title: string;
  /** K. 4 = the four stems; see the note in `vizfx.ts`. */
  speciesCount: number;
  /** per-layer display names, index-aligned with the stems channel */
  layerNames: readonly string[];
  /** shipped per-layer hues, sRGB hex */
  paletteHex: readonly string[];
  /** orbiting emitter clusters per layer. Structural — outside θ, in `extras`. */
  emittersPerLayer: number;
  layerSlots: readonly VizSlot[];
  globalSlots: readonly VizSlot[];
  macros: readonly VizMacro[];
  /** WGSL for the two passes that make this visual what it is */
  warpWgsl: string;
  drawWgsl: string;
}

export function vectorLength(v: VizFxVisual): number {
  return v.layerSlots.length * v.speciesCount + v.globalSlots.length;
}

/** Registry name of a slot. The panel binds `config.params` by exactly this key. */
export function slotName(slot: VizSlot, layer: number): string {
  return layer < 0 ? slot.key : `layer${layer}.${slot.key}`;
}

function defaultOf(slot: VizSlot, layer: number): number {
  return typeof slot.def === 'function' ? slot.def(layer) : slot.def;
}

/** Every slot in vector order, paired with its layer index (-1 for a global). */
export function eachSlot(v: VizFxVisual): { slot: VizSlot; layer: number }[] {
  const out: { slot: VizSlot; layer: number }[] = [];
  for (let l = 0; l < v.speciesCount; l++) {
    for (const slot of v.layerSlots) out.push({ slot, layer: l });
  }
  for (const slot of v.globalSlots) out.push({ slot, layer: -1 });
  return out;
}

/** Human-readable slot names, in vector order. The workbench and the panel index by these. */
export function fieldNames(v: VizFxVisual): string[] {
  return eachSlot(v).map(({ slot, layer }) => slotName(slot, layer));
}

export function fieldClasses(v: VizFxVisual): Uint8Array {
  const all = eachSlot(v);
  const out = new Uint8Array(all.length);
  for (let i = 0; i < all.length; i++) out[i] = (all[i] as { slot: VizSlot }).slot.cls;
  return out;
}

export function modulationSlots(v: VizFxVisual): (ModSpec | null)[] {
  return eachSlot(v).map(({ slot }) => slot.mod);
}

export function modulationMask(v: VizFxVisual): Uint8Array {
  const slots = modulationSlots(v);
  const out = new Uint8Array(slots.length);
  for (let i = 0; i < slots.length; i++) out[i] = slots[i] ? 1 : 0;
  return out;
}

/**
 * The shipped parameter store: slot name → default.
 *
 * A plain object rather than a typed interface, and that is the whole trick
 * that keeps this declarative — the panel binds `config.params` with the slot's
 * own name as the tweakpane key, so adding a slot adds a slider with no
 * interface to widen. It also survives `structuredClone` (explorer tiles) and
 * `JSON.stringify` unchanged, which an accessor-based scheme would not.
 */
export function defaultParams(v: VizFxVisual): Record<string, number> {
  const out: Record<string, number> = {};
  for (const { slot, layer } of eachSlot(v)) out[slotName(slot, layer)] = defaultOf(slot, layer);
  return out;
}

/** Default macro rig: every multiplier neutral. */
export function defaultMacros(v: VizFxVisual): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of v.macros) out[m.key] = 1;
  return out;
}

/** θ ← the live store. Allocates; the per-tick path is `effectiveTheta`. */
export function paramsToVector(
  v: VizFxVisual,
  params: Record<string, number>,
  out?: Float64Array,
): Float64Array {
  const all = eachSlot(v);
  const dst = out ?? new Float64Array(all.length);
  for (let i = 0; i < all.length; i++) {
    const { slot, layer } = all[i] as { slot: VizSlot; layer: number };
    const name = slotName(slot, layer);
    const x = params[name];
    dst[i] = typeof x === 'number' && Number.isFinite(x) ? x : defaultOf(slot, layer);
  }
  return dst;
}

/**
 * The live store ← θ, clamped into each slot's hard bound.
 *
 * `mask`, when given, restricts the write to the slots it marks — that is how a
 * modulated run leaves the excluded slots (brightness, exposure, gamma) under
 * the panel's control instead of stamping a stale copy over every edit.
 *
 * The clamp is here rather than in a separate `clampVector` because this is the
 * only door into the store: the modulator, a loaded file, an explorer candidate
 * and `applyBase` all arrive through it, so one clamp covers all four.
 */
export function applyVector(
  v: VizFxVisual,
  params: Record<string, number>,
  vec: ArrayLike<number>,
  mask?: Uint8Array,
): void {
  const all = eachSlot(v);
  for (let i = 0; i < all.length; i++) {
    if (mask !== undefined && mask[i] !== 1) continue;
    const { slot, layer } = all[i] as { slot: VizSlot; layer: number };
    const raw = vec[i];
    const x = typeof raw === 'number' && Number.isFinite(raw) ? raw : defaultOf(slot, layer);
    params[slotName(slot, layer)] = x < slot.min ? slot.min : x > slot.max ? slot.max : x;
  }
}

/**
 * θ as the **GPU** sees it: the live store with the macro rig folded in, written
 * into a reused f32 buffer.
 *
 * The macro multiply happens here and nowhere else, which is what lets the
 * shader read one flat array and lets a macro edit be visible on the very next
 * tick without touching θ. It is also why the shader must never re-apply a
 * macro: by the time a value reaches `th()` it is already effective.
 *
 * Clamped to the hard bound again after the multiply, deliberately: a macro at
 * 2 on a slot already sitting at its ceiling must saturate rather than hand the
 * shader a value no slider could produce.
 */
export function effectiveTheta(
  v: VizFxVisual,
  params: Record<string, number>,
  macros: Record<string, number>,
  out: Float32Array,
): Float32Array {
  const all = eachSlot(v);
  for (let i = 0; i < all.length; i++) {
    const { slot, layer } = all[i] as { slot: VizSlot; layer: number };
    const raw = params[slotName(slot, layer)];
    let x = typeof raw === 'number' && Number.isFinite(raw) ? raw : defaultOf(slot, layer);
    if (slot.macro !== undefined) {
      const m = macros[slot.macro];
      x *= typeof m === 'number' && Number.isFinite(m) ? m : 1;
    }
    out[i] = x < slot.min ? slot.min : x > slot.max ? slot.max : x;
  }
  return out;
}

/**
 * The WGSL prelude: one `TH_*` constant per slot plus the two accessors.
 *
 * Generated rather than hand-written because it is the one place the shader and
 * the table could silently disagree — a hand-maintained `const TH_zoom = 14u`
 * is correct exactly until someone inserts a slot above it, and the failure
 * mode is a shader that compiles and reads the wrong parameter. Here a renamed
 * slot is a WGSL compile error naming the old constant, which is loud.
 *
 * Concatenated *between* `common.wgsl` (which declares `theta`) and the pass
 * body (which uses the constants), so nothing in the generated text forward-
 * references anything. WGSL is order-independent at module scope and this would
 * work in any order; ordering it anyway costs nothing and removes a bet.
 *
 * The accessors are generated here rather than living in `common.wgsl` for one
 * reason: `thL` needs `TH_PER_LAYER`, which is a property of the visual's table
 * and not of the chassis.
 */
export function wgslPrelude(v: VizFxVisual): string {
  const lines: string[] = [
    '// GENERATED from the visual\'s slot table (sim/vizfx/slots.ts). Do not hand-edit.',
    `const TH_PER_LAYER: u32 = ${v.layerSlots.length}u;`,
    `const TH_COUNT: u32 = ${vectorLength(v)}u;`,
  ];
  v.layerSlots.forEach((slot, i) => {
    lines.push(`const TH_L_${slot.key}: u32 = ${i}u;`);
  });
  const base = v.layerSlots.length * v.speciesCount;
  v.globalSlots.forEach((slot, i) => {
    lines.push(`const TH_${slot.key}: u32 = ${base + i}u;`);
  });
  lines.push(
    '/** One global θ slot, macro-applied and clamped by the CPU. */',
    'fn th(i: u32) -> f32 { return theta[i]; }',
    '/** Layer `l`\'s copy of a per-layer slot. */',
    'fn thL(l: u32, off: u32) -> f32 { return theta[l * TH_PER_LAYER + off]; }',
  );
  return lines.join('\n');
}
