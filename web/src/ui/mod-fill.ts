/**
 * The blue band behind a modulated slider's knob: where the music can take this
 * parameter, given where you have put its base.
 *
 * With VST semantics in the modulator (see `mapping/modulation.ts` — a drag
 * re-anchors the base instead of being stamped over), a modulated slider stopped
 * being a readout and became a control with a *range*. That range is invisible
 * in a plain slider: the knob wanders, the number changes, and nothing on screen
 * says how far it can go or that the value under your finger is a centre rather
 * than a value. A synth draws the modulation depth on the knob for exactly this
 * reason, so this draws it on the track.
 *
 * The span is the honest reachable one, and it is an intersection of three
 * things rather than just `base ± half`:
 *
 *   base ± half        additive, or base·e^±half multiplicative — the bound
 *                      `tanh` guarantees, whatever the depth or the gains
 *   ∩ [spec.lo, hi]    `computeTarget` clamps its target into the authored
 *                      modulation range, so past that edge the parameter simply
 *                      stops
 *   ∩ [slider min,max] the pixels belong to the widget, not to θ, and the two
 *                      ranges differ by design (physarum's panel binds the hard
 *                      θ bound, plife's binds the ModSpec range)
 *
 * ## Why it also owns the `markApplied` pairing
 *
 * Because both jobs are "the thing that has to happen after tweakpane has
 * refreshed its bindings", and there are 26 `pane.refresh()` call sites across
 * the two panels, the workbench and the explorer folder. Refresh is not a
 * passive read — it pushes every bound number through its widget's `step` and
 * writes the rounded value back — so a missed pairing is not a stale band, it is
 * the modulator reading its own rounding as a human edit and re-anchoring dozens
 * of bases onto `base ± excursion` twice a second. Rather than 26 places
 * remembering a rule, the pairing is installed on the pane once, here.
 * `main.ts` needs no hook of its own: its periodic `panel.refresh()` ends in
 * `pane.refresh()` like everything else.
 */
import type { Pane } from 'tweakpane';
import type { Modulator } from '../mapping/modulation';
import type { ModSpec } from '../mapping/modspec';

/**
 * The only thing wanted from a tweakpane `BindingApi` is its DOM root, so that
 * is all this asks for — structurally, rather than importing a generic API type
 * whose type parameters every call site would then have to satisfy.
 */
export interface BindingElement {
  readonly element: HTMLElement;
}

/**
 * The number-valued keys of an object.
 *
 * Needed because the panels bind a species' θ fields through one generic helper,
 * and a `keyof SpeciesConfig` that still admits `name: string` makes tweakpane
 * resolve `addBinding` against its *string monitor* overload — which demands a
 * `readonly` flag no slider has. Narrowing the key narrows the overload.
 */
export type NumericKey<T> = {
  [K in keyof T]-?: T[K] extends number ? K : never;
}[keyof T];

/**
 * A θ slider's numeric range, passed once to the binding and once to its band —
 * the two have to agree or the band would be drawn in the wrong coordinates.
 *
 * A type alias rather than an interface, deliberately: tweakpane's input params
 * extend `Record<string, unknown>`, and TypeScript only grants an *implicit*
 * index signature to anonymous object types, never to a declared interface.
 */
export type SliderParams = {
  min: number;
  max: number;
  step: number;
  label: string;
};

/** tweakpane 4's slider track. `position: relative`, so the band can live inside it. */
const TRACK_SELECTOR = '.tp-sldv_t';

export interface ModBands {
  /**
   * Decorate one slider. Silently does nothing when the slot is unmodulated
   * (`slot < 0`, or a `null` ModSpec) or when the binding drew no slider —
   * "no band on an unmodulated slot, ever" is then the default rather than a
   * rule each call site has to apply.
   */
  add(binding: BindingElement, slot: number, min: number, max: number): void;
  /** Re-place every band. Cheap; runs on the pane's own refresh cadence. */
  refresh(): void;
}

interface Band {
  slot: number;
  spec: ModSpec;
  min: number;
  max: number;
  el: HTMLElement;
}

/** Slot names → index, for panels that know a field's name but not its offset. */
export function slotLookup(names: readonly string[]): (name: string) => number {
  const index = new Map<string, number>();
  names.forEach((name, i) => {
    if (!index.has(name)) index.set(name, i);
  });
  return (name: string): number => index.get(name) ?? -1;
}

export function createModBands(pane: Pane, modulator: Modulator): ModBands {
  const bands: Band[] = [];

  const refresh = (): void => {
    // No band in manual mode: nothing is modulating, so a band would be drawing a
    // range that is not in play. Hidden rather than dimmed — dimmed reads as
    // "reduced", and the honest statement is "this slider is absolute right now",
    // which the workbench's status line already makes in words.
    const showing = modulator.mode === 'modulated' && modulator.available;
    for (const b of bands) {
      if (!showing) {
        b.el.style.display = 'none';
        continue;
      }
      const base = modulator.baseValue(b.slot);
      const { half, mult, lo, hi } = b.spec;
      let a = mult ? base * Math.exp(-half) : base - half;
      let z = mult ? base * Math.exp(half) : base + half;
      if (a > z) [a, z] = [z, a];
      a = a < lo ? lo : a > hi ? hi : a;
      z = z < lo ? lo : z > hi ? hi : z;
      a = a < b.min ? b.min : a > b.max ? b.max : a;
      z = z < b.min ? b.min : z > b.max ? b.max : z;
      const span = b.max - b.min;
      const left = span > 0 ? (a - b.min) / span : 0;
      const width = span > 0 ? Math.max((z - a) / span, 0) : 0;
      b.el.style.display = '';
      b.el.style.left = `${(left * 100).toFixed(3)}%`;
      b.el.style.width = `${(width * 100).toFixed(3)}%`;
    }
  };

  // The pairing described in the header. Bound first so the original is still
  // reachable after the property shadows the prototype method.
  const inner = pane.refresh.bind(pane);
  pane.refresh = (): void => {
    inner();
    modulator.markApplied();
    refresh();
  };

  return {
    add(binding: BindingElement, slot: number, min: number, max: number): void {
      const spec = slot >= 0 ? modulator.spec(slot) : null;
      if (!spec) return;
      const track = binding.element.querySelector(TRACK_SELECTOR);
      if (!(track instanceof HTMLElement)) return;
      const el = document.createElement('div');
      el.className = 'mod-band';
      // First child, so paint order puts the band under tweakpane's own knob and
      // fill rather than over them: both are position:absolute with no z-index,
      // and in that case DOM order decides.
      track.insertBefore(el, track.firstChild);
      bands.push({ slot, spec, min, max, el });
    },
    refresh,
  };
}
