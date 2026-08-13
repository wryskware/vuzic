import type { PanelContainer } from './panel';

/**
 * The word the *mode* selector uses for the vizfx family.
 *
 * Presentation only, and exported so that the two other places that have to say
 * it — main's status line and its `?sim=` alias — say the same string. It is
 * deliberately **not** an id: no sim answers to it, no autosave slot is named
 * after it, and `switchTo` is never called with it. Every visual in the family
 * keeps its own id in the URL and in localStorage, because their θ tables differ
 * and a shared slot would apply one visual's saved vector to another's registry.
 */
export const MILKDROP_MODE = 'milkdrop';

export interface SimPanelHost {
  /**
   * Every substrate that can be selected. This is main's `SIMS` passed straight
   * through, not a copy of it: the list is also the `?sim=` whitelist and the set
   * of autosave slots, and a picker that restated it would be a second place to
   * forget when a visual is added.
   */
  ids: readonly string[];
  /**
   * The milkdrop repertoire, in cycle order — a subset of `ids`, from
   * `sim/vizfx/visuals.ts`. Two uses here: it is the preset dropdown's option
   * list, and it is what the mode selector subtracts from `ids` to decide which
   * entries are modes in their own right. So a visual added to the repertoire
   * file appears under milkdrop and *stops* appearing as a top-level mode, with
   * no edit here.
   */
  presets: readonly string[];
  /** The one on screen. `Sim.simId`, which is also its `?sim=` value. */
  current: string;
  /**
   * Which preset picking "milkdrop" from the mode selector means — main's memory
   * of the last visual that actually ran, not `presets[0]`. Flicking to physarum
   * and back mid-audition should return the visual you were looking at; the
   * panel cannot remember it itself because a swap rebuilds the panel.
   */
  entryPreset: string;
  /** Switch. See `switchSim` in main.ts for what that does and does not rebuild. */
  switchTo(id: string): void;
  /** Whether a section boundary advances to the next preset. Session-local. */
  autoAdvance: boolean;
  setAutoAdvance(on: boolean): void;
}

const optionsOf = (ids: readonly string[]): Record<string, string> =>
  Object.fromEntries(ids.map((id) => [id, id]));

/**
 * The substrate picker, directly under the track picker on the play tab.
 *
 * Those two are the same kind of decision — *what am I looking at*, before any
 * question of how it is tuned — and until now one of them was only reachable by
 * hand-editing the query string. So it sits with the track, above the world seed,
 * and not on the `sim` tab: that tab is the running substrate's own physics, and
 * a control that replaces the substrate outright does not belong among the
 * controls that shape it.
 *
 * ## Two levels, one flat set of ids underneath
 *
 * The vizfx visuals are *presented* as one mode called milkdrop with a preset
 * inside it, rather than as N peers of physarum and plife. That is a claim about
 * what the human is choosing between — a substrate, and then which of the family
 * of screen-space visuals is running — and it keeps a mode list of three from
 * growing by one every time a warp/draw pair lands.
 *
 * Underneath, nothing is grouped: `switchTo` always receives a concrete visual
 * id, which is what `?sim=` carries and what the autosave slot is named after.
 * The grouping exists in this file and in main's status line, and nowhere that
 * persists anything.
 *
 * Options are labelled with the bare ids rather than with each substrate's
 * display title (`physarum`, not `terrarium · physarum`). The id is what `?sim=`
 * takes, what the autosave slot is named after and what the status line prints,
 * so labelling with anything else would introduce a fourth name for the same
 * thing purely for the dropdown's benefit. `milkdrop` is the one exception and
 * pays for itself: it is the name of a mode that has no id.
 *
 * Unlike the track picker this does **not** necessarily reload — `switchTo` is
 * main's live swap, which falls back to nothing more drastic than staying put.
 * The widget cannot tell the difference and deliberately does not try to: on a
 * successful swap the whole panel is rebuilt underneath it, so everything shown
 * here — including whether the milkdrop folder exists at all — is re-derived
 * from the live sim rather than remembered here. That rebuild is triggered by
 * `switchSim`, which only disposes this panel in a continuation after an await,
 * never synchronously inside the change handler that started it.
 */
export function createSimFolder(container: PanelContainer, host: SimPanelHost): void {
  const folder = container.addFolder({ title: 'visual' });
  const presets = new Set(host.presets);
  const active = presets.has(host.current);
  const modes = [
    ...host.ids.filter((id) => !presets.has(id)),
    ...(host.presets.length > 0 ? [MILKDROP_MODE] : []),
  ];

  const state = {
    mode: active ? MILKDROP_MODE : host.current,
    preset: active ? host.current : (host.presets[0] ?? ''),
    auto: host.autoAdvance,
  };

  folder.addBinding(state, 'mode', { label: 'running', options: optionsOf(modes) }).on(
    'change',
    (ev) => {
      const id = ev.value === MILKDROP_MODE ? host.entryPreset : String(ev.value);
      if (id !== host.current) host.switchTo(id);
    },
  );

  // The preset row only exists while a visual from the family is on screen.
  // Showing it greyed the rest of the time would be a control that answers a
  // question nobody asked — and it is one click away in either direction.
  if (!active) return;

  const milkdrop = folder.addFolder({ title: MILKDROP_MODE });
  milkdrop.addBinding(state, 'preset', { label: 'preset', options: optionsOf(host.presets) }).on(
    'change',
    (ev) => {
      if (ev.value !== host.current) host.switchTo(String(ev.value));
    },
  );
  // Session-local by design: it is a way of watching, not a mapping decision, and
  // nothing here writes localStorage. It also survives a swap without being
  // persisted, because the flag lives in main and only its *display* is rebuilt.
  milkdrop
    .addBinding(state, 'auto', { label: 'advance on section' })
    .on('change', (ev) => host.setAutoAdvance(Boolean(ev.value)));
}
