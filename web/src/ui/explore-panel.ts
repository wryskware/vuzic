/**
 * The explorer folder: the way in and out of 9-up mode, and the two knobs that
 * decide what the eight candidates are.
 *
 * Structurally a sibling of `impulses-panel.ts` — a small folder factory that
 * both substrates' panels mount, rather than anything either of them owns. It
 * knows nothing about the rig or the search; it drives an `ExplorerPanelHost`
 * that main.ts implements, exactly as the workbench drives a `WorkbenchHost`.
 * That is what keeps this file free of GPU lifecycle: entering the mode is an
 * async operation that can fail, and a panel is the wrong place to reason about
 * that.
 *
 * ## Why the controls exist at all
 *
 * The mode is a hill climb whose objective function is a human eye, so the two
 * things a human has to be able to change mid-climb are **how far** a candidate
 * may stray (step) and **what is allowed to vary** (subspace). Freezing three of
 * the four mod groups and climbing in the fourth is how "I like the shapes,
 * find me better colours of motion" becomes an actual search — without it every
 * generation moves everything and nothing is attributable.
 *
 * ## One exit, and why there is no "adopt"
 *
 * There used to be two ways out — adopt the centre or discard it — because the
 * grid was a sandbox that only wrote back on the way out. It is not one any more:
 * every pick applies its θ to the live sim immediately, and every slider in the
 * folders below this one edits the θ the grid is centred on. So the centre is
 * *already* what you have, "exit" is only the act of stopping looking at nine of
 * them, and the thing that genuinely needed a button is the opposite — a way to
 * throw the whole run away and get back the θ you walked in with. Hence one exit
 * and one revert, rather than two exits that mean different things.
 *
 * ## Widgets that are not there
 *
 * No per-candidate readout of θ, and no numeric diff against the centre. The
 * mode's whole claim is that you judge by looking; a wall of numbers next to it
 * invites judging by reading, which is the thing the workbench already does
 * better. The log carries the numbers for anything downstream that wants them.
 */
import type { ButtonApi, FolderApi } from 'tweakpane';
import { EXPLORER_SUBSPACES, type ExplorerSubspace } from '../explore/search';
import type { PanelContainer } from './panel';

export interface ExplorerPanelHost {
  /** false when WebGPU is unavailable — there is nothing to run nine of */
  readonly available: boolean;
  readonly active: boolean;
  /** 0 at the first generation, +1 per pick/reroll */
  readonly generation: number;
  readonly logSize: number;
  readonly step: number;
  readonly subspace: ExplorerSubspace;
  enter(): void;
  /**
   * Close the grid. There is nothing to adopt: the centre has been the live sim's
   * θ since the moment it became the centre.
   */
  exit(): void;
  /** put the live sim back to the θ it had when the mode was entered, and regrow the grid there */
  revert(): void;
  reroll(): void;
  back(): void;
  /** log a 'like' against the current centre without moving the search */
  like(): void;
  exportLog(): void;
  setStep(v: number): void;
  setSubspace(s: ExplorerSubspace): void;
}

export interface ExplorePanelHandle {
  refresh(): void;
  dispose(): void;
}

interface UiState {
  status: string;
  step: number;
  subspace: ExplorerSubspace;
}

/** `{ all: 'all', structure: 'structure', … }` — tweakpane's list wants label → value. */
function subspaceOptions(): Record<string, ExplorerSubspace> {
  const out: Record<string, ExplorerSubspace> = {};
  for (const s of EXPLORER_SUBSPACES) out[s] = s;
  return out;
}

export function createExplorePanel(
  container: PanelContainer,
  host: ExplorerPanelHost,
): ExplorePanelHandle {
  const ui: UiState = {
    status: host.available ? 'off' : 'no webgpu',
    step: host.step,
    subspace: host.subspace,
  };

  // Still a folder even though it now has a tab page to itself: it owns its own
  // disposal, and a page cannot be disposed.
  const root = container.addFolder({ title: 'explorer · 9-up search', expanded: true });
  root.addBinding(ui, 'status', { readonly: true, label: '' });

  const enter = root.addButton({ title: '⊞ enter explorer (9-up)' });
  enter.on('click', () => host.enter());
  enter.disabled = !host.available;

  // Step and subspace are live in *both* states on purpose: setting them before
  // entering is how you start a run somewhere other than the default, and the
  // host holds them for the search that does not exist yet.
  root
    .addBinding(ui, 'step', { min: 0.05, max: 1, step: 0.01, label: 'mutation step σ' })
    .on('change', (ev) => host.setStep(ev.value));
  root
    .addBinding(ui, 'subspace', { options: subspaceOptions(), label: 'vary (frozen: rest)' })
    .on('change', (ev) => host.setSubspace(ev.value));

  // Everything below only means anything with nine tiles on screen, so it is
  // hidden rather than disabled while off — a folder of dead buttons reads as
  // broken, and this one is expanded by default.
  const live = [
    button(root, '⟳ reroll candidates  (r)', () => host.reroll()),
    button(root, '↶ back one generation  (⌫)', () => host.back()),
    button(root, '♥ like this centre (log only)', () => host.like()),
    button(root, '⟲ revert to entry θ', () => host.revert()),
    button(root, '✕ exit explorer  (esc)', () => host.exit()),
  ];

  // The log outlives the mode — it is preference data, not session state — so
  // its readout and its export stay visible with the tiles gone. Exporting after
  // you have left is the normal case, not the exception.
  const logUi = { log: '0 entries' };
  root.addBinding(logUi, 'log', { readonly: true, label: 'log' });
  button(root, '⭳ export explorer log (.jsonl)', () => host.exportLog());

  const setActive = (active: boolean): void => {
    for (const b of live) b.hidden = !active;
    enter.hidden = active;
  };
  setActive(false);

  return {
    refresh(): void {
      if (!host.available) {
        ui.status = 'no webgpu';
        return;
      }
      // The active line names the other half of the loop on purpose: the sliders
      // below being live is the least discoverable thing about the mode.
      ui.status = host.active
        ? `gen ${host.generation} · pick a tile · sliders recentre`
        : 'off · live sim runs';
      ui.step = host.step;
      ui.subspace = host.subspace;
      logUi.log = `${host.logSize} entries`;
      setActive(host.active);
    },
    dispose(): void {
      root.dispose();
    },
  };
}

function button(root: FolderApi, title: string, onClick: () => void): ButtonApi {
  const b = root.addButton({ title });
  b.on('click', onClick);
  return b;
}
