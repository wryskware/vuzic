/**
 * The live control panel for a vizfx visual.
 *
 * A sibling of `panel.ts` and `plife-panel.ts` in placement and conventions, and
 * deliberately *not* a sibling in construction: those two hand-write a slider per
 * θ field because their θ is structurally varied (physarum's nested adaptive
 * triples, plife's three K² blocks), and describing that as data would be more
 * machinery than two hand-written panels. A vizfx visual's θ is a flat list of
 * scalars with bounds, a label and a folder already attached to each — the
 * `VizSlot` table in `sim/vizfx/slots.ts` — so a hand-written panel here would be
 * a second copy of a table that already exists, kept in step by hope.
 *
 * So this file builds the θ folders from the visual's own table and knows the
 * name of no parameter. Everything *around* the θ sliders — the tab scaffolding,
 * the workbench, the impulse folder, the track picker, the explorer folder, the
 * HDR chain — is the shared machinery, imported exactly as `plife-panel.ts`
 * imports it and taking the identical options object.
 *
 * ## Slider ranges
 *
 * Same rule both older panels use, applied here from the table:
 *
 * - a slot the music moves gets its **`ModSpec` lo/hi** — the range the modulator
 *   itself wanders inside. A slider wider than that would spend most of its
 *   travel somewhere the music never goes, and dragging it there would be
 *   overwritten on the next tick anyway.
 * - a slot excluded from modulation gets its **hard min/max**, because nothing
 *   else is ever going to write it and the whole authored range is usable.
 *
 * ## What is not here
 *
 * Slots with no `folder` (see `VizSlot.folder`). `exposure` and `gamma` are θ but
 * the shared HDR folder already binds them, and two widgets on one number is a
 * bug waiting for a drag.
 */
import { Pane, type FolderApi } from 'tweakpane';
import { saveModulationLocal } from '../mapping/persist';
import type { ImpulseEngine } from '../sim/impulses';
import { ENERGY_RANGE, MAX_EMITTERS_PER_LAYER } from '../sim/vizfx/config';
import type { VizSlot } from '../sim/vizfx/slots';
import { slotName } from '../sim/vizfx/slots';
import type { VizFxSim } from '../sim/vizfx/vizfx';
import { randomSeed, setPinnedSeed, syncUrlSeed } from '../sim/seed';
import {
  createExplorePanel,
  type ExplorePanelHandle,
  type ExplorerPanelHost,
} from './explore-panel';
import { createImpulsePanel, type ImpulsePanelHandle } from './impulses-panel';
import { createModBands, slotLookup, type ModBands } from './mod-fill';
import { createPanelTabs, type PanelContainer, type PanelHandle, type PanelWorkbench } from './panel';
import { addPaletteFolder } from './palette-folder';
import { addRenderFolder } from './render-folder';
import { createExportFolder, type ExportPanelHost } from './export-panel';
import { createSimFolder, type SimPanelHost } from './sim-panel';
import { createTrackFolder, type TrackPanelHost } from './track-panel';
import { createWorkbench, type WorkbenchHandle } from './workbench';

interface RunState {
  seed: string;
  field: string;
  steps: string;
}

export function createVizFxPanel(
  sim: VizFxSim,
  opts: {
    pinned: boolean;
    /** rewind the transport: a restart must not resume against an arbitrary timeline position */
    onRestart?: () => void;
    /** omit to get a panel with no modulation layer at all */
    workbench?: PanelWorkbench;
    /** omit to hide the events folder entirely */
    impulses?: ImpulseEngine;
    /** omit to hide the explorer folder entirely */
    explorer?: ExplorerPanelHost;
    /** omit to hide the track picker entirely */
    tracks?: TrackPanelHost;
    /** omit when the local native export API is not part of this host */
    exports?: ExportPanelHost;
    /** omit to hide the substrate picker entirely */
    sims?: SimPanelHost;
  },
): PanelHandle {
  const config = sim.config;
  const visual = sim.visual;
  const k = config.speciesCount;

  const pane = new Pane({ title: visual.title });
  // Created before any binding: it decorates each slider as it is built, and it
  // installs the `markApplied` pairing on `pane.refresh` (see mod-fill.ts) that
  // every refresh below then inherits for free.
  const bands: ModBands | null = opts.workbench
    ? createModBands(pane, opts.workbench.modulator)
    : null;
  /** θ slot names are the registry's, and the registry's names are the table's. */
  const slotOf = slotLookup(sim.registry().names);
  const tabs = createPanelTabs(pane, {
    explorer: opts.explorer !== undefined,
    workbench: opts.workbench !== undefined,
  });

  const state: RunState = { seed: String(sim.currentSeed), field: '—', steps: '—' };

  /**
   * One θ slider plus its modulation band, from a table row.
   *
   * This is the whole of the panel's θ knowledge: the binding key *is* the
   * registry name, so a slot lookup needs nothing plumbed through — and a table
   * row whose name does not resolve draws no band rather than silently
   * decorating a neighbouring slider.
   */
  const bindSlot = (into: PanelContainer, slot: VizSlot, layer: number): void => {
    const name = slotName(slot, layer);
    // Modulated slots take the ModSpec range; the sliders own the rest outright
    // and get the whole hard bound. See the header.
    const lo = slot.mod ? Math.min(slot.mod.lo, slot.mod.hi) : slot.min;
    const hi = slot.mod ? Math.max(slot.mod.lo, slot.mod.hi) : slot.max;
    const b = into.addBinding(config.params, name, {
      min: lo,
      max: hi,
      step: slot.step,
      label: slot.label,
    });
    bands?.add(b, slotOf(name), lo, hi);
  };

  // ── explore ────────────────────────────────────────────────────────────────
  const explorer: ExplorePanelHandle | null =
    opts.explorer && tabs.explore ? createExplorePanel(tabs.explore, opts.explorer) : null;

  // ── play ───────────────────────────────────────────────────────────────────
  if (opts.tracks) createTrackFolder(tabs.play, opts.tracks);
  const exportFolder = opts.exports ? createExportFolder(tabs.play, opts.exports) : null;
  // And directly under it, the other "what am I looking at" decision.
  if (opts.sims) createSimFolder(tabs.play, opts.sims);

  // Created before the macros folder because tweakpane orders by mount order and
  // the seed belongs above them; the workbench fills it in below.
  const seedFolder = opts.workbench
    ? tabs.play.addFolder({ title: 'world seed (wiring + personality + emitters)' })
    : null;

  // The macro rig, the energy lane and the emitter count all ride the mapping
  // file's opaque `extras` block, so an edit to any of them has to reach the same
  // autosave the workbench's own bindings use. Done directly rather than through
  // a new callback — the panel already holds both halves — and debounced, because
  // these are sliders and a drag would otherwise write localStorage on every
  // pointer move. It re-serialises the *whole* block rather than the field that
  // changed, which is why one saver covers three folders.
  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  const persistExtras = (): void => {
    const wb = opts.workbench;
    if (!wb) return;
    if (saveTimer !== null) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      wb.modulator.config.extras = sim.serializeExtras();
      saveModulationLocal(wb.modulator.config, sim.simId);
    }, 400);
  };

  // Multipliers that compose outside θ, exactly where stem-follow and the impulse
  // lanes do — which is the whole reason the modulator can never write over them.
  // Everything on the sim tab is θ (mirrored, overwritten on the next tick while
  // modulating) or structural; this is the only block that is unconditionally
  // yours.
  const macros = tabs.play.addFolder({ title: 'macros · performance layer' });
  for (const m of visual.macros) {
    macros
      .addBinding(config.macros, m.key, {
        min: m.min,
        max: m.max,
        step: 0.01,
        label: m.label,
      })
      .on('change', persistExtras);
  }
  macros.addButton({ title: 'reset macros to 1' }).on('click', () => {
    for (const m of visual.macros) config.macros[m.key] = 1;
    persistExtras();
    pane.refresh();
  });

  let workbench: WorkbenchHandle | null = null;
  if (opts.workbench && seedFolder && tabs.data) {
    workbench = createWorkbench(
      { pane, seed: seedFolder, play: tabs.play, map: tabs.map, data: tabs.data },
      {
        ...opts.workbench,
        pinned: opts.pinned,
        restart: () => opts.onRestart?.(),
        // A reroll or a file load rewrites the live config wholesale; the palette
        // widgets are bound to a proxy and must be re-read.
        onConfigReplaced: () => {
          refreshPalette();
          pane.refresh();
        },
      },
    );
  }

  const run = tabs.play.addFolder({ title: 'run', expanded: !opts.workbench });
  if (!opts.workbench) {
    run.addBinding(state, 'seed', { readonly: true });
    run.addButton({ title: 'restart (same seed)' }).on('click', () => {
      sim.reseed(sim.currentSeed);
      opts.onRestart?.();
    });
    run.addButton({ title: 'reseed + restart' }).on('click', () => {
      const seed = randomSeed();
      sim.reseed(seed);
      opts.onRestart?.();
      state.seed = String(seed);
      syncUrlSeed(seed);
      setPinnedSeed(opts.pinned ? seed : null);
      pane.refresh();
    });
  }
  run.addBinding(config, 'paused');
  run.addButton({ title: 'single step' }).on('click', () => {
    config.paused = true;
    sim.requestSingleStep();
    pane.refresh();
  });
  // Steps per clock tick. Above 1 the field decays and zooms faster *per second*
  // without becoming coarser, because the step is a fixed 1/60 s whatever this
  // says — see the note on `VizFxSim.tick`.
  run.addBinding(config, 'speed', { min: 0, max: 4, step: 0.25, label: 'steps / tick' });
  run.addBinding(state, 'field', { readonly: true, label: 'field' });
  run.addBinding(state, 'steps', { readonly: true, label: 'steps this tick' });

  // ── map ────────────────────────────────────────────────────────────────────
  // Below the workbench's own folders: the energy lane and the impulse lane are
  // the two ways the music reaches this substrate *without* going through the
  // driver bank.
  //
  // The energy lane is outside θ entirely, which is why every widget here binds
  // directly with no mask or mode caveat: the modulator never writes any of it,
  // in either mode.
  const energy = tabs.map.addFolder({ title: 'energy · stems → presence', expanded: false });
  energy
    .addBinding(config.energy, 'followStems', { label: 'follow stems (off = all layers full)' })
    .on('change', persistExtras);
  energy
    .addBinding(config.energy, 'floor', {
      min: ENERGY_RANGE.floor.min,
      max: ENERGY_RANGE.floor.max,
      step: 0.01,
      label: 'floor (silent = this ×)',
    })
    .on('change', persistExtras);
  energy
    .addBinding(config.energy, 'curve', {
      min: ENERGY_RANGE.curve.min,
      max: ENERGY_RANGE.curve.max,
      step: 0.01,
      label: 'curve (exponent)',
    })
    .on('change', persistExtras);
  energy
    .addBinding(config.energy, 'smoothingMs', {
      min: ENERGY_RANGE.smoothingMs.min,
      max: ENERGY_RANGE.smoothingMs.max,
      step: 50,
      label: 'smoothing (ms)',
    })
    .on('change', persistExtras);
  // Per-layer readout of what the lane is actually doing: the smoothed stem level
  // and the presence derived from it. This is what tells you whether the floor is
  // doing what you meant.
  const energyUi: Record<string, string> = {};
  for (let i = 0; i < k; i++) {
    const key = `e${i}`;
    energyUi[key] = '—';
    energy.addBinding(energyUi, key, {
      readonly: true,
      label: `${i} ${config.species[i]?.name ?? ''}`,
    });
  }

  let impulsePanel: ImpulsePanelHandle | null = null;
  if (opts.impulses) {
    impulsePanel = createImpulsePanel(
      tabs.map,
      opts.impulses,
      (i) => config.species[i]?.name ?? `${i}`,
    );
  }

  // ── sim ────────────────────────────────────────────────────────────────────
  // Structural first, then the global θ folders in the table's own order, then
  // one folder per layer. The order is deliberate: the emitter count changes what
  // the constellation *is*, and the θ below it describes how that constellation
  // is flown.
  const structure = tabs.sim.addFolder({ title: 'structure  (never modulated)' });
  structure
    .addBinding(config, 'emittersPerLayer', {
      min: 1,
      max: MAX_EMITTERS_PER_LAYER,
      step: 1,
      label: 'emitters / layer (redraws the constellation)',
    })
    .on('change', persistExtras);

  // One folder per distinct `folder` string, in first-mention order — so the
  // table's own grouping is the panel's, and adding a slot to a new folder needs
  // nothing here.
  const folders = new Map<string, FolderApi>();
  for (const slot of visual.globalSlots) {
    if (slot.folder === undefined) continue;
    let f = folders.get(slot.folder);
    if (!f) {
      f = tabs.sim.addFolder({ title: slot.folder, expanded: folders.size === 0 });
      folders.set(slot.folder, f);
    }
    bindSlot(f, slot, -1);
  }

  const layersRoot = tabs.sim.addFolder({
    title: opts.workbench ? 'layers  (blue band = where the music can take it)' : 'layers',
  });
  for (let i = 0; i < k; i++) {
    const f = layersRoot.addFolder({
      title: `${i} · ${config.species[i]?.name ?? ''}`,
      expanded: i === 0,
    });
    for (const slot of visual.layerSlots) bindSlot(f, slot, i);
  }

  // ── look ───────────────────────────────────────────────────────────────────
  // Static art direction, outside the modulation registry entirely. Editing these
  // edits the object the config file serialises.
  // Palette v2, in the folder builder every substrate's panel shares.
  const refreshPalette = addPaletteFolder(tabs.look, {
    palette: config.palette,
    speciesCount: k,
    speciesName: (i) => config.species[i]?.name ?? '',
    invalidate: () => sim.invalidatePalette(),
    onChange: persistExtras,
  });

  // `exposure` and `gamma` are θ slots with no `folder`, so nothing above bound
  // them; this is their one widget. The range comes from the table rather than
  // from a literal here, so the slider and the hard θ bound cannot drift apart.
  const exposureSlot = visual.globalSlots.find((s) => s.key === 'exposure');
  const refreshRender = addRenderFolder(tabs.look, {
    render: config.render,
    // `params` carries `exposure` and `gamma` as ordinary own properties (the
    // store is keyed by slot name), so it satisfies the host's shape directly and
    // the folder's bindings write straight into θ. The cast is the whole of the
    // adaptation: a wrapper object would be a second writable copy of two numbers.
    config: config.params as unknown as { exposure: number; gamma: number },
    exposureRange: {
      min: exposureSlot?.min ?? 0.002,
      max: exposureSlot?.max ?? 2,
      step: exposureSlot?.step ?? 0.001,
    },
    renderPasses: () => sim.stats().renderPasses,
    autoExposureState: () => sim.post.autoExposureState,
    invalidatePalette: () => sim.invalidatePalette(),
    // The render block is shared by reference with the mapping config and
    // exposure/gamma ride `extras`, so this one call persists the whole folder.
    onChange: persistExtras,
    // No soil field in this substrate, so the ember underlay has nothing to
    // draw from.
  });

  return {
    refresh(): void {
      const st = sim.stats();
      state.seed = String(sim.currentSeed);
      state.field = `${st.fieldW}×${st.fieldH} · ${st.activeRings} ring${
        st.activeRings === 1 ? '' : 's'
      }`;
      state.steps = String(st.stepsThisTick);

      // The two arrays are the sim's own live state, held by reference and
      // rewritten in place every tick — reading them here is the whole of the
      // energy readout, and nothing polls them anywhere else.
      const es = sim.energyState();
      for (let i = 0; i < k; i++) {
        energyUi[`e${i}`] =
          `stem ${(es.level[i] ?? 0).toFixed(2)} · presence ${(es.energy[i] ?? 1).toFixed(2)}`;
      }

      refreshPalette();
      refreshRender();
      impulsePanel?.refresh();
      workbench?.refresh();
      explorer?.refresh();
      // One call for the whole pane: tweakpane refreshes every binding it owns,
      // including the pages that are not on screen, so the visible tab is always
      // current the instant it is selected.
      pane.refresh();
    },
    dispose(): void {
      if (saveTimer !== null) clearTimeout(saveTimer);
      impulsePanel?.dispose();
      // Unsubscribes only: an export already running is owned by the session and
      // survives this panel.
      exportFolder?.dispose();
      workbench?.dispose();
      explorer?.dispose();
      pane.dispose();
    },
  };
}
