import { Pane, type FolderApi, type TabPageApi } from 'tweakpane';
import { saveModulationLocal } from '../mapping/persist';
import type { PhysarumSim } from '../sim/physarum/physarum';
import {
  defaultPhysarumMacros,
  MAX_SPECIES_SCALE,
  MIN_SPECIES_SCALE,
  PHYSARUM_MACRO_LABELS,
  PHYSARUM_MACRO_RANGE,
  type AdaptiveTriple,
  type SpeciesConfig,
} from '../sim/physarum/config';
import type { ImpulseEngine } from '../sim/impulses';
import { randomSeed, setPinnedSeed, syncUrlSeed } from '../sim/seed';
import {
  createExplorePanel,
  type ExplorePanelHandle,
  type ExplorerPanelHost,
} from './explore-panel';
import { createImpulsePanel, type ImpulsePanelHandle } from './impulses-panel';
import {
  createModBands,
  slotLookup,
  type ModBands,
  type NumericKey,
  type SliderParams,
} from './mod-fill';
import { addRenderFolder } from './render-folder';
import { createExportFolder, type ExportPanelHost } from './export-panel';
import { createSimFolder, type SimPanelHost } from './sim-panel';
import { createTrackFolder, type TrackPanelHost } from './track-panel';
import { createWorkbench, type WorkbenchHandle, type WorkbenchHost } from './workbench';

type Folder = FolderApi;

export interface PanelHandle {
  refresh(): void;
  dispose(): void;
}

/**
 * Anything a folder or a widget can be mounted into.
 *
 * `FolderApi` and `TabPageApi` expose the same container surface, and since the
 * panel became tabbed the shared builders (the HDR chain, the impulse lane, the
 * explorer, the workbench) are handed whichever one their content belongs on
 * rather than the pane itself. Typing them by the surface instead of by `Pane`
 * is what lets one builder mount into a tab page, a folder, or a subfolder
 * without knowing which it got.
 */
export type PanelContainer = Pick<FolderApi, 'addFolder' | 'addBinding' | 'addButton'>;

/**
 * The six task surfaces the panel is organised into. They are tasks, not
 * subsystems: "play" is what you touch while the music runs, "map" is how the
 * music reaches the sim, "sim" is the substrate's own physics, and so on. The
 * old flat column was ordered by the order the features were *built*, which is
 * the one ordering nobody is ever looking for.
 *
 * `explore` and `data` are null when their owning module was not supplied —
 * tweakpane's pages are fixed at creation, so an absent module means the page is
 * never created rather than created and left empty.
 */
export interface PanelTabs {
  play: TabPageApi;
  explore: TabPageApi | null;
  map: TabPageApi;
  sim: TabPageApi;
  look: TabPageApi;
  data: TabPageApi | null;
}

/**
 * Build the one tab bar both substrates' panels hang everything off. Shared
 * rather than duplicated because the tab *set* is a property of the workbench as
 * a whole, not of physarum or plife: the two sims differ in which folders land
 * on `sim` and `map`, never in which tabs exist.
 */
export function createPanelTabs(
  pane: Pane,
  present: { explorer: boolean; workbench: boolean },
): PanelTabs {
  const keys: string[] = ['play'];
  if (present.explorer) keys.push('explore');
  keys.push('map', 'sim', 'look');
  if (present.workbench) keys.push('data');

  const tab = pane.addTab({ pages: keys.map((title) => ({ title })) });
  const byKey = new Map<string, TabPageApi>();
  keys.forEach((key, i) => {
    const page = tab.pages[i];
    if (page) byKey.set(key, page);
  });
  const need = (key: string): TabPageApi => {
    const page = byKey.get(key);
    if (!page) throw new Error(`panel tab "${key}" was not created`);
    return page;
  };

  return {
    play: need('play'),
    explore: byKey.get('explore') ?? null,
    map: need('map'),
    sim: need('sim'),
    look: need('look'),
    data: byKey.get('data') ?? null,
  };
}

/**
 * Everything the workbench needs from the app, minus what the panel supplies
 * itself. `restart` comes from the panel's own `onRestart`.
 */
export type PanelWorkbench = Omit<WorkbenchHost, 'onConfigReplaced' | 'pinned' | 'restart'>;

interface RunState {
  seed: string;
  pin: boolean;
  agents: string;
  grid: string;
}

/**
 * The live control panel. Everything here binds straight to the live config
 * object; the sim re-reads it every step, so there are no change handlers except
 * where GPU state has to be rebuilt (matrix upload, reseed).
 *
 * Seed controls appear in the `run` folder **only when there is no workbench**.
 * With one, the seed is the world seed — wiring, personality and agent placement
 * at once — and it gets its own folder at the top of the play tab, in one place
 * rather than two.
 */
export function createPanel(
  sim: PhysarumSim,
  opts: {
    pinned: boolean;
    /** rewind the transport: a restart must not resume against an arbitrary timeline position */
    onRestart?: () => void;
    /** omit to get the phase-4 panel with no modulation layer at all */
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
  const k = config.speciesCount;

  const pane = new Pane({ title: 'terrarium · physarum' });
  // Created before any binding: it decorates each slider as it is built, and it
  // installs the `markApplied` pairing on `pane.refresh` (see mod-fill.ts) that
  // every refresh below then inherits for free. Null without a workbench —
  // there is no modulator to draw a range from, and nothing to protect.
  const bands: ModBands | null = opts.workbench
    ? createModBands(pane, opts.workbench.modulator)
    : null;
  /** θ slot names are the registry's, so a field's own name is the lookup key. */
  const slotOf = slotLookup(sim.registry().names);
  const tabs = createPanelTabs(pane, {
    explorer: opts.explorer !== undefined,
    workbench: opts.workbench !== undefined,
  });

  const state: RunState = {
    seed: String(sim.currentSeed),
    pin: opts.pinned,
    agents: '—',
    grid: '—',
  };

  // ── explore ────────────────────────────────────────────────────────────────
  // Its own tab rather than the first folder of the play tab: the mode suspends
  // the live sim, so while it is running every other tab describes something you
  // are not looking at. A tab you switch away from says that better than a folder
  // you scroll past.
  const explorer: ExplorePanelHandle | null =
    opts.explorer && tabs.explore ? createExplorePanel(tabs.explore, opts.explorer) : null;

  // ── play ───────────────────────────────────────────────────────────────────
  // The track picker is the first thing on the tab: it is the one control here
  // that belongs to neither the sim nor the mapping, and picking a song comes
  // before shaping what it drives.
  if (opts.tracks) createTrackFolder(tabs.play, opts.tracks);
  const exportFolder = opts.exports ? createExportFolder(tabs.play, opts.exports) : null;
  // And directly under it, the other "what am I looking at" decision. Same shape,
  // same reason for the placement, and it is mounted from all three panels
  // identically — see `createSimFolder`.
  if (opts.sims) createSimFolder(tabs.play, opts.sims);

  // The world seed is created here, above the macros, rather than by the
  // workbench — the workbench fills it in. Ordering is mount order in tweakpane,
  // and this is the one folder that has to sit above a folder the panel owns.
  const seedFolder = opts.workbench
    ? tabs.play.addFolder({ title: 'world seed (wiring + personality + sim)' })
    : null;

  // Five multipliers that compose outside θ, exactly where stem-follow and the
  // impulse lanes do, which is the whole reason the modulator can never write
  // over them. Everything on the sim tab is either θ (mirrored, overwritten on
  // the next tick while modulating) or structural; this is the only block that
  // is unconditionally yours.
  //
  // Persistence: the macros, the soil block and the direct stems→deposit path
  // all ride the mapping file's opaque `extras` block, so an edit to any of them
  // has to reach the same autosave the workbench's own bindings use. The panel
  // does it directly rather than through a new callback — it already holds both
  // halves (the sim, for the snapshot, and the modulator's config, for the file)
  // — and debounces, because these are sliders and a drag would otherwise write
  // localStorage on every pointer move.
  //
  // It re-serialises the *whole* block rather than the field that changed, which
  // is why one saver covers three folders and why adding a fourth needs nothing
  // here beyond an `.on('change', …)`.
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

  const macros = tabs.play.addFolder({ title: 'macros · performance layer' });
  for (const { key, label } of PHYSARUM_MACRO_LABELS) {
    const r = PHYSARUM_MACRO_RANGE[key];
    macros
      .addBinding(config.macros, key, { min: r.min, max: r.max, step: 0.01, label })
      .on('change', persistExtras);
  }
  macros.addButton({ title: 'reset macros to 1' }).on('click', () => {
    Object.assign(config.macros, defaultPhysarumMacros());
    persistExtras();
    pane.refresh();
  });

  // The workbench mounts into three tabs at once: the modulation headline lands
  // on play (right here, under the macros), the driver bank and everything that
  // explains it lands on map, and the file/snapshot/log trio lands on data.
  let workbench: WorkbenchHandle | null = null;
  if (opts.workbench && seedFolder && tabs.data) {
    workbench = createWorkbench(
      { pane, seed: seedFolder, play: tabs.play, map: tabs.map, data: tabs.data },
      {
        ...opts.workbench,
        pinned: opts.pinned,
        restart: () => opts.onRestart?.(),
        // A reroll or a file load rewrites the live config wholesale; the species
        // and matrix widgets are bound to that object and must be re-read.
        onConfigReplaced: () => {
          syncMatrixProxy();
          pane.refresh();
        },
      },
    );
  }

  // Last on the tab and collapsed when there is a workbench: with one, this is
  // two readouts and a pause button. Without one it also carries the seed, which
  // is worth having open.
  const run = tabs.play.addFolder({ title: 'run', expanded: !opts.workbench });
  if (!opts.workbench) {
    run.addBinding(state, 'seed', { readonly: true });
    run.addBinding(state, 'pin', { label: 'pin seed' }).on('change', (ev) => {
      setPinnedSeed(ev.value ? sim.currentSeed : null);
      syncUrlSeed(ev.value ? sim.currentSeed : null);
    });
    run.addButton({ title: 'restart (same seed)' }).on('click', () => {
      sim.reseed(sim.currentSeed);
      opts.onRestart?.();
    });
    run.addButton({ title: 'reseed + restart' }).on('click', () => {
      const seed = randomSeed();
      sim.reseed(seed);
      opts.onRestart?.();
      state.seed = String(seed);
      // Keep an existing ?seed= pointing at the live world; without this a URL
      // param outranks both the reroll and the pin on the next reload.
      syncUrlSeed(seed);
      if (state.pin) setPinnedSeed(seed);
      pane.refresh();
    });
  }
  run.addBinding(config, 'paused');
  run.addButton({ title: 'single step' }).on('click', () => {
    config.paused = true;
    sim.requestSingleStep();
    pane.refresh();
  });
  run.addBinding(config, 'speed', { min: 0, max: 4, step: 0.25, label: 'sim speed' });
  run.addBinding(state, 'agents', { readonly: true, label: 'agents alive' });
  run.addBinding(state, 'grid', { readonly: true, label: 'sim grid' });

  // ── map ────────────────────────────────────────────────────────────────────
  // Below the workbench's own folders, which mounted above: the direct stem path
  // and the impulse lane are the two ways the music reaches the sim *without*
  // going through the driver bank, so they read as the tail of the same story.
  //
  // Stems bypass the modulator by design: the projections do character, stems do
  // "an instrument just came in".
  const music = tabs.map.addFolder({ title: 'music · stems (direct path)', expanded: false });
  // Both persist through `persistExtras`: they ride the mapping file's `extras`
  // block now, and an edit that only lived in memory would be lost on the next
  // reload while the macro next to it survived.
  music.addBinding(config, 'stemDrive', { label: 'stems → deposit' }).on('change', persistExtras);
  music
    .addBinding(config, 'stemGain', { min: 0, max: 6, step: 0.1, label: 'stem gain' })
    .on('change', persistExtras);

  let impulsePanel: ImpulsePanelHandle | null = null;
  if (opts.impulses) {
    impulsePanel = createImpulsePanel(tabs.map, opts.impulses, (i) => config.species[i]?.name ?? `${i}`);
  }

  // ── sim ────────────────────────────────────────────────────────────────────
  // θ, but out of the registry since 2026-08-08 (see mapping/preset.ts): absolute
  // in both modes, so no band. `bands.add` is still called, so the day it goes
  // back into the registry the band appears without anyone remembering this line.
  const sense = tabs.sim.addFolder({ title: 'sense response (absolute — never modulated)' });
  const senseGain: SliderParams = { min: 0.02, max: 4, step: 0.02, label: 'sense gain (x)' };
  bands?.add(
    sense.addBinding(config, 'senseGain', senseGain),
    slotOf('senseGain'),
    senseGain.min,
    senseGain.max,
  );

  // Track-scale memory. Structural, like the grid: outside θ, so nothing here is
  // modulated or slewed — you set it once for a track and let it run.
  const soil = tabs.sim.addFolder({ title: 'soil · track-scale memory', expanded: false });
  soil.addBinding(config.soil, 'debugView', { label: 'debug view (soil only)' });
  // τ ≈ 1/(1-decay) ticks; 0.999 ≈ 17 s and 0.9999 ≈ 3 min at 60 fps, which is the
  // range where "minute three remembers minute one" actually lives.
  soil
    .addBinding(config.soil, 'decay', {
      min: 0.99,
      max: 0.99999,
      step: 0.00001,
      label: 'decay / tick',
    })
    .on('change', persistExtras);
  soil
    .addBinding(config.soil, 'accum', {
      min: 0,
      max: 0.02,
      step: 0.0001,
      label: 'accumulation',
    })
    .on('change', persistExtras);
  soil
    .addBinding(config.soil, 'depositBias', {
      min: 0,
      max: 3,
      step: 0.01,
      label: 'deposit bias',
    })
    .on('change', persistExtras);
  soil
    .addBinding(config.soil, 'senseBias', {
      min: 0,
      max: 4,
      step: 0.01,
      label: 'sense bias',
    })
    .on('change', persistExtras);
  soil.addButton({ title: 'clear soil (keep world)' }).on('click', () => {
    sim.clearSoil();
  });

  const speciesRoot = tabs.sim.addFolder({
    title: opts.workbench ? 'species  (blue band = where the music can take it)' : 'species',
  });
  for (let i = 0; i < k; i++) {
    const s = config.species[i];
    if (!s) continue;
    addSpeciesFolder(speciesRoot, i, s, bands, slotOf);
  }

  const matrixRoot = tabs.sim.addFolder({ title: 'sense matrix M', expanded: false });
  const proxy: Record<string, number> = {};
  for (let i = 0; i < k; i++) {
    const row = matrixRoot.addFolder({
      title: `${i} ${config.species[i]?.name ?? ''} senses…`,
      expanded: false,
    });
    for (let j = 0; j < k; j++) {
      const key = `m${i}_${j}`;
      proxy[key] = config.matrix[i * k + j] ?? 0;
      const cell = row
        .addBinding(proxy, key, {
          label: `← ${j} ${config.species[j]?.name ?? ''}`,
          min: -2,
          max: 2,
          step: 0.01,
        })
        .on('change', (ev) => {
          config.matrix[i * k + j] = ev.value;
          sim.uploadMatrix();
        });
      // Every cell of M is modulated (the diagonal on its own, kept positive
      // spec), so every cell gets a band. The slider is the ±2 hard bound while
      // the spec wanders in ±1.7, which is exactly why the band is clamped to the
      // spec range and not just to base ± half.
      bands?.add(cell, slotOf(`M[${i}][${j}]`), -2, 2);
    }
  }

  // ── look ───────────────────────────────────────────────────────────────────
  // Static art direction, outside the modulation registry entirely (plan.md
  // Revision 2). Editing these edits the object the config file serialises.
  const palette = tabs.look.addFolder({ title: 'palette (static — never modulated)', expanded: false });
  const colorProxy: Record<string, string> = {};
  for (let i = 0; i < k; i++) {
    const key = `c${i}`;
    colorProxy[key] = config.palette.colors[i] ?? '#ffffff';
    palette
      .addBinding(colorProxy, key, { label: `${i} ${config.species[i]?.name ?? ''}` })
      .on('change', (ev) => {
        config.palette.colors[i] = ev.value;
        sim.invalidatePalette();
      });
  }
  // The sim linearises the palette once and caches it, so every edit here has to
  // say so — these are the only widgets that touch it.
  palette
    .addBinding(config.palette, 'saturation', { min: 0, max: 2, step: 0.01 })
    .on('change', () => sim.invalidatePalette());
  palette
    .addBinding(config.palette, 'brightness', { min: 0, max: 2, step: 0.01 })
    .on('change', () => sim.invalidatePalette());

  // The HDR chain is shared with every other substrate's panel; physarum's only
  // additions to it are the soil underlay and its own scene-exposure bound.
  const refreshRender = addRenderFolder(tabs.look, {
    render: config.render,
    config,
    // min matches the θ bound in mapping/preset.ts
    exposureRange: { min: 0.005, max: 1.5, step: 0.001 },
    renderPasses: () => sim.stats().renderPasses,
    autoExposureState: () => sim.post.autoExposureState,
    invalidatePalette: () => sim.invalidatePalette(),
    onChange: persistExtras,
    soil: true,
  });

  /** The matrix and palette widgets edit proxies, so a load or reroll has to be pulled back in. */
  function syncMatrixProxy(): void {
    for (let i = 0; i < k; i++) {
      colorProxy[`c${i}`] = config.palette.colors[i] ?? '#ffffff';
      for (let j = 0; j < k; j++) {
        proxy[`m${i}_${j}`] = config.matrix[i * k + j] ?? 0;
      }
    }
  }

  return {
    refresh(): void {
      const st = sim.stats();
      state.agents = `${st.aliveAgents.toLocaleString()} / ${st.totalAgents.toLocaleString()}`;
      state.grid = `${st.gridW}×${st.gridH}×${k}`;
      state.seed = String(sim.currentSeed);
      syncMatrixProxy();
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

function addSpeciesFolder(
  root: Folder,
  index: number,
  s: SpeciesConfig,
  bands: ModBands | null,
  slotOf: (name: string) => number,
): void {
  const f = root.addFolder({ title: `${index} · ${s.name}`, expanded: index === 0 });
  /**
   * One species θ slider plus its band. The registry names a species slot
   * `species<i>.<field>` and the field *is* the config key, so the slot lookup
   * needs nothing plumbed through — which is what keeps this honest when a slot
   * is added or reordered: a wrong name resolves to -1 and draws no band, rather
   * than silently decorating a neighbouring slider.
   */
  const bind = <K extends NumericKey<SpeciesConfig> & string>(
    key: K,
    params: SliderParams,
  ): void => {
    const b = f.addBinding(s, key, params);
    bands?.add(b, slotOf(`species${index}.${String(key)}`), params.min, params.max);
  };
  // First control in the folder, and the only per-species slider that changes the
  // *shape* of the network rather than its light or its rate. It is here, at the
  // top, because the thing it replaces was two AdaptiveTriples three folders down
  // — sensor dist p1/p2 and move dist p1/p2 — which nobody reaches for mid-track
  // and which cannot be moved together by hand at all.
  bind('scale', {
    min: MIN_SPECIES_SCALE,
    max: MAX_SPECIES_SCALE,
    step: 0.01,
    label: 'scale ×  (sensor + move dist)',
  });
  // Honest label, both ways round (Revision 4): the modulator never writes these
  // any more, so with stem-follow off they are absolute, and with it on they are
  // the base it scales. Which one is live is stated in the stem-follow folder.
  // No band on either, for the same reason.
  bind('brightness', { min: 0, max: 2, step: 0.01, label: 'brightness (base × stem-follow)' });
  bind('intensity', { min: 0, max: 4, step: 0.01, label: 'intensity (manual)' });
  bind('deposit', { min: 0, max: 6, step: 0.01, label: 'deposit' });
  bind('decay', { min: 0.8, max: 1, step: 0.001, label: 'decay' });
  bind('aliveFraction', { min: 0, max: 1, step: 0.01, label: 'alive fraction' });
  bind('diffuseCentre', { min: 0.111, max: 1, step: 0.005, label: 'blur centre w' });

  const a = f.addFolder({ title: 'intensity-adaptive  p1 + p2·xᵖ³', expanded: false });
  const triple = (label: string, field: string, t: AdaptiveTriple, lo: number, hi: number): void =>
    addTriple(a, label, `species${index}.${field}`, t, lo, hi, bands, slotOf);
  triple('sensor dist', 'sensorDist', s.sensorDist, 0, 80);
  triple('sensor angle', 'sensorAngle', s.sensorAngle, -Math.PI, Math.PI);
  triple('rotate angle', 'rotate', s.rotate, -Math.PI, Math.PI);
  triple('move dist', 'moveDist', s.moveDist, 0, 8);
}

/**
 * One `p1 + p2·xᵖ³` curve. p1 and p2 are modulated and get bands; **p3 does not
 * and never will** — it is the shape of the response rather than a value, and
 * plan.md Revision 3 keeps exponents fixed. `bands.add` no-ops on it because its
 * registry slot carries no ModSpec, so the exclusion is stated in exactly one
 * place (mapping/preset.ts) rather than restated here.
 */
function addTriple(
  parent: Folder,
  label: string,
  slotPrefix: string,
  t: AdaptiveTriple,
  lo: number,
  hi: number,
  bands: ModBands | null,
  slotOf: (name: string) => number,
): void {
  const g = parent.addFolder({ title: label, expanded: false });
  const step = (hi - lo) / 400;
  const p1: SliderParams = { min: lo, max: hi, step, label: 'p1 base' };
  const p2: SliderParams = { min: lo - (hi - lo), max: hi, step, label: 'p2 gain' };
  const p3: SliderParams = { min: 0.1, max: 4, step: 0.01, label: 'p3 exp' };
  bands?.add(g.addBinding(t, 'p1', p1), slotOf(`${slotPrefix}.p1`), p1.min, p1.max);
  bands?.add(g.addBinding(t, 'p2', p2), slotOf(`${slotPrefix}.p2`), p2.min, p2.max);
  bands?.add(g.addBinding(t, 'p3', p3), slotOf(`${slotPrefix}.p3`), p3.min, p3.max);
}
