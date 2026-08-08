import { Pane } from 'tweakpane';
import { saveModulationLocal } from '../mapping/persist';
import type { PhysarumSim } from '../sim/physarum/physarum';
import {
  defaultPhysarumMacros,
  PHYSARUM_MACRO_LABELS,
  PHYSARUM_MACRO_RANGE,
  type AdaptiveTriple,
  type SpeciesConfig,
} from '../sim/physarum/config';
import type { ImpulseEngine } from '../sim/impulses';
import { randomSeed, setPinnedSeed, syncUrlSeed } from '../sim/seed';
import { createImpulsePanel, type ImpulsePanelHandle } from './impulses-panel';
import { addRenderFolder } from './render-folder';
import { createWorkbench, type WorkbenchHandle, type WorkbenchHost } from './workbench';

type Folder = ReturnType<Pane['addFolder']>;

export interface PanelHandle {
  refresh(): void;
  dispose(): void;
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
 * Seed controls appear here **only when there is no workbench**. With one, the
 * seed is the world seed — wiring, personality and agent placement at once — and
 * it lives in the modulation folder, in one place rather than two.
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
  },
): PanelHandle {
  const config = sim.config;
  const k = config.speciesCount;

  const pane = new Pane({ title: 'terrarium · physarum' });

  const state: RunState = {
    seed: String(sim.currentSeed),
    pin: opts.pinned,
    agents: '—',
    grid: '—',
  };

  // ── macros: the performance layer, first because it is what you reach for ──
  //
  // Five multipliers that compose outside θ, exactly where stem-follow and the
  // impulse lanes do, which is the whole reason the modulator can never write
  // over them. Everything below this folder is either θ (mirrored, overwritten
  // on the next tick while modulating) or structural; this is the only block
  // that is unconditionally yours.
  //
  // Persistence: macros ride the mapping file's opaque `extras` block, so an
  // edit here has to reach the same autosave the workbench's own bindings use.
  // The panel does it directly rather than through a new callback — it already
  // holds both halves (the sim, for the snapshot, and the modulator's config,
  // for the file) — and debounces, because these are sliders and a drag would
  // otherwise write localStorage on every pointer move.
  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  const persistMacros = (): void => {
    const wb = opts.workbench;
    if (!wb) return;
    if (saveTimer !== null) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      wb.modulator.config.extras = sim.serializeExtras();
      saveModulationLocal(wb.modulator.config, sim.simId);
    }, 400);
  };

  const macros = pane.addFolder({ title: 'macros · always yours (never modulated)' });
  for (const { key, label } of PHYSARUM_MACRO_LABELS) {
    const r = PHYSARUM_MACRO_RANGE[key];
    macros
      .addBinding(config.macros, key, { min: r.min, max: r.max, step: 0.01, label })
      .on('change', persistMacros);
  }
  macros.addButton({ title: 'reset macros to 1' }).on('click', () => {
    Object.assign(config.macros, defaultPhysarumMacros());
    persistMacros();
    pane.refresh();
  });

  const run = pane.addFolder({ title: 'run' });
  if (!opts.workbench) {
    run.addBinding(state, 'seed', { readonly: true });
    run.addBinding(state, 'pin', { label: 'pin seed' }).on('change', (ev) => {
      setPinnedSeed(ev.value ? sim.currentSeed : null);
      syncUrlSeed(ev.value ? sim.currentSeed : null);
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
    run.addButton({ title: 'restart (same seed)' }).on('click', () => {
      sim.reseed(sim.currentSeed);
      opts.onRestart?.();
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

  // The HDR chain is shared with every other substrate's panel; physarum's only
  // additions to it are the soil underlay and its own scene-exposure bound.
  const refreshRender = addRenderFolder(pane, {
    render: config.render,
    config,
    // min matches the θ bound in mapping/preset.ts
    exposureRange: { min: 0.005, max: 1.5, step: 0.001 },
    renderPasses: () => sim.stats().renderPasses,
    autoExposureState: () => sim.post.autoExposureState,
    invalidatePalette: () => sim.invalidatePalette(),
    soil: true,
  });

  const sense = pane.addFolder({ title: 'sense response', expanded: false });
  sense.addBinding(config, 'senseGain', {
    min: 0.02,
    max: 4,
    step: 0.02,
    label: 'sense gain (x)',
  });

  // Stems bypass the modulator by design: the projections do character, stems do
  // "an instrument just came in".
  const music = pane.addFolder({ title: 'music · stems (direct path)', expanded: false });
  music.addBinding(config, 'stemDrive', { label: 'stems → deposit' });
  music.addBinding(config, 'stemGain', { min: 0, max: 6, step: 0.1, label: 'stem gain' });

  // Track-scale memory. Structural, like the grid: outside θ, so nothing here is
  // modulated or slewed — you set it once for a track and let it run.
  const soil = pane.addFolder({ title: 'soil · track-scale memory', expanded: false });
  soil.addBinding(config.soil, 'debugView', { label: 'debug view (soil only)' });
  // τ ≈ 1/(1-decay) ticks; 0.999 ≈ 17 s and 0.9999 ≈ 3 min at 60 fps, which is the
  // range where "minute three remembers minute one" actually lives.
  soil.addBinding(config.soil, 'decay', {
    min: 0.99,
    max: 0.99999,
    step: 0.00001,
    label: 'decay / tick',
  });
  soil.addBinding(config.soil, 'accum', {
    min: 0,
    max: 0.02,
    step: 0.0001,
    label: 'accumulation',
  });
  soil.addBinding(config.soil, 'depositBias', {
    min: 0,
    max: 3,
    step: 0.01,
    label: 'deposit bias',
  });
  soil.addBinding(config.soil, 'senseBias', {
    min: 0,
    max: 4,
    step: 0.01,
    label: 'sense bias',
  });
  soil.addButton({ title: 'clear soil (keep world)' }).on('click', () => {
    sim.clearSoil();
  });

  // Above the modulation folder on purpose: impulses are the fastest-moving lane.
  let impulsePanel: ImpulsePanelHandle | null = null;
  if (opts.impulses) {
    impulsePanel = createImpulsePanel(pane, opts.impulses, (i) => config.species[i]?.name ?? `${i}`);
  }

  let workbench: WorkbenchHandle | null = null;
  if (opts.workbench) {
    workbench = createWorkbench(pane, {
      ...opts.workbench,
      pinned: opts.pinned,
      restart: () => opts.onRestart?.(),
      // A reroll or a file load rewrites the live config wholesale; the species
      // and matrix widgets below are bound to that object and must be re-read.
      onConfigReplaced: () => {
        syncMatrixProxy();
        pane.refresh();
      },
    });
  }

  // Static art direction, outside the modulation registry entirely (plan.md
  // Revision 2). Editing these edits the object the config file serialises.
  const palette = pane.addFolder({ title: 'palette (static — never modulated)', expanded: false });
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

  const speciesRoot = pane.addFolder({
    title: opts.workbench ? 'species  (live values while modulating)' : 'species',
  });
  for (let i = 0; i < k; i++) {
    const s = config.species[i];
    if (!s) continue;
    addSpeciesFolder(speciesRoot, i, s);
  }

  const matrixRoot = pane.addFolder({ title: 'sense matrix M', expanded: false });
  const proxy: Record<string, number> = {};
  for (let i = 0; i < k; i++) {
    const row = matrixRoot.addFolder({
      title: `${i} ${config.species[i]?.name ?? ''} senses…`,
      expanded: false,
    });
    for (let j = 0; j < k; j++) {
      const key = `m${i}_${j}`;
      proxy[key] = config.matrix[i * k + j] ?? 0;
      row
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
    }
  }

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
      pane.refresh();
    },
    dispose(): void {
      if (saveTimer !== null) clearTimeout(saveTimer);
      impulsePanel?.dispose();
      workbench?.dispose();
      pane.dispose();
    },
  };
}

function addSpeciesFolder(root: Folder, index: number, s: SpeciesConfig): void {
  const f = root.addFolder({ title: `${index} · ${s.name}`, expanded: index === 0 });
  // Honest label, both ways round (Revision 4): the modulator never writes these
  // any more, so with stem-follow off they are absolute, and with it on they are
  // the base it scales. Which one is live is stated in the stem-follow folder.
  f.addBinding(s, 'brightness', {
    min: 0,
    max: 2,
    step: 0.01,
    label: 'brightness (base × stem-follow)',
  });
  f.addBinding(s, 'intensity', { min: 0, max: 4, step: 0.01, label: 'intensity (manual)' });
  f.addBinding(s, 'deposit', { min: 0, max: 6, step: 0.01 });
  f.addBinding(s, 'decay', { min: 0.8, max: 1, step: 0.001 });
  f.addBinding(s, 'aliveFraction', { min: 0, max: 1, step: 0.01, label: 'alive fraction' });
  f.addBinding(s, 'diffuseCentre', { min: 0.111, max: 1, step: 0.005, label: 'blur centre w' });

  const a = f.addFolder({ title: 'intensity-adaptive  p1 + p2·xᵖ³', expanded: false });
  addTriple(a, 'sensor dist', s.sensorDist, 0, 80);
  addTriple(a, 'sensor angle', s.sensorAngle, -Math.PI, Math.PI);
  addTriple(a, 'rotate angle', s.rotate, -Math.PI, Math.PI);
  addTriple(a, 'move dist', s.moveDist, 0, 8);
}

function addTriple(
  parent: Folder,
  label: string,
  t: AdaptiveTriple,
  lo: number,
  hi: number,
): void {
  const g = parent.addFolder({ title: label, expanded: false });
  g.addBinding(t, 'p1', { min: lo, max: hi, step: (hi - lo) / 400, label: 'p1 base' });
  g.addBinding(t, 'p2', { min: lo - (hi - lo), max: hi, step: (hi - lo) / 400, label: 'p2 gain' });
  g.addBinding(t, 'p3', { min: 0.1, max: 4, step: 0.01, label: 'p3 exp' });
}
