import { Pane } from 'tweakpane';
import type { PhysarumSim } from '../sim/physarum/physarum';
import type { AdaptiveTriple, SpeciesConfig } from '../sim/physarum/config';
import { MAX_BLOOM_LEVELS, TONEMAPS, type ToneMap } from '../sim/render/config';
import type { ImpulseEngine } from '../sim/impulses';
import { randomSeed, setPinnedSeed } from '../sim/seed';
import { createImpulsePanel, type ImpulsePanelHandle } from './impulses-panel';
import { createWorkbench, type WorkbenchHandle, type WorkbenchHost } from './workbench';

type Folder = ReturnType<Pane['addFolder']>;

export interface PanelHandle {
  refresh(): void;
  dispose(): void;
}

/** Everything the phase-5 workbench needs from the app, minus what the panel supplies itself. */
export type PanelWorkbench = Omit<WorkbenchHost, 'onConfigReplaced'>;

interface RunState {
  seed: string;
  pin: boolean;
  agents: string;
  grid: string;
}

/**
 * The embryo of the phase-5 workbench. Everything here binds straight to the live
 * config object; the sim re-reads it every step, so there are no change handlers
 * except where GPU state has to be rebuilt (matrix upload, reseed).
 */
export function createPanel(
  sim: PhysarumSim,
  opts: {
    pinned: boolean;
    /** rewind the transport: a restart must not resume against an arbitrary timeline position */
    onRestart?: () => void;
    /** omit to get the phase-4 panel with no mapping layer at all */
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

  const run = pane.addFolder({ title: 'run' });
  run.addBinding(state, 'seed', { readonly: true });
  run.addBinding(state, 'pin', { label: 'pin seed' }).on('change', (ev) => {
    setPinnedSeed(ev.value ? sim.currentSeed : null);
  });
  run.addButton({ title: 'reseed + restart' }).on('click', () => {
    const seed = randomSeed();
    sim.reseed(seed);
    opts.onRestart?.();
    state.seed = String(seed);
    if (state.pin) setPinnedSeed(seed);
    pane.refresh();
  });
  run.addButton({ title: 'restart (same seed)' }).on('click', () => {
    sim.reseed(sim.currentSeed);
    opts.onRestart?.();
  });
  run.addBinding(config, 'paused');
  run.addButton({ title: 'single step' }).on('click', () => {
    config.paused = true;
    sim.requestSingleStep();
    pane.refresh();
  });
  run.addBinding(config, 'speed', { min: 0, max: 4, step: 0.25, label: 'sim speed' });
  run.addBinding(state, 'agents', { readonly: true, label: 'agents alive' });
  run.addBinding(state, 'grid', { readonly: true, label: 'sim grid' });

  const refreshRender = addRenderFolder(pane, sim);

  const sense = pane.addFolder({ title: 'sense response', expanded: false });
  sense.addBinding(config, 'senseGain', {
    min: 0.02,
    max: 4,
    step: 0.02,
    label: 'sense gain (x)',
  });

  // Stems bypass the simplex by design: the mapping layer does character, stems
  // do "an instrument just came in".
  const music = pane.addFolder({ title: 'music · stems (direct path)', expanded: false });
  music.addBinding(config, 'stemDrive', { label: 'stems → deposit' });
  music.addBinding(config, 'stemGain', { min: 0, max: 6, step: 0.1, label: 'stem gain' });

  // Track-scale memory. Structural, like the grid: outside θ, so nothing here is
  // blended or slewed — you set it once for a track and let it run.
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

  // Above the mapping folder on purpose: impulses are the fastest-moving lane and
  // the one being tuned in this round.
  let impulsePanel: ImpulsePanelHandle | null = null;
  if (opts.impulses) {
    impulsePanel = createImpulsePanel(pane, opts.impulses, (i) => config.species[i]?.name ?? `${i}`);
  }

  let workbench: WorkbenchHandle | null = null;
  if (opts.workbench) {
    workbench = createWorkbench(pane, {
      ...opts.workbench,
      // A load/refit/solo rewrites the live config wholesale; the species and
      // matrix widgets below are bound to that object and must be re-read.
      onConfigReplaced: () => {
        syncMatrixProxy();
        pane.refresh();
      },
    });
  }

  // Static art direction, outside the anchor system entirely (plan.md Revision 2).
  // Editing these edits the object the mapping file serialises.
  const palette = pane.addFolder({ title: 'palette (static — never blended)', expanded: false });
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
    title: opts.workbench ? 'species  (live values in mapped mode)' : 'species',
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

  /** The matrix and palette widgets edit proxies, so a load/refit has to be pulled back in. */
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
      impulsePanel?.dispose();
      workbench?.dispose();
      pane.dispose();
    },
  };
}

/**
 * Phase 7's controls, in one folder because they are one instrument: the whole
 * chain runs at once and the only way to tune a grade is to watch it move.
 *
 * Ordered the way the pixels flow — scene exposure, adaptation, bloom, tone,
 * grade, feedback — so scrolling down the folder walks the pipeline.
 */
function addRenderFolder(pane: Pane, sim: PhysarumSim): () => void {
  const config = sim.config;
  const r = config.render;
  const readout = { passes: '—', adapt: '—' };

  const root = pane.addFolder({ title: 'render · HDR chain (phase 7)', expanded: true });
  root.addBinding(readout, 'passes', { readonly: true, label: 'render passes' });
  // The measured HDR mean is the number to aim `target mean` at; the gain is
  // where the controller has settled. Both are one or two frames stale.
  root.addBinding(readout, 'adapt', { readonly: true, label: 'gain / mean' });

  // Scene exposure and gamma stay in θ: they are the two grading knobs the
  // mapper is allowed to move, and moving them per anchor predates phase 7.
  const scene = root.addFolder({ title: 'scene exposure (θ)', expanded: true });
  // min matches the θ bound in mapping/preset.ts; below it the mapper clamps back
  scene.addBinding(config, 'exposure', { min: 0.005, max: 1.5, step: 0.001, label: 'exposure' });
  scene.addBinding(config, 'gamma', { min: 1, max: 3, step: 0.05, label: 'display gamma' });
  scene.addBinding(r.grade, 'exposureEv', { min: -8, max: 8, step: 0.05, label: 'trim (stops)' });

  const auto = root.addFolder({ title: 'auto-exposure', expanded: false });
  auto.addBinding(r.grade, 'autoExposure', { label: 'enabled' });
  auto.addBinding(r.grade, 'autoTarget', { min: 0.01, max: 2, step: 0.01, label: 'target mean' });
  auto.addBinding(r.grade, 'autoTau', { min: 0.2, max: 30, step: 0.1, label: 'τ (seconds)' });
  auto.addBinding(r.grade, 'autoMinGain', { min: 0.01, max: 1, step: 0.01, label: 'min gain' });
  auto.addBinding(r.grade, 'autoMaxGain', { min: 1, max: 32, step: 0.5, label: 'max gain' });

  const bloom = root.addFolder({ title: 'bloom', expanded: true });
  bloom.addBinding(r.bloom, 'enabled');
  bloom.addBinding(r.bloom, 'threshold', { min: 0, max: 4, step: 0.01 });
  bloom.addBinding(r.bloom, 'knee', { min: 0.01, max: 2, step: 0.01, label: 'soft knee' });
  bloom.addBinding(r.bloom, 'intensity', { min: 0, max: 4, step: 0.01 });
  bloom.addBinding(r.bloom, 'levels', { min: 1, max: MAX_BLOOM_LEVELS, step: 1, label: 'mips' });

  const tone = root.addFolder({ title: 'tone + grade', expanded: true });
  tone.addBinding(r.grade, 'tonemap', {
    options: Object.fromEntries(TONEMAPS.map((t) => [t, t])) as Record<string, ToneMap>,
  });
  tone.addBinding(r.grade, 'blackPoint', { min: 0, max: 0.3, step: 0.001, label: 'black point' });
  tone.addBinding(r.grade, 'contrast', { min: 0.5, max: 2.5, step: 0.01 });
  tone.addBinding(r.grade, 'pivot', { min: 0.05, max: 0.8, step: 0.01, label: 'contrast pivot' });
  tone.addBinding(r.grade, 'saturation', { min: 0, max: 2, step: 0.01 });
  tone.addBinding(r.grade, 'vignette', { min: 0, max: 1, step: 0.01 });

  const ground = root.addFolder({ title: 'soil underlay', expanded: false });
  ground.addBinding(r.grade, 'soilTint', { min: 0, max: 1.5, step: 0.005, label: 'strength' });
  // Cached alongside the palette, and invalidated the same way.
  ground
    .addBinding(r.grade, 'soilColor', { label: 'colour (static)' })
    .on('change', () => sim.invalidatePalette());

  // Render-domain only: this cannot touch the trail field, so it is safe to
  // push. It is also the one control here that trades legibility for looks.
  const fb = root.addFolder({ title: 'feedback (render-domain trail)', expanded: false });
  fb.addBinding(r.feedback, 'amount', { min: 0, max: 0.95, step: 0.01 });
  fb.addBinding(r.feedback, 'zoom', { min: 0.98, max: 1.02, step: 0.0005 });

  return (): void => {
    readout.passes = String(sim.stats().renderPasses);
    const a = sim.post.autoExposureState;
    readout.adapt = `${a.gain.toFixed(2)}×  ·  ${a.mean.toFixed(4)}`;
  };
}

function addSpeciesFolder(root: Folder, index: number, s: SpeciesConfig): void {
  const f = root.addFolder({ title: `${index} · ${s.name}`, expanded: index === 0 });
  f.addBinding(s, 'brightness', { min: 0, max: 2, step: 0.01, label: 'brightness (θ, fast)' });
  f.addBinding(s, 'intensity', { min: 0, max: 4, step: 0.01 });
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
