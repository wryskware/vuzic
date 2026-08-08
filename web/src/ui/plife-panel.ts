/**
 * The live control panel for particle life.
 *
 * Structurally a sibling of `panel.ts` rather than a generalisation of it: the
 * two sims share the workbench, the impulse folder and the HDR chain (all three
 * are imported here, not re-implemented), and differ in exactly the part that
 * *should* differ — which knobs exist and what they mean. Trying to drive both
 * from one table-driven panel would mean describing every slider as data, which
 * is more machinery than two hand-written folders and reads worse.
 *
 * The conventions are `panel.ts`'s, deliberately:
 *
 * - Everything binds straight to the live config object. The sim re-reads it
 *   every substep, so there are no change handlers except where GPU state has to
 *   be rebuilt — here that is exactly two places, the attraction matrix (which
 *   lives in its own buffer) and the palette (which is parsed and cached).
 * - Proxy objects for the matrix and the palette, resynced in `refresh()`, since
 *   a reroll or a file load rewrites the live config wholesale.
 * - Seed controls appear **only when there is no workbench**. With one, the seed
 *   is the world seed and it lives there, in one place rather than two.
 *
 * ## Slider ranges
 *
 * Two different bounds exist in `plife/preset.ts` and the panel picks between
 * them on purpose:
 *
 * - a slot the music moves gets its **`ModSpec` lo/hi** — the range the
 *   modulator itself wanders inside. A slider wider than that would spend most
 *   of its travel somewhere the music never goes, and dragging it there would be
 *   overwritten on the next tick anyway.
 * - a slot excluded from modulation (brightness, intensity, stretch, exposure,
 *   gamma) gets its **hard min/max**, because nothing else is ever going to
 *   write it and the whole authored range is usable.
 */
import { Pane } from 'tweakpane';
import { saveModulationLocal } from '../mapping/persist';
import type { ImpulseEngine } from '../sim/impulses';
import {
  defaultPlifeMacros,
  MACRO_LABELS,
  MACRO_RANGE,
  type PlifeSpeciesConfig,
} from '../sim/plife/config';
import type { PlifeSim } from '../sim/plife/plife';
import { coupled } from '../sim/plife/preset';
import { randomSeed, setPinnedSeed, syncUrlSeed } from '../sim/seed';
import { createImpulsePanel, type ImpulsePanelHandle } from './impulses-panel';
import type { PanelHandle, PanelWorkbench } from './panel';
import { addRenderFolder } from './render-folder';
import { createWorkbench, type WorkbenchHandle } from './workbench';

type Folder = ReturnType<Pane['addFolder']>;

interface RunState {
  seed: string;
  pin: boolean;
  particles: string;
  grid: string;
}

export function createPlifePanel(
  sim: PlifeSim,
  opts: {
    pinned: boolean;
    /** rewind the transport: a restart must not resume against an arbitrary timeline position */
    onRestart?: () => void;
    /** omit to get a panel with no modulation layer at all */
    workbench?: PanelWorkbench;
    /** omit to hide the events folder entirely */
    impulses?: ImpulseEngine;
  },
): PanelHandle {
  const config = sim.config;
  const k = config.speciesCount;

  const pane = new Pane({ title: 'terrarium · particle life' });

  const state: RunState = {
    seed: String(sim.currentSeed),
    pin: opts.pinned,
    particles: '—',
    grid: '—',
  };

  // ── macros: the performance layer, first because it is what you reach for ──
  //
  // Seven multipliers that compose outside θ, exactly where stem-follow and the
  // impulse lanes do, which is the whole reason the modulator can never write
  // over them. Everything below this folder is either θ (mirrored, overwritten
  // on the next tick while modulating) or generation settings; this is the only
  // block that is unconditionally yours.
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
  for (const { key, label } of MACRO_LABELS) {
    const r = MACRO_RANGE[key];
    macros
      .addBinding(config.macros, key, { min: r.min, max: r.max, step: 0.01, label })
      .on('change', persistMacros);
  }
  macros.addButton({ title: 'reset macros to 1' }).on('click', () => {
    Object.assign(config.macros, defaultPlifeMacros());
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
  // "alive" is the summed population *target*, not a GPU read-back — see the note
  // on `PlifeSim.aliveCount`. Whatever is still fading out is not counted.
  run.addBinding(state, 'particles', { readonly: true, label: 'particles alive' });
  run.addBinding(state, 'grid', { readonly: true, label: 'grid' });

  const refreshRender = addRenderFolder(pane, {
    render: config.render,
    config,
    // The hard θ bound from plife/preset.ts's global table, not physarum's: a
    // particle splat is already a small alpha-weighted contribution, so this sim
    // lives around 1.0 rather than around 0.01.
    exposureRange: { min: 0.05, max: 4, step: 0.01 },
    renderPasses: () => sim.stats().renderPasses,
    autoExposureState: () => sim.post.autoExposureState,
    invalidatePalette: () => sim.invalidatePalette(),
    // No soil field in this sim, so the ember underlay has nothing to draw from.
  });

  // The population lane. Outside θ entirely (like physarum's soil block), which
  // is why every widget here binds directly with no mask or mode caveat: the
  // modulator never writes any of it, in either mode.
  const pop = pane.addFolder({ title: 'population · stems → colonies', expanded: true });
  pop.addBinding(config.population, 'followStems', { label: 'follow stems (off = θ absolute)' });
  pop.addBinding(config.population, 'floor', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'floor (silent = this ×)',
  });
  pop.addBinding(config.population, 'curve', {
    min: 0.2,
    max: 4,
    step: 0.01,
    label: 'curve (exponent)',
  });
  // Deliberately allowed much slower than stem-follow's brightness smoothing: a
  // colony that sheds a third of its members every bar reads as flicker.
  pop.addBinding(config.population, 'smoothingMs', {
    min: 100,
    max: 5000,
    step: 50,
    label: 'smoothing (ms)',
  });
  pop.addBinding(config.population, 'riseTau', {
    min: 0.05,
    max: 3,
    step: 0.05,
    label: 'rise τ (s)',
  });
  pop.addBinding(config.population, 'fallTau', {
    min: 0.1,
    max: 8,
    step: 0.1,
    label: 'fall τ (s)',
  });

  const accent = pop.addFolder({ title: 'accents · novelty', expanded: false });
  accent.addBinding(config.population.accent, 'enabled', { label: 'novelty → accents' });
  accent.addBinding(config.population.accent, 'floor', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'floor (plain section)',
  });
  accent.addBinding(config.population.accent, 'boost', {
    min: 0,
    max: 3,
    step: 0.01,
    label: 'boost (headroom ×)',
  });
  accent.addBinding(config.population.accent, 'smoothingMs', {
    min: 100,
    max: 5000,
    step: 50,
    label: 'smoothing (ms)',
  });

  // Per-species readout of what the lane is actually doing: the integer target,
  // the stem multiplier behind it, and (accents only) the novelty multiplier.
  // This is what tells you whether the floor is doing what you meant.
  const popUi: Record<string, string> = {};
  for (let i = 0; i < k; i++) {
    const key = `p${i}`;
    popUi[key] = '—';
    pop.addBinding(popUi, key, {
      readonly: true,
      label: `${i} ${config.species[i]?.name ?? ''}`,
    });
  }

  // Both of these are in θ and both are modulated, so in modulated mode the
  // sliders mirror the live value and an edit lasts until the next tick — the
  // same contract every θ slot in the species folders has.
  const physics = pane.addFolder({ title: 'physics  (live values while modulating)' });
  physics.addBinding(config, 'forceGain', {
    min: 0.2,
    max: 4,
    step: 0.01,
    label: 'force gain ×',
  });
  physics.addBinding(config, 'maxSpeed', {
    min: 0.05,
    max: 1,
    step: 0.005,
    label: 'max speed (world/s)',
  });

  const speciesRoot = pane.addFolder({
    title: opts.workbench ? 'species  (live values while modulating)' : 'species',
  });
  for (let i = 0; i < k; i++) {
    const s = config.species[i];
    if (!s) continue;
    addSpeciesFolder(speciesRoot, i, s);
  }

  // How a seed *draws* the matrix below, rather than what is in it right now.
  // Nothing here is θ and nothing here is modulated: these act once, at reroll
  // time, in `genmatrix.ts`. Editing one changes nothing on screen until a new
  // draw happens — which is the entire reason the redraw button exists.
  const gen = pane.addFolder({ title: 'matrix · seeded generation', expanded: false });
  gen.addBinding(config.matrixGen, 'sigma', {
    min: 0,
    max: 1.2,
    step: 0.01,
    label: 'sigma (draw scale)',
  });
  gen.addBinding(config.matrixGen, 'symmetry', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'symmetry (1 = no chase)',
  });
  gen.addBinding(config.matrixGen, 'selfBiasAccent', {
    min: -1,
    max: 1,
    step: 0.01,
    label: 'accent self bias (− disperse)',
  });
  gen.addBinding(config.matrixGen, 'selfBias', {
    min: -1,
    max: 1,
    step: 0.01,
    label: 'self bias (− filigree)',
  });
  gen.addBinding(config.matrixGen, 'accentGain', {
    min: 0,
    max: 2,
    step: 0.01,
    label: 'accent gain ×',
  });
  gen.addBinding(config.matrixGen.rMin, 'lo', { min: 0.002, max: 0.01, step: 0.0005, label: 'r-min lo' });
  gen.addBinding(config.matrixGen.rMin, 'hi', { min: 0.002, max: 0.01, step: 0.0005, label: 'r-min hi' });
  gen.addBinding(config.matrixGen.rMax, 'lo', { min: 0.005, max: 0.02, step: 0.0005, label: 'r-max lo' });
  gen.addBinding(config.matrixGen.rMax, 'hi', { min: 0.005, max: 0.02, step: 0.0005, label: 'r-max hi' });
  if (opts.workbench) {
    const modulator = opts.workbench.modulator;
    // A workbench *reroll* already draws a fresh matrix — a new seed is a new
    // world and the draw is part of what "new world" means. This button is the
    // other half of the tuning loop: re-draw under the settings above while
    // keeping the SAME seed, so the world, the wiring and the personality are
    // unchanged and the only thing that moved is the matrix. `setSeed` is the
    // whole path — it re-runs rewire (jitter, then the generator) and applyBase.
    gen.addButton({ title: 'redraw from settings (same seed)' }).on('click', () => {
      modulator.setSeed(sim.currentSeed);
      syncMatrixProxy();
      pane.refresh();
    });
  }

  // The interaction matrix. Unlike physarum's M, which the sim re-reads from the
  // config every step, this one lives in its own GPU buffer — so every edit has
  // to push, and `uploadInteractions` is the narrowest public way to do it.
  //
  // These sliders now edit a *drawn* matrix, not the shipped constants: the seed
  // generates the whole block (see the folder above), so a reroll or a redraw
  // replaces every number here wholesale. Hand edits are still real — they go
  // straight to the GPU — they just do not survive the next draw.
  const matrixRoot = pane.addFolder({ title: 'attraction matrix A (drawn from the seed)', expanded: false });
  const proxy: Record<string, number> = {};
  for (let i = 0; i < k; i++) {
    const row = matrixRoot.addFolder({
      title: `${i} ${config.species[i]?.name ?? ''} feels…`,
      expanded: false,
    });
    for (let j = 0; j < k; j++) {
      const key = `a${i}_${j}`;
      proxy[key] = config.attraction[i * k + j] ?? 0;
      // The uncoupled cells are the primary/secondary partition made visible.
      // They are still editable — `mod: null` means "the sliders own this" — but
      // the music will never move them, and a cell you drag off zero by hand is
      // a cell you have taken out of the partition yourself.
      const uncoupled = coupled(i, j, k) ? '' : ' (uncoupled)';
      row
        .addBinding(proxy, key, {
          label: `← ${j} ${config.species[j]?.name ?? ''}${uncoupled}`,
          // The ModSpec range, not the ±2 hard bound: past about ±1.2 the pair
          // force overwhelms the hard-core term and the pair collapses.
          min: -1.2,
          max: 1.2,
          step: 0.01,
        })
        .on('change', (ev) => {
          config.attraction[i * k + j] = ev.value;
          sim.uploadInteractions();
        });
    }
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
      // A reroll or a file load rewrites the live config wholesale; the matrix
      // and palette widgets above are bound to proxies and must be re-read.
      onConfigReplaced: () => {
        syncMatrixProxy();
        pane.refresh();
      },
    });
  }

  /** The matrix and palette widgets edit proxies, so a load or reroll has to be pulled back in. */
  function syncMatrixProxy(): void {
    for (let i = 0; i < k; i++) {
      colorProxy[`c${i}`] = config.palette.colors[i] ?? '#ffffff';
      for (let j = 0; j < k; j++) {
        proxy[`a${i}_${j}`] = config.attraction[i * k + j] ?? 0;
      }
    }
  }

  return {
    refresh(): void {
      const st = sim.stats();
      state.particles = `${st.aliveParticles.toLocaleString()} / ${st.totalParticles.toLocaleString()}`;
      state.grid = `${st.gridW}×${st.gridH} cells`;
      state.seed = String(sim.currentSeed);

      // The three arrays are the sim's own live state, held by reference and
      // rewritten in place every tick — reading them here is the whole of the
      // population readout, and nothing polls them anywhere else.
      const ps = sim.popState();
      for (let i = 0; i < k; i++) {
        const target = ps.target[i] ?? 0;
        const line =
          `tgt ${target.toLocaleString()} · pop ×${(ps.popMul[i] ?? 1).toFixed(2)}`;
        popUi[`p${i}`] =
          config.species[i]?.role === 'secondary'
            ? `${line} · acc ×${(ps.accentMul[i] ?? 1).toFixed(2)}`
            : line;
      }

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

function addSpeciesFolder(root: Folder, index: number, s: PlifeSpeciesConfig): void {
  const f = root.addFolder({
    title: `${index} · ${s.name}${s.role === 'secondary' ? ' (accent)' : ''}`,
    expanded: index === 0,
  });
  // Honest label, both ways round, exactly as in physarum's panel: the modulator
  // never writes brightness, so with stem-follow off it is absolute and with it
  // on it is the base being scaled. Which one is live is stated in the workbench's
  // stem-follow folder.
  f.addBinding(s, 'brightness', {
    min: 0,
    max: 2,
    step: 0.01,
    label: 'brightness (base × stem-follow)',
  });
  f.addBinding(s, 'intensity', { min: 0, max: 4, step: 0.01, label: 'intensity (manual)' });
  // Same story one level down: θ owns this number, and the population lane's two
  // multipliers scale it rather than replacing it.
  f.addBinding(s, 'aliveFraction', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'alive fraction (base × pop-follow)',
  });
  f.addBinding(s, 'radiusScale', { min: 0.5, max: 2, step: 0.01, label: 'reach ×' });
  f.addBinding(s, 'forceScale', { min: 0.3, max: 3, step: 0.01, label: 'force ×' });
  f.addBinding(s, 'friction', { min: 0.8, max: 8, step: 0.05, label: 'friction (1/s)' });
  f.addBinding(s, 'wander', { min: 0.002, max: 0.15, step: 0.001, label: 'wander' });
  f.addBinding(s, 'size', { min: 0.0008, max: 0.006, step: 0.0001, label: 'sprite size' });
  f.addBinding(s, 'stretch', { min: 0, max: 8, step: 0.05, label: 'velocity stretch' });
}
