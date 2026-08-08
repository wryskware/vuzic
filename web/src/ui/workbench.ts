/**
 * The workbench — dev tooling rather than a viewer feature, and after plan.md
 * Revision 3 a much smaller instrument than it was.
 *
 * The anchor era's loop was *scrub → tweak → capture → save*, i.e. authoring a
 * preset per cluster. The user's verdict on that, verbatim: "I do not care about
 * scenes. I want real time reactivity and morphing parameters." So the loop here
 * is now
 *
 *   reroll → watch → pin
 *
 * and the only continuous knobs are how hard and how fast the music pushes.
 *
 * UX decisions worth knowing:
 *
 * - **modulate vs manual.** Manual is absolute: the sliders rule and nothing
 *   writes over them (phase-4 behaviour). Modulated hands the registry's slots to
 *   the projection engine — the sliders then mirror the live values, and an edit
 *   lasts until the next tick. Slots outside the registry (the p3 exponents,
 *   exposure, gamma, stemGain) stay editable in *both* modes.
 * - **one seed UI, here.** Seed, pin and reroll live in this folder and nowhere
 *   else. The seed drives wiring, personality and the world at once, so splitting
 *   it across two panels was two half-truths.
 * - **freeze** holds the current ẑ. Parameters stop morphing and settle where
 *   they are, which is how you look at one configuration long enough to judge it.
 * - **snapshot/restore** copies agents + trails + deposit to spare GPU memory and
 *   puts them back byte for byte, and rewinds the transport to the moment of the
 *   snapshot, which is what makes A/B a real comparison rather than two different
 *   points in the song.
 *
 * Caveats, honestly:
 *
 * - Seeking the transport alone does NOT rewind the world. The sim is stateful by
 *   design, so jumping back to 1:32 replays the music against whatever the trails
 *   have become. Snapshot and restore is the only true rewind.
 * - Only one snapshot slot exists; a second snapshot overwrites the first.
 * - Restoring puts back sim state *and the seed*, so it restores the personality
 *   too — the A/B you can still run is depth/speed, which are not seeded.
 * - While the transport is paused the sim free-runs on wall-clock pacing, so two
 *   restores separated by idle time diverge slightly. Pause the sim (run ▸ paused)
 *   for a frozen comparison.
 */
import type { Pane } from 'tweakpane';
import type { PhysarumSim } from '../sim/physarum/physarum';
import type { Modulator } from '../mapping/modulation';
import { MOD_GROUPS, type ModGroup } from '../mapping/preset';
import type { TuningLog, TuningAction } from '../mapping/tuninglog';
import { randomSeed, setPinnedSeed, syncUrlSeed } from '../sim/seed';
import {
  clearModulationLocal,
  defaultModulationConfig,
  downloadText,
  modulationFits,
  parseModulation,
  pickTextFile,
  saveModulationLocal,
  serializeModulation,
} from '../mapping/persist';

export interface WorkbenchHost {
  sim: PhysarumSim;
  modulator: Modulator;
  log: TuningLog;
  trackId: string;
  /** was this run's seed pinned when it started? */
  pinned: boolean;
  /** transport position in seconds */
  time(): number;
  /** sim tick the transport is at */
  tick(): number;
  seek(seconds: number): void;
  /** drop ringing impulse envelopes and rewind the transport */
  restart(): void;
  /** the panel's own species/matrix widgets are bound to the live config and must be re-read */
  onConfigReplaced?: () => void;
}

export interface WorkbenchHandle {
  refresh(): void;
  dispose(): void;
}

interface UiState {
  enabled: boolean;
  status: string;
  source: string;
  seed: string;
  pin: boolean;
  depth: number;
  responseSpeed: number;
  frozen: boolean;
  slewFast: number;
  slewMedium: number;
  slewSlow: number;
  boundaryOn: boolean;
  snapFraction: number;
  respawnFraction: number;
  autosave: boolean;
  file: string;
  snapshotInfo: string;
  logInfo: string;
}

const GROUP_LABELS: Record<ModGroup, string> = {
  structure: 'structure (sensors, motion)',
  matrix: 'matrix M (who follows whom)',
  population: 'population (deposit, alive)',
  brightness: 'brightness (light)',
  decay: 'decay (trail memory)',
};

export function createWorkbench(pane: Pane, host: WorkbenchHost): WorkbenchHandle {
  const { modulator, sim, log } = host;
  const cfg = modulator.config;
  const ui: UiState = {
    enabled: cfg.enabled && modulator.available,
    status: '—',
    source: modulator.sourceLabel,
    seed: String(sim.currentSeed),
    pin: host.pinned,
    depth: cfg.depth,
    responseSpeed: cfg.responseSpeed,
    frozen: false,
    slewFast: cfg.slew.fast,
    slewMedium: cfg.slew.medium,
    slewSlow: cfg.slew.slow,
    boundaryOn: cfg.boundary.enabled,
    snapFraction: cfg.boundary.snapFraction,
    respawnFraction: cfg.boundary.respawnFraction,
    autosave: true,
    file: '—',
    snapshotInfo: 'none',
    logInfo: '0 entries',
  };
  const groupUi: Record<string, number> = {};
  const excursionUi: Record<string, string> = {};

  const autosave = (): void => {
    if (!ui.autosave) return;
    ui.file = saveModulationLocal(modulator.config) ? 'autosaved' : 'autosave failed';
  };

  const record = (action: TuningAction, note?: string): void => {
    log.append({
      tick: host.tick(),
      time: host.time(),
      action,
      seed: sim.currentSeed,
      source: modulator.sourceLabel,
      theta: modulator.currentTheta(),
      ...(note === undefined ? {} : { note }),
    });
    refreshLog();
  };

  const pullFromConfig = (): void => {
    const c = modulator.config;
    ui.enabled = modulator.mode === 'modulated';
    ui.depth = c.depth;
    ui.responseSpeed = c.responseSpeed;
    ui.slewFast = c.slew.fast;
    ui.slewMedium = c.slew.medium;
    ui.slewSlow = c.slew.slow;
    ui.boundaryOn = c.boundary.enabled;
    ui.snapFraction = c.boundary.snapFraction;
    ui.respawnFraction = c.boundary.respawnFraction;
    for (const g of MOD_GROUPS) groupUi[g] = c.groupDepth[g];
  };

  // ── modulation ─────────────────────────────────────────────────────────────
  const root = pane.addFolder({ title: 'modulation (revision 3)', expanded: true });

  root.addBinding(ui, 'enabled', { label: 'modulate (off = manual)' }).on('change', (ev) => {
    const want = ev.value ? 'modulated' : 'manual';
    // tweakpane re-emits 'change' from refresh(), and refresh() runs every 30
    // frames to mirror the live mode; without this guard the mirror looks like a
    // user action.
    if ((want === 'modulated') === (modulator.mode === 'modulated')) return;
    if (want === 'modulated' && !modulator.available) {
      ui.enabled = false;
      ui.status = `unavailable — ${modulator.unavailableReason}`;
      pane.refresh();
      return;
    }
    modulator.setMode(want);
    modulator.config.enabled = want === 'modulated';
    autosave();
    host.onConfigReplaced?.();
  });
  root.addBinding(ui, 'status', { readonly: true, label: '' });
  root.addBinding(ui, 'source', { readonly: true, label: 'input' });

  // ── the world seed. The ONE place seeds are shown or changed. ───────────────
  const world = root.addFolder({ title: 'world seed (wiring + personality + sim)', expanded: true });
  world.addBinding(ui, 'seed', { readonly: true, label: 'seed' });
  world.addBinding(ui, 'pin', { label: 'pin (survives reload)' }).on('change', (ev) => {
    // localStorage and ?seed= are pinned together: unticking has to clear both or
    // a URL param would keep resurrecting the old world on reload.
    setPinnedSeed(ev.value ? sim.currentSeed : null);
    syncUrlSeed(ev.value ? sim.currentSeed : null);
    record(ev.value ? 'pin' : 'unpin');
  });
  world.addButton({ title: '🎲 reroll world (new seed)' }).on('click', () => {
    // sim.reseed fires onSeedChange, which is what re-keys the modulator's
    // projections and personality and the impulse hotspots. One act, one seed.
    const seed = randomSeed();
    sim.reseed(seed);
    host.restart();
    ui.seed = String(seed);
    // An existing ?seed= must follow the reroll, or the address bar names a world
    // that is no longer running and wins again on the next reload.
    syncUrlSeed(seed);
    if (ui.pin) setPinnedSeed(seed);
    record('reroll');
    host.onConfigReplaced?.();
    pane.refresh();
  });
  world.addButton({ title: 're-run this world (same seed)' }).on('click', () => {
    sim.reseed(sim.currentSeed);
    host.restart();
    host.onConfigReplaced?.();
    pane.refresh();
  });

  // ── depth ──────────────────────────────────────────────────────────────────
  const depth = root.addFolder({ title: 'depth', expanded: true });
  depth
    .addBinding(ui, 'depth', { min: 0, max: 4, step: 0.01, label: 'global ×' })
    .on('change', (ev) => {
      modulator.config.depth = ev.value;
      autosave();
    });
  for (const g of MOD_GROUPS) {
    groupUi[g] = cfg.groupDepth[g];
    depth
      .addBinding(groupUi, g, { min: 0, max: 3, step: 0.01, label: GROUP_LABELS[g] })
      .on('change', (ev) => {
        modulator.config.groupDepth[g] = ev.value;
        autosave();
      });
  }

  // ── response ───────────────────────────────────────────────────────────────
  const response = root.addFolder({ title: 'response speed', expanded: false });
  response
    .addBinding(ui, 'responseSpeed', { min: 0.05, max: 8, step: 0.05, label: 'speed ×' })
    .on('change', (ev) => {
      modulator.config.responseSpeed = ev.value;
      autosave();
    });
  response.addBinding(ui, 'frozen', { label: 'freeze input (hold ẑ)' }).on('change', (ev) => {
    modulator.frozen = ev.value;
  });
  const bindSlew = (
    key: 'slewFast' | 'slewMedium' | 'slewSlow',
    field: 'fast' | 'medium' | 'slow',
    label: string,
    max: number,
  ): void => {
    response.addBinding(ui, key, { min: 0, max, step: 0.01, label }).on('change', (ev) => {
      modulator.config.slew[field] = ev.value;
      modulator.syncRates();
      autosave();
    });
  };
  bindSlew('slewFast', 'fast', 'fast τ (light/deposit/angle)', 2);
  bindSlew('slewMedium', 'medium', 'medium τ (geometry)', 10);
  bindSlew('slewSlow', 'slow', 'slow τ (decay/alive/M)', 60);

  // ── live readout ───────────────────────────────────────────────────────────
  const live = root.addFolder({ title: 'live excursion (mean |tanh|)', expanded: true });
  for (const g of MOD_GROUPS) {
    excursionUi[g] = '—';
    live.addBinding(excursionUi, g, { readonly: true, label: GROUP_LABELS[g] });
  }

  // ── section boundaries (an event, not a scene) ─────────────────────────────
  const boundary = root.addFolder({ title: 'section boundaries (optional)', expanded: false });
  boundary.addBinding(ui, 'boundaryOn', { label: 'step + re-seed on boundary' }).on('change', (ev) => {
    modulator.config.boundary.enabled = ev.value;
    autosave();
  });
  boundary
    .addBinding(ui, 'snapFraction', { min: 0, max: 1, step: 0.01, label: 'slow snap' })
    .on('change', (ev) => {
      modulator.config.boundary.snapFraction = ev.value;
      autosave();
    });
  boundary
    .addBinding(ui, 'respawnFraction', { min: 0, max: 1, step: 0.01, label: 'respawn frac' })
    .on('change', (ev) => {
      modulator.config.boundary.respawnFraction = ev.value;
      autosave();
    });
  // Fire a boundary event by hand, to see what a section change does without
  // waiting for one. The key walks so repeated presses are not identical.
  let manualReseedKey = 1000;
  boundary.addButton({ title: 'test: re-seed now' }).on('click', () => {
    sim.partialReseed(manualReseedKey++, modulator.config.boundary.respawnFraction);
  });

  // ── modulation file ────────────────────────────────────────────────────────
  const file = pane.addFolder({ title: 'modulation file (depths + palette + grade)', expanded: false });
  file.addBinding(ui, 'file', { readonly: true, label: '' });
  file.addBinding(ui, 'autosave', { label: 'autosave (localStorage)' });
  file.addButton({ title: 'download modulation.json' }).on('click', () => {
    downloadText(`modulation-${host.trackId}.json`, serializeModulation(modulator.config));
    record('save');
    ui.file = 'downloaded';
  });
  file.addButton({ title: 'load modulation.json…' }).on('click', () => {
    void pickTextFile().then((text) => {
      if (text === null) return;
      try {
        const parsed = parseModulation(text);
        if (!modulationFits(parsed, sim.config.speciesCount)) {
          ui.file = `rejected: authored for K=${parsed.speciesCount}`;
          pane.refresh();
          return;
        }
        modulator.setConfig(parsed);
        modulator.setMode(parsed.enabled && modulator.available ? 'modulated' : 'manual');
        pullFromConfig();
        ui.file = 'loaded';
        host.onConfigReplaced?.();
        pane.refresh();
      } catch (err) {
        ui.file = String((err as Error).message);
        pane.refresh();
      }
    });
  });
  file.addButton({ title: 'save to browser now' }).on('click', () => {
    ui.file = saveModulationLocal(modulator.config) ? 'saved' : 'save failed';
    pane.refresh();
  });
  file.addButton({ title: 'reset to defaults' }).on('click', () => {
    clearModulationLocal();
    modulator.setConfig(defaultModulationConfig(sim.config));
    modulator.setMode(modulator.available ? 'modulated' : 'manual');
    pullFromConfig();
    ui.file = 'reset';
    host.onConfigReplaced?.();
    pane.refresh();
  });

  // ── A/B ────────────────────────────────────────────────────────────────────
  const ab = pane.addFolder({ title: 'A/B  ·  sim state snapshot', expanded: false });
  ab.addBinding(ui, 'snapshotInfo', { readonly: true, label: '' });
  let snapshotTime = 0;
  ab.addButton({ title: 'snapshot state' }).on('click', () => {
    if (!sim.snapshot()) return;
    snapshotTime = host.time();
    ui.snapshotInfo = `taken at ${snapshotTime.toFixed(2)}s`;
    record('snapshot');
    pane.refresh();
  });
  ab.addButton({ title: 'restore + rewind' }).on('click', () => {
    if (!sim.restoreSnapshot()) {
      ui.snapshotInfo = 'no snapshot';
      pane.refresh();
      return;
    }
    host.seek(snapshotTime);
    ui.snapshotInfo = `restored to ${snapshotTime.toFixed(2)}s`;
    record('restore');
    host.onConfigReplaced?.();
    pane.refresh();
  });
  ab.addButton({ title: 'drop snapshot (free GPU memory)' }).on('click', () => {
    sim.clearSnapshot();
    ui.snapshotInfo = 'none';
    pane.refresh();
  });

  // ── tuning log ─────────────────────────────────────────────────────────────
  const logFolder = pane.addFolder({ title: 'tuning log (seed preferences)', expanded: false });
  logFolder.addBinding(ui, 'logInfo', { readonly: true, label: '' });
  logFolder.addButton({ title: 'log: keep this world' }).on('click', () => {
    record('keep');
    pane.refresh();
  });
  logFolder.addButton({ title: 'log: discard this world' }).on('click', () => {
    record('discard');
    pane.refresh();
  });
  logFolder.addButton({ title: 'export .jsonl' }).on('click', () => {
    downloadText(`tuning-${host.trackId}.jsonl`, log.toJsonl(), 'application/x-ndjson');
  });
  logFolder.addButton({ title: 'clear log' }).on('click', () => {
    log.clear();
    refreshLog();
    pane.refresh();
  });

  function refreshLog(): void {
    ui.logInfo = `${log.size} entries`;
  }
  refreshLog();
  pullFromConfig();

  return {
    refresh(): void {
      const ex = modulator.excursions();
      for (const g of MOD_GROUPS) {
        const depthOf = modulator.config.depth * modulator.config.groupDepth[g];
        excursionUi[g] =
          modulator.mode === 'modulated'
            ? `${(ex[g] * 100).toFixed(0)}%  ·  ×${depthOf.toFixed(2)}`
            : '— (manual)';
      }
      if (!modulator.available) {
        ui.status = `unavailable — ${modulator.unavailableReason}`;
      } else if (modulator.mode === 'modulated') {
        ui.status = modulator.frozen
          ? `frozen · ${modulator.modulatedCount} params held`
          : `modulating ${modulator.modulatedCount} params from ${modulator.sourceLabel}`;
      } else {
        ui.status = 'manual · sliders are absolute';
      }
      ui.enabled = modulator.mode === 'modulated';
      ui.seed = String(sim.currentSeed);
      if (sim.hasSnapshot && ui.snapshotInfo === 'none') ui.snapshotInfo = 'held';
    },
    dispose(): void {
      root.dispose();
      file.dispose();
      ab.dispose();
      logFolder.dispose();
    },
  };
}
