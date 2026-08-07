/**
 * The workbench — phase 5's explicit deliverable, and dev tooling rather than a
 * viewer feature. Everything here exists to make the loop
 *
 *   scrub → tweak → snapshot → tweak again → restore → compare → capture → save
 *
 * fast enough that a human will actually do it a hundred times in an evening.
 *
 * UX decisions worth knowing:
 *
 * - **manual vs mapped.** In manual the sliders rule (phase-4 behaviour). In
 *   mapped the simplex writes θ into the same config object the sliders are bound
 *   to, so they turn into a live read-out: you can see the blend moving, and an
 *   edit lasts until the next tick. That is intentional — mapped mode is for
 *   watching, solo is for tuning.
 * - **solo** loads one anchor's preset exactly and then *stops* the mapper, which
 *   hands the sliders back. Tune, then "capture" to store them into that anchor.
 *   Leaving solo resumes blending from wherever you left the config, so there is
 *   no step.
 * - **snapshot/restore** copies agents + trails + deposit to spare GPU memory and
 *   puts them back byte for byte, and it also rewinds the transport to the
 *   moment of the snapshot, which is what makes A/B a real comparison rather
 *   than two different points in the song.
 *
 * Caveats of the scrub → tweak → replay loop, honestly:
 *
 * - Seeking the transport alone does NOT rewind the world. The sim is stateful by
 *   design, so jumping back to 1:32 replays the music against whatever the trails
 *   have become. Take a snapshot at the moment you care about and restore it —
 *   that is the only true rewind.
 * - Only one snapshot slot exists. A second snapshot overwrites the first.
 * - Restoring puts back sim state and the seed, not θ. That asymmetry is the
 *   point of A/B: same world, different parameters.
 * - While the transport is paused the sim free-runs on wall-clock pacing, so two
 *   restores separated by idle time diverge slightly. Pause the sim (run ▸ paused)
 *   if you want a frozen comparison.
 * - With the panel open, tweakpane quantises the live config to its slider steps
 *   on every refresh, so a captured preset can differ from the blend in the third
 *   decimal. Visually irrelevant; it does not accumulate.
 */
import type { Pane } from 'tweakpane';
import type { PhysarumSim } from '../sim/physarum/physarum';
import type { Mapper, MappingMode } from '../mapping/mapper';
import type { MappingConfig } from '../mapping/types';
import type { TuningLog } from '../mapping/tuninglog';
import {
  clearMappingLocal,
  downloadText,
  parseMapping,
  pickTextFile,
  saveMappingLocal,
  serializeMapping,
  mappingFits,
} from '../mapping/persist';

type Folder = ReturnType<Pane['addFolder']>;

export interface WorkbenchHost {
  sim: PhysarumSim;
  mapper: Mapper;
  log: TuningLog;
  trackId: string;
  /** transport position in seconds */
  time(): number;
  /** sim tick the transport is at */
  tick(): number;
  seek(seconds: number): void;
  /** re-run k-means against the loaded timeline, keeping tuned presets */
  refit(k: number): MappingConfig;
  /** discard all tuning and rebuild starter presets from the templates */
  resetToFitted(): MappingConfig;
  /** the panel's own species/matrix widgets need rebinding when K-shaped state changes */
  onConfigReplaced?: () => void;
}

export interface WorkbenchHandle {
  refresh(): void;
  dispose(): void;
}

interface UiState {
  mode: MappingMode;
  status: string;
  temperature: number;
  slewFast: number;
  slewMedium: number;
  slewSlow: number;
  boundaryOn: boolean;
  snapFraction: number;
  respawnFraction: number;
  anchorCount: number;
  autosave: boolean;
  file: string;
  snapshotInfo: string;
  logInfo: string;
}

export function createWorkbench(pane: Pane, host: WorkbenchHost): WorkbenchHandle {
  const { mapper, sim, log } = host;
  const ui: UiState = {
    mode: mapper.mode,
    status: '—',
    temperature: mapper.config.temperature,
    slewFast: mapper.config.slew.fast,
    slewMedium: mapper.config.slew.medium,
    slewSlow: mapper.config.slew.slow,
    boundaryOn: mapper.config.boundary.enabled,
    snapFraction: mapper.config.boundary.snapFraction,
    respawnFraction: mapper.config.boundary.respawnFraction,
    anchorCount: mapper.config.kmeans.k,
    autosave: true,
    file: '—',
    snapshotInfo: 'none',
    logInfo: '0 entries',
  };

  const autosave = (): void => {
    if (!ui.autosave) return;
    ui.file = saveMappingLocal(mapper.config) ? 'autosaved' : 'autosave failed';
  };

  const pullFromConfig = (): void => {
    const c = mapper.config;
    ui.temperature = c.temperature;
    ui.slewFast = c.slew.fast;
    ui.slewMedium = c.slew.medium;
    ui.slewSlow = c.slew.slow;
    ui.boundaryOn = c.boundary.enabled;
    ui.snapFraction = c.boundary.snapFraction;
    ui.respawnFraction = c.boundary.respawnFraction;
    ui.anchorCount = c.kmeans.k;
    ui.mode = mapper.mode;
  };

  // ── mapping ────────────────────────────────────────────────────────────────
  const root = pane.addFolder({ title: 'mapping (phase 5)', expanded: true });

  root
    .addBinding(ui, 'mode', {
      label: 'mode',
      options: { manual: 'manual', mapped: 'mapped' } as Record<string, MappingMode>,
    })
    .on('change', (ev) => {
      const mode = ev.value;
      // tweakpane re-emits 'change' from refresh(), and refresh() runs every 30
      // frames to mirror the live mode. Without this guard the mirror would look
      // like a user action and clear solo half a second after every solo click.
      if (mode === mapper.mode) return;
      if (mode === 'mapped' && !mapper.available) {
        ui.mode = 'manual';
        ui.status = `unavailable — ${mapper.unavailableReason}`;
        pane.refresh();
        return;
      }
      if (mode === 'mapped') {
        // Stems keep their own direct path into deposit; the simplex only does
        // character, so mapped mode is not "alive" without it.
        sim.config.stemDrive = true;
      }
      mapper.setSolo(null);
      mapper.setMode(mode);
      host.onConfigReplaced?.();
    });
  root.addBinding(ui, 'status', { readonly: true, label: '' });

  root
    .addBinding(ui, 'temperature', { min: 0.01, max: 3, step: 0.01, label: 'temperature ×σ²' })
    .on('change', (ev) => {
      mapper.config.temperature = ev.value;
      autosave();
    });

  const slew = root.addFolder({ title: 'slew (seconds, per class)', expanded: false });
  const bindSlew = (
    key: 'slewFast' | 'slewMedium' | 'slewSlow',
    field: 'fast' | 'medium' | 'slow',
    label: string,
    max: number,
  ): void => {
    slew
      .addBinding(ui, key, { min: 0, max, step: 0.01, label })
      .on('change', (ev) => {
        mapper.config.slew[field] = ev.value;
        mapper.syncRates();
        autosave();
      });
  };
  bindSlew('slewFast', 'fast', 'fast τ (deposit/angle/brightness)', 2);
  bindSlew('slewMedium', 'medium', 'medium τ (geometry)', 10);
  bindSlew('slewSlow', 'slow', 'slow τ (decay/alive/M)', 60);

  const boundary = root.addFolder({ title: 'section boundaries', expanded: false });
  boundary.addBinding(ui, 'boundaryOn', { label: 'steps + re-seed' }).on('change', (ev) => {
    mapper.config.boundary.enabled = ev.value;
    autosave();
  });
  boundary
    .addBinding(ui, 'snapFraction', { min: 0, max: 1, step: 0.01, label: 'slow snap' })
    .on('change', (ev) => {
      mapper.config.boundary.snapFraction = ev.value;
      autosave();
    });
  boundary
    .addBinding(ui, 'respawnFraction', { min: 0, max: 1, step: 0.01, label: 'respawn frac' })
    .on('change', (ev) => {
      mapper.config.boundary.respawnFraction = ev.value;
      autosave();
    });
  // Fire a boundary event by hand, to see what a section change does without
  // waiting for one. The key walks so repeated presses are not identical.
  let manualReseedKey = 1000;
  boundary.addButton({ title: 'test: re-seed now' }).on('click', () => {
    sim.partialReseed(manualReseedKey++, mapper.config.boundary.respawnFraction);
  });

  // ── anchors ────────────────────────────────────────────────────────────────
  const anchorRoot = pane.addFolder({ title: 'anchors', expanded: true });
  anchorRoot
    .addBinding(ui, 'anchorCount', { min: 2, max: 16, step: 1, label: 'k (needs refit)' })
    .on('change', (ev) => {
      mapper.config.kmeans.k = Math.round(ev.value);
    });
  anchorRoot.addButton({ title: 'refit anchors (keeps presets)' }).on('click', () => {
    host.refit(Math.round(ui.anchorCount));
    rebuildAnchors();
    pullFromConfig();
    autosave();
    host.onConfigReplaced?.();
    pane.refresh();
  });
  anchorRoot.addButton({ title: 'clear solo' }).on('click', () => {
    mapper.setSolo(null);
    pane.refresh();
  });

  interface AnchorRow {
    folder: Folder;
    state: { name: string; readout: string };
  }
  let rows: AnchorRow[] = [];

  function rebuildAnchors(): void {
    for (const r of rows) r.folder.dispose();
    rows = [];
    mapper.config.anchors.forEach((anchor, index) => {
      const state = { name: anchor.name, readout: '—' };
      const f = anchorRoot.addFolder({ title: `${index} · ${anchor.name}`, expanded: false });
      f.addBinding(state, 'readout', { readonly: true, label: 'w / d²' });
      f.addBinding(state, 'name', { label: 'name' }).on('change', (ev) => {
        anchor.name = ev.value;
        f.title = `${index} · ${ev.value}`;
        autosave();
      });
      f.addButton({ title: 'solo (tune this preset)' }).on('click', () => {
        mapper.setSolo(index);
        ui.mode = 'mapped';
        host.onConfigReplaced?.();
        pane.refresh();
      });
      f.addButton({ title: 'apply (blend continues)' }).on('click', () => {
        mapper.applyAnchor(index);
        host.onConfigReplaced?.();
        pane.refresh();
      });
      f.addButton({ title: 'capture current params →' }).on('click', () => {
        mapper.captureInto(index);
        log.append({
          tick: host.tick(),
          time: host.time(),
          action: 'capture',
          anchor: index,
          anchorName: anchor.name,
          z: mapper.currentZ(),
          theta: mapper.currentTheta(),
        });
        autosave();
        refreshLog();
      });
      rows.push({ folder: f, state });
    });
  }
  rebuildAnchors();

  // ── mapping file ───────────────────────────────────────────────────────────
  const file = pane.addFolder({ title: 'mapping file (this IS the art direction)', expanded: false });
  file.addBinding(ui, 'file', { readonly: true, label: '' });
  file.addBinding(ui, 'autosave', { label: 'autosave (localStorage)' });
  file.addButton({ title: 'download mapping.json' }).on('click', () => {
    downloadText(`mapping-${host.trackId}.json`, serializeMapping(mapper.config));
    log.append({
      tick: host.tick(),
      time: host.time(),
      action: 'save',
      anchor: mapper.solo,
      anchorName: null,
      z: mapper.currentZ(),
      theta: mapper.currentTheta(),
    });
    ui.file = 'downloaded';
    refreshLog();
  });
  file.addButton({ title: 'load mapping.json…' }).on('click', () => {
    void pickTextFile().then((text) => {
      if (text === null) return;
      try {
        const parsed = parseMapping(text);
        const latentDims = host.mapper.config.latentDims;
        if (!mappingFits(parsed, sim.config.speciesCount, latentDims)) {
          ui.file = `rejected: authored for K=${parsed.speciesCount}/${parsed.latentDims}d`;
          pane.refresh();
          return;
        }
        mapper.setConfig(parsed);
        rebuildAnchors();
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
    ui.file = saveMappingLocal(mapper.config) ? 'saved' : 'save failed';
    pane.refresh();
  });
  file.addButton({ title: 'reset to fitted defaults' }).on('click', () => {
    clearMappingLocal();
    host.resetToFitted();
    rebuildAnchors();
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
    log.append({
      tick: host.tick(),
      time: snapshotTime,
      action: 'snapshot',
      anchor: mapper.solo,
      anchorName: null,
      z: mapper.currentZ(),
      theta: mapper.currentTheta(),
    });
    refreshLog();
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
    log.append({
      tick: host.tick(),
      time: snapshotTime,
      action: 'restore',
      anchor: mapper.solo,
      anchorName: null,
      z: mapper.currentZ(),
      theta: mapper.currentTheta(),
    });
    refreshLog();
    pane.refresh();
  });
  ab.addButton({ title: 'drop snapshot (free GPU memory)' }).on('click', () => {
    sim.clearSnapshot();
    ui.snapshotInfo = 'none';
    pane.refresh();
  });

  // ── tuning log ─────────────────────────────────────────────────────────────
  const logFolder = pane.addFolder({ title: 'tuning log', expanded: false });
  logFolder.addBinding(ui, 'logInfo', { readonly: true, label: '' });
  const logVerdict = (action: 'keep' | 'discard'): void => {
    log.append({
      tick: host.tick(),
      time: host.time(),
      action,
      anchor: mapper.solo,
      anchorName: mapper.solo === null ? null : (mapper.config.anchors[mapper.solo]?.name ?? null),
      z: mapper.currentZ(),
      theta: mapper.currentTheta(),
    });
    refreshLog();
    pane.refresh();
  };
  logFolder.addButton({ title: 'log: keep this look' }).on('click', () => logVerdict('keep'));
  logFolder.addButton({ title: 'log: discard this look' }).on('click', () => logVerdict('discard'));
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

  return {
    refresh(): void {
      const w = mapper.weights;
      const d = mapper.distances;
      rows.forEach((row, i) => {
        const weight = w[i] ?? 0;
        row.state.readout = `${(weight * 100).toFixed(1)}%  ·  d²=${(d[i] ?? 0).toFixed(2)}`;
      });

      if (!mapper.available) {
        ui.status = `unavailable — ${mapper.unavailableReason}`;
      } else if (mapper.solo !== null) {
        const name = mapper.config.anchors[mapper.solo]?.name ?? '';
        ui.status = `solo ${mapper.solo} (${name}) · sliders live, capture to keep`;
      } else if (mapper.mode === 'mapped') {
        let top = 0;
        for (let i = 1; i < w.length; i++) if ((w[i] as number) > (w[top] as number)) top = i;
        ui.status = `mapped · nearest ${top} (${mapper.config.anchors[top]?.name ?? ''}) ${(
          (w[top] ?? 0) * 100
        ).toFixed(0)}%`;
      } else {
        ui.status = 'manual · sliders rule';
      }
      ui.mode = mapper.mode;
      if (sim.hasSnapshot && ui.snapshotInfo === 'none') ui.snapshotInfo = 'held';
    },
    dispose(): void {
      root.dispose();
      anchorRoot.dispose();
      file.dispose();
      ab.dispose();
      logFolder.dispose();
    },
  };
}
