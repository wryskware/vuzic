/**
 * The workbench — dev tooling rather than a viewer feature, and after plan.md
 * Revision 4 an instrument you can actually read.
 *
 * **The drivers section is the point of this round.** ~16 named channels, each
 * with a live meter and a gain slider: mute the ones that are not doing anything
 * you like, boost the ones that are. That is the whole tuning surface for the
 * projection layer — nobody tunes weights, and nobody should have to.
 *
 * **Brightness is no longer in it.** It has its own section (stem-follow), which
 * is not modulation at all: a species is as bright as its own instrument is
 * loud. That section works in manual mode and with every driver muted.
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
 * **It is no longer one column.** The panel is tabbed, and the workbench's
 * pieces belong to three different tasks: the mode switch and the two knobs you
 * turn mid-track go on *play*, the driver bank and everything that explains what
 * it is doing go on *map*, and the file/snapshot/log trio goes on *data*. It
 * stayed one function taking four containers rather than becoming four exported
 * builders because the pieces share one closure — `autosave()`, `record()`,
 * `pullFromConfig()` and a single `refresh()` — and splitting it would have meant
 * four copies of that state or a fifth object to thread it through.
 *
 * UX decisions worth knowing:
 *
 * - **modulate vs manual.** Manual is absolute: the sliders rule and nothing
 *   writes over them (phase-4 behaviour). Modulated hands the registry's slots to
 *   the projection engine — the sliders then mirror the live values, and an edit
 *   lasts until the next tick. Slots outside the registry (the p3 exponents,
 *   exposure, gamma, stemGain) stay editable in *both* modes.
 * - **one seed UI.** Seed, pin and reroll live in one folder and nowhere else.
 *   The seed drives wiring, personality and the world at once, so splitting it
 *   across two panels was two half-truths.
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
import type { PanelContainer } from './panel';
import type { ModTarget } from '../mapping/target';
import { MAX_DRIVER_GAIN, type Modulator } from '../mapping/modulation';
import { MOD_GROUPS, type ModGroup } from '../mapping/modspec';
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
  /**
   * Typed at the seam, not at physarum: everything this folder touches — the
   * seed, the snapshot pair, the palette invalidation, the boundary reseed — is
   * on `ModTarget`, so the workbench is already whatever-sim tooling.
   */
  sim: ModTarget;
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
  /** drop ringing impulse envelopes; the transport keeps its position */
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
  followOn: boolean;
  followFloor: number;
  followCurve: number;
  followSmoothing: number;
  followStatus: string;
  autosave: boolean;
  file: string;
  snapshotInfo: string;
  logInfo: string;
}

const GROUP_LABELS: Record<ModGroup, string> = {
  structure: 'structure (sensors, motion)',
  matrix: 'matrix M (who follows whom)',
  population: 'population (deposit, alive)',
  decay: 'decay (trail memory)',
};

/**
 * A z value as a signed bar. The meters are what make the bank interpretable —
 * "chorus-ness is pinned high and pc-1 just swung negative" is a sentence you can
 * only say if you can see it — so they are drawn wide enough to read at a glance
 * rather than printed as four decimals nobody parses in motion.
 */
const METER_HALF = 5;

function meterBar(z: number, muted: boolean): string {
  const v = Number.isFinite(z) ? Math.min(Math.max(z / 3, -1), 1) : 0;
  const n = Math.round(Math.abs(v) * METER_HALF);
  const left = v < 0 ? '█'.repeat(n).padStart(METER_HALF, ' ') : ' '.repeat(METER_HALF);
  const right = v > 0 ? '█'.repeat(n).padEnd(METER_HALF, ' ') : ' '.repeat(METER_HALF);
  // Deliberately narrow: tweakpane's readonly field is ~16 monospace characters
  // wide and clips silently, so a wider meter loses the number off the end.
  const tail = muted ? 'MUTE' : `${z < 0 ? '-' : '+'}${Math.abs(z).toFixed(1)}`;
  return `${left}│${right}${tail}`;
}

/**
 * The structure drivers' names are written for the meter labels, where there is
 * a whole row per driver. Three of them have to share one wiring line, so they
 * get short forms; the pc-N names are already short enough to leave alone.
 */
const DRIVER_ABBREV: Readonly<Record<string, string>> = {
  'novelty·4bar': 'nov4',
  'novelty·16bar': 'nov16',
  'chorus-ness': 'chorus',
};

function shortDriver(name: string): string {
  const bare = name.replace(' (absent)', '');
  return DRIVER_ABBREV[bare] ?? bare;
}

/**
 * Where each piece of the workbench mounts.
 *
 * `seed` is the one container the workbench does *not* create: it has to sit
 * above the macros folder on the play tab, tweakpane orders by mount order, and
 * the macros folder belongs to the panel. So the panel creates the folder and
 * the workbench fills it. Its lifetime is therefore the panel's — `dispose()`
 * below leaves it alone, and the pane's own dispose takes it down.
 */
export interface WorkbenchMounts {
  /** the pane itself: buttons that write UI state have to force the redraw themselves */
  pane: Pane;
  /** play tab — the world-seed folder, created by the panel, filled in here */
  seed: PanelContainer;
  /** play tab — the modulation headline: mode, status, global depth, response speed */
  play: PanelContainer;
  /** map tab — drivers, per-group depth, wiring, excursion, slew detail, boundaries, stem-follow */
  map: PanelContainer;
  /** data tab — modulation file, A/B snapshot, tuning log */
  data: PanelContainer;
}

export function createWorkbench(mounts: WorkbenchMounts, host: WorkbenchHost): WorkbenchHandle {
  const { pane } = mounts;
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
    followOn: cfg.stemFollow.enabled,
    followFloor: cfg.stemFollow.floor,
    followCurve: cfg.stemFollow.curve,
    followSmoothing: cfg.stemFollow.smoothingMs,
    followStatus: '—',
    autosave: true,
    file: '—',
    snapshotInfo: 'none',
    logInfo: '0 entries',
  };
  const groupUi: Record<string, number> = {};
  const excursionUi: Record<string, string> = {};
  /** driver index → meter text / gain, keyed by index so duplicate names cannot collide */
  const meterUi: Record<string, string> = {};
  const gainUi: Record<string, number> = {};
  const followUi: Record<string, string> = {};
  /** group → "pc-1 34% · chorus 22% · pc-4 11%" */
  const wiringUi: Record<string, string> = {};

  /**
   * The sim's opaque extras block is a *snapshot* in the config, not a shared
   * object (unlike the palette and the render block), so anything that writes
   * the file has to re-take it first — otherwise a macro edit made after the
   * last `setConfig` would be saved at its old value.
   */
  const refreshExtras = (): void => {
    const extras = sim.serializeExtras?.();
    if (extras) modulator.config.extras = extras;
  };

  const autosave = (): void => {
    if (!ui.autosave) return;
    refreshExtras();
    ui.file = saveModulationLocal(modulator.config, sim.simId) ? 'autosaved' : 'autosave failed';
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
    ui.followOn = c.stemFollow.enabled;
    ui.followFloor = c.stemFollow.floor;
    ui.followCurve = c.stemFollow.curve;
    ui.followSmoothing = c.stemFollow.smoothingMs;
    for (const g of MOD_GROUPS) groupUi[g] = c.groupDepth[g];
    for (let d = 0; d < modulator.driverCount; d++) gainUi[`g${d}`] = modulator.driverGain(d);
  };

  // ── play · modulation headline ─────────────────────────────────────────────
  // Four widgets and no more: is the music driving this at all, what is it doing
  // right now, how hard does it push, how fast does it move. Those are the ones
  // you touch with the track running. Per-group depth, the slew constants and the
  // driver bank itself are tuning, not performance, and live on the map tab.
  const root = mounts.play.addFolder({ title: 'modulation', expanded: true });

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
  // The two continuous knobs of the whole modulation layer: how hard the music
  // pushes and how fast it moves. Their per-group trims are on the map tab.
  root
    .addBinding(ui, 'depth', { min: 0, max: 4, step: 0.01, label: 'depth × (global)' })
    .on('change', (ev) => {
      modulator.config.depth = ev.value;
      autosave();
    });
  root
    .addBinding(ui, 'responseSpeed', { min: 0.05, max: 8, step: 0.05, label: 'speed ×' })
    .on('change', (ev) => {
      modulator.config.responseSpeed = ev.value;
      autosave();
    });

  // ── play · the world seed. The ONE place seeds are shown or changed. ────────
  const world = mounts.seed;
  world.addBinding(ui, 'seed', { readonly: true, label: 'seed' });
  world.addBinding(ui, 'pin', { label: 'pin (survives reload)' }).on('change', (ev) => {
    // localStorage and ?seed= are pinned together: unticking has to clear both or
    // a URL param would keep resurrecting the old world on reload.
    setPinnedSeed(ev.value ? sim.currentSeed : null);
    syncUrlSeed(ev.value ? sim.currentSeed : null);
    record(ev.value ? 'pin' : 'unpin');
  });
  // Full scatter, and that is its whole meaning: "re-run" is "start this world
  // again from the beginning", so it clears and re-scatters exactly as a fresh
  // load would. It is the only button in the app that reproduces a load.
  world.addButton({ title: '↻ re-run this world (same seed · rescatters)' }).on('click', () => {
    sim.reseed(sim.currentSeed);
    host.restart();
    host.onConfigReplaced?.();
    pane.refresh();
  });
  // Last in the folder on purpose: it is the one button here that replaces every
  // rule the world runs by, and there is no undo for it.
  world.addButton({ title: '🎲 reroll world (new seed · keeps the matter)' }).on('click', () => {
    // sim.reseed fires onSeedChange, which is what re-keys the modulator's
    // projections and personality and the impulse hotspots. One act, one seed.
    //
    // `keepWorld`, and this is the difference from the button above it. A reroll
    // used to rescatter, so the two buttons differed only in which seed they
    // restarted from and every reroll cost you the world you had accumulated —
    // which on physarum is minutes of trail network and on plife is whatever
    // arrangement you were auditioning against. Now the new seed's rules land on
    // the matter that is already there: the matrix is redrawn, the personality
    // and the wiring are restamped, and you watch the world you have reorganise
    // itself under them. "New physics, same matter" — the one below is the one
    // that means "start over".
    //
    // Not reproducible mid-run, deliberately, and the sims' own reseed comments
    // say so: a pinned seed reproduces a run *from load*, where the scatter does
    // happen. Rerolling in place is a performance gesture against whatever the
    // world is at that instant.
    const seed = randomSeed();
    sim.reseed(seed, { keepWorld: true });
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

  // ── map · the driver bank (Revision 4) ─────────────────────────────────────
  // Two widgets per driver: a meter you read and a gain you turn. The meter is
  // the reason this round exists — it is what turns "something changed" into
  // "chorus-ness went up", which is the only way to decide what to mute.
  //
  // The one folder on the map tab that opens expanded: it is the instrument, and
  // a tab whose instrument is behind a disclosure triangle is a tab you do not
  // use. Everything under it is collapsed to keep it reachable.
  const drivers = mounts.map.addFolder({
    title: `drivers · gains + meters (${modulator.driverCount} inputs · ±3σ)`,
    expanded: true,
  });
  for (let d = 0; d < modulator.driverCount; d++) {
    const key = `g${d}`;
    meterUi[key] = '—';
    gainUi[key] = modulator.driverGain(d);
    drivers.addBinding(meterUi, key, { readonly: true, label: modulator.driverName(d) });
    drivers
      .addBinding(gainUi, key, { min: 0, max: MAX_DRIVER_GAIN, step: 0.01, label: '↳ gain' })
      .on('change', (ev) => {
        modulator.setDriverGain(d, ev.value);
        autosave();
      });
  }
  if (modulator.driverCount > 0) {
    const setAll = (v: number): void => {
      modulator.setAllDriverGains(v);
      for (let d = 0; d < modulator.driverCount; d++) gainUi[`g${d}`] = modulator.driverGain(d);
      autosave();
      pane.refresh();
    };
    // "Mute all" is a diagnostic, not a preset: it is how you check that what you
    // are looking at is the projections and not the stem lane or the impulses.
    drivers.addButton({ title: 'mute all (silence the projections)' }).on('click', () => setAll(0));
    drivers.addButton({ title: 'reset gains to 1' }).on('click', () => setAll(1));
  }

  // ── map · depth, per group ─────────────────────────────────────────────────
  // The global multiplier is on the play tab; these four are the trim behind it —
  // "let the music move the colours but leave the geometry alone" is a sentence
  // you say once per track, not once per chorus.
  const depth = mounts.map.addFolder({ title: 'depth · per group', expanded: false });
  for (const g of MOD_GROUPS) {
    groupUi[g] = cfg.groupDepth[g];
    depth
      .addBinding(groupUi, g, { min: 0, max: 3, step: 0.01, label: GROUP_LABELS[g] })
      .on('change', (ev) => {
        modulator.config.groupDepth[g] = ev.value;
        autosave();
      });
  }

  // ── map · wiring readout ───────────────────────────────────────────────────
  // The other half of the drivers folder: the gains say how loud each input is,
  // this says where it lands. Without it a reroll rewires everything invisibly
  // and there is no way to answer "why did muting pc-3 do nothing to the matrix".
  // Recomputed in refresh(), so it follows both a reroll and a gain edit.
  const wiring = mounts.map.addFolder({
    title: 'wiring · what this seed listens to',
    expanded: false,
  });
  for (const g of MOD_GROUPS) {
    wiringUi[g] = '—';
    wiring.addBinding(wiringUi, g, { readonly: true, label: GROUP_LABELS[g] });
  }

  // ── map · live readout ─────────────────────────────────────────────────────
  const live = mounts.map.addFolder({ title: 'live excursion (mean |tanh|)', expanded: false });
  for (const g of MOD_GROUPS) {
    excursionUi[g] = '—';
    live.addBinding(excursionUi, g, { readonly: true, label: GROUP_LABELS[g] });
  }

  // ── map · response detail ──────────────────────────────────────────────────
  // `speed ×` scales all three of these at once and is on the play tab; what is
  // left here is the shape of the response rather than its rate.
  const response = mounts.map.addFolder({ title: 'response · slew detail', expanded: false });
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

  // ── map · section boundaries (an event, not a scene) ───────────────────────
  const boundary = mounts.map.addFolder({ title: 'section boundaries (optional)', expanded: false });
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

  // ── map · brightness · stem-follow (Revision 4, NOT modulation) ────────────
  const follow = mounts.map.addFolder({ title: 'brightness · stem-follow', expanded: false });
  follow.addBinding(ui, 'followStatus', { readonly: true, label: '' });
  follow.addBinding(ui, 'followOn', { label: 'follow stems (off = absolute)' }).on('change', (ev) => {
    modulator.config.stemFollow.enabled = ev.value;
    autosave();
  });
  follow
    .addBinding(ui, 'followFloor', { min: 0, max: 1, step: 0.01, label: 'floor (silent = this ×)' })
    .on('change', (ev) => {
      modulator.config.stemFollow.floor = ev.value;
      autosave();
    });
  follow
    .addBinding(ui, 'followCurve', { min: 0.2, max: 4, step: 0.01, label: 'curve (exponent)' })
    .on('change', (ev) => {
      modulator.config.stemFollow.curve = ev.value;
      autosave();
    });
  follow
    .addBinding(ui, 'followSmoothing', { min: 50, max: 2000, step: 10, label: 'smoothing (ms)' })
    .on('change', (ev) => {
      modulator.config.stemFollow.smoothingMs = ev.value;
      autosave();
    });
  // Per-species readout: activity, the multiplier it produces, and the effective
  // light. This is what tells you whether the floor is doing what you meant.
  for (let s = 0; s < sim.config.speciesCount; s++) {
    const key = `f${s}`;
    followUi[key] = '—';
    follow.addBinding(followUi, key, {
      readonly: true,
      label: `${s} ${sim.config.species[s]?.name ?? ''}`,
    });
  }

  // ── data · modulation file ─────────────────────────────────────────────────
  const file = mounts.data.addFolder({
    title: 'modulation file (depths + palette + grade)',
    expanded: true,
  });
  file.addBinding(ui, 'file', { readonly: true, label: '' });
  file.addBinding(ui, 'autosave', { label: 'autosave (localStorage)' });
  file.addButton({ title: 'download modulation.json' }).on('click', () => {
    refreshExtras();
    downloadText(`modulation-${host.trackId}.json`, serializeModulation(modulator.config));
    record('save');
    ui.file = 'downloaded';
  });
  file.addButton({ title: 'load modulation.json…' }).on('click', () => {
    void pickTextFile().then((text) => {
      if (text === null) return;
      try {
        const parsed = parseModulation(text);
        if (!modulationFits(parsed, sim.config.speciesCount, sim.simId)) {
          ui.file = `rejected: authored for ${parsed.sim} K=${parsed.speciesCount}`;
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
    refreshExtras();
    ui.file = saveModulationLocal(modulator.config, sim.simId) ? 'saved' : 'save failed';
    pane.refresh();
  });
  file.addButton({ title: 'reset to defaults' }).on('click', () => {
    clearModulationLocal(sim.simId);
    modulator.setConfig(defaultModulationConfig(sim.config, sim.simId));
    modulator.setMode(modulator.available ? 'modulated' : 'manual');
    pullFromConfig();
    ui.file = 'reset';
    host.onConfigReplaced?.();
    pane.refresh();
  });

  // ── data · A/B ─────────────────────────────────────────────────────────────
  const ab = mounts.data.addFolder({ title: 'A/B  ·  sim state snapshot', expanded: false });
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

  // ── data · tuning log ──────────────────────────────────────────────────────
  const logFolder = mounts.data.addFolder({ title: 'tuning log (seed preferences)', expanded: false });
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
      // Meters first — they are live in *both* modes, because the bank is an
      // instrument panel before it is an input, and reading the music with
      // modulation off is a legitimate thing to want.
      for (let d = 0; d < modulator.driverCount; d++) {
        const key = `g${d}`;
        const gain = modulator.driverGain(d);
        meterUi[key] = meterBar(modulator.driverValue(d), gain === 0);
        gainUi[key] = gain;
      }

      const sf = modulator.stemFollow;
      const on = modulator.config.stemFollow.enabled;
      ui.followStatus = on
        ? 'ON — species brightness sliders are the BASE, scaled by their stem'
        : 'OFF — species brightness sliders are absolute';
      ui.followOn = on;
      for (let s = 0; s < sf.speciesCount; s++) {
        const key = `f${s}`;
        if (sf.stemOf(s) < 0) {
          followUi[key] = 'no stem — brightness absolute';
          continue;
        }
        const mul = sf.multiplier[s] ?? 1;
        const base = sim.config.species[s]?.brightness ?? 1;
        followUi[key] =
          `act ${(sf.activity[s] ?? 0).toFixed(2)} · ×${mul.toFixed(2)} → ${(base * mul).toFixed(2)}`;
      }

      // Wiring: cheap enough at 30-frame cadence, and it has to be recomputed
      // rather than cached because a gain edit changes it without any event.
      for (const row of modulator.groupDriverWeights()) {
        wiringUi[row.group] =
          row.top.length === 0
            ? '— (nothing wired)'
            : row.top
                .map((t) => `${shortDriver(t.name)} ${(t.share * 100).toFixed(0)}%`)
                .join(' · ');
      }

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
      // Every folder this function created, across all three tabs. `mounts.seed`
      // is deliberately absent: the panel created it and the pane's dispose owns
      // it (see `WorkbenchMounts`).
      root.dispose();
      drivers.dispose();
      depth.dispose();
      wiring.dispose();
      live.dispose();
      response.dispose();
      boundary.dispose();
      follow.dispose();
      file.dispose();
      ab.dispose();
      logFolder.dispose();
    },
  };
}
