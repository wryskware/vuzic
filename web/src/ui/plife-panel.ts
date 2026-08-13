/**
 * The live control panel for particle life.
 *
 * Structurally a sibling of `panel.ts` rather than a generalisation of it: the
 * two sims share the tab scaffolding, the workbench, the impulse folder and the
 * HDR chain (all four are imported here, not re-implemented), and differ in
 * exactly the part that *should* differ — which knobs exist, what they mean, and
 * which of the six tabs they belong on. Trying to drive both from one
 * table-driven panel would mean describing every slider as data, which is more
 * machinery than two hand-written panels and reads worse.
 *
 * The conventions are `panel.ts`'s, deliberately:
 *
 * - Everything binds straight to the live config object. The sim re-reads it
 *   every substep, so there are no change handlers except where GPU state has to
 *   be rebuilt — here that is exactly two places, the attraction matrix (which
 *   lives in its own buffer) and the palette (which is parsed and cached).
 * - Proxy objects for the matrix and the palette, resynced in `refresh()`, since
 *   a reroll or a file load rewrites the live config wholesale.
 * - Seed controls appear in `run` **only when there is no workbench**. With one,
 *   the seed is the world seed and it gets its own folder at the top of the play
 *   tab, in one place rather than two.
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
import { Pane, type FolderApi } from 'tweakpane';
import { saveModulationLocal } from '../mapping/persist';
import type { ImpulseEngine } from '../sim/impulses';
import {
  BUDGET_FPS_RANGE,
  BUDGET_MIN,
  BUDGET_STEP,
  defaultPlifeMacros,
  FAR_GAIN_RANGE,
  FAR_SCALE_RANGE,
  MACRO_LABELS,
  MACRO_RANGE,
  MAX_MIN_R,
  MAX_NEAR_STENCIL,
  MAX_REACH_BRUTE,
  PAIR_SEARCH_MODES,
  R_CAP,
  type PairSearch,
  type PlifeSpeciesConfig,
} from '../sim/plife/config';
import type { PlifeSim } from '../sim/plife/plife';
import { coupled } from '../sim/plife/preset';
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
import { createPanelTabs, type PanelHandle, type PanelWorkbench } from './panel';
import { addRenderFolder } from './render-folder';
import { createExportFolder, type ExportPanelHost } from './export-panel';
import { createSimFolder, type SimPanelHost } from './sim-panel';
import { createTrackFolder, type TrackPanelHost } from './track-panel';
import { createWorkbench, type WorkbenchHandle } from './workbench';

type Folder = FolderApi;

interface RunState {
  seed: string;
  pin: boolean;
  particles: string;
  grid: string;
}

/** The budget folder's two readouts. Both derived; neither is ever written back. */
interface BudgetState {
  effective: string;
  alive: string;
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

  const pane = new Pane({ title: 'terrarium · particle life' });
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
    particles: '—',
    grid: '—',
  };
  const budgetState: BudgetState = { effective: '—', alive: '—' };

  // ── explore ────────────────────────────────────────────────────────────────
  // Its own tab, the same placement physarum's panel uses and for the same
  // reason: while the grid is up, every other tab is describing something that
  // is not on screen.
  const explorer: ExplorePanelHandle | null =
    opts.explorer && tabs.explore ? createExplorePanel(tabs.explore, opts.explorer) : null;

  // ── play ───────────────────────────────────────────────────────────────────
  // The track picker is the first thing on the tab: it is the one control here
  // that belongs to neither the sim nor the mapping, and picking a song comes
  // before shaping what it drives.
  if (opts.tracks) createTrackFolder(tabs.play, opts.tracks);
  if (opts.exports) createExportFolder(tabs.play, opts.exports);
  // And directly under it, the other "what am I looking at" decision.
  if (opts.sims) createSimFolder(tabs.play, opts.sims);

  // Created before the macros folder because tweakpane orders by mount order and
  // the seed belongs above them; the workbench fills it in below.
  const seedFolder = opts.workbench
    ? tabs.play.addFolder({ title: 'world seed (wiring + personality + sim)' })
    : null;

  // Seven multipliers that compose outside θ, exactly where stem-follow and the
  // impulse lanes do, which is the whole reason the modulator can never write
  // over them. Everything on the sim tab is either θ (mirrored, overwritten on
  // the next tick while modulating) or generation settings; this is the only
  // block that is unconditionally yours.
  //
  // Persistence: the macros, the matrix-generation settings and the population
  // lane all ride the mapping file's opaque `extras` block, so an edit to any of
  // them has to reach the same autosave the workbench's own bindings use. The
  // panel does it directly rather than through a new callback — it already holds
  // both halves (the sim, for the snapshot, and the modulator's config, for the
  // file) — and debounces, because these are sliders and a drag would otherwise
  // write localStorage on every pointer move.
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
  for (const { key, label } of MACRO_LABELS) {
    const r = MACRO_RANGE[key];
    macros
      .addBinding(config.macros, key, { min: r.min, max: r.max, step: 0.01, label })
      .on('change', persistExtras);
  }
  macros.addButton({ title: 'reset macros to 1' }).on('click', () => {
    Object.assign(config.macros, defaultPlifeMacros());
    persistExtras();
    pane.refresh();
  });

  // Mounts into three tabs at once: the modulation headline lands here under the
  // macros, the driver bank and its explanations land on map, and the
  // file/snapshot/log trio lands on data.
  let workbench: WorkbenchHandle | null = null;
  if (opts.workbench && seedFolder && tabs.data) {
    workbench = createWorkbench(
      { pane, seed: seedFolder, play: tabs.play, map: tabs.map, data: tabs.data },
      {
        ...opts.workbench,
        pinned: opts.pinned,
        restart: () => opts.onRestart?.(),
        // A reroll or a file load rewrites the live config wholesale; the matrix
        // and palette widgets are bound to proxies and must be re-read.
        onConfigReplaced: () => {
          syncMatrixProxy();
          pane.refresh();
        },
      },
    );
  }

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
  // "alive" is the summed population *target*, not a GPU read-back — see the note
  // on `PlifeSim.aliveCount`. Whatever is still fading out is not counted.
  run.addBinding(state, 'particles', { readonly: true, label: 'particles alive' });
  run.addBinding(state, 'grid', { readonly: true, label: 'grid' });

  // ── map ────────────────────────────────────────────────────────────────────
  // Below the workbench's own folders, which mounted above: the population lane
  // and the impulse lane are the two ways the music reaches this sim *without*
  // going through the driver bank.
  //
  // The population lane is outside θ entirely (like physarum's soil block), which
  // is why every widget here binds directly with no mask or mode caveat: the
  // modulator never writes any of it, in either mode.
  //
  // Every widget below persists through `persistExtras`, for the same reason the
  // macros do: the lane now rides the mapping file's `extras` block, and an edit
  // that only lived in memory would be lost on the next reload while the macro
  // next to it survived.
  const pop = tabs.map.addFolder({ title: 'population · stems → colonies', expanded: false });
  pop
    .addBinding(config.population, 'followStems', { label: 'follow stems (off = θ absolute)' })
    .on('change', persistExtras);
  pop
    .addBinding(config.population, 'floor', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'floor (silent = this ×)',
    })
    .on('change', persistExtras);
  pop
    .addBinding(config.population, 'curve', {
      min: 0.2,
      max: 4,
      step: 0.01,
      label: 'curve (exponent)',
    })
    .on('change', persistExtras);
  // Deliberately allowed much slower than stem-follow's brightness smoothing: a
  // colony that sheds a third of its members every bar reads as flicker.
  pop
    .addBinding(config.population, 'smoothingMs', {
      min: 100,
      max: 5000,
      step: 50,
      label: 'smoothing (ms)',
    })
    .on('change', persistExtras);
  pop
    .addBinding(config.population, 'riseTau', {
      min: 0.05,
      max: 3,
      step: 0.05,
      label: 'rise τ (s)',
    })
    .on('change', persistExtras);
  pop
    .addBinding(config.population, 'fallTau', {
      min: 0.1,
      max: 8,
      step: 0.1,
      label: 'fall τ (s)',
    })
    .on('change', persistExtras);

  const accent = pop.addFolder({ title: 'accents · novelty', expanded: false });
  accent
    .addBinding(config.population.accent, 'enabled', { label: 'novelty → accents' })
    .on('change', persistExtras);
  accent
    .addBinding(config.population.accent, 'floor', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'floor (plain section)',
    })
    .on('change', persistExtras);
  accent
    .addBinding(config.population.accent, 'boost', {
      min: 0,
      max: 3,
      step: 0.01,
      label: 'boost (headroom ×)',
    })
    .on('change', persistExtras);
  accent
    .addBinding(config.population.accent, 'smoothingMs', {
      min: 100,
      max: 5000,
      step: 50,
      label: 'smoothing (ms)',
    })
    .on('change', persistExtras);

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

  let impulsePanel: ImpulsePanelHandle | null = null;
  if (opts.impulses) {
    impulsePanel = createImpulsePanel(tabs.map, opts.impulses, (i) => config.species[i]?.name ?? `${i}`);
  }

  // ── sim ────────────────────────────────────────────────────────────────────
  // Both are θ and neither is modulated any more (user call, 2026-08-08 — see
  // plife/preset.ts): these two are absolute in both modes, which is why they
  // get no modulation band. `bands.add` is still called on them, so the day one
  // of them goes back into the registry the band appears without anyone
  // remembering this line.
  const physics = tabs.sim.addFolder({ title: 'physics  (absolute — never modulated)' });
  const forceGain = { min: 0.2, max: 4, step: 0.01, label: 'force gain ×' };
  bands?.add(
    physics.addBinding(config, 'forceGain', forceGain),
    slotOf('forceGain'),
    forceGain.min,
    forceGain.max,
  );
  const maxSpeed = { min: 0.05, max: 1, step: 0.005, label: 'max speed (world/s)' };
  bands?.add(
    physics.addBinding(config, 'maxSpeed', maxSpeed),
    slotOf('maxSpeed'),
    maxSpeed.min,
    maxSpeed.max,
  );

  // How far the near lane reaches, in grid cells. Structural, outside θ, and the
  // one control on this tab that changes what the force pass *searches* rather
  // than what it finds — which is why it states its own reach in the label
  // instead of leaving "2" to be decoded. Persisted through `persistExtras` like
  // every other extras-block knob.
  //
  // Costed rather than free: the search window is (2s+1)² cells, so 3 is 5.4× the
  // pair work of 1. The label says cells and world units; the frame counter says
  // the rest.
  const reach = tabs.sim.addFolder({ title: 'reach  (structural — never modulated)' });
  // The pair-search mode, first in the folder because it decides what every knob
  // under it means: in grid mode the stencil is the reach cap, in brute mode the
  // stencil is ignored entirely and the cap is half the torus.
  //
  // The label states the cost rather than hiding it. Brute is O(N²): at ~33 k
  // alive that is ~1.1 × 10⁹ pair tests per substep against grid mode's ~10⁷, so
  // it is a deliberate trade of frame time for the only thing that buys radii
  // past 0.06 — and radii past 0.06 are the only way this sim makes structures
  // bigger than a filigree strand.
  //
  // Starting points for large radii, measured rather than guessed (see the
  // stability sweep in the commit that added this): at r-max ≈ 0.15 the shipped
  // force settings hold; at 0.25–0.3 the summed tent over a few thousand
  // neighbours pins everything at maxSpeed, and the working settings are the
  // `force` macro around 0.3 with `agility` at or below 1. Reach for the macros
  // before the θ sliders — they survive rerolls and modulation.
  // The brute label also carries the one caveat that is not visible from here:
  // the explorer's nine tiles stay on the grid search whatever this says (see
  // `PlifeSim.forceGridSearch`), so at large radii the 9-up under-represents
  // what the full-size sim does with the candidate you pick.
  const pairSearchLabel = (): string =>
    config.field.pairSearch === 'brute'
      ? `pair search  (N², cap ${sim.nearReach.toFixed(2)}, live ${sim.liveReach.toFixed(3)}; 9-up stays grid)`
      : 'pair search  (brute = N², large radii)';
  const modeBinding = reach.addBinding(config.field, 'pairSearch', {
    options: Object.fromEntries(PAIR_SEARCH_MODES.map((m) => [m, m])) as Record<string, PairSearch>,
    label: pairSearchLabel(),
  });
  const stencilLabel = (): string =>
    config.field.pairSearch === 'brute'
      ? 'near stencil  (grid only — ignored in brute)'
      : `near stencil  (reach = s × ${R_CAP} = ${sim.nearReach.toFixed(3)})`;
  const stencilBinding = reach.addBinding(config.field, 'nearStencil', {
    min: 1,
    max: MAX_NEAR_STENCIL,
    step: 1,
    label: stencilLabel(),
  });
  // The far lane. Two knobs, both outside θ, both riding the extras block — and
  // deliberately in the same folder as the stencil, because the stencil is σ1:
  // it is the seam the two lanes meet at, and moving it moves the far lane's
  // inner scale with it. That is one knob doing one job, not a coupling to
  // remember.
  //
  // seam: neither of these is modulated, and a θ/modulation lane for them is
  // explicitly future work (see `PlifeFieldConfig`). If it lands, they move into
  // preset.ts's slot table and this folder keeps only the stencil.
  const farGain = { min: FAR_GAIN_RANGE.min, max: FAR_GAIN_RANGE.max, step: 0.01 };
  const farScaleLabel = (): string =>
    `far scale σ2  (effective ${sim.farSigma.toFixed(3)})`;
  const scaleBinding = reach.addBinding(config.field, 'farScale', {
    min: FAR_SCALE_RANGE.min,
    max: FAR_SCALE_RANGE.max,
    step: 0.005,
    label: farScaleLabel(),
  });
  // Mode changes live: `runStep` picks the pipeline per substep, so there is no
  // rebuild and no state to migrate — the particles keep their positions and
  // velocities and only the force pass they are stepped by changes. All three
  // labels are derived from the mode, so all three are rebuilt here.
  modeBinding.on('change', () => {
    modeBinding.label = pairSearchLabel();
    stencilBinding.label = stencilLabel();
    scaleBinding.label = farScaleLabel();
    persistExtras();
  });
  stencilBinding.on('change', () => {
    // Both labels carry a derived number, and both only move when a slider does
    // — so they are rebuilt here rather than in `refresh()`, where they would
    // rebuild a string sixty times a second to say the same thing. The stencil
    // touches both, because it is σ1 and σ1 is σ2's floor.
    stencilBinding.label = stencilLabel();
    scaleBinding.label = farScaleLabel();
    persistExtras();
  });
  reach
    .addBinding(config.field, 'farGain', { ...farGain, label: 'far gain  (0 = lane off)' })
    .on('change', persistExtras);
  scaleBinding.on('change', () => {
    scaleBinding.label = farScaleLabel();
    persistExtras();
  });

  // ── budget · the particle ceiling and its governor ─────────────────────────
  //
  // Directly under `reach`, because the two are the same subject from opposite
  // ends: `reach` decides how expensive one particle is (a stencil, or an O(N²)
  // pass) and this decides how many of them there may be. Outside θ, never
  // modulated, persisted through `persistExtras` like every other extras knob —
  // except the governor's own state, which is session-only by design (see
  // `PlifeBudgetConfig`) and appears here only as a readout.
  //
  // The brute lane is what makes this folder worth having: cost there is ∝ N², so
  // halving the budget quarters the frame time and the governor's ×0.85 step is a
  // ~28% cut in work per adjustment rather than 15%.
  const budget = tabs.sim.addFolder({ title: 'budget  (structural — never modulated)' });
  // Ceiling on Σ(all species targets), not a per-species number: the clamp scales
  // every species by one factor, so the mix you authored is what you keep.
  budget
    .addBinding(config.budget, 'cap', {
      min: BUDGET_MIN,
      max: config.maxParticles,
      // All three commensurate on purpose — see BUDGET_MIN's note. A stepped
      // tweakpane slider snaps to a grid anchored at one end, so a step that does
      // not divide the span writes back a value neither end sits on.
      step: BUDGET_STEP,
      label: `particle cap  (Σ targets; pool ${config.maxParticles.toLocaleString()})`,
    })
    .on('change', persistExtras);
  budget
    .addBinding(config.budget, 'adaptive', { label: 'adaptive  (governor on)' })
    .on('change', persistExtras);
  // Below floor the governor sheds fast; at or near ideal it grows slowly; in
  // between it holds. Both are sliders rather than readouts because the right
  // pair is a per-machine, per-display judgement — 60/120 is the author's panel,
  // not a universal truth.
  budget
    .addBinding(config.budget, 'floorFps', {
      min: BUDGET_FPS_RANGE.min,
      max: BUDGET_FPS_RANGE.max,
      step: 1,
      label: 'floor fps  (under this → shed)',
    })
    .on('change', persistExtras);
  budget
    .addBinding(config.budget, 'idealFps', {
      min: BUDGET_FPS_RANGE.min,
      max: BUDGET_FPS_RANGE.max,
      step: 1,
      label: 'ideal fps  (met → grow back)',
    })
    .on('change', persistExtras);
  // Live state, never persisted. `effective` is what the clamp actually used this
  // frame and `alive` is what came out of it, so the pair reads as "the ceiling,
  // and whether it is binding" — when they are equal the budget is what is
  // shaping the population, and when alive is well below it something else is.
  budget.addBinding(budgetState, 'effective', { readonly: true, label: 'effective budget' });
  budget.addBinding(budgetState, 'alive', { readonly: true, label: '↳ alive (target)' });

  const speciesRoot = tabs.sim.addFolder({
    title: opts.workbench ? 'species  (blue band = where the music can take it)' : 'species',
  });
  // Each returns a title sync: the folders are collapsed by default, so without
  // it "which species are off" is only visible after expanding all eight.
  const speciesTitles: (() => void)[] = [];
  for (let i = 0; i < k; i++) {
    const s = config.species[i];
    if (!s) continue;
    speciesTitles.push(addSpeciesFolder(speciesRoot, i, s, bands, slotOf, persistExtras));
  }

  // How a seed *draws* the matrix below, rather than what is in it right now.
  // Nothing here is θ and nothing here is modulated: these act once, at reroll
  // time, in `genmatrix.ts`. Editing one changes nothing on screen until a new
  // draw happens — which is the entire reason the redraw button exists.
  const gen = tabs.sim.addFolder({ title: 'matrix · seeded generation', expanded: false });
  gen.addBinding(config.matrixGen, 'sigma', {
    min: 0,
    max: 1.2,
    step: 0.01,
    label: 'sigma (draw scale)',
  }).on('change', persistExtras);
  gen.addBinding(config.matrixGen, 'symmetry', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'symmetry (1 = no chase)',
  }).on('change', persistExtras);
  gen.addBinding(config.matrixGen, 'selfBiasAccent', {
    min: -1,
    max: 1,
    step: 0.01,
    label: 'accent self bias (− disperse)',
  }).on('change', persistExtras);
  gen.addBinding(config.matrixGen, 'selfBias', {
    min: -1,
    max: 1,
    step: 0.01,
    label: 'self bias (− filigree)',
  }).on('change', persistExtras);
  gen.addBinding(config.matrixGen, 'accentGain', {
    min: 0,
    max: 2,
    step: 0.01,
    label: 'accent gain ×',
  }).on('change', persistExtras);
  // The hard-core band runs to MAX_MIN_R (0.05), not to the grid cell size. A
  // core is a *fraction of a reach*, and with brute mode's reaches an 0.02 core
  // inside an 0.3 tent is a pinhole — see MAX_MIN_R. Small values stay the norm;
  // the top of this slider is for large-radius worlds only.
  gen.addBinding(config.matrixGen.rMin, 'lo', { min: 0.002, max: MAX_MIN_R, step: 0.0005, label: 'r-min lo' }).on('change', persistExtras);
  gen.addBinding(config.matrixGen.rMin, 'hi', { min: 0.002, max: MAX_MIN_R, step: 0.0005, label: 'r-min hi' }).on('change', persistExtras);
  // The outer-radius band runs to the authored ceiling (half the torus), not to
  // any one mode's reach cap: this is the hand-tuning path to structures larger
  // than one filigree strand. Drawing above the *current* mode's cap is legal
  // and simply saturates — the shader truncates it — so in grid mode anything
  // over 0.06 does nothing and the mode dropdown above is what unlocks it.
  gen.addBinding(config.matrixGen.rMax, 'lo', { min: 0.005, max: MAX_REACH_BRUTE, step: 0.0005, label: 'r-max lo' }).on('change', persistExtras);
  gen.addBinding(config.matrixGen.rMax, 'hi', { min: 0.005, max: MAX_REACH_BRUTE, step: 0.0005, label: 'r-max hi' }).on('change', persistExtras);
  if (opts.workbench) {
    const modulator = opts.workbench.modulator;
    // A workbench *reroll* already draws a fresh matrix — a new seed is a new
    // world and the draw is part of what "new world" means. This button is the
    // other half of the tuning loop: re-draw under the settings above while
    // keeping the SAME seed, so the world, the wiring and the personality are
    // unchanged and the only thing that moved is the matrix. `setSeed` is the
    // whole path — it re-runs rewire (jitter, then the generator) and applyBase.
    //
    // Last in the folder: it is the one control here that replaces every number
    // in the matrix below, including hand edits.
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
  const matrixRoot = tabs.sim.addFolder({ title: 'attraction matrix A (drawn from the seed)', expanded: false });
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
      const cell = row
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
      // Only the coupled cells carry a ModSpec, so only they get a band — which
      // makes the primary/secondary partition visible on the sliders themselves
      // rather than only in the "(uncoupled)" suffix.
      bands?.add(cell, slotOf(`A[${i}][${j}]`), -1.2, 1.2);
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

  const refreshRender = addRenderFolder(tabs.look, {
    render: config.render,
    config,
    // The hard θ bound from plife/preset.ts's global table, not physarum's: a
    // particle splat is already a small alpha-weighted contribution, so this sim
    // lives around 1.0 rather than around 0.01.
    exposureRange: { min: 0.05, max: 4, step: 0.01 },
    renderPasses: () => sim.stats().renderPasses,
    autoExposureState: () => sim.post.autoExposureState,
    invalidatePalette: () => sim.invalidatePalette(),
    onChange: persistExtras,
    // No soil field in this sim, so the ember underlay has nothing to draw from.
  });

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

      // "governed" vs "cap" is the one thing the two numbers cannot say by
      // themselves: an effective budget equal to the cap means the governor has
      // not intervened, and one below it means the frame rate pulled it there.
      const governed = config.budget.adaptive && st.effectiveBudget < config.budget.cap;
      budgetState.effective =
        `${st.effectiveBudget.toLocaleString()}` +
        (governed ? ` (governed · ${sim.governorFps.toFixed(0)} fps)` : ' (cap)');
      budgetState.alive =
        `${st.aliveParticles.toLocaleString()}` +
        (st.aliveParticles >= st.effectiveBudget ? ' — budget binding' : '');

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

      for (const sync of speciesTitles) sync();

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
      workbench?.dispose();
      explorer?.dispose();
      pane.dispose();
    },
  };
}

/** @returns a sync for the folder title, which carries the on/off state. */
function addSpeciesFolder(
  root: Folder,
  index: number,
  s: PlifeSpeciesConfig,
  bands: ModBands | null,
  slotOf: (name: string) => number,
  persistExtras: () => void,
): () => void {
  const title = (): string =>
    `${index} · ${s.name}${s.role === 'secondary' ? ' (accent)' : ''}${s.enabled ? '' : '  — off'}`;
  const f = root.addFolder({ title: title(), expanded: index === 0 });
  // First control in the folder, above every θ slider, because it decides
  // whether the rest of them matter. Outside θ entirely — it rides the `extras`
  // block, hence `persistExtras` rather than the workbench's θ autosave — which
  // is what makes "four primaries and one accent" survive a chorus, a reroll,
  // an explorer pick and a reload. Switching it off drains the species over the
  // population lane's fall-τ instead of blanking it, so the frame stays honest.
  f.addBinding(s, 'enabled', { label: 'enabled (off = drains to 0)' }).on(
    'change',
    persistExtras,
  );
  /**
   * One species θ slider plus its band. The registry names a species slot
   * `species<i>.<field>` and the field *is* the config key, so the slot lookup
   * needs nothing plumbed through — which is what keeps this honest when a slot
   * is added or reordered: a wrong name resolves to -1 and draws no band, rather
   * than silently decorating a neighbouring slider.
   */
  const bind = <K extends NumericKey<PlifeSpeciesConfig> & string>(
    key: K,
    params: SliderParams,
  ): void => {
    const b = f.addBinding(s, key, params);
    bands?.add(b, slotOf(`species${index}.${String(key)}`), params.min, params.max);
  };
  // Honest label, both ways round, exactly as in physarum's panel: the modulator
  // never writes brightness, so with stem-follow off it is absolute and with it
  // on it is the base being scaled. Which one is live is stated in the workbench's
  // stem-follow folder. No band, for the same reason.
  bind('brightness', { min: 0, max: 2, step: 0.01, label: 'brightness (base × stem-follow)' });
  bind('intensity', { min: 0, max: 4, step: 0.01, label: 'intensity (manual)' });
  // Same story one level down: θ owns this number, and the population lane's two
  // multipliers scale it rather than replacing it.
  bind('aliveFraction', {
    min: 0,
    max: 1,
    step: 0.01,
    label: 'alive fraction (base × pop-follow)',
  });
  bind('radiusScale', { min: 0.5, max: 2, step: 0.01, label: 'reach ×' });
  bind('forceScale', { min: 0.3, max: 3, step: 0.01, label: 'force ×' });
  bind('friction', { min: 0.8, max: 8, step: 0.05, label: 'friction (1/s)' });
  bind('wander', { min: 0.002, max: 0.15, step: 0.001, label: 'wander' });
  bind('size', { min: 0.0008, max: 0.006, step: 0.0001, label: 'sprite size' });
  bind('stretch', { min: 0, max: 8, step: 0.05, label: 'velocity stretch' });
  // `pane.refresh()` re-reads bindings but knows nothing about folder titles, so
  // the caller drives this from its own refresh — which is also what makes the
  // marker correct after a *load* or an explorer promotion, where the flag
  // changed without anyone clicking the checkbox.
  return () => {
    f.title = title();
  };
}
