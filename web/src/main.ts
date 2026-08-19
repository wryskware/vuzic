import './style.css';
import { AudioClock } from './audio/clock';
import { DebugOverlay } from './debug/overlay';
import { ExplorerLog, type ExplorerAction } from './explore/log';
import { ExplorerRig, TILE_COUNT } from './explore/rig';
import { ExplorerSearch, type CandidateSet, type ExplorerSubspace } from './explore/search';
import {
  initGpu,
  renderUnsupportedPage,
  type BrowserGpuContext,
} from './gpu/browser-context';
import { buildDriverBank, MAX_CONTINUOUS_TICK_GAP } from './mapping/modulation';
import { captureBrowserExportRecipe } from './export/browser-recipe';
import { createLocalExportClient } from './export/client';
import { createExportSession } from './export/session';
import {
  defaultModulationConfig,
  downloadText,
  loadModulationLocal,
  modulationFits,
} from './mapping/persist';
import type { ModTarget } from './mapping/target';
import { TuningLog } from './mapping/tuninglog';
import { PhysarumSim } from './sim/physarum/physarum';
import { PlifeSim } from './sim/plife/plife';
import { isSeedPinned, resolveSeed, setPinnedSeed, syncUrlSeed } from './sim/seed';
import type { RenderFrame, Sim } from './sim/types';
import { VIZFX_IDS } from './sim/vizfx/ids';
import { VizFxSim } from './sim/vizfx/vizfx';
import { buildSimBundle, buildSimBundleFromRecipe, type SimBundle } from './runtime/sim-bundle';
import {
  consumePendingProfile,
  profileOutput,
  PROFILE_RENDERER_BUILD,
} from './ui/profiles';
import { consumeBootPreset } from './ui/presets';
import type { ExportRecipe } from './runtime/recipe';
import { invalidateIfStale, rememberCachedTrack } from './timeline/cache';
import {
  buildCatalog,
  DEFAULT_AUDIO_FILE,
  fetcherFor,
  type TrackEntry,
} from './timeline/catalog';
import { loadTimeline } from './timeline/loader';
import { TimelineSampler, type FeaturesFrame } from './timeline/sampler';
import { SECONDS_PER_TICK, TICK_HZ } from './timing';
import type { ExplorerPanelHost } from './ui/explore-panel';
import { planKey, type TargetInfo } from './ui/keys';
import { createPanel, type PanelHandle } from './ui/panel';
import { createPlifePanel } from './ui/plife-panel';
import { MILKDROP_MODE } from './ui/sim-panel';
import { createVizFxPanel } from './ui/vizfx-panel';

const DEFAULT_TRACK = 'free-fall';
const FALLBACK_TRACK = 'synthetic';
const DEFAULT_SIM = 'plife';
/**
 * Every `?sim=` value that resolves to a real substrate.
 *
 * The two simulations are spelled out; the vizfx visuals are appended from the
 * repertoire (`sim/vizfx/visuals.ts`), because each visual is its own entry here
 * — its own θ vector, its own autosave slot — and the repertoire file is the one
 * place a new visual is declared. Only string-ness is relied on below (`includes`
 * and the picker's option list), so the derivation costs no type safety that the
 * old literal tuple was actually providing.
 */
const SIMS: readonly string[] = ['physarum', 'plife', ...VIZFX_IDS];
/**
 * How long a milkdrop preset is guaranteed before a section boundary may replace
 * it. 12 s.
 *
 * Section boundaries are not evenly spaced: an intro tag, a two-bar fill lifted
 * into its own segment, or a bridge that the analysis split in two can arrive
 * within seconds of each other, and honouring every one of them would be a
 * strobe of substrate swaps rather than a slideshow. 12 s is comfortably shorter
 * than a real verse or chorus (15–30 s) — so consecutive *sections* each still
 * get their own visual — and long enough that a cluster of short segments
 * collapses to one transition.
 *
 * Measured from the last committed swap of any kind, deliberately including the
 * manual ones: picking a preset by hand and having a boundary take it away one
 * second later is exactly the thrash this exists to prevent.
 */
const AUTO_ADVANCE_DWELL_MS = 12_000;
// Eight 120 Hz ticks preserves the old four-60-Hz-tick catch-up window.
const MAX_FREE_TICKS_PER_FRAME = 8;
const PANEL_REFRESH_FRAMES = 30;

/**
 * Where explorer mode's mutation step starts. Mid-range on purpose: at 0.05 the
 * eight candidates are indistinguishable from the centre and there is nothing to
 * choose between, and at 1.0 the first generation has already left the
 * neighbourhood you liked. 0.3 is a visible difference you can still attribute.
 */
const EXPLORER_STEP_DEFAULT = 0.3;

/**
 * App ticks per *plife* tile tick in explorer mode. 2, i.e. half the live sim's
 * new 120 Hz physics rate (the tiles remain a 60-state-per-second fallback).
 *
 * Particle life's world is normalised rather than measured in pixels, so a small
 * tile is not a cheap tile: nine tiles is nine full populations and nine full
 * force passes, on a sim whose own header already warns that one of them will
 * not hold 60 fps everywhere. Halving the substep count is the sanctioned
 * fallback and it is honest about what it costs — the tile worlds evolve at half
 * speed, all nine equally, so what you are comparing is unaffected even though
 * what you are watching is slower.
 *
 * The next lever, deliberately *not* pulled, is `maxParticles`: it is the
 * biggest one, but colony density is part of what a candidate looks like, and a
 * tile with a ninth of the particles would be judging a world that is not the
 * one adopting the θ would give you. Physarum has no such problem — its world
 * *is* the pixel grid, so a third-size tile is a ninth of the cells, and scaling
 * its agent pool to match keeps the density identical.
 */
const PLIFE_TILE_TICK_EVERY = 2;

/** Floor on a physarum tile's agent pool, so a tiny window cannot starve a tile. */
const MIN_TILE_AGENTS = 4096;

/**
 * How long the live θ has to hold still after a panel edit before the grid is
 * regenerated around it.
 *
 * A slider drag is a *stream* of writes, not one, and every regeneration reseeds
 * and restarts nine worlds — so recentering on each observed change would mean
 * dragging a slider across nine flickering tiles that never live long enough to
 * show what they are. Half a second is comfortably past the end of a drag and
 * short enough that "I moved it, the grid followed" still reads as one gesture.
 * The observation cadence is `PANEL_REFRESH_FRAMES`, so the true latency is one
 * poll interval on top of this.
 */
const EXPLORER_EDIT_SETTLE_MS = 500;

/**
 * Time constant of the frame-rate EMA, in ms.
 *
 * One number, two consumers — the status bar's readout and plife's budget
 * governor — and it is main's rather than either consumer's because the frame
 * rate is a property of *this loop*: it covers the frames where nine explorer
 * tiles were drawn and none of the live sim, and both substrates need the same
 * segment in the same status line. A sim that measured its own frame time would
 * be measuring how often something chose to call it.
 *
 * 400 ms is about 25 frames at 60 Hz. Long enough that one compositor hitch or
 * one shader compile does not move the number a human can read, short enough
 * that dragging the pair-search dropdown to brute shows the consequence inside a
 * second — which is exactly the loop the governor is also closing.
 */
const FPS_EMA_TAU_MS = 400;

/**
 * How often the *displayed* fps is re-read from the EMA. ~2 Hz.
 *
 * The EMA is already smooth; this is about the text. A number that changes sixty
 * times a second is unreadable however stable the underlying value is, and the
 * status line is rebuilt every frame — so the integer is latched here and the
 * string is assembled from the latch.
 */
const FPS_DISPLAY_MS = 500;

const stage = document.getElementById('stage') as HTMLCanvasElement;
const overlayCanvas = document.getElementById('overlay') as HTMLCanvasElement;
const playButton = document.getElementById('play') as HTMLButtonElement;
const haltButton = document.getElementById('halt') as HTMLButtonElement;
const statusEl = document.getElementById('status') as HTMLElement;
const fatalHost = document.getElementById('fatal-host') as HTMLElement;

/** Timelines live at `timelines/<id>/`; `?track=` picks one, restricted to a dir name. */
function requestedTrack(): string {
  const param = new URLSearchParams(location.search).get('track');
  return param !== null && /^[A-Za-z0-9._-]+$/.test(param) ? param : DEFAULT_TRACK;
}

/**
 * Change track by changing the URL and reloading.
 *
 * The timeline is threaded into the sampler, the clock, the driver bank, the
 * modulator, the impulse engine and the sim itself, each built from it exactly
 * once at startup. Swapping it live would mean a second construction path for
 * every one of those, kept in step with the first forever, to save a reload that
 * costs under a second on localhost and comes back with the autosaved mapping
 * restored anyway. Every other parameter is preserved, so `?seed=` and `?sim=`
 * survive the switch — "same world, different song" is one click.
 */
function switchTrack(id: string): void {
  const url = new URL(location.href);
  url.searchParams.set('track', id);
  location.href = url.toString();
}

/**
 * `?sim=` picks the simulation, the same way `?track=` picks the timeline. It is
 * also the persistence key (`ModTarget.simId`), so an unknown value has to be
 * refused rather than passed through — otherwise `?sim=typo` would quietly start
 * physarum against an empty autosave slot named after the typo.
 *
 * `?sim=milkdrop` is accepted as an *alias* for the first visual in the
 * repertoire, because the mode picker calls the family by that name and a URL
 * that cannot say what the UI says is a small trap. It is resolved to a concrete
 * id here, before anything downstream exists — so the string never reaches
 * `buildSimBundle`, never becomes a `ModTarget.simId`, and can never name an
 * autosave slot. `switchSim` writes the concrete id back into the URL on the
 * first swap, so the alias is an entry point and not a state.
 */
function requestedSim(): string {
  const param = new URLSearchParams(location.search).get('sim');
  if (param === null || !/^[a-z-]+$/.test(param)) return DEFAULT_SIM;
  return param === MILKDROP_MODE ? (VIZFX_IDS[0] ?? DEFAULT_SIM) : param;
}

/**
 * Exact per-slot equality of two θ vectors.
 *
 * Exact, and not a tolerance: this answers "did a human move a control", the two
 * vectors being compared are both `currentVector()` reads of the same config, and
 * every write between them is a plain assignment of the number a widget produced.
 * A tolerance here would only ever hide a real edit at the fine end of a slider.
 * Length is compared first because a reroll can in principle land between two
 * reads of a sim whose registry length is a function of K.
 */
function sameTheta(a: Float64Array, b: Float64Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

async function main(): Promise<void> {
  /**
   * The authored state this boot starts from, if it did not start from the
   * autosave: a preset string (staged by the workbench, or carried in a `#p=`
   * fragment — `ui/presets.ts`) or a saved profile (`ui/profiles.ts`).
   *
   * Read first, because between them they decide four things every construction
   * below is built from: which substrate, which seed, which authored state,
   * and — presets only — which track. **Both channels are consumed
   * unconditionally, and only then does one of them win**: a slot left behind by
   * its own failed apply would re-apply on every reload, which turns one bad
   * string into an app that cannot be started without devtools.
   *
   * A preset outranks a profile because a preset is only ever armed one action
   * before the reload that reads it, and the two can only be set at once by a
   * session that pressed both buttons in the same instant.
   */
  const boot = await consumeBootPreset();
  if (boot.error !== null) console.warn(`preset: ignoring an unusable preset — ${boot.error}`);
  const staged = consumePendingProfile();
  let profile = boot.preset?.recipe ?? staged;
  /**
   * Advisory, and only a preset has one. A profile carries a real track id too,
   * but loading a profile has never switched tracks, and quietly changing that
   * here would be a second feature riding in on this one.
   */
  let trackHint = boot.preset?.plan.trackHint ?? null;
  if (profile && !SIMS.includes(profile.sim)) {
    // Saved state for a substrate this build does not have. Dropping it whole is
    // the only coherent answer: its simulation config, θ and palette are all
    // that sim's, so there is nothing left to partially honour — including its
    // track, which was chosen to go with a look that is not being applied.
    console.warn(`profile: ignoring saved state for unknown sim "${profile.sim}"`);
    profile = null;
    trackHint = null;
  }
  if (profile) {
    console.info(
      boot.preset !== null
        ? `preset: applying a ${boot.preset.source} preset for ${profile.sim}`
        : `profile: applying a saved ${profile.sim} profile`,
    );
  }

  // A profile always replays its own seed — a particle-life matrix is generated
  // from it, so the same seed *is* half of "the look I saved". Its pin state is
  // restored alongside, into both channels, so the next reload does what the
  // profile said rather than what the previous session had pinned.
  const { seed, pinned } = profile ? { seed: profile.seed, pinned: profile.seedPinned } : resolveSeed();
  if (profile) {
    setPinnedSeed(profile.seedPinned ? profile.seed : null);
    syncUrlSeed(profile.seedPinned ? profile.seed : null);
  }

  // Bundled tracks, whatever a local analysis server offers, and whatever is
  // still in the offline cache from a previous session — one list, resolved
  // before anything is built, because the timeline is a construction argument to
  // half the app. The probe is short and its failure is normal.
  const catalog = await buildCatalog();
  const fallbackEntry: TrackEntry = {
    id: FALLBACK_TRACK,
    title: FALLBACK_TRACK,
    duration: 0,
    version: '',
    base: `${import.meta.env.BASE_URL}timelines/${FALLBACK_TRACK}`,
    hasAudio: false,
    audioFile: DEFAULT_AUDIO_FILE,
    source: 'bundled',
  };
  const pick = (id: string): TrackEntry =>
    catalog.tracks.find((t) => t.id === id) ?? {
      ...fallbackEntry,
      id,
      title: id,
      base: `${import.meta.env.BASE_URL}timelines/${id}`,
    };

  // A preset's track is a *hint*: the music never travels with the look, so a
  // preset made against a track this origin does not have still applies — it
  // just applies to whatever is playing, and says so.
  if (trackHint !== null && !catalog.tracks.some((t) => t.id === trackHint)) {
    console.info(`preset: no local track "${trackHint}"; keeping the current one`);
    trackHint = null;
  }
  let entry = pick(trackHint ?? requestedTrack());
  let track = entry.id;
  let timeline;
  try {
    // A server track whose content moved is evicted before the read-through
    // below can answer from a stale copy of half of it.
    if (entry.source !== 'bundled') await invalidateIfStale(entry);
    timeline = await loadTimeline(entry.base, fetcherFor(entry));
    if (entry.source !== 'bundled') rememberCachedTrack(entry);
  } catch (err) {
    // A missing or broken timeline must not dead-end the app: the synthetic one
    // ships with the repo and always loads.
    if (track === FALLBACK_TRACK) throw err;
    console.warn(`timeline "${track}" failed to load; falling back to "${FALLBACK_TRACK}"`, err);
    entry = pick(FALLBACK_TRACK);
    track = entry.id;
    timeline = await loadTimeline(entry.base, fetcherFor(entry));
  }
  const sampler = new TimelineSampler(timeline, SECONDS_PER_TICK);
  console.info(`simulation clock: ${TICK_HZ} Hz fixed timestep`);

  const clock = new AudioClock(
    {
      duration: timeline.manifest.track.duration,
      beats: timeline.manifest.beats,
      downbeats: timeline.manifest.downbeats,
    },
    {
      secondsPerTick: SECONDS_PER_TICK,
      audioUrl: `${entry.base}/${entry.audioFile}`,
      fetcher: fetcherFor(entry),
    },
  );
  // Fetch ahead of the first gesture; the click-track fallback costs nothing if
  // this 404s.
  clock.preload();

  const overlay = new DebugOverlay(overlayCanvas, sampler, {
    onSeek: (t) => clock.seek(t),
  });

  // The modulation input: ~16 named drivers built from this timeline's own
  // structure channels and its 64-dim latent channel, variance-reordered and
  // z-scored once, here (plan.md Revision 4). No sidecar, no background upgrade.
  //
  // Timeline-shaped, not substrate-shaped, so it is built once out here and
  // handed to every Modulator a sim swap goes on to construct — the same reason
  // the sampler, the clock and the overlay above it are not rebuilt either.
  const drivers = buildDriverBank(timeline);
  const tuningLog = new TuningLog();
  // Client and session are both main-lifetime, and the session is the reason the
  // panel below can be thrown away and rebuilt by a substrate swap while a
  // render is still going: it, not the folder, owns the job.
  const exportClient = createLocalExportClient();
  const exportSession = createExportSession(exportClient);
  if (sampler.events.length === 0) {
    console.info(`timeline "${track}" has no events array; impulses idle until test-fired`);
  }

  /**
   * Browser policy around the shared construction path. Startup calls it;
   * `switchSim` calls it.
   *
   * That it is *one* function is the whole point rather than a tidiness
   * preference. The order inside the shared builder is load-bearing in three
   * places (the accent channels before the first tick, `applyBase` before
   * `setMode`, `onSeedChange` before anything can reseed), and every one is the kind
   * of constraint that a second, hand-kept-in-step copy silently loses six months
   * later. If a live swap had been written as its own sequence it would have been
   * a slowly diverging duplicate of this; instead the only thing the swap owns is
   * *when* to call this and what to tear down first.
   *
   * `sim.init(ctx)` is deliberately **not** in here. It is the one step whose
   * failure has to be recoverable differently by the two callers — at startup it
   * means "this browser cannot run the app", during a swap it means "keep the sim
   * you already have" — so it stays at each call site where that choice is made.
   */
  const buildBrowserSimBundle = (id: string, forSeed: number, authored?: ExportRecipe): SimBundle => {
    // A staged profile is a complete recipe, and rehydrating one is already a
    // solved problem with a canonical implementation — the same one the export
    // worker boots from. Delegating keeps "what a recipe means" in one place
    // instead of growing a browser-flavoured second reading of it here.
    const bundle = authored
      ? buildSimBundleFromRecipe({ recipe: authored, sampler, drivers, secondsPerTick: SECONDS_PER_TICK })
      : buildSimBundle({
          id,
          seed: forSeed,
          sampler,
          drivers,
          secondsPerTick: SECONDS_PER_TICK,
          // `sim.simId` and not the requested id: it is the autosave slot's name,
          // and taking it from the constructed object keeps an unrecognised id on
          // physarum's mapping instead of opening an empty slot named after a typo.
          resolveModulationConfig: (candidate) => {
            const stored = loadModulationLocal(candidate.simId);
            return stored && modulationFits(stored, candidate.config.speciesCount, candidate.simId)
              ? stored
              : defaultModulationConfig(candidate.config, candidate.simId);
          },
        });
    console.info(
      `modulation: ${bundle.sim.simId} · ${bundle.modulator.sourceLabel} → ${bundle.modulator.modulatedCount} parameters, ` +
        `seed ${forSeed}${pinned ? ' (pinned)' : ''}`,
    );
    return bundle;
  };

  // Resolve which sim to build before building it, so a second substrate slots
  // in here rather than being threaded through the twenty call sites below —
  // those all talk to `ModTarget` now and do not care which one they got.
  //
  // A staged profile names its own substrate and wins over `?sim=`: it was saved
  // as one world, and loading it into a different one would be loading nothing
  // it actually contains.
  const wantedSim = profile?.sim ?? requestedSim();
  if (!(SIMS as readonly string[]).includes(wantedSim)) {
    console.warn(`sim "${wantedSim}" not available; using ${DEFAULT_SIM}`);
  }
  // `let` and not `const`, because the substrate picker replaces all three at
  // once (`switchSim`). Every closure below reads them rather than capturing
  // them, which is what makes the swap invisible to the frame loop.
  //
  // The profile is passed **only** to this first build. A later substrate swap
  // is a request for that substrate's own autosaved mapping, not a second
  // application of a profile the user has since edited away from.
  let { sim, impulses, modulator } = buildBrowserSimBundle(wantedSim, seed, profile ?? undefined);

  // ── milkdrop mode ──────────────────────────────────────────────────────────
  //
  // "milkdrop" is what the picker calls the vizfx family, and it is presentation
  // only: every visual keeps its own `simId`, its own `?sim=` value and its own
  // autosave slot, and everything below deals in those concrete ids. The two
  // things the mode actually owns are this block.
  //
  /**
   * Whether a section boundary advances to the next preset. **On by default**,
   * and that is a judgement call worth stating:
   *
   * - it is the behaviour the feature was asked for, and the one that makes the
   *   mode a slideshow rather than a picker with a spare button;
   * - it is inert at a repertoire of one, which is what ships today — so the
   *   default cannot surprise anyone until a second visual exists, by which
   *   point it is the requested behaviour;
   * - the flag is session-local (below), so defaulting it *off* would mean
   *   turning it on again after every reload, which is real friction on a
   *   feature someone explicitly wanted.
   *
   * What it costs is that a swap can arrive mid-tune, rebuilding the panel under
   * a slider. The gates in `noteSectionBoundary` are what keep that rare — never
   * while paused, never while exploring, never inside the dwell window — and the
   * toggle is one click away in the milkdrop folder.
   *
   * Session-local on purpose: it is a way of watching, not a mapping decision,
   * so it has no business in the autosave slot that holds θ.
   */
  let autoAdvance = true;
  /**
   * Which visual "milkdrop" means when it is picked from the mode selector: the
   * last one that actually ran this session, not the first in the repertoire.
   * Held here because the panel that displays it is destroyed by every swap.
   */
  let milkdropEntry = sim instanceof VizFxSim ? sim.simId : (VIZFX_IDS[0] ?? '');
  /** `performance.now()` of the last committed swap — the dwell window's origin. */
  let lastSwapAt = performance.now();
  /** Segment index last seen by the boundary detector; -2 is "never looked". */
  let lastSegment = -2;
  /** Sim tick last seen by the boundary detector, for the playback/seek test. */
  let lastBoundaryTick = -1;
  /**
   * What the detector decided about the most recent segment change, for scripted
   * checks — the gate chain is otherwise invisible from outside, and "it did not
   * swap" is the same observation whether the logic ran or never fired at all.
   */
  let lastBoundaryDecision: { time: number; segment: number; action: string } | null = null;

  /** The next entry in repertoire order, wrapping. Equal to `id` when there is one. */
  const nextVisualAfter = (id: string): string => {
    const i = (VIZFX_IDS as readonly string[]).indexOf(id);
    return VIZFX_IDS[(i + 1) % VIZFX_IDS.length] ?? id;
  };

  let gpu: BrowserGpuContext | null = null;
  let panel: PanelHandle | null = null;

  // ── explorer mode: nine live worlds, one θ hill climb ──────────────────────
  //
  // Everything stateful about the mode lives here rather than in the rig,
  // because the mode is a *transport* decision as much as a rendering one: while
  // it is active the fixed-step pump stops feeding the live sim entirely (see
  // `frame` below), and that is main's business, not the rig's.
  //
  // Everything is built lazily, on the first enter. That is not just thrift: it
  // keeps a substrate that is never explored from paying for nine of itself, and
  // it means a failure to open is a failure to enter rather than a failure to
  // start the app.
  //
  // ## The grid and the panel are one system
  //
  // The mode used to be a sandbox: tile configs were cloned on the way in, the
  // chosen θ was copied back on the way out, and a slider moved in between went
  // to a frozen sim nobody could see. That is two searches over the same vector
  // pretending to be one, and it fails the obvious test — "I adjusted a parameter
  // and the exploration ignored me".
  //
  // So the centre θ is *always* the live sim's θ, in both directions:
  //
  //   grid → live   every pick/reroll/back writes `set.center` straight into the
  //                 live config (`explorerAdvance`), so the workbench sliders are a
  //                 readout of the current centre while the mode runs, and there
  //                 is nothing left to adopt on exit.
  //   live → grid   an edit that did *not* come from the grid — a slider, a file
  //                 load, a reroll — is detected by polling `currentVector` and
  //                 recentres the search on it (`explorerPollEdits`), so the nine
  //                 tiles regrow around wherever the human just put the centre.
  //
  // The live sim is still not ticked while the mode is open (see `advance`), so
  // "live" here means its config and its GPU-side θ state, not its motion.
  //
  // Entering therefore forces modulation to manual, up front rather than on the
  // way out: the live config is being written throughout, and a modulated tick on
  // exit would stamp over every one of those writes. It is deliberately *not*
  // restored afterwards — the human has spent the whole mode hand-authoring a θ,
  // and handing it straight back to the music is the one outcome that makes all
  // of that work invisible.
  let explorerRig: ExplorerRig | null = null;
  let explorerSearch: ExplorerSearch | null = null;
  let explorerLog: ExplorerLog | null = null;
  /** the generation on screen; also what 'like' is about */
  let explorerSet: CandidateSet | null = null;
  let explorerActive = false;
  /** open() is async; without this a double click builds two rigs */
  let explorerOpening = false;
  let explorerStep = EXPLORER_STEP_DEFAULT;
  let explorerSubspace: ExplorerSubspace = 'all';
  /** θ as it stood at `explorerEnter`; the revert button's whole content */
  let explorerEntryTheta: Float64Array | null = null;
  /**
   * The live θ as it was after the last write *this code* made, read back from
   * the sim rather than assumed. That read-back is the point: it means the
   * edit detector compares like with like, and any clamping or round-tripping
   * `applyTheta` might do can never be mistaken for a human moving a slider.
   */
  let explorerAppliedTheta: Float64Array | null = null;
  /** `performance.now()` of the most recently *observed* external θ change, or null */
  let explorerEditAt: number | null = null;

  /**
   * One tile's simulation: the same substrate as the live sim, with the same
   * channels wired, and a *copy* of its current config.
   *
   * The copy is the point. A tile built from the shipped defaults would differ
   * from the live world in every non-θ way at once — macros, the HDR chain, the
   * palette, the population lane — and the mode's claim is that the only
   * difference between the centre tile and what you were just watching is that
   * the music is driving nothing.
   *
   * The copy is a *starting* point, not a freeze: `ExplorerRig.syncStyle` keeps
   * everything-that-is-not-θ tracking the live config for as long as the mode is
   * open, so a palette or macro edit made while looking at nine tiles is visible
   * in all nine. What the clone still buys is ownership — nine tiles editing one
   * shared config object would be nine tiles that cannot differ at all.
   *
   * No Modulator and no ImpulseEngine, by design. Raw θ, unmediated, is the
   * whole reason the mode exists — a slew limiter between the candidate and the
   * screen would mean the nine tiles are nine smoothed approximations of nine
   * parameter sets rather than the parameter sets themselves.
   */
  const createExplorerTile = (): Sim & ModTarget => {
    if (sim instanceof VizFxSim) {
      // A vizfx tile is the cheapest of the three by a wide margin: its cost is
      // three full-screen passes at the tile's own pixel count, so nine
      // third-size tiles are exactly the work of one full-size sim. Nothing to
      // force off and no tick divisor, unlike plife.
      //
      // The tile takes the live sim's *visual* by reference — it is frozen data
      // (a slot table and two WGSL strings), so there is nothing to clone — and a
      // structuredClone of its config, for the reason the block above states.
      const tile = new VizFxSim(sim.visual, sim.currentSeed, structuredClone(sim.config));
      tile.setStemChannel(sampler.getChannel('stems'));
      return tile;
    }
    if (sim instanceof PlifeSim) {
      const tile = new PlifeSim(sim.currentSeed, structuredClone(sim.config));
      // Nine tiles never run the brute pair search, whatever the live sim is
      // set to. Nine times an O(N²) force pass measured at ~212 ms/frame with
      // 13 k alive — the mode works in the grid, it is just not a search tool at
      // 4.7 fps. See `PlifeSim.forceGridSearch` for why this is a flag and not a
      // value on the cloned config (`syncStyle` would overwrite the latter).
      tile.forceGridSearch = true;
      // Nine tiles, one canvas, one frame time. Nine governors reading that one
      // number would each blame themselves for it and each shed 15%, so the grid
      // would lose two thirds of its population in four seconds and then all
      // grow back together — one measurement counted nine times, oscillating.
      // A tile therefore runs at `budget.cap` flat, which is the same ceiling the
      // live sim is working under and the one the human actually set; the cap
      // slider is what makes the mode cheaper. Same flag mechanism, and the same
      // reason for it, as `forceGridSearch` — `syncStyle` fans the extras block
      // into all nine twice a second, so `budget.adaptive = false` on the clone
      // would not survive.
      tile.forceNoGovernor = true;
      tile.setStemChannel(sampler.getChannel('stems'));
      tile.setAccentChannels(sampler.getChannel('novelty16'), sampler.getChannel('actChorus'));
      return tile;
    }
    const cfg = structuredClone(sim.config);
    // Physarum's world is the pixel grid, so a 1/3-scale tile has 1/9 the cells;
    // scaling the pool by the same factor keeps agents-per-cell — which is what
    // the look is actually made of — identical to the full-screen sim.
    cfg.maxAgents = Math.max(MIN_TILE_AGENTS, Math.round(cfg.maxAgents / TILE_COUNT));
    const tile = new PhysarumSim(sim.currentSeed, cfg);
    tile.setStemChannel(sampler.getChannel('stems'));
    return tile;
  };

  const explorerRecord = (
    action: ExplorerAction,
    set: CandidateSet,
    pickedIndex?: number,
  ): void => {
    explorerLog?.append({
      action,
      generation: set.generation,
      subspace: explorerSubspace,
      step: explorerStep,
      genSeed: set.genSeed,
      ...(pickedIndex === undefined ? {} : { pickedIndex }),
      theta: Array.from(set.center),
    });
  };

  /**
   * Take a fresh reading of the live θ as "what the mode last put there", so the
   * edit detector has something honest to compare against.
   *
   * Read back rather than assumed, and read back **after the panel has been
   * refreshed**, because the panel is not a passive mirror: tweakpane refreshes a
   * bound number through the widget's own `step` constraint and writes the
   * quantised value back into the config. A candidate θ is a continuum of raw
   * floats, so the first refresh after a pick silently rounds dozens of slots by
   * up to half a step — and taking the reference before that made the mode's own
   * rounding look like a human moving sliders, which recentred the search and
   * re-rolled all nine tiles about a second after every pick.
   */
  const explorerMarkApplied = (): void => {
    explorerAppliedTheta = sim.currentVector();
    explorerEditAt = null;
  };

  /**
   * Put a generation on screen, on the live sim, and in the log.
   *
   * The order is load-bearing twice over. The live sim is written *before* the
   * tiles sync their style from it, so a generation's tiles are styled from the
   * config state their θ came out of; and the applied-θ reference is taken *after*
   * `panel.refresh()`, for the quantisation reason on `explorerMarkApplied`.
   *
   * No mask on the write: a centre is a complete θ, including the slots the
   * modulator is never allowed to move, and all of it is what is on screen in the
   * middle tile.
   */
  const explorerAdvance = (
    set: CandidateSet,
    action: ExplorerAction,
    pickedIndex?: number,
  ): void => {
    explorerSet = set;
    sim.applyTheta(set.center);
    explorerRig?.setCandidates(set);
    // Freshly reseeded tiles start from their clone's art direction, which may be
    // several palette edits stale by now; one sync here means a new generation is
    // never briefly the wrong colour while it waits for the next poll.
    explorerRig?.syncStyle(sim);
    explorerRecord(action, set, pickedIndex);
    panel?.refresh();
    explorerMarkApplied();
  };

  const explorerEnter = async (): Promise<void> => {
    const ctx = gpu;
    if (explorerActive || explorerOpening || !ctx) return;
    explorerOpening = true;
    try {
      // The log persists across runs of the mode — it is preference data, not
      // session state. The search does not: a new entry is a new climb, and it
      // starts from wherever the live sim is now, under the current seed.
      explorerLog ??= new ExplorerLog(sim.simId);
      explorerEntryTheta = sim.currentVector();
      explorerSearch = new ExplorerSearch(sim.registry(), sim.currentSeed);
      explorerSearch.setStep(explorerStep);
      explorerSearch.setSubspace(explorerSubspace);
      explorerRig ??= new ExplorerRig({
        gpu: ctx,
        createTile: createExplorerTile,
        tickEvery: sim instanceof PlifeSim ? PLIFE_TILE_TICK_EVERY : 1,
      });
      await explorerRig.open();
      explorerRig.layout(ctx.width, ctx.height);
      explorerActive = true;
      // Up front, not on the way out — from here on the live config is being
      // written by the mode and a modulated tick would overwrite all of it (see
      // the block comment above). After `open()` succeeded, though: a failed
      // enter must leave the app exactly as it found it, and silently killing
      // modulation would not be that.
      modulator.setMode('manual');
      // The scrub strip is 300 px of 82%-opaque canvas across the bottom of the
      // stage: left up it would cover the bottom row of tiles *and* swallow
      // their clicks. Space and the arrow keys still drive the transport, which
      // is the part of it that matters here.
      overlayCanvas.style.display = 'none';
      explorerAdvance(explorerSearch.start(sim.currentVector()), 'start');
    } catch (err) {
      // A half-open rig would render nine black tiles forever, so the failure is
      // unwound completely and the app goes back to the live sim.
      console.error('explorer: could not enter', err);
      explorerRig?.close();
      explorerSearch = null;
      explorerSet = null;
      explorerActive = false;
      explorerEntryTheta = null;
      explorerAppliedTheta = null;
      explorerEditAt = null;
      overlayCanvas.style.display = '';
    } finally {
      explorerOpening = false;
      panel?.refresh();
    }
  };

  /**
   * Stop looking at nine worlds. Nothing is logged; on plife, the centre tile's
   * *world* comes with you.
   *
   * There is no adopt/discard pair, because the centre has been the live sim's θ
   * since the moment it became the centre — so exit is a view change, not a
   * commit. What it is not is a pure view change: θ alone does not reconstitute a
   * colony arrangement, and exiting used to give you the parameters you picked
   * attached to the unrelated world the live sim had been frozen in since entry.
   * `adoptParticleState` closes that gap for the one substrate where it is a
   * clean copy.
   *
   * **plife promotes, physarum and vizfx do not**, and the asymmetry is a
   * decision:
   *
   * - plife's tiles are `structuredClone`s of the live config, so they run the
   *   same `maxParticles` in the same normalised world space, and per-particle
   *   state is one 32-byte struct in one buffer. Tile and live sim are literally
   *   buffer-compatible; the copy is exact.
   * - physarum's world *is* its pixel grid, and a tile's grid is a third the size
   *   in each axis (`createExplorerTile` scales the agent pool to match). Agent
   *   positions, the trail field and the soil field are all in grid units, so
   *   there is nothing size-compatible to copy and nothing meaningful to
   *   resample. It also needs it least: the look is carried by the trail field,
   *   which re-forms from the agents within seconds of exit.
   * - a vizfx tile's world *is* a texture at the tile's own size, so the copy is
   *   not merely size-incompatible, it is meaning-incompatible: every θ radius is
   *   expressed in aspect-corrected screen units, and rescaling the field would
   *   change what all of them meant. The same objection `VizFxSim.ensureField`
   *   states about a canvas resize. The picked θ still comes with you and rebuilds
   *   a comparable nebula over the field's own memory, about a second.
   *
   * Ordering is load-bearing twice. The copy is encoded *before* `close()`,
   * which disposes the tile's buffers; and it is submitted from a DOM event
   * handler, never from inside `frame()`, so it can never land between the passes
   * of a `tick` — every sim step is encoded and submitted synchronously inside
   * one `advance` call, and JS gives us no way to interleave with that. The queue
   * therefore sees: every tile step, then this copy, then the first live step.
   *
   * `explorerRevert` deliberately does not promote: revert means "throw this
   * away", and handing back the world of a run you just undid would be the
   * opposite of that.
   *
   * Modulation stays manual, deliberately. See the block comment above.
   */
  const explorerExit = (): void => {
    if (!explorerActive) return;
    const center = explorerRig?.centerSim ?? null;
    if (sim instanceof PlifeSim && center instanceof PlifeSim) sim.adoptParticleState(center);
    explorerActive = false;
    explorerRig?.close();
    overlayCanvas.style.display = '';
    explorerSearch = null;
    explorerSet = null;
    explorerEntryTheta = null;
    explorerAppliedTheta = null;
    explorerEditAt = null;
    panel?.refresh();
  };

  /**
   * The undo for the whole run: put the live sim back to the θ it had when the
   * mode was entered, and regrow the grid around it.
   *
   * `recenter` rather than `start` so the climb keeps its history — a revert is
   * itself undoable with `back()`, which matters because it is the one button
   * here that can throw away twenty generations of work.
   *
   * Available only while the mode is active. Offering it afterwards would mean
   * holding an entry θ that a later panel edit has already made a lie, and "revert
   * to something you were at ten minutes and forty slider moves ago" is not an
   * undo, it is a trap.
   */
  const explorerRevert = (): void => {
    if (!explorerActive || !explorerSearch || !explorerEntryTheta) return;
    explorerAdvance(explorerSearch.recenter(explorerEntryTheta), 'revert');
  };

  /**
   * Notice a θ edit that did not come from the grid, and recentre the search on it.
   *
   * The live config has exactly two writers while the mode is open: this file
   * (through `explorerAdvance`, which records what it wrote) and the workbench
   * panel's own bindings (which write the config object directly and tell nobody).
   * So "θ differs from the last thing we applied" *is* "the human moved something",
   * and no extra plumbing through the panel is needed to know it — which is the
   * reason it is done this way rather than by a change callback on every widget.
   *
   * Exact per-slot comparison, not an epsilon: the panel writes whatever number
   * the slider produced, and a tolerance would silently swallow the fine end of a
   * step-0.001 control.
   *
   * The debounce is restarted on every observed change, so a drag — which is a
   * stream of them — regenerates once at the end rather than once per poll.
   */
  const explorerPollEdits = (now: number): void => {
    if (!explorerActive || !explorerSearch || !explorerAppliedTheta) return;
    const live = sim.currentVector();
    if (!sameTheta(live, explorerAppliedTheta)) {
      // Track the newest observation, not the last applied one: while a drag is in
      // flight every poll should see "changed again" and push the deadline out.
      explorerAppliedTheta = live;
      explorerEditAt = now;
      return;
    }
    if (explorerEditAt === null || now - explorerEditAt < EXPLORER_EDIT_SETTLE_MS) return;
    explorerAdvance(explorerSearch.recenter(live), 'recenter');
  };

  const explorerPick = (index: number): void => {
    if (!explorerActive || !explorerSearch) return;
    explorerAdvance(explorerSearch.pick(index), 'pick', index);
  };

  const explorerReroll = (): void => {
    if (!explorerActive || !explorerSearch) return;
    explorerAdvance(explorerSearch.reroll(), 'reroll');
  };

  const explorerBack = (): void => {
    if (!explorerActive || !explorerSearch) return;
    const set = explorerSearch.back();
    // Null at the root. A no-op is the honest answer — re-rolling instead would
    // dress up "there is nothing behind this" as an undo.
    if (set) explorerAdvance(set, 'back');
  };

  const explorerHost: ExplorerPanelHost = {
    get available(): boolean {
      return gpu !== null;
    },
    get active(): boolean {
      return explorerActive;
    },
    get generation(): number {
      return explorerSet?.generation ?? 0;
    },
    get logSize(): number {
      return explorerLog?.size ?? 0;
    },
    get step(): number {
      return explorerStep;
    },
    get subspace(): ExplorerSubspace {
      return explorerSubspace;
    },
    enter(): void {
      void explorerEnter();
    },
    exit: explorerExit,
    revert: explorerRevert,
    reroll: explorerReroll,
    back: explorerBack,
    like(): void {
      if (!explorerActive || !explorerSet) return;
      explorerRecord('like', explorerSet);
      panel?.refresh();
    },
    exportLog(): void {
      if (!explorerLog) return;
      downloadText(
        `explorer-${sim.simId}-${timeline.manifest.track.id}.jsonl`,
        explorerLog.exportJsonl(),
        'application/x-ndjson',
      );
    },
    // Both knobs are mirrored here as well as pushed into the search, because
    // they are editable before any search exists — and both only take effect
    // from the next generated set anyway, so there is nothing to re-apply.
    setStep(v: number): void {
      explorerStep = v;
      explorerSearch?.setStep(v);
    },
    setSubspace(s: ExplorerSubspace): void {
      explorerSubspace = s;
      explorerSearch?.setSubspace(s);
    },
  };

  /**
   * Build the workbench for whichever substrate is live *now*.
   *
   * A panel binds tweakpane widgets to its sim's config fields by name, so each
   * substrate has its own. Everything *around* the widgets — the workbench, the
   * impulse folder, the HDR chain, the two pickers — is shared, so the three
   * calls differ only in which factory they name and take an identical options
   * object.
   *
   * Every one of those options is read at call time rather than captured once:
   * `sim`, `impulses` and `modulator` are the three things a swap replaces, and a
   * panel built against the previous trio would be a live set of sliders wired to
   * a disposed sim. The hosts that are *not* rebuilt (the explorer host, the two
   * `switchTo` callbacks) are closures over `let`s for the same reason.
   */
  const buildPanel = (): PanelHandle => {
    const panelOpts = {
      // The workbench can pin/unpin/reroll after startup. Rebuilding the panel
      // for a substrate swap must describe the live world, not resolveSeed()'s
      // startup snapshot.
      pinned: isSeedPinned(sim.currentSeed),
      // Both reroll buttons land here, after sim.reseed() — which has already
      // re-keyed the hotspots via onSeedChange. What is left is dropping any
      // envelope still ringing. The transport deliberately keeps its position:
      // a reroll is "new world, same song", and rewinding to 0:00 on every
      // reroll made auditioning seeds against a chorus impossible (user call —
      // the original "never resume against an arbitrary position" rule lost to
      // practice). The overlay scrub and arrow keys are the way back to the top.
      onRestart: (): void => {
        impulses.reset();
      },
      impulses,
      explorer: explorerHost,
      tracks: {
        tracks: catalog.tracks,
        current: track,
        serverUp: catalog.server,
        serverCount: catalog.serverCount,
        switchTo: switchTrack,
      },
      exports: {
        session: exportSession,
        trackId: entry.id,
        // `sim`, `modulator`, and `impulses` are the live `let`s replaced as one
        // unit by a substrate swap. Capture reads them only on the button click.
        capture: (rendererBuild: string, output: ExportRecipe['output']) =>
          captureBrowserExportRecipe({
            rendererBuild,
            track: entry,
            source: { sim, modulator, impulses },
            output,
          }),
      },
      sims: {
        ids: SIMS,
        presets: VIZFX_IDS,
        current: sim.simId,
        entryPreset: milkdropEntry,
        switchTo: (id: string): void => void switchSim(id),
        // Read at build time, written through: the panel is rebuilt by every
        // swap, so the value it shows is re-derived from this `let` rather than
        // being the panel's own copy of the truth.
        autoAdvance,
        setAutoAdvance: (on: boolean): void => {
          autoAdvance = on;
        },
      },
      workbench: {
        sim,
        modulator,
        log: tuningLog,
        trackId: timeline.manifest.track.id,
        time: () => clock.time,
        tick: () => clock.simTick,
        seek: (t: number) => clock.seek(t),
        // The same capture the export button takes, from the same live `let`s,
        // with the two export-only fields stubbed. `contentVersion` is one of
        // them in effect: a profile carries the track id as a note to a human
        // and never replays a timeline, so an unsynced track must not be a
        // reason you cannot save the look you just tuned.
        captureProfile: (): ExportRecipe =>
          captureBrowserExportRecipe({
            rendererBuild: PROFILE_RENDERER_BUILD,
            track: { id: entry.id, version: entry.version === '' ? 'unversioned' : entry.version },
            source: { sim, modulator, impulses },
            output: profileOutput(),
          }),
      },
    };
    return sim instanceof PhysarumSim
      ? createPanel(sim, panelOpts)
      : sim instanceof PlifeSim
        ? createPlifePanel(sim, panelOpts)
        : createVizFxPanel(sim, panelOpts);
  };

  /** open() is async; without this a double click builds two substrates */
  let swapping = false;

  /**
   * Change substrate **without a reload**, unlike `switchTrack` above.
   *
   * The two are not the same problem, which is why they get opposite answers. A
   * track reaches into the sampler, the clock, the driver bank *and* the sim, and
   * rebuilding it live would mean a second construction path for each. A
   * substrate reaches into exactly three objects — itself, the impulse engine
   * sized to its species count, and the modulator keyed to its θ registry — and
   * those three already have a single construction path (`buildSimBundle`) that
   * this function calls rather than reimplements. The timeline, the sampler, the
   * clock, the catalog, the driver bank, the overlay and the GPU context are all
   * substrate-independent and stay exactly where they are.
   *
   * What that buys is the thing a reload cannot: **the music does not stop**. The
   * whole point of a substrate picker is to answer "which of these looks better
   * against this bar", and a reload puts you back at 0:00 with a cold audio
   * element every time you ask.
   *
   * ## The order here is the careful part
   *
   * 1. **Explorer first.** Its nine tiles are the *old* substrate, its rig's tick
   *    divisor was chosen for the old substrate, and its log is keyed to the old
   *    `simId`. So the mode is exited, the rig dropped (its `close()` disposes the
   *    tiles and releases the layered target) and the log dropped, before anything
   *    else moves. Re-entering after a swap rebuilds all three against the new sim.
   *
   *    One visible consequence, and it is the right one: exiting explorer mode
   *    normally leaves modulation in `manual`, because the human has just spent
   *    the mode hand-authoring a θ and handing it straight back to the music would
   *    make that work invisible. A swap re-reads the *new* substrate's autosave
   *    slot and comes back modulated — correctly, because the θ that rule exists
   *    to protect belongs to the substrate that just went away.
   * 2. **Build and `init` the new sim while the old one is still running.** The
   *    frame loop keeps ticking and rendering the outgoing sim throughout the
   *    await — it reads `sim`, which has not moved yet — so there is no window
   *    where the loop can touch a sim whose pipelines do not exist. Two substrates
   *    hold GPU memory at once for the duration; explorer mode already runs ten at
   *    once, so this is not the expensive case.
   * 3. **Commit, then dispose.** The three `let`s are reassigned together and the
   *    panel is rebuilt before the old sim's buffers are destroyed. Both happen in
   *    a promise continuation, never inside `frame()` — JS gives no way to
   *    interleave with a synchronous `advance`/`render`/`submit` — so the queue
   *    sees every one of the old sim's passes submitted before its textures go.
   *
   * The seed carried across is `sim.currentSeed` and not the startup `seed`: a
   * reroll moves the former and not the latter, and "switch substrate" must mean
   * the world you are looking at, not the one you started with. A pinned `?seed=`
   * is untouched by that, so it keeps meaning exactly what it meant.
   *
   * `?sim=` is updated with `replaceState` rather than `pushState`: this is not a
   * navigation — nothing about it should make Back mean "return to physarum" when
   * Back has meant "leave the app" all session. What it does buy is that a manual
   * reload, and every `switchTrack`/`syncUrlSeed` write after it, lands on the
   * substrate you are actually looking at.
   */
  const switchSim = async (id: string): Promise<void> => {
    // The GPU guard is not defensive dressing: with no device there is no panel,
    // so the only caller that can reach this is a scripted one.
    if (swapping || !gpu || id === sim.simId) return;
    if (!(SIMS as readonly string[]).includes(id)) {
      console.warn(`sim "${id}" not available; staying on "${sim.simId}"`);
      return;
    }
    swapping = true;
    let next: SimBundle | null = null;
    try {
      explorerExit();
      explorerRig?.close();
      explorerRig = null;
      explorerLog = null;

      next = buildBrowserSimBundle(id, sim.currentSeed);
      await next.sim.init(gpu);
      // Within the vizfx family the look is one preference, not one per visual:
      // exposure, the grade and the adapted gain all carry across the swap, so a
      // section-boundary auto-advance neither resets the sliders to the shipped
      // defaults nor spends its first seconds racing the exposure controller up
      // from 1× against an empty field. After `init` on purpose — init's own
      // resetAutoExposure runs first and the seed here overrides it.
      if (sim instanceof VizFxSim && next.sim instanceof VizFxSim) next.sim.adoptLook(sim);

      const outgoing = sim;
      const outgoingPanel = panel;
      ({ sim, impulses, modulator } = next);
      // Cleared at the commit point so the catch below cannot dispose the sim it
      // just installed. Everything after this line — `buildPanel`, the disposals,
      // `replaceState` — throws into a world where `next.sim` *is* the live sim,
      // and `next?.sim.dispose()` there would destroy the running substrate's
      // buffers and leave nothing to fall back to. Before it, `next` is the
      // half-built bundle nobody has seen and disposing it is exactly right.
      next = null;
      // Both before `buildPanel`, which reads them. The dwell origin is stamped
      // on *every* committed swap, hand-driven or automatic, so a manual preset
      // pick gets the same protection from the next boundary that an automatic
      // one does; the entry memory only moves when a visual actually ran, so a
      // failed `init` cannot redirect the mode selector to something unusable.
      lastSwapAt = performance.now();
      if (sim instanceof VizFxSim) milkdropEntry = sim.simId;
      panel = buildPanel();
      outgoingPanel?.dispose();
      outgoing.dispose();
      panel.refresh();

      const url = new URL(location.href);
      url.searchParams.set('sim', id);
      history.replaceState(null, '', url.toString());
    } catch (err) {
      // Two shapes of failure, one recovery. Before the commit (the common one —
      // `init` threw) the running app is exactly as it was, except for the picker,
      // which has already moved itself to the value that failed. After it, the new
      // sim is live and it is the *old* one that may not have been disposed. Both
      // are answered by rebuilding the panel against whatever `sim` is now, which
      // is the same rebuild the success path does and the thing that makes the
      // picker tell the truth again.
      console.error(`sim: could not switch to "${id}"; staying on "${sim.simId}"`, err);
      next?.sim.dispose();
      panel?.dispose();
      panel = buildPanel();
      panel.refresh();
    } finally {
      swapping = false;
      // A halted loop draws nothing, so a swap made while halted would leave the
      // previous substrate on screen. One frame, then back to spending nothing.
      needsRedraw = true;
    }
  };

  try {
    gpu = await initGpu(stage);
    await sim.init(gpu);
    panel = buildPanel();
    panel.refresh();
  } catch (err) {
    console.error(err);
    gpu = null;
    renderUnsupportedPage(fatalHost, err, () => {
      stage.style.display = 'none';
    });
  }

  // Dev-only handle: the workbench is driven by hand, but scripted checks
  // (snapshot/restore fidelity, mapping behaviour across sections) need a way in.
  if (import.meta.env.DEV) {
    (globalThis as unknown as Record<string, unknown>)['terrarium'] = {
      // Getters for the three a substrate swap replaces. Captured values would
      // hand a scripted check the disposed sim of whatever was running when the
      // page loaded, which is the kind of stale handle that makes a check pass
      // against nothing.
      get sim() {
        return sim;
      },
      get modulator() {
        return modulator;
      },
      get impulses() {
        return impulses;
      },
      sampler,
      clock,
      tuningLog,
      // The explorer is driven by clicking tiles, which is exactly the kind of
      // thing a scripted check cannot do; the host is the whole surface.
      explorer: explorerHost,
      switchSim,
      // The transition lane is otherwise unobservable from outside: it fires at
      // most once per section and its whole contract is about the boundaries it
      // *declines*. `lastBoundaryDecision` is how a scripted check tells "played
      // through a boundary and skipped because the repertoire is one" apart from
      // "the detector never ran". Writing `autoAdvance` here moves the flag, not
      // the checkbox — the panel re-derives it on the next rebuild.
      milkdrop: {
        get autoAdvance(): boolean {
          return autoAdvance;
        },
        set autoAdvance(on: boolean) {
          autoAdvance = on;
        },
        get repertoire(): readonly string[] {
          return VIZFX_IDS;
        },
        get lastBoundary(): { time: number; segment: number; action: string } | null {
          return lastBoundaryDecision;
        },
      },
    };
  }

  playButton.addEventListener('click', () => void clock.toggle());
  haltButton.addEventListener('click', () => setHalted(!halted));

  // --- view toggles: session-only visibility, CSS class on <body> -------------
  // Hiding is never a teardown: the panel and overlay keep running underneath
  // the class, so re-showing is instant and a `Space` still plays while the
  // bar is off-screen. Deliberately not persisted — a fresh load shows all.
  const togglePanel = (): void => {
    document.body.classList.toggle('panel-hidden');
  };
  const toggleTimeline = (): void => {
    document.body.classList.toggle('timeline-hidden');
    // `DebugOverlay.resize` sizes the backing store from `clientWidth`, which is
    // 0 while the strip is `display:none` — so it collapses to 1×1 for as long
    // as the strip is off. A running workbench repaints at full size on the
    // first frame back; a halted one draws only when asked, and would show that
    // 1×1 stretched across the strip until the next resize. One frame each way
    // is the whole fix.
    needsRedraw = true;
  };
  const toggleFullscreen = (): void => {
    // Not while exploring. The grid is driven by single clicks that re-lay it
    // out underneath the cursor, so a double there is already ambiguous — going
    // fullscreen on top of it is one surprise too many. Same rule the `s`/`t`
    // claims follow in `planKey`: the explorer is a locked-in GUI.
    if (explorerActive) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen();
  };

  const targetInfo = (ev: KeyboardEvent): TargetInfo => {
    const t = ev.target;
    return {
      tagName: t instanceof Element ? t.tagName : undefined,
      isContentEditable: t instanceof HTMLElement ? t.isContentEditable : false,
    };
  };

  // All key → action mapping lives in `planKey` (ui/keys.ts), which is unit
  // tested in isolation. The handler only adapts the two DOM facts `ev` carries
  // and runs the action. `preventDefault` is part of the plan so the test can
  // pin which keys are claimed; the editable-target guard is inside `planKey`.
  window.addEventListener('keydown', (ev) => {
    const plan = planKey(ev.code, ev.shiftKey, explorerActive, targetInfo(ev));
    if (plan.preventDefault) ev.preventDefault();
    switch (plan.kind) {
      case 'halt':
        // Live in both modes and whichever substrate is running: it is a statement
        // about the machine, not about the world.
        setHalted(!halted);
        return;
      case 'play-toggle':
        void clock.toggle();
        return;
      case 'seek':
        clock.seek(clock.time + plan.delta);
        return;
      case 'explorer-reroll':
        // Explorer's keys are claimed only while the mode is active, so 'r',
        // Backspace and Escape keep meaning nothing the rest of the time. The
        // transport keys above stay live in both modes — the music still drives
        // the nine tiles, so seeking to the chorus is exactly what you want to
        // do here.
        explorerReroll();
        return;
      case 'explorer-back':
        explorerBack();
        return;
      case 'explorer-exit':
        explorerExit();
        return;
      case 'toggle-panel':
        togglePanel();
        return;
      case 'toggle-timeline':
        toggleTimeline();
        return;
      case 'none':
        return;
      default:
        // `keys.ts` says a new binding is a branch there plus one here; this is
        // what makes that true. A `KeyPlan` kind with no case above stops being
        // assignable to `never` and fails the typecheck instead of silently
        // doing nothing.
        plan satisfies never;
        return;
    }
  });

  // A double-click on the terrarium (not the workbench, not a button) is the
  // fullscreen affordance: it goes where the eye is, and leaves every UI
  // surface the double-click might have hit alone.
  stage.addEventListener('dblclick', toggleFullscreen);
  overlayCanvas.addEventListener('dblclick', toggleFullscreen);

  const onResize = (): void => {
    gpu?.resize();
    overlay.resize();
    if (gpu) explorerRig?.layout(gpu.width, gpu.height);
    // A resize clears the canvas's drawing buffer, so a halted workbench would
    // go black and stay black. One frame puts the frozen world back.
    needsRedraw = true;
  };
  window.addEventListener('resize', onResize);
  // `resize` does not necessarily fire when the viewport changes by entering or
  // leaving fullscreen, so the canvases get the same one-frame refresh explicitly.
  document.addEventListener('fullscreenchange', onResize);

  // Explorer routing. Both listeners are registered once and guard on the mode
  // rather than being attached and detached around it — one fewer thing that can
  // survive an exit. `offsetX/Y` is CSS px relative to the canvas, which is what
  // `tileAt` documents itself as taking.
  stage.addEventListener('pointermove', (ev) => {
    if (!explorerActive || !explorerRig) return;
    explorerRig.setHover(explorerRig.tileAt(ev.offsetX, ev.offsetY));
  });
  stage.addEventListener('pointerleave', () => {
    if (explorerActive) explorerRig?.setHover(null);
  });
  stage.addEventListener('click', (ev) => {
    if (!explorerActive || !explorerRig) return;
    const grid = explorerRig.tileAt(ev.offsetX, ev.offsetY);
    if (grid === null) return;
    // The centre is the incumbent, so clicking it would mean "promote the thing
    // that is already promoted". Deliberately inert rather than a hidden reroll.
    const candidate = ExplorerRig.candidateAt(grid);
    if (candidate !== null) explorerPick(candidate);
  });

  const gpuState = gpu ? 'webgpu ok' : 'webgpu unavailable';

  // The transport only pumps ticks while audio plays. Phase 4 is driven by sliders,
  // so when the transport is idle the sim free-runs on the same fixed timestep.
  // Idle free-running is wall-clock paced, so how far the world drifts while paused
  // is not reproducible; only an uninterrupted run from a pinned seed is.
  //
  // stepIndex is the PCG key and counts sim steps, not transport position: it never
  // rewinds, so a seek backwards or a spell of free-running cannot re-issue hash keys
  // the run has already consumed.
  let stepIndex = 0;
  let freeAccum = 0;
  let lastNow = performance.now();
  let frameCount = 0;

  /**
   * The full stop, as distinct from `run ▸ paused` in the panel.
   *
   * `paused` is a *substrate* control: it zeroes the substep count and leaves
   * the render chain running, which is what makes it useful for tuning the look
   * of a still field. It is not a way to stop spending anything — the fade pass,
   * the whole particle draw, the bloom pyramid and the grade all still run every
   * rAF frame, which is most of the per-frame cost. Halting is the other
   * control: no ticks, no GPU work, no presented frame.
   *
   * It exists for one concrete situation: an export rendering on this machine
   * should not be competing with a workbench nobody is looking at.
   *
   * The rAF loop keeps running while halted rather than being cancelled. A
   * halted frame costs one callback and an early return, un-halting is
   * immediate, and there is no "who restarts the loop" question to get wrong on
   * a sim switch or a track change.
   *
   * The canvas keeps its last presented image for free as long as nothing calls
   * `getCurrentTexture()`, so a halted workbench still shows the world it froze
   * on. What that does NOT survive is a resize, which clears the drawing buffer
   * — hence `needsRedraw`, which buys exactly one more frame.
   */
  let halted = false;
  let needsRedraw = false;

  const setHalted = (next: boolean): void => {
    if (next === halted) return;
    halted = next;
    // Halting stops the world, and a transport that kept running behind a frozen
    // screen would be one: the features would advance, the modulator and the
    // impulse envelopes would not, and resuming would land the sim somewhere the
    // music had moved on from. Resuming deliberately does not restart playback —
    // "unhalt" is not "play", and which one you wanted is not guessable.
    if (halted && clock.isPlaying) clock.pause();
    haltButton.textContent = halted ? 'resume' : 'halt';
    // Both of these are rebuilt every frame by the loop, which is exactly what a
    // halted loop is not doing — so the state that halting itself caused has to
    // be published here or the chrome would keep describing the frame before it.
    // Resuming leaves them alone: the next frame rewrites both.
    if (halted) {
      playButton.textContent = 'play';
      statusEl.textContent = `${statusEl.textContent} · HALTED`;
    }
  };

  /**
   * The frame-rate EMA, held as a frame *time* in ms because that is the quantity
   * that is actually being averaged: averaging the reciprocal would weight a
   * single 200 ms hitch as one sample of "5 fps" instead of as the 12 frames'
   * worth of time it really cost, and the governor would under-react to exactly
   * the stalls it exists to catch.
   *
   * Seeded at 60 Hz so the first half-second reads as a plausible number rather
   * than as a spike, and the coefficient is derived from the measured dt on every
   * frame (`1 - exp(-dt/τ)`) so the smoothing means the same wall-clock duration
   * on a 60 Hz panel and a 240 Hz one.
   */
  let frameMs = 1000 / 60;
  /** the latched integer the status line prints, and when it was latched */
  let fpsShown = 0;
  let fpsShownAt = 0;

  /**
   * One fixed step. `stepIndex` has already been incremented by the caller — it
   * is the PCG key and counts sim steps, not transport position.
   *
   * The modulation layer runs on the same fixed timestep as the sim, ahead of
   * it: ẑ → projections → tanh → slew → config, then the sim reads that config
   * for its step. Impulses run after the modulator and before the sim: they are
   * a separate lane applied on top of whatever the slew limiter just wrote,
   * never through it, so a transient is not smoothed into a ramp.
   *
   * In explorer mode the whole lane is diverted to the rig. The live sim is not
   * ticked, not modulated and not impulsed — it freezes where it stands, so exit
   * resumes an accumulated world rather than restarting one. Its *config* is very
   * much live throughout (picks write it, sliders write it), which is a different
   * thing from its motion and the reason the two can be separated at all. The
   * modulator and the impulse engine are skipped for a second reason on top of
   * that: both are stateful smoothers, and running them against a sim nobody is
   * stepping would leave them describing a world that never happened.
   */
  /**
   * Milkdrop's transition engine: watch the transport cross a section boundary
   * and hand the next preset to `switchSim`.
   *
   * ## Crossed, not merely arrived at
   *
   * The detector's whole job is to tell "the song moved from the verse into the
   * chorus" apart from "somebody dropped the playhead into the chorus", and it
   * does it exactly the way `EventCursor` and the Modulator already do: one tick
   * forward is playback, the same tick again is an idle transport, and anything
   * else — a restart, an arrow key, an overlay scrub, an A/B restore — is a
   * jump. A jump **relocates** `lastSegment` without firing, for the reason the
   * cursor gives about walking a gap: a seek is a statement about where you want
   * to be, not a claim to have travelled through everything in between, and a
   * scrub across four sections would otherwise be four swaps in four frames.
   * The first sight of a segment is a relocation too — there is no boundary
   * behind the first frame of the session.
   *
   * ## Tracking is unconditional; only the response is gated
   *
   * Same reasoning as `Modulator.trackSegment`'s: short-circuiting on the toggle
   * would let `lastSegment` go stale while the toggle was off, so turning it back
   * on mid-track would read the next tick as a crossing and swap the substrate at
   * an arbitrary point in the song. Everything that can say "not now" therefore
   * says it *after* the state has been advanced — and is recorded, because from
   * the outside "no swap happened" looks identical whether the chain ran and
   * declined or never fired at all.
   *
   * The gates, in order after the crossing itself: milkdrop must be the running
   * mode (a `VizFxSim` — physarum and plife have no repertoire to advance
   * through), the toggle must be on, explorer mode must be closed (a swap exits
   * it and drops the rig, which would end a climb the human is in the middle of),
   * the transport must be *playing* (a free-running idle sim is not crossing
   * anything the music did), no swap may be in flight, and the current preset
   * must have had its dwell.
   *
   * A boundary that lands mid-swap is **dropped, not queued**. `switchSim`'s own
   * guard would refuse it anyway, but a boundary is a moment: a transition that
   * fired 400 ms late because an `init` was still awaiting would be a transition
   * in the wrong place, which is worse than the one that did not happen.
   *
   * At a repertoire of one — today — the last gate always wins and this is inert
   * by construction rather than by being switched off.
   */
  const noteSectionBoundary = (features: FeaturesFrame): void => {
    const gap = features.tick - lastBoundaryTick;
    const continuous = lastBoundaryTick >= 0 && gap >= 0 && gap <= MAX_CONTINUOUS_TICK_GAP;
    lastBoundaryTick = features.tick;

    const seg = sampler.segmentIndexAt(features.time);
    // Outside the segment map, or still in the one we were in: nothing happened.
    if (seg < 0 || seg === lastSegment) return;
    const first = lastSegment === -2;
    lastSegment = seg;

    const decide = (action: string): void => {
      lastBoundaryDecision = { time: features.time, segment: seg, action };
    };
    if (first || !continuous) {
      decide(first ? 'first' : 'seek');
      return;
    }
    if (!(sim instanceof VizFxSim)) {
      decide('not-milkdrop');
      return;
    }
    if (!autoAdvance) {
      decide('off');
      return;
    }
    if (explorerActive) {
      decide('exploring');
      return;
    }
    if (!clock.isPlaying) {
      decide('idle');
      return;
    }
    if (swapping) {
      decide('swapping');
      return;
    }
    const dwell = performance.now() - lastSwapAt;
    if (dwell < AUTO_ADVANCE_DWELL_MS) {
      decide('dwell');
      console.info(
        `milkdrop: section ${seg} at ${features.time.toFixed(1)}s — held, ` +
          `${(dwell / 1000).toFixed(1)}s into a ${AUTO_ADVANCE_DWELL_MS / 1000}s dwell`,
      );
      return;
    }
    const next = nextVisualAfter(sim.simId);
    if (next === sim.simId) {
      // One visual in the repertoire. Logged rather than silent: this is the
      // only externally visible sign that the lane is armed and working.
      decide('single');
      console.info(
        `milkdrop: section ${seg} at ${features.time.toFixed(1)}s — ` +
          `"${sim.simId}" is the whole repertoire, staying`,
      );
      return;
    }
    decide('advance');
    console.info(`milkdrop: section ${seg} at ${features.time.toFixed(1)}s → "${next}"`);
    void switchSim(next);
  };

  const advance = (tick: number, features: FeaturesFrame): void => {
    // Ahead of the explorer branch, because segment tracking is unconditional
    // even though the response is not — see the header. The mode's own gate is
    // inside.
    noteSectionBoundary(features);
    if (explorerActive) {
      explorerRig?.tick(features, stepIndex);
      return;
    }
    modulator.update(features, SECONDS_PER_TICK);
    impulses.update(tick, SECONDS_PER_TICK);
    sim.tick(features, stepIndex);
  };

  /**
   * Everything that touches the GPU and the overlay, in one place so the halted
   * path can buy a single frame without duplicating it.
   */
  const draw = (renderFrame: RenderFrame): void => {
    if (gpu) {
      gpu.resize();
      const encoder = gpu.device.createCommandEncoder();
      const view = gpu.gpuCanvasContext.getCurrentTexture().createView();
      if (explorerActive && explorerRig) {
        // Cheaper than a resize listener and strictly more complete: this also
        // catches a DPR change and a CSS-driven resize, and it is a no-op when
        // the canvas has not moved.
        explorerRig.layout(gpu.width, gpu.height);
        explorerRig.render(encoder, view, renderFrame);
      } else {
        sim.render(encoder, view, renderFrame);
      }
      gpu.device.queue.submit([encoder.finish()]);
    }

    // The scrub strip is hidden while exploring (it would cover the bottom row),
    // so there is nothing to draw into.
    if (!explorerActive) {
      overlay.draw(clock.tickTime, {
        tick: clock.simTick,
        playing: clock.isPlaying,
        seed: sim.currentSeed,
      });
    }
  };

  const frame = (now: DOMHighResTimeStamp): void => {
    const wallDelta = Math.min((now - lastNow) / 1000, 0.25);
    lastNow = now;
    // Render consumers share one host-owned clock sample. The first-frame value
    // preserves their historical 60 Hz seed; subsequent frames retain measured,
    // variable-rate rAF timing (including the existing 0.25 s background clamp).
    const renderFrame: RenderFrame = {
      frameIndex: frameCount,
      timeSeconds: now / 1000,
      deltaSeconds: frameCount === 0 ? 1 / 60 : wallDelta,
    };

    if (halted) {
      // Ahead of every other lane, and it returns: no ticks, no modulation, no
      // governor sample (an empty loop measures the display's refresh rate, and
      // a budget grown against a frame nobody rendered is a lie), no panel
      // refresh, no status rebuild. `lastNow` is already updated above, so the
      // first frame back is a normal one rather than a long one.
      if (needsRedraw) {
        needsRedraw = false;
        draw(renderFrame);
      }
      requestAnimationFrame(frame);
      return;
    }

    // The frame-rate lane. Same clamped `wallDelta` the free-run pump uses, so a
    // backgrounded tab's two-second gap is one 0.25 s sample rather than a
    // measurement that would read as 0.5 fps and floor the governor on the first
    // frame back.
    const dtMs = wallDelta * 1000;
    frameMs += (dtMs - frameMs) * (1 - Math.exp(-dtMs / FPS_EMA_TAU_MS));
    const fps = 1000 / Math.max(frameMs, 1e-3);
    if (now - fpsShownAt >= FPS_DISPLAY_MS) {
      fpsShown = Math.round(fps);
      fpsShownAt = now;
    }
    // Not while exploring: the number on screen then describes nine tiles, and
    // the live sim's governor must not size a world nobody is stepping against a
    // frame time nobody is spending on it. The tiles have no governor at all
    // (`forceNoGovernor`), so the whole mode is governor-free by construction.
    if (sim instanceof PlifeSim && !explorerActive) sim.noteFrameRate(fps, now);

    clock.pump((tick) => {
      stepIndex++;
      advance(tick, sampler.sampleAt(tick));
    });

    if (!clock.isPlaying) {
      freeAccum += wallDelta;
      let ran = 0;
      while (freeAccum >= SECONDS_PER_TICK && ran < MAX_FREE_TICKS_PER_FRAME) {
        freeAccum -= SECONDS_PER_TICK;
        stepIndex++;
        // clock.simTick does not move while idle, so no timeline event re-fires;
        // envelopes still decay, which is what makes test-fire work while paused.
        advance(clock.simTick, sampler.sampleAt(clock.simTick));
        ran++;
      }
      if (freeAccum > SECONDS_PER_TICK * MAX_FREE_TICKS_PER_FRAME) freeAccum = 0;
    } else {
      freeAccum = 0;
    }

    draw(renderFrame);

    if (frameCount % PANEL_REFRESH_FRAMES === 0) {
      // The panel's own cadence, shared by the two things that keep the grid and
      // the workbench describing the same world. Both are cheap (a θ read and
      // nine value-copies) but neither is free, and neither needs to be more
      // responsive than the panel that drives it.
      if (explorerActive) {
        // Style first: a recentre regenerates the tiles and syncs them itself, so
        // doing it the other way round would style nine tiles that are about to
        // be replaced.
        explorerRig?.syncStyle(sim);
        explorerPollEdits(now);
      }
      // No modulator hook alongside this one, deliberately. The modulator has the
      // same "re-take the reference after the panel has quantised its bindings"
      // requirement the explorer solves with `explorerMarkApplied`, but there are
      // 26 `pane.refresh()` call sites across the panels and the workbench and
      // only one of them is here — so `ui/mod-fill.ts` installs the pairing on the
      // pane itself, and this call inherits it like every other.
      panel?.refresh();
    }
    frameCount++;

    playButton.textContent = clock.isPlaying ? 'pause' : 'play';
    // The middle segment is the sim's own (`Sim.status`); everything around it is
    // the same whichever substrate is running.
    // While exploring, the sim segment describes a frozen world nobody is
    // looking at, so the explorer segment replaces it rather than joining it —
    // what is on screen is nine other worlds and their search state.
    // The modulation mode is named in both branches, unlike everything else here:
    // entering the mode forces it to manual and leaving does not put it back, so
    // it is a state change the human did not ask for and has to be able to see.
    // The fps segment goes last, in both variants and therefore in both sims: it
    // is the one number here that is about the machine rather than about the
    // world, so it sits at the end where the eye can find it without reading the
    // rest — and it is the segment you watch while dragging the pair-search mode
    // or the particle cap.
    // "milkdrop · nebula" rather than "nebula": the picker calls the family
    // milkdrop and the status line has to agree with the control that got you
    // here. Assembled here and not inside the sim — the grouping is main's
    // presentation decision, and `Sim.name` stays the visual's own id, which is
    // what `?sim=` carries and what the autosave slot is named after.
    const simLabel = sim instanceof VizFxSim ? `${MILKDROP_MODE} · ${sim.name}` : sim.name;
    statusEl.textContent = explorerActive
      ? `${track} · ${clock.sourceKind} audio · sim "${simLabel}" · ` +
        `explorer gen ${explorerSet?.generation ?? 0} · ${explorerSubspace} · ` +
        `step ${explorerStep.toFixed(2)} · ${explorerLog?.size ?? 0} logged · ` +
        `${modulator.mode} · ${gpuState} · ${fpsShown} fps`
      : `${track} · ${clock.sourceKind} audio · sim "${simLabel}" ` +
        `${sim.status()} · ` +
        `${modulator.mode}${
          modulator.mode === 'modulated' ? ` ${modulator.sourceLabel}` : ''
        } · ${gpuState} · ${fpsShown} fps`;

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err: unknown) => {
  console.error(err);
  renderUnsupportedPage(fatalHost, err);
});
