import './style.css';
import { AudioClock } from './audio/clock';
import { DebugOverlay } from './debug/overlay';
import { ExplorerLog, type ExplorerAction } from './explore/log';
import { ExplorerRig, TILE_COUNT } from './explore/rig';
import { ExplorerSearch, type CandidateSet, type ExplorerSubspace } from './explore/search';
import { initGpu, renderUnsupportedPage, type GpuContext } from './gpu/context';
import { buildDriverBank, Modulator } from './mapping/modulation';
import {
  defaultModulationConfig,
  downloadText,
  loadModulationLocal,
  modulationFits,
} from './mapping/persist';
import type { ModTarget } from './mapping/target';
import { TuningLog } from './mapping/tuninglog';
import { ImpulseEngine } from './sim/impulses';
import { defaultConfig } from './sim/physarum/config';
import { PhysarumSim } from './sim/physarum/physarum';
import { defaultPlifeConfig } from './sim/plife/config';
import { PlifeSim } from './sim/plife/plife';
import { resolveSeed } from './sim/seed';
import type { Sim } from './sim/types';
import { invalidateIfStale, rememberCachedTrack } from './timeline/cache';
import { buildCatalog, fetcherFor, type TrackEntry } from './timeline/catalog';
import { loadTimeline } from './timeline/loader';
import { TimelineSampler, type FeaturesFrame } from './timeline/sampler';
import type { ExplorerPanelHost } from './ui/explore-panel';
import { createPanel, type PanelHandle } from './ui/panel';
import { createPlifePanel } from './ui/plife-panel';

const DEFAULT_TRACK = 'free-fall';
const FALLBACK_TRACK = 'synthetic';
const DEFAULT_SIM = 'physarum';
/** Every `?sim=` value that resolves to a real substrate. */
const SIMS = ['physarum', 'plife'] as const;
const SECONDS_PER_TICK = 1 / 60;
const MAX_FREE_TICKS_PER_FRAME = 4;
const PANEL_REFRESH_FRAMES = 30;

/**
 * Where explorer mode's mutation step starts. Mid-range on purpose: at 0.05 the
 * eight candidates are indistinguishable from the centre and there is nothing to
 * choose between, and at 1.0 the first generation has already left the
 * neighbourhood you liked. 0.3 is a visible difference you can still attribute.
 */
const EXPLORER_STEP_DEFAULT = 0.3;

/**
 * Fixed steps per *plife* tile tick in explorer mode. 2, i.e. half rate.
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

const stage = document.getElementById('stage') as HTMLCanvasElement;
const overlayCanvas = document.getElementById('overlay') as HTMLCanvasElement;
const playButton = document.getElementById('play') as HTMLButtonElement;
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
 */
function requestedSim(): string {
  const param = new URLSearchParams(location.search).get('sim');
  return param !== null && /^[a-z-]+$/.test(param) ? param : DEFAULT_SIM;
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
  const { seed, pinned } = resolveSeed();

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
    source: 'bundled',
  };
  const pick = (id: string): TrackEntry =>
    catalog.tracks.find((t) => t.id === id) ?? {
      ...fallbackEntry,
      id,
      title: id,
      base: `${import.meta.env.BASE_URL}timelines/${id}`,
    };

  let entry = pick(requestedTrack());
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

  const clock = new AudioClock(
    {
      duration: timeline.manifest.track.duration,
      beats: timeline.manifest.beats,
      downbeats: timeline.manifest.downbeats,
    },
    {
      secondsPerTick: SECONDS_PER_TICK,
      audioUrl: `${entry.base}/audio.wav`,
      fetcher: fetcherFor(entry),
    },
  );
  // Fetch ahead of the first gesture; the click-track fallback costs nothing if
  // this 404s.
  clock.preload();

  const overlay = new DebugOverlay(overlayCanvas, sampler, {
    onSeek: (t) => clock.seek(t),
  });

  // Resolve which sim to build before building it, so a second substrate slots
  // in here rather than being threaded through the twenty call sites below —
  // those all talk to `ModTarget` now and do not care which one they got.
  const wantedSim = requestedSim();
  if (!(SIMS as readonly string[]).includes(wantedSim)) {
    console.warn(`sim "${wantedSim}" not available; using ${DEFAULT_SIM}`);
  }
  // Typed as the union rather than as `Sim`: everything below this line talks to
  // either `Sim` or `ModTarget`, both of which both substrates satisfy, and the
  // union is only needed for the two physarum-specific call sites (the stems
  // channel and the parameter panel) that are guarded by an instanceof.
  const sim: PhysarumSim | PlifeSim =
    wantedSim === 'plife'
      ? new PlifeSim(seed, defaultPlifeConfig())
      : new PhysarumSim(seed, defaultConfig(4));
  // Both substrates read the stems channel directly now, for different things:
  // physarum drives deposit with it, plife drives population. Neither is the
  // Modulator's stem-follow lane, which owns brightness and is wired separately.
  sim.setStemChannel(sampler.getChannel('stems'));
  if (sim instanceof PlifeSim) {
    // Novelty → accent population. Deliberately a direct wire rather than a
    // seeded projection: the projections give the accents character, this is
    // what makes "the chorus arrived" legible on every seed.
    sim.setAccentChannels(sampler.getChannel('novelty16'), sampler.getChannel('actChorus'));
  }
  const simId = sim.simId;

  // The impulse lane. It exists whether or not the timeline carries events: with
  // none, every multiplier stays at 1 and the workbench's test-fire buttons are
  // still the way to tune the responses.
  const impulses = new ImpulseEngine(
    seed,
    sim.config.speciesCount,
    sampler.events,
    SECONDS_PER_TICK,
  );
  sim.setImpulses(impulses.state);
  if (sampler.events.length === 0) {
    console.info(`timeline "${track}" has no events array; impulses idle until test-fired`);
  }

  // The modulation input: ~16 named drivers built from this timeline's own
  // structure channels and its 64-dim latent channel, variance-reordered and
  // z-scored once, here (plan.md Revision 4). No sidecar, no background upgrade.
  const drivers = buildDriverBank(timeline);
  const stored = loadModulationLocal(simId);
  const modConfig =
    stored && modulationFits(stored, sim.config.speciesCount, simId)
      ? stored
      : defaultModulationConfig(sim.config, simId);
  const modulator = new Modulator(
    sim,
    sampler,
    drivers,
    modConfig,
    seed,
  );
  // Before `setMode`, and that order is now load-bearing. Turning modulation on
  // adopts whatever θ the config currently holds as the base of every modulated
  // slot (the VST rule: modulation breathes around the value you own), so the
  // seeded personality has to be *in* the config by then or the first act of the
  // app would be to adopt the shipped defaults and throw the seed's personality
  // away. Stamping it here rather than waiting for `sim.init`'s onSeedChange also
  // covers the WebGPU-unavailable path, where that callback never runs.
  modulator.applyBase();
  modulator.setMode(modConfig.enabled && modulator.available ? 'modulated' : 'manual');
  console.info(
    `modulation: ${modulator.sourceLabel} → ${modulator.modulatedCount} parameters, ` +
      `seed ${seed}${pinned ? ' (pinned)' : ''}`,
  );
  const tuningLog = new TuningLog();

  // One seed, three consumers. Hotspot placement, projection wiring and the
  // seeded personality all re-key from here, so a reseed, a reroll and a snapshot
  // restore each move all three together without any caller remembering to.
  sim.onSeedChange = (s) => {
    impulses.setSeed(s);
    modulator.setSeed(s);
  };

  let gpu: GpuContext | null = null;
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
    if (sim instanceof PlifeSim) {
      const tile = new PlifeSim(sim.currentSeed, structuredClone(sim.config));
      // Nine tiles never run the brute pair search, whatever the live sim is
      // set to. Nine times an O(N²) force pass measured at ~212 ms/frame with
      // 13 k alive — the mode works in the grid, it is just not a search tool at
      // 4.7 fps. See `PlifeSim.forceGridSearch` for why this is a flag and not a
      // value on the cloned config (`syncStyle` would overwrite the latter).
      tile.forceGridSearch = true;
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
   * **plife promotes, physarum does not**, and the asymmetry is a decision:
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

  try {
    gpu = await initGpu(stage);
    await sim.init(gpu);
    // A panel binds tweakpane widgets to its sim's config fields by name, so each
    // substrate has its own. Everything *around* the widgets — the workbench, the
    // impulse folder, the HDR chain — is shared, so the two calls differ only in
    // which factory they name and take identical options.
    const panelOpts = {
      pinned,
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
      workbench: {
        sim,
        modulator,
        log: tuningLog,
        trackId: timeline.manifest.track.id,
        time: () => clock.time,
        tick: () => clock.simTick,
        seek: (t: number) => clock.seek(t),
      },
    };
    panel =
      sim instanceof PhysarumSim ? createPanel(sim, panelOpts) : createPlifePanel(sim, panelOpts);
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
      sim,
      modulator,
      sampler,
      clock,
      tuningLog,
      impulses,
      // The explorer is driven by clicking tiles, which is exactly the kind of
      // thing a scripted check cannot do; the host is the whole surface.
      explorer: explorerHost,
    };
  }

  playButton.addEventListener('click', () => void clock.toggle());
  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement) return;
    // Explorer's keys are claimed only while the mode is active, so 'r',
    // Backspace and Escape keep meaning nothing the rest of the time. The
    // transport keys below stay live in both modes — the music still drives the
    // nine tiles, so seeking to the chorus is exactly what you want to do here.
    if (explorerActive) {
      if (ev.code === 'KeyR') {
        ev.preventDefault();
        explorerReroll();
        return;
      }
      if (ev.code === 'Backspace') {
        ev.preventDefault();
        explorerBack();
        return;
      }
      if (ev.code === 'Escape') {
        ev.preventDefault();
        explorerExit();
        return;
      }
    }
    if (ev.code === 'Space') {
      ev.preventDefault();
      void clock.toggle();
    } else if (ev.code === 'ArrowLeft') {
      clock.seek(clock.time - (ev.shiftKey ? 10 : 2));
    } else if (ev.code === 'ArrowRight') {
      clock.seek(clock.time + (ev.shiftKey ? 10 : 2));
    }
  });
  window.addEventListener('resize', () => {
    gpu?.resize();
    overlay.resize();
    if (gpu) explorerRig?.layout(gpu.width, gpu.height);
  });

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
  const advance = (tick: number, features: FeaturesFrame): void => {
    if (explorerActive) {
      explorerRig?.tick(features, stepIndex);
      return;
    }
    modulator.update(features, SECONDS_PER_TICK);
    impulses.update(tick, SECONDS_PER_TICK);
    sim.tick(features, stepIndex);
  };

  const frame = (): void => {
    const now = performance.now();
    const wallDelta = Math.min((now - lastNow) / 1000, 0.25);
    lastNow = now;

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

    if (gpu) {
      gpu.resize();
      const encoder = gpu.device.createCommandEncoder();
      const view = gpu.gpuCanvasContext.getCurrentTexture().createView();
      if (explorerActive && explorerRig) {
        // Cheaper than a resize listener and strictly more complete: this also
        // catches a DPR change and a CSS-driven resize, and it is a no-op when
        // the canvas has not moved.
        explorerRig.layout(gpu.width, gpu.height);
        explorerRig.render(encoder, view);
      } else {
        sim.render(encoder, view);
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
    statusEl.textContent = explorerActive
      ? `${track} · ${clock.sourceKind} audio · sim "${sim.name}" · ` +
        `explorer gen ${explorerSet?.generation ?? 0} · ${explorerSubspace} · ` +
        `step ${explorerStep.toFixed(2)} · ${explorerLog?.size ?? 0} logged · ` +
        `${modulator.mode} · ${gpuState}`
      : `${track} · ${clock.sourceKind} audio · sim "${sim.name}" ` +
        `${sim.status()} · ` +
        `${modulator.mode}${
          modulator.mode === 'modulated' ? ` ${modulator.sourceLabel}` : ''
        } · ${gpuState}`;

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err: unknown) => {
  console.error(err);
  renderUnsupportedPage(fatalHost, err);
});
