import './style.css';
import { AudioClock } from './audio/clock';
import { DebugOverlay } from './debug/overlay';
import { initGpu, renderUnsupportedPage, type GpuContext } from './gpu/context';
import { buildDriverBank, Modulator } from './mapping/modulation';
import {
  defaultModulationConfig,
  loadModulationLocal,
  modulationFits,
} from './mapping/persist';
import { TuningLog } from './mapping/tuninglog';
import { ImpulseEngine } from './sim/impulses';
import { defaultConfig } from './sim/physarum/config';
import { PhysarumSim } from './sim/physarum/physarum';
import { defaultPlifeConfig } from './sim/plife/config';
import { PlifeSim } from './sim/plife/plife';
import { resolveSeed } from './sim/seed';
import { loadTimeline } from './timeline/loader';
import { TimelineSampler } from './timeline/sampler';
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
 * `?sim=` picks the simulation, the same way `?track=` picks the timeline. It is
 * also the persistence key (`ModTarget.simId`), so an unknown value has to be
 * refused rather than passed through — otherwise `?sim=typo` would quietly start
 * physarum against an empty autosave slot named after the typo.
 */
function requestedSim(): string {
  const param = new URLSearchParams(location.search).get('sim');
  return param !== null && /^[a-z-]+$/.test(param) ? param : DEFAULT_SIM;
}

async function main(): Promise<void> {
  const { seed, pinned } = resolveSeed();

  const timelineUrl = (id: string): string => `${import.meta.env.BASE_URL}timelines/${id}`;
  let track = requestedTrack();
  let timeline;
  try {
    timeline = await loadTimeline(timelineUrl(track));
  } catch (err) {
    // A missing or broken timeline must not dead-end the app: the synthetic one
    // ships with the repo and always loads.
    if (track === FALLBACK_TRACK) throw err;
    console.warn(`timeline "${track}" failed to load; falling back to "${FALLBACK_TRACK}"`, err);
    track = FALLBACK_TRACK;
    timeline = await loadTimeline(timelineUrl(track));
  }
  const sampler = new TimelineSampler(timeline, SECONDS_PER_TICK);

  const clock = new AudioClock(
    {
      duration: timeline.manifest.track.duration,
      beats: timeline.manifest.beats,
      downbeats: timeline.manifest.downbeats,
    },
    { secondsPerTick: SECONDS_PER_TICK, audioUrl: `${timelineUrl(track)}/audio.wav` },
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
  // The sim has not been initialised yet (that happens below, and fires
  // onSeedChange), but in the WebGPU-unavailable path it never will — so stamp
  // the personality on now rather than depending on a callback that may not run.
  modulator.applyBase();

  let gpu: GpuContext | null = null;
  let panel: PanelHandle | null = null;
  try {
    gpu = await initGpu(stage);
    await sim.init(gpu);
    // A panel binds tweakpane widgets to its sim's config fields by name, so each
    // substrate has its own. Everything *around* the widgets — the workbench, the
    // impulse folder, the HDR chain — is shared, so the two calls differ only in
    // which factory they name and take identical options.
    const panelOpts = {
      pinned,
      // Both restart buttons land here, after sim.reseed() — which has already
      // re-keyed the hotspots via onSeedChange. What is left is dropping any
      // envelope still ringing and putting the transport back to the top.
      onRestart: (): void => {
        impulses.reset();
        clock.seek(0);
      },
      impulses,
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
    };
  }

  playButton.addEventListener('click', () => void clock.toggle());
  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement) return;
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

  const frame = (): void => {
    const now = performance.now();
    const wallDelta = Math.min((now - lastNow) / 1000, 0.25);
    lastNow = now;

    // The modulation layer runs on the same fixed timestep as the sim, ahead of
    // it: ẑ → projections → tanh → slew → config, then the sim reads that config
    // for its step. Impulses run after the modulator and before the sim: they are
    // a separate lane applied on top of whatever the slew limiter just wrote,
    // never through it, so a transient is not smoothed into a ramp.
    clock.pump((tick) => {
      stepIndex++;
      const features = sampler.sampleAt(tick);
      modulator.update(features, SECONDS_PER_TICK);
      impulses.update(tick, SECONDS_PER_TICK);
      sim.tick(features, stepIndex);
    });

    if (!clock.isPlaying) {
      freeAccum += wallDelta;
      let ran = 0;
      while (freeAccum >= SECONDS_PER_TICK && ran < MAX_FREE_TICKS_PER_FRAME) {
        freeAccum -= SECONDS_PER_TICK;
        stepIndex++;
        const features = sampler.sampleAt(clock.simTick);
        modulator.update(features, SECONDS_PER_TICK);
        // clock.simTick does not move while idle, so no timeline event re-fires;
        // envelopes still decay, which is what makes test-fire work while paused.
        impulses.update(clock.simTick, SECONDS_PER_TICK);
        sim.tick(features, stepIndex);
        ran++;
      }
      if (freeAccum > SECONDS_PER_TICK * MAX_FREE_TICKS_PER_FRAME) freeAccum = 0;
    } else {
      freeAccum = 0;
    }

    if (gpu) {
      gpu.resize();
      const encoder = gpu.device.createCommandEncoder();
      sim.render(encoder, gpu.gpuCanvasContext.getCurrentTexture().createView());
      gpu.device.queue.submit([encoder.finish()]);
    }

    overlay.draw(clock.tickTime, {
      tick: clock.simTick,
      playing: clock.isPlaying,
      seed: sim.currentSeed,
    });

    if (panel && frameCount % PANEL_REFRESH_FRAMES === 0) panel.refresh();
    frameCount++;

    playButton.textContent = clock.isPlaying ? 'pause' : 'play';
    // The middle segment is the sim's own (`Sim.status`); everything around it is
    // the same whichever substrate is running.
    statusEl.textContent =
      `${track} · ${clock.sourceKind} audio · sim "${sim.name}" ` +
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
