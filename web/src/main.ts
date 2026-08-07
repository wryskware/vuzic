import './style.css';
import { AudioClock } from './audio/clock';
import { DebugOverlay } from './debug/overlay';
import { initGpu, renderUnsupportedPage, type GpuContext } from './gpu/context';
import { fitMapping, latentMatrix, refitCenters } from './mapping/anchors';
import { Mapper } from './mapping/mapper';
import { loadMappingLocal, mappingFits, saveMappingLocal } from './mapping/persist';
import { TuningLog } from './mapping/tuninglog';
import { ImpulseEngine } from './sim/impulses';
import { defaultConfig } from './sim/physarum/config';
import { PhysarumSim } from './sim/physarum/physarum';
import { resolveSeed } from './sim/seed';
import type { Sim } from './sim/types';
import { loadTimeline } from './timeline/loader';
import { TimelineSampler } from './timeline/sampler';
import { createPanel, type PanelHandle } from './ui/panel';

const DEFAULT_TRACK = 'free-fall';
const FALLBACK_TRACK = 'synthetic';
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

  const physarum = new PhysarumSim(seed, defaultConfig(4));
  physarum.setStemChannel(sampler.getChannel('stems'));
  const sim: Sim = physarum;

  // The impulse lane. It exists whether or not the timeline carries events: with
  // none, every multiplier stays at 1 and the workbench's test-fire buttons are
  // still the way to tune the responses.
  const impulses = new ImpulseEngine(
    seed,
    physarum.config.speciesCount,
    sampler.events,
    SECONDS_PER_TICK,
  );
  physarum.setImpulses(impulses.state);
  // Hotspot placement hashes the seed, and the engine keeps its own copy — so it
  // follows every seed change the sim makes, restores from a snapshot included.
  physarum.onSeedChange = (s) => impulses.setSeed(s);
  if (sampler.events.length === 0) {
    console.info(`timeline "${track}" has no events array; impulses idle until test-fired`);
  }

  // Anchors come from the track: k-means over its own latent channel, run once
  // here (a few ms) and identical on every load. A saved mapping wins if it was
  // authored for this K and this latent width; otherwise it cannot be applied
  // field-for-field and we refit rather than guess.
  const latent = latentMatrix(timeline);
  const fitDefault = (): ReturnType<typeof fitMapping> =>
    fitMapping(latent, { base: physarum.config });
  const stored = loadMappingLocal();
  const mappingConfig =
    stored && mappingFits(stored, physarum.config.speciesCount, latent?.dims ?? 0)
      ? stored
      : fitDefault();
  const mapper = new Mapper(physarum, sampler, mappingConfig);
  const tuningLog = new TuningLog();

  let gpu: GpuContext | null = null;
  let panel: PanelHandle | null = null;
  try {
    gpu = await initGpu(stage);
    await sim.init(gpu);
    panel = createPanel(physarum, {
      pinned,
      // Both restart buttons land here, after sim.reseed() — which has already
      // re-keyed the hotspots via onSeedChange. What is left is dropping any
      // envelope still ringing and putting the transport back to the top.
      onRestart: () => {
        impulses.reset();
        clock.seek(0);
      },
      impulses,
      workbench: {
        sim: physarum,
        mapper,
        log: tuningLog,
        trackId: timeline.manifest.track.id,
        time: () => clock.time,
        tick: () => clock.simTick,
        seek: (t) => clock.seek(t),
        refit: (k) => {
          mapper.config.kmeans.k = k;
          const next = refitCenters(mapper.config, latent, physarum.config);
          mapper.setConfig(next);
          saveMappingLocal(next);
          return next;
        },
        resetToFitted: () => {
          const next = fitDefault();
          mapper.setConfig(next);
          return next;
        },
      },
    });
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
      sim: physarum,
      mapper,
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

    // The mapping layer runs on the same fixed timestep as the sim, ahead of it:
    // z → simplex → slew → config, then the sim reads that config for its step.
    // Impulses run after the mapper and before the sim: they are a separate lane
    // applied on top of whatever the mapper's slew limiter just wrote, never
    // through it, so a transient is not smoothed into a ramp.
    clock.pump((tick) => {
      stepIndex++;
      const features = sampler.sampleAt(tick);
      mapper.update(features, SECONDS_PER_TICK);
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
        mapper.update(features, SECONDS_PER_TICK);
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
      seed: physarum.currentSeed,
    });

    if (panel && frameCount % PANEL_REFRESH_FRAMES === 0) panel.refresh();
    frameCount++;

    playButton.textContent = clock.isPlaying ? 'pause' : 'play';
    const st = physarum.stats();
    statusEl.textContent =
      `${track} · ${clock.sourceKind} audio · sim "${sim.name}" ` +
      `${st.gridW}×${st.gridH}×${physarum.config.speciesCount} · ` +
      `${st.aliveAgents.toLocaleString()} agents · seed ${physarum.currentSeed} · ` +
      `${mapper.solo !== null ? `solo ${mapper.solo}` : mapper.mode} · ${gpuState}`;

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err: unknown) => {
  console.error(err);
  renderUnsupportedPage(fatalHost, err);
});
