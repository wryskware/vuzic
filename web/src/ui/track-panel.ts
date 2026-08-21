import type { PanelContainer } from './panel';
import { DEMO_BUILD } from '../runtime/demo';
import {
  pollJob,
  uploadForAnalysis,
  type JobState,
  type TrackEntry,
} from '../timeline/catalog';

/** How often the upload progress readout asks the server where the job got to. */
const POLL_MS = 1500;

export interface TrackPanelHost {
  /** Everything available: bundled, server, and cached-from-server. */
  tracks: readonly TrackEntry[];
  /** The one currently playing. */
  current: string;
  /** Did a server answer the startup probe? Drives the upload button's presence. */
  serverUp: boolean;
  /** How many tracks it offered, before the merge folded them into the bundled ones. */
  serverCount: number;
  /** Switch tracks. See the note on `switchTo` below for why this reloads. */
  switchTo(id: string): void;
}

export interface TrackPanelHandle {
  refresh(): void;
}

function label(entry: TrackEntry): string {
  const m = Math.floor(entry.duration / 60);
  const s = Math.round(entry.duration % 60)
    .toString()
    .padStart(2, '0');
  // The source is in the label rather than in a separate column because it is
  // the one thing that predicts behaviour: a `server` track stops loading when
  // the server does, unless it has become a `cached` one by being played.
  const mark = entry.source === 'bundled' ? '' : ` · ${entry.source}`;
  return `${entry.title} (${m}:${s}${mark})`;
}

/**
 * The track picker, and the upload that feeds it.
 *
 * Mounted at the top of the play tab by both substrates' panels — a track is the
 * one thing on screen that belongs to neither the sim nor the mapping, and it is
 * the first decision you make, so it sits above the world seed.
 *
 * Switching **reloads the page** with a new `?track=`. The timeline is not a
 * value the app holds, it is threaded into the sampler, the clock, the driver
 * bank, the modulator, the impulse engine and both sims at construction time,
 * and every one of those is built from it exactly once. Rebuilding them live
 * would mean a second construction path for each, kept in step with the first
 * forever, to save a reload that costs under a second on localhost and restores
 * the autosaved mapping on the way back up. `?seed=` and `?sim=` are preserved,
 * so "same world, different song" is one click.
 */
export function createTrackFolder(
  container: PanelContainer,
  host: TrackPanelHost,
): TrackPanelHandle {
  const folder = container.addFolder({ title: 'track' });
  const state = {
    track: host.current,
    server: host.serverUp ? `up · ${host.serverCount} track(s)` : 'offline',
    job: 'idle',
  };

  const options = Object.fromEntries(host.tracks.map((t) => [label(t), t.id]));
  // A track that loaded from a URL the catalog does not know about (a bundled
  // directory added without re-running sync-data, say) still has to be
  // selectable, or the list would silently show the wrong current value.
  if (!host.tracks.some((t) => t.id === host.current)) options[`${host.current} (unlisted)`] = host.current;

  folder.addBinding(state, 'track', { label: 'playing', options }).on('change', (ev) => {
    if (ev.value !== host.current) host.switchTo(String(ev.value));
  });
  // The readout is for whoever is running a pipeline beside the app; in the
  // published cut it is a permanent "offline" against a server the build was
  // never going to talk to.
  if (!DEMO_BUILD) {
    folder.addBinding(state, 'server', { readonly: true, label: 'analysis server' });
  }

  // The upload half only exists when a server answered. With none there is
  // nothing to POST to, and a button that always fails is worse than no button.
  if (host.serverUp) {
    const jobMonitor = folder.addBinding(state, 'job', { readonly: true, label: 'upload' });

    // A real <input type="file"> rather than a tweakpane widget: tweakpane has no
    // file control, and the browser will not open a picker except from a user
    // gesture on an input, so the button forwards its click to a hidden one.
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.wav,.mp3,audio/wav,audio/mpeg';
    input.style.display = 'none';
    document.body.appendChild(input);

    const setJob = (text: string): void => {
      state.job = text;
      jobMonitor.refresh();
    };

    const watch = async (jobId: string, trackId: string): Promise<void> => {
      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        let job: JobState;
        try {
          job = await pollJob(jobId);
        } catch (err) {
          setJob('lost the server');
          console.warn('upload: polling failed', err);
          return;
        }
        if (job.status === 'error') {
          setJob(`failed: ${job.error}`.slice(0, 60));
          console.error(`analysis job ${jobId} failed`, job.error);
          return;
        }
        if (job.status === 'done') {
          setJob(`done — switching to "${trackId}"`);
          // Straight to it: the point of uploading a track is to see it, and the
          // reload is the same one the picker does.
          host.switchTo(trackId);
          return;
        }
        setJob(`${job.stage || job.status} ${Math.round(job.progress * 100)}%`);
      }
    };

    input.addEventListener('change', () => {
      const file = input.files?.[0];
      input.value = ''; // so re-picking the same file fires `change` again
      if (!file) return;
      setJob(`uploading ${file.name}`);
      void uploadForAnalysis(file)
        .then(({ jobId, trackId }) => {
          setJob(`queued as "${trackId}"`);
          return watch(jobId, trackId);
        })
        .catch((err: unknown) => {
          setJob('upload rejected');
          console.error('upload failed', err);
        });
    });

    folder.addButton({ title: 'analyze a wav/mp3…' }).on('click', () => input.click());
  }

  return {
    refresh(): void {
      state.track = host.current;
    },
  };
}
