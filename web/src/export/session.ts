import type { ExportRecipe } from '../runtime/recipe.ts';
import type { ExportCapabilities, ExportJob, LocalExportClient } from './client.ts';

/**
 * Where the one export slot currently is.
 *
 * `submitting` is the browser-side gap between the click and the server's first
 * job response; every other phase mirrors an `ExportJobStatus`, so a panel never
 * has to know whether a value came from the POST or from a poll.
 */
export type ExportPhase =
  | 'idle'
  | 'submitting'
  | 'queued'
  | 'running'
  | 'done'
  | 'error'
  | 'cancelled';

export interface ExportDownload {
  /** absolute, already resolved through the client's `downloadUrl` */
  url: string;
  filename: string;
}

/**
 * The whole of what a panel needs to draw itself, flat and immutable.
 *
 * Flat rather than a discriminated union because the consumer is tweakpane: it
 * wants a status string, a disabled flag and a link, and a union would make
 * every one of those a switch. Immutable because it is replayed to late
 * subscribers — a state object handed out at 40% must not later read as done.
 */
export interface ExportSessionState {
  phase: ExportPhase;
  /** true while a submission or job is in flight; this is the dedupe gate */
  busy: boolean;
  progress: number;
  /** the server's stage word, or the status when the server sends none */
  stage: string;
  /** non-empty only in the `error` phase */
  error: string;
  job: ExportJob | null;
  /** set only once a `done` job carried a download path */
  download: ExportDownload | null;
}

export interface ExportSession {
  /** the current state, without subscribing */
  state(): ExportSessionState;
  /**
   * Subscribe to state changes. The listener is called **immediately** with the
   * current state, so a panel rebuilt halfway through a render — or long after
   * one finished — paints the right thing without asking a second question.
   * Returns an unsubscribe function; unsubscribing never touches the job.
   */
  subscribe(listener: (state: ExportSessionState) => void): () => void;
  /** Probed once per session; a failed probe is not cached, so a rebuild retries. */
  capabilities(): Promise<ExportCapabilities>;
  /**
   * Capture a recipe and submit it.
   *
   * Returns `false` — with no state change and no notification — when a job is
   * already in flight. That is the duplicate-submission guard, and it is the
   * *only* reason for a `false`: an attempt that is accepted and then fails
   * during capture or submission returns `true` and reports itself through the
   * `error` phase, where a later subscriber can still see it.
   *
   * `capture` is a thunk rather than a recipe so that capturing the live sim
   * happens inside the guard, and so a capture throw becomes replayable state
   * instead of a panel-local message that the next rebuild loses.
   */
  start(trackId: string, capture: () => ExportRecipe): boolean;
  /**
   * Ask the server to stop the job holding the slot.
   *
   * Returns `false` when there is nothing to cancel. The slot is *not* released
   * here: it reopens when the poll delivers the terminal `cancelled` state, so
   * a panel cannot start a second render while the first worker is still being
   * torn down — and so the cancelled state is replayable to a panel built after
   * it, exactly like `done` and `error`.
   *
   * A cancel clicked before the server has answered the submission is
   * remembered and applied as soon as the job id exists.
   */
  cancel(): boolean;
}

export interface ExportSessionOptions {
  intervalMs?: number;
  wait?: (ms: number) => Promise<void>;
}

const IDLE: ExportSessionState = {
  phase: 'idle',
  busy: false,
  progress: 0,
  stage: '',
  error: '',
  job: null,
  download: null,
};

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Main-lifetime owner of the export job lifecycle.
 *
 * It exists because the control panel does not live as long as a render does:
 * switching simulations disposes the panel, and with it every `let` the old UI
 * used to remember that a job was running and where the finished file went. The
 * session holds that state beside the equally long-lived `LocalExportClient`,
 * and panels become pure subscribers that can come and go mid-job.
 *
 * It *composes* the client rather than re-implementing it — all HTTP, parsing
 * and polling stay in `client.ts`; what lives here is the single-slot state
 * machine and the fan-out.
 */
export function createExportSession(
  client: LocalExportClient,
  options: ExportSessionOptions = {},
): ExportSession {
  const listeners = new Set<(state: ExportSessionState) => void>();
  let state: ExportSessionState = IDLE;
  let probe: Promise<ExportCapabilities> | null = null;
  /** A cancel that arrived while the submission was still in flight. */
  let cancelPending = false;

  const emit = (next: ExportSessionState): void => {
    state = next;
    // Copied first: a listener is free to unsubscribe from inside its own call
    // (a panel disposing on the completion it just drew), and mutating the set
    // mid-iteration would drop the panel after it.
    for (const listener of [...listeners]) listener(state);
  };

  const fromJob = (job: ExportJob): ExportSessionState => ({
    phase: job.status,
    busy: job.status === 'queued' || job.status === 'running',
    progress: job.progress,
    stage: job.stage || job.status,
    error: job.status === 'error' ? job.error || job.message || 'worker error' : '',
    job,
    download: job.status === 'done' && job.downloadUrl
      ? { url: client.downloadUrl(job.downloadUrl), filename: job.filename ?? '' }
      : null,
  });

  const requestCancel = (jobId: string): void => {
    void client.cancel(jobId).catch((error: unknown) => {
      // The job keeps running and the slot stays held: the button is still
      // there, so the honest recovery is to let the user press it again.
      console.error('export cancel failed', error);
    });
  };

  const fail = (text: string): void => {
    emit({ ...IDLE, phase: 'error', error: text, job: state.job });
  };

  return {
    state(): ExportSessionState {
      return state;
    },
    subscribe(listener): () => void {
      listeners.add(listener);
      listener(state);
      return () => {
        listeners.delete(listener);
      };
    },
    async capabilities(): Promise<ExportCapabilities> {
      if (!probe) {
        probe = client.capabilities().catch((error: unknown) => {
          // A probe that failed because the local server was not up yet must not
          // poison every later panel; only successes are memoised.
          probe = null;
          throw error;
        });
      }
      return probe;
    },
    start(trackId, capture): boolean {
      if (state.busy) return false;
      cancelPending = false;
      emit({ ...IDLE, phase: 'submitting', busy: true, stage: 'submitting' });
      let recipe: ExportRecipe;
      try {
        recipe = capture();
      } catch (error) {
        fail(`capture failed: ${message(error)}`);
        return true;
      }
      void client.start(trackId, recipe).then(async (queued) => {
        emit(fromJob(queued));
        if (cancelPending) {
          cancelPending = false;
          requestCancel(queued.jobId);
        }
        const completed = await client.watch(
          queued.jobId,
          (job) => emit(fromJob(job)),
          options,
        );
        emit(fromJob(completed));
      }).catch((error: unknown) => {
        fail(message(error));
        console.error('video export failed', error);
      });
      return true;
    },
    cancel(): boolean {
      if (!state.busy) return false;
      const jobId = state.job?.jobId;
      if (jobId === undefined) {
        // Still submitting: the job exists on the server or is about to, and
        // `start` applies this the moment it learns the id.
        cancelPending = true;
      } else {
        requestCancel(jobId);
      }
      // Advisory only — `busy` is untouched, so this cannot open the slot early.
      emit({ ...state, stage: 'cancelling' });
      return true;
    },
  };
}
