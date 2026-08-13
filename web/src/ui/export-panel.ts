import type { ButtonApi, FolderApi } from 'tweakpane';

import type { ExportCapabilities, ExportJob } from '../export/client.ts';
import { DEFAULT_MASTERING_PEAK_NITS, DEFAULT_PAPER_WHITE_NITS } from '../export/hdr.ts';
import {
  SDR_DEBUG_TRANSPORT,
  isHdrExportProfile,
  requiredEncoder,
} from '../export/profiles.ts';
import type { ExportSession, ExportSessionState } from '../export/session.ts';
import type { ExportRecipe } from '../runtime/recipe.ts';
import type { PanelContainer } from './panel.ts';

export const EXPORT_BUTTON_LABEL = 'Render video with current settings';
export const EXPORT_CANCEL_LABEL = 'Cancel render';

/**
 * Profiles offered in the panel, in the order they are offered.
 *
 * The HDR entries are the real PQ path and say so; the SDR entries remain
 * labelled "debug" because they still are. Every one of them is capability-gated
 * against what the local renderer actually advertises, so an entry the server
 * cannot serve leaves the button disabled with a reason rather than failing a
 * render minutes in.
 */
export const EXPORT_PROFILE_CHOICES = [
  {
    profile: 'hevc-hdr10-1080p120',
    label: '1080p / 120 fps / HDR10 (HEVC Main10, PQ)',
  },
  {
    profile: 'hevc-hdr10-2160p120',
    label: '4K / 120 fps / HDR10 (HEVC Main10, PQ)',
  },
  {
    profile: 'av1-sdr-debug-1080p120',
    label: '1080p / 120 fps / SDR debug',
  },
  {
    profile: 'av1-sdr-debug-2160p120',
    label: '4K / 120 fps / SDR debug',
  },
] as const satisfies readonly { profile: ExportRecipe['output']['profile']; label: string }[];

/** Kept under the old name for the SDR-only callers and tests that predate HDR. */
export const SDR_DEBUG_PROFILES = EXPORT_PROFILE_CHOICES.filter(
  (choice) => !isHdrExportProfile(choice.profile),
);
export type SdrDebugProfile = ExportRecipe['output']['profile'];
export const DEFAULT_EXPORT_PROFILE: SdrDebugProfile = 'hevc-hdr10-1080p120';
export const DEFAULT_SDR_DEBUG_PROFILE: SdrDebugProfile = 'av1-sdr-debug-1080p120';

/**
 * The output section of a recipe for one profile.
 *
 * Paper white and mastering peak travel in the recipe rather than in the worker
 * so they can be art-directed later without a new schema; they are ignored by
 * the SDR path and are the whole HDR luminance policy for the HDR one.
 */
export function debugExportOutput(profile: SdrDebugProfile): ExportRecipe['output'] {
  return {
    profile,
    encoder: requiredEncoder(profile),
    paperWhiteNits: DEFAULT_PAPER_WHITE_NITS,
    masteringPeakNits: DEFAULT_MASTERING_PEAK_NITS,
  };
}

export interface ExportPanelHost {
  /** main-lifetime; outlives this folder and every panel rebuild */
  session: ExportSession;
  trackId: string;
  capture(rendererBuild: string, output: ExportRecipe['output']): ExportRecipe;
}

/** Unsubscribes from the session. It does **not** cancel an in-flight job. */
export interface ExportPanelHandle {
  dispose(): void;
}

interface UiState {
  profile: SdrDebugProfile;
  status: string;
}

/**
 * Whether the local renderer can serve one profile.
 *
 * The server only lists a profile once its whole path exists — encoder, 10-bit
 * support, built worker — so membership in `profiles` is the authority. The SDR
 * profiles additionally check the transport string they were shipped with, which
 * is what keeps an older server from being asked for a transport it never had.
 */
export function exportProfileAvailable(
  capabilities: ExportCapabilities,
  profile: SdrDebugProfile,
): boolean {
  if (
    !capabilities.available ||
    !capabilities.profiles.includes(profile) ||
    capabilities.rendererBuild.length === 0
  ) {
    return false;
  }
  if (!capabilities.encoders.includes(requiredEncoder(profile))) return false;
  if (isHdrExportProfile(profile)) return true;
  return capabilities.transport === SDR_DEBUG_TRANSPORT;
}

export function debugExportAvailable(
  capabilities: ExportCapabilities,
  profile: SdrDebugProfile = DEFAULT_SDR_DEBUG_PROFILE,
): boolean {
  return exportProfileAvailable(capabilities, profile);
}

export function exportProgressLabel(job: ExportJob): string {
  const stage = job.stage || job.status;
  return `${stage} ${Math.round(job.progress * 100)}%`;
}

/**
 * The status line for a session state, or `null` when the session has nothing
 * to say and the capability line should show instead.
 *
 * Split out of the folder so the wording is testable without a DOM: it is the
 * only thing a rebuilt panel has to reproduce exactly from replayed state.
 */
export function exportSessionLabel(state: ExportSessionState): string | null {
  // Checked before the phase: a cancel that has been sent but not yet confirmed
  // is still a queued/running job, and "render 62%" would read as if the click
  // did nothing.
  if (state.busy && state.stage === 'cancelling') return 'cancelling…';
  switch (state.phase) {
    case 'idle':
      return null;
    case 'submitting':
      return 'capturing current settings…';
    case 'queued':
    case 'running':
      return state.job ? exportProgressLabel(state.job) : state.stage;
    case 'cancelled':
      return 'cancelled';
    case 'done':
      return state.download
        ? `done · ${state.download.filename || 'video ready'}`
        : 'done, but the server returned no download link';
    case 'error':
      return `failed: ${state.error}`;
  }
}

/**
 * Visibility and enablement of the cancel button, as a function of state alone.
 *
 * Split out for the same reason the status line is: it is the one control whose
 * correctness is a lifecycle question rather than a layout one, and a rebuilt
 * panel has to arrive at the same answer from replayed state as the panel that
 * was disposed mid-render.
 */
export function exportCancelButton(state: ExportSessionState): {
  hidden: boolean;
  disabled: boolean;
} {
  return { hidden: !state.busy, disabled: state.stage === 'cancelling' };
}

/**
 * Local native-render job UI. No public/remote compute path is implied here.
 *
 * The folder owns no job state. Everything it draws below the profile selector
 * is replayed from `host.session`, which outlives it — so a simulation swap can
 * dispose this whole folder mid-render and the rebuilt one picks the job back up
 * at whatever percentage it has reached, download link included.
 */
export function createExportFolder(
  container: PanelContainer,
  host: ExportPanelHost,
): ExportPanelHandle {
  const folder = container.addFolder({ title: 'video export', expanded: false }) as FolderApi;
  const ui: UiState = {
    profile: DEFAULT_EXPORT_PROFILE,
    status: 'checking local renderer…',
  };
  const profile = folder.addBinding(ui, 'profile', {
    label: 'output',
    options: Object.fromEntries(
      EXPORT_PROFILE_CHOICES.map((choice) => [choice.label, choice.profile]),
    ),
  });
  const status = folder.addBinding(ui, 'status', { readonly: true, label: 'status' });
  const render = folder.addButton({ title: EXPORT_BUTTON_LABEL }) as ButtonApi;
  render.disabled = true;
  // Only meaningful while the slot is held, and hidden rather than disabled the
  // rest of the time: a permanently greyed cancel button next to the render
  // button reads as a broken control.
  const cancel = folder.addButton({ title: EXPORT_CANCEL_LABEL }) as ButtonApi;
  cancel.hidden = true;

  const download = document.createElement('a');
  download.textContent = 'Download rendered video';
  download.hidden = true;
  download.download = '';
  download.style.display = 'block';
  download.style.padding = '6px 8px';
  download.style.color = 'var(--tp-label-foreground-color, #b9c7d8)';
  download.style.textDecoration = 'underline';
  folder.element.appendChild(download);

  let capabilities: ExportCapabilities | null = null;
  let probeFailed = false;
  let disposed = false;
  let session: ExportSessionState = host.session.state();

  const setStatus = (value: string): void => {
    ui.status = value.slice(0, 120);
    status.refresh();
  };

  const capabilityLabel = (): string => {
    if (probeFailed) return 'local export server unavailable';
    if (!capabilities) return 'checking local renderer…';
    if (!capabilities.available) return capabilities.reason || 'local export unavailable';
    if (!exportProfileAvailable(capabilities, ui.profile)) {
      const label =
        EXPORT_PROFILE_CHOICES.find((choice) => choice.profile === ui.profile)?.label ?? ui.profile;
      return `${label} is unavailable on the local renderer`;
    }
    return `ready${capabilities.gpu ? ` · ${capabilities.gpu}` : ''}`;
  };

  /** The one place the widgets are written, from capabilities + replayed state. */
  const paint = (): void => {
    if (disposed) return;
    const ready = capabilities !== null && exportProfileAvailable(capabilities, ui.profile);
    render.disabled = session.busy || !ready;
    profile.disabled = session.busy;
    // Hidden when there is nothing to cancel; disabled while a cancel is
    // already in flight, because a second click would only be absorbed.
    const cancelState = exportCancelButton(session);
    cancel.hidden = cancelState.hidden;
    cancel.disabled = cancelState.disabled;
    setStatus(exportSessionLabel(session) ?? capabilityLabel());
    if (session.download) {
      download.href = session.download.url;
      download.download = session.download.filename;
      download.hidden = false;
    } else {
      download.hidden = true;
    }
  };

  profile.on('change', () => paint());

  void host.session.capabilities().then((value) => {
    capabilities = value;
    paint();
  }).catch((error: unknown) => {
    probeFailed = true;
    paint();
    console.info('export capability probe failed', error);
  });

  render.on('click', () => {
    if (!capabilities || !exportProfileAvailable(capabilities, ui.profile)) return;
    const rendererBuild = capabilities.rendererBuild;
    const output = debugExportOutput(ui.profile);
    // The session runs the capture inside its own duplicate guard, so a click
    // that arrives while an earlier render is still going is simply refused.
    host.session.start(host.trackId, () => host.capture(rendererBuild, output));
  });

  cancel.on('click', () => {
    host.session.cancel();
  });

  // Subscribing last: the callback paints, and it fires once immediately with
  // whatever the session already holds.
  const unsubscribe = host.session.subscribe((next) => {
    session = next;
    paint();
  });

  return {
    dispose(): void {
      disposed = true;
      unsubscribe();
    },
  };
}
