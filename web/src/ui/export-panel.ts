import type { ButtonApi, FolderApi } from 'tweakpane';

import type { ExportCapabilities, ExportJob } from '../export/client.ts';
import type { ExportSession, ExportSessionState } from '../export/session.ts';
import type { ExportRecipe } from '../runtime/recipe.ts';
import type { PanelContainer } from './panel.ts';

export const EXPORT_BUTTON_LABEL = 'Render video with current settings';
export const SDR_DEBUG_PROFILES = [
  {
    profile: 'av1-sdr-debug-1080p120',
    label: '1080p / 120 fps / SDR debug (recommended)',
  },
  {
    profile: 'av1-sdr-debug-2160p120',
    label: '4K / 120 fps / SDR debug',
  },
] as const satisfies readonly { profile: ExportRecipe['output']['profile']; label: string }[];
export type SdrDebugProfile = (typeof SDR_DEBUG_PROFILES)[number]['profile'];
export const DEFAULT_SDR_DEBUG_PROFILE: SdrDebugProfile = 'av1-sdr-debug-1080p120';

export function debugExportOutput(profile: SdrDebugProfile): ExportRecipe['output'] {
  return {
    profile,
    encoder: 'av1_nvenc',
    paperWhiteNits: 203,
    masteringPeakNits: 1000,
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

export function debugExportAvailable(
  capabilities: ExportCapabilities,
  profile: SdrDebugProfile = DEFAULT_SDR_DEBUG_PROFILE,
): boolean {
  return capabilities.available &&
    capabilities.profiles.includes(profile) &&
    capabilities.encoders.includes('av1_nvenc') &&
    capabilities.transport === 'sdr-rgba8-av1-debug' &&
    capabilities.rendererBuild.length > 0;
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
  switch (state.phase) {
    case 'idle':
      return null;
    case 'submitting':
      return 'capturing current settings…';
    case 'queued':
    case 'running':
      return state.job ? exportProgressLabel(state.job) : state.stage;
    case 'done':
      return state.download
        ? `done · ${state.download.filename || 'video ready'}`
        : 'done, but the server returned no download link';
    case 'error':
      return `failed: ${state.error}`;
  }
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
    profile: DEFAULT_SDR_DEBUG_PROFILE,
    status: 'checking local renderer…',
  };
  const profile = folder.addBinding(ui, 'profile', {
    label: 'output',
    options: Object.fromEntries(SDR_DEBUG_PROFILES.map((choice) => [choice.label, choice.profile])),
  });
  const status = folder.addBinding(ui, 'status', { readonly: true, label: 'status' });
  const render = folder.addButton({ title: EXPORT_BUTTON_LABEL }) as ButtonApi;
  render.disabled = true;

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
    if (!debugExportAvailable(capabilities, ui.profile)) {
      const label = SDR_DEBUG_PROFILES.find((choice) => choice.profile === ui.profile)?.label ?? ui.profile;
      return `${label} is unavailable on the local renderer`;
    }
    return `ready${capabilities.gpu ? ` · ${capabilities.gpu}` : ''}`;
  };

  /** The one place the widgets are written, from capabilities + replayed state. */
  const paint = (): void => {
    if (disposed) return;
    const ready = capabilities !== null && debugExportAvailable(capabilities, ui.profile);
    render.disabled = session.busy || !ready;
    profile.disabled = session.busy;
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
    if (!capabilities || !debugExportAvailable(capabilities, ui.profile)) return;
    const rendererBuild = capabilities.rendererBuild;
    const output = debugExportOutput(ui.profile);
    // The session runs the capture inside its own duplicate guard, so a click
    // that arrives while an earlier render is still going is simply refused.
    host.session.start(host.trackId, () => host.capture(rendererBuild, output));
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
