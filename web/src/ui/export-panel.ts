import type { ButtonApi, FolderApi } from 'tweakpane';

import type { LocalExportClient, ExportCapabilities, ExportJob } from '../export/client.ts';
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
  client: LocalExportClient;
  trackId: string;
  capture(rendererBuild: string, output: ExportRecipe['output']): ExportRecipe;
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

/** Local native-render job UI. No public/remote compute path is implied here. */
export function createExportFolder(container: PanelContainer, host: ExportPanelHost): void {
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
  let busy = false;
  const setStatus = (value: string): void => {
    ui.status = value.slice(0, 120);
    status.refresh();
  };

  const updateAvailability = (): void => {
    if (!capabilities) return;
    const ready = debugExportAvailable(capabilities, ui.profile);
    render.disabled = busy || !ready;
    if (busy) return;
    if (!capabilities.available) {
      setStatus(capabilities.reason || 'local export unavailable');
    } else if (!ready) {
      const label = SDR_DEBUG_PROFILES.find((choice) => choice.profile === ui.profile)?.label ?? ui.profile;
      setStatus(`${label} is unavailable on the local renderer`);
    } else {
      const gpu = capabilities.gpu ? ` · ${capabilities.gpu}` : '';
      setStatus(`ready${gpu}`);
    }
  };

  profile.on('change', () => updateAvailability());

  void host.client.capabilities().then((value) => {
    capabilities = value;
    updateAvailability();
  }).catch((error: unknown) => {
    setStatus('local export server unavailable');
    console.info('export capability probe failed', error);
  });

  render.on('click', () => {
    if (busy || !capabilities || !debugExportAvailable(capabilities, ui.profile)) return;
    busy = true;
    render.disabled = true;
    profile.disabled = true;
    download.hidden = true;
    setStatus('capturing current settings…');
    let recipe: ExportRecipe;
    try {
      recipe = host.capture(capabilities.rendererBuild, debugExportOutput(ui.profile));
    } catch (error) {
      busy = false;
      profile.disabled = false;
      updateAvailability();
      setStatus(`capture failed: ${(error as Error).message}`);
      return;
    }
    void host.client.start(host.trackId, recipe).then((queued) => {
      setStatus(exportProgressLabel(queued));
      return host.client.watch(queued.jobId, (job) => setStatus(exportProgressLabel(job)));
    }).then((completed) => {
      busy = false;
      profile.disabled = false;
      updateAvailability();
      if (completed.status === 'error') {
        setStatus(`failed: ${completed.error || completed.message || 'worker error'}`);
        return;
      }
      if (!completed.downloadUrl) {
        setStatus('done, but the server returned no download link');
        return;
      }
      setStatus(`done · ${completed.filename || 'video ready'}`);
      download.href = host.client.downloadUrl(completed.downloadUrl);
      download.hidden = false;
    }).catch((error: unknown) => {
      busy = false;
      profile.disabled = false;
      updateAvailability();
      setStatus(`failed: ${(error as Error).message}`);
      console.error('video export failed', error);
    });
  });
}
