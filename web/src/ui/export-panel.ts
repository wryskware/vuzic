import type { ButtonApi, FolderApi } from 'tweakpane';

import type { LocalExportClient, ExportCapabilities, ExportJob } from '../export/client.ts';
import type { ExportRecipe } from '../runtime/recipe.ts';
import type { PanelContainer } from './panel.ts';

export const DEBUG_EXPORT_LABEL = '1080p / 120 fps / SDR debug';
export const EXPORT_BUTTON_LABEL = 'Render video with current settings';
export const DEBUG_EXPORT_OUTPUT = {
  profile: 'hdr10-1080p120',
  encoder: 'av1_nvenc',
  paperWhiteNits: 203,
  masteringPeakNits: 1000,
} as const;

export interface ExportPanelHost {
  client: LocalExportClient;
  trackId: string;
  capture(rendererBuild: string, output: ExportRecipe['output']): ExportRecipe;
}

interface UiState {
  format: string;
  status: string;
}

export function debugExportAvailable(capabilities: ExportCapabilities): boolean {
  return capabilities.available &&
    capabilities.profiles.includes('hdr10-1080p120') &&
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
  const ui: UiState = { format: DEBUG_EXPORT_LABEL, status: 'checking local renderer…' };
  folder.addBinding(ui, 'format', { readonly: true, label: 'format' });
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

  void host.client.capabilities().then((value) => {
    capabilities = value;
    if (!debugExportAvailable(value)) {
      setStatus(value.reason || 'local export unavailable');
      return;
    }
    const gpu = value.gpu ? ` · ${value.gpu}` : '';
    setStatus(`ready${gpu}`);
    render.disabled = false;
  }).catch((error: unknown) => {
    setStatus('local export server unavailable');
    console.info('export capability probe failed', error);
  });

  render.on('click', () => {
    if (busy || !capabilities || !debugExportAvailable(capabilities)) return;
    busy = true;
    render.disabled = true;
    download.hidden = true;
    setStatus('capturing current settings…');
    let recipe: ExportRecipe;
    try {
      recipe = host.capture(capabilities.rendererBuild, DEBUG_EXPORT_OUTPUT);
    } catch (error) {
      busy = false;
      render.disabled = false;
      setStatus(`capture failed: ${(error as Error).message}`);
      return;
    }
    void host.client.start(host.trackId, recipe).then((queued) => {
      setStatus(exportProgressLabel(queued));
      return host.client.watch(queued.jobId, (job) => setStatus(exportProgressLabel(job)));
    }).then((completed) => {
      busy = false;
      render.disabled = false;
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
      render.disabled = capabilities !== null && !debugExportAvailable(capabilities);
      setStatus(`failed: ${(error as Error).message}`);
      console.error('video export failed', error);
    });
  });
}
