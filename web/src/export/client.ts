import { EXPORT_PROFILES, type ExportRecipe, type ExportProfile } from '../runtime/recipe.ts';
import { SERVER_BASE } from '../timeline/catalog.ts';

export interface ExportCapabilities {
  available: boolean;
  profiles: ExportProfile[];
  encoders: string[];
  rendererBuild: string;
  gpu: string;
  backend: string;
  transport: string;
  reason: string;
}

export type ExportJobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled';

/** Statuses a job never leaves; `watch` stops here and the busy slot reopens. */
export const TERMINAL_EXPORT_STATUSES: readonly ExportJobStatus[] = ['done', 'error', 'cancelled'];

/** Flat `/jobs/{id}` response; completion metadata is additive. */
export interface ExportJob {
  jobId: string;
  kind: 'export';
  trackId: string;
  title: string;
  status: ExportJobStatus;
  stage: string;
  message: string;
  error: string;
  progress: number;
  exportId: string;
  filename?: string;
  byteSize?: number;
  duration?: number;
  width?: number;
  height?: number;
  frameRate?: number;
  frames?: number;
  codec?: string;
  audio?: boolean;
  transport?: string;
  downloadUrl?: string;
  /** Present on completed HDR10 exports; absent on the SDR debug profiles. */
  dynamicRange?: string;
  pixelFormat?: string;
  colorPrimaries?: string;
  colorTransfer?: string;
  colorMatrix?: string;
  colorRange?: string;
  paperWhiteNits?: number;
  masteringPeakNits?: number;
}

export interface LocalExportClient {
  capabilities(): Promise<ExportCapabilities>;
  start(trackId: string, recipe: ExportRecipe): Promise<ExportJob>;
  job(jobId: string): Promise<ExportJob>;
  /**
   * Ask the server to cancel a job. Returns the job as the server then sees it —
   * `cancelled` for a queued job, still `running` for one whose worker is being
   * stopped, or an untouched `done` for a job that beat the request. Cancelling
   * an already-terminal job is not an error.
   */
  cancel(jobId: string): Promise<ExportJob>;
  watch(
    jobId: string,
    onUpdate: (job: ExportJob) => void,
    options?: { intervalMs?: number; wait?: (ms: number) => Promise<void> },
  ): Promise<ExportJob>;
  downloadUrl(path: string): string;
}

function object(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${name}: malformed server response`);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function parseCapabilities(value: unknown): ExportCapabilities {
  const row = object(value, 'export capabilities');
  return {
    available: row['available'] === true,
    profiles: strings(row['profiles']).filter((profile): profile is ExportProfile =>
      EXPORT_PROFILES.includes(profile as ExportProfile),
    ),
    encoders: strings(row['encoders']),
    rendererBuild: text(row['rendererBuild']),
    gpu: text(row['gpu']),
    backend: text(row['backend']),
    transport: text(row['transport']),
    reason: text(row['reason']),
  };
}

function parseJob(value: unknown): ExportJob {
  const row = object(value, 'export job');
  const status = row['status'];
  if (!['queued', 'running', 'done', 'error', 'cancelled'].includes(status as string)) {
    throw new Error('export job: invalid status');
  }
  if (row['kind'] !== 'export') throw new Error('export job: response is not an export job');
  const progress = typeof row['progress'] === 'number' && Number.isFinite(row['progress'])
    ? Math.min(Math.max(row['progress'], 0), 1)
    : 0;
  const job: ExportJob = {
    jobId: text(row['jobId']),
    kind: 'export',
    trackId: text(row['trackId']),
    title: text(row['title']),
    status: status as ExportJobStatus,
    stage: text(row['stage']),
    message: text(row['message']),
    error: text(row['error']),
    progress,
    exportId: text(row['exportId']),
  };
  for (const key of [
    'filename',
    'codec',
    'transport',
    'downloadUrl',
    'dynamicRange',
    'pixelFormat',
    'colorPrimaries',
    'colorTransfer',
    'colorMatrix',
    'colorRange',
  ] as const) {
    if (typeof row[key] === 'string') job[key] = row[key];
  }
  for (const key of [
    'byteSize',
    'duration',
    'width',
    'height',
    'frameRate',
    'frames',
    'paperWhiteNits',
    'masteringPeakNits',
  ] as const) {
    if (typeof row[key] === 'number' && Number.isFinite(row[key])) job[key] = row[key];
  }
  if (typeof row['audio'] === 'boolean') job.audio = row['audio'];
  if (!job.jobId) throw new Error('export job: missing jobId');
  return job;
}

async function errorMessage(response: Response): Promise<string> {
  const body = await response.text().catch(() => '');
  if (body) {
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === 'string') return parsed.detail;
    } catch {
      return body.slice(0, 300);
    }
  }
  return `HTTP ${response.status}`;
}

export function createLocalExportClient(
  fetcher: typeof fetch = globalThis.fetch.bind(globalThis),
  serverBase = SERVER_BASE,
): LocalExportClient {
  const request = async (path: string, init?: RequestInit): Promise<unknown> => {
    const response = await fetcher(`${serverBase}${path}`, init);
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json() as Promise<unknown>;
  };
  return {
    async capabilities(): Promise<ExportCapabilities> {
      return parseCapabilities(await request('/exports/capabilities'));
    },
    async start(trackId: string, recipe: ExportRecipe): Promise<ExportJob> {
      return parseJob(await request('/exports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trackId, recipe }),
      }));
    },
    async job(jobId: string): Promise<ExportJob> {
      return parseJob(await request(`/jobs/${encodeURIComponent(jobId)}`));
    },
    async cancel(jobId: string): Promise<ExportJob> {
      return parseJob(await request(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' }));
    },
    async watch(jobId, onUpdate, options = {}): Promise<ExportJob> {
      const intervalMs = options.intervalMs ?? 1500;
      const wait = options.wait ?? ((ms: number) => new Promise((resolve) => setTimeout(resolve, ms)));
      for (;;) {
        await wait(intervalMs);
        const current = await this.job(jobId);
        onUpdate(current);
        if (TERMINAL_EXPORT_STATUSES.includes(current.status)) return current;
      }
    },
    downloadUrl(path: string): string {
      return new URL(path, `${serverBase}/`).toString();
    },
  };
}
