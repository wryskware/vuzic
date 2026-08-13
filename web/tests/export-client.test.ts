import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createLocalExportClient, type ExportJob } from '../src/export/client.ts';
import type { ExportRecipe } from '../src/runtime/recipe.ts';
import {
  DEBUG_EXPORT_LABEL,
  DEBUG_EXPORT_OUTPUT,
  EXPORT_BUTTON_LABEL,
  debugExportAvailable,
  exportProgressLabel,
} from '../src/ui/export-panel.ts';

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function job(status: ExportJob['status'], progress: number): ExportJob {
  return {
    jobId: 'job-1',
    kind: 'export',
    trackId: 'pink-loop',
    title: 'Pink Loop',
    status,
    stage: status === 'running' ? 'render' : status,
    message: '',
    error: '',
    progress,
    exportId: 'export-1',
  };
}

test('local export client probes capabilities and posts only trackId plus recipe', async () => {
  const requests: Array<{ url: string; init?: RequestInit }> = [];
  const recipe = { version: 2, rendererBuild: 'build-1' } as ExportRecipe;
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), ...(init === undefined ? {} : { init }) });
    if (String(url).endsWith('/exports/capabilities')) {
      return json({
        available: true,
        profiles: ['hdr10-1080p120'],
        encoders: ['av1_nvenc'],
        rendererBuild: 'build-1',
        gpu: 'Fixture GPU',
        backend: 'd3d12',
        transport: 'sdr-rgba8-av1-debug',
        reason: '',
      });
    }
    return json(job('queued', 0));
  }) as typeof fetch;
  const client = createLocalExportClient(fetcher, 'http://localhost:8765');

  const capabilities = await client.capabilities();
  assert.equal(capabilities.transport, 'sdr-rgba8-av1-debug');
  assert.equal(capabilities.rendererBuild, 'build-1');
  await client.start('pink-loop', recipe);

  assert.equal(requests[0]?.url, 'http://localhost:8765/exports/capabilities');
  assert.equal(requests[1]?.url, 'http://localhost:8765/exports');
  assert.equal(requests[1]?.init?.method, 'POST');
  assert.deepEqual(JSON.parse(String(requests[1]?.init?.body)), {
    trackId: 'pink-loop',
    recipe,
  });
});

test('watch polls the flat job response through completion and reports every update', async () => {
  const responses = [job('running', 0.375), {
    ...job('done', 1),
    filename: 'pink-loop.mp4',
    downloadUrl: '/exports/export-1/download',
  }];
  const fetcher = (async () => json(responses.shift())) as typeof fetch;
  const client = createLocalExportClient(fetcher, 'http://localhost:8765');
  const updates: ExportJob[] = [];
  const completed = await client.watch('job-1', (value) => updates.push(value), {
    intervalMs: 0,
    wait: async () => {},
  });

  assert.deepEqual(updates.map((value) => value.status), ['running', 'done']);
  assert.equal(updates[0]?.progress, 0.375);
  assert.equal(completed.downloadUrl, '/exports/export-1/download');
  assert.equal(
    client.downloadUrl(completed.downloadUrl!),
    'http://localhost:8765/exports/export-1/download',
  );
});

test('server error detail is surfaced instead of a generic failed fetch', async () => {
  const fetcher = (async () => json({ detail: 'renderer build changed' }, 409)) as typeof fetch;
  const client = createLocalExportClient(fetcher, 'http://localhost:8765');
  await assert.rejects(client.capabilities(), /renderer build changed/);
});

test('the UI advertises only the honest, currently supported SDR debug path', () => {
  assert.equal(EXPORT_BUTTON_LABEL, 'Render video with current settings');
  assert.equal(DEBUG_EXPORT_LABEL, '1080p / 120 fps / SDR debug');
  assert.deepEqual(DEBUG_EXPORT_OUTPUT, {
    profile: 'hdr10-1080p120',
    encoder: 'av1_nvenc',
    paperWhiteNits: 203,
    masteringPeakNits: 1000,
  });
  const base = {
    available: true,
    profiles: ['hdr10-1080p120'] as const,
    encoders: ['av1_nvenc'],
    rendererBuild: 'build-1',
    gpu: '',
    backend: '',
    transport: 'sdr-rgba8-av1-debug',
    reason: '',
  };
  assert.equal(debugExportAvailable({ ...base, profiles: [...base.profiles] }), true);
  assert.equal(debugExportAvailable({ ...base, profiles: [], encoders: [] }), false);
  assert.equal(debugExportAvailable({ ...base, transport: 'hdr10-p010' }), false);
  assert.equal(exportProgressLabel(job('running', 0.376)), 'render 38%');
});
