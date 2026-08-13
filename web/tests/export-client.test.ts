import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createLocalExportClient, type ExportJob } from '../src/export/client.ts';
import type { ExportRecipe } from '../src/runtime/recipe.ts';
import {
  DEFAULT_SDR_DEBUG_PROFILE,
  EXPORT_BUTTON_LABEL,
  SDR_DEBUG_PROFILES,
  debugExportOutput,
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
  const recipe = { version: 3, rendererBuild: 'build-1' } as ExportRecipe;
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), ...(init === undefined ? {} : { init }) });
    if (String(url).endsWith('/exports/capabilities')) {
      return json({
        available: true,
        profiles: ['av1-sdr-debug-1080p120', 'av1-sdr-debug-2160p120'],
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
  assert.deepEqual(capabilities.profiles, [
    'av1-sdr-debug-1080p120',
    'av1-sdr-debug-2160p120',
  ]);
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

test('cancel is a DELETE on the job with no body, and watch stops on cancelled', async () => {
  const requests: Array<{ url: string; init?: RequestInit }> = [];
  const responses = [
    { ...job('running', 0.5), status: 'running' },
    { ...job('cancelled', 0.5), status: 'cancelled', stage: 'cancelled' },
  ];
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), ...(init === undefined ? {} : { init }) });
    if (init?.method === 'DELETE') return json({ ...job('running', 0.5), stage: 'cancelling' });
    return json(responses.shift());
  }) as typeof fetch;
  const client = createLocalExportClient(fetcher, 'http://localhost:8765');

  const acknowledged = await client.cancel('job 1/2');
  assert.equal(requests[0]?.url, 'http://localhost:8765/jobs/job%201%2F2');
  assert.equal(requests[0]?.init?.method, 'DELETE');
  assert.equal(requests[0]?.init?.body, undefined);
  // The DELETE answers with the job as it stands, not with a bare 204: the
  // panel needs to know whether the slot is already free.
  assert.equal(acknowledged.status, 'running');
  assert.equal(acknowledged.stage, 'cancelling');

  const updates: ExportJob[] = [];
  const final = await client.watch('job-1', (value) => updates.push(value), {
    intervalMs: 0,
    wait: async () => {},
  });
  assert.deepEqual(updates.map((value) => value.status), ['running', 'cancelled']);
  assert.equal(final.status, 'cancelled');
});

test('server error detail is surfaced instead of a generic failed fetch', async () => {
  const fetcher = (async () => json({ detail: 'renderer build changed' }, 409)) as typeof fetch;
  const client = createLocalExportClient(fetcher, 'http://localhost:8765');
  await assert.rejects(client.capabilities(), /renderer build changed/);
});

test('the UI defaults to recommended 1080p and offers honest 1080p and 4K SDR choices', () => {
  assert.equal(EXPORT_BUTTON_LABEL, 'Render video with current settings');
  assert.equal(DEFAULT_SDR_DEBUG_PROFILE, 'av1-sdr-debug-1080p120');
  assert.deepEqual(SDR_DEBUG_PROFILES, [
    {
      profile: 'av1-sdr-debug-1080p120',
      label: '1080p / 120 fps / SDR debug',
    },
    {
      profile: 'av1-sdr-debug-2160p120',
      label: '4K / 120 fps / SDR debug',
    },
  ]);
  assert.deepEqual(debugExportOutput('av1-sdr-debug-1080p120'), {
    profile: 'av1-sdr-debug-1080p120',
    encoder: 'av1_nvenc',
    paperWhiteNits: 203,
    masteringPeakNits: 1000,
  });
  assert.equal(
    debugExportOutput('av1-sdr-debug-2160p120').profile,
    'av1-sdr-debug-2160p120',
  );
  const base = {
    available: true,
    profiles: ['av1-sdr-debug-1080p120'] as const,
    encoders: ['av1_nvenc'],
    rendererBuild: 'build-1',
    gpu: '',
    backend: '',
    transport: 'sdr-rgba8-av1-debug',
    reason: '',
  };
  assert.equal(debugExportAvailable({ ...base, profiles: [...base.profiles] }), true);
  assert.equal(
    debugExportAvailable(
      { ...base, profiles: ['av1-sdr-debug-1080p120', 'av1-sdr-debug-2160p120'] },
      'av1-sdr-debug-2160p120',
    ),
    true,
  );
  assert.equal(
    debugExportAvailable({ ...base, profiles: [...base.profiles] }, 'av1-sdr-debug-2160p120'),
    false,
  );
  assert.equal(debugExportAvailable({ ...base, profiles: [], encoders: [] }), false);
  assert.equal(debugExportAvailable({ ...base, transport: 'hdr10-p010' }), false);
  assert.equal(exportProgressLabel(job('running', 0.376)), 'render 38%');
});
