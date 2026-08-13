import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { ExportCapabilities, ExportJob, LocalExportClient } from '../src/export/client.ts';
import { createExportSession, type ExportSessionState } from '../src/export/session.ts';
import type { ExportRecipe } from '../src/runtime/recipe.ts';
import { exportSessionLabel } from '../src/ui/export-panel.ts';

const RECIPE = { version: 3, rendererBuild: 'build-1' } as ExportRecipe;

function job(status: ExportJob['status'], progress: number, extra: Partial<ExportJob> = {}): ExportJob {
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
    ...extra,
  };
}

/**
 * A client whose polling is driven by the test rather than by a timer: `watch`
 * hands back one queued response per `step()`, so a test can stop the job
 * exactly at the point a panel would be disposed.
 */
function fakeClient(updates: ExportJob[], options: {
  start?: () => Promise<ExportJob>;
  capabilities?: () => Promise<ExportCapabilities>;
} = {}): { client: LocalExportClient; step: () => Promise<void> } {
  const pending = [...updates];
  let release: (() => void) | null = null;
  const gate = (): Promise<void> => new Promise((resolve) => {
    release = resolve;
  });
  const client: LocalExportClient = {
    capabilities: options.capabilities ?? (async () => {
      throw new Error('not used');
    }),
    start: options.start ?? (async () => job('queued', 0)),
    async job() {
      throw new Error('not used');
    },
    async watch(_jobId, onUpdate) {
      for (;;) {
        await gate();
        const next = pending.shift();
        if (!next) throw new Error('watch ran out of fixture updates');
        onUpdate(next);
        if (next.status === 'done' || next.status === 'error') return next;
      }
    },
    downloadUrl(path: string): string {
      return `http://localhost:8765${path}`;
    },
  };
  return {
    client,
    /** Let exactly one poll through, then drain the emits it caused. */
    step: async (): Promise<void> => {
      for (let attempt = 0; attempt < 50 && !release; attempt += 1) await settle();
      if (!release) throw new Error('watch is not waiting for a poll');
      release();
      release = null;
      await settle();
    },
  };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setImmediate(resolve));
}

const DONE = job('done', 1, {
  filename: 'pink-loop.mp4',
  downloadUrl: '/exports/export-1/download',
});

test('a panel disposed mid-job is replaced by one that sees the live job and its completion', async () => {
  const fake = fakeClient([job('running', 0.5), DONE]);
  const session = createExportSession(fake.client);

  // First panel: subscribes, starts, watches the job reach 50%.
  const first: ExportSessionState[] = [];
  const disposeFirst = session.subscribe((state) => first.push(state));
  assert.equal(first[0]?.phase, 'idle');
  assert.equal(session.start('pink-loop', () => RECIPE), true);
  await settle();
  await fake.step();
  assert.equal(first.at(-1)?.progress, 0.5);

  // The simulation swap: the panel goes away, the render does not.
  disposeFirst();
  const seenAfterDispose = first.length;

  // Rebuilt panel: replay must show the in-flight job, not idle.
  const second: ExportSessionState[] = [];
  session.subscribe((state) => second.push(state));
  assert.equal(second[0]?.phase, 'running');
  assert.equal(second[0]?.busy, true);
  assert.equal(second[0]?.progress, 0.5);
  assert.equal(exportSessionLabel(second[0]!), 'render 50%');

  await fake.step();
  const final = second.at(-1)!;
  assert.equal(final.phase, 'done');
  assert.equal(final.busy, false);
  assert.deepEqual(final.download, {
    url: 'http://localhost:8765/exports/export-1/download',
    filename: 'pink-loop.mp4',
  });
  assert.equal(exportSessionLabel(final), 'done · pink-loop.mp4');
  assert.equal(first.length, seenAfterDispose, 'the disposed panel got no further updates');
});

test('a panel built after the render finished replays the done state and download metadata', async () => {
  const fake = fakeClient([DONE]);
  const session = createExportSession(fake.client);
  session.start('pink-loop', () => RECIPE);
  await settle();
  await fake.step();
  assert.equal(session.state().phase, 'done');

  const late: ExportSessionState[] = [];
  session.subscribe((state) => late.push(state));
  assert.equal(late.length, 1);
  assert.equal(late[0]?.phase, 'done');
  assert.equal(late[0]?.busy, false);
  assert.equal(late[0]?.download?.url, 'http://localhost:8765/exports/export-1/download');
  assert.equal(late[0]?.download?.filename, 'pink-loop.mp4');
});

test('a second start while a job is queued or running is refused without disturbing the job', async () => {
  let starts = 0;
  const fake = fakeClient([job('running', 0.25), DONE], {
    start: async () => {
      starts += 1;
      return job('queued', 0);
    },
  });
  const session = createExportSession(fake.client);
  const seen: ExportSessionState[] = [];
  session.subscribe((state) => seen.push(state));

  assert.equal(session.start('pink-loop', () => RECIPE), true);
  // Refused while still submitting…
  assert.equal(session.start('pink-loop', () => RECIPE), false);
  await settle();
  assert.equal(session.state().phase, 'queued');
  // …and refused again once queued, and once running.
  assert.equal(session.start('pink-loop', () => RECIPE), false);
  await fake.step();
  assert.equal(session.state().phase, 'running');
  assert.equal(session.start('pink-loop', () => RECIPE), false);
  assert.equal(starts, 1, 'only one submission reached the client');

  await fake.step();
  assert.equal(session.state().phase, 'done');
  // The gate opens again once the slot is free.
  assert.equal(session.state().busy, false);
});

test('a failed submission is replayable error state for a panel built afterwards', async () => {
  const fake = fakeClient([], {
    start: async () => {
      throw new Error('renderer build changed');
    },
  });
  const session = createExportSession(fake.client);
  session.start('pink-loop', () => RECIPE);
  await settle();

  const late: ExportSessionState[] = [];
  session.subscribe((state) => late.push(state));
  assert.equal(late[0]?.phase, 'error');
  assert.equal(late[0]?.busy, false);
  assert.equal(late[0]?.error, 'renderer build changed');
  assert.equal(exportSessionLabel(late[0]!), 'failed: renderer build changed');
  // The failure released the slot.
  assert.equal(session.start('pink-loop', () => RECIPE), true);
});

test('a capture throw becomes error state instead of a lost panel-local message', async () => {
  const fake = fakeClient([]);
  const session = createExportSession(fake.client);
  assert.equal(session.start('pink-loop', () => {
    throw new Error('no live sim');
  }), true);
  const state = session.state();
  assert.equal(state.phase, 'error');
  assert.equal(state.error, 'capture failed: no live sim');
  assert.equal(exportSessionLabel(state), 'failed: capture failed: no live sim');
});

test('a worker error job replays as error state with the worker message', async () => {
  const fake = fakeClient([job('error', 0.4, { error: 'ffmpeg exited 1' })]);
  const session = createExportSession(fake.client);
  session.start('pink-loop', () => RECIPE);
  await settle();
  await fake.step();

  const late: ExportSessionState[] = [];
  session.subscribe((state) => late.push(state));
  assert.equal(late[0]?.phase, 'error');
  assert.equal(late[0]?.error, 'ffmpeg exited 1');
  assert.equal(late[0]?.download, null);
});

test('unsubscribing stops notifications for that listener only', async () => {
  const fake = fakeClient([job('running', 0.5), DONE]);
  const session = createExportSession(fake.client);
  const kept: ExportSessionState[] = [];
  const dropped: ExportSessionState[] = [];
  session.subscribe((state) => kept.push(state));
  const unsubscribe = session.subscribe((state) => dropped.push(state));
  unsubscribe();
  const droppedCount = dropped.length;

  session.start('pink-loop', () => RECIPE);
  await settle();
  await fake.step();
  await fake.step();

  assert.equal(dropped.length, droppedCount);
  assert.equal(kept.at(-1)?.phase, 'done');
  // Unsubscribing twice is harmless.
  unsubscribe();
});

test('capabilities are probed once, and a failed probe is retried by the next panel', async () => {
  let calls = 0;
  const ok: ExportCapabilities = {
    available: true,
    profiles: ['av1-sdr-debug-1080p120'],
    encoders: ['av1_nvenc'],
    rendererBuild: 'build-1',
    gpu: 'Fixture GPU',
    backend: 'd3d12',
    transport: 'sdr-rgba8-av1-debug',
    reason: '',
  };
  const fake = fakeClient([], {
    capabilities: async () => {
      calls += 1;
      if (calls === 1) throw new Error('server down');
      return ok;
    },
  });
  const session = createExportSession(fake.client);
  await assert.rejects(session.capabilities(), /server down/);
  assert.equal((await session.capabilities()).gpu, 'Fixture GPU');
  assert.equal((await session.capabilities()).gpu, 'Fixture GPU');
  assert.equal(calls, 2, 'the success was memoised, the failure was not');
});
