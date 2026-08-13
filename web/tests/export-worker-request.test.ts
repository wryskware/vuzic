import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import {
  loadExportWorkerRequest,
  MAX_EXPORT_RANGE_SECONDS,
  parseExportWorkerRequest,
  validateExportWorkerRequest,
  type ExportWorkerRequest,
} from '../src/export/worker-request.ts';
import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import { presetFromConfig, presetToVector } from '../src/sim/plife/preset.ts';
import type { ExportRecipe } from '../src/runtime/recipe.ts';

function recipe(): ExportRecipe {
  const { render, ...simulation } = defaultPlifeConfig();
  const { render: _sharedRender, ...modulation } = defaultModulationConfig(
    { ...simulation, render },
    'plife',
  );
  return {
    version: 2,
    rendererBuild: 'test-build',
    track: { id: 'pink-loop', contentVersion: 'sha256-deadbeef' },
    sim: 'plife',
    seed: 1234,
    seedPinned: true,
    simulation,
    modulation,
    modulationBase: Array.from(presetToVector(presetFromConfig({ ...simulation, render }), simulation.speciesCount)),
    impulses: defaultImpulseConfig(),
    render,
    particleBudget: simulation.budget.cap,
    presentation: { mode: 'single', autoAdvance: false },
    output: {
      profile: 'hdr10-1080p120',
      encoder: 'av1_nvenc',
      paperWhiteNits: 203,
      masteringPeakNits: 1000,
    },
  };
}

function request(): ExportWorkerRequest {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../test-worker-files');
  return {
    version: 1,
    recipe: recipe(),
    runtime: {
      ffmpegExecutable: join(root, process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg'),
      workingDirectory: root,
    },
    timeline: {
      jsonPath: join(root, 'timeline.json'),
      binaryPath: join(root, 'timeline.bin'),
    },
    audioPath: join(root, 'track.wav'),
    output: {
      path: join(root, 'render.mp4'),
      profile: 'hdr10-1080p120',
    },
    range: { startSeconds: 4, durationSeconds: 10 },
  };
}

test('worker request round-trips the strict versioned internal contract', () => {
  const value = request();
  validateExportWorkerRequest(value);
  assert.deepEqual(parseExportWorkerRequest(JSON.stringify(value)), value);
});

test('worker request requires server-resolved paths and matching bounded output', () => {
  const relative = request() as unknown as { timeline: { jsonPath: string } };
  relative.timeline.jsonPath = 'timeline.json';
  assert.throws(() => validateExportWorkerRequest(relative), /jsonPath.*absolute native path/);

  const ambientFfmpeg = request();
  ambientFfmpeg.runtime.ffmpegExecutable = 'ffmpeg';
  assert.throws(
    () => validateExportWorkerRequest(ambientFfmpeg),
    /ffmpegExecutable.*absolute native path/,
  );

  const ambientCwd = request();
  ambientCwd.runtime.workingDirectory = '.';
  assert.throws(
    () => validateExportWorkerRequest(ambientCwd),
    /workingDirectory.*absolute native path/,
  );

  const mismatch = request();
  mismatch.recipe.output.profile = 'hdr10-2160p120';
  assert.throws(() => validateExportWorkerRequest(mismatch), /output\.profile.*must match/);

  const tooLong = request();
  tooLong.range.startSeconds = MAX_EXPORT_RANGE_SECONDS - 1;
  tooLong.range.durationSeconds = 2;
  assert.throws(() => validateExportWorkerRequest(tooLong), /must end no later/);
});

test('worker request rejects schema drift and loads one bounded request file', async () => {
  const unknown = request() as unknown as Record<string, unknown>;
  unknown['ffmpegArgs'] = ['-anything'];
  assert.throws(() => validateExportWorkerRequest(unknown), /ffmpegArgs.*not supported/);

  const directory = await mkdtemp(join(tmpdir(), 'terrarium-worker-request-'));
  try {
    const path = join(directory, 'request.json');
    const value = request();
    await writeFile(path, JSON.stringify(value), 'utf8');
    assert.deepEqual(await loadExportWorkerRequest(path), value);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
