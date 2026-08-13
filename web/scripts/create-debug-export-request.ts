import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import { validateExportWorkerRequest, type ExportWorkerRequest } from '../src/export/worker-request.ts';
import { defaultModulationConfig } from '../src/mapping/persist.ts';
import { baseVector } from '../src/mapping/modulation.ts';
import { defaultImpulseConfig } from '../src/sim/impulses.ts';
import { defaultPlifeConfig } from '../src/sim/plife/config.ts';
import { seedMatrixBase } from '../src/sim/plife/genmatrix.ts';
import {
  modulationSlots,
  presetFromConfig,
  presetToVector,
} from '../src/sim/plife/preset.ts';
import {
  EXPORT_PROFILES,
  EXPORT_RECIPE_VERSION,
  type ExportProfile,
  type ExportRecipe,
} from '../src/runtime/recipe.ts';

function option(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`${name} needs a value`);
  return value;
}

function finiteOption(name: string, fallback: number, min: number): number {
  const value = Number(option(name, String(fallback)));
  if (!Number.isFinite(value) || value < min) throw new Error(`${name} must be >= ${min}`);
  return value;
}

function profileOption(): ExportProfile {
  const value = option('--profile', 'av1-sdr-debug-1080p120');
  if (!EXPORT_PROFILES.includes(value as ExportProfile)) {
    throw new Error(`--profile must be one of ${EXPORT_PROFILES.join(', ')}`);
  }
  return value as ExportProfile;
}

const repoRoot = resolve(import.meta.dirname, '../..');
const trackRoot = resolve(option('--track-root', resolve(repoRoot, 'data/timelines/pink-loop')));
const requestPath = resolve(option('--request', resolve(repoRoot, 'web/exports/debug-request.json')));
const outputPath = resolve(option('--output', resolve(repoRoot, 'web/exports/pink-loop-plife-debug.mp4')));
const ffmpegValue = option('--ffmpeg', process.env['FFMPEG_PATH'] ?? '');
if (!ffmpegValue) throw new Error('--ffmpeg or FFMPEG_PATH must name the absolute FFmpeg executable');
const ffmpegExecutable = resolve(ffmpegValue);
const startSeconds = finiteOption('--start', 0, 0);
const durationSeconds = finiteOption('--duration', 5, Number.MIN_VALUE);
const seed = Math.floor(finiteOption('--seed', 0x5eed_120, 0));
const profile = profileOption();

const fullSimulation = defaultPlifeConfig();
fullSimulation.budget.adaptive = false;
const { render, ...simulation } = fullSimulation;
const { render: _sharedRender, ...modulation } = defaultModulationConfig(fullSimulation, 'plife');
const slots = modulationSlots(fullSimulation.speciesCount);
const defaults = presetToVector(
  presetFromConfig(fullSimulation),
  fullSimulation.speciesCount,
);
const seededBase = baseVector(seed, defaults, slots, new Float64Array(slots.length));
seedMatrixBase(seed, fullSimulation, seededBase);

const recipe: ExportRecipe = {
  version: EXPORT_RECIPE_VERSION,
  rendererBuild: 'phase2-native-worker',
  track: { id: 'pink-loop', contentVersion: 'local-engineering' },
  sim: 'plife',
  seed,
  seedPinned: true,
  simulation,
  modulation,
  modulationBase: Array.from(seededBase),
  impulses: defaultImpulseConfig(),
  render,
  particleBudget: simulation.budget.cap,
  presentation: { mode: 'single', autoAdvance: false },
  output: {
    profile,
    encoder: 'av1_nvenc',
    paperWhiteNits: 203,
    masteringPeakNits: 1000,
  },
};

const request: ExportWorkerRequest = {
  version: 1,
  recipe,
  runtime: {
    ffmpegExecutable,
    workingDirectory: repoRoot,
  },
  timeline: {
    jsonPath: resolve(trackRoot, 'timeline.json'),
    binaryPath: resolve(trackRoot, 'timeline.bin'),
  },
  audioPath: resolve(trackRoot, 'audio.wav'),
  output: {
    path: outputPath,
    profile: recipe.output.profile,
  },
  range: { startSeconds, durationSeconds },
};

validateExportWorkerRequest(request);
await mkdir(dirname(requestPath), { recursive: true });
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(requestPath, `${JSON.stringify(request, null, 2)}\n`, { flag: 'wx' });
process.stdout.write(`${requestPath}\n`);
