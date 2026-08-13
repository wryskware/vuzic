import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';

import {
  AV1_DEBUG_FPS,
  buildAv1DebugFfmpegArgs,
  partialOutputPath,
  type Av1DebugEncoderOptions,
} from '../src/export/ffmpeg-encoder.ts';

function options(): Av1DebugEncoderOptions {
  return {
    ffmpegExecutable: resolve('tools/ffmpeg.exe'),
    workingDirectory: resolve('work'),
    outputPath: resolve('exports/debug.mp4'),
    width: 1920,
    height: 1080,
  };
}

test('AV1 debug arguments accept RGBA8 and enforce constant 120 fps', () => {
  const value = options();
  const args = buildAv1DebugFfmpegArgs(value);
  assert.equal(AV1_DEBUG_FPS, 120);
  assert.deepEqual(args.slice(0, 8), [
    '-hide_banner',
    '-loglevel',
    'warning',
    '-nostdin',
    '-n',
    '-f',
    'rawvideo',
    '-pixel_format',
  ]);
  assert.equal(args[args.indexOf('-pixel_format') + 1], 'rgba');
  assert.equal(args[args.indexOf('-video_size') + 1], '1920x1080');
  assert.equal(args[args.indexOf('-framerate') + 1], '120');
  assert.equal(args[args.indexOf('-c:v') + 1], 'av1_nvenc');
  assert.equal(args[args.indexOf('-r') + 1], '120');
  assert.equal(args[args.indexOf('-fps_mode') + 1], 'cfr');
  assert.equal(args.at(-1), `${value.outputPath}.partial`);
  assert.equal(partialOutputPath(value.outputPath), `${value.outputPath}.partial`);
});

test('AV1 debug arguments contain no caller-provided argument escape hatch', () => {
  const args = buildAv1DebugFfmpegArgs({ ...options(), preset: 'p7', cq: 18 });
  assert.equal(args[args.indexOf('-preset') + 1], 'p7');
  assert.equal(args[args.indexOf('-cq') + 1], '18');
  assert.equal(args[args.lastIndexOf('-f') + 1], 'mp4');
  assert.ok(args.includes('bt709'));
});

test('AV1 debug encoder rejects unbounded dimensions, quality, and paths', () => {
  assert.throws(() => buildAv1DebugFfmpegArgs({ ...options(), width: 0 }), /width/);
  assert.throws(() => buildAv1DebugFfmpegArgs({ ...options(), cq: 52 }), /cq/);
  assert.throws(
    () => buildAv1DebugFfmpegArgs({ ...options(), outputPath: 'relative.mp4' }),
    /absolute path/,
  );
  assert.throws(
    () => buildAv1DebugFfmpegArgs({ ...options(), outputPath: resolve('exports/debug.mkv') }),
    /end in .mp4/,
  );
});
