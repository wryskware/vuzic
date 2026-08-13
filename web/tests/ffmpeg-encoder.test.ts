import assert from 'node:assert/strict';
import { access, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';

import {
  AV1_DEBUG_FPS,
  Av1DebugEncoder,
  MAX_AV1_DEBUG_RANGE_SECONDS,
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
  assert.ok(args.includes('-an'));
  assert.ok(!args.includes('-c:a'));
  assert.equal(args.at(-1), `${value.outputPath}.partial`);
  assert.equal(partialOutputPath(value.outputPath), `${value.outputPath}.partial`);
});

test('trusted WAV audio is trimmed to the requested range, reset to zero, and muxed as AAC-LC', () => {
  const value = {
    ...options(),
    audioPath: resolve('audio/source.wav'),
    startSeconds: 12.25,
    durationSeconds: 3.5,
  };
  const args = buildAv1DebugFfmpegArgs(value);
  assert.ok(!args.includes('-an'));
  assert.equal(args[args.indexOf('-i', args.indexOf('-i') + 1) + 1], value.audioPath);
  assert.deepEqual(
    [args[args.indexOf('-map') + 1], args[args.indexOf('-map', args.indexOf('-map') + 1) + 1]],
    ['0:v:0', '1:a:0'],
  );
  assert.equal(
    args[args.indexOf('-af') + 1],
    'atrim=start=12.25:duration=3.5,asetpts=PTS-STARTPTS',
  );
  assert.equal(args[args.indexOf('-c:a') + 1], 'aac');
  assert.equal(args[args.indexOf('-profile:a') + 1], 'aac_low');
  assert.equal(args[args.indexOf('-ar') + 1], '48000');
  assert.equal(args[args.indexOf('-b:a') + 1], '192k');
  assert.equal(args[args.indexOf('-t') + 1], '3.5');
  assert.equal(args[args.indexOf('-framerate') + 1], '120');
  assert.equal(args[args.indexOf('-fps_mode') + 1], 'cfr');
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
  assert.throws(
    () => buildAv1DebugFfmpegArgs({ ...options(), audioPath: resolve('audio/source.wav') }),
    /must be supplied together/,
  );
  assert.throws(
    () => buildAv1DebugFfmpegArgs({ ...options(), startSeconds: 0, durationSeconds: 1 }),
    /must be supplied together/,
  );
  assert.throws(
    () =>
      buildAv1DebugFfmpegArgs({
        ...options(),
        audioPath: 'relative.wav',
        startSeconds: 0,
        durationSeconds: 1,
      }),
    /audioPath.*absolute/,
  );
  assert.throws(
    () =>
      buildAv1DebugFfmpegArgs({
        ...options(),
        audioPath: resolve('audio/source.mp3'),
        startSeconds: 0,
        durationSeconds: 1,
      }),
    /audioPath.*\.wav/,
  );
  assert.throws(
    () =>
      buildAv1DebugFfmpegArgs({
        ...options(),
        audioPath: resolve('audio/source.wav'),
        startSeconds: -1,
        durationSeconds: 1,
      }),
    /startSeconds/,
  );
  assert.throws(
    () =>
      buildAv1DebugFfmpegArgs({
        ...options(),
        audioPath: resolve('audio/source.wav'),
        startSeconds: 0,
        durationSeconds: 0,
      }),
    /durationSeconds/,
  );
  assert.throws(
    () =>
      buildAv1DebugFfmpegArgs({
        ...options(),
        audioPath: resolve('audio/source.wav'),
        startSeconds: MAX_AV1_DEBUG_RANGE_SECONDS,
        durationSeconds: 1,
      }),
    /must end no later/,
  );
});

test('an early child exit is reported through write backpressure and cleans the partial', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'lmt-ffmpeg-encoder-test-'));
  const outputPath = join(directory, 'failed.mp4');
  try {
    // Node is intentionally not FFmpeg: it spawns successfully, rejects the
    // fixed FFmpeg arguments, and closes stdin to exercise the EPIPE path.
    const encoder = await Av1DebugEncoder.start({
      ffmpegExecutable: process.execPath,
      workingDirectory: directory,
      outputPath,
      width: 256,
      height: 256,
    });
    await assert.rejects(
      encoder.writeFrame(new Uint8Array(encoder.bytesPerFrame)),
      /FFmpeg frame write failed/,
    );
    assert.equal(encoder.state, 'failed');
    await assert.rejects(access(`${outputPath}.partial`), { code: 'ENOENT' });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
