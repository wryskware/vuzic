import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';

import {
  buildHdr10FfmpegArgs,
  partialOutputPath,
  type Hdr10EncoderOptions,
} from '../src/export/ffmpeg-encoder.ts';
import { p010Layout } from '../src/export/hdr.ts';
import {
  EXPORT_PROFILE_SPECS,
  HDR10_TRANSPORT,
  SDR_DEBUG_TRANSPORT,
  exportProfileSpec,
  isHdrExportProfile,
  requiredEncoder,
} from '../src/export/profiles.ts';
import { EXPORT_PROFILES } from '../src/runtime/recipe.ts';
import {
  EXPORT_PROFILE_CHOICES,
  DEFAULT_EXPORT_PROFILE,
  debugExportOutput,
  exportProfileAvailable,
} from '../src/ui/export-panel.ts';

function options(): Hdr10EncoderOptions {
  return {
    ffmpegExecutable: resolve('tools/ffmpeg.exe'),
    workingDirectory: resolve('work'),
    outputPath: resolve('exports/hdr.mp4'),
    width: 3840,
    height: 2160,
  };
}

function valueAfter(args: readonly string[], flag: string): string | undefined {
  return args[args.indexOf(flag) + 1];
}

test('every schema profile has a spec, and only the PQ path is labelled hdr10', () => {
  assert.deepEqual(
    EXPORT_PROFILE_SPECS.map((spec) => spec.id),
    [...EXPORT_PROFILES],
  );
  for (const spec of EXPORT_PROFILE_SPECS) {
    const hdr = spec.dynamicRange === 'hdr10';
    assert.equal(spec.id.includes('hdr10'), hdr, `${spec.id} is labelled dishonestly`);
    assert.equal(spec.transport, hdr ? HDR10_TRANSPORT : SDR_DEBUG_TRANSPORT);
    assert.equal(spec.inputPixelFormat, hdr ? 'p010le' : 'rgba');
    assert.equal(spec.encoder, 'av1_nvenc');
    assert.equal(spec.fps, 120);
    // 4:2:0 chroma decimation needs both dimensions even for every profile.
    assert.equal(spec.width % 2, 0);
    assert.equal(spec.height % 2, 0);
    assert.equal(isHdrExportProfile(spec.id), hdr);
    assert.equal(requiredEncoder(spec.id), spec.encoder);
  }
  assert.equal(exportProfileSpec('av1-hdr10-2160p120').width, 3840);
  assert.equal(exportProfileSpec('av1-hdr10-1080p120').height, 1080);
});

test('HDR10 arguments feed P010 straight into 10-bit AV1 with the full colour description', () => {
  const value = options();
  const args = buildHdr10FfmpegArgs(value);

  // Input is already the encoder's pixel format, so FFmpeg inserts no scaler
  // and cannot silently convert range or matrix behind the pipeline's back.
  assert.equal(valueAfter(args, '-pixel_format'), 'p010le');
  assert.equal(valueAfter(args, '-video_size'), '3840x2160');
  assert.equal(valueAfter(args, '-framerate'), '120');
  assert.equal(valueAfter(args, '-c:v'), 'av1_nvenc');
  assert.equal(valueAfter(args, '-pix_fmt'), 'p010le');
  assert.equal(valueAfter(args, '-color_primaries'), 'bt2020');
  assert.equal(valueAfter(args, '-color_trc'), 'smpte2084');
  assert.equal(valueAfter(args, '-colorspace'), 'bt2020nc');
  assert.equal(valueAfter(args, '-color_range'), 'tv');
  assert.equal(valueAfter(args, '-r'), '120');
  assert.equal(valueAfter(args, '-fps_mode'), 'cfr');
  assert.equal(args.at(-1), partialOutputPath(value.outputPath));
  assert.equal(valueAfter(args, '-f'), 'rawvideo');
  assert.equal(args[args.lastIndexOf('-f') + 1], 'mp4');

  // AV1 Main already covers 10 bits. Asking for a profile or an HEVC-shaped tag
  // is the mistake this pipeline is not allowed to reintroduce.
  assert.ok(!args.includes('-profile:v'));
  assert.ok(!args.includes('-tag:v'));
  assert.ok(!args.some((arg) => arg.includes('hevc')));
});

test('the AV1 sequence header is restamped, because NVENC writes only matrix and range', () => {
  const args = buildHdr10FfmpegArgs(options());
  const bsf = valueAfter(args, '-bsf:v') ?? '';
  assert.ok(bsf.startsWith('av1_metadata='));
  assert.ok(bsf.includes('color_primaries=9'));
  assert.ok(bsf.includes('transfer_characteristics=16'));
  assert.ok(bsf.includes('matrix_coefficients=9'));
  assert.ok(bsf.includes('color_range=tv'));
});

test('ST 2086 rides in as real input side data, before the -i it describes', () => {
  const args = buildHdr10FfmpegArgs(options());
  const flag = args.indexOf('-mastering_display');
  assert.ok(flag >= 0, 'mastering display metadata must be requested');
  assert.ok(flag < args.indexOf('-i'), '-mastering_display is an input option');
  // BT.2020 primaries and D65 in 0.00002 units; 1000 nits in 0.0001 cd/m².
  assert.equal(
    args[flag + 1],
    'G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)',
  );

  const dim = buildHdr10FfmpegArgs({ ...options(), masteringPeakNits: 600 });
  assert.ok(dim[dim.indexOf('-mastering_display') + 1]?.endsWith('L(6000000,1)'));
  assert.throws(
    () => buildHdr10FfmpegArgs({ ...options(), masteringPeakNits: 20_000 }),
    /masteringPeakNits/,
  );

  // MaxCLL/MaxFALL are measurements this pipeline does not make, so it makes no
  // claim about them rather than inventing one.
  assert.ok(!args.includes('-content_light'));
});

test('HDR10 output is a normal faststart MP4 needing no post-hoc container surgery', () => {
  const args = buildHdr10FfmpegArgs(options());
  assert.equal(valueAfter(args, '-movflags'), '+faststart');
});

test('HDR10 keeps the deterministic AAC-LC audio behaviour unchanged', () => {
  const args = buildHdr10FfmpegArgs({
    ...options(),
    audioPath: resolve('audio/source.wav'),
    startSeconds: 4,
    durationSeconds: 2.5,
  });
  assert.equal(
    valueAfter(args, '-af'),
    'atrim=start=4:duration=2.5,asetpts=PTS-STARTPTS',
  );
  assert.equal(valueAfter(args, '-c:a'), 'aac');
  assert.equal(valueAfter(args, '-profile:a'), 'aac_low');
  assert.equal(valueAfter(args, '-ar'), '48000');
  assert.equal(valueAfter(args, '-t'), '2.5');
  assert.ok(!args.includes('-an'));

  const silent = buildHdr10FfmpegArgs(options());
  assert.ok(silent.includes('-an'));
  assert.ok(!silent.includes('-c:a'));
});

test('HDR10 arguments expose no caller-controlled surface and reject bad geometry', () => {
  assert.throws(() => buildHdr10FfmpegArgs({ ...options(), width: 1921 }), /even dimensions/);
  assert.throws(() => buildHdr10FfmpegArgs({ ...options(), height: 0 }), /height/);
  assert.throws(() => buildHdr10FfmpegArgs({ ...options(), cq: 52 }), /cq/);
  assert.throws(
    () => buildHdr10FfmpegArgs({ ...options(), outputPath: 'relative.mp4' }),
    /absolute path/,
  );
  // Frame size the encoder will demand of every write.
  assert.equal(p010Layout(3840, 2160).byteLength, 3840 * 2160 * 3);
});

test('the panel offers HDR first, honestly labelled, and pairs each id with its encoder', () => {
  assert.equal(DEFAULT_EXPORT_PROFILE, 'av1-hdr10-1080p120');
  assert.deepEqual(
    EXPORT_PROFILE_CHOICES.map((choice) => choice.profile),
    [
      'av1-hdr10-1080p120',
      'av1-hdr10-2160p120',
      'av1-sdr-debug-1080p120',
      'av1-sdr-debug-2160p120',
    ],
  );
  for (const choice of EXPORT_PROFILE_CHOICES) {
    const hdr = isHdrExportProfile(choice.profile);
    assert.equal(/HDR10/.test(choice.label), hdr, `${choice.profile} label is dishonest`);
    assert.equal(/SDR debug/.test(choice.label), !hdr);
  }
  assert.deepEqual(debugExportOutput('av1-hdr10-2160p120'), {
    profile: 'av1-hdr10-2160p120',
    encoder: 'av1_nvenc',
    paperWhiteNits: 203,
    masteringPeakNits: 1000,
  });
});

test('an HDR profile is offered only when the server advertises its whole path', () => {
  const base = {
    available: true,
    profiles: ['av1-hdr10-1080p120', 'av1-sdr-debug-1080p120'] as string[],
    encoders: ['av1_nvenc'],
    rendererBuild: 'build-1',
    gpu: '',
    backend: '',
    transport: SDR_DEBUG_TRANSPORT,
    reason: '',
  } as const;
  const caps = (patch: Partial<typeof base>): Parameters<typeof exportProfileAvailable>[0] =>
    ({ ...base, ...patch }) as unknown as Parameters<typeof exportProfileAvailable>[0];

  assert.equal(exportProfileAvailable(caps({}), 'av1-hdr10-1080p120'), true);
  assert.equal(exportProfileAvailable(caps({}), 'av1-sdr-debug-1080p120'), true);
  // Not advertised: an older or less capable renderer.
  assert.equal(exportProfileAvailable(caps({}), 'av1-hdr10-2160p120'), false);
  // Advertised, but the encoder it needs is gone.
  assert.equal(exportProfileAvailable(caps({ encoders: [] }), 'av1-hdr10-1080p120'), false);
  assert.equal(exportProfileAvailable(caps({ available: false }), 'av1-hdr10-1080p120'), false);
  assert.equal(exportProfileAvailable(caps({ rendererBuild: '' }), 'av1-hdr10-1080p120'), false);
  // The SDR transport string still gates the SDR profiles, and only those.
  assert.equal(
    exportProfileAvailable(caps({ transport: 'something-else' }), 'av1-sdr-debug-1080p120'),
    false,
  );
  assert.equal(
    exportProfileAvailable(caps({ transport: 'something-else' }), 'av1-hdr10-1080p120'),
    true,
  );
});
