import { spawn, type ChildProcess } from 'node:child_process';
import { access, rename, rm, stat } from 'node:fs/promises';
import { extname, isAbsolute } from 'node:path';

import {
  DEFAULT_MASTERING_PEAK_NITS,
  MAX_MASTERING_PEAK_NITS,
  MIN_MASTERING_PEAK_NITS,
  masteringDisplayMetadata,
  p010Layout,
} from './hdr.ts';

export const AV1_DEBUG_FPS = 120;
export const EXPORT_FPS = 120;
export const MAX_AV1_DEBUG_RANGE_SECONDS = 6 * 60 * 60;
const MAX_STDERR_CHARACTERS = 64 * 1024;

export type Av1NvencPreset = 'p1' | 'p2' | 'p3' | 'p4' | 'p5' | 'p6' | 'p7';
export type FfmpegEncoderState =
  | 'running'
  | 'finishing'
  | 'complete'
  | 'failed'
  | 'aborted';

export interface Av1DebugEncoderOptions {
  /** Trusted, server-configured absolute path to FFmpeg. */
  readonly ffmpegExecutable: string;
  /** Trusted, server-configured absolute worker directory. */
  readonly workingDirectory: string;
  /** Trusted, unique absolute final path ending in .mp4. */
  readonly outputPath: string;
  readonly width: number;
  readonly height: number;
  /**
   * Omit this and both range fields for the explicit video-only debug mode.
   * Supplying audio requires the complete trusted WAV/range tuple.
   */
  readonly audioPath?: string;
  readonly startSeconds?: number;
  readonly durationSeconds?: number;
  readonly preset?: Av1NvencPreset;
  /** NVENC constant-quality target, conservatively project-bounded to 0-51. */
  readonly cq?: number;
}

export interface Hdr10EncoderOptions extends Av1DebugEncoderOptions {
  /** Peak the ST 2086 mastering-display metadata will claim. */
  readonly masteringPeakNits?: number;
}

interface ValidatedEncoderOptions {
  readonly ffmpegExecutable: string;
  readonly workingDirectory: string;
  readonly outputPath: string;
  readonly width: number;
  readonly height: number;
  readonly audio: {
    readonly path: string;
    readonly startSeconds: number;
    readonly durationSeconds: number;
  } | null;
  readonly preset: Av1NvencPreset;
  readonly cq: number;
}

interface ProcessExit {
  readonly code: number | null;
  readonly signal: NodeJS.Signals | null;
}

function validateDimension(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 8192) {
    throw new RangeError(`${name} must be an integer in [1, 8192]`);
  }
  return value;
}

function validateAudioRange(options: Av1DebugEncoderOptions): ValidatedEncoderOptions['audio'] {
  const supplied = [
    options.audioPath !== undefined,
    options.startSeconds !== undefined,
    options.durationSeconds !== undefined,
  ];
  if (supplied.every((value) => !value)) return null;
  if (!supplied.every(Boolean)) {
    throw new Error(
      'audioPath, startSeconds, and durationSeconds must be supplied together',
    );
  }

  const audioPath = options.audioPath as string;
  const startSeconds = options.startSeconds as number;
  const durationSeconds = options.durationSeconds as number;
  if (!isAbsolute(audioPath)) throw new Error('audioPath must be an absolute path');
  if (extname(audioPath).toLowerCase() !== '.wav') {
    throw new Error('audioPath must end in .wav');
  }
  if (
    !Number.isFinite(startSeconds) ||
    startSeconds < 0 ||
    startSeconds > MAX_AV1_DEBUG_RANGE_SECONDS
  ) {
    throw new RangeError(
      `startSeconds must be finite and in [0, ${MAX_AV1_DEBUG_RANGE_SECONDS}]`,
    );
  }
  if (
    !Number.isFinite(durationSeconds) ||
    durationSeconds <= 0 ||
    durationSeconds > MAX_AV1_DEBUG_RANGE_SECONDS
  ) {
    throw new RangeError(
      `durationSeconds must be finite and in (0, ${MAX_AV1_DEBUG_RANGE_SECONDS}]`,
    );
  }
  if (startSeconds + durationSeconds > MAX_AV1_DEBUG_RANGE_SECONDS) {
    throw new RangeError(
      `audio range must end no later than ${MAX_AV1_DEBUG_RANGE_SECONDS} seconds`,
    );
  }
  return { path: audioPath, startSeconds, durationSeconds };
}

function validateOptions(
  options: Av1DebugEncoderOptions,
  defaultCq: number,
): ValidatedEncoderOptions {
  for (const [name, value] of [
    ['ffmpegExecutable', options.ffmpegExecutable],
    ['workingDirectory', options.workingDirectory],
    ['outputPath', options.outputPath],
  ] as const) {
    if (!isAbsolute(value)) throw new Error(`${name} must be an absolute path`);
  }
  if (!options.outputPath.toLowerCase().endsWith('.mp4')) {
    throw new Error('outputPath must end in .mp4');
  }
  const preset = options.preset ?? 'p5';
  if (!/^p[1-7]$/.test(preset)) throw new Error('preset must be p1 through p7');
  const cq = options.cq ?? defaultCq;
  if (!Number.isSafeInteger(cq) || cq < 0 || cq > 51) {
    throw new RangeError('cq must be an integer in [0, 51]');
  }
  return {
    ffmpegExecutable: options.ffmpegExecutable,
    workingDirectory: options.workingDirectory,
    outputPath: options.outputPath,
    width: validateDimension(options.width, 'width'),
    height: validateDimension(options.height, 'height'),
    audio: validateAudioRange(options),
    preset,
    cq,
  };
}

export function partialOutputPath(outputPath: string): string {
  return `${outputPath}.partial`;
}

/**
 * `extra` carries per-input options such as `-mastering_display`, which FFmpeg
 * only accepts before the `-i` they describe.
 */
function rawInputArgs(
  pixelFormat: string,
  value: ValidatedEncoderOptions,
  extra: readonly string[] = [],
): string[] {
  return [
    '-hide_banner',
    '-loglevel',
    'warning',
    '-nostdin',
    '-n',
    '-f',
    'rawvideo',
    '-pixel_format',
    pixelFormat,
    '-video_size',
    `${value.width}x${value.height}`,
    '-framerate',
    String(EXPORT_FPS),
    ...extra,
    '-i',
    'pipe:0',
  ];
}

function audioInputArgs(value: ValidatedEncoderOptions): string[] {
  if (!value.audio) return ['-map', '0:v:0', '-an'];
  return [
    '-i',
    value.audio.path,
    '-map',
    '0:v:0',
    '-map',
    '1:a:0',
    '-af',
    `atrim=start=${value.audio.startSeconds}:duration=${value.audio.durationSeconds},` +
      'asetpts=PTS-STARTPTS',
  ];
}

function audioCodecArgs(value: ValidatedEncoderOptions): string[] {
  if (!value.audio) return [];
  return [
    '-c:a',
    'aac',
    '-profile:a',
    'aac_low',
    '-ar',
    '48000',
    '-b:a',
    '192k',
    '-t',
    String(value.audio.durationSeconds),
  ];
}

/** Fixed, server-owned argument set for the temporary SDR engineering profile. */
export function buildAv1DebugFfmpegArgs(options: Av1DebugEncoderOptions): string[] {
  const value = validateOptions(options, 24);
  const args = [...rawInputArgs('rgba', value), ...audioInputArgs(value)];
  args.push(
    '-c:v',
    'av1_nvenc',
    '-preset',
    value.preset,
    '-tune',
    'hq',
    '-rc',
    'vbr',
    '-cq',
    String(value.cq),
    '-b:v',
    '0',
    '-pix_fmt',
    'yuv420p',
    '-color_primaries',
    'bt709',
    '-color_trc',
    'bt709',
    '-colorspace',
    'bt709',
    '-r',
    String(EXPORT_FPS),
    '-fps_mode',
    'cfr',
  );
  args.push(...audioCodecArgs(value));
  args.push(
    '-movflags',
    '+faststart',
    '-f',
    'mp4',
    partialOutputPath(value.outputPath),
  );
  return args;
}

/**
 * ISO/IEC 23091-2 code points for the signal this pipeline produces. AV1 uses
 * the same CICP registry as HEVC, so these are also the values the AV1 sequence
 * header's `color_config` (spec section 6.4.2) carries.
 */
export const COLOUR_PRIMARIES_BT2020 = 9;
export const TRANSFER_CHARACTERISTICS_SMPTE2084 = 16;
export const MATRIX_COEFFICIENTS_BT2020_NCL = 9;

/**
 * `-mastering_display` value: SMPTE ST 2086 in FFmpeg's documented format,
 * `G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)`, chromaticities in 0.00002 units and
 * luminance in 0.0001 cd/m² — exactly the units `masteringDisplayMetadata`
 * already returns.
 */
export function masteringDisplayArgument(masteringPeakNits: number): string {
  const m = masteringDisplayMetadata(masteringPeakNits);
  return (
    `G(${m.greenX},${m.greenY})` +
    `B(${m.blueX},${m.blueY})` +
    `R(${m.redX},${m.redY})` +
    `WP(${m.whiteX},${m.whiteY})` +
    `L(${m.maxLuminance},${m.minLuminance})`
  );
}

/**
 * Fixed, server-owned argument set for the real HDR10 profile.
 *
 * Four details are load-bearing:
 *
 * - the input is already P010LE, so FFmpeg inserts no scaler and performs no
 *   range or matrix conversion between the packing shader and NVENC;
 * - AV1 Main covers 10-bit, so unlike HEVC there is no `main10` profile to
 *   request — asking for one is what makes an AV1 command line look strange;
 * - NVENC writes only the matrix and range into the AV1 sequence header, so the
 *   `av1_metadata` bitstream filter restamps the full colour description with
 *   BT.2020 primaries and the ST 2084 transfer. This is the direct analogue of
 *   `hevc_metadata` and has shipped in FFmpeg for years;
 * - `-mastering_display` is an *input* option. It attaches ST 2086 side data to
 *   the raw stream, which the MP4 muxer then writes as a real `mdcv` box. That
 *   is why this profile can ask for `+faststart` like any other modern encode
 *   instead of reserving a trailing `moov` for post-hoc container surgery.
 *
 * No `-content_light` is passed: MaxCLL and MaxFALL must be measured from the
 * encoded content, and this pipeline does not measure them. Both are optional in
 * HDR10, and claiming an unmeasured number would be worse than omitting it.
 */
export function buildHdr10FfmpegArgs(options: Hdr10EncoderOptions): string[] {
  const value = validateOptions(options, 20);
  if (value.width % 2 !== 0 || value.height % 2 !== 0) {
    throw new RangeError('HDR10 4:2:0 output requires even dimensions');
  }
  const masteringPeakNits = validateMasteringPeak(options.masteringPeakNits);
  const args = [
    ...rawInputArgs('p010le', value, [
      '-mastering_display',
      masteringDisplayArgument(masteringPeakNits),
    ]),
    ...audioInputArgs(value),
  ];
  args.push(
    '-c:v',
    'av1_nvenc',
    '-preset',
    value.preset,
    '-tune',
    'hq',
    '-rc',
    'vbr',
    '-cq',
    String(value.cq),
    '-b:v',
    '0',
    '-pix_fmt',
    'p010le',
    '-color_primaries',
    'bt2020',
    '-color_trc',
    'smpte2084',
    '-colorspace',
    'bt2020nc',
    '-color_range',
    'tv',
    '-bsf:v',
    `av1_metadata=color_primaries=${COLOUR_PRIMARIES_BT2020}:` +
      `transfer_characteristics=${TRANSFER_CHARACTERISTICS_SMPTE2084}:` +
      `matrix_coefficients=${MATRIX_COEFFICIENTS_BT2020_NCL}:color_range=tv`,
    '-r',
    String(EXPORT_FPS),
    '-fps_mode',
    'cfr',
  );
  args.push(...audioCodecArgs(value));
  args.push('-movflags', '+faststart', '-f', 'mp4', partialOutputPath(value.outputPath));
  return args;
}

function validateMasteringPeak(value: number | undefined): number {
  const peak = value ?? DEFAULT_MASTERING_PEAK_NITS;
  if (
    !Number.isFinite(peak) ||
    peak < MIN_MASTERING_PEAK_NITS ||
    peak > MAX_MASTERING_PEAK_NITS
  ) {
    throw new RangeError('masteringPeakNits is outside the supported range');
  }
  return peak;
}

async function assertPathAbsent(path: string, name: string): Promise<void> {
  try {
    await access(path);
  } catch (error: unknown) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return;
    throw error;
  }
  throw new Error(`${name} already exists: ${path}`);
}

function waitForSpawn(child: ChildProcess): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSpawn = (): void => {
      child.off('error', onError);
      resolve();
    };
    const onError = (error: Error): void => {
      child.off('spawn', onSpawn);
      reject(error);
    };
    child.once('spawn', onSpawn);
    child.once('error', onError);
  });
}

interface FfmpegEncoderSpec {
  readonly ffmpegExecutable: string;
  readonly workingDirectory: string;
  readonly outputPath: string;
  readonly bytesPerFrame: number;
  readonly args: readonly string[];
  readonly audioPath: string | null;
  /**
   * Runs on the `.partial` file after a clean FFmpeg exit and before the atomic
   * rename, so a failure here can never publish a half-described HDR file.
   */
  readonly finalize?: (partialPath: string) => Promise<void>;
}

/**
 * One FFmpeg child fed tightly packed frames over stdin.
 *
 * The argument list is chosen entirely by the builders above; nothing reaching
 * this class comes from a browser. Frame writes are sequential and awaited so
 * pipe backpressure is the render loop's throttle.
 */
export class FfmpegFrameEncoder {
  readonly outputPath: string;
  readonly partialPath: string;
  readonly bytesPerFrame: number;

  private readonly child: ChildProcess;
  private readonly exitPromise: Promise<ProcessExit>;
  private readonly finalizeOutput: ((partialPath: string) => Promise<void>) | null;
  private stderrTail = '';
  private lifecycle: FfmpegEncoderState = 'running';
  private writeInProgress = false;
  private writtenFrames = 0;

  private constructor(child: ChildProcess, spec: FfmpegEncoderSpec) {
    this.child = child;
    this.outputPath = spec.outputPath;
    this.partialPath = partialOutputPath(spec.outputPath);
    this.bytesPerFrame = spec.bytesPerFrame;
    this.finalizeOutput = spec.finalize ?? null;
    this.exitPromise = new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) => resolve({ code, signal }));
    });
    // A later write/finish/abort observes this promise. The immediate handler
    // prevents a spawn error from becoming an unhandled rejection meanwhile.
    void this.exitPromise.catch(() => undefined);
    // Individual writes still receive their callback error, while this listener
    // prevents an early FFmpeg exit from turning the pipe's EPIPE event into an
    // uncaught process-level exception before writeFrame can clean up.
    child.stdin?.on('error', () => undefined);
    child.stderr?.on('data', (chunk: Buffer | string) => {
      this.stderrTail += chunk.toString();
      if (this.stderrTail.length > MAX_STDERR_CHARACTERS) {
        this.stderrTail = this.stderrTail.slice(-MAX_STDERR_CHARACTERS);
      }
    });
  }

  static async start(spec: FfmpegEncoderSpec): Promise<FfmpegFrameEncoder> {
    const partialPath = partialOutputPath(spec.outputPath);
    await Promise.all([
      assertPathAbsent(spec.outputPath, 'final output'),
      assertPathAbsent(partialPath, 'partial output'),
    ]);
    if (spec.audioPath) {
      const audioInfo = await stat(spec.audioPath);
      if (!audioInfo.isFile()) throw new Error(`audioPath must identify a regular file`);
    }
    const child = spawn(spec.ffmpegExecutable, [...spec.args], {
      cwd: spec.workingDirectory,
      windowsHide: true,
      shell: false,
      stdio: ['pipe', 'ignore', 'pipe'],
    });
    const encoder = new FfmpegFrameEncoder(child, spec);
    try {
      await waitForSpawn(child);
      if (!child.stdin || !child.stderr) {
        throw new Error('FFmpeg did not expose the required stdin/stderr pipes');
      }
      return encoder;
    } catch (error: unknown) {
      await rm(partialPath, { force: true });
      throw error;
    }
  }

  get state(): FfmpegEncoderState {
    return this.lifecycle;
  }

  get framesWritten(): number {
    return this.writtenFrames;
  }

  /** Write exactly one tightly packed frame and await pipe backpressure. */
  async writeFrame(frame: Uint8Array): Promise<void> {
    if (this.lifecycle !== 'running') {
      throw new Error(`cannot write a frame while encoder is ${this.lifecycle}`);
    }
    if (this.writeInProgress) throw new Error('frame writes must be awaited sequentially');
    if (frame.byteLength !== this.bytesPerFrame) {
      throw new RangeError(`frame has ${frame.byteLength} bytes; expected ${this.bytesPerFrame}`);
    }
    const stdin = this.child.stdin;
    if (!stdin) throw new Error('FFmpeg stdin is unavailable');

    this.writeInProgress = true;
    try {
      await new Promise<void>((resolve, reject) => {
        stdin.write(frame, (error: Error | null | undefined) => {
          if (error) reject(error);
          else resolve();
        });
      });
      this.writtenFrames++;
    } catch (error: unknown) {
      await this.failAndCleanUp();
      throw this.encoderError('FFmpeg frame write failed', error);
    } finally {
      this.writeInProgress = false;
    }
  }

  /** Finish stdin, require a clean FFmpeg exit, then atomically publish the file. */
  async finish(): Promise<void> {
    if (this.lifecycle !== 'running') {
      throw new Error(`cannot finish encoder while it is ${this.lifecycle}`);
    }
    if (this.writeInProgress) throw new Error('cannot finish while a frame write is in progress');
    this.lifecycle = 'finishing';
    try {
      const stdin = this.child.stdin;
      if (!stdin) throw new Error('FFmpeg stdin is unavailable');
      await new Promise<void>((resolve, reject) => {
        stdin.once('error', reject);
        stdin.end(() => resolve());
      });
      const result = await this.exitPromise;
      if (result.code !== 0) {
        throw new Error(
          `FFmpeg exited with code ${String(result.code)} signal ${result.signal ?? 'none'}`,
        );
      }
      if (this.finalizeOutput) await this.finalizeOutput(this.partialPath);
      await rename(this.partialPath, this.outputPath);
      this.lifecycle = 'complete';
    } catch (error: unknown) {
      await this.failAndCleanUp();
      throw this.encoderError('FFmpeg encode failed', error);
    }
  }

  /** Terminate an unfinished encoder and remove only its known partial output. */
  async abort(): Promise<void> {
    if (this.lifecycle === 'complete' || this.lifecycle === 'aborted') return;
    this.lifecycle = 'aborted';
    this.child.stdin?.destroy();
    if (this.child.exitCode === null && this.child.signalCode === null) this.child.kill();
    await this.exitPromise.catch(() => undefined);
    await rm(this.partialPath, { force: true });
  }

  private async failAndCleanUp(): Promise<void> {
    this.lifecycle = 'failed';
    this.child.stdin?.destroy();
    if (this.child.exitCode === null && this.child.signalCode === null) this.child.kill();
    await this.exitPromise.catch(() => undefined);
    await rm(this.partialPath, { force: true });
  }

  private encoderError(message: string, cause: unknown): Error {
    const detail = cause instanceof Error ? cause.message : String(cause);
    const diagnostics = this.stderrTail.trim();
    return new Error(`${message}: ${detail}${diagnostics ? `\n${diagnostics}` : ''}`, { cause });
  }
}

/** The temporary RGBA8 → AV1 SDR engineering transport. */
export const Av1DebugEncoder = {
  async start(options: Av1DebugEncoderOptions): Promise<FfmpegFrameEncoder> {
    const value = validateOptions(options, 24);
    return FfmpegFrameEncoder.start({
      ffmpegExecutable: value.ffmpegExecutable,
      workingDirectory: value.workingDirectory,
      outputPath: value.outputPath,
      bytesPerFrame: value.width * value.height * 4,
      args: buildAv1DebugFfmpegArgs(options),
      audioPath: value.audio?.path ?? null,
    });
  },
};
export type Av1DebugEncoder = FfmpegFrameEncoder;

/**
 * The real P010 → AV1 10-bit HDR10 transport.
 *
 * There is no finalize step: every byte of colour description this file needs is
 * written by FFmpeg itself, in the sequence header and in the muxer's `mdcv`.
 */
export const Hdr10Encoder = {
  async start(options: Hdr10EncoderOptions): Promise<FfmpegFrameEncoder> {
    const value = validateOptions(options, 20);
    validateMasteringPeak(options.masteringPeakNits);
    // Exactly what the packing shader writes: a 16-bit luma plane plus a
    // half-height interleaved 16-bit chroma plane, three bytes per pixel.
    const bytesPerFrame = p010Layout(value.width, value.height).byteLength;
    return FfmpegFrameEncoder.start({
      ffmpegExecutable: value.ffmpegExecutable,
      workingDirectory: value.workingDirectory,
      outputPath: value.outputPath,
      bytesPerFrame,
      args: buildHdr10FfmpegArgs(options),
      audioPath: value.audio?.path ?? null,
    });
  },
};
export type Hdr10Encoder = FfmpegFrameEncoder;
