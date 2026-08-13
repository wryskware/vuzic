import { spawn, type ChildProcess } from 'node:child_process';
import { access, rename, rm } from 'node:fs/promises';
import { isAbsolute } from 'node:path';

export const AV1_DEBUG_FPS = 120;
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
  readonly preset?: Av1NvencPreset;
  /** NVENC constant-quality target, conservatively project-bounded to 0-51. */
  readonly cq?: number;
}

interface ValidatedEncoderOptions {
  readonly ffmpegExecutable: string;
  readonly workingDirectory: string;
  readonly outputPath: string;
  readonly width: number;
  readonly height: number;
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

function validateOptions(options: Av1DebugEncoderOptions): ValidatedEncoderOptions {
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
  const cq = options.cq ?? 24;
  if (!Number.isSafeInteger(cq) || cq < 0 || cq > 51) {
    throw new RangeError('cq must be an integer in [0, 51]');
  }
  return {
    ffmpegExecutable: options.ffmpegExecutable,
    workingDirectory: options.workingDirectory,
    outputPath: options.outputPath,
    width: validateDimension(options.width, 'width'),
    height: validateDimension(options.height, 'height'),
    preset,
    cq,
  };
}

export function partialOutputPath(outputPath: string): string {
  return `${outputPath}.partial`;
}

/** Fixed, server-owned argument set for the temporary SDR engineering profile. */
export function buildAv1DebugFfmpegArgs(options: Av1DebugEncoderOptions): string[] {
  const value = validateOptions(options);
  return [
    '-hide_banner',
    '-loglevel',
    'warning',
    '-nostdin',
    '-n',
    '-f',
    'rawvideo',
    '-pixel_format',
    'rgba',
    '-video_size',
    `${value.width}x${value.height}`,
    '-framerate',
    String(AV1_DEBUG_FPS),
    '-i',
    'pipe:0',
    '-an',
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
    String(AV1_DEBUG_FPS),
    '-fps_mode',
    'cfr',
    '-movflags',
    '+faststart',
    '-f',
    'mp4',
    partialOutputPath(value.outputPath),
  ];
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

export class Av1DebugEncoder {
  readonly outputPath: string;
  readonly partialPath: string;
  readonly bytesPerFrame: number;

  private readonly child: ChildProcess;
  private readonly exitPromise: Promise<ProcessExit>;
  private stderrTail = '';
  private lifecycle: FfmpegEncoderState = 'running';
  private writeInProgress = false;
  private writtenFrames = 0;

  private constructor(child: ChildProcess, options: ValidatedEncoderOptions) {
    this.child = child;
    this.outputPath = options.outputPath;
    this.partialPath = partialOutputPath(options.outputPath);
    this.bytesPerFrame = options.width * options.height * 4;
    this.exitPromise = new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) => resolve({ code, signal }));
    });
    // A later write/finish/abort observes this promise. The immediate handler
    // prevents a spawn error from becoming an unhandled rejection meanwhile.
    void this.exitPromise.catch(() => undefined);
    child.stderr?.on('data', (chunk: Buffer | string) => {
      this.stderrTail += chunk.toString();
      if (this.stderrTail.length > MAX_STDERR_CHARACTERS) {
        this.stderrTail = this.stderrTail.slice(-MAX_STDERR_CHARACTERS);
      }
    });
  }

  static async start(options: Av1DebugEncoderOptions): Promise<Av1DebugEncoder> {
    const value = validateOptions(options);
    const partialPath = partialOutputPath(value.outputPath);
    await Promise.all([
      assertPathAbsent(value.outputPath, 'final output'),
      assertPathAbsent(partialPath, 'partial output'),
    ]);
    const child = spawn(value.ffmpegExecutable, buildAv1DebugFfmpegArgs(value), {
      cwd: value.workingDirectory,
      windowsHide: true,
      shell: false,
      stdio: ['pipe', 'ignore', 'pipe'],
    });
    const encoder = new Av1DebugEncoder(child, value);
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

  /** Write exactly one tightly packed RGBA8 frame and await pipe backpressure. */
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
