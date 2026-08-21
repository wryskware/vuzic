import { execFile } from 'node:child_process';
import { rm } from 'node:fs/promises';

import postCommonWgsl from '../sim/render/shaders/post-common.wgsl?raw';
import exposureWgsl from '../sim/render/shaders/exposure.wgsl?raw';
import gradeHdrWgsl from '../sim/render/shaders/grade-hdr.wgsl?raw';
import hdrPackWgsl from './shaders/hdr-pack.wgsl?raw';
import { buildDriverBank } from '../mapping/modulation.ts';
import { buildSimBundleFromRecipe, type SimBundle } from '../runtime/sim-bundle.ts';
import { SECONDS_PER_TICK } from '../timing.ts';
import { TimelineSampler } from '../timeline/sampler.ts';
import gate0Wgsl from './shaders/gate0.wgsl?raw';

import { createDawnContext, type DawnRuntimeContext } from './dawn-context.ts';
import { ExportClock, exportFrameCount } from './export-clock.ts';
import {
  Av1DebugEncoder,
  Hdr10Encoder,
  partialOutputPath,
  type FfmpegFrameEncoder,
} from './ffmpeg-encoder.ts';
import { FrameReadbackRing, PackedFrameReadbackRing } from './frame-readback.ts';
import { Hdr10FramePacker } from './hdr-output.ts';
import { loadTimelineFromFiles } from './node-timeline-loader.ts';
import { exportProfileSpec } from './profiles.ts';
import {
  loadExportWorkerRequest,
} from './worker-request.ts';

interface ProbeOptions {
  width: number;
  height: number;
  frames: number;
  ringSize: number;
  backend?: string;
}

interface ReadbackSlot {
  buffer: GPUBuffer;
  pending: Promise<number> | null;
}

interface FfmpegCapabilities {
  version: string;
  encoders: string[];
  /** Encoders this build can actually feed 10-bit 4:2:0; the HDR gate. */
  tenBitEncoders: string[];
}

const emit = (message: Record<string, unknown>): void => {
  process.stdout.write(`${JSON.stringify(message)}\n`);
};

// stdout is a machine protocol. Shared runtime diagnostics were written for a
// browser console, so route every non-error console lane to stderr in this host.
const stderrDiagnostic = console.error.bind(console);
console.log = stderrDiagnostic;
console.info = stderrDiagnostic;
console.debug = stderrDiagnostic;

let activeStage = 'startup';

/**
 * Exit code for "a cancellation signal reached me", distinct from the 1 a crash
 * produces so the supervising server can tell a cancelled render from a broken
 * one even when it was not the party that asked.
 */
export const CANCELLED_EXIT_CODE = 130;
/** Ceiling on cooperative shutdown; the server force-kills the tree after its own. */
const SHUTDOWN_DEADLINE_MS = 3000;
const CANCEL_SIGNALS = ['SIGBREAK', 'SIGTERM', 'SIGINT'] as const;

let cancelling = false;
/**
 * What a cancellation has to undo, registered as it comes into existence.
 *
 * A signal can arrive at any await in a render that is minutes long, so the
 * handler cannot reach into `runRequest`'s locals — the request path publishes
 * them here instead, and cleanup runs against whatever exists at that instant.
 */
const cancellationCleanups = new Set<() => Promise<void>>();

export function registerCancellationCleanup(cleanup: () => Promise<void>): () => void {
  cancellationCleanups.add(cleanup);
  return () => {
    cancellationCleanups.delete(cleanup);
  };
}

export function isCancelling(): boolean {
  return cancelling;
}

async function shutdownForCancellation(signal: string): Promise<void> {
  emit({
    type: 'error',
    stage: activeStage,
    cancelled: true,
    message: `export cancelled by ${signal}`,
  });
  // Cleanup is best-effort under a deadline: FFmpeg has already been signalled
  // by the same console control event on Windows, and a stuck unlink must not
  // be the reason a cancelled job needs force-killing.
  await Promise.race([
    Promise.allSettled([...cancellationCleanups].map((cleanup) => cleanup())),
    new Promise((resolve) => setTimeout(resolve, SHUTDOWN_DEADLINE_MS)),
  ]);
}

function installCancellationHandlers(): void {
  for (const signal of CANCEL_SIGNALS) {
    process.on(signal, () => {
      if (cancelling) return;
      cancelling = true;
      void shutdownForCancellation(signal).finally(() => {
        process.exit(CANCELLED_EXIT_CODE);
      });
    });
  }
}

function execFileText(executable: string, args: readonly string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      executable,
      [...args],
      { windowsHide: true, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`${executable} failed: ${stderr.trim() || error.message}`));
          return;
        }
        resolve(`${stdout}\n${stderr}`);
      },
    );
  });
}

async function probeFfmpeg(): Promise<FfmpegCapabilities> {
  const [versionText, encodersText] = await Promise.all([
    execFileText('ffmpeg', ['-hide_banner', '-version']),
    execFileText('ffmpeg', ['-hide_banner', '-encoders']),
  ]);
  const encoders = ['av1_nvenc'].filter((name) =>
    new RegExp(`\\b${name}\\b`).test(encodersText),
  );
  if (!encoders.includes('av1_nvenc')) {
    throw new Error('FFmpeg does not expose the required av1_nvenc encoder');
  }
  // Advertising an HDR profile this build cannot actually describe would turn a
  // capability question into a failure minutes into a render, so ask first. Two
  // things are required and both are build-time facts: the encoder must accept
  // P010 input, and `av1_metadata` must exist to stamp the sequence header's
  // colour description. AV1 Main already covers 10 bits, so unlike HEVC there is
  // no separate profile to look for.
  const bsfsText = await execFileText('ffmpeg', ['-hide_banner', '-bsfs']).catch(() => '');
  const hasAv1Metadata = /\bav1_metadata\b/.test(bsfsText);
  const tenBitEncoders: string[] = [];
  for (const name of encoders) {
    const help = await execFileText('ffmpeg', ['-hide_banner', '-h', `encoder=${name}`]).catch(
      () => '',
    );
    const pixelFormats = /Supported pixel formats:([^\n]*)/.exec(help)?.[1] ?? '';
    if (/\bp010le\b/.test(pixelFormats) && hasAv1Metadata) tenBitEncoders.push(name);
  }
  return {
    version: versionText.split(/\r?\n/, 1)[0] ?? 'unknown FFmpeg',
    encoders,
    tenBitEncoders,
  };
}

const positiveInteger = (raw: string | undefined, fallback: number, name: string, max: number): number => {
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 1 || value > max) {
    throw new Error(`${name} must be an integer in [1, ${max}]`);
  }
  return value;
};

function optionValue(args: readonly string[], name: string): string | undefined {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  if (value === undefined || value.startsWith('--')) throw new Error(`${name} needs a value`);
  return value;
}

function parseProbeOptions(args: readonly string[]): ProbeOptions {
  const backend = optionValue(args, '--backend');
  if (backend !== undefined && !/^[a-z0-9-]+$/i.test(backend)) {
    throw new Error('--backend contains unsupported characters');
  }
  return {
    width: positiveInteger(optionValue(args, '--width'), 256, '--width', 8192),
    height: positiveInteger(optionValue(args, '--height'), 144, '--height', 8192),
    frames: positiveInteger(optionValue(args, '--frames'), 6, '--frames', 600),
    ringSize: positiveInteger(optionValue(args, '--ring-size'), 3, '--ring-size', 16),
    ...(backend === undefined ? {} : { backend }),
  };
}

async function compileCurrentProjectShader(device: GPUDevice): Promise<void> {
  const module = device.createShaderModule({
    label: 'gate0.current-project-exposure',
    code: `${postCommonWgsl}\n${exposureWgsl}`,
  });
  const messages = await module.getCompilationInfo();
  const errors = messages.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `current project WGSL failed to compile: ${errors.map((message) => message.message).join('; ')}`,
    );
  }

  device.pushErrorScope('validation');
  device.createComputePipeline({
    label: 'gate0.current-project-exposure',
    layout: 'auto',
    compute: { module, entryPoint: 'measure' },
  });
  const validationError = await device.popErrorScope();
  if (validationError) throw validationError;

  // The HDR output path is a capability, so its shaders are proven at probe time
  // rather than several minutes into a 4K render.
  await compileOrThrow(device, 'gate0.hdr-grade', `${postCommonWgsl}\n${gradeHdrWgsl}`);
  const packModule = await compileOrThrow(device, 'gate0.hdr-pack', hdrPackWgsl);
  device.pushErrorScope('validation');
  device.createComputePipeline({
    label: 'gate0.hdr-pack',
    layout: 'auto',
    compute: { module: packModule, entryPoint: 'pack' },
  });
  const packError = await device.popErrorScope();
  if (packError) throw packError;
}

async function compileOrThrow(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `${label} failed to compile: ${errors.map((message) => message.message).join('; ')}`,
    );
  }
  return module;
}

async function checksumReadback(buffer: GPUBuffer): Promise<number> {
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  let hash = 2166136261;
  // Sampling keeps a 4K probe cheap while still covering the whole mapped frame.
  const stride = Math.max(1, Math.floor(bytes.length / 16_384));
  for (let i = 0; i < bytes.length; i += stride) {
    hash ^= bytes[i] ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  buffer.unmap();
  return hash >>> 0;
}

async function runProbe(options: ProbeOptions): Promise<void> {
  activeStage = 'gate0-probe';
  let ctx: DawnRuntimeContext | null = null;
  const startedAt = performance.now();
  try {
    ctx = await createDawnContext(options);
    const ffmpeg = await probeFfmpeg();
    emit({
      type: 'ready',
      adapter: ctx.adapterName,
      backend: ctx.backend,
      float32Filterable: ctx.float32Filterable,
      ffmpeg: ffmpeg.version,
      encoders: ffmpeg.encoders,
      tenBitEncoders: ffmpeg.tenBitEncoders,
    });

    const { device, width, height } = ctx;
    await compileCurrentProjectShader(device);

    const module = device.createShaderModule({ label: 'gate0.hdr-pattern', code: gate0Wgsl });
    const pipeline = device.createRenderPipeline({
      label: 'gate0.hdr-pattern',
      layout: 'auto',
      vertex: { module, entryPoint: 'vsMain' },
      fragment: { module, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    const params = device.createBuffer({
      label: 'gate0.params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      label: 'gate0.params',
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: params } }],
    });
    const target = device.createTexture({
      label: 'gate0.rgba16float',
      size: { width, height },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const targetView = target.createView();

    const bytesPerPixel = 8;
    const unpaddedRowBytes = width * bytesPerPixel;
    const bytesPerRow = Math.ceil(unpaddedRowBytes / 256) * 256;
    const readbackBytes = bytesPerRow * height;
    const ring: ReadbackSlot[] = Array.from({ length: Math.min(options.ringSize, options.frames) }, (_, index) => ({
      buffer: device.createBuffer({
        label: `gate0.readback.${index}`,
        size: readbackBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: null,
    }));
    const checksums: number[] = [];

    for (let frame = 0; frame < options.frames; frame++) {
      const slot = ring[frame % ring.length];
      if (!slot) throw new Error('readback ring unexpectedly empty');
      if (slot.pending) {
        checksums.push(await slot.pending);
        slot.pending = null;
      }

      const phase = options.frames === 1 ? 0 : frame / (options.frames - 1);
      device.queue.writeBuffer(params, 0, new Float32Array([phase, width, height, 0]));
      const encoder = device.createCommandEncoder({ label: `gate0.frame.${frame}` });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      encoder.copyTextureToBuffer(
        { texture: target },
        { buffer: slot.buffer, bytesPerRow, rowsPerImage: height },
        { width, height },
      );
      device.queue.submit([encoder.finish()]);
      slot.pending = checksumReadback(slot.buffer);
      emit({ type: 'progress', stage: 'render-readback', frame: frame + 1, frames: options.frames });
    }

    for (const slot of ring) {
      if (slot.pending) checksums.push(await slot.pending);
      slot.buffer.destroy();
    }
    target.destroy();
    params.destroy();

    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    const totalBytes = readbackBytes * options.frames;
    emit({
      type: 'result',
      stage: 'gate0-probe',
      format: 'rgba16float',
      width,
      height,
      frames: options.frames,
      ringSize: ring.length,
      readbackBytes: totalBytes,
      readbackMiBPerSecond: totalBytes / (1024 * 1024) / Math.max(elapsedSeconds, 1e-6),
      distinctChecksums: new Set(checksums).size,
      elapsedSeconds,
    });
  } finally {
    ctx?.dispose();
  }
}

/**
 * Render one request.
 *
 * Two transports live here and they diverge at exactly one place: what the
 * simulation's final grade writes into, and therefore what reaches FFmpeg.
 *
 * - `sdr-rgba8-av1-debug` renders the 8-bit display grade into `rgba8unorm` and
 *   reads whole RGBA frames back. Temporary, honest, unchanged.
 * - `hdr10-p010-compute` renders the scene-linear BT.2020/PQ grade into
 *   `rgba16float` and converts on the GPU, so what crosses PCIe is the 10-bit
 *   4:2:0 the encoder wants — 24.9 MB per 4K frame rather than 66.4 MB.
 */
async function runRequest(requestPath: string): Promise<void> {
  activeStage = 'request';
  const request = await loadExportWorkerRequest(requestPath);
  const profile = exportProfileSpec(request.output.profile);
  if (request.recipe.output.encoder !== profile.encoder) {
    throw new Error(
      `profile ${profile.id} requires encoder ${profile.encoder}, ` +
        `not ${request.recipe.output.encoder}`,
    );
  }
  const hdr = profile.dynamicRange === 'hdr10';

  const timeline = await loadTimelineFromFiles(request.timeline);
  const rangeEndSeconds = request.range.startSeconds + request.range.durationSeconds;
  if (rangeEndSeconds > timeline.manifest.track.duration + 1e-9) {
    throw new Error(
      `requested range ends at ${rangeEndSeconds}s, beyond the ` +
        `${timeline.manifest.track.duration}s timeline`,
    );
  }

  const { width, height } = profile;
  const firstOutputFrame = exportFrameCount(request.range.startSeconds);
  const clock = new ExportClock(rangeEndSeconds);
  const outputFrames = clock.frameCount - firstOutputFrame;
  if (outputFrames < 1) throw new Error('requested range contains no 120 Hz output frames');

  let ctx: DawnRuntimeContext | null = null;
  let bundle: SimBundle | null = null;
  let target: GPUTexture | null = null;
  let readback: FrameReadbackRing | null = null;
  let packedReadback: PackedFrameReadbackRing | null = null;
  let packer: Hdr10FramePacker | null = null;
  let encoder: FfmpegFrameEncoder | null = null;
  const startedAt = performance.now();
  // Registered before the GPU exists: a cancel during device creation still has
  // to leave no `.partial` behind, and `rm --force` on an absent path is free.
  const releaseCleanup = registerCancellationCleanup(async () => {
    if (encoder && encoder.state !== 'complete') await encoder.abort();
    await rm(partialOutputPath(request.output.path), { force: true });
  });

  try {
    activeStage = 'gpu-init';
    ctx = await createDawnContext({
      width,
      height,
      format: hdr ? 'rgba16float' : 'rgba8unorm',
      ...(hdr
        ? {
            hdrOutput: {
              paperWhiteNits: request.recipe.output.paperWhiteNits,
              masteringPeakNits: request.recipe.output.masteringPeakNits,
            },
          }
        : {}),
    });
    // Checked, not assumed: if the HDR policy failed to reach the context the
    // post chain would quietly build the 8-bit SDR grade, the packing pass would
    // read gamma-encoded display values as if they were luminance, and the job
    // would publish a correctly tagged HDR file containing an SDR image.
    if (hdr && !ctx.hdrOutput) {
      throw new Error('HDR profile requested but the GPU context carries no HDR output policy');
    }
    const sampler = new TimelineSampler(timeline, SECONDS_PER_TICK);
    bundle = buildSimBundleFromRecipe({
      recipe: request.recipe,
      sampler,
      drivers: buildDriverBank(timeline),
      secondsPerTick: SECONDS_PER_TICK,
    });
    await bundle.sim.init(ctx);

    target = ctx.device.createTexture({
      label: hdr ? 'export.hdr10-target' : 'export.sdr-debug-target',
      size: { width, height },
      format: hdr ? 'rgba16float' : 'rgba8unorm',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        // The HDR path never copies the texture: a compute pass reads it and
        // writes the encoder-ready bytes, so only the packed buffer is read back.
        (hdr ? GPUTextureUsage.TEXTURE_BINDING : GPUTextureUsage.COPY_SRC),
    });
    const targetView = target.createView();

    const audioArgs =
      request.audioPath === undefined
        ? {}
        : {
            audioPath: request.audioPath,
            startSeconds: request.range.startSeconds,
            durationSeconds: request.range.durationSeconds,
          };

    if (hdr) {
      packer = new Hdr10FramePacker(ctx.device, { width, height });
      packedReadback = new PackedFrameReadbackRing(ctx.device, {
        byteLength: packer.byteLength,
        ringSize: 3,
        label: 'export.hdr10-readback',
      });
      encoder = await Hdr10Encoder.start({
        ffmpegExecutable: request.runtime.ffmpegExecutable,
        workingDirectory: request.runtime.workingDirectory,
        outputPath: request.output.path,
        width,
        height,
        masteringPeakNits: request.recipe.output.masteringPeakNits,
        ...audioArgs,
      });
    } else {
      readback = new FrameReadbackRing(ctx.device, {
        width,
        height,
        bytesPerPixel: 4,
        ringSize: 3,
        label: 'export.sdr-debug-readback',
      });
      encoder = await Av1DebugEncoder.start({
        ffmpegExecutable: request.runtime.ffmpegExecutable,
        workingDirectory: request.runtime.workingDirectory,
        outputPath: request.output.path,
        width,
        height,
        ...audioArgs,
      });
    }

    emit({
      type: 'ready',
      adapter: ctx.adapterName,
      backend: ctx.backend,
      profile: request.output.profile,
      transport: profile.transport,
      encoder: profile.encoder,
      dynamicRange: profile.dynamicRange,
      ...(hdr
        ? {
            paperWhiteNits: request.recipe.output.paperWhiteNits,
            masteringPeakNits: request.recipe.output.masteringPeakNits,
          }
        : {}),
      audio: request.audioPath !== undefined,
      width,
      height,
      frames: outputFrames,
    });

    activeStage = 'render';
    let stepIndex = 0;
    while (!clock.done) {
      const frame = clock.advanceThenRender(
        // The offline path stays FIXED-STEP and is deliberately the one place
        // that still does. `ExportClock` walks an exact integer cadence and hands
        // every simulation tick to this callback in order, so the dt handed down
        // is the constant `SECONDS_PER_TICK` rather than anything measured — the
        // live loop's variable-dt contract is about a display nobody is watching
        // here. Same recipe, same seed, same machine, same file.
        (tick) => {
          stepIndex++;
          const features = sampler.sampleAt(tick);
          bundle?.modulator.update(features, SECONDS_PER_TICK);
          bundle?.impulses.update(tick, SECONDS_PER_TICK);
          bundle?.sim.tick(features, stepIndex, SECONDS_PER_TICK);
        },
        (renderFrame) => {
          const command = ctx?.device.createCommandEncoder({
            label: `export.render.frame.${renderFrame.frameIndex}`,
          });
          if (!command || !bundle || !ctx) throw new Error('render resources were disposed early');
          bundle.sim.render(command, targetView, renderFrame);
          // Same submission as the grade that produced the pixels: the packing
          // pass reads the target as a texture, so no extra synchronisation and
          // no intermediate copy is involved.
          packer?.pack(command, targetView);
          ctx.device.queue.submit([command.finish()]);
        },
      );
      if (!frame) break;
      if (frame.frameIndex < firstOutputFrame) continue;

      const outputIndex = frame.frameIndex - firstOutputFrame;
      const completed = packedReadback
        ? await packedReadback.enqueue((packer as Hdr10FramePacker).packedBuffer, outputIndex)
        : await (readback as FrameReadbackRing).enqueue(target, outputIndex);
      if (completed) await encoder.writeFrame(completed.data);
      const rendered = frame.frameIndex - firstOutputFrame + 1;
      if (rendered === outputFrames || rendered % 30 === 0) {
        emit({ type: 'progress', stage: 'render', frame: rendered, frames: outputFrames });
      }
    }

    activeStage = 'encode';
    const trailing = packedReadback
      ? await packedReadback.flush()
      : await (readback as FrameReadbackRing).flush();
    for (const completed of trailing) {
      await encoder.writeFrame(completed.data);
    }
    if (encoder.framesWritten !== outputFrames) {
      throw new Error(
        `encoder received ${encoder.framesWritten} frames; expected ${outputFrames}`,
      );
    }
    await encoder.finish();

    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    activeStage = 'complete';
    emit({
      type: 'result',
      path: request.output.path,
      profile: request.output.profile,
      transport: profile.transport,
      codec: 'av1',
      dynamicRange: profile.dynamicRange,
      pixelFormat: hdr ? 'p010le' : 'yuv420p',
      ...(hdr
        ? {
            colorPrimaries: 'bt2020',
            colorTransfer: 'smpte2084',
            colorMatrix: 'bt2020nc',
            colorRange: 'tv',
            paperWhiteNits: request.recipe.output.paperWhiteNits,
            masteringPeakNits: request.recipe.output.masteringPeakNits,
          }
        : {}),
      audio: request.audioPath !== undefined,
      width,
      height,
      fps: 120,
      frames: outputFrames,
      elapsedSeconds,
    });
  } finally {
    releaseCleanup();
    if (encoder && encoder.state !== 'complete') await encoder.abort();
    await readback?.dispose();
    await packedReadback?.dispose();
    packer?.dispose();
    target?.destroy();
    bundle?.sim.dispose();
    ctx?.dispose();
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const requestPath = optionValue(args, '--request');
  const probe = args.includes('--probe');
  if (probe === (requestPath !== undefined)) {
    throw new Error('provide exactly one of --probe or --request <absolute-request.json>');
  }
  if (probe) await runProbe(parseProbeOptions(args));
  else {
    installCancellationHandlers();
    await runRequest(requestPath as string);
  }
}

main().catch((error: unknown) => {
  // A cancellation unwinds the render as an error too; the signal handler owns
  // that report and the exit code, so this must not overwrite either.
  if (cancelling) return;
  const message = error instanceof Error ? error.message : String(error);
  emit({ type: 'error', stage: activeStage, message });
  process.exitCode = 1;
});
