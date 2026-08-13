import { execFile } from 'node:child_process';

import postCommonWgsl from '../sim/render/shaders/post-common.wgsl?raw';
import exposureWgsl from '../sim/render/shaders/exposure.wgsl?raw';
import gate0Wgsl from './shaders/gate0.wgsl?raw';

import { createDawnContext, type DawnRuntimeContext } from './dawn-context.ts';

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
}

const emit = (message: Record<string, unknown>): void => {
  process.stdout.write(`${JSON.stringify(message)}\n`);
};

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
  const encoders = ['hevc_nvenc', 'av1_nvenc'].filter((name) =>
    new RegExp(`\\b${name}\\b`).test(encodersText),
  );
  if (!encoders.includes('hevc_nvenc')) {
    throw new Error('FFmpeg does not expose the required hevc_nvenc encoder');
  }
  return {
    version: versionText.split(/\r?\n/, 1)[0] ?? 'unknown FFmpeg',
    encoders,
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

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (!args.includes('--probe')) {
    throw new Error(
      'This milestone supports --probe only; request-file rendering is added after shared runtime extraction',
    );
  }
  await runProbe(parseProbeOptions(args));
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  emit({ type: 'error', stage: 'gate0-probe', message });
  process.exitCode = 1;
});
