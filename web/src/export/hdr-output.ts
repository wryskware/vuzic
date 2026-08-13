/**
 * HDR10 output pipeline orchestration for the export host.
 *
 * The render target the simulation grades into is an `rgba16float` surface
 * holding *linear BT.2020 luminance normalised to the ST 2084 domain* — see
 * `sim/render/shaders/grade-hdr.wgsl`. This module owns the second half: one
 * compute dispatch that PQ-encodes it, matrixes it to BT.2020 non-constant
 * luminance Y'CbCr, decimates chroma to 4:2:0 and packs 10-bit samples into
 * P010LE, straight into a buffer FFmpeg can consume.
 *
 * Doing the conversion here rather than on the CPU is the transport decision the
 * proposal asks for: at 4K120 an RGBA16F readback is 66.4 MB/frame against
 * P010's 24.9 MB, and the packing is per-pixel work on data that is already
 * resident on the device.
 */
import { p010Layout, type P010Layout } from './hdr.ts';
import hdrPackWgsl from './shaders/hdr-pack.wgsl?raw';

/** u32 slots in the packing uniform; must match `PackParams` in hdr-pack.wgsl. */
const PACK_PARAM_WORDS = 8;
const PACK_WORKGROUP = 8;

export interface Hdr10FramePackerOptions {
  readonly width: number;
  readonly height: number;
  readonly label?: string;
}

export class Hdr10FramePacker {
  readonly layout: P010Layout;

  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private readonly packed: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private boundView: GPUTextureView | null = null;
  private disposed = false;

  constructor(device: GPUDevice, options: Hdr10FramePackerOptions) {
    this.device = device;
    this.layout = p010Layout(options.width, options.height);
    const label = options.label ?? 'export.hdr10.pack';

    this.bindGroupLayout = device.createBindGroupLayout({
      label: `${label}.layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          // textureLoad only; the pass never samples, so filterability is moot.
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ label, code: hdrPackWgsl }),
        entryPoint: 'pack',
      },
    });

    this.params = device.createBuffer({
      label: `${label}.params`,
      size: PACK_PARAM_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.params,
      0,
      new Uint32Array([
        this.layout.width,
        this.layout.height,
        this.layout.lumaWordsPerRow,
        this.layout.chromaWordsPerRow,
        this.layout.chromaWordOffset,
        0,
        0,
        0,
      ]),
    );
    this.packed = device.createBuffer({
      label: `${label}.p010`,
      size: this.layout.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  /** Encoder-ready bytes per packed frame. */
  get byteLength(): number {
    return this.layout.byteLength;
  }

  /**
   * The buffer the last `pack` wrote. Safe to reuse for the next frame as soon
   * as a copy out of it has been submitted, because both live in the same queue.
   */
  get packedBuffer(): GPUBuffer {
    return this.packed;
  }

  /** Record the conversion for one graded frame. The caller owns submission. */
  pack(encoder: GPUCommandEncoder, source: GPUTextureView): void {
    if (this.disposed) throw new Error('HDR10 frame packer is disposed');
    if (source !== this.boundView || !this.bindGroup) {
      this.bindGroup = this.device.createBindGroup({
        label: 'export.hdr10.pack.bind',
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: source },
          { binding: 2, resource: { buffer: this.packed } },
        ],
      });
      this.boundView = source;
    }
    const pass = encoder.beginComputePass({ label: 'export.hdr10.pack' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.layout.width / 2 / PACK_WORKGROUP),
      Math.ceil(this.layout.height / 2 / PACK_WORKGROUP),
    );
    pass.end();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.bindGroup = null;
    this.boundView = null;
    this.params.destroy();
    this.packed.destroy();
  }
}
