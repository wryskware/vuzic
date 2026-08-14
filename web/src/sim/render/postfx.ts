/**
 * The HDR post chain (phase 7).
 *
 * The sim composites into `hdrTargetView` — an rgba16float surface at canvas
 * resolution — and then hands the encoder here. Nothing between the trail
 * textures and the final grade pass is 8-bit or clamped, which is the structural
 * fix for "trails clip to white at high stem drive": there is no longer a
 * surface anywhere upstream that *can* clip.
 *
 * Per frame:
 *
 *   [sim composites → hdr[cur]]
 *   measure   compute, 1 workgroup, 4096 samples → auto-exposure state buffer
 *   prefilter hdr[cur] → A0        (half res, soft-knee threshold)
 *   downsample A0 → A1 → A2        (one pass each, L-1 total)
 *   for i = L-1 … 0: blurH A[i]→B[i], blurV B[i]→A[i], then additively
 *                    upsample A[i] into A[i-1]
 *   grade     hdr[cur] + A0 → swapchain
 *
 * Then `cur` flips, so next frame's compositor can read this frame's HDR as the
 * feedback source. The ping-pong exists for that one feature; with feedback at 0
 * it costs one extra texture and nothing else.
 *
 * Two deliberate omissions:
 *
 * - No per-pass uniform buffers. Every chain pass reads its source's size from
 *   `textureDimensions()`, and the two blur directions are two entry points, so
 *   the whole chain shares one 80-byte params buffer written once per frame.
 * - No CPU readback for auto-exposure. The controller's state lives in a GPU
 *   storage buffer that the compositor reads one frame later; a readback would
 *   be a queue stall for a number that is allowed to be one frame stale.
 */
import type { GpuRuntimeContext } from '../../gpu/runtime-context';
import type { RenderFrame } from '../types';
import { MAX_BLOOM_LEVELS, tonemapIndex, type RenderConfig } from './config';

import postCommonWgsl from './shaders/post-common.wgsl?raw';
import bloomWgsl from './shaders/bloom.wgsl?raw';
import gradeWgsl from './shaders/grade.wgsl?raw';
import gradeHdrWgsl from './shaders/grade-hdr.wgsl?raw';
import exposureWgsl from './shaders/exposure.wgsl?raw';

/** f32 slots in the params uniform; must match `Post` in post-common.wgsl. */
const PARAM_WORDS = 24;
/** bytes in the auto-exposure state buffer: gain, mean, init, pad */
const AUTO_BYTES = 16;

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';

interface Level {
  a: GPUTexture;
  b: GPUTexture;
  aView: GPUTextureView;
  bView: GPUTextureView;
  /** blurH source = a, blurV source = b, upsample source = a */
  bindA: GPUBindGroup;
  bindB: GPUBindGroup;
}

export class PostFx {
  /** Set by the owner each frame; θ owns gamma, PostFx only applies it. */
  gamma = 2.2;

  private ctx: GpuRuntimeContext | null = null;
  private cfg: RenderConfig;

  private sampler!: GPUSampler;
  private paramsBuf!: GPUBuffer;
  private autoBuf!: GPUBuffer;
  private readonly paramsData = new Float32Array(PARAM_WORDS);

  private chainLayout!: GPUBindGroupLayout;
  private gradeLayout!: GPUBindGroupLayout;
  private measureLayout!: GPUBindGroupLayout;

  private prefilterPipeline!: GPURenderPipeline;
  private downsamplePipeline!: GPURenderPipeline;
  private blurHPipeline!: GPURenderPipeline;
  private blurVPipeline!: GPURenderPipeline;
  private upsamplePipeline!: GPURenderPipeline;
  private gradePipeline!: GPURenderPipeline;
  private measurePipeline!: GPUComputePipeline;

  private width = 0;
  private height = 0;
  private version = 0;

  /** ping-pong so the compositor can read the previous frame for feedback */
  private hdr: GPUTexture[] = [];
  private hdrViews: GPUTextureView[] = [];
  private prefilterBind: GPUBindGroup[] = [];
  private gradeBind: GPUBindGroup[] = [];
  private measureBind: GPUBindGroup[] = [];
  private levels: Level[] = [];
  private parity = 0;

  private passes = 0;

  /**
   * A copy of the controller's state, pulled back for the workbench readout only.
   * Nothing in the render path waits on it: the map is async, at most one is in
   * flight, and the numbers are simply the last ones that arrived.
   */
  private autoRead!: GPUBuffer;
  private autoReadBusy = false;
  /** dispose() may land while a readback is in flight; whoever finishes last frees it. */
  private disposed = false;
  private autoGain = 1;
  private autoMean = 0;

  constructor(config: RenderConfig) {
    this.cfg = config;
  }

  get config(): RenderConfig {
    return this.cfg;
  }

  /** Bumped whenever the textures are reallocated; the sim rebinds when it changes. */
  get sizeVersion(): number {
    return this.version;
  }

  /** Render passes issued by `run()` last frame — the compositor is not counted. */
  get passCount(): number {
    return this.passes;
  }

  get currentParity(): number {
    return this.parity;
  }

  /** Where the sim composites this frame. */
  get hdrTargetView(): GPUTextureView {
    return this.hdrViews[this.parity] as GPUTextureView;
  }

  /** Either ping-pong surface by index; parity `p` composites into `p` and reads `1 - p`. */
  hdrViewAt(index: number): GPUTextureView {
    return this.hdrViews[index & 1] as GPUTextureView;
  }

  get autoBuffer(): GPUBuffer {
    return this.autoBuf;
  }

  get linearSampler(): GPUSampler {
    return this.sampler;
  }

  init(ctx: GpuRuntimeContext): void {
    this.ctx = ctx;
    const { device } = ctx;

    this.sampler = device.createSampler({
      label: 'postfx.linear',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.paramsBuf = device.createBuffer({
      label: 'postfx.params',
      size: PARAM_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.autoBuf = device.createBuffer({
      label: 'postfx.autoExposure',
      size: AUTO_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.autoRead = device.createBuffer({
      label: 'postfx.autoExposure.read',
      size: AUTO_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // gain 1, mean 0, init 0 — "uninitialised", so the first real frame snaps.
    device.queue.writeBuffer(this.autoBuf, 0, new Float32Array([1, 0, 0, 0]));

    const F = GPUShaderStage.FRAGMENT;
    const filterable: GPUTextureBindingLayout = { sampleType: 'float', viewDimension: '2d' };

    this.chainLayout = device.createBindGroupLayout({
      label: 'postfx.chainLayout',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: filterable },
      ],
    });
    this.gradeLayout = device.createBindGroupLayout({
      label: 'postfx.gradeLayout',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: filterable },
        { binding: 3, visibility: F, texture: filterable },
      ],
    });
    this.measureLayout = device.createBindGroupLayout({
      label: 'postfx.measureLayout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          // textureLoad only; never sampled, so filterability is irrelevant here
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const src = (body: string): GPUShaderModule =>
      device.createShaderModule({ code: `${postCommonWgsl}\n${body}` });
    const bloomModule = src(bloomWgsl);
    // The HDR10 export host replaces the final pass outright rather than
    // post-processing the SDR grade's output: the display transform, the
    // primaries and the transfer function are all different, and the SDR pass
    // has already thrown away everything above diffuse white by the time it ends.
    const gradeModule = src(ctx.hdrOutput ? gradeHdrWgsl : gradeWgsl);

    const chainPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.chainLayout],
    });
    const chain = (
      label: string,
      entryPoint: string,
      blend?: GPUBlendState,
    ): GPURenderPipeline =>
      device.createRenderPipeline({
        label,
        layout: chainPipelineLayout,
        vertex: { module: bloomModule, entryPoint: 'vsMain' },
        fragment: {
          module: bloomModule,
          entryPoint,
          targets: [blend ? { format: HDR_FORMAT, blend } : { format: HDR_FORMAT }],
        },
        primitive: { topology: 'triangle-list' },
      });

    this.prefilterPipeline = chain('postfx.prefilter', 'fsPrefilter');
    this.downsamplePipeline = chain('postfx.downsample', 'fsDownsample');
    this.blurHPipeline = chain('postfx.blurH', 'fsBlurH');
    this.blurVPipeline = chain('postfx.blurV', 'fsBlurV');
    this.upsamplePipeline = chain('postfx.upsample', 'fsUpsample', {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    });

    this.gradePipeline = device.createRenderPipeline({
      label: ctx.hdrOutput ? 'postfx.grade.hdr10' : 'postfx.grade',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.gradeLayout] }),
      vertex: { module: gradeModule, entryPoint: 'vsMain' },
      fragment: { module: gradeModule, entryPoint: 'fsMain', targets: [{ format: ctx.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.measurePipeline = device.createComputePipeline({
      label: 'postfx.measure',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.measureLayout] }),
      compute: { module: src(exposureWgsl), entryPoint: 'measure' },
    });
  }

  /** Allocate (or reallocate) for the current canvas size. True if anything moved. */
  ensureSize(width: number, height: number): boolean {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.width && h === this.height) return false;
    const ctx = this.ctx;
    if (!ctx) return false;
    const { device } = ctx;

    this.releaseTextures();
    this.width = w;
    this.height = h;

    const attach =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

    this.hdr = [0, 1].map((i) =>
      device.createTexture({
        label: `postfx.hdr${i}`,
        size: { width: w, height: h },
        format: HDR_FORMAT,
        usage: attach,
      }),
    );
    this.hdrViews = this.hdr.map((t) => t.createView());

    this.levels = [];
    for (let i = 0; i < MAX_BLOOM_LEVELS; i++) {
      const lw = Math.max(1, Math.floor(w / 2 ** (i + 1)));
      const lh = Math.max(1, Math.floor(h / 2 ** (i + 1)));
      const make = (suffix: string): GPUTexture =>
        device.createTexture({
          label: `postfx.bloom${i}${suffix}`,
          size: { width: lw, height: lh },
          format: HDR_FORMAT,
          usage: attach,
        });
      const a = make('a');
      const b = make('b');
      const aView = a.createView();
      const bView = b.createView();
      this.levels.push({
        a,
        b,
        aView,
        bView,
        bindA: this.chainBind(aView, `postfx.bloom${i}a`),
        bindB: this.chainBind(bView, `postfx.bloom${i}b`),
      });
    }

    this.prefilterBind = this.hdrViews.map((v, i) => this.chainBind(v, `postfx.prefilter${i}`));
    this.gradeBind = this.hdrViews.map((v, i) =>
      device.createBindGroup({
        label: `postfx.grade${i}`,
        layout: this.gradeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: v },
          { binding: 3, resource: (this.levels[0] as Level).aView },
        ],
      }),
    );
    this.measureBind = this.hdrViews.map((v, i) =>
      device.createBindGroup({
        label: `postfx.measure${i}`,
        layout: this.measureLayout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: v },
          { binding: 2, resource: { buffer: this.autoBuf } },
        ],
      }),
    );

    this.version++;
    return true;
  }

  /**
   * Measure, bloom, grade. `targetView` is the swapchain. Flips the ping-pong on
   * the way out, so `hdrTargetView` is a fresh surface next frame and
   * `previousHdrView` is the one just graded.
   */
  run(encoder: GPUCommandEncoder, targetView: GPUTextureView, frame: RenderFrame): void {
    if (!this.ctx || this.levels.length === 0) return;
    this.writeParams(Math.min(frame.deltaSeconds, 0.25));

    const cur = this.parity;
    let passes = 0;

    const measure = encoder.beginComputePass({ label: 'postfx.measure' });
    measure.setPipeline(this.measurePipeline);
    measure.setBindGroup(0, this.measureBind[cur] as GPUBindGroup);
    measure.dispatchWorkgroups(1);
    measure.end();
    passes++;

    const bloom = this.cfg.bloom;
    const levelCount = bloom.enabled
      ? Math.min(Math.max(Math.round(bloom.levels), 1), MAX_BLOOM_LEVELS)
      : 0;

    if (levelCount > 0) {
      const draw = (
        label: string,
        pipeline: GPURenderPipeline,
        bind: GPUBindGroup,
        view: GPUTextureView,
        additive = false,
      ): void => {
        const pass = encoder.beginRenderPass({
          label,
          colorAttachments: [
            {
              view,
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: additive ? 'load' : 'clear',
              storeOp: 'store',
            },
          ],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.draw(3);
        pass.end();
        passes++;
      };

      const l0 = this.levels[0] as Level;
      draw(
        'postfx.prefilter',
        this.prefilterPipeline,
        this.prefilterBind[cur] as GPUBindGroup,
        l0.aView,
      );
      for (let i = 1; i < levelCount; i++) {
        const prev = this.levels[i - 1] as Level;
        const lv = this.levels[i] as Level;
        draw(`postfx.downsample${i}`, this.downsamplePipeline, prev.bindA, lv.aView);
      }
      for (let i = levelCount - 1; i >= 0; i--) {
        const lv = this.levels[i] as Level;
        draw(`postfx.blurH${i}`, this.blurHPipeline, lv.bindA, lv.bView);
        draw(`postfx.blurV${i}`, this.blurVPipeline, lv.bindB, lv.aView);
        if (i > 0) {
          const up = this.levels[i - 1] as Level;
          draw(`postfx.upsample${i}`, this.upsamplePipeline, lv.bindA, up.aView, true);
        }
      }
    }

    const grade = encoder.beginRenderPass({
      label: 'postfx.grade',
      colorAttachments: [
        {
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    grade.setPipeline(this.gradePipeline);
    grade.setBindGroup(0, this.gradeBind[cur] as GPUBindGroup);
    grade.draw(3);
    grade.end();
    passes++;

    this.passes = passes;
    this.parity = 1 - this.parity;

    if (!this.autoReadBusy && !this.disposed) {
      this.autoReadBusy = true;
      encoder.copyBufferToBuffer(this.autoBuf, 0, this.autoRead, 0, AUTO_BYTES);
      // A macrotask, not a microtask: the caller submits this encoder synchronously
      // after render(), and mapAsync only waits for work already submitted.
      setTimeout(() => void this.pullAutoState(), 0);
    }
  }

  /** Adapted gain and measured mean luminance, for the workbench readout. */
  get autoExposureState(): { gain: number; mean: number } {
    return { gain: this.autoGain, mean: this.autoMean };
  }

  private async pullAutoState(): Promise<void> {
    try {
      // dispose() ran between the copy being encoded and this macrotask: the
      // source buffer is already gone, so there is nothing left to read.
      if (this.disposed) return;
      await this.autoRead.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(this.autoRead.getMappedRange().slice(0));
      this.autoRead.unmap();
      this.autoGain = v[0] ?? 1;
      this.autoMean = v[1] ?? 0;
    } catch {
      // device lost, or the buffer was destroyed mid-flight; the readout is
      // cosmetic, so a failure just leaves the last values in place.
    } finally {
      this.autoReadBusy = false;
      // dispose() skipped the buffer because this readback held it; free it here.
      if (this.disposed) this.autoRead?.destroy();
    }
  }

  /** Re-snap auto-exposure on the next frame (used by reseed / restart). */
  resetAutoExposure(): void {
    this.ctx?.device.queue.writeBuffer(this.autoBuf, 0, new Float32Array([1, 0, 0, 0]));
  }

  /**
   * Start the controller at a caller-supplied gain instead of 1.
   *
   * A live substrate swap hands the incoming chain the outgoing chain's adapted
   * gain: the new sim starts on an empty field, and a controller re-hunting from
   * 1× against black races to its rail and renders the first seconds badly
   * over-exposed. Seeding it where the outgoing controller had settled makes the
   * moment after a swap exposed like the moment before it. Clamped to the live
   * rails so a stale or foreign value cannot park the controller out of range.
   */
  seedAutoExposure(gain: number): void {
    const g = this.cfg.grade;
    const lo = Math.max(g.autoMinGain, 1e-3);
    const hi = Math.max(g.autoMaxGain, lo);
    const seeded = Math.min(Math.max(Number.isFinite(gain) ? gain : 1, lo), hi);
    this.ctx?.device.queue.writeBuffer(this.autoBuf, 0, new Float32Array([seeded, 0, 0, 0]));
  }

  dispose(): void {
    this.disposed = true;
    this.releaseTextures();
    this.paramsBuf?.destroy();
    this.autoBuf?.destroy();
    // An in-flight readback still holds autoRead; pullAutoState's finally destroys
    // it instead, so exactly one of the two paths frees it and neither leaks.
    if (!this.autoReadBusy) this.autoRead?.destroy();
    this.ctx = null;
  }

  private chainBind(view: GPUTextureView, label: string): GPUBindGroup {
    const { device } = this.ctx as GpuRuntimeContext;
    return device.createBindGroup({
      label,
      layout: this.chainLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: view },
      ],
    });
  }

  private writeParams(dt: number): void {
    const { device } = this.ctx as GpuRuntimeContext;
    const g = this.cfg.grade;
    const b = this.cfg.bloom;
    const p = this.paramsData;
    p[0] = this.width;
    p[1] = this.height;
    p[2] = Math.max(b.threshold, 0);
    p[3] = Math.max(b.knee, 1e-4);
    p[4] = b.enabled ? Math.max(b.intensity, 0) : 0;
    p[5] = g.exposureEv;
    p[6] = tonemapIndex(g.tonemap);
    p[7] = g.blackPoint;
    p[8] = Math.max(g.contrast, 0);
    p[9] = g.pivot;
    p[10] = Math.max(g.saturation, 0);
    p[11] = g.vignette;
    // The display transfer function stays where it always was: PhysarumConfig.gamma,
    // which is part of θ. This buffer only carries the value.
    p[12] = this.gamma;
    p[13] = g.autoExposure ? 1 : 0;
    p[14] = Math.max(g.autoTarget, 1e-5);
    p[15] = Math.max(g.autoTau, 0.01);
    p[16] = Math.max(g.autoMinGain, 1e-3);
    p[17] = Math.max(g.autoMaxGain, g.autoMinGain);
    p[18] = dt;
    p[19] = 0;
    // Zero for every host that is not exporting HDR10. grade.wgsl never reads
    // these two; grade-hdr.wgsl reads nothing else to decide what absolute
    // luminance a graded 1.0 means.
    const hdr = this.ctx?.hdrOutput;
    p[20] = hdr ? hdr.paperWhiteNits : 0;
    p[21] = hdr ? hdr.masteringPeakNits : 0;
    // Headroom, on the other hand, is read by both grades and means the same
    // thing in each: how far above diffuse white the target can go. The export
    // knows it from its mastering policy; the browser measures it from the
    // display and gets 1 whenever the swapchain is 8-bit or the panel is SDR.
    p[22] = hdr
      ? Math.max(hdr.masteringPeakNits / hdr.paperWhiteNits, 1)
      : Math.max(this.ctx?.displayHeadroom ?? 1, 1);
    p[23] = 0;
    device.queue.writeBuffer(this.paramsBuf, 0, p);
  }

  private releaseTextures(): void {
    for (const t of this.hdr) t.destroy();
    for (const l of this.levels) {
      l.a.destroy();
      l.b.destroy();
    }
    this.hdr = [];
    this.hdrViews = [];
    this.levels = [];
    this.prefilterBind = [];
    this.gradeBind = [];
    this.measureBind = [];
  }
}
