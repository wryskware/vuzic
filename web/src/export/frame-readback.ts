const COPY_BYTES_PER_ROW_ALIGNMENT = 256;

export interface FrameReadbackLayout {
  readonly width: number;
  readonly height: number;
  readonly bytesPerPixel: number;
  readonly unpaddedBytesPerRow: number;
  readonly paddedBytesPerRow: number;
  /** Size of the tightly packed frame delivered to the consumer. */
  readonly bytesPerFrame: number;
  /** Size of each GPU staging buffer, including row padding. */
  readonly stagingBytesPerFrame: number;
}

export interface FrameReadbackRingOptions {
  readonly width: number;
  readonly height: number;
  readonly bytesPerPixel: number;
  readonly ringSize?: number;
  readonly label?: string;
}

export interface ReadbackFrame {
  readonly frameIndex: number;
  /** Tightly packed rows. No WebGPU copy-row padding remains. */
  readonly data: Uint8Array;
}

interface ReadbackSlot {
  readonly buffer: GPUBuffer;
  pending: Promise<ReadbackFrame> | null;
}

function positiveSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}

function checkedProduct(left: number, right: number, name: string): number {
  const value = left * right;
  if (!Number.isSafeInteger(value)) {
    throw new RangeError(`${name} exceeds the safe integer range`);
  }
  return value;
}

export function alignTo(value: number, alignment: number): number {
  positiveSafeInteger(value, 'value');
  positiveSafeInteger(alignment, 'alignment');
  const aligned = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(aligned)) {
    throw new RangeError('aligned value exceeds the safe integer range');
  }
  return aligned;
}

export function frameReadbackLayout(
  width: number,
  height: number,
  bytesPerPixel: number,
): FrameReadbackLayout {
  positiveSafeInteger(width, 'width');
  positiveSafeInteger(height, 'height');
  positiveSafeInteger(bytesPerPixel, 'bytesPerPixel');
  const unpaddedBytesPerRow = checkedProduct(width, bytesPerPixel, 'row byte count');
  const paddedBytesPerRow = alignTo(unpaddedBytesPerRow, COPY_BYTES_PER_ROW_ALIGNMENT);
  return {
    width,
    height,
    bytesPerPixel,
    unpaddedBytesPerRow,
    paddedBytesPerRow,
    bytesPerFrame: checkedProduct(unpaddedBytesPerRow, height, 'frame byte count'),
    stagingBytesPerFrame: checkedProduct(paddedBytesPerRow, height, 'staging byte count'),
  };
}

/** Copy a mapped staging buffer into encoder-ready, tightly packed rows. */
export function stripReadbackRowPadding(
  source: Uint8Array,
  layout: FrameReadbackLayout,
): Uint8Array {
  if (source.byteLength < layout.stagingBytesPerFrame) {
    throw new RangeError(
      `staging buffer has ${source.byteLength} bytes; expected ${layout.stagingBytesPerFrame}`,
    );
  }
  const output = new Uint8Array(layout.bytesPerFrame);
  if (layout.paddedBytesPerRow === layout.unpaddedBytesPerRow) {
    output.set(source.subarray(0, layout.bytesPerFrame));
    return output;
  }
  for (let row = 0; row < layout.height; row++) {
    const sourceStart = row * layout.paddedBytesPerRow;
    const outputStart = row * layout.unpaddedBytesPerRow;
    output.set(
      source.subarray(sourceStart, sourceStart + layout.unpaddedBytesPerRow),
      outputStart,
    );
  }
  return output;
}

/**
 * Bounded asynchronous texture readback.
 *
 * `enqueue` submits a copy after all previously submitted render work. Once the
 * ring wraps, it waits for and returns the frame formerly occupying that slot;
 * the caller can then await encoder backpressure before enqueueing again. A
 * slot is always unmapped before it is reused for a GPU copy.
 */
export class FrameReadbackRing {
  readonly layout: FrameReadbackLayout;
  readonly ringSize: number;

  private readonly device: GPUDevice;
  private readonly slots: ReadbackSlot[];
  private nextSlot = 0;
  private operationInProgress = false;
  private disposed = false;

  constructor(device: GPUDevice, options: FrameReadbackRingOptions) {
    this.device = device;
    this.layout = frameReadbackLayout(options.width, options.height, options.bytesPerPixel);
    this.ringSize = positiveSafeInteger(options.ringSize ?? 3, 'ringSize');
    const label = options.label ?? 'export.readback';
    this.slots = Array.from({ length: this.ringSize }, (_, index) => ({
      buffer: device.createBuffer({
        label: `${label}.${index}`,
        size: this.layout.stagingBytesPerFrame,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: null,
    }));
  }

  get pendingCount(): number {
    return this.slots.reduce((count, slot) => count + (slot.pending ? 1 : 0), 0);
  }

  async enqueue(texture: GPUTexture, frameIndex: number): Promise<ReadbackFrame | null> {
    this.beginOperation();
    try {
      if (!Number.isSafeInteger(frameIndex) || frameIndex < 0) {
        throw new RangeError('frameIndex must be a non-negative safe integer');
      }
      const slot = this.slots[this.nextSlot];
      if (!slot) throw new Error('readback ring unexpectedly has no slot');

      let completed: ReadbackFrame | null = null;
      if (slot.pending) {
        const pending = slot.pending;
        slot.pending = null;
        completed = await pending;
      }

      const encoder = this.device.createCommandEncoder({
        label: `export.readback.frame.${frameIndex}`,
      });
      encoder.copyTextureToBuffer(
        { texture },
        {
          buffer: slot.buffer,
          bytesPerRow: this.layout.paddedBytesPerRow,
          rowsPerImage: this.layout.height,
        },
        { width: this.layout.width, height: this.layout.height },
      );
      this.device.queue.submit([encoder.finish()]);

      // mapAsync is ordered after the submitted COPY_DST use. getMappedRange is
      // reached only once that GPU write has completed, and readSlot unmaps the
      // buffer before enqueue can reuse this slot.
      const pending = this.readSlot(slot.buffer, frameIndex);
      // A ring slot may not be revisited for several frames. Keep the original
      // rejection observable to enqueue/flush while preventing an early device
      // loss from becoming an unhandled rejection in the meantime.
      void pending.catch(() => undefined);
      slot.pending = pending;
      this.nextSlot = (this.nextSlot + 1) % this.slots.length;
      return completed;
    } catch (error: unknown) {
      await this.dispose();
      throw error;
    } finally {
      this.operationInProgress = false;
    }
  }

  /** Return every in-flight frame in frame-index order. The result stays ring-bounded. */
  async flush(): Promise<ReadbackFrame[]> {
    this.beginOperation();
    try {
      const frames: ReadbackFrame[] = [];
      for (const slot of this.slots) {
        if (!slot.pending) continue;
        const pending = slot.pending;
        slot.pending = null;
        frames.push(await pending);
      }
      frames.sort((left, right) => left.frameIndex - right.frameIndex);
      return frames;
    } catch (error: unknown) {
      await this.dispose();
      throw error;
    } finally {
      this.operationInProgress = false;
    }
  }

  /** Destroy all staging resources. Safe on a failed or cancelled render. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const pending = this.slots.flatMap((slot) => (slot.pending ? [slot.pending] : []));
    for (const slot of this.slots) {
      slot.pending = null;
      slot.buffer.destroy();
    }
    await Promise.allSettled(pending);
  }

  private beginOperation(): void {
    if (this.disposed) throw new Error('frame readback ring is disposed');
    if (this.operationInProgress) {
      throw new Error('frame readback operations must be awaited sequentially');
    }
    this.operationInProgress = true;
  }

  private async readSlot(buffer: GPUBuffer, frameIndex: number): Promise<ReadbackFrame> {
    let mapped = false;
    try {
      await buffer.mapAsync(GPUMapMode.READ, 0, this.layout.stagingBytesPerFrame);
      mapped = true;
      const bytes = new Uint8Array(
        buffer.getMappedRange(0, this.layout.stagingBytesPerFrame),
      );
      return { frameIndex, data: stripReadbackRowPadding(bytes, this.layout) };
    } finally {
      if (mapped) buffer.unmap();
    }
  }
}

export interface PackedReadbackRingOptions {
  /** Exact tightly packed frame size produced by the GPU packing pass. */
  readonly byteLength: number;
  readonly ringSize?: number;
  readonly label?: string;
}

/**
 * The same bounded readback discipline as `FrameReadbackRing`, for frames a
 * compute pass has already packed into encoder-ready bytes.
 *
 * There is no row padding to strip here: `copyBufferToBuffer` has none of
 * `copyTextureToBuffer`'s 256-byte row alignment rule, which is half the reason
 * the HDR path packs on the GPU. The source buffer may be reused for the next
 * frame as soon as the copy is submitted, because the copy is ordered inside the
 * same queue as the pass that wrote it.
 */
export class PackedFrameReadbackRing {
  readonly byteLength: number;
  readonly ringSize: number;

  private readonly device: GPUDevice;
  private readonly slots: ReadbackSlot[];
  private nextSlot = 0;
  private operationInProgress = false;
  private disposed = false;

  constructor(device: GPUDevice, options: PackedReadbackRingOptions) {
    this.device = device;
    this.byteLength = positiveSafeInteger(options.byteLength, 'byteLength');
    if (this.byteLength % 4 !== 0) throw new RangeError('byteLength must be a multiple of 4');
    this.ringSize = positiveSafeInteger(options.ringSize ?? 3, 'ringSize');
    const label = options.label ?? 'export.packed-readback';
    this.slots = Array.from({ length: this.ringSize }, (_, index) => ({
      buffer: device.createBuffer({
        label: `${label}.${index}`,
        size: this.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: null,
    }));
  }

  get pendingCount(): number {
    return this.slots.reduce((count, slot) => count + (slot.pending ? 1 : 0), 0);
  }

  async enqueue(source: GPUBuffer, frameIndex: number): Promise<ReadbackFrame | null> {
    this.beginOperation();
    try {
      if (!Number.isSafeInteger(frameIndex) || frameIndex < 0) {
        throw new RangeError('frameIndex must be a non-negative safe integer');
      }
      const slot = this.slots[this.nextSlot];
      if (!slot) throw new Error('readback ring unexpectedly has no slot');

      let completed: ReadbackFrame | null = null;
      if (slot.pending) {
        const pending = slot.pending;
        slot.pending = null;
        completed = await pending;
      }

      const encoder = this.device.createCommandEncoder({
        label: `export.packed-readback.frame.${frameIndex}`,
      });
      encoder.copyBufferToBuffer(source, 0, slot.buffer, 0, this.byteLength);
      this.device.queue.submit([encoder.finish()]);

      const pending = this.readSlot(slot.buffer, frameIndex);
      void pending.catch(() => undefined);
      slot.pending = pending;
      this.nextSlot = (this.nextSlot + 1) % this.slots.length;
      return completed;
    } catch (error: unknown) {
      await this.dispose();
      throw error;
    } finally {
      this.operationInProgress = false;
    }
  }

  async flush(): Promise<ReadbackFrame[]> {
    this.beginOperation();
    try {
      const frames: ReadbackFrame[] = [];
      for (const slot of this.slots) {
        if (!slot.pending) continue;
        const pending = slot.pending;
        slot.pending = null;
        frames.push(await pending);
      }
      frames.sort((left, right) => left.frameIndex - right.frameIndex);
      return frames;
    } catch (error: unknown) {
      await this.dispose();
      throw error;
    } finally {
      this.operationInProgress = false;
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const pending = this.slots.flatMap((slot) => (slot.pending ? [slot.pending] : []));
    for (const slot of this.slots) {
      slot.pending = null;
      slot.buffer.destroy();
    }
    await Promise.allSettled(pending);
  }

  private beginOperation(): void {
    if (this.disposed) throw new Error('packed readback ring is disposed');
    if (this.operationInProgress) {
      throw new Error('frame readback operations must be awaited sequentially');
    }
    this.operationInProgress = true;
  }

  private async readSlot(buffer: GPUBuffer, frameIndex: number): Promise<ReadbackFrame> {
    let mapped = false;
    try {
      await buffer.mapAsync(GPUMapMode.READ, 0, this.byteLength);
      mapped = true;
      const bytes = new Uint8Array(buffer.getMappedRange(0, this.byteLength));
      return { frameIndex, data: new Uint8Array(bytes) };
    } finally {
      if (mapped) buffer.unmap();
    }
  }
}
