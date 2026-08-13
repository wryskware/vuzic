import { create, globals } from 'webgpu';

import type { GpuHdrOutput, GpuRuntimeContext } from '../gpu/runtime-context.ts';

export interface DawnContextOptions {
  width: number;
  height: number;
  backend?: string;
  /** Caller-owned final target format; simulations keep their internal HDR surfaces. */
  format?: GPUTextureFormat;
  /**
   * Absolute-luminance policy for an HDR10 export. Supplying it is what makes
   * the shared post chain emit the scene-linear BT.2020/PQ grade instead of the
   * 8-bit display grade, so it must reach the context object itself — a host
   * that sets it and a context that drops it would produce an HDR-tagged SDR
   * file, which is the one outcome this whole path exists to avoid.
   */
  hdrOutput?: GpuHdrOutput;
}

export interface DawnRuntimeContext extends GpuRuntimeContext {
  readonly adapterName: string;
  readonly backend: string;
  dispose(): void;
}

/** Construct the canvas-free GPU context used by the Node export host. */
export async function createDawnContext(options: DawnContextOptions): Promise<DawnRuntimeContext> {
  Object.assign(globalThis, globals);

  const backend = options.backend ?? (process.platform === 'win32' ? 'd3d12' : 'vulkan');
  let nativeGpu: GPU | null = create([`backend=${backend}`]);
  const adapter = await nativeGpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    nativeGpu = null;
    throw new Error(`Dawn did not find a high-performance ${backend} adapter`);
  }

  const wantFloat32Filterable = adapter.features.has('float32-filterable');
  const device = await adapter.requestDevice({
    requiredFeatures: wantFloat32Filterable ? (['float32-filterable'] as GPUFeatureName[]) : [],
  });
  const float32Filterable = device.features.has('float32-filterable');

  device.addEventListener('uncapturederror', (event) => {
    console.error('webgpu:', (event as GPUUncapturedErrorEvent).error.message);
  });

  const info = adapter.info;
  const adapterName =
    info.description || info.device || info.architecture || info.vendor || 'unknown adapter';

  return {
    adapter,
    device,
    width: Math.max(1, Math.floor(options.width)),
    height: Math.max(1, Math.floor(options.height)),
    format: options.format ?? 'rgba16float',
    ...(options.hdrOutput ? { hdrOutput: options.hdrOutput } : {}),
    float32Filterable,
    adapterName,
    backend,
    dispose() {
      device.destroy();
      // dawn.node keeps its native event loop alive while the object returned by
      // create() is reachable. Dropping our last reference lets a probe/worker
      // terminate normally after the device is destroyed.
      nativeGpu = null;
    },
  };
}
