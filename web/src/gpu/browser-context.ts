import type { GpuRuntimeContext } from './runtime-context';

/** Browser-owned GPU state, including the canvas swapchain and its lifecycle. */
export interface BrowserGpuContext extends GpuRuntimeContext {
  canvas: HTMLCanvasElement;
  gpuCanvasContext: GPUCanvasContext;
  resize(): void;
  destroy(): void;
}

export class WebGpuUnavailableError extends Error {}

export async function initGpu(canvas: HTMLCanvasElement): Promise<BrowserGpuContext> {
  if (!('gpu' in navigator) || navigator.gpu === undefined) {
    throw new WebGpuUnavailableError(
      'This browser does not expose navigator.gpu. WebGPU is required.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnavailableError(
      'No WebGPU adapter available. The GPU may be blocklisted or hardware acceleration disabled.',
    );
  }

  const wantFloat32Filterable = adapter.features.has('float32-filterable');
  const device = await adapter.requestDevice({
    requiredFeatures: wantFloat32Filterable ? (['float32-filterable'] as GPUFeatureName[]) : [],
  });
  const float32Filterable = device.features.has('float32-filterable');

  // Without this a WGSL compile error is completely silent: pipeline creation is
  // validated asynchronously, so the app keeps running and simply draws nothing.
  // A black canvas with a clean console cost real debugging time; it should not
  // be possible twice.
  device.addEventListener('uncapturederror', (ev) => {
    console.error('webgpu:', (ev as GPUUncapturedErrorEvent).error.message);
  });

  const gpuCanvasContext = canvas.getContext('webgpu');
  if (!gpuCanvasContext) {
    throw new WebGpuUnavailableError('Could not acquire a "webgpu" canvas context.');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  gpuCanvasContext.configure({ device, format, alphaMode: 'opaque' });

  const ctx: BrowserGpuContext = {
    adapter,
    device,
    canvas,
    gpuCanvasContext,
    format,
    float32Filterable,
    width: 1,
    height: 1,
    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (w === ctx.width && h === ctx.height) return;
      canvas.width = w;
      canvas.height = h;
      ctx.width = w;
      ctx.height = h;
    },
    destroy() {
      device.destroy();
    },
  };
  ctx.resize();
  return ctx;
}

export function renderUnsupportedPage(
  host: HTMLElement,
  err: unknown,
  onContinue?: () => void,
): void {
  const message = err instanceof Error ? err.message : String(err);
  host.innerHTML = '';
  const box = document.createElement('div');
  box.className = 'fatal';
  box.innerHTML = `
    <h1>WebGPU required</h1>
    <p></p>
    <p class="hint">Try a current Chrome, Edge, or Firefox on a machine with hardware
    acceleration enabled. On Linux, Chrome may need <code>--enable-unsafe-webgpu</code>.</p>
  `;
  (box.querySelector('p') as HTMLElement).textContent = message;
  if (onContinue) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'continue without the simulation';
    btn.addEventListener('click', () => {
      box.remove();
      onContinue();
    });
    box.appendChild(btn);
  }
  host.appendChild(box);
}
