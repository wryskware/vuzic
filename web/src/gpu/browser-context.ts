import { configureCanvas, displayHeadroom, hdrOverride, type CanvasHdrState } from './hdr-canvas';
import type { GpuRuntimeContext } from './runtime-context';

/** Browser-owned GPU state, including the canvas swapchain and its lifecycle. */
export interface BrowserGpuContext extends GpuRuntimeContext {
  canvas: HTMLCanvasElement;
  gpuCanvasContext: GPUCanvasContext;
  /** What the swapchain configuration actually got, for the readout. */
  hdrCanvas: CanvasHdrState;
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

  const override = hdrOverride(window.location.search);
  const hdrCanvas = configureCanvas(gpuCanvasContext, device, override !== 'off');
  console.info(`webgpu: swapchain ${hdrCanvas.format} — ${hdrCanvas.detail}`);

  const ctx: BrowserGpuContext = {
    adapter,
    device,
    canvas,
    gpuCanvasContext,
    hdrCanvas,
    format: hdrCanvas.format,
    float32Filterable,
    // 1 on an SDR swapchain, and on an extended one until a display with real
    // headroom is under the window. The grade reads this every frame.
    displayHeadroom: 1,
    width: 1,
    height: 1,
    resize() {
      // Headroom is refreshed here rather than from a media-query listener
      // because it is not a boolean: it drifts with panel brightness and with
      // what else is on screen, and resize() already runs once per frame.
      ctx.displayHeadroom = !hdrCanvas.extended
        ? 1
        : typeof override === 'number'
          ? override
          : displayHeadroom();

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
