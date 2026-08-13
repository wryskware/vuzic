/**
 * GPU state shared by every renderer host.
 *
 * This contract deliberately has no DOM or swapchain members: browser previews
 * and headless exporters can construct the same simulations and render them to
 * caller-supplied texture views.
 */
export interface GpuRuntimeContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
  /** Granted only if the adapter offered it; the sim must fall back to manual bilinear otherwise. */
  float32Filterable: boolean;
  /** Render-target dimensions in physical pixels, maintained by the owning host. */
  width: number;
  height: number;
}
