/**
 * GPU state shared by every renderer host.
 *
 * This contract deliberately has no DOM or swapchain members: browser previews
 * and headless exporters can construct the same simulations and render them to
 * caller-supplied texture views.
 */
/**
 * Absolute-luminance policy for a host whose final target is an HDR10 signal.
 *
 * Present only on the export host. Its presence is what selects the scene-linear
 * BT.2020/PQ output path over the 8-bit SDR grade, so a browser preview cannot
 * accidentally acquire it.
 */
export interface GpuHdrOutput {
  /** Diffuse-white luminance the graded 1.0 maps to. */
  readonly paperWhiteNits: number;
  /** Mastering-display peak the highlight roll-off and metadata assume. */
  readonly masteringPeakNits: number;
}

export interface GpuRuntimeContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
  /** Set only by a host that wants the HDR10 final transform. */
  hdrOutput?: GpuHdrOutput;
  /** Granted only if the adapter offered it; the sim must fall back to manual bilinear otherwise. */
  float32Filterable: boolean;
  /** Render-target dimensions in physical pixels, maintained by the owning host. */
  width: number;
  height: number;
}
