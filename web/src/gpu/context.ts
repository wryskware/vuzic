/**
 * Compatibility entry point for code that still imports the original context
 * module. New platform-neutral renderer code should import GpuRuntimeContext
 * from runtime-context; browser hosts should import from browser-context.
 */
export type { GpuRuntimeContext } from './runtime-context';
export {
  initGpu,
  renderUnsupportedPage,
  WebGpuUnavailableError,
  type BrowserGpuContext,
  /** @deprecated Use BrowserGpuContext for browser hosts. */
  type BrowserGpuContext as GpuContext,
} from './browser-context';
