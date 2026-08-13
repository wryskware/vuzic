import type { ExportProfile } from '../runtime/recipe.ts';
import { exportProfileSpec } from './profiles.ts';

export interface SdrDebugProfileDimensions {
  readonly width: number;
  readonly height: number;
  readonly fps: 120;
}

/**
 * Exact native render dimensions for an export profile.
 *
 * Kept under its original name because the SDR engineering path still calls it;
 * the table it reads now covers every profile, HDR included.
 */
export function sdrDebugProfileDimensions(profile: ExportProfile): SdrDebugProfileDimensions {
  const spec = exportProfileSpec(profile);
  return { width: spec.width, height: spec.height, fps: spec.fps };
}
