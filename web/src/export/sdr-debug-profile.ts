import type { ExportProfile } from '../runtime/recipe.ts';

export interface SdrDebugProfileDimensions {
  readonly width: number;
  readonly height: number;
  readonly fps: 120;
}

/** Exact native render dimensions for the temporary AV1 SDR engineering profiles. */
export function sdrDebugProfileDimensions(profile: ExportProfile): SdrDebugProfileDimensions {
  switch (profile) {
    case 'av1-sdr-debug-2160p120':
      return { width: 3840, height: 2160, fps: 120 };
    case 'av1-sdr-debug-1080p120':
      return { width: 1920, height: 1080, fps: 120 };
  }
}
