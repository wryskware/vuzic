import type { Modulator } from '../mapping/modulation.ts';
import type { ModulationConfig } from '../mapping/types.ts';
import type { ImpulseConfig, ImpulseEngine } from '../sim/impulses.ts';
import { isSeedPinned } from '../sim/seed.ts';
import type { TrackEntry } from '../timeline/catalog.ts';
import {
  captureExportRecipe,
  type CapturableSimulationConfig,
} from '../runtime/capture-recipe.ts';
import type { ExportRecipe } from '../runtime/recipe.ts';

interface BrowserCaptureSim {
  readonly simId: string;
  readonly currentSeed: number;
  readonly config: CapturableSimulationConfig;
  serializeExtras?(): Record<string, unknown>;
}

export interface BrowserRecipeSource {
  sim: BrowserCaptureSim;
  modulator: Pick<Modulator, 'config' | 'baseValues'>;
  impulses: Pick<ImpulseEngine, 'config'>;
}

export interface BrowserExportRecipeOptions {
  rendererBuild: string;
  track: Pick<TrackEntry, 'id' | 'version'>;
  source: BrowserRecipeSource;
  output: ExportRecipe['output'];
  /** Injectable only to keep the browser adapter testable without DOM storage. */
  currentPinState?: (seed: number) => boolean;
}

/** Capture the exact current browser session without consulting autosave. */
export function captureBrowserExportRecipe(options: BrowserExportRecipeOptions): ExportRecipe {
  const { rendererBuild, track, source } = options;
  if (track.version === '') {
    throw new Error('selected track has no content version; reload after syncing timeline data');
  }
  return captureExportRecipe({
    rendererBuild,
    track: { id: track.id, contentVersion: track.version },
    simId: source.sim.simId,
    seed: source.sim.currentSeed,
    seedPinned: (options.currentPinState ?? isSeedPinned)(source.sim.currentSeed),
    simulation: source.sim.config,
    modulation: source.modulator.config as ModulationConfig,
    modulationBase: source.modulator.baseValues(),
    impulses: source.impulses.config as ImpulseConfig,
    serializeExtras: () => source.sim.serializeExtras?.(),
    output: options.output,
  });
}
