import type { ModulationConfig } from '../mapping/types.ts';
import type { ImpulseConfig } from '../sim/impulses.ts';
import type { PhysarumConfig } from '../sim/physarum/config.ts';
import type { PlifeConfig } from '../sim/plife/config.ts';
import type { VizFxConfig } from '../sim/vizfx/config.ts';
import { isVizFxId } from '../sim/vizfx/ids.ts';
import {
  EXPORT_RECIPE_VERSION,
  validateExportRecipe,
  type ExportRecipe,
  type RecipeModulationConfig,
  type SimulationBaseConfig,
} from './recipe.ts';

export type CapturableSimulationConfig = PhysarumConfig | PlifeConfig | VizFxConfig;

export interface ExportRecipeCaptureState {
  rendererBuild: string;
  track: {
    id: string;
    contentVersion: string;
  };
  simId: string;
  seed: number;
  seedPinned: boolean;
  simulation: CapturableSimulationConfig;
  modulation: ModulationConfig;
  /** Authored base, not the time-varying theta currently written into the sim. */
  modulationBase: ArrayLike<number>;
  impulses: ImpulseConfig;
  /** Re-read from the sim at capture time; autosave may not have run yet. */
  serializeExtras?: () => Record<string, unknown> | undefined;
  output: ExportRecipe['output'];
}

function particleBudget(simId: string, config: CapturableSimulationConfig): number {
  if (simId === 'physarum' && 'maxAgents' in config) return config.maxAgents;
  if (simId === 'plife' && 'budget' in config) return config.budget.cap;
  if (isVizFxId(simId) && 'params' in config) return 0;
  throw new Error(`export recipe capture: config does not match sim "${simId}"`);
}

/**
 * Snapshot live authored state into an independently owned recipe.
 *
 * This function has no DOM, storage, or local-persistence dependency. Its
 * browser adapter decides where each live value comes from; this layer owns the
 * important invariant that no live object is mutated or retained by the result.
 */
export function captureExportRecipe(state: ExportRecipeCaptureState): ExportRecipe {
  const clonedSimulation = structuredClone(state.simulation);
  const { render, ...simulation } = clonedSimulation;
  const clonedModulation = structuredClone(state.modulation);
  const extras = state.serializeExtras?.();
  if (extras === undefined) delete clonedModulation.extras;
  else clonedModulation.extras = structuredClone(extras);
  const { render: _sharedRender, ...modulation } = clonedModulation;
  void _sharedRender;

  const recipe: ExportRecipe = {
    version: EXPORT_RECIPE_VERSION,
    rendererBuild: state.rendererBuild,
    track: structuredClone(state.track),
    sim: state.simId,
    seed: state.seed,
    seedPinned: state.seedPinned,
    simulation: simulation as SimulationBaseConfig,
    modulation: modulation as RecipeModulationConfig,
    modulationBase: Array.from(state.modulationBase),
    impulses: structuredClone(state.impulses),
    render,
    particleBudget: particleBudget(state.simId, clonedSimulation),
    presentation: { mode: 'single', autoAdvance: false },
    output: structuredClone(state.output),
  };
  validateExportRecipe(recipe);
  return recipe;
}
