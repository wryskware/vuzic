import { type DriverBank, Modulator } from '../mapping/modulation';
import type { ModulationConfig } from '../mapping/types';
import { ImpulseEngine, type ImpulseConfig } from '../sim/impulses';
import { defaultConfig, type PhysarumConfig } from '../sim/physarum/config';
import { PhysarumSim } from '../sim/physarum/physarum';
import { defaultPlifeConfig, type PlifeConfig } from '../sim/plife/config';
import { PlifeSim } from '../sim/plife/plife';
import type { VizFxConfig } from '../sim/vizfx/config';
import { VIZFX_VISUALS } from '../sim/vizfx/visuals';
import { VizFxSim } from '../sim/vizfx/vizfx';
import type { TimelineSampler } from '../timeline/sampler';
import {
  validateExportRecipe,
  type ExportRecipe,
  type RecipeModulationConfig,
  type SimulationBaseConfig,
} from './recipe';

/** Every concrete simulation currently constructible by the shared runtime. */
export type RuntimeSim = PhysarumSim | PlifeSim | VizFxSim;
export type RuntimeSimulationConfig = PhysarumConfig | PlifeConfig | VizFxConfig;

/** A substrate and the two substrate-shaped lanes that must be rebuilt with it. */
export interface SimBundle {
  sim: RuntimeSim;
  impulses: ImpulseEngine;
  modulator: Modulator;
}

export interface BuildSimBundleOptions {
  id: string;
  seed: number;
  sampler: TimelineSampler;
  drivers: DriverBank | null;
  secondsPerTick: number;
  /** Complete authored state. Omitted by the browser to retain shipped defaults. */
  simulationConfig?: RuntimeSimulationConfig;
  /** Complete authored event-response state. Omitted by the browser for defaults. */
  impulseConfig?: ImpulseConfig;
  /** Exact authored modulation centre. Omitted by the browser for seeded personality. */
  modulationBase?: ArrayLike<number>;
  /**
   * Host policy for selecting authored modulation state. Browser hosts may read
   * local persistence; export hosts can return recipe state. The shared runtime
   * deliberately knows neither source.
   */
  resolveModulationConfig(sim: RuntimeSim): ModulationConfig;
}

export interface BuildRecipeSimBundleOptions {
  recipe: ExportRecipe;
  sampler: TimelineSampler;
  drivers: DriverBank | null;
  secondsPerTick: number;
}

interface RecipeRuntimeState {
  id: string;
  seed: number;
  simulationConfig: RuntimeSimulationConfig;
  impulseConfig: ImpulseConfig;
  modulationConfig: ModulationConfig;
  modulationBase: number[];
}

function owns(value: object, key: string): boolean {
  return Object.hasOwn(value, key);
}

function assertSimulationConfigMatches(
  id: string,
  config: RuntimeSimulationConfig,
  visual: (typeof VIZFX_VISUALS)[number] | undefined,
): void {
  if (id === 'plife') {
    if (!owns(config, 'maxParticles') || owns(config, 'maxAgents') || owns(config, 'params')) {
      throw new Error('simulation config for "plife" is not a particle-life config');
    }
    return;
  }
  if (visual) {
    if (!owns(config, 'params') || owns(config, 'maxAgents') || owns(config, 'maxParticles')) {
      throw new Error(`simulation config for "${id}" is not a VizFX config`);
    }
    if (config.speciesCount !== visual.speciesCount) {
      throw new Error(
        `simulation config for "${id}" has ${config.speciesCount} species; expected ${visual.speciesCount}`,
      );
    }
    return;
  }
  if (id !== 'physarum') {
    throw new Error(`cannot apply an authored simulation config to unknown sim "${id}"`);
  }
  if (!owns(config, 'maxAgents') || owns(config, 'maxParticles') || owns(config, 'params')) {
    throw new Error('simulation config for "physarum" is not a physarum config');
  }
}

/**
 * Rehydrate the recipe without handing any of its supposedly immutable objects
 * to a runtime that edits configs in place. Palette and render are deliberately
 * re-shared: they are one live object in the browser, even though JSON has to
 * encode the palette in both the simulation and modulation blocks.
 */
function runtimeStateFromRecipe(recipe: ExportRecipe): RecipeRuntimeState {
  validateExportRecipe(recipe);
  const base = structuredClone(recipe.simulation) as SimulationBaseConfig;
  const render = structuredClone(recipe.render);
  const simulationConfig = { ...base, render } as RuntimeSimulationConfig;
  const savedModulation = structuredClone(recipe.modulation) as RecipeModulationConfig;
  const modulationConfig: ModulationConfig = {
    ...savedModulation,
    palette: simulationConfig.palette,
    render: simulationConfig.render,
  };
  return {
    id: recipe.sim,
    seed: recipe.seed,
    simulationConfig,
    impulseConfig: structuredClone(recipe.impulses),
    modulationConfig,
    modulationBase: recipe.modulationBase.slice(),
  };
}

/**
 * Construct one simulation and every substrate-shaped input lane around it.
 *
 * Ordering here is load-bearing: direct timeline channels are wired before the
 * first tick, the seeded base is applied before modulation mode is enabled, and
 * the coordinated seed callback is installed before any host can initialise or
 * reseed the simulation. GPU initialisation remains the host's responsibility.
 */
export function buildSimBundle(options: BuildSimBundleOptions): SimBundle {
  const {
    id,
    seed,
    sampler,
    drivers,
    secondsPerTick,
    simulationConfig,
    impulseConfig,
    modulationBase,
    resolveModulationConfig,
  } = options;
  const visual = VIZFX_VISUALS.find((candidate) => candidate.id === id);
  if (simulationConfig) assertSimulationConfigMatches(id, simulationConfig, visual);
  const sim: RuntimeSim =
    id === 'plife'
      ? new PlifeSim(seed, (simulationConfig as PlifeConfig | undefined) ?? defaultPlifeConfig())
      : visual
        ? new VizFxSim(visual, seed, simulationConfig as VizFxConfig | undefined)
        : new PhysarumSim(
            seed,
            (simulationConfig as PhysarumConfig | undefined) ?? defaultConfig(4),
          );

  // This is separate from the Modulator's stem-follow brightness lane: each
  // substrate consumes its direct stem channel for its own world mechanics.
  sim.setStemChannel(sampler.getChannel('stems'));
  if (sim instanceof PlifeSim) {
    sim.setAccentChannels(sampler.getChannel('novelty16'), sampler.getChannel('actChorus'));
  }

  const impulses = new ImpulseEngine(
    seed,
    sim.config.speciesCount,
    sampler.events,
    secondsPerTick,
    impulseConfig,
  );
  sim.setImpulses(impulses.state);

  const modConfig = resolveModulationConfig(sim);
  const modulator = new Modulator(sim, sampler, drivers, modConfig, seed);
  // Construction rewires the seed internally but has not written it to the
  // target yet. A recipe replaces that internal centre before the one and only
  // apply, so neither seed personality nor a captured music excursion leaks in.
  if (modulationBase) modulator.restoreBase(modulationBase);
  modulator.applyBase();
  modulator.setMode(modConfig.enabled && modulator.available ? 'modulated' : 'manual');

  // Hotspots, projection wiring and seeded personality are one coherent world.
  sim.onSeedChange = (nextSeed) => {
    impulses.setSeed(nextSeed);
    modulator.setSeed(nextSeed);
  };

  return { sim, impulses, modulator };
}

/** Canonical canvas/localStorage-free construction entry point for export hosts. */
export function buildSimBundleFromRecipe(options: BuildRecipeSimBundleOptions): SimBundle {
  const { recipe, sampler, drivers, secondsPerTick } = options;
  const state = runtimeStateFromRecipe(recipe);
  return buildSimBundle({
    id: state.id,
    seed: state.seed,
    sampler,
    drivers,
    secondsPerTick,
    simulationConfig: state.simulationConfig,
    impulseConfig: state.impulseConfig,
    modulationBase: state.modulationBase,
    resolveModulationConfig: () => state.modulationConfig,
  });
}
