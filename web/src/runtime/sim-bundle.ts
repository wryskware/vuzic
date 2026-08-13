import { type DriverBank, Modulator } from '../mapping/modulation';
import type { ModulationConfig } from '../mapping/types';
import { ImpulseEngine } from '../sim/impulses';
import { defaultConfig } from '../sim/physarum/config';
import { PhysarumSim } from '../sim/physarum/physarum';
import { defaultPlifeConfig } from '../sim/plife/config';
import { PlifeSim } from '../sim/plife/plife';
import { VIZFX_VISUALS } from '../sim/vizfx/visuals';
import { VizFxSim } from '../sim/vizfx/vizfx';
import type { TimelineSampler } from '../timeline/sampler';

/** Every concrete simulation currently constructible by the shared runtime. */
export type RuntimeSim = PhysarumSim | PlifeSim | VizFxSim;

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
  /**
   * Host policy for selecting authored modulation state. Browser hosts may read
   * local persistence; export hosts can return recipe state. The shared runtime
   * deliberately knows neither source.
   */
  resolveModulationConfig(sim: RuntimeSim): ModulationConfig;
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
  const { id, seed, sampler, drivers, secondsPerTick, resolveModulationConfig } = options;
  const visual = VIZFX_VISUALS.find((candidate) => candidate.id === id);
  const sim: RuntimeSim =
    id === 'plife'
      ? new PlifeSim(seed, defaultPlifeConfig())
      : visual
        ? new VizFxSim(visual, seed)
        : new PhysarumSim(seed, defaultConfig(4));

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
  );
  sim.setImpulses(impulses.state);

  const modConfig = resolveModulationConfig(sim);
  const modulator = new Modulator(sim, sampler, drivers, modConfig, seed);
  // Enabling modulation adopts the target's current theta as its base. Stamp
  // the seeded personality first so it cannot be replaced by shipped defaults.
  modulator.applyBase();
  modulator.setMode(modConfig.enabled && modulator.available ? 'modulated' : 'manual');

  // Hotspots, projection wiring and seeded personality are one coherent world.
  sim.onSeedChange = (nextSeed) => {
    impulses.setSeed(nextSeed);
    modulator.setSeed(nextSeed);
  };

  return { sim, impulses, modulator };
}
