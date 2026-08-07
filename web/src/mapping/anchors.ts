/**
 * Anchors come from the track itself: k-means over the loaded timeline's latent
 * channel, one anchor per cluster, each carrying a full parameter preset.
 *
 * The presets generated here are *placeholders with a pulse* — deterministic
 * perturbations of the species templates in config.ts, wide enough that mapped
 * mode visibly changes character between sections out of the box. They are not
 * art direction. Hand-tuning them in the workbench is the point of phase 5.
 */
import { defaultConfig, type PhysarumConfig } from '../sim/physarum/config.ts';
import type { Timeline } from '../timeline/types.ts';
import { kmeans, makeRng } from './kmeans.ts';
import {
  clampVector,
  clonePreset,
  presetFromConfig,
  presetToVector,
  vectorToPreset,
  type Preset,
} from './preset.ts';
import { DEFAULT_SLEW } from './slew.ts';
import { MAPPING_VERSION, type Anchor, type MappingConfig } from './types.ts';

/** Fixed literals. Anchors and starter presets must not move between runs. */
export const KMEANS_SEED = 0x5eed_1e57;
export const PRESET_SEED = 0x9e37_79b9;
export const DEFAULT_ANCHOR_COUNT = 8;
export const DEFAULT_TEMPERATURE = 0.35;

export interface LatentMatrix {
  data: Float32Array;
  n: number;
  dims: number;
}

/**
 * Pack the latent channel out of the interleaved timeline into a contiguous
 * n x dims matrix. Copies ~600 KB once at load; k-means then reads it linearly.
 */
export function latentMatrix(timeline: Timeline, channelName = 'latent'): LatentMatrix | null {
  const c = timeline.channels.get(channelName);
  if (!c || c.dims === 0) return null;
  const n = timeline.manifest.grid.frames;
  const out = new Float32Array(n * c.dims);
  for (let i = 0; i < n; i++) {
    const src = i * timeline.stride + c.offset;
    out.set(timeline.data.subarray(src, src + c.dims), i * c.dims);
  }
  return { data: out, n, dims: c.dims };
}

function jitter(rng: () => number, spread: number): number {
  return 1 + (rng() * 2 - 1) * spread;
}

function offset(rng: () => number, spread: number): number {
  return (rng() * 2 - 1) * spread;
}

/**
 * Starter preset for anchor `index`. Deterministic in (index, k): seeded jitter
 * around the templates. Colour is no longer part of this — the palette is static
 * (Revision 2), so what distinguishes anchors is shape and light.
 */
export function starterPreset(base: PhysarumConfig, index: number, anchorCount: number): Preset {
  const p = presetFromConfig(base);
  if (index === 0) return p; // anchor 0 is the untouched default — a known-good home base
  const rng = makeRng((PRESET_SEED ^ (index * 0x85eb_ca6b)) >>> 0);
  // Brightness walks systematically across the anchor set as well as jittering, so
  // moving between anchors is visible as light now that it is not visible as hue.
  const brightStep = 0.7 + 0.6 * ((index / Math.max(anchorCount, 1)) % 1);

  // Structure jitters hard, exposure barely: shape is what should differ between
  // anchors, and compounding intensity x deposit x exposure just blows the
  // compositor out. The exact spreads are a starting point, not a claim.
  for (const s of p.species) {
    s.brightness *= brightStep * jitter(rng, 0.25);
    s.intensity *= jitter(rng, 0.18);
    s.deposit *= jitter(rng, 0.22);
    s.decay += offset(rng, 0.05);
    s.aliveFraction *= jitter(rng, 0.35);
    s.diffuseCentre += offset(rng, 0.25);
    s.sensorDist.p1 *= jitter(rng, 0.6);
    s.sensorDist.p2 *= jitter(rng, 0.6);
    s.sensorAngle.p1 += offset(rng, 0.3);
    s.sensorAngle.p2 += offset(rng, 0.2);
    s.rotate.p1 *= jitter(rng, 0.5);
    s.rotate.p2 *= jitter(rng, 0.5);
    s.moveDist.p1 *= jitter(rng, 0.4);
    s.moveDist.p2 *= jitter(rng, 0.4);
  }

  const k = p.species.length;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const at = i * k + j;
      const v = p.matrix[at] ?? 0;
      // self-attraction stays positive or the species stops forming networks at all
      p.matrix[at] = i === j ? Math.max(0.4, v * jitter(rng, 0.3)) : v + offset(rng, 0.45);
    }
  }
  p.senseGain *= jitter(rng, 0.4);
  p.exposure *= jitter(rng, 0.15);
  p.stemGain *= jitter(rng, 0.35);

  // Round-trip through the vector so every slot lands inside its authored range;
  // the blend can then never leave it either.
  return vectorToPreset(clampVector(presetToVector(p, k), k), k);
}

/** Mean squared distance from each centre to its nearest other centre. */
export function nearestNeighbourScale(centers: readonly number[][]): number {
  if (centers.length < 2) return 0;
  let total = 0;
  for (let i = 0; i < centers.length; i++) {
    const a = centers[i] as number[];
    let best = Infinity;
    for (let j = 0; j < centers.length; j++) {
      if (i === j) continue;
      const b = centers[j] as number[];
      let s = 0;
      for (let d = 0; d < a.length; d++) {
        const delta = (a[d] as number) - (b[d] as number);
        s += delta * delta;
      }
      if (s < best) best = s;
    }
    total += best;
  }
  return total / centers.length;
}

export interface FitOptions {
  k?: number;
  seed?: number;
  iterations?: number;
  base?: PhysarumConfig;
}

/**
 * Fit anchors to a track. `latent` null (no such channel) yields a config with no
 * anchors; the mapper then reports mapped mode as unavailable rather than
 * pretending to blend.
 */
export function fitMapping(latent: LatentMatrix | null, opts: FitOptions = {}): MappingConfig {
  const base = opts.base ?? defaultConfig();
  const k = Math.max(1, Math.floor(opts.k ?? DEFAULT_ANCHOR_COUNT));
  const seed = opts.seed ?? KMEANS_SEED;
  const iterations = opts.iterations ?? 60;

  const cfg: MappingConfig = {
    version: MAPPING_VERSION,
    speciesCount: base.speciesCount,
    // Shared by reference with the live config: a palette edit in the workbench is
    // an edit to the thing that gets serialised, with no sync step to forget.
    palette: base.palette,
    // Shared by reference for the same reason as the palette.
    render: base.render,
    latentDims: latent?.dims ?? 0,
    kmeans: { k, seed, iterations },
    temperature: DEFAULT_TEMPERATURE,
    distanceScale: 1,
    slew: { ...DEFAULT_SLEW },
    boundary: { enabled: true, snapFraction: 0.6, respawnFraction: 0.12 },
    anchors: [],
  };
  if (!latent || latent.n === 0) return cfg;

  const fit = kmeans(latent.data, latent.n, latent.dims, k, { seed, iterations });
  // Temperature is measured against how far apart *neighbouring anchors* are, not
  // against within-cluster scatter. On a track whose sections are cleanly
  // separated the two differ by two orders of magnitude, and scaling by the
  // scatter makes every softmax one-hot — the blend would never do anything.
  cfg.distanceScale = Math.max(nearestNeighbourScale(fit.centers), fit.meanSqDist, 1e-6);
  cfg.anchors = fit.centers.map((center, i) => ({
    id: `a${i}`,
    name: `anchor ${i}`,
    center,
    preset: starterPreset(base, i, fit.centers.length),
  }));
  return cfg;
}

/**
 * Re-fit centres for an existing mapping, keeping the presets that are already
 * tuned. Anchor m keeps preset m; extra anchors get starter presets.
 */
export function refitCenters(
  cfg: MappingConfig,
  latent: LatentMatrix | null,
  base: PhysarumConfig,
): MappingConfig {
  const next = fitMapping(latent, {
    k: cfg.kmeans.k,
    seed: cfg.kmeans.seed,
    iterations: cfg.kmeans.iterations,
    base,
  });
  next.temperature = cfg.temperature;
  next.slew = { ...cfg.slew };
  next.boundary = { ...cfg.boundary };
  next.anchors = next.anchors.map((a, i) => {
    const old = cfg.anchors[i];
    return old ? ({ ...a, name: old.name, preset: clonePreset(old.preset) } as Anchor) : a;
  });
  return next;
}
