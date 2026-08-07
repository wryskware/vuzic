/**
 * Seeded k-means over the track's own latent timeline. Pure TS, no deps, no
 * Math.random anywhere: the same timeline always yields the same anchors, which
 * is what makes a saved MappingConfig mean the same thing on the next load.
 *
 * Runs once at load. ~2000 frames x 64 dims x k=8 x 40 iterations is a few
 * milliseconds; nothing here is worth optimising further.
 */

/** mulberry32 — small, fast, and deterministic across engines for a u32 seed. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface KMeansOptions {
  seed?: number;
  iterations?: number;
  /** stop when no centre moves more than this (squared) */
  tolerance?: number;
}

export interface KMeansResult {
  /** k centres, each `dims` long */
  centers: number[][];
  /** one cluster index per input row */
  assignments: Int32Array;
  /** rows per cluster */
  counts: Int32Array;
  /** Σ ‖x − c(x)‖² */
  inertia: number;
  /** mean ‖x − c(x)‖² — the natural scale for the simplex temperature */
  meanSqDist: number;
}

function sqDist(data: Float32Array, row: number, dims: number, c: number[]): number {
  let s = 0;
  const o = row * dims;
  for (let d = 0; d < dims; d++) {
    const delta = (data[o + d] as number) - (c[d] as number);
    s += delta * delta;
  }
  return s;
}

function rowOf(data: Float32Array, row: number, dims: number): number[] {
  const out = new Array<number>(dims);
  const o = row * dims;
  for (let d = 0; d < dims; d++) out[d] = data[o + d] as number;
  return out;
}

/** k-means++ seeding from a seeded PRNG; D² sampling with a deterministic tie-break. */
function seedCenters(
  data: Float32Array,
  n: number,
  dims: number,
  k: number,
  rng: () => number,
): number[][] {
  const centers: number[][] = [rowOf(data, Math.min(n - 1, Math.floor(rng() * n)), dims)];
  const best = new Float64Array(n);
  for (let i = 0; i < n; i++) best[i] = sqDist(data, i, dims, centers[0] as number[]);

  while (centers.length < k) {
    let total = 0;
    for (let i = 0; i < n; i++) total += best[i] as number;
    let pick = n - 1;
    if (total <= 0) {
      // Degenerate (all points identical, or fewer distinct points than k):
      // walk deterministically instead of sampling a zero-mass distribution.
      pick = centers.length % n;
    } else {
      let target = rng() * total;
      for (let i = 0; i < n; i++) {
        target -= best[i] as number;
        if (target <= 0) {
          pick = i;
          break;
        }
      }
    }
    const c = rowOf(data, pick, dims);
    centers.push(c);
    for (let i = 0; i < n; i++) {
      const d = sqDist(data, i, dims, c);
      if (d < (best[i] as number)) best[i] = d;
    }
  }
  return centers;
}

export function kmeans(
  data: Float32Array,
  n: number,
  dims: number,
  k: number,
  opts: KMeansOptions = {},
): KMeansResult {
  const seed = opts.seed ?? 0;
  const iterations = opts.iterations ?? 60;
  const tolerance = opts.tolerance ?? 1e-9;
  const kk = Math.max(1, Math.min(k, Math.max(1, n)));

  const rng = makeRng(seed);
  let centers = seedCenters(data, n, dims, kk, rng);
  const assignments = new Int32Array(n);
  const counts = new Int32Array(kk);
  const sums = new Float64Array(kk * dims);
  const bestDist = new Float64Array(n);

  for (let iter = 0; iter < iterations; iter++) {
    counts.fill(0);
    sums.fill(0);

    for (let i = 0; i < n; i++) {
      let bi = 0;
      let bd = Infinity;
      for (let c = 0; c < kk; c++) {
        const d = sqDist(data, i, dims, centers[c] as number[]);
        // strict < keeps the lowest index on ties, so assignment is order-stable
        if (d < bd) {
          bd = d;
          bi = c;
        }
      }
      assignments[i] = bi;
      bestDist[i] = bd;
      counts[bi] = (counts[bi] as number) + 1;
      const o = i * dims;
      const so = bi * dims;
      for (let d = 0; d < dims; d++) sums[so + d] = (sums[so + d] as number) + (data[o + d] as number);
    }

    // Empty clusters take the single worst-fit point — deterministic, and it
    // stops a cluster from silently vanishing and skewing the softmax.
    for (let c = 0; c < kk; c++) {
      if ((counts[c] as number) > 0) continue;
      let worst = 0;
      let worstD = -1;
      for (let i = 0; i < n; i++) {
        if ((bestDist[i] as number) > worstD && (counts[assignments[i] as number] as number) > 1) {
          worstD = bestDist[i] as number;
          worst = i;
        }
      }
      const from = assignments[worst] as number;
      counts[from] = (counts[from] as number) - 1;
      counts[c] = 1;
      const o = worst * dims;
      for (let d = 0; d < dims; d++) {
        sums[from * dims + d] = (sums[from * dims + d] as number) - (data[o + d] as number);
        sums[c * dims + d] = data[o + d] as number;
      }
      assignments[worst] = c;
      bestDist[worst] = 0;
    }

    let shift = 0;
    for (let c = 0; c < kk; c++) {
      const cnt = counts[c] as number;
      if (cnt === 0) continue;
      const cen = centers[c] as number[];
      for (let d = 0; d < dims; d++) {
        const v = (sums[c * dims + d] as number) / cnt;
        const delta = v - (cen[d] as number);
        shift += delta * delta;
        cen[d] = v;
      }
    }
    if (shift <= tolerance) break;
  }

  // Final assignment pass against the settled centres, then a stable re-ordering
  // by cluster size so anchor 0 is always the track's dominant character.
  const order = Array.from({ length: kk }, (_, i) => i).sort((a, b) => {
    const d = (counts[b] as number) - (counts[a] as number);
    if (d !== 0) return d;
    const ca = centers[a] as number[];
    const cb = centers[b] as number[];
    for (let i = 0; i < dims; i++) {
      const delta = (ca[i] as number) - (cb[i] as number);
      if (delta !== 0) return delta;
    }
    return a - b;
  });
  centers = order.map((from) => centers[from] as number[]);
  const sortedCounts = new Int32Array(kk);
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    let bi = 0;
    let bd = Infinity;
    for (let c = 0; c < kk; c++) {
      const d = sqDist(data, i, dims, centers[c] as number[]);
      if (d < bd) {
        bd = d;
        bi = c;
      }
    }
    assignments[i] = bi;
    sortedCounts[bi] = (sortedCounts[bi] as number) + 1;
    inertia += bd;
  }

  return {
    centers,
    assignments,
    counts: sortedCounts,
    inertia,
    meanSqDist: n > 0 ? inertia / n : 0,
  };
}
