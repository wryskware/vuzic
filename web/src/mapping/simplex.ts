/**
 * The preset simplex: w = softmax(−‖z − cₘ‖² / T).
 *
 * Structurally an RBF network whose training data is the hand-tuned presets.
 * The only property that matters downstream is that w is non-negative and sums
 * to 1, because that is what keeps θ = Σ wₘ θₘ inside the convex hull of
 * known-good parameter sets — the sim can never be driven somewhere no human
 * ever looked at.
 */

export function squaredDistances(
  z: ArrayLike<number>,
  centers: readonly number[][],
  out: Float64Array,
): Float64Array {
  for (let m = 0; m < centers.length; m++) {
    const c = centers[m] as number[];
    let s = 0;
    const n = Math.min(c.length, z.length);
    for (let d = 0; d < n; d++) {
      const delta = (z[d] as number) - (c[d] as number);
      s += delta * delta;
    }
    out[m] = s;
  }
  return out;
}

/**
 * Weights for a latent vector. `temperature` is absolute (same units as the
 * squared distances) — callers scale it by the k-means mean squared distance so
 * the knob means the same thing on every track.
 *
 * T → 0 degenerates to one-hot on the nearest anchor, which is exactly what
 * "solo this anchor" wants; it is handled explicitly rather than left to
 * exp(−big) underflow.
 */
export function softmaxWeights(
  z: ArrayLike<number>,
  centers: readonly number[][],
  temperature: number,
  out: Float64Array,
  scratch?: Float64Array,
): Float64Array {
  const m = centers.length;
  if (m === 0) return out;
  const d2 = scratch && scratch.length >= m ? scratch : new Float64Array(m);
  squaredDistances(z, centers, d2);

  let nearest = 0;
  let best = Infinity;
  for (let i = 0; i < m; i++) {
    if ((d2[i] as number) < best) {
      best = d2[i] as number;
      nearest = i;
    }
  }

  if (!(temperature > 1e-12)) {
    out.fill(0);
    out[nearest] = 1;
    return out;
  }

  // shift by the minimum before exp: the largest term is exactly 1, so nothing
  // overflows and the nearest anchor never underflows to a zero-sum
  let sum = 0;
  for (let i = 0; i < m; i++) {
    const e = Math.exp(-((d2[i] as number) - best) / temperature);
    out[i] = e;
    sum += e;
  }
  if (!(sum > 0) || !Number.isFinite(sum)) {
    out.fill(0);
    out[nearest] = 1;
    return out;
  }
  for (let i = 0; i < m; i++) out[i] = (out[i] as number) / sum;
  return out;
}

/** One-hot on `index`. Used by the workbench's anchor solo. */
export function oneHot(index: number, m: number, out: Float64Array): Float64Array {
  out.fill(0);
  if (index >= 0 && index < m) out[index] = 1;
  return out;
}
