// plasma — the warp pass.
//
// One fullscreen resample of the previous field: for each output pixel, work out
// where that pixel's light *came from* one step ago, fetch it, soften it, dim it
// and rotate its hue.
//
// Nebula's warp is where its structure is *made* — twelve blobs become spiral
// arms because a hundred warped copies of them are on screen at once. Plasma's
// warp is where its structure is *deformed*: the draw pass already paints a
// full-frame interference pattern, and this pass drags that pattern through a
// slow, large-scale current so it folds into itself instead of standing still.
//
// The transform is written as an inverse map (destination → source) because that
// is the only direction a fragment shader can express — forward-mapping would
// need a scatter — so it reads backwards from the effect. It is the composition
// of exactly two maps, in this order:
//
//   1. zoom    a pure radial scale about the frame centre. Slow. This is the one
//              area-changing term and it is priced accordingly (see below).
//   2. flow    a displacement field built from a STREAM FUNCTION, which makes it
//              divergence-free by construction and its area Jacobian exactly
//              computable rather than a bound somebody guessed at.
//
// There is deliberately no rotation and no swirl. A rigid rotation of a
// full-frame band field reads as the camera turning rather than as the medium
// moving, and a radius-keyed differential twist would fight the flow for the
// same job while adding a third Jacobian to keep honest. The flow does all of
// it, and being time-dependent (`flowSpeed`) it does it without ever settling.

// ── the stream function ───────────────────────────────────────────────────────
//
// Three sinusoids. Not two (two plane waves interfere into a stationary
// chequerboard of eddies, which after ten seconds is visibly a lattice) and not
// six (the extra terms cost the same per pixel and produce a flow whose
// correlation length is shorter than a fringe, which shreds the pattern instead
// of carrying it).

const FLOW_WAVES: u32 = 3u;

/**
 * Relative spatial-frequency multipliers of the three waves, and their
 * amplitude weights (which sum to 1, so `flow` is the whole displacement).
 *
 * Irrational-ish ratios rather than 1 : 2 : 3, so the three never come back into
 * register and the current genuinely never repeats. Amplitude falls with
 * frequency for the same reason a natural flow spectrum does: the large scale
 * has to carry, the small scales only have to stop it from looking like a single
 * sine. This weighting is also what keeps `Σ aᵢmᵢ² = 2.10` small, and that sum
 * is the coefficient of the Jacobian correction derived below.
 */
const WAVE_M0: f32 = 1.0;
const WAVE_M1: f32 = 1.53;
const WAVE_M2: f32 = 2.17;
const WAVE_A0: f32 = 0.5;
const WAVE_A1: f32 = 0.32;
const WAVE_A2: f32 = 0.18;

/**
 * Angular separation of the three wave vectors, radians. The golden angle, so
 * that no two of them are near-parallel (which would collapse the flow to one
 * dimension and produce visible banding in the current itself) and no two are
 * near-perpendicular for long.
 */
const GOLDEN_ANGLE: f32 = 2.39996323;

/** How much of `pulseShock` is spent spiking the flow rather than the phase. */
const PULSE_FLOW: f32 = 0.55;

/** The window height the `blurRadius` slot's texel count is quoted at. */
const BLUR_REFERENCE_HEIGHT: f32 = 1080.0;

/**
 * A cheap symmetric blur at `radius` texels: centre plus four diagonals, equal
 * weights. Same construction as nebula's, for the same two reasons.
 *
 * Diagonals rather than an axis-aligned cross, because with linear filtering four
 * diagonal taps cover the neighbourhood more evenly — an axial cross leaves the
 * corners unsampled and the accumulated field grows a faint plus-shaped
 * anisotropy over a few hundred steps, which is exactly the timescale this
 * substrate runs on. And the offset is scaled against a 1080-line reference so
 * that `blurRadius` describes a fraction of the *image* rather than a count of
 * device pixels; without that it is the one θ slot whose meaning changes with the
 * window, and a 1/3-size explorer tile would be judging a blur three times
 * stronger than the candidate gives you full-screen.
 *
 * Worth stating for the energy argument below: the five weights sum to exactly
 * 1, so this is a normalised low-pass. It moves light around and never creates
 * or destroys any, which is why it does not appear in the loop gain.
 */
fn blurAt(uv: vec2f, radius: f32) -> vec3f {
  let scale = max(g.res.y / BLUR_REFERENCE_HEIGHT, 0.25);
  let o = g.invRes * max(radius, 0.0) * scale;
  var s = textureSampleLevel(field, fieldSamp, uv, 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(o.x, o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(-o.x, o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(o.x, -o.y), 0.0).rgb;
  s = s + textureSampleLevel(field, fieldSamp, uv + vec2f(-o.x, -o.y), 0.0).rgb;
  return s * 0.2;
}

@fragment
fn fsWarp(in: FsIn) -> @location(0) vec4f {
  let p = toCentre(in.uv);

  // The event lane, in the geometry. `pulse` is 0 at rest and rises with the
  // deposit envelope of whatever just fired. Here it spikes the flow amplitude,
  // so a hit is a gust; in the draw pass the same slot shifts the phase of every
  // fringe. Together that is the visual's transient idiom — the whole frame
  // ripples rather than one spot lighting up.
  let pulse = clamp(g.pulse, 0.0, 4.0);
  let shockAmt = max(th(TH_pulseShock), 0.0);

  // ── map 1: the radial breath ────────────────────────────────────────────────
  //
  // A pure scale about the frame centre. `zoomRate` is a per-step *rate* (the
  // stored number is the growth, not 1 + growth — see the slot's comment), so
  // the factor is built here. Floored well above 0 because it divides a radius.
  let zoom = max(1.0 + th(TH_zoomRate), 0.25);
  let q = p / zoom;

  // ── map 2: the divergence-free flow ─────────────────────────────────────────
  //
  // ψ̂(x) = Σᵢ (aᵢ/k)·sin(kᵢ·x + φᵢ)   with kᵢ = k·mᵢ·uᵢ, |uᵢ| = 1, Σaᵢ = 1
  //
  // and the displacement is the perpendicular gradient of `flow · ψ̂`:
  //
  //   F = ∇⊥ψ̂ = (∂ψ̂/∂y, −∂ψ̂/∂x) = Σᵢ aᵢmᵢ·cos(argᵢ)·(u_iy, −u_ix)
  //
  // The 1/k in ψ̂ is why `flow` can be authored in screen heights: it makes
  // |F| ≤ Σ aᵢmᵢ = 1.38 regardless of what `flowSize` is set to, so the slot is a
  // distance and not a distance-times-a-frequency.
  let k = TAU / max(th(TH_flowSize), 0.05);
  let amp = max(th(TH_flow), 0.0) * (1.0 + PULSE_FLOW * shockAmt * pulse);

  // Seeded: a reroll gets a different current through the same machinery, and it
  // is drawn from the same PCG the CPU places the wave sources with, so "seeded"
  // means one thing across the project. Uniform across the frame by construction
  // — every fragment hashes the same (seed, i).
  let phase0 = rand01(hash3(g.seed, 0x5f1c9a3bu, 1u)) * TAU;
  let spin = g.time * max(th(TH_flowSpeed), 0.0);

  var wm = array<f32, 3>(WAVE_M0, WAVE_M1, WAVE_M2);
  var wa = array<f32, 3>(WAVE_A0, WAVE_A1, WAVE_A2);

  var disp = vec2f(0.0);
  // The three independent entries of the Hessian of ψ̂, divided by −k. Named for
  // the sums they are: sxx = Σ aᵢmᵢ²·u_ix²·sin(argᵢ), and so on.
  var sxx = 0.0;
  var syy = 0.0;
  var sxy = 0.0;
  for (var i = 0u; i < FLOW_WAVES; i = i + 1u) {
    let m = wm[i];
    let a = wa[i];
    let dirAng = phase0 + f32(i) * GOLDEN_ANGLE;
    let u = vec2f(cos(dirAng), sin(dirAng));
    // Each wave slides at its own rate, so the three drift out of and back into
    // register and the eddy pattern rearranges itself rather than translating.
    let arg =
      dot(u, q) * k * m
      + spin * (0.7 + 0.35 * f32(i))
      + rand01(hash3(g.seed, 0x5f1c9a3bu, i + 7u)) * TAU;
    let c = cos(arg);
    let s = sin(arg);
    disp = disp + a * m * c * vec2f(u.y, -u.x);
    let am2 = a * m * m * s;
    sxx = sxx + am2 * u.x * u.x;
    syy = syy + am2 * u.y * u.y;
    sxy = sxy + am2 * u.x * u.y;
  }

  let src = toUv(q + amp * disp);

  // The sampler is mirror-repeat, so a source coordinate pushed outside the frame
  // by the flow folds back instead of smearing the edge texel across the border —
  // which is what clamp-to-edge does, and over a few hundred accumulation steps
  // it builds a bright picture-frame that never decays.
  var c = textureSampleLevel(field, fieldSamp, src, 0.0).rgb;
  let mixAmt = clamp(th(TH_blurMix), 0.0, 1.0);
  if (mixAmt > 0.002) {
    c = mix(c, blurAt(src, th(TH_blurRadius)), mixAmt);
  }

  // ── the resample has to conserve light ──────────────────────────────────────
  //
  // This is a *gather*: each destination pixel copies the value it finds at
  // `src`. Total light therefore transforms as
  //
  //     ∫ c_dst d(dst) = ∫ c_src(s) / |det ∂src/∂dst| ds
  //
  // so to conserve it the sampled value must be MULTIPLIED by the map's area
  // Jacobian J = |det ∂src/∂dst|. Get it wrong and the resample is not a
  // transport, it is an amplifier compounding inside a feedback loop: nebula's
  // table records exactly that failure, where the loop gain was (1 − fade)·zoom²
  // = 0.997 instead of 0.99, the field equilibrated 47× too high, and the frame
  // came out as a white plate with an *inverted* radial profile because light was
  // being manufactured in flight and piled up against a boundary it cannot cross.
  //
  // The map here is a composition of two, so the Jacobians multiply (chain rule),
  // and each is handled exactly.
  //
  // ## J₁ — the zoom
  //
  // map₁(dst) = dst / zoom, so ∂map₁/∂dst = I/zoom and det = 1/zoom². Floored at
  // 1 in the denominator so the term can only ever DIM. `zoomRate` may legally go
  // negative — the documented inward-flow variant — and there the honest Jacobian
  // is greater than 1, which would put the loop gain back above 1 for exactly the
  // reason above with the sign flipped. A drain that concentrates light and loses
  // a little of it is a drain; a drain that manufactures it is the bug.
  let jZoom = 1.0 / max(zoom * zoom, 1.0);
  //
  // ## J₂ — the flow, exactly
  //
  // For src = s + Ψ_⊥(s) with Ψ = flow·ψ̂ and Ψ_⊥ = (Ψ_y, −Ψ_x):
  //
  //   DF = [ Ψ_yx   Ψ_yy ]      tr(DF) = Ψ_yx − Ψ_xy = 0   ← EXACTLY, because
  //        [ −Ψ_xx  −Ψ_xy ]                                   mixed partials commute
  //
  //   det(I + DF) = 1 + tr(DF) + det(DF) = 1 + (Ψ_xx·Ψ_yy − Ψ_xy²)
  //
  // The first-order term vanishing is the whole reason the displacement is built
  // from a stream function rather than from "some sinusoids" or from fbm: a
  // general displacement field has a first-order Jacobian term A·div F that is
  // linear in the amplitude and does not average away, and there is no honest
  // cheap way to compute it. Here the leading correction is second order and
  // written out below in closed form.
  //
  // With ψ̂ = Σ (aᵢ/k) sin(argᵢ) the second derivatives are
  //
  //   ψ̂_xx = −k·Σ aᵢmᵢ²·u_ix²·sin(argᵢ)       (= −k·sxx)
  //   ψ̂_yy = −k·Σ aᵢmᵢ²·u_iy²·sin(argᵢ)       (= −k·syy)
  //   ψ̂_xy = −k·Σ aᵢmᵢ²·u_ix·u_iy·sin(argᵢ)   (= −k·sxy)
  //
  // so det(Hess ψ̂) = k²·(sxx·syy − sxy²) and, since Ψ = flow·ψ̂ scales the whole
  // Hessian linearly,
  //
  //   J₂ = 1 + flow²·k²·(sxx·syy − sxy²)
  //
  // — three multiply-adds already accumulated in the loop above. Note it is a
  // *product* of two sums, so a single wave gives exactly J₂ = 1 (one sinusoidal
  // shear is area-preserving to all orders); the correction is entirely the
  // interaction between the three.
  //
  // ## How big it actually is
  //
  // The Hessian is M = Σ cᵢ·uᵢuᵢᵀ with cᵢ = aᵢmᵢ²·sin(argᵢ), and for a sum of
  // rank-1 terms det M = Σ_{i<j} cᵢcⱼ·sin²(θᵢⱼ). With the golden-angle
  // separations that is bounded by 0.88, six times tighter than the naive
  // |sxx||syy| + sxy² bound of 5.5. So
  //
  //   |J₂ − 1| ≤ 0.88 · flow² · k²
  //
  //   defaults      flow 0.008, flowSize 1.4 (k² = 20.1)        → 0.11 %
  //   mod extremes  flow 0.014, flowSize 0.7 (k² = 80.6)        → 1.4 %
  //   …with a capped pulse and pulseShock at its ceiling (×3.2) → 14 %
  //
  // At the defaults this is a rounding error against `fade` = 2 %/step; at the
  // corner it is a real extra loss, which is why `flow`'s mod ceiling is where it
  // is (see the slot's comment). Beyond the mod ranges — a hand-set θ, or `scale`
  // at 0 pinning `flowSize` to its hard floor — the term can exceed 1 and the map
  // is genuinely folded, i.e. not injective; the clamp is what makes that
  // degenerate rather than divergent.
  //
  // ## The invariant
  //
  // Both factors are clamped to ≤ 1, so for EVERY θ this table can produce
  //
  //   loop gain = clamp(1 − fade, 0, 0.9995) · J₁ · J₂ ≤ 1 − fade
  //
  // and the field's memory is `1/fade` steps or shorter, never longer. Clamping
  // J₂ at 1 rather than letting it exceed 1 means the flow is a net *loss* — the
  // converging half of each eddy dims and the diverging half does not brighten —
  // and at the defaults that costs about 0.05 %/step, which is 2.5 % of the fade
  // rate and invisible. That asymmetry is the price of the guarantee and it is
  // the right way round: a feedback loop may lose light for a reason nobody
  // notices, and may never gain it.
  //
  // One coordinate note: the aspect correction cancels. `toCentre` is a linear
  // map A and `toUv` is A⁻¹, so det(∂src_uv/∂dst_uv) = det(A⁻¹)·det(DM)·det(A) =
  // det(DM) — the Jacobian computed in centred coordinates is the Jacobian in uv.
  let jFlow = clamp(1.0 + amp * amp * k * k * (sxx * syy - sxy * sxy), 0.0, 1.0);
  let jacobian = jZoom * jFlow;

  // Decay last, so the blur mixes light that has not yet been dimmed twice.
  // Clamped strictly under 1: at a fade of exactly 0 the field is a perfect
  // integrator and a pass that paints most of the frame every step saturates it
  // into a white plate within seconds, with no path back down. The θ bound
  // already stops that; this stops a macro or an arithmetic slip from getting
  // round it.
  c = c * (clamp(1.0 - th(TH_fade), 0.0, 0.9995) * jacobian);

  // The chroma lane. Applied to the FEEDBACK rather than to the draw, which is
  // what makes it accumulate: light painted now is the layer's own colour, and it
  // walks around the hue circle over the following second as the flow carries it.
  // In a band field that shows up as a gradient *across* each fringe — the
  // leading edge is the layer's hue and the trailing edge is where it has drifted
  // to — which is structure the palette alone cannot produce.
  return vec4f(hueRotate(c, th(TH_chromaShift)), 1.0);
}
