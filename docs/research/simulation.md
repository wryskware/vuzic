# Research — Simulation Substrate & the Mapping Problem (surveyed 2026-08)

Candidate visual systems for the terrarium, judged on three axes: genuine
emergence, robustness under continuous parameter modulation, and
implementability by one developer. Decisions live in `docs/plan.md`.

## The axis nobody thinks about first, which decides everything

Most of these systems look great in a demo where a human hand-tunes parameters
and waits. This project instead **sweeps parameters continuously from a music
timeline for four minutes without supervision**. That reframes the choice
entirely: what matters is not "how beautiful at its best" but "how does it
behave across a whole path through parameter space, including the parts you
didn't test".

By that measure the field separates sharply.

## Continuous cellular automata

### Lenia — conceptually perfect, practically hostile
`A ← clip(A + dt·G(K∗A))` with a radial ring kernel and Gaussian growth. Organic
membranous creatures; strong but *fragile* identity.

The problem is the parameter space. *Visualizing the Structure of Lenia
Parameter Space* (arXiv 2601.01932, Jan 2026) mapped >10⁵ systems across the
μ–σ plane and found soliton-bearing zones sit **only in a thin band at the
stable/metastable transition**, with a phase diagram "strikingly resembling the
phase transition of water" and reported fractal structure at the boundary.
*Looking for Complexity at Phase Boundaries* (arXiv 2402.17848) found 30.5%
"interesting" *on* the boundary versus 14% from random sampling — meaning even
on the good shell, ~70% of samples are uninteresting.

The sensorimotor-agency work (Science Advances 2024) states plainly that
"changing some parameters too much can easily break the dynamic"; a perturbation
study (arXiv 2605.30708) found occlusion leads to "death, metamorphosis, or
explosion" depending on extent.

**Interesting behaviour lives on a thin fractal shell, so the midpoint of two
good parameter sets is usually bad.** That breaks every safe interpolation
scheme. Not viable as a v1 substrate.

*(**Asymptotic Lenia** — Kawaguchi et al. 2021 — replaces the discontinuous
`clip` with an asymptotic relaxation toward a target state, removing
discretisation artifacts and improving structural stability. Meaningfully better
if one insists on CA.)*

### Flow-Lenia — the intellectually correct answer
Plantec, Hamon, Etcheverry, Chan, Oudeyer, Moulin-Frier. ALIFE 2023 **Best
Paper**; journal version *Artificial Life* 31(2):228–248 (arXiv 2506.08569).
https://github.com/erwanplantec/FlowLenia (JAX)

Mass-conservative Lenia. Activations are *concentrations of matter*. Compute an
affinity map U, then a velocity field `F = α_weighted(∇U − ∇A)` where α makes
the negative concentration gradient dominate in dense regions (the anti-blowup
term). Matter is transported by **reintegration tracking** — mass-conserving
semi-Lagrangian advection.

Two properties make it uniquely relevant:

1. **Parameter localisation.** Update-rule parameters are attached as **spatial
   maps that flow with the matter**. When matter from cells with different
   parameters meets, parameters mix by mass-weighted average or softmax-by-
   incoming-mass. This is a native, peer-reviewed mechanism for *"the rules vary
   in space and time and are carried around by the stuff"* — essentially the
   architecture the brief describes.
2. **Mass conservation as an intrinsic regulariser.** Total mass per channel is
   exactly constant, so the system **cannot globally explode or die**. Bad
   parameters degrade a region, not the world. This is the strongest robustness
   argument in the entire CA family.

Performance: **255 µs ± 3.11 µs per step on a Tesla T4** at research resolution;
128² and 1024² both demonstrated; the community treats 256² as the working size.
dt = 0.2.

Cost: **no WebGPU implementation exists** — JAX only. Build from paper.
Reintegration tracking is tractable via Moroz's writeup
(https://michaelmoroz.github.io/Reintegration-Tracking/): keep neighbour radius
R=1 by capping velocity and substepping, use **box distributions** so cell
overlap is an analytic rectangle intersection with no inner loop and no
discontinuities.

Risk: published outputs are scientific, not art-directed. "Might not look good
enough" is a real possibility.

### Particle Lenia
Mordvintsev, Niklasson, Randazzo. Energy-based reformulation: particles create a
Lenia field, a growth field, and a soft-core repulsion field; each particle
gradient-descends its *own* energy. Six parameters.

Mild parameter sensitivity by CA standards — bad parameters give boring or
dispersed states rather than NaN. Strongly multistable (different seeds reach
different final states), which means hysteresis, which is a natural fit for
"returns to a related state when inputs return".

But the reference notebook runs **200 particles**, upscaled to 800. That is not
a terrarium. Scaling needs spatial hashing, and the visual character (crisp
discrete dots) reads more geometric than the brief wants.

### Neural CA — disqualified
The rule *is* learned weights. There is no safe continuous parameter axis to
modulate; perturbing weights goes out of distribution and the pattern collapses.
Making this work would require training a conditional NCA where the conditioning
vector is the latent — a whole ML project. Note as a v2 seam, not a v1 option.

## Agent and particle systems

### Physarum / Jones model — and Sage Jenson's "36 Points"
Two coupled layers: an agent list (position, heading) and a trail map. Per tick:
sense three points ahead and at ±sensor angle → rotate toward the strongest →
step → deposit → 3×3 diffuse → multiplicative decay.

**Jenson's key extension** (reverse-engineered in detail at
https://bleuje.com/physarum-explanation/): make the four core parameters
*functions of locally sensed trail intensity x*:

```
sensor distance = p1 + p2·x^p3
sensor angle    = p4 + p5·x^p6
rotation angle  = p7 + p8·x^p9
move distance   = p10 + p11·x^p12
```

plus trail-sampling offsets → ~15–20 parameters. "36 Points" = 36 named presets
(26 letters + 10 digits), each ~20 numbers, **each producing a qualitatively
distinct organism from identical code**. Agents behave differently in dense
versus sparse regions — second-order emergence, not a lookup.

**Robustness is the standout property. Physarum essentially cannot die and
cannot explode.** The trail map is bounded by deposit/decay equilibrium; agents
always move. The literature confirms it: "the simplest models are sufficient to
reproduce the networking behavior … with little sensitivity and high
robustness"; convergence happens "regardless of the initial structure of the
network or of the initial mass distribution". Sensitivity analyses identify
sensor angle, sensor distance, and movement parameters as dominant, well-behaved
shape determinants. **You can sweep these live, arbitrarily, and always get
structure.** For a solo developer this is worth more than any other property
here.

**Two distinct memory reservoirs:**
- The trail map is an exponential integrator — literally an EMA of history, with
  tunable duration via decay rate (τ ≈ 1/(1−decay); decay 0.97 ≈ 33-frame
  memory, 0.995 ≈ 200 frames).
- The network topology is a slow, hysteretic structure that persists long after
  the agents that built it. Reconfiguration under parameter shift is *gradual
  and visible* — old channels fade while new ones grow. Returning to a prior
  parameter set on top of an existing network **reinforces surviving structure
  rather than rebuilding**, which is precisely the requested recurrence
  behaviour.

Performance, documented: Jenson gets **5–10M particles** real-time on a GTX
1070; bleuje reports **5.8M at 60 fps** on mid-range hardware and **13.1M** on
newer cards at 1920×1088; Klein's WebGL2 browser version does 1M+.

Existing WebGPU code to start from:
- https://github.com/SuboptimalEng/slime-sim-webgpu (TypeScript + WGSL + Vite)
- https://github.com/tom-strowger/physarum-rust (Rust/wgpu, runs in Chrome)
- https://apps.amandaghassaei.com/gpu-io/examples/physarum/ (WebGL2 reference)

Rigorous formulation: Monte Carlo Physarum Machine, Elek et al., *Artificial
Life* 28(1):22 — also documents that "more interconnected polyphorms are much
more persistent in time".

Visual character: organic, filamentary, vascular, dendritic; at high particle
counts individual points vanish leaving smoke/dune/thunderhead aggregate
densities. Extremely photogenic without expensive rendering.

**The one real criticism: ubiquity.** Physarum is the most-cloned GPU sim of the
last five years. Novelty must come from the latent driving and the rendering,
not the substrate.

### Boids, curl noise, particle life, fluids — why each falls short
- **Boids**: essentially no memory; a pure function of current positions. Fails
  the brief unless paired with a deposited field, at which point you have
  reinvented physarum. (Spatial-hash implementations do reach 67M boids at
  30 fps — useful as a *layer*.)
- **Curl noise**: zero state, analytic function of (x, t). **This is exactly the
  class of thing the brief warns against** — it would work identically driven by
  an FFT magnitude. Garnish only.
- **Particle Life** (lisyarus, 260k particles in browser WebGPU): the K×K signed
  interaction matrix is a genuinely elegant latent-modulation target — any matrix
  produces *some* behaviour, none produce NaN. But no substrate field, and the
  visual reads toy rather than terrarium. Its GPU spatial-binning recipe
  (count → prefix sum → scatter) is worth copying regardless.
  https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html
- **SPH / MLS-MPM fluids** (WebGPU-Ocean: ~100k particles on integrated
  graphics, ~300k on discrete): **fluids forget.** Momentum decays, the field
  homogenises, no long-term identity. Poor as a primary substrate, excellent as
  a transport layer. Steal one trick regardless — P2G scatter uses `atomicAdd`
  on **i32 fixed-point** because WGSL has no float atomics, and integer addition
  is associative, making it **bit-exactly order-independent and therefore
  deterministic**.

## Reaction–diffusion

### Gray-Scott — the best-mapped parameter space in the survey
Two parameters only: F (feed), k (kill). Munafo's *xmorphia*
(http://www.mrob.com/pub/comp/xmorphia/pearson-classes.html) extends Pearson's
14 classes to ~18 named regimes: chaotic wavelets, Turing hexagons, mitosing
spots, negatons, solitons, non-branching worms, BZ-like spirals, and class-4
"diverse localized structures".

Critically, **the interesting region is a connected 2D band, not a fractal
shell** — the exact opposite of Lenia's situation. Path-planning inside it is
trivial. And the two axes are roughly monotone in meaning: F traverses
no-reaction → chaotic oscillation → regular periodic → steady → stable; k
traverses all-blue → negatons → stripes+spots → solid → all-red. Ideal for
binding two smooth latent dims.

**Weakness: memory.** Gray-Scott relaxes to attractors and largely forgets
initial conditions. The literature is clear that **hysteresis, not bistability,
is what gives RD systems memory** ("bistability alone, without hysteresis, does
not result in stable patterns" — arXiv 1311.1737; DCDS 2020). The fix is cheap
and worth remembering for any substrate: **add a slow third field with
bistability that integrates where structures have been and biases F/k locally.**

Highest WebGPU feasibility of anything surveyed. Best reference: Codrops,
*Reaction-Diffusion Compute Shader in WebGPU*
(https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/)
— texture ping-pong, workgroup size 64, 2×2 tile per thread, workgroup-shared
prefetch cache, simulation at a fraction of canvas resolution.

### Multi-scale Turing (McCabe) — the beauty-per-effort winner
Convolve with activator and inhibitor kernels at N radii; at each pixel pick the
scale with **minimum variation** and step the field by ±Δ for that scale;
renormalise. Choosing *minimum* variation rather than maximum response is what
keeps it stable — it prevents runaway local contrast.

Best implementation in existence: Ricky Reusser's WebGPU version,
https://rreusser.github.io/notebooks/multiscale-turing-patterns/ — 256² grid,
**FFT convolution with packed FFTs** (four real transforms per pass by storing
the field as `vec4` treated as two complex numbers), kernels evaluated
**analytically in Fourier space** (Gaussian's FT is a Gaussian, disc's is a
Bessel J₁ — zero kernel memory), and **f16 storage buffers**. Softology's
alternative optimisation approximates multi-radius blurs with **mipmaps**:
2048² went from ~0.66 s/iter to **~0.025 s/iter (40 fps)** — the pragmatic route
if you don't want to write an FFT.

Per-scale parameters: activator radius, inhibitor/activator ratio, kernel shape,
step amount, weight, and **symmetry (1–8 fold rotational order)**. That's
5–6 knobs × N scales = 25–40 parameters, **essentially all of them safe** — one
of the best expressiveness-to-risk ratios available. The integer symmetry knob is
a perfect discrete target for section-boundary events.

Visually the most painterly of everything surveyed — fingerprint-and-nebula
marbling. Memory is moderate: structure persists and reorganises gradually, but
it doesn't accumulate history the way a trail map does.

## Comparison

| System | Visual | Memory | Param robustness | WebGPU cost | Existing code |
|---|---|---|---|---|---|
| Lenia | organic creatures | strong but brittle | **very poor** (fractal shell) | med | none mature |
| Flow-Lenia | flowing protoplasm | **excellent** (mass + advected params) | **good** (conservation) | med-high | none (JAX) |
| Particle Lenia | crisp cells | good, multistable | good | low–med | tiny |
| Neural CA | learned/specific | strong | n/a (weights) | low | WebGL2 |
| **Physarum / 36 Points** | vascular, dune-like | **excellent** (trail + network) | **excellent** (can't die) | **low** (5–13M @60) | several |
| Boids | flocks | none | excellent | low | official sample |
| Curl noise | wispy | **none** | trivial | very low | many |
| Particle Life | colored clusters | moderate | excellent | low (260k) | lisyarus |
| MLS-MPM fluid | fluid | poor (forgets) | good | med | WebGPU-Ocean |
| Gray-Scott | spots/worms/spirals | weak without extra field | **good, fully mapped** | **very low** | Codrops |
| Multi-scale Turing | marbled, painterly | moderate | **very good** | low | Reusser |

## The mapping problem: latent → parameters without chaos

The thinnest and most valuable part of the literature.

**Preset simplex with weighted blending — the strongest practical prior art.**
Jenson's 36 Points does exactly this: hand-tuned parameter sets with
**Gaussian/RBF-weighted blending by distance**. Generalise: hold M presets
θ₁..θ_M each known good, place anchors c_m in latent space, map z → weights
`w = softmax(−‖z − c_m‖² / T)`, take `θ = Σ w_m θ_m`. This **guarantees you stay
inside the convex hull of known-good parameters** — the single most important
safety property. It works because physarum/Turing/Gray-Scott good regions are
locally convex, and fails for Lenia for exactly the reason described above.

**Interpolate spherically, not linearly.** Tom White, *Sampling Generative
Networks* (arXiv 1609.04468): linear interpolation in high-dim space "traverses
locations that are extremely unlikely given the prior" — use **slerp**. Also
documents latent dead zones. The synth-preset analogue with a fully worked
pipeline is SPINVAE (https://gwendal-lv.github.io/spinvae2/).

**Discover the safe set offline, once per system — not per track.**
- **IMGEP / curiosity-driven diversity search** (Reinke, Etcheverry, Oudeyer
  2020) — samples *goals in a learned behaviour space* and inverts to
  parameters, precisely because the parameter→behaviour map is wildly
  non-linear. http://developmentalsystems.org/intrinsically_motivated_discovery_of_diverse_patterns
- **Phase Transition Finder** (arXiv 2402.17848) — bisect between a dead phase
  and a live phase to land on the interesting boundary.
- **ParamExplorer** (arXiv 2512.16529, Dec 2025) — generic framework for
  exploring generative-art parameter spaces via Optuna/Hyperband, with explicit
  methods for smooth transitions "while maintaining visual coherence and
  avoiding abrupt, jarring changes". Closest citable how-to for this problem.

**Parameters as a spatial field carried by the simulation** (Flow-Lenia's
contribution, §above). Applies beyond Flow-Lenia: physarum agents can carry
per-agent parameter vectors inherited and blended from a trail-aligned parameter
field. Effects: a bad injection is local rather than global; old rules persist as
long as their matter does; re-injecting a prior latent reinforces surviving
structure that already carries similar rules. **This is the best published answer
to "recurrence must produce recognisably related visual states".**

**Rate limiting and timescale separation.** Not formalised anywhere but implied
throughout. Each simulation has a relaxation time τ_sim; parameter motion must be
slow relative to it or the system never settles into anything recognisable. A
10 Hz latent timeline should be **low-pass filtered / slew-limited before
touching parameters, with different cutoffs per parameter**: fast parameters
(deposit strength, sensor angle, colour) can track 10 Hz; slow ones (decay rate,
kernel radius, feed rate) should track section scale, 0.01–0.1 Hz. Discrete
boundary events are the right trigger for the slow ones — a step, not a ramp.

**Hysteresis is the memory mechanism.** Confirmed across the RD literature:
bistability alone gives no memory, hysteresis does. Practically, recurrence
requires at least one slow bistable or integrating field whose state depends on
history — a trail map, a mass field, a "soil/scar" layer. Without that, no amount
of clever parameter mapping produces recurrence.

**Deterministic re-seeding — a cheap trick worth its weight.** Key the PRNG on
`(global_seed, section_id, agent_index)` rather than frame count for any
*structural* randomness (respawn positions, injection sites, species assignment).
Then the second chorus injects matter at bit-identically the same places as the
first, and the entire difference in outcome is the accumulated state — exactly
the "same idea, older world" effect.

## Practical WebGPU notes

**Browser status (2026).** WebGPU is Baseline across all major browsers: Chrome
113+, Firefox 141 (Windows, Jul 2025) / 145 (macOS), Safari 26.0 (Sept 2025).
Coverage ~70% (Jan 2026) rising toward ~82%. Part of Interop 2026. Shipping
WebGPU-only is defensible; still needs a graceful unsupported-browser page.

**Storage textures vs buffers.** Measured performance is **essentially
identical** — the data structure "mainly impacts how data is stored, not how it
is computed". Choose by shape: buffers for one scalar per cell or odd struct
layouts; textures when you need filtered sampling (bilinear diffusion, gradient
reads, mip pyramids). Physarum wants **textures** — the trail map is sampled at
fractional sensor positions, so bilinear filtering is free. Read-write storage
textures now exist (gated on `readonly_and_readwrite_storage_textures`, formats
limited to r32float/sint/uint unless `texture-formats-tier2`), but that path is
narrow: **plan ping-pong as the portable default**.

**Sizing.** Guaranteed limits: `maxComputeWorkgroupSizeX/Y = 256`,
`maxComputeInvocationsPerWorkgroup = 256`, `maxComputeWorkgroupStorageSize =
16384 B`, `maxStorageBufferBindingSize = 128 MiB` (real devices expose 128 MB
mobile to 4 GB desktop — always read `adapter.limits`). Use **64 threads per
workgroup** (8×8 for 2D grids); most GPUs run 64 in lockstep. Process **multiple
pixels per thread** to amortise dispatch and index math. Use
**workgroup-shared prefetch** (tile + halo) for any convolution.

**Dispatch budget.** Measured per-dispatch overhead is **0.08–0.35 ms** across
vendors/backends/browsers, with CPU-side submission ~40% of it (arXiv
2604.02344). At 16.6 ms/frame that caps you at ~40–80 dispatches from overhead
alone — budget **≤ 15–20 compute passes per frame** and batch aggressively.
Profile with `timestamp-query`, noting it is **pass-level granularity only** and
returns the sum since last resolve.

**Determinism — the honest picture.** **Cross-GPU bit-exactness is not
achievable and should not be promised.** Divergence sources: f32
non-associativity in parallel reductions; implementation-defined FMA
contraction; and — most relevant here — **transcendentals (`exp`, `pow`, `sin`,
`inverseSqrt`) have implementation-defined ULP bounds in WGSL and differ across
vendors**, and every CA in this survey is built on `exp`. WebGPU benchmarking
practice validates to max absolute difference < 1e-4, not bit-equality.

Float atomics are the worst offender and WGSL lacks them anyway; the
**i32 fixed-point atomicAdd** workaround (scale ~1e7) is simultaneously the
determinism fix, since integer addition is exactly associative and therefore
order-independent. Use it for physarum trail deposit.

**Same-device determinism is achievable, and is what to promise** ("deterministic
on your machine; visually equivalent everywhere"). Requires: fixed timestep, no
wall-clock reads in shaders, no float atomics, fixed dispatch order, and no
frame-count-derived randomness for structural decisions.

**Fixed timestep is mandatory regardless.** Accumulate real elapsed time, run N
integer sim steps of fixed dt, and **index the latent timeline by sim-tick, not
wall clock**. The run becomes a pure function of (track, seed, tick count), and a
slow frame costs visual smoothness rather than simulation state. Precedents:
Flow-Lenia's dt = 0.2, WebGPU-Ocean's 2 sim steps per frame.

**PRNG.** Jarzynski & Olano, *Hash Functions for GPU Rendering* (JCGT 2020) —
`pcg2d`/`pcg3d`/`pcg4d`, single-u32 state, ideal for shaders. Seed per invocation
by hashing **(global_seed, sim_tick, agent_index)** — sim_tick, never frame
index, or determinism breaks the moment frame rate varies.

## Notable prior art, and a gap

- **Sage Jenson (mxsage)** — C++/GLSL/openFrameworks, all state GPU-resident,
  custom renderer. Technical writing is thin; bleuje's reverse-engineering is the
  actually-useful source. https://sagejenson.com
- **Refik Anadol** — worth being precise about: *Unsupervised* is StyleGAN2-ADA
  with a bespoke latent-space browser, and the visual is produced by
  **interpolating between latent vectors**. There is no emergent simulation
  underneath. It is the aesthetic reference, not the architectural one.
- **compute.toys** (https://compute.toys) — Shadertoy for compute shaders; best
  place to prototype a WGSL sim kernel and read others' persistent-state tricks.
- **SwissGL** (https://github.com/google/swissgl) — Mordvintsev's <1000-LOC
  WebGL2 wrapper; the ALIFE 2024 tutorial walks Particle Lenia from scratch.
- **three.js WebGPURenderer + TSL** — realistic option for a solo dev who wants
  rendering solved, at the cost of control over the compute pipeline.

The only audio-reactive physarum found (github.com/bu3nAmigue/physarum-audio-
reactive) maps audio amplitude to sensor offset and step size — confirming those
are the natural knobs, and confirming that **nobody has done the structural or
latent-driven version.**
