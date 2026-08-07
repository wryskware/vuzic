# Research: Music Structure Analysis (the "recurrence" stream)

Surveyed 2026-08. Goal: boundaries, section labels, graded section similarity
("chorus A ≈ chorus A′"), beat grid, and continuous novelty/transition signals —
offline Python, deterministic.

## Supervised segmentation

### All-In-One (`allin1`, Kim & Nam, WASPAA 2023)
- One model → tempo, beats, downbeats, bar position, functional segment
  boundaries + labels (intro/verse/chorus/bridge/...), **plus frame-level
  activation curves at 100 fps** (continuous "chorus-ness / boundary-ness" —
  exactly the graded signals the simulation wants). Per-stem embeddings too.
- ~27 s per hour of audio on RTX 4090. **MIT.**
- **Effectively frozen since ~2023.** Pain points: NATTEN builds from source on
  Windows; PyTorch 2.x issues; madmom dependency aging. Mitigations: community
  fork `all-in-one-fix` (PyPI, PyTorch 2.x + demucs-infer), or run in
  WSL2/Docker. Convert MP3→WAV first (decoder offset ~20–40 ms).
- https://github.com/mir-aidj/all-in-one

### SongFormer (ASLP-lab, Oct 2025) — current SotA boundaries+labels
- Fuses MuQ + MusicFM features at 30 s and 420 s windows. Label accuracy
  **0.807 vs All-In-One's 0.740** on SongFormBench-Harmonix; new SotA HR.5F.
- No beats/downbeats — structure only. Whole song in 2–4 s on an L40.
- CC-BY-4.0, actively maintained. git clone + conda py3.10 setup.
- https://github.com/ASLP-lab/SongFormer

### MSAF — dormant; mine it for algorithms, don't depend on it.

## Classical self-similarity (the unsupervised recurrence machinery)
All cheap, deterministic, and implemented in librosa/libfmp:
- **SSM + Foote novelty** (checkerboard kernel): continuous novelty curve;
  peak height = boundary confidence. Multiple kernel widths (4/8/16 bars) give
  a multi-scale event hierarchy.
- **Time-lag / recurrence stripes**: per-frame pointer "now repeating time t₀,
  strength s". `librosa.segment.recurrence_matrix(mode='affinity')` +
  `librosa.segment.path_enhance`.
- **Laplacian/spectral clustering** (McFee & Ellis 2014): multi-level
  segmentation; the eigenvector coordinates are a smooth per-beat "structural
  position" vector — directly usable as latent axes.
  https://librosa.org/doc/main/auto_examples/plot_segmentation.html
- **libfmp** (`pip install libfmp`, MIT): reference implementations of the FMP
  Ch. 4 toolbox incl. transposition-invariant SSMs, structure features,
  audio thumbnailing (finds "most repeated segment" ≈ chorus, unsupervised).

### Key counterintuitive finding
SMC 2026 study (arXiv 2603.27218): for SSM-based structure analysis, **MERT and
MuQ underperform**; ~⅓ of deep embeddings fail to beat traditional features;
best embedding was MATPAC++; **CBM** (arXiv 2311.18604) is the best downstream
segmenter. Practical takeaway: compute recurrence over **chroma/mel first**,
optionally stacked with an embedding-SSM — do not assume the character-stream
model is the right recurrence substrate. This empirically validates the hybrid
split in scaffolding-notes §1.

## Graded section similarity
- **2D Fourier Magnitude Coefficients** (Nieto & Bello 2014): per-segment chroma
  patch → 2D-FFT magnitude → key- and phase-shift-invariant fingerprint. The
  pairwise cosine distances ARE the graded similarity — no hard thresholding.
- Combine: functional label (chorus identity) + acoustic segment similarity
  (chorus A vs evolved chorus A″) + spectral-clustering eigenvector distance.
- Hand the simulation the full segment×segment similarity matrix plus each new
  segment's nearest-past-segment with score.

## Beat / downbeat
- **Beat This!** (CPJKU, ISMIR 2024): `pip install beat-this`, MIT (code and
  weights), maintained, CPU-friendly. GTZAN beat F1 0.890 / downbeat 0.772 —
  clearly above madmom and BeatNet. Recommended master clock.
  https://github.com/CPJKU/beat_this
- **madmom**: semi-abandoned (PyPI 0.16.1 from 2018, broken on py≥3.10),
  models CC-BY-NC-SA. Avoid as a direct dependency.
- BeatNet: real-time focus, weaker offline. Skip.

## Tension / continuous event signals
- Best graded signals: All-In-One activations (supervised) + Foote novelty at
  multiple scales (unsupervised) + `scipy.signal.find_peaks(prominence=...)`
  for confidence. Keep the continuous curve as "transition energy" — it gives
  the ramp INTO a boundary (pre-chorus buildup → anticipatory visuals).
- TenseMusic (PLOS One 2024) is the only off-the-shelf audio tension tool;
  research-grade, classical-optimized; treat as experimental garnish.
- Pragmatic tension proxy: loudness slope + onset density + novelty + HCDF.

## Harmonic color (cheap axis)
- Zero-new-deps option: beat-synchronous CQT chroma + Krumhansl key correlation
  per segment + HCDF (harmonic change detection) via librosa/libfmp. Likely
  sufficient for v1.
- crema (McFee) if actual chord symbols wanted (best quality/effort, aging TF
  dep). Essentia is maintained but AGPL. Skip chord-extractor/autochord/BTC.

## Recommended stack
1. **All-In-One** (via fix fork or WSL/Docker) — beats/downbeats/sections/
   labels/activations in one shot, MIT.
2. **Beat This!** — run anyway; prefer its beats on disagreement; its grid is
   the master clock.
3. **SongFormer** (optional, GPU) — override All-In-One's labels/boundaries.
4. **Custom recurrence layer (~200 lines, librosa+libfmp)** — chroma SSM +
   path_enhance → multi-scale Foote novelty; time-lag repeat pointers;
   Laplacian eigen-coordinates; 2D-FMC segment similarity matrix. This is the
   part no supervised model provides and the direct mechanism for
   "returning chorus → recognizably returning world."
5. Chroma/key/HCDF harmonic color axis.

## Timeline implication
Three natural strata: (i) sparse events (boundaries w/ confidence + label +
similarity-to-past vector), (ii) beat grid (beat/downbeat/bar phase),
(iii) dense curves at 10–100 fps (novelty, label activations, structural
eigen-coordinates, harmonic color). All deterministic at inference.
