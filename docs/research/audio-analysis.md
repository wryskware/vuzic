# Research — Audio Analysis SotA (surveyed 2026-08)

Findings from a survey of music-understanding models and structure-analysis
tooling, current through early 2026. This document is evidence, not decisions;
decisions live in `docs/plan.md`.

## The headline finding

The two halves of the problem want **different feature substrates**, and this is
empirically supported rather than a matter of taste:

- **"What does this moment feel like"** is best served by a self-supervised
  music foundation model (MuQ / MERT / MusicFM).
- **"When does this moment come back"** is best served by classical
  self-similarity over chroma — and modern embeddings are measurably *worse* at
  it.

That second claim is the surprise. *Unsupervised Evaluation of Deep Audio
Embeddings for Music Structure Analysis* (SMC 2026, arXiv 2603.27218) tested
nine pretrained models by building self-similarity matrices from each and
segmenting them. MERT and MuQ were **not** top performers; roughly a third of
deep embeddings failed to beat traditional acoustic features, and on SALAMI
plain barwise log-mel beat everything. Music-only-pretrained models
"consistently underperform" for SSM-based structure.

The hybrid split the scaffolding notes leaned toward is therefore the
evidence-backed choice, not just the pragmatic one.

## Music foundation models (the "character" stream)

### MuQ — recommended primary
Tencent AI Lab, Jan 2025, arXiv 2501.01108 (TASLP 2025).
- Mel-RVQ self-supervised targets; 12 Conformer layers, **1024-dim, 25 Hz**, 310M params.
- SotA among open music SSL models on MARBLE-style probing — beats MERT and MusicFM on nearly every task.
- `pip install muq`; `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")`, `output_hidden_states=True`.
- Requires 24 kHz input; **fp32 recommended** (NaN risk in fp16).
- Code MIT, **weights CC-BY-NC-4.0**.
- https://github.com/tencent-ailab/muq · https://huggingface.co/OpenMuQ

**MuQ-MuLan** (~700M) is a contrastive music-text model built on the same
backbone — a music-specialised CLAP equivalent, SotA zero-shot tagging on
MagnaTagATune. Gives text-anchor projection in the same family.

### Layer selection matters more than model selection
*Layer-wise Investigation of Large-Scale Self-Supervised Music Representation
Models* (arXiv 2505.16306) probed MusicFM and MuQ per layer on MARBLE. Best
layer by task, for the 12-layer models:

| Task | MusicFM | MuQ |
|---|---|---|
| Pitch | 1 | 2 |
| Singer identity | 1 | 1 |
| Instrument / timbre | 3 | 3 |
| Key / harmony | 3 | 5 |
| Genre | 5 | 6 |
| Emotion (EmoMusic V/A) | 5 | 6 — R² ≈ 0.77 / 0.63 |
| Vocal technique | 6 | 7 |
| Frame-level structure (Harmonix) | 6 | **9–12** |

Representations evolve acoustic → semantic monotonically. The best layer is
almost never the last. Single-layer selection usually beats learned weighted
sums. **Practical consequence: extract two layers — ~6 for character, ~9–12 for
structural identity.**

### Alternatives
- **MERT-v1-95M / 330M** (m-a-p, 2023) — 75 Hz, 768/1024-dim, native
  `transformers` support, most battle-tested literature. CC-BY-NC-4.0. Best
  choice if minimising dependencies matters more than ~5% probe quality; 95M
  runs fine on CPU.
- **MusicFM** (arXiv 2311.03318) — 25 Hz, 1024-dim, **MIT licensed**, which is
  its trump card. Slightly below MuQ. Manual checkpoint download, repo code not
  on PyPI. https://github.com/minzwon/musicfm
- **SoniDo** (Sony, NeurIPS 2024) — not openly released. Skip.
- **HeartMuLa / HeartCodec** (Jan 2026) — 12.5 Hz codec explicitly designed to
  capture long-range structure. Very new, generation-focused, unproven for
  analysis probing. Watch, don't bet on.

### Not recommended as the semantic stream
Neural codec latents — **EnCodec** (75 Hz, 128-dim), **DAC** (~86 Hz, MIT),
**Stable Audio Open VAE** (64-dim @ 21.5 Hz), **Music2Latent** (Sony CSL, ISMIR
2024 — coincidentally ~10 Hz × 64-dim, exactly our target rate). All are
reconstruction-oriented: smooth and continuous, but probing literature
consistently finds them **entangled** (content/timbre/pitch mixed) and
underperforming SSL models on semantic MIR. A repeated motif lands nearby only
if it is *acoustically* near, which is not what "recognisably related" means
here. Viable as an optional low-level texture channel, not as the backbone.

**MusicGen internals** do robustly encode tempo/key/chords (arXiv 2410.00872)
but extraction requires codec tokens plus a 24–48-layer LM forward pass. Not
worth it versus MuQ.

## Text-anchor semantic axes (the interpretable overlay)

Projecting audio embeddings against text prompt *pairs* — `sim("tense") −
sim("calm")`, `sim("sparse") − sim("dense")` — yields interpretable,
art-directable axes over time.

- **LAION-CLAP**, music checkpoint `music_audioset_epoch_15_esc_90.14.pt`.
  512-dim joint space, 10 s windows at 48 kHz. `pip install laion_clap`, ~150M
  params, permissive. https://github.com/LAION-AI/CLAP
- Clip-level, not frame-level — but a **sliding 10 s window at 1–2 s hop** gives
  a 0.5–1 Hz semantic trajectory that is inherently smooth from window overlap.
  Complements, does not replace, the 10 Hz SSL stream.
- **Use anchor pairs and z-score per track, never raw cosines.** EmotionRankCLAP
  (arXiv 2505.23732) shows raw CLAP similarity is not calibrated as a scale;
  COMET (arXiv 2605.29628) documents a systematic audio-text modality gap.
- Prior art that the mechanic works: **Text2FX** (arXiv 2409.18847) found that
  moving through CLAP space along text-defined directions is perceptually
  consistent and traces smooth trajectories. **MMVA** (arXiv 2501.01094) does
  continuous valence-arousal in a contrastive space. Stability's
  `stable-audio-metrics` industrialises prompt-cosine scoring.
- Nobody has shipped this as a *visualisation timeline*. Novel assembly of
  well-attested pieces.

## Structure, recurrence, and rhythm

### Beat This! — recommended master clock
Foscarin, Schlüter, Widmer, ISMIR 2024. `pip install beat-this`.
- Beat + downbeat timestamps, 50 fps internally, tempo derivable.
- GTZAN (unseen): **beat F1 0.890, downbeat F1 0.772** — clearly above madmom's
  TCN/DBN and BeatNet (0.754 / 0.467).
- ~78 MB model, fine on CPU, fast on GPU. **MIT**, actively maintained.
- https://github.com/CPJKU/beat_this

### All-In-One — recommended structure extractor
Taejun Kim & Juhan Nam, WASPAA 2023. `pip install allin1`.
- One call yields tempo, beats, downbeats, beat-position-in-bar, segment
  boundaries **and** functional labels (intro/verse/chorus/bridge/inst/solo/
  outro/break), plus **frame-level activations at 100 fps** and per-stem
  embeddings. Those activation curves are graded "chorus-ness / boundary-ness"
  signals — exactly the continuous confidence a simulation wants.
- Demucs stem separation + dilated neighborhood attention (NATTEN).
- ~27 s per hour of audio on an RTX 4090. **MIT**.
- **Maintenance: effectively frozen** (last real activity ~2023). Known pain:
  NATTEN must be built from source on Windows; PyTorch 2.x incompatibilities;
  ageing madmom dependency. Community fork `all-in-one-fix` (PyPI v2.0.4) fixes
  PyTorch 2.x and uses maintained `demucs-infer`.
- Quirk: MP3 decoder offsets of 20–40 ms — **convert to WAV first**.
- https://github.com/mir-aidj/all-in-one

### SongFormer — optional label-quality upgrade
ASLP-lab, Oct 2025, arXiv 2510.02797.
- Boundaries + 8-class functional labels. **No beats/downbeats** — structure only.
- Fuses MuQ + MusicFM features at 30 s and 420 s windows.
- Functional label accuracy **0.807 vs All-In-One's 0.740** on
  SongFormBench-Harmonix. Beats Gemini 2.5 Pro on the same benchmark.
- Whole song in 2–4 s on an L40. **CC-BY-4.0**, actively maintained.
- https://github.com/ASLP-lab/SongFormer

### The recurrence layer — custom, ~200 lines
No supervised model gives graded "chorus A ≈ chorus A′ with strength s". Build
it on `librosa` + `libfmp` (MIT, `pip install libfmp` — reference
implementations of the whole FMP Chapter 4 structure toolbox):

1. **Beat-synchronous chroma** → affinity recurrence matrix
   (`librosa.segment.recurrence_matrix(mode='affinity')` +
   `librosa.segment.path_enhance` for multi-tempo diagonal smoothing).
2. **Foote novelty** (Gaussian checkerboard kernel down the SSM diagonal) at
   *multiple kernel widths* — 4 / 8 / 16 bars — giving a multi-scale continuous
   transition-energy curve. Peak prominence via
   `scipy.signal.find_peaks(prominence=...)` is a natural confidence value.
3. **Time-lag stripe analysis** (Goto's RefraiD lineage) → per-beat pointer
   "you are now re-living time t₀, strength s". This is *the* mechanism for
   "returning chorus brings back a recognisable world".
4. **Laplacian spectral clustering** (McFee & Ellis 2014) — eigenvectors of the
   recurrence-graph Laplacian give a multi-level segmentation *and* the
   eigenvector coordinates are themselves a smooth, graded "structural position"
   vector per beat. Worked example:
   https://librosa.org/doc/main/auto_examples/plot_segmentation.html
5. **2D Fourier Magnitude Coefficients** (Nieto & Bello, ICASSP 2014) —
   key-transposition- and phase-shift-invariant per-segment fingerprints.
   Pairwise cosine distances *are* the graded similarity; no thresholding into
   hard labels needed.

Also available in libfmp: **scape-plot audio thumbnailing**, a fitness measure
that finds "the most repeated segment" (≈ the chorus) without any labels.

### Harmonic colour, cheaply
Beat-synchronous CQT chroma + per-segment Krumhansl/Temperley key correlation +
**HCDF** (harmonic change detection function). All in librosa/libfmp, zero new
model dependencies. Probably sufficient for a "harmonic colour" axis.

If actual chord symbols prove useful: **crema** (McFee, 602-class, per-frame
root/bass posteriors — great continuous signal, but ageing TF/keras dependency)
or **Essentia** `KeyExtractor` (actively maintained, **AGPL-3.0** — fine for
non-distributed offline tooling, note the license).

### Avoid
- **MSAF** — dormant, dependency rot with modern librosa/numpy. Mine it for
  algorithms; don't depend on it.
- **madmom** — PyPI still at 0.16.1 (Nov 2018), broken on Python ≥3.10 without
  patches, and **models are CC BY-NC-SA 4.0**. Semi-abandoned.
- **BeatNet** — real-time focus we don't need, weaker offline accuracy, madmom +
  PyAudio install friction.
- **pychorus / DeepChorus** — subsumed by the SSM work above.

### Emotion / mood trajectories
Dynamic MER is a fragmented area with no dominant pretrained "V-A over time"
checkpoint. DEAM/MediaEval (1,802 tracks, **continuous V-A at 2 Hz**) is the
natural training target. Recent models report DEAM RMSE ≈ 0.035–0.044 but rarely
ship weights.

Don't adopt a bespoke emotion model. Either train a tiny ridge/MLP probe from
MuQ layer-6 frames → DEAM 2 Hz V-A (about a day of work; foundation-model probes
hit the practical ceiling of R² ≈ 0.63–0.77 on EmoMusic), or get V-A-like axes
for free from the CLAP text-anchor pairs. Essentia's MusiCNN-based
arousal/valence regressors are the lowest-effort coarse baseline.

## Dimensionality reduction

**PCA is the right default**, and this is not a close call for this project:
linear, deterministic, distance-preserving (so repeated motifs stay near each
other), inherits the smoothness of its input, and exports as a plain projection
matrix — which matters for the determinism requirement.

Fit per-track after per-dimension standardisation; 8–24 components of a MuQ
layer typically retain most variance.

**Avoid vanilla UMAP/t-SNE**: cluster-forming, trajectory-fragmenting,
seed-sensitive, and they distort global distances — all fatal for a continuous
world-state driver. If PCA proves too flat, the nonlinear options that *preserve
trajectories* are **PHATE** (documented to keep time-continuous trajectories
where UMAP fragments them) and **Parametric UMAP** / **Aligned-UMAP**.

Smoothing: overlapping-window averaging to 10 Hz already low-passes; add EMA or
Savitzky-Golay on the reduced trajectory. Beat-synchronous averaging is the
musically principled alternative.

## Prior art in music visualisation from learned representations

- **Deep Music Visualizer** (Siegelman 2019) — chromagram + volume → BigGAN
  vectors. Its *audio* side is DSP, not learned. Precisely the trap this project
  avoids; useful only as latent-navigation precedent.
- **NEBULA** (SMC 2025, TAMLab Linz) — PCA over RAVE latents for interactive
  exploration, with OSC. https://github.com/tamlablinz/RAVE_PCA
- **Latent Terrain / nn_terrain** (Zheng, QMUL) — dissects neural-audio-
  autoencoder latent spaces into navigable 2D terrains.
  https://jasper-zheng.github.io/nn_terrain

Both validate "PCA of learned audio latents → low-dim navigable space" as
artistic practice, but for *synthesis control*, not structural display.

**Nothing found drives a real-time visual simulation from music-understanding
embeddings.** Everything in this space is either FFT-driven or built on
codec/synthesis latents. The concept appears genuinely unoccupied.
