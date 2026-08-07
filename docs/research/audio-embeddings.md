# Research: Music Embedding Models (the "character" stream)

Surveyed 2026-08. Goal: a time-varying latent representation of one track,
extracted offline in Python, reduced to ~8–24 smooth dims at ~10 Hz, such that
similar musical moments land near each other.

## Self-supervised music models (primary candidates)

### MuQ (Tencent AI Lab, Jan 2025) — top pick
- SSL with Mel-RVQ targets; 12 Conformer layers, 1024-dim, 25 Hz, 310M params.
- SotA among open music SSL models on MARBLE-style probing (beats MERT and
  MusicFM on nearly every task).
- `pip install muq`; `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")` with
  `output_hidden_states=True`. Strict 24 kHz input, fp32 recommended (fp16 NaNs).
- License: code MIT, **weights CC-BY-NC-4.0**. Full track in seconds on a
  consumer GPU, minutes on CPU.
- https://github.com/tencent-ailab/muq · https://huggingface.co/OpenMuQ

### MERT (m-a-p, 2023)
- HuBERT-style; v1-95M (12×768) and v1-330M (24×1024), 75 Hz, 24 kHz input.
- Native HF `transformers` support; the most battle-tested downstream
  literature. License CC-BY-NC-4.0. 95M runs comfortably on CPU.
- https://huggingface.co/m-a-p/MERT-v1-330M

### MusicFM (Won et al., 2023)
- BEST-RQ-style, 330M, 25 Hz, 1024-dim. Slightly below MuQ on probes.
- **MIT license (incl. weights)** — the fallback if CC-BY-NC ever bites.
- Manual checkpoint download; repo code not pip-packaged.
- https://github.com/minzwon/musicfm

### Layer-wise probing (arXiv 2505.16306 — which layers capture what)
For 12-layer MuQ: pitch L2, timbre L3, key L5, genre/emotion L6 (EmoMusic
R² ≈ 0.77/0.63), vocal technique L7, **music structure L9–12**. Representations
evolve acoustic → semantic; best layer is almost never the last; single-layer
beats learned layer-weighting. Implication: extract **layer ~6 for "what a
moment feels like"** and **layers ~9–12 for structural identity**; concatenating
two layers is legitimate.

### Watch list
- MuQ-MuLan (~700M): CLAP-equivalent in the MuQ family, SotA zero-shot tagging.
- HeartMuLa / HeartCodec (Jan 2026, arXiv 2601.10547): 12.5 Hz codec designed
  for long-range structure. Unproven for analysis; watch, don't bet.
- MARBLE benchmark is the scoreboard: https://marble-bm.shef.ac.uk

## CLAP family (semantic text-anchor axes)

- **LAION-CLAP** music checkpoint (`music_audioset_epoch_15_esc_90.14.pt`):
  512-dim joint audio-text space, 10 s windows @ 48 kHz. Clip-level, but a
  sliding window (10 s / 1–2 s hop) yields a smooth 0.5–1 Hz mood trajectory.
  `pip install laion_clap`, ~150M params, permissive. https://github.com/LAION-AI/CLAP
- **Text-anchor trick:** project audio embeddings onto contrastive prompt
  *pairs* — score = sim("tense") − sim("calm"), z-scored per track. Use pairs,
  not raw cosines (modality gap; CLAP similarity is not calibrated as a scale —
  EmotionRankCLAAP arXiv 2505.23732, COMET arXiv 2605.29628).
- Prior art validating the mechanic: Text2FX (arXiv 2409.18847 — motion along
  CLAP text directions is perceptually consistent and smooth), MMVA
  (arXiv 2501.01094), Stability's clap_score metric. **Nobody has shipped
  anchor-prompt mood timelines for visualization** — novel-but-safe assembly.
- Microsoft CLAP 2023 (`msclap`, MIT) exists but LAION's music checkpoint and
  MuQ-MuLan beat it on music.

## Codec / generative latents — rejected as the semantic stream
- EnCodec (75 Hz), DAC (~86 Hz): reconstruction-oriented, entangle
  content/timbre/pitch, underperform SSL on semantic MIR. Baselines to beat.
- **Music2Latent** (Sony CSL, ISMIR 2024): 44.1 kHz → ~10 Hz × 64-dim continuous
  latents (`pip install music2latent`, CC-BY-NC). Smooth but semantically
  shallow — optional *auxiliary texture channel*, not the backbone.
- MusicGen internals encode theory (arXiv 2410.00872) but extraction is awkward.
- Jukebox/CALM: historical proof, superseded at ~1/20 the compute.

## Emotion trajectories
- Dynamic valence-arousal is a fragmented area; no dominant pretrained
  checkpoint. DEAM has continuous 2 Hz V-A annotations (1,802 tracks).
- Practical routes: (a) tiny ridge/MLP probe from MuQ layer-6 frames → DEAM
  V-A (~a day of work, matches published R² ≈ 0.63–0.77 ceiling); (b) CLAP
  text-anchor pairs give V-A-like axes for free. Essentia's MusiCNN V-A
  regressors are the lowest-effort baseline.
- Modern recipe reference: arXiv 2502.03979 (unified MER over MERT features).

## Dimensionality reduction → smooth low-dim trajectory
- **PCA is the right default:** linear, deterministic, distance-preserving (so
  motif recurrence survives), exportable as a projection matrix (fits the
  determinism requirement). Per-track fit after per-dim standardization;
  8–24 PCs retain most variance.
- Avoid vanilla UMAP/t-SNE (fragment trajectories, seed-sensitive, distort
  global distances). If PCA proves too flat: PHATE (preserves smooth temporal
  trajectories) or Parametric UMAP.
- Temporal smoothing: 25 Hz → 10 Hz pooling already low-passes; add EMA or
  Savitzky-Golay; beat-synchronous averaging is the musically principled option.
- Prior art for "PCA of learned audio latents as navigable space": NEBULA
  (SMC 2025, github.com/tamlablinz/RAVE_PCA), Latent Terrain
  (jasper-zheng.github.io/nn_terrain).

## Prior art gap
No project found that drives a real-time visual simulation from
music-understanding embeddings (MERT/MuQ/CLAP-class) rather than codec latents
or FFT. The terrarium concept appears genuinely underexplored.

## Recommendation
1. **MuQ backbone**: layer ~6 (character) + layer ~9–12 (structure feed),
   25 Hz → 10 Hz pooling, standardize, PCA → 8–24 dims, light smoothing.
   Fallback with identical pipeline shape: MusicFM (MIT).
2. **LAION-CLAP text-anchor axes** as an interpretable, art-directable overlay
   (sliding window, anchor pairs, z-scored per track).
3. Optional: DEAM-probed V-A channel; Music2Latent texture channel.
