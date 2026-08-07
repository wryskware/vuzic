"""Stage 4 — the character stream (MuQ layer ~6).

plan.md Decision 1: "MuQ layer selection is not a detail." Published per-layer
probing (arXiv 2505.16306) puts genre/emotion at layer 6 for the 12-layer MuQ;
the last layer is almost never the right one. 25 Hz -> 10 Hz pooling,
per-dimension standardisation, PCA to 64 dims with a fixed sign convention, then
light zero-phase smoothing.

PCA here is a compact default input and a debug visualisation, not the contract
(Decision 2, commitment 1) — `--embedding` dumps the raw pooled 1024-dim layer-6
features alongside, for a future learned mapping.
"""

from __future__ import annotations

import numpy as np

from ..context import Context, log
from ..dsp import alpha_for_tau, ema_zero_phase, pool_to_grid
from ..optional import require

NAME = "character"

MUQ_SR = 24000
MUQ_RATE = 25.0
CHUNK_SECONDS = 30.0
CHUNK_OVERLAP = 1.0
SMOOTH_TAU = 0.25


def standardize(x: np.ndarray) -> np.ndarray:
    """Per-dimension zero mean, unit variance. Constant dims collapse to zero."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic PCA. Returns (scores, components, explained_variance_ratio).

    SVD leaves each component's sign arbitrary, so an unconstrained PCA flips
    axes between runs and between tracks and the visual mapping flips with it.
    Convention: the loading of largest magnitude in each component is positive
    (ties broken by lowest index, which is what argmax gives).
    """
    x = np.asarray(x, dtype=np.float64)
    n_samples, n_features = x.shape
    k = int(min(n_components, n_features, max(1, n_samples)))
    centred = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:k]
    flip = components[np.arange(k), np.argmax(np.abs(components), axis=1)] < 0
    components = np.where(flip[:, None], -components, components)
    scores = centred @ components.T
    total = float(np.sum(s**2))
    ratio = (s[:k] ** 2) / total if total > 0 else np.zeros(k)
    return scores, components, ratio


def to_latent(features: np.ndarray, n_dims: int, hop_seconds: float) -> np.ndarray:
    """Pooled embeddings -> standardized -> PCA(n_dims) -> smoothed, (T, n_dims)."""
    scores, _, _ = pca(standardize(features), n_dims)
    scale = float(np.sqrt(np.mean(scores[:, 0] ** 2))) if scores.shape[1] else 0.0
    if scale > 1e-8:
        # One global scale, not per-component whitening: whitening would amplify
        # the near-noise tail components to the same amplitude as PC1.
        scores = scores / scale
    scores = ema_zero_phase(scores, alpha_for_tau(SMOOTH_TAU, hop_seconds))
    if scores.shape[1] < n_dims:
        scores = np.pad(scores, ((0, 0), (0, n_dims - scores.shape[1])))
    return scores


def _extract_muq(ctx: Context) -> tuple[np.ndarray, float]:
    """Returns (features, seconds_of_audio_covered)."""
    MuQ = require("muq", "MuQ")  # before torch: muq's advice is the more useful message
    torch = require("torch")
    import librosa

    device = ctx.cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("character: MuQ %s layer %d on %s", ctx.cfg.muq_model, ctx.cfg.muq_layer, device)

    y = librosa.load(str(ctx.wav_path), sr=MUQ_SR, mono=True)[0]
    model = MuQ.from_pretrained(ctx.cfg.muq_model)
    model = model.to(device).eval()  # fp32 only: fp16 NaNs (docs/research)

    chunk = int(CHUNK_SECONDS * MUQ_SR)
    overlap = int(CHUNK_OVERLAP * MUQ_SR)
    drop = int(round(CHUNK_OVERLAP * MUQ_RATE))
    pieces: list[np.ndarray] = []
    start = 0
    covered = 0
    while start < len(y):
        end = min(len(y), start + chunk)
        seg = y[start:end]
        if seg.size < MUQ_SR // 2:
            break
        with torch.no_grad():
            wav = torch.from_numpy(seg[None, :]).float().to(device)
            out = model(wav, output_hidden_states=True)
            hidden = out.hidden_states[ctx.cfg.muq_layer]
            feats = hidden[0].float().cpu().numpy()
        if pieces:
            feats = feats[drop:]
        pieces.append(feats)
        covered = end
        if end >= len(y):
            break
        start = end - overlap
    if not pieces:
        raise RuntimeError("MuQ produced no frames (track too short?)")
    return np.concatenate(pieces, axis=0), covered / MUQ_SR


def frame_rate(n_frames: int, seconds: float) -> float:
    """Measured feature rate, falling back to the nominal one."""
    if seconds <= 0 or n_frames <= 0:
        return MUQ_RATE
    return n_frames / seconds


def run(ctx: Context) -> None:
    raw, covered = _extract_muq(ctx)
    # Measured, not assumed: MUQ_RATE is nominal, and chunk border effects or a
    # different model would drift the latent stream out of alignment with the
    # beat grid linearly over the track, silently.
    rate = frame_rate(raw.shape[0], covered)
    if abs(rate - MUQ_RATE) > 0.02 * MUQ_RATE:
        ctx.note(
            f"character: MuQ frame rate measured at {rate:.3f} Hz, not the "
            f"nominal {MUQ_RATE:.1f} Hz; pooling with the measured rate"
        )
    pooled = pool_to_grid(raw, rate, ctx.times, ctx.cfg.hop_seconds)
    ctx.embedding = pooled.astype(np.float32)
    ctx.latent = to_latent(pooled, ctx.cfg.latent_dims, ctx.cfg.hop_seconds)
    log.info(
        "character: %d raw frames @ %.2f Hz -> %d pooled x %d dims -> latent %s",
        raw.shape[0],
        rate,
        pooled.shape[0],
        pooled.shape[1],
        ctx.latent.shape,
    )
