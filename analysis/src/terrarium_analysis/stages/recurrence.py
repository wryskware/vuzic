"""Stage 5 — recurrence (optional; librosa only, no model weights).

Computed over chroma rather than embeddings: the 2026 SSM study
(arXiv 2603.27218) found music-pretrained embeddings measurably *worse* at
self-similarity than traditional features. plan.md demotes this stream to
"emit when cheap, design nothing around it" — nothing downstream depends on it.

Emits novelty4 / novelty16 (multi-scale Foote novelty, 4 and 16 bars),
recurTime / recurStr (time-lag repeat pointer: "you are re-living time t0, with
strength s") and the 2D-FMC segment similarity matrix.

recurTime is in SECONDS (0 when no repeat is found); every other channel in the
file is 0..1.
"""

from __future__ import annotations

import numpy as np

from ..context import Context, log
from ..dsp import alpha_for_tau, ema_zero_phase, foote_novelty, robust_norm

NAME = "recurrence"

MAX_SSM_FRAMES = 6000  # 10 minutes at 10 Hz; the SSM is O(n^2)
MIN_LAG_SECONDS = 4.0
FMC_PATCH = (12, 64)
FMC_QUADRANT = (6, 32)


def _chroma_on_grid(ctx: Context) -> np.ndarray:
    import librosa

    hop = max(1, int(round(ctx.mono_sr * ctx.cfg.hop_seconds)))
    chroma = librosa.feature.chroma_cqt(y=ctx.mono, sr=ctx.mono_sr, hop_length=hop)
    if chroma.shape[1] >= ctx.frames:
        chroma = chroma[:, : ctx.frames]
    else:
        pad = ctx.frames - chroma.shape[1]
        chroma = np.pad(chroma, ((0, 0), (0, pad)), mode="edge")
    return np.asarray(chroma, dtype=np.float64)


def _ssm(chroma: np.ndarray) -> np.ndarray:
    norm = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-9)
    return np.clip(norm.T @ norm, 0.0, 1.0)


def fmc_similarity(chroma: np.ndarray, segments: list[dict], hop_seconds: float) -> np.ndarray:
    """2D Fourier Magnitude Coefficients (Nieto & Bello 2014) segment similarity.

    The 2D-FFT magnitude of a chroma patch is invariant to key transposition and
    to phase within the segment, so the pairwise cosines are a graded similarity
    without any thresholding.
    """
    n = len(segments)
    if n == 0:
        return np.zeros((0, 0))
    feats = np.zeros((n, FMC_QUADRANT[0] * FMC_QUADRANT[1]), dtype=np.float64)
    for i, seg in enumerate(segments):
        a = max(0, int(seg["start"] / hop_seconds))
        b = min(chroma.shape[1], max(a + 2, int(seg["end"] / hop_seconds)))
        patch = chroma[:, a:b]
        if patch.shape[1] < 2:
            continue
        idx = np.linspace(0, patch.shape[1] - 1, FMC_PATCH[1])
        resized = np.stack(
            [np.interp(idx, np.arange(patch.shape[1]), patch[k]) for k in range(patch.shape[0])]
        )
        mag = np.abs(np.fft.fft2(resized))[: FMC_QUADRANT[0], : FMC_QUADRANT[1]]
        # Log compression before the cosine, as in MSAF's ffmc2d: without it the
        # low-order coefficients dominate and every pair comes out near 1.
        feats[i] = np.log1p(mag).ravel()
    norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
    feats = feats / norms
    return np.clip(feats @ feats.T, 0.0, 1.0)


def repeat_pointer(
    affinity: np.ndarray, times: np.ndarray, min_lag_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame nearest repeat: (time of best match in seconds, strength 0..1)."""
    a = np.array(affinity, dtype=np.float64, copy=True)
    n = a.shape[0]
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    a[np.abs(rows - cols) < min_lag_frames] = 0.0
    best = np.argmax(a, axis=1)
    strength = a[np.arange(n), best]
    pointer = np.where(strength > 0, times[best], 0.0)
    return pointer, strength


def run(ctx: Context) -> None:
    import librosa

    chroma = _chroma_on_grid(ctx)
    stride = max(1, int(np.ceil(ctx.frames / MAX_SSM_FRAMES)))
    if stride > 1:
        ctx.note(
            f"recurrence: {ctx.frames} frames exceeds the {MAX_SSM_FRAMES}-frame SSM cap; "
            f"computing at 1/{stride} resolution and interpolating back"
        )
    sub = chroma[:, ::stride]
    sub_times = ctx.times[::stride]

    ssm = _ssm(sub)
    bar_frames = max(2, int(round(ctx.bar_period / (ctx.cfg.hop_seconds * stride))))
    nov4 = robust_norm(foote_novelty(ssm, 4 * bar_frames))
    nov16 = robust_norm(foote_novelty(ssm, 16 * bar_frames))

    stacked = librosa.feature.stack_memory(sub, n_steps=4, delay=max(1, bar_frames // 4))
    rec = librosa.segment.recurrence_matrix(
        stacked, mode="affinity", metric="cosine", sym=True, width=max(3, bar_frames)
    )
    rec = librosa.segment.path_enhance(np.asarray(rec, dtype=np.float64), n=15, window="hann")
    min_lag = max(2, int(round(MIN_LAG_SECONDS / (ctx.cfg.hop_seconds * stride))))
    pointer, strength = repeat_pointer(rec, sub_times, min_lag)
    strength = robust_norm(strength)

    alpha = alpha_for_tau(ctx.bar_period, ctx.cfg.hop_seconds * stride)
    ctx.novelty4 = np.interp(ctx.times, sub_times, nov4)
    ctx.novelty16 = np.interp(ctx.times, sub_times, nov16)
    ctx.recur_str = np.clip(np.interp(ctx.times, sub_times, ema_zero_phase(strength, alpha)), 0, 1)
    # The pointer is a hard jump, not a curve: interpolate nearest, never linear.
    # searchsorted returns an insertion point, which rounds up and would bias the
    # pointer forward by up to a sub-frame whenever stride > 1; round instead.
    sub_hop = ctx.cfg.hop_seconds * stride
    nearest = np.clip(
        np.round((ctx.times - sub_times[0]) / sub_hop).astype(int), 0, sub_times.size - 1
    )
    ctx.recur_time = pointer[nearest]

    if ctx.segments:
        ctx.segment_similarity = fmc_similarity(chroma, ctx.segments, ctx.cfg.hop_seconds)

    log.info(
        "recurrence: novelty peaks %.2f/%.2f, mean repeat strength %.3f, similarity %s",
        float(ctx.novelty4.max()),
        float(ctx.novelty16.max()),
        float(ctx.recur_str.mean()),
        None if ctx.segment_similarity is None else ctx.segment_similarity.shape,
    )
