"""Small numpy signal helpers shared by the stages. No optional dependencies."""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def alpha_for_tau(tau_seconds: float, hop_seconds: float) -> float:
    """EMA coefficient whose impulse response decays with time constant `tau`."""
    if tau_seconds <= 0:
        return 1.0
    return float(1.0 - np.exp(-hop_seconds / tau_seconds))


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA along axis 0."""
    x = np.asarray(x, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.empty_like(x)
    if x.shape[0] == 0:
        return out
    acc = x[0].copy()
    out[0] = acc
    for i in range(1, x.shape[0]):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def ema_asym(x: np.ndarray, alpha_up: float, alpha_down: float) -> np.ndarray:
    """EMA with separate attack and release rates.

    Asymmetry is the point: an instrument entering must read as a step, so the
    attack is fast; the release is slow so a gap between phrases does not make
    the channel flicker.
    """
    x = np.asarray(x, dtype=np.float64)
    au = float(np.clip(alpha_up, 0.0, 1.0))
    ad = float(np.clip(alpha_down, 0.0, 1.0))
    out = np.empty_like(x)
    if x.shape[0] == 0:
        return out
    acc = x[0].copy()
    out[0] = acc
    for i in range(1, x.shape[0]):
        d = x[i] - acc
        acc = acc + np.where(d > 0, au, ad) * d
        out[i] = acc
    return out


def ema_zero_phase(x: np.ndarray, alpha: float) -> np.ndarray:
    """Forward+backward EMA: smooths without shifting events in time."""
    fwd = ema(x, alpha)
    bwd = ema(fwd[::-1], alpha)[::-1]
    return bwd


def robust_norm(x: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Percentile-based normalisation to 0..1, clipped."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi - lo < EPS:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


SILENCE_DB = -70.0


def db_level(rms: np.ndarray, dynamic_range_db: float = 45.0, peak_pct: float = 99.0) -> np.ndarray:
    """RMS -> 0..1 loudness in the dB domain.

    Linear RMS makes a present-but-quiet instrument indistinguishable from
    silence; dB with a fixed window below the track's own peak makes an entrance
    a clean step, which is the whole job of the stems channel.

    The window is relative to the stem's own peak, so a stem that is silent for
    the whole track would otherwise normalise its noise floor up to 1.0 —
    `SILENCE_DB` is the absolute cutoff that stops that.

    The peak percentile is taken over the audible frames only. Over all frames it
    would erase a stem that plays in under (100 - peak_pct)% of the track — a
    lead that enters for a few seconds is exactly the event this exists to show.
    """
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size == 0:
        return rms
    db = 20.0 * np.log10(np.maximum(rms, EPS))
    audible = db > SILENCE_DB
    if not audible.any():
        return np.zeros_like(db)
    peak = float(np.percentile(db[audible], peak_pct))
    floor = max(peak - dynamic_range_db, SILENCE_DB)
    if peak - floor < EPS:
        return np.zeros_like(db)
    return np.clip((db - floor) / (peak - floor), 0.0, 1.0)


def resample_curve(values: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (T,) or (T, D) curve onto `dst_times`."""
    values = np.asarray(values, dtype=np.float64)
    src_times = np.asarray(src_times, dtype=np.float64)
    dst_times = np.asarray(dst_times, dtype=np.float64)
    if values.ndim == 1:
        if values.size == 0:
            return np.zeros_like(dst_times)
        return np.interp(dst_times, src_times, values)
    out = np.empty((dst_times.size, values.shape[1]), dtype=np.float64)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(dst_times, src_times, values[:, d])
    return out


def pool_to_grid(
    x: np.ndarray, src_rate: float, dst_times: np.ndarray, hop_seconds: float
) -> np.ndarray:
    """Mean-pool a (T, D) feature sequence onto a coarser regular grid.

    Each output frame averages the source frames whose centres fall inside its
    hop-wide window; windows that catch no source frame fall back to the nearest
    one. This is the 25 Hz -> 10 Hz step, where the ratio is not an integer.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_src = x.shape[0]
    dst_times = np.asarray(dst_times, dtype=np.float64)
    out = np.zeros((dst_times.size, x.shape[1]), dtype=np.float64)
    if n_src == 0:
        return out
    src_times = np.arange(n_src, dtype=np.float64) / float(src_rate)
    lo = np.searchsorted(src_times, dst_times - hop_seconds * 0.5, side="left")
    hi = np.searchsorted(src_times, dst_times + hop_seconds * 0.5, side="left")
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)
    for i in range(dst_times.size):
        a, b = int(lo[i]), int(hi[i])
        if b > a:
            out[i] = (csum[b] - csum[a]) / (b - a)
        else:
            out[i] = x[min(max(a, 0), n_src - 1)]
    return out


def checkerboard_kernel(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel for Foote novelty (size must be even)."""
    size = max(2, int(size) // 2 * 2)
    half = size // 2
    axis = np.arange(-half, half, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(axis, axis)
    taper = np.exp(-(gx**2 + gy**2) / (2.0 * (half / 2.0) ** 2))
    sign = np.sign(gx * gy)
    return sign * taper


def foote_novelty(ssm: np.ndarray, kernel_size: int) -> np.ndarray:
    """Foote novelty: correlate a checkerboard kernel down the SSM diagonal."""
    ssm = np.asarray(ssm, dtype=np.float64)
    n = ssm.shape[0]
    kernel = checkerboard_kernel(kernel_size)
    half = kernel.shape[0] // 2
    padded = np.pad(ssm, half, mode="edge")
    nov = np.zeros(n, dtype=np.float64)
    for i in range(n):
        patch = padded[i : i + 2 * half, i : i + 2 * half]
        nov[i] = float(np.sum(patch * kernel))
    nov = np.maximum(nov, 0.0)
    return nov


def segment_label_curve(
    segments: list[dict], times: np.ndarray, label: str, ramp_seconds: float = 0.3
) -> np.ndarray:
    """Fallback graded activation from hard segment labels (smoothed step)."""
    times = np.asarray(times, dtype=np.float64)
    out = np.zeros_like(times)
    for seg in segments:
        if label not in str(seg.get("label", "")).lower():
            continue
        inside = (times >= seg["start"]) & (times < seg["end"])
        out[inside] = 1.0
    if ramp_seconds > 0 and times.size > 1:
        hop = float(times[1] - times[0])
        out = ema_zero_phase(out, alpha_for_tau(ramp_seconds, hop))
    return np.clip(out, 0.0, 1.0)
