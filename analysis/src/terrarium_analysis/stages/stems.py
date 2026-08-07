"""Stage 3b — per-stem activity curves (the "instrument comes in" signal).

plan.md Decision 2, commitment 2: stems are the highest-value channel in the
file, and they are smoothed here so the runtime never has to.

The curve for one stem is a dB-domain loudness level blended with an onset
density, then run through an asymmetric EMA at beat scale: fast attack so an
entrance is a step, slow release so a rest between phrases does not flicker.
Linear RMS is deliberately avoided — it buries a quiet-but-present instrument
at the bottom of the range, which is exactly the event this channel exists to
show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..context import Context, log
from ..dsp import EPS, SILENCE_DB, alpha_for_tau, db_level, ema_asym, robust_norm
from ..timeline import STEM_NAMES

NAME = "stems"

ONSET_WEIGHT = 0.3
DYNAMIC_RANGE_DB = 45.0
ATTACK_BEATS = 0.5
RELEASE_BEATS = 2.0
GATE_KNEE = 0.15


def activity_from_curves(
    rms: np.ndarray,
    onset: np.ndarray,
    hop_seconds: float,
    beat_period: float,
    onset_weight: float = ONSET_WEIGHT,
    dynamic_range_db: float = DYNAMIC_RANGE_DB,
    attack_beats: float = ATTACK_BEATS,
    release_beats: float = RELEASE_BEATS,
) -> np.ndarray:
    """Combine per-frame RMS and onset strength into a 0..1 activity curve.

    Pure numpy so it is unit-testable without audio; the librosa feature
    extraction lives in `stem_activity`.
    """
    rms = np.asarray(rms, dtype=np.float64)
    onset = np.asarray(onset, dtype=np.float64)
    if rms.size == 0:
        return rms
    level = db_level(rms, dynamic_range_db=dynamic_range_db)
    onset_n = robust_norm(onset) if onset.size == rms.size else np.zeros_like(level)
    # Onset density is a boost on top of the level, not a blend with it: a
    # sustained pad with no transients must still be able to reach 1.0.
    # The gate is load-bearing: robust_norm rescales whatever it is given to
    # 0..1, so a stem that never plays would have its demucs bleed stretched to
    # full range and read as permanently ~0.15 active. Onset may only lift a
    # stem that the level term already says is audible.
    db = 20.0 * np.log10(np.maximum(rms, EPS))
    gate = np.where(db > SILENCE_DB, np.clip(level / GATE_KNEE, 0.0, 1.0), 0.0)
    raw = np.clip(level + onset_weight * onset_n * gate, 0.0, 1.0)
    smoothed = ema_asym(
        raw,
        alpha_for_tau(attack_beats * beat_period, hop_seconds),
        alpha_for_tau(release_beats * beat_period, hop_seconds),
    )
    return np.clip(smoothed, 0.0, 1.0)


def pool_onset(
    onset_env: np.ndarray,
    onset_times: np.ndarray,
    times: np.ndarray,
    hop_seconds: float,
) -> np.ndarray:
    """Max-pool a fine onset envelope onto the coarse grid.

    Max, not mean: a transient must survive the rate reduction, and averaging
    would smear it away. The bins are CENTRED on the grid times — grid frame i is
    the instant t = i*hop, not the interval after it — so the onset term lines up
    with the RMS term instead of leading it by half a hop.
    """
    onset = np.zeros(times.size, dtype=np.float64)
    if times.size == 0:
        return onset
    bounds = np.concatenate([times - hop_seconds * 0.5, times[-1:] + hop_seconds * 0.5])
    edges = np.searchsorted(onset_times, bounds)
    for i in range(times.size):
        a, b = int(edges[i]), int(edges[i + 1])
        if b > a:
            onset[i] = float(onset_env[a:b].max())
    return onset


def stem_activity(
    y: np.ndarray,
    sr: int,
    times: np.ndarray,
    hop_seconds: float,
    beat_period: float,
) -> np.ndarray:
    """Activity curve for one stem signal, sampled on `times`."""
    import librosa

    grid_hop = max(1, int(round(sr * hop_seconds)))
    fine_hop = max(1, int(round(sr * 0.01)))

    rms = librosa.feature.rms(
        y=y, frame_length=grid_hop * 4, hop_length=grid_hop, center=True
    )[0]
    rms_times = np.arange(rms.size, dtype=np.float64) * hop_seconds

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=fine_hop)
    onset_times = np.arange(onset_env.size, dtype=np.float64) * (fine_hop / sr)
    onset = pool_onset(onset_env, onset_times, times, hop_seconds)

    rms_on_grid = np.interp(times, rms_times, rms) if rms.size else np.zeros_like(times)
    return activity_from_curves(rms_on_grid, onset, hop_seconds, beat_period)


def resolve_stem_paths(ctx: Context) -> dict[str, Path]:
    """Stem WAVs from the structure stage, or from an explicit --stems-dir.

    Public because the events stage reads the same four files.
    """
    if ctx.cfg.stems_dir is not None:
        d = Path(ctx.cfg.stems_dir)
        found = {}
        for name in STEM_NAMES:
            for path in d.rglob(f"{name}.wav"):
                found[name] = path
                break
        return found
    return dict(ctx.stem_paths)


def run(ctx: Context) -> None:
    import librosa

    paths = resolve_stem_paths(ctx)
    if not paths:
        ctx.note(
            "stems: no stem files. They come from All-In-One's demucs pass "
            "(--skip structure removes them), or pass --stems-dir DIR containing "
            "bass/drums/vocals/other.wav."
        )
        return

    out = np.zeros((ctx.frames, len(STEM_NAMES)), dtype=np.float64)
    beat_period = ctx.beat_period
    for i, name in enumerate(STEM_NAMES):
        path = paths.get(name)
        if path is None:
            continue
        y, sr = librosa.load(str(path), sr=ctx.mono_sr, mono=True)
        out[:, i] = stem_activity(y, sr, ctx.times, ctx.cfg.hop_seconds, beat_period)
        log.info(
            "stems: %-6s mean %.3f  p95 %.3f  (%s)",
            name,
            float(out[:, i].mean()),
            float(np.percentile(out[:, i], 95)),
            path.name,
        )
    ctx.stems = out
