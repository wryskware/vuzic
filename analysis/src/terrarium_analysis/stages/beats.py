"""Stage 2 — beat grid from Beat This! (the master clock).

plan.md Decision 1: Beat This! beats override All-In-One's where they disagree.
This stage runs before `structure`, and `structure` only fills in beats it finds
missing, so the override is structural rather than a comparison rule.
"""

from __future__ import annotations

import numpy as np

from ..context import Context, log
from ..optional import require

NAME = "beats"


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        torch = require("torch")
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tempo_from_beats(beats: np.ndarray) -> float:
    beats = np.asarray(beats, dtype=np.float64)
    if beats.size < 2:
        return 0.0
    ibi = np.diff(beats)
    ibi = ibi[(ibi > 0.15) & (ibi < 2.0)]
    if ibi.size == 0:
        return 0.0
    return float(60.0 / np.median(ibi))


def run(ctx: Context) -> None:
    File2Beats = require("beat_this.inference", "File2Beats")
    device = _resolve_device(ctx.cfg.device)
    log.info("beats: Beat This! on %s", device)
    f2b = File2Beats(device=device, dbn=False)
    beats, downbeats = f2b(str(ctx.wav_path))

    ctx.beats = np.asarray(beats, dtype=np.float64)
    ctx.downbeats = np.asarray(downbeats, dtype=np.float64)
    ctx.tempo = tempo_from_beats(ctx.beats)
    log.info(
        "beats: %d beats, %d downbeats, tempo %.2f BPM",
        ctx.beats.size,
        ctx.downbeats.size,
        ctx.tempo,
    )
