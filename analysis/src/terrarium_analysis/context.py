"""Run configuration and the object every stage reads from and writes to."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("terrarium")

STAGE_NAMES = ("audio", "beats", "structure", "stems", "events", "character", "recurrence")


@dataclass
class Config:
    input_path: Path
    out_dir: Path
    track_id: str = ""
    hop_seconds: float = 0.1
    seed: int = 0
    device: str = "auto"
    skip: tuple[str, ...] = ()
    stems_dir: Path | None = None
    dump_embedding: bool = False
    muq_model: str = "OpenMuQ/MuQ-large-msd-iter"
    muq_layer: int = 6
    latent_dims: int = 64
    analysis_sr: int = 22050
    keep_byproducts: bool = True

    def enabled(self, stage: str) -> bool:
        return stage not in self.skip


@dataclass
class Context:
    cfg: Config
    rng: np.random.Generator

    # stage 1 — audio
    wav_path: Path | None = None
    sample_rate: int = 48000
    duration: float = 0.0
    mono: np.ndarray | None = None  # at cfg.analysis_sr, for librosa work
    mono_sr: int = 22050

    # grid
    frames: int = 0
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # stage 2 — beats
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downbeats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tempo: float = 0.0

    # stage 3 — structure
    segments: list[dict] = field(default_factory=list)
    act_chorus: np.ndarray | None = None
    stem_paths: dict[str, Path] = field(default_factory=dict)

    # stage 4 — stems
    stems: np.ndarray | None = None  # (frames, 4)

    # stage 4b — events: sparse {t, kind, strength}, sorted by t
    events: list[dict] = field(default_factory=list)

    # stage 5 — character
    latent: np.ndarray | None = None  # (frames, 64)
    embedding: np.ndarray | None = None  # (frames, 1024) raw pooled

    # stage 6 — recurrence
    novelty4: np.ndarray | None = None
    novelty16: np.ndarray | None = None
    recur_time: np.ndarray | None = None
    recur_str: np.ndarray | None = None
    segment_similarity: np.ndarray | None = None

    notes: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)

    @property
    def beat_period(self) -> float:
        """Seconds per beat; falls back to 120 BPM when no beat grid exists."""
        if self.beats.size >= 2:
            d = float(np.median(np.diff(self.beats)))
            if 0.15 < d < 2.0:
                return d
        if self.tempo > 0:
            return 60.0 / self.tempo
        return 0.5

    @property
    def bar_period(self) -> float:
        if self.downbeats.size >= 2:
            d = float(np.median(np.diff(self.downbeats)))
            if d > 0.5:
                return d
        return 4.0 * self.beat_period

    def note(self, msg: str) -> None:
        log.warning(msg)
        self.notes.append(msg)


def build_grid(ctx: Context) -> None:
    hop = ctx.cfg.hop_seconds
    ctx.frames = max(1, int(np.floor(ctx.duration / hop + 1e-6)))
    ctx.times = np.arange(ctx.frames, dtype=np.float64) * hop
