"""Stage 3c — the sparse transient event stream (plan.md Revision 2, item 2).

Beat-to-beat reactivity without a runtime FFT. The dense stem curves say *what
is playing*; this stage says *what just hit*, with a source label the runtime
cannot recover from a spectrum: a kick and a bass note occupy the same octave,
and only the demucs separation tells them apart.

Detection is band-split onset picking on the stems the structure stage already
wrote:

    drums  <120 Hz            -> kick
    drums  120 Hz .. 2 kHz    -> snare  (snare / clap / rim)
    drums  >4 kHz             -> hat
    bass   30 .. 500 Hz       -> bass   (note onsets)
    vocals 150 Hz .. 6 kHz    -> vocal

Everything here is a deterministic function of the stem audio — no RNG, no
thresholds drawn from a distribution. The silence gate is the same idea as
`stems.py`: demucs bleed in a stem that is not playing must produce nothing, so
a peak only survives where the band is both absolutely audible and loud relative
to its own dynamic range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..context import Context, log
from ..dsp import EPS, SILENCE_DB, db_level
from .stems import resolve_stem_paths

NAME = "events"

# The contract's kind vocabulary, in emission-tie order.
KINDS = ("kick", "snare", "hat", "bass", "vocal")

ONSET_HOP = 256  # 11.6 ms at 22.05 kHz — a 16th note at 130 BPM is 115 ms
FILTER_ORDER = 4

# peak_pick geometry, in frames of ONSET_HOP.
PRE_MAX = 3
POST_MAX = 3
PRE_AVG = 12
POST_AVG = 12
# Threshold above the local mean, in units of the band's own 99th percentile.
DELTA = 0.10

GATE_KNEE = 0.10  # fraction of the band's dynamic range below which nothing fires
# Backtracking moves a peak to the start of its transient, but `onset_backtrack`
# walks all the way to the previous envelope minimum — on a slow-attack low note
# that lands 80 ms early, which at 130 BPM is a sixth of a beat. Bounded here.
MAX_BACKTRACK = 0.04
STRENGTH_PCT = 90.0  # robust scale for strength; a plain max is one loud hit away from useless
MIN_STRENGTH = 0.15  # a detected event that maps to ~0 is a wasted event (Revision 2, item 3)

# Sanity bounds for kicks per beat inside drum-active regions (guard, not a filter).
KICK_PER_BEAT_LO = 0.25
KICK_PER_BEAT_HI = 2.0
DRUMS_ACTIVE = 0.5


@dataclass
class BandSpec:
    """One detector: a stem, a band, and how close two hits may be."""

    kind: str
    stem: str
    lo: float | None  # highpass edge, Hz
    hi: float | None  # lowpass edge, Hz
    min_gap: float  # seconds between two events of this kind
    onset_kwargs: dict = field(default_factory=dict)


SPECS: tuple[BandSpec, ...] = (
    BandSpec("kick", "drums", None, 120.0, 0.09),
    BandSpec("snare", "drums", 120.0, 2000.0, 0.09),
    BandSpec("hat", "drums", 4000.0, None, 0.06),
    # Note onsets: a mel view capped near the bass register keeps the envelope
    # from being dominated by the click of the pluck's upper harmonics.
    BandSpec("bass", "bass", 30.0, 500.0, 0.10, {"fmax": 500.0, "n_mels": 48}),
    BandSpec("vocal", "vocals", 150.0, 6000.0, 0.10),
)


def band_filter(
    y: np.ndarray, sr: int, lo: float | None, hi: float | None, order: int = FILTER_ORDER
) -> np.ndarray:
    """Zero-phase Butterworth band split.

    `sosfiltfilt`, not `sosfilt`: a causal filter delays the transient by tens of
    milliseconds and the delay differs per band, which would smear the three drum
    kinds apart in time. Zero-phase costs nothing offline.
    """
    from scipy.signal import butter, sosfiltfilt

    y = np.asarray(y, dtype=np.float64)
    nyq = 0.5 * sr
    lo_n = None if not lo else float(np.clip(lo / nyq, 1e-4, 0.99))
    hi_n = None if not hi else float(np.clip(hi / nyq, 1e-4, 0.99))
    if lo_n is not None and hi_n is not None and lo_n >= hi_n:
        hi_n = None
    if lo_n is not None and hi_n is not None:
        sos = butter(order, [lo_n, hi_n], btype="bandpass", output="sos")
    elif hi_n is not None:
        sos = butter(order, hi_n, btype="lowpass", output="sos")
    elif lo_n is not None:
        sos = butter(order, lo_n, btype="highpass", output="sos")
    else:
        return y
    if y.size < 3 * (order * 2 + 1):
        return y
    return np.asarray(sosfiltfilt(sos, y), dtype=np.float64)


def band_curves(
    y: np.ndarray, sr: int, spec: BandSpec, hop: int = ONSET_HOP
) -> tuple[np.ndarray, np.ndarray]:
    """(onset envelope, RMS) for one band, on the same frame grid."""
    import librosa

    yb = band_filter(y, sr, spec.lo, spec.hi)
    if yb.size == 0:
        return np.zeros(0), np.zeros(0)
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=hop, **spec.onset_kwargs)
    rms = librosa.feature.rms(y=yb, frame_length=hop * 4, hop_length=hop, center=True)[0]
    n = min(env.size, rms.size)
    return np.asarray(env[:n], dtype=np.float64), np.asarray(rms[:n], dtype=np.float64)


def dedupe(times: np.ndarray, strengths: np.ndarray, min_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Collapse same-kind events closer than `min_gap`, keeping the earlier time.

    Backtracking to the onset start can land two picked peaks on the same
    transient; the transient begins at the earlier of the two, and its strength
    is the louder of the two.
    """
    times = np.asarray(times, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if times.size == 0:
        return times, strengths
    order = np.argsort(times, kind="stable")
    keep_t: list[float] = []
    keep_s: list[float] = []
    for i in order:
        t, s = float(times[i]), float(strengths[i])
        if keep_t and t - keep_t[-1] < min_gap:
            keep_s[-1] = max(keep_s[-1], s)
            continue
        keep_t.append(t)
        keep_s.append(s)
    return np.asarray(keep_t), np.asarray(keep_s)


def detect(
    env: np.ndarray,
    rms: np.ndarray,
    frame_seconds: float,
    min_gap: float,
    delta: float = DELTA,
    gate_knee: float = GATE_KNEE,
    strength_pct: float = STRENGTH_PCT,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick, gate, backtrack and normalise one band. Returns (times, strengths).

    Pure numpy plus librosa's peak picker — unit-testable without audio.
    """
    import librosa

    empty = (np.zeros(0), np.zeros(0))
    env = np.asarray(env, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    if env.size == 0:
        return empty

    # Scale by a high percentile rather than the max so one outlier hit does not
    # push the whole band below the picking threshold.
    scale = float(np.percentile(env, 99.0))
    if scale < EPS:
        return empty
    norm = env / scale

    wait = max(1, int(round(min_gap / max(frame_seconds, EPS))))
    peaks = np.asarray(
        librosa.util.peak_pick(
            norm,
            pre_max=PRE_MAX,
            post_max=POST_MAX,
            pre_avg=PRE_AVG,
            post_avg=POST_AVG,
            delta=delta,
            wait=wait,
        ),
        dtype=int,
    )
    if peaks.size == 0:
        return empty

    # Silence gate — the stems.py rule. Absolute audibility first (a stem that
    # never plays is demucs bleed and must produce nothing at all), then a
    # relative knee so the tail of a decaying section stops firing.
    if rms.size:
        db = 20.0 * np.log10(np.maximum(rms, EPS))
        level = db_level(rms)
        audible = (db > SILENCE_DB) & (level > gate_knee)
        peaks = peaks[audible[np.clip(peaks, 0, audible.size - 1)]]
    if peaks.size == 0:
        return empty

    hits = env[peaks]
    s_scale = float(np.percentile(hits, strength_pct))
    if s_scale < EPS:
        s_scale = float(hits.max())
    if s_scale < EPS:
        return empty
    strengths = MIN_STRENGTH + (1.0 - MIN_STRENGTH) * np.clip(hits / s_scale, 0.0, 1.0)

    starts = np.asarray(librosa.onset.onset_backtrack(peaks, env), dtype=int)
    limit = int(np.ceil(MAX_BACKTRACK / max(frame_seconds, EPS)))
    starts = np.maximum(starts, peaks - limit)
    times = starts.astype(np.float64) * frame_seconds
    return dedupe(times, strengths, min_gap)


def kicks_per_active_beat(
    kick_times: np.ndarray,
    beats: np.ndarray,
    drums: np.ndarray | None,
    times: np.ndarray,
    threshold: float = DRUMS_ACTIVE,
) -> tuple[float, int, int]:
    """(kicks per beat, kicks, beats) inside drum-active regions.

    The guard from the task brief: four-on-the-floor at 130 BPM should give
    roughly one kick per beat where the drums are playing. Reported, never used
    to filter — a detector that quietly rewrites its own output to satisfy a
    sanity check is worse than one that is visibly wrong.
    """
    kick_times = np.asarray(kick_times, dtype=np.float64)
    beats = np.asarray(beats, dtype=np.float64)
    if drums is None or times.size == 0 or beats.size == 0:
        return float("nan"), int(kick_times.size), int(beats.size)
    drums = np.asarray(drums, dtype=np.float64)
    active = drums > threshold

    def _active_at(t: np.ndarray) -> np.ndarray:
        idx = np.clip(np.round(t / max(times[1] - times[0], EPS)).astype(int), 0, active.size - 1)
        return active[idx]

    n_beats = int(_active_at(beats).sum())
    n_kicks = int(_active_at(kick_times).sum()) if kick_times.size else 0
    ratio = n_kicks / n_beats if n_beats else float("nan")
    return ratio, n_kicks, n_beats


def run(ctx: Context) -> None:
    import librosa

    paths = resolve_stem_paths(ctx)
    if not paths:
        ctx.note(
            "events: no stem files. They come from All-In-One's demucs pass "
            "(--skip structure removes them), or pass --stems-dir DIR containing "
            "bass/drums/vocals/other.wav."
        )
        return

    frame_seconds = ONSET_HOP / float(ctx.mono_sr)
    cache: dict[str, np.ndarray] = {}
    per_kind: dict[str, np.ndarray] = {}
    events: list[dict] = []

    for spec in SPECS:
        path = paths.get(spec.stem)
        if path is None:
            continue
        if spec.stem not in cache:
            cache[spec.stem] = librosa.load(str(path), sr=ctx.mono_sr, mono=True)[0]
        env, rms = band_curves(cache[spec.stem], ctx.mono_sr, spec)
        times, strengths = detect(env, rms, frame_seconds, spec.min_gap)
        per_kind[spec.kind] = times
        events.extend(
            {"t": float(t), "kind": spec.kind, "strength": float(s)}
            for t, s in zip(times, strengths)
        )
        rate = times.size / ctx.duration * 60.0 if ctx.duration > 0 else 0.0
        log.info(
            "events: %-5s %4d  (%.0f/min, mean strength %.2f, %s)",
            spec.kind,
            times.size,
            rate,
            float(strengths.mean()) if strengths.size else 0.0,
            spec.stem,
        )

    events.sort(key=lambda e: (e["t"], KINDS.index(e["kind"])))
    ctx.events = events

    drums = ctx.stems[:, 1] if ctx.stems is not None else None
    ratio, n_kicks, n_beats = kicks_per_active_beat(
        per_kind.get("kick", np.zeros(0)), ctx.beats, drums, ctx.times
    )
    log.info(
        "events: %d total; %d kicks against %d beats in drum-active regions (%.2f per beat)",
        len(events),
        n_kicks,
        n_beats,
        ratio,
    )
    if np.isfinite(ratio) and not (KICK_PER_BEAT_LO <= ratio <= KICK_PER_BEAT_HI):
        ctx.note(
            f"events: {ratio:.2f} kicks per beat in drum-active regions is outside "
            f"[{KICK_PER_BEAT_LO}, {KICK_PER_BEAT_HI}] — check the kick band threshold"
        )
