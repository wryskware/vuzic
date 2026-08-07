"""The transient stream: the right kind, at the right time, with no RNG anywhere.

Built on synthetic audio with known click times per band, because that is the
only ground truth available without hand-labelling a real track.
"""

from __future__ import annotations

import numpy as np
import pytest

from terrarium_analysis.stages.events import (
    KINDS,
    SPECS,
    band_curves,
    band_filter,
    dedupe,
    detect,
    kicks_per_active_beat,
)
from terrarium_analysis.timeline import EVENT_KINDS

SR = 22050
ONSET_HOP = 256
FRAME = ONSET_HOP / SR


def _spec(kind: str):
    return next(s for s in SPECS if s.kind == kind)


def _click(n: int, freq: float, decay: float = 40.0, sr: int = SR) -> np.ndarray:
    """A short decaying tone burst — a stand-in for one drum hit."""
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)


def _noise_burst(n: int, seed: int, decay: float = 60.0, sr: int = SR) -> np.ndarray:
    # Fixed generator: the fixture must be identical every run, and the detector
    # itself never draws a random number.
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    return rng.standard_normal(n) * np.exp(-decay * t)


def _place(duration: float, hits: list[tuple[float, np.ndarray]], sr: int = SR) -> np.ndarray:
    y = np.zeros(int(duration * sr), dtype=np.float64)
    for t, w in hits:
        i = int(round(t * sr))
        j = min(i + w.size, y.size)
        y[i:j] += w[: j - i]
    return y


def _detect_kind(y: np.ndarray, kind: str):
    spec = _spec(kind)
    env, rms = band_curves(y, SR, spec)
    return detect(env, rms, FRAME, spec.min_gap)


def _matched(found: np.ndarray, expected: list[float], tol: float = 0.05) -> int:
    return sum(1 for e in expected if found.size and np.abs(found - e).min() <= tol)


# --- the contract ---------------------------------------------------------


def test_kind_vocabulary_matches_the_timeline_contract():
    assert KINDS == EVENT_KINDS
    assert tuple(s.kind for s in SPECS) == EVENT_KINDS


# --- band splitting -------------------------------------------------------


def test_band_filter_keeps_its_band_and_rejects_the_others():
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 60 * t)
    high = np.sin(2 * np.pi * 8000 * t)
    kick_band = band_filter(low + high, SR, None, 120.0)
    hat_band = band_filter(low + high, SR, 4000.0, None)
    # measured over the interior, away from the filter's edge transients
    assert np.abs(kick_band[2000:-2000]).max() > 0.8
    assert np.abs(hat_band[2000:-2000]).max() > 0.8
    assert np.abs(band_filter(high, SR, None, 120.0)[2000:-2000]).max() < 0.05
    assert np.abs(band_filter(low, SR, 4000.0, None)[2000:-2000]).max() < 0.05


def test_band_filter_is_zero_phase_so_bands_do_not_drift_apart():
    """A causal filter would delay the kick band relative to the hat band."""
    y = _place(2.0, [(1.0, _click(4000, 60.0) + _noise_burst(4000, 1) * 0.5)])
    lo = band_filter(y, SR, None, 120.0)
    hi = band_filter(y, SR, 4000.0, None)
    assert abs(int(np.argmax(np.abs(lo))) - int(np.argmax(np.abs(hi)))) < int(0.02 * SR)


# --- detection ------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, times",
    [
        ("kick", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        ("snare", [1.0, 2.0, 3.0]),
        ("hat", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
    ],
)
def test_a_click_train_is_found_at_the_times_it_was_placed(kind, times):
    hits = {
        "kick": lambda i: _click(6000, 55.0, 30.0) * 1.0,
        "snare": lambda i: band_filter(_noise_burst(4000, 10 + i), SR, 200.0, 1800.0),
        "hat": lambda i: band_filter(_noise_burst(1500, 20 + i, decay=180.0), SR, 6000.0, None) * 3,
    }[kind]
    y = _place(4.0, [(t, hits(i)) for i, t in enumerate(times)])
    found, strengths = _detect_kind(y, kind)
    assert _matched(found, times) == len(times), (kind, found)
    assert found.size <= len(times) + 1  # no phantom hits between the real ones
    assert strengths.min() >= 0.0 and strengths.max() <= 1.0
    assert np.all(np.diff(found) > 0)  # sorted, deduped


def test_bands_do_not_steal_each_others_hits():
    """A kick must not register as a hat, and a hat must not register as a kick."""
    kicks = [0.5, 1.5, 2.5]
    hats = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    y = _place(
        3.5,
        [(t, _click(6000, 55.0, 30.0)) for t in kicks]
        + [
            (t, band_filter(_noise_burst(1500, 30 + i, decay=180.0), SR, 6000.0, None) * 3)
            for i, t in enumerate(hats)
        ],
    )
    kick_t, _ = _detect_kind(y, "kick")
    hat_t, _ = _detect_kind(y, "hat")
    assert _matched(kick_t, kicks) == len(kicks)
    assert kick_t.size <= len(kicks) + 1
    assert _matched(hat_t, hats) == len(hats)
    # none of the hat detections sit on a kick that had no hat
    assert not any(abs(h - 0.5) < 0.03 and h not in hats for h in hat_t)


def test_strength_ranks_loud_hits_above_quiet_ones():
    times = [0.5, 1.0, 1.5, 2.0]
    gains = [1.0, 0.25, 1.0, 0.25]
    y = _place(3.0, [(t, _click(6000, 55.0, 30.0) * g) for t, g in zip(times, gains)])
    found, strengths = _detect_kind(y, "kick")
    assert found.size == 4
    assert strengths[0] > strengths[1] and strengths[2] > strengths[3]
    assert strengths.max() <= 1.0


def test_strength_uses_a_robust_percentile_not_the_max():
    """One freak-loud hit must not push every other event to near zero."""
    times = [0.5, 1.0, 1.5, 2.0, 2.5]
    y = _place(
        3.2,
        [(t, _click(6000, 55.0, 30.0) * (12.0 if t == 1.5 else 1.0)) for t in times],
    )
    _, strengths = _detect_kind(y, "kick")
    ordinary = np.sort(strengths)[:-1]
    assert ordinary.min() > 0.3


def test_silence_produces_no_events():
    for kind in ("kick", "snare", "hat", "bass", "vocal"):
        found, strengths = _detect_kind(np.zeros(int(4.0 * SR)), kind)
        assert found.size == 0 and strengths.size == 0


def test_a_stem_that_only_contains_bleed_produces_no_events():
    """stems.py's silence philosophy: demucs bleed below -70 dBFS is not music."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(int(4.0 * SR)) * 1e-5
    y[int(1.0 * SR) : int(1.0 * SR) + 400] *= 40.0  # a bleed transient, still inaudible
    for kind in ("kick", "snare", "hat"):
        found, _ = _detect_kind(y, kind)
        assert found.size == 0, kind


def test_events_stop_where_the_instrument_stops():
    """The breakdown case: hits in the first half only, nothing after."""
    times = [0.5, 1.0, 1.5, 2.0]
    y = _place(8.0, [(t, _click(6000, 55.0, 30.0)) for t in times])
    found, _ = _detect_kind(y, "kick")
    assert found.size == len(times)
    assert found.max() < 2.3


def test_backtracking_never_runs_far_ahead_of_the_transient():
    """Unbounded onset_backtrack put a slow bass note 80 ms early; MAX_BACKTRACK caps it."""
    notes = [0.5, 1.0, 1.5, 2.0, 2.5]
    y = _place(3.5, [(t, _click(8000, 55.0, 12.0)) for t in notes])
    found, _ = _detect_kind(y, "bass")
    offsets = np.array([found[np.abs(found - t).argmin()] - t for t in notes])
    assert offsets.min() > -0.05
    assert offsets.max() < 0.03


def test_detection_is_deterministic():
    y = _place(4.0, [(t, _click(6000, 55.0, 30.0)) for t in (0.5, 1.0, 1.5, 2.0)])
    a_t, a_s = _detect_kind(y, "kick")
    b_t, b_s = _detect_kind(y.copy(), "kick")
    assert np.array_equal(a_t, b_t) and np.array_equal(a_s, b_s)


def test_bass_note_onsets_are_found_on_a_synthetic_bassline():
    notes = [(0.5, 55.0), (1.0, 65.4), (1.5, 55.0), (2.0, 82.4), (2.5, 55.0)]
    y = _place(3.5, [(t, _click(8000, f, 12.0)) for t, f in notes])
    found, _ = _detect_kind(y, "bass")
    assert _matched(found, [t for t, _ in notes]) >= 4
    assert found.size <= len(notes) + 1


def test_vocal_onsets_are_found_on_synthetic_syllables():
    times = [0.5, 1.2, 2.0, 2.8]
    y = _place(
        3.6,
        [
            (t, band_filter(_noise_burst(6000, 50 + i, decay=12.0), SR, 300.0, 3000.0) * 2)
            for i, t in enumerate(times)
        ],
    )
    found, _ = _detect_kind(y, "vocal")
    assert _matched(found, times, tol=0.08) == len(times)


# --- dedupe ---------------------------------------------------------------


def test_dedupe_collapses_near_simultaneous_hits_keeping_the_earlier_time():
    t = np.array([1.000, 1.020, 1.500, 1.530, 2.000])
    s = np.array([0.4, 0.9, 0.8, 0.2, 0.5])
    ot, os_ = dedupe(t, s, 0.06)
    assert np.allclose(ot, [1.0, 1.5, 2.0])
    assert np.allclose(os_, [0.9, 0.8, 0.5])  # strength is the max of the pair


def test_dedupe_keeps_hits_further_apart_than_the_gap():
    t = np.array([1.0, 1.07, 1.14])
    ot, _ = dedupe(t, np.ones(3), 0.06)
    assert ot.size == 3


def test_dedupe_sorts_and_handles_empty():
    ot, os_ = dedupe(np.array([2.0, 1.0]), np.array([0.2, 0.9]), 0.06)
    assert np.allclose(ot, [1.0, 2.0]) and np.allclose(os_, [0.9, 0.2])
    assert dedupe(np.zeros(0), np.zeros(0), 0.06)[0].size == 0


# --- the quality guard ----------------------------------------------------


def test_kicks_per_active_beat_counts_only_drum_active_regions():
    times = np.arange(600) * 0.1  # 60 s grid
    drums = np.zeros(600)
    drums[300:] = 1.0  # drums enter at 30 s
    beats = np.arange(0.0, 60.0, 0.5)  # 120 BPM
    kicks = np.arange(30.0, 60.0, 0.5)  # four on the floor after the entrance
    ratio, n_kicks, n_beats = kicks_per_active_beat(kicks, beats, drums, times)
    assert n_beats == 60 and n_kicks == 60
    assert ratio == pytest.approx(1.0)


def test_kicks_per_active_beat_without_a_stems_curve_is_nan():
    ratio, _, _ = kicks_per_active_beat(np.zeros(3), np.zeros(3), None, np.arange(10) * 0.1)
    assert np.isnan(ratio)


# --- end to end through the stage ----------------------------------------


def test_stage_emits_sorted_contract_shaped_events(tmp_path):
    import soundfile as sf

    from terrarium_analysis.context import Config, Context, build_grid
    from terrarium_analysis.stages import events as events_stage

    kicks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    hats = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    drums = _place(
        4.0,
        [(t, _click(6000, 55.0, 30.0)) for t in kicks]
        + [
            (t, band_filter(_noise_burst(1500, 60 + i, decay=180.0), SR, 6000.0, None) * 3)
            for i, t in enumerate(hats)
        ],
    )
    stem_dir = tmp_path / "demix"
    stem_dir.mkdir()
    sf.write(stem_dir / "drums.wav", drums, SR)
    sf.write(stem_dir / "bass.wav", np.zeros(int(4.0 * SR)), SR)
    sf.write(stem_dir / "vocals.wav", np.zeros(int(4.0 * SR)), SR)
    sf.write(stem_dir / "other.wav", np.zeros(int(4.0 * SR)), SR)

    cfg = Config(input_path=tmp_path / "x.wav", out_dir=tmp_path, stems_dir=stem_dir)
    ctx = Context(cfg=cfg, rng=np.random.default_rng(0))
    ctx.duration = 4.0
    ctx.mono_sr = SR
    build_grid(ctx)
    events_stage.run(ctx)

    assert ctx.events
    ts = [e["t"] for e in ctx.events]
    assert ts == sorted(ts)
    assert {e["kind"] for e in ctx.events} <= set(EVENT_KINDS)
    assert all(0.0 <= e["strength"] <= 1.0 for e in ctx.events)
    assert all(0.0 <= e["t"] <= 4.0 for e in ctx.events)
    # the two silent stems contribute nothing
    assert not [e for e in ctx.events if e["kind"] in ("bass", "vocal")]
    found_kicks = np.array([e["t"] for e in ctx.events if e["kind"] == "kick"])
    assert _matched(found_kicks, kicks) == len(kicks)


def test_manifest_round_trips_events(tmp_path):
    from terrarium_analysis.timeline import (
        TOTAL_DIMS,
        Timeline,
        manifest_dict,
        read_timeline,
        write_timeline,
    )

    tl = Timeline(
        track_id="t",
        duration=2.0,
        sample_rate=48000,
        hop_seconds=0.1,
        frames=20,
        beats=np.array([0.0, 0.5]),
        downbeats=np.array([0.0]),
        tempo=120.0,
        segments=[],
        data=np.zeros((20, TOTAL_DIMS), dtype=np.float32),
        events=[
            {"t": 1.0, "kind": "snare", "strength": 0.5},
            {"t": 0.25, "kind": "kick", "strength": 1.0},
        ],
    )
    doc = manifest_dict(tl)
    assert [e["t"] for e in doc["events"]] == [0.25, 1.0]
    assert list(doc)[-1] == "events"  # last, matching the shared fixture

    write_timeline(tl, tmp_path)
    back = read_timeline(tmp_path)
    assert back.events == [
        {"t": 0.25, "kind": "kick", "strength": 1.0},
        {"t": 1.0, "kind": "snare", "strength": 0.5},
    ]


def test_shared_fixture_events_conform_to_the_contract():
    """The fixture is what web/ conforms to; if it carries events, they are legal."""
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[2] / "data" / "timelines" / "synthetic" / "timeline.json"
    )
    doc = json.loads(fixture.read_text(encoding="utf-8"))
    assert doc["version"] == 2  # the field is additive; the version does not move
    ev = doc.get("events")
    if not ev:
        return
    ts = [e["t"] for e in ev]
    assert ts == sorted(ts)
    assert all(e["kind"] in EVENT_KINDS for e in ev)
    assert all(0.0 <= float(e["strength"]) <= 1.0 for e in ev)
    assert all(0.0 <= float(e["t"]) <= doc["track"]["duration"] for e in ev)


def test_manifest_omits_events_when_there_are_none():
    from terrarium_analysis.timeline import TOTAL_DIMS, Timeline, manifest_dict

    tl = Timeline(
        track_id="t",
        duration=2.0,
        sample_rate=48000,
        hop_seconds=0.1,
        frames=20,
        beats=np.zeros(0),
        downbeats=np.zeros(0),
        tempo=0.0,
        segments=[],
        data=np.zeros((20, TOTAL_DIMS), dtype=np.float32),
    )
    assert "events" not in manifest_dict(tl)
