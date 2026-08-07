"""The stems channel is the "instrument comes in" signal; an entrance must be a step."""

from __future__ import annotations

import numpy as np

import pytest

from terrarium_analysis.stages.stems import activity_from_curves, pool_onset, stem_activity

HOP = 0.1
BEAT = 0.5


def _entrance(frames=600, enter=200, level=0.2):
    rms = np.full(frames, 1e-5)
    rms[enter:] = level
    return rms


def test_entrance_is_a_clean_step():
    rms = _entrance()
    act = activity_from_curves(rms, np.zeros_like(rms), HOP, BEAT)
    assert act[:190].max() < 0.05
    # within a beat of the entrance the channel is most of the way up
    assert act[200 + int(BEAT / HOP)] > 0.5
    assert act[-1] > 0.9


def test_quiet_but_present_stem_is_not_buried():
    """Linear RMS would put -30 dB below peak at 0.03; the dB mapping must not."""
    frames = 400
    rms = np.full(frames, 1.0)
    rms[200:] = 10 ** (-30 / 20)
    act = activity_from_curves(rms, np.zeros(frames), HOP, BEAT)
    assert 0.25 < act[-1] < 0.5


def test_release_is_slower_than_attack():
    frames = 800
    rms = np.full(frames, 1e-5)
    rms[200:400] = 0.5
    act = activity_from_curves(rms, np.zeros(frames), HOP, BEAT)
    rise = np.argmax(act > 0.5) - 200
    fall = np.argmax(act[400:] < 0.5)
    assert rise > 0 and fall > rise


def test_output_is_bounded_and_finite():
    rng = np.random.default_rng(0)
    rms = np.abs(rng.standard_normal(500)) * 0.3
    onset = np.abs(rng.standard_normal(500))
    act = activity_from_curves(rms, onset, HOP, BEAT)
    assert np.isfinite(act).all()
    assert act.min() >= 0.0 and act.max() <= 1.0


def test_silence_stays_zero():
    act = activity_from_curves(np.zeros(200), np.zeros(200), HOP, BEAT)
    assert np.allclose(act, 0.0)


def test_a_stem_that_never_plays_stays_zero_despite_bleed_onsets():
    """robust_norm stretches demucs bleed to 0..1; the onset term must be gated."""
    rng = np.random.default_rng(0)
    rms = np.full(500, 1e-4)
    onset = np.abs(rng.standard_normal(500)) * 0.01
    act = activity_from_curves(rms, onset, HOP, BEAT)
    assert act.max() < 1e-9


def test_a_short_entrance_is_not_erased_by_the_peak_percentile():
    """A lead that plays for one second is under 1% of frames but is the event."""
    frames = 2000
    rms = np.full(frames, 1e-8)
    rms[1000:1010] = 0.5
    act = activity_from_curves(rms, np.zeros(frames), HOP, BEAT)
    assert act[:990].max() < 1e-9
    assert act[1000:1015].max() > 0.5


def test_onset_density_lifts_a_percussive_stem():
    frames = 400
    rms = np.full(frames, 1.0)
    rms[200:] = 0.05  # ~26 dB below peak, so the level term is not saturated
    onset = np.zeros(frames)
    onset[200::5] = 1.0
    flat = activity_from_curves(rms, np.zeros(frames), HOP, BEAT)
    percussive = activity_from_curves(rms, onset, HOP, BEAT)
    assert percussive[-1] > flat[-1]
    assert percussive[-1] <= 1.0


@pytest.mark.parametrize(
    "click_time, expected_frame",
    [
        (0.0, 0),  # the very first grid instant
        (1.23, 12),  # nearer 1.2 than 1.3
        (1.27, 13),  # nearer 1.3 — half-open [i*hop, (i+1)*hop) bins would say 12
        (1.98, 20),  # nearer 2.0 than 1.9
    ],
)
def test_onset_pooling_bins_are_centred_on_the_grid(click_time, expected_frame):
    """A click at time t belongs to the grid frame whose instant t*hop is nearest."""
    fine = 0.01
    onset_times = np.arange(400) * fine
    onset_env = np.zeros(onset_times.size)
    onset_env[int(round(click_time / fine))] = 1.0
    times = np.arange(30) * HOP
    onset = pool_onset(onset_env, onset_times, times, HOP)
    assert onset.argmax() == expected_frame
    assert onset.sum() == 1.0  # the click lands in exactly one bin


def test_onset_pooling_handles_an_empty_grid():
    assert pool_onset(np.ones(10), np.arange(10) * 0.01, np.zeros(0), HOP).size == 0


def test_stem_activity_on_synthetic_audio_tracks_an_entrance():
    sr = 22050
    frames = 300
    t = np.arange(frames * int(sr * HOP)) / sr
    y = np.sin(2 * np.pi * 220 * t).astype(np.float32) * 0.5
    y[: 150 * int(sr * HOP)] = 0.0
    times = np.arange(frames) * HOP
    act = stem_activity(y, sr, times, HOP, BEAT)
    assert act.shape == (frames,)
    assert act[:145].max() < 0.15
    assert act[-1] > 0.85
