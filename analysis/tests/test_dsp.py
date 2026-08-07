from __future__ import annotations

import numpy as np

from terrarium_analysis.dsp import (
    alpha_for_tau,
    db_level,
    ema,
    ema_asym,
    ema_zero_phase,
    foote_novelty,
    pool_to_grid,
    resample_curve,
    robust_norm,
    segment_label_curve,
)


def test_alpha_for_tau_matches_the_exponential_it_names():
    hop, tau = 0.1, 0.5
    a = alpha_for_tau(tau, hop)
    # ema seeds on x[0], so the step starts from an explicit zero sample
    step = ema(np.concatenate([[0.0], np.ones(int(tau / hop))]), a)
    assert abs(step[-1] - (1 - np.exp(-1))) < 1e-9


def test_ema_converges_and_preserves_a_constant():
    x = np.full(50, 3.0)
    assert np.allclose(ema(x, 0.3), 3.0)
    assert ema(np.concatenate([np.zeros(1), np.ones(200)]), 0.2)[-1] > 0.999


def test_ema_is_causal():
    x = np.zeros(20)
    x[10] = 1.0
    y = ema(x, 0.5)
    assert np.allclose(y[:10], 0.0)
    assert y[10] > 0


def test_ema_zero_phase_does_not_shift_a_peak():
    x = np.zeros(101)
    x[50] = 1.0
    y = ema_zero_phase(x, 0.3)
    assert int(np.argmax(y)) == 50
    assert np.allclose(y, y[::-1], atol=1e-6)  # edge seeding, not phase, breaks exactness


def test_ema_asym_attack_and_release_differ():
    x = np.concatenate([np.zeros(10), np.ones(40), np.zeros(40)])
    y = ema_asym(x, alpha_up=0.5, alpha_down=0.05)
    rise = int(np.argmax(y > 0.5)) - 10
    fall = int(np.argmax(y[50:] < 0.5))
    assert 0 < rise < fall


def test_ema_asym_handles_multiple_columns():
    x = np.stack([np.arange(10.0), np.arange(10.0)[::-1]], axis=1)
    y = ema_asym(x, 1.0, 1.0)
    assert np.allclose(y, x)


def test_db_level_is_a_step_for_a_step():
    rms = np.concatenate([np.full(50, 1e-6), np.full(50, 0.5)])
    lvl = db_level(rms, dynamic_range_db=45.0)
    assert lvl[:50].max() < 1e-9
    assert lvl[-1] > 0.99


def test_db_level_is_not_linear():
    rms = np.array([1.0, 10 ** (-20 / 20), 10 ** (-40 / 20)])
    lvl = db_level(rms, dynamic_range_db=45.0, peak_pct=100.0)
    assert abs(lvl[1] - (45 - 20) / 45) < 1e-6
    assert lvl[2] > 0.1  # linear would give 0.01


def test_db_level_anchors_on_audible_frames_not_all_frames():
    rms = np.full(2000, 1e-8)
    rms[1000:1010] = 0.5
    lvl = db_level(rms)
    assert lvl[1000:1010].min() > 0.99
    assert lvl[:1000].max() == 0.0


def test_db_level_of_an_entirely_silent_stem_is_zero():
    assert np.allclose(db_level(np.full(100, 1e-9)), 0.0)


def test_robust_norm_clips_outliers_into_range():
    x = np.concatenate([np.linspace(0, 1, 100), [500.0]])
    y = robust_norm(x)
    assert y.min() == 0.0 and y.max() == 1.0
    assert np.all(np.diff(y[:100]) >= -1e-12)


def test_robust_norm_of_a_constant_is_zero():
    assert np.allclose(robust_norm(np.full(10, 7.0)), 0.0)


def test_pool_to_grid_averages_25hz_into_10hz_bins():
    src = np.arange(25, dtype=float)[:, None]  # 1 s at 25 Hz
    dst = np.arange(10) * 0.1
    out = pool_to_grid(src, 25.0, dst, 0.1)
    assert out.shape == (10, 1)
    # source frames at 0.00 and 0.04 s fall in the window centred on 0.0 (-0.05..0.05)
    assert out[0, 0] == np.mean([0, 1])
    # 0.08 and 0.12 s fall in the window centred on 0.1 (0.05..0.15)
    assert out[1, 0] == np.mean([2, 3])
    assert np.all(np.diff(out[:, 0]) > 0)


def test_pool_to_grid_falls_back_to_nearest_when_a_bin_is_empty():
    src = np.array([[1.0], [2.0]])
    out = pool_to_grid(src, 1.0, np.array([0.0, 0.1, 0.2]), 0.05)
    assert out[0, 0] == 1.0
    assert np.isfinite(out).all()


def test_pool_to_grid_handles_1d_input():
    out = pool_to_grid(np.arange(10.0), 10.0, np.array([0.0, 0.5]), 0.1)
    assert out.shape == (2, 1)


def test_resample_curve_1d_and_2d():
    src_t = np.array([0.0, 1.0, 2.0])
    assert np.allclose(resample_curve(np.array([0.0, 10.0, 20.0]), src_t, np.array([0.5])), 5.0)
    two = resample_curve(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0.0, 1.0]), np.array([0.5]))
    assert two.shape == (1, 2)
    assert np.allclose(two, [[1.0, 2.0]])


def test_foote_novelty_peaks_at_a_block_boundary():
    n = 120
    labels = np.zeros(n)
    labels[60:] = 1.0
    ssm = (labels[:, None] == labels[None, :]).astype(float)
    nov = foote_novelty(ssm, 32)
    assert abs(int(np.argmax(nov)) - 60) <= 2
    assert nov.min() >= 0.0


def test_segment_label_curve_is_one_inside_the_labelled_segment():
    times = np.arange(100) * 0.1
    segs = [
        {"start": 0.0, "end": 3.0, "label": "verse"},
        {"start": 3.0, "end": 7.0, "label": "chorus"},
        {"start": 7.0, "end": 10.0, "label": "verse"},
    ]
    curve = segment_label_curve(segs, times, "chorus")
    assert curve[50] > 0.9
    assert curve[5] < 0.2
    assert curve.max() <= 1.0
