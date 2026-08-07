"""PCA must be deterministic and sign-stable: the visual mapping is downstream of it."""

from __future__ import annotations

import numpy as np

from terrarium_analysis.stages.character import MUQ_RATE, frame_rate, pca, standardize, to_latent


def _synthetic_features(n=400, d=32, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, n)
    basis = rng.standard_normal((3, d))
    scores = np.stack([np.sin(t), np.cos(t * 0.5), np.sin(t * 0.25)], axis=1)
    return scores @ basis + 0.05 * rng.standard_normal((n, d))


def test_frame_rate_is_measured_not_assumed():
    """A hardcoded rate drifts the latent stream out of time alignment linearly."""
    assert frame_rate(2500, 100.0) == 25.0
    assert frame_rate(2600, 100.0) == 26.0
    assert frame_rate(0, 100.0) == MUQ_RATE
    assert frame_rate(2500, 0.0) == MUQ_RATE


def test_pca_is_bitwise_repeatable():
    x = standardize(_synthetic_features())
    a, ca, ra = pca(x, 8)
    b, cb, rb = pca(x, 8)
    assert np.array_equal(a, b)
    assert np.array_equal(ca, cb)
    assert np.array_equal(ra, rb)


def test_sign_convention_largest_loading_is_positive():
    x = standardize(_synthetic_features())
    _, comps, _ = pca(x, 8)
    for row in comps:
        assert row[np.argmax(np.abs(row))] > 0


def test_sign_convention_survives_a_flipped_input():
    """Negating the data flips SVD's arbitrary signs; the convention must absorb it."""
    x = standardize(_synthetic_features())
    _, comps_a, _ = pca(x, 6)
    _, comps_b, _ = pca(-x, 6)
    assert np.allclose(comps_a, comps_b, atol=1e-9)


def test_components_are_orthonormal_and_variance_is_ordered():
    x = standardize(_synthetic_features())
    scores, comps, ratio = pca(x, 6)
    assert np.allclose(comps @ comps.T, np.eye(6), atol=1e-9)
    assert np.all(np.diff(ratio) <= 1e-12)
    var = scores.var(axis=0)
    assert np.all(np.diff(var) <= 1e-9)


def test_pca_reconstructs_a_rank_3_signal():
    x = standardize(_synthetic_features())
    scores, comps, ratio = pca(x, 3)
    recon = scores @ comps + x.mean(axis=0, keepdims=True)
    assert ratio.sum() > 0.95
    assert np.abs(recon - x).mean() < 0.15


def test_standardize_zero_mean_unit_var_and_constant_dims_collapse():
    x = _synthetic_features()
    x[:, 0] = 4.2
    z = standardize(x)
    assert np.allclose(z[:, 0], 0.0)
    assert np.allclose(z[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z[:, 1:].std(axis=0), 1.0, atol=1e-12)


def test_to_latent_pads_to_requested_dims_and_is_repeatable():
    x = _synthetic_features(n=120, d=12)
    a = to_latent(x, 64, 0.1)
    b = to_latent(x, 64, 0.1)
    assert a.shape == (120, 64)
    assert np.array_equal(a, b)
    assert np.allclose(a[:, 12:], 0.0)  # only 12 real components exist
    assert np.isfinite(a).all()


def test_to_latent_is_scaled_and_smoothed():
    x = _synthetic_features(n=600, d=32)
    latent = to_latent(x, 64, 0.1)
    assert 0.5 < np.sqrt(np.mean(latent[:, 0] ** 2)) < 1.5
    raw, _, _ = pca(standardize(x), 64)
    raw = raw / np.sqrt(np.mean(raw[:, 0] ** 2))
    assert np.abs(np.diff(latent[:, 0])).mean() < np.abs(np.diff(raw[:, 0])).mean()
