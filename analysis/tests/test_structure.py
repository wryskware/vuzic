"""Structure-stage glue that is testable without All-In-One installed.

Everything here is the part of `stages/structure.py` that reads the model
package's output rather than producing it: the label vocabulary lookup, the
activation-matrix orientation, the RNG pinning that makes demucs reproducible,
and the 100 fps -> 10 fps decimation of the chorus activation.
"""

from __future__ import annotations

import random
import types

import numpy as np

from terrarium_analysis.dsp import pool_to_grid, resample_curve
from terrarium_analysis.stages import structure

LABELS = ("start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus")


def _fake_package(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_label_vocabulary_finds_harmonix_labels():
    """The installed package exposes HARMONIX_LABELS, not LABELS.

    Both `allin1` 1.1.0 and `allin1_infer` 3.1.0 name it that way; looking only
    for `LABELS` silently returns None, which drops every confidence to 1.0 and
    falls actChorus back to a step function derived from the hard labels.
    """
    pkg = _fake_package("fake_allin1", HARMONIX_LABELS=list(LABELS))
    assert structure._label_vocabulary(pkg) == LABELS


def test_label_vocabulary_still_accepts_plain_labels():
    pkg = _fake_package("fake_allin1_b", LABELS=list(LABELS))
    assert structure._label_vocabulary(pkg) == LABELS


def test_label_vocabulary_returns_none_when_absent():
    assert structure._label_vocabulary(_fake_package("fake_allin1_c")) is None


def test_label_activations_accepts_the_native_orientation():
    """All-In-One softmaxes over dim 0, so `label` arrives as (n_labels, T)."""
    acts = {"label": np.arange(10 * 7, dtype=np.float64).reshape(10, 7)}
    out = structure._label_activations(acts, LABELS)
    assert out.shape == (10, 7)
    assert np.array_equal(out, acts["label"])


def test_label_activations_transposes_the_other_orientation():
    acts = {"label": np.arange(10 * 7, dtype=np.float64).reshape(7, 10)}
    out = structure._label_activations(acts, LABELS)
    assert out.shape == (10, 7)


def test_label_activations_none_without_a_vocabulary():
    acts = {"label": np.zeros((10, 7))}
    assert structure._label_activations(acts, None) is None
    assert structure._label_activations({}, LABELS) is None


def test_seeding_pins_the_stdlib_rng_demucs_draws_its_shift_from():
    """demucs' `shifts=1` offset comes from the unseeded stdlib `random`."""
    structure._seed_everything(1234)
    first = [random.random() for _ in range(4)]
    structure._seed_everything(1234)
    assert [random.random() for _ in range(4)] == first
    structure._seed_everything(5678)
    assert [random.random() for _ in range(4)] != first


def test_activation_pooling_averages_instead_of_decimating():
    """A 100 fps curve resampled to 10 fps must not alias.

    The curve is 1 on every 10th source frame and 0 elsewhere — the pathological
    case, and not a contrived one: the grid times ARE multiples of the hop, so
    point sampling lands on the spikes every time and reports a curve that is
    flat 1.0, ten times the true duty cycle. Mean pooling returns 0.1.
    """
    fps, hop, frames = 100.0, 0.1, 20
    times = np.arange(frames) * hop
    row = np.zeros(int(frames * hop * fps))
    row[::10] = 1.0

    # Frame 0's window is half-width ([-hop/2, hop/2) clipped at t=0), so it
    # legitimately sees 5 source frames rather than 10; every interior frame
    # must land on the duty cycle.
    pooled = pool_to_grid(row, fps, times, hop)[:, 0]
    assert np.allclose(pooled[1:], 0.1, atol=0.02)

    src_times = np.arange(row.size) / fps
    aliased = resample_curve(row, src_times, times)
    assert aliased.min() > 0.9  # what the old point-sampling path produced
