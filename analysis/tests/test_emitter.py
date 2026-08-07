"""The timeline layout is the contract with web/; these tests are the contract test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from terrarium_analysis.timeline import (
    CHANNELS,
    TOTAL_DIMS,
    Timeline,
    channel_index,
    manifest_dict,
    pack,
    read_timeline,
    write_timeline,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "timelines" / "synthetic" / "timeline.json"


def test_matches_shared_fixture_layout():
    """Ground truth is data/timelines/synthetic/timeline.json, not this file."""
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert doc["version"] == 2
    assert doc["channels"] == [dict(c) for c in CHANNELS]
    assert sum(c["dims"] for c in doc["channels"]) == TOTAL_DIMS


def test_fixture_bin_is_frames_by_total_dims():
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw = (FIXTURE.parent / "timeline.bin").read_bytes()
    assert len(raw) == doc["grid"]["frames"] * TOTAL_DIMS * 4


def _hand_built(frames: int) -> tuple[np.ndarray, dict]:
    """An expected buffer built by explicit indexing, independent of pack()."""
    rng = np.random.default_rng(7)
    stems = rng.random((frames, 4))
    latent = rng.standard_normal((frames, 64))
    nov4 = rng.random(frames)
    nov16 = rng.random(frames)
    act = rng.random(frames)
    rt = rng.random(frames) * 100.0
    rs = rng.random(frames)

    expected = np.zeros((frames, TOTAL_DIMS), dtype=np.float32)
    for f in range(frames):
        for d in range(4):
            expected[f, 0 + d] = stems[f, d]
        for d in range(64):
            expected[f, 4 + d] = latent[f, d]
        expected[f, 68] = nov4[f]
        expected[f, 69] = nov16[f]
        expected[f, 70] = act[f]
        expected[f, 71] = rt[f]
        expected[f, 72] = rs[f]

    channels = {
        "stems": stems,
        "latent": latent,
        "novelty4": nov4,
        "novelty16": nov16,
        "actChorus": act,
        "recurTime": rt,
        "recurStr": rs,
    }
    return expected, channels


def test_pack_matches_hand_built_buffer():
    frames = 37
    expected, channels = _hand_built(frames)
    assert np.array_equal(pack(frames, channels), expected)


def test_bin_round_trip_is_little_endian_row_major(tmp_path):
    frames = 37
    expected, channels = _hand_built(frames)
    tl = Timeline(
        track_id="unit",
        duration=frames * 0.1,
        sample_rate=48000,
        hop_seconds=0.1,
        frames=frames,
        beats=np.array([0.0, 0.5, 1.0]),
        downbeats=np.array([0.0, 2.0]),
        tempo=120.0,
        segments=[{"start": 0.0, "end": 3.7, "label": "intro", "confidence": 0.5}],
        data=pack(frames, channels),
        segment_similarity=np.array([[1.0]]),
    )
    write_timeline(tl, tmp_path)

    raw = (tmp_path / "timeline.bin").read_bytes()
    assert len(raw) == frames * TOTAL_DIMS * 4
    flat = np.frombuffer(raw, dtype="<f4")
    assert np.array_equal(flat, expected.reshape(-1))  # row-major
    # explicit index arithmetic, the way the web loader will do it
    off, dims = channel_index("latent")
    for f in (0, 5, frames - 1):
        for d in (0, 63):
            assert flat[f * TOTAL_DIMS + off + d] == expected[f, off + d]
    assert dims == 64

    back = read_timeline(tmp_path)
    assert back.frames == frames
    assert np.array_equal(back.data, expected)
    assert back.channel("stems").shape == (frames, 4)


def test_manifest_key_order_and_optional_similarity():
    tl = Timeline(
        track_id="t",
        duration=1.0,
        sample_rate=48000,
        hop_seconds=0.1,
        frames=10,
        beats=np.array([0.0]),
        downbeats=np.array([0.0]),
        tempo=120.0,
        segments=[],
        data=np.zeros((10, TOTAL_DIMS), dtype=np.float32),
    )
    doc = manifest_dict(tl)
    assert list(doc) == ["version", "track", "grid", "beats", "downbeats", "tempo", "segments", "channels"]
    # `segmentSimilarity` and `events` are the two optional keys; everything else
    # appears in the fixture's order.
    optional = {"segmentSimilarity", "events"}
    fixture_keys = list(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert [k for k in fixture_keys if k not in optional] == list(doc)


def test_pack_rejects_wrong_width():
    with pytest.raises(ValueError):
        pack(4, {"stems": np.zeros((4, 3))})


def test_pack_holds_last_value_for_short_channels_and_scrubs_nan():
    frames = 6
    short = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, np.nan, np.inf, 8.0]])
    out = pack(frames, {"stems": short})
    assert out[1, 1] == 0.0 and out[1, 2] == 0.0
    assert np.array_equal(out[5, :4], out[1, :4])
