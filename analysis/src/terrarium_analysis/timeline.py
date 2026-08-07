"""The timeline contract (format version 2).

The channel table below is the byte-layout half of the contract between
analysis/ and web/. It must stay identical to
`data/timelines/synthetic/timeline.json`, which is the shared fixture both sides
conform to; `tests/test_emitter.py` asserts that against the real file.

The layout is fixed even when a stage is skipped: a skipped stage writes zeros
rather than removing its channel, so `totalDims` is always 73 and the web side
never has to branch on which stages ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VERSION = 2

CHANNELS: tuple[dict, ...] = (
    {"name": "stems", "dims": 4, "offset": 0},  # bass, drums, vocals, other
    {"name": "latent", "dims": 64, "offset": 4},  # MuQ L6, standardized + PCA-64
    {"name": "novelty4", "dims": 1, "offset": 68},
    {"name": "novelty16", "dims": 1, "offset": 69},
    {"name": "actChorus", "dims": 1, "offset": 70},
    {"name": "recurTime", "dims": 1, "offset": 71},
    {"name": "recurStr", "dims": 1, "offset": 72},
)

TOTAL_DIMS = 73
STEM_NAMES = ("bass", "drums", "vocals", "other")

# The sparse transient stream (plan.md Revision 2, item 2). Additive to v2: the
# `events` array is OPTIONAL, and absent or empty means "no events" — a reader
# that predates it is still correct.
EVENT_KINDS = ("kick", "snare", "hat", "bass", "vocal")


def channel_index(name: str) -> tuple[int, int]:
    """(offset, dims) for a channel name."""
    for ch in CHANNELS:
        if ch["name"] == name:
            return ch["offset"], ch["dims"]
    raise KeyError(name)


def _check_layout() -> None:
    cursor = 0
    for ch in CHANNELS:
        if ch["offset"] != cursor:
            raise AssertionError(f"channel {ch['name']} offset {ch['offset']} != {cursor}")
        cursor += ch["dims"]
    if cursor != TOTAL_DIMS:
        raise AssertionError(f"channels sum to {cursor}, expected {TOTAL_DIMS}")


_check_layout()


@dataclass
class Timeline:
    track_id: str
    duration: float
    sample_rate: int
    hop_seconds: float
    frames: int
    beats: np.ndarray
    downbeats: np.ndarray
    tempo: float
    segments: list[dict]
    data: np.ndarray  # (frames, TOTAL_DIMS) float32, row-major
    segment_similarity: np.ndarray | None = None
    events: list[dict] = field(default_factory=list)  # {t, kind, strength}, sorted by t
    extra: dict = field(default_factory=dict)

    def channel(self, name: str) -> np.ndarray:
        off, dims = channel_index(name)
        return self.data[:, off : off + dims]


def pack(frames: int, channels: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble named channel arrays into the (frames, 73) row-major buffer."""
    out = np.zeros((frames, TOTAL_DIMS), dtype=np.float32)
    for name, values in channels.items():
        if values is None:
            continue
        off, dims = channel_index(name)
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] != dims:
            raise ValueError(f"channel '{name}' has {arr.shape[1]} dims, expected {dims}")
        n = min(frames, arr.shape[0])
        out[:n, off : off + dims] = np.nan_to_num(
            arr[:n], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        if n < frames and n > 0:
            out[n:, off : off + dims] = out[n - 1, off : off + dims]
    return out


def manifest_dict(tl: Timeline) -> dict:
    """The JSON manifest, key order matching the fixture."""
    doc: dict = {
        "version": VERSION,
        "track": {
            "id": tl.track_id,
            "duration": round(float(tl.duration), 3),
            "sampleRate": int(tl.sample_rate),
        },
        "grid": {"hopSeconds": round(float(tl.hop_seconds), 6), "frames": int(tl.frames)},
        "beats": [round(float(t), 3) for t in np.asarray(tl.beats).ravel()],
        "downbeats": [round(float(t), 3) for t in np.asarray(tl.downbeats).ravel()],
        "tempo": round(float(tl.tempo), 3),
        "segments": [
            {
                "start": round(float(s["start"]), 3),
                "end": round(float(s["end"]), 3),
                "label": str(s["label"]),
                "confidence": round(float(s.get("confidence", 1.0)), 3),
            }
            for s in tl.segments
        ],
    }
    if tl.segment_similarity is not None and len(tl.segments) > 0:
        sim = np.asarray(tl.segment_similarity, dtype=np.float64)
        doc["segmentSimilarity"] = [[round(float(v), 3) for v in row] for row in sim]
    doc["channels"] = [dict(ch) for ch in CHANNELS]
    if tl.events:
        # Last, matching the shared fixture: the array is the longest thing in the
        # file, and the channel table stays readable at the top of a diff.
        # Milliseconds are the useful resolution; the key is omitted entirely when
        # there are no events, per the optional half of the contract.
        doc["events"] = [
            {
                "t": round(float(e["t"]), 3),
                "kind": str(e["kind"]),
                "strength": round(float(e.get("strength", 1.0)), 3),
            }
            for e in sorted(tl.events, key=lambda e: float(e["t"]))
        ]
    return doc


def write_timeline(tl: Timeline, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "timeline.json"
    bin_path = out / "timeline.bin"
    data = np.ascontiguousarray(tl.data, dtype="<f4")
    if data.shape != (tl.frames, TOTAL_DIMS):
        raise ValueError(f"data shape {data.shape} != ({tl.frames}, {TOTAL_DIMS})")
    json_path.write_text(json.dumps(manifest_dict(tl), indent=2) + "\n", encoding="utf-8")
    bin_path.write_bytes(data.tobytes(order="C"))
    return json_path, bin_path


def read_timeline(path: str | Path) -> Timeline:
    """Read a timeline directory (or its timeline.json) back into memory."""
    p = Path(path)
    json_path = p if p.suffix == ".json" else p / "timeline.json"
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    frames = int(doc["grid"]["frames"])
    total = sum(int(c["dims"]) for c in doc["channels"])
    raw = np.frombuffer((json_path.parent / "timeline.bin").read_bytes(), dtype="<f4")
    if raw.size != frames * total:
        raise ValueError(f"timeline.bin has {raw.size} floats, expected {frames * total}")
    sim = doc.get("segmentSimilarity")
    return Timeline(
        track_id=doc["track"]["id"],
        duration=float(doc["track"]["duration"]),
        sample_rate=int(doc["track"]["sampleRate"]),
        hop_seconds=float(doc["grid"]["hopSeconds"]),
        frames=frames,
        beats=np.asarray(doc.get("beats", []), dtype=np.float64),
        downbeats=np.asarray(doc.get("downbeats", []), dtype=np.float64),
        tempo=float(doc.get("tempo", 0.0)),
        segments=list(doc.get("segments", [])),
        data=np.array(raw, dtype=np.float32).reshape(frames, total),
        segment_similarity=np.asarray(sim, dtype=np.float64) if sim else None,
        events=list(doc.get("events", []) or []),
        extra={"channels": doc["channels"], "version": doc["version"]},
    )
