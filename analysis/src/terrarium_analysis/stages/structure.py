"""Stage 3a — structure and stems from All-In-One.

Yields boundaries, functional labels, the 100 fps activation curves (which
plan.md values above the hard labels: a label is a step function and steps look
like glitches), and the demucs stems it separated on the way. The stems are the
input to stage 3b.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..context import Context, log
from ..dsp import pool_to_grid, segment_label_curve
from ..optional import have, require
from ..timeline import STEM_NAMES

NAME = "structure"

ACTIVATION_FPS = 100.0  # allin1's spectrogram fps is literally 44100/441

# In package-preference order. `allin1_infer` (PyPI `all-in-one-infer`, the
# renamed `all-in-one-fix`) reimplements NATTEN's neighborhood attention in pure
# PyTorch and depends on madmom-infer / demucs-infer, so it installs against any
# torch 2.x — including the cu128 builds a Blackwell GPU needs. The original
# `allin1` calls `natten.functional.natten1dqkrpb`, an API deleted in NATTEN
# 0.17, and drags in a madmom that does not build on Python 3.10+.
PACKAGES = ("allin1_infer", "allin1")

# Attribute names holding the functional-label vocabulary, in lookup order.
LABEL_ATTRS = ("HARMONIX_LABELS", "LABELS")


def _load_allin1():
    """Import the first available All-In-One package, or raise with install help."""
    for name in PACKAGES:
        if have(name):
            return require(name)
    return require(PACKAGES[0])  # raises MissingDependency with the install command


def _label_vocabulary(allin1) -> tuple[str, ...] | None:
    """The installed package's label order, or None if it does not expose one.

    Never guessed: this order indexes rows of the activation matrix, and a wrong
    guess silently returns some other label's curve as actChorus. Any 10-label
    ordering passes the shape check, so nothing downstream would look broken.
    """
    roots = [allin1]
    for sub in ("config", "postprocessing", "typings"):
        try:
            roots.append(__import__(f"{allin1.__name__}.{sub}", fromlist=[sub]))
        except Exception:  # noqa: BLE001
            continue
    for mod in roots:
        for attr in LABEL_ATTRS:
            labels = getattr(mod, attr, None)
            if labels:
                return tuple(str(x) for x in labels)
    return None


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _label_activations(acts: dict, labels: tuple[str, ...] | None) -> np.ndarray | None:
    """Return (n_labels, T); All-In-One's orientation has varied between versions."""
    raw = acts.get("label")
    if raw is None or labels is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == len(labels):
        return arr
    if arr.shape[1] == len(labels):
        return arr.T
    return None


def _find_stems(demix_dir: Path, stem_name: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    candidates = list(demix_dir.rglob(f"{stem_name}/*.wav")) + list(demix_dir.rglob("*.wav"))
    for path in candidates:
        key = path.stem.lower()
        if key in STEM_NAMES and key not in found:
            found[key] = path
    return found


def _resolve_device(name: str) -> str:
    """A concrete device string.

    `None` is not an option here: All-In-One's demix step shells out to
    `demucs.separate --device {device}`, so a None would arrive as the literal
    string "None" and fail inside a subprocess, several minutes in.
    """
    if name != "auto":
        return name
    try:
        torch = require("torch")
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    """Pin the RNGs All-In-One's demucs pass reads from.

    demucs' `apply_model` defaults to `shifts=1`, which draws a random time
    offset from the **unseeded stdlib `random` module** (demucs-infer documents
    this and tells callers to seed immediately before the call). All-In-One does
    not pass `shifts`, so without this the four stem WAVs — and therefore the
    stem curves, and All-In-One's own activations, which are computed from the
    demixed audio — change on every run. Measured before this seeding: stems
    max |diff| 0.28, actChorus max |diff| 0.086 between two same-seed runs.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        torch = require("torch")
    except Exception:  # noqa: BLE001
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(ctx: Context) -> None:
    allin1 = _load_allin1()
    _seed_everything(ctx.cfg.seed)
    demix_dir = ctx.cfg.out_dir / "demix"
    spec_dir = ctx.cfg.out_dir / "spec"
    device = _resolve_device(ctx.cfg.device)
    log.info(
        "structure: %s on %s, device %s (this also runs demucs)",
        allin1.__name__,
        ctx.wav_path.name,
        device,
    )

    result = allin1.analyze(
        paths=str(ctx.wav_path),
        out_dir=str(ctx.cfg.out_dir / "allin1"),
        demix_dir=str(demix_dir),
        spec_dir=str(spec_dir),
        device=device,
        include_activations=True,
        keep_byproducts=ctx.cfg.keep_byproducts,
    )
    if isinstance(result, (list, tuple)):
        result = result[0]

    labels = _label_vocabulary(allin1)
    acts = _as_dict(getattr(result, "activations", None))
    label_acts = _label_activations(acts, labels)
    if labels is None:
        ctx.note(
            "All-In-One exposes no LABELS vocabulary; activation rows cannot be "
            "matched to labels, so confidences and actChorus fall back to the "
            "hard segment labels"
        )

    segments = []
    for seg in getattr(result, "segments", []) or []:
        start = float(getattr(seg, "start", seg["start"] if isinstance(seg, dict) else 0.0))
        end = float(getattr(seg, "end", seg["end"] if isinstance(seg, dict) else 0.0))
        label = str(getattr(seg, "label", seg["label"] if isinstance(seg, dict) else "?"))
        conf = 1.0
        if label_acts is not None and label in labels:
            row = label_acts[labels.index(label)]
            a = int(start * ACTIVATION_FPS)
            b = max(a + 1, int(end * ACTIVATION_FPS))
            window = row[a : min(b, row.shape[0])]
            if window.size:
                conf = float(np.clip(window.mean(), 0.0, 1.0))
        segments.append({"start": start, "end": end, "label": label, "confidence": conf})
    ctx.segments = segments

    if label_acts is not None and "chorus" in labels:
        row = label_acts[labels.index("chorus")]
        # Mean-pooled, not point-sampled: 100 fps -> 10 fps by interpolation
        # keeps 1 frame in 10 and aliases the other 9 into it, which shows up as
        # a fuzzy band on this curve. plan.md Decision 1 wants the *graded*
        # activation, so the decimation has to average rather than decimate.
        ctx.act_chorus = np.clip(
            pool_to_grid(row, ACTIVATION_FPS, ctx.times, ctx.cfg.hop_seconds)[:, 0], 0.0, 1.0
        )
    elif segments:
        ctx.act_chorus = segment_label_curve(segments, ctx.times, "chorus")
        ctx.note("All-In-One returned no label activations; actChorus derived from labels")

    if ctx.beats.size == 0:
        ctx.beats = np.asarray(getattr(result, "beats", []) or [], dtype=np.float64)
        ctx.downbeats = np.asarray(getattr(result, "downbeats", []) or [], dtype=np.float64)
        ctx.tempo = float(getattr(result, "bpm", 0.0) or 0.0)
        if ctx.beats.size:
            ctx.note("using All-In-One beats (Beat This! stage was skipped or empty)")
    else:
        a1_beats = np.asarray(getattr(result, "beats", []) or [], dtype=np.float64)
        if a1_beats.size:
            offsets = np.abs(
                ctx.beats[:, None] - a1_beats[None, :]
            ).min(axis=1) if a1_beats.size else np.zeros(0)
            log.info(
                "structure: All-In-One beats kept as reference only "
                "(median |offset| from Beat This! = %.3f s over %d beats)",
                float(np.median(offsets)) if offsets.size else float("nan"),
                ctx.beats.size,
            )

    ctx.stem_paths = _find_stems(demix_dir, Path(ctx.wav_path).stem)
    missing = [s for s in STEM_NAMES if s not in ctx.stem_paths]
    if missing:
        ctx.note(f"demucs stems not found for: {', '.join(missing)} (looked under {demix_dir})")
    log.info(
        "structure: %d segments, %d stems, activations=%s",
        len(ctx.segments),
        len(ctx.stem_paths),
        "yes" if label_acts is not None else "no",
    )
