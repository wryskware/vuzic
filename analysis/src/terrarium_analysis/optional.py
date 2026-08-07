"""Lazy loading of the heavy, optional model dependencies.

Nothing here imports torch or a model package at module scope. `require()` is
called from inside a stage's `run()`, so a light-only environment can still
import the package, print CLI help, and run every stage that does not need a
model.
"""

from __future__ import annotations

import importlib
from typing import Any

WSL2_NOTE = (
    "Windows note (docs/plan.md, 'Windows gotcha'): the ORIGINAL `allin1` package\n"
    "  needs NATTEN's pre-0.17 C++ API (removed upstream; only ever a source build\n"
    "  on Windows) and an ageing madmom that does not build on Python 3.10+.\n"
    "  `all-in-one-infer` — the renamed, maintained `all-in-one-fix` fork — drops\n"
    "  both: neighborhood attention is reimplemented in pure PyTorch, and it uses\n"
    "  madmom-infer / demucs-infer. That is what this stage prefers.\n"
    "  The tested path is still WSL2 (Ubuntu), because analysis is offline and\n"
    "  runs once per track, so the round trip costs nothing at runtime:\n"
    "      wsl -d Ubuntu\n"
    "      cd ~/lmt/analysis   # an ext4 copy; /mnt/c I/O is slow\n"
    "      uv venv --python 3.10 && uv pip install -e '.[dev,beats,structure,character]'"
)

_ALLIN1_SPEC: dict[str, Any] = {
    "feature": "structure (boundaries, labels, 100 fps activations) and demucs stems",
    "install": "uv pip install all-in-one-infer   # maintained All-In-One fork, any torch 2.x",
    "extra": "structure",
    "notes": WSL2_NOTE
    + "\n  Already have demucs stems elsewhere? Point --stems-dir at the folder\n"
    "  containing bass/drums/vocals/other.wav and --skip structure.",
}

_SPECS: dict[str, dict[str, Any]] = {
    "beat_this": {
        "feature": "beat / downbeat tracking (Beat This!, the master clock)",
        "install": "uv pip install beat-this",
        "extra": "beats",
        "notes": (
            "Pulls torch. MIT code and weights; the ~78 MB checkpoint downloads on\n"
            "  first use. CPU is fine. Skip this stage with --skip beats (a synthetic\n"
            "  grid is emitted instead and the beat channel loses its meaning)."
        ),
    },
    # Both spellings map to the same advice: `allin1_infer` is what the stage
    # loads first, `allin1` is the original package it falls back to.
    "allin1_infer": _ALLIN1_SPEC,
    "allin1": _ALLIN1_SPEC,
    "muq": {
        "feature": "the character stream (MuQ layer-6 embeddings)",
        "install": "uv pip install muq torch",
        "extra": "character",
        "notes": (
            "MuQ-large-msd-iter is ~310M params; code MIT, weights CC-BY-NC-4.0\n"
            "  (accepted for this project). Strict 24 kHz input, fp32 only (fp16 NaNs).\n"
            "  Minutes on CPU for a 4-minute track, seconds on a GPU."
        ),
    },
    "torch": {
        "feature": "model inference",
        "install": "uv pip install torch",
        "extra": "character",
        "notes": "Install the CUDA build from https://pytorch.org if a GPU is present.",
    },
}


class MissingDependency(RuntimeError):
    """Raised when an optional model dependency is needed but absent."""


def _message(top: str, module: str, exc: BaseException) -> str:
    spec = _SPECS.get(top, {})
    feature = spec.get("feature", f"the '{top}' stage")
    install = spec.get("install", f"uv pip install {top}")
    extra = spec.get("extra")
    lines = [
        f"'{module}' is required for {feature}, and is not importable.",
        f"  underlying error: {type(exc).__name__}: {exc}",
        "",
        "  install:",
        f"      {install}",
    ]
    if extra:
        lines.append(f"      (equivalently: uv pip install -e '.[{extra}]' from analysis/)")
    notes = spec.get("notes")
    if notes:
        lines += ["", "  " + notes]
    lines += [
        "",
        "  Heavy dependencies are deliberately optional: the package imports and the",
        "  CLI runs with only numpy/scipy/librosa/soundfile/matplotlib installed.",
    ]
    return "\n".join(lines)


def require(module: str, attr: str | None = None) -> Any:
    """Import an optional dependency, or raise with the install instructions.

    `module` may be dotted (``beat_this.inference``); the advice is looked up
    from its top-level package.
    """
    top = module.split(".", 1)[0]
    try:
        mod = importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - some model packages raise SystemExit/OSError
        raise MissingDependency(_message(top, module, exc)) from exc
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise MissingDependency(
            f"'{module}.{attr}' not found — the installed version of '{top}' is not the "
            f"expected one.\n{_message(top, module, exc)}"
        ) from exc


def have(module: str) -> bool:
    """True if `module` can be imported, without raising if it cannot."""
    try:
        importlib.import_module(module)
    except BaseException:  # noqa: BLE001
        return False
    return True
