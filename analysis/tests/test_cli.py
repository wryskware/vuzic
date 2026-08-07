"""Light-deps-only guarantees: the package imports, the CLI runs, models are lazy."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import terrarium_analysis
from terrarium_analysis.cli import build_parser, main
from terrarium_analysis.optional import MissingDependency, have, require

HEAVY_MODULES = ("torch", "muq", "allin1", "allin1_infer", "beat_this")


def _module_scope_imports(node: ast.AST):
    """Imports that run at import time — function bodies are the lazy escape hatch."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.Import):
            for alias in child.names:
                yield alias.name
        elif isinstance(child, ast.ImportFrom):
            if child.level == 0 and child.module:
                yield child.module
        else:
            yield from _module_scope_imports(child)


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "terrarium-analyze" in out
    assert "--skip" in out
    assert "73 dims" in out


def test_no_heavy_module_is_imported_by_importing_the_package():
    # Only meaningful on a machine where the extras ARE installed, so it is backed
    # by the static scan below.
    for mod in HEAVY_MODULES:
        assert mod not in sys.modules


def test_no_heavy_module_is_imported_at_module_scope_anywhere_in_the_package():
    """The lazy-import contract, checked against source rather than sys.modules.

    The extras are not installed in CI, so importing the package could never have
    pulled them in; this asserts the property that actually matters.
    """
    root = Path(terrarium_analysis.__file__).parent
    offenders = [
        f"{path.relative_to(root)}: {name}"
        for path in sorted(root.rglob("*.py"))
        for name in _module_scope_imports(ast.parse(path.read_text(encoding="utf-8")))
        if name.split(".")[0] in HEAVY_MODULES
    ]
    assert offenders == []


def test_skip_choices_cover_every_optional_stage():
    p = build_parser()
    action = next(a for a in p._actions if a.dest == "skip")
    assert set(action.choices) == {
        "beats",
        "structure",
        "stems",
        "events",
        "character",
        "recurrence",
    }


@pytest.mark.parametrize("module", ["allin1_infer", "allin1"])
def test_missing_dependency_message_quotes_the_install_and_wsl2_guidance(module):
    """Both spellings must give the same, installable advice.

    The stage prefers `allin1_infer`; `allin1` stays mapped so an old traceback
    still points at a package that exists. `all-in-one-fix` no longer has files
    on PyPI — it was renamed to `all-in-one-infer` — so naming it here would
    hand the user a command that cannot succeed.
    """
    if have(module):
        pytest.skip(f"{module} is installed here; there is no missing-dep message to check")
    with pytest.raises(MissingDependency) as exc:
        require(module)
    msg = str(exc.value)
    assert "uv pip install all-in-one-infer" in msg
    assert "WSL2" in msg
    assert "NATTEN" in msg


def test_missing_dependency_message_for_each_optional_stage():
    for mod, needle in (
        ("beat_this.inference", "beat-this"),
        ("muq", "uv pip install muq torch"),
    ):
        if have(mod):
            continue
        with pytest.raises(MissingDependency) as exc:
            require(mod)
        assert needle in str(exc.value)


def test_latent_dims_is_rejected_before_any_stage_runs(capsys):
    """pack() would raise only after MuQ inference, discarding the whole run."""
    with pytest.raises(SystemExit) as exc:
        main(["in.wav", "-o", "out", "--latent-dims", "32"])
    assert exc.value.code == 2
    assert "--latent-dims must be 64" in capsys.readouterr().err


def test_parser_defaults():
    args = build_parser().parse_args(["in.wav", "-o", "out"])
    assert args.hop == 0.1
    assert args.seed is None  # drawn fresh per run
    assert args.skip == []
    assert args.muq_layer == 6
