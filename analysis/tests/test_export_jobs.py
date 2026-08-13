from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from terrarium_analysis.export_jobs import (
    CANCELLED_WORKER_EXIT_CODE,
    MAX_DIAGNOSTIC_BYTES,
    CancellationHandle,
    ExportCancelled,
    ExportProcessError,
    ExportSettings,
    _json_lines,
    compact_diagnostic_log,
    probe_export_capabilities,
    run_export_worker,
)


def settings(tmp_path: Path) -> ExportSettings:
    web = tmp_path / "web"
    worker = web / "dist-worker" / "worker.mjs"
    node = tmp_path / "bin" / "node.exe"
    ffmpeg = tmp_path / "bin" / "ffmpeg.exe"
    for path, content in (
        (worker, b"worker bundle"),
        (node, b"node"),
        (ffmpeg, b"ffmpeg"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return ExportSettings.discover(
        tmp_path,
        node=node,
        ffmpeg=ffmpeg,
        worker=worker,
        output_dir=tmp_path / "exports",
    )


def test_settings_are_server_owned_absolute_paths_and_build_hash(tmp_path):
    cfg = settings(tmp_path)
    assert cfg.unavailable_reason() == ""
    assert cfg.node_executable.is_absolute()
    assert cfg.ffmpeg_executable.is_absolute()
    assert cfg.worker_path.is_absolute()
    assert cfg.output_dir.is_absolute()
    assert cfg.renderer_build() == hashlib.sha256(b"worker bundle").hexdigest()[:16]


def test_missing_runtime_reports_a_capability_reason(tmp_path):
    cfg = ExportSettings.discover(
        tmp_path,
        node=tmp_path / "missing-node",
        ffmpeg=tmp_path / "missing-ffmpeg",
    )
    result = probe_export_capabilities(cfg)
    assert result["available"] is False
    assert "Node" in result["reason"]


def test_probe_timeout_becomes_an_unavailable_reason(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )
    result = probe_export_capabilities(cfg)
    assert result["available"] is False
    assert "timed out" in result["reason"]


def test_probe_parses_only_the_worker_ndjson_protocol(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    stdout = "\n".join(
        (
            json.dumps(
                {
                    "type": "ready",
                    "adapter": "Test GPU",
                    "backend": "d3d12",
                    "encoders": ["hevc_nvenc", "av1_nvenc"],
                }
            ),
            json.dumps({"type": "result", "stage": "gate0-probe"}),
        )
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )
    result = probe_export_capabilities(cfg)
    assert result["available"] is True
    assert result["gpu"] == "Test GPU"
    assert result["backend"] == "d3d12"
    assert result["transport"] == "sdr-rgba8-av1-debug"
    assert result["profiles"] == [
        "av1-sdr-debug-2160p120",
        "av1-sdr-debug-1080p120",
    ]


def test_protocol_rejects_human_text_on_stdout():
    with pytest.raises(ExportProcessError, match="non-JSON stdout"):
        _json_lines("modulation: hello")


def test_completed_diagnostic_log_retains_only_a_bounded_tail(tmp_path):
    path = tmp_path / "worker.log"
    content = b"prefix" + b"x" * (MAX_DIAGNOSTIC_BYTES + 100)
    path.write_bytes(content)
    compact_diagnostic_log(path)
    assert path.stat().st_size == MAX_DIAGNOSTIC_BYTES
    assert path.read_bytes() == content[-MAX_DIAGNOSTIC_BYTES:]


def test_worker_supervisor_forwards_progress_and_requires_result(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    lines = iter(
        (
            json.dumps({"type": "ready", "adapter": "Test GPU"}) + "\n",
            json.dumps({"type": "progress", "stage": "render", "frame": 3, "frames": 4})
            + "\n",
            json.dumps({"type": "result", "frames": 4, "path": "out.mp4"}) + "\n",
        )
    )

    class FakeProcess:
        stdout = lines

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    seen = []
    result = run_export_worker(cfg, request, tmp_path / "worker.log", seen.append)
    assert [message["type"] for message in seen] == ["ready", "progress", "result"]
    assert result["frames"] == 4


def test_worker_supervisor_surfaces_structured_failure(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    class FakeProcess:
        stdout = iter((json.dumps({"type": "error", "message": "GPU exploded"}) + "\n",))

        def wait(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(ExportProcessError, match="GPU exploded"):
        run_export_worker(cfg, request, tmp_path / "worker.log", lambda _: None)


def test_a_worker_exit_code_of_130_is_cancellation_not_failure(tmp_path, monkeypatch):
    # The server may not be the party that stopped the tree — an operator can
    # kill it by hand — so the code the worker exits with has to carry the
    # distinction on its own.
    cfg = settings(tmp_path)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    class FakeProcess:
        stdout = iter(
            (json.dumps({"type": "error", "message": "export cancelled by SIGBREAK"}) + "\n",)
        )

        def wait(self):
            return CANCELLED_WORKER_EXIT_CODE

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(ExportCancelled, match="SIGBREAK"):
        run_export_worker(cfg, request, tmp_path / "worker.log", lambda _: None)


def test_a_cancel_before_the_spawn_never_starts_a_worker(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    spawned = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: spawned.append(args) or None
    )
    handle = CancellationHandle()
    handle.cancel()
    with pytest.raises(ExportCancelled, match="before the worker started"):
        run_export_worker(cfg, request, tmp_path / "worker.log", lambda _: None, cancel=handle)
    assert spawned == []


# --- real process trees -------------------------------------------------------
#
# These spawn actual processes, because the whole point of the feature is what
# the operating system does to a *descendant* of the worker, and no fake of
# Popen can be evidence of that. Python stands in for Node: `run_export_worker`
# only knows it is launching `<node_executable> <worker_path> --request <path>`,
# and the interpreter satisfies that shape.

HEARTBEAT_CHILD = """
import sys, time
path = sys.argv[1]
while True:
    with open(path, 'ab') as stream:
        stream.write(b'.')
    time.sleep(0.02)
"""

WORKER_TEMPLATE = """
import json, subprocess, sys, time
{preamble}
index = sys.argv.index('--request')
request = json.loads(open(sys.argv[index + 1], encoding='utf-8').read())
child = subprocess.Popen([sys.executable, request['childScript'], request['heartbeat']])
print(json.dumps({{'type': 'ready', 'child': child.pid}}), flush=True)
time.sleep(120)
"""

IGNORE_SIGNALS = """
import signal
for name in ('SIGINT', 'SIGBREAK', 'SIGTERM'):
    handler = getattr(signal, name, None)
    if handler is not None:
        try:
            signal.signal(handler, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
"""


def process_tree_fixture(tmp_path: Path, preamble: str = "") -> tuple[ExportSettings, Path, Path]:
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    worker = web / "worker.py"
    worker.write_text(WORKER_TEMPLATE.format(preamble=preamble), encoding="utf-8")
    child_script = tmp_path / "child.py"
    child_script.write_text(HEARTBEAT_CHILD, encoding="utf-8")
    heartbeat = tmp_path / "heartbeat.bin"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"childScript": str(child_script), "heartbeat": str(heartbeat)}),
        encoding="utf-8",
    )
    cfg = ExportSettings(
        repo_root=tmp_path,
        web_dir=web,
        worker_path=worker,
        node_executable=Path(sys.executable),
        ffmpeg_executable=Path(sys.executable),
        output_dir=tmp_path / "exports",
    )
    return cfg, request, heartbeat


def run_until_ready(cfg, request, diagnostic, handle) -> tuple[threading.Thread, list, dict]:
    seen: list[dict] = []
    outcome: dict = {}

    def body() -> None:
        try:
            outcome["result"] = run_export_worker(
                cfg, request, diagnostic, seen.append, cancel=handle
            )
        except BaseException as exc:  # noqa: BLE001 — the test asserts on the type
            outcome["error"] = exc

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not seen:
        time.sleep(0.02)
    assert seen, f"worker never reported ready: {outcome}"
    return thread, seen, outcome


def assert_tree_is_dead(heartbeat: Path) -> None:
    """The grandchild proves its own liveness by appending to a file."""

    settled = heartbeat.stat().st_size if heartbeat.exists() else 0
    time.sleep(1.0)
    after = heartbeat.stat().st_size if heartbeat.exists() else 0
    assert after == settled, "a descendant of the cancelled worker is still running"


@pytest.mark.parametrize(
    "preamble,label",
    [("", "cooperative"), (IGNORE_SIGNALS, "force")],
    ids=["cooperative-shutdown", "force-kill-fallback"],
)
def test_cancelling_kills_the_worker_and_every_descendant(tmp_path, preamble, label):
    # Two workers, one indistinguishable from the other at the API: one takes
    # the hint, one ignores every signal it can. Both have to be gone, and so
    # does the child neither of them was asked about.
    cfg, request, heartbeat = process_tree_fixture(tmp_path, preamble)
    handle = CancellationHandle(grace_seconds=2.0)
    thread, seen, outcome = run_until_ready(cfg, request, tmp_path / "worker.log", handle)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not heartbeat.exists():
        time.sleep(0.02)
    assert heartbeat.exists(), f"the {label} worker never started its child"

    handle.cancel()
    thread.join(timeout=30)
    assert not thread.is_alive(), "run_export_worker did not return after cancellation"
    assert isinstance(outcome.get("error"), ExportCancelled), outcome
    assert seen[0]["type"] == "ready"
    assert_tree_is_dead(heartbeat)


def test_a_cancel_racing_the_spawn_still_kills_the_tree(tmp_path):
    # The window this closes: `DELETE` lands while `Popen` is still returning,
    # so `cancel()` sees no process and `attach()` has to be the one that kills.
    cfg, request, heartbeat = process_tree_fixture(tmp_path)
    handle = CancellationHandle(grace_seconds=2.0)
    seen: list[dict] = []
    outcome: dict = {}

    def body() -> None:
        try:
            run_export_worker(cfg, request, tmp_path / "worker.log", seen.append, cancel=handle)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=body, daemon=True)
    # Latched between the pre-spawn check and `attach`, as close to the spawn as
    # a test can get without reaching inside the function.
    thread.start()
    handle.cancel()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), ExportCancelled), outcome
    if heartbeat.exists():
        assert_tree_is_dead(heartbeat)
