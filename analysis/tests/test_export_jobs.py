from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from terrarium_analysis.export_jobs import (
    MAX_DIAGNOSTIC_BYTES,
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
