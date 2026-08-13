"""The HTTP surface, minus the pipeline.

Nothing here runs an analysis: the pipeline has its own tests and a real job
needs a GPU and a minute. What is worth asserting is everything around it —
the naming rules the browser's `?track=` regex depends on, the upload gate, the
served-file whitelist, and that a listing carries the fields the cache keys on.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="install the `server` extra")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import terrarium_analysis.server as server_module  # noqa: E402
from terrarium_analysis.export_jobs import ExportSettings  # noqa: E402
from terrarium_analysis.server import (  # noqa: E402
    SERVED_FILES,
    Job,
    StageWatcher,
    create_app,
    sniff_audio,
    slugify,
    unique_slug,
)

WAV_HEAD = b"RIFF\x24\x08\x00\x00WAVEfmt "


def export_settings(tmp_path):
    web_dir = tmp_path / "web"
    worker = web_dir / "dist-export" / "worker.js"
    worker.parent.mkdir(parents=True)
    worker.write_text("// fixture worker", encoding="utf-8")
    node = tmp_path / "node.exe"
    ffmpeg = tmp_path / "ffmpeg.exe"
    node.write_bytes(b"node")
    ffmpeg.write_bytes(b"ffmpeg")
    return ExportSettings(
        repo_root=tmp_path,
        web_dir=web_dir,
        worker_path=worker,
        node_executable=node,
        ffmpeg_executable=ffmpeg,
        output_dir=tmp_path / "exports",
    )


def export_recipe(track_id, track_version, renderer_build):
    return {
        "version": 2,
        "rendererBuild": renderer_build,
        "track": {"id": track_id, "contentVersion": track_version},
        "sim": "plife",
        "seed": 42,
        "seedPinned": True,
        "simulation": {},
        "modulationBase": [0.5],
        "modulation": {},
        "impulses": {},
        "render": {},
        "particleBudget": 1000,
        "presentation": {"mode": "single", "autoAdvance": False},
        "output": {
            "profile": "hdr10-1080p120",
            "encoder": "av1_nvenc",
            "paperWhiteNits": 203,
            "masteringPeakNits": 1000,
        },
    }


def write_track(root, name, *, frames=3, duration=0.3, title=None, audio=True):
    d = root / name
    d.mkdir(parents=True)
    manifest = {
        "version": 2,
        "track": {"id": title or name, "duration": duration, "sampleRate": 48000},
        "grid": {"hopSeconds": 0.1, "frames": frames},
        "beats": [],
        "downbeats": [],
        "tempo": 120.0,
        "segments": [],
        "channels": [{"name": "stems", "dims": 73, "offset": 0}],
        "events": [{"t": 0.1, "kind": "kick", "strength": 1.0}],
    }
    (d / "timeline.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "timeline.bin").write_bytes(np.zeros(frames * 73, dtype="<f4").tobytes())
    if audio:
        (d / "audio.wav").write_bytes(WAV_HEAD + b"\x00" * 32)
    return d


@pytest.fixture
def client(tmp_path):
    write_track(tmp_path, "fixture-track", title="Fixture Track")
    with TestClient(create_app(tmp_path)) as c:
        yield c


def test_slugify_produces_a_valid_track_param():
    # The web side validates ?track= against [A-Za-z0-9._-]+; anything this
    # function emits has to survive that or the track is unreachable by URL.
    assert slugify("Pink Loop (2).wav") == "pink-loop-2"
    assert slugify("  ??? .mp3") == "track"
    assert slugify("Free Fall (Remastered).wav") == "free-fall-remastered"


def test_unique_slug_steps_around_an_existing_track(tmp_path):
    write_track(tmp_path, "song")
    assert unique_slug(tmp_path, "song") == "song-2"
    write_track(tmp_path, "song-2")
    assert unique_slug(tmp_path, "song") == "song-3"


@pytest.mark.parametrize(
    ("head", "suffix", "ok"),
    [
        (WAV_HEAD, ".wav", True),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00", ".mp3", True),
        (b"\xff\xfb\x90\x00", ".mp3", True),  # tagless MP3, raw frame sync
        (WAV_HEAD, ".exe", False),  # right bytes, wrong extension
        (b"MZ\x90\x00" + b"\x00" * 12, ".wav", False),  # right extension, wrong bytes
        (b"", ".wav", False),
    ],
)
def test_sniff_audio(head, suffix, ok):
    assert sniff_audio(head, suffix) is ok


def test_tracks_lists_the_fields_the_cache_keys_on(client):
    tracks = client.get("/tracks").json()["tracks"]
    assert [t["id"] for t in tracks] == ["fixture-track"]
    t = tracks[0]
    assert t["title"] == "Fixture Track"  # the manifest's id, not the directory
    assert t["duration"] == pytest.approx(0.3)
    assert t["hasAudio"] is True
    assert len(t["version"]) == 16


def test_version_changes_only_when_the_timeline_content_changes(client, tmp_path):
    first = client.get("/tracks").json()["tracks"][0]["version"]
    # A touch is not a change: the fingerprint memoises, but the value behind it
    # is a content hash, so re-running the pipeline to the same bytes is a no-op.
    (tmp_path / "fixture-track" / "timeline.bin").write_bytes(
        (tmp_path / "fixture-track" / "timeline.bin").read_bytes()
    )
    assert client.get("/tracks").json()["tracks"][0]["version"] == first
    (tmp_path / "fixture-track" / "timeline.bin").write_bytes(b"\x00" * 4)
    assert client.get("/tracks").json()["tracks"][0]["version"] != first


def test_an_unfinished_directory_is_not_a_track(client, tmp_path):
    (tmp_path / "claimed-but-empty").mkdir()
    assert [t["id"] for t in client.get("/tracks").json()["tracks"]] == ["fixture-track"]


@pytest.mark.parametrize("name", sorted(SERVED_FILES))
def test_served_files_carry_their_media_type(client, name):
    res = client.get(f"/tracks/fixture-track/{name}")
    if name == "run.json":  # the fixture has no run.json
        assert res.status_code == 404
        return
    assert res.status_code == 200
    assert res.headers["content-type"].split(";")[0] == SERVED_FILES[name]


def test_only_whitelisted_files_are_served(client):
    assert client.get("/tracks/fixture-track/embedding.bin").status_code == 404
    assert client.get("/tracks/..%2F..%2Fsecrets/timeline.json").status_code == 404
    assert client.get("/tracks/nope/timeline.json").status_code == 404


def test_analyze_rejects_a_non_audio_upload(client, tmp_path):
    res = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415
    res = client.post("/analyze", files={"file": ("evil.wav", b"MZ" + b"\x00" * 20, "audio/wav")})
    assert res.status_code == 415
    # And nothing was claimed on disk for either.
    assert [t["id"] for t in client.get("/tracks").json()["tracks"]] == ["fixture-track"]


def test_stage_watcher_reads_progress_off_the_pipeline_log():
    job = Job(id="j", track_id="t", title="T")
    watcher = StageWatcher(job)
    logger = logging.getLogger("terrarium.test-watcher")
    logger.setLevel(logging.INFO)
    logger.addHandler(watcher)
    try:
        logger.info("beats: 440 beats, 132 downbeats, tempo 125.00 BPM")
        assert job.stage == "beats"
        assert "440 beats" in job.message
        first = job.progress
        logger.info("character: MuQ layer 6 on cuda")
        assert job.stage == "character"
        assert job.progress > first
        # A line that is not a stage announcement still updates the message.
        logger.info("wrote timeline.json")
        assert job.stage == "character"
        assert job.message == "wrote timeline.json"
    finally:
        logger.removeHandler(watcher)


def test_stage_watcher_does_not_shadow_the_handler_lock():
    """The deadlock that ate the first live run.

    `logging.Handler` owns `self.lock` and `Handler.handle` acquires it around
    `emit`. A watcher that stored an application lock under that name had
    `handle` take it and `emit` take it again — a non-reentrant self-deadlock on
    the very first line the pipeline logged, with the server's event loop stuck
    behind it. Assert the shape, not the symptom: the handler's lock must still
    be the reentrant one `createLock` made.
    """
    watcher = StageWatcher(Job(id="j", track_id="t", title="T"))
    assert isinstance(watcher.lock, type(threading.RLock()))
    watcher.acquire()
    watcher.acquire()  # would hang on a plain Lock
    watcher.release()
    watcher.release()


def test_unknown_job_is_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_cors_allows_a_vite_dev_origin(client):
    res = client.get("/tracks", headers={"Origin": "http://localhost:5175"})
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5175"
    res = client.get("/tracks", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in res.headers


def test_export_capabilities_are_probed_once(monkeypatch, tmp_path):
    write_track(tmp_path, "fixture-track")
    settings = export_settings(tmp_path)
    calls = []

    def probe(actual):
        calls.append(actual)
        return {
            "available": True,
            "profiles": ["hdr10-1080p120"],
            "encoders": ["av1_nvenc"],
            "rendererBuild": actual.renderer_build(),
            "transport": "sdr-rgba8-av1-debug",
            "reason": "",
        }

    monkeypatch.setattr(server_module, "probe_export_capabilities", probe)
    with TestClient(create_app(tmp_path, export_settings=settings)) as export_client:
        first = export_client.get("/exports/capabilities")
        second = export_client.get("/exports/capabilities")
    assert first.status_code == second.status_code == 200
    assert first.json()["rendererBuild"] == settings.renderer_build()
    assert calls == [settings]


def test_export_job_runs_with_server_owned_paths_and_downloads(monkeypatch, tmp_path):
    write_track(tmp_path, "fixture-track", duration=0.3)
    settings = export_settings(tmp_path)
    renderer_build = settings.renderer_build()

    monkeypatch.setattr(
        server_module,
        "probe_export_capabilities",
        lambda _settings: {
            "available": True,
            "profiles": ["hdr10-1080p120"],
            "encoders": ["av1_nvenc"],
            "rendererBuild": renderer_build,
            "transport": "sdr-rgba8-av1-debug",
            "reason": "",
        },
    )

    requests = []

    def run_worker(actual_settings, request_path, diagnostic_path, on_message):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        requests.append((actual_settings, request, diagnostic_path))
        on_message({"type": "ready", "adapter": "Fixture GPU", "transport": "fixture"})
        on_message({"type": "progress", "stage": "render", "frame": 18, "frames": 36})
        output = tmp_path / "exports" / request["output"]["path"].split("\\")[-1]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture-mp4")
        return {
            "type": "result",
            "frames": 36,
            "width": 1920,
            "height": 1080,
            "audio": True,
            "transport": "sdr-rgba8-av1-debug",
        }

    monkeypatch.setattr(server_module, "run_export_worker", run_worker)
    with TestClient(create_app(tmp_path, export_settings=settings)) as export_client:
        track = export_client.get("/tracks").json()["tracks"][0]
        response = export_client.post(
            "/exports",
            json={
                "trackId": "fixture-track",
                "recipe": export_recipe("fixture-track", track["version"], renderer_build),
                "range": {"startSeconds": 0, "durationSeconds": 0.3},
            },
        )
        assert response.status_code == 200, response.text
        queued = response.json()
        assert queued["kind"] == "export"
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            completed = export_client.get(f"/jobs/{queued['jobId']}").json()
            if completed["status"] in {"done", "error"}:
                break
            time.sleep(0.01)
        assert completed["status"] == "done", completed
        export_id = completed["exportId"]
        assert export_client.get(f"/exports/{export_id}").json() == completed
        download = export_client.get(completed["downloadUrl"])

    assert download.status_code == 200
    assert download.content == b"fixture-mp4"
    assert completed["audio"] is True
    assert completed["duration"] == pytest.approx(0.3)
    actual_settings, request, diagnostic_path = requests[0]
    assert actual_settings == settings
    assert request["runtime"]["ffmpegExecutable"] == str(settings.ffmpeg_executable)
    assert request["audioPath"] == str((tmp_path / "fixture-track" / "audio.wav").resolve())
    assert diagnostic_path.parent.parent == tmp_path / ".work"


def test_export_rejects_stale_track_or_renderer(monkeypatch, tmp_path):
    write_track(tmp_path, "fixture-track")
    settings = export_settings(tmp_path)
    renderer_build = settings.renderer_build()
    monkeypatch.setattr(
        server_module,
        "probe_export_capabilities",
        lambda _settings: {
            "available": True,
            "profiles": ["hdr10-1080p120"],
            "encoders": ["av1_nvenc"],
            "rendererBuild": renderer_build,
            "reason": "",
        },
    )
    with TestClient(create_app(tmp_path, export_settings=settings)) as export_client:
        track = export_client.get("/tracks").json()["tracks"][0]
        stale_track = export_recipe("fixture-track", "stale", renderer_build)
        stale_build = export_recipe("fixture-track", track["version"], "stale-build")
        assert export_client.post(
            "/exports", json={"trackId": "fixture-track", "recipe": stale_track}
        ).status_code == 409
        assert export_client.post(
            "/exports", json={"trackId": "fixture-track", "recipe": stale_build}
        ).status_code == 409
