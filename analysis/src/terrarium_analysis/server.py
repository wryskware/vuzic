"""`terrarium-server` — the pipeline behind an HTTP API.

The offline pipeline stays offline: this is the same `pipeline.run` the CLI
calls, wrapped so the web workbench can hand it a file and then load the result
the same way it loads a track that shipped in `data/timelines/`. Nothing here is
on the runtime path either — the browser talks to it exactly twice per track
(list, then fetch) and works with it absent.

Shape of the thing:

    POST /analyze            multipart `file=` .wav/.mp3  -> {jobId, trackId}
    GET  /jobs/{jobId}                                    -> {status, stage, ...}
    GET  /tracks                                          -> [{id, title, ...}]
    GET  /tracks/{id}/timeline.json | timeline.bin | audio.wav | run.json

**Polling, not a websocket.** A job is one long CPU/GPU burst with six coarse
transitions in it, the client is a panel that redraws at human speed, and a
poll loop is four lines on both sides with no reconnect story. A websocket
would buy sub-second progress nobody can act on.

The analysis stages need the heavy optional dependencies (torch, allin1_infer,
muq) — see README, "run the model stages under WSL2". Run this server from an
environment that has them, or every model stage is skipped and the timeline it
produces is mostly zeros. `--strict` turns that into a job failure instead,
which is the right default for a server nobody is watching; it is off here only
so the server behaves like the CLI it wraps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import secrets
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .context import STAGE_NAMES, Config
from .export_jobs import (
    EXPORT_PROFILE_TABLE,
    CancellationHandle,
    ExportCancelled,
    ExportProcessError,
    ExportSettings,
    probe_export_capabilities,
    run_export_worker,
)

log = logging.getLogger("terrarium.server")

DEFAULT_PORT = 8765
"""Not 8000: that is the first port every other Python server in the world takes."""

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
"""~25 minutes of 48 kHz stereo WAV. A cap, not a target — the point is that a
stray multi-gigabyte POST cannot fill the disk before anything looks at it."""

ALLOWED_SUFFIXES = {".wav", ".mp3"}

#: Files a track directory is allowed to serve, and what to call them on the way
#: out. Everything else in there (`embedding.bin`, `plots/`, demix byproducts) is
#: for humans reading the repo, so the route is a whitelist rather than a
#: `StaticFiles` mount that would publish whatever happened to be written.
SERVED_FILES = {
    "timeline.json": "application/json",
    "timeline.bin": "application/octet-stream",
    "audio.wav": "audio/wav",
    "run.json": "application/json",
}

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
JobKind = Literal["analysis", "export"]

#: Once a job reaches one of these it never moves again, and every caller —
#: the poll loop, the download route, `DELETE /jobs/{id}` — keys off that.
TERMINAL_STATUSES = frozenset({"done", "error", "cancelled"})


def sniff_audio(head: bytes, suffix: str) -> bool:
    """Is this actually a WAV or an MP3?

    Extension and magic both, because they fail in different directions: the
    extension is what the pipeline dispatches decoding on, and the magic is the
    only thing that stops a renamed .exe from reaching a decoder. Neither is a
    security boundary on its own and this server binds localhost, but a 400 with
    "not audio" is a better answer than a librosa stack trace.
    """
    if suffix not in ALLOWED_SUFFIXES:
        return False
    if suffix == ".wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    # MP3: an ID3v2 tag, or a raw frame sync (11 set bits) for a tagless file.
    return head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def slugify(name: str) -> str:
    """`"Pink Loop (2).wav"` -> `"pink-loop-2"`.

    The result is also a URL path segment and a directory name, and the web side
    validates `?track=` against `[A-Za-z0-9._-]+`, so the character class here is
    the intersection of all three rather than anything prettier.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")
    return slug or "track"


def unique_slug(data_dir: Path, base: str) -> str:
    """A re-upload of the same song gets its own directory rather than silently
    replacing a track the browser may already be holding in its cache."""
    slug, n = base, 2
    while (data_dir / slug).exists():
        slug, n = f"{base}-{n}", n + 1
    return slug


@dataclass
class Job:
    id: str
    track_id: str
    title: str
    kind: JobKind = "analysis"
    export_id: str = ""
    status: JobStatus = "queued"
    stage: str = ""
    message: str = ""
    error: str = ""
    #: 0..1, by stage index. Coarse and honest: the stages are wildly unequal
    #: (structure and character are ~80 % of the wall time between them), so this
    #: is "how far through the list", not "how far through the work".
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    #: Guards the three fields below and the status transitions that read them.
    #: Narrow on purpose: it is only ever held for a few assignments, never
    #: across the worker or a `terminate`, so it cannot invert against the
    #: logging handler that also writes `stage`/`message`.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Latched by `DELETE /jobs/{id}`. A running job that carries this ends
    #: `cancelled` even if the worker happened to succeed in the same instant.
    cancel_requested: bool = False
    #: True once the export body has taken the job off the queue. Before it, a
    #: cancel is a pure queue removal; after it, a cancel must reach a process.
    started: bool = False
    cancel_handle: CancellationHandle | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        value = {
            "jobId": self.id,
            "kind": self.kind,
            "trackId": self.track_id,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "progress": round(self.progress, 3),
        }
        if self.kind == "export":
            value["exportId"] = self.export_id
            value.update(self.result)
        return value


class StageWatcher(logging.Handler):
    """Progress, read off the log the stages already write.

    The alternative was a callback threaded through `pipeline.run` and every
    stage — a runtime concern pushed into offline code whose only caller today
    does not want it. Stages announce themselves as `"<name>: ..."` on the
    `terrarium` logger, which is a contract the plots and the README already
    depend on, so watching it costs nothing and changes nothing.

    **It takes no lock, and must not.** `logging.Handler.handle` already holds
    the handler's own lock across `emit`, so any application lock acquired in
    here inverts against every thread that logs while holding that same lock —
    and `Handler` owns an attribute literally called `self.lock`, so an
    application lock stored under that name is not merely a hazard, it is
    acquired by `handle` and then re-acquired here, which deadlocks the server
    on the first line the pipeline logs. (It did.) Nothing here needs one: these
    are single attribute assignments read by a progress poll, and the worst a
    racing reader can see is a `stage` from one line and a `message` from the
    next.
    """

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # a broken format string must not kill the run
            return
        job, head = self.job, msg.split(":", 1)[0].strip()
        job.message = msg.splitlines()[0][:200]
        if head in STAGE_NAMES:
            job.stage = head
            job.progress = (STAGE_NAMES.index(head) + 1) / (len(STAGE_NAMES) + 1)


@dataclass
class Store:
    data_dir: Path
    strict: bool = False
    jobs: dict[str, Job] = field(default_factory=dict)
    exports: dict[str, Job] = field(default_factory=dict)
    #: `job id -> (snapshotted settings, request path)`, so a cancel arriving
    #: before the worker body runs can remove exactly the snapshots that body
    #: would have removed.
    export_inputs: dict[str, tuple[ExportSettings, Path]] = field(default_factory=dict)
    export_settings: ExportSettings | None = None
    export_capabilities: dict[str, Any] | None = None
    export_capabilities_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Guards the slug claim only — the read-modify-write of "is this directory
    #: taken, take it". Job *fields* are deliberately not under it: they are
    #: written from the worker and from a logging handler, and a lock spanning
    #: those two is how the first version of this file deadlocked.
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: One at a time. Analysis and export both want the whole GPU, so this is the
    #: local GPU lease as well as the analysis queue.
    pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="terrarium-job")
    )
    #: `dir name -> (fingerprint, listing)`. See `track_entry`.
    cache: dict[str, tuple[tuple[int, int, int, int], dict[str, Any]]] = field(default_factory=dict)


def timeline_content_version(manifest_path: Path, binary_path: Path) -> str:
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    h.update(binary_path.read_bytes())
    return h.hexdigest()[:16]


def track_entry(store: Store, d: Path) -> dict[str, Any] | None:
    """One `/tracks` row, or None if the directory is not a finished track.

    `version` is what the browser caches on. It is a hash of the two files that
    define the track's content (`timeline.json` + `timeline.bin`), not an mtime:
    re-running the pipeline with the same seed rewrites both files with identical
    bytes, and a cache that invalidated on that would re-download a multi-megabyte
    track every time someone re-ran the analysis to no effect.

    Hashing is memoised on (size, mtime_ns) of both files, so a `/tracks` call
    against unchanged tracks reads no bytes at all.
    """
    manifest_path, bin_path = d / "timeline.json", d / "timeline.bin"
    try:
        ms, bs = manifest_path.stat(), bin_path.stat()
    except OSError:
        return None

    fingerprint = (ms.st_size, ms.st_mtime_ns, bs.st_size, bs.st_mtime_ns)
    cached = store.cache.get(d.name)
    if cached and cached[0] == fingerprint:
        return cached[1]

    try:
        manifest = json.loads(manifest_path.read_bytes())
        track = manifest["track"]
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("track %s: unreadable timeline.json, skipping", d.name)
        return None

    audio = d / "audio.wav"

    entry = {
        "id": d.name,
        # The manifest's own id is the human title ("Free Fall (Remastered)");
        # the directory name is the URL key. They are deliberately different.
        "title": str(track.get("id") or d.name),
        "duration": float(track.get("duration") or 0.0),
        "frames": int(manifest.get("grid", {}).get("frames") or 0),
        "tempo": float(manifest.get("tempo") or 0.0),
        "events": len(manifest.get("events") or ()),
        "hasAudio": audio.is_file(),
        "version": timeline_content_version(manifest_path, bin_path),
    }
    store.cache[d.name] = (fingerprint, entry)
    return entry


def analyze_job(store: Store, job: Job, src: Path, work_root: Path) -> None:
    """The worker body. Runs on the single-slot pool, off the event loop."""
    from .pipeline import run  # deferred exactly as the CLI defers it (librosa is slow)

    watcher = StageWatcher(job)
    pipeline_log = logging.getLogger("terrarium")
    pipeline_log.addHandler(watcher)
    tmp = Path(tempfile.mkdtemp(prefix=f"{job.track_id}-", dir=work_root))
    try:
        job.status = "running"
        cfg = Config(
            input_path=src,
            out_dir=tmp,
            track_id=job.title,
            seed=secrets.randbelow(2**31),
        )
        run(cfg, strict=store.strict)

        dest = store.data_dir / job.track_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("timeline.json", "timeline.bin", "run.json"):
            shutil.move(str(tmp / name), str(dest / name))
        # An MP3 upload is decoded to `audio.wav` by the audio stage; a WAV one is
        # not, because it already is one. Either way the served name is audio.wav,
        # which is the filename the web clock appends to a timeline base URL.
        produced = tmp / "audio.wav"
        if produced.is_file():
            shutil.move(str(produced), str(dest / "audio.wav"))
        else:
            shutil.move(str(src), str(dest / "audio.wav"))
        # Logged before the fields are set, not after: this logger is a child of
        # `terrarium`, so the watcher is still listening and would otherwise
        # overwrite the tidy final message with its own absolute-path log line.
        log.info("job %s: wrote %s", job.id, dest)
        job.status, job.progress, job.stage = "done", 1.0, "done"
        job.message = f"wrote {dest.name}"
    except Exception as exc:  # noqa: BLE001 — the job owns the failure, the server survives it
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        # Give the slug back if nothing was written into it, so a retry of the
        # same file does not walk up through -2, -3, -4 collecting empty rubble.
        dest = store.data_dir / job.track_id
        if dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
    finally:
        pipeline_log.removeHandler(watcher)
        shutil.rmtree(tmp, ignore_errors=True)
        src.unlink(missing_ok=True)


MAX_EXPORT_RECIPE_BYTES = 8 * 1024 * 1024
EXPORT_PROFILES = set(EXPORT_PROFILE_TABLE)

#: Absolute-luminance bounds, matching `web/src/export/hdr.ts`.  Python is the
#: security boundary for these two, so it enforces them even though TypeScript
#: validates the same recipe twice more downstream.
MIN_PAPER_WHITE_NITS = 80.0
MAX_PAPER_WHITE_NITS = 1000.0
MIN_MASTERING_PEAK_NITS = 100.0
MAX_MASTERING_PEAK_NITS = 10_000.0


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        raise HTTPException(422, f"{name} must be finite and in {minimum}..{maximum}")
    return float(value)


def validate_export_submission(
    payload: Any,
    track: dict[str, Any],
    renderer_build: str,
    available_profiles: Iterable[str] | None = None,
) -> tuple[dict[str, Any], float, float]:
    """Validate the browser-owned envelope before any paths are attached.

    TypeScript remains the complete recipe-schema authority and the worker
    validates it again.  Python validates the security/resource boundary and
    the identity fields it owns: track, build, profile, encoder, and range.
    """

    if not isinstance(payload, dict) or set(payload) - {"trackId", "recipe", "range"}:
        raise HTTPException(422, "export body has unsupported fields")
    track_id = payload.get("trackId")
    if track_id != track["id"]:
        raise HTTPException(409, "trackId does not match the selected track")
    recipe = payload.get("recipe")
    if not isinstance(recipe, dict):
        raise HTTPException(422, "recipe must be an object")
    try:
        encoded = json.dumps(recipe, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"recipe is not finite JSON: {exc}") from exc
    if len(encoded) > MAX_EXPORT_RECIPE_BYTES:
        raise HTTPException(413, f"recipe exceeds {MAX_EXPORT_RECIPE_BYTES} bytes")
    # 6 is current (plife's per-particle `luma` block); 3 (pre-HSLuv palette),
    # 4 and 5 (impulse `wiggle`) are still accepted because `liftExportRecipe`
    # in the Node worker walks an older recipe up to the current schema
    # losslessly on the way in. This layer does not validate the recipe's
    # contents at all — that is `validateExportRecipe`'s job in the worker — so
    # it only has to refuse versions the worker cannot read at all.
    if recipe.get("version") not in (3, 4, 5, 6):
        raise HTTPException(422, "recipe version 3, 4, 5 or 6 is required")
    if recipe.get("rendererBuild") != renderer_build:
        raise HTTPException(409, "renderer build changed; refresh the app and capture again")
    identity = recipe.get("track")
    if not isinstance(identity, dict) or identity.get("id") != track_id:
        raise HTTPException(409, "recipe track does not match trackId")
    content_version = identity.get("contentVersion")
    if content_version != track["version"]:
        raise HTTPException(409, "track content changed; refresh the app and capture again")
    output = recipe.get("output")
    if not isinstance(output, dict):
        raise HTTPException(422, "recipe.output must be an object")
    profile = output.get("profile")
    if profile not in EXPORT_PROFILES:
        raise HTTPException(422, "unsupported export profile")
    if available_profiles is not None and profile not in set(available_profiles):
        # The profile exists in the schema but this renderer cannot serve it —
        # no 10-bit encoder, say.  Refusing here is the whole point of gating
        # capabilities: the alternative is failing minutes into a render.
        raise HTTPException(422, f"profile {profile} is not available on this renderer")
    encoder, _dynamic_range, _transport = EXPORT_PROFILE_TABLE[str(profile)]
    if output.get("encoder") != encoder:
        raise HTTPException(422, f"profile {profile} requires encoder {encoder}")
    paper_white = _number(
        output.get("paperWhiteNits"),
        "recipe.output.paperWhiteNits",
        MIN_PAPER_WHITE_NITS,
        MAX_PAPER_WHITE_NITS,
    )
    mastering_peak = _number(
        output.get("masteringPeakNits"),
        "recipe.output.masteringPeakNits",
        MIN_MASTERING_PEAK_NITS,
        MAX_MASTERING_PEAK_NITS,
    )
    if mastering_peak < paper_white:
        raise HTTPException(
            422, "recipe.output.masteringPeakNits must be at least paperWhiteNits"
        )
    if not isinstance(recipe.get("sim"), str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", recipe["sim"]
    ):
        raise HTTPException(422, "recipe.sim is invalid")
    seed = recipe.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF_FFFF:
        raise HTTPException(422, "recipe.seed must be a uint32")

    range_value = payload.get("range")
    if range_value is None:
        start, duration = 0.0, float(track["duration"])
    else:
        if not isinstance(range_value, dict) or set(range_value) != {"startSeconds", "durationSeconds"}:
            raise HTTPException(422, "range requires startSeconds and durationSeconds")
        start = _number(range_value["startSeconds"], "range.startSeconds", 0, 6 * 60 * 60)
        duration = _number(
            range_value["durationSeconds"], "range.durationSeconds", 1e-9, 6 * 60 * 60
        )
    if start + duration > float(track["duration"]) + 1e-9:
        raise HTTPException(422, "export range extends beyond the track")
    return recipe, start, duration


def snapshot_export_inputs(
    directory: Path,
    settings: ExportSettings,
    private: Path,
    worker_private: Path,
    expected_track_version: str,
    expected_renderer_build: str,
) -> tuple[Path, Path, Path, ExportSettings]:
    """Freeze every mutable file a queued native job will later consume."""

    manifest = private / "timeline.json"
    binary = private / "timeline.bin"
    audio = private / "audio.wav"
    worker_private.mkdir(parents=True, exist_ok=False)
    worker = worker_private / "worker.mjs"
    shutil.copyfile(directory / "timeline.json", manifest)
    shutil.copyfile(directory / "timeline.bin", binary)
    if timeline_content_version(manifest, binary) != expected_track_version:
        raise ValueError("track content changed while the export was queued")
    shutil.copyfile(directory / "audio.wav", audio)
    shutil.copyfile(settings.worker_path, worker)
    job_settings = replace(settings, worker_path=worker)
    if job_settings.renderer_build() != expected_renderer_build:
        raise ValueError("renderer build changed while the export was queued")
    return manifest, binary, audio, job_settings


def discard_export_snapshots(settings: ExportSettings, request_path: Path) -> None:
    """Drop every private snapshot a job took, keeping only its bounded log.

    Shared by the normal terminal path and by cancelling a job that never
    started, so a queued cancel leaves exactly the same residue on disk as a
    completed or failed render does.
    """

    for name in ("request.json", "timeline.json", "timeline.bin", "audio.wav"):
        (request_path.parent / name).unlink(missing_ok=True)
    settings.worker_path.unlink(missing_ok=True)
    try:
        settings.worker_path.parent.rmdir()
        settings.worker_path.parent.parent.rmdir()
    except OSError:
        pass


def cancel_export_job(job: Job, settings: ExportSettings, request_path: Path) -> dict[str, Any]:
    """Cancel one export job.  Idempotent, and safe against natural completion.

    The lock decides one thing: whether this call is the one that flips the job
    out of a live state.  A job that already reached a terminal status wins —
    a render that finished a millisecond before the click keeps its file and
    stays `done` — and everything after that point is cleanup that a second
    caller would simply repeat harmlessly.

    Termination happens *outside* the lock because it blocks for up to the
    cooperative grace period, and nothing else may be blocked behind that.
    """

    with job.lock:
        if job.status in TERMINAL_STATUSES:
            return job.as_dict()
        job.cancel_requested = True
        handle = job.cancel_handle
        never_started = not job.started
        if never_started:
            # The worker body will find `cancel_requested` and return without
            # touching status, so this call owns the whole terminal transition.
            job.status, job.stage = "cancelled", "cancelled"
            job.message = "cancelled before the render started"
        else:
            job.stage, job.message = "cancelling", "stopping the native worker"
    if never_started:
        discard_export_snapshots(settings, request_path)
    elif handle is not None:
        handle.cancel()
    log.info("export job %s: cancelled", job.id)
    return job.as_dict()


def export_job(
    settings: ExportSettings,
    job: Job,
    request_path: Path,
    diagnostic_path: Path,
    output_path: Path,
) -> None:

    def on_message(message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "ready":
            job.stage = "render"
            job.message = f"{message.get('adapter', 'GPU')} · {message.get('transport', 'worker')}"
        elif kind == "progress":
            job.stage = str(message.get("stage") or "render")[:64]
            frame, frames = message.get("frame"), message.get("frames")
            if isinstance(frame, int) and isinstance(frames, int) and frames > 0:
                job.progress = min(max(frame / frames, 0.0), 0.999)
                job.message = f"frame {frame:,} / {frames:,}"
        elif kind == "error":
            job.message = str(message.get("message") or "worker failed")[:200]

    with job.lock:
        if job.cancel_requested:
            # Cancelled while queued: the endpoint already published the
            # terminal state and removed the snapshots, and the worker must
            # never start.  Nothing left for this slot to do.
            return
        job.started = True
        job.cancel_handle = handle = CancellationHandle()
        job.status, job.stage, job.message = "running", "startup", "starting native worker"

    succeeded = False
    cancelled = False
    terminal_error = ""
    try:
        result = run_export_worker(
            settings, request_path, diagnostic_path, on_message, cancel=handle
        )
        if not output_path.is_file():
            raise ExportProcessError("worker reported success but output is missing")
        frames = int(result.get("frames") or 0)
        metadata: dict[str, Any] = {
            "filename": job.result["filename"],
            "byteSize": output_path.stat().st_size,
            "duration": frames / 120,
            "width": int(result.get("width") or 0),
            "height": int(result.get("height") or 0),
            "frameRate": 120,
            "frames": frames,
            "codec": str(result.get("codec") or "av1"),
            "audio": bool(result.get("audio")),
            "transport": str(result.get("transport") or "sdr-rgba8-av1-debug"),
            "downloadUrl": f"/exports/{job.export_id}/download",
        }
        # The colour summary is reported by the worker that chose the encoder
        # arguments, not re-derived here, so a completed job describes the file
        # that was actually written rather than the one this module expected.
        if result.get("dynamicRange") == "hdr10":
            metadata.update(
                {
                    "dynamicRange": "hdr10",
                    "pixelFormat": str(result.get("pixelFormat") or "p010le"),
                    "colorPrimaries": str(result.get("colorPrimaries") or ""),
                    "colorTransfer": str(result.get("colorTransfer") or ""),
                    "colorMatrix": str(result.get("colorMatrix") or ""),
                    "colorRange": str(result.get("colorRange") or ""),
                    "paperWhiteNits": float(result.get("paperWhiteNits") or 0),
                    "masteringPeakNits": float(result.get("masteringPeakNits") or 0),
                }
            )
        else:
            metadata["dynamicRange"] = "sdr"
            metadata["pixelFormat"] = str(result.get("pixelFormat") or "yuv420p")
        job.result.update(metadata)
        job.message = f"wrote {job.result['filename']}"
        succeeded = True
    except ExportCancelled:
        log.info("export job %s: worker stopped for cancellation", job.id)
        cancelled = True
    except Exception as exc:  # noqa: BLE001 — one failed export must not stop the server
        log.exception("export job %s failed", job.id)
        terminal_error = f"{type(exc).__name__}: {exc}"
    finally:
        # The bounded worker log is the only retained private diagnostic. Track,
        # audio, worker, and request snapshots can be large and are never public.
        discard_export_snapshots(settings, request_path)
    # A cancel that lands in the sliver between the worker's last frame and this
    # line still wins: the job the user cancelled must not resolve to a
    # downloadable file, so the finished output goes with it. The other side of
    # that race — cancel arriving after this job is already `done` — is refused
    # by the endpoint, which is why the resolution is unambiguous either way.
    if job.cancel_requested:
        succeeded, cancelled = False, True
    if cancelled:
        output_path.unlink(missing_ok=True)
        Path(f"{output_path}.partial").unlink(missing_ok=True)
    if succeeded:
        job.status, job.stage, job.progress = "done", "done", 1.0
    elif cancelled:
        job.status, job.stage = "cancelled", "cancelled"
        job.message = "cancelled"
        job.result.pop("downloadUrl", None)
    else:
        job.status, job.stage, job.error = "error", "error", terminal_error


def create_app(
    data_dir: Path,
    strict: bool = False,
    export_settings: ExportSettings | None = None,
) -> FastAPI:
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if export_settings is None:
        repo_root = Path(__file__).resolve().parents[3]
        export_settings = ExportSettings.discover(repo_root)
    store = Store(data_dir=data_dir, strict=strict, export_settings=export_settings)
    # Uploads and pipeline byproducts land next to the tracks so the finishing
    # move is a rename rather than a cross-device copy of ~50 MB.
    work_root = data_dir / ".work"
    work_root.mkdir(exist_ok=True)

    app = FastAPI(title="terrarium analysis server", version="1")
    # Dev-only, and deliberately loose within that: vite picks 5174 or 5175 when
    # 5173 is taken, so pinning one origin means the day the port shifts the
    # workbench silently loses the server. Localhost of any port, nothing else.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    def track_dir(track_id: str) -> Path:
        # `..`, absolute paths and separators all die here: the id has to be one
        # of our own directory names, compared against the listing.
        d = data_dir / track_id
        if not re.fullmatch(r"[A-Za-z0-9._-]+", track_id) or not d.is_dir():
            raise HTTPException(404, "no such track")
        return d

    def capabilities_sync() -> dict[str, Any]:
        with store.export_capabilities_lock:
            assert store.export_settings is not None
            current_build = store.export_settings.renderer_build()
            cached_build = str((store.export_capabilities or {}).get("rendererBuild") or "")
            if store.export_capabilities is None or cached_build != current_build:
                store.export_capabilities = probe_export_capabilities(store.export_settings)
            return dict(store.export_capabilities)

    @app.get("/tracks")
    async def list_tracks() -> dict[str, Any]:
        def scan() -> list[dict[str, Any]]:
            out = [
                e
                for d in sorted(data_dir.iterdir())
                if d.is_dir() and not d.name.startswith(".")
                for e in (track_entry(store, d),)
                if e is not None
            ]
            return out

        return {"tracks": await run_in_threadpool(scan)}

    @app.get("/tracks/{track_id}/{name}")
    async def get_file(track_id: str, name: str) -> FileResponse:
        media = SERVED_FILES.get(name)
        if media is None:
            raise HTTPException(404, "not served")
        path = track_dir(track_id) / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type=media)

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        head = await file.read(16)
        if not sniff_audio(head, suffix):
            raise HTTPException(415, "expected a .wav or .mp3 upload")

        base = slugify(file.filename or "track")
        with store.lock:
            track_id = unique_slug(data_dir, base)
            # Claimed immediately: the analysis takes a minute, and without the
            # directory on disk two uploads of the same song a second apart would
            # both win the same slug and the second would overwrite the first.
            (data_dir / track_id).mkdir(parents=True)

        upload = work_root / f"{track_id}{suffix}"
        size = len(head)
        with upload.open("wb") as fh:
            fh.write(head)
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    upload.unlink(missing_ok=True)
                    (data_dir / track_id).rmdir()
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
                fh.write(chunk)

        job = Job(id=uuid.uuid4().hex[:12], track_id=track_id, title=Path(file.filename or "").stem)
        store.jobs[job.id] = job
        store.pool.submit(analyze_job, store, job, upload, work_root)
        log.info("job %s: queued %s (%d bytes) as %s", job.id, file.filename, size, track_id)
        return job.as_dict()

    @app.get("/exports/capabilities")
    async def export_capabilities() -> dict[str, Any]:
        return await run_in_threadpool(capabilities_sync)

    @app.post("/exports")
    async def create_export(payload: dict[str, Any]) -> dict[str, Any]:
        track_id = payload.get("trackId") if isinstance(payload, dict) else None
        if not isinstance(track_id, str):
            raise HTTPException(422, "trackId is required")
        directory = track_dir(track_id)
        entry = track_entry(store, directory)
        if entry is None:
            raise HTTPException(409, "track is incomplete")
        capabilities = await run_in_threadpool(capabilities_sync)
        if not capabilities.get("available"):
            raise HTTPException(503, str(capabilities.get("reason") or "export unavailable"))
        recipe, start, duration = validate_export_submission(
            payload,
            entry,
            str(capabilities.get("rendererBuild") or ""),
            capabilities.get("profiles") or [],
        )
        audio = directory / "audio.wav"
        if not audio.is_file():
            raise HTTPException(409, "track has no audio.wav")

        settings = store.export_settings
        assert settings is not None and settings.node_executable is not None
        assert settings.ffmpeg_executable is not None
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        export_id = uuid.uuid4().hex[:16]
        filename = f"{track_id}-{recipe['sim']}-{recipe['seed']}-{export_id}.mp4"
        output_path = (settings.output_dir / filename).resolve()
        if output_path.parent != settings.output_dir:
            raise HTTPException(500, "resolved export path escaped output directory")

        private = work_root / f"export-{job_id}"
        private.mkdir(parents=False, exist_ok=False)
        worker_private = settings.web_dir / ".export-work" / f"export-{job_id}"
        request_path = private / "request.json"
        diagnostic_path = private / "worker.log"
        try:
            manifest, binary, audio_snapshot, job_settings = await run_in_threadpool(
                snapshot_export_inputs,
                directory,
                settings,
                private,
                worker_private,
                str(entry["version"]),
                str(capabilities["rendererBuild"]),
            )
        except ValueError as exc:
            shutil.rmtree(private, ignore_errors=True)
            shutil.rmtree(worker_private, ignore_errors=True)
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            shutil.rmtree(private, ignore_errors=True)
            shutil.rmtree(worker_private, ignore_errors=True)
            raise HTTPException(500, f"could not snapshot export inputs: {exc}") from exc
        request = {
            "version": 1,
            "recipe": recipe,
            "runtime": {
                "ffmpegExecutable": str(settings.ffmpeg_executable),
                "workingDirectory": str(settings.repo_root),
            },
            "timeline": {
                "jsonPath": str(manifest),
                "binaryPath": str(binary),
            },
            "audioPath": str(audio_snapshot),
            "output": {
                "path": str(output_path),
                "profile": recipe["output"]["profile"],
            },
            "range": {"startSeconds": start, "durationSeconds": duration},
        }
        try:
            with request_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    request,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                stream.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            shutil.rmtree(private, ignore_errors=True)
            shutil.rmtree(worker_private, ignore_errors=True)
            raise HTTPException(500, f"could not write export request: {exc}") from exc

        job = Job(
            id=job_id,
            track_id=track_id,
            title=str(entry["title"]),
            kind="export",
            export_id=export_id,
            result={
                "filename": filename,
                "recipeVersion": recipe["version"],
                "rendererBuild": recipe["rendererBuild"],
                "trackVersion": recipe["track"]["contentVersion"],
                "sim": recipe["sim"],
                "seed": recipe["seed"],
            },
        )
        store.jobs[job.id] = job
        store.exports[export_id] = job
        store.export_inputs[job.id] = (job_settings, request_path)
        store.pool.submit(
            export_job,
            job_settings,
            job,
            request_path,
            diagnostic_path,
            output_path,
        )
        log.info("export job %s: queued %s as %s", job.id, track_id, filename)
        return job.as_dict()

    @app.get("/exports/{export_id}")
    async def get_export(export_id: str) -> dict[str, Any]:
        job = store.exports.get(export_id)
        if job is None:
            raise HTTPException(404, "no such export")
        return job.as_dict()

    @app.get("/exports/{export_id}/download")
    async def download_export(export_id: str) -> FileResponse:
        job = store.exports.get(export_id)
        if job is None:
            raise HTTPException(404, "no such export")
        if job.status == "cancelled":
            raise HTTPException(404, "export was cancelled; there is no file")
        if job.status != "done":
            raise HTTPException(409, "export is not complete")
        settings = store.export_settings
        assert settings is not None
        path = settings.output_dir / str(job.result["filename"])
        if not path.is_file():
            raise HTTPException(410, "export file is missing")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = store.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.as_dict()

    @app.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel an export job. Idempotent, and never blocks the event loop.

        An unknown job is a 404 and a terminal one is simply itself again, so a
        double click, a retry after a dropped response, and a cancel that lost
        the race to completion all produce the same untroubled answer.
        """

        job = store.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.kind != "export":
            raise HTTPException(409, "only export jobs can be cancelled")
        inputs = store.export_inputs.get(job_id)
        if inputs is None:
            raise HTTPException(500, "export job has no recorded inputs")
        # Off the loop: this may spend the cooperative grace period waiting for
        # a worker to put its FFmpeg down.
        return await run_in_threadpool(cancel_export_job, job, inputs[0], inputs[1])

    return app


def default_data_dir() -> Path:
    """`<repo>/data/timelines` — the directory `web/scripts/sync-data.mjs` reads.

    Resolved from this file (`analysis/src/terrarium_analysis/server.py`) so that
    a server started from anywhere writes where the rest of the repo looks.
    """
    return Path(__file__).resolve().parents[3] / "data" / "timelines"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="terrarium-server",
        description="HTTP front end for the offline analysis pipeline.",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--data-dir", type=Path, default=None, help="track store (default <repo>/data/timelines)")
    p.add_argument("--node", type=Path, default=None, help="Node executable for native exports")
    p.add_argument("--ffmpeg", type=Path, default=None, help="FFmpeg executable for native exports")
    p.add_argument("--export-worker", type=Path, default=None, help="built Node export worker")
    p.add_argument("--export-dir", type=Path, default=None, help="completed video output directory")
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail a job on a missing model dependency instead of zeroing its channels",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    data_dir = (args.data_dir or default_data_dir()).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    export_settings = ExportSettings.discover(
        repo_root,
        node=args.node,
        ffmpeg=args.ffmpeg,
        worker=args.export_worker,
        output_dir=args.export_dir,
    )
    log.info("serving %s on http://%s:%d", data_dir, args.host, args.port)
    if reason := export_settings.unavailable_reason():
        log.warning("native export unavailable: %s", reason)
    else:
        log.info("native exports will be written to %s", export_settings.output_dir)

    import uvicorn

    uvicorn.run(
        create_app(data_dir, strict=args.strict, export_settings=export_settings),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
