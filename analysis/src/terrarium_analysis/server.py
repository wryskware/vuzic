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
import re
import secrets
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .context import STAGE_NAMES, Config

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

JobStatus = Literal["queued", "running", "done", "error"]


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
    status: JobStatus = "queued"
    stage: str = ""
    message: str = ""
    error: str = ""
    #: 0..1, by stage index. Coarse and honest: the stages are wildly unequal
    #: (structure and character are ~80 % of the wall time between them), so this
    #: is "how far through the list", not "how far through the work".
    progress: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.id,
            "trackId": self.track_id,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "progress": round(self.progress, 3),
        }


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
    #: Guards the slug claim only — the read-modify-write of "is this directory
    #: taken, take it". Job *fields* are deliberately not under it: they are
    #: written from the worker and from a logging handler, and a lock spanning
    #: those two is how the first version of this file deadlocked.
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: One at a time. Both heavy stages want the whole GPU, so a second
    #: concurrent analysis would not finish sooner — it would just make the first
    #: one slower and put two demucs models in VRAM at once.
    pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyze")
    )
    #: `dir name -> (fingerprint, listing)`. See `track_entry`.
    cache: dict[str, tuple[tuple[int, int, int, int], dict[str, Any]]] = field(default_factory=dict)


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

    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    h.update(bin_path.read_bytes())
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
        "version": h.hexdigest()[:16],
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


def create_app(data_dir: Path, strict: bool = False) -> FastAPI:
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir=data_dir, strict=strict)
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
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def track_dir(track_id: str) -> Path:
        # `..`, absolute paths and separators all die here: the id has to be one
        # of our own directory names, compared against the listing.
        d = data_dir / track_id
        if not re.fullmatch(r"[A-Za-z0-9._-]+", track_id) or not d.is_dir():
            raise HTTPException(404, "no such track")
        return d

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

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = store.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.as_dict()

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
    log.info("serving %s on http://%s:%d", data_dir, args.host, args.port)

    import uvicorn

    uvicorn.run(create_app(data_dir, strict=args.strict), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
