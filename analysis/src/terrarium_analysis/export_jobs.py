"""Native headless-export process discovery and supervision.

The FastAPI server remains the only listener.  A Node/Dawn worker and the
FFmpeg process it owns are short-lived children of one export job; neither opens
a port.  This module deliberately knows nothing about FastAPI so process and
protocol behavior can be tested without an HTTP server.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MAX_PROTOCOL_LINE_BYTES = 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
PROBE_TIMEOUT_SECONDS = 30

#: How long a cancelled worker gets to shut down cooperatively — abort FFmpeg,
#: delete its `.partial`, exit — before it is killed. Long enough for an FFmpeg
#: that is mid-write to close its file, short enough that a user who clicked
#: cancel does not think the button is broken.
CANCEL_GRACE_SECONDS = 5.0
#: Bound on the force-kill itself, so a wedged `taskkill` cannot hold the request.
KILL_TIMEOUT_SECONDS = 10.0
#: The worker's own exit code for "a cancellation signal reached me". Distinct
#: from 1 (crash) so the supervisor can tell the two apart even if the flag on
#: this side was never set — an operator killing the tree by hand, say.
CANCELLED_WORKER_EXIT_CODE = 130


class ExportProcessError(RuntimeError):
    """A worker could not start, violated its protocol, or exited unsuccessfully."""


class ExportCancelled(ExportProcessError):
    """The job was cancelled; the worker was stopped rather than failing."""


@dataclass(frozen=True)
class ExportSettings:
    """Server-owned native paths.  None of these values come from the browser."""

    repo_root: Path
    web_dir: Path
    worker_path: Path
    node_executable: Path | None
    ffmpeg_executable: Path | None
    output_dir: Path

    @classmethod
    def discover(
        cls,
        repo_root: Path,
        *,
        node: Path | None = None,
        ffmpeg: Path | None = None,
        worker: Path | None = None,
        output_dir: Path | None = None,
    ) -> "ExportSettings":
        root = repo_root.resolve()
        web_dir = (root / "web").resolve()

        def executable(explicit: Path | None, name: str) -> Path | None:
            candidate = str(explicit) if explicit is not None else shutil.which(name)
            return Path(candidate).resolve() if candidate else None

        return cls(
            repo_root=root,
            web_dir=web_dir,
            worker_path=(worker or web_dir / "dist-worker" / "worker.mjs").resolve(),
            node_executable=executable(node, "node"),
            ffmpeg_executable=executable(ffmpeg, "ffmpeg"),
            output_dir=(output_dir or root / "data" / "exports").resolve(),
        )

    def unavailable_reason(self) -> str:
        if self.node_executable is None or not self.node_executable.is_file():
            return "Node executable not found"
        if self.ffmpeg_executable is None or not self.ffmpeg_executable.is_file():
            return "FFmpeg executable not found"
        if not self.worker_path.is_file():
            return f"headless worker not built: {self.worker_path}"
        if not self.web_dir.is_dir():
            return f"web working directory not found: {self.web_dir}"
        return ""

    def renderer_build(self) -> str:
        if not self.worker_path.is_file():
            return ""
        digest = hashlib.sha256(self.worker_path.read_bytes()).hexdigest()
        return digest[:16]

    def child_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # The probe invokes `ffmpeg` internally. Put the server-selected binary's
        # directory first while the actual render request still carries its exact
        # absolute path.
        if self.ffmpeg_executable is not None:
            env["PATH"] = os.pathsep.join(
                (str(self.ffmpeg_executable.parent), env.get("PATH", ""))
            )
        return env


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _worker_popen_kwargs() -> dict[str, Any]:
    """Spawn the render worker as the root of its own killable process tree.

    The worker starts FFmpeg, FFmpeg may start nothing or something, and none of
    them are ours to enumerate by hand — so the child gets a group/session of its
    own and cancellation addresses the group.

    Windows: `CREATE_NEW_PROCESS_GROUP` makes the worker's PID a process-group id
    that `GenerateConsoleCtrlEvent` (Python's `send_signal(CTRL_BREAK_EVENT)`)
    can target, and simultaneously detaches it from the console's Ctrl+C — so a
    Ctrl+C in the server's terminal no longer half-kills a render behind its
    back. Deliberately *not* `CREATE_NO_WINDOW`, unlike the probe: that flag
    gives the child no console at all, and a process with no console can neither
    be sent nor receive a console control event, which would leave force-kill as
    the only option. Node inheriting the server's console opens no window.
    """

    if os.name == "nt":
        return {"creationflags": int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))}
    return {"start_new_session": True}


def _taskkill(pid: int) -> None:
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    executable = Path(system_root) / "System32" / "taskkill.exe"
    try:
        subprocess.run(
            [str(executable), "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=KILL_TIMEOUT_SECONDS,
            shell=False,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        # Already gone, or taskkill is unavailable.  Either way the caller's
        # bounded wait below is what decides whether the tree actually died.
        return


def terminate_process_tree(
    process: "subprocess.Popen[Any]",
    grace_seconds: float = CANCEL_GRACE_SECONDS,
) -> None:
    """Cooperative stop, then force, applied to the whole tree.  Never raises.

    Two stages on purpose: the worker owns an FFmpeg child writing an MP4, and a
    hard kill of Node alone would orphan that FFmpeg and leave a `.partial`
    behind.  The cooperative stage lets the worker do its own cleanup; the force
    stage is what guarantees nothing survives when it will not.
    """

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError, AttributeError):
            # No console, or the process is already reaped: fall straight
            # through to the force stage.
            pass
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        _taskkill(process.pid)
    else:
        try:
            group = os.getpgid(process.pid)
        except (OSError, AttributeError):
            group = process.pid
        for sig, timeout in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, KILL_TIMEOUT_SECONDS)):
            try:
                os.killpg(group, sig)
            except (OSError, AttributeError):
                pass
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue
        return
    try:
        process.wait(timeout=KILL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass


class CancellationHandle:
    """The join point between an HTTP cancel and a worker that may not exist yet.

    Cancellation and process start race in both directions — a `DELETE` can land
    while `Popen` is still returning — so both sides go through one lock, and the
    handle kills on `attach` if the request already arrived.  `cancelled` is
    latched: once set it never clears, which is what makes double-cancel a no-op
    and lets the supervisor classify the worker's exit after the fact.
    """

    def __init__(self, grace_seconds: float = CANCEL_GRACE_SECONDS) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._process: "subprocess.Popen[Any] | None" = None
        self._grace_seconds = grace_seconds

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def attach(self, process: "subprocess.Popen[Any]") -> None:
        with self._lock:
            self._process = process
            already = self._cancelled
        if already:
            terminate_process_tree(process, self._grace_seconds)

    def detach(self) -> None:
        with self._lock:
            self._process = None

    def cancel(self) -> None:
        """Latch the request and stop any attached tree.  Safe to call twice."""

        with self._lock:
            self._cancelled = True
            process = self._process
        if process is not None:
            terminate_process_tree(process, self._grace_seconds)


def _json_lines(text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if len(raw.encode("utf-8", errors="replace")) > MAX_PROTOCOL_LINE_BYTES:
            raise ExportProcessError("worker emitted an oversized protocol line")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExportProcessError(f"worker emitted non-JSON stdout: {raw[:200]}") from exc
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ExportProcessError("worker emitted a malformed protocol message")
        messages.append(message)
    return messages


def probe_export_capabilities(settings: ExportSettings) -> dict[str, Any]:
    """Run one short native probe.  The caller owns caching this result."""

    reason = settings.unavailable_reason()
    base: dict[str, Any] = {
        "available": False,
        "profiles": [],
        "encoders": [],
        "rendererBuild": settings.renderer_build(),
        "reason": reason,
    }
    if reason:
        return base

    assert settings.node_executable is not None
    try:
        completed = subprocess.run(
            [
                str(settings.node_executable),
                str(settings.worker_path),
                "--probe",
                "--width",
                "64",
                "--height",
                "36",
                "--frames",
                "3",
                "--ring-size",
                "2",
            ],
            cwd=settings.web_dir,
            env=settings.child_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
            shell=False,
            creationflags=_creation_flags(),
        )
    except subprocess.TimeoutExpired:
        return {**base, "reason": f"worker probe timed out after {PROBE_TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {**base, "reason": f"worker probe could not start: {exc}"}
    try:
        messages = _json_lines(completed.stdout)
    except ExportProcessError as exc:
        return {**base, "reason": str(exc)}

    error = next((m for m in messages if m.get("type") == "error"), None)
    ready = next((m for m in messages if m.get("type") == "ready"), None)
    result = next((m for m in messages if m.get("type") == "result"), None)
    if completed.returncode != 0 or error or not ready or not result:
        detail = str((error or {}).get("message") or completed.stderr[-2000:]).strip()
        return {**base, "reason": detail or f"worker probe exited {completed.returncode}"}

    encoders = [
        encoder
        for encoder in ready.get("encoders", [])
        if encoder in ("hevc_nvenc", "av1_nvenc")
    ]
    return {
        "available": "av1_nvenc" in encoders,
        "profiles": ["av1-sdr-debug-2160p120", "av1-sdr-debug-1080p120"],
        "gpu": str(ready.get("adapter") or ""),
        "backend": str(ready.get("backend") or ""),
        "encoders": encoders,
        "rendererBuild": settings.renderer_build(),
        "transport": "sdr-rgba8-av1-debug",
        "reason": "" if "av1_nvenc" in encoders else "AV1 NVENC is unavailable",
    }


def diagnostic_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_DIAGNOSTIC_BYTES))
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def compact_diagnostic_log(path: Path) -> None:
    """Retain at most the useful tail after a worker exits."""

    try:
        if path.stat().st_size <= MAX_DIAGNOSTIC_BYTES:
            return
        with path.open("rb") as stream:
            stream.seek(-MAX_DIAGNOSTIC_BYTES, os.SEEK_END)
            tail = stream.read()
        path.write_bytes(tail)
    except OSError:
        # Diagnostics must never hide the actual worker outcome.
        return


def run_export_worker(
    settings: ExportSettings,
    request_path: Path,
    diagnostic_path: Path,
    on_message: Callable[[dict[str, Any]], None],
    cancel: CancellationHandle | None = None,
) -> dict[str, Any]:
    """Run one request and forward each validated NDJSON stdout message."""

    reason = settings.unavailable_reason()
    if reason:
        raise ExportProcessError(reason)
    request = request_path.resolve()
    if not request.is_file():
        raise ExportProcessError(f"request file not found: {request}")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    assert settings.node_executable is not None
    # Checked before the spawn as well as after the wait: a cancel that arrives
    # in the gap between "queued job dequeued" and "worker started" must not
    # start a render nobody wants.
    if cancel is not None and cancel.cancelled:
        raise ExportCancelled("export cancelled before the worker started")

    result: dict[str, Any] | None = None
    worker_error = ""
    try:
        with diagnostic_path.open("w", encoding="utf-8", errors="replace") as diagnostics:
            process = subprocess.Popen(
                [
                    str(settings.node_executable),
                    str(settings.worker_path),
                    "--request",
                    str(request),
                ],
                cwd=settings.web_dir,
                env=settings.child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                **_worker_popen_kwargs(),
            )
            assert process.stdout is not None
            if cancel is not None:
                cancel.attach(process)
            try:
                for raw in process.stdout:
                    messages = _json_lines(raw)
                    for message in messages:
                        on_message(message)
                        if message["type"] == "result":
                            result = message
                        elif message["type"] == "error":
                            worker_error = str(message.get("message") or "worker failed")
                return_code = process.wait()
            except BaseException:
                # A protocol violation or callback failure must not leave the
                # native worker running after Python rejects its output.
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    try:
                        process.kill()
                    except OSError:
                        pass
                raise
            finally:
                if cancel is not None:
                    cancel.detach()
    finally:
        compact_diagnostic_log(diagnostic_path)

    if (cancel is not None and cancel.cancelled) or return_code == CANCELLED_WORKER_EXIT_CODE:
        raise ExportCancelled(worker_error or "export cancelled")
    if return_code != 0 or worker_error:
        detail = worker_error or diagnostic_tail(diagnostic_path)
        raise ExportProcessError(detail or f"worker exited with code {return_code}")
    if result is None:
        raise ExportProcessError("worker exited successfully without a result message")
    return result
