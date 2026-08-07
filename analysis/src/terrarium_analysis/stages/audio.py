"""Stage 1 — load the track, guarantee a WAV for the model stages.

Everything downstream is given a WAV path. Per plan.md Decision 6, MP3 decoder
offsets of 20-40 ms are a known All-In-One quirk, so a non-WAV input is decoded
once here and every later stage reads the decoded file, keeping all stages on
one time base even if that base is shifted from the original MP3.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from ..context import Context, build_grid, log

NAME = "audio"

WAV_SUFFIXES = {".wav", ".wave"}


def _decode_with_soundfile(src: Path, dst: Path) -> bool:
    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        log.info("soundfile could not decode %s (%s); trying ffmpeg", src.name, exc)
        return False
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return True


def _decode_with_ffmpeg(src: Path, dst: Path) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:a", "pcm_s16le", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        log.error("ffmpeg failed: %s", exc.stderr.decode("utf-8", "replace")[:500])
        return False
    return dst.exists()


def run(ctx: Context) -> None:
    src = Path(ctx.cfg.input_path)
    if not src.exists():
        raise FileNotFoundError(f"input audio not found: {src}")

    if src.suffix.lower() in WAV_SUFFIXES:
        wav = src
    else:
        wav = ctx.cfg.out_dir / "audio.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        ok = _decode_with_soundfile(src, wav) or _decode_with_ffmpeg(src, wav)
        if not ok:
            raise RuntimeError(
                f"could not decode {src.name} to WAV. Install ffmpeg, or convert manually:\n"
                f"    ffmpeg -i '{src}' -c:a pcm_s16le '{wav}'"
            )
        ctx.note(
            f"decoded {src.suffix} -> {wav.name}: lossy decoders introduce a 20-40 ms "
            "leading offset (plan.md Decision 6). All stages share this offset, but the "
            "timeline may sit that far from the original file; prefer a WAV master."
        )

    info = sf.info(str(wav))
    ctx.wav_path = wav
    ctx.sample_rate = int(info.samplerate)
    ctx.duration = float(info.frames) / float(info.samplerate)

    import librosa  # light dep, but slow to import — keep it inside the stage

    ctx.mono_sr = int(ctx.cfg.analysis_sr)
    ctx.mono = np.asarray(
        librosa.load(str(wav), sr=ctx.mono_sr, mono=True)[0], dtype=np.float32
    )

    build_grid(ctx)
    if not ctx.cfg.track_id:
        ctx.cfg.track_id = src.stem
    log.info(
        "audio: %s  %.2fs  %d Hz  %d ch -> %d frames @ %.3fs",
        wav.name,
        ctx.duration,
        ctx.sample_rate,
        info.channels,
        ctx.frames,
        ctx.cfg.hop_seconds,
    )
