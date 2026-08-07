"""Stage 6 — assemble the context into timeline.json + timeline.bin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .context import Context, log
from .timeline import EVENT_KINDS, Timeline, write_timeline


def build(ctx: Context) -> Timeline:
    frames = ctx.frames
    zeros = np.zeros(frames, dtype=np.float64)
    channels = {
        "stems": ctx.stems if ctx.stems is not None else np.zeros((frames, 4)),
        "latent": ctx.latent if ctx.latent is not None else np.zeros((frames, 64)),
        "novelty4": ctx.novelty4 if ctx.novelty4 is not None else zeros,
        "novelty16": ctx.novelty16 if ctx.novelty16 is not None else zeros,
        "actChorus": ctx.act_chorus if ctx.act_chorus is not None else zeros,
        "recurTime": ctx.recur_time if ctx.recur_time is not None else zeros,
        "recurStr": ctx.recur_str if ctx.recur_str is not None else zeros,
    }
    from .timeline import pack

    return Timeline(
        track_id=ctx.cfg.track_id or Path(ctx.cfg.input_path).stem,
        duration=ctx.duration,
        sample_rate=ctx.sample_rate,
        hop_seconds=ctx.cfg.hop_seconds,
        frames=frames,
        beats=ctx.beats,
        downbeats=ctx.downbeats,
        tempo=ctx.tempo,
        segments=ctx.segments,
        data=pack(frames, channels),
        segment_similarity=ctx.segment_similarity,
        events=list(ctx.events),
    )


def emit(ctx: Context) -> Timeline:
    tl = build(ctx)
    out_dir = Path(ctx.cfg.out_dir)
    json_path, bin_path = write_timeline(tl, out_dir)

    if ctx.cfg.dump_embedding and ctx.embedding is not None:
        emb = np.ascontiguousarray(ctx.embedding, dtype="<f4")
        (out_dir / "embedding.bin").write_bytes(emb.tobytes(order="C"))
        (out_dir / "embedding.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "frames": int(emb.shape[0]),
                    "dims": int(emb.shape[1]),
                    "hopSeconds": ctx.cfg.hop_seconds,
                    "source": f"{ctx.cfg.muq_model} layer {ctx.cfg.muq_layer}, pooled to grid",
                    "dtype": "float32-le",
                    "layout": "row-major frames x dims",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log.info("wrote embedding.bin (%d x %d)", emb.shape[0], emb.shape[1])

    run_doc = {
        "trackId": tl.track_id,
        "input": str(ctx.cfg.input_path),
        "wav": str(ctx.wav_path) if ctx.wav_path else None,
        "seed": int(ctx.cfg.seed),
        "hopSeconds": ctx.cfg.hop_seconds,
        "frames": tl.frames,
        "device": ctx.cfg.device,
        "stagesRun": list(ctx.ran),
        "stagesSkipped": list(ctx.cfg.skip),
        "muq": {"model": ctx.cfg.muq_model, "layer": ctx.cfg.muq_layer},
        "events": {
            "total": len(tl.events),
            **{
                kind: sum(1 for e in tl.events if e["kind"] == kind)
                for kind in EVENT_KINDS
            },
        },
        "notes": list(ctx.notes),
    }
    (out_dir / "run.json").write_text(json.dumps(run_doc, indent=2) + "\n", encoding="utf-8")

    log.info(
        "wrote %s (%d frames x %d dims, %d events) and %s",
        json_path.name,
        tl.frames,
        tl.data.shape[1],
        len(tl.events),
        bin_path.name,
    )
    return tl
