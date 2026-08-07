"""Stage orchestration."""

from __future__ import annotations

import numpy as np

from .context import Config, Context, log
from .emitter import emit
from .optional import MissingDependency
from .stages import STAGES
from .timeline import Timeline


def run(cfg: Config, strict: bool = False) -> Timeline:
    """Run every enabled stage in order, then emit.

    A stage whose optional dependency is missing is reported and skipped rather
    than aborting the run (unless `strict`): a timeline with zeroed channels is
    still useful downstream, and the phase-1 loop is "run what you have, look at
    the plots".
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(cfg=cfg, rng=np.random.default_rng(cfg.seed))

    for stage in STAGES:
        name = stage.NAME
        if name != "audio" and not cfg.enabled(name):
            log.info("stage %-10s skipped (--skip %s)", name, name)
            continue
        try:
            stage.run(ctx)
            ctx.ran.append(name)
        except MissingDependency as exc:
            if strict or name == "audio":
                raise
            ctx.note(f"stage '{name}' skipped - missing dependency")
            log.error("stage %-10s SKIPPED\n%s", name, exc)

    return emit(ctx)
