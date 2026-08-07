"""Pipeline stages.

Every stage module exposes the same two names:

    NAME: str
    run(ctx: Context) -> None

A stage reads what earlier stages left on the context and writes its own
results back onto it. Stages other than `audio` are individually skippable; a
skipped stage leaves its channels at zero.
"""

from . import audio, beats, character, events, recurrence, stems, structure

STAGES = (audio, beats, structure, stems, events, character, recurrence)

__all__ = [
    "STAGES",
    "audio",
    "beats",
    "structure",
    "stems",
    "events",
    "character",
    "recurrence",
]
