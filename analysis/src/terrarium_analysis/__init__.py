"""Offline analysis pipeline for the Latent Music Terrarium.

Only light dependencies (numpy/scipy/librosa/soundfile) are imported at module
scope anywhere in this package. Every model dependency is loaded lazily inside
the stage that needs it, so the package imports and the CLI runs help with a
light-only environment.
"""

from .timeline import CHANNELS, TOTAL_DIMS, Timeline, channel_index, read_timeline

__all__ = ["CHANNELS", "TOTAL_DIMS", "Timeline", "channel_index", "read_timeline"]
__version__ = "0.1.0"
