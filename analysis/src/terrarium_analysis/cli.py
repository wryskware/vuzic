"""`terrarium-analyze` — one track in, timeline.json + timeline.bin out."""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
from pathlib import Path

from .context import STAGE_NAMES, Config
from .timeline import TOTAL_DIMS, VERSION, channel_index

LATENT_DIMS = channel_index("latent")[1]

EPILOG = """\
stages (all but `audio` can be skipped):
  audio       decode to WAV, build the 10 Hz frame grid          (always runs)
  beats       Beat This!  -> beats, downbeats, tempo             (needs beat-this)
  structure   All-In-One  -> segments, activations, demucs stems (needs all-in-one-fix)
  stems       demucs stems -> 4 activity curves                  (librosa only)
  events      those stems  -> sparse kick/snare/hat/bass/vocal   (librosa only)
  character   MuQ layer 6 -> 64-dim PCA latent                   (needs muq + torch)
  recurrence  chroma SSM  -> novelty, repeat pointers            (librosa only)

examples:
  terrarium-analyze tracks/song.wav -o out/song
  terrarium-analyze tracks/song.wav -o out/song --skip character --skip recurrence
  terrarium-analyze tracks/song.wav -o out/song --stems-dir demix/htdemucs/song --skip structure
  terrarium-analyze tracks/song.wav -o out/song --embedding --seed 1234
  terrarium-plot out/song

A missing optional dependency skips its stage (leaving its channels at zero) and
prints the install command; --strict turns that into an error instead. The
output layout is fixed at %d dims either way.
""" % TOTAL_DIMS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terrarium-analyze",
        description="Offline analysis for the Latent Music Terrarium (timeline format v2).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="input audio (WAV preferred; MP3 is converted)")
    p.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    p.add_argument("--track-id", default="", help="track id in timeline.json (default: filename)")
    p.add_argument("--hop", type=float, default=0.1, help="frame hop in seconds (default 0.1)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed; drawn fresh each run and printed if omitted (pin it to replay)",
    )
    p.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=[s for s in STAGE_NAMES if s != "audio"],
        help="skip a stage (repeatable)",
    )
    p.add_argument(
        "--stems-dir",
        type=Path,
        default=None,
        help="use existing demucs stems (bass/drums/vocals/other.wav) instead of All-In-One's",
    )
    p.add_argument(
        "--embedding",
        action="store_true",
        help="also dump the raw pooled 1024-dim MuQ layer-6 features as embedding.bin",
    )
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (default auto)")
    p.add_argument("--muq-model", default="OpenMuQ/MuQ-large-msd-iter")
    p.add_argument("--muq-layer", type=int, default=6, help="MuQ hidden layer (default 6)")
    p.add_argument(
        "--latent-dims",
        type=int,
        default=LATENT_DIMS,
        help=f"PCA dims (must be {LATENT_DIMS} for v{VERSION})",
    )
    p.add_argument("--strict", action="store_true", help="fail instead of skipping on a missing dep")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Rejected here, not at pack() time: the layout is fixed, and the emit step
    # runs after MuQ inference, so a late failure discards every stage's work.
    if args.latent_dims != LATENT_DIMS:
        parser.error(
            f"--latent-dims must be {LATENT_DIMS}: the timeline v{VERSION} layout "
            "fixes the 'latent' channel width"
        )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    log = logging.getLogger("terrarium")
    log.info("seed %d%s", seed, "" if args.seed is not None else "  (random; pin with --seed)")

    cfg = Config(
        input_path=args.input,
        out_dir=args.out,
        track_id=args.track_id,
        hop_seconds=args.hop,
        seed=seed,
        device=args.device,
        skip=tuple(args.skip),
        stems_dir=args.stems_dir,
        dump_embedding=args.embedding,
        muq_model=args.muq_model,
        muq_layer=args.muq_layer,
        latent_dims=args.latent_dims,
    )

    from .pipeline import run  # deferred: pulls librosa, which is slow to import

    tl = run(cfg, strict=args.strict)
    print(f"{args.out}/timeline.json  {tl.frames} frames x {tl.data.shape[1]} dims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
