"""`terrarium-plot` — the phase-1 validation plots.

plan.md phase 1: "Plot everything before writing any renderer... Stems are the
first thing to check now — if bass entering doesn't produce an obvious step in
the bass channel, fix it here, because nothing downstream can."

Five figures, written as PNGs next to the timeline:
  stems.png    per-stem activity as small multiples over the waveform envelope
  events.png   the sparse transient stream against the stem curves and beat grid
  novelty.png  novelty4 / novelty16 / actChorus with segment boundaries
  latent.png   PC1-PC2 trajectory, coloured by time, faceted per segment label
  ssm.png      the segment similarity matrix (only if segmentSimilarity exists)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .timeline import STEM_NAMES, Timeline, read_timeline

# Categorical slots in fixed order (5 slots validated together on this surface:
# worst adjacent CVD dE 9.1 protan, normal-vision 22.9, lightness and chroma
# floors pass). Slots below 3:1 contrast carry a visible direct label, which is
# the relief rule. Never cycled: a sixth series would need a re-validated set.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a4fd6")
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SEQ = ("#eaf1fb", "#9dc0ea", "#2a78d6", "#1a4f92", "#0d2f57")  # one hue, light -> dark


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "figure.dpi": 130,
        }
    )
    return plt


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("terrarium_seq", list(SEQ))


def _boundaries(ax, tl: Timeline, label_y: float | None = None) -> None:
    for seg in tl.segments:
        if seg["start"] <= 0:
            continue  # the track start is not a boundary, and its label collides
        ax.axvline(seg["start"], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
        if label_y is not None:
            ax.text(
                seg["start"] + 0.4,
                label_y,
                seg["label"],
                fontsize=7,
                color=INK_2,
                va="top",
                ha="left",
                rotation=90,
            )


def plot_stems(tl: Timeline, wav: Path | None, out: Path):
    plt = _mpl()
    times = np.arange(tl.frames) * tl.hop_seconds
    stems = tl.channel("stems")

    env_t, env = None, None
    if wav is not None and wav.exists():
        import librosa

        y, sr = librosa.load(str(wav), sr=8000, mono=True)
        step = max(1, int(sr * tl.hop_seconds))
        n = (len(y) // step) * step
        env = np.abs(y[:n].reshape(-1, step)).max(axis=1)
        env = env / max(env.max(), 1e-9)
        env_t = np.arange(env.size) * tl.hop_seconds

    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    for i, (name, ax) in enumerate(zip(STEM_NAMES, axes)):
        if env is not None:
            ax.fill_between(env_t, 0, env, color=GRID, lw=0, zorder=0, label="waveform envelope")
        ax.plot(times, stems[:, i], color=SERIES[i], lw=2.0, solid_capstyle="round", zorder=2)
        ax.text(
            0.004,
            0.86,
            name,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="600",
            color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2},
        )
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks([0, 0.5, 1])
        _boundaries(ax, tl, label_y=1.0 if i == 0 else None)
    axes[0].set_title("Stem activity vs waveform — does an entrance read as a step?", loc="left")
    if env is not None:
        axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xlabel("seconds")
    axes[-1].set_xlim(0, max(times[-1], 1))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


EVENT_KINDS = ("kick", "snare", "hat", "bass", "vocal")
# Which stem curve belongs behind each kind's lane.
EVENT_STEM = {"kick": 1, "snare": 1, "hat": 1, "bass": 0, "vocal": 2}
ZOOM_SECONDS = 8.0


def _events_by_kind(tl: Timeline) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in EVENT_KINDS:
        sel = [e for e in tl.events if e.get("kind") == kind]
        out[kind] = (
            np.array([float(e["t"]) for e in sel]),
            np.array([float(e.get("strength", 1.0)) for e in sel]),
        )
    return out


def _zoom_window(kicks: np.ndarray, beats: np.ndarray, duration: float, width: float) -> float:
    """Start of the `width`-second window with the most beat-locked kicks.

    Scored on kicks within 60 ms of a beat rather than on raw count: the raw
    count picks the outro drum roll, which is the one place a listener would not
    expect the kicks to sit on the grid, and this figure exists to answer whether
    they do everywhere else. Ties go to the earliest window, so it is stable.
    """
    if kicks.size == 0:
        return 0.0
    if beats.size:
        locked = kicks[np.abs(kicks[:, None] - beats[None, :]).min(axis=1) <= 0.06]
    else:
        locked = kicks
    if locked.size == 0:
        locked = kicks
    starts = np.arange(0.0, max(duration - width, 0.0) + 0.5, 0.5)
    if starts.size == 0:
        return 0.0
    counts = [int(((locked >= s) & (locked < s + width)).sum()) for s in starts]
    return float(starts[int(np.argmax(counts))])


def plot_events(tl: Timeline, out: Path):
    """The sparse transient stream — one lane per kind, plus a beat-grid zoom.

    The question this figure exists to answer: do the kicks land on the beat
    grid where the drums are playing, and do the events disappear where the
    instrument does?
    """
    plt = _mpl()
    times = np.arange(tl.frames) * tl.hop_seconds
    stems = tl.channel("stems")
    by_kind = _events_by_kind(tl)
    total = sum(t.size for t, _ in by_kind.values())

    fig = plt.figure(figsize=(13, 10.5))
    # Explicit margins rather than tight_layout: the gridspec's own hspace is
    # load-bearing here (the spacer row), and tight_layout warns and fights it.
    gs = fig.add_gridspec(
        7,
        1,
        height_ratios=[1, 1, 1, 1, 1, 0.25, 2.0],
        hspace=0.35,
        left=0.055,
        right=0.985,
        top=0.945,
        bottom=0.055,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]

    for i, (kind, ax) in enumerate(zip(EVENT_KINDS, axes)):
        t_ev, s_ev = by_kind[kind]
        ax.fill_between(times, 0, stems[:, EVENT_STEM[kind]], color=GRID, lw=0, zorder=0)
        if t_ev.size:
            ax.vlines(t_ev, 0, s_ev, color=SERIES[i], lw=0.9, zorder=2)
            ax.scatter(t_ev, s_ev, s=9, color=SERIES[i], lw=0, zorder=3)
        # Direct label, not a legend lookup: the two low-contrast slots need it.
        ax.text(
            0.004,
            0.86,
            f"{kind}  ·  {t_ev.size}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="600",
            color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2},
        )
        ax.set_ylim(-0.02, 1.08)
        ax.set_yticks([0, 1])
        ax.set_xlim(0, max(times[-1], 1))
        _boundaries(ax, tl, label_y=1.05 if i == 0 else None)
        if i < 4:
            ax.set_xticklabels([])
    axes[0].set_title(
        f"Transient events over stem activity — {total} events, "
        "bar height is strength (grey = that stem's activity curve)",
        loc="left",
    )
    axes[4].set_xlabel("seconds")

    # --- zoom: does a kick sit on a beat? -----------------------------------
    beats = np.asarray(tl.beats, dtype=np.float64)
    downs = np.asarray(tl.downbeats, dtype=np.float64)
    kick_t = by_kind["kick"][0]
    if kick_t.size == 0:
        kick_t = np.concatenate([t for t, _ in by_kind.values()])
    start = _zoom_window(kick_t, beats, tl.duration, ZOOM_SECONDS)
    end = start + ZOOM_SECONDS
    axz = fig.add_subplot(gs[6, 0])
    for b in beats[(beats >= start) & (beats <= end)]:
        axz.axvline(b, color=GRID, lw=1.0, zorder=0)
    for b in downs[(downs >= start) & (downs <= end)]:
        axz.axvline(b, color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=1)
    for i, kind in enumerate(EVENT_KINDS):
        t_ev, s_ev = by_kind[kind]
        sel = (t_ev >= start) & (t_ev <= end)
        base = 4 - i
        if sel.any():
            axz.vlines(
                t_ev[sel], base, base + 0.85 * s_ev[sel], color=SERIES[i], lw=2.0, zorder=3
            )
            axz.scatter(
                t_ev[sel],
                base + 0.85 * s_ev[sel],
                s=18,
                color=SERIES[i],
                lw=0,
                zorder=4,
                label=kind,
            )
        axz.text(start + 0.02, base + 0.30, kind, fontsize=8, color=INK_2, ha="left")
    axz.set_ylim(-0.15, 5.95)
    axz.set_yticks([])
    axz.set_xlim(start, end)
    axz.set_xlabel("seconds")
    axz.set_title(
        f"Zoom {start:.1f}–{end:.1f} s — grey lines are beats, dashed are downbeats",
        loc="left",
    )
    axz.legend(
        loc="upper center",
        ncols=5,
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=1.0,
    )
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_novelty(tl: Timeline, out: Path):
    plt = _mpl()
    times = np.arange(tl.frames) * tl.hop_seconds
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)

    ax.plot(times, tl.channel("novelty4")[:, 0], color=SERIES[0], lw=2.0, label="novelty4 (4 bars)")
    ax.plot(
        times, tl.channel("novelty16")[:, 0], color=SERIES[1], lw=2.0, label="novelty16 (16 bars)"
    )
    ax.set_ylabel("novelty")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    ax.set_title("Foote novelty and chorus activation against segment boundaries", loc="left")
    _boundaries(ax, tl, label_y=ax.get_ylim()[1])

    ax2.fill_between(times, 0, tl.channel("actChorus")[:, 0], color=SERIES[0], alpha=0.25, lw=0)
    ax2.plot(times, tl.channel("actChorus")[:, 0], color=SERIES[0], lw=2.0, label="actChorus")
    ax2.plot(times, tl.channel("recurStr")[:, 0], color=SERIES[2], lw=2.0, label="recurStr")
    ax2.set_ylabel("activation")
    ax2.set_xlabel("seconds")
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    _boundaries(ax2, tl)
    ax2.set_xlim(0, max(times[-1], 1))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _label_at(tl: Timeline, times: np.ndarray) -> np.ndarray:
    labels = np.array(["-"] * times.size, dtype=object)
    for seg in tl.segments:
        inside = (times >= seg["start"]) & (times < seg["end"])
        labels[inside] = seg["label"]
    return labels


def plot_latent(tl: Timeline, out: Path):
    plt = _mpl()
    times = np.arange(tl.frames) * tl.hop_seconds
    latent = tl.channel("latent")
    labels = _label_at(tl, times)
    uniq = [x for x in dict.fromkeys(labels.tolist()) if x != "-"]

    # Faceted per label rather than one scatter with N categorical hues: with
    # every pair on screen at once no 5+ hue set clears the CVD floors, and the
    # question here ("do the three choruses land in the same neighbourhood?") is
    # answered better by facets anyway.
    ncols = max(1, min(4, len(uniq)))
    nrows = 1 + int(np.ceil(len(uniq) / ncols)) if uniq else 1
    fig = plt.figure(figsize=(13, 3.6 * nrows))
    gs = fig.add_gridspec(nrows, ncols)

    ax = fig.add_subplot(gs[0, :])
    sc = ax.scatter(
        latent[:, 0], latent[:, 1], c=times, cmap=_sequential_cmap(), s=9, lw=0, alpha=0.9
    )
    ax.plot(latent[:, 0], latent[:, 1], color=MUTED, lw=0.4, alpha=0.35, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent trajectory (MuQ L6 -> PCA), coloured by time", loc="left")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("seconds", color=INK_2)
    cb.outline.set_visible(False)

    for i, name in enumerate(uniq):
        axf = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        axf.scatter(latent[:, 0], latent[:, 1], c=GRID, s=6, lw=0)
        sel = labels == name
        axf.scatter(latent[sel, 0], latent[sel, 1], c=SERIES[0], s=9, lw=0)
        axf.set_title(f"{name}  ({int(sel.sum())} frames)", loc="left", fontsize=9)
        axf.set_xticks([])
        axf.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_ssm(tl: Timeline, out: Path):
    if tl.segment_similarity is None or tl.segment_similarity.size == 0:
        return None
    plt = _mpl()
    sim = tl.segment_similarity
    names = [f"{i}:{s['label']}" for i, s in enumerate(tl.segments)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(names), 0.9 + 0.5 * len(names)))
    im = ax.imshow(sim, cmap=_sequential_cmap(), vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    ax.set_title("Segment similarity (2D-FMC cosine)", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terrarium-plot",
        description="Phase-1 validation plots for a timeline directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  terrarium-plot out/song\n  terrarium-plot out/song --wav tracks/song.wav",
    )
    p.add_argument("timeline", type=Path, help="directory containing timeline.json + timeline.bin")
    p.add_argument("--wav", type=Path, default=None, help="audio to draw behind the stem curves")
    p.add_argument("--out", type=Path, default=None, help="plot directory (default <timeline>/plots)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tl = read_timeline(args.timeline)
    root = args.timeline if args.timeline.is_dir() else args.timeline.parent
    out_dir = args.out or (root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = args.wav
    if wav is None:
        guess = root / "audio.wav"
        wav = guess if guess.exists() else None

    written = [
        plot_stems(tl, wav, out_dir / "stems.png"),
        plot_events(tl, out_dir / "events.png") if tl.events else None,
        plot_novelty(tl, out_dir / "novelty.png"),
        plot_latent(tl, out_dir / "latent.png"),
        plot_ssm(tl, out_dir / "ssm.png"),
    ]
    for path in written:
        if path is not None:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
