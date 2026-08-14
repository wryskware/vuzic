# analysis/ — the offline pipeline

One track in, `timeline.json` + `timeline.bin` out. Phase 1 of `docs/plan.md`.
Runs once per track, offline; nothing here is on the runtime path.

```
song.wav
  ├─ audio       decode to WAV, build the 10 Hz frame grid
  ├─ beats       Beat This!   → beats, downbeats, tempo          (master clock)
  ├─ structure   All-In-One   → segments, labels, 100 fps activations, demucs stems
  ├─ stems       those stems  → 4 activity curves                (the "instrument entered" signal)
  ├─ events      those stems  → sparse kick/snare/hat/bass/vocal (the "that just hit" signal)
  ├─ character   MuQ layer 6  → 25 Hz → 10 Hz → PCA-64 latent
  ├─ recurrence  chroma SSM   → novelty4/16, repeat pointers, segment similarity
  └─ emitter     → timeline.json + timeline.bin (format v2, 73 dims/frame)
```

The byte layout is the contract with `web/`, and
`data/timelines/synthetic/timeline.json` is the shared fixture both sides
conform to. `tests/test_emitter.py` asserts this package's layout against that
file — if the fixture changes, the tests fail, which is the point.

## Environment

Light dependencies only (numpy, scipy, librosa, soundfile, matplotlib, pytest).
The package imports and both CLIs run with nothing else installed.

```bash
cd analysis
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Then `.venv/Scripts/terrarium-analyze --help` (Windows) or
`.venv/bin/terrarium-analyze --help` (WSL2/Linux), or prefix commands with
`uv run`.

### The heavy dependencies are optional, on purpose

`torch`, `beat_this`, `allin1_infer`, `muq` and demucs are imported lazily, inside the
stage that needs them. A missing one prints its exact install command and skips
that stage, leaving its channels at zero; `--strict` turns that into an error.

```bash
uv pip install -e ".[beats]"      # beat-this + torch
uv pip install -e ".[structure]"  # all-in-one-infer — see the Windows note below
uv pip install -e ".[character]"  # muq + torch
```

### Windows: run the model stages under WSL2

`docs/plan.md`, "Windows gotcha": the **original** `allin1` package calls
`natten.functional.natten1dqkrpb`, an API NATTEN deleted in 0.17, and drags in a
madmom that does not build on Python 3.10+. Neither is installable today.

**`all-in-one-infer` is what this package uses instead.** It is the renamed,
maintained `all-in-one-fix` fork (`all-in-one-fix` itself has no files left on
PyPI): neighborhood attention reimplemented in pure PyTorch, plus `madmom-infer`
and `demucs-infer`. No NATTEN, no compiled extension, no torch upper bound — the
last part is what makes a Blackwell GPU workable, since that needs cu128 wheels
and therefore torch ≥ 2.7. The package imports as `allin1_infer`; the stage
falls back to `allin1` if only the original is present.

**Verified working environment** (WSL2 Ubuntu 24.04, RTX 5090 / sm_120,
2026-08-06 — a full 4:32 track through all five stages in 60 s warm, 83 s cold):

| | version | notes |
|---|---|---|
| Python | 3.10.19 | `uv venv --python 3.10` fetches it |
| torch / torchaudio | `2.8.0+cu128` | cu128 index; sm_120 needs ≥ 2.7 |
| all-in-one-infer | 3.1.0 | imports as `allin1_infer` |
| madmom-infer | 0.2.0 | pulled by the above |
| demucs-infer | 4.2.2 | pulled by the above |
| beat-this | 1.1.0 | now on PyPI; the git URL is no longer needed |
| muq | 0.1.0 | with transformers 5.14.1, nnAudio 0.3.4 |
| NATTEN | **not installed** | `all-in-one-infer` does not use it |

```bash
wsl -d Ubuntu
# work on the ext4 disk — /mnt/c I/O is slow enough to dominate the run
cp -r /mnt/c/Users/<you>/wryskware/latent-music-terrarium/analysis ~/lmt/analysis
cd ~/lmt/analysis
uv venv --python 3.10 && . .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128"
uv pip install all-in-one-infer beat-this muq
terrarium-analyze "tracks/song.wav" -o out/song --seed 1234 --embedding
terrarium-plot out/song --wav "tracks/song.wav"
```

Then copy `timeline.json` / `timeline.bin` (and `embedding.bin`) back into
`data/timelines/<track>/` on the Windows side and run `npm run sync-data` in
`web/`. Alternatively run only the structure stage in WSL2 and reuse its stems:

```bash
# in WSL2 — writes out/song/demix/htdemucs/song/{bass,drums,vocals,other}.wav
terrarium-analyze tracks/song.wav -o out/song --skip character --skip recurrence
# back on Windows — everything except structure, reusing those stems
terrarium-analyze tracks/song.wav -o out/song --skip structure \
    --stems-dir out/song/demix/htdemucs/song
```

Model weights download on first use into `~/.cache/huggingface` and
`~/.cache/torch`: All-In-One's eight harmonix folds, htdemucs, the Beat This!
checkpoint, and MuQ-large (~1.3 GB).

Wall time on that machine, warm caches, a 272 s track — **60 s end to end**
(~63 s since the events stage landed).
Measured per stage as `total − 2.3 s` (the interpreter start plus the audio
stage, timed on its own):

| stage | device | net seconds |
|---|---|---|
| audio + startup | CPU | 2.3 (the baseline subtracted below) |
| beats | CUDA | 2.9 |
| structure | CUDA | 30.2 — htdemucs plus eight harmonix folds |
| stems | CPU | 1.7 |
| events | CPU | 2.8 — measured as the difference of two runs with and without it |
| character | CUDA | 20.1 — MuQ-large over 30 s chunks |
| recurrence | CPU | 5.9 |

## Tracks

Drop audio in `analysis/tracks/` — gitignored, nothing there is committed. WAV
is preferred: lossy decoders introduce a 20–40 ms leading offset (a known
All-In-One quirk, `plan.md` Decision 6). MP3 and friends are decoded to
`<out>/audio.wav` automatically, via soundfile or ffmpeg, with a warning.

Pick a track with obvious structure and clearly separable stems (phase 0).

## Usage

```bash
terrarium-analyze tracks/song.wav -o out/song
terrarium-plot out/song
```

Stage skipping — every stage but `audio` can be skipped, repeatably:

```bash
# no models installed: stems from an existing demucs folder, plus recurrence
terrarium-analyze tracks/song.wav -o out/song \
    --skip beats --skip structure --skip character --stems-dir demix/song

# structure and stems only, quickly
terrarium-analyze tracks/song.wav -o out/song --skip character --skip recurrence

# raw 1024-dim MuQ layer-6 features alongside, for a future learned mapping
terrarium-analyze tracks/song.wav -o out/song --embedding

# pin the seed to replay a run exactly (it is random and printed otherwise)
terrarium-analyze tracks/song.wav -o out/song --seed 1234
```

Other flags: `--hop` (frame rate, default 0.1 s), `--device auto|cpu|cuda`,
`--muq-layer` (default 6 — genre/emotion; the last layer is the wrong answer),
`--track-id`, `--strict`, `-v`.

## Output

```
out/song/
  timeline.json    manifest: track, grid, beats, downbeats, tempo, segments,
                   segmentSimilarity (optional), channel table
  timeline.bin     Float32Array, frames × 73, row-major, little-endian
  embedding.bin    optional raw pooled 1024-dim MuQ features (--embedding)
  run.json         seed, stages run/skipped, warnings — the reproducibility record
  audio.wav        only when the input was not already WAV
  plots/           terrarium-plot output
```

| channel | dims | offset | |
|---|---|---|---|
| `stems` | 4 | 0 | bass, drums, vocals, other — 0..1, smoothed at beat scale |
| `latent` | 64 | 4 | MuQ L6, standardized + PCA-64, PC1 scaled to unit RMS |
| `novelty4` | 1 | 68 | Foote novelty, 4-bar kernel, 0..1 |
| `novelty16` | 1 | 69 | Foote novelty, 16-bar kernel, 0..1 |
| `actChorus` | 1 | 70 | All-In-One chorus activation, 0..1 |
| `recurTime` | 1 | 71 | **seconds** — best matching earlier moment, 0 if none |
| `recurStr` | 1 | 72 | strength of that match, 0..1 |

Skipped stages leave their channels at zero rather than removing them: the
layout is always 73 dims, so the web side never branches on which stages ran.

### The `events` array (optional, additive to v2)

`plan.md` Revision 2, item 2 — beat-to-beat reactivity without a runtime FFT.
`timeline.json` gains an optional top-level array, sorted by `t` ascending:

```jsonc
"events": [
  { "t": 69.412, "kind": "kick", "strength": 1.0 },
  { "t": 69.643, "kind": "hat",  "strength": 0.62 }
]
```

`kind` is one of `kick | snare | hat | bass | vocal`; `strength` is 0..1;
`t` is seconds, rounded to the millisecond. **The field is optional and the
version stays 2** — absent or empty means "no events", and a reader that
predates it is still correct.

Detection (`stages/events.py`), all of it a deterministic function of the demucs
stems, no RNG:

| kind | stem | band | min gap |
|---|---|---|---|
| `kick` | drums | lowpass 120 Hz | 90 ms |
| `snare` | drums | 120 Hz – 2 kHz | 90 ms |
| `hat` | drums | highpass 4 kHz | 60 ms |
| `bass` | bass | 30–500 Hz, mel capped at 500 Hz | 100 ms |
| `vocal` | vocals | 150 Hz – 6 kHz | 100 ms |

Zero-phase Butterworth split (a causal filter would delay each band by a
different amount and smear the kinds apart in time), `librosa` onset strength at
an 11.6 ms hop, `peak_pick`, then:

- **the silence gate is `stems.py`'s** — a peak survives only where the band is
  above −70 dBFS *and* above 10 % of its own dynamic range, so demucs bleed in a
  stem that is not playing produces nothing at all;
- **backtracking to the onset start is bounded to 40 ms** (`MAX_BACKTRACK`).
  Unbounded, `onset_backtrack` walks to the previous envelope minimum, which put
  slow bass notes 80 ms early — a sixth of a beat at 130 BPM;
- **strength is normalised on the band's 90th percentile, not its max**, with a
  0.15 floor: one freak-loud hit must not flatten every other event, and a
  detected event that maps to ~0 is a wasted event;
- near-simultaneous same-kind peaks are collapsed to the earlier time with the
  louder strength.

The stage logs per-kind counts and, as a sanity guard, kicks per beat inside
drum-active regions (`stems` drums curve > 0.5); outside 0.25–2.0 it adds a note
to `run.json`. The guard never filters the output — a detector that rewrites
itself to satisfy its own check is worse than one that is visibly wrong.

## Validation plots

`terrarium-plot out/song` writes five PNGs into `out/song/plots/`:

- `stems.png` — per-stem activity over the waveform envelope. **Look at this
  first.** If bass entering is not an obvious step in the bass channel, fix it
  here; nothing downstream can.
- `events.png` — one lane per event kind over its stem's activity curve (bar
  height is strength), plus a zoom on the 8 s window with the most beat-locked
  kicks, drawn against the beat and downbeat grid. The questions: do the kicks
  sit on the grid, and do the events stop where the instrument stops? Written
  only when the timeline has events.
- `novelty.png` — novelty4 / novelty16 / actChorus / recurStr against boundaries.
- `latent.png` — PC1–PC2 trajectory coloured by time, faceted per segment label
  (do the three choruses land in the same neighbourhood?).
- `ssm.png` — the 2D-FMC segment similarity matrix, when present.

Pass `--wav tracks/song.wav` to draw the waveform when the input was already WAV
(there is no `audio.wav` in the output directory in that case).

### What the first real run showed, and what is still weak

Free Fall (Remastered), 272.5 s, 130.4 BPM, 19 segments. Read alongside
`data/timelines/free-fall/plots/`.

- **`stems.png` is the strong one.** Drums silent until ~40 s, a build to ~0.6
  through 40–68 s, a hard step to 1.0 at 69 s (a labelled boundary), zero
  through the 147–170 s breakdown, back to 1.0 at 178 s. Vocals step in at 22 s
  and gate per phrase. Bass notches to 0 at the drops. Entrances read as steps —
  the phase-1 question is answered yes.
- **`events.png` is the other strong one.** 3506 events: 361 kick, 742 snare,
  916 hat, 907 bass, 580 vocal. In drum-active regions there are 359 kicks
  against 380 beats — **0.94 kicks per beat**, which is what four-on-the-floor
  should give. 343 of 361 kicks fall in a single eighth-of-a-beat bin (96 %
  within 60 ms of a beat), sitting a systematic ~24 ms early: the onset envelope
  leads the transient because its analysis window is 93 ms long. Snares and hats
  land on eighths, bass onsets are syncopated after the beat, vocal onsets are
  phase-uniform — all as expected. The 147–170 s breakdown drops to 1 kick and
  4 snares; the pre-40 s intro has no kick or snare at all.
- **`other` is the weak stem**: p05 0.34, below 0.1 for 0.6 % of the track. It
  is the pads/synths residue and it is essentially always on, so it carries
  little event information. Expect it to drive slow texture, not entrances.
- **`actChorus` tracks the labelled choruses** (peaks ~0.85 at 115–128 s and
  238–262 s, ~0.05–0.2 in intro and verses), which is also the proof that the
  activation rows are matched to the right labels.
- **`novelty4/16` go quiet through 70–130 s and 180–235 s.** Those sections are
  harmonically static, so a chroma SSM genuinely has nothing to say there;
  `robust_norm` over the whole track then lets the churny intro compress
  everything else. novelty16 does still peak at the big boundaries (69 s, 145 s,
  175 s).
- **`ssm.png` is close to useless as shipped**: off-diagonal cosine mean 0.909,
  max 0.953, so the whole matrix is one flat block. The 2D-FMC features share a
  large constant component (low-order FFT magnitude) that `log1p` does not
  remove, and cosine on top of that saturates. Fixes if it ever matters: drop
  the DC bin, or centre the feature matrix across segments before the cosine.
  Left alone for now — plan.md demotes recurrence to "emit when cheap, design
  nothing around it".
- **PC ordering is not variance ordering** after `to_latent`'s smoothing. PCA
  is applied first, then a 0.25 s zero-phase EMA, and that EMA attenuates PC2
  and PC4 to ~25–33 % of their amplitude while leaving PC1/PC3/PC5 at ~91–96 %
  — so the emitted `latent[:,1]` is *smaller* than `latent[:,2]`. Nothing is
  lost (all 64 dims ship), but `latent.png`'s PC1–PC2 view is squashed and any
  mapping that assumes "PC1 and PC2 are the two biggest axes" is wrong.
  Underlying cause: those components carry beat-rate (~2.2 Hz) content that the
  smoother is meant to remove.
- **The beat grid is clean except in the intro.** 534 of 592 inter-beat
  intervals sit in 0.45–0.50 s; the ~30 short ones (0.10–0.35 s) all fall in
  6.8–20.4 s, where the intro is sparse. Downbeats average 4.09 beats apart.

### The second track: Pink Loop

209.6 s, 125.0 BPM, 13 segments, 2095 frames, seed 1234 — the same command as
above, 62.6 s wall on the same machine with warm caches. Read alongside
`data/timelines/pink-loop/plots/`.

- **The 115–143 s break is the headline.** Drums fall from 1.0 to 0 over ~2 s at
  115 s and come back as a hard step at 143 s; the event lanes go with them (the
  break holds 1 snare and a handful of bass onsets against 1.5 k events in the
  rest of the track). That is "the drums dropped out" as a step, which is the
  phase-1 standard.
- **Vocals gate per phrase** — four blocks (22–37 s, 63–85 s, 126–145 s, and a
  short 176 s tag) with clean zeros between, and 164 vocal events that sit only
  inside them.
- **Bass notches to ~0.15–0.5 at each chorus boundary** (17 s, 53 s, 97 s,
  159 s, 189 s) and to 0 through 137–143 s, so the drop is legible in two
  channels at once.
- **1.11 kicks per beat** in drum-active regions (421 kicks / 379 beats) — inside
  the 0.25–2.0 guard band, and the zoom panel of `events.png` shows them landing
  on the beat lines. Slightly above free-fall's 0.94; the excess is not
  investigated further, since the guard's job is to catch a detector that has
  come loose, and 1.11 is not that.
- **`other` is again the flat one** (mean 0.82, rarely below 0.5): same pads
  residue, same conclusion — slow texture, not entrances.

## Determinism

Same track + same `--seed` + same machine ⇒ **bit-identical `timeline.bin`**,
and bit-identical demucs stem WAVs, verified across two runs into fresh output
directories. Re-verified when the events stage landed: adding it left
`timeline.bin` and `embedding.bin` byte-for-byte identical to the previous run
(the events live in the JSON, not the dense buffer). The detector itself draws
no random numbers, so the event list is a pure function of the stems. Change the seed and the stems change, because the seed reaches
demucs (below); everything else in the file is a deterministic function of the
audio.

The one thing that had to be pinned for this: demucs' `apply_model` defaults to
`shifts=1`, which draws a random time offset from the **unseeded stdlib
`random`** module, and All-In-One does not pass `shifts`. Before pinning it, two
runs with the same seed differed by up to 0.28 in the stem channels and 0.086 in
`actChorus` (All-In-One reads the demixed audio, so the stems' noise propagates
into its activations). `stages/structure.py` now seeds `random`, `numpy` and
`torch` from `cfg.seed` immediately before calling `analyze`.

MuQ inference on CUDA turned out to be bit-reproducible run to run without any
help: `embedding.bin` matched exactly even before the seeding fix.

## The server

`terrarium-server` is the same pipeline behind an HTTP API, so the web workbench
can hand it a file and then load the result exactly the way it loads a track
that shipped in `data/timelines/`. It is not on the runtime path: the browser
talks to it twice per track (list, then fetch) and works with it absent.

```bash
cd analysis
uv run --extra server terrarium-server            # http://127.0.0.1:8765
uv run --extra server terrarium-server --help     # analysis and native-export options
```

It writes into `<repo>/data/timelines/<slug>/` by default — the directory
`web/scripts/sync-data.mjs` copies from — so an uploaded track is a real track:
`npm run sync-data` bundles it like any other, and until then the web side
fetches it from the server.

| route | |
|---|---|
| `POST /analyze` | multipart `file=` a `.wav` or `.mp3` → `{jobId, trackId, …}`; 415 on anything else, 413 over 256 MB |
| `GET /jobs/{id}` | `{status: queued\|running\|done\|error, stage, progress, message, error}` |
| `GET /tracks` | `{tracks: [{id, title, duration, frames, tempo, events, hasAudio, version}]}` |
| `GET /tracks/{id}/{file}` | `timeline.json`, `timeline.bin`, `audio.wav`, `run.json` — nothing else |
| `GET /exports/capabilities` | native Windows worker/GPU/AV1 availability and renderer build id |
| `POST /exports` | immutable recipe v3 for an analyzed `trackId` → queued export job |
| `GET /exports/{id}` | export job metadata |
| `GET /exports/{id}/download` | completed MP4 only; never an arbitrary filesystem path |

- **Polling, not a websocket.** A job is one long burst with six coarse
  transitions in it and the client is a panel that redraws at human speed.
  Progress is read off the log the stages already write (`"<stage>: …"`), which
  is why no callback had to be threaded through `pipeline.run`.
- **`version` is a content hash** of `timeline.json` + `timeline.bin`, not an
  mtime. It is what the browser's offline cache keys on, and re-running the
  pipeline with the same seed produces identical bytes — an mtime would evict a
  multi-megabyte track for a run that changed nothing.
- **One job at a time.** Both heavy stages want the whole GPU.
- **Uploaded ids are slugs** (`"Pink Loop (2).wav"` → `pink-loop-2`), unique by
  suffix, and a re-upload never overwrites an existing track — the browser may
  be holding it in cache.
- **CORS** allows `http://localhost:<any>` and `127.0.0.1`, because vite moves to
  5174/5175 when 5173 is taken. Dev-only, and the bind is localhost by default.

### Where to run it

Wherever the model extras are, which on this machine means WSL2 (see the Windows
note above). The Windows side has the light deps only, so a job started there
skips every model stage and produces a timeline of zeros; `--strict` turns that
into a failed job instead. Pointing the WSL2 server at the Windows repo works
and is what was used to verify this:

```bash
wsl -d Ubuntu
cd ~/lmt/analysis && . .venv/bin/activate
uv pip install -e ".[dev,server]"
# 0.0.0.0, not 127.0.0.1: WSL2's localhost forwarding only picks up a listener
# bound to all interfaces. The browser on Windows then reaches it at localhost.
terrarium-server --host 0.0.0.0 --port 8765 \
    --data-dir /mnt/c/Users/<you>/wryskware/latent-music-terrarium/data/timelines
```

A 40 s excerpt goes queued → running → done in ~30 s that way, writing
`timeline.json` (55 KB), `timeline.bin` (400 × 73 × 4 = 116 800 B), `run.json`
and `audio.wav` into the track directory.

### Native Windows video export checkpoint

Run the API **natively on Windows** when using video export. The Node worker
uses Dawn/D3D12, FFmpeg uses NVENC, and the completed multi-gigabyte file stays
on the Windows filesystem. This avoids routing rendered frames or finished
videos through WSL. Heavy model analysis may still run in WSL as described
above; only the much smaller analyzed track artifacts need to cross that seam.

The current browser checkpoint emits **constant 120 fps AV1 NVENC SDR-debug
video with AAC-LC audio** at either 1920×1080 or 3840×2160. The workbench
labels both choices explicitly and defaults to the more practical 1080p path.
Neither is HDR. Real HDR10 now ships alongside them as
`av1-hdr10-1080p120` / `av1-hdr10-2160p120` — scene-linear PQ/BT.2020 graded and
GPU-packed to P010, encoded as 10-bit AV1 with the colour description in the
sequence header and ST 2086 in the container. Hosted/product compute, restart
recovery, disk quotas, and retention policy are later work.

Prerequisites on Windows:

- Node 22+ and the repository's pinned `webgpu` dependency;
- FFmpeg with `av1_nvenc` and AAC support, plus a compatible NVIDIA driver;
- the light Python/server environment from this directory; and
- an analyzed track containing `timeline.json`, `timeline.bin`, and
  `audio.wav` under `data/timelines/<track-id>/`.

Build the native worker from PowerShell:

```powershell
cd C:\path\to\latent-music-terrarium\web
npm ci
npm run build:worker
```

Then start FastAPI from a native Windows PowerShell. Pass all four native paths
explicitly so capability failures are actionable and output placement is never
inferred from WSL or the ambient shell:

```powershell
cd C:\path\to\latent-music-terrarium\analysis
uv run --extra server terrarium-server `
  --host 127.0.0.1 `
  --port 8765 `
  --data-dir "C:\path\to\latent-music-terrarium\data\timelines" `
  --node "C:\Program Files\nodejs\node.exe" `
  --ffmpeg "C:\tools\ffmpeg\bin\ffmpeg.exe" `
  --export-worker "C:\path\to\latent-music-terrarium\web\dist-worker\worker.mjs" `
  --export-dir "C:\path\to\latent-music-terrarium\data\exports"
```

Open the Vite app normally, expand `video export`, and wait for the capability
probe to report ready before pressing **Render video with current settings**.
The server owns the output filename, writes a `.partial` file during encoding,
publishes the MP4 only after success, and returns a download link. Node and
FFmpeg are per-job child processes; neither opens another HTTP port. Each queued
job snapshots its renderer bundle, timeline, and WAV so later source changes
cannot alter what it renders; those potentially large private snapshots are
removed at the terminal job state, while the bounded worker log is retained.

This checkpoint has no cancellation endpoint and no automatic cleanup policy.
Stop the server only when no export is running, and manage completed files in
the directory passed to `--export-dir` manually.

## Tests

```bash
uv run --extra dev --extra server pytest -q
```

Covers the emitter layout (against a hand-built buffer and against the shared
fixture), PCA determinism and its sign convention, the smoothing helpers, the
stem-activity computation, the structure stage's glue (label vocabulary lookup,
activation-matrix orientation, RNG pinning, activation decimation), the event
detector (band isolation and zero-phase alignment, click trains recovered at the
times they were placed, bands not stealing each other's hits, strength ordering
and its robust scale, bounded backtracking, silence and bleed producing nothing,
dedupe, the kicks-per-beat guard, and the manifest round trip), and the
guarantee that importing the package pulls in no model dependency. All on
synthetic inputs — no audio, no weights, no network.

The server's own tests (`test_server.py`, skipped without the `server` extra)
cover everything around the pipeline rather than the pipeline: the slug rules
the browser's `?track=` regex depends on, the upload gate, the served-file
whitelist and its media types, the content-hash `version` staying put across a
byte-identical rewrite, CORS, and the progress watcher — including that it does
not shadow `logging.Handler.lock`, which is what deadlocked the first live run.
