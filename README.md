# Latent Music Terrarium

A browser-based audiovisual experience that turns a song into a living world.

Instead of reacting to volume or an FFT, it listens the way you do: music-
understanding models analyze the track offline — beats, sections, stems,
instrument entrances, a learned "character" of the sound — and compress all of
that into a compact timeline. In the browser, a WebGPU particle simulation
*lives on* that timeline. It has state and memory: it evolves continuously
rather than being recomputed frame by frame, so a drop hits like weather
arriving, not like a bar graph jumping.

Two halves, one contract:

- **`analysis/`** — the offline pipeline (Python). One track in,
  `timeline.json` + `timeline.bin` out: a 10 Hz, 73-dimensional latent
  timeline. Runs once per track; nothing here is on the runtime path.
- **`web/`** — the real-time side (TypeScript + WebGPU). Consumes the
  timeline, drives compute-shader simulations — particle life is the main
  act — and hosts the workbench where the look is tuned.

## Run it

You need [Node](https://nodejs.org/) 22+ and a browser with WebGPU
(recent Chrome or Edge; Firefox and Safari support is newer and less tested).
No Python required for this part — analyzed timelines for several tracks are
committed.

```bash
cd web
npm install
npm run dev
```

Open the printed URL (usually `http://localhost:5173`). The bundled tracks
ship **without audio** (rights), so they play silently — the timeline itself
keeps time and the visuals run exactly as they would with sound. To hear
music, process one of your own tracks (next section).

### Around the app

- **Track picker** — in the play tab; switches between analyzed tracks.
- **Space** play/pause · **h** halt · **arrow keys** seek.
- **s** toggles the settings workbench · **t** toggles the timeline and top
  bar · **double-click** the stage toggles fullscreen.
- **Workbench** (Tweakpane panel) — every knob of the simulation, palette,
  and music mapping, live. Tweaks autosave per sim; named save profiles live
  under the data folder. The whole panel is a first-class part of the
  project, not a debug leftover.
- **URL parameters**:
  - `?sim=` — `plife` (particle life, the main act), `physarum` (the
    current default), or one of the shader visuals (`nebula`, `tunnel`,
    `kaleido`, `plasma`).
  - `?track=` — a track id from the picker.
  - `?seed=` — pin the world seed. Seeds are random per run by default; the
    same track + seed replays the same world, which is what makes tuning
    loops possible.

## Process your own music

The pipeline turns a song into a timeline in about a minute of GPU time. Two
ways to run it.

### The easy way: the local server

`terrarium-server` wraps the pipeline in a local HTTP API. Start it, and the
web app grows an upload affordance — drop a WAV in the browser and it comes
back analyzed, with audio.

```bash
cd analysis
uv venv --python 3.10
uv pip install -e ".[dev,server,beats,structure,character]"
uv run --extra server terrarium-server
```

Then start the web dev server as above. The browser probes
`127.0.0.1:8765` once at startup; everything works without the server, it
just does more with it.

Or start both at once from the repo root, which is what `dev.ps1` is for:

```powershell
./dev.ps1                  # server, then vite once the port answers
./dev.ps1 -NoServer        # web only
./dev.ps1 -- -v            # extra args go to terrarium-server
```

```bash
./dev.sh                   # the same thing, for bash/zsh
./dev.sh --no-server
./dev.sh -- -v
```

Either one holds vite back until the server is listening (the browser only probes
once, and the models take a while to load), and Ctrl-C stops both.

### The manual way: the CLI

Same install as above (the `server` extra is optional here), then:

```bash
cd analysis
uv run terrarium-analyze path/to/song.wav -o ../data/timelines/my-song
cd ../web && npm run dev   # sync-data picks the new track up automatically
```

Output lands in `data/timelines/<id>/`; `npm run dev` and `npm run build`
copy every track found there into the app and rebuild the picker's index.

### What to expect

- **Model downloads**: the first full run fetches ~1.3 GB of weights
  (Beat This!, All-In-One, htdemucs, MuQ) into your Hugging Face / Torch
  caches.
- **Speed**: roughly a minute for a 4–5 minute track on a modern NVIDIA GPU;
  CPU-only works but is much slower.
- **Windows**: the light dependencies run natively, but the model stages
  should run under **WSL2** — see the Windows section of
  [`analysis/README.md`](analysis/README.md), which also covers running the
  stages piecemeal, reusing stems, determinism, and the timeline format in
  detail.
- **Format**: WAV in is best. MP3 works (it is decoded for you) but carries a
  small known timing offset from one of the models.
- Missing heavy dependencies don't crash the pipeline — a stage that can't
  import its model prints the install command and skips, leaving its
  channels at zero.

## Finding your way around

| Path | What lives there |
| --- | --- |
| `analysis/` | The offline pipeline: one stage per musical question (beats, structure, stems, events, character, recurrence) and the emitter that writes the timeline. Its README is the deep documentation. |
| `web/src/sim/` | The substrates. `plife/` is particle life — compute-shader simulation, per-particle luminance, HDR rendering. `physarum/` and `vizfx/` are secondary. |
| `web/src/mapping/` | Timeline → simulation: modulation routing, persistence machinery. |
| `web/src/ui/` | The workbench: panels, autosave, save profiles, palette folder. |
| `web/src/timeline/` | Loading, caching, and the track catalog (bundled + server). |
| `data/timelines/` | Analyzed tracks (timelines committed; audio not). |
| `docs/` | `plan.md` is the settled implementation plan, `roadmap.md` the current sequencing, `research/` the survey behind the stack choices, `handoffs/` scoped briefs for feature threads. |

The timeline byte layout is the contract between the two halves;
`data/timelines/synthetic/` is the shared fixture both sides' tests conform
to.

## Developing

```bash
cd web
npm test                # node --test — the full suite
npm run typecheck       # tsc --noEmit (tests are outside tsconfig — run both)

cd analysis
uv run --extra dev --extra server pytest -q
```

Things the codebase is opinionated about, which you'll meet quickly:

- **Persistence is by construction.** Config blocks are declared in a
  per-sim `BlockTable` (`web/src/mapping/blocks.ts`); adding a block without
  declaring how it persists is a compile error, and opting out requires a
  written reason. Don't fight this — it exists because "my tweak didn't
  save" was paid for four times.
- **Looks beat fidelity.** The deciding factor for any mapping change is
  that obvious musical events produce obvious visual effects and the result
  looks good — not analytical faithfulness.
- **Determinism is tooling, not product.** Seeds are random by default and
  pinnable so that scrub–tweak–replay loops work.

Fork it, add a substrate, retune the mapping — the seams between analysis,
timeline, and simulation are deliberately clean.

## Status

Active development, working toward a public demo. Part of the
[Wryskware](https://wryskware.dev) ecosystem — an independent repo with its
own stack and deployment.

No license has been chosen yet; until one is, all rights reserved.
