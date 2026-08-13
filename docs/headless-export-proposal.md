# Headless 120 fps HDR Export — Implementation Proposal

**Status:** local Windows SDR/debug checkpoint operational through the browser; HDR Gate 0 remains open
**Date:** 2026-08-13
**Primary profile:** 3840×2160, constant 120 fps, HDR10  
**Secondary profile:** 1920×1080, constant 120 fps, HDR10

Implemented in the first pass on 2026-08-12:

- native Dawn/D3D12 worker bundle and structured Gate 0 probe;
- compilation of a current project WGSL shader, offscreen `rgba16float`
  rendering, and ring-buffered 4K readback;
- FFmpeg/NVENC capability detection;
- canvas-free GPU runtime context plus browser surface adapter;
- explicit host-supplied render-frame timing;
- shared simulation/modulation/impulse bundle construction;
- versioned, bounded export recipe validation and deterministic 120 Hz export
  scheduling primitives.

Implemented in the second pass on 2026-08-12:

- strict, bounded `--request <absolute-request.json>` worker protocol with
  server-owned native paths and no FFmpeg argument surface;
- one shared timeline validator behind browser fetch and Node filesystem
  transports;
- recipe-driven construction of the complete concrete simulation, modulation,
  impulse, palette, and render state without reading browser persistence;
- start-from-zero deterministic range scheduling, native offscreen rendering,
  a bounded three-slot staging/readback ring, encoder backpressure, structured
  progress, failure cleanup, `.partial` output, and atomic publication;
- an explicitly temporary RGBA8 → AV1 NVENC SDR/debug transport; and
- a real `pink-loop` Particle Life artifact: 1920×1080, constant 120 fps, 600
  frames / 5.000 seconds, AV1 Main `yuv420p`, rendered and encoded natively on
  Windows in 7.75 seconds on the development machine.

Implemented in the local-product checkpoint on 2026-08-13:

- export recipe v3 and browser capture of the concrete simulation, seed/pin
  state, modulation config and authored base vector, impulse config, palette,
  render config, and fixed particle budget; the captured base is distinct from
  any transient music-driven excursion visible when the button is pressed;
- native Windows FastAPI supervision of a short-lived Node/Dawn worker and its
  FFmpeg child, with build-aware capability probing, immutable per-job
  worker/timeline/audio snapshots, bounded NDJSON progress, bounded diagnostics,
  and atomic output publication;
- `POST /exports`, job/export polling, completed-file metadata, and download
  routes, backed by the server's existing single-worker executor;
- a browser workbench control labelled **Render video with current settings**,
  with explicit 1080p/4K 120 fps SDR-debug choices, progress, and a completed
  download link; and
- deterministic audio range selection and AAC-LC muxing from the analyzed
  track's trusted `audio.wav` into the debug MP4.

The v3 4K SDR-debug smoke rendered 240 frames / 2.000 seconds at 3840×2160 in
5.70 seconds on the development machine. FFprobe reported AV1 Main `yuv420p`
at constant `120/1` plus 48 kHz AAC, with both streams starting at zero and
ending at exactly 2.000 seconds. This proves the local 4K engineering path; it
does not satisfy any HDR acceptance criterion.

Two identical requests produced the same first decoded frame, the same frame
count, and visibly matching large-scale seeded composition, but their later
decoded pixels were not bit-identical. Particle Life is chaotic enough to
amplify minute GPU/encoder differences immediately; strict same-device pixel
repeatability therefore remains an explicit investigation rather than a claimed
Phase 2 property.

Still outstanding before Gate 0 exits: perform scene-linear BT.2020/PQ
conversion, feed P010 or another validated 10-bit transport through an AV1 or
HEVC Main10 encoder, create the 5–10 second 4K120 test file, and inspect its
constant frame rate, HDR metadata, levels, and playback on the target display.

The current file is **not HDR**, despite the future-profile identifier still
traveling through the internal request: it is RGBA8 readback encoded as AV1 Main
`yuv420p`, with AAC-LC audio. Proper 4K/HDR profiles, production compute and
deployment, cancellation/process-tree handling, restart recovery, disk quotas,
and an explicit retention policy remain later work. The current UI/API is a
local engineering checkpoint, not a published rendering service.

## Executive decision

Build the exporter as a **headless Node worker using native Dawn/WebGPU**, launched
and supervised by the existing Python/FastAPI server for each export job.

The browser and worker must import one shared TypeScript renderer. There must not
be a Python, Rust, or second TypeScript reimplementation of Particle Life or any
other simulation. The browser remains the interactive preview and authoring
surface; Node replaces the canvas and audio clock with an offscreen render target
and deterministic export clock.

The public deployment remains one API service:

```text
browser
   │ HTTP
   ▼
Python / FastAPI                         the only public listener
   │ starts one export job
   ▼
Node headless render worker              no HTTP port, no Chromium
   │ streams frames
   ▼
FFmpeg / NVENC                           encoding and audio mux only
   │
   ▼
server-controlled exports directory
```

This is a multi-process implementation, but not a multi-service deployment.
There is no proxy, service discovery, second port, or independently managed Node
daemon. Python owns the child processes and exposes job status and downloads
through the API it already serves.

## Why this architecture

### The synchronization problem is more important than the porting problem

A native Rust/C++ renderer or a Python `wgpu` renderer could perform the work.
The unacceptable cost is that it would duplicate the CPU side of the visual
runtime:

- simulation construction and configuration;
- buffer, bind-group, and pipeline layout orchestration;
- modulation and driver-bank behavior;
- impulse envelopes and event consumption;
- timeline sampling;
- seed and population behavior;
- simulation-specific render passes and post-processing.

That code changes whenever a visual is added or tuned. Sharing WGSL alone would
not keep two renderers behaviorally synchronized.

Node can run the existing TypeScript against a native WebGPU implementation.
Dawn is the implementation beneath Chromium, but it can also run directly over
D3D12 or Vulkan without Chromium or a DOM. The `webgpu` Node package publishes
the Dawn Node binding. It should be pinned to an exact tested version because
the binding is less mature than browser WebGPU.

### The repository already has a useful seam

The current `Sim` contract separates fixed simulation ticks from rendering:

```ts
tick(frame, simTick)
render(encoder, targetView)
```

The simulation classes already render into a supplied `GPUTextureView`, not
directly into a canvas. Most browser coupling is concentrated in:

- `web/src/main.ts` — DOM, rAF, Web Audio, runtime construction, UI;
- `web/src/gpu/context.ts` — canvas acquisition and resize;
- `web/src/audio/clock.ts` — preview playback clock;
- preset import/export UI.

Particle Life, Physarum, VizFX, mapping, modulation, impulses, timeline code,
and WGSL are mostly platform-neutral already. The meaningful leaks are
wall-clock reads in render feedback/auto-exposure and Vite's `?raw` shader
imports. Both are contained refactors rather than ports.

## Product behavior

### Browser workflow

1. The author chooses or uploads/analyzes a track.
2. They select a simulation, seed, modulation settings, and render settings in
   the existing browser workbench.
3. They preview interactively at the canvas's natural variable frame rate.
4. They press **Export** and select one of:
   - `4K · 120 fps · HDR10`
   - `1080p · 120 fps · HDR10`
5. The browser submits an immutable export recipe to FastAPI.
6. The UI polls the existing job endpoint and shows queue/render/encode
   progress.
7. On completion it offers a download link. The same file is available in the
   configured local exports directory.

An export always starts from simulation time zero. Scrubbing the preview does
not change the export start unless a future explicit range-export feature is
added.

### Track inputs

Version 1 exports an **analyzed track ID**, not an arbitrary bare WAV. The
simulation needs `timeline.json` and `timeline.bin` in addition to audio.

An uploaded WAV therefore follows the existing sequence:

```text
upload WAV → analysis job → track ID → export job
```

The UI can later offer a convenience operation that chains those jobs, but the
renderer itself should consume the same finished track contract as the browser.
This avoids a second analysis path.

## Timing model

### Correction to preserve

The preview canvas is not capped at 60 fps. It renders at the variable rate of
`requestAnimationFrame`, and its displayed FPS is the actual observed render
rate.

The app clock is independently fixed at 120 Hz (implemented after this proposal
was first drafted):

- `TICK_HZ = 120` and `SECONDS_PER_TICK = 1 / 120` in `web/src/timing.ts`;
- the audio clock drains zero, one, or several fixed ticks per browser frame;
- Particle Life uses a true 1/120-second tick/substep;
- Physarum and VizFX preserve their authored 60-step-per-second behavior by
  consuming one model step every two app ticks;
- render feedback and auto-exposure currently use variable render-frame time.

At a 120 Hz preview, Particle Life can produce a distinct integrated state for
every rendered frame. Physarum and VizFX preserve their existing 60 Hz model
cadence, while render-domain feedback and grading still run on every frame. At a
lower preview rate, multiple app ticks may run before one render.

### Export parity mode

The first exporter must preserve that relationship:

```text
output cadence:       120 frames/second
app-clock cadence:     120 ticks/second
render delta:          1/120 second
app tick delta:         1/120 second
```

For output frame `n`:

```text
frameTime     = n / 120
targetAppTick = floor(n * 120 / 120) = n
```

The worker advances every missing app tick in order, then renders one frame.
Particle Life integrates on each tick; the existing substrate cadence gate keeps
Physarum and VizFX at 60 model steps/second. Integer counters or rational
arithmetic must be used; accumulating a floating-point `1/120` for an entire
track is unnecessary and invites drift.

The output frame count is:

```text
ceil(trackDurationSeconds * 120)
```

This produces a frame beginning at every 120 Hz sample time before the end of
the audio. The muxer uses the audio as the authoritative duration and trims the
container cleanly at the end.

### 120 Hz Particle Life correction

The later, settled 120 Hz app-clock change deliberately made Particle Life a
true 120 Hz integrator while compensating Physarum and VizFX back to their
authored 60 Hz step cadence. Export parity means reproducing that current runtime
behavior; it must not revive the proposal's original 60 Hz Particle Life state.

### Explicit render time

Render-dependent code must not call `performance.now()` internally. Extend the
render contract with an explicit frame description, for example:

```ts
interface RenderFrame {
  frameIndex: number;
  timeSeconds: number;
  deltaSeconds: number;
}
```

The browser supplies measured rAF time. The exporter supplies exactly 1/120.
Snapshot-age timestamps that are purely UI concerns can remain outside the
deterministic runtime.

The Particle Life adaptive FPS governor is disabled during export. Export uses
the authored particle budget captured in the recipe; it must not reduce quality
because an offline frame took longer than 8.33 ms to compute.

## One shared renderer

### Required extraction

Extract a platform-neutral runtime from `main.ts`. A suitable ownership split is:

```text
TerrariumRuntime
├── TimelineSampler / driver bank
├── selected simulation
├── Modulator
├── ImpulseEngine
├── fixed-tick advance
└── render to caller-supplied GPUTextureView

BrowserHost
├── DOM and panels
├── Web Audio clock
├── rAF timing
├── canvas swapchain
└── local persistence and sim switching UI

ExportHost
├── recipe validation
├── Dawn device
├── fixed 120 Hz clock
├── offscreen targets / readback ring
└── encoder transport and progress
```

The existing `buildSimBundle` logic is specifically important to share. It
contains ordering constraints for stem/accent channels, base values, modulation
mode, seed changes, and impulse wiring. Copying that construction sequence into
the worker would recreate the synchronization problem this design is intended
to solve.

### GPU context split

`GpuContext` currently requires `HTMLCanvasElement` and `GPUCanvasContext` even
though simulations chiefly need the device, dimensions, format, and feature
flags. Split it into a core context and browser surface:

```ts
interface GpuRuntimeContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  width: number;
  height: number;
  format: GPUTextureFormat;
  float32Filterable: boolean;
}

interface BrowserGpuContext extends GpuRuntimeContext {
  canvas: HTMLCanvasElement;
  gpuCanvasContext: GPUCanvasContext;
  resize(): void;
}
```

Node constructs the core context with fixed export dimensions. It renders to
offscreen textures and never creates a canvas.

### Shader packaging

The source shaders remain the only shader copies. Add a Node-targeted build
entry through Vite's SSR/library build, or another bundler configuration that
supports the existing `*.wgsl?raw` imports. Externalize the native `webgpu`
package from that bundle.

Do not add a runtime script that copies WGSL into a second directory. Build-time
embedding from the same files is the synchronization guarantee.

### Recipe-driven state

The worker must not infer authoring state from browser `localStorage`. The
browser serializes the current state into a versioned recipe. The same schema
should also back preset import/export so there is one definition of a renderable
state.

At minimum the recipe captures:

- schema version and renderer build ID;
- track ID and track content version;
- concrete simulation ID;
- seed and whether it was pinned;
- complete simulation/base configuration;
- modulation configuration and driver gains;
- impulse/event-response configuration;
- render/post-processing configuration;
- fixed authored particle budget;
- output profile and encoder choice.

Milkdrop/VizFX auto-advance is explicit. The current recipe exports only the
currently selected concrete visual. If sequence export is added, the recipe must
contain the ordered repertoire, boundary behavior, and dwell policy; it must not
depend on mutable browser session state.

## Proposed API

Use the same polling model as analysis jobs.

### Capabilities

```http
GET /exports/capabilities
```

Example response:

```json
{
  "available": true,
  "profiles": ["av1-sdr-debug-2160p120", "av1-sdr-debug-1080p120"],
  "gpu": "NVIDIA GeForce RTX 5090",
  "backend": "d3d12",
  "encoders": ["hevc_nvenc", "av1_nvenc"],
  "rendererBuild": "<build-id>",
  "reason": ""
}
```

The server can compute this once at startup with a short worker probe and cache
it. Missing Node, Dawn, FFmpeg, GPU adapter, or 10-bit encoder support should
disable the button with a useful reason rather than failing after a long render.

### Start export

```http
POST /exports
Content-Type: application/json
```

Conceptual request:

```jsonc
{
  "trackId": "pink-loop",
  "recipe": {
    "version": 3,
    "rendererBuild": "<build-id>",
    "track": {"id": "pink-loop", "contentVersion": "<track-content-hash>"},
    "sim": "plife",
    "seed": 123456789,
    "simulation": { "...": "complete current state" },
    "modulation": { "...": "complete current state" },
    "impulses": { "...": "complete current state" },
    "render": { "...": "complete current state" },
    "output": {
      "profile": "av1-sdr-debug-2160p120",
      "encoder": "av1_nvenc",
      "paperWhiteNits": 203,
      "masteringPeakNits": 1000
    }
  }
}
```

Response uses the existing job vocabulary, extended with `kind` and export
metadata:

```json
{
  "jobId": "a1b2c3d4e5f6",
  "kind": "export",
  "status": "queued",
  "stage": "queued",
  "progress": 0,
  "trackId": "pink-loop",
  "exportId": "<opaque-id>"
}
```

### Poll and download

```http
GET /jobs/{jobId}
GET /exports/{exportId}
GET /exports/{exportId}/download
DELETE /jobs/{jobId}             # cancellation; can be a follow-up milestone
```

Completed job metadata should include filename, byte size, duration, resolution,
frame rate, codec, HDR metadata summary, and download URL. Do not expose an
absolute server filesystem path through a published API. The server log can
print it for local use, and the exports directory is configurable.

## Python-to-Node job protocol

Launch the worker with `shell=False`, an explicit executable, explicit working
directory, and a small fixed argument set:

```text
node <built-worker.mjs> --request <absolute-internal-request.json>
```

Paths and the large recipe live in the request file, not in command-line JSON.
Python creates that file inside the job's private work directory after validating
all browser-controlled values.

The worker writes newline-delimited JSON messages to stdout:

```jsonc
{"type":"ready","adapter":"...","backend":"d3d12"}
{"type":"progress","stage":"render","frame":1200,"frames":14400}
{"type":"progress","stage":"mux","progress":0.98}
{"type":"result","path":"<internal path>","frames":14400}
{"type":"error","stage":"render","message":"..."}
```

Human-readable diagnostics go to stderr. Python is the authority for public job
status and translates these events into the existing polling response.

The worker writes `filename.mp4.partial` and atomically renames it only after
FFmpeg exits successfully and output validation passes. Failed/cancelled jobs
remove partial output but retain a bounded diagnostic log.

Node owns its FFmpeg child and handles termination. Python owns Node. Cancellation
must terminate the whole job process tree; on Windows this may require a Job
Object or an explicit cooperative shutdown period before force termination.

## GPU scheduling

Analysis and export both want most of the GPU. Running demucs/MuQ and a 4K render
simultaneously will produce poor latency and may exhaust VRAM even on a large
card.

Add one server-level GPU lease, serialized by default across:

- model-heavy analysis stages;
- the complete lifetime of a Node export worker.

Separate job queues may still exist for clean status reporting, but only one
GPU-heavy job should hold the lease. Multi-GPU routing can be added later by
assigning a lease/device ID per worker.

## HDR10 output

### Current gap

The renderer already composites and blooms in `rgba16float`, so it preserves
high-range scene values internally. Its current final grade pass tone-maps,
clamps to 0–1, and gamma-encodes for an 8-bit browser swapchain. Labeling that
output HDR would only create an HDR-tagged SDR video.

### Required HDR path

Add a separate final output path that performs:

1. creative exposure/bloom/grade in scene-linear RGB;
2. scene-linear source primaries to BT.2020 conversion;
3. HDR highlight mapping with configurable paper white and mastering peak;
4. ST 2084/PQ encoding;
5. BT.2020 non-constant-luminance RGB-to-YCbCr conversion;
6. 4:2:0 chroma downsampling;
7. 10-bit P010 packing;
8. Main10 video encoding with matching container metadata.

Recommended initial defaults:

- paper white: 203 nits;
- mastering peak: 1000 nits;
- color primaries: BT.2020;
- transfer: SMPTE ST 2084/PQ;
- matrix: BT.2020 non-constant luminance;
- range: limited, unless the entire conversion/encoder path is deliberately
  validated as full range.

Paper white and mastering peak belong in the export recipe so the defaults can
be art-directed later. Static mastering-display and content-light metadata must
match the chosen policy. If MaxCLL/MaxFALL are claimed from content, measure them;
do not invent measured values.

### Encoder profiles

Default to **HEVC Main10 in MP4** for the broadest HDR playback compatibility.
Offer AV1 Main10 as an opt-in where supported. Both profiles use constant 120 fps
timestamps.

Audio should be encoded from the track WAV at 48 kHz, using AAC-LC for the MP4
deliverable. An optional archival Matroska profile can retain lossless audio
later.

### Frame transport

The simplest correctness path is asynchronous GPU readback through a ring of
staging buffers, with FFmpeg consuming frames under backpressure. Do not map the
same buffer while the GPU is writing it.

At 4K120, transfers are substantial:

- RGBA16F: about 66.4 MB/frame, 8.0 GB/s at real time;
- RGB48: about 49.8 MB/frame, 6.0 GB/s;
- P010: about 24.9 MB/frame, 3.0 GB/s.

These numbers make GPU-side conversion to P010 desirable, but real-time export
is not a requirement. Implement the clear correctness path first and measure it.
If readback dominates, add a compute pass that writes encoder-ready P010 into a
storage/readback buffer. True zero-copy WebGPU-texture-to-NVENC requires native
D3D12/Vulkan resource interop that the standard Node WebGPU surface does not
promise; it should be treated as an optimization requiring a native addon, not
as a prerequisite for the exporter.

A prior synthetic local encoder probe on the RTX 5090 reached approximately
131 fps for 4K120 AV1 Main10 at NVENC preset P5 and 99 fps at P7. That proves the
encoder is in the right class, not that the complete simulation/readback/HDR
pipeline will render in real time.

## Build and deployment

### One deployable API, several packaged runtimes

The backend deployment needs:

- the existing Python environment and FastAPI application;
- a supported Node runtime;
- the built headless-render worker bundle;
- a pinned native Dawn Node package for the host platform;
- FFmpeg with the chosen 10-bit hardware encoder;
- a compatible GPU driver exposed to all child processes;
- a writable export/work directory with retention limits.

This can live in one container or machine image and start only Uvicorn. Node and
FFmpeg are job children, not daemons. A reverse proxy remains optional hosting
infrastructure, not part of the application architecture.

### Platform risk

Dawn supports native D3D12 and Vulkan, but the exact target environment still
needs a smoke test. Windows/D3D12 is likely the easiest local path. A Linux or
WSL deployment needs a working Vulkan adapter visible to the Node binding as
well as NVENC access.

Do not commit to a hosting provider or container base image until Gate 0 below
has compiled a current project shader, rendered offscreen, and encoded Main10 on
the actual target environment.

## Security and operational limits

The export endpoint may eventually be reachable on a hosted service, so it must
not be a general-purpose renderer or FFmpeg command surface.

- Server selects all input/output paths.
- Track ID is resolved through the same whitelist logic as existing track files.
- Output filename is sanitized and made unique; the browser cannot provide a
  directory.
- Profile, resolution, frame rate, codec, duration, particle cap, and HDR values
  are schema-bounded.
- No user-provided FFmpeg arguments, shader source, module path, or executable.
- Request recipe and track version are immutable after queueing.
- Maximum queued jobs, maximum track duration, disk quota, and retention policy
  are configurable.
- Downloads are served only from completed export records, not arbitrary paths.
- Child environment and working directory are explicit.
- Partial files never appear as successful downloads.

## Proposed source layout

Names are illustrative; preserve repository conventions during implementation.

```text
web/src/runtime/
  terrarium-runtime.ts       shared construction, advance, and render
  recipe.ts                  versioned runtime/export state schema

web/src/gpu/
  runtime-context.ts         canvas-free GPU contract
  browser-context.ts         current canvas/swapchain behavior

web/src/export/
  worker.ts                  Node CLI and structured progress protocol
  dawn-context.ts            native adapter/device setup
  export-clock.ts            rational 120/60 scheduler
  frame-readback.ts          staging ring and encoder backpressure
  hdr-output.ts              HDR output pipeline orchestration
  shaders/                   only new HDR conversion/packing shaders

analysis/src/terrarium_analysis/
  export_jobs.py             validation, queue, child supervision, downloads

data/exports/                default local output; ignored by source control
```

The existing simulation and shader directories remain shared and are imported
by both browser and worker entries.

## Delivery plan

### Gate 0 — prove the native platform

**Checkpoint:** partially proven on native Windows/D3D12. Current project WGSL,
offscreen render/readback, and AV1 NVENC work, but the required Main10 HDR file
and display/metadata inspection do not yet exist. Gate 0 remains open.

Before refactoring the application:

- pin and load the Dawn Node package;
- enumerate the intended high-performance adapter;
- prove D3D12 on Windows and/or Vulkan on the deployment target;
- compile at least one current project WGSL shader;
- render an offscreen `rgba16float` test sequence;
- read back a buffered sequence without per-frame queue stalls;
- produce a short 4K120 Main10 file through the intended encoder;
- inspect pixel format, constant frame rate, color metadata, and visible HDR
  behavior.

**Exit condition:** a 5–10 second test file plays as HDR and the adapter/encoder
path works in the actual environment. Throughput is measured but need not yet be
real time.

### Phase 1 — extract the shared runtime

**Checkpoint:** complete. Browser and Node share the runtime contexts,
simulation construction, timing inputs, recipe validation, timeline behavior,
post-processing, and source WGSL.

- Split canvas-free GPU context from browser surface.
- Extract `buildSimBundle` and fixed advance logic from `main.ts`.
- Inject render-frame timing into simulation/post-processing code.
- Create a versioned recipe serializer/deserializer.
- Keep the browser behavior and variable rAF timing unchanged.
- Add a Node-targeted bundle that consumes the same WGSL imports.

**Exit condition:** the browser still behaves as before, and a Node smoke program
constructs the same selected simulation from a serialized browser recipe.

### Phase 2 — deterministic 1080p120 worker

**Checkpoint:** the SDR/debug engineering path is operational at 1080p120 with
AV1 NVENC, AAC-LC, progress, bounded readback/backpressure, and atomic
publication. Same-device strict later-frame pixel repeatability is still under
investigation and is not claimed.

- Implement fixed 120 render / 120 app-clock scheduling; existing substrate
  cadence keeps Physarum/VizFX model steps at 60 Hz.
- Render from time zero through a short selected range.
- Disable adaptive preview quality behavior.
- Add staging-buffer readback, progress, backpressure, and failure cleanup.
- Initially allow an SDR/debug encode only as an engineering aid; it is not the
  product deliverable.

**Exit condition:** repeated exports on the same device/build have the same frame
count and visually matching evolution, with no canvas or Chromium process.

### Phase 3 — FastAPI orchestration

**Checkpoint:** the local Windows subset is operational: build-aware capability
probe, immutable queued inputs, create, poll, metadata, download, one
server-owned executor, and private native request paths. Cancellation, restart
recovery, quotas, and retention are not implemented, so this phase is not
production-complete.

- Add capabilities, create-export, metadata, poll, and download routes.
- Add a single export queue and shared GPU lease with analysis.
- Validate recipes and track content versions.
- Supervise Node/FFmpeg process lifetime and partial output.
- Make output root, queue limits, and retention configurable.

**Exit condition:** an API request starts a job, polling reflects structured
progress, and the completed file is downloadable or discoverable locally.

### Phase 4 — real HDR10 and audio

**Checkpoint:** audio trimming and AAC-LC muxing are implemented for the debug
MP4. Scene-linear BT.2020/PQ, P010, Main10 validation, HDR metadata, and both
real output profiles remain outstanding.

- Implement the scene-linear-to-BT.2020/PQ path.
- Validate P010 and Main10 encoding.
- Attach correct stream/container color metadata.
- Mux the server track WAV with deterministic start/end sync.
- Add 1080p120 and 4K120 HDR profiles.

**Exit condition:** both profiles pass the acceptance checks below on an HDR
display and with `ffprobe`/MediaInfo.

### Phase 5 — browser export UX

**Checkpoint:** a local-only 1080p/4K 120 fps SDR-debug selector captures
recipe v3, checks capabilities, submits the job, reports progress/errors, and
exposes the download. HDR selection, cancellation, and published-compute UX
remain outstanding.

- Add the export button/profile selector.
- Capture the complete current recipe explicitly.
- Show unavailable reasons, queue/render/encode progress, errors, cancellation,
  and download.
- Warn if the browser build and server renderer build are incompatible.

### Phase 6 — optimize only from measurements

Possible optimizations, in order:

1. deeper staging/readback ring and larger encoder writes;
2. GPU-side P010 conversion and packing;
3. overlap simulation frame `n+1` with mapping/encoding frame `n`;
4. persistent Node worker to amortize device/pipeline initialization;
5. native zero-copy NVENC interop.

Do not begin with step 5. It creates the largest platform-specific maintenance
surface and does nothing for renderer synchronization.

## Acceptance criteria

### Shared behavior

- Browser and worker import the same simulation, modulation, impulse, timeline,
  post-processing, configuration, and WGSL modules.
- No copied or translated simulation implementation exists.
- Existing browser preview remains variable-rate and audio-clock-driven.
- Export uses the exact serialized authoring state and reports its recipe/build
  versions in output metadata or a sidecar manifest.

### Timing and media

- Output reports constant 120/1 fps, not variable frame rate or nominal 120 with
  irregular timestamps.
- The app clock advances at 120 ticks/second in parity mode; Particle Life
  integrates at 120 Hz and Physarum/VizFX retain 60 model steps/second.
- Frame count is derived deterministically from audio duration.
- Audio starts at time zero and remains perceptually synchronized at early,
  middle, and final track landmarks.
- A one-minute export contains exactly 7,200 output frames before any final
  mux-duration trim.

### HDR

- Video is 10-bit Main10 with a 4:2:0 10-bit pixel format.
- Stream/container metadata reports BT.2020 primaries, PQ transfer, and the
  intended BT.2020 matrix/range.
- HDR is generated from unclamped scene-linear values, not from the existing
  8-bit SDR grade.
- A test ramp and representative bright simulation frame show no unexpected
  clipping, banding, range lift/crush, or chroma offset.
- The file triggers HDR playback on the target display/player.

### Jobs and deployment

- Only FastAPI listens publicly; Node and FFmpeg open no ports.
- Only one GPU-heavy job runs by default.
- Failure and cancellation terminate child processes and leave no downloadable
  partial file.
- The completed file can be downloaded through FastAPI and found in the
  configured export directory.
- A server restart does not mistake a stale partial for a completed export.

## Known risks and mitigations

| Risk | Assessment | Mitigation |
|---|---|---|
| Native Dawn Node binding differs from browser WebGPU | Medium | Gate 0, exact version pin, small platform adapter, compile all shaders in CI/smoke tests |
| WSL/Linux cannot expose the desired adapter/backend | Medium | Prove target environment first; use Windows/D3D12 locally if needed |
| 4K120 readback is slower than real time | High likelihood, acceptable | Offline job semantics, staging ring, GPU P010 path after measurement |
| HDR looks like bright SDR or has incorrect levels | Medium/high | Separate HDR grade, test patterns, metadata inspection, real HDR display review |
| Browser and worker state serialization drifts | Medium | One versioned recipe schema shared by both builds; reject incompatible versions |
| Analysis and render compete for VRAM | High if unscheduled | One FastAPI-owned GPU lease by default |
| Node or FFmpeg orphaned on cancellation | Medium, platform-specific | Cooperative shutdown plus process-group/Job-Object cleanup |
| Cross-GPU output is not bit-identical | Expected | Promise deterministic timing and same-device/build repeatability, not cross-GPU bit identity |

## Explicitly rejected or deferred approaches

- **Headless Chromium:** feasible and maximizes browser fidelity, but unnecessary
  once native Node WebGPU is proven and contrary to the desired renderer process
  model.
- **Python/Rust/C++ simulation port:** rejected because it creates two CPU-side
  renderers that must be manually synchronized.
- **Embedding Node/V8 into Python:** rejected as much more fragile than a
  supervised child process.
- **Persistent Node HTTP sidecar:** deferred. It adds a port and lifecycle with
  no initial product benefit. A persistent non-HTTP worker can be considered if
  cold-start cost becomes material.
- **Native zero-copy encoder integration first:** deferred until readback is
  measured and shown to be the limiting cost.
- **120 Hz physics silently enabled for export:** rejected; it changes the
  simulation rather than merely increasing output temporal resolution.

## External technical basis

- [Dawn](https://github.com/google/dawn) is a native WebGPU implementation with
  D3D12, Vulkan, Metal, and OpenGL backends and is the implementation underlying
  Chromium WebGPU.
- [WebGPU.org's implementation list](https://webgpu.org/) lists Dawn as available
  for Node and native use.
- The [`webgpu` npm package](https://www.npmjs.com/package/webgpu) publishes a
  prebuilt Dawn Node binding. Treat its exact tested release as a project-pinned
  native dependency.

## Handoff summary

The implementation should optimize for **one visual runtime**, not one operating
system process. FastAPI remains the single public API and job authority. A Node
worker exists solely to execute the same TypeScript/WebGPU renderer the browser
previews, with an offscreen target and exact clock. FFmpeg exists solely to
encode and mux that output.

The first irreversible decision is not the API or codec; it is the shared-runtime
extraction. If an implementation begins by copying `buildSimBundle`, rewriting
Particle Life, or teaching Python the mapping system, it has violated the central
constraint of this proposal.
