# Headless export handoff

## Correction — 2026-08-13 (HDR10 is AV1, not HEVC)

The HDR10 work described in the next section shipped on `hevc_nvenc` because the
implementing session concluded AV1 could not carry an HDR10 colour description.
**That conclusion was wrong**, and the two caveats it produced are both gone. The
cause was an old local FFmpeg, not a codec limitation. On FFmpeg 9 (verified by
running the pipeline's own generated argument list end to end):

- `av1_metadata` is the direct analogue of `hevc_metadata` and writes primaries,
  transfer, matrix and range into the AV1 sequence header. It is not new — it has
  been in FFmpeg since 4.3. The claim that "AV1 has no VUI bitstream filter
  equivalent" was never true;
- the CLI's `-mastering_display` **input** option attaches real ST 2086 side data,
  which the MP4 muxer writes as a proper `mdcv` box. The 254-line hand-rolled
  `moov` box-surgery module (`web/src/export/mp4-hdr-metadata.ts`) and the
  no-faststart constraint it forced both existed only to work around the absence
  of this option, and are deleted;
- AV1 needs no HDR-specific profile: Main already covers 10-bit, so there is no
  `main10` to request and no `-tag:v` to set.

Current state: profiles are `av1-hdr10-2160p120` / `av1-hdr10-1080p120`
(1080p is the default), `hevc_nvenc` is gone from the codebase entirely, and HDR
MP4s are ordinary `+faststart` files. The old `hevc-hdr10-*` ids are rejected
rather than aliased — a recipe is a reproduction contract, and re-pointing an id
at a different codec would make it lie. Capability gating now requires
`av1_nvenc` *and* P010 input *and* the `av1_metadata` filter.

Verified output: `codec=av1 profile=Main pix_fmt=yuv420p10le color_range=tv
color_space=bt2020nc color_transfer=smpte2084 color_primaries=bt2020`, ST 2086
mastering metadata at 1000 nits, CFR 120. Suites 298 web / 146 python, all green.

Still-open caveats from below: HDR playback on a real HDR display remains
unverified (ffprobe and decoded-pixel statistics only); no MaxCLL/MaxFALL, which
would have to be measured and this pipeline does not measure them; 4K120 HDR
throughput; analysis-job cancellation returns 409 by design.

## Update — 2026-08-13 (later the same day)

All three "Next priorities" below are implemented, tested, and committed:

- `6db254e` — persistent `ExportSession` controller; job state, download link,
  and duplicate-submission guard survive panel/simulation rebuilds.
- `e972202` — `DELETE /jobs/{id}` cancellation with terminal `cancelled` state,
  cooperative `CTRL_BREAK_EVENT` then bounded `taskkill /T /F` process-tree
  cleanup, worker SIGBREAK/SIGTERM/SIGINT handling with `.partial` removal, and
  a cancel button in the panel. Smoke-verified mid-render on native Windows.
- `7804257` — real HDR10: scene-linear HDR grade (paper white 203 / mastering
  peak 1000, recipe-bounded) → GPU PQ/BT.2020/P010 packing compute pass → HEVC
  Main10 NVENC with restamped VUI and a post-encode ST 2086 `mdcv` box. New
  profiles `hevc-hdr10-2160p120` / `hevc-hdr10-1080p120` (now the default),
  capability-gated on 10-bit encoder support; SDR debug profiles unchanged.

Suites: 305 web / 146 python, all green. Known caveats: HDR playback on a real
HDR display is still unverified (ffprobe + decoded-pixel statistics only); HDR
MP4s have no faststart (the mastering box is injected into the trailing
`moov`); no AV1 HDR profile (av1_nvenc drops VUI color tags with no bitstream
filter to restore them); 4K120 HDR measured ~8.7× slower than real time;
analysis-job cancellation returns 409 by design. Remaining product work is
unchanged from "Lower-priority hardening" below plus those caveats.

The rest of this document describes the state *before* that work and is kept
for context.

---

Status as of 2026-08-13: the first useful native-Windows rendering path is implemented, committed, and tested. A browser session can submit its current seed and settings to the local FastAPI service and render a full track as an AV1/AAC MP4. Both 1080p120 and honest SDR 4K120 profiles work. The remaining work is hardening and true HDR, not a blocker to creating renders now.

## Checkpoints

The work is on `main` in these commits, newest first:

- `da09c8b` — `web: add honest 4k sdr export profile`
- `06f91f7` — `web: render current sessions through native export API`
- `48ab645` — `web: render real tracks in native export worker`
- `ce22ade` — `web: start native headless export runtime`
- `7724aea` — `web: add 120 Hz timing and milkdrop visual repertoire`

Do not amend or squash these while resuming; they are useful recovery points.

## What works

- The browser captures a versioned recipe containing the current seed, live pin state, concrete simulation/config/extras, authored modulation state, impulses, rendering configuration, fixed particle budget, output profile, and exact timeline content version.
- Manual mode captures the current theta. Modulated mode captures authored base values rather than a transient modulation excursion.
- The browser UI has a **Render video with current settings** control and a profile selector:
  - `av1-sdr-debug-1080p120` (default/recommended): 1920x1080, CFR 120
  - `av1-sdr-debug-2160p120`: 3840x2160, CFR 120
- FastAPI exposes capabilities, job submission/status, export metadata, and download endpoints.
- Exports run through native Windows Node and FFmpeg, avoiding large WSL/Windows filesystem transfers.
- The worker renders real tracks deterministically, and FFmpeg muxes trimmed WAV audio as AAC-LC, 48 kHz, 192 kbps, with audio/video starting at zero and having deterministic duration.
- Jobs snapshot the exact worker bundle, timeline pair, and WAV at submission time. A queued job therefore cannot silently change when source assets are rebuilt.
- A single shared queue prevents competing GPU exports. Private job snapshots are removed at terminal state; a bounded worker log is retained.
- Renderer capabilities invalidate when the worker bundle changes.

Current output is deliberately labeled SDR debug. It is AV1 Main/yuv420p, not HDR10.

## Native Windows development flow

Build the browser and worker first:

```powershell
cd web
npm run build
```

Start the API from native Windows, supplying local paths for Node, FFmpeg, the worker, timelines, and exports. The exact executable locations are machine-specific:

```powershell
cd ..\analysis
uv run --extra server terrarium-server `
  --host 127.0.0.1 `
  --port 8765 `
  --data-dir "C:\path\to\latent-music-terrarium\data\timelines" `
  --node "C:\Program Files\nodejs\node.exe" `
  --ffmpeg "C:\path\to\ffmpeg.exe" `
  --export-worker "C:\path\to\latent-music-terrarium\web\dist-worker\worker.mjs" `
  --export-dir "C:\path\to\latent-music-terrarium\data\exports"
```

In another native Windows terminal:

```powershell
cd web
npm run dev
```

Open the app, choose/load a track, establish the desired seed and settings, expand the video-export section, select 1080p or 4K, and click **Render video with current settings**. The submitted export covers the full track; there is no UI range selector yet.

## Validation evidence

- Web tests: 255/255 passed.
- Strict TypeScript, browser Vite build, and worker SSR build passed. The only build warning was the existing client chunk-size warning.
- Python tests: 122 passed, with one benign Starlette/httpx deprecation warning.
- Native API 1080p smoke: 30 frames / 0.25 seconds, 1920x1080 AV1 with audio, job completed, snapshots cleaned, only bounded `worker.log` retained.
- Native direct 1080p smoke: 1.0 second, 1920x1080 AV1 yuv420p at 120 fps plus AAC 48 kHz; both streams start at 0 and end at 1.0 second; 3,987,977 bytes.
- Native direct 4K smoke: 240 frames / 2.0 seconds, 3840x2160 AV1 Main yuv420p at 120 fps plus AAC 48 kHz; both streams start at 0 and end at 2.0 seconds. Total elapsed time was 5.70 seconds including startup.

The 4K sample suggests roughly 2.85x slower than real time on this machine for a short render. A 3.5-minute track might therefore take around 10 minutes, but that is only an initial extrapolation; measure a longer steady-state render before treating it as a forecast.

Generated smoke outputs live under `web/exports/` and are ignored. Do not commit them.

## Next priorities

### 1. Persist the UI job lifecycle

This is the smallest and highest-value follow-up. `createExportFolder` currently owns local busy/status/watch state. Rebuilding the control panel (for example, after switching simulations) disposes that UI, loses its download link, and may allow a duplicate submission while the original job is still running.

Add a main-lifetime `ExportSession` or controller beside the long-lived `LocalExportClient`. It should support subscription/replay, preserve in-flight/completed/error state across panel rebuilds, deduplicate starts, and expose download metadata to late subscribers. Add tests for panel disposal, late subscription, duplicate submission prevention, completion/error replay, and unsubscribe behavior.

An incomplete untracked draft of `web/src/export/session.ts` was intentionally deleted during wrap-up. Reimplement from the committed state rather than assuming that scratch design was correct.

### 2. Cancellation and process-tree cleanup

Cancellation is not implemented. Add a cancellation endpoint and terminal `cancelled` state for queued and running jobs. On Windows, start the worker in a new process group, try a cooperative `CTRL_BREAK_EVENT`, wait briefly, then use `taskkill.exe /PID <pid> /T /F` as a bounded fallback. On POSIX, use a new session/process group and `killpg`. The Node worker should handle `SIGBREAK`, `SIGTERM`, and `SIGINT`, abort FFmpeg, remove `.partial` output, and make downloads unavailable for cancelled jobs.

Test queued cancellation, active cancellation, process descendants, partial-file cleanup, idempotency, and races with natural completion. Until this lands, do not stop the service while an export is running.

### 3. True HDR transport

4K is working, but the current transport is RGBA8 to AV1 Main/yuv420p SDR. True HDR needs a scene-linear BT.2020/PQ pipeline, a higher-precision frame transport, P010/Main10 encoding, correct stream/container color metadata, and inspection on an HDR-capable display/player. Keep the profile names honest throughout this work. See `docs/headless-export-proposal.md` for the intended direction.

## Lower-priority hardening and product work

- Add a UI range/sample selector for short test renders.
- If WAV content can change independently of `timeline.json` and `timeline.bin`, include the WAV identity in browser/server content-version negotiation. Submission-time snapshotting already protects queued jobs after acceptance.
- Decide retention, restart recovery, quotas, hosted compute, and publishing infrastructure later. These were explicitly out of scope for this pass.
- A future published-app button can use the same recipe/job contract; only the execution and storage backend needs to change.

## Repository notes for the next thread

- Read `CLAUDE.md`, `docs/plan.md`, and `docs/headless-export-proposal.md` before editing.
- Use `codebase-memory` after identifying symbols when callers, dependencies, impact, or execution paths matter. Then read exact source before editing.
- Preserve the pre-existing untracked directories `data/timelines/bug-fix-rush/allin1/` and `data/timelines/bug-fix-rush/spec/`; they are unrelated user work.
- Run focused tests while iterating, then finish with the complete web and Python suites and a real native Windows smoke render for any process-management or encoding change.

Suggested next-thread request:

> Resume from `docs/handoffs/headless-export-current-state.md`. First implement and test a persistent browser export-session controller across panel/simulation rebuilds. Then commit that checkpoint. If budget remains, implement cross-platform cancellation/process-tree cleanup as a separate commit. Preserve unrelated untracked timeline directories and use the codebase-memory MCP server according to AGENTS.md.
