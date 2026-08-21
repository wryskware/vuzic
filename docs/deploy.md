# Deploying the demo

The public cut of the app lives at **https://dreams.wryskware.dev/**. This is
the runbook for pushing a new one. `docs/roadmap.md` item 10 records *why* the
pieces are shaped this way; this file is the *how*.

## The one command

```bash
tools/deploy.sh
```

From anywhere in the repo. It builds the publish cut, uploads it, and
invalidates the CDN. It is idempotent — run it as often as you like.

```bash
tools/deploy.sh --no-build   # upload whatever is already in web/dist
```

Use `--no-build` when you have already built (or when you are re-uploading to
test a CDN or DNS change and want to be sure the bytes did not move).

## Prerequisites

- `aws` CLI, authenticated to account `040630189325` with permission to write
  the bucket and create CloudFront invalidations.
- `ffmpeg` on `PATH` — `sync-data` transcodes each bundled `audio.wav` to AAC.
  Without it the build still succeeds, but every track silently falls back to
  the click track. **Watch the build output for the transcode warning.**
- `npm ci` done in `web/`.

## What "the publish cut" means

`npm run build:publish` (in `web/`), which is three steps:

1. `node scripts/sync-data.mjs --publish` — copies `data/timelines/` into
   `web/public/timelines/`, but only the track ids listed in
   **`data/publish.json`**, and only the files named in that script's
   allowlist (`timeline.json`, `timeline.bin`, plus the transcoded
   `audio.m4a`). Everything else in a track directory — stems, model
   activations, the source wav — stays home. This is what keeps `dist` at
   ~27 MB instead of 519 MB.
2. `tsc --noEmit` — the typecheck. Note it covers `src` and `scripts` only;
   `tests/` is not typechecked, so run `npm test` yourself.
3. `vite build --mode demo` — picks up `web/.env.demo`, which sets
   `VITE_DEMO=1`, which is what `DEMO_BUILD` in `web/src/runtime/demo.ts`
   reads. That flag is the whole difference between the demo and a local
   build: the demo declines to probe for `terrarium-server` and declines to
   hand the panel an export host, so the upload row and the export folder
   vanish on their own.

**To add or remove a track from the demo, edit `data/publish.json` and
redeploy.** Nothing else needs to change.

`web/.env.demo` is committed against `.gitignore`'s `.env.*` via an explicit
`!web/.env.demo` negation. If it ever goes missing, `build:publish` still
succeeds and quietly ships the full app as if it were the demo — so if the
export panel shows up on the live site, that file is the first thing to check.

## What the upload does

Three `aws s3 sync` passes, in an order chosen so the site is consistent at
every instant — content first, the document that points at it last:

| what | cache-control | `--delete`? |
| --- | --- | --- |
| `dist/assets` | `public, max-age=31536000, immutable` | no |
| `dist/timelines` | `public, max-age=86400` | yes |
| everything else (`index.html`, …) | `no-cache` | yes |

Vite's assets are content-hashed, so their names change with their bytes and
they can live forever; they are not deleted, so a visitor mid-deploy still
finds the assets the old index references. Timelines are *not* content-hashed
(`free-fall/audio.m4a` keeps its name across a re-analysis), hence the one-day
ceiling. `no-cache` on the document means "revalidate", not "do not store" — a
deploy is visible on the next load rather than whenever a stale copy expires.

Then `create-invalidation --paths '/*'`, which covers the window the
one-day timeline TTL leaves open. The script prints the invalidation id;
propagation is typically under a minute.

## The infrastructure, for when something is wrong

Standing up on 2026-08-19; none of it needs to be touched for a routine
deploy.

| piece | value |
| --- | --- |
| AWS account | `040630189325` |
| S3 bucket | `wryskware-terrarium-site` (private) |
| CloudFront distribution | `E2ISYNUAG82OYT` |
| CloudFront domain | `dwjp9zs5pyebf.cloudfront.net` |
| ACM certificate (us-east-1) | `arn:aws:acm:us-east-1:040630189325:certificate/b5f29792-ab2e-44a6-b10c-759a10213fa6` |
| Route 53 hosted zone | `Z1014033376GTUZEXE6AE` (`wryskware.dev`) |
| CloudFront's alias zone id | `Z2FDTNDATAQYW2` (a global constant, not ours) |

The bucket is private and read only by the distribution through an origin
access control — there is no public bucket policy and no S3 website endpoint.
Don't "fix" a 403 by making the bucket public; check the OAC and the
distribution's default root object instead.

`dreams.wryskware.dev` is an A **and** AAAA alias to the distribution. TLS
floor is `TLSv1.2_2021`, `sni-only`. The `cloudfront.net` name keeps working
independently, which makes it the right thing to `curl` when you want to know
whether a problem is the site or DNS:

```bash
curl -sI https://dwjp9zs5pyebf.cloudfront.net/ | head -1
curl -sI --resolve dreams.wryskware.dev:443:<ip> https://dreams.wryskware.dev/ | head -1
```

(A local resolver that has cached an NXDOMAIN from before the record existed
will report "could not resolve host" long after the record is live. `nslookup`
disagreeing with `curl` means a stale negative cache, not an outage.)

The zone also carries Spaceship's email forwarding for the apex: an SPF `TXT`
and two `MX` at preference 0. **Route 53 models both MX entries as one record
set with two values** — creating them as two sets is rejected, which does not
match how Spaceship's own instructions describe them.

## Before you push a cut

- `npm test` in `web/` (the build does not run it).
- Load the built site locally: `npm run preview` after `build:publish`, not
  `npm run dev` — dev serves the full app and will not show you what the demo
  build actually removes.
- Confirm on the live site afterwards: no request to `localhost:8765`, no
  export folder in the panel, no "analysis server" row, and audio actually
  plays (a click track instead of music means the ffmpeg step was skipped).
