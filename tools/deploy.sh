#!/usr/bin/env bash
#
# Build the publish cut and push it to the static demo.
#
# The demo is a private S3 bucket read only by one CloudFront distribution
# (origin access control, no public bucket, no website endpoint). Everything
# here is idempotent: run it as often as you like.
#
# Usage:  tools/deploy.sh            build, upload, invalidate
#         tools/deploy.sh --no-build upload what is already in web/dist
set -euo pipefail

BUCKET=wryskware-terrarium-site
DISTRIBUTION=E2ISYNUAG82OYT

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$here/web"

if [[ "${1:-}" != "--no-build" ]]; then
  # build:publish, not build: the allowlist in data/publish.json decides what
  # ships, and `build` deliberately bundles everything for local work.
  npm run build:publish
fi

if [[ ! -f dist/index.html ]]; then
  echo "deploy: web/dist has no index.html — build first" >&2
  exit 1
fi

# Three cache classes, in the order that leaves the site consistent at every
# moment: content-addressed things first, the document that points at them last.
# A visitor mid-deploy either sees the old index (whose assets are still there,
# because --delete has not run on them yet) or the new one.

# Vite's assets are content-hashed, so their names change whenever their bytes
# do and they can be cached forever.
aws s3 sync dist/assets "s3://$BUCKET/assets" \
  --cache-control 'public, max-age=31536000, immutable' \
  --only-show-errors

# Timelines are not content-hashed — `free-fall/audio.m4a` keeps its name across
# a re-analysis — so they get a day, and the invalidation below covers the case
# where that is not fast enough.
aws s3 sync dist/timelines "s3://$BUCKET/timelines" --delete \
  --cache-control 'public, max-age=86400' \
  --only-show-errors

# Everything else, index.html included. `no-cache` means "revalidate", not "do
# not store": the browser still holds it and a 304 is cheap, but a deploy is
# visible on the next load rather than whenever a stale copy happens to expire.
aws s3 sync dist "s3://$BUCKET" --delete \
  --exclude 'assets/*' --exclude 'timelines/*' \
  --cache-control 'no-cache' \
  --only-show-errors

id=$(aws cloudfront create-invalidation --distribution-id "$DISTRIBUTION" \
  --paths '/*' --query 'Invalidation.Id' --output text)

echo "deploy: uploaded; invalidation $id"
echo "deploy: https://dreams.wryskware.dev/"
