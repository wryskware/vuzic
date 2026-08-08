import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../data/timelines');
const dst = resolve(here, '../public/timelines');

/**
 * Files the browser never asks for. `embedding.*` is the ~11 MB raw MuQ sidecar:
 * plan.md Revision 4 took it out of the runtime (the driver bank is built from
 * the timeline's own latent channel), so copying it into `public/` only bloated
 * the dev server and the dist bundle. It stays in `data/` — it is the training
 * input for the distilled-NN mapping in "Later" — it just does not ship.
 *
 * `plots/` is analysis QA output, likewise for humans reading the repo.
 */
const SKIP = new Set(['embedding.json', 'embedding.bin', 'plots']);

const entries = await readdir(src, { withFileTypes: true }).catch(() => {
  console.error(`sync-data: no source directory at ${src}`);
  process.exit(1);
});

await mkdir(dst, { recursive: true });
for (const e of entries) {
  if (!e.isDirectory()) continue;
  await cp(join(src, e.name), join(dst, e.name), {
    recursive: true,
    filter: (path) => !SKIP.has(basename(path)),
  });
  // A previous sync may have copied them before the skip list existed.
  for (const name of SKIP) await rm(join(dst, e.name, name), { recursive: true, force: true });
  console.log(`sync-data: ${e.name}`);
}
