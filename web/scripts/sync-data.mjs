import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
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
const index = [];
for (const e of entries) {
  if (!e.isDirectory() || e.name.startsWith('.')) continue;
  await cp(join(src, e.name), join(dst, e.name), {
    recursive: true,
    filter: (path) => !SKIP.has(basename(path)),
  });
  // A previous sync may have copied them before the skip list existed.
  for (const name of SKIP) await rm(join(dst, e.name, name), { recursive: true, force: true });

  // The track picker needs to know what shipped, and the set of bundled tracks
  // is a property of this directory rather than of any list in the source — so
  // it is written here instead of being maintained by hand in TypeScript.
  // Malformed timelines are skipped rather than fatal: `sync-data` runs before
  // every `dev` and `build`, and a half-written analysis output must not stop
  // the dev server from starting.
  const manifest = await readFile(join(src, e.name, 'timeline.json'), 'utf8')
    .then(JSON.parse)
    .catch(() => null);
  if (manifest?.track) {
    index.push({
      id: e.name,
      title: String(manifest.track.id ?? e.name),
      duration: Number(manifest.track.duration) || 0,
      hasAudio: await stat(join(dst, e.name, 'audio.wav')).then(
        () => true,
        () => false,
      ),
    });
  } else {
    console.warn(`sync-data: ${e.name} has no readable timeline.json; not listed`);
  }
  console.log(`sync-data: ${e.name}`);
}

index.sort((a, b) => a.title.localeCompare(b.title));
await writeFile(join(dst, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(`sync-data: index.json (${index.length} tracks)`);
