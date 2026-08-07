import { cp, mkdir, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../data/timelines');
const dst = resolve(here, '../public/timelines');

const entries = await readdir(src, { withFileTypes: true }).catch(() => {
  console.error(`sync-data: no source directory at ${src}`);
  process.exit(1);
});

await mkdir(dst, { recursive: true });
for (const e of entries) {
  if (!e.isDirectory()) continue;
  await cp(join(src, e.name), join(dst, e.name), { recursive: true });
  console.log(`sync-data: ${e.name}`);
}
