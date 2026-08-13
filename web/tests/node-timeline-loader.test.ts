import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import { loadTimelineFromFiles } from '../src/export/node-timeline-loader.ts';

test('Node filesystem loading constructs the shipped timeline through shared validation', async () => {
  const timeline = await loadTimelineFromFiles({
    jsonPath: fileURLToPath(new URL('../../data/timelines/synthetic/timeline.json', import.meta.url)),
    binaryPath: fileURLToPath(new URL('../../data/timelines/synthetic/timeline.bin', import.meta.url)),
  });
  assert.equal(timeline.manifest.track.id, 'synthetic');
  assert.equal(timeline.manifest.grid.frames, 2100);
  assert.equal(timeline.stride, 73);
  assert.equal(timeline.data.length, 2100 * 73);
  assert.equal(timeline.channels.get('latent')?.dims, 64);
  assert.ok(timeline.events.length > 100);
});

test('Node filesystem loading reports manifest JSON and shared binary-size failures', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'terrarium-timeline-'));
  try {
    const jsonPath = join(directory, 'timeline.json');
    const binaryPath = join(directory, 'timeline.bin');
    await writeFile(binaryPath, new Uint8Array(0));
    await writeFile(jsonPath, '{', 'utf8');
    await assert.rejects(
      loadTimelineFromFiles({ jsonPath, binaryPath }),
      /timeline\.json: invalid JSON/,
    );

    await writeFile(
      jsonPath,
      JSON.stringify({
        version: 2,
        track: { id: 'tiny', duration: 1, sampleRate: 48_000 },
        grid: { hopSeconds: 0.1, frames: 1 },
        beats: [],
        downbeats: [],
        tempo: 120,
        segments: [],
        channels: [{ name: 'latent', dims: 1, offset: 0 }],
      }),
      'utf8',
    );
    await assert.rejects(
      loadTimelineFromFiles({ jsonPath, binaryPath }),
      /timeline\.bin: expected 4 bytes.*got 0/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
