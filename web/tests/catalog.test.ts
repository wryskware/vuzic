/**
 * The track catalog's merge rules.
 *
 * Nothing here touches the network or storage: `probeServer`, `buildCatalog` and
 * `cache.ts` all need a browser, and what is actually worth pinning is the part
 * that decides *which* copy of a track wins when the same id is bundled, offered
 * by a server, and sitting in the offline cache — which is pure.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  SERVER_BASE,
  fetcherFor,
  mergeCatalog,
  serverEntry,
  type TrackEntry,
} from '../src/timeline/catalog.ts';

function bundled(id: string, title = id): TrackEntry {
  return {
    id,
    title,
    duration: 100,
    version: '',
    base: `/timelines/${id}`,
    hasAudio: true,
    source: 'bundled',
  };
}

function cached(id: string, version: string): TrackEntry {
  return { ...serverEntry({ id, title: id, duration: 100, version }), source: 'cached' };
}

test('serverEntry builds a URL the loader can append filenames to', () => {
  const e = serverEntry({ id: 'pink-loop', title: 'Pink Loop', duration: 209.56, version: 'abc' });
  assert.equal(e.base, `${SERVER_BASE}/tracks/pink-loop`);
  assert.equal(e.source, 'server');
  assert.equal(e.title, 'Pink Loop');
  assert.equal(e.duration, 209.56);
  // Absent in the JSON means absent, not truthy.
  assert.equal(e.hasAudio, false);
  assert.equal(serverEntry({ id: 'x' }).title, 'x');
});

test('bundled beats server beats cached for the same id', () => {
  const merged = mergeCatalog(
    [bundled('pink-loop', 'Pink Loop')],
    [serverEntry({ id: 'pink-loop', title: 'Pink Loop', version: 'v2' })],
    [cached('pink-loop', 'v1')],
  );
  assert.equal(merged.length, 1);
  assert.equal(merged[0]?.source, 'bundled');
});

test('a server track that is not bundled survives, and the cache fills the gap', () => {
  const merged = mergeCatalog(
    [bundled('free-fall', 'Free Fall')],
    [serverEntry({ id: 'uploaded', title: 'Uploaded', version: 'v1' })],
    [cached('older', 'v0')],
  );
  assert.deepEqual(
    merged.map((t) => [t.id, t.source]),
    [
      ['free-fall', 'bundled'],
      ['older', 'cached'],
      ['uploaded', 'server'],
    ],
  );
});

test('with no server the cached copy is what is left', () => {
  const merged = mergeCatalog([bundled('synthetic')], [], [cached('uploaded', 'v1')]);
  assert.deepEqual(
    merged.map((t) => t.id),
    ['synthetic', 'uploaded'],
  );
});

test('the list is sorted by title, not by source', () => {
  const merged = mergeCatalog([bundled('b', 'Zebra'), bundled('a', 'Aardvark')], [], []);
  assert.deepEqual(
    merged.map((t) => t.title),
    ['Aardvark', 'Zebra'],
  );
});

test('only bundled tracks bypass the cache, via a BOUND fetch', () => {
  // The read-through needs a browser, so this is shape, not behaviour: a bundled
  // track is never routed through the cache (it is already local), server and
  // cached tracks are. The bound-copy assertions are a shipped-bug regression
  // guard: fetcherFor once returned bare `fetch`, whose brand check fails as a
  // *rejected promise* when AudioClock calls it as `this.fetcher(...)` — every
  // bundled track silently lost its audio to the click fallback. Node's fetch
  // is not receiver-branded, so the bug itself is unobservable here; the
  // not-reference-equal check pins the fix at the only level node can see.
  const f = fetcherFor(bundled('x'));
  assert.notEqual(f, fetch, 'bundled fetcher must be a bound copy, never bare fetch');
  assert.equal(f.name, 'bound fetch');
  assert.notEqual(fetcherFor(serverEntry({ id: 'x' })), fetch);
  assert.equal(fetcherFor(cached('x', 'v1')), fetcherFor(serverEntry({ id: 'x' })));
});
