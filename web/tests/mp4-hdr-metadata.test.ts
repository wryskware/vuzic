import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  COLOUR_PRIMARIES_BT2020,
  MATRIX_COEFFICIENTS_BT2020_NCL,
  TRANSFER_CHARACTERISTICS_SMPTE2084,
  colrBox,
  injectHdrContainerMetadata,
  mdcvBox,
  patchMoovHdrMetadata,
} from '../src/export/mp4-hdr-metadata.ts';

function box(type: string, payload: Buffer): Buffer {
  const out = Buffer.alloc(8 + payload.length);
  out.writeUInt32BE(out.length, 0);
  out.write(type, 4, 'latin1');
  payload.copy(out, 8);
  return out;
}

/** A minimal but structurally real moov with one video and one audio track. */
function fixtureMoov(): Buffer {
  const hvcC = box('hvcC', Buffer.alloc(12, 7));
  const hvc1 = box('hvc1', Buffer.concat([Buffer.alloc(78, 0), hvcC]));
  const videoStsd = box(
    'stsd',
    Buffer.concat([Buffer.from([0, 0, 0, 0, 0, 0, 0, 1]), hvc1]),
  );
  const audioStsd = box(
    'stsd',
    Buffer.concat([Buffer.from([0, 0, 0, 0, 0, 0, 0, 1]), box('mp4a', Buffer.alloc(28))]),
  );
  const trak = (stsd: Buffer): Buffer =>
    box(
      'trak',
      Buffer.concat([
        box('tkhd', Buffer.alloc(84)),
        box('mdia', Buffer.concat([box('minf', box('stbl', Buffer.concat([stsd, box('stts', Buffer.alloc(16))])))])),
      ]),
    );
  return box('moov', Buffer.concat([box('mvhd', Buffer.alloc(100)), trak(audioStsd), trak(videoStsd)]));
}

function findBox(buffer: Buffer, type: string): number {
  return buffer.indexOf(Buffer.from(type, 'latin1'));
}

test('the colr box describes BT.2020 primaries, PQ transfer, and limited range', () => {
  const colr = colrBox();
  assert.equal(colr.length, 19);
  assert.equal(colr.readUInt32BE(0), 19);
  assert.equal(colr.toString('latin1', 4, 8), 'colr');
  assert.equal(colr.toString('latin1', 8, 12), 'nclx');
  assert.equal(colr.readUInt16BE(12), COLOUR_PRIMARIES_BT2020);
  assert.equal(colr.readUInt16BE(14), TRANSFER_CHARACTERISTICS_SMPTE2084);
  assert.equal(colr.readUInt16BE(16), MATRIX_COEFFICIENTS_BT2020_NCL);
  // Bit 7 is full_range_flag; limited range means it is clear.
  assert.equal(colr.readUInt8(18) & 0x80, 0);
});

test('the mdcv box stores green, blue, red order and the policy luminance', () => {
  const mdcv = mdcvBox(1000);
  assert.equal(mdcv.length, 32);
  assert.equal(mdcv.toString('latin1', 4, 8), 'mdcv');
  const values = Array.from({ length: 8 }, (_, index) => mdcv.readUInt16BE(8 + index * 2));
  assert.deepEqual(values, [8500, 39850, 6550, 2300, 35400, 14600, 15635, 16450]);
  assert.equal(mdcv.readUInt32BE(24), 10_000_000);
  assert.equal(mdcv.readUInt32BE(28), 1);
  assert.equal(mdcvBox(4000).readUInt32BE(24), 40_000_000);
});

test('patching a moov adds both boxes to the video entry and nothing else', () => {
  const moov = fixtureMoov();
  const patched = patchMoovHdrMetadata(moov, 1000);

  assert.equal(patched.length, moov.length + 19 + 32);
  assert.equal(patched.readUInt32BE(0), patched.length);
  assert.equal(patched.toString('latin1', 4, 8), 'moov');

  // Exactly one of each, and both inside the hvc1 sample entry.
  const colrAt = findBox(patched, 'colr');
  const mdcvAt = findBox(patched, 'mdcv');
  const hvc1At = findBox(patched, 'hvc1');
  assert.ok(hvc1At > 0 && colrAt > hvc1At && mdcvAt > colrAt);
  assert.equal(patched.indexOf(Buffer.from('colr'), colrAt + 1), -1);
  assert.equal(patched.indexOf(Buffer.from('mdcv'), mdcvAt + 1), -1);

  // The audio track's sample entry is untouched.
  assert.ok(findBox(patched, 'mp4a') > 0);
});

test('every ancestor size is corrected, so the patched tree still parses', () => {
  const patched = patchMoovHdrMetadata(fixtureMoov(), 1000);
  const walk = (buffer: Buffer, start: number, end: number, depth: number): void => {
    let offset = start;
    while (offset + 8 <= end) {
      const size = buffer.readUInt32BE(offset);
      const type = buffer.toString('latin1', offset + 4, offset + 8);
      assert.ok(size >= 8 && offset + size <= end, `${type} size ${size} overruns its parent`);
      if (depth < 5 && ['trak', 'mdia', 'minf', 'stbl'].includes(type)) {
        walk(buffer, offset + 8, offset + size, depth + 1);
      }
      offset += size;
    }
    assert.equal(offset, end, 'a container has trailing bytes its children do not cover');
  };
  walk(patched, 8, patched.length, 0);
});

test('re-patching is idempotent rather than duplicating the boxes', () => {
  const once = patchMoovHdrMetadata(fixtureMoov(), 1000);
  const twice = patchMoovHdrMetadata(once, 1000);
  assert.deepEqual(twice, once);
});

test('a moov with no video sample entry is a failure, not a silent no-op', () => {
  const audioOnly = box(
    'moov',
    box(
      'trak',
      box(
        'mdia',
        box(
          'minf',
          box(
            'stbl',
            box('stsd', Buffer.concat([Buffer.from([0, 0, 0, 0, 0, 0, 0, 1]), box('mp4a', Buffer.alloc(28))])),
          ),
        ),
      ),
    ),
  );
  assert.throws(() => patchMoovHdrMetadata(audioOnly, 1000), /no video sample entry/);
});

test('injection rewrites a trailing moov in place and leaves media data alone', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'lmt-mp4-hdr-'));
  try {
    const path = join(directory, 'sample.mp4');
    const ftyp = box('ftyp', Buffer.from('isom\0\0\0isomiso2mp41', 'latin1'));
    const mdat = box('mdat', Buffer.alloc(4096, 0xa5));
    const moov = fixtureMoov();
    await writeFile(path, Buffer.concat([ftyp, mdat, moov]));

    await injectHdrContainerMetadata(path, 1000);

    const patched = await readFile(path);
    assert.equal(patched.length, ftyp.length + mdat.length + moov.length + 19 + 32);
    // Byte-for-byte identical prefix: every stco/co64 offset in the file is
    // still correct because nothing before moov moved.
    assert.deepEqual(patched.subarray(0, ftyp.length + mdat.length), Buffer.concat([ftyp, mdat]));
    const tail = patched.subarray(ftyp.length + mdat.length);
    assert.equal(tail.toString('latin1', 4, 8), 'moov');
    assert.equal(tail.readUInt32BE(0), tail.length);
    assert.ok(findBox(tail, 'mdcv') > 0);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('injection refuses a leading moov rather than shifting every chunk offset', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'lmt-mp4-hdr-'));
  try {
    const path = join(directory, 'faststart.mp4');
    await writeFile(
      path,
      Buffer.concat([box('ftyp', Buffer.alloc(16)), fixtureMoov(), box('mdat', Buffer.alloc(256))]),
    );
    await assert.rejects(injectHdrContainerMetadata(path, 1000), /not the last box/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
