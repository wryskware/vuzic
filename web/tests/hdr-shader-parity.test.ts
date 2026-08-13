import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import {
  BT2020_KB,
  BT2020_KG,
  BT2020_KR,
  BT709_TO_BT2020,
  LIMITED_CHROMA_MAX_10,
  LIMITED_CHROMA_MIN_10,
  LIMITED_CHROMA_OFFSET_10,
  LIMITED_CHROMA_SCALE_10,
  LIMITED_LUMA_MAX_10,
  LIMITED_LUMA_MIN_10,
  LIMITED_LUMA_OFFSET_10,
  LIMITED_LUMA_SCALE_10,
  PQ_C1,
  PQ_C2,
  PQ_C3,
  PQ_M1,
  PQ_M2,
  PQ_MAX_NITS,
} from '../src/export/hdr.ts';

const shader = (relative: string): string =>
  fileURLToPath(new URL(`../src/${relative}`, import.meta.url));

async function read(relative: string): Promise<string> {
  return readFile(shader(relative), 'utf8');
}

/** Pull `const NAME: f32 = 1.25;` out of a WGSL source. */
function wgslConstant(source: string, name: string): number {
  const match = new RegExp(`const\\s+${name}\\s*:\\s*f32\\s*=\\s*(-?[0-9.]+)\\s*;`).exec(source);
  assert.ok(match, `hdr-pack.wgsl declares no f32 constant named ${name}`);
  return Number((match as RegExpExecArray)[1]);
}

/**
 * The colour standards exist twice on purpose — once in TypeScript where they
 * can be unit-tested against published reference values, once in WGSL where the
 * GPU can execute them. This test is what stops the two copies from drifting;
 * without it the tested numbers and the shipped numbers are unrelated.
 */
test('the packing shader carries exactly the PQ constants the tested module does', async () => {
  const source = await read('export/shaders/hdr-pack.wgsl');
  assert.equal(wgslConstant(source, 'PQ_M1'), PQ_M1);
  assert.equal(wgslConstant(source, 'PQ_M2'), PQ_M2);
  assert.equal(wgslConstant(source, 'PQ_C1'), PQ_C1);
  assert.equal(wgslConstant(source, 'PQ_C2'), PQ_C2);
  assert.equal(wgslConstant(source, 'PQ_C3'), PQ_C3);
});

test('the packing shader carries the BT.2020 matrix and quantisation the module does', async () => {
  const source = await read('export/shaders/hdr-pack.wgsl');
  assert.equal(wgslConstant(source, 'BT2020_KR'), BT2020_KR);
  assert.equal(wgslConstant(source, 'BT2020_KG'), BT2020_KG);
  assert.equal(wgslConstant(source, 'BT2020_KB'), BT2020_KB);

  assert.equal(wgslConstant(source, 'LUMA_OFFSET_10'), LIMITED_LUMA_OFFSET_10);
  assert.equal(wgslConstant(source, 'LUMA_SCALE_10'), LIMITED_LUMA_SCALE_10);
  assert.equal(wgslConstant(source, 'CHROMA_OFFSET_10'), LIMITED_CHROMA_OFFSET_10);
  assert.equal(wgslConstant(source, 'CHROMA_SCALE_10'), LIMITED_CHROMA_SCALE_10);
  assert.equal(wgslConstant(source, 'LUMA_MIN_10'), LIMITED_LUMA_MIN_10);
  assert.equal(wgslConstant(source, 'LUMA_MAX_10'), LIMITED_LUMA_MAX_10);
  assert.equal(wgslConstant(source, 'CHROMA_MIN_10'), LIMITED_CHROMA_MIN_10);
  assert.equal(wgslConstant(source, 'CHROMA_MAX_10'), LIMITED_CHROMA_MAX_10);
});

test('the HDR grade carries the BT.2020 primaries matrix, transposed for WGSL', async () => {
  const source = await read('sim/render/shaders/grade-hdr.wgsl');
  const block = /const BT709_TO_BT2020: mat3x3f = mat3x3f\(([\s\S]*?)\);/.exec(source);
  assert.ok(block, 'grade-hdr.wgsl declares no BT709_TO_BT2020 matrix');
  const numbers = ((block as RegExpExecArray)[1] as string)
    .match(/-?\d+\.\d+/g)
    ?.map(Number);
  assert.equal(numbers?.length, 9);

  // mat3x3f takes columns, so WGSL entry (column c, row r) is TS row r, column c.
  for (let column = 0; column < 3; column++) {
    for (let row = 0; row < 3; row++) {
      assert.equal(
        (numbers as number[])[column * 3 + row],
        (BT709_TO_BT2020[row] as readonly number[])[column],
        `matrix entry column ${column} row ${row} drifted`,
      );
    }
  }
});

test('the HDR grade normalises to the same ST 2084 peak the module uses', async () => {
  const source = await read('sim/render/shaders/grade-hdr.wgsl');
  assert.equal(wgslConstant(source, 'PQ_MAX_NITS'), PQ_MAX_NITS);
  // The HDR grade must not gamma-encode; that is the packing pass's job and
  // doing it twice is exactly the "HDR-tagged SDR" failure to avoid.
  assert.ok(!/p\.gamma/.test(source), 'the HDR grade must not apply a display gamma');
});

test('the HDR grade and the SDR grade read the same params buffer layout', async () => {
  const common = await read('sim/render/shaders/post-common.wgsl');
  const fields = /struct Post \{([\s\S]*?)\n\}/.exec(common)?.[1] ?? '';
  const words = fields.match(/:\s*(f32|vec2f)/g) ?? [];
  const slots = words.reduce((count, word) => count + (word.includes('vec2f') ? 2 : 1), 0);
  // PostFx writes this many f32; a mismatch silently corrupts every grade value.
  assert.equal(slots, 24);
  for (const name of ['hdrPaperWhite', 'hdrPeak', 'hdrHeadroom']) {
    assert.ok(fields.includes(name), `post-common.wgsl is missing ${name}`);
  }
});
