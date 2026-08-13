import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

/**
 * The HDR output policy has to survive the trip from the worker into the
 * context object, because that object is the *only* thing the shared post chain
 * consults when it decides whether to build the SDR grade or the PQ grade.
 *
 * This was a real defect: the option was accepted by `createDawnContext` and
 * dropped on the way out, and the resulting file was a correctly tagged HDR10
 * container holding an 8-bit SDR grade — indistinguishable from success in
 * every container-level check. It is asserted at the source level because the
 * only other way to see it is a native GPU render.
 */
async function source(relative: string): Promise<string> {
  return readFile(fileURLToPath(new URL(`../src/${relative}`, import.meta.url)), 'utf8');
}

test('the Dawn context forwards the HDR output policy it was given', async () => {
  const text = await source('export/dawn-context.ts');
  assert.ok(/hdrOutput\?: GpuHdrOutput/.test(text), 'the option is not declared');
  assert.ok(
    /\.\.\.\(options\.hdrOutput \? \{ hdrOutput: options\.hdrOutput \} : \{\}\)/.test(text),
    'the returned context does not carry options.hdrOutput',
  );
});

test('the post chain selects its final pass from the context, not from a flag', async () => {
  const text = await source('sim/render/postfx.ts');
  assert.ok(/ctx\.hdrOutput \? gradeHdrWgsl : gradeWgsl/.test(text));
  // The luminance policy must reach the params buffer, or the HDR grade would
  // clamp everything to a peak of zero.
  assert.ok(/hdr \? hdr\.paperWhiteNits : 0/.test(text));
  assert.ok(/hdr \? hdr\.masteringPeakNits : 0/.test(text));
});

test('the worker refuses an HDR job whose context lost the policy', async () => {
  const text = await source('export/worker.ts');
  assert.ok(
    /if \(hdr && !ctx\.hdrOutput\) \{[\s\S]*?throw new Error/.test(text),
    'the worker does not guard against a dropped HDR policy',
  );
});
