/**
 * Static HDR10 colour metadata for an MP4 the encoder could not describe.
 *
 * NVENC writes the matrix coefficients and the range into the bitstream and
 * nothing else, and this FFmpeg build exposes no way to attach SMPTE ST 2086
 * mastering-display metadata to a hardware-encoded stream — `-master_display`
 * belongs to libx265, and the `sidedata` filter can only select and delete. The
 * transfer function and primaries are recovered by forcing the HEVC VUI through
 * the `hevc_metadata` bitstream filter (see `ffmpeg-encoder.ts`); the mastering
 * display volume has no such lever and is written here, into the container, as
 * ISO/IEC 23001-17 `mdcv`. A matching ISO/IEC 14496-12 `colr` box goes in beside
 * it so a player that trusts the container rather than the elementary stream
 * still sees BT.2020/PQ.
 *
 * Nothing here is measured from content: `mdcv` describes the display this
 * render was *graded for*, which is the same paper-white/peak policy the
 * highlight roll-off used. No `clli` box is written, because MaxCLL and MaxFALL
 * would have to be measured and this pipeline does not measure them.
 *
 * The edit is deliberately confined to a trailing `moov`. FFmpeg writes the
 * index last unless `+faststart` is requested, so growing `moov` moves no media
 * data and every `stco`/`co64` chunk offset in the file stays correct. That is
 * why the HDR profiles do not ask for faststart.
 */
import { open } from 'node:fs/promises';

import { masteringDisplayMetadata } from './hdr.ts';

/** ISO/IEC 23091-2 code points for the signal this pipeline produces. */
export const COLOUR_PRIMARIES_BT2020 = 9;
export const TRANSFER_CHARACTERISTICS_SMPTE2084 = 16;
export const MATRIX_COEFFICIENTS_BT2020_NCL = 9;

/** Video sample entries whose colour description these boxes belong to. */
const VIDEO_SAMPLE_ENTRIES = new Set(['hvc1', 'hev1', 'av01', 'avc1', 'avc3']);

interface BoxHeader {
  readonly type: string;
  readonly start: number;
  readonly headerSize: number;
  readonly size: number;
}

function readHeader(buffer: Buffer, offset: number, limit: number): BoxHeader {
  if (offset + 8 > limit) throw new Error('mp4: truncated box header');
  const declared = buffer.readUInt32BE(offset);
  const type = buffer.toString('latin1', offset + 4, offset + 8);
  if (declared === 1) {
    if (offset + 16 > limit) throw new Error('mp4: truncated 64-bit box header');
    const large = buffer.readBigUInt64BE(offset + 8);
    if (large > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('mp4: box larger than 2^53');
    return { type, start: offset, headerSize: 16, size: Number(large) };
  }
  const size = declared === 0 ? limit - offset : declared;
  if (size < 8 || offset + size > limit) throw new Error(`mp4: box ${type} has an invalid size`);
  return { type, start: offset, headerSize: 8, size };
}

function children(buffer: Buffer, start: number, end: number): BoxHeader[] {
  const found: BoxHeader[] = [];
  let offset = start;
  while (offset + 8 <= end) {
    const header = readHeader(buffer, offset, end);
    found.push(header);
    offset += header.size;
  }
  return found;
}

function box(type: string, payload: Buffer): Buffer {
  const out = Buffer.alloc(8 + payload.length);
  out.writeUInt32BE(out.length, 0);
  out.write(type, 4, 'latin1');
  payload.copy(out, 8);
  return out;
}

export function colrBox(): Buffer {
  const payload = Buffer.alloc(11);
  payload.write('nclx', 0, 'latin1');
  payload.writeUInt16BE(COLOUR_PRIMARIES_BT2020, 4);
  payload.writeUInt16BE(TRANSFER_CHARACTERISTICS_SMPTE2084, 6);
  payload.writeUInt16BE(MATRIX_COEFFICIENTS_BT2020_NCL, 8);
  // Bit 7 is full_range_flag; limited range, so the whole byte is zero.
  payload.writeUInt8(0, 10);
  return box('colr', payload);
}

/**
 * ISO/IEC 23001-17 MasteringDisplayColourVolume.
 *
 * Chromaticities are stored green, blue, red — the HEVC SEI order, not RGB.
 */
export function mdcvBox(masteringPeakNits: number): Buffer {
  const m = masteringDisplayMetadata(masteringPeakNits);
  const payload = Buffer.alloc(24);
  const order = [m.greenX, m.greenY, m.blueX, m.blueY, m.redX, m.redY, m.whiteX, m.whiteY];
  order.forEach((value, index) => payload.writeUInt16BE(value, index * 2));
  payload.writeUInt32BE(m.maxLuminance, 16);
  payload.writeUInt32BE(m.minLuminance, 20);
  return box('mdcv', payload);
}

/**
 * Replace the child boxes of `type` in `container` with `replacement`.
 *
 * There is at most one of each of these in a sample entry, so "replace or
 * append" is the whole policy: re-running the injector on an already-patched
 * file produces the same bytes rather than a second box.
 */
function upsertChild(
  parent: Buffer,
  childStart: number,
  type: string,
  replacement: Buffer,
): Buffer {
  const existing = children(parent, childStart, parent.length).filter((c) => c.type === type);
  if (existing.length === 0) return Buffer.concat([parent, replacement]);
  if (existing.length > 1) throw new Error(`mp4: multiple ${type} boxes in one sample entry`);
  const found = existing[0] as BoxHeader;
  return Buffer.concat([
    parent.subarray(0, found.start),
    replacement,
    parent.subarray(found.start + found.size),
  ]);
}

/** Fixed VisualSampleEntry fields between the box header and its child boxes. */
const VISUAL_SAMPLE_ENTRY_FIELDS = 78;

function patchSampleEntry(entry: Buffer, masteringPeakNits: number): Buffer {
  const childStart = 8 + VISUAL_SAMPLE_ENTRY_FIELDS;
  if (entry.length < childStart) throw new Error('mp4: truncated visual sample entry');
  let patched = upsertChild(entry, childStart, 'colr', colrBox());
  patched = upsertChild(patched, childStart, 'mdcv', mdcvBox(masteringPeakNits));
  patched.writeUInt32BE(patched.length, 0);
  return patched;
}

function patchStsd(stsd: Buffer, masteringPeakNits: number): Buffer | null {
  // FullBox(version, flags) then entry_count, then the sample entries.
  const entriesStart = 16;
  if (stsd.length < entriesStart) throw new Error('mp4: truncated stsd');
  const entries = children(stsd, entriesStart, stsd.length);
  const target = entries.findIndex((entry) => VIDEO_SAMPLE_ENTRIES.has(entry.type));
  if (target < 0) return null;
  const found = entries[target] as BoxHeader;
  const patchedEntry = patchSampleEntry(
    Buffer.from(stsd.subarray(found.start, found.start + found.size)),
    masteringPeakNits,
  );
  const out = Buffer.concat([
    stsd.subarray(0, found.start),
    patchedEntry,
    stsd.subarray(found.start + found.size),
  ]);
  out.writeUInt32BE(out.length, 0);
  return out;
}

/** Rewrite one child of a container box, fixing the container's own size. */
function patchChild(
  container: Buffer,
  childStart: number,
  type: string,
  patch: (child: Buffer) => Buffer | null,
): Buffer | null {
  for (const header of children(container, childStart, container.length)) {
    if (header.type !== type) continue;
    const patched = patch(
      Buffer.from(container.subarray(header.start, header.start + header.size)),
    );
    if (!patched) continue;
    const out = Buffer.concat([
      container.subarray(0, header.start),
      patched,
      container.subarray(header.start + header.size),
    ]);
    out.writeUInt32BE(out.length, 0);
    return out;
  }
  return null;
}

/** Patch the one video track's sample entry inside a whole `moov` box. */
export function patchMoovHdrMetadata(moov: Buffer, masteringPeakNits: number): Buffer {
  const patched = patchChild(moov, 8, 'trak', (trak) =>
    patchChild(trak, 8, 'mdia', (mdia) =>
      patchChild(mdia, 8, 'minf', (minf) =>
        patchChild(minf, 8, 'stbl', (stbl) =>
          patchChild(stbl, 8, 'stsd', (stsd) => patchStsd(stsd, masteringPeakNits)),
        ),
      ),
    ),
  );
  if (!patched) throw new Error('mp4: no video sample entry found in moov');
  return patched;
}

/**
 * Write `colr` and `mdcv` into the trailing `moov` of an existing MP4, in place.
 *
 * Throws rather than guessing if `moov` is not the last top-level box: growing a
 * leading `moov` would shift every chunk offset in the file, and silently
 * producing an unplayable render is far worse than failing the job.
 */
export async function injectHdrContainerMetadata(
  path: string,
  masteringPeakNits: number,
): Promise<void> {
  const handle = await open(path, 'r+');
  try {
    const { size: fileSize } = await handle.stat();
    let offset = 0;
    let moovStart = -1;
    let moovSize = 0;
    const header = Buffer.alloc(16);
    while (offset + 8 <= fileSize) {
      const { bytesRead } = await handle.read(header, 0, 16, offset);
      if (bytesRead < 8) break;
      const declared = header.readUInt32BE(0);
      const type = header.toString('latin1', 4, 8);
      let boxSize: number;
      if (declared === 1) {
        if (bytesRead < 16) throw new Error('mp4: truncated 64-bit box header');
        const large = header.readBigUInt64BE(8);
        if (large > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('mp4: box larger than 2^53');
        boxSize = Number(large);
      } else {
        boxSize = declared === 0 ? fileSize - offset : declared;
      }
      if (boxSize < 8 || offset + boxSize > fileSize) {
        throw new Error(`mp4: top-level box ${type} has an invalid size`);
      }
      if (type === 'moov') {
        moovStart = offset;
        moovSize = boxSize;
      }
      offset += boxSize;
    }
    if (moovStart < 0) throw new Error('mp4: no moov box found');
    if (moovStart + moovSize !== fileSize) {
      throw new Error('mp4: moov is not the last box; refusing to shift media offsets');
    }

    const moov = Buffer.alloc(moovSize);
    await handle.read(moov, 0, moovSize, moovStart);
    const patched = patchMoovHdrMetadata(moov, masteringPeakNits);
    await handle.write(patched, 0, patched.length, moovStart);
    await handle.truncate(moovStart + patched.length);
  } finally {
    await handle.close();
  }
}
