/**
 * The one table that says what an export profile actually is.
 *
 * Profile identifiers are load-bearing and deliberately literal: an id may only
 * say `hdr10` if it renders through the scene-linear BT.2020/PQ path and encodes
 * a 4:2:0 10-bit stream with matching colour metadata. The temporary engineering
 * profiles keep saying `sdr-debug` for exactly the same reason.
 *
 * **Every profile here is AV1.** An earlier revision routed HDR10 through
 * `hevc_nvenc` on the belief that AV1 could not carry the colour description —
 * that was wrong, and it was wrong because the FFmpeg build was old. On FFmpeg 9
 * the `av1_metadata` bitstream filter writes primaries, transfer, matrix and
 * range into the AV1 sequence header exactly as `hevc_metadata` does for the
 * HEVC VUI, and the `-mastering_display` / `-content_light` input options give
 * the MP4 muxer real ST 2086 side data. AV1 needs no HDR-specific profile
 * either: Main already covers 10-bit, so there is no `main10` to ask for.
 */
import type { ExportEncoder, ExportProfile } from '../runtime/recipe.ts';

export type ExportDynamicRange = 'sdr' | 'hdr10';

/**
 * How frames reach FFmpeg. This is a capability string, not decoration: the
 * browser and the server both refuse a profile whose transport the built worker
 * does not advertise.
 */
export const SDR_DEBUG_TRANSPORT = 'sdr-rgba8-av1-debug';
export const HDR10_TRANSPORT = 'hdr10-p010-compute';

export interface ExportProfileSpec {
  readonly id: ExportProfile;
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly fps: 120;
  readonly encoder: ExportEncoder;
  readonly dynamicRange: ExportDynamicRange;
  readonly transport: typeof SDR_DEBUG_TRANSPORT | typeof HDR10_TRANSPORT;
  /** FFmpeg raw input pixel format the worker feeds for this profile. */
  readonly inputPixelFormat: 'rgba' | 'p010le';
}

export const EXPORT_PROFILE_SPECS: readonly ExportProfileSpec[] = [
  {
    id: 'av1-hdr10-2160p120',
    label: '4K / 120 fps / HDR10 (AV1 10-bit, PQ)',
    width: 3840,
    height: 2160,
    fps: 120,
    encoder: 'av1_nvenc',
    dynamicRange: 'hdr10',
    transport: HDR10_TRANSPORT,
    inputPixelFormat: 'p010le',
  },
  {
    id: 'av1-hdr10-1080p120',
    label: '1080p / 120 fps / HDR10 (AV1 10-bit, PQ)',
    width: 1920,
    height: 1080,
    fps: 120,
    encoder: 'av1_nvenc',
    dynamicRange: 'hdr10',
    transport: HDR10_TRANSPORT,
    inputPixelFormat: 'p010le',
  },
  {
    id: 'av1-sdr-debug-2160p120',
    label: '4K / 120 fps / SDR debug',
    width: 3840,
    height: 2160,
    fps: 120,
    encoder: 'av1_nvenc',
    dynamicRange: 'sdr',
    transport: SDR_DEBUG_TRANSPORT,
    inputPixelFormat: 'rgba',
  },
  {
    id: 'av1-sdr-debug-1080p120',
    label: '1080p / 120 fps / SDR debug',
    width: 1920,
    height: 1080,
    fps: 120,
    encoder: 'av1_nvenc',
    dynamicRange: 'sdr',
    transport: SDR_DEBUG_TRANSPORT,
    inputPixelFormat: 'rgba',
  },
];

export function exportProfileSpec(profile: ExportProfile): ExportProfileSpec {
  const spec = EXPORT_PROFILE_SPECS.find((candidate) => candidate.id === profile);
  if (!spec) throw new Error(`unknown export profile: ${profile}`);
  return spec;
}

export function isHdrExportProfile(profile: ExportProfile): boolean {
  return exportProfileSpec(profile).dynamicRange === 'hdr10';
}

/** Encoders that must be present before a profile can be offered at all. */
export function requiredEncoder(profile: ExportProfile): ExportEncoder {
  return exportProfileSpec(profile).encoder;
}
