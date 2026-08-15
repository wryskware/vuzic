/**
 * Extended-range (HDR) swapchain configuration for the browser preview.
 *
 * The app has been scene-referred and unbounded from the trail textures all the
 * way to the final grade since phase 7 — the only reason highlights ever landed
 * on diffuse white is that the *swapchain* was 8-bit. Chrome can hand us a
 * float swapchain instead:
 *
 *   format      rgba16float
 *   toneMapping { mode: 'extended' }
 *
 * With `mode: 'standard'` (the default) the compositor clamps the surface to
 * [0, 1] and a float swapchain buys nothing. With `'extended'`, values above 1
 * survive into the display pipeline, so a bloom core can be genuinely brighter
 * than paper white instead of merely being white.
 *
 * Two things about the encoding, because getting either wrong produces a
 * plausible-looking but badly wrong image:
 *
 * - The values are still in the canvas colour space (`srgb`), with its transfer
 *   function — a float swapchain is *not* scene-linear. So `grade.wgsl` keeps
 *   its gamma encode exactly as it was; the only change is that it no longer
 *   clamps the result at 1.0.
 * - 1.0 still means SDR diffuse white. Everything above it consumes the
 *   display's headroom, which is what `headroom` below measures.
 *
 * The configuration is attempted whenever the browser supports it, regardless of
 * whether an HDR display is attached: on an SDR display the headroom is 1, the
 * grade collapses to bit-for-bit the SDR curve it always applied, and moving the
 * window to an HDR display later costs nothing but a changed uniform. The
 * alternative — reconfiguring on display change — would mean rebuilding the
 * grade pipeline, whose target format is baked in.
 */

/** No display gets more than this, and a bogus report should not be believed. */
const MAX_HEADROOM = 16;

/**
 * Assumed headroom when the display says it is HDR but the browser exposes no
 * number. ~2 stops: about what a 1000-nit panel has over a 200-nit paper white,
 * and conservative enough that a dimmer one only gets tone-mapped back down.
 */
const ASSUMED_HEADROOM = 4;

export interface CanvasHdrState {
  /** The format the swapchain was actually configured with. */
  readonly format: GPUTextureFormat;
  /** True when the compositor accepted extended-range tone mapping. */
  readonly extended: boolean;
  /** Why, for the console line and the workbench readout. */
  readonly detail: string;
}

let dynamicRange: MediaQueryList | null = null;

function hdrDisplay(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  dynamicRange ??= window.matchMedia('(dynamic-range: high)');
  return dynamicRange.matches;
}

/**
 * Ratio of the display's peak luminance to SDR white, right now. 1 means SDR —
 * either the panel, or the browser, or the user's own override.
 *
 * Cheap enough to call every frame, which is how it is called: headroom is not
 * a constant. It moves when the window crosses monitors, when the panel changes
 * brightness, and (on macOS) whenever another HDR surface appears on screen.
 */
export function displayHeadroom(): number {
  const reported = (screen as { highDynamicRangeHeadroom?: number }).highDynamicRangeHeadroom;
  if (typeof reported === 'number' && Number.isFinite(reported)) {
    return Math.min(Math.max(reported, 1), MAX_HEADROOM);
  }
  return hdrDisplay() ? ASSUMED_HEADROOM : 1;
}

/**
 * Session-only headroom override, for the workbench's preview toggle.
 *
 * `?hdr=<number>` already pins the headroom, but it costs a reload, and tuning
 * two renditions of the same look is a loop you run dozens of times — the whole
 * argument for the workbench existing. This is that override made live.
 *
 * Three deliberate properties:
 *
 * - **Module state, not config.** Both readers — the grade
 *   (`PostFx.writeParams`) and the particle luminance lane (`writeGlobals`) —
 *   reach it through `GpuRuntimeContext.displayHeadroom`, which
 *   `BrowserGpuContext.resize` recomputes once a frame. Putting it here means
 *   one knob moves both, so what you tune is what you see; a per-sim setting
 *   would have moved one of them and quietly lied about the other.
 * - **Never persisted.** It is a claim about the panel you are looking at, in
 *   the same class as `effectiveBudget`. A saved "pretend this is SDR" would
 *   open every future session mid-experiment.
 * - **Only honoured on an extended swapchain.** An 8-bit surface genuinely has
 *   headroom 1, and pretending otherwise would drive the grade past a ceiling
 *   the compositor is about to clamp — a preview of nothing. Forcing 1 on an
 *   extended surface, which is the direction that matters for tuning the SDR
 *   rendition on an HDR machine, is exact.
 */
let previewHeadroom: number | null = null;

/** `null` restores the measured value. Values are clamped like a real report. */
export function setPreviewHeadroom(value: number | null): void {
  previewHeadroom =
    value === null || !Number.isFinite(value) ? null : Math.min(Math.max(value, 1), MAX_HEADROOM);
}

export function previewHeadroomOverride(): number | null {
  return previewHeadroom;
}

/**
 * `?hdr=off` forces the 8-bit swapchain; `?hdr=<number>` pins the headroom
 * instead of trusting the display, which is the only way to compare the two
 * grades on one machine. Anything else leaves the automatic behaviour alone.
 */
export function hdrOverride(search: string): 'off' | number | null {
  const value = new URLSearchParams(search).get('hdr');
  if (value === null) return null;
  if (value === 'off' || value === '0' || value === 'false') return 'off';
  const pinned = Number(value);
  if (Number.isFinite(pinned) && pinned >= 1) return Math.min(pinned, MAX_HEADROOM);
  return null;
}

/**
 * Configure the swapchain, preferring an extended-range float surface. Falls
 * back to the preferred 8-bit format if anything about that is unsupported —
 * `configure()` throws on an unknown format, and an older Chrome accepts the
 * `toneMapping` dictionary while ignoring it, so the mode is read back rather
 * than assumed.
 */
export function configureCanvas(
  context: GPUCanvasContext,
  device: GPUDevice,
  allowHdr: boolean,
): CanvasHdrState {
  const sdr = (detail: string): CanvasHdrState => {
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    return { format, extended: false, detail };
  };

  if (!allowHdr) return sdr('disabled by ?hdr=off');

  try {
    context.configure({
      device,
      format: 'rgba16float',
      alphaMode: 'opaque',
      toneMapping: { mode: 'extended' },
    });
  } catch (err) {
    return sdr(`rgba16float swapchain rejected: ${err instanceof Error ? err.message : err}`);
  }

  const mode = context.getConfiguration?.()?.toneMapping?.mode;
  if (mode !== 'extended') {
    return sdr(`extended tone mapping unsupported (reported "${mode ?? 'none'}")`);
  }

  return {
    format: 'rgba16float',
    extended: true,
    detail: hdrDisplay()
      ? `extended range, display headroom ${displayHeadroom().toFixed(2)}x`
      : 'extended range, SDR display (headroom 1x until one is attached)',
  };
}
