/**
 * The phase-7 render chain's folder, shared by every substrate's panel.
 *
 * This started life as a private function inside `panel.ts` and moved here the
 * moment a second sim needed it. Nothing about the HDR chain is physarum's: the
 * `RenderConfig` block is already sim-agnostic (see `sim/render/config.ts`), the
 * auto-exposure controller lives in `PostFx`, and the only sim-specific pieces
 * are the two readouts, the scene exposure/gamma pair and physarum's soil
 * underlay. Those four are the whole of `RenderFolderHost` — everything else in
 * here is the same instrument whoever is playing it.
 */
import type { Pane } from 'tweakpane';
import { MAX_BLOOM_LEVELS, TONEMAPS, type RenderConfig, type ToneMap } from '../sim/render/config';

export interface RenderFolderHost {
  render: RenderConfig;
  /** the θ scene-exposure/gamma pair — both sims have these fields on their config */
  config: { exposure: number; gamma: number };
  /**
   * Hard bounds for the scene-exposure slider. Per-sim rather than a constant
   * here: physarum composites an accumulated trail field and lives around 0.01,
   * plife splats alpha-weighted sprites and lives around 1.0, so a single range
   * would give one of them a slider that is unusable across nine tenths of its
   * travel. Each caller passes the bound its own θ registry states.
   */
  exposureRange: { min: number; max: number; step: number };
  renderPasses(): number;
  autoExposureState(): { gain: number; mean: number };
  invalidatePalette(): void;
  /** physarum's soil-underlay controls; omit for sims with no soil */
  soil?: boolean;
}

/**
 * Phase 7's controls, in one folder because they are one instrument: the whole
 * chain runs at once and the only way to tune a grade is to watch it move.
 *
 * Ordered the way the pixels flow — scene exposure, adaptation, bloom, tone,
 * grade, feedback — so scrolling down the folder walks the pipeline.
 */
export function addRenderFolder(pane: Pane, host: RenderFolderHost): () => void {
  const config = host.config;
  const r = host.render;
  const readout = { passes: '—', adapt: '—' };

  const root = pane.addFolder({ title: 'render · HDR chain (phase 7)', expanded: true });
  root.addBinding(readout, 'passes', { readonly: true, label: 'render passes' });
  // The measured HDR mean is the number to aim `target mean` at; the gain is
  // where the controller has settled. Both are one or two frames stale.
  root.addBinding(readout, 'adapt', { readonly: true, label: 'gain / mean' });

  // Scene exposure and gamma are in θ but **excluded from modulation** — see the
  // exclusion note in the owning sim's preset registry. They are yours, always.
  const scene = root.addFolder({ title: 'scene exposure (manual only)', expanded: true });
  // min/max match the θ bound in the caller's preset registry
  scene.addBinding(config, 'exposure', { ...host.exposureRange, label: 'exposure' });
  scene.addBinding(config, 'gamma', { min: 1, max: 3, step: 0.05, label: 'display gamma' });
  scene.addBinding(r.grade, 'exposureEv', { min: -8, max: 8, step: 0.05, label: 'trim (stops)' });

  const auto = root.addFolder({ title: 'auto-exposure', expanded: false });
  auto.addBinding(r.grade, 'autoExposure', { label: 'enabled' });
  auto.addBinding(r.grade, 'autoTarget', { min: 0.01, max: 2, step: 0.01, label: 'target mean' });
  auto.addBinding(r.grade, 'autoTau', { min: 0.2, max: 30, step: 0.1, label: 'τ (seconds)' });
  auto.addBinding(r.grade, 'autoMinGain', { min: 0.01, max: 1, step: 0.01, label: 'min gain' });
  auto.addBinding(r.grade, 'autoMaxGain', { min: 1, max: 32, step: 0.5, label: 'max gain' });

  const bloom = root.addFolder({ title: 'bloom', expanded: true });
  bloom.addBinding(r.bloom, 'enabled');
  bloom.addBinding(r.bloom, 'threshold', { min: 0, max: 4, step: 0.01 });
  bloom.addBinding(r.bloom, 'knee', { min: 0.01, max: 2, step: 0.01, label: 'soft knee' });
  bloom.addBinding(r.bloom, 'intensity', { min: 0, max: 4, step: 0.01 });
  bloom.addBinding(r.bloom, 'levels', { min: 1, max: MAX_BLOOM_LEVELS, step: 1, label: 'mips' });

  const tone = root.addFolder({ title: 'tone + grade', expanded: true });
  tone.addBinding(r.grade, 'tonemap', {
    options: Object.fromEntries(TONEMAPS.map((t) => [t, t])) as Record<string, ToneMap>,
  });
  tone.addBinding(r.grade, 'blackPoint', { min: 0, max: 0.3, step: 0.001, label: 'black point' });
  tone.addBinding(r.grade, 'contrast', { min: 0.5, max: 2.5, step: 0.01 });
  tone.addBinding(r.grade, 'pivot', { min: 0.05, max: 0.8, step: 0.01, label: 'contrast pivot' });
  tone.addBinding(r.grade, 'saturation', { min: 0, max: 2, step: 0.01 });
  tone.addBinding(r.grade, 'vignette', { min: 0, max: 1, step: 0.01 });

  // Only drawn for a sim that has a soil field to draw it from. `soilTint` is
  // forced to 0 in plife's shipped render config for the same reason, so a sim
  // without soil neither shows the controls nor silently carries the effect.
  if (host.soil === true) {
    const ground = root.addFolder({ title: 'soil underlay', expanded: false });
    ground.addBinding(r.grade, 'soilTint', { min: 0, max: 1.5, step: 0.005, label: 'strength' });
    // Cached alongside the palette, and invalidated the same way.
    ground
      .addBinding(r.grade, 'soilColor', { label: 'colour (static)' })
      .on('change', () => host.invalidatePalette());
  }

  // Render-domain only: this cannot touch the trail field, so it is safe to
  // push. It is also the one control here that trades legibility for looks.
  const fb = root.addFolder({ title: 'feedback (render-domain trail)', expanded: false });
  fb.addBinding(r.feedback, 'amount', { min: 0, max: 0.95, step: 0.01 });
  fb.addBinding(r.feedback, 'zoom', { min: 0.98, max: 1.02, step: 0.0005 });

  return (): void => {
    readout.passes = String(host.renderPasses());
    const a = host.autoExposureState();
    readout.adapt = `${a.gain.toFixed(2)}×  ·  ${a.mean.toFixed(4)}`;
  };
}
