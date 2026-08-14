/**
 * The palette folder, shared by every substrate's panel.
 *
 * It started as three near-identical copies (physarum, plife, vizfx) of "K
 * colour pickers plus saturation plus brightness". Palette v2 turns that into a
 * mode selector, an arc, a global rotation, a cycle and a catalog, which is
 * enough surface that three copies would have drifted within a week. Nothing in
 * here is any one sim's: `Palette` is sim-agnostic (see `sim/palette.ts`) and
 * shared by reference with `ModulationConfig`, so a picker edit *is* an edit to
 * the object that gets serialised.
 *
 * The one thing the host has to supply beyond the palette itself is
 * `invalidate()` — the substrates cache the linearised colours and nothing polls,
 * so a widget that does not say it changed something changes nothing on screen.
 */
import {
  harmonizePalette,
  syncPaletteColors,
  MAX_HUE_RATE_DEG_PER_SEC,
  PALETTE_SPACES,
  type Palette,
  type PaletteSpace,
} from '../sim/palette.ts';
import { PALETTE_CATALOG, applyPaletteEntry, paletteCatalogEntry } from '../sim/palettes.ts';
import { saveNow, type PersistedContainer } from './autosave.ts';

export interface PaletteFolderHost {
  /** the live object, shared by reference with the mapping config */
  palette: Palette;
  speciesCount: number;
  /**
   * How many species the primaries' arc covers. plife's `PRIMARY_COUNT`; omit on
   * a substrate whose species are all primaries, where it defaults to K and the
   * accents' arc is neither shown nor read.
   */
  groupSize?: number;
  /** for picker labels; a sim with unnamed species can return '' */
  speciesName(index: number): string;
  /** the substrate's `invalidatePalette` — the cache is not polled */
  invalidate(): void;
}

/**
 * What each space buys, in the dropdown itself — the choice is a look, not a
 * technicality, and the label is the only place that is explained at the point
 * of use.
 */
const SPACE_LABELS: Record<PaletteSpace, string> = {
  hsl: 'hsl — vivid, uneven brightness',
  hsluv: 'hsluv — even brightness, flatter',
  oklch: 'oklch — even chroma, gamut-mapped',
};

/** Returns the panel's refresh hook: pulls proxy state and visibility back in. */
export function addPaletteFolder(
  container: PersistedContainer,
  host: PaletteFolderHost,
): () => void {
  const p = host.palette;
  const k = Math.max(1, host.speciesCount);
  const groupSize = Math.max(1, Math.min(host.groupSize ?? k, k));

  const root = container.addFolder({
    title: 'palette (static — never modulated)',
    expanded: false,
  });

  // One handler for the whole folder rather than per binding: tweakpane bubbles
  // change up through ancestor containers. Everything below then only has to add
  // the *sim-side* effect it needs (invalidate, or a colours resync).
  // Set while `refresh()` writes widget state back from the palette. Tweakpane
  // re-emits `change` from `refresh()`, which would otherwise bounce straight
  // back into this handler and autosave on every programmatic sync.
  let refreshing = false;

  root.on('change', () => {
    if (refreshing) return;
    host.invalidate();
  });

  // ── catalog ────────────────────────────────────────────────────────────────
  // A dropdown that *acts* rather than binds: the palette has no "which preset"
  // field, because a preset is a starting point you then edit, and remembering
  // the name would imply the palette still is that preset when it no longer is.
  // The proxy resets itself to the placeholder after every load for that reason.
  const CATALOG_NONE = '—';
  const catalogProxy = { entry: CATALOG_NONE };
  root
    .addBinding(catalogProxy, 'entry', {
      label: 'load catalog',
      options: {
        [CATALOG_NONE]: CATALOG_NONE,
        ...Object.fromEntries(PALETTE_CATALOG.map((e) => [`${e.name} — ${e.note}`, e.name])),
      },
    })
    .on('change', (ev) => {
      const entry = paletteCatalogEntry(ev.value);
      if (!entry) return;
      applyPaletteEntry(p, entry, k, groupSize);
      catalogProxy.entry = CATALOG_NONE;
      // An entry can switch the mode, so the visibility has to follow it.
      refresh();
    });

  // Switching authority re-derives the colours immediately: going to arc mode
  // must repaint from the arc rather than leave the old hexes on screen until
  // the first slider move, and the two sub-folders swap with it.
  root
    .addBinding(p, 'mode', {
      label: 'mode',
      options: { arc: 'arc', custom: 'custom' },
    })
    .on('change', () => {
      syncPaletteColors(p, k, undefined, groupSize);
      refresh();
    });

  // Which cylindrical space the arcs and the hue shift work in. It changes what
  // the same two arc numbers *mean*, so the derived colours are rebuilt on the
  // spot rather than waiting for the next slider move.
  root
    .addBinding(p, 'space', {
      label: 'colour space',
      options: Object.fromEntries(PALETTE_SPACES.map((s) => [SPACE_LABELS[s], s])),
    })
    .on('change', () => {
      syncPaletteColors(p, k, undefined, groupSize);
      refresh();
    });

  // ── arc ────────────────────────────────────────────────────────────────────
  // vuzic's scheme: one start, one sweep, one shared saturation and lightness.
  // Editing any of the four re-derives the serialised colour list so the file on
  // disk always shows what is on screen.
  const arcFolder = root.addFolder({ title: 'arc', expanded: true });
  const arcChanged = (): void => {
    syncPaletteColors(p, k, undefined, groupSize);
    syncColorProxy();
  };
  /**
   * One group's four knobs. Two groups exist: the primaries and, on a substrate
   * whose species come in stem-keyed pairs, the accents. They are independent
   * arcs — nothing here derives one from the other — because a single arc spread
   * across all K put each accent on its primary's complement.
   */
  const addArcControls = (folder: PersistedContainer, arc: typeof p.arc): void => {
    folder
      .addBinding(arc, 'hueStartDeg', { label: 'start (°)', min: 0, max: 360, step: 1 })
      .on('change', arcChanged);
    // Signed, because the sign *is* the direction control: -360..360 covers both
    // ways round the wheel without a second widget. Full range at either end
    // divides the whole wheel across the *group*, which is the case that looks
    // like vuzic.
    folder
      .addBinding(arc, 'hueRangeDeg', { label: 'spread (°)', min: -360, max: 360, step: 1 })
      .on('change', arcChanged);
    // The colour-space pair, not the linear-space trim in `global` below. 100
    // means "as colourful as this hue gets" in all three spaces — the gamut
    // boundary in HSLuv, gamut-mapped to it in OKLCh, full saturation in HSL.
    folder
      .addBinding(arc, 'sat', { label: 'saturation / chroma', min: 0, max: 100, step: 1 })
      .on('change', arcChanged);
    folder
      .addBinding(arc, 'light', { label: 'lightness', min: 0, max: 100, step: 1 })
      .on('change', arcChanged);
  };

  // Only split the folder when the substrate actually has a second group.
  // Physarum and vizfx have no accent species, so `groupSize` is K there and a
  // second set of controls would be four knobs wired to nothing.
  const hasAccents = groupSize < k;
  const primaryFolder = hasAccents
    ? arcFolder.addFolder({ title: `primaries (${groupSize})`, expanded: true })
    : arcFolder;
  addArcControls(primaryFolder, p.arc);
  if (hasAccents) {
    const accentFolder = arcFolder.addFolder({
      title: `accents (${k - groupSize})`,
      expanded: true,
    });
    addArcControls(accentFolder, p.accentArc);
  }

  // ── custom ─────────────────────────────────────────────────────────────────
  const customFolder = root.addFolder({ title: 'colours', expanded: true });
  const colorProxy: Record<string, string> = {};
  for (let i = 0; i < k; i++) {
    const key = `c${i}`;
    colorProxy[key] = p.colors[i] ?? '#ffffff';
    customFolder
      .addBinding(colorProxy, key, { label: `${i} ${host.speciesName(i)}` })
      .on('change', (ev) => {
        p.colors[i] = ev.value;
      });
  }
  // Not a binding: it computes new values for the pickers rather than editing
  // one, so it has to push them back through the proxy and ask for a refresh.
  customFolder.addButton({ title: 'harmonize (shared HSLuv S/L)' }).on('click', () => {
    harmonizePalette(p, k);
    syncColorProxy();
    host.invalidate();
    // A button is not a binding, so the container's wrapper does not see it —
    // this is the one place in the folder that has to ask for a save by hand,
    // and it is conspicuous rather than forgettable.
    saveNow(container);
  });

  // ── global ─────────────────────────────────────────────────────────────────
  const global = root.addFolder({ title: 'global', expanded: true });
  // Applies in both modes and rotates in HSLuv hue space, so it preserves every
  // colour's saturation and lightness. It is the *authored base* of the cycle,
  // never the current effective angle — an export replays from sim time zero and
  // reads exactly this number.
  global.addBinding(p, 'hueShiftDeg', { label: 'hue shift (°)', min: 0, max: 360, step: 1 });
  // The knob the user described from vuzic: 0 is a fixed set of hues, and
  // anything above it is a slow lap of the wheel. 0.05°/s steps because the
  // interesting settings are down at "one lap per few minutes" — 1°/s is already
  // a six-minute lap, and the top of the range is a 36-second one.
  global.addBinding(p, 'hueRateDegPerSec', {
    label: 'cycle °/s (0 = fixed)',
    min: 0,
    max: MAX_HUE_RATE_DEG_PER_SEC,
    step: 0.05,
  });
  // The two v1 knobs, unchanged in meaning: a linear-space lerp against luminance
  // and a linear-space multiply, applied after the colour has been chosen. Named
  // apart from the arc's HSLuv pair on purpose — they are different instruments.
  global.addBinding(p, 'saturation', { label: 'trim: saturation', min: 0, max: 2, step: 0.01 });
  global.addBinding(p, 'brightness', { label: 'trim: brightness', min: 0, max: 2, step: 0.01 });

  function syncColorProxy(): void {
    for (let i = 0; i < k; i++) colorProxy[`c${i}`] = p.colors[i] ?? '#ffffff';
  }

  /**
   * Pull proxy state and folder visibility back in.
   *
   * Called once during construction, not only by the host: the shipped palettes
   * are all in custom mode, so a folder that only ever hid itself on a *later*
   * reroll or config load would show the arc controls from startup — live
   * sliders wired to a block that custom mode does not read, which reads as
   * "the arc does nothing".
   */
  function refresh(): void {
    syncColorProxy();
    // Only one of the two authorities is ever live, and showing the other is how
    // you end up editing hexes that the arc overwrites on its next change.
    arcFolder.hidden = p.mode !== 'arc';
    customFolder.hidden = p.mode !== 'custom';
    // The pickers are bound to a proxy, so rewriting it is not enough — the
    // widgets have to be told to re-read. Cheap: this folder is a few dozen rows.
    refreshing = true;
    try {
      root.refresh();
    } finally {
      refreshing = false;
    }
  }

  refresh();
  return refresh;
}
