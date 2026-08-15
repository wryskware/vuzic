/**
 * The workbench's "events" folder — the tuning surface for the impulse lane.
 *
 * The per-kind test-fire buttons are the point of this panel: tuning a snare
 * splash by waiting for the song to reach a snare is unusable, so every kind can
 * be fired synthetically at full strength, right now, with the transport stopped.
 * A test fire goes through exactly the same path as a timeline event (same
 * envelope, same hashed hotspots), so what you tune is what you get.
 */
import { MAX_WIGGLE, type ImpulseEngine, type ResponseConfig } from '../sim/impulses';
import { EVENT_KINDS, type EventKind } from '../timeline/types';
import type { PersistedContainer } from './autosave';

export interface ImpulsePanelHandle {
  refresh(): void;
  dispose(): void;
}

const SPECIES_HINT: Record<EventKind, string> = {
  kick: 'deposit burst + flash',
  snare: 'radial splash',
  hat: 'fast shimmer',
  bass: 'sensor pop',
  vocal: 'wide pulse',
};

/**
 * The container is a `PersistedContainer` and that is not a convention — it is
 * the type. Every binding below saves because the container makes it save, and
 * a caller cannot pass a folder that does not, which is what this folder's own
 * history argues for: until 2026-08-13 it bound the live `ImpulseConfig` and
 * nothing ever wrote it anywhere, so every splash radius and decay τ tuned here
 * died on refresh.
 */
export function createImpulsePanel(
  container: PersistedContainer,
  engine: ImpulseEngine,
  speciesName: (index: number) => string,
): ImpulsePanelHandle {
  const cfg = engine.config;
  /**
   * **Session-only, by declaration.** These three are bound to *this* object and
   * not to `ImpulseConfig`, and that is the opt-out: nothing serialises this
   * object, so nothing can restore it. Said here rather than by being quietly
   * absent from a save list — the same rule `mapping/blocks.ts` enforces one
   * level up for whole config blocks, and `block-registry.test.ts` asserts that
   * `hold` never appears in `defaultImpulseConfig()`.
   *
   * `status` and `levels` are readouts `refresh()` rewrites every frame.
   * `hold` is a tuning *stance*, not a setting: it pins one kind's envelope open
   * so the lane can be judged as a state rather than as a 200 ms transient, and a
   * reload that restored it would bring back a world stuck permanently mid-hit
   * with nothing on screen to explain it.
   */
  const sessionOnly = {
    status: '—',
    levels: '—',
    /** '' = nothing held; otherwise the kind whose envelope is pinned open. */
    hold: '',
  };

  // Collapsed by default now that it shares the map tab with the driver bank:
  // six folders' worth of per-kind tuning under an expanded header pushed
  // everything else off the bottom of the pane.
  // Wrapped, so every binding below saves by construction — including the two
  // readonly readouts' neighbours, and including whatever a future event kind
  // adds. The readouts themselves are skipped by `persisting` from their own
  // `readonly` flag, which matters: `refresh()` rewrites both every frame.
  const root = container.addFolder({ title: 'events (impulse lane)', expanded: false });
  root.addBinding(sessionOnly, 'status', { readonly: true, label: '' });
  root.addBinding(cfg, 'enabled', { label: 'impulses on' });
  root.addBinding(cfg, 'gain', { min: 0, max: 3, step: 0.05, label: 'global gain' });
  root.addBinding(sessionOnly, 'levels', { readonly: true, label: 'env' });
  root.addButton({ title: 'test: fire all kinds' }).on('click', () => {
    for (const kind of EVENT_KINDS) engine.testFire(kind);
  });

  // The isolation control, and the answer to "is this lane doing anything?".
  //
  // Every lane of a response rides one 90–380 ms envelope, so by eye they are
  // one event and you cannot tell the flash from the wiggle from the splash.
  // Holding a kind pins its envelope open: the transient becomes a state the
  // field settles into, the per-lane depths below become a mixing desk you can
  // solo (zero the ones you are not asking about), and switching the hold off
  // is a clean A/B against the world you are looking at.
  //
  // Runtime only — deliberately not in `ImpulseConfig`; see `sessionOnly` above.
  root
    .addBinding(sessionOnly, 'hold', {
      label: 'hold (isolate)',
      options: {
        off: '',
        ...Object.fromEntries(EVENT_KINDS.map((k) => [k, k])),
      } as Record<string, string>,
    })
    .on('change', (ev) => {
      engine.setHold(ev.value === '' ? null : (ev.value as EventKind));
    });

  for (const kind of EVENT_KINDS) {
    const r = cfg.responses[kind];
    const f = root.addFolder({ title: `${kind} · ${SPECIES_HINT[kind]}`, expanded: false });
    f.addButton({ title: `▶ test-fire ${kind}` }).on('click', () => engine.testFire(kind));
    f.addBinding(r, 'enabled');
    f.addBinding(r, 'species', {
      min: -1,
      max: 7,
      step: 1,
      label: 'species (-1 = all)',
    }).on('change', (ev) => {
      f.title = `${kind} → ${ev.value < 0 ? 'all' : speciesName(ev.value)}`;
    });
    f.addBinding(r, 'decayMs', { min: 20, max: 1500, step: 5, label: 'decay τ (ms)' });
    addDepths(f, r);
  }

  return {
    refresh(): void {
      const n = engine.eventCount;
      const held = engine.heldKind;
      sessionOnly.status = n === 0 ? 'timeline has no events' : `${n} events · ${engine.activeSplashes} splashes live`;
      // Loud, because a held envelope is a world that is not what the music is
      // doing, and forgetting one open is exactly how "the sim looks wrong" gets
      // reported against a build that is fine.
      if (held) sessionOnly.status = `⏸ HOLDING "${held}" · ${sessionOnly.status}`;
      sessionOnly.levels = EVENT_KINDS.map((k) => `${k[0]}${engine.levelOf(k).toFixed(2)}`).join(' ');
    },
    dispose(): void {
      root.dispose();
    },
  };
}

function addDepths(parent: PersistedContainer, r: ResponseConfig): void {
  parent.addBinding(r, 'deposit', { min: 0, max: 6, step: 0.05, label: 'deposit burst ×' });
  parent.addBinding(r, 'flash', { min: 0, max: 4, step: 0.05, label: 'brightness flash ×' });
  parent.addBinding(r, 'sensor', { min: 0, max: 3, step: 0.05, label: 'sensor pop ×' });
  // The wiggle lane. The depth is in multiples of the matrix's own strongest
  // cell, so 1 already displaces the row by as much as the whole matrix is
  // worth, and the shipped depths are 3-4. The plife macro `wiggle` scales all
  // of them at once; this is the per-kind share, and it does nothing in
  // substrates with no interaction matrix.
  parent.addBinding(r, 'wiggle', {
    min: 0,
    max: MAX_WIGGLE,
    step: 0.05,
    label: 'matrix wiggle  (x max|A|)',
  });

  const s = parent.addFolder({ title: 'radial splash', expanded: false });
  s.addBinding(r, 'splashCount', { min: 0, max: 8, step: 1, label: 'discs / event' });
  s.addBinding(r, 'splashRadius', { min: 0.02, max: 0.8, step: 0.01, label: 'radius (of short axis)' });
  s.addBinding(r, 'splashPush', { min: 0, max: 20, step: 0.1, label: 'push (cells/tick)' });
  s.addBinding(r, 'splashSwirl', { min: -2, max: 2, step: 0.01, label: 'swirl (rad)' });
}
