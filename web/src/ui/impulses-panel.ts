/**
 * The workbench's "events" folder — the tuning surface for the impulse lane.
 *
 * The per-kind test-fire buttons are the point of this panel: tuning a snare
 * splash by waiting for the song to reach a snare is unusable, so every kind can
 * be fired synthetically at full strength, right now, with the transport stopped.
 * A test fire goes through exactly the same path as a timeline event (same
 * envelope, same hashed hotspots), so what you tune is what you get.
 */
import type { FolderApi } from 'tweakpane';
import type { ImpulseEngine, ResponseConfig } from '../sim/impulses';
import { EVENT_KINDS, type EventKind } from '../timeline/types';
import type { PanelContainer } from './panel';

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

export function createImpulsePanel(
  container: PanelContainer,
  engine: ImpulseEngine,
  speciesName: (index: number) => string,
): ImpulsePanelHandle {
  const cfg = engine.config;
  const ui = {
    status: '—',
    levels: '—',
  };

  // Collapsed by default now that it shares the map tab with the driver bank:
  // six folders' worth of per-kind tuning under an expanded header pushed
  // everything else off the bottom of the pane.
  const root = container.addFolder({ title: 'events (impulse lane)', expanded: false });
  root.addBinding(ui, 'status', { readonly: true, label: '' });
  root.addBinding(cfg, 'enabled', { label: 'impulses on' });
  root.addBinding(cfg, 'gain', { min: 0, max: 3, step: 0.05, label: 'global gain' });
  root.addBinding(ui, 'levels', { readonly: true, label: 'env' });
  root.addButton({ title: 'test: fire all kinds' }).on('click', () => {
    for (const kind of EVENT_KINDS) engine.testFire(kind);
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
      ui.status = n === 0 ? 'timeline has no events' : `${n} events · ${engine.activeSplashes} splashes live`;
      ui.levels = EVENT_KINDS.map((k) => `${k[0]}${engine.levelOf(k).toFixed(2)}`).join(' ');
    },
    dispose(): void {
      root.dispose();
    },
  };
}

function addDepths(parent: FolderApi, r: ResponseConfig): void {
  parent.addBinding(r, 'deposit', { min: 0, max: 6, step: 0.05, label: 'deposit burst ×' });
  parent.addBinding(r, 'flash', { min: 0, max: 4, step: 0.05, label: 'brightness flash ×' });
  parent.addBinding(r, 'sensor', { min: 0, max: 3, step: 0.05, label: 'sensor pop ×' });

  const s = parent.addFolder({ title: 'radial splash', expanded: false });
  s.addBinding(r, 'splashCount', { min: 0, max: 8, step: 1, label: 'discs / event' });
  s.addBinding(r, 'splashRadius', { min: 0.02, max: 0.8, step: 0.01, label: 'radius (of short axis)' });
  s.addBinding(r, 'splashPush', { min: 0, max: 20, step: 0.1, label: 'push (cells/tick)' });
  s.addBinding(r, 'splashSwirl', { min: -2, max: 2, step: 0.01, label: 'swirl (rad)' });
}
