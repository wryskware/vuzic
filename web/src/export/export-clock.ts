import { TICK_HZ } from '../timing.ts';

/** Deterministic parity clock: 120 output frames/s over the 120 Hz app clock. */
export const EXPORT_RENDER_HZ = 120;
export const EXPORT_SIMULATION_HZ = TICK_HZ;
export const EXPORT_RENDER_DELTA_SECONDS = 1 / EXPORT_RENDER_HZ;

export interface ExportRenderFrame {
  frameIndex: number;
  timeSeconds: number;
  deltaSeconds: number;
  targetSimulationTick: number;
}

function safeCounter(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
  return value;
}

/** One sample at each 120 Hz frame time strictly before the audio ends. */
export function exportFrameCount(durationSeconds: number): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds < 0) {
    throw new RangeError('durationSeconds must be a finite non-negative number');
  }
  return safeCounter(Math.ceil(durationSeconds * EXPORT_RENDER_HZ), 'frame count');
}

/** Exact integer cadence; no accumulated floating-point timestep participates. */
export function targetSimulationTick(frameIndex: number): number {
  safeCounter(frameIndex, 'frameIndex');
  // The current parity ratio is 1:1. Keep the rational form explicit so this
  // remains drift-free if the output and app rates diverge in a future mode.
  const divisor = EXPORT_RENDER_HZ / EXPORT_SIMULATION_HZ;
  if (Number.isSafeInteger(divisor)) return Math.floor(frameIndex / divisor);
  const tick = Math.floor(frameIndex * EXPORT_SIMULATION_HZ / EXPORT_RENDER_HZ);
  return safeCounter(tick, 'target simulation tick');
}

/**
 * Stateful scheduler with the ordering in its API name and implementation:
 * every missing simulation tick runs in order, then exactly one frame renders.
 */
export class ExportClock {
  readonly frameCount: number;
  private nextFrameIndex = 0;
  private currentSimulationTick = 0;

  constructor(durationSeconds: number) {
    this.frameCount = exportFrameCount(durationSeconds);
  }

  get frameIndex(): number {
    return this.nextFrameIndex;
  }

  get simulationTick(): number {
    return this.currentSimulationTick;
  }

  get done(): boolean {
    return this.nextFrameIndex >= this.frameCount;
  }

  advanceThenRender(
    advance: (simulationTick: number) => void,
    render: (frame: ExportRenderFrame) => void,
  ): ExportRenderFrame | null {
    if (this.done) return null;

    const frameIndex = this.nextFrameIndex;
    const targetTick = targetSimulationTick(frameIndex);
    while (this.currentSimulationTick < targetTick) {
      this.currentSimulationTick++;
      advance(this.currentSimulationTick);
    }

    const frame: ExportRenderFrame = {
      frameIndex,
      timeSeconds: frameIndex / EXPORT_RENDER_HZ,
      deltaSeconds: EXPORT_RENDER_DELTA_SECONDS,
      targetSimulationTick: targetTick,
    };
    render(frame);
    this.nextFrameIndex++;
    return frame;
  }
}
