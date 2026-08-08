import type { GpuContext } from '../gpu/context';
import type { FeaturesFrame } from '../timeline/sampler';

/**
 * The seam the simulation slots into. The main loop owns the audio clock, the sampler and
 * the frame pacing; a Sim only advances its own state and draws.
 *
 * `tick` is called once per fixed sim step (possibly several times per rAF, possibly zero)
 * with the timeline sampled at that tick. `render` is called at most once per rAF.
 * `frame.values` is a reused buffer — read it, do not retain it.
 */
export interface Sim {
  readonly name: string;
  init(ctx: GpuContext): Promise<void>;
  tick(frame: FeaturesFrame, simTick: number): void;
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void;
  /**
   * The sim-specific middle of the app's status line — grid, population, seed,
   * whatever this substrate's readout is. It lives here rather than in main
   * because the string is entirely made of numbers only the sim has, and a
   * second substrate has a different set of them; main owns the track, audio,
   * mode and gpu segments, which are the same whatever is running.
   */
  status(): string;
  dispose(): void;
}

/** Placeholder until phase 4. Clears the canvas; tints slightly with stem energy. */
export class NullSim implements Sim {
  readonly name = 'null';
  private stems = new Float32Array(4);
  private stemOffset = 0;

  async init(_ctx: GpuContext): Promise<void> {
    void _ctx;
  }

  setStemOffset(offset: number): void {
    this.stemOffset = offset;
  }

  tick(frame: FeaturesFrame, _simTick: number): void {
    void _simTick;
    for (let i = 0; i < 4; i++) {
      this.stems[i] = frame.values[this.stemOffset + i] ?? 0;
    }
  }

  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    const energy = ((this.stems[0] ?? 0) + (this.stems[1] ?? 0) + (this.stems[2] ?? 0) + (this.stems[3] ?? 0)) / 4;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          clearValue: {
            r: 0.02 + 0.05 * energy,
            g: 0.025 + 0.04 * energy,
            b: 0.04 + 0.09 * energy,
            a: 1,
          },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.end();
  }

  /** Nothing to report: the status line simply loses its middle segment. */
  status(): string {
    return '';
  }

  dispose(): void {}
}
