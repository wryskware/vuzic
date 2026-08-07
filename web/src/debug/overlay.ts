import type { TimelineSampler } from '../timeline/sampler';
import { STEM_NAMES } from '../timeline/types';

interface Series {
  label: string;
  color: string;
  channel: string;
  dim: number;
  width: number;
}

interface Lane {
  title: string;
  series: Series[];
  /** fraction of the plot height */
  weight: number;
  min: number;
  max: number;
  /** draw a zero reference line when the range straddles zero */
  zeroLine: boolean;
}

/** plot gutters in CSS pixels; multiplied by dpr when drawing to the backing store */
const PAD_L = 62;
const PAD_R = 10;

const STEM_COLORS = ['#ffa23a', '#ff5f7e', '#57d6ff', '#a684ff'];
const LATENT_COLORS = ['#8ce99a', '#74c0fc', '#ffd43b', '#ff8787', '#b197fc', '#63e6be'];
const NOVELTY_COLORS = ['#ffffff', '#ffd43b'];

const SEGMENT_COLORS: Record<string, string> = {
  intro: '#3d4a63',
  verse: '#2f5d50',
  chorus: '#6b3b5e',
  bridge: '#5c4a2a',
  outro: '#3d3d4a',
};

export interface OverlayOptions {
  /** seconds of history shown to the left of the playhead */
  historySeconds?: number;
  /** seconds of lookahead shown to the right */
  futureSeconds?: number;
  /** number of latent dims plotted */
  latentDims?: number;
  onSeek?: (seconds: number) => void;
}

export class DebugOverlay {
  private readonly canvas: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D;
  private readonly sampler: TimelineSampler;
  private readonly lanes: Lane[];
  private readonly history: number;
  private readonly future: number;
  private w = 1;
  private h = 1;
  private dpr = 1;

  constructor(canvas: HTMLCanvasElement, sampler: TimelineSampler, opts: OverlayOptions = {}) {
    this.canvas = canvas;
    this.sampler = sampler;
    this.history = opts.historySeconds ?? 10;
    this.future = opts.futureSeconds ?? 4;

    const g = canvas.getContext('2d', { alpha: true });
    if (!g) throw new Error('overlay: no 2d context');
    this.g = g;

    const latentCount = Math.min(opts.latentDims ?? 6, sampler.getChannel('latent')?.dims ?? 0);

    const stemSeries: Series[] = STEM_NAMES.filter(() => sampler.hasChannel('stems')).map(
      (name, i) => ({
        label: name,
        color: STEM_COLORS[i] as string,
        channel: 'stems',
        dim: i,
        width: 2,
      }),
    );

    const latentSeries: Series[] = Array.from({ length: latentCount }, (_, i) => ({
      label: `z${i}`,
      color: LATENT_COLORS[i % LATENT_COLORS.length] as string,
      channel: 'latent',
      dim: i,
      width: 1,
    }));

    const noveltySeries: Series[] = (['novelty4', 'novelty16'] as const)
      .filter((n) => sampler.hasChannel(n))
      .map((n, i) => ({
        label: n,
        color: NOVELTY_COLORS[i] as string,
        channel: n,
        dim: 0,
        width: 1.5,
      }));

    const lanes: Lane[] = [];
    if (stemSeries.length)
      lanes.push({ title: 'stems', series: stemSeries, weight: 0.42, ...range(sampler, stemSeries) });
    if (latentSeries.length)
      lanes.push({
        title: `latent (${latentCount} of ${sampler.getChannel('latent')?.dims ?? 0} dims)`,
        series: latentSeries,
        weight: 0.36,
        ...range(sampler, latentSeries),
      });
    if (noveltySeries.length)
      lanes.push({
        title: 'novelty',
        series: noveltySeries,
        weight: 0.22,
        ...range(sampler, noveltySeries),
      });
    this.lanes = lanes;

    if (opts.onSeek) {
      const seek = opts.onSeek;
      canvas.addEventListener('pointerdown', (ev) => {
        // Inverse of draw()'s x(t): the plot occupies rect minus the same gutters.
        const rect = canvas.getBoundingClientRect();
        const plotW = Math.max(rect.width - PAD_L - PAD_R, 1);
        const frac = (ev.clientX - rect.left - PAD_L) / plotW;
        const span = this.history + this.future;
        seek(this.lastTime + (frac - this.history / span) * span);
      });
    }
    this.resize();
  }

  private lastTime = 0;

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this.w && h === this.h && dpr === this.dpr) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.w = w;
    this.h = h;
    this.dpr = dpr;
  }

  draw(time: number, info: { tick: number; playing: boolean; seed: number }): void {
    this.resize();
    this.lastTime = time;

    const g = this.g;
    const s = this.dpr;
    const W = this.w;
    const H = this.h;
    const padL = PAD_L * s;
    const padR = PAD_R * s;
    const padT = 34 * s;
    const padB = 20 * s;
    const plotW = Math.max(1, W - padL - padR);
    const plotH = Math.max(1, H - padT - padB);

    const span = this.history + this.future;
    const t0 = time - this.history;
    const t1 = time + this.future;
    const x = (t: number) => padL + ((t - t0) / span) * plotW;

    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(8,9,14,0.82)';
    g.fillRect(0, 0, W, H);

    this.drawSegments(g, x, t0, t1, padT, plotH, s, padL, plotW);
    this.drawBeats(g, x, t0, t1, padT, plotH, s);

    let y = padT;
    for (const lane of this.lanes) {
      const laneH = plotH * lane.weight;
      this.drawLane(g, lane, x, t0, t1, y, laneH - 8 * s, s, padL, plotW, time);
      y += laneH;
    }

    // playhead
    const px = x(time);
    g.strokeStyle = '#ffffff';
    g.lineWidth = 1.5 * s;
    g.beginPath();
    g.moveTo(px, padT - 12 * s);
    g.lineTo(px, padT + plotH);
    g.stroke();

    this.drawHeader(g, time, info, s, W, padL);
  }

  private drawHeader(
    g: CanvasRenderingContext2D,
    time: number,
    info: { tick: number; playing: boolean; seed: number },
    s: number,
    W: number,
    padL: number,
  ): void {
    const seg = this.sampler.segmentAt(time);
    const beat = this.sampler.beatStateAt(time);
    const bar = beat.downbeatIndex >= 0 ? beat.downbeatIndex + 1 : 0;
    g.font = `${11 * s}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.textBaseline = 'top';
    g.fillStyle = '#e6e9f0';
    const left = `${info.playing ? '▶' : '‖'} ${fmtTime(time)} / ${fmtTime(
      this.sampler.duration,
    )}   tick ${info.tick}   bar ${bar} beat ${beat.index >= 0 ? (beat.index % 4) + 1 : '-'}`;
    g.fillText(left, padL, 8 * s);

    const label = seg ? `${seg.label} (${seg.confidence.toFixed(2)})` : '—';
    const right = `${label}   ${this.sampler.timeline.manifest.tempo.toFixed(1)} BPM   seed ${info.seed}`;
    g.textAlign = 'right';
    g.fillStyle = '#9aa3b8';
    g.fillText(right, W - PAD_R * s, 8 * s);
    g.textAlign = 'left';
  }

  private drawSegments(
    g: CanvasRenderingContext2D,
    x: (t: number) => number,
    t0: number,
    t1: number,
    padT: number,
    plotH: number,
    s: number,
    padL: number,
    plotW: number,
  ): void {
    g.save();
    g.beginPath();
    g.rect(padL, padT - 14 * s, plotW, plotH + 14 * s);
    g.clip();

    for (const seg of this.sampler.timeline.manifest.segments) {
      if (seg.end < t0 || seg.start > t1) continue;
      const xa = x(seg.start);
      const xb = x(seg.end);
      g.fillStyle = `${SEGMENT_COLORS[seg.label] ?? '#444a5c'}55`;
      g.fillRect(xa, padT, xb - xa, plotH);

      g.strokeStyle = '#c9d1e8';
      g.lineWidth = 1 * s;
      g.setLineDash([4 * s, 3 * s]);
      g.beginPath();
      g.moveTo(xa, padT - 14 * s);
      g.lineTo(xa, padT + plotH);
      g.stroke();
      g.setLineDash([]);

      g.font = `${10 * s}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      g.fillStyle = '#c9d1e8';
      g.textBaseline = 'bottom';
      g.fillText(seg.label, xa + 4 * s, padT - 2 * s);
      g.textBaseline = 'top';
    }
    g.restore();
  }

  private drawBeats(
    g: CanvasRenderingContext2D,
    x: (t: number) => number,
    t0: number,
    t1: number,
    padT: number,
    plotH: number,
    s: number,
  ): void {
    const { beats, downbeats } = this.sampler.timeline.manifest;
    const downSet = new Set(downbeats);
    for (const b of beats) {
      if (b < t0) continue;
      if (b > t1) break;
      const down = downSet.has(b);
      g.strokeStyle = down ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.07)';
      g.lineWidth = (down ? 1.2 : 0.8) * s;
      g.beginPath();
      g.moveTo(x(b), padT + plotH - (down ? 14 : 7) * s);
      g.lineTo(x(b), padT + plotH);
      g.stroke();
    }
  }

  private drawLane(
    g: CanvasRenderingContext2D,
    lane: Lane,
    x: (t: number) => number,
    t0: number,
    t1: number,
    top: number,
    height: number,
    s: number,
    padL: number,
    plotW: number,
    now: number,
  ): void {
    const hop = this.sampler.timeline.manifest.grid.hopSeconds;
    const frames = this.sampler.timeline.manifest.grid.frames;
    const f0 = Math.max(0, Math.floor(t0 / hop) - 1);
    const f1 = Math.min(frames - 1, Math.ceil(t1 / hop) + 1);
    const scale = (v: number) =>
      top + height - ((v - lane.min) / Math.max(lane.max - lane.min, 1e-6)) * height;

    g.strokeStyle = 'rgba(255,255,255,0.12)';
    g.lineWidth = 1 * s;
    g.beginPath();
    g.moveTo(padL, top + height);
    g.lineTo(padL + plotW, top + height);
    g.stroke();

    if (lane.zeroLine && lane.min < 0 && lane.max > 0) {
      g.strokeStyle = 'rgba(255,255,255,0.10)';
      g.setLineDash([2 * s, 4 * s]);
      g.beginPath();
      g.moveTo(padL, scale(0));
      g.lineTo(padL + plotW, scale(0));
      g.stroke();
      g.setLineDash([]);
    }

    g.font = `${10 * s}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.fillStyle = '#7d879c';
    g.textBaseline = 'top';
    g.fillText(lane.title, 6 * s, top);
    g.fillText(lane.max.toFixed(1), 6 * s, top + 12 * s);
    g.fillText(lane.min.toFixed(1), 6 * s, top + height - 10 * s);

    g.save();
    g.beginPath();
    g.rect(padL, top - 2 * s, plotW, height + 4 * s);
    g.clip();

    for (const ser of lane.series) {
      g.strokeStyle = ser.color;
      g.lineWidth = ser.width * s;
      g.beginPath();
      for (let f = f0; f <= f1; f++) {
        const px = x(f * hop);
        const py = scale(this.sampler.rawAt(f, ser.channel, ser.dim));
        if (f === f0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.stroke();
    }
    g.restore();

    // legend with live values at the playhead
    let lx = padL + 4 * s;
    const ly = top + 1 * s;
    for (const ser of lane.series) {
      const v = this.sampler.rawAt(Math.round(now / hop), ser.channel, ser.dim);
      const text = `${ser.label} ${v.toFixed(2)}`;
      g.fillStyle = ser.color;
      g.fillRect(lx, ly + 3 * s, 8 * s, 3 * s);
      g.fillText(text, lx + 11 * s, ly);
      lx += g.measureText(text).width + 24 * s;
    }
  }
}

function range(
  sampler: TimelineSampler,
  series: Series[],
): { min: number; max: number; zeroLine: boolean } {
  const frames = sampler.timeline.manifest.grid.frames;
  let min = Infinity;
  let max = -Infinity;
  for (let f = 0; f < frames; f++) {
    for (const s of series) {
      const v = sampler.rawAt(f, s.channel, s.dim);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!isFinite(min) || !isFinite(max)) return { min: 0, max: 1, zeroLine: false };
  const pad = (max - min) * 0.08 || 0.1;
  return { min: min - pad, max: max + pad, zeroLine: min < 0 };
}

function fmtTime(t: number): string {
  const m = Math.floor(t / 60);
  const sec = t - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
}
