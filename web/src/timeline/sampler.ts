import type { ChannelSpec, Segment, Timeline, TimelineEvent } from './types';

/** Largest i with arr[i] <= x, or -1. */
function lowerIndex(arr: readonly number[], x: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if ((arr[mid] as number) <= x) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

export interface FeaturesFrame {
  /** sim tick this frame was sampled for */
  tick: number;
  /** track time in seconds */
  time: number;
  /** interpolated row, length = timeline.stride. Reused between calls — do not retain. */
  values: Float32Array;
}

export interface BeatState {
  /** index of the most recent beat, -1 before the first */
  index: number;
  /** 0..1 position between that beat and the next */
  phase: number;
  /** most recent beat is also a downbeat */
  isDownbeat: boolean;
  /** index of the most recent downbeat, -1 before the first */
  downbeatIndex: number;
  /** seconds since the most recent beat */
  sinceBeat: number;
}

/**
 * Half-open index range into the event array: `[start, end)`. Because events are
 * sorted by `t` and the cursor only ever walks forward, everything that fires in
 * one tick window is contiguous, so a range beats allocating a list per tick.
 */
export interface EventRange {
  start: number;
  end: number;
}

/**
 * Playback advances the cursor one tick at a time, so any other movement is a
 * seek and relocates by binary search. The two paths are *not* equivalent: the
 * walk fires everything it steps over, so a forward jump that walked would dump
 * the whole gap's worth of events into one tick (two seconds of hats and kicks
 * firing together). The relocate path discards the skipped events instead, which
 * is what a seek should do — and it is what a backwards seek has always done.
 */
const MAX_WALK_GAP_TICKS = 1;

/**
 * Pointer walk over the sparse event stream, indexed by sim tick like everything
 * else. Events are sorted, so steady playback is O(events fired); a seek re-locates
 * the pointer by binary search and never replays what it skipped over.
 */
export class EventCursor {
  readonly events: readonly TimelineEvent[];
  readonly secondsPerTick: number;
  private ptr = 0;
  private lastTick = -1;

  constructor(events: readonly TimelineEvent[], secondsPerTick: number) {
    this.events = events;
    this.secondsPerTick = secondsPerTick;
  }

  /**
   * The tick an event lands on. Nearest tick rather than floor: at 60 Hz a tick is
   * 16.7 ms and rounding halves the worst-case timing error, which is audible-scale
   * on a transient. Deterministic, and monotonic in `t`, which is what the walk needs.
   */
  tickOf(index: number): number {
    const e = this.events[index];
    return e === undefined ? Number.POSITIVE_INFINITY : Math.round(e.t / this.secondsPerTick);
  }

  get position(): number {
    return this.ptr;
  }

  /** Forget where we are; the next advance() relocates. */
  reset(): void {
    this.ptr = 0;
    this.lastTick = -1;
  }

  /**
   * Events firing at `tick`, i.e. every event not yet consumed whose tick is <= it.
   * Calling twice with the same tick returns an empty range — a paused transport
   * re-ticking the same position does not re-fire.
   */
  advance(tick: number): EventRange {
    const gap = tick - this.lastTick;
    if (this.lastTick < 0 || gap < 0 || gap > MAX_WALK_GAP_TICKS) {
      this.relocate(tick);
    }
    const start = this.ptr;
    const n = this.events.length;
    while (this.ptr < n && this.tickOf(this.ptr) <= tick) this.ptr++;
    this.lastTick = tick;
    return { start, end: this.ptr };
  }

  /** First index whose tick is >= `tick`; everything strictly before it is skipped. */
  private relocate(tick: number): void {
    let lo = 0;
    let hi = this.events.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.tickOf(mid) < tick) lo = mid + 1;
      else hi = mid;
    }
    this.ptr = lo;
  }
}

export class TimelineSampler {
  readonly timeline: Timeline;
  readonly secondsPerTick: number;
  private readonly row: Float32Array;
  private readonly frame: FeaturesFrame;

  constructor(timeline: Timeline, secondsPerTick: number) {
    this.timeline = timeline;
    this.secondsPerTick = secondsPerTick;
    this.row = new Float32Array(timeline.stride);
    this.frame = { tick: 0, time: 0, values: this.row };
  }

  get duration(): number {
    return this.timeline.manifest.track.duration;
  }

  /** Sparse transient events; empty when the timeline carries none. */
  get events(): readonly TimelineEvent[] {
    return this.timeline.events;
  }

  /** A cursor over this timeline's events on this sampler's tick rate. */
  eventCursor(): EventCursor {
    return new EventCursor(this.timeline.events, this.secondsPerTick);
  }

  get totalTicks(): number {
    return Math.ceil(this.duration / this.secondsPerTick);
  }

  tickToTime(tick: number): number {
    return tick * this.secondsPerTick;
  }

  /** The timeline is indexed by sim tick, never by wall clock. */
  sampleAt(tick: number): FeaturesFrame {
    return this.sampleAtTime(tick * this.secondsPerTick, tick);
  }

  sampleAtTime(time: number, tick = time / this.secondsPerTick): FeaturesFrame {
    const { data, stride } = this.timeline;
    const { hopSeconds, frames } = this.timeline.manifest.grid;

    const pos = Math.min(Math.max(time / hopSeconds, 0), frames - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, frames - 1);
    const t = pos - i0;

    const a = i0 * stride;
    const b = i1 * stride;
    const row = this.row;
    if (t === 0 || i0 === i1) {
      row.set(data.subarray(a, a + stride));
    } else {
      for (let d = 0; d < stride; d++) {
        const va = data[a + d] as number;
        row[d] = va + ((data[b + d] as number) - va) * t;
      }
    }

    this.frame.tick = tick;
    this.frame.time = time;
    return this.frame;
  }

  getChannel(name: string): ChannelSpec | undefined {
    return this.timeline.channels.get(name);
  }

  hasChannel(name: string): boolean {
    return this.timeline.channels.has(name);
  }

  /** View into a sampled frame for one channel. Zero-copy; valid until the next sample. */
  channel(frame: FeaturesFrame, name: string): Float32Array {
    const c = this.timeline.channels.get(name);
    if (!c) throw new Error(`timeline: no channel "${name}"`);
    return frame.values.subarray(c.offset, c.offset + c.dims);
  }

  /** Single value from a channel (defaults to the first dim, for 1-dim channels). */
  value(frame: FeaturesFrame, name: string, dim = 0): number {
    const c = this.timeline.channels.get(name);
    if (!c || dim >= c.dims) return 0;
    return frame.values[c.offset + dim] as number;
  }

  /** Raw stored value at a grid frame index — for plotting whole-track curves. */
  rawAt(frameIndex: number, name: string, dim = 0): number {
    const c = this.timeline.channels.get(name);
    if (!c || dim >= c.dims) return 0;
    const i = Math.min(Math.max(frameIndex, 0), this.timeline.manifest.grid.frames - 1);
    return this.timeline.data[i * this.timeline.stride + c.offset + dim] as number;
  }

  segmentIndexAt(time: number): number {
    const segs = this.timeline.manifest.segments;
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i] as Segment;
      if (time >= s.start && time < s.end) return i;
    }
    return time >= this.duration ? segs.length - 1 : -1;
  }

  segmentAt(time: number): Segment | undefined {
    const i = this.segmentIndexAt(time);
    return i < 0 ? undefined : this.timeline.manifest.segments[i];
  }

  beatStateAt(time: number): BeatState {
    const { beats, downbeats } = this.timeline.manifest;
    const index = lowerIndex(beats, time);
    const downbeatIndex = lowerIndex(downbeats, time);
    if (index < 0) {
      return { index: -1, phase: 0, isDownbeat: false, downbeatIndex, sinceBeat: 0 };
    }
    const b0 = beats[index] as number;
    const b1 = (beats[index + 1] ?? b0 + 60 / this.timeline.manifest.tempo) as number;
    const span = Math.max(b1 - b0, 1e-6);
    const db = downbeats[downbeatIndex];
    return {
      index,
      phase: Math.min((time - b0) / span, 1),
      isDownbeat: db !== undefined && Math.abs(db - b0) < 1e-6,
      downbeatIndex,
      sinceBeat: time - b0,
    };
  }
}
