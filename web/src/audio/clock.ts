export interface ClickTrackSource {
  duration: number;
  beats: readonly number[];
  downbeats: readonly number[];
}

export interface AudioClockOptions {
  /** fixed simulation timestep in seconds */
  secondsPerTick?: number;
  /** cap on ticks run in one rAF, so a stalled tab does not spiral */
  maxTicksPerFrame?: number;
  /**
   * Optional track audio, by convention `<timelineBaseUrl>/audio.wav`. If it is
   * missing or undecodable the click track built from the beat grid is used
   * instead, so a timeline without a rendered audio file still plays.
   */
  audioUrl?: string;
  /**
   * How to fetch that audio. Defaults to `fetch`; a server track passes the
   * Cache-API read-through so its 40 MB of WAV is stored on first play and
   * answered locally afterwards, including with the server gone.
   */
  fetcher?: typeof fetch;
}

/** Which buffer the transport ended up playing. */
export type AudioSourceKind = 'track' | 'click';

const DEFAULT_TICK = 1 / 60;

/**
 * AudioContext.currentTime is the master clock. Sim ticks are derived from it by a
 * fixed-timestep accumulator; rAF only decides when to drain the accumulator.
 */
export class AudioClock {
  readonly secondsPerTick: number;
  readonly duration: number;
  private readonly maxTicksPerFrame: number;
  private readonly source: ClickTrackSource;
  private readonly audioUrl: string | null;
  private readonly fetcher: typeof fetch;

  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private node: AudioBufferSourceNode | null = null;
  private encoded: Promise<ArrayBuffer | null> | null = null;
  /** Memoised: decodeAudioData detaches its input, so it must run exactly once. */
  private buffer: Promise<AudioBuffer> | null = null;
  private kind: AudioSourceKind = 'click';
  /** Bumped by every play() so a superseded start (pause/seek mid-await) bails out. */
  private startId = 0;

  /** ctx.currentTime that corresponds to track time 0 (only meaningful while playing) */
  private originTime = 0;
  private pausedAt = 0;
  private playing = false;
  private tick = 0;

  constructor(source: ClickTrackSource, opts: AudioClockOptions = {}) {
    this.source = source;
    this.duration = source.duration;
    this.secondsPerTick = opts.secondsPerTick ?? DEFAULT_TICK;
    this.maxTicksPerFrame = opts.maxTicksPerFrame ?? 8;
    this.audioUrl = opts.audioUrl ?? null;
    // Bound, because it is stored on `this` and called as `this.fetcher(...)`:
    // native `fetch` throws "Illegal invocation" if its receiver is not `window`.
    this.fetcher = opts.fetcher ?? globalThis.fetch.bind(globalThis);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** 'click' until the track audio has been fetched and decoded. */
  get sourceKind(): AudioSourceKind {
    return this.kind;
  }

  get simTick(): number {
    return this.tick;
  }

  /** Track time in seconds, straight off the audio clock. */
  get time(): number {
    // `node` is only set once play() has finished starting a source; between the
    // playing flag being claimed and that point the origin is stale, so report paused.
    if (!this.playing || !this.ctx || !this.node) return this.pausedAt;
    return Math.min(Math.max(this.ctx.currentTime - this.originTime, 0), this.duration);
  }

  /**
   * ctx.currentTime is when a sample is handed to the graph, not when it is heard.
   * Without this the sim leads the audio by 20-50 ms on WASAPI and drops land early.
   */
  private outputDelay(ctx: AudioContext): number {
    const extra = (ctx as AudioContext & { outputLatency?: number }).outputLatency;
    return ctx.baseLatency + (Number.isFinite(extra) ? (extra as number) : 0);
  }

  /** Time the sim has actually caught up to; use this for rendering, not `time`. */
  get tickTime(): number {
    return this.tick * this.secondsPerTick;
  }

  private ensureContext(): AudioContext {
    if (this.ctx) return this.ctx;
    const ctx = new AudioContext();
    const gain = ctx.createGain();
    gain.gain.value = 0.5;
    gain.connect(ctx.destination);
    this.ctx = ctx;
    this.gain = gain;
    return ctx;
  }

  /**
   * Start fetching the track audio. Safe (and worth calling) before any user
   * gesture: it touches no AudioContext, so the first play does not stall on a
   * multi-megabyte download.
   */
  preload(): void {
    if (this.encoded || this.audioUrl === null) return;
    this.encoded = this.fetcher(this.audioUrl)
      .then((res) => {
        // The dev server answers a missing file with index.html at 200, so "not
        // an HTML page" is the real test for "the track has audio".
        const html = res.headers.get('content-type')?.startsWith('text/html') ?? false;
        return res.ok && !html ? res.arrayBuffer() : null;
      })
      .catch(() => null);
  }

  /** Track audio if it exists and decodes, click track otherwise. Resolved once. */
  private ensureBuffer(ctx: AudioContext): Promise<AudioBuffer> {
    this.buffer ??= this.decodeBuffer(ctx);
    return this.buffer;
  }

  private async decodeBuffer(ctx: AudioContext): Promise<AudioBuffer> {
    this.preload();
    const encoded = this.encoded === null ? null : await this.encoded;
    if (encoded) {
      try {
        const decoded = await ctx.decodeAudioData(encoded);
        this.kind = 'track';
        return decoded;
      } catch (err) {
        console.warn('audio: decode failed, falling back to click track', err);
      }
    }
    return buildClickTrack(ctx, this.source);
  }

  /** Must be called from a user gesture the first time (autoplay policy). */
  async play(): Promise<void> {
    // Claim the transport before the first await: two activations in the same frame
    // (button + Space) would otherwise both pass the guard and start two sources,
    // and only the second would be stoppable.
    if (this.playing) return;
    this.playing = true;
    const id = ++this.startId;
    const ctx = this.ensureContext();
    try {
      if (ctx.state !== 'running') await ctx.resume();
    } catch (err) {
      this.playing = false;
      throw err;
    }
    if (!this.playing || this.startId !== id) return;

    // Decoding a real track takes long enough for a pause or a seek to land
    // first. A seek restarts play() immediately, so "still playing" is not
    // enough — only the newest start may claim the node.
    const buffer = await this.ensureBuffer(ctx);
    if (!this.playing || this.startId !== id) return;

    if (this.pausedAt >= this.duration - 1e-3) this.pausedAt = 0;

    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(this.gain as GainNode);
    node.start(0, this.pausedAt);
    this.originTime = ctx.currentTime + this.outputDelay(ctx) - this.pausedAt;
    this.node = node;
    this.tick = Math.floor(this.pausedAt / this.secondsPerTick);
  }

  pause(): void {
    if (!this.playing) return;
    this.pausedAt = this.time;
    this.stopNode();
    this.playing = false;
  }

  async toggle(): Promise<void> {
    if (this.playing) this.pause();
    else await this.play();
  }

  seek(seconds: number): void {
    const t = Math.min(Math.max(seconds, 0), this.duration);
    const wasPlaying = this.playing;
    if (wasPlaying) {
      this.stopNode();
      this.playing = false;
    }
    this.pausedAt = t;
    this.tick = Math.floor(t / this.secondsPerTick);
    if (wasPlaying) void this.play();
  }

  private stopNode(): void {
    if (!this.node) return;
    try {
      this.node.stop();
    } catch {
      // already stopped
    }
    this.node.disconnect();
    this.node = null;
  }

  /**
   * Drain the accumulator. Calls `run` once per whole sim tick that the audio clock has
   * passed, in order. Returns the number of ticks run.
   */
  pump(run: (simTick: number) => void): number {
    if (!this.playing) return 0;

    const now = this.time;
    if (now >= this.duration) {
      this.pause();
      this.pausedAt = this.duration;
      return 0;
    }

    const target = Math.floor(now / this.secondsPerTick);
    let count = 0;
    while (this.tick < target && count < this.maxTicksPerFrame) {
      this.tick++;
      run(this.tick);
      count++;
    }
    // A long stall would otherwise leave the sim permanently behind the audio.
    if (this.tick < target - this.maxTicksPerFrame) this.tick = target;
    return count;
  }
}

/**
 * The synthetic track has no audio file, so the beat grid is rendered into an audible
 * click track: short decaying sines, downbeats a fifth up and louder.
 */
export function buildClickTrack(ctx: BaseAudioContext, src: ClickTrackSource): AudioBuffer {
  const sr = ctx.sampleRate;
  const frames = Math.ceil((src.duration + 0.25) * sr);
  const buffer = ctx.createBuffer(1, frames, sr);
  const out = buffer.getChannelData(0);

  const downbeatSet = new Set(src.downbeats.map((t) => Math.round(t * 1000)));
  const clickLen = Math.floor(0.06 * sr);

  for (const t of src.beats) {
    const accent = downbeatSet.has(Math.round(t * 1000));
    const freq = accent ? 1600 : 1000;
    const amp = accent ? 0.9 : 0.4;
    const decay = accent ? 45 : 70;
    const start = Math.floor(t * sr);
    for (let i = 0; i < clickLen; i++) {
      const j = start + i;
      if (j >= frames) break;
      const s = i / sr;
      out[j] = (out[j] as number) + Math.sin(2 * Math.PI * freq * s) * amp * Math.exp(-decay * s);
    }
  }
  return buffer;
}
