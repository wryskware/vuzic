import { SECONDS_PER_TICK } from '../timing.ts';

export interface ClickTrackSource {
  duration: number;
  beats: readonly number[];
  downbeats: readonly number[];
}

export interface AudioClockOptions {
  /** seconds per timeline sample — the rate the analysis was written at */
  secondsPerTick?: number;
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

/**
 * AudioContext.currentTime is the master clock for *where in the track we are*.
 *
 * It is no longer a simulation clock. The world advances once per rendered frame
 * on measured wall time (see `main.ts`), and this class only answers "which
 * timeline sample does the audio position correspond to right now" — a lookup,
 * not an accumulator. There is nothing to drain and nothing to catch up: the
 * position is read fresh from `ctx.currentTime` every frame, so a long frame
 * lands on a later sample rather than owing a backlog of skipped ones.
 */
export class AudioClock {
  readonly secondsPerTick: number;
  readonly duration: number;
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

  constructor(source: ClickTrackSource, opts: AudioClockOptions = {}) {
    this.source = source;
    this.duration = source.duration;
    // The default is the timeline's own sample rate rather than a second local
    // literal, so a caller that omits the option cannot silently index the
    // analysis at a rate it was not written at.
    this.secondsPerTick = opts.secondsPerTick ?? SECONDS_PER_TICK;
    this.audioUrl = opts.audioUrl ?? null;
    // Bound unconditionally — including a caller-supplied fetcher — because it
    // is stored on `this` and called as `this.fetcher(...)`: native `fetch`
    // with a non-Window receiver fails its brand check, and as a REJECTED
    // PROMISE rather than a throw, so an unbound copy slips straight through
    // `preload()`'s catch into the silent click-track fallback. That exact bug
    // shipped once (an unbound `fetch` out of `fetcherFor`); binding here makes
    // the seam immune to the next caller who forgets.
    this.fetcher = (opts.fetcher ?? globalThis.fetch).bind(globalThis);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** 'click' until the track audio has been fetched and decoded. */
  get sourceKind(): AudioSourceKind {
    return this.kind;
  }

  /**
   * The timeline sample the transport is on, derived from the audio position
   * rather than counted. Frozen with the transport, so a paused workbench keeps
   * reading the same features frame and no timeline event re-fires.
   */
  get simTick(): number {
    return Math.floor(this.time / this.secondsPerTick);
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

  /** Track time quantised to the timeline grid — what the scrub strip reads. */
  get tickTime(): number {
    return this.simTick * this.secondsPerTick;
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
   * The timeline sample this rendered frame should read, plus the end-of-track
   * stop. Called exactly once per frame.
   *
   * This is what replaced `pump()`. The old contract handed the caller a *run*
   * of ticks to simulate so the sim could stay glued to the audio grid; there is
   * no grid to stay glued to any more, so a frame simply asks where the audio is
   * and samples there. Skipping timeline samples between two frames is not a
   * backlog — it is what playing a 120 Hz analysis on a 60 Hz display means, and
   * it was always what the sampler's interpolation was for.
   */
  sampleTick(): number {
    if (this.playing && this.time >= this.duration) {
      this.pause();
      this.pausedAt = this.duration;
    }
    return this.simTick;
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
