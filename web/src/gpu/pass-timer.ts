/**
 * GPU pass timing, via `timestamp-query` — the only clock that answers "what
 * does this pass cost" rather than "how long did the browser take to get back
 * to us". rAF deltas measure the frame; these measure the dispatches.
 *
 * ## Shape
 *
 * One `PassTimer` owns one query set and a small ring of readback buffers. Each
 * pass that wants timing asks `begin(label)` for a `timestampWrites` descriptor
 * (or `undefined` when the feature is missing or the frame's budget is spent —
 * both of which the caller treats identically: pass it through and move on).
 * Once per frame the owner calls `finishFrame()`, which resolves everything
 * recorded since the last call into a free ring slot and maps it asynchronously.
 * If every slot is still in flight the frame's numbers are dropped rather than
 * awaited: this is a meter, and a meter that stalls the pipeline it measures is
 * reporting on itself.
 *
 * Ordering is what makes one query set enough: the resolve is submitted after
 * the frame's passes and before the next frame's, so queue order guarantees it
 * reads this frame's values even though the indices are reused immediately.
 *
 * ## Reading the numbers
 *
 * Durations are summed per label per frame (a label used by several passes — a
 * substepped force pass, a three-pass grid rebuild — reports the frame's
 * total), then averaged over `WINDOW` recorded frames. Browsers quantize
 * timestamps (Chrome to ~100 µs) so individual small passes are noisy; the
 * averaging window is what makes the numbers worth reading.
 */

const MAX_PASSES_PER_FRAME = 64;
const RING = 4;
/**
 * Wall-clock milliseconds per published average. Time-based rather than
 * frame-counted on purpose: a throttled or GPU-saturated tab may complete only
 * a couple of readbacks a second, and a fixed frame count would leave the
 * meter reading "—" exactly when someone is staring at it wondering why.
 */
const PUBLISH_MS = 500;

interface RingSlot {
  resolve: GPUBuffer;
  read: GPUBuffer;
  busy: boolean;
}

export class PassTimer {
  private readonly device: GPUDevice;
  private readonly querySet: GPUQuerySet | null = null;
  private readonly ring: RingSlot[] = [];

  /** Labels of the current frame's passes, in query-pair order. */
  private labels: string[] = [];

  /** label → summed ns over the window, and how many frames the window holds. */
  private sums = new Map<string, number>();
  private frames = 0;
  private lastPublish = 0;

  /** label → average ms per frame, republished every WINDOW frames. */
  private report = new Map<string, number>();

  constructor(device: GPUDevice) {
    this.device = device;
    if (!device.features.has('timestamp-query')) return;
    this.querySet = device.createQuerySet({
      label: 'passTimer.queries',
      type: 'timestamp',
      count: 2 * MAX_PASSES_PER_FRAME,
    });
    for (let i = 0; i < RING; i++) {
      this.ring.push({
        resolve: device.createBuffer({
          label: `passTimer.resolve${i}`,
          size: 8 * 2 * MAX_PASSES_PER_FRAME,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        read: device.createBuffer({
          label: `passTimer.read${i}`,
          size: 8 * 2 * MAX_PASSES_PER_FRAME,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        busy: false,
      });
    }
  }

  /** False when the adapter has no `timestamp-query`; every call degrades to a no-op. */
  get enabled(): boolean {
    return this.querySet !== null;
  }

  /**
   * Claim a query pair for one pass. The return value plugs straight into a
   * compute or render pass descriptor's `timestampWrites` (the two interfaces
   * are structurally identical); `undefined` means "not this frame" and is a
   * legal descriptor value, so callers never branch.
   */
  begin(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.querySet || this.labels.length >= MAX_PASSES_PER_FRAME) return undefined;
    const n = this.labels.length;
    this.labels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: 2 * n,
      endOfPassWriteIndex: 2 * n + 1,
    };
  }

  /**
   * `begin`, but folded into a pass descriptor — the shape call sites want
   * under `exactOptionalPropertyTypes`, where `timestampWrites: undefined` is
   * not a legal way to spell "no timing". One method per pass flavour rather
   * than a generic: the two `timestampWrites` interfaces are structurally
   * identical, but a generic constrained on one of them defeats TS's literal
   * inference at every call site.
   */
  timed(desc: GPUComputePassDescriptor, label: string): GPUComputePassDescriptor {
    const tw = this.begin(label);
    return tw ? { ...desc, timestampWrites: tw } : desc;
  }

  /** `timed`, for render passes (the far lane's splat and blurs). */
  timedRender(desc: GPURenderPassDescriptor, label: string): GPURenderPassDescriptor {
    const tw = this.begin(label);
    return tw ? { ...desc, timestampWrites: tw } : desc;
  }

  /**
   * Resolve and read back everything recorded since the last call. Its own
   * encoder and submit, so the owner can call it after any number of
   * already-submitted encoders and queue order does the sequencing.
   */
  finishFrame(): void {
    const qs = this.querySet;
    const used = this.labels.length;
    if (!qs || used === 0) return;
    const frameLabels = this.labels;
    this.labels = [];

    const slot = this.ring.find((s) => !s.busy);
    if (!slot) return; // all readbacks in flight — drop the frame, never wait

    slot.busy = true;
    const encoder = this.device.createCommandEncoder({ label: 'passTimer.resolve' });
    encoder.resolveQuerySet(qs, 0, 2 * used, slot.resolve, 0);
    encoder.copyBufferToBuffer(slot.resolve, 0, slot.read, 0, 8 * 2 * used);
    this.device.queue.submit([encoder.finish()]);

    slot.read
      .mapAsync(GPUMapMode.READ, 0, 8 * 2 * used)
      .then(() => {
        const ts = new BigInt64Array(slot.read.getMappedRange(0, 8 * 2 * used));
        for (let i = 0; i < used; i++) {
          const begin = ts[2 * i] as bigint;
          const end = ts[2 * i + 1] as bigint;
          // A pair can legitimately come back 0/0 (device discarded the write);
          // max() also shrugs off any quantization-order oddity.
          const ns = Math.max(Number(end - begin), 0);
          const label = frameLabels[i] as string;
          this.sums.set(label, (this.sums.get(label) ?? 0) + ns);
        }
        slot.read.unmap();
        slot.busy = false;

        this.frames += 1;
        const now = performance.now();
        if (now - this.lastPublish >= PUBLISH_MS) {
          this.report = new Map(
            [...this.sums].map(([label, ns]) => [label, ns / this.frames / 1e6]),
          );
          this.sums = new Map();
          this.frames = 0;
          this.lastPublish = now;
        }
      })
      .catch((err: unknown) => {
        // Device loss mid-map, or a bug in the read path. Logged (once per
        // slot-cycle at worst) rather than swallowed: a meter that dies
        // silently shows '—' forever and reads as "still warming up".
        console.warn('passTimer: readback failed —', err);
        slot.busy = false;
      });
  }

  /**
   * Average GPU milliseconds per recorded frame, by label, over the last
   * published window. Empty until the first readback lands (~a second).
   */
  averages(): ReadonlyMap<string, number> {
    return this.report;
  }

  /**
   * `performance.now()` of the last publish, 0 before the first. Consumers
   * making decisions on `averages()` (the budget governor) should treat a stale
   * publish as no data — a paused sim stops recording but keeps its last map.
   */
  get publishedAt(): number {
    return this.lastPublish;
  }
}
