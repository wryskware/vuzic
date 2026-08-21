/**
 * Explorer mode's 9-up rig: nine live simulations on one canvas.
 *
 * The centre tile is the current θ; the eight around it are mutated candidates.
 * Clicking one promotes it and generates eight fresh mutations around it — hill
 * climbing with the eye as the objective function. This file owns the *rendering
 * and lifecycle* half of that; `search.ts` owns which nine vectors exist and
 * `log.ts` records what was picked.
 *
 * ## Why nine whole sims and not one sim drawn nine times
 *
 * These substrates are stateful and continuous — that is the point of the
 * project — so "θ at this instant" is not enough to draw a tile with. Two
 * candidates that differ only in decay look identical for the first second and
 * diverge over the next twenty; the tile has to *be* a running world, not a
 * snapshot evaluated at nine parameter settings. So each tile gets its own sim
 * instance, its own GPU state, and its own post chain.
 *
 * ## The virtual-context seam
 *
 * Nothing in `sim/**` changed to make this work, and nothing needed to. A `Sim`
 * already renders to whatever texture view it is handed (main.ts is the only
 * place that touches the swapchain), and it already reads `ctx.width/height`
 * afresh every frame to size its post surfaces. So a tile runs on a *virtual*
 * `GpuRuntimeContext`: the real device, format and adapter, with `width` and
 * `height` replaced by the tile's own pixel size. The
 * rig owns those two numbers; a canvas resize updates them and each sim's own
 * `post.ensureSize` does the reallocation on its next frame.
 *
 * One consequence worth stating: both substrates fix their world at `init` time
 * (physarum snaps a grid, plife snaps a world aspect) and never rescale it,
 * because rescaling would throw away everything they have accumulated. Tiles
 * inherit that: a tile's grid is chosen for the tile size at `open()`, and a
 * later window resize only changes how it is composited — exactly the behaviour
 * the full-screen sim already has.
 *
 * ## The layered target
 *
 * The nine tiles render into nine array layers of one texture rather than into
 * nine separate textures. Layers are what let the compositor be a single
 * `draw(4, 9)` with the per-tile state (rect, border, centre/hover marking) read
 * from one uniform array by `instance_index` — nine textures would mean nine
 * bind groups and nine draws for the same picture. Every tile is therefore
 * exactly the same size, and the remainder pixels become an outer margin rather
 * than a fatter last column.
 *
 * ## Ordering: reseed, then apply
 *
 * `setCandidates` reseeds a tile *before* writing its θ, never after. A reseed
 * is "new world" — it re-scatters agents/particles and, for a sim wired to a
 * Modulator, re-draws whatever it draws from the seed. Applying θ afterwards is
 * what guarantees the candidate's own numbers are the ones running. (Tiles are
 * deliberately Modulator-less and ImpulseEngine-less: raw parameters, unmediated,
 * is the whole point of the mode.)
 *
 * A fresh scatter per generation is intended, not a compromise. Initial
 * conditions do not carry meaning in either substrate, and cloning nine worth of
 * GPU state per pick would cost far more than it bought.
 *
 * ## θ is the grid's; everything else is the live sim's (`syncStyle`)
 *
 * A tile differs from the live world in exactly one intended way: its θ. Every
 * *other* thing that decides what a frame looks like — the palette, the HDR
 * chain, the macro rig — has to keep tracking the live config while the mode is
 * open, because the workbench panel is live during exploration and an art
 * direction edit that only landed on the invisible full-screen sim would look
 * like the panel was broken. That is `syncStyle`: main pushes the live sim in on
 * every generation and on the panel's own refresh cadence, and the rig fans it
 * out to nine tiles.
 *
 * Two properties make that safe to run twice a second rather than only on change:
 *
 *   - it copies **values**, never object references. A tile that shared the live
 *     palette array would be un-tunable — every tile would be the live world's
 *     colours by identity, which is fine, right up until something wants them to
 *     differ.
 *   - `applyExtras` is idempotent in both substrates: it clamps numbers into the
 *     live config and does no GPU work and no redraw (plife restores its
 *     `matrixGen` *settings* and explicitly does not re-draw the matrix from
 *     them). So a style sync cannot rescatter a tile, and — the ordering question
 *     `setCandidates` would otherwise raise — it cannot disturb the candidate θ
 *     that was applied before it either, because nothing it writes is in θ.
 */
import type { BrowserGpuContext } from '../gpu/browser-context.ts';
import type { GpuRuntimeContext } from '../gpu/runtime-context.ts';
import type { ModTarget } from '../mapping/target.ts';
import { hash3 } from '../sim/impulses.ts';
import type { Palette } from '../sim/palette.ts';
import { mergeRenderConfig } from '../sim/render/config.ts';
import type { RenderFrame, Sim } from '../sim/types.ts';
import type { FeaturesFrame } from '../timeline/sampler.ts';
import { CANDIDATE_COUNT, type CandidateSet } from './search.ts';

import compositeWgsl from './shaders/composite.wgsl?raw';

/** 3×3. Fixed: the shader's uniform array is sized to it. */
export const TILE_COUNT = 9;

/** Grid position of the current θ. Row-major, so this is the middle of the 3×3. */
export const CENTER_TILE = 4;

/**
 * Candidate index → grid position, row-major:
 *
 *     0 1 2
 *     3 [4] 5
 *     6 7 8
 *
 * i.e. candidates fill the ring in reading order and skip the centre. Stated as
 * a table rather than computed so that "which tile was candidate 5" has exactly
 * one answer, in one place — the log records the index, and a training loader
 * reading that log has to be able to reconstruct the picture.
 */
const GRID_OF_CANDIDATE: readonly number[] = [0, 1, 2, 3, 5, 6, 7, 8];

/** f32 slots per Tile in the uniform: rect(4) + border(4) + mark(4). */
const TILE_WORDS = 12;
/** f32 slots before the tile array: tileSize(2) + pad(2). */
const HEADER_WORDS = 4;

/** Gap between tiles, and the outer margin, in CSS px. */
const DEFAULT_GUTTER_CSS = 10;

/**
 * Border presentation, in CSS px and in the swapchain's own (already display
 * encoded) colour space — the tile textures hold graded output, so a literal
 * here is the colour you get.
 */
const BORDER_IDLE: readonly [number, number, number] = [0.18, 0.2, 0.28];
const BORDER_CENTER: readonly [number, number, number] = [0.25, 0.82, 1.0];
const BORDER_HOVER: readonly [number, number, number] = [0.95, 0.96, 1.0];
const WIDTH_IDLE_CSS = 1;
const WIDTH_ACTIVE_CSS = 2;
const BRACKET_LEN_CSS = 20;
const BRACKET_EXTRA_CSS = 2;

/** The gutter colour. Matches `--bg` in style.css so the chrome stays one piece. */
const BACKDROP: GPUColor = { r: 0.024, g: 0.027, b: 0.043, a: 1 };

/** A tile's on-screen rectangle in device px. */
interface TileRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface TileEntry {
  sim: Sim & ModTarget;
  /** the virtual context this tile's sim reads its size from; mutated on layout */
  ctx: GpuRuntimeContext;
}

export interface ExplorerRigOptions {
  gpu: BrowserGpuContext;
  /**
   * Builds one tile's simulation — same substrate as the live sim, same channel
   * wiring, not yet `init`ed. A factory rather than a class reference because
   * only main.ts knows which substrate is running and what its channels are, and
   * the rig deliberately does not learn either.
   */
  createTile: () => Sim & ModTarget;
  /** gap between tiles and outer margin, CSS px */
  gutterCss?: number;
}

export class ExplorerRig {
  private readonly gpu: BrowserGpuContext;
  private readonly createTile: () => Sim & ModTarget;
  private readonly gutterCss: number;

  private tiles: TileEntry[] = [];
  private opened = false;

  /** device px per CSS px, refreshed with the geometry; drives border widths */
  private pxScale = 1;
  private width = 0;
  private height = 0;
  private tileW = 1;
  private tileH = 1;
  private rects: TileRect[] = [];
  private hovered: number | null = null;

  /** one 2d-array texture, nine layers; see the header note */
  private target: GPUTexture | null = null;
  private layerViews: GPUTextureView[] = [];
  private bindGroup: GPUBindGroup | null = null;

  // Device-lifetime, built on first open and kept across close/open: a shader
  // module and a 448-byte uniform are not worth rebuilding every time the mode
  // is entered, and nothing in them depends on the tile size.
  private pipeline: GPURenderPipeline | null = null;
  private bindLayout: GPUBindGroupLayout | null = null;
  private sampler: GPUSampler | null = null;
  private uniformBuf: GPUBuffer | null = null;
  private readonly uniformData = new Float32Array(HEADER_WORDS + TILE_COUNT * TILE_WORDS);

  constructor(opts: ExplorerRigOptions) {
    this.gpu = opts.gpu;
    this.createTile = opts.createTile;
    this.gutterCss = Math.max(0, opts.gutterCss ?? DEFAULT_GUTTER_CSS);
  }

  get isOpen(): boolean {
    return this.opened;
  }

  /**
   * The centre tile's simulation — the incumbent, the one with the brackets
   * round it — or null when the mode is closed.
   *
   * Exposed for exit-time state promotion: the centre tile is the world the
   * human has been choosing, and main.ts copies it onto the live sim before
   * `close()` disposes it. Typed as the rig's own `Sim & ModTarget` and nothing
   * narrower, because the rig deliberately does not know which substrate it is
   * running — deciding whether a given substrate can be promoted at all, and how,
   * belongs to the caller that built the tiles.
   */
  get centerSim(): (Sim & ModTarget) | null {
    return this.opened ? this.tiles[CENTER_TILE]?.sim ?? null : null;
  }

  /** Candidate index → grid position. Inverse of `candidateAt`. */
  static gridOfCandidate(index: number): number {
    return GRID_OF_CANDIDATE[index] ?? CENTER_TILE;
  }

  /** Grid position → candidate index, or null for the centre tile. */
  static candidateAt(grid: number): number | null {
    if (grid === CENTER_TILE || grid < 0 || grid >= TILE_COUNT) return null;
    return grid < CENTER_TILE ? grid : grid - 1;
  }

  /**
   * Build the nine sims and the layered target. Async because `Sim.init` is —
   * pipeline creation is the slow part and the nine are started together rather
   * than in series.
   *
   * Failure is unwound rather than left half-built: a rig with four initialised
   * tiles would render garbage and leak the other five.
   */
  async open(): Promise<void> {
    if (this.opened) return;
    this.ensureDeviceResources();
    this.computeGeometry(this.gpu.width, this.gpu.height);
    this.allocTarget();

    try {
      for (let i = 0; i < TILE_COUNT; i++) {
        this.tiles.push({ sim: this.createTile(), ctx: this.makeTileContext() });
      }
      await Promise.all(this.tiles.map((t) => t.sim.init(t.ctx)));
    } catch (err) {
      for (const t of this.tiles) t.sim.dispose();
      this.tiles = [];
      this.releaseTarget();
      throw err;
    }

    this.opened = true;
    this.writeUniform();
  }

  /** Dispose the nine sims (each frees its own GPU state) and the shared target. */
  close(): void {
    if (!this.opened) return;
    this.opened = false;
    for (const t of this.tiles) t.sim.dispose();
    this.tiles = [];
    this.releaseTarget();
    this.hovered = null;
  }

  /**
   * Re-lay the grid for a new canvas size, in device px. Idempotent, so the
   * frame loop can call it unconditionally — which is where it is called from,
   * because that covers a DPR change and a CSS-driven resize as well as a window
   * `resize` event.
   */
  layout(width: number, height: number): void {
    if (!this.opened) return;
    if (width === this.width && height === this.height) return;
    const prevW = this.tileW;
    const prevH = this.tileH;
    this.computeGeometry(width, height);
    if (this.tileW !== prevW || this.tileH !== prevH) {
      this.allocTarget();
      // The sims pick this up themselves: every one of them calls
      // `post.ensureSize(ctx.width, ctx.height)` at the top of its own `render`,
      // so the reallocation happens on their next frame with no rig involvement.
      for (const t of this.tiles) {
        t.ctx.width = this.tileW;
        t.ctx.height = this.tileH;
      }
    }
    this.writeUniform();
  }

  /**
   * Advance every tile by one step of `dt`, exactly like the live sim — same
   * world time, same cadence, so what the grid shows is what adopting the θ
   * would give you. `frame.values` is shared and read-only here, as everywhere.
   */
  tick(frame: FeaturesFrame, stepIndex: number, dt: number): void {
    if (!this.opened) return;
    for (const t of this.tiles) t.sim.tick(frame, stepIndex, dt);
  }

  /** Nine sim renders into nine layers, then one composite pass onto the canvas. */
  render(
    encoder: GPUCommandEncoder,
    canvasView: GPUTextureView,
    frame: RenderFrame,
  ): void {
    if (!this.opened || !this.pipeline || !this.bindGroup) return;

    for (let i = 0; i < TILE_COUNT; i++) {
      const tile = this.tiles[i];
      const view = this.layerViews[i];
      if (!tile || !view) continue;
      tile.sim.render(encoder, view, frame);
    }

    const pass = encoder.beginRenderPass({
      label: 'explorer.composite',
      colorAttachments: [
        { view: canvasView, clearValue: BACKDROP, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(4, TILE_COUNT);
    pass.end();
  }

  /**
   * Adopt a generation: the centre tile takes `set.center`, the eight others take
   * the candidates in `GRID_OF_CANDIDATE` order, and every tile restarts from a
   * seed derived from `(genSeed, grid)` so the same generation is the same nine
   * worlds on a replay.
   */
  setCandidates(set: CandidateSet): void {
    if (!this.opened) return;
    this.applyToTile(CENTER_TILE, set.center, set.genSeed);
    for (let i = 0; i < CANDIDATE_COUNT; i++) {
      const theta = set.candidates[i];
      if (!theta) continue;
      this.applyToTile(ExplorerRig.gridOfCandidate(i), theta, set.genSeed);
    }
  }

  /**
   * Copy everything-that-is-not-θ from the live sim onto all nine tiles: the
   * macro rig and any other opaque per-sim extras, the palette, and the render
   * chain. See the header for why this exists and why it is safe to call on a
   * timer.
   *
   * Ordering against `setCandidates` is "candidates first, style second", and it
   * is a free choice rather than a forced one: nothing `applyExtras` writes lives
   * in θ (physarum's macros and plife's macros + `matrixGen` are all outside the
   * registry — that is what the extras channel *is*), so neither order can clobber
   * the other. Style-last is chosen because it is the order that stays correct if
   * a future substrate ever does derive GPU state from an extra.
   *
   * `from` is the live `ModTarget` rather than a config snapshot so that the
   * opaque extras block never has to be understood here — the rig moves it from
   * one sim to nine without learning its shape, exactly as the mapping layer does.
   */
  syncStyle(from: ModTarget): void {
    if (!this.opened) return;
    // Serialised once and applied nine times: it is a fresh plain object per call
    // and every `applyExtras` copies field by field out of it, so there is no
    // aliasing risk in sharing one across the nine.
    const extras = from.serializeExtras?.();
    for (const t of this.tiles) {
      t.sim.applyExtras?.(extras);
      copyPalette(from.config.palette, t.sim.config.palette);
      // Value copy: `mergeRenderConfig` is three `Object.assign`s over three flat
      // blocks of primitives, so the tile keeps its own grade/bloom/feedback
      // objects and the panel's widgets stay bound to the live ones.
      mergeRenderConfig(t.sim.config.render, from.config.render);
      // Both substrates cache the linearised palette; nothing else here needs a
      // push, since a tile re-reads its config every step exactly as the live sim does.
      t.sim.invalidatePalette();
    }
  }

  /** Highlight a tile under the pointer, or none. Cheap enough to call per move. */
  setHover(grid: number | null): void {
    const next = grid !== null && grid >= 0 && grid < TILE_COUNT ? grid : null;
    if (next === this.hovered) return;
    this.hovered = next;
    if (this.opened) this.writeUniform();
  }

  /**
   * Which tile is at a point, given in **CSS px relative to the canvas** (i.e.
   * a pointer event's `offsetX`/`offsetY` on the stage). Null in the gutter or
   * the outer margin, which is what makes a miss a miss rather than a wrong pick.
   */
  tileAt(cssX: number, cssY: number): number | null {
    if (!this.opened) return null;
    const x = cssX * this.pxScale;
    const y = cssY * this.pxScale;
    for (let i = 0; i < this.rects.length; i++) {
      const r = this.rects[i];
      if (!r) continue;
      if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return i;
    }
    return null;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private applyToTile(grid: number, theta: Float64Array, genSeed: number): void {
    const tile = this.tiles[grid];
    if (!tile) return;
    // Reseed FIRST — see the header note. A reseed is a new world; θ is what that
    // world is made of, and it has to be written after the world exists.
    tile.sim.reseed(tileSeed(genSeed, grid));
    // No mask: a candidate is a complete θ, including the slots the modulator is
    // not allowed to move, and all of it is what is being judged.
    tile.sim.applyTheta(theta);
  }

  /**
   * A tile's canvas-free GPU context: the real device and capabilities with the
   * tile's own dimensions. The browser surface remains owned by the rig.
   */
  private makeTileContext(): GpuRuntimeContext {
    const tile: GpuRuntimeContext = {
      adapter: this.gpu.adapter,
      device: this.gpu.device,
      format: this.gpu.format,
      float32Filterable: this.gpu.float32Filterable,
      // Tiles grade into their own float texture, so they can carry the same
      // headroom the live view does and the composite passes it straight
      // through to the swapchain. A candidate should be judged as it will look.
      displayHeadroom: this.gpu.displayHeadroom ?? 1,
      width: this.tileW,
      height: this.tileH,
    };
    return tile;
  }

  private ensureDeviceResources(): void {
    if (this.pipeline) return;
    const { device } = this.gpu;

    this.sampler = device.createSampler({
      label: 'explorer.sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.uniformBuf = device.createBuffer({
      label: 'explorer.tiles',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const F = GPUShaderStage.FRAGMENT;
    this.bindLayout = device.createBindGroupLayout({
      label: 'explorer.compositeLayout',
      entries: [
        // The vertex stage reads the rects out of the same uniform, so it is
        // visible to both stages rather than to the fragment stage alone.
        { binding: 0, visibility: GPUShaderStage.VERTEX | F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ],
    });

    const module = device.createShaderModule({
      label: 'explorer.composite',
      code: compositeWgsl,
    });
    this.pipeline = device.createRenderPipeline({
      label: 'explorer.composite',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] }),
      vertex: { module, entryPoint: 'vsMain' },
      fragment: { module, entryPoint: 'fsMain', targets: [{ format: this.gpu.format }] },
      primitive: { topology: 'triangle-strip' },
    });
  }

  /**
   * Tile size and the nine rectangles, in device px. Every tile is the same size
   * (the array texture requires it, and an uneven grid reads as a bug anyway);
   * the pixels left over by the division become extra outer margin, split evenly.
   */
  private computeGeometry(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    const canvas = this.gpu.canvas;
    this.pxScale = canvas.clientWidth > 0 ? canvas.width / canvas.clientWidth : 1;

    const gap = Math.round(this.gutterCss * this.pxScale);
    // Two inner gaps plus a margin on each side = four gaps of horizontal budget.
    this.tileW = Math.max(1, Math.floor((this.width - 4 * gap) / 3));
    this.tileH = Math.max(1, Math.floor((this.height - 4 * gap) / 3));

    const spanW = 3 * this.tileW + 2 * gap;
    const spanH = 3 * this.tileH + 2 * gap;
    const originX = Math.max(0, Math.floor((this.width - spanW) / 2));
    const originY = Math.max(0, Math.floor((this.height - spanH) / 2));

    this.rects = [];
    for (let i = 0; i < TILE_COUNT; i++) {
      this.rects.push({
        x: originX + (i % 3) * (this.tileW + gap),
        y: originY + Math.floor(i / 3) * (this.tileH + gap),
        w: this.tileW,
        h: this.tileH,
      });
    }
  }

  private allocTarget(): void {
    this.releaseTarget();
    const { device } = this.gpu;
    const texture = device.createTexture({
      label: 'explorer.tiles',
      size: { width: this.tileW, height: this.tileH, depthOrArrayLayers: TILE_COUNT },
      format: this.gpu.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.target = texture;
    // A single-layer 2d view per tile is what a render pass can attach to; the
    // 2d-array view is what the compositor samples.
    this.layerViews = Array.from({ length: TILE_COUNT }, (_, i) =>
      texture.createView({
        label: `explorer.tile${i}`,
        dimension: '2d',
        baseArrayLayer: i,
        arrayLayerCount: 1,
      }),
    );
    this.bindGroup = device.createBindGroup({
      label: 'explorer.composite',
      layout: this.bindLayout as GPUBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf as GPUBuffer } },
        { binding: 1, resource: this.sampler as GPUSampler },
        { binding: 2, resource: texture.createView({ dimension: '2d-array' }) },
      ],
    });
  }

  private releaseTarget(): void {
    this.target?.destroy();
    this.target = null;
    this.layerViews = [];
    this.bindGroup = null;
  }

  private writeUniform(): void {
    const d = this.uniformData;
    d[0] = this.tileW;
    d[1] = this.tileH;
    d[2] = 0;
    d[3] = 0;
    for (let i = 0; i < TILE_COUNT; i++) {
      const r = this.rects[i];
      const o = HEADER_WORDS + i * TILE_WORDS;
      if (!r) continue;
      // Pixel rect → clip space. y is flipped: framebuffer y grows downward,
      // clip-space y grows upward, so the rect's top edge is the larger value.
      d[o + 0] = (r.x / this.width) * 2 - 1;
      d[o + 1] = 1 - (r.y / this.height) * 2;
      d[o + 2] = ((r.x + r.w) / this.width) * 2 - 1;
      d[o + 3] = 1 - ((r.y + r.h) / this.height) * 2;

      const isCenter = i === CENTER_TILE;
      const isHover = i === this.hovered;
      const rgb = isHover ? BORDER_HOVER : isCenter ? BORDER_CENTER : BORDER_IDLE;
      d[o + 4] = rgb[0];
      d[o + 5] = rgb[1];
      d[o + 6] = rgb[2];
      d[o + 7] = (isHover || isCenter ? WIDTH_ACTIVE_CSS : WIDTH_IDLE_CSS) * this.pxScale;
      // Corner brackets mark the centre and only the centre, so "which one is
      // the incumbent" survives the pointer moving over it and recolouring it.
      d[o + 8] = isCenter ? BRACKET_LEN_CSS * this.pxScale : 0;
      d[o + 9] = isCenter ? BRACKET_EXTRA_CSS * this.pxScale : 0;
      d[o + 10] = 0;
      d[o + 11] = 0;
    }
    this.gpu.device.queue.writeBuffer(this.uniformBuf as GPUBuffer, 0, d);
  }
}

/**
 * Palette → palette, by value.
 *
 * `colors` is copied element-wise into the destination's own array rather than
 * assigned: the tile's palette object is what its config holds and what its
 * `refreshPalette` reads, and handing it the live array would mean nine tiles and
 * the live sim sharing one set of strings by identity.
 */
function copyPalette(src: Palette, dst: Palette): void {
  dst.colors.length = src.colors.length;
  for (let i = 0; i < src.colors.length; i++) dst.colors[i] = src.colors[i] as string;
  dst.saturation = src.saturation;
  dst.brightness = src.brightness;
}

/**
 * A tile's world seed. Derived from the generation's own RNG key so that the
 * nine worlds of a generation are reproducible from what the log already
 * records — `genSeed` alone reconstructs all nine θs *and* all nine seeds.
 *
 * `hash3` is the sim layer's own PCG mixer (it mirrors common.wgsl), reused here
 * rather than reinvented so "seeded" means the same thing everywhere.
 */
function tileSeed(genSeed: number, grid: number): number {
  return hash3(genSeed >>> 0, grid + 1, 0x9e3779b9);
}
