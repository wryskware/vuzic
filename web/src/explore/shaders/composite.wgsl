// The 9-up compositor: one instanced quad per tile, sampling that tile's layer
// of the shared array texture and stamping a border on top.
//
// One draw call for the whole grid is why the tiles live in a *layered* texture
// rather than in nine separate ones: with nine textures each tile would need its
// own bind group and its own draw, and the per-tile state (which one is the
// centre, which one the pointer is over) would have to be a uniform rebind per
// tile. Here it is `instance_index` into one uniform array, and the pass is a
// single `draw(4, 9)`.

struct Tile {
  /** clip-space rect: x0, y0 (top), x1, y1 (bottom) */
  rect: vec4f,
  /** border colour rgb + width in device px */
  border: vec4f,
  /** corner bracket: length in device px (0 = none), extra width in device px */
  mark: vec4f,
}

struct Params {
  /** one tile's size in device px — the border is measured in pixels, not uv */
  tileSize: vec2f,
  pad: vec2f,
  tiles: array<Tile, 9>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tiles: texture_2d_array<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) layer: u32,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  // triangle-strip quad, in the order (0,0) (1,0) (0,1) (1,1). v = 0 is the
  // tile's TOP edge, which is what makes `uv` usable as a texture coordinate
  // directly: the sim rendered into this layer with the same convention.
  let u = f32(vi & 1u);
  let v = f32(vi >> 1u);
  let r = P.tiles[ii].rect;
  var out: VSOut;
  out.pos = vec4f(mix(r.x, r.z, u), mix(r.y, r.w, v), 0.0, 1.0);
  out.uv = vec2f(u, v);
  out.layer = ii;
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let t = P.tiles[in.layer];
  // The quad covers exactly `tileSize` device pixels and the layer is exactly
  // that size, so this is a 1:1 blit and the linear filter is an identity.
  let col = textureSample(tiles, samp, in.uv, i32(in.layer)).rgb;

  // Distance to the nearest edge, in device px, per axis. Working in pixels
  // rather than in uv is what keeps the border one pixel wide on every tile
  // whatever the aspect ratio.
  let p = in.uv * P.tileSize;
  let dx = min(p.x, P.tileSize.x - p.x);
  let dy = min(p.y, P.tileSize.y - p.y);

  let w = t.border.w;
  let onEdge = min(dx, dy) < w;
  // L-shaped corner brackets, thicker than the plain edge. This is the centre
  // tile's mark: a colour difference alone is not enough when the tile under it
  // is itself brightly coloured, and the brackets read at a glance.
  let bw = w + t.mark.y;
  let len = t.mark.x;
  let onBracket = len > 0.0 && ((dx < bw && dy < len) || (dy < bw && dx < len));

  return vec4f(select(col, t.border.rgb, onEdge || onBracket), 1.0);
}
