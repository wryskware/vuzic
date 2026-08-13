struct ProbeParams {
  phase: f32,
  width: f32,
  height: f32,
  pad: f32,
}

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> params: ProbeParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VsOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = corners[vertexIndex];
  var out: VsOut;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let aspect = params.width / max(params.height, 1.0);
  let p = (in.uv - vec2f(0.5)) * vec2f(aspect, 1.0);
  let centre = vec2f(0.42 * sin(params.phase * 6.2831853), 0.28 * cos(params.phase * 6.2831853));
  let highlight = 5.0 * exp(-36.0 * dot(p - centre, p - centre));
  let base = vec3f(0.08 + in.uv.x * 0.7, 0.04 + in.uv.y * 0.5, 0.12);
  // Values intentionally exceed 1.0: the Gate 0 target must prove that the
  // offscreen path preserves scene-linear HDR rather than silently clamping.
  return vec4f(base + highlight * vec3f(1.0, 0.42, 0.16), 1.0);
}
