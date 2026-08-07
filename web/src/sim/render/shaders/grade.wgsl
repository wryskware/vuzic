// The only pass in the app that produces display-referred pixels. Everything
// upstream of here is linear and unbounded.

@group(0) @binding(0) var<uniform> p: Post;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var hdrTex: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;

@fragment
fn fsMain(in: PostVSOut) -> @location(0) vec4f {
  var c = textureSampleLevel(hdrTex, samp, in.uv, 0.0).rgb;

  // Bloom joins *before* the tone map, on purpose: that is what turns an impulse
  // flash into a splash of light instead of a plate of clipped white. The tone
  // map then compresses the combined value, so a 10x flash lands as a bright
  // glowing core with a halo rather than as saturated 1.0.
  c = c + textureSampleLevel(bloomTex, samp, in.uv, 0.0).rgb * p.bloomIntensity;

  // Exposure is not applied here — the compositor already did it, so that
  // auto-exposure measures the fully-exposed frame. `p.exposureEv` reaches this
  // module only as documentation of what that multiply was.
  c = tonemapSelect(c, p.tonemap);

  // Deep blacks, deliberately here rather than in the tone curve.
  let black = clamp(p.blackPoint, 0.0, 0.9);
  c = max(c - vec3f(black), vec3f(0.0)) / (1.0 - black);

  // Contrast about a low pivot: the image is mostly dark, so pivoting at 0.5
  // would just dim everything. A low pivot keeps the filaments and drops the
  // haze between them.
  c = max((c - vec3f(p.pivot)) * p.contrast + vec3f(p.pivot), vec3f(0.0));

  let l = luma(c);
  c = max(mix(vec3f(l), c, p.saturation), vec3f(0.0));

  // Aspect-corrected so the falloff is circular, not elliptical.
  let aspect = max(p.viewport.x, 1.0) / max(p.viewport.y, 1.0);
  let d = length((in.uv - vec2f(0.5)) * vec2f(aspect, 1.0));
  c = c * mix(1.0, smoothstep(1.15, 0.30, d), clamp(p.vignette, 0.0, 1.0));

  // The swapchain format is *not* sRGB (getPreferredCanvasFormat returns
  // bgra8unorm), so the transfer function is applied by hand, matching the
  // pow(2.2) decode the palette uses on the way in.
  return vec4f(pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / max(p.gamma, 0.01))), 1.0);
}
