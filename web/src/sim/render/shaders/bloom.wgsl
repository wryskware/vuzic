// The bloom chain: one thresholding prefilter that also halves resolution, a
// box downsample per extra level, a separable 9-tap Gaussian per level, and a
// tent upsample that is additively blended into the level above.
//
// Every pass shares one bind group shape — params, sampler, one source texture —
// so the source's size is read from the texture itself (`textureDimensions`)
// rather than passed in. That is why there are no per-pass uniforms and no
// dynamic offsets anywhere in this file.

@group(0) @binding(0) var<uniform> p: Post;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;

fn srcTexel() -> vec2f {
  return 1.0 / max(vec2f(textureDimensions(src)), vec2f(1.0));
}

/** 2x2 bilinear taps = a 4x4 tent over the source. */
fn box4(uv: vec2f, r: vec2f) -> vec3f {
  var s = textureSampleLevel(src, samp, uv + vec2f(-r.x, -r.y), 0.0).rgb;
  s = s + textureSampleLevel(src, samp, uv + vec2f(r.x, -r.y), 0.0).rgb;
  s = s + textureSampleLevel(src, samp, uv + vec2f(-r.x, r.y), 0.0).rgb;
  s = s + textureSampleLevel(src, samp, uv + vec2f(r.x, r.y), 0.0).rgb;
  return s * 0.25;
}

// Soft-knee threshold (Karis). A hard threshold makes bloom pop on and off as a
// filament crosses it, which reads as flicker; the knee makes the transition
// quadratic instead.
@fragment
fn fsPrefilter(in: PostVSOut) -> @location(0) vec4f {
  let c = box4(in.uv, srcTexel() * 0.5);
  let br = max(c.r, max(c.g, c.b));
  let knee = max(p.knee, 1e-4);
  let soft = clamp(br - p.threshold + knee, 0.0, 2.0 * knee);
  let contrib = max(soft * soft / (4.0 * knee), br - p.threshold) / max(br, 1e-5);
  return vec4f(c * max(contrib, 0.0), 1.0);
}

@fragment
fn fsDownsample(in: PostVSOut) -> @location(0) vec4f {
  return vec4f(box4(in.uv, srcTexel() * 0.5), 1.0);
}

// 9-tap Gaussian as 5 bilinear taps; sigma ~2 texels at the level's own scale.
fn gauss(uv: vec2f, dir: vec2f) -> vec3f {
  let t = srcTexel() * dir;
  var s = textureSampleLevel(src, samp, uv, 0.0).rgb * 0.2270270270;
  s = s + (textureSampleLevel(src, samp, uv + t * 1.3846153846, 0.0).rgb
         + textureSampleLevel(src, samp, uv - t * 1.3846153846, 0.0).rgb) * 0.3162162162;
  s = s + (textureSampleLevel(src, samp, uv + t * 3.2307692308, 0.0).rgb
         + textureSampleLevel(src, samp, uv - t * 3.2307692308, 0.0).rgb) * 0.0702702703;
  return s;
}

@fragment
fn fsBlurH(in: PostVSOut) -> @location(0) vec4f {
  return vec4f(gauss(in.uv, vec2f(1.0, 0.0)), 1.0);
}

@fragment
fn fsBlurV(in: PostVSOut) -> @location(0) vec4f {
  return vec4f(gauss(in.uv, vec2f(0.0, 1.0)), 1.0);
}

// 3x3 tent, drawn with additive blending into the next level up.
@fragment
fn fsUpsample(in: PostVSOut) -> @location(0) vec4f {
  let t = srcTexel();
  var s = textureSampleLevel(src, samp, in.uv, 0.0).rgb * 4.0;
  s = s + (textureSampleLevel(src, samp, in.uv + vec2f(t.x, 0.0), 0.0).rgb
         + textureSampleLevel(src, samp, in.uv - vec2f(t.x, 0.0), 0.0).rgb
         + textureSampleLevel(src, samp, in.uv + vec2f(0.0, t.y), 0.0).rgb
         + textureSampleLevel(src, samp, in.uv - vec2f(0.0, t.y), 0.0).rgb) * 2.0;
  s = s + (textureSampleLevel(src, samp, in.uv + t, 0.0).rgb
         + textureSampleLevel(src, samp, in.uv - t, 0.0).rgb
         + textureSampleLevel(src, samp, in.uv + vec2f(t.x, -t.y), 0.0).rgb
         + textureSampleLevel(src, samp, in.uv + vec2f(-t.x, t.y), 0.0).rgb);
  return vec4f(s * (1.0 / 16.0), 1.0);
}
