struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VOut;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = pos[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn hash33(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

// ---------------------------------------------------------------------------
// Dual Kawase bloom — two passes, rotationally symmetric by design.
//
// The key insight: instead of separable H+V Gaussian (which smears point
// sources into horizontal pills), Kawase samples 4 diagonal neighbours at
// offset (n + 0.5) texels. Because the taps are symmetric around 45° axes
// the blur is isotropic — point sources stay circular after both passes.
//
// Pass 1 (fs_kawase_down): downsample 2x + first Kawase blur (offset=0).
//   Reads from full-res scene (Rgba16Float, bloom intensity in alpha channel).
//   Writes to half-res bloom_tex_1.
//
// Pass 2 (fs_kawase_up): upsample back to full-res + second Kawase blur (offset=1).
//   Reads from half-res bloom_tex_1.
//   Writes to full-res bloom_tex_2.
//
// Two passes with offsets 0 and 1 approximate a Gaussian with sigma ~2px.
// For a wider bloom you'd add more passes, but 2 is already much better than
// the old separable approach for point sources.
// ---------------------------------------------------------------------------

// Kawase 4-tap diagonal sample at (offset + 0.5) texels.
// Alpha channel carries bloom weight from the raytrace pass; we preserve it
// through both blur passes so the composite can still use it.
fn kawase4(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, offset: f32) -> vec4<f32> {
    let ts  = 1.0 / vec2<f32>(textureDimensions(tex));
    let o   = (offset + 0.5) * ts;
    let s00 = textureSample(tex, samp, uv + vec2<f32>( o.x,  o.y));
    let s10 = textureSample(tex, samp, uv + vec2<f32>(-o.x,  o.y));
    let s01 = textureSample(tex, samp, uv + vec2<f32>( o.x, -o.y));
    let s11 = textureSample(tex, samp, uv + vec2<f32>(-o.x, -o.y));
    return (s00 + s10 + s01 + s11) * 0.25;
}

// Pass 1: threshold + Kawase offset=0 → half-res bloom_tex_1.
// We downsample via bilinear (the sampler already does this since we're
// writing to a half-res target) and apply the first Kawase kernel.
// Threshold is applied here so only bright pixels contribute to bloom.
@fragment
fn fs_kawase_down(in: VOut) -> @location(0) vec4<f32> {
    // Sample with Kawase offset=0 (taps at ±0.5 texel diagonals).
    // Since bloom_tex_1 is half-res, each texel here covers 2x2 scene pixels —
    // the bilinear sampler gives us a free 2x2 box prefilter on top of Kawase.
    let s = kawase4(input_texture, input_sampler, in.uv, 0.0);

    // Alpha channel from raytrace = pre-computed bloom brightness.
    // Use it as a mask: pixels below threshold contribute nothing.
    let bloom_weight = s.a;
    let bloom_threshold = 0.4;
    let mask = max(0.0, bloom_weight - bloom_threshold) / (1.0 - bloom_threshold + 0.0001);

    // Soft knee: smoothly ramp contribution above threshold
    let knee = smoothstep(0.0, 1.0, mask);

    return vec4<f32>(s.rgb * knee, knee);
}

// Pass 2: Kawase offset=1 + upsample → full-res bloom_tex_2.
// Reading from half-res bloom_tex_1 into a full-res target: bilinear gives
// free upsampling, Kawase gives additional isotropic spread.
@fragment
fn fs_kawase_up(in: VOut) -> @location(0) vec4<f32> {
    return kawase4(input_texture, input_sampler, in.uv, 1.0);
}

// ---------------------------------------------------------------------------
// Composite pass — unchanged bind group layout, same bindings as before.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var scene_texture:  texture_2d<f32>;
@group(0) @binding(3) var bloom_texture:  texture_2d<f32>;
@group(0) @binding(4) var screen_sampler: sampler;

struct CompositeGlobalInfo {
    global_offset: vec2<f32>,
    global_scale:  vec2<f32>,
};
@group(0) @binding(5) var<uniform> global_info: CompositeGlobalInfo;

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    // 1. Chromatic aberration
    let center_uv  = in.uv - vec2<f32>(0.5);
    let aberration = 0.0025;
    let r_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv * (1.0 + aberration)).r;
    let g_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv).g;
    let b_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv * (1.0 - aberration)).b;
    let scene = vec3<f32>(r_col, g_col, b_col);

    let bloom         = textureSample(bloom_texture, screen_sampler, in.uv).rgb;
    let center_vector = vec2<f32>(0.5) - in.uv;

    // 2. Ghost flares (reflected off lens elements)
    var ghost_sum = vec3<f32>(0.0);
    for (var i = 1; i < 4; i++) {
        let ghost_uv = in.uv + center_vector * (f32(i) * 0.35);
        if (ghost_uv.x >= 0.0 && ghost_uv.x <= 1.0 && ghost_uv.y >= 0.0 && ghost_uv.y <= 1.0) {
            let w = 1.0 - abs(f32(i) - 2.0) / 2.0;
            ghost_sum += textureSample(bloom_texture, screen_sampler, ghost_uv).rgb * w * 0.22;
        }
    }

    // 3. Anamorphic horizontal streak (simulates anamorphic lens flares)
    var streak = vec3<f32>(0.0);
    let tex_size    = vec2<f32>(textureDimensions(bloom_texture));
    let streak_step = 4.0 / tex_size.x;
    for (var j = -8; j <= 8; j++) {
        let streak_uv = in.uv + vec2<f32>(f32(j) * streak_step, 0.0);
        if (streak_uv.x >= 0.0 && streak_uv.x <= 1.0) {
            // Gaussian weight along streak, narrow vertically
            let w = exp(-f32(j * j) * 0.06);
            streak += textureSample(bloom_texture, screen_sampler, streak_uv).rgb * w * 0.04;
        }
    }

    let combined = scene + (bloom * 0.8) + ghost_sum + streak;

    // Global UV for effects that span the full tiled image
    let global_uv        = in.uv * global_info.global_scale + global_info.global_offset;
    let global_center_uv = global_uv - vec2<f32>(0.5);

    // 4. Film grain — seeded from global pixel position so tiling is seamless
    let full_image_size = tex_size / global_info.global_scale;
    let grain = (hash33(vec3<f32>(global_uv * full_image_size, 0.0)) - 0.5) * 0.012;

    return vec4<f32>(combined + grain, 1.0);
}