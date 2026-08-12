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
// ---------------------------------------------------------------------------

fn kawase4(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, offset: f32) -> vec4<f32> {
    let ts  = 1.0 / vec2<f32>(textureDimensions(tex));
    let o   = (offset + 0.5) * ts;
    let s00 = textureSample(tex, samp, uv + vec2<f32>( o.x,  o.y));
    let s10 = textureSample(tex, samp, uv + vec2<f32>(-o.x,  o.y));
    let s01 = textureSample(tex, samp, uv + vec2<f32>( o.x, -o.y));
    let s11 = textureSample(tex, samp, uv + vec2<f32>(-o.x, -o.y));
    return (s00 + s10 + s01 + s11) * 0.25;
}

@fragment
fn fs_kawase_down(in: VOut) -> @location(0) vec4<f32> {
    let s = kawase4(input_texture, input_sampler, in.uv, 0.0);
    let bloom_weight = s.a;
    let bloom_threshold = 0.4;
    let mask = max(0.0, bloom_weight - bloom_threshold) / (1.0 - bloom_threshold + 0.0001);
    let knee = smoothstep(0.0, 1.0, mask);
    return vec4<f32>(s.rgb * knee, knee);
}

@fragment
fn fs_kawase_up(in: VOut) -> @location(0) vec4<f32> {
    return kawase4(input_texture, input_sampler, in.uv, 1.0);
}

// ---------------------------------------------------------------------------
// Composite pass with ACES Tonemapping
// ---------------------------------------------------------------------------

@group(0) @binding(2) var scene_texture:  texture_2d<f32>;
@group(0) @binding(3) var bloom_texture:  texture_2d<f32>;
@group(0) @binding(4) var screen_sampler: sampler;

struct CompositeGlobalInfo {
    global_offset: vec2<f32>,
    global_scale:  vec2<f32>,
};
@group(0) @binding(5) var<uniform> global_info: CompositeGlobalInfo;

// Standard fitted ACES tonemapping curve (Narkowicz 2015)
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

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

    // 3. Anamorphic horizontal streak
    var streak = vec3<f32>(0.0);
    let tex_size    = vec2<f32>(textureDimensions(bloom_texture));
    let streak_step = 4.0 / tex_size.x;
    for (var j = -8; j <= 8; j++) {
        let streak_uv = in.uv + vec2<f32>(f32(j) * streak_step, 0.0);
        if (streak_uv.x >= 0.0 && streak_uv.x <= 1.0) {
            let w = exp(-f32(j * j) * 0.06);
            streak += textureSample(bloom_texture, screen_sampler, streak_uv).rgb * w * 0.04;
        }
    }

    let combined = scene + (bloom * 0.8) + ghost_sum + streak;

    // Apply ACES filmic tonemapping curve
    let tonemapped = aces_tonemap(combined);

    // Global UV for effects that span the full tiled image
    let global_uv        = in.uv * global_info.global_scale + global_info.global_offset;
    
    // 4. Film grain — applied after tonemapping (or you can apply before tonemapping if preferred)
    let full_image_size = tex_size / global_info.global_scale;
    let grain = (hash33(vec3<f32>(global_uv * full_image_size, 0.0)) - 0.5) * 0.012;

    return vec4<f32>(tonemapped + grain, 1.0);
}