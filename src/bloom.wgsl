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

// Pseudo-random hash helper for film grain
fn hash33(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

@fragment
fn fs_horizontal_blur(in: VOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(input_texture));
    let texel_size = 1.0 / tex_size;
    
    var result = vec4<f32>(0.0);
    let w0 = 0.227027;
    let w1 = 0.316216;
    let w2 = 0.070270;
    
    result += textureSample(input_texture, input_sampler, in.uv) * w0;
    
    let offset1 = vec2<f32>(texel_size.x * 2.0, 0.0);
    let s1_pos = textureSample(input_texture, input_sampler, in.uv + offset1);
    let s1_neg = textureSample(input_texture, input_sampler, in.uv - offset1);
    result += vec4<f32>((s1_pos.rgb * s1_pos.a) + (s1_neg.rgb * s1_neg.a), 1.0) * w1;

    let offset2 = vec2<f32>(texel_size.x * 4.0, 0.0);
    let s2_pos = textureSample(input_texture, input_sampler, in.uv + offset2);
    let s2_neg = textureSample(input_texture, input_sampler, in.uv - offset2);
    result += vec4<f32>((s2_pos.rgb * s2_pos.a) + (s2_neg.rgb * s2_neg.a), 1.0) * w2;

    return result;
}

@fragment
fn fs_vertical_blur(in: VOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(input_texture));
    let texel_size = 1.0 / tex_size;
    
    var result = vec4<f32>(0.0);
    let w0 = 0.227027;
    let w1 = 0.316216;
    let w2 = 0.070270;
    
    result += textureSample(input_texture, input_sampler, in.uv) * w0;
    
    let offset1 = vec2<f32>(0.0, texel_size.y * 2.0);
    let s1_pos = textureSample(input_texture, input_sampler, in.uv + offset1);
    let s1_neg = textureSample(input_texture, input_sampler, in.uv - offset1);
    result += (s1_pos + s1_neg) * w1;

    let offset2 = vec2<f32>(0.0, texel_size.y * 4.0);
    let s2_pos = textureSample(input_texture, input_sampler, in.uv + offset2);
    let s2_neg = textureSample(input_texture, input_sampler, in.uv - offset2);
    result += (s2_pos + s2_neg) * w2;

    return result;
}

@group(0) @binding(2) var scene_texture: texture_2d<f32>;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var screen_sampler: sampler;

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    // 1. Chromatic Aberration (Subtle color channel separation)
    let center_uv = in.uv - vec2<f32>(0.5);
    let aberration = 0.0025;
    
    let r_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv * (1.0 + aberration)).r;
    let g_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv).g;
    let b_col = textureSample(scene_texture, screen_sampler, 0.5 + center_uv * (1.0 - aberration)).b;
    let scene = vec3<f32>(r_col, g_col, b_col);

    let bloom = textureSample(bloom_texture, screen_sampler, in.uv).rgb;
    let center_vector = vec2<f32>(0.5) - in.uv;
    
    // 2. Ghost Flares
    var ghost_sum = vec3<f32>(0.0);
    for (var i = 1; i < 4; i++) {
        let ghost_uv = in.uv + center_vector * (f32(i) * 0.35);
        if (ghost_uv.x >= 0.0 && ghost_uv.x <= 1.0 && ghost_uv.y >= 0.0 && ghost_uv.y <= 1.0) {
            let sample_weight = 1.0 - abs(f32(i) - 2.0) / 2.0;
            ghost_sum += textureSample(bloom_texture, screen_sampler, ghost_uv).rgb * sample_weight * 0.15;
        }
    }
    
    // 3. Anamorphic Streak
    var streak = vec3<f32>(0.0);
    let tex_size = vec2<f32>(textureDimensions(bloom_texture));
    let streak_step = 4.0 / tex_size.x;
    for (var j = -5; j <= 5; j++) {
        let streak_uv = in.uv + vec2<f32>(f32(j) * streak_step, 0.0);
        if (streak_uv.x >= 0.0 && streak_uv.x <= 1.0) {
            let weight = 1.0 / (1.0 + abs(f32(j)) * 0.5);
            streak += textureSample(bloom_texture, screen_sampler, streak_uv).rgb * weight * 0.08;
        }
    }

    let combined = scene + (bloom * 0.7) + streak;

    // 4. Vignetting (Darker corners)
    let uv_dist = length(center_uv);
    let vignette = clamp(1.0 - uv_dist * uv_dist * 1.2, 0.0, 1.0);
    
    // 5. Film Grain (Organic texture using hash33)
    let grain = (hash33(vec3<f32>(in.uv * tex_size, 0.0)) - 0.5) * 0.01;

    let final_color = (combined * vignette) + grain;
    return vec4<f32>(final_color, 1.0);
}