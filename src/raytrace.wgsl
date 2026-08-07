struct Uniforms {
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

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
    out.uv = pos[vi];
    return out;
}

fn hash33(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

// 3D Noise helper for organic clumps
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash33(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash33(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash33(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash33(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash33(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash33(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash33(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash33(i + vec3<f32>(1.0, 1.0, 1.0));

    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}

// Fractal Brownian Motion
fn fbm(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        val += amp * noise3d(pos);
        pos *= 2.03;
        amp *= 0.5;
    }
    return val;
}

// Polynomial fit to Planckian locus
fn blackbody(temp_units: f32) -> vec3<f32> {
    let kelvin = clamp(temp_units, 1.0, 35.0) * 150.0 + 1400.0;
    let t100 = kelvin / 100.0;

    var r: f32;
    var g: f32;
    var b: f32;

    if (t100 <= 66.0) {
        r = 1.0;
        g = clamp(0.3900815787 * log(t100) - 0.6318414437, 0.0, 1.0);
    } else {
        r = clamp(1.29293618 * pow(t100 - 60.0, -0.1332047592), 0.0, 1.0);
        g = clamp(1.12989086 * pow(t100 - 60.0, -0.0755148492), 0.0, 1.0);
    }

    if (t100 >= 66.0) {
        b = 1.0;
    } else if (t100 <= 19.0) {
        b = 0.0;
    } else {
        b = clamp(0.5432067891 * log(t100 - 10.0) - 1.19625408914, 0.0, 1.0);
    }

    return vec3<f32>(r, g, b);
}

// Ridged noise helper for sharp gas tendrils and filaments
fn ridged_fbm(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        let n = noise3d(pos);
        let ridge = 1.0 - abs(n * 2.0 - 1.0);
        val += ridge * ridge * amp;
        pos *= 2.1;
        amp *= 0.5;
    }
    return val;
}

fn sample_accretion_disk(pos: vec3<f32>, vel: vec3<f32>) -> vec4<f32> {
    let pitch_angle = 0.28;
    let cos_p = cos(pitch_angle);
    let sin_p = sin(pitch_angle);

    let pitched_pos = vec3<f32>(
        pos.x,
        pos.y * cos_p - pos.z * sin_p,
        pos.y * sin_p + pos.z * cos_p
    );

    let pitched_vel = vec3<f32>(
        vel.x,
        vel.y * cos_p - vel.z * sin_p,
        vel.y * sin_p + vel.z * cos_p
    );

    let r = length(pitched_pos.xz);
    let r_isco = 2.6;
    let r_outer = 24.0; // Extended bounds to allow gas stream to drift outward smoothly

    if (r < r_isco || r > r_outer) {
        return vec4<f32>(0.0);
    }

    let dilated_time = u.time * sqrt(max(0.1, 1.0 - 1.0 / r));

    let spin = 0.45;
    let phi = atan2(pitched_pos.z, pitched_pos.x);
    let frame_drag = spin / (r * r * r + spin * spin);
    let shear = (3.5 + frame_drag * 2.5) / (r * sqrt(r));
    let twisted_phi = phi - shear * dilated_time + frame_drag * 1.8;

    let warp_p = vec3<f32>(r * cos(twisted_phi), pitched_pos.y * 6.0, r * sin(twisted_phi));
    
    let macro_clumps = ridged_fbm(warp_p * 1.2);
    let micro_strands = ridged_fbm(warp_p * 4.5 + vec3<f32>(dilated_time * 0.5));
    
    var clumpy_texture = macro_clumps * 0.6 + micro_strands * 0.4;
    clumpy_texture = smoothstep(0.2, 0.85, clumpy_texture);

    let height_bump = noise3d(warp_p * 2.5 + vec3<f32>(0.0, dilated_time, 0.0)) * 0.6;
    let disk_scale = (0.015 + 0.010 * (r - r_isco)) * (0.6 + height_bump);
    
    let y_offset = pitched_pos.y + (noise3d(warp_p * 3.0) - 0.5) * 0.03;
    let height = exp(-(y_offset * y_offset) / (disk_scale * disk_scale));

    // Smooth inner/outer transition masks to eliminate blocky artifacts
    let base_radial = smoothstep(r_isco, r_isco + 0.4, r) * (1.0 - smoothstep(14.0, 22.0, r));
    let raw_density = base_radial * clumpy_texture * height;
    let uneven_density = pow(max(0.0, raw_density), 1.3);

    // Smoothly fading corona gas stretching past the main disk edge
    let corona_radial = smoothstep(r_isco, r_isco + 0.5, r) * (1.0 - smoothstep(15.0, 24.0, r));
    let corona_scale = disk_scale * 4.0;
    let corona_height = exp(-(y_offset * y_offset) / (corona_scale * corona_scale));
    let corona_noise = fbm(warp_p * 0.5 + vec3<f32>(dilated_time * 0.1)); // Lower frequency noise for smooth gas
    let corona_density = corona_radial * corona_height * corona_noise * 0.10;

    let total_density = uneven_density + corona_density;
    if (total_density < 0.0005) { return vec4<f32>(0.0); }

    let v_orbital = sqrt(0.5 / r);
    let disk_vel = vec3<f32>(-sin(phi), 0.0, cos(phi)) * v_orbital;
    
    let gamma = 1.0 / sqrt(max(0.001, 1.0 - v_orbital * v_orbital));
    let cos_theta = dot(normalize(disk_vel), -pitched_vel);
    let doppler = 1.0 / (gamma * (1.0 - v_orbital * cos_theta));

    let beaming_factor = pow(clamp(doppler, 0.12, 3.8), 3.8);

    let base_temp = 12.5 * pow(r_isco / r, 0.75);
    let rs = 1.0; 
    let grav_shift = sqrt(max(0.15, 1.0 - rs / r));
    let combined_shift = grav_shift * doppler;
    let effective_temp = base_temp * mix(1.0, combined_shift, 0.5);

    var color = blackbody(effective_temp);
    color *= beaming_factor * grav_shift * 3.8;

    let corona_color = mix(color * 0.4, vec3<f32>(0.6, 0.8, 1.0), 0.5);
    let final_color = mix(color, corona_color, corona_density / (total_density + 0.0001));

    return vec4<f32>(final_color * total_density, total_density * 0.45);
}

fn planet_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash33(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash33(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash33(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash33(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash33(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash33(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash33(i + vec3<f32>(1.0, 1.0, 1.1));
    let n111 = hash33(i + vec3<f32>(1.0, 1.0, 1.0));

    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}

fn tonemap(hdr_color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = hdr_color * exposure;
    let mapped = exposed / (1.0 + exposed);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

fn sample_background_planet(ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    let orbit_speed = u.time * 0.05;
    let orbit_radius = 160.0;

    let planet_center = vec3<f32>(
        cos(orbit_speed) * orbit_radius,
        25.0,
        sin(orbit_speed) * orbit_radius
    );
    let planet_radius = 16.0;

    let oc = ray_pos - planet_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - planet_radius * planet_radius;
    let discriminant = b * b - c;

    if (discriminant < 0.0) {
        return vec4<f32>(0.0);
    }

    let t = -b - sqrt(discriminant);
    if (t < 0.0) {
        return vec4<f32>(0.0);
    }

    let hit_pos = ray_pos + ray_dir * t;
    let normal = normalize(hit_pos - planet_center);

    let light_dir = normalize(vec3<f32>(0.5, 0.7, 0.5));
    let NdotL = max(0.2, dot(normal, light_dir));

    let sample_p = normal * 5.0;
    let bands = sin(normal.y * 16.0 + planet_noise(sample_p) * 2.0) * 0.5 + 0.5;

    let deep_blue = vec3<f32>(0.08, 0.30, 0.85);
    let gold_band = vec3<f32>(0.90, 0.75, 0.40);
    let planet_color = mix(deep_blue, gold_band, bands);

    let fresnel = pow(1.0 - max(0.0, dot(-ray_dir, normal)), 2.5);
    let atmos = vec3<f32>(0.2, 0.6, 1.4) * fresnel * 3.0;

    return vec4<f32>(planet_color * NdotL * 2.2 + atmos, 1.0);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let inv_proj = transpose(u.inv_proj);
    let inv_view = transpose(u.inv_view);

    let target_pos = inv_proj * vec4<f32>(in.uv.x, in.uv.y, 1.0, 1.0);
    let ray_dir_view = normalize(target_pos.xyz / target_pos.w);
    var dir = normalize((inv_view * vec4<f32>(ray_dir_view, 0.0)).xyz);
    var pos = u.cam_pos.xyz;

    let jitter = hash33(vec3<f32>(in.uv * 1000.0, u.time));
    pos += dir * (jitter * 0.005);

    let rs = 1.0;
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;
    var min_r = 1000.0;

    var disk_light_accum = vec3<f32>(0.0);
    var inner_disk_proximity = 0.0;

    for (var step = 0; step < 500; step++) {
        let r2 = dot(pos, pos);
        let r = sqrt(r2);
        
        min_r = min(min_r, r);

        if (r <= rs * 1.01) {
            return vec4<f32>(tonemap(color, 1.6), 1.0);
        }

        if (r > 100.0) {
            let planet_sample = sample_background_planet(pos, dir);
            
            if (planet_sample.a > 0.0) {
                let gravitational_redshift = sqrt(max(0.05, 1.0 - rs / min_r));
                color += planet_sample.rgb * gravitational_redshift * transmittance;
            } else {
                let scale1 = 1600.0;
                let id1 = floor(dir * scale1);
                let uv1 = fract(dir * scale1) - vec3<f32>(0.5);
                let h1 = hash33(id1);
                
                var star_light = vec3<f32>(0.0);
                
                if (h1 > 0.995) {
                    let dist1 = length(uv1);
                    let intensity = pow((h1 - 0.995) / 0.005, 2.0) * 2.0;
                    let star_color = mix(vec3<f32>(0.7, 0.8, 1.0), vec3<f32>(1.0, 0.9, 0.7), hash33(id1 + 1.0));
                    star_light += star_color * smoothstep(0.3, 0.0, dist1) * intensity;
                }

                let scale2 = 480.0;
                let id2 = floor(dir * scale2 + 42.0);
                let uv2 = fract(dir * scale2 + 42.0) - vec3<f32>(0.5);
                let h2 = hash33(id2);

                if (h2 > 0.985) {
                    let dist2 = length(uv2);
                    let intensity = pow((h2 - 0.985) / 0.015, 1.5) * 3.5;
                    let star_color = mix(vec3<f32>(0.6, 0.8, 1.0), vec3<f32>(1.0, 0.95, 0.8), hash33(id2 + 50.0));
                    let glow = exp(-dist2 * dist2 * 16.0);
                    star_light += star_color * glow * intensity;
                }

                color += star_light * transmittance;
            }
            break;
        }

        // Adaptive Step Sizing (Clean & Smooth)
        let photon_sphere_r = 1.5 * rs;
        let dist_from_photon_sphere = abs(r - photon_sphere_r);
        let photon_sphere_tightening = smoothstep(1.2, 0.0, dist_from_photon_sphere);
        let in_disk_zone = smoothstep(24.0, 20.0, r) * smoothstep(2.0, 2.6, r);

        var h_step = clamp((r - rs) * 0.04, 0.003, 0.5);
        h_step = mix(h_step, h_step * 0.15, photon_sphere_tightening);
        h_step = mix(h_step, h_step * 0.35, in_disk_zone);

        let disk = sample_accretion_disk(pos, dir);
        let density = disk.a;

        if (density > 0.0) {
            let opacity_scale = 8.0; 
            let optical_depth = density * opacity_scale * h_step;
            let step_transmittance = exp(-optical_depth);
            let step_factor = 1.0 - step_transmittance;
            
            let sample_color = disk.rgb * transmittance * step_factor;
            color += sample_color;
            disk_light_accum += sample_color;

            transmittance *= step_transmittance;
            if (transmittance < 0.005) { break; }
        }

        let dist_to_isco = abs(r - 3.0);
        inner_disk_proximity += exp(-dist_to_isco * 1.5) * h_step;

        let L = cross(pos, dir);
        let L2 = dot(L, L);
        let base_accel = -1.5 * rs * L2 * pos / (r2 * r2 * r);
        let dispersion_multiplier = 1.0 + (rs / r2) * 0.02;
        let accel = base_accel * dispersion_multiplier;

        dir = normalize(dir + accel * h_step);
        pos += dir * h_step;
    }

    let disk_bloom = disk_light_accum * 0.35;
    let isco_glare = vec3<f32>(1.0, 0.85, 0.5) * inner_disk_proximity * 0.08;

    color += disk_bloom + isco_glare;

    let final_color = tonemap(color, 1.6);
    return vec4<f32>(final_color, 1.0);
}