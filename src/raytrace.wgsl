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
        let ridge = 1.0 - abs(n * 2.0 - 1.0); // Sharp ridges instead of smooth waves
        val += ridge * ridge * amp;            // Square it for extra contrast
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
    let r_outer = 15.0;

    if (r < r_isco || r > r_outer) {
        return vec4<f32>(0.0);
    }

    let dilated_time = u.time * sqrt(max(0.1, 1.0 - 1.0 / r));

    let phi = atan2(pitched_pos.z, pitched_pos.x);
    let shear = 3.5 / (r * sqrt(r));
    let twisted_phi = phi - shear * dilated_time;

    let warp_p = vec3<f32>(r * cos(twisted_phi), pitched_pos.y * 6.0, r * sin(twisted_phi));
    
    // -------------------------------------------------------------
    // 1. RIDGED FILAMENTS & MICRO-CLUMPS
    // -------------------------------------------------------------
    let macro_clumps = ridged_fbm(warp_p * 1.2);
    let micro_strands = ridged_fbm(warp_p * 4.5 + vec3<f32>(dilated_time * 0.5));
    
    var clumpy_texture = macro_clumps * 0.6 + micro_strands * 0.4;
    
    // Hard cutoff: zero out weak density so there is empty space between bumps
    clumpy_texture = smoothstep(0.2, 0.85, clumpy_texture);

    // -------------------------------------------------------------
    // 2. BUMPY DISK THICKNESS (Displaces height vertically)
    // -------------------------------------------------------------
    let height_bump = noise3d(warp_p * 2.5 + vec3<f32>(0.0, dilated_time, 0.0)) * 0.6;
    let disk_scale = (0.015 + 0.010 * (r - r_isco)) * (0.6 + height_bump);
    
    // Evaluate height profile with perturbed vertical coordinate
    let y_offset = pitched_pos.y + (noise3d(warp_p * 3.0) - 0.5) * 0.03;
    let height = exp(-(y_offset * y_offset) / (disk_scale * disk_scale));

    if (height < 0.001) { return vec4<f32>(0.0); }

    let base_radial = smoothstep(r_isco, r_isco + 0.3, r) * (1.0 - smoothstep(r_outer - 1.0, r_outer, r));
    
    // Combine into sharp, non-linear density
    let raw_density = base_radial * clumpy_texture * height;
    if (raw_density < 0.002) { return vec4<f32>(0.0); }
    
    let uneven_density = pow(raw_density, 1.3);

    // -------------------------------------------------------------
    // RELATIVISTIC DOPPLER & BLACKBODY COLORING
    // -------------------------------------------------------------
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

    return vec4<f32>(color * uneven_density, uneven_density * 0.45);
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
    let n011 = hash33(i + vec3<f32>(0.0, 1.0, 1.0));
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

    // Ray dithering to prevent step banding
    let jitter = hash33(vec3<f32>(in.uv * 1000.0, u.time));
    pos += dir * (jitter * 0.005);

    let rs = 1.0;
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;
    var min_r = 1000.0;

    // Tracks total light energy accumulated from the accretion disk
    var disk_light_accum = vec3<f32>(0.0);
    // Tracks proximity to the bright inner disk edge (near ISCO)
    var inner_disk_proximity = 0.0;

    for (var step = 0; step < 500; step++) {
        let r2 = dot(pos, pos);
        let r = sqrt(r2);
        
        min_r = min(min_r, r);

        // 1. Captured by Event Horizon -> Pure Pitch Black (No Glow)
        if (r <= rs * 1.01) {
            return vec4<f32>(tonemap(color, 1.6), 1.0);
        }

        // 2. Escape to Deep Space
        if (r > 100.0) {
            let planet_sample = sample_background_planet(pos, dir);
            
            if (planet_sample.a > 0.0) {
                let gravitational_redshift = sqrt(max(0.05, 1.0 - rs / min_r));
                color += planet_sample.rgb * gravitational_redshift * transmittance;
            } else {
                let star_scale = 160.0;
                let grid = dir * star_scale;
                let cell_id = floor(grid);
                let cell_uv = fract(grid) - vec3<f32>(0.5);

                let h = hash33(cell_id);
                var star_brightness = 0.0;

                if (h > 0.993) {
                    let dist = length(cell_uv);
                    let star_shape = smoothstep(0.25, 0.0, dist); 
                    let star_intensity = (h - 0.993) / 0.007; 
                    star_brightness = star_shape * star_intensity * 1.5;
                }

                color += vec3<f32>(star_brightness) * transmittance;
            }
            break;
        }

        // Adaptive Step Sizing
        let photon_sphere_r = 1.5 * rs;
        let dist_from_photon_sphere = abs(r - photon_sphere_r);
        let photon_sphere_tightening = smoothstep(1.2, 0.0, dist_from_photon_sphere);
        let in_disk_zone = smoothstep(16.0, 14.0, r) * smoothstep(2.0, 2.6, r);

        var h_step = clamp((r - rs) * 0.04, 0.003, 0.5);
        h_step = mix(h_step, h_step * 0.15, photon_sphere_tightening);
        h_step = mix(h_step, h_step * 0.35, in_disk_zone);

        // 3. Volumetric Accretion Disk Sampling
        let disk = sample_accretion_disk(pos, dir);
        if (disk.a > 0.0) {
            let sample_color = disk.rgb * transmittance;
            color += sample_color;
            disk_light_accum += sample_color; // Store disk radiance for bloom

            transmittance *= (1.0 - disk.a);
            if (transmittance < 0.005) { break; }
        }

        // Measure how close the ray grazed to the scorching inner edge of the disk (r ~ 2.6 - 4.0)
        let dist_to_isco = abs(r - 3.0);
        inner_disk_proximity += exp(-dist_to_isco * 1.5) * h_step;

        // 4. Geodesic Light Ray Bending
        let L = cross(pos, dir);
        let L2 = dot(L, L);
        let accel = -1.5 * rs * L2 * pos / (r2 * r2 * r);

        dir = normalize(dir + accel * h_step);
        pos += dir * h_step;
    }

    // -------------------------------------------------------------
    // ACCRETION DISK BLOOM & PHOTON RING BLEED
    // -------------------------------------------------------------
    // 1. Direct light halo: intense glowing pixels bleed outward proportionally
    let disk_bloom = disk_light_accum * 0.35;

    // 2. Inner-edge thermal glare: subtle optical bleed from grazing the ISCO zone
    let isco_glare = vec3<f32>(1.0, 0.85, 0.5) * inner_disk_proximity * 0.08;

    color += disk_bloom + isco_glare;

    let final_color = tonemap(color, 1.6);
    return vec4<f32>(final_color, 1.0);
}