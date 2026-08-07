struct Uniforms {
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    time: f32,
    _pad0: f32,
    tile_offset: vec2<f32>,
    tile_scale: vec2<f32>,
    _pad_tail: vec4<f32>,
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
    out.uv = pos[vi] * u.tile_scale + u.tile_offset;
    return out;
}

fn hash33(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn hash33v(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        hash33(p),
        hash33(p + vec3<f32>(17.3, 4.7, 11.9)),
        hash33(p + vec3<f32>(31.1, 8.3, 22.7))
    );
}

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

// Blackbody from raw Kelvin — used for star surface
fn blackbody_kelvin(kelvin: f32) -> vec3<f32> {
    let t100 = clamp(kelvin, 1000.0, 40000.0) / 100.0;

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

fn tonemap(hdr_color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = hdr_color * exposure;
    let mapped = exposed / (1.0 + exposed);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

// Voronoi-based solar granulation
// Returns (cell_id_hash, distance_to_edge)
fn voronoi(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);

    var min_dist = 8.0;
    var min_id = 0.0;
    var second_dist = 8.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let cell = i + neighbor;
            let jitter = hash33v(vec3<f32>(cell, 0.0)).xy * 0.5 + 0.25;
            let offset = neighbor + jitter - f;
            let d = dot(offset, offset);
            if (d < min_dist) {
                second_dist = min_dist;
                min_dist = d;
                min_id = hash33(vec3<f32>(cell, 1.0));
            } else if (d < second_dist) {
                second_dist = d;
            }
        }
    }

    return vec2<f32>(min_id, sqrt(second_dist) - sqrt(min_dist));
}

// Solar granulation: hot bright cell centers, dark cool edges
// normal is the surface normal in world space, used to project to 2D surface coords
fn granulation(normal: vec3<f32>, time: f32) -> f32 {
    // Build a stable tangent frame from normal
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    // Project to surface 2D coords, slow drift simulates convection turnover (~10min/granule)
    let drift = time * 0.003;
    let uv = vec2<f32>(dot(normal, tangent), dot(normal, bitangent)) * 14.0 + drift;

    // Coarse granulation (~1000km cells)
    let v_coarse = voronoi(uv);
    // Fine intergranular lanes
    let v_fine = voronoi(uv * 3.1 + vec2<f32>(7.3, 2.1));

    // Edge proximity = dark intergranular lane
    let lane_coarse = smoothstep(0.0, 0.35, v_coarse.y);
    let lane_fine   = smoothstep(0.0, 0.25, v_fine.y);

    // Cell brightness varies by convective upwelling strength (hash per cell)
    let cell_bright = mix(0.75, 1.0, v_coarse.x);

    // Combine: interior bright, lanes dark
    let gran = lane_coarse * lane_fine * cell_bright;

    // Subtle supergranulation modulation (large slow cells)
    let super_gran = fbm(normal * 2.5 + vec3<f32>(drift * 0.1)) * 0.15 + 0.85;

    return gran * super_gran;
}

// Limb darkening — Eddington approximation, u parameter controls strength
// cos_theta = dot(-ray_dir, surface_normal)
fn limb_darkening(cos_theta: f32, u_ld: f32) -> f32 {
    // Standard linear limb darkening law: I(theta) = 1 - u*(1 - cos_theta)
    // Wavelength-averaged u ~ 0.6 for sun-like stars
    return 1.0 - u_ld * (1.0 - max(0.0, cos_theta));
}

// Solar prominence: arching filament of plasma above surface
// Returns emitted color+alpha along a ray that passed near the surface
fn prominence(normal: vec3<f32>, ray_dir: vec3<f32>, time: f32) -> vec4<f32> {
    // Prominences live in the chromosphere, just above the photosphere
    // We model a few fixed arch shapes in surface-normal space
    var prom_emit = vec3<f32>(0.0);
    var prom_alpha = 0.0;

    // Use a noise field to place and animate prominences organically
    let arch_noise = ridged_fbm(normal * 5.0 + vec3<f32>(time * 0.01, 0.0, 0.0));
    let arch_mask = smoothstep(0.55, 0.85, arch_noise);

    if (arch_mask > 0.0) {
        // Height above surface: use a separate noise driven "arch" profile
        let height_profile = sin(arch_noise * 3.14159) * arch_mask;
        // Prominence plasma is ~8000-30000K, cooler than photosphere, reddish-pink
        let prom_temp = mix(8000.0, 20000.0, hash33(floor(normal * 4.0)));
        let prom_color = blackbody_kelvin(prom_temp);
        // Modulate by height — thicker in the arch body
        let thickness = height_profile * arch_mask * 0.6;
        prom_emit = prom_color * thickness * 3.5;
        prom_alpha = thickness * 0.7;
    }

    return vec4<f32>(prom_emit, prom_alpha);
}

// Chromosphere: thin hot halo just above the photosphere
// Seen mainly at limb. Returns additive glow color.
fn chromosphere(cos_theta: f32, gran: f32) -> vec3<f32> {
    // Chromosphere is ~10,000K, blue-ish emission lines
    // Brightens dramatically toward limb (1 - cos_theta)
    let limb_factor = pow(1.0 - max(0.0, cos_theta), 4.0);
    // Chromospheric spicules — thin dark-bright streaks
    let spicule_temp = 10000.0 + gran * 5000.0;
    let chrom_color = blackbody_kelvin(spicule_temp) * vec3<f32>(0.6, 0.8, 1.4); // bias toward Ca II H&K blue
    return chrom_color * limb_factor * 1.2;
}

// Star surface shading: granulation + limb darkening + chromosphere + prominences
// stellar_kelvin: photospheric temperature
// normal: outward surface normal
// ray_dir: incident ray direction (toward surface)
fn shade_star_surface(normal: vec3<f32>, ray_dir: vec3<f32>, stellar_kelvin: f32, time: f32) -> vec3<f32> {
    let cos_theta = max(0.0, dot(-ray_dir, normal));

    // Granulation pattern (convection cells)
    let gran = granulation(normal, time);

    // Local temperature variation from granulation
    // Hot upwelling cells ~+300K, cool lanes ~-500K
    let gran_temp = stellar_kelvin + (gran - 0.7) * 800.0;
    var surface_color = blackbody_kelvin(clamp(gran_temp, 2000.0, 50000.0));

    // Starspots: dark cool regions (~3500-4500K for G-type)
    // Use a slow fbm to place spots, stable over convection timescale
    let spot_noise = fbm(normal * 3.0 + vec3<f32>(time * 0.0005));
    let spot_mask = smoothstep(0.72, 0.62, spot_noise);
    let spot_temp = stellar_kelvin * 0.65; // umbra ~65% of photospheric temp
    let spot_color = blackbody_kelvin(clamp(spot_temp, 2000.0, 50000.0));
    surface_color = mix(surface_color, spot_color, spot_mask * 0.8);

    // Limb darkening (wavelength-averaged u=0.6 for G-type, adjust per spectral class)
    let u_ld = select(0.55, 0.65, stellar_kelvin < 7000.0); // cooler stars darker limb
    let ld = limb_darkening(cos_theta, u_ld);
    surface_color *= ld;

    // Chromosphere glow at limb
    let chrom = chromosphere(cos_theta, gran);
    surface_color += chrom;

    // Solar flare: occasional hot bright patches (~50,000K white)
    let flare_noise = ridged_fbm(normal * 8.0 + vec3<f32>(time * 0.05));
    let flare_mask = smoothstep(0.88, 0.96, flare_noise) * smoothstep(0.5, 0.0, spot_mask);
    let flare_color = blackbody_kelvin(50000.0) * 4.0;
    surface_color = mix(surface_color, flare_color, flare_mask);

    // Prominence emission (additive, seen above surface)
    let prom = prominence(normal, ray_dir, time);
    surface_color += prom.rgb;

    // Overall surface brightness
    return surface_color * 2.5;
}

// ---- Stellar spectral classes ----
// Returns (temperature_kelvin, radius_scale, color_tint)
// Used for background stars to give each a physically correct color
struct StarClass {
    kelvin: f32,
    brightness: f32,
    color: vec3<f32>,
};

fn star_class_from_hash(h: f32) -> StarClass {
    var sc: StarClass;
    // IMF-weighted distribution: mostly K/G/M, rare O/B
    if (h < 0.60) {
        // M dwarf: 2500-3700K, dim red
        sc.kelvin = mix(2500.0, 3700.0, h / 0.60);
        sc.brightness = 0.4;
        sc.color = blackbody_kelvin(sc.kelvin);
    } else if (h < 0.85) {
        // K dwarf: 3700-5200K, orange
        sc.kelvin = mix(3700.0, 5200.0, (h - 0.60) / 0.25);
        sc.brightness = 0.75;
        sc.color = blackbody_kelvin(sc.kelvin);
    } else if (h < 0.95) {
        // G dwarf (sun-like): 5200-6000K, yellow-white
        sc.kelvin = mix(5200.0, 6000.0, (h - 0.85) / 0.10);
        sc.brightness = 1.0;
        sc.color = blackbody_kelvin(sc.kelvin);
    } else if (h < 0.98) {
        // F/A: 6000-10000K, white-blue
        sc.kelvin = mix(6000.0, 10000.0, (h - 0.95) / 0.03);
        sc.brightness = 2.0;
        sc.color = blackbody_kelvin(sc.kelvin);
    } else {
        // B/O: 10000-40000K, blue — very rare, very bright
        sc.kelvin = mix(10000.0, 40000.0, (h - 0.98) / 0.02);
        sc.brightness = 5.0;
        sc.color = blackbody_kelvin(sc.kelvin);
    }
    return sc;
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

fn sample_accretion_disk(pos: vec3<f32>, vel: vec3<f32>) -> vec4<f32> {
    let pitch_angle = 0.3;
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
    let r_outer = 24.0;

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

    let base_radial = smoothstep(r_isco, r_isco + 0.4, r) * (1.0 - smoothstep(14.0, 22.0, r));
    let raw_density = base_radial * clumpy_texture * height;
    let uneven_density = pow(max(0.0, raw_density), 1.3);

    let corona_radial = smoothstep(r_isco, r_isco + 0.5, r) * (1.0 - smoothstep(15.0, 24.0, r));
    let corona_scale = disk_scale * 4.0;
    let corona_height = exp(-(y_offset * y_offset) / (corona_scale * corona_scale));
    let corona_noise = fbm(warp_p * 0.5 + vec3<f32>(dilated_time * 0.1));
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

    let scattering_albedo = 0.75;
    let scattering_phase = 0.85 + 0.25 * pow(max(0.0, dot(normalize(pitched_pos), -pitched_vel)), 2.0);
    let scattered_light = mix(color * 0.5, vec3<f32>(0.5, 0.75, 1.0) * 2.5, 0.6) * scattering_phase * scattering_albedo;

    let corona_color = mix(color * 0.4, scattered_light, 0.7);
    let final_color = mix(color, corona_color, corona_density / (total_density + 0.001));

    return vec4<f32>(final_color * total_density, total_density * 0.45);
}



// ---- Enhanced star field ----
// All lattice lookups done in spherical (phi, theta) space so cells are
// uniform angular patches — no axis-aligned rectangular distortion.
fn sample_starfield(dir: vec3<f32>) -> vec3<f32> {
    var star_light = vec3<f32>(0.0);

    // Project direction to spherical coords: uniform 2D domain, no distortion
    let phi   = atan2(dir.z, dir.x);            // [-pi, pi]
    let theta = asin(clamp(dir.y, -1.0, 1.0));  // [-pi/2, pi/2]
    let sph   = vec2<f32>(phi, theta);

    // --- Scale 1: dense small stars ---
    let scale1 = 400.0;
    let cell1   = sph * scale1;
    let id1_2d  = floor(cell1);
    let raw1    = fract(cell1) - 0.5;
    let seed1   = vec3<f32>(id1_2d, 0.0);
    let h1      = hash33(seed1);
    let j1      = (hash33v(seed1 + vec3<f32>(42.1, 7.3, 0.0)) - vec3<f32>(0.5)).xy * 0.4;
    let uv1     = raw1 - j1;
    let dist1   = length(uv1);

    if (h1 > 0.999) {
        let sc        = star_class_from_hash(hash33(seed1 + vec3<f32>(3.7, 1.1, 5.3)));
        let intensity = pow((h1 - 0.995) / 0.005, 2.0) * 2.0 * sc.brightness;
        let core      = smoothstep(0.15, 0.0, dist1) * intensity;
        let ring1     = smoothstep(0.28, 0.18, dist1) * smoothstep(0.18, 0.28, dist1) * intensity * 0.15;
        let ring2     = smoothstep(0.42, 0.34, dist1) * smoothstep(0.34, 0.42, dist1) * intensity * 0.07;
        star_light   += sc.color * (core + ring1 + ring2);
    }

    // --- Scale 2: medium stars ---
    let scale2 = 200.0;
    let cell2   = sph * scale2 + 89.0;
    let id2_2d  = floor(cell2);
    let raw2    = fract(cell2) - 0.5;
    let seed2   = vec3<f32>(id2_2d, 1.0);
    let h2      = hash33(seed2);
    let j2      = (hash33v(seed2 + vec3<f32>(13.7, 55.1, 0.0)) - vec3<f32>(0.5)).xy * 0.4;
    let uv2     = raw2 - j2;
    let dist2   = length(uv2);

    if (h2 > 0.995) {
        let sc2        = star_class_from_hash(hash33(seed2 + vec3<f32>(9.1, 4.4, 2.8)));
        let intensity2 = pow((h2 - 0.985) / 0.015, 1.5) * 3.5 * sc2.brightness;
        let glow       = exp(-dist2 * dist2 * 16.0);
        let spike_x    = exp(-abs(uv2.x) * 40.0) * exp(-uv2.y * uv2.y * 200.0) * 0.3;
        let spike_y    = exp(-abs(uv2.y) * 40.0) * exp(-uv2.x * uv2.x * 200.0) * 0.3;
        let spikes     = (spike_x + spike_y) * smoothstep(0.5, 0.8, h2);
        star_light    += sc2.color * (glow + spikes) * intensity2;
    }

    // --- Scale 3: rare giant/supergiant stars ---
    let scale3 = 1.0;
    let cell3   = sph * scale3 + 99.99999999999;
    let id3_2d  = floor(cell3);
    let raw3    = fract(cell3) - 0.5;
    let seed3   = vec3<f32>(id3_2d, 2.0);
    let h3      = hash33(seed3);
    let j3      = (hash33v(seed3 + vec3<f32>(77.3, 22.9, 0.0)) - vec3<f32>(0.5)).xy * 0.4;
    let uv3     = raw3 - j3;
    let dist3   = length(uv3);

    if (h3 > 0.9985) {
        let sc3        = star_class_from_hash(hash33(seed3 + vec3<f32>(6.6, 3.3, 8.8)));
        let intensity3 = pow((h3 - 0.995) / 0.005, 1.2) * 8.0 * sc3.brightness;
        let glow3      = exp(-dist3 * dist3 * 6.0);
        let halo3      = exp(-dist3 * dist3 * 1.5) * 0.15;
        let spike_x3   = exp(-abs(uv3.x) * 20.0) * exp(-uv3.y * uv3.y * 80.0) * 0.5;
        let spike_y3   = exp(-abs(uv3.y) * 20.0) * exp(-uv3.x * uv3.x * 80.0) * 0.5;
        let diag1      = exp(-abs(uv3.x + uv3.y) * 28.0) * exp(-(uv3.x - uv3.y) * (uv3.x - uv3.y) * 80.0) * 0.25;
        let diag2      = exp(-abs(uv3.x - uv3.y) * 28.0) * exp(-(uv3.x + uv3.y) * (uv3.x + uv3.y) * 80.0) * 0.25;
        star_light    += sc3.color * (glow3 + halo3 + spike_x3 + spike_y3 + diag1 + diag2) * intensity3;
    }

    return star_light;
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
            color += sample_starfield(dir) * transmittance;
            break;
        }

        // Adaptive Step Sizing
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

    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bloom_threshold = 0.75;
    let bloom_factor = smoothstep(bloom_threshold, bloom_threshold + 0.4, luminance);
    let bright_highlights = color * bloom_factor * 1.5;

    let final_color = tonemap(color, 1.6);
    return vec4<f32>(final_color, max(max(bright_highlights.r, bright_highlights.g), bright_highlights.b));
}