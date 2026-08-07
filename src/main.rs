use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};
use image::{ImageBuffer, Rgba};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RaytraceUniforms {
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    time: f32,
    _pad0: f32,
    tile_offset: [f32; 2],
    tile_scale: [f32; 2],
    // WGSL inserts an 8-byte alignment gap here because _pad_tail (vec4,
    // align 16) can't start at byte 168 -- it gets pushed to 176. Mirror
    // that gap explicitly so Rust's #[repr(C)] layout matches byte-for-byte.
    _pad_gap: [f32; 2],
    _pad_tail: [f32; 4],
}

fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    device.create_texture(&desc)
}

/// Renders one tile of the final image at `tile_w`x`tile_h` resolution (well
/// under the GPU's max 2D texture dimension, e.g. <= 8192) and reads it back
/// into a CPU-side RGBA8 buffer. `tile_offset`/`tile_scale` window the ray
/// directions into the sub-rectangle of NDC space this tile covers, so
/// adjacent tiles line up pixel-perfectly with no visible seams (same camera,
/// same math, just a different slice of the frustum per tile).
async fn render_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    raytrace_pipeline: &wgpu::RenderPipeline,
    h_blur_pipeline: &wgpu::RenderPipeline,
    v_blur_pipeline: &wgpu::RenderPipeline,
    composite_pipeline: &wgpu::RenderPipeline,
    post_bgl: &wgpu::BindGroupLayout,
    composite_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ray_bgl: &wgpu::BindGroupLayout,
    tile_w: u32,
    tile_h: u32,
    ray_u: RaytraceUniforms,
    // (global_offset_x, global_offset_y, global_scale_x, global_scale_y):
    // where this tile sits within the full stitched image, in [0,1] uv
    // space, so screen-space-global effects (vignette, film grain) in the
    // composite shader are computed relative to the whole image instead of
    // each tile independently. [0,0,1,1] for a non-tiled full-frame render.
    composite_global_info: [f32; 4],
) -> Result<Vec<u8>, String> {
    let scene_tex = create_hdr_texture(device, tile_w, tile_h, "Tile Scene Texture");
    let scene_view = scene_tex.create_view(&Default::default());

    let bloom_tex_1 = create_hdr_texture(device, tile_w / 2, tile_h / 2, "Tile Bloom 1");
    let bloom_view_1 = bloom_tex_1.create_view(&Default::default());

    let bloom_tex_2 = create_hdr_texture(device, tile_w / 2, tile_h / 2, "Tile Bloom 2");
    let bloom_view_2 = bloom_tex_2.create_view(&Default::default());

    let output_desc = wgpu::TextureDescriptor {
        label: Some("Tile Output Texture"),
        size: wgpu::Extent3d { width: tile_w, height: tile_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let output_tex = device.create_texture(&output_desc);
    let output_view = output_tex.create_view(&Default::default());

    let ray_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tile Ray Uniform Buffer"),
        contents: bytemuck::bytes_of(&ray_u),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let ray_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile Ray BG"),
        layout: ray_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: ray_uniform_buffer.as_entire_binding() }],
    });

    let h_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile H Blur BG"),
        layout: post_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });

    let v_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile V Blur BG"),
        layout: post_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bloom_view_1) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });

    let composite_global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tile Composite Global Info Buffer"),
        contents: bytemuck::bytes_of(&composite_global_info),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile Composite BG"),
        layout: composite_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bloom_view_2) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 5, resource: composite_global_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Tile Encoder") });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Tile Raytrace Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &scene_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(raytrace_pipeline);
        rpass.set_bind_group(0, &ray_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Tile H Blur Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &bloom_view_1,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(h_blur_pipeline);
        rpass.set_bind_group(0, &h_blur_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Tile V Blur Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &bloom_view_2,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(v_blur_pipeline);
        rpass.set_bind_group(0, &v_blur_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Tile Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(composite_pipeline);
        rpass.set_bind_group(0, &composite_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    let unaligned_bytes_per_row = tile_w * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unaligned_bytes_per_row + align - 1) & !(align - 1);
    let buffer_size = (padded_bytes_per_row * tile_h) as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tile Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &output_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(tile_h),
            },
        },
        wgpu::Extent3d { width: tile_w, height: tile_h, depth_or_array_layers: 1 },
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    if receiver.recv().unwrap().is_err() {
        return Err("Failed to map buffer for tile screenshot".into());
    }

    let data = buffer_slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((tile_w * tile_h * 4) as usize);
    for row in 0..tile_h {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + (tile_w * 4) as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    staging_buffer.unmap();

    // Explicitly drop all tile-local GPU resources and force a poll so
    // wgpu actually reclaims their VRAM before the next tile starts
    // allocating. Without this, resource destruction can be deferred until
    // some later poll, and repeated per-tile allocation can pile up VRAM
    // pressure across many tiles until an allocation stalls.
    drop(scene_view);
    drop(scene_tex);
    drop(bloom_view_1);
    drop(bloom_tex_1);
    drop(bloom_view_2);
    drop(bloom_tex_2);
    drop(output_view);
    drop(output_tex);
    drop(ray_uniform_buffer);
    drop(composite_global_buffer);
    drop(staging_buffer);
    device.poll(wgpu::Maintain::Wait).panic_on_timeout();

    Ok(pixels)
}

/// Renders a `width`x`height` image by stitching together tiles no larger
/// than `max_tile_dim` on a side (kept comfortably under the GPU's
/// max_texture_dimension_2d, since consumer cards generally cap out around
/// 8K-ish 2D texture support). Each tile is rendered fully offscreen in VRAM
/// (small, tile-sized), read back to a small CPU staging buffer, then copied
/// into its place in one big RAM-resident image buffer. The full final image
/// therefore lives only in RAM, never in VRAM -- so e.g. a 128K shot's ~34GB
/// RGBA8 buffer is bounded by system RAM, not GPU memory.
async fn take_tiled_screenshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    raytrace_pipeline: &wgpu::RenderPipeline,
    h_blur_pipeline: &wgpu::RenderPipeline,
    v_blur_pipeline: &wgpu::RenderPipeline,
    composite_pipeline: &wgpu::RenderPipeline,
    ray_bgl: &wgpu::BindGroupLayout,
    post_bgl: &wgpu::BindGroupLayout,
    composite_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    yaw: f32,
    pitch: f32,
    camera_distance: f32,
    time: f32,
    width: u32,
    height: u32,
    max_tile_dim: u32,
    out_path: &str,
) -> Result<(), String> {
    let aspect = width as f32 / height as f32;

    // Pick a tile size that divides the image into a whole number of tiles
    // and stays under max_tile_dim (and under the device's actual texture
    // limit, as a safety net in case max_tile_dim was set too high).
    let device_limit = device.limits().max_texture_dimension_2d;
    let safe_tile_dim = max_tile_dim.min(device_limit);

    let tiles_x = (width + safe_tile_dim - 1) / safe_tile_dim;
    let tiles_y = (height + safe_tile_dim - 1) / safe_tile_dim;
    let tile_w = (width + tiles_x - 1) / tiles_x;
    let tile_h = (height + tiles_y - 1) / tiles_y;
    // Round tile dims up to a multiple of 4 to keep things friendly with
    // texture/bytes-per-row alignment; edge tiles get clipped to the exact
    // remainder so the final image is still exactly width x height.
    let tile_w = ((tile_w + 3) & !3).min(safe_tile_dim);
    let tile_h = ((tile_h + 3) & !3).min(safe_tile_dim);

    println!(
        "Rendering {}x{} image as {}x{} tiles of up to {}x{} each ({} tiles total)...",
        width, height, tiles_x, tiles_y, tile_w, tile_h, tiles_x * tiles_y
    );

    let approx_ram_bytes = (width as u64) * (height as u64) * 4;
    println!(
        "Final image will occupy ~{:.2} GB of RAM before encoding.",
        approx_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Full image buffer lives in RAM only. Never touches VRAM as a whole --
    // only individual tiles are ever resident on the GPU.
    let mut full_pixels: Vec<u8> = vec![0u8; approx_ram_bytes as usize];
    let full_stride = (width * 4) as usize;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile_w;
            let y0 = ty * tile_h;
            let this_tile_w = tile_w.min(width - x0);
            let this_tile_h = tile_h.min(height - y0);

            // Render at the full tile_w/tile_h (so uv windowing math is
            // consistent across tiles), then only copy the valid
            // this_tile_w/this_tile_h region for edge tiles that would
            // otherwise overshoot the image bounds.
            let tile_index = ty * tiles_x + tx + 1;
            let tile_start = Instant::now();
            println!(
                "  Tile {}/{}: rendering region ({}, {}) size {}x{}...",
                tile_index, tiles_x * tiles_y, x0, y0, this_tile_w, this_tile_h
            );

            // Map this tile's pixel rect to a sub-rectangle of NDC [-1,1].
            // NDC x in [-1,1] maps to pixel x in [0,width]; y is flipped
            // (NDC +1 = top of screen = pixel row 0).
            let scale_x = tile_w as f32 / width as f32;
            let scale_y = tile_h as f32 / height as f32;
            let ndc_x0 = (x0 as f32 / width as f32) * 2.0 - 1.0;
            let ndc_y1 = 1.0 - (y0 as f32 / height as f32) * 2.0; // top edge, NDC y (flipped)
            let offset_x = ndc_x0 + scale_x;
            let offset_y = ndc_y1 - scale_y;

            let ray_u = build_ray_uniforms_tiled(
                yaw,
                pitch,
                camera_distance,
                aspect,
                time,
                [offset_x, offset_y],
                [scale_x, scale_y],
            );

            // Composite-pass global info uses plain [0,1] uv space (matching
            // bloom.wgsl's own vertex shader, which already outputs uv in
            // [0,1] with y pre-flipped) -- so this is just the tile's
            // top-left corner and size as a fraction of the full image,
            // no NDC conversion needed (unlike the raytrace uniforms above).
            let composite_global_info = [
                x0 as f32 / width as f32,
                y0 as f32 / height as f32,
                scale_x,
                scale_y,
            ];

            device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            device.push_error_scope(wgpu::ErrorFilter::Validation);

            let tile_pixels = render_tile(
                device,
                queue,
                raytrace_pipeline,
                h_blur_pipeline,
                v_blur_pipeline,
                composite_pipeline,
                post_bgl,
                composite_bgl,
                sampler,
                ray_bgl,
                tile_w,
                tile_h,
                ray_u,
                composite_global_info,
            )
            .await?;

            // Pop the error scopes (validation first, since it was pushed
            // last) and surface anything the GPU driver flagged instead of
            // letting the process silently hang or continue in a bad state.
            if let Some(err) = pollster::block_on(device.pop_error_scope()) {
                return Err(format!("GPU validation error on tile {}/{}: {}", tile_index, tiles_x * tiles_y, err));
            }
            if let Some(err) = pollster::block_on(device.pop_error_scope()) {
                return Err(format!(
                    "GPU ran out of memory rendering tile {}/{} ({}x{} tile) -- try a smaller max_tile_dim: {}",
                    tile_index, tiles_x * tiles_y, tile_w, tile_h, err
                ));
            }

            println!(
                "  Tile {}/{} done in {:.1}s",
                tile_index, tiles_x * tiles_y, tile_start.elapsed().as_secs_f32()
            );

            // Blit this tile's valid region into the full image buffer.
            let copy_bytes = (this_tile_w * 4) as usize;
            for row in 0..this_tile_h {
                let src_start = (row * tile_w) as usize * 4;
                let dst_row = y0 + row;
                let dst_start = dst_row as usize * full_stride + (x0 as usize * 4);
                full_pixels[dst_start..dst_start + copy_bytes]
                    .copy_from_slice(&tile_pixels[src_start..src_start + copy_bytes]);
            }
        }
    }

    println!("Encoding final {}x{} PNG to {}... (this may take a while for very large images)", width, height, out_path);
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, full_pixels)
        .ok_or("Failed to create image buffer from stitched pixels")?;

    img.save(out_path).map_err(|e| e.to_string())?;
    println!("Successfully saved screenshot to {}!", out_path);

    Ok(())
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Curved Geodesic Black Hole Raytracer (P=8K, O=16K tiled, I=128K tiled)")
            .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 1000.0))
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .unwrap();

    let mut config = surface.get_default_config(&adapter, 1000, 1000).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    let mut width = config.width.max(1);
    let mut height = config.height.max(1);
    let scene_tex = create_hdr_texture(&device, width, height, "Scene Texture");
    let bloom_tex_1 = create_hdr_texture(&device, width / 2, height / 2, "Bloom Tex 1");
    let bloom_tex_2 = create_hdr_texture(&device, width / 2, height / 2, "Bloom Tex 2");

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let mut camera_distance: f32 = 18.0;
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.20;
    let mut is_dragging = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    // While true, live preview redraws are paused so the GPU's full attention
    // goes to the (potentially many, potentially huge) tile renders for a
    // screenshot capture instead of competing with the 60fps preview loop.
    let mut capturing = false;

    let start_time = Instant::now();
    let ray_uniforms = build_ray_uniforms(yaw, pitch, camera_distance, 1.0, 0.0);
    let ray_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Raytrace Uniform Buffer"),
        contents: bytemuck::bytes_of(&ray_uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let raytrace_shader = device.create_shader_module(wgpu::include_wgsl!("raytrace.wgsl"));
    let bloom_shader = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));

    let ray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Ray Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let ray_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Ray BG"),
        layout: &ray_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: ray_uniform_buffer.as_entire_binding() }],
    });

    let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Post Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { 
                binding: 0, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, 
                count: None 
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 1, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), 
                count: None 
            },
        ],
    });

    let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Composite Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { 
                binding: 2, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, 
                count: None 
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 3, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, 
                count: None 
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 4, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), 
                count: None 
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let make_post_bg = |dev: &wgpu::Device, layout: &wgpu::BindGroupLayout, tex_view: &wgpu::TextureView, samp: &wgpu::Sampler| {
        dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tex_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
            ],
        })
    };

    let mut scene_view = scene_tex.create_view(&Default::default());
    let mut bloom_view_1 = bloom_tex_1.create_view(&Default::default());
    let mut bloom_view_2 = bloom_tex_2.create_view(&Default::default());

    let mut h_blur_bg = make_post_bg(&device, &post_bgl, &scene_view, &sampler);
    let mut v_blur_bg = make_post_bg(&device, &post_bgl, &bloom_view_1, &sampler);

    // Default composite global-info uniform for normal (non-tiled) live
    // rendering: offset (0,0), scale (1,1), i.e. "this tile IS the whole
    // image." Only the tiled screenshot path (render_tile) uses non-default
    // values, computed per-tile.
    let composite_global_default: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let composite_global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Composite Global Info Buffer"),
        contents: bytemuck::bytes_of(&composite_global_default),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Composite BG"),
        layout: &composite_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bloom_view_2) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 5, resource: composite_global_buffer.as_entire_binding() },
        ],
    });

    let make_pipeline = |dev: &wgpu::Device, layout: &wgpu::PipelineLayout, shader: &wgpu::ShaderModule, vs_entry: &str, fs_entry: &str, format: wgpu::TextureFormat| {
        dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs_entry),
            layout: Some(layout),
            vertex: wgpu::VertexState { module: shader, entry_point: vs_entry, buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: fs_entry,
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    };

    let ray_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&ray_bgl], push_constant_ranges: &[] });
    let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&post_bgl], push_constant_ranges: &[] });
    let comp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&composite_bgl], push_constant_ranges: &[] });

    let raytrace_pipeline = make_pipeline(&device, &ray_pipeline_layout, &raytrace_shader, "vs_main", "fs_main", wgpu::TextureFormat::Rgba16Float);
    let h_blur_pipeline = make_pipeline(&device, &post_pipeline_layout, &bloom_shader, "vs_main", "fs_horizontal_blur", wgpu::TextureFormat::Rgba16Float);
    let v_blur_pipeline = make_pipeline(&device, &post_pipeline_layout, &bloom_shader, "vs_main", "fs_vertical_blur", wgpu::TextureFormat::Rgba16Float);
    let composite_pipeline = make_pipeline(&device, &comp_pipeline_layout, &bloom_shader, "vs_main", "fs_composite", config.format);

    // Arc-wrapped handles so a capture can be rendered on its own worker
    // thread without blocking the winit/GUI thread. wgpu's Device/Queue and
    // pipeline/layout objects are Send + Sync, so sharing them across threads
    // for concurrent submission is sound; wgpu serializes actual GPU command
    // submission internally.
    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);
    let raytrace_pipeline_arc = Arc::new(raytrace_pipeline);
    let h_blur_pipeline_arc = Arc::new(h_blur_pipeline);
    let v_blur_pipeline_arc = Arc::new(v_blur_pipeline);
    let composite_pipeline_arc = Arc::new(composite_pipeline);
    let ray_bgl_arc = Arc::new(ray_bgl);
    let post_bgl_arc = Arc::new(post_bgl);
    let composite_bgl_arc = Arc::new(composite_bgl);
    let sampler_arc = Arc::new(sampler);

    // Shadow the original bindings with the Arc-wrapped versions so all the
    // existing `&device`, `&raytrace_pipeline`, etc. call sites below keep
    // working unchanged (Arc<T> derefs to &T).
    let device = device_arc.clone();
    let queue = queue_arc.clone();
    let raytrace_pipeline = raytrace_pipeline_arc.clone();
    let h_blur_pipeline = h_blur_pipeline_arc.clone();
    let v_blur_pipeline = v_blur_pipeline_arc.clone();
    let composite_pipeline = composite_pipeline_arc.clone();
    let ray_bgl = ray_bgl_arc.clone();
    let post_bgl = post_bgl_arc.clone();
    let composite_bgl = composite_bgl_arc.clone();
    let sampler = sampler_arc.clone();

    // Channel the capture worker thread uses to report completion back to
    // the GUI thread without either side blocking on the other.
    let (capture_tx, capture_rx) = std::sync::mpsc::channel::<(Result<(), String>, String)>();

    let target_frame_time = std::time::Duration::from_secs_f32(1.0 / 60.0);
    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, target| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => target.exit(),
        Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
            width = new_size.width.max(1);
            height = new_size.height.max(1);
            config.width = width;
            config.height = height;
            surface.configure(&device, &config);

            let scene_tex = create_hdr_texture(&device, width, height, "Scene Texture");
            let bloom_tex_1 = create_hdr_texture(&device, width / 2, height / 2, "Bloom 1");
            let bloom_tex_2 = create_hdr_texture(&device, width / 2, height / 2, "Bloom 2");

            scene_view = scene_tex.create_view(&Default::default());
            bloom_view_1 = bloom_tex_1.create_view(&Default::default());
            bloom_view_2 = bloom_tex_2.create_view(&Default::default());

            h_blur_bg = make_post_bg(&device, &post_bgl, &scene_view, &sampler);
            v_blur_bg = make_post_bg(&device, &post_bgl, &bloom_view_1, &sampler);
            composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite BG"),
                layout: &composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bloom_view_2) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: composite_global_buffer.as_entire_binding() },
                ],
            });
        }
        Event::WindowEvent { event: WindowEvent::KeyboardInput { event: key_event, .. }, .. } => {
            if key_event.state == ElementState::Pressed {
                // P = 8K single-shot (no tiling needed, fits under GPU texture limits).
                // O = 16K tiled capture. I = 128K tiled capture (huge; RAM-bound, see warning).
                let capture: Option<(u32, u32, u32, &str)> = match key_event.physical_key {
                    // max_tile_dim kept well under the ~8K-ish practical
                    // ceiling on consumer GPUs (rather than right at it) so
                    // there's VRAM headroom left for the bloom/output/staging
                    // textures that get allocated alongside the main tile,
                    // and so repeated per-tile allocation across many tiles
                    // (128K = 144 of them) doesn't creep up against the
                    // limit and stall the driver.
                    PhysicalKey::Code(KeyCode::KeyP) => Some((7680, 4320, 4096, "black_hole_8k.png")),
                    PhysicalKey::Code(KeyCode::KeyO) => Some((15360, 8640, 4096, "black_hole_16k.png")),
                    PhysicalKey::Code(KeyCode::KeyI) => Some((122880, 69120, 4096, "black_hole_128k.png")),
                    _ => None,
                };

                if let Some((w, h, max_tile, out_path)) = capture {
                    if capturing {
                        println!(">>> A capture is already in progress, ignoring key press.");
                    } else {
                        let current_time = start_time.elapsed().as_secs_f32();

                        // Pause the live preview loop so the GPU isn't splitting
                        // time between 60fps preview frames and capture tiles.
                        capturing = true;

                        if w > 8192 || h > 8192 {
                            let approx_gb = (w as u64 * h as u64 * 4) as f64 / (1024.0 * 1024.0 * 1024.0);
                            println!(
                                ">>> Triggering {}x{} tiled capture -- this needs ~{:.1} GB of RAM for the final image, plus time to render and stitch every tile. Live preview paused. Rendering on a background thread so the window stays responsive...",
                                w, h, approx_gb
                            );
                        } else {
                            println!(">>> Triggering {}x{} screenshot capture on a background thread...", w, h);
                        }

                        // Clone Arc handles (cheap refcount bumps) to move
                        // into the worker thread. The GUI/winit thread keeps
                        // its own clones and is never blocked by the capture.
                        let device = device_arc.clone();
                        let queue = queue_arc.clone();
                        let raytrace_pipeline = raytrace_pipeline_arc.clone();
                        let h_blur_pipeline = h_blur_pipeline_arc.clone();
                        let v_blur_pipeline = v_blur_pipeline_arc.clone();
                        let composite_pipeline = composite_pipeline_arc.clone();
                        let ray_bgl = ray_bgl_arc.clone();
                        let post_bgl = post_bgl_arc.clone();
                        let composite_bgl = composite_bgl_arc.clone();
                        let sampler = sampler_arc.clone();
                        let tx = capture_tx.clone();
                        let out_path_owned = out_path.to_string();

                        std::thread::spawn(move || {
                            let res = pollster::block_on(take_tiled_screenshot(
                                &device,
                                &queue,
                                &raytrace_pipeline,
                                &h_blur_pipeline,
                                &v_blur_pipeline,
                                &composite_pipeline,
                                &ray_bgl,
                                &post_bgl,
                                &composite_bgl,
                                &sampler,
                                yaw,
                                pitch,
                                camera_distance,
                                current_time,
                                w,
                                h,
                                max_tile,
                                &out_path_owned,
                            ));
                            // Ignore send errors: if the GUI thread is gone,
                            // there's nothing left to report to.
                            let _ = tx.send((res, out_path_owned));
                        });
                    }
                }
            }
        }
        Event::WindowEvent { event: WindowEvent::MouseInput { button: MouseButton::Left, state, .. }, .. } => {
            is_dragging = state == ElementState::Pressed;
            if !is_dragging { last_mouse_pos = None; }
        }
        Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
            if is_dragging {
                if let Some((lx, ly)) = last_mouse_pos {
                    yaw += (position.x - lx) as f32 * 0.005;
                    pitch = (pitch + (position.y - ly) as f32 * 0.005).clamp(-1.4, 1.4);
                }
                last_mouse_pos = Some((position.x, position.y));
            }
        }
        Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => {
            let scroll = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05 };
            camera_distance = (if scroll > 0.0 { camera_distance * 0.9 } else { camera_distance * 1.1 }).clamp(3.0, 80.0);
        }
        Event::AboutToWait => {
            // Non-blocking check: has the background capture thread finished?
            // try_recv never blocks, so the GUI event loop keeps pumping
            // window events (moves, resizes, close requests) the entire time
            // a capture is rendering, instead of freezing and risking the
            // OS/driver flagging the window as unresponsive.
            if let Ok((res, out_path)) = capture_rx.try_recv() {
                match res {
                    Ok(()) => println!(">>> Screenshot saved successfully to {}!", out_path),
                    Err(e) => eprintln!(">>> Error capturing screenshot: {}", e),
                }
                capturing = false;
                last_frame_time = Instant::now() - target_frame_time;
                window.request_redraw();
            }

            if !capturing && last_frame_time.elapsed() >= target_frame_time { window.request_redraw(); }
        }
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
            if capturing { return; }
            last_frame_time = Instant::now();
            let time = start_time.elapsed().as_secs_f32();
            let aspect = width as f32 / height as f32;

            let ray_u = build_ray_uniforms(yaw, pitch, camera_distance, aspect, time);
            queue.write_buffer(&ray_uniform_buffer, 0, bytemuck::bytes_of(&ray_u));

            let frame = match surface.get_current_texture() {
                Ok(f) => f,
                Err(_) => return,
            };
            let surface_view = frame.texture.create_view(&Default::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Pass Encoder") });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Raytrace Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &scene_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&raytrace_pipeline);
                rpass.set_bind_group(0, &ray_bg, &[]);
                rpass.draw(0..3, 0..1);
            }

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Horizontal Blur"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &bloom_view_1,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&h_blur_pipeline);
                rpass.set_bind_group(0, &h_blur_bg, &[]);
                rpass.draw(0..3, 0..1);
            }

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Vertical Blur"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &bloom_view_2,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&v_blur_pipeline);
                rpass.set_bind_group(0, &v_blur_bg, &[]);
                rpass.draw(0..3, 0..1);
            }

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Composite Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&composite_pipeline);
                rpass.set_bind_group(0, &composite_bg, &[]);
                rpass.draw(0..3, 0..1);
            }

            queue.submit(Some(encoder.finish()));
            frame.present();
        }
        _ => {}
    }).unwrap();
}

fn transpose_4x4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn build_ray_uniforms(yaw: f32, pitch: f32, distance: f32, aspect: f32, time: f32) -> RaytraceUniforms {
    build_ray_uniforms_tiled(yaw, pitch, distance, aspect, time, [0.0, 0.0], [1.0, 1.0])
}

fn build_ray_uniforms_tiled(
    yaw: f32,
    pitch: f32,
    distance: f32,
    aspect: f32,
    time: f32,
    tile_offset: [f32; 2],
    tile_scale: [f32; 2],
) -> RaytraceUniforms {
    let eye_x = distance * pitch.cos() * yaw.sin();
    let eye_y = distance * pitch.sin();
    let eye_z = distance * pitch.cos() * yaw.cos();
    let eye = [eye_x, eye_y, eye_z];

    let view = lookat(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let proj = perspective(60.0_f32.to_radians(), aspect, 0.1, 1000.0);

    RaytraceUniforms {
        inv_view: transpose_4x4(invert_4x4(view)),
        inv_proj: transpose_4x4(invert_4x4(proj)),
        cam_pos: [eye_x, eye_y, eye_z, 0.0],
        time,
        _pad0: 0.0,
        tile_offset,
        tile_scale,
        _pad_gap: [0.0; 2],
        _pad_tail: [0.0; 4],
    }
}

fn perspective(fov_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_rad * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (far - near), 1.0],
        [0.0, 0.0, -(near * far) / (far - near), 0.0],
    ]
}

fn lookat(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let norm = |v: [f32; 3]| {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0001 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 0.0] }
    };
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let f = norm(sub(target, eye));
    let s = norm(cross(up, f));
    let u = cross(f, s);

    [
        [s[0], u[0], f[0], 0.0],
        [s[1], u[1], f[1], 0.0],
        [s[2], u[2], f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), -dot(f, eye), 1.0],
    ]
}

fn invert_4x4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut inv = [[0.0; 4]; 4];
    let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    let det = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

    inv[0][0] = (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * det;
    inv[0][1] = (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * det;
    inv[0][2] = (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * det;
    inv[0][3] = (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * det;
    inv[1][0] = (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * det;
    inv[1][1] = (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * det;
    inv[1][2] = (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * det;
    inv[1][3] = (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * det;
    inv[2][0] = (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * det;
    inv[2][1] = (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * det;
    inv[2][2] = (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * det;
    inv[2][3] = (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * det;
    inv[3][0] = (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * det;
    inv[3][1] = (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * det;
    inv[3][2] = (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * det;
    inv[3][3] = (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * det;

    inv
}