use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RaytraceUniforms {
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Curved Geodesic Black Hole Raytracer with Bloom")
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

    let start_time = Instant::now();
    let ray_uniforms = build_ray_uniforms(yaw, pitch, camera_distance, 1.0, 0.0);
    let ray_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Raytrace Uniform Buffer"),
        contents: bytemuck::bytes_of(&ray_uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Shaders & Pipelines setup
    let raytrace_shader = device.create_shader_module(wgpu::include_wgsl!("raytrace.wgsl"));
    let bloom_shader = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));

    let ray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Ray Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
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
                ty: wgpu::BindingType::Texture { 
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                    view_dimension: wgpu::TextureViewDimension::D2, 
                    multisampled: false 
                }, 
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
                ty: wgpu::BindingType::Texture { 
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                    view_dimension: wgpu::TextureViewDimension::D2, 
                    multisampled: false 
                }, 
                count: None 
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 3, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Texture { 
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                    view_dimension: wgpu::TextureViewDimension::D2, 
                    multisampled: false 
                }, 
                count: None 
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 4, 
                visibility: wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), 
                count: None 
            },
        ],
    });

    // Helper function to create bind groups
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
    
    let mut composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Composite BG"),
        layout: &composite_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bloom_view_2) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let make_pipeline = |dev: &wgpu::Device, layout: &wgpu::PipelineLayout, shader: &wgpu::ShaderModule, entry: &str, format: wgpu::TextureFormat| {
        dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            vertex: wgpu::VertexState { module: shader, entry_point: "vs_main", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: entry,
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

    let raytrace_pipeline = make_pipeline(&device, &ray_pipeline_layout, &raytrace_shader, "fs_main", wgpu::TextureFormat::Rgba16Float);
    let h_blur_pipeline = make_pipeline(&device, &post_pipeline_layout, &bloom_shader, "fs_horizontal_blur", wgpu::TextureFormat::Rgba16Float);
    let v_blur_pipeline = make_pipeline(&device, &post_pipeline_layout, &bloom_shader, "fs_vertical_blur", wgpu::TextureFormat::Rgba16Float);
    let composite_pipeline = make_pipeline(&device, &comp_pipeline_layout, &bloom_shader, "fs_composite", config.format);

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
                ],
            });
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
            if last_frame_time.elapsed() >= target_frame_time { window.request_redraw(); }
        }
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
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

            // Pass 1: Render black hole raytracer to offscreen HDR scene texture
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

            // Pass 2: Horizontal Blur (extracts and expands highlights horizontally)
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

            // Pass 3: Vertical Blur (smooths vertically to create camera lens glare)
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

            // Pass 4: Composite / Final Screen Output
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

// Keep helper math functions from original main.rs (transpose_4x4, build_ray_uniforms, perspective, lookat, invert_4x4)
fn transpose_4x4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn build_ray_uniforms(yaw: f32, pitch: f32, distance: f32, aspect: f32, time: f32) -> RaytraceUniforms {
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
        _pad1: 0.0,
        _pad2: 0.0,
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