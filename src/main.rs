use std::sync::Arc;
use std::time::{Duration, Instant};
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

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Curved Geodesic Black Hole Raytracer")
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

    // Camera orbit parameters
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

    // Load WGSL Raytracing Shader
    let raytrace_shader = device.create_shader_module(wgpu::include_wgsl!("raytrace.wgsl"));

    // Pipeline Layout
    let ray_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Raytrace Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let ray_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raytrace Bind Group"),
        layout: &ray_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ray_uniform_buffer.as_entire_binding(),
        }],
    });

    let ray_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Raytrace Pipeline Layout"),
        bind_group_layouts: &[&ray_bind_group_layout],
        push_constant_ranges: &[],
    });

    let raytrace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Raytrace Pipeline"),
        layout: Some(&ray_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &raytrace_shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &raytrace_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let target_frame_time = Duration::from_secs_f32(1.0 / 60.0);
    let mut last_frame_time = Instant::now();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => target.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                config.width = new_size.width.max(1);
                config.height = new_size.height.max(1);
                surface.configure(&device, &config);
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { button: MouseButton::Left, state, .. },
                ..
            } => {
                is_dragging = state == ElementState::Pressed;
                if !is_dragging {
                    last_mouse_pos = None;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                if is_dragging {
                    if let Some((last_x, last_y)) = last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        yaw += dx * 0.005;
                        pitch = (pitch + dy * 0.005).clamp(-1.4, 1.4);
                    }
                    last_mouse_pos = Some((position.x, position.y));
                }
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.05,
                };
                if scroll > 0.0 {
                    camera_distance *= 0.9;
                } else if scroll < 0.0 {
                    camera_distance *= 1.1;
                }
                camera_distance = camera_distance.clamp(3.0, 80.0);
            }
            Event::AboutToWait => {
                if last_frame_time.elapsed() >= target_frame_time {
                    window.request_redraw();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                last_frame_time = Instant::now();
                let time = start_time.elapsed().as_secs_f32();
                let aspect = config.width as f32 / config.height as f32;

                let ray_u = build_ray_uniforms(yaw, pitch, camera_distance, aspect, time);
                queue.write_buffer(&ray_uniform_buffer, 0, bytemuck::bytes_of(&ray_u));

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Outdated) => return,
                    Err(e) => panic!("Surface error: {:?}", e),
                };

                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

                // Pure GR Raytracing Render Pass
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Main Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.005, g: 0.008, b: 0.02, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    rpass.set_pipeline(&raytrace_pipeline);
                    rpass.set_bind_group(0, &ray_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        })
        .unwrap();
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
        if len > 0.0001 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 0.0]
        }
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
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