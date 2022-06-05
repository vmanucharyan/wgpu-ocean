use image::GenericImageView;
use wgpu::util::DeviceExt;

use crate::camera;
use crate::generate_plane::generate_plane;
use crate::ocean::{OceanCascade, OceanCascadeParameters};
use crate::vertex::Vertex;

const SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
  view_position: [f32; 4],
  view_proj: [[f32; 4]; 4],
  view: [[f32; 4]; 4],
  proj: [[f32; 4]; 4],
  inverse_view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
  fn new() -> Self {
    use cgmath::SquareMatrix;
    Self {
      view_position: cgmath::Vector4::unit_x().into(),
      view_proj: cgmath::Matrix4::identity().into(),
      view: cgmath::Matrix4::identity().into(),
      proj: cgmath::Matrix4::identity().into(),
      inverse_view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
    use cgmath::prelude::*;
    use cgmath::Matrix;

    self.view_position = camera.position.to_homogeneous().into();
    self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    self.view = (camera.calc_matrix()).transpose().into();
    self.proj = (projection.calc_matrix()).transpose().into();
    self.inverse_view_proj = (projection.calc_matrix() * camera.calc_matrix())
      .inverse_transform()
      .unwrap_or(cgmath::Matrix4::identity())
      .into();
  }
}

pub struct Renderer {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  config: wgpu::SurfaceConfiguration,
  pub size: winit::dpi::PhysicalSize<u32>,
  multisampled_framebuffer: wgpu::TextureView,

  render_pipeline: wgpu::RenderPipeline,
  vertex_buffer: wgpu::Buffer,
  index_buffer: wgpu::Buffer,
  num_indices: u32,

  // camera
  camera: camera::Camera,
  projection: camera::Projection,
  pub camera_controller: camera::CameraController,
  camera_uniform: CameraUniform,
  camera_buffer: wgpu::Buffer,
  camera_bind_group: wgpu::BindGroup,

  heightmap_bind_group: wgpu::BindGroup,
  ocean_surface: OceanCascade,
  ocean_params: OceanCascadeParameters,
  ocean_initialized: bool,
  pub mouse_pressed: bool,
}

impl Renderer {
  pub async fn new<'b>(window: &'b winit::window::Window) -> Renderer {
    let size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(window) };
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          features: wgpu::Features::POLYGON_MODE_LINE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::PUSH_CONSTANTS,
          limits: wgpu::Limits {
            max_push_constant_size: 256,
            ..wgpu::Limits::default()
          },
          label: None,
        },
        None, // Trace path
      )
      .await
      .unwrap();

    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format: surface.get_preferred_format(&adapter).unwrap(),
      width: size.width,
      height: size.height,
      present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    let multisampled_framebuffer =
      Self::create_multisampled_framebuffer(&device, &config, SAMPLE_COUNT);

    // Render pipeline (shaders)
    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("ocean_shader.wgsl").into()),
    });

    let camera = camera::Camera::new((0.0, 40.0, 0.0), cgmath::Deg(160.0), cgmath::Deg(-20.0));
    let projection =
      camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 1.0, 5000.0);
    let camera_controller = camera::CameraController::new(20.0, 1.0);

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera, &projection);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("camera_bind_group_layout"),
      });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("camera_bind_group"),
    });

    // foam texture
    let foam_img = image::load_from_memory(include_bytes!("./assets/foam.jpg")).unwrap();
    let foam_bytes = foam_img.to_rgba8().to_vec();
    let (foam_width, foam_height) = foam_img.dimensions();

    let foam_texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Foam texture"),
      size: wgpu::Extent3d {
        width: foam_width,
        height: foam_height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 6,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba8UnormSrgb,
      usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
      wgpu::ImageCopyTexture {
        texture: &foam_texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      &foam_bytes,
      wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: std::num::NonZeroU32::new(4 * foam_width),
        rows_per_image: std::num::NonZeroU32::new(foam_height),
      },
      wgpu::Extent3d {
        width: foam_width,
        height: foam_height,
        depth_or_array_layers: 1,
      },
    );

    for i in 1..=5 {
      let downsample_factor = u32::pow(2, i);
      let width = foam_width / downsample_factor;
      let height = foam_height / downsample_factor;

      let mip_bytes = foam_img.thumbnail(width, height).to_rgba8().to_vec();

      queue.write_texture(
        wgpu::ImageCopyTexture {
          texture: &foam_texture,
          mip_level: i,
          origin: wgpu::Origin3d::ZERO,
          aspect: wgpu::TextureAspect::All,
        },
        &mip_bytes,
        wgpu::ImageDataLayout {
          offset: 0,
          bytes_per_row: std::num::NonZeroU32::new(4 * width),
          rows_per_image: std::num::NonZeroU32::new(height),
        },
        wgpu::Extent3d {
          width: width,
          height: height,
          depth_or_array_layers: 1,
        },
      );
    }

    // ocean
    let ocean_size = 256;
    let ocean_initialized = false;

    let ocean_params = OceanCascadeParameters {
      size: ocean_size,
      wind_speed: 10.0,
      wind_direction: -20.0,
      swell: 0.4,
    };

    let derivatives_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      address_mode_u: wgpu::AddressMode::Repeat,
      address_mode_v: wgpu::AddressMode::Repeat,
      address_mode_w: wgpu::AddressMode::Repeat,
      mag_filter: wgpu::FilterMode::Linear,
      min_filter: wgpu::FilterMode::Linear,
      mipmap_filter: wgpu::FilterMode::Linear,
      anisotropy_clamp: std::num::NonZeroU8::new(8),
      ..Default::default()
    });

    let ocean_surface = OceanCascade::new(&device, ocean_size, ocean_params);
    let texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
          // displacement 0
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // derivatives 0
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // derivatives sampler
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
          },
          // displacement 1
          wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // derivatives 1
          wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // displacement 2
          wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // derivatives 2
          wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
          // foam
          wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              multisampled: false,
            },
            count: None,
          },
        ],
        label: Some("texture_bind_group_layout"),
      });

    let heightmap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &texture_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_0.displacement_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_0.derivatives_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::Sampler(&derivatives_sampler),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_1.displacement_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 4,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_1.derivatives_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 5,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_2.displacement_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 6,
          resource: wgpu::BindingResource::TextureView(
            &ocean_surface.cascade_2.derivatives_texture().create_view(
              &wgpu::TextureViewDescriptor {
                ..Default::default()
              },
            ),
          ),
        },
        wgpu::BindGroupEntry {
          binding: 7,
          resource: wgpu::BindingResource::TextureView(&foam_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
      ],
      label: Some("Texture bind group"),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Render Pipeline Layout"),
      bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
      push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: "vs_main",
        buffers: &[Vertex::desc()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: "fs_main",
        targets: &[wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        }],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::default(),
        cull_mode: Some(wgpu::Face::Front),
        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
        polygon_mode: wgpu::PolygonMode::Fill,
        // Requires Features::DEPTH_CLIP_CONTROL
        unclipped_depth: false,
        // Requires Features::CONSERVATIVE_RASTERIZATION
        conservative: false,
      },
      depth_stencil: None,
      multisample: wgpu::MultisampleState {
        count: SAMPLE_COUNT,
        mask: !0,
        alpha_to_coverage_enabled: false,
      },
      multiview: None,
    });

    let (plane_mesh_vertices, plane_mesh_indices) = generate_plane(2.3, 512);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Vertex Buffer"),
      contents: bytemuck::cast_slice(&plane_mesh_vertices),
      usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Index buffer"),
      contents: bytemuck::cast_slice(&plane_mesh_indices),
      usage: wgpu::BufferUsages::INDEX,
    });

    let num_indices = plane_mesh_indices.len() as u32;

    Renderer {
      surface,
      device,
      queue,
      config,
      size,

      render_pipeline,
      vertex_buffer,
      index_buffer,
      num_indices,
      camera,
      projection,
      camera_controller,
      camera_uniform,
      camera_bind_group,
      camera_buffer,

      heightmap_bind_group,
      ocean_params,
      ocean_surface,
      ocean_initialized,

      mouse_pressed: false,
      multisampled_framebuffer,
    }
  }

  pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.projection.resize(new_size.width, new_size.height);
      self.surface.configure(&self.device, &self.config);
      self.multisampled_framebuffer =
        Self::create_multisampled_framebuffer(&self.device, &self.config, SAMPLE_COUNT);
    }
  }

  pub fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
    use winit::{
      event::ElementState, event::KeyboardInput, event::MouseButton, event::WindowEvent,
    };

    match event {
      WindowEvent::KeyboardInput {
        input:
          KeyboardInput {
            virtual_keycode: Some(key),
            state,
            ..
          },
        ..
      } => self.camera_controller.process_keyboard(*key, *state),
      WindowEvent::MouseWheel { delta, .. } => {
        self.camera_controller.process_scroll(delta);
        true
      }
      WindowEvent::MouseInput {
        button: MouseButton::Left,
        state,
        ..
      } => {
        self.mouse_pressed = *state == ElementState::Pressed;
        true
      }
      _ => false,
    }
  }

  pub fn update(&mut self, dt: std::time::Duration) {
    self.camera_controller.update_camera(&mut self.camera, dt);
    self
      .camera_uniform
      .update_view_proj(&self.camera, &self.projection);
    self.queue.write_buffer(
      &self.camera_buffer,
      0,
      bytemuck::cast_slice(&[self.camera_uniform]),
    );
  }

  pub fn render(&mut self, time: f32, dt: std::time::Duration) -> Result<(), wgpu::SurfaceError> {
    let output = self.surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
      ..Default::default()
    });

    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
      });

    if !self.ocean_initialized {
      self.ocean_surface.init(&mut encoder, &self.queue);
      self.ocean_initialized = true;
    }

    {
      self
        .ocean_surface
        .dispatch(&mut encoder, &self.queue, time, dt);
    }

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[wgpu::RenderPassColorAttachment {
          view: &self.multisampled_framebuffer,
          resolve_target: Some(&view),
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
              r: 0.5,
              g: 0.5,
              b: 0.5,
              a: 1.0,
            }),
            store: true,
          },
        }],
        depth_stencil_attachment: None,
      });

      render_pass.set_pipeline(&self.render_pipeline);

      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.heightmap_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
      render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

      render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }

    // submit will accept anything that implements IntoIter
    self.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    Ok(())
  }

  fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
  ) -> wgpu::TextureView {
    let multisampled_texture_extent = wgpu::Extent3d {
      width: config.width,
      height: config.height,
      depth_or_array_layers: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
      size: multisampled_texture_extent,
      mip_level_count: 1,
      sample_count,
      dimension: wgpu::TextureDimension::D2,
      format: config.format,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      label: None,
    };

    device
      .create_texture(multisampled_frame_descriptor)
      .create_view(&wgpu::TextureViewDescriptor::default())
  }
}
