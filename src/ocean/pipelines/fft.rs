use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
  ping_pong: u32,
  step: u32,
  size: u32,
}

pub struct FFT {
  size: u32,
  buffer: wgpu::Texture,

  precompute_pipeline: wgpu::ComputePipeline,
  horizontal_step_pipeline: wgpu::ComputePipeline,
  vertical_step_pipeline: wgpu::ComputePipeline,
  scale_pipeline: wgpu::ComputePipeline,
  permute_pipeline: wgpu::ComputePipeline,
  swap_pipeline: wgpu::ComputePipeline,

  parameters_buffer: wgpu::Buffer,
  precompute_data_texture: wgpu::Texture,
  texture_bind_group: wgpu::BindGroup,
  parameters_bind_group: wgpu::BindGroup,
}

impl FFT {
  pub fn init(
    size: u32,
    device: &wgpu::Device,
    input: &wgpu::Texture,
    input_b: &wgpu::Texture,
  ) -> Self {
    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("FFT shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/fft.wgsl").into()),
    });

    let texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT texture bind group layout"),
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadWrite,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadWrite,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadWrite,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadWrite,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadWrite,
            },
            count: None,
          },
        ],
      });

    let parameters_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT Parameters bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::COMPUTE,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
      });

    let parameters = Parameters {
      ping_pong: 0,
      step: 0,
      size,
    };

    let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("FFT parameters buffer"),
      contents: bytemuck::cast_slice(&[parameters]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let parameters_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("FFT parameters"),
      layout: &parameters_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: parameters_buffer.as_entire_binding(),
      }],
    });

    let precompute_data_texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("FFT precompute buffer"),
      size: wgpu::Extent3d {
        width: (size as f64).log(2.0) as u32,
        height: size,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba32Float,
      usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    let buffer = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("FFT Buffer"),
      size: wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba32Float,
      usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    let buffer_b = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("FFT Buffer B"),
      size: wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba32Float,
      usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("FFT texture bind group"),
      layout: &texture_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&precompute_data_texture.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&input.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&buffer.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&input_b.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 4,
          resource: wgpu::BindingResource::TextureView(&buffer_b.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
      ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("FFT pipeline layout"),
      bind_group_layouts: &[&texture_bind_group_layout],
      push_constant_ranges: &[wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..12,
      }],
    });

    let precompute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("FFT - Calculate twiddle factors and input indices"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "calculat_twiddle_factors_and_input_indices",
    });

    let horizontal_step_pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT - Horizontal step"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "horizontal_step_inverse_fft",
      });

    let vertical_step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("FFT - Vertical step"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "vertical_step_inverse_fft",
    });

    let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("FFT - Scale"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "scale",
    });

    let permute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("FFT - Permute"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "permute",
    });

    let swap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("FFT - Swap"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "swap",
    });

    return Self {
      size,
      buffer,
      precompute_pipeline,
      parameters_buffer,
      precompute_data_texture,
      texture_bind_group,
      parameters_bind_group,
      horizontal_step_pipeline,
      vertical_step_pipeline,
      scale_pipeline,
      permute_pipeline,
      swap_pipeline,
    };
  }

  pub fn precompute(&self, encoder: &mut wgpu::CommandEncoder) {
    let log_size = (self.size as f64).log(2.0) as u32;

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: Some("FFT precompute"),
    });

    let parameters = Parameters {
      ping_pong: 0,
      size: self.size,
      step: 0,
    };

    compute_pass.set_pipeline(&self.precompute_pipeline);
    compute_pass.set_bind_group(0, &self.texture_bind_group, &[]);

    compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));

    compute_pass.dispatch(log_size, self.size / 2 / 8, 1);
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
    let log_size = (self.size as f64).log(2.0) as u32;
    let mut ping_pong = 0u32;

    let mut compute_pass =
      encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FFT") });

    compute_pass.set_bind_group(0, &self.texture_bind_group, &[]);

    compute_pass.set_pipeline(&self.horizontal_step_pipeline);
    for i in 0..log_size {
      ping_pong = if ping_pong == 0 { 1 } else { 0 };
      let parameters = Parameters {
        ping_pong,
        size: self.size,
        step: i,
      };

      compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
      compute_pass.dispatch(self.size / 16, self.size / 16, 1);
    }

    compute_pass.set_pipeline(&self.vertical_step_pipeline);
    for i in 0..log_size {
      ping_pong = if ping_pong == 0 { 1 } else { 0 };
      let parameters = Parameters {
        ping_pong,
        size: self.size,
        step: i,
      };

      compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
      compute_pass.dispatch(self.size / 16, self.size / 16, 1);
    }

    if ping_pong == 1 {
      compute_pass.set_pipeline(&self.swap_pipeline);
      compute_pass.set_push_constants(
        0,
        bytemuck::cast_slice(&[Parameters {
          ping_pong: 0,
          size: self.size,
          step: 0,
        }]),
      );
      compute_pass.dispatch(self.size / 16, self.size / 16, 1);
    }

    compute_pass.set_pipeline(&self.permute_pipeline);
    compute_pass.set_push_constants(
      0,
      bytemuck::cast_slice(&[Parameters {
        ping_pong: 0,
        size: self.size,
        step: 0,
      }]),
    );
    compute_pass.dispatch(self.size / 16, self.size / 16, 1);
  }
}
