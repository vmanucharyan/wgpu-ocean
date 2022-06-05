#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
  lambda: f32,
  delta_time: f32,
}

pub struct WavesDataMergePipeline {
  size: u32,
  lambda: f32,
  textures_bind_group: wgpu::BindGroup,
  pipeline: wgpu::ComputePipeline,
  blur_turbulence_pipeline: wgpu::ComputePipeline,
}

impl WavesDataMergePipeline {
  pub fn init<'a>(
    device: &wgpu::Device,
    size: u32,
    lambda: f32,
    amp_dx_dz_texture: &'a wgpu::Texture,
    amp_dyx_dyz_texture: &'a wgpu::Texture,
    displacement_texture: &'a wgpu::Texture,
    derivatives_texture: &'a wgpu::Texture,
  ) -> Self {
    let textures_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Waves data merge - texture bind group layout"),
        entries: &[
          // amp_dx_dz_texture
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadOnly,
            },
            count: None,
          },
          // amp_dyx_dyz_texture
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadOnly,
            },
            count: None,
          },
          // out_displacement
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
          // out_derivatives
          wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::WriteOnly,
            },
            count: None,
          },
        ],
      });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Waves data merge pipeline layout"),
      bind_group_layouts: &[&textures_bind_group_layout],
      push_constant_ranges: &[wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..8,
      }],
    });

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Waves data merge shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/waves_data_merge.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Waves data merge pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "merge",
    });

    let blur_turbulence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Blur turbulence pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "blur_turbulence",
    });

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Waves data merge - textures"),
      layout: &textures_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&amp_dx_dz_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&amp_dyx_dyz_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&displacement_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 0,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&derivatives_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 0,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },     
      ],
    });

    Self {
      size,
      lambda,
      textures_bind_group,
      pipeline,
      blur_turbulence_pipeline,
    }
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, dt: std::time::Duration) {
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: Some("Waves data merge"),
    });

    let parameters = Parameters {
      lambda: self.lambda,
      delta_time: dt.as_secs_f32(),
    };

    compute_pass.set_pipeline(&self.pipeline);
    compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
    compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
    compute_pass.dispatch(self.size / 16, self.size / 16, 1);
  }
}
