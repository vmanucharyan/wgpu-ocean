pub struct GenerateMipmapsPipeline {
  size: u32,
  textures_bind_group: wgpu::BindGroup,
  pipeline: wgpu::ComputePipeline,
}

impl GenerateMipmapsPipeline {
  pub fn init<'a>(
    device: &wgpu::Device,
    size: u32,
    displacement_texture: &'a wgpu::Texture,
    derivatives_texture: &'a wgpu::Texture,
  ) -> Self {
    let textures_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("generate mipmaps - texture bind group layout"),
        entries: &[
          // displacement
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
          // displacement_0
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::WriteOnly,
            },
            count: None,
          },
          // displacement_1
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::WriteOnly,
            },
            count: None,
          },
          // displacement_2
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
          // derivatives
          wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadOnly,
            },
            count: None,
          },
          // derivatives_1
          wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::WriteOnly,
            },
            count: None,
          },
          // derivatives_2
          wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::WriteOnly,
            },
            count: None,
          },
          // derivatives_3
          wgpu::BindGroupLayoutEntry {
            binding: 7,
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
      label: Some("Generate mipmaps pipeline layout"),
      bind_group_layouts: &[&textures_bind_group_layout],
      push_constant_ranges: &[wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..8,
      }],
    });

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Generate mipmaps shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/generate_mipmaps.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Generate mipmaps pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "main",
    });

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Generate mipmaps textures"),
      layout: &textures_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&displacement_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 0,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&displacement_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 1,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&displacement_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 2,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&displacement_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 3,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 4,
          resource: wgpu::BindingResource::TextureView(&derivatives_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 0,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 5,
          resource: wgpu::BindingResource::TextureView(&derivatives_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 1,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 6,
          resource: wgpu::BindingResource::TextureView(&derivatives_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 2,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 7,
          resource: wgpu::BindingResource::TextureView(&derivatives_texture.create_view(
            &wgpu::TextureViewDescriptor {
              base_mip_level: 3,
              mip_level_count: std::num::NonZeroU32::new(1),
              ..Default::default()
            },
          )),
        },
      ],
    });

    Self {
      size,
      textures_bind_group,
      pipeline,
    }
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: Some("Generate mipmaps"),
    });

    compute_pass.set_pipeline(&self.pipeline);
    compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
    compute_pass.dispatch(self.size / 16, self.size / 16, 1);
  }
}
