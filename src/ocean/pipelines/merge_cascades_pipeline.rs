use std::rc::Rc;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
  lambda: f32,
  delta_time: f32,
}

pub struct MergeCascadesPipeline {
  size: u32,

  pub cascade_0: Rc<wgpu::Texture>,
  pub cascade_1: Rc<wgpu::Texture>,
  pub cascade_2: Rc<wgpu::Texture>,

  pub merged_displacement: wgpu::Texture,

  textures_bind_group: wgpu::BindGroup,
  pipeline: wgpu::ComputePipeline,
}

impl MergeCascadesPipeline {
  pub fn init(
    device: &wgpu::Device,
    size: u32,
    cascade_0: Rc<wgpu::Texture>,
    cascade_1: Rc<wgpu::Texture>,
    cascade_2: Rc<wgpu::Texture>,
  ) -> Self {
    let textures_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Waves data merge - texture bind group layout"),
        entries: &[
          // cascade_0
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
          // cascade_1
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
          // cascade_2
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadOnly,
            },
            count: None,
          },
          // merged_displacement
          wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
              view_dimension: wgpu::TextureViewDimension::D2,
              format: wgpu::TextureFormat::Rgba32Float,
              access: wgpu::StorageTextureAccess::ReadOnly,
            },
            count: None,
          },
        ],
      });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Cascade merge pipeline layout"),
      bind_group_layouts: &[&textures_bind_group_layout],
      push_constant_ranges: &[wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..8,
      }],
    });

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Merge cascades shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/waves_data_merge.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Merge cascades pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "merge",
    });

    let texture_size = wgpu::Extent3d {
      width: size,
      height: size,
      depth_or_array_layers: 1,
    };

    let merged_displacement = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Cascade 0"),
      size: texture_size,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba32Float,
      usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Merge cascades bind group"),
      layout: &textures_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&cascade_0.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&cascade_1.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&cascade_2.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&merged_displacement.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
      ],
    });

    Self {
      size,
      merged_displacement,
      cascade_0,
      cascade_1,
      cascade_2,
      textures_bind_group,
      pipeline,
    }
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: Some("Waves data merge"),
    });

    compute_pass.set_pipeline(&self.pipeline);
    compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
    compute_pass.dispatch(self.size / 16, self.size / 16, 1);
  }
}
