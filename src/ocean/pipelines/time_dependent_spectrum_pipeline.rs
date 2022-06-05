use crate::ocean::utils::compute_work_group_count;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
  time: f32,
}

pub struct TimeDependentSpectrumPipeline {
  size: u32,
  textures_bind_group: wgpu::BindGroup,
  pipeline: wgpu::ComputePipeline,
}

impl TimeDependentSpectrumPipeline {
  pub fn init<'a>(
    size: u32,
    device: &wgpu::Device,

    h0_texture: &'a wgpu::Texture,
    waves_data_texture: &'a wgpu::Texture,
    amp_dx_dz_texture: &'a wgpu::Texture,
    amp_dyx_dyz_texture: &'a wgpu::Texture,
  ) -> Self {
    let texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texture bind group layout"),
        entries: &[
          // h0_texture
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
          // waves_data_texture
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
          // amp_dx_dz_texture
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
          // amp_dyx_dyz_texture
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

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Texture bind group"),
      layout: &texture_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&h0_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&waves_data_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&amp_dx_dz_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&amp_dyx_dyz_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
      ],
    });

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Time-dependent spectrum shader"),
      source: wgpu::ShaderSource::Wgsl(
        include_str!("./shaders/time_dependent_spectrum.wgsl").into(),
      ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Time-dependent spectrum pipeline layout"),
      bind_group_layouts: &[&texture_bind_group_layout],
      push_constant_ranges: &[wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..4,
      }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Time-dependent spectrum pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: "calculate_amplitudes",
    });

    Self {
      size,
      textures_bind_group,
      pipeline,
    }
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, time: f32) {
    let (dispatch_width, dispatch_height) =
      compute_work_group_count((self.size, self.size), (16, 16));

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: Some("Calculate time-dependent spectrum"),
    });

    let params = Params { time };

    compute_pass.set_pipeline(&self.pipeline);
    compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
    compute_pass.set_push_constants(0, bytemuck::cast_slice(&[params]));
    compute_pass.dispatch(dispatch_width, dispatch_height, 1);
  }
}
