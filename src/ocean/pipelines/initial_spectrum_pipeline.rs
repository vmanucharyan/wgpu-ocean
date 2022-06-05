use crate::ocean::ocean_parameters::OceanSpectrumParameters;
use crate::ocean::utils::clamp;

const WG_COUNT: u32 = 16;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
  size: u32,
  length_scale: f32,
  cut_off_low: f32,
  cut_off_high: f32,
  gravity_acceleration: f32,
  depth: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectrumParamers {
  scale: f32,
  angle: f32,
  spread_blend: f32,
  swell: f32,
  alpha: f32,
  peak_omega: f32,
  gamma: f32,
  short_waves_fade: f32,
}

impl SpectrumParamers {
  fn from_ocean_parameters(o: OceanSpectrumParameters) -> Self {
    Self {
      scale: o.scale,
      angle: o.wind_direction / 180.0 * std::f32::consts::PI,
      spread_blend: o.spread_blend,
      swell: clamp(o.swell, 0.01, 1.0),
      alpha: Self::jonswap_alpha(9.81, o.fetch, o.wind_speed),
      peak_omega: Self::jonswap_peak_frequency(9.81, o.fetch, o.wind_speed),
      gamma: o.peak_enhancement,
      short_waves_fade: o.short_waves_fade,
    }
  }

  fn jonswap_alpha(g: f32, fetch: f32, wind_speed: f32) -> f32 {
    0.076 * f32::powf(g * fetch / wind_speed / wind_speed, -0.22)
  }

  fn jonswap_peak_frequency(g: f32, fetch: f32, wind_speed: f32) -> f32 {
    22.0 * f32::powf(wind_speed * fetch / g / g, -0.33)
  }
}

pub struct InitialSpectrumPipeline {
  size: u32,
  textures_bind_group: wgpu::BindGroup,
  calculate_initial_spectrum_pipeline: wgpu::ComputePipeline,
  calculate_conjugated_spectrum_pipeline: wgpu::ComputePipeline,

  texture_size: wgpu::Extent3d,
  noise_texture: wgpu::Texture,
  parameters_buffer: wgpu::Buffer,
  parameters_bind_group: wgpu::BindGroup,

  noise_data: Vec<f32>,
}

impl InitialSpectrumPipeline {
  pub fn init(
    size: u32,
    wave_params: OceanSpectrumParameters,
    device: &wgpu::Device,
    h0k_texture: &wgpu::Texture,
    waves_data_texture: &wgpu::Texture,
    h0_texture: &wgpu::Texture,
  ) -> Self {
    use wgpu::util::DeviceExt;

    let texture_size = wgpu::Extent3d {
      width: size,
      height: size,
      depth_or_array_layers: 1,
    };

    let parameters = Parameters {
      size: wave_params.size,
      length_scale: wave_params.length_scale,
      cut_off_low: wave_params.cut_off_low,
      cut_off_high: wave_params.cut_off_high,
      gravity_acceleration: wave_params.gravity_acceleration,
      depth: wave_params.depth,
    };

    let spectrum_parameters = SpectrumParamers::from_ocean_parameters(wave_params);

    let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Parameters Buffer"),
      contents: bytemuck::cast_slice(&[parameters]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let spectrum_parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Spectrum parameters buffer"),
      contents: bytemuck::cast_slice(&[spectrum_parameters]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let parameters_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IS - Parameters bind group layout"),
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Uniform,
              has_dynamic_offset: false,
              min_binding_size: None,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Uniform,
              has_dynamic_offset: false,
              min_binding_size: None,
            },
            count: None,
          },
        ],
      });

    let parameters_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Parameters bind group"),
      layout: &parameters_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: parameters_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: spectrum_parameters_buffer.as_entire_binding(),
        },
      ],
    });

    let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Noise texture"),
      size: texture_size,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba32Float,
      usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    let texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IS - Texture bind group layout"),
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
              view_dimension: wgpu::TextureViewDimension::D2,
              sample_type: wgpu::TextureSampleType::Float { filterable: false },
              multisampled: false,
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
        ],
      });

    let noise_data = generate_noise_data(size as usize);

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
      label: Some("Initial spectrum shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/initial_spectrum.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Initial spectrum pipeline layout"),
      bind_group_layouts: &[&texture_bind_group_layout, &parameters_bind_group_layout],
      push_constant_ranges: &[],
    });

    let calculate_initial_spectrum_pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Initial spectrum pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "calculate_initial_spectrum",
      });

    let calculate_conjugated_spectrum_pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Calculate conjugated spectrum pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "calculate_conjugated_spectrum",
      });

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("IS - Texture bind group"),
      layout: &texture_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&noise_texture.create_view(
            &wgpu::TextureViewDescriptor {
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(&h0_texture.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::TextureView(&waves_data_texture.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
        wgpu::BindGroupEntry {
          binding: 3,
          resource: wgpu::BindingResource::TextureView(&h0k_texture.create_view(
            &wgpu::TextureViewDescriptor {
              format: Some(wgpu::TextureFormat::Rgba32Float),
              ..Default::default()
            },
          )),
        },
      ],
    });

    Self {
      size,
      noise_data,
      texture_size,
      noise_texture,
      textures_bind_group,
      calculate_initial_spectrum_pipeline,
      calculate_conjugated_spectrum_pipeline,
      parameters_buffer,
      parameters_bind_group,
    }
  }

  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
    queue.write_texture(
      wgpu::ImageCopyTexture {
        texture: &self.noise_texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      &bytemuck::cast_slice(&self.noise_data),
      wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: std::num::NonZeroU32::new(16 * self.size),
        rows_per_image: std::num::NonZeroU32::new(self.size),
      },
      self.texture_size,
    );

    {
      let (dispatch_width, dispatch_height) = compute_work_group_count(
        (self.texture_size.width, self.texture_size.height),
        (WG_COUNT, WG_COUNT),
      );
      let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Calculate Initial Spectrum"),
      });
      compute_pass.set_pipeline(&self.calculate_initial_spectrum_pipeline);
      compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
      compute_pass.set_bind_group(1, &self.parameters_bind_group, &[]);
      compute_pass.dispatch(dispatch_width, dispatch_height, 1);
    }

    {
      let (dispatch_width, dispatch_height) = compute_work_group_count(
        (self.texture_size.width, self.texture_size.height),
        (WG_COUNT, WG_COUNT),
      );

      let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Calculate Conjugated Spectrum"),
      });

      compute_pass.set_pipeline(&self.calculate_conjugated_spectrum_pipeline);
      compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
      compute_pass.set_bind_group(1, &self.parameters_bind_group, &[]);
      compute_pass.dispatch(dispatch_width, dispatch_height, 1);
    }
  }
}

fn generate_noise_data(size: usize) -> Vec<f32> {
  use rand::prelude::*;

  let mut rng = rand::thread_rng();
  let mut buf: Vec<f32> = vec![0 as f32; 4 * size * size];
  for i in 0..4 * size * size {
    buf[i] = rng.gen();
  }

  return buf;
}

fn compute_work_group_count(
  (width, height): (u32, u32),
  (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
  let x = (width + workgroup_width - 1) / workgroup_width;
  let y = (height + workgroup_height - 1) / workgroup_height;

  return (x, y);
}
