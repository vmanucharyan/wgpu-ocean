use crate::ocean::OceanSpectrumParameters;
use crate::ocean::OceanSurface;

pub struct OceanCascade {
  pub cascade_0: OceanSurface,
  pub cascade_1: OceanSurface,
  pub cascade_2: OceanSurface,
}

#[derive(Clone, Copy)]
pub struct OceanCascadeParameters {
  pub size: u32,
  pub wind_speed: f32,
  pub wind_direction: f32,
  pub swell: f32,
}

impl OceanCascade {
  pub fn new(device: &wgpu::Device, size: u32, params: OceanCascadeParameters) -> Self {
    let surface_params = OceanSpectrumParameters {
      size: params.size,
      wind_speed: params.wind_speed,
      wind_direction: params.wind_direction,
      swell: params.swell,
      ..Default::default()
    };

    let length_scale_0 = 500.0;
    let length_scale_1 = 85.0;
    let length_scale_2 = 10.0;

    let boundary_1 = 2.0 * std::f32::consts::PI / length_scale_1 * 6.0;
    let boundary_2 = 2.0 * std::f32::consts::PI / length_scale_2 * 6.0;

    let params_0 = OceanSpectrumParameters {
      cut_off_low: 0.0001,
      cut_off_high: boundary_1,
      length_scale: length_scale_0,
      ..surface_params
    };

    let params_1 = OceanSpectrumParameters {
      cut_off_low: boundary_1,
      cut_off_high: boundary_2,
      length_scale: length_scale_1,
      ..surface_params
    };

    let params_2 = OceanSpectrumParameters {
      cut_off_low: boundary_2,
      cut_off_high: 9999.0,
      length_scale: length_scale_2,
      ..surface_params
    };

    let cascade_0 = OceanSurface::new(device, size, params_0);
    let cascade_1 = OceanSurface::new(device, size, params_1);
    let cascade_2 = OceanSurface::new(device, size, params_2);

    Self {
      cascade_0,
      cascade_1,
      cascade_2,
    }
  }

  pub fn init(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
    self.cascade_0.init(encoder, queue);
    self.cascade_1.init(encoder, queue);
    self.cascade_2.init(encoder, queue);
  }

  pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, time: f32, dt: std::time::Duration) {
    self.cascade_0.dispatch(encoder, queue, time, dt);
    self.cascade_1.dispatch(encoder, queue, time, dt);
    self.cascade_2.dispatch(encoder, queue, time, dt);
  }
}
