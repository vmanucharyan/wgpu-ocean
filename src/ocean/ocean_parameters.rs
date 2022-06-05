#[derive(Clone, Copy)]
pub struct OceanSpectrumParameters {
  pub size: u32,
  pub gravity_acceleration: f32,
  pub length_scale: f32,
  pub depth: f32,
  pub cut_off_low: f32,
  pub cut_off_high: f32,

  pub scale: f32,
  pub wind_speed: f32,
  pub wind_direction: f32,
  pub fetch: f32,
  pub spread_blend: f32,
  pub swell: f32,
  pub peak_enhancement: f32,
  pub short_waves_fade: f32,
}

impl Default for OceanSpectrumParameters {
  fn default() -> OceanSpectrumParameters {
    OceanSpectrumParameters {
      size: 256u32,
      length_scale: 150.0,
      cut_off_low: 0.0001,
      cut_off_high: 9999.0,
      gravity_acceleration: 9.81,
      depth: 500.0,
      scale: 1.0,
      wind_speed: 0.5,
      wind_direction: 200.0,
      fetch: 100000.0,
      spread_blend: 1.0,
      swell: 0.7,
      peak_enhancement: 3.3,
      short_waves_fade: 0.01,
    }
  }
}
