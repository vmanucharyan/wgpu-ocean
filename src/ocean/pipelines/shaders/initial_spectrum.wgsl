// https://github.com/gasgiant/FFT-Ocean/blob/main/Assets/ComputeShaders/InitialSpectrum.compute

struct Parameters {
  size: u32;
  length_scale: f32;
  cut_off_low: f32;
  cut_off_high: f32;
  gravity_acceleration: f32;
  depth: f32;
};

struct SpectrumParamers {
  scale: f32;
  angle: f32;
  spread_blend: f32;
  swell: f32;
  alpha: f32;
  peak_omega: f32;
  gamma: f32;
  short_waves_fade: f32;
};

// initial spectrum
[[group(0), binding(0)]]
var noise: texture_2d<f32>;

[[group(0), binding(1)]]
var h0_texture: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(2)]]
var waves_data_texture: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(3)]]
var h0k_texture: texture_storage_2d<rgba32float, read_write>;

[[group(1), binding(0)]]
var<uniform> parameters: Parameters;

[[group(1), binding(1)]]
var<uniform> spectrum_0: SpectrumParamers;

let PI: f32 = 3.14159265358979323846264338;

fn frequency(k: f32, g: f32, depth: f32) -> f32 {
    return sqrt(g * k * tanh(min(k * depth, 20.0)));
}

fn frequency_derivative(k: f32, g: f32, depth: f32) -> f32 {
    let th = tanh(min(k * depth, 20.0));
    let ch = cosh(k * depth);

    return g * (depth * k / ch / ch + th) / frequency(k, g, depth) / 2.0;
}

fn normalization_factor(s: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;

    if (s < 5.0) {
        return -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163;
    } else {
        return -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.90e-02 * s + 3.93e-01;
    }
}

fn donelan_banner_beta(x: f32) -> f32 {
    if (x < 0.95) {
        return 2.61 * pow(abs(x), 1.3);
    }

    if (x < 1.6) {
        return 2.28 * pow(abs(x), -1.3);
    }

    let p = -0.4 + 0.8393 * exp(-0.567 * log(x * x));

    return pow(10.0, p);
}

fn donelan_banner(theta: f32, omega: f32, peak_omega: f32) -> f32 {
    let beta = donelan_banner_beta(omega / peak_omega);
    let sech = 1.0 / cosh(beta * theta);

    return beta / 2.0 / tanh(beta * 3.1416) * sech * sech;
}

fn cosine_2s(theta: f32, s: f32) -> f32 {
    return normalization_factor(s) * pow(abs(cos(0.5 * theta)), 2.0 * s);
}

fn spread_power(omega: f32, peak_omega: f32) -> f32 {
    if (omega > peak_omega) {
        return 9.77 * pow(abs(omega / peak_omega), -2.5);
    } else {
        return 6.97 * pow(abs(omega / peak_omega), 5.0);
    }
}

fn direction_spectrum(theta: f32, omega: f32, pars: SpectrumParamers) -> f32 {
    let s = spread_power(omega, pars.peak_omega) + 16.0 * tanh(min(omega / pars.peak_omega, 20.0)) * pars.swell * pars.swell;

    return mix(2.0 / 3.1415 * cos(theta) * cos(theta), cosine_2s(theta - pars.angle, s), pars.spread_blend);
}

fn tma_correction(omega: f32, g: f32, depth: f32) -> f32 {
    let omega_h = omega * sqrt(depth / g);

    if (omega_h <= 1.0) {
        return 0.5 * omega_h * omega_h;
    }

    if (omega_h < 2.0) {
        return 1.0 - 0.5 * (2.0 - omega_h) * (2.0 - omega_h);
    }

    return 1.0;
}

fn jonswap(omega: f32, g: f32, depth: f32, pars: SpectrumParamers) -> f32 {
    var sigma: f32 = 0.0;

    if (omega <= pars.peak_omega) {
        sigma = 0.07;
    } else {
        sigma = 0.09;
    };

    let r = exp(-(omega - pars.peak_omega) * (omega - pars.peak_omega) 
        / 2.0 / sigma / sigma / pars.peak_omega / pars.peak_omega);

    let one_over_omega = 1.0 / omega;
    let peak_omega_over_omega = pars.peak_omega / omega;

    return pars.scale * tma_correction(omega, g, depth) * pars.alpha * g * g 
        * one_over_omega * one_over_omega * one_over_omega * one_over_omega * one_over_omega 
        * exp(-1.25 * peak_omega_over_omega * peak_omega_over_omega * peak_omega_over_omega * peak_omega_over_omega) 
        * pow(abs(pars.gamma), r);
}

fn short_waves_fade(k_length: f32, pars: SpectrumParamers) -> f32 {
    return exp(-pars.short_waves_fade * pars.short_waves_fade * k_length * k_length);
}

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
 
[[stage(compute), workgroup_size(16, 16)]]
fn calculate_initial_spectrum(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let delta_k = 2.0 * PI / parameters.length_scale;
    let nx = i32(id.x) - i32(parameters.size / 2u);
    let nz = i32(id.y) - i32(parameters.size / 2u);

    let k = vec2<f32>(f32(nx), f32(nz)) * delta_k;
    let k_length = length(k);
    let coords = vec2<i32>(id.xy);

    if (k_length > parameters.cut_off_high || k_length < parameters.cut_off_low) {
        textureStore(h0k_texture, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(waves_data_texture, coords, vec4<f32>(k.x, 1.0, k.y, 0.0));
        return;
    }

    let k_angle = atan2(k.y, k.x);
    let omega = frequency(k_length, parameters.gravity_acceleration, parameters.depth);
    let w = vec4<f32>(k.x, 1.0 / k_length, k.y, omega);

    textureStore(waves_data_texture, coords, w);

    let d_omega_dk = frequency_derivative(k_length, parameters.gravity_acceleration, parameters.depth);

    let j0 = jonswap(
        omega,
        parameters.gravity_acceleration,
        parameters.depth,
        spectrum_0
    );

    let d0 = direction_spectrum(
        k_angle,
        omega,
        spectrum_0
    );

    let s0 = short_waves_fade(k_length, spectrum_0);
    let spectrum_val = j0 * d0 * s0;
    let n = textureLoad(noise, coords, 0).xy;
    let h0k_val = n * sqrt(2.0 * spectrum_val * abs(d_omega_dk) / k_length * delta_k * delta_k);

    textureStore(h0k_texture, coords, vec4<f32>(h0k_val, 0.0, 1.0));
}

[[stage(compute), workgroup_size(16, 16)]]
fn calculate_conjugated_spectrum(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);
    let h0k = textureLoad(h0k_texture, coords).xy;
    let h0minusk_coords = vec2<i32>(
        i32((parameters.size - id.x) % parameters.size),
        i32((parameters.size - id.y) % parameters.size)
    );
    let h0minusk = textureLoad(h0k_texture, h0minusk_coords).xy;

    textureStore(h0_texture, coords, vec4<f32>(h0k.x, h0k.y, h0minusk.x, -h0minusk.y));
}
