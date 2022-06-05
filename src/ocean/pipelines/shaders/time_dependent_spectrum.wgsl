[[group(0), binding(0)]]
var h0_texture: texture_storage_2d<rgba32float, read>;

[[group(0), binding(1)]]
var waves_data_texture: texture_storage_2d<rgba32float, read>;

[[group(0), binding(2)]]
var amp_dx_dz__dy_dxz_texture: texture_storage_2d<rgba32float, write>;

[[group(0), binding(3)]]
var amp_dyx_dyz__dxx_dzz_texture: texture_storage_2d<rgba32float, write>;

struct Params {
    time: f32;
};

var<push_constant> params: Params;

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

[[stage(compute), workgroup_size(16, 16)]]
fn calculate_amplitudes(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);

    let wave = textureLoad(waves_data_texture, coords);
    let h0 = textureLoad(h0_texture, coords);

    let phase = wave.w * params.time;
    let exponent = vec2<f32>(cos(phase), sin(phase));

    let h = complex_mult(h0.xy, exponent) + complex_mult(h0.zw, vec2<f32>(exponent.x, -exponent.y));
    let ih = vec2<f32>(-h.y, h.x);

    let displacement_x = ih * wave.x * wave.y;
    let displacement_y = h;
    let displacement_z = ih * wave.z * wave.y;

    let displacement_x_dx = -h * wave.x * wave.x * wave.y;
    let displacement_y_dx = ih * wave.x;
    let displacement_z_dx = -h * wave.x * wave.z * wave.y;

    let displacement_y_dz = ih * wave.z;
    let displacement_z_dz = -h * wave.z * wave.z * wave.y;

    let dx_dz = vec2<f32>(
        displacement_x.x - displacement_z.y,
        displacement_x.y + displacement_z.x
    );

    let dy_dxz = vec2<f32>(
        displacement_y.x - displacement_z_dx.y,
        displacement_y.y + displacement_z_dx.x,
    );

    let dyx_dyz = vec2<f32>(
        displacement_y_dx.x - displacement_y_dz.y,
        displacement_y_dx.y + displacement_y_dz.x,
    );

    let dxx_dzz = vec2<f32>(
        displacement_x_dx.x - displacement_y_dz.y,
        displacement_x_dx.y + displacement_z_dz.x,
    );

    textureStore(amp_dx_dz__dy_dxz_texture, coords, vec4<f32>(dx_dz, dy_dxz));
    textureStore(amp_dyx_dyz__dxx_dzz_texture, coords, vec4<f32>(dyx_dyz, dxx_dzz));
}
