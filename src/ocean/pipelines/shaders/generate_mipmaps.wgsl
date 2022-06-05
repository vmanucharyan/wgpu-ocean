[[group(0), binding(0)]]
var t_displacement: texture_storage_2d<rgba32float, read>;

[[group(0), binding(1)]]
var t_displacement_1: texture_storage_2d<rgba32float, write>;

[[group(0), binding(2)]]
var t_displacement_2: texture_storage_2d<rgba32float, write>;

[[group(0), binding(3)]]
var t_displacement_3: texture_storage_2d<rgba32float, write>;

[[group(0), binding(4)]]
var t_derivatives: texture_storage_2d<rgba32float, read>;

[[group(0), binding(5)]]
var t_derivatives_1: texture_storage_2d<rgba32float, write>;

[[group(0), binding(6)]]
var t_derivatives_2: texture_storage_2d<rgba32float, write>;

[[group(0), binding(7)]]
var t_derivatives_3: texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);
    let displacement = textureLoad(t_displacement, coords);

    textureStore(
        t_displacement_1,
        coords / 2,
        displacement,
    );

    textureStore(
        t_displacement_2,
        coords / 4,
        displacement,
    );

    textureStore(
        t_displacement_3,
        coords / 8,
        displacement,
    );

    let derivatives = textureLoad(t_derivatives, coords);

    textureStore(
        t_derivatives_1,
        coords / 2,
        derivatives,
    );

    textureStore(
        t_derivatives_2,
        coords / 4,
        derivatives,
    );

    textureStore(
        t_derivatives_3,
        coords / 8,
        derivatives,
    );
}
