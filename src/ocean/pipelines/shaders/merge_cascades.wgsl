[[group(0), binding(0)]]
var cascade_0: texture_storage_2d<rgba32float, read>;

[[group(0), binding(1)]]
var cascade_1: texture_storage_2d<rgba32float, read>;

[[group(0), binding(2)]]
var cascade_2: texture_storage_2d<rgba32float, read>;

[[group(0), binding(3)]]
var out: texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(16, 16)]]
fn merge_cascades(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let c0 = textureLoad(cascade_0, vec2<i32>(id.xy));
    let c1 = textureLoad(cascade_1, vec2<i32>(id.xy));
    let c2 = textureLoad(cascade_2, vec2<i32>(id.xy));

    textureStore(out, vec2<i32>(id.xy), c0 + c1 + c2);
}
