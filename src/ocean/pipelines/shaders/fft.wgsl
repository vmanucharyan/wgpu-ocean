[[group(0), binding(0)]]
var precompute_buffer: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(1)]]
var buffer_a_0: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(2)]]
var buffer_a_1: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(3)]]
var buffer_b_0: texture_storage_2d<rgba32float, read_write>;

[[group(0), binding(4)]]
var buffer_b_1: texture_storage_2d<rgba32float, read_write>;

struct Params {
  ping_pong: u32;
  step: u32;
  size: u32;
};

var<push_constant> params: Params;

let PI = 3.1415926;

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

fn complex_exp(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(a.y), sin(a.y)) * exp(a.x);
}

[[stage(compute), workgroup_size(1, 8)]]
fn calculat_twiddle_factors_and_input_indices(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let b = params.size >> (id.x + 1u);
    let mult = 2.0 * PI * vec2<f32>(0.0, 1.0) / f32(params.size);
    let i = (2u * b * (id.y / b) + id.y % b) % params.size;
    let twiddle = complex_exp((-mult * f32((id.y / b) * b)));

    textureStore(
        precompute_buffer,
        vec2<i32>(id.xy),
        vec4<f32>(twiddle.x, twiddle.y, f32(i), f32(i + b))
    );

    textureStore(
        precompute_buffer,
        vec2<i32>(i32(id.x), i32(id.y + params.size / 2u)),
        vec4<f32>(-twiddle.x, -twiddle.y, f32(i), f32(i + b)),
    );
}

[[stage(compute), workgroup_size(16, 16)]]
fn horizontal_step_inverse_fft(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let data = textureLoad(
        precompute_buffer,
        vec2<i32>(i32(params.step), i32(id.x))
    );

    let input_indices = vec2<i32>(data.ba);

    if (bool(params.ping_pong)) {
        {
            let bx = textureLoad(
                buffer_a_0,
                vec2<i32>(input_indices.x, i32(id.y))
            );

            let by = textureLoad(
                buffer_a_0,
                vec2<i32>(input_indices.y, i32(id.y)),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_a_1,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }

            {
            let bx = textureLoad(
                buffer_b_0,
                vec2<i32>(input_indices.x, i32(id.y))
            );

            let by = textureLoad(
                buffer_b_0,
                vec2<i32>(input_indices.y, i32(id.y)),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_b_1,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }
    } else {
            {
            let bx = textureLoad(
                buffer_a_1,
                vec2<i32>(input_indices.x, i32(id.y)),
            );

            let by = textureLoad(
                buffer_a_1,
                vec2<i32>(input_indices.y, i32(id.y)),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_a_0,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }

        {
            let bx = textureLoad(
                buffer_b_1,
                vec2<i32>(input_indices.x, i32(id.y)),
            );

            let by = textureLoad(
                buffer_b_1,
                vec2<i32>(input_indices.y, i32(id.y)),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_b_0,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn vertical_step_inverse_fft(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let data = textureLoad(
        precompute_buffer,
        vec2<i32>(i32(params.step), i32(id.y))
    );

    let input_indices = vec2<i32>(data.ba);

    if (bool(params.ping_pong)) {
        {
            let bx = textureLoad(
                buffer_a_0,
                vec2<i32>(i32(id.x), input_indices.x)
            );

            let by = textureLoad(
                buffer_a_0,
                vec2<i32>(i32(id.x), input_indices.y),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_a_1,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }

        {
            let bx = textureLoad(
                buffer_b_0,
                vec2<i32>(i32(id.x), input_indices.x)
            );

            let by = textureLoad(
                buffer_b_0,
                vec2<i32>(i32(id.x), input_indices.y),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_b_1,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }
    } else {
        {
            let bx = textureLoad(
                buffer_a_1,
                vec2<i32>(i32(id.x), input_indices.x)
            );

            let by = textureLoad(
                buffer_a_1,
                vec2<i32>(i32(id.x), input_indices.y),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_a_0,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }

        {
            let bx = textureLoad(
                buffer_b_1,
                vec2<i32>(i32(id.x), input_indices.x)
            );

            let by = textureLoad(
                buffer_b_1,
                vec2<i32>(i32(id.x), input_indices.y),
            );

            let bx0 = bx.xy;
            let by0 = by.xy;
            let bx1 = bx.zw;
            let by1 = by.zw;

            let res0 = bx0 + complex_mult(vec2<f32>(data.r, -data.g), by0);
            let res1 = bx1 + complex_mult(vec2<f32>(data.r, -data.g), by1);

            textureStore(
                buffer_b_0,
                vec2<i32>(id.xy),
                vec4<f32>(res0, res1),
            );
        }
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn scale(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let b0 = textureLoad(
        buffer_a_0,
        vec2<i32>(id.xy),
    );

    textureStore(
        buffer_a_0,
        vec2<i32>(id.xy),
        b0 / f32(params.size) / f32(params.size),
    );
}

[[stage(compute), workgroup_size(16, 16)]]
fn permute(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let b0 = textureLoad(
        buffer_a_0,
        vec2<i32>(id.xy),
    );

    textureStore(
        buffer_a_0,
        vec2<i32>(id.xy),
        b0 * (1.0 - 2.0 * f32((id.x + id.y) % 2u)),
    );

    let bb = textureLoad(
        buffer_b_0,
        vec2<i32>(id.xy),
    );

    textureStore(
        buffer_b_0,
        vec2<i32>(id.xy),
        bb * (1.0 - 2.0 * f32((id.x + id.y) % 2u)),
    );
}

[[stage(compute), workgroup_size(16, 16)]]
fn swap(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let b1 = textureLoad(
        buffer_a_1,
        vec2<i32>(id.xy),
    );

    textureStore(
        buffer_a_0,
        vec2<i32>(id.xy),
        b1,
    );

    let b1 = textureLoad(
        buffer_b_1,
        vec2<i32>(id.xy),
    );

    textureStore(
        buffer_b_0,
        vec2<i32>(id.xy),
        b1,
    );
}
