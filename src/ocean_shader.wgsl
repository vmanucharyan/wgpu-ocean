// Vertex shader

struct CameraUniform {
    pos: vec3<f32>;
    view_proj: mat4x4<f32>;
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
    inverse_view_proj: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;

[[group(1), binding(0)]]
var t_displacement_0: texture_2d<f32>;

[[group(1), binding(1)]]
var t_derivatives_0: texture_2d<f32>;

[[group(1), binding(2)]]
var s_derivatives: sampler;

[[group(1), binding(3)]]
var t_displacement_1: texture_2d<f32>;

[[group(1), binding(4)]]
var t_derivatives_1: texture_2d<f32>;

[[group(1), binding(5)]]
var t_displacement_2: texture_2d<f32>;

[[group(1), binding(6)]]
var t_derivatives_2: texture_2d<f32>;

[[group(1), binding(7)]]
var t_foam: texture_2d<f32>;

let SKY_COLOR = vec3<f32>(0.9, 0.9, 0.9);

let OCEAN_BASE_COLOR = vec3<f32>(0.0, 0.10, 0.18);
let OCEAN_WATER_COLOR = vec3<f32>(0.48, 0.54, 0.36);

let OCEAN_COLOR = vec3<f32>(0.0, 0.38, 0.53);
// let OCEAN_COLOR = vec3<f32>(0.0, 0.0, 1.0);
let SUN_DIR = vec3<f32>(-1.0, 1.0, 1.0);
let LOD_SCALE = 15.0;
let LENGTH_SCALE = vec3<f32>(500.0, 85.0, 10.0);

let PI: f32 = 3.14159265358979323846264338;
let INFINITE = 100000.0;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] color: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
    [[location(1)]] uv_0: vec2<f32>;
    [[location(2)]] uv_1: vec2<f32>;
    [[location(3)]] uv_2: vec2<f32>;
    [[location(4)]] world_pos: vec3<f32>;
    [[location(5)]] view_vector: vec3<f32>;
    [[location(6)]] lod_scales: vec3<f32>;
};

fn screen_to_world(screen_uv: vec3<f32>) -> vec3<f32> {
    let w = camera.inverse_view_proj * vec4<f32>(screen_uv.xyz, 1.0);
    return w.xyz * (1.0 / w.w);
}

fn intercept_plane(source: vec3<f32>, dir: vec3<f32>, normal: vec3<f32>, height: f32) -> vec3<f32> {
    // Compute the distance between the source and the surface, following a ray, then return the intersection
    // http://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S09/lectures/11_ray_tracing.pdf

    let distance = (-height - dot(normal, source)) / dot(normal, dir);

    if (distance < 0.0) {
        return source + dir * distance;
    } else {
        return - (vec3<f32>(source.x, height, source.z) + vec3<f32>(dir.x, height, dir.z) * INFINITE);
    }
}

[[stage(vertex)]]
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let screen_uv = model.position.xz;

    let ray_origin = screen_to_world(vec3<f32>(screen_uv, 0.0));
    let ray_end = screen_to_world(vec3<f32>(screen_uv, 1.0));
    let ray = normalize(ray_origin - ray_end);
    let surface_point = vec3<f32>(0.0);
    let surface_normal = vec3<f32>(0.0, 1.0, 0.0);

    var dist = dot(surface_point - ray_origin, surface_normal) / dot(ray, surface_normal);
    var world_pos = vec3<f32>(0.0);

    if (dist < 0.0) {
        world_pos = ray_origin + dist * ray;
    } else {
        world_pos = ray_origin - INFINITE * ray;
    }

    let view_dist = abs(length(camera.pos - world_pos));
    let lod_c0 = min(LOD_SCALE * LENGTH_SCALE.x / view_dist, 1.0);
    let lod_c1 = min(LOD_SCALE * LENGTH_SCALE.y / view_dist, 1.0);
    let lod_c2 = min(LOD_SCALE * LENGTH_SCALE.z / view_dist, 1.0);

    let near = view_dist < 200.0;
    let mid = view_dist < 2000.0;

    let ocean_uv_0 = world_pos.xz / LENGTH_SCALE.x;
    let ocean_uv_1 = world_pos.xz / LENGTH_SCALE.y;
    let ocean_uv_2 = world_pos.xz / LENGTH_SCALE.z;

    var d = textureDimensions(t_displacement_0);

    var tex_coord_0 = vec2<i32>(
        i32(ocean_uv_0.x * f32(d.x)) % d.x,
        i32(ocean_uv_0.y * f32(d.y)) % d.y,
    );

    var tex_coord_1 = vec2<i32>(
        i32(ocean_uv_1.x * f32(d.x)) % d.x,
        i32(ocean_uv_1.y * f32(d.y)) % d.y,
    );

    var tex_coord_2 = vec2<i32>(
        i32(ocean_uv_2.x * f32(d.x)) % d.x,
        i32(ocean_uv_2.y * f32(d.y)) % d.y,
    );

    var displacement = vec4<f32>(0.0);
    displacement = displacement + textureSampleLevel(t_displacement_0, s_derivatives, ocean_uv_0, 0.0) * lod_c0;
    if (mid) {
        displacement = displacement + textureSampleLevel(t_displacement_1, s_derivatives, ocean_uv_1, 0.0) * lod_c1;
    }
    if (near) {
        displacement = displacement + textureSampleLevel(t_displacement_2, s_derivatives, ocean_uv_2, 0.0) * lod_c2;
    }

    var pos = world_pos + displacement.xyz;

    out.color = model.color;
    out.uv_0 = ocean_uv_0;
    out.uv_1 = ocean_uv_1;
    out.uv_2 = ocean_uv_2;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.view_vector = normalize(camera.pos - pos);
    out.lod_scales = vec3<f32>(lod_c0, lod_c1, lod_c2);

    return out;
}

fn hdr(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return 1.0 - exp(-color * exposure);
}

fn diffuse(n: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

fn specular(n: vec3<f32>, l: vec3<f32>, e: vec3<f32>, s: f32) -> f32 {
    let nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

fn getSkyColor(e: vec3<f32>) -> vec3<f32> {
    let ey = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return vec3<f32>(pow(1.0 - ey, 2.0), 1.0 - ey, 0.6 + (1.0 - ey) * 0.4) * 1.1;
}

fn getSeaColor(p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, eye: vec3<f32>, dist: vec3<f32>) -> vec3<f32> {  
    var fresnel_factor = dot(n, -eye);
    fresnel_factor = max(fresnel_factor, 0.0);
    fresnel_factor = 1.0 - fresnel_factor;
    fresnel_factor = pow(fresnel_factor, 5.0) * 0.5;

    let reflected = SKY_COLOR;
    let refracted = OCEAN_BASE_COLOR + diffuse(n, l, 80.0) * OCEAN_WATER_COLOR * 0.12;

    var color = (1.0 - fresnel_factor) * refracted + fresnel_factor * reflected;

    let atten = max(1.0 - dot(dist, dist) * 0.00001, 0.0);
    color = color + OCEAN_WATER_COLOR * (p.y - 0.1) * 0.04 * atten;
    color = color + vec3<f32>(1.0, 1.0, 1.0) * 0.05 * specular(n,l,eye,1200.0);

    return color;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let distance = abs(length(camera.pos - in.world_pos));
    let near = distance < 300.0;
    let mid = distance < 2000.0;

    let d0 = textureSample(t_derivatives_0, s_derivatives, in.uv_0);
    let d1 = textureSample(t_derivatives_1, s_derivatives, in.uv_1);
    let d2 = textureSample(t_derivatives_2, s_derivatives, in.uv_2);

    var d = vec4<f32>(0.0);
    d = d + d0 * in.lod_scales.x;

    if (mid) {
        d = d + d1 * in.lod_scales.y;
    }

    if (near) {
        d = d + d2 * in.lod_scales.z;
    }

    let j0 = textureSample(t_displacement_0, s_derivatives, in.uv_0).w * 0.6;
    let j1 = textureSample(t_displacement_1, s_derivatives, in.uv_1).w * 0.17;
    let j2 = textureSample(t_displacement_2, s_derivatives, in.uv_2).w * 0.23;

    let turbulence = clamp((-(j0 + j1 + j2) + 0.84) * 2.4, 0.0, 1.0);
    var slope = vec2<f32>(d.x / (1.0 + d.z), d.y / (1.0 + d.w));
    var normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));

    let fog_range = vec2<f32>(200.0, 10000.0);
    let fog_factor = clamp((distance - fog_range.x) / (fog_range.y - fog_range.x), 0.0, 1.0);

    let foam_color = textureSample(t_foam, s_derivatives, in.uv_1);
    let foam = turbulence;

    let light = normalize(SUN_DIR);
    let color = getSeaColor(in.world_pos, normal, light, normalize(in.world_pos - camera.pos), camera.pos - in.world_pos);

    return vec4<f32>(color + fog_factor + foam, 1.0);
}
