// infer_gen.wgsl -- WebGPU: fill a packed float storage buffer with the
// deterministic generator (same two lines as infer_hash.glsl).

@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@group(0) @binding(1) var<uniform> p : vec4<u32>; // n, 0, 0, seed

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= p.x) { return; }
    let h = ((i + p.w) * 2654435761u) >> 8u;
    o[i] = f32(h & 0xFFFFu) / 65536.0 - 0.5;
}
