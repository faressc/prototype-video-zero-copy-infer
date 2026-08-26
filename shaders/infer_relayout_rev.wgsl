// infer_relayout_rev.wgsl -- WebGPU: the DEVICE_COPY in the other
// direction. A packed float buffer (an engine's output) becomes an
// imported dma-buf byte image (RGBA8, one texel = one float's bytes)
// that a Vulkan, GL or CPU consumer owns. unpack4x8unorm is exact per
// byte and the rgba8unorm store rounds k/255 back to k, so the
// consumer reads the engine's bits unchanged. Texels past the element
// count (rows are padded to 64 bytes) are left alone.

@group(0) @binding(0) var tex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> i : array<f32>;
@group(0) @binding(2) var<uniform> p : vec4<u32>; // img_w, img_h, elements

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let idx = id.y * p.x + id.x;
    if (id.x >= p.x || id.y >= p.y || idx >= p.z) { return; }
    textureStore(tex, vec2<i32>(i32(id.x), i32(id.y)), unpack4x8unorm(bitcast<u32>(i[idx])));
}
