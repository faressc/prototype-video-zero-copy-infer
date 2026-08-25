// infer_relayout.wgsl -- WebGPU: the DEVICE_COPY. An imported dma-buf
// byte image (RGBA8, one texel = one float's bytes) becomes the packed
// float buffer ONNX Runtime consumes. pack4x8unorm undoes the
// producer's unpackUnorm4x8 exactly.

@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
@group(0) @binding(2) var<uniform> p : vec4<u32>; // img_w, img_h

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x >= p.x || id.y >= p.y) { return; }
    let t = textureLoad(tex, vec2<i32>(i32(id.x), i32(id.y)), 0);
    o[id.y * p.x + id.x] = bitcast<f32>(pack4x8unorm(t));
}
