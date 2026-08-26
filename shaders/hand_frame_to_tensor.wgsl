// hand_frame_to_tensor.wgsl -- FrameToTensor on the GPU: the camera's
// dma-buf, imported into the shared Dawn device, resampled through an
// affine into the packed float buffer ONNX Runtime's WebGPU EP has bound.
// One pass serves both model inputs -- the detector's letterbox and the
// landmark stage's rotated crop -- because the only difference between
// them is p.affine (see hand/hand_affine.h).
//
// The C twin is hand/hand_frame_cpu.c and the two must agree: same
// bilinear convention (pixel k covers [k, k+1), centre at k+0.5),
// same clamp-to-edge, same chroma coordinates (half of luma's, "centre
// siting"), same YUV coefficients (camera/nv12_convert.h's
// nv12_yuv_params, handed over in p.coef / p.range). hello_hand
// --compare-frame-to-tensor asserts the agreement.
//
// textureLoad, not textureSampleLevel: a plane view of an imported
// biplanar texture is bound as unfilterable-float on this driver, and
// hand-writing the bilinear tap costs four loads while removing the
// filterability question entirely -- and it is what lets the C twin be
// formula-identical rather than approximately similar.

struct Params {
    affine : vec4<f32>,   // m0 m1 m3 m4
    extra  : vec4<f32>,   // m2, m5, norm lo, norm span
    sizes  : vec4<u32>,   // tensor w, tensor h, frame w, frame h
    coef   : vec4<f32>,   // r<-v, g<-u, g<-v, b<-u
    range  : vec4<f32>,   // y offset, y scale, chroma scale, unused
    flags  : vec4<u32>,   // x: 1 = YUYV packed, y: 1 = BORDER_REPLICATE, else unused
}

// plane 0: NV12 luma (r8unorm) or, in YUYV mode, the packed plane as
// half-width RGBA. plane 1: NV12 chroma (rg8unorm, half size); in YUYV
// mode it is plane 0 again, because a bind group must fill every entry.
@group(0) @binding(0) var plane0 : texture_2d<f32>;
@group(0) @binding(1) var plane1 : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out_tensor : array<f32>;
@group(0) @binding(3) var<uniform> p : Params;

// One bilinear tap. `dim` is the plane's size in samples, `q` a
// continuous image coordinate in that plane's own scale.
fn tap0(q : vec2<f32>, dim : vec2<i32>) -> vec4<f32> {
    let f = q - vec2<f32>(0.5, 0.5);
    let i0 = floor(f);
    let a = f - i0;
    let lo = vec2<i32>(i0);
    let c0 = clamp(lo, vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    let c1 = clamp(lo + vec2<i32>(1, 1), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    let t00 = textureLoad(plane0, vec2<i32>(c0.x, c0.y), 0);
    let t10 = textureLoad(plane0, vec2<i32>(c1.x, c0.y), 0);
    let t01 = textureLoad(plane0, vec2<i32>(c0.x, c1.y), 0);
    let t11 = textureLoad(plane0, vec2<i32>(c1.x, c1.y), 0);
    return mix(mix(t00, t10, a.x), mix(t01, t11, a.x), a.y);
}

fn tap1(q : vec2<f32>, dim : vec2<i32>) -> vec4<f32> {
    let f = q - vec2<f32>(0.5, 0.5);
    let i0 = floor(f);
    let a = f - i0;
    let lo = vec2<i32>(i0);
    let c0 = clamp(lo, vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    let c1 = clamp(lo + vec2<i32>(1, 1), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    let t00 = textureLoad(plane1, vec2<i32>(c0.x, c0.y), 0);
    let t10 = textureLoad(plane1, vec2<i32>(c1.x, c0.y), 0);
    let t01 = textureLoad(plane1, vec2<i32>(c0.x, c1.y), 0);
    let t11 = textureLoad(plane1, vec2<i32>(c1.x, c1.y), 0);
    return mix(mix(t00, t10, a.x), mix(t01, t11, a.x), a.y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tw = p.sizes.x;
    let th = p.sizes.y;
    if (gid.x >= tw || gid.y >= th) { return; }

    let fw = i32(p.sizes.z);
    let fh = i32(p.sizes.w);
    let lo = p.extra.z;
    let span = p.extra.w;
    let base = (gid.y * tw + gid.x) * 3u;

    // destination pixel centre -> frame pixel coordinate
    let u = f32(gid.x) + 0.5;
    let v = f32(gid.y) + 0.5;
    let x = p.affine.x * u + p.affine.y * v + p.extra.x;
    let y = p.affine.z * u + p.affine.w * v + p.extra.y;

    // Outside the frame: black for the detector (BORDER_ZERO, which is
    // what its letterbox bars are), the nearest edge pixel for the
    // landmark crop (BORDER_REPLICATE, MediaPipe's default). tap0/tap1
    // already clamp their coordinates, so REPLICATE is simply not taking
    // this branch -- see hand/hand_frame.h for why the two differ.
    if (p.flags.y == 0u && (x < 0.0 || y < 0.0 || x >= f32(fw) || y >= f32(fh))) {
        out_tensor[base + 0u] = lo;
        out_tensor[base + 1u] = lo;
        out_tensor[base + 2u] = lo;
        return;
    }

    // unorm8 texels arrive in [0,1]; the coefficients are in 0..255 units
    var yy : f32;
    var uu : f32;
    var vv : f32;
    if (p.flags.x == 0u) {
        // NV12: luma at full resolution, chroma at half in both axes
        yy = tap0(vec2<f32>(x, y), vec2<i32>(fw, fh)).r * 255.0;
        let ch = tap1(vec2<f32>(x * 0.5, y * 0.5), vec2<i32>(fw / 2, fh / 2));
        uu = ch.r * 255.0;
        vv = ch.g * 255.0;
    } else {
        // YUYV as half-width RGBA: texel = (Y0, U, Y1, V). Nearest by
        // construction -- interpolating across a pair boundary would mix
        // Y0 with Y1 -- so the parity pick is exact, as in the C twin.
        let xi = min(i32(x), fw - 1);
        let yi = min(i32(y), fh - 1);
        let t = textureLoad(plane0, vec2<i32>(xi / 2, yi), 0);
        yy = select(t.r, t.b, (xi & 1) == 1) * 255.0;
        uu = t.g * 255.0;
        vv = t.a * 255.0;
    }

    yy = (yy - p.range.x) * p.range.y;
    uu = (uu - 128.0) * p.range.z;
    vv = (vv - 128.0) * p.range.z;
    let rgb = clamp(vec3<f32>(yy + p.coef.x * vv,
                              yy - p.coef.y * uu - p.coef.z * vv,
                              yy + p.coef.w * uu) * (1.0 / 255.0),
                    vec3<f32>(0.0, 0.0, 0.0),
                    vec3<f32>(1.0, 1.0, 1.0));
    out_tensor[base + 0u] = lo + rgb.r * span;
    out_tensor[base + 1u] = lo + rgb.g * span;
    out_tensor[base + 2u] = lo + rgb.b * span;
}
