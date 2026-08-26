/* hand_frame_wgpu.c -- FrameToTensor as one compute pass on the shared
 * Dawn device: the camera's dma-buf in, the WGPUBuffer ONNX Runtime has
 * bound out, no host memory anywhere.
 *
 * Two things here deviate from infer_ctx_wgpu.c's style on purpose.
 *
 * The bind group layout is written out instead of taken from
 * wgpuComputePipelineGetBindGroupLayout. A plane view of an imported
 * multiplanar texture must be bound as unfilterable-float; an auto-
 * derived layout infers plain float from the shader's texture_2d<f32>
 * and then rejects the view. Declaring it is the fix, and it is also
 * documentation of what the shader really needs.
 *
 * The dispatch does NOT wait, and nothing fences it against ORT. Both go
 * to the one queue of the one device the process shares with the WebGPU
 * EP, so the tensor is written before the kernels that read it by queue
 * order alone. That is the same argument the `identity` edge rests on
 * (README §20's table); it is why this pass is nearly free.
 */
#include <stdlib.h>
#include <string.h>

#include <drm_fourcc.h>

#include "hand_frame.h"
#include "hand_frame_to_tensor_wgsl.h" /* generated from shaders/ by embed_glsl */
#include "infer_util.h"
#include "nv12_convert.h"

#define SV(s) (WGPUStringView){(s), WGPU_STRLEN}

/* mirrors struct Params in the WGSL: five vec4s, all 16-byte aligned so
 * there is no padding to get wrong */
struct pass_params {
    float affine[4]; /* m0 m1 m3 m4 */
    float extra[4];  /* m2, m5, norm lo, norm span */
    uint32_t sizes[4];
    float coef[4];
    float range[4];    /* y off, y scale, chroma scale, unused */
    uint32_t flags[4]; /* yuyv, border_replicate, unused, unused */
};

struct hand_frame_pass {
    WGPUBindGroupLayout bgl;
    WGPUPipelineLayout layout;
    WGPUComputePipeline pipeline;
    WGPUBuffer uniform;
    /* the bind group is rebuilt only when one of the three things it
     * names changes -- and under graph capture the buffer never does */
    WGPUBindGroup bg;
    WGPUTextureView bg_p0, bg_p1;
    WGPUBuffer bg_buf;
};

struct hand_frame_pass* hand_frame_pass_create(struct infer_ctx_wgpu* c) {
    struct hand_frame_pass* p = calloc(1, sizeof *p);

    WGPUBindGroupLayoutEntry e[4];
    for (int i = 0; i < 4; i++) {
        e[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        e[i].binding = (uint32_t)i;
        e[i].visibility = WGPUShaderStage_Compute;
    }
    /* the two planes: unfilterable-float, because that is what a plane
     * view of a biplanar import is (and textureLoad needs no more) */
    e[0].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    e[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    e[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    e[2].buffer.type = WGPUBufferBindingType_Storage;
    e[3].buffer.type = WGPUBufferBindingType_Uniform;
    e[3].buffer.minBindingSize = sizeof(struct pass_params);

    WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bd.label = SV("hand frame_to_tensor");
    bd.entryCount = 4;
    bd.entries = e;
    p->bgl = wgpuDeviceCreateBindGroupLayout(c->device, &bd);

    WGPUPipelineLayoutDescriptor ld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    ld.bindGroupLayoutCount = 1;
    ld.bindGroupLayouts = &p->bgl;
    p->layout = wgpuDeviceCreatePipelineLayout(c->device, &ld);

    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = SV(hand_frame_to_tensor_wgsl);
    WGPUShaderModuleDescriptor sd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    sd.nextInChain = &src.chain;
    sd.label = SV("hand_frame_to_tensor");
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(c->device, &sd);
    WGPUComputePipelineDescriptor pd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pd.label = SV("hand_frame_to_tensor");
    pd.layout = p->layout;
    pd.compute.module = mod;
    pd.compute.entryPoint = SV("main");
    p->pipeline = wgpuDeviceCreateComputePipeline(c->device, &pd);
    wgpuShaderModuleRelease(mod);

    WGPUBufferDescriptor ud = WGPU_BUFFER_DESCRIPTOR_INIT;
    ud.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ud.size = sizeof(struct pass_params);
    p->uniform = wgpuDeviceCreateBuffer(c->device, &ud);

    if (!p->bgl || !p->layout || !p->pipeline || !p->uniform) {
        fprintf(stderr, "hand: frame_to_tensor pipeline creation failed\n");
        hand_frame_pass_destroy(p);
        return NULL;
    }
    return p;
}

void hand_frame_pass_destroy(struct hand_frame_pass* p) {
    if (!p) { return; }
    if (p->bg) { wgpuBindGroupRelease(p->bg); }
    if (p->uniform) { wgpuBufferRelease(p->uniform); }
    if (p->pipeline) { wgpuComputePipelineRelease(p->pipeline); }
    if (p->layout) { wgpuPipelineLayoutRelease(p->layout); }
    if (p->bgl) { wgpuBindGroupLayoutRelease(p->bgl); }
    free(p);
}

int hand_frame_to_tensor_wgpu(struct infer_ctx_wgpu* c,
                              struct hand_frame_pass* p,
                              struct infer_wgpu_frame_import* im,
                              const struct infer_frame* f,
                              struct hand_affine a,
                              struct hand_norm n,
                              enum hand_border border,
                              struct infer_tensor* t) {
    int errors0 = c->errors;
    INFER_CHECK(t->domain == INFER_DOMAIN_WGPU && t->mem.wgpu.buffer, "hand: WGPU tensor expected");
    INFER_CHECK(t->desc.ndim == 4 && t->desc.dims[3] == 3, "hand: tensor is not [1,H,W,3]");
    const uint32_t th = (uint32_t)t->desc.dims[1], tw = (uint32_t)t->desc.dims[2];
    const int yuyv = f->drm_format == DRM_FORMAT_YUYV;

    WGPUTextureView p0 = infer_wgpu_frame_plane(im, 0);
    /* YUYV has one plane; a bind group must still fill every entry, so
     * the shader is handed plane 0 twice and ignores the second */
    WGPUTextureView p1 = infer_wgpu_frame_plane(im, 1);
    if (!p1) { p1 = p0; }
    INFER_CHECK(p0, "hand: frame import has no plane 0");

    if (p->bg_p0 != p0 || p->bg_p1 != p1 || p->bg_buf != t->mem.wgpu.buffer) {
        if (p->bg) { wgpuBindGroupRelease(p->bg); }
        WGPUBindGroupEntry be[4] = {
            {.binding = 0, .textureView = p0},
            {.binding = 1, .textureView = p1},
            {.binding = 2, .buffer = t->mem.wgpu.buffer, .size = WGPU_WHOLE_SIZE},
            {.binding = 3, .buffer = p->uniform, .size = sizeof(struct pass_params)},
        };
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = p->bgl;
        bgd.entryCount = 4;
        bgd.entries = be;
        p->bg = wgpuDeviceCreateBindGroup(c->device, &bgd);
        INFER_CHECK(p->bg, "hand: frame_to_tensor bind group failed");
        p->bg_p0 = p0;
        p->bg_p1 = p1;
        p->bg_buf = t->mem.wgpu.buffer;
    }

    struct pass_params pp;
    memset(&pp, 0, sizeof pp);
    pp.affine[0] = a.m[0];
    pp.affine[1] = a.m[1];
    pp.affine[2] = a.m[3];
    pp.affine[3] = a.m[4];
    pp.extra[0] = a.m[2];
    pp.extra[1] = a.m[5];
    pp.extra[2] = n.lo;
    pp.extra[3] = n.hi - n.lo;
    pp.sizes[0] = tw;
    pp.sizes[1] = th;
    /* the frame's own luma size, not the import's: for YUYV the import is
     * half as wide and the shader converts back through the parity pick */
    pp.sizes[2] = f->width;
    pp.sizes[3] = f->height;
    nv12_yuv_params(f->bt709, f->full_range, pp.coef, pp.range);
    pp.flags[0] = yuyv ? 1u : 0u;
    pp.flags[1] = border == HAND_BORDER_REPLICATE ? 1u : 0u;
    wgpuQueueWriteBuffer(c->queue, p->uniform, 0, &pp, sizeof pp);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(cp, p->pipeline);
    wgpuComputePassEncoderSetBindGroup(cp, 0, p->bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cp, (tw + 7) / 8, (th + 7) / 8, 1);
    wgpuComputePassEncoderEnd(cp);
    wgpuComputePassEncoderRelease(cp);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    /* submitted, not complete: the queue orders us before ORT's kernels */
    return c->errors != errors0 ? -1 : 0;
}
