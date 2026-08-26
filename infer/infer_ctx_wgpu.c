/* infer_ctx_wgpu.c -- the WebGPU domain on Dawn: the one device this
 * process shares with ONNX Runtime's WebGPU EP, the generator pass, the
 * MapRead readback, and the dma-buf import that is the DEVICE_COPY edge
 * in both directions: a foreign byte image read into a packed float
 * buffer (an input), or a packed float buffer written into a foreign
 * byte image (an output a consumer owns).
 *
 * Dawn's C API is future-based: every asynchronous call returns a
 * WGPUFuture and we block on it with wgpuInstanceWaitAny (the instance
 * is created with TimedWaitAny for that). The dma-buf import is Dawn-
 * native: SharedTextureMemory over a DmaBuf descriptor, BeginAccess /
 * EndAccess with Vulkan image-layout states and SharedFences that are
 * sync files on the outside.
 */
#include "infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <drm_fourcc.h>

#include "infer_util.h"
#include "infer_gen_wgsl.h"
#include "infer_relayout_wgsl.h"
#include "infer_relayout_rev_wgsl.h"

#define SV(s) (WGPUStringView){(s), WGPU_STRLEN}
#define SV_ARGS(s) \
    ((int)((s).length == WGPU_STRLEN ? strlen((s).data ? (s).data : "") : (s).length)), \
        ((s).data ? (s).data : "")

/* ------------------------------------------------------------------ */
/* callbacks + waiting                                                */
/* ------------------------------------------------------------------ */

static void on_adapter(WGPURequestAdapterStatus st,
                       WGPUAdapter a,
                       WGPUStringView msg,
                       void* u1,
                       void* u2) {
    (void)u2;
    if (st != WGPURequestAdapterStatus_Success) {
        fprintf(stderr, "wgpu: no adapter: %.*s\n", SV_ARGS(msg));
    }
    *(WGPUAdapter*)u1 = a;
}

static void on_device(WGPURequestDeviceStatus st,
                      WGPUDevice d,
                      WGPUStringView msg,
                      void* u1,
                      void* u2) {
    (void)u2;
    if (st != WGPURequestDeviceStatus_Success) {
        fprintf(stderr, "wgpu: no device: %.*s\n", SV_ARGS(msg));
    }
    *(WGPUDevice*)u1 = d;
}

static void on_error(WGPUDevice const* d, WGPUErrorType t, WGPUStringView msg, void* u1, void* u2) {
    (void)d;
    (void)u2;
    struct infer_ctx_wgpu* c = u1;
    c->errors++;
    fprintf(stderr, "wgpu error %d: %.*s\n", (int)t, SV_ARGS(msg));
}

static void on_lost(WGPUDevice const* d,
                    WGPUDeviceLostReason r,
                    WGPUStringView msg,
                    void* u1,
                    void* u2) {
    (void)d;
    (void)u1;
    (void)u2;
    if (r != WGPUDeviceLostReason_Destroyed) {
        fprintf(stderr, "wgpu device lost (%d): %.*s\n", (int)r, SV_ARGS(msg));
    }
}

static void on_map(WGPUMapAsyncStatus st, WGPUStringView msg, void* u1, void* u2) {
    (void)u2;
    if (st != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "wgpu map failed: %.*s\n", SV_ARGS(msg));
    }
    *(int*)u1 = st == WGPUMapAsyncStatus_Success;
}

static int wait_future(struct infer_ctx_wgpu* c, WGPUFuture f) {
    WGPUFutureWaitInfo w = WGPU_FUTURE_WAIT_INFO_INIT;
    w.future = f;
    WGPUWaitStatus s = wgpuInstanceWaitAny(c->instance, 1, &w, UINT64_MAX);
    return s == WGPUWaitStatus_Success ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* pipelines                                                          */
/* ------------------------------------------------------------------ */

static WGPUComputePipeline make_pipeline(struct infer_ctx_wgpu* c, const char* wgsl, const char* label) {
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = SV(wgsl);
    WGPUShaderModuleDescriptor sd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    sd.nextInChain = &src.chain;
    sd.label = SV(label);
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(c->device, &sd);
    WGPUComputePipelineDescriptor pd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pd.label = SV(label);
    pd.compute.module = mod;
    pd.compute.entryPoint = SV("main");
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(c->device, &pd);
    wgpuShaderModuleRelease(mod);
    return p;
}

WGPUBuffer infer_wgpu_create_buffer(struct infer_ctx_wgpu* c, uint64_t bytes, WGPUBufferUsage extra) {
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst | extra;
    bd.size = infer_align((size_t)bytes, 16);
    return wgpuDeviceCreateBuffer(c->device, &bd);
}

/* ------------------------------------------------------------------ */
/* bring-up                                                           */
/* ------------------------------------------------------------------ */

static int setup_common(struct infer_ctx_wgpu* c) {
    c->queue = wgpuDeviceGetQueue(c->device);
    c->has_dmabuf = wgpuDeviceHasFeature(c->device, WGPUFeatureName_SharedTextureMemoryDmaBuf);
    c->has_sync_fd = wgpuDeviceHasFeature(c->device, WGPUFeatureName_SharedFenceSyncFD);
    c->gen_pipeline = make_pipeline(c, infer_gen_wgsl, "infer_gen");
    c->relayout_pipeline = make_pipeline(c, infer_relayout_wgsl, "infer_relayout");
    c->relayout_rev_pipeline = make_pipeline(c, infer_relayout_rev_wgsl, "infer_relayout_rev");
    INFER_CHECK(c->gen_pipeline && c->relayout_pipeline && c->relayout_rev_pipeline,
                "wgpu: pipeline creation failed");
    return 0;
}

int infer_ctx_wgpu_init(struct infer_ctx_wgpu* c) {
    memset(c, 0, sizeof *c);
    c->owned = 1;

    WGPUInstanceFeatureName ifeat[] = {WGPUInstanceFeatureName_TimedWaitAny};
    WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
    id.requiredFeatureCount = 1;
    id.requiredFeatures = ifeat;
    c->instance = wgpuCreateInstance(&id);
    INFER_CHECK(c->instance, "wgpu: no instance");

    WGPURequestAdapterOptions ao = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    ao.backendType = WGPUBackendType_Vulkan;
    ao.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo aci = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    aci.mode = WGPUCallbackMode_WaitAnyOnly;
    aci.callback = on_adapter;
    aci.userdata1 = &c->adapter;
    wait_future(c, wgpuInstanceRequestAdapter(c->instance, &ao, aci));
    INFER_CHECK(c->adapter, "wgpu: no Vulkan adapter");

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    wgpuAdapterGetInfo(c->adapter, &info);
    fprintf(stderr, "wgpu: %.*s (%.*s)\n", SV_ARGS(info.device), SV_ARGS(info.description));
    wgpuAdapterInfoFreeMembers(info);

    /* what the edges need + what ORT's kernels like, if the adapter has it */
    static const WGPUFeatureName wanted[] = {
        WGPUFeatureName_SharedTextureMemoryDmaBuf,
        WGPUFeatureName_SharedFenceSyncFD,
        WGPUFeatureName_DawnMultiPlanarFormats,
        WGPUFeatureName_ShaderF16,
        WGPUFeatureName_Subgroups,
        WGPUFeatureName_TimestampQuery,
    };
    WGPUFeatureName req[8];
    size_t nreq = 0;
    for (size_t i = 0; i < sizeof wanted / sizeof wanted[0]; i++) {
        if (wgpuAdapterHasFeature(c->adapter, wanted[i])) { req[nreq++] = wanted[i]; }
    }
    const char* toggles[] = {"allow_unsafe_apis"};
    WGPUDawnTogglesDescriptor tg = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
    tg.enabledToggleCount = 1;
    tg.enabledToggles = toggles;
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.nextInChain = &tg.chain;
    dd.requiredFeatureCount = nreq;
    dd.requiredFeatures = req;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    dd.uncapturedErrorCallbackInfo.userdata1 = c;
    dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dd.deviceLostCallbackInfo.callback = on_lost;
    WGPURequestDeviceCallbackInfo dci = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dci.mode = WGPUCallbackMode_WaitAnyOnly;
    dci.callback = on_device;
    dci.userdata1 = &c->device;
    wait_future(c, wgpuAdapterRequestDevice(c->adapter, &dd, dci));
    INFER_CHECK(c->device, "wgpu: no device");
    return setup_common(c);
}

int infer_ctx_wgpu_borrow(struct infer_ctx_wgpu* c, WGPUInstance instance, WGPUDevice device) {
    memset(c, 0, sizeof *c);
    c->instance = instance;
    c->device = device;
    wgpuInstanceAddRef(instance);
    wgpuDeviceAddRef(device);
    return setup_common(c);
}

void infer_ctx_wgpu_fini(struct infer_ctx_wgpu* c) {
    if (c->gen_pipeline) { wgpuComputePipelineRelease(c->gen_pipeline); }
    if (c->relayout_pipeline) { wgpuComputePipelineRelease(c->relayout_pipeline); }
    if (c->relayout_rev_pipeline) { wgpuComputePipelineRelease(c->relayout_rev_pipeline); }
    if (c->queue) { wgpuQueueRelease(c->queue); }
    if (c->device) { wgpuDeviceRelease(c->device); }
    if (c->adapter) { wgpuAdapterRelease(c->adapter); }
    if (c->instance) { wgpuInstanceRelease(c->instance); }
    memset(c, 0, sizeof *c);
}

/* ------------------------------------------------------------------ */
/* the tensor: a packed float buffer on the shared device              */
/* ------------------------------------------------------------------ */

int infer_wgpu_alloc(struct infer_ctx_wgpu* c, const struct infer_desc* d, struct infer_tensor* t) {
    if (t->mem.wgpu.buffer) { return 0; }
    t->domain = INFER_DOMAIN_WGPU;
    t->desc = *d;
    t->desc.row_pitch_bytes = d->img_w * 4; /* packed */
    t->owned = 1;
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
    t->mem.wgpu.buffer = infer_wgpu_create_buffer(c, infer_desc_bytes_packed(d), 0);
    INFER_CHECK(t->mem.wgpu.buffer, "wgpu: buffer creation failed");
    return 0;
}

int infer_wgpu_gen(struct infer_ctx_wgpu* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t) {
    size_t n = infer_desc_elements(d);
    /* c->errors is cumulative for the life of the device, so an error
     * any earlier caller provoked -- a dma-buf import this driver
     * rejects, say -- would otherwise fail every later operation on a
     * healthy device. Only errors raised between here and the return
     * belong to this call. */
    int errors0 = c->errors;
    if (infer_wgpu_alloc(c, d, t) < 0) { return -1; }
    if (!t->mem.wgpu.bind_group) {
        WGPUBufferDescriptor ud = WGPU_BUFFER_DESCRIPTOR_INIT;
        ud.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        ud.size = 16;
        t->mem.wgpu.uniform = wgpuDeviceCreateBuffer(c->device, &ud);
        WGPUBindGroupEntry entries[2] = {
            {.binding = 0, .buffer = t->mem.wgpu.buffer, .size = n * sizeof(float)},
            {.binding = 1, .buffer = t->mem.wgpu.uniform, .size = 16},
        };
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(c->gen_pipeline, 0);
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = bgl;
        bgd.entryCount = 2;
        bgd.entries = entries;
        t->mem.wgpu.bind_group = wgpuDeviceCreateBindGroup(c->device, &bgd);
        wgpuBindGroupLayoutRelease(bgl);
        INFER_CHECK(t->mem.wgpu.uniform && t->mem.wgpu.bind_group, "wgpu: generator resources");
    }
    uint32_t params[4] = {(uint32_t)n, 0, 0, seed};
    wgpuQueueWriteBuffer(c->queue, t->mem.wgpu.uniform, 0, params, sizeof params);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, c->gen_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, t->mem.wgpu.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)((n + 255) / 256), 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    /* same queue as every consumer on this device: ordering is the
     * queue's; nothing to export */
    return c->errors != errors0 ? -1 : 0;
}

int infer_wgpu_readback(struct infer_ctx_wgpu* c, const struct infer_tensor* t, float* dst) {
    size_t bytes = infer_desc_bytes_packed(&t->desc);
    WGPUBufferDescriptor sd = WGPU_BUFFER_DESCRIPTOR_INIT;
    sd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    sd.size = infer_align(bytes, 4);
    WGPUBuffer staging = wgpuDeviceCreateBuffer(c->device, &sd);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, t->mem.wgpu.buffer, 0, staging, 0, sd.size);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    int ok = 0;
    WGPUBufferMapCallbackInfo mi = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mi.mode = WGPUCallbackMode_WaitAnyOnly;
    mi.callback = on_map;
    mi.userdata1 = &ok;
    wait_future(c, wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, sd.size, mi));
    if (ok) {
        const void* p = wgpuBufferGetConstMappedRange(staging, 0, sd.size);
        memcpy(dst, p, bytes);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
    return ok ? 0 : -1;
}

void infer_wgpu_release(struct infer_ctx_wgpu* c, struct infer_tensor* t) {
    (void)c;
    if (!t->owned) { return; }
    if (t->mem.wgpu.bind_group) { wgpuBindGroupRelease(t->mem.wgpu.bind_group); }
    if (t->mem.wgpu.uniform) { wgpuBufferRelease(t->mem.wgpu.uniform); }
    if (t->mem.wgpu.buffer) { wgpuBufferRelease(t->mem.wgpu.buffer); }
}

/* ------------------------------------------------------------------ */
/* dma-buf import + relayout (the DEVICE_COPY edge body, both ways)   */
/* ------------------------------------------------------------------ */

struct infer_wgpu_import {
    struct infer_ctx_wgpu* c;
    WGPUSharedTextureMemory mem;
    WGPUTexture tex;
    WGPUTextureView view;
    WGPUBuffer uniform;
    WGPUBindGroup bg;
    WGPUBuffer bg_buf; /* which buffer / pass the bind group was built for */
    int bg_pass;
    int writable;
    int layout; /* VkImageLayout the image is in when we next acquire it */
    uint32_t img_w, img_h;
};

struct infer_wgpu_import* infer_wgpu_import_create(struct infer_ctx_wgpu* c,
                                                   int dmabuf_fd,
                                                   const struct infer_desc* d,
                                                   int writable) {
    if (!c->has_dmabuf) {
        fprintf(stderr, "wgpu: SharedTextureMemoryDmaBuf unavailable\n");
        return NULL;
    }
    int errors0 = c->errors;
    struct infer_wgpu_import* im = calloc(1, sizeof *im);
    im->c = c;
    im->img_w = d->img_w;
    im->img_h = d->img_h;
    im->writable = writable;
    im->bg_pass = -1;
    /* an input's image is in whatever layout its producer released it
     * in (GENERAL, by the Vulkan writer's barrier); an output's has
     * never been used by anyone and its contents are irrelevant */
    im->layout = writable ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;

    WGPUSharedTextureMemoryDmaBufPlane planes[4];
    for (uint32_t p = 0; p < d->planes && p < 4; p++) {
        planes[p] = (WGPUSharedTextureMemoryDmaBufPlane){dmabuf_fd, d->plane_offset[p], d->plane_pitch[p]};
    }
    WGPUSharedTextureMemoryDmaBufDescriptor dma = WGPU_SHARED_TEXTURE_MEMORY_DMA_BUF_DESCRIPTOR_INIT;
    dma.size = (WGPUExtent3D){d->img_w, d->img_h, 1};
    dma.drmFormat = DRM_FORMAT_ABGR8888; /* byte 0 = R: a float's bytes in texel order */
    dma.drmModifier = d->drm_modifier;   /* whatever the owner's allocator chose */
    dma.planeCount = d->planes;
    dma.planes = planes;
    WGPUSharedTextureMemoryDescriptor sd = WGPU_SHARED_TEXTURE_MEMORY_DESCRIPTOR_INIT;
    sd.nextInChain = &dma.chain;
    sd.label = SV(writable ? "infer dma-buf output" : "infer dma-buf input");
    im->mem = wgpuDeviceImportSharedTextureMemory(c->device, &sd);
    WGPUSharedTextureMemoryProperties props = WGPU_SHARED_TEXTURE_MEMORY_PROPERTIES_INIT;
    if (!im->mem || wgpuSharedTextureMemoryGetProperties(im->mem, &props) != WGPUStatus_Success ||
        c->errors != errors0) {
        fprintf(stderr, "wgpu: dma-buf import failed\n");
        infer_wgpu_import_destroy(im);
        return NULL;
    }
    /* read: sampled by the relayout pass; write: stored by the reverse
     * pass (rgba8unorm is a core storage format) */
    WGPUTextureUsage want = writable ? (WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopyDst)
                                     : (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc);
    if ((props.usage & want) != want) {
        fprintf(stderr,
                "wgpu: dma-buf import does not allow %s (usage 0x%llx)\n",
                writable ? "storage writes" : "texture reads",
                (unsigned long long)props.usage);
        infer_wgpu_import_destroy(im);
        return NULL;
    }
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.usage = want;
    td.size = props.size;
    td.format = props.format;
    im->tex = wgpuSharedTextureMemoryCreateTexture(im->mem, &td);
    im->view = wgpuTextureCreateView(im->tex, NULL);

    WGPUBufferDescriptor ud = WGPU_BUFFER_DESCRIPTOR_INIT;
    ud.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ud.size = 16;
    im->uniform = wgpuDeviceCreateBuffer(c->device, &ud);
    uint32_t params[4] = {d->img_w, d->img_h, (uint32_t)infer_desc_elements(d), 0};
    wgpuQueueWriteBuffer(c->queue, im->uniform, 0, params, sizeof params);
    if (c->errors != errors0) {
        infer_wgpu_import_destroy(im);
        return NULL;
    }
    return im;
}

static int export_fence_fd(WGPUSharedFence f) {
    WGPUSharedFenceSyncFDExportInfo sfd = WGPU_SHARED_FENCE_SYNC_FD_EXPORT_INFO_INIT;
    WGPUSharedFenceExportInfo info = WGPU_SHARED_FENCE_EXPORT_INFO_INIT;
    info.nextInChain = &sfd.chain;
    wgpuSharedFenceExportInfo(f, &info);
    return info.type == WGPUSharedFenceType_SyncFD && sfd.handle >= 0 ? dup(sfd.handle) : -1;
}

int infer_wgpu_import_access(struct infer_ctx_wgpu* c,
                             struct infer_wgpu_import* im,
                             int acquire_fd,
                             enum infer_wgpu_pass pass,
                             WGPUBuffer buf,
                             int* release_fd) {
    *release_fd = -1;
    int errors0 = c->errors;
    INFER_CHECK((pass == INFER_WGPU_BUF_TO_TEX) == (im->writable != 0),
                "wgpu: import was created for the other direction");
    if (im->bg_buf != buf || im->bg_pass != (int)pass) {
        if (im->bg) { wgpuBindGroupRelease(im->bg); }
        /* the two passes bind the same three things -- the image, the
         * packed buffer, the size -- and differ only in who reads whom */
        WGPUComputePipeline pl = pass == INFER_WGPU_TEX_TO_BUF ? c->relayout_pipeline
                                                                : c->relayout_rev_pipeline;
        /* the packed buffer holds the elements, the image may hold more
         * texels: bind the buffer whole, the shaders guard on the count */
        WGPUBindGroupEntry entries[3] = {
            {.binding = 0, .textureView = im->view},
            {.binding = 1, .buffer = buf, .size = WGPU_WHOLE_SIZE},
            {.binding = 2, .buffer = im->uniform, .size = 16},
        };
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pl, 0);
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = bgl;
        bgd.entryCount = 3;
        bgd.entries = entries;
        im->bg = wgpuDeviceCreateBindGroup(c->device, &bgd);
        wgpuBindGroupLayoutRelease(bgl);
        im->bg_buf = buf;
        im->bg_pass = (int)pass;
    }

    /* acquire: the fence that gates our access -- the producer's "data
     * valid" for an input, the consumer's "done reading" for an output
     * -- as a SharedFence (Dawn dups the fd) */
    WGPUSharedFence fence = NULL;
    uint64_t one = 1;
    if (acquire_fd >= 0 && c->has_sync_fd) {
        WGPUSharedFenceSyncFDDescriptor fdd = WGPU_SHARED_FENCE_SYNC_FD_DESCRIPTOR_INIT;
        fdd.handle = acquire_fd;
        WGPUSharedFenceDescriptor fd = WGPU_SHARED_FENCE_DESCRIPTOR_INIT;
        fd.nextInChain = &fdd.chain;
        fence = wgpuDeviceImportSharedFence(c->device, &fd);
    } else if (acquire_fd >= 0) {
        infer_wait_sync_fd(acquire_fd, 1000); /* no fence import: host wait */
    }
    WGPUSharedTextureMemoryVkImageLayoutBeginState bl =
        WGPU_SHARED_TEXTURE_MEMORY_VK_IMAGE_LAYOUT_BEGIN_STATE_INIT;
    bl.oldLayout = im->layout;
    bl.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    WGPUSharedTextureMemoryBeginAccessDescriptor ba =
        WGPU_SHARED_TEXTURE_MEMORY_BEGIN_ACCESS_DESCRIPTOR_INIT;
    ba.nextInChain = &bl.chain;
    ba.initialized = 1; /* an output is overwritten whole: no lazy clear wanted */
    ba.fenceCount = fence ? 1 : 0;
    ba.fences = &fence;
    ba.signaledValueCount = fence ? 1 : 0;
    ba.signaledValues = &one;
    WGPUStatus bs = wgpuSharedTextureMemoryBeginAccess(im->mem, im->tex, &ba);
    if (fence) { wgpuSharedFenceRelease(fence); }
    INFER_CHECK(bs == WGPUStatus_Success, "wgpu: BeginAccess failed");

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(cp, pass == INFER_WGPU_TEX_TO_BUF ? c->relayout_pipeline
                                                                        : c->relayout_rev_pipeline);
    wgpuComputePassEncoderSetBindGroup(cp, 0, im->bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cp, (im->img_w + 15) / 16, (im->img_h + 15) / 16, 1);
    wgpuComputePassEncoderEnd(cp);
    wgpuComputePassEncoderRelease(cp);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    /* release: Dawn's fence for "done with the image" -- done reading
     * an input, done writing an output -- handed back as a sync file */
    WGPUSharedTextureMemoryVkImageLayoutEndState el =
        WGPU_SHARED_TEXTURE_MEMORY_VK_IMAGE_LAYOUT_END_STATE_INIT;
    WGPUSharedTextureMemoryEndAccessState ea = WGPU_SHARED_TEXTURE_MEMORY_END_ACCESS_STATE_INIT;
    ea.nextInChain = &el.chain;
    WGPUStatus es = wgpuSharedTextureMemoryEndAccess(im->mem, im->tex, &ea);
    if (es == WGPUStatus_Success && ea.fenceCount > 0) { *release_fd = export_fence_fd(ea.fences[0]); }
    wgpuSharedTextureMemoryEndAccessStateFreeMembers(ea);
    INFER_CHECK(es == WGPUStatus_Success, "wgpu: EndAccess failed");
    /* an output image is ours alone between accesses: remember the
     * layout Dawn left it in. An input's producer re-transitions it. */
    if (im->writable) { im->layout = el.newLayout ? el.newLayout : VK_IMAGE_LAYOUT_GENERAL; }
    if (getenv("INFER_WGPU_TRACE") && *release_fd >= 0) {
        /* how long after EndAccess does Dawn's release fence signal? */
        uint64_t w0 = infer_now_ns();
        int r = infer_wait_sync_fd(*release_fd, 200);
        fprintf(stderr, "  [wgpu] release fd signaled %s after EndAccess (%.1f us); layout %d -> %d\n",
                r == 0 ? "" : "NOT (200 ms timeout)", (double)(infer_now_ns() - w0) / 1e3,
                el.oldLayout, el.newLayout);
    }
    /* The release fence is the exportable semaphore Dawn signals from the
     * relayout submission itself (ImportedTextureBase::OnBeforeSubmit /
     * OnAfterSubmit in TextureVk.cpp), so it signals as soon as the
     * relayout finishes -- ~100 us after EndAccess on Honeykrisp when the
     * queue is otherwise idle. If it seems late, something submitted
     * BEFORE the relayout is still running (an engine whose Run()
     * returned before its GPU work completed); measure with
     * INFER_WGPU_TRACE=1 before blaming the fence. */
    return c->errors != errors0 ? -1 : 0;
}

static void on_work_done(WGPUQueueWorkDoneStatus st, WGPUStringView msg, void* u1, void* u2) {
    (void)msg;
    (void)u2;
    *(int*)u1 = (int)st;
}

int infer_wgpu_wait_idle(struct infer_ctx_wgpu* c) {
    int status = -1;
    WGPUQueueWorkDoneCallbackInfo ci = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    ci.mode = WGPUCallbackMode_WaitAnyOnly;
    ci.callback = on_work_done;
    ci.userdata1 = &status;
    WGPUFutureWaitInfo w = WGPU_FUTURE_WAIT_INFO_INIT;
    w.future = wgpuQueueOnSubmittedWorkDone(c->queue, ci);
    return wgpuInstanceWaitAny(c->instance, 1, &w, UINT64_MAX) == WGPUWaitStatus_Success &&
                   status == (int)WGPUQueueWorkDoneStatus_Success
               ? 0
               : -1;
}

void infer_wgpu_import_destroy(void* p) {
    struct infer_wgpu_import* im = p;
    if (!im) { return; }
    if (im->bg) { wgpuBindGroupRelease(im->bg); }
    if (im->uniform) { wgpuBufferRelease(im->uniform); }
    if (im->view) { wgpuTextureViewRelease(im->view); }
    if (im->tex) { wgpuTextureRelease(im->tex); }
    if (im->mem) { wgpuSharedTextureMemoryRelease(im->mem); }
    free(im);
}

/* ------------------------------------------------------------------ */
/* stage five: a camera frame imported for sampling                   */
/* ------------------------------------------------------------------ */

/* A camera's NV12 dma-buf imports as a biplanar texture. The measured
 * usage on this driver is CopySrc | TextureBinding only --
 * DawnMultiPlanarFormats is present, MultiPlanarFormatExtendedUsages is
 * not -- and the question that mattered was whether that is enough to
 * create per-plane SAMPLED views. Measured 2026-08-26 on Honeykrisp
 * (Mesa 26.1.7, M1): it is. `hello_hand --probe` reports it, because it
 * is a driver fact and the next driver may say otherwise.
 *
 * There is no device-side fallback, and that is a finding rather than an
 * omission. The obvious one -- copy the plane aspects into textures Dawn
 * owns -- is refused outright: CommandEncoder.cpp's
 * APICopyTextureToTexture has a blanket "Copying between a multiplanar
 * texture and another texture is currently not allowed" on either side
 * of the copy. CopyTextureToBuffer carries no such rule, so a driver
 * that refused the views could still be served by copying each plane
 * into a buffer and sampling buffers instead of textures -- at the cost
 * of a second shader that no driver here can exercise. An untested rung
 * is worse than the host path (convert on the CPU, wgpuQueueWriteBuffer:
 * the registry's cpu -> wgpu row), which every CPU-EP run exercises, so
 * this returns NULL and lets the caller take that. */
struct infer_wgpu_frame_import {
    struct infer_ctx_wgpu* c;
    WGPUSharedTextureMemory mem;
    WGPUTexture tex;          /* the imported frame */
    WGPUTextureView plane[2]; /* what the caller's pass samples */
    int layout;               /* VkImageLayout the image is in when we next acquire it */
    uint32_t w, h;
    int planes;
    int in_access;
};

/* The per-plane format of a biplanar frame; single-plane frames are
 * sampled through the format they were imported with. */
static WGPUTextureFormat plane_format(int plane) {
    return plane == 0 ? WGPUTextureFormat_R8Unorm : WGPUTextureFormat_RG8Unorm;
}

static int make_plane_views(struct infer_wgpu_frame_import* im, WGPUTextureFormat tex_format) {
    int errors0 = im->c->errors;
    for (int p = 0; p < im->planes; p++) {
        WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.mipLevelCount = 1;
        vd.arrayLayerCount = 1;
        if (im->planes > 1) {
            /* a plane of a multiplanar texture: the aspect selects it and
             * the view's format is the PLANE's, not the texture's */
            vd.aspect = p == 0 ? WGPUTextureAspect_Plane0Only : WGPUTextureAspect_Plane1Only;
            vd.format = plane_format(p);
        } else {
            vd.aspect = WGPUTextureAspect_All;
            vd.format = tex_format;
        }
        im->plane[p] = wgpuTextureCreateView(im->tex, &vd);
        if (!im->plane[p] || im->c->errors != errors0) { return -1; }
    }
    return 0;
}

struct infer_wgpu_frame_import* infer_wgpu_frame_import_create(struct infer_ctx_wgpu* c,
                                                               const struct infer_frame* f) {
    if (!c->has_dmabuf) {
        fprintf(stderr, "wgpu: SharedTextureMemoryDmaBuf unavailable\n");
        return NULL;
    }
    int errors0 = c->errors;
    struct infer_wgpu_frame_import* im = calloc(1, sizeof *im);
    im->c = c;
    im->w = f->width;
    im->h = f->height;
    im->planes = f->planes < 1 ? 1 : f->planes > 2 ? 2 : f->planes;
    /* the frame's producer (the ISP, through V4L2) never told Vulkan
     * about a layout: whatever Dawn finds, it must treat as undefined
     * the first time and as what it left behind after that */
    im->layout = VK_IMAGE_LAYOUT_UNDEFINED;

    WGPUSharedTextureMemoryDmaBufPlane planes[4];
    for (int p = 0; p < im->planes; p++) {
        planes[p] =
            (WGPUSharedTextureMemoryDmaBufPlane){f->dmabuf_fd[0], f->offset[p], f->pitch[p]};
    }
    WGPUSharedTextureMemoryDmaBufDescriptor dma = WGPU_SHARED_TEXTURE_MEMORY_DMA_BUF_DESCRIPTOR_INIT;
    dma.size = (WGPUExtent3D){f->width, f->height, 1};
    dma.drmFormat = f->drm_format;
    dma.drmModifier = f->drm_modifier;
    dma.planeCount = (size_t)im->planes;
    dma.planes = planes;
    WGPUSharedTextureMemoryDescriptor sd = WGPU_SHARED_TEXTURE_MEMORY_DESCRIPTOR_INIT;
    sd.nextInChain = &dma.chain;
    sd.label = SV("infer camera frame");
    im->mem = wgpuDeviceImportSharedTextureMemory(c->device, &sd);
    WGPUSharedTextureMemoryProperties props = WGPU_SHARED_TEXTURE_MEMORY_PROPERTIES_INIT;
    if (!im->mem || wgpuSharedTextureMemoryGetProperties(im->mem, &props) != WGPUStatus_Success ||
        c->errors != errors0) {
        fprintf(stderr, "wgpu: frame dma-buf import failed (fourcc 0x%08x)\n", f->drm_format);
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }
    if (!(props.usage & WGPUTextureUsage_CopySrc)) {
        fprintf(stderr,
                "wgpu: frame import grants neither reads nor copies (usage 0x%llx)\n",
                (unsigned long long)props.usage);
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }

    if (!(props.usage & WGPUTextureUsage_TextureBinding)) {
        fprintf(stderr,
                "wgpu: frame import is not readable by a shader (usage 0x%llx); "
                "the caller must convert on the host\n",
                (unsigned long long)props.usage);
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    /* ask for exactly what the import granted, no more: requesting a
     * usage the memory does not allow is an error */
    td.usage = props.usage & (WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding);
    td.dimension = WGPUTextureDimension_2D;
    td.size = props.size;
    td.format = props.format;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    im->tex = wgpuSharedTextureMemoryCreateTexture(im->mem, &td);
    if (!im->tex || c->errors != errors0) {
        fprintf(stderr, "wgpu: frame texture creation failed\n");
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }
    if (make_plane_views(im, props.format) < 0) {
        fprintf(stderr,
                "wgpu: this driver refuses per-plane views of an imported %s frame; "
                "the caller must convert on the host\n",
                im->planes > 1 ? "multiplanar" : "single-plane");
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }
    if (infer_verbose) {
        fprintf(stderr,
                "wgpu: frame %ux%u fourcc 0x%08x %d plane(s), usage 0x%llx, plane views ok\n",
                f->width,
                f->height,
                f->drm_format,
                im->planes,
                (unsigned long long)props.usage);
    }
    if (c->errors != errors0) {
        infer_wgpu_frame_import_destroy(im);
        return NULL;
    }
    return im;
}

WGPUTextureView infer_wgpu_frame_plane(const struct infer_wgpu_frame_import* im, int plane) {
    return plane >= 0 && plane < im->planes ? im->plane[plane] : NULL;
}

int infer_wgpu_frame_begin(struct infer_ctx_wgpu* c,
                           struct infer_wgpu_frame_import* im,
                           int acquire_fd) {
    int errors0 = c->errors;
    INFER_CHECK(!im->in_access, "wgpu: frame import already in an access bracket");

    WGPUSharedFence fence = NULL;
    uint64_t one = 1;
    if (acquire_fd >= 0 && c->has_sync_fd) {
        WGPUSharedFenceSyncFDDescriptor fdd = WGPU_SHARED_FENCE_SYNC_FD_DESCRIPTOR_INIT;
        fdd.handle = acquire_fd;
        WGPUSharedFenceDescriptor fd = WGPU_SHARED_FENCE_DESCRIPTOR_INIT;
        fd.nextInChain = &fdd.chain;
        fence = wgpuDeviceImportSharedFence(c->device, &fd);
    } else if (acquire_fd >= 0) {
        infer_wait_sync_fd(acquire_fd, 1000); /* no fence import: host wait */
    }
    WGPUSharedTextureMemoryVkImageLayoutBeginState bl =
        WGPU_SHARED_TEXTURE_MEMORY_VK_IMAGE_LAYOUT_BEGIN_STATE_INIT;
    bl.oldLayout = im->layout;
    bl.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    WGPUSharedTextureMemoryBeginAccessDescriptor ba =
        WGPU_SHARED_TEXTURE_MEMORY_BEGIN_ACCESS_DESCRIPTOR_INIT;
    ba.nextInChain = &bl.chain;
    ba.initialized = 1; /* the ISP filled it; no lazy clear wanted */
    ba.concurrentRead = 0;
    ba.fenceCount = fence ? 1 : 0;
    ba.fences = &fence;
    ba.signaledValueCount = fence ? 1 : 0;
    ba.signaledValues = &one;
    WGPUStatus bs = wgpuSharedTextureMemoryBeginAccess(im->mem, im->tex, &ba);
    if (fence) { wgpuSharedFenceRelease(fence); }
    INFER_CHECK(bs == WGPUStatus_Success, "wgpu: frame BeginAccess failed");
    im->in_access = 1;
    return c->errors != errors0 ? -1 : 0;
}

int infer_wgpu_frame_end(struct infer_ctx_wgpu* c,
                         struct infer_wgpu_frame_import* im,
                         int* release_fd) {
    *release_fd = -1;
    int errors0 = c->errors;
    INFER_CHECK(im->in_access, "wgpu: frame import is not in an access bracket");
    WGPUSharedTextureMemoryVkImageLayoutEndState el =
        WGPU_SHARED_TEXTURE_MEMORY_VK_IMAGE_LAYOUT_END_STATE_INIT;
    WGPUSharedTextureMemoryEndAccessState ea = WGPU_SHARED_TEXTURE_MEMORY_END_ACCESS_STATE_INIT;
    ea.nextInChain = &el.chain;
    WGPUStatus es = wgpuSharedTextureMemoryEndAccess(im->mem, im->tex, &ea);
    if (es == WGPUStatus_Success && ea.fenceCount > 0) { *release_fd = export_fence_fd(ea.fences[0]); }
    wgpuSharedTextureMemoryEndAccessStateFreeMembers(ea);
    im->in_access = 0;
    INFER_CHECK(es == WGPUStatus_Success, "wgpu: frame EndAccess failed");
    /* We only ever read the frame, so the layout Dawn leaves it in is
     * what the next acquire must declare. The producer is V4L2, which
     * knows nothing of layouts and does not re-transition it. */
    im->layout = el.newLayout ? el.newLayout : VK_IMAGE_LAYOUT_GENERAL;
    return c->errors != errors0 ? -1 : 0;
}

void infer_wgpu_frame_import_destroy(void* p) {
    struct infer_wgpu_frame_import* im = p;
    if (!im) { return; }
    for (int i = 0; i < 2; i++) {
        if (im->plane[i]) { wgpuTextureViewRelease(im->plane[i]); }
    }
    if (im->tex) { wgpuTextureRelease(im->tex); }
    if (im->mem) { wgpuSharedTextureMemoryRelease(im->mem); }
    free(im);
}

/* ------------------------------------------------------------------ */
/* stage five: the readback, polled instead of waited on              */
/* ------------------------------------------------------------------ */

enum { RB_IDLE = 0, RB_MAPPING = 1, RB_READY = 2, RB_FAILED = 3 };

struct infer_wgpu_readback {
    WGPUBuffer staging;
    size_t capacity; /* allocated, 4-byte aligned */
    size_t bytes;    /* what the pending submit asked for */
    int state;
};

static void on_map_state(WGPUMapAsyncStatus st, WGPUStringView msg, void* u1, void* u2) {
    (void)u2;
    if (st != WGPUMapAsyncStatus_Success) { fprintf(stderr, "wgpu map failed: %.*s\n", SV_ARGS(msg)); }
    *(int*)u1 = st == WGPUMapAsyncStatus_Success ? RB_READY : RB_FAILED;
}

struct infer_wgpu_readback* infer_wgpu_readback_create(struct infer_ctx_wgpu* c, size_t bytes) {
    struct infer_wgpu_readback* rb = calloc(1, sizeof *rb);
    rb->capacity = infer_align(bytes, 4);
    WGPUBufferDescriptor sd = WGPU_BUFFER_DESCRIPTOR_INIT;
    sd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    sd.size = rb->capacity;
    sd.label = SV("infer polled readback");
    rb->staging = wgpuDeviceCreateBuffer(c->device, &sd);
    if (!rb->staging) {
        free(rb);
        return NULL;
    }
    return rb;
}

int infer_wgpu_readback_inflight(const struct infer_wgpu_readback* rb) {
    return rb->state != RB_IDLE;
}

int infer_wgpu_readback_submit(struct infer_ctx_wgpu* c,
                              struct infer_wgpu_readback* rb,
                              WGPUBuffer src,
                              size_t bytes) {
    int errors0 = c->errors;
    /* A MapRead buffer cannot be re-mapped while mapped, and nothing may
     * be copied into it while mapped: the previous result must have been
     * drained by poll() first. */
    INFER_CHECK(rb->state == RB_IDLE, "wgpu: readback already in flight");
    size_t n = infer_align(bytes, 4);
    INFER_CHECK(n <= rb->capacity, "wgpu: readback is %zu bytes, needs %zu", rb->capacity, n);
    rb->bytes = bytes;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, rb->staging, 0, n);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    rb->state = RB_MAPPING;
    WGPUBufferMapCallbackInfo mi = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    /* AllowProcessEvents, not WaitAnyOnly: the callback must be able to
     * fire from wgpuInstanceProcessEvents in poll() rather than from a
     * blocking wait -- that is the whole point of this object */
    mi.mode = WGPUCallbackMode_AllowProcessEvents;
    mi.callback = on_map_state;
    mi.userdata1 = &rb->state;
    wgpuBufferMapAsync(rb->staging, WGPUMapMode_Read, 0, n, mi);
    return c->errors != errors0 ? -1 : 0;
}

int infer_wgpu_readback_poll(struct infer_ctx_wgpu* c, struct infer_wgpu_readback* rb, float* dst) {
    if (rb->state == RB_IDLE) { return 0; }
    if (rb->state == RB_MAPPING) {
        wgpuInstanceProcessEvents(c->instance);
        if (rb->state == RB_MAPPING) { return 0; } /* still on the GPU */
    }
    if (rb->state == RB_FAILED) {
        rb->state = RB_IDLE;
        return -1;
    }
    const void* p = wgpuBufferGetConstMappedRange(rb->staging, 0, infer_align(rb->bytes, 4));
    if (p) { memcpy(dst, p, rb->bytes); }
    wgpuBufferUnmap(rb->staging);
    rb->state = RB_IDLE;
    return p ? 1 : -1;
}

void infer_wgpu_readback_destroy(struct infer_wgpu_readback* rb) {
    if (!rb) { return; }
    if (rb->state == RB_READY) { wgpuBufferUnmap(rb->staging); }
    if (rb->staging) { wgpuBufferRelease(rb->staging); }
    free(rb);
}
