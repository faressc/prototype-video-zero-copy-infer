/* infer_ctx_wgpu.c -- the WebGPU domain on Dawn: the one device this
 * process shares with ONNX Runtime's WebGPU EP, the generator pass, the
 * MapRead readback, and the dma-buf import + relayout that is the
 * DEVICE_COPY edge from OpenGL/Vulkan tensors.
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
    bd.size = infer_align((size_t)bytes, 4);
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
    INFER_CHECK(c->gen_pipeline && c->relayout_pipeline, "wgpu: pipeline creation failed");
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
    if (c->queue) { wgpuQueueRelease(c->queue); }
    if (c->device) { wgpuDeviceRelease(c->device); }
    if (c->adapter) { wgpuAdapterRelease(c->adapter); }
    if (c->instance) { wgpuInstanceRelease(c->instance); }
    memset(c, 0, sizeof *c);
}

/* ------------------------------------------------------------------ */
/* producer + readback                                                */
/* ------------------------------------------------------------------ */

int infer_wgpu_gen(struct infer_ctx_wgpu* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t) {
    size_t n = infer_desc_elements(d);
    if (!t->mem.wgpu.buffer) {
        t->domain = INFER_DOMAIN_WGPU;
        t->desc = *d;
        t->owned = 1;
        t->ready.sync_fd = -1;
        t->released.sync_fd = -1;
        t->mem.wgpu.buffer = infer_wgpu_create_buffer(c, n * sizeof(float), 0);
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
        INFER_CHECK(t->mem.wgpu.buffer && t->mem.wgpu.uniform && t->mem.wgpu.bind_group,
                    "wgpu: generator resources");
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
    return c->errors ? -1 : 0;
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
/* dma-buf import + relayout (the DEVICE_COPY edge body)              */
/* ------------------------------------------------------------------ */

struct infer_wgpu_import {
    struct infer_ctx_wgpu* c;
    WGPUSharedTextureMemory mem;
    WGPUTexture tex;
    WGPUTextureView view;
    WGPUBuffer uniform;
    WGPUBindGroup bg;
    WGPUBuffer bg_dst; /* which dst the bind group was built for */
    uint32_t img_w, img_h;
};

struct infer_wgpu_import* infer_wgpu_import_create(struct infer_ctx_wgpu* c,
                                                   int dmabuf_fd,
                                                   const struct infer_desc* d) {
    if (!c->has_dmabuf) {
        fprintf(stderr, "wgpu: SharedTextureMemoryDmaBuf unavailable\n");
        return NULL;
    }
    struct infer_wgpu_import* im = calloc(1, sizeof *im);
    im->c = c;
    im->img_w = d->img_w;
    im->img_h = d->img_h;

    WGPUSharedTextureMemoryDmaBufPlane planes[4];
    for (uint32_t p = 0; p < d->planes && p < 4; p++) {
        planes[p] = (WGPUSharedTextureMemoryDmaBufPlane){dmabuf_fd, d->plane_offset[p], d->plane_pitch[p]};
    }
    WGPUSharedTextureMemoryDmaBufDescriptor dma = WGPU_SHARED_TEXTURE_MEMORY_DMA_BUF_DESCRIPTOR_INIT;
    dma.size = (WGPUExtent3D){d->img_w, d->img_h, 1};
    dma.drmFormat = DRM_FORMAT_ABGR8888; /* byte 0 = R: a float's bytes in texel order */
    dma.drmModifier = d->drm_modifier;   /* whatever the producer's allocator chose */
    dma.planeCount = d->planes;
    dma.planes = planes;
    WGPUSharedTextureMemoryDescriptor sd = WGPU_SHARED_TEXTURE_MEMORY_DESCRIPTOR_INIT;
    sd.nextInChain = &dma.chain;
    sd.label = SV("infer dma-buf tensor");
    im->mem = wgpuDeviceImportSharedTextureMemory(c->device, &sd);
    WGPUSharedTextureMemoryProperties props = WGPU_SHARED_TEXTURE_MEMORY_PROPERTIES_INIT;
    if (!im->mem || wgpuSharedTextureMemoryGetProperties(im->mem, &props) != WGPUStatus_Success ||
        c->errors) {
        fprintf(stderr, "wgpu: dma-buf import failed\n");
        infer_wgpu_import_destroy(im);
        return NULL;
    }
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.usage = props.usage & (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc);
    td.size = props.size;
    td.format = props.format;
    im->tex = wgpuSharedTextureMemoryCreateTexture(im->mem, &td);
    im->view = wgpuTextureCreateView(im->tex, NULL);

    WGPUBufferDescriptor ud = WGPU_BUFFER_DESCRIPTOR_INIT;
    ud.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ud.size = 16;
    im->uniform = wgpuDeviceCreateBuffer(c->device, &ud);
    uint32_t params[4] = {d->img_w, d->img_h, 0, 0};
    wgpuQueueWriteBuffer(c->queue, im->uniform, 0, params, sizeof params);
    if (c->errors) {
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

int infer_wgpu_import_relayout(struct infer_ctx_wgpu* c,
                               struct infer_wgpu_import* im,
                               int acquire_fd,
                               WGPUBuffer dst,
                               int* release_fd) {
    *release_fd = -1;
    if (im->bg_dst != dst) {
        if (im->bg) { wgpuBindGroupRelease(im->bg); }
        WGPUBindGroupEntry entries[3] = {
            {.binding = 0, .textureView = im->view},
            {.binding = 1, .buffer = dst, .size = (uint64_t)im->img_w * im->img_h * 4},
            {.binding = 2, .buffer = im->uniform, .size = 16},
        };
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(c->relayout_pipeline, 0);
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = bgl;
        bgd.entryCount = 3;
        bgd.entries = entries;
        im->bg = wgpuDeviceCreateBindGroup(c->device, &bgd);
        wgpuBindGroupLayoutRelease(bgl);
        im->bg_dst = dst;
    }

    /* acquire: the producer's completion as a SharedFence (Dawn dups the fd) */
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
    bl.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    bl.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    WGPUSharedTextureMemoryBeginAccessDescriptor ba =
        WGPU_SHARED_TEXTURE_MEMORY_BEGIN_ACCESS_DESCRIPTOR_INIT;
    ba.nextInChain = &bl.chain;
    ba.initialized = 1;
    ba.fenceCount = fence ? 1 : 0;
    ba.fences = &fence;
    ba.signaledValueCount = fence ? 1 : 0;
    ba.signaledValues = &one;
    WGPUStatus bs = wgpuSharedTextureMemoryBeginAccess(im->mem, im->tex, &ba);
    if (fence) { wgpuSharedFenceRelease(fence); }
    INFER_CHECK(bs == WGPUStatus_Success, "wgpu: BeginAccess failed");

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, c->relayout_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, im->bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (im->img_w + 15) / 16, (im->img_h + 15) / 16, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(c->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    /* release: Dawn's fence for "done reading", handed back as a sync file */
    WGPUSharedTextureMemoryVkImageLayoutEndState el =
        WGPU_SHARED_TEXTURE_MEMORY_VK_IMAGE_LAYOUT_END_STATE_INIT;
    WGPUSharedTextureMemoryEndAccessState ea = WGPU_SHARED_TEXTURE_MEMORY_END_ACCESS_STATE_INIT;
    ea.nextInChain = &el.chain;
    WGPUStatus es = wgpuSharedTextureMemoryEndAccess(im->mem, im->tex, &ea);
    if (es == WGPUStatus_Success && ea.fenceCount > 0) { *release_fd = export_fence_fd(ea.fences[0]); }
    wgpuSharedTextureMemoryEndAccessStateFreeMembers(ea);
    INFER_CHECK(es == WGPUStatus_Success, "wgpu: EndAccess failed");
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
    return c->errors ? -1 : 0;
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
