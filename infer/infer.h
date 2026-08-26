/* infer.h -- typeless tensors across memory domains, a probed edge
 * registry that moves them between domains at an explicit cost, and an
 * ONNX Runtime adapter that consumes them. Stage four (README §20).
 *
 * The shape is anira v3's: a tensor is a memory handle in some DOMAIN
 * (CPU, OpenGL ES, Vulkan, WebGPU) plus a descriptor that is the only
 * type the memory has -- no pixel formats, no strides in disguise. An
 * EDGE converts a tensor from one domain to another and is tagged with
 * what it costs: a handle hand-over (ZERO_COPY), one GPU pass or copy
 * (DEVICE_COPY), or a trip through host memory (HOST_COPY). Which edges
 * exist is decided by PROBING the drivers, never by the platform name.
 * The engine adapter says which domain it wants its input in and which
 * domain it produces its outputs in; the caller asks the registry for
 * both conversions -- producer -> engine input, engine output ->
 * consumer -- and gets told the price of each. Completion is a token
 * on the consumer's tensor, never the engine call returning.
 *
 * Domains and their handles:
 *   CPU   malloc'd floats
 *   GL    a gbm_bo (linear dma-buf) imported as an EGLImage texture
 *   VK    exportable VkDeviceMemory bound to a linear DRM-modifier
 *         VkImage and, when the driver allows, to a VkBuffer alias
 *   WGPU  a WGPUBuffer on the one Dawn device the process shares with
 *         ONNX Runtime's WebGPU execution provider
 *   DMABUF a dma-buf fd that no API in this process owns -- the shape a
 *         camera, decoder or compositor buffer arrives in (here: a
 *         dma-heap allocation the CPU fills, so exactness still checks)
 *
 * The GL and VK tensors carry a second view of the same bytes: a 2D
 * "byte image" of img_w x img_h RGBA8 texels where each texel holds the
 * four little-endian bytes of one float. That is the currency Dawn's
 * dma-buf import accepts (textures only, 8/16-bit formats only), so a
 * float tensor travels into WebGPU as an RGBA8 image and one WGSL pass
 * turns texels back into a packed float buffer -- the DEVICE_COPY. The
 * same import runs the other way for outputs: Dawn writes the engine's
 * packed float buffer into a byte image the consumer owns, and hands
 * the consumer a fence.
 */
#ifndef HELLO_WAYLAND_INFER_H
#define HELLO_WAYLAND_INFER_H

#include <stddef.h>
#include <stdint.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>
#include <vulkan/vulkan.h>
#include <webgpu/webgpu.h>

/* ------------------------------------------------------------------ */
/* vocabulary                                                         */
/* ------------------------------------------------------------------ */

enum infer_domain {
    INFER_DOMAIN_CPU,
    INFER_DOMAIN_GL,
    INFER_DOMAIN_VK,
    INFER_DOMAIN_WGPU,
    INFER_DOMAIN_DMABUF,
    INFER_DOMAIN_COUNT,
};

enum infer_ep {
    INFER_EP_CPU,
    INFER_EP_WEBGPU,
    INFER_EP_COUNT,
};

enum infer_cost {
    INFER_COST_ZERO_COPY,   /* handle hand-over or memory import; no data moves */
    INFER_COST_DEVICE_COPY, /* one GPU pass or copy-engine op */
    INFER_COST_HOST_COPY,   /* readback and/or upload through host memory */
    INFER_COST_UNAVAILABLE, /* the probe said no */
};

/* per-allocation chatter (the "vk: tensor ..." / "gl: tensor ..." lines)
 * only when set; bring-up lines print regardless */
extern int infer_verbose;

const char* infer_domain_name(enum infer_domain d);
const char* infer_ep_name(enum infer_ep ep);
const char* infer_cost_name(enum infer_cost c);
int infer_domain_parse(const char* s, enum infer_domain* out);
int infer_ep_parse(const char* s, enum infer_ep* out);

/* ------------------------------------------------------------------ */
/* the tensor                                                         */
/* ------------------------------------------------------------------ */

enum { INFER_MAX_DIMS = 8 };

/* float32 only for now; dtype is here so the descriptor stays the one
 * place a tensor's type lives */
enum infer_dtype { INFER_F32 };

struct infer_desc {
    enum infer_dtype dtype;
    int ndim;
    int64_t dims[INFER_MAX_DIMS]; /* the model's order (NHWC for ours) */
    /* the byte-image view used by the dma-buf edges: img_h rows of
     * img_w floats, img_w a multiple of 16 (64-byte rows, what linear-
     * image importers accept), img_w * img_h >= the element count --
     * the tail past it is never data. Packed tensors have
     * row_pitch_bytes == img_w * 4. */
    uint32_t img_w, img_h;
    uint32_t row_pitch_bytes; /* meaningful for linear memory only */
    /* how the exported dma-buf is laid out: the DRM modifier the
     * allocator chose (tiled on Apple GPUs -- a linear image cannot be
     * written by shaders there without a shadow copy) and the planes
     * the importer must be told about */
    uint64_t drm_modifier;
    uint32_t planes;
    uint32_t plane_offset[4], plane_pitch[4];
};

size_t infer_desc_elements(const struct infer_desc* d);
size_t infer_desc_bytes_packed(const struct infer_desc* d); /* elements * 4 */
size_t infer_desc_bytes_image(const struct infer_desc* d);  /* img_w * img_h * 4 */
/* Fill img_w/img_h/row_pitch (packed, rows 64-byte aligned) from dims:
 * the tensor's own rows when the tensor is [1, H, ...] and they are
 * aligned, else an exact aligned factorisation, else a padded tail. */
void infer_desc_set_image(struct infer_desc* d);

/* A cross-API completion signal. The sync file (dma-fence fd) is the
 * lingua franca on Linux: Vulkan exports/imports it as a semaphore, EGL
 * as a native fence sync, Dawn as a SharedFence. -1 = already signaled
 * / nothing to wait for. */
struct infer_sync {
    int sync_fd; /* owned; close() on reset */
};
void infer_sync_reset(struct infer_sync* s);
/* Take ownership of fd (may be -1), dropping what was there. */
void infer_sync_set(struct infer_sync* s, int fd);

struct infer_mem_cpu {
    float* ptr;
};
struct infer_mem_gl {
    struct gbm_bo* bo;
    int dmabuf_fd; /* owned dup of the bo's fd */
    EGLImageKHR image;
    GLuint rbo; /* the bo as a renderbuffer: what the generator draws into */
    GLuint fbo;
};
struct infer_mem_vk {
    VkDeviceMemory memory;
    VkImage image;            /* linear, DRM_FORMAT_MOD_LINEAR, R8G8B8A8_UNORM */
    VkBuffer buffer;          /* alias over the same memory; VK_NULL_HANDLE if not allowed */
    VkImageView storage_view; /* for the imageStore fallback writer */
    VkDeviceSize size;
    int dmabuf_fd;       /* owned */
    int host_visible;    /* memory type is HOST_VISIBLE|HOST_COHERENT */
    uint32_t mem_type;   /* the memory type index the allocation came from */
    int alias_ok;        /* buffer alias exists and is the writer's target */
    VkSemaphore done;    /* signaled by the last write, SYNC_FD exportable */
    VkSemaphore release; /* imported from the consumer's fence, waited by the next write */
    VkFence fence;
    int first_write;
};
struct infer_mem_wgpu {
    WGPUBuffer buffer;
    WGPUBuffer uniform;       /* generator parameters */
    WGPUBindGroup bind_group; /* generator bind group over buffer + uniform */
};
struct infer_mem_dmabuf {
    int fd;      /* owned; from /dev/dma_heap/system here */
    void* map;   /* mmap of the whole buffer (CPU side of a UMA machine) */
    size_t size; /* bytes allocated (page rounded) */
};

struct infer_tensor {
    enum infer_domain domain;
    struct infer_desc desc;
    union {
        struct infer_mem_cpu cpu;
        struct infer_mem_gl gl;
        struct infer_mem_vk vk;
        struct infer_mem_wgpu wgpu;
        struct infer_mem_dmabuf dmabuf;
    } mem;
    struct infer_sync ready;    /* producer -> consumer: data valid once signaled */
    struct infer_sync released; /* consumer -> producer: may overwrite once signaled */
    int owned;
    /* PRODUCER-private state (the Vulkan writer's descriptor set); freed
     * with the tensor. Never edge state: an edge's cached import lives in
     * the caller's infer_edge_cache, keyed by the tensor it imported, so
     * a tensor can be an edge source and an edge destination in the same
     * plan without the two uses fighting over one slot. */
    void* priv;
    void (*priv_free)(void* p);
};

/* A camera/decoder frame -- image data with a pixel format. NOT a
 * tensor and never convertible into one by an edge: an edge preserves
 * what the bytes MEAN and changes where they live, while frame -> tensor
 * changes the meaning (pixels in a colour space become normalised NHWC
 * floats). That is a PASS, and it is stage five's business
 * (hand/hand_frame_*.c). */
struct infer_frame {
    int dmabuf_fd[4];
    uint32_t offset[4], pitch[4];
    uint64_t drm_modifier;
    uint32_t drm_format; /* fourcc */
    uint32_t width, height;
    uint8_t planes;
    /* Which YUV->RGB matrix and which quantisation range these bytes
     * were captured with. A property of the FRAME, not of whoever
     * consumes it -- otherwise every consumer re-derives it from the
     * driver and they drift. */
    uint8_t bt709, full_range;
    /* A CPU mapping when the owner has one (V4L2 MMAP buffers do, via
     * camera_map()); NULL for a device-only frame. The CPU FrameToTensor
     * reads this, the WebGPU one imports dmabuf_fd instead. */
    void* map;
    struct infer_sync ready;
};

/* ------------------------------------------------------------------ */
/* contexts: one per GPU API, owned (we create) or borrowed            */
/* ------------------------------------------------------------------ */

/* The DRM format modifiers this GPU's Vulkan driver can import as an
 * R8G8B8A8_UNORM image -- which is what Dawn's Vulkan backend accepts.
 * Brings up its own throwaway instance, so a GL-only context can ask
 * without a Vulkan domain being initialised. Returns the count. */
int infer_vk_importable_modifiers(uint64_t* out, int max);

struct infer_ctx_gl {
    int owned;
    int drm_fd;
    struct gbm_device* gbm;
    EGLDisplay dpy;
    EGLConfig cfg;
    EGLContext ctx;
    PFNEGLCREATEIMAGEKHRPROC create_image;
    PFNEGLDESTROYIMAGEKHRPROC destroy_image;
    PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC image_target_rbo;
    PFNEGLCREATESYNCKHRPROC create_sync;
    PFNEGLDESTROYSYNCKHRPROC destroy_sync;
    PFNEGLWAITSYNCKHRPROC wait_sync;
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fence_fd;
    int has_dmabuf_import, has_modifiers, has_native_fence;
    /* layouts both renderable here and importable by Vulkan (so by
     * Dawn), preferred first; see negotiate_modifier. Empty = let the
     * driver choose. Tried per allocation: a layout can be refused for
     * a particular size. */
    uint64_t bo_modifiers[64];
    int bo_modifier_count;
    int bo_negotiated; /* the list is only meaningful once this is set */
    GLuint gen_program;
    GLuint upload_tex, upload_fbo; /* host -> bo: texture upload + blit (lazily made) */
};

struct infer_ctx_vk {
    int owned;
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkDescriptorSetLayout buf_set_layout, img_set_layout;
    VkPipelineLayout buf_layout, img_layout;
    VkPipeline gen_buf_pipeline, gen_img_pipeline;
    VkDescriptorPool pool;
    PFN_vkGetMemoryFdKHR get_memory_fd;
    PFN_vkGetSemaphoreFdKHR get_semaphore_fd;
    PFN_vkImportSemaphoreFdKHR import_semaphore_fd;
    int has_dmabuf_export, has_drm_modifier, has_sync_fd, has_foreign_queue;
};

struct infer_ctx_wgpu {
    int owned;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUComputePipeline gen_pipeline;
    WGPUComputePipeline relayout_pipeline;     /* byte image -> packed floats */
    WGPUComputePipeline relayout_rev_pipeline; /* packed floats -> byte image */
    int has_dmabuf, has_sync_fd;
    int errors; /* uncaptured errors seen (the callback counts them) */
};

struct infer_ctx {
    struct infer_ctx_gl gl;
    struct infer_ctx_vk vk;
    struct infer_ctx_wgpu wgpu;
    unsigned have; /* bit per INFER_DOMAIN_* that initialised */
};

enum {
    INFER_WANT_GL = 1u << INFER_DOMAIN_GL,
    INFER_WANT_VK = 1u << INFER_DOMAIN_VK,
    INFER_WANT_WGPU = 1u << INFER_DOMAIN_WGPU,
    INFER_WANT_DMABUF = 1u << INFER_DOMAIN_DMABUF, /* needs /dev/dma_heap/system */
};

int infer_ctx_gl_init(struct infer_ctx_gl* c, const char* render_node);
int infer_ctx_gl_borrow(struct infer_ctx_gl* c,
                        EGLDisplay dpy,
                        EGLContext ctx,
                        struct gbm_device* gbm);
void infer_ctx_gl_fini(struct infer_ctx_gl* c);

int infer_ctx_vk_init(struct infer_ctx_vk* c);
int infer_ctx_vk_borrow(struct infer_ctx_vk* c,
                        VkInstance instance,
                        VkPhysicalDevice phys,
                        VkDevice device,
                        uint32_t queue_family,
                        VkQueue queue);
void infer_ctx_vk_fini(struct infer_ctx_vk* c);

int infer_ctx_wgpu_init(struct infer_ctx_wgpu* c);
int infer_ctx_wgpu_borrow(struct infer_ctx_wgpu* c, WGPUInstance instance, WGPUDevice device);
void infer_ctx_wgpu_fini(struct infer_ctx_wgpu* c);

/* Best effort: bring up what `want` asks for; c->have says what came up. */
int infer_ctx_init_all(struct infer_ctx* c, unsigned want);
void infer_ctx_fini_all(struct infer_ctx* c);

/* ------------------------------------------------------------------ */
/* producers and readback                                             */
/* ------------------------------------------------------------------ */

/* The deterministic generator, identical in C and all three shader
 * languages: f(i) for flat element index i. */
float infer_gen_value(uint32_t i, uint32_t seed);

/* Allocate a tensor in domain d (out zeroed) without filling it: what a
 * consumer does to own the destination of an output edge (the doc's
 * bind_output), and what the harness does for the engine's own output
 * buffers. No-op if out already holds memory. */
int infer_tensor_alloc(struct infer_ctx* c,
                       enum infer_domain d,
                       const struct infer_desc* desc,
                       struct infer_tensor* out);

/* "Get a tensor from X": allocate (first call, out zeroed) or reuse
 * (later calls) a tensor in domain d and fill it ON that domain with
 * infer_gen_value. Sets out->ready. */
int infer_tensor_gen(struct infer_ctx* c,
                     enum infer_domain d,
                     const struct infer_desc* desc,
                     uint32_t seed,
                     struct infer_tensor* out);

/* The consumer's wait on `ready` -- the output_ready token, seen from
 * the host. What that token IS depends on the domain: a sync file for
 * the dma-buf domains (GL, VK, DMABUF written by a device), the queue's
 * work-done future for WGPU (no exportable fence exists for a plain
 * buffer; ordering on the same queue is the token), nothing for CPU.
 * A dma-buf written by the CPU (the CPU engine through the mmap
 * hand-over) has no fence either: completing it means closing the CPU
 * write window and flushing, so a device reader sees the bytes. */
int infer_tensor_wait_ready(struct infer_ctx* c, struct infer_tensor* t);

/* Any domain -> packed host floats (waits for ready). Verification only. */
int infer_tensor_readback(struct infer_ctx* c, const struct infer_tensor* t, float* dst);

void infer_tensor_release(struct infer_ctx* c, struct infer_tensor* t);

/* per-domain implementations (used by infer_tensor.c and the edges) */
int infer_gl_alloc(struct infer_ctx_gl* c, const struct infer_desc* d, struct infer_tensor* t);
int infer_gl_gen(struct infer_ctx_gl* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t);
/* packed host floats -> the bo, through a texture upload and a blit
 * into the tensor's FBO (no driver maps every renderable layout) */
int infer_gl_upload(struct infer_ctx_gl* c, const float* src, struct infer_tensor* t);
int infer_gl_readback(struct infer_ctx_gl* c, const struct infer_tensor* t, float* dst);
void infer_gl_release(struct infer_ctx_gl* c, struct infer_tensor* t);
int infer_vk_alloc(struct infer_ctx_vk* c, const struct infer_desc* d, struct infer_tensor* t);
int infer_vk_gen(struct infer_ctx_vk* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t);
/* packed host floats -> the exportable memory: a map when it is
 * host-visible, a staging copy when it is device-local */
int infer_vk_upload(struct infer_ctx_vk* c, const float* src, struct infer_tensor* t);
int infer_vk_readback(struct infer_ctx_vk* c, const struct infer_tensor* t, float* dst);
void infer_vk_release(struct infer_ctx_vk* c, struct infer_tensor* t);
int infer_wgpu_alloc(struct infer_ctx_wgpu* c, const struct infer_desc* d, struct infer_tensor* t);
int infer_wgpu_gen(struct infer_ctx_wgpu* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t);
int infer_wgpu_readback(struct infer_ctx_wgpu* c, const struct infer_tensor* t, float* dst);
void infer_wgpu_release(struct infer_ctx_wgpu* c, struct infer_tensor* t);
/* the foreign dma-buf: a dma-heap allocation written through its mmap
 * (bracketed by DMA_BUF_IOCTL_SYNC so the GPU sees it on a UMA machine) */
int infer_dmabuf_available(void);
int infer_dmabuf_alloc(const struct infer_desc* d, struct infer_tensor* t);
int infer_dmabuf_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t);
int infer_dmabuf_readback(const struct infer_tensor* t, float* dst);
void infer_dmabuf_release(struct infer_tensor* t);
/* CPU access window for the consumer side of the mmap (start=1 / end=0) */
int infer_dmabuf_sync(const struct infer_tensor* t, int start, int write);
/* Cache maintenance for a CPU-side access to dma-buf pages a non-
 * snooping device reads or wrote: a no-op where DMA_BUF_IOCTL_SYNC does
 * the work (arm64), clflush on x86 where the ioctl does nothing. */
void infer_dmabuf_cpu_cache_sync(const void* p, size_t bytes);
/* block until everything submitted to the shared queue so far has
 * executed: the WGPU domain's only host-visible completion token */
int infer_wgpu_wait_idle(struct infer_ctx_wgpu* c);
/* a Storage|CopySrc|CopyDst buffer on the shared device, sized to a
 * multiple of 16 bytes (ORT's own allocation granularity; harmless) */
WGPUBuffer infer_wgpu_create_buffer(struct infer_ctx_wgpu* c,
                                    uint64_t bytes,
                                    WGPUBufferUsage extra);
/* A dma-buf byte image imported into the Dawn device, used in ONE
 * direction: readable (an input the engine consumes: texels -> packed
 * float buffer) or writable (an output the consumer owns: packed float
 * buffer -> texels). The import is the expensive part and is cached by
 * the edge in an infer_edge_cache; the access is one WGSL pass between
 * BeginAccess and EndAccess, with a SharedFence in (the fence that
 * gates our access) and a sync file out (Dawn's fence for the other
 * side). Both DEVICE_COPY edge bodies are this one function. */
struct infer_wgpu_import;
enum infer_wgpu_pass {
    INFER_WGPU_TEX_TO_BUF, /* import -> buf: the input edge */
    INFER_WGPU_BUF_TO_TEX, /* buf -> import: the output edge */
};
struct infer_wgpu_import* infer_wgpu_import_create(struct infer_ctx_wgpu* c,
                                                   int dmabuf_fd,
                                                   const struct infer_desc* d,
                                                   int writable);
int infer_wgpu_import_access(struct infer_ctx_wgpu* c,
                             struct infer_wgpu_import* im,
                             int acquire_fd, /* sync file gating our access, -1 = none */
                             enum infer_wgpu_pass pass,
                             WGPUBuffer buf,
                             int* release_fd /* out: Dawn's fence for the other side */);
void infer_wgpu_import_destroy(void* im);

/* ------------------------------------------------------------------ */
/* stage five: a camera frame imported for sampling, and a polled     */
/* readback                                                           */
/* ------------------------------------------------------------------ */

/* A dma-buf holding PIXELS (not a byte image) imported into the Dawn
 * device to be READ by a compute pass. Two differences from
 * infer_wgpu_import above, both forced by what a camera frame is:
 *
 *   - the format is the frame's fourcc, so NV12 arrives as a biplanar
 *     texture and the caller samples per-plane views, rather than
 *     ABGR8888 with the relayout pipelines bolted on;
 *   - begin and end are SEPARATE calls, because stage five records TWO
 *     passes over one frame in a single access bracket (the detector's
 *     letterbox and the landmark stage's rotated crop).
 *
 * Returns NULL when the driver will not let a compute shader read the
 * frame -- the caller's fallback is then the host path (convert on the
 * CPU and wgpuQueueWriteBuffer, which is the registry's cpu -> wgpu row).
 * There is deliberately no device-side fallback: Dawn refuses
 * CopyTextureToTexture on a multiplanar texture outright
 * (CommandEncoder.cpp, "Copying between a multiplanar texture and
 * another texture is currently not allowed"), so the only other device
 * route would be CopyTextureToBuffer per plane aspect plus a second
 * shader that samples buffers instead of textures -- untestable on a
 * driver that grants the views, and an untested rung is worse than a
 * host path that every CPU-EP run exercises.
 *
 * The import is the expensive part; cache one per camera buffer and keep
 * it for the camera's lifetime. */
struct infer_wgpu_frame_import;
struct infer_wgpu_frame_import* infer_wgpu_frame_import_create(struct infer_ctx_wgpu* c,
                                                               const struct infer_frame* f);
/* plane 0 = luma (r8unorm), plane 1 = chroma (rg8unorm, half size). The
 * views are valid between begin and end. */
WGPUTextureView infer_wgpu_frame_plane(const struct infer_wgpu_frame_import* im, int plane);
/* BeginAccess, gated by acquire_fd (-1 = nothing to wait for). For the
 * COPY path this also records the plane copies, so the views the caller
 * samples are ready when its own pass runs. */
int infer_wgpu_frame_begin(struct infer_ctx_wgpu* c,
                           struct infer_wgpu_frame_import* im,
                           int acquire_fd);
/* EndAccess. *release_fd is Dawn's "done reading this frame" fence as a
 * sync file -- what the frame's owner (V4L2, through cam_stream) must
 * see signal before it may hand the buffer back to the device. */
int infer_wgpu_frame_end(struct infer_ctx_wgpu* c,
                         struct infer_wgpu_frame_import* im,
                         int* release_fd);
void infer_wgpu_frame_import_destroy(void* im);

/* A readback that is POLLED instead of waited on: infer_wgpu_readback
 * blocks in wgpuInstanceWaitAny(UINT64_MAX), which a render loop cannot
 * do. submit() copies into a MapRead staging buffer and starts the map;
 * poll() returns 1 and fills dst once the map completed, 0 while it is
 * still in flight, -1 on error. One buffer is in flight at a time: a
 * MapRead buffer cannot be re-mapped while mapped, and nothing may be
 * copied into it while mapped, so submit() refuses until poll() drained
 * the previous one. This is the consumer's ready token, polled. */
struct infer_wgpu_readback;
struct infer_wgpu_readback* infer_wgpu_readback_create(struct infer_ctx_wgpu* c, size_t bytes);
int infer_wgpu_readback_submit(struct infer_ctx_wgpu* c,
                               struct infer_wgpu_readback* rb,
                               WGPUBuffer src,
                               size_t bytes);
int infer_wgpu_readback_poll(struct infer_ctx_wgpu* c, struct infer_wgpu_readback* rb, float* dst);
int infer_wgpu_readback_inflight(const struct infer_wgpu_readback* rb);
void infer_wgpu_readback_destroy(struct infer_wgpu_readback* rb);

/* ------------------------------------------------------------------ */
/* the edge registry                                                  */
/* ------------------------------------------------------------------ */

/* What an edge keeps between calls -- the imports it paid for -- lives
 * here, owned by the caller (anira: the compiled plan), keyed by the
 * tensor whose memory was imported. Under producer-side multi-buffering
 * the same edge sees a rotating set of tensors and must hit the cache
 * for each of them, so this is a small map, not one slot. */
enum { INFER_EDGE_CACHE_SLOTS = 32 };
struct infer_edge_cache {
    struct {
        const struct infer_tensor* key;
        void* state;
        void (*free_fn)(void* state);
    } slot[INFER_EDGE_CACHE_SLOTS];
    int count;
};
void infer_edge_cache_init(struct infer_edge_cache* k);
void* infer_edge_cache_get(const struct infer_edge_cache* k, const struct infer_tensor* key);
int infer_edge_cache_put(struct infer_edge_cache* k,
                         const struct infer_tensor* key,
                         void* state,
                         void (*free_fn)(void* state));
/* drop every import; call before the tensors it refers to are released */
void infer_edge_cache_fini(struct infer_edge_cache* k);

struct infer_edge;
typedef int (*infer_edge_probe_fn)(struct infer_ctx* c);
typedef int (*infer_edge_convert_fn)(struct infer_ctx* c,
                                     struct infer_edge_cache* k,
                                     struct infer_tensor* src,
                                     struct infer_tensor* dst);

/* A row is keyed (from, to) = (the domain the data is in, the domain
 * it is wanted in). Two kinds of body:
 *
 *   copy      convert(c, k, src, dst) moves src's data into dst. Run
 *             after the writer finished with src: after the producer
 *             (input edge) or after the engine (output edge).
 *   handover  nothing moves. convert(c, k, src, dst) makes dst a non-
 *             owning VIEW of src's memory, so whoever works through dst
 *             touches src's bytes. Run BEFORE the engine, and always
 *             with the engine's tensor as dst: on the input side the
 *             engine's input becomes a view of the producer's tensor;
 *             on the output side the engine's output becomes a view of
 *             the CONSUMER's tensor -- the engine writes straight into
 *             memory the consumer owns (the doc's bind_output). For an
 *             output row keyed (engine domain -> consumer domain) that
 *             call is therefore convert(c, k, consumer, engine).
 */
struct infer_edge {
    enum infer_domain src, dst;
    const char* name;
    enum infer_cost cost;
    int handover;
    infer_edge_probe_fn probe; /* NULL = always available */
    infer_edge_convert_fn convert;
    int available; /* filled by infer_registry_probe */
};

enum { INFER_MAX_EDGES = 24 };

struct infer_registry {
    struct infer_edge edges[INFER_MAX_EDGES];
    int count;
};

void infer_registry_init(struct infer_registry* r);
void infer_registry_probe(struct infer_registry* r, struct infer_ctx* c);
const struct infer_edge* infer_registry_find(const struct infer_registry* r,
                                             enum infer_domain src,
                                             enum infer_domain dst);
/* Run one edge's body (see the copy / handover rule above). */
int infer_edge_apply(const struct infer_edge* e,
                     struct infer_ctx* c,
                     struct infer_edge_cache* k,
                     struct infer_tensor* src,
                     struct infer_tensor* dst);
/* find + apply for the input side: src's domain -> dst_domain. dst is
 * allocated on first use and reused after. Returns the edge taken, or
 * NULL when none is available / the conversion failed. */
const struct infer_edge* infer_edge_convert(const struct infer_registry* r,
                                            struct infer_edge_cache* k,
                                            struct infer_ctx* c,
                                            struct infer_tensor* src,
                                            enum infer_domain dst_domain,
                                            struct infer_tensor* dst);

/* ------------------------------------------------------------------ */
/* the ONNX Runtime adapter                                           */
/* ------------------------------------------------------------------ */

struct infer_engine_ort;

enum { INFER_ENGINE_MAX_OUTPUTS = 8 };

/* the domain the EP reads its input from, and the one it writes its
 * outputs to: host memory for the CPU EP, WGPUBuffers on our Dawn device
 * for the WebGPU EP. The caller owns both sides. */
enum infer_domain infer_engine_ort_input_domain(enum infer_ep ep);
enum infer_domain infer_engine_ort_output_domain(enum infer_ep ep);
/* The WebGPU EP runs with graph capture: its command buffers are
 * recorded once and replayed. That is legitimate only because the
 * caller binds the same tensors every run -- a replay re-dispatches
 * the bind groups it captured. INFER_ORT_OPTS="key=value,..." appends
 * or overrides EP options for experiments. */
struct infer_engine_ort* infer_engine_ort_create(struct infer_ctx* c,
                                                 enum infer_ep ep,
                                                 const char* model_path);
/* the model's first input, dims with dynamic entries forced to 1 */
int infer_engine_ort_input_desc(const struct infer_engine_ort* e, struct infer_desc* d);
size_t infer_engine_ort_output_count(const struct infer_engine_ort* e);
/* output i, dims with dynamic entries forced to 1 -- a bound output
 * must have exactly this shape */
int infer_engine_ort_output_desc(const struct infer_engine_ort* e, size_t i, struct infer_desc* d);
/* t must be in infer_engine_ort_input_domain(ep) */
int infer_engine_ort_bind_input(struct infer_engine_ort* e, const struct infer_tensor* t);
/* t must be in infer_engine_ort_output_domain(ep) with output i's
 * shape; the engine writes it in place, nothing is copied or freed */
int infer_engine_ort_bind_output(struct infer_engine_ort* e,
                                 size_t i,
                                 const struct infer_tensor* t);
/* Submits the inference. For the CPU EP the outputs are complete on
 * return; for the WebGPU EP they are complete only when the queue
 * reaches them -- completion is a fence, never this call returning
 * (infer_tensor_wait_ready on the consumer's tensor). */
int infer_engine_ort_run(struct infer_engine_ort* e);
void infer_engine_ort_destroy(struct infer_engine_ort* e);

#endif
