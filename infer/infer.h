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
 * The engine adapter says which domain it wants its input in; the
 * caller asks the registry for that conversion and gets told the price.
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
 * turns texels back into a packed float buffer -- the DEVICE_COPY.
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
     * img_w floats; packed tensors have row_pitch_bytes == img_w * 4 */
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
size_t infer_desc_bytes_packed(const struct infer_desc* d);
/* Fill img_w/img_h/row_pitch (packed) from dims: dims[1] rows when the
 * tensor is [1, H, ...], else a single row. */
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
    /* edge-private state kept between conversions into this tensor
     * (e.g. the cached dma-buf import); freed with the tensor */
    void* edge_state;
    void (*edge_state_free)(void* state);
    const void* edge_state_src; /* which source tensor the state was built for */
};

/* A camera/decoder frame -- image data with a pixel format. Stage five
 * turns it into a tensor through a FrameToTensor pass; declared here
 * so the vocabulary is complete. */
struct infer_frame {
    int dmabuf_fd[4];
    uint32_t offset[4], pitch[4];
    uint64_t drm_modifier;
    uint32_t drm_format; /* fourcc */
    uint32_t width, height;
    uint8_t planes;
    struct infer_sync ready;
};

/* ------------------------------------------------------------------ */
/* contexts: one per GPU API, owned (we create) or borrowed            */
/* ------------------------------------------------------------------ */

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
    GLuint gen_program;
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
    WGPUComputePipeline gen_pipeline, relayout_pipeline;
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

/* "Get a tensor from X": allocate (first call, out zeroed) or reuse
 * (later calls) a tensor in domain d and fill it ON that domain with
 * infer_gen_value. Sets out->ready. */
int infer_tensor_gen(struct infer_ctx* c,
                     enum infer_domain d,
                     const struct infer_desc* desc,
                     uint32_t seed,
                     struct infer_tensor* out);

/* Any domain -> packed host floats (waits for ready). Verification only. */
int infer_tensor_readback(struct infer_ctx* c, const struct infer_tensor* t, float* dst);

void infer_tensor_release(struct infer_ctx* c, struct infer_tensor* t);

/* per-domain implementations (used by infer_tensor.c and the edges) */
int infer_gl_gen(struct infer_ctx_gl* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t);
int infer_gl_readback(struct infer_ctx_gl* c, const struct infer_tensor* t, float* dst);
void infer_gl_release(struct infer_ctx_gl* c, struct infer_tensor* t);
int infer_vk_gen(struct infer_ctx_vk* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t);
int infer_vk_readback(struct infer_ctx_vk* c, const struct infer_tensor* t, float* dst);
void infer_vk_release(struct infer_ctx_vk* c, struct infer_tensor* t);
int infer_wgpu_gen(struct infer_ctx_wgpu* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t);
int infer_wgpu_readback(struct infer_ctx_wgpu* c, const struct infer_tensor* t, float* dst);
void infer_wgpu_release(struct infer_ctx_wgpu* c, struct infer_tensor* t);
/* the foreign dma-buf: a dma-heap allocation written through its mmap
 * (bracketed by DMA_BUF_IOCTL_SYNC so the GPU sees it on a UMA machine) */
int infer_dmabuf_available(void);
int infer_dmabuf_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t);
int infer_dmabuf_readback(const struct infer_tensor* t, float* dst);
void infer_dmabuf_release(struct infer_tensor* t);
/* CPU access window for the consumer side of the mmap (start=1 / end=0) */
int infer_dmabuf_sync(const struct infer_tensor* t, int start, int write);
/* block until everything submitted to the shared queue so far has
 * executed (diagnostics: separates "submitted" from "finished") */
int infer_wgpu_wait_idle(struct infer_ctx_wgpu* c);
/* a Storage|CopySrc|CopyDst buffer on the shared device */
WGPUBuffer infer_wgpu_create_buffer(struct infer_ctx_wgpu* c,
                                    uint64_t bytes,
                                    WGPUBufferUsage extra);
/* dma-buf (byte image) -> packed float buffer on the Dawn device: the
 * DEVICE_COPY edge body. state caches the import between calls. */
struct infer_wgpu_import;
struct infer_wgpu_import* infer_wgpu_import_create(struct infer_ctx_wgpu* c,
                                                   int dmabuf_fd,
                                                   const struct infer_desc* d);
int infer_wgpu_import_relayout(struct infer_ctx_wgpu* c,
                               struct infer_wgpu_import* im,
                               int acquire_fd, /* producer's sync file, -1 = none */
                               WGPUBuffer dst,
                               int* release_fd /* out: Dawn's fence for the producer */);
void infer_wgpu_import_destroy(void* im);

/* ------------------------------------------------------------------ */
/* the edge registry                                                  */
/* ------------------------------------------------------------------ */

struct infer_edge;
typedef int (*infer_edge_probe_fn)(struct infer_ctx* c);
typedef int (*infer_edge_convert_fn)(struct infer_ctx* c,
                                     struct infer_tensor* src,
                                     struct infer_tensor* dst);

struct infer_edge {
    enum infer_domain src, dst;
    const char* name;
    enum infer_cost cost;
    infer_edge_probe_fn probe; /* NULL = always available */
    infer_edge_convert_fn convert;
    int available; /* filled by infer_registry_probe */
};

enum { INFER_MAX_EDGES = 16 };

struct infer_registry {
    struct infer_edge edges[INFER_MAX_EDGES];
    int count;
};

void infer_registry_init(struct infer_registry* r);
void infer_registry_probe(struct infer_registry* r, struct infer_ctx* c);
const struct infer_edge* infer_registry_find(const struct infer_registry* r,
                                             enum infer_domain src,
                                             enum infer_domain dst);
/* Convert src into dst_domain along the registered edge. dst is
 * allocated on first use and reused after. Returns the edge taken, or
 * NULL when none is available / the conversion failed. */
const struct infer_edge* infer_edge_convert(const struct infer_registry* r,
                                            struct infer_ctx* c,
                                            struct infer_tensor* src,
                                            enum infer_domain dst_domain,
                                            struct infer_tensor* dst);

/* ------------------------------------------------------------------ */
/* the ONNX Runtime adapter                                           */
/* ------------------------------------------------------------------ */

struct infer_engine_ort;

enum infer_domain infer_engine_ort_input_domain(enum infer_ep ep);
struct infer_engine_ort* infer_engine_ort_create(struct infer_ctx* c,
                                                 enum infer_ep ep,
                                                 const char* model_path);
/* the model's first input, dims with dynamic entries forced to 1 */
int infer_engine_ort_input_desc(const struct infer_engine_ort* e, struct infer_desc* d);
/* t must be in infer_engine_ort_input_domain(ep) */
int infer_engine_ort_bind_input(struct infer_engine_ort* e, const struct infer_tensor* t);
/* Synchronous: returns when all outputs are on the host. */
int infer_engine_ort_run(struct infer_engine_ort* e);
/* all outputs, concatenated, as left by the last run */
const float* infer_engine_ort_output(const struct infer_engine_ort* e, size_t* count);
void infer_engine_ort_destroy(struct infer_engine_ort* e);

#endif
