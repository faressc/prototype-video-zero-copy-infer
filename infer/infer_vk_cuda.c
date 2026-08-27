/* infer_vk_cuda.c -- the VK <-> CUDA bridge: DEVICE_COPY without a
 * dma-buf, because CUDA has none to offer.
 *
 * The problem this file solves: a Vulkan tensor's memory is allocated
 * exportable as DMA_BUF (that is what Dawn imports), and on NVIDIA a
 * dma-buf's compatibleHandleTypes is dma-buf alone -- the same
 * allocation can never also be exported as the OPAQUE_FD that
 * cudaImportExternalMemory accepts. So the tensor itself does not
 * cross. What crosses is a second VkBuffer per tensor, allocated
 * OPAQUE_FD-exportable, imported into CUDA exactly once and cached by
 * the edge (keyed by the VK tensor, like the Dawn imports): one Vulkan
 * copy moves the tensor into or out of that buffer, and the CUDA side
 * touches the very same pages through its mapping -- no second copy of
 * substance (a device-local D2D into the engine's own tensor, so the
 * bound addresses stay fixed for graph capture) and no host memory
 * anywhere. Ordering crosses the API line the same way the memory does:
 * two binary semaphores, exported as opaque fds and imported into CUDA,
 * one per direction (VK signals -> the stream waits; the stream signals
 * -> VK waits). The producer/consumer side within Vulkan is honoured
 * the way infer_vk_readback/upload do it -- a host wait on the tensor's
 * fence and sync file -- and the copy submit's completion goes back out
 * as a SYNC_FD on `released` (input direction) or `ready` (output
 * direction), so the rest of the registry sees the usual tokens.
 *
 * Everything CUDA here is the runtime API's external-resource set
 * (cudaImportExternalMemory / Semaphore), which takes ownership of the
 * fds it imports. Without INFER_HAVE_CUDA the probe says no and the
 * registry falls back to the via_host rows.
 */
#include "infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "infer_util.h"

#ifndef INFER_HAVE_CUDA

int infer_vk_cuda_available(struct infer_ctx* c) {
    (void)c;
    return 0;
}
int infer_vk_cuda_to_cuda(struct infer_ctx* c,
                          struct infer_edge_cache* k,
                          struct infer_tensor* src,
                          struct infer_tensor* dst) {
    (void)c;
    (void)k;
    (void)src;
    (void)dst;
    INFER_CHECK(0, "vk<->cuda: built without CUDA");
    return -1;
}
int infer_vk_cuda_from_cuda(struct infer_ctx* c,
                            struct infer_edge_cache* k,
                            struct infer_tensor* src,
                            struct infer_tensor* dst) {
    return infer_vk_cuda_to_cuda(c, k, src, dst);
}

struct infer_vk_cuda_buf* infer_vk_cuda_buf_create(struct infer_ctx* c, size_t bytes) {
    (void)c;
    (void)bytes;
    return NULL;
}
VkBuffer infer_vk_cuda_buf_vk(const struct infer_vk_cuda_buf* b) {
    (void)b;
    return VK_NULL_HANDLE;
}
void* infer_vk_cuda_buf_ptr(const struct infer_vk_cuda_buf* b) {
    (void)b;
    return NULL;
}
VkSemaphore infer_vk_cuda_buf_semaphore(const struct infer_vk_cuda_buf* b) {
    (void)b;
    return VK_NULL_HANDLE;
}
int infer_vk_cuda_buf_stream_wait(struct infer_ctx* c, struct infer_vk_cuda_buf* b) {
    (void)c;
    (void)b;
    return -1;
}
int infer_vk_cuda_buf_from_tensor(struct infer_ctx* c,
                                  struct infer_vk_cuda_buf* b,
                                  struct infer_tensor* t) {
    (void)c;
    (void)b;
    (void)t;
    return -1;
}
int infer_vk_cuda_buf_to_tensor(struct infer_ctx* c,
                                struct infer_vk_cuda_buf* b,
                                struct infer_tensor* t,
                                int* ready_fd) {
    (void)c;
    (void)b;
    (void)t;
    if (ready_fd) { *ready_fd = -1; }
    return -1;
}
int infer_vk_cuda_buf_stream_signal(struct infer_ctx* c, struct infer_vk_cuda_buf* b) {
    (void)c;
    (void)b;
    return -1;
}
int infer_vk_cuda_buf_read_into(struct infer_ctx* c,
                                struct infer_vk_cuda_buf* b,
                                const struct infer_desc* d,
                                struct infer_tensor* dst) {
    (void)c;
    (void)b;
    (void)d;
    (void)dst;
    return -1;
}
int infer_vk_cuda_buf_write_from(struct infer_ctx* c,
                                 struct infer_vk_cuda_buf* b,
                                 const struct infer_desc* d,
                                 const struct infer_tensor* src) {
    (void)c;
    (void)b;
    (void)d;
    (void)src;
    return -1;
}
void infer_vk_cuda_buf_destroy(void* b) {
    (void)b;
}

#else

#include <cuda_runtime_api.h>

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        cudaError_t cuda_check_r = (expr);                                                      \
        if (cuda_check_r != cudaSuccess) {                                                      \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,                       \
                    cudaGetErrorString(cuda_check_r));                                          \
            return -1;                                                                          \
        }                                                                                       \
    } while (0)

/* ------------------------------------------------------------------ */
/* the probe                                                          */
/* ------------------------------------------------------------------ */

/* Can this driver export a TRANSFER buffer's memory and a semaphore as
 * OPAQUE_FD? (The extensions are the ones the dma-buf path already
 * enables -- VK_KHR_external_memory_fd / _semaphore_fd -- only the
 * handle type differs.) Cached per physical device; the CUDA side needs
 * no probe beyond the domain being up, which the caller checked. */
int infer_vk_cuda_available(struct infer_ctx* c) {
    static VkPhysicalDevice probed = VK_NULL_HANDLE;
    static int ok = 0;
    if (c->vk.phys == VK_NULL_HANDLE) { return 0; }
    if (probed == c->vk.phys) { return ok; }
    probed = c->vk.phys;
    ok = 0;
    VkPhysicalDeviceExternalBufferInfo bi = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkExternalBufferProperties bp = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(c->vk.phys, &bi, &bp);
    VkPhysicalDeviceExternalSemaphoreInfo si = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkExternalSemaphoreProperties sp = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(c->vk.phys, &si, &sp);
    ok = (bp.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) &&
         (sp.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) &&
         c->vk.get_memory_fd && c->vk.get_semaphore_fd;
    if (infer_verbose) {
        fprintf(stderr, "vk<->cuda: opaque-fd export %s\n", ok ? "yes" : "NO");
    }
    return ok;
}

/* ------------------------------------------------------------------ */
/* the bridge, one per VK tensor, owned by the edge cache             */
/* ------------------------------------------------------------------ */

struct vk_cuda_bridge {
    /* enough context to destroy ourselves from the cache's free_fn */
    VkDevice dev;
    VkQueue queue;
    VkCommandPool pool;
    struct infer_ctx_cuda* cuda;

    VkBuffer buf; /* the crossing: OPAQUE_FD-exportable, byte-image sized */
    VkDeviceMemory mem;
    VkDeviceSize size;
    VkCommandBuffer cmd; /* our own: the ctx's c->cmd is for synchronous users */
    VkFence fence;
    VkFence armed_fence; /* the fence the last cmd submit carried: ours
                          * (to_cuda) or the consumer tensor's own
                          * (from_cuda -- infer_vk_readback waits THAT
                          * one, per the !first_write contract) */
    int fence_armed;
    VkSemaphore vk_done;     /* VK signals, CUDA waits (opaque fd) */
    VkSemaphore cuda_done;   /* CUDA signals, VK waits (opaque fd) */
    VkSemaphore release_sem; /* SYNC_FD out: the copy submit's completion */
    int has_release;

    cudaExternalMemory_t ext_mem;
    void* dev_ptr; /* the buffer's pages, as seen by the stream */
    cudaExternalSemaphore_t ext_vk_done, ext_cuda_done;
};

static uint32_t mem_type_of(const VkPhysicalDeviceMemoryProperties* p,
                            uint32_t bits,
                            VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < p->memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (p->memoryTypes[i].propertyFlags & req) == req) { return i; }
    }
    return UINT32_MAX;
}

static void bridge_destroy(void* p) {
    struct vk_cuda_bridge* b = p;
    if (!b) { return; }
    if (b->fence_armed) { vkWaitForFences(b->dev, 1, &b->armed_fence, VK_TRUE, UINT64_MAX); }
    if (b->cuda) { cudaStreamSynchronize((cudaStream_t)b->cuda->stream); }
    if (b->ext_vk_done) { cudaDestroyExternalSemaphore(b->ext_vk_done); }
    if (b->ext_cuda_done) { cudaDestroyExternalSemaphore(b->ext_cuda_done); }
    if (b->dev_ptr) { cudaFree(b->dev_ptr); }
    if (b->ext_mem) { cudaDestroyExternalMemory(b->ext_mem); }
    if (b->cmd) { vkFreeCommandBuffers(b->dev, b->pool, 1, &b->cmd); }
    if (b->fence) { vkDestroyFence(b->dev, b->fence, NULL); }
    if (b->vk_done) { vkDestroySemaphore(b->dev, b->vk_done, NULL); }
    if (b->cuda_done) { vkDestroySemaphore(b->dev, b->cuda_done, NULL); }
    if (b->release_sem) { vkDestroySemaphore(b->dev, b->release_sem, NULL); }
    if (b->buf) { vkDestroyBuffer(b->dev, b->buf, NULL); }
    if (b->mem) { vkFreeMemory(b->dev, b->mem, NULL); }
    free(b);
}

/* an opaque-fd-exportable semaphore, its fd handed to CUDA (which owns
 * the fd from then on; reference semantics, so import once serves every
 * later signal/wait pair) */
static int semaphore_pair(VkDevice dev,
                          struct infer_ctx_vk* v,
                          VkSemaphore* sem,
                          cudaExternalSemaphore_t* ext) {
    VkExportSemaphoreCreateInfo esi = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                                       .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esi};
    VK_CHECK(vkCreateSemaphore(dev, &sci, NULL, sem));
    VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                   .semaphore = *sem,
                                   .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
    int fd = -1;
    INFER_CHECK(v->get_semaphore_fd(dev, &gfi, &fd) == VK_SUCCESS,
                "vk<->cuda: semaphore fd export failed");
    struct cudaExternalSemaphoreHandleDesc d;
    memset(&d, 0, sizeof d);
    d.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    d.handle.fd = fd;
    cudaError_t r = cudaImportExternalSemaphore(ext, &d);
    if (r != cudaSuccess) {
        close(fd); /* import failed: the fd is still ours */
        INFER_CHECK(0, "vk<->cuda: cudaImportExternalSemaphore: %s", cudaGetErrorString(r));
    }
    return 0;
}

static struct vk_cuda_bridge* bridge_of(struct infer_ctx* c,
                                        struct infer_edge_cache* k,
                                        struct infer_tensor* t) {
    struct vk_cuda_bridge* b = infer_edge_cache_get(k, t);
    if (b) { return b; }
    struct infer_ctx_vk* v = &c->vk;
    b = calloc(1, sizeof *b);
    if (!b) { return NULL; }
    b->dev = v->device;
    b->queue = v->queue;
    b->pool = v->cmd_pool;
    b->cuda = &c->cuda;
    b->size = (VkDeviceSize)t->desc.row_pitch_bytes * t->desc.img_h;

    /* the crossing buffer, exportable the one way CUDA imports */
    VkExternalMemoryBufferCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext,
        .size = b->size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(b->dev, &bci, NULL, &b->buf) != VK_SUCCESS) { goto fail; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(b->dev, b->buf, &req);
    uint32_t type = mem_type_of(&v->mem_props, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) { type = mem_type_of(&v->mem_props, req.memoryTypeBits, 0); }
    if (type == UINT32_MAX) { goto fail; }
    VkExportMemoryAllocateInfo exp = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .pNext = &exp,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type};
    if (vkAllocateMemory(b->dev, &mai, NULL, &b->mem) != VK_SUCCESS) { goto fail; }
    if (vkBindBufferMemory(b->dev, b->buf, b->mem, 0) != VK_SUCCESS) { goto fail; }

    /* into CUDA, once: the import owns the fd, the mapping is the pages */
    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                .memory = b->mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    int fd = -1;
    if (v->get_memory_fd(b->dev, &gfi, &fd) != VK_SUCCESS) { goto fail; }
    struct cudaExternalMemoryHandleDesc hd;
    memset(&hd, 0, sizeof hd);
    hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd;
    hd.size = req.size;
    if (cudaImportExternalMemory(&b->ext_mem, &hd) != cudaSuccess) {
        close(fd);
        fprintf(stderr, "vk<->cuda: cudaImportExternalMemory failed\n");
        goto fail;
    }
    struct cudaExternalMemoryBufferDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.offset = 0;
    bd.size = b->size;
    if (cudaExternalMemoryGetMappedBuffer(&b->dev_ptr, b->ext_mem, &bd) != cudaSuccess) {
        fprintf(stderr, "vk<->cuda: mapping the imported memory failed\n");
        goto fail;
    }

    /* one command buffer + fence of our own (c->cmd belongs to the
     * synchronous paths), and the semaphores */
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = b->pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(b->dev, &cai, &b->cmd) != VK_SUCCESS) { goto fail; }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(b->dev, &fci, NULL, &b->fence) != VK_SUCCESS) { goto fail; }
    if (semaphore_pair(b->dev, v, &b->vk_done, &b->ext_vk_done) < 0) { goto fail; }
    if (semaphore_pair(b->dev, v, &b->cuda_done, &b->ext_cuda_done) < 0) { goto fail; }
    if (v->has_sync_fd) {
        VkExportSemaphoreCreateInfo esi = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esi};
        if (vkCreateSemaphore(b->dev, &sci, NULL, &b->release_sem) == VK_SUCCESS) {
            b->has_release = 1;
        }
    }
    if (infer_edge_cache_put(k, t, b, bridge_destroy) < 0) { goto fail; }
    if (infer_verbose) {
        fprintf(stderr, "vk<->cuda: bridge %zu bytes (type %u) for tensor %ux%u\n",
                (size_t)b->size, type, t->desc.img_w, t->desc.img_h);
    }
    return b;

fail:
    bridge_destroy(b);
    return NULL;
}

/* the pitched<->packed stream copies, defined with the edge bodies below */
static int d2d_from_pitched(void* dst, const void* src, const struct infer_desc* d, cudaStream_t s);
static int d2d_to_pitched(void* dst, const void* src, const struct infer_desc* d, cudaStream_t s);

/* ------------------------------------------------------------------ */
/* the standalone shared buffer (a pass's destination)                 */
/* ------------------------------------------------------------------ */

struct infer_vk_cuda_buf {
    VkDevice dev;
    VkQueue queue;
    VkCommandPool pool;
    struct infer_ctx_vk* vk; /* for get_semaphore_fd; outlives us (cache order) */
    struct infer_ctx_cuda* cuda;
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    VkSemaphore sem; /* binary, one direction per plan: writer signals, reader waits */
    cudaExternalMemory_t ext_mem;
    void* dev_ptr;
    cudaExternalSemaphore_t ext_sem;
    /* for the tensor<->buffer copies the wgpu bridge composes */
    VkCommandBuffer cmd;
    VkFence fence;
    int fence_armed;
    VkSemaphore ready_sem; /* SYNC_FD out of to_tensor, when the driver can */
    int has_ready;
};

void infer_vk_cuda_buf_destroy(void* p) {
    struct infer_vk_cuda_buf* b = p;
    if (!b) { return; }
    if (b->cuda && b->cuda->stream) { cudaStreamSynchronize((cudaStream_t)b->cuda->stream); }
    if (b->fence_armed) { vkWaitForFences(b->dev, 1, &b->fence, VK_TRUE, UINT64_MAX); }
    if (b->dev) { vkDeviceWaitIdle(b->dev); } /* the writer's submit may be in flight */
    if (b->ext_sem) { cudaDestroyExternalSemaphore(b->ext_sem); }
    if (b->dev_ptr) { cudaFree(b->dev_ptr); }
    if (b->ext_mem) { cudaDestroyExternalMemory(b->ext_mem); }
    if (b->ready_sem) { vkDestroySemaphore(b->dev, b->ready_sem, NULL); }
    if (b->fence) { vkDestroyFence(b->dev, b->fence, NULL); }
    if (b->cmd) { vkFreeCommandBuffers(b->dev, b->pool, 1, &b->cmd); }
    if (b->sem) { vkDestroySemaphore(b->dev, b->sem, NULL); }
    if (b->buf) { vkDestroyBuffer(b->dev, b->buf, NULL); }
    if (b->mem) { vkFreeMemory(b->dev, b->mem, NULL); }
    free(b);
}

struct infer_vk_cuda_buf* infer_vk_cuda_buf_create(struct infer_ctx* c, size_t bytes) {
    if (!(c->have & INFER_WANT_CUDA) || !infer_vk_cuda_available(c)) { return NULL; }
    struct infer_ctx_vk* v = &c->vk;
    struct infer_vk_cuda_buf* b = calloc(1, sizeof *b);
    if (!b) { return NULL; }
    b->dev = v->device;
    b->queue = v->queue;
    b->pool = v->cmd_pool;
    b->vk = v;
    b->cuda = &c->cuda;
    b->size = bytes;
    VkExternalMemoryBufferCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(b->dev, &bci, NULL, &b->buf) != VK_SUCCESS) { goto fail; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(b->dev, b->buf, &req);
    uint32_t type = mem_type_of(&v->mem_props, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) { type = mem_type_of(&v->mem_props, req.memoryTypeBits, 0); }
    if (type == UINT32_MAX) { goto fail; }
    VkExportMemoryAllocateInfo exp = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .pNext = &exp,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type};
    if (vkAllocateMemory(b->dev, &mai, NULL, &b->mem) != VK_SUCCESS) { goto fail; }
    if (vkBindBufferMemory(b->dev, b->buf, b->mem, 0) != VK_SUCCESS) { goto fail; }
    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                .memory = b->mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    int fd = -1;
    if (v->get_memory_fd(b->dev, &gfi, &fd) != VK_SUCCESS) { goto fail; }
    struct cudaExternalMemoryHandleDesc hd;
    memset(&hd, 0, sizeof hd);
    hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd;
    hd.size = req.size;
    if (cudaImportExternalMemory(&b->ext_mem, &hd) != cudaSuccess) {
        close(fd);
        fprintf(stderr, "vk<->cuda: buf import failed\n");
        goto fail;
    }
    struct cudaExternalMemoryBufferDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.offset = 0;
    bd.size = bytes;
    if (cudaExternalMemoryGetMappedBuffer(&b->dev_ptr, b->ext_mem, &bd) != cudaSuccess) {
        fprintf(stderr, "vk<->cuda: buf mapping failed\n");
        goto fail;
    }
    if (semaphore_pair(b->dev, v, &b->sem, &b->ext_sem) < 0) { goto fail; }
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = b->pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(b->dev, &cai, &b->cmd) != VK_SUCCESS) { goto fail; }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(b->dev, &fci, NULL, &b->fence) != VK_SUCCESS) { goto fail; }
    if (v->has_sync_fd) {
        VkExportSemaphoreCreateInfo esi = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        VkSemaphoreCreateInfo semci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                       .pNext = &esi};
        if (vkCreateSemaphore(b->dev, &semci, NULL, &b->ready_sem) == VK_SUCCESS) {
            b->has_ready = 1;
        }
    }
    if (infer_verbose) {
        fprintf(stderr, "vk<->cuda: shared buffer %zu bytes (type %u)\n", bytes, type);
    }
    return b;
fail:
    infer_vk_cuda_buf_destroy(b);
    return NULL;
}

/* the VK copy between a VK tensor and the shared buffer, one submit on
 * our own command buffer, serialised by our own fence */
static int buf_tensor_copy(struct infer_vk_cuda_buf* b,
                           struct infer_tensor* t,
                           int to_buf,
                           VkSemaphore wait,
                           const VkSemaphore* sig,
                           uint32_t nsig) {
    struct infer_mem_vk* m = &t->mem.vk;
    if (b->fence_armed) {
        VK_CHECK(vkWaitForFences(b->dev, 1, &b->fence, VK_TRUE, UINT64_MAX));
        b->fence_armed = 0;
    }
    VK_CHECK(vkResetFences(b->dev, 1, &b->fence));
    VK_CHECK(vkResetCommandBuffer(b->cmd, 0));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(b->cmd, &bi));
    VkDeviceSize bytes = (VkDeviceSize)t->desc.row_pitch_bytes * t->desc.img_h;
    if (bytes > b->size) { bytes = b->size; }
    if (m->alias_ok) {
        VkBufferCopy region = {.size = bytes};
        if (to_buf) {
            vkCmdCopyBuffer(b->cmd, m->buffer, b->buf, 1, &region);
        } else {
            vkCmdCopyBuffer(b->cmd, b->buf, m->buffer, 1, &region);
        }
    } else {
        /* the image path: a foreign writer (Dawn) filled it, so establish
         * GENERAL the way the frame acquire does */
        VkImageMemoryBarrier ib = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = to_buf ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = m->first_write ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m->image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        VkBufferImageCopy region = {.bufferRowLength = t->desc.row_pitch_bytes / 4,
                                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    .imageExtent = {t->desc.img_w, t->desc.img_h, 1}};
        if (to_buf) {
            vkCmdCopyImageToBuffer(b->cmd, m->image, VK_IMAGE_LAYOUT_GENERAL, b->buf, 1, &region);
        } else {
            vkCmdCopyBufferToImage(b->cmd, b->buf, m->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        m->first_write = 0;
    }
    VK_CHECK(vkEndCommandBuffer(b->cmd));
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = wait != VK_NULL_HANDLE ? 1u : 0u,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &b->cmd,
        .signalSemaphoreCount = nsig,
        .pSignalSemaphores = sig,
    };
    VK_CHECK(vkQueueSubmit(b->queue, 1, &si, b->fence));
    b->fence_armed = 1;
    return 0;
}

int infer_vk_cuda_buf_from_tensor(struct infer_ctx* c,
                                  struct infer_vk_cuda_buf* b,
                                  struct infer_tensor* t) {
    (void)c;
    INFER_CHECK(b && t->domain == INFER_DOMAIN_VK, "vk<->cuda: from_tensor wants a vk tensor");
    /* the writer's completion, host-side (Dawn's fence on `ready`) */
    if (t->ready.sync_fd >= 0) {
        INFER_CHECK(infer_wait_sync_fd(t->ready.sync_fd, 2000) == 0,
                    "vk<->cuda: staging ready fence timed out");
        infer_sync_reset(&t->ready);
    }
    VkSemaphore sig[1] = {b->sem};
    return buf_tensor_copy(b, t, 1, VK_NULL_HANDLE, sig, 1);
}

int infer_vk_cuda_buf_to_tensor(struct infer_ctx* c,
                                struct infer_vk_cuda_buf* b,
                                struct infer_tensor* t,
                                int* ready_fd) {
    (void)c;
    INFER_CHECK(b && t->domain == INFER_DOMAIN_VK, "vk<->cuda: to_tensor wants a vk tensor");
    if (ready_fd) { *ready_fd = -1; }
    VkSemaphore sig[1] = {b->ready_sem};
    const uint32_t nsig = b->has_ready ? 1u : 0u;
    if (buf_tensor_copy(b, t, 0, b->sem, sig, nsig) < 0) { return -1; }
    if (b->has_ready && ready_fd) {
        VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                       .semaphore = b->ready_sem,
                                       .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        if (b->vk->get_semaphore_fd(b->dev, &gfi, ready_fd) != VK_SUCCESS) { *ready_fd = -1; }
    } else if (!b->has_ready) {
        /* no exportable fence: complete it here */
        VK_CHECK(vkWaitForFences(b->dev, 1, &b->fence, VK_TRUE, UINT64_MAX));
        b->fence_armed = 0;
    }
    return 0;
}

int infer_vk_cuda_buf_stream_signal(struct infer_ctx* c, struct infer_vk_cuda_buf* b) {
    INFER_CHECK(b, "vk<->cuda: no shared buffer");
    struct cudaExternalSemaphoreSignalParams sp;
    memset(&sp, 0, sizeof sp);
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&b->ext_sem, &sp, 1, (cudaStream_t)c->cuda.stream));
    return 0;
}

int infer_vk_cuda_buf_read_into(struct infer_ctx* c,
                                struct infer_vk_cuda_buf* b,
                                const struct infer_desc* d,
                                struct infer_tensor* dst) {
    INFER_CHECK(b && dst->mem.cuda.ptr, "vk<->cuda: read_into wants an allocated cuda tensor");
    return d2d_from_pitched(dst->mem.cuda.ptr, b->dev_ptr, d, (cudaStream_t)c->cuda.stream);
}

int infer_vk_cuda_buf_write_from(struct infer_ctx* c,
                                 struct infer_vk_cuda_buf* b,
                                 const struct infer_desc* d,
                                 const struct infer_tensor* src) {
    INFER_CHECK(b && src->mem.cuda.ptr, "vk<->cuda: write_from wants an allocated cuda tensor");
    return d2d_to_pitched(b->dev_ptr, src->mem.cuda.ptr, d, (cudaStream_t)c->cuda.stream);
}

VkBuffer infer_vk_cuda_buf_vk(const struct infer_vk_cuda_buf* b) {
    return b ? b->buf : VK_NULL_HANDLE;
}

void* infer_vk_cuda_buf_ptr(const struct infer_vk_cuda_buf* b) {
    return b ? b->dev_ptr : NULL;
}

VkSemaphore infer_vk_cuda_buf_semaphore(const struct infer_vk_cuda_buf* b) {
    return b ? b->sem : VK_NULL_HANDLE;
}

int infer_vk_cuda_buf_stream_wait(struct infer_ctx* c, struct infer_vk_cuda_buf* b) {
    INFER_CHECK(b, "vk<->cuda: no shared buffer");
    struct cudaExternalSemaphoreWaitParams wp;
    memset(&wp, 0, sizeof wp);
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&b->ext_sem, &wp, 1, (cudaStream_t)c->cuda.stream));
    return 0;
}

/* ------------------------------------------------------------------ */
/* the two copies                                                     */
/* ------------------------------------------------------------------ */

/* the staging pages hold the byte image (rows pitch apart); the engine's
 * tensor is packed. In practice pitch == img_w*4 (rows are 64-byte
 * aligned by infer_desc_set_image) and this is one flat copy; the padded
 * case copies the full rows in one 2D op and the tail row after it. */
static int d2d_from_pitched(void* dst, const void* src, const struct infer_desc* d, cudaStream_t s) {
    const size_t n = infer_desc_elements(d);
    const size_t wbytes = (size_t)d->img_w * 4;
    if (d->row_pitch_bytes == wbytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, s));
        return 0;
    }
    const size_t rows = n / d->img_w, tail = n % d->img_w;
    if (rows) {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, wbytes, src, d->row_pitch_bytes, wbytes, rows,
                                     cudaMemcpyDeviceToDevice, s));
    }
    if (tail) {
        CUDA_CHECK(cudaMemcpyAsync((char*)dst + rows * wbytes,
                                   (const char*)src + rows * d->row_pitch_bytes, tail * 4,
                                   cudaMemcpyDeviceToDevice, s));
    }
    return 0;
}

static int d2d_to_pitched(void* dst, const void* src, const struct infer_desc* d, cudaStream_t s) {
    const size_t n = infer_desc_elements(d);
    const size_t wbytes = (size_t)d->img_w * 4;
    if (d->row_pitch_bytes == wbytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, s));
        return 0;
    }
    const size_t rows = n / d->img_w, tail = n % d->img_w;
    if (rows) {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, d->row_pitch_bytes, src, wbytes, wbytes, rows,
                                     cudaMemcpyDeviceToDevice, s));
    }
    if (tail) {
        CUDA_CHECK(cudaMemcpyAsync((char*)dst + rows * d->row_pitch_bytes,
                                   (const char*)src + rows * wbytes, tail * 4,
                                   cudaMemcpyDeviceToDevice, s));
    }
    return 0;
}

/* record + submit the VK half: tensor -> staging (to_staging=1) or
 * staging -> tensor. Waits `wait` (may be NULL), signals `sig[0..nsig)`,
 * arms the bridge fence. The tensor's image path copies in GENERAL, the
 * layout every writer here leaves it in -- the same arrangement
 * infer_vk_readback/upload rely on. */
static int vk_copy(struct infer_ctx_vk* v,
                   struct vk_cuda_bridge* b,
                   struct infer_tensor* t,
                   int to_staging,
                   VkSemaphore wait,
                   const VkSemaphore* sig,
                   uint32_t nsig,
                   VkFence fence) {
    struct infer_mem_vk* m = &t->mem.vk;
    if (b->fence_armed) {
        VK_CHECK(vkWaitForFences(b->dev, 1, &b->armed_fence, VK_TRUE, UINT64_MAX));
        b->fence_armed = 0;
    }
    VK_CHECK(vkResetFences(b->dev, 1, &fence));
    VK_CHECK(vkResetCommandBuffer(b->cmd, 0));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(b->cmd, &bi));
    if (m->alias_ok) {
        VkBufferCopy region = {.size = b->size};
        if (to_staging) {
            vkCmdCopyBuffer(b->cmd, m->buffer, b->buf, 1, &region);
        } else {
            vkCmdCopyBuffer(b->cmd, b->buf, m->buffer, 1, &region);
        }
    } else {
        VkBufferImageCopy region = {.bufferRowLength = t->desc.row_pitch_bytes / 4,
                                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    .imageExtent = {t->desc.img_w, t->desc.img_h, 1}};
        if (to_staging) {
            vkCmdCopyImageToBuffer(b->cmd, m->image, VK_IMAGE_LAYOUT_GENERAL, b->buf, 1, &region);
        } else {
            VkImageMemoryBarrier ib = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = m->first_write ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = m->image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
            vkCmdCopyBufferToImage(b->cmd, b->buf, m->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
    }
    VK_CHECK(vkEndCommandBuffer(b->cmd));
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = wait ? 1 : 0,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &b->cmd,
        .signalSemaphoreCount = nsig,
        .pSignalSemaphores = sig,
    };
    VK_CHECK(vkQueueSubmit(v->queue, 1, &si, fence));
    b->armed_fence = fence;
    b->fence_armed = 1;
    return 0;
}

/* the input direction: the producer's VK tensor into the engine's CUDA
 * tensor. VK copies tensor -> staging and signals; the stream waits and
 * copies staging -> dst, ordered ahead of the kernels the coming run
 * submits. The copy submit's own completion goes back as src->released. */
int infer_vk_cuda_to_cuda(struct infer_ctx* c,
                          struct infer_edge_cache* k,
                          struct infer_tensor* src,
                          struct infer_tensor* dst) {
    if (infer_tensor_alloc(c, INFER_DOMAIN_CUDA, &src->desc, dst) < 0) { return -1; }
    struct vk_cuda_bridge* b = bridge_of(c, k, src);
    INFER_CHECK(b, "vk->cuda: no bridge");
    /* the producer's completion, honoured the way infer_vk_readback does */
    if (src->ready.sync_fd >= 0) {
        INFER_CHECK(infer_wait_sync_fd(src->ready.sync_fd, 2000) == 0,
                    "vk->cuda: ready fence timed out");
    }
    if (!src->mem.vk.first_write) {
        VK_CHECK(vkWaitForFences(c->vk.device, 1, &src->mem.vk.fence, VK_TRUE, UINT64_MAX));
    }
    VkSemaphore sig[2] = {b->vk_done, b->release_sem};
    if (vk_copy(&c->vk, b, src, 1, VK_NULL_HANDLE, sig, b->has_release ? 2u : 1u, b->fence) < 0) {
        return -1;
    }
    if (b->has_release) {
        VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                       .semaphore = b->release_sem,
                                       .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        int fd = -1;
        if (c->vk.get_semaphore_fd(c->vk.device, &gfi, &fd) != VK_SUCCESS) { fd = -1; }
        infer_sync_set(&src->released, fd);
    }
    cudaStream_t s = (cudaStream_t)c->cuda.stream;
    struct cudaExternalSemaphoreWaitParams wp;
    memset(&wp, 0, sizeof wp);
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&b->ext_vk_done, &wp, 1, s));
    if (d2d_from_pitched(dst->mem.cuda.ptr, b->dev_ptr, &src->desc, s) < 0) { return -1; }
    infer_sync_reset(&dst->ready); /* the stream is the token */
    return 0;
}

/* the output direction, called as convert(consumer, engine) inverted by
 * the registry contract for copy edges: src is the engine's CUDA tensor,
 * dst the consumer's VK tensor. The stream copies src -> staging and
 * signals; VK waits on the device, copies staging -> tensor, and the
 * submit's completion becomes dst->ready as a sync file. */
int infer_vk_cuda_from_cuda(struct infer_ctx* c,
                            struct infer_edge_cache* k,
                            struct infer_tensor* src,
                            struct infer_tensor* dst) {
    INFER_CHECK(dst->mem.vk.memory, "cuda->vk: the consumer's vk tensor is not allocated");
    struct vk_cuda_bridge* b = bridge_of(c, k, dst);
    INFER_CHECK(b, "cuda->vk: no bridge");
    /* the consumer may still be reading the previous contents, and our
     * own previous copy into it may still be in flight */
    if (dst->released.sync_fd >= 0) {
        infer_wait_sync_fd(dst->released.sync_fd, 1000);
        infer_sync_reset(&dst->released);
    }
    cudaStream_t s = (cudaStream_t)c->cuda.stream;
    if (d2d_to_pitched(b->dev_ptr, src->mem.cuda.ptr, &src->desc, s) < 0) { return -1; }
    struct cudaExternalSemaphoreSignalParams sp;
    memset(&sp, 0, sizeof sp);
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&b->ext_cuda_done, &sp, 1, s));
    VkSemaphore sig[1] = {dst->mem.vk.done};
    const uint32_t nsig = c->vk.has_sync_fd ? 1u : 0u;
    /* the submit carries the TENSOR's fence: setting first_write = 0
     * below promises infer_vk_readback (and vk->cuda's source wait) a
     * fence the last writer armed, and the bridge's own fence is not
     * the one they look at */
    if (vk_copy(&c->vk, b, dst, 0, b->cuda_done, sig, nsig, dst->mem.vk.fence) < 0) { return -1; }
    dst->mem.vk.first_write = 0;
    if (nsig) {
        VkSemaphoreGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                       .semaphore = dst->mem.vk.done,
                                       .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
        int fd = -1;
        if (c->vk.get_semaphore_fd(c->vk.device, &gfi, &fd) != VK_SUCCESS) { fd = -1; }
        infer_sync_set(&dst->ready, fd);
    } else {
        VK_CHECK(vkWaitForFences(b->dev, 1, &dst->mem.vk.fence, VK_TRUE, UINT64_MAX));
        b->fence_armed = 0;
        infer_sync_set(&dst->ready, -1);
    }
    return 0;
}

#endif /* INFER_HAVE_CUDA */
