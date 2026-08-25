/* infer_tensor.c -- names, descriptors, sync files, the deterministic
 * generator, the CPU domain, and the per-domain dispatch for
 * gen / readback / release.
 */
#include "infer.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "infer_util.h"

/* ------------------------------------------------------------------ */
/* names                                                              */
/* ------------------------------------------------------------------ */

static const char* const domain_names[INFER_DOMAIN_COUNT] = {"cpu", "gl", "vk", "wgpu", "dmabuf"};
static const char* const ep_names[INFER_EP_COUNT] = {"cpu", "webgpu"};
static const char* const cost_names[] = {"ZERO_COPY", "DEVICE_COPY", "HOST_COPY", "UNAVAILABLE"};

const char* infer_domain_name(enum infer_domain d) {
    return d < INFER_DOMAIN_COUNT ? domain_names[d] : "?";
}
const char* infer_ep_name(enum infer_ep ep) {
    return ep < INFER_EP_COUNT ? ep_names[ep] : "?";
}
const char* infer_cost_name(enum infer_cost c) {
    return c <= INFER_COST_UNAVAILABLE ? cost_names[c] : "?";
}
int infer_domain_parse(const char* s, enum infer_domain* out) {
    for (int i = 0; i < INFER_DOMAIN_COUNT; i++) {
        if (strcmp(s, domain_names[i]) == 0) {
            *out = (enum infer_domain)i;
            return 0;
        }
    }
    return -1;
}
int infer_ep_parse(const char* s, enum infer_ep* out) {
    for (int i = 0; i < INFER_EP_COUNT; i++) {
        if (strcmp(s, ep_names[i]) == 0) {
            *out = (enum infer_ep)i;
            return 0;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* descriptor                                                         */
/* ------------------------------------------------------------------ */

size_t infer_desc_elements(const struct infer_desc* d) {
    size_t n = 1;
    for (int i = 0; i < d->ndim; i++) { n *= (size_t)(d->dims[i] > 0 ? d->dims[i] : 1); }
    return n;
}

size_t infer_desc_bytes_packed(const struct infer_desc* d) {
    return infer_desc_elements(d) * sizeof(float);
}

void infer_desc_set_image(struct infer_desc* d) {
    size_t n = infer_desc_elements(d);
    if (d->ndim >= 3 && d->dims[0] == 1 && d->dims[1] > 0) {
        d->img_h = (uint32_t)d->dims[1];
        d->img_w = (uint32_t)(n / (size_t)d->dims[1]);
    } else {
        d->img_h = 1;
        d->img_w = (uint32_t)n;
    }
    d->row_pitch_bytes = d->img_w * 4;
    d->drm_modifier = 0; /* DRM_FORMAT_MOD_LINEAR */
    d->planes = 1;
    d->plane_offset[0] = 0;
    d->plane_pitch[0] = d->row_pitch_bytes;
}

/* ------------------------------------------------------------------ */
/* sync                                                               */
/* ------------------------------------------------------------------ */

void infer_sync_reset(struct infer_sync* s) {
    if (s->sync_fd >= 0) { close(s->sync_fd); }
    s->sync_fd = -1;
}

void infer_sync_set(struct infer_sync* s, int fd) {
    infer_sync_reset(s);
    s->sync_fd = fd;
}

/* ------------------------------------------------------------------ */
/* generator + CPU domain                                             */
/* ------------------------------------------------------------------ */

float infer_gen_value(uint32_t i, uint32_t seed) {
    /* uint32 wraparound arithmetic, then one exact int->float
     * conversion of a 16-bit value, a power-of-two divide and a
     * subtraction that stays exact: bit-identical everywhere */
    uint32_t h = ((i + seed) * 2654435761u) >> 8;
    return (float)(h & 0xFFFFu) / 65536.0f - 0.5f;
}

static int cpu_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t) {
    size_t n = infer_desc_elements(d);
    if (!t->mem.cpu.ptr) {
        t->domain = INFER_DOMAIN_CPU;
        t->desc = *d;
        t->mem.cpu.ptr = malloc(n * sizeof(float));
        t->owned = 1;
        t->ready.sync_fd = -1;
        t->released.sync_fd = -1;
        INFER_CHECK(t->mem.cpu.ptr, "out of memory");
    }
    for (size_t i = 0; i < n; i++) { t->mem.cpu.ptr[i] = infer_gen_value((uint32_t)i, seed); }
    return 0;
}

/* ------------------------------------------------------------------ */
/* the foreign dma-buf domain                                         */
/* ------------------------------------------------------------------ */

#define DMA_HEAP_PATH "/dev/dma_heap/system"

int infer_dmabuf_available(void) {
    int fd = open(DMA_HEAP_PATH, O_RDWR | O_CLOEXEC);
    if (fd < 0) { return 0; }
    close(fd);
    return 1;
}

int infer_dmabuf_sync(const struct infer_tensor* t, int start, int write) {
    struct dma_buf_sync s = {
        .flags = (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) |
                 (write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ),
    };
    if (ioctl(t->mem.dmabuf.fd, DMA_BUF_IOCTL_SYNC, &s) < 0) {
        perror("DMA_BUF_IOCTL_SYNC");
        return -1;
    }
    return 0;
}

int infer_dmabuf_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t) {
    size_t n = infer_desc_elements(d);
    if (!t->mem.dmabuf.map) {
        int heap = open(DMA_HEAP_PATH, O_RDWR | O_CLOEXEC);
        INFER_CHECK(heap >= 0, "cannot open " DMA_HEAP_PATH);
        size_t page = (size_t)sysconf(_SC_PAGESIZE);
        size_t size = (n * sizeof(float) + page - 1) / page * page;
        struct dma_heap_allocation_data a = {.len = size, .fd_flags = O_RDWR | O_CLOEXEC};
        int r = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &a);
        close(heap);
        INFER_CHECK(r == 0, "DMA_HEAP_IOCTL_ALLOC failed");
        void* map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
        if (map == MAP_FAILED) {
            close(a.fd);
            INFER_CHECK(0, "mmap of the dma-buf failed");
        }
        t->domain = INFER_DOMAIN_DMABUF;
        t->desc = *d;
        t->desc.row_pitch_bytes = d->img_w * 4; /* packed, linear */
        t->desc.drm_modifier = 0;
        t->desc.planes = 1;
        t->desc.plane_offset[0] = 0;
        t->desc.plane_pitch[0] = t->desc.row_pitch_bytes;
        t->mem.dmabuf.fd = a.fd;
        t->mem.dmabuf.map = map;
        t->mem.dmabuf.size = size;
        t->owned = 1;
        t->ready.sync_fd = -1;
        t->released.sync_fd = -1;
    }
    /* the previous consumer may still be reading: a foreign producer
     * (camera, decoder) would honour `released` the same way */
    if (t->released.sync_fd >= 0) {
        struct pollfd p = {.fd = t->released.sync_fd, .events = POLLIN};
        while (poll(&p, 1, -1) < 0) {}
        infer_sync_reset(&t->released);
    }
    /* DMA_BUF_SYNC_START also waits for every fence attached to the
     * buffer's reservation (implicit sync) -- everything the GPU ever
     * did with it, not just what `released` describes. INFER_DMABUF_NOSYNC=1
     * skips the bracket to measure that cost (writes still land: the
     * mapping is coherent on this machine). */
    static int nosync = -1;
    if (nosync < 0) { nosync = getenv("INFER_DMABUF_NOSYNC") != NULL; }
    if (!nosync && infer_dmabuf_sync(t, 1, 1) < 0) { return -1; }
    float* f = t->mem.dmabuf.map;
    for (size_t i = 0; i < n; i++) { f[i] = infer_gen_value((uint32_t)i, seed); }
    if (!nosync && infer_dmabuf_sync(t, 0, 1) < 0) { return -1; }
    infer_sync_reset(&t->ready); /* CPU writes are complete on return */
    return 0;
}

int infer_dmabuf_readback(const struct infer_tensor* t, float* dst) {
    if (infer_dmabuf_sync(t, 1, 0) < 0) { return -1; }
    memcpy(dst, t->mem.dmabuf.map, infer_desc_bytes_packed(&t->desc));
    return infer_dmabuf_sync(t, 0, 0);
}

void infer_dmabuf_release(struct infer_tensor* t) {
    if (!t->owned) { return; }
    if (t->mem.dmabuf.map) { munmap(t->mem.dmabuf.map, t->mem.dmabuf.size); }
    if (t->mem.dmabuf.fd >= 0) { close(t->mem.dmabuf.fd); }
}

/* ------------------------------------------------------------------ */
/* dispatch                                                           */
/* ------------------------------------------------------------------ */

int infer_tensor_gen(struct infer_ctx* c,
                     enum infer_domain d,
                     const struct infer_desc* desc,
                     uint32_t seed,
                     struct infer_tensor* out) {
    if (d != INFER_DOMAIN_CPU && !(c->have & (1u << d))) {
        fprintf(stderr, "domain %s is not initialised\n", infer_domain_name(d));
        return -1;
    }
    switch (d) {
    case INFER_DOMAIN_CPU: return cpu_gen(desc, seed, out);
    case INFER_DOMAIN_GL: return infer_gl_gen(&c->gl, desc, seed, out);
    case INFER_DOMAIN_VK: return infer_vk_gen(&c->vk, desc, seed, out);
    case INFER_DOMAIN_WGPU: return infer_wgpu_gen(&c->wgpu, desc, seed, out);
    case INFER_DOMAIN_DMABUF: return infer_dmabuf_gen(desc, seed, out);
    default: return -1;
    }
}

int infer_tensor_readback(struct infer_ctx* c, const struct infer_tensor* t, float* dst) {
    switch (t->domain) {
    case INFER_DOMAIN_CPU:
        memcpy(dst, t->mem.cpu.ptr, infer_desc_bytes_packed(&t->desc));
        return 0;
    case INFER_DOMAIN_GL: return infer_gl_readback(&c->gl, t, dst);
    case INFER_DOMAIN_VK: return infer_vk_readback(&c->vk, t, dst);
    case INFER_DOMAIN_WGPU: return infer_wgpu_readback(&c->wgpu, t, dst);
    case INFER_DOMAIN_DMABUF: return infer_dmabuf_readback(t, dst);
    default: return -1;
    }
}

void infer_tensor_release(struct infer_ctx* c, struct infer_tensor* t) {
    if (t->edge_state && t->edge_state_free) { t->edge_state_free(t->edge_state); }
    t->edge_state = NULL;
    switch (t->domain) {
    case INFER_DOMAIN_CPU:
        if (t->owned) { free(t->mem.cpu.ptr); }
        break;
    case INFER_DOMAIN_GL: infer_gl_release(&c->gl, t); break;
    case INFER_DOMAIN_VK: infer_vk_release(&c->vk, t); break;
    case INFER_DOMAIN_WGPU: infer_wgpu_release(&c->wgpu, t); break;
    case INFER_DOMAIN_DMABUF: infer_dmabuf_release(t); break;
    default: break;
    }
    infer_sync_reset(&t->ready);
    infer_sync_reset(&t->released);
    memset(t, 0, sizeof *t);
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
}

/* ------------------------------------------------------------------ */
/* contexts                                                           */
/* ------------------------------------------------------------------ */

int infer_ctx_init_all(struct infer_ctx* c, unsigned want) {
    memset(c, 0, sizeof *c);
    /* WebGPU first: it is the device ONNX Runtime will share */
    if ((want & INFER_WANT_WGPU) && infer_ctx_wgpu_init(&c->wgpu) == 0) { c->have |= INFER_WANT_WGPU; }
    if ((want & INFER_WANT_GL) && infer_ctx_gl_init(&c->gl, NULL) == 0) { c->have |= INFER_WANT_GL; }
    if ((want & INFER_WANT_VK) && infer_ctx_vk_init(&c->vk) == 0) { c->have |= INFER_WANT_VK; }
    if ((want & INFER_WANT_DMABUF) && infer_dmabuf_available()) { c->have |= INFER_WANT_DMABUF; }
    c->have |= 1u << INFER_DOMAIN_CPU;
    return (c->have & want) == want ? 0 : -1;
}

void infer_ctx_fini_all(struct infer_ctx* c) {
    if (c->have & INFER_WANT_GL) { infer_ctx_gl_fini(&c->gl); }
    if (c->have & INFER_WANT_VK) { infer_ctx_vk_fini(&c->vk); }
    if (c->have & INFER_WANT_WGPU) { infer_ctx_wgpu_fini(&c->wgpu); }
    c->have = 0;
}
