/* infer_tensor.c -- names, descriptors, sync files, the deterministic
 * generator, the CPU domain, the foreign dma-buf domain, and the
 * per-domain dispatch for alloc / gen / wait / readback / release.
 */
#include "infer.h"

#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h> /* _mm_clflush, _mm_mfence: SSE2, baseline on x86-64 */
#endif

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

int infer_verbose = 0;

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

size_t infer_desc_bytes_image(const struct infer_desc* d) {
    return (size_t)d->img_w * d->img_h * 4;
}

/* The byte image is our encoding, not the tensor's shape, so its rows
 * can be whatever the importers want: 64-byte aligned. A linear image
 * with a packed odd pitch (the palm detector's 18-float rows, 72 bytes)
 * is accepted by Dawn's dma-buf import and then written at a pitch the
 * driver rounded on its own, so the bytes land elsewhere -- measured on
 * Honeykrisp as max_abs_err 175 on every dma-heap output. The Vulkan
 * domain never saw it because it picks an aligned pitch, GBM because
 * it picks its own. Rows of 16 texels or a multiple, then: the tensor's
 * own rows when they qualify (so the README's 576x192 stays 576x192),
 * else an exact factorisation as square as possible, else padding at
 * the end that no shader reads as data (they guard on the element
 * count). The floats stay packed and contiguous in all three cases,
 * which is what keeps the CPU hand-over zero-copy. */
void infer_desc_set_image(struct infer_desc* d) {
    enum { MAX_DIM = 8192, ROW = 16 }; /* 16 RGBA8 texels = 64 bytes */
    size_t n = infer_desc_elements(d);
    size_t w = 0, h = 0;
    if (d->ndim >= 3 && d->dims[0] == 1 && d->dims[1] > 0) {
        size_t hh = (size_t)d->dims[1], ww = n / hh;
        if (ww % ROW == 0 && ww <= MAX_DIM && hh <= MAX_DIM) {
            w = ww;
            h = hh;
        }
    }
    if (!w) {
        size_t best_diff = (size_t)-1;
        for (size_t ww = ROW; ww <= n && ww <= MAX_DIM; ww += ROW) {
            if (n % ww) { continue; }
            size_t hh = n / ww;
            if (hh > MAX_DIM) { continue; }
            size_t diff = ww > hh ? ww - hh : hh - ww;
            if (diff < best_diff) {
                best_diff = diff;
                w = ww;
                h = hh;
            }
        }
    }
    if (!w) {
        w = ROW;
        while (w * w < n && w < MAX_DIM) { w += ROW; }
        h = (n + w - 1) / w;
    }
    d->img_w = (uint32_t)w;
    d->img_h = (uint32_t)h;
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

static int cpu_alloc(const struct infer_desc* d, struct infer_tensor* t) {
    if (t->mem.cpu.ptr) { return 0; }
    t->domain = INFER_DOMAIN_CPU;
    t->desc = *d;
    t->desc.row_pitch_bytes = d->img_w * 4; /* packed */
    t->mem.cpu.ptr = malloc(infer_desc_bytes_packed(d));
    t->owned = 1;
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
    INFER_CHECK(t->mem.cpu.ptr, "out of memory");
    return 0;
}

static int cpu_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t) {
    if (cpu_alloc(d, t) < 0) { return -1; }
    size_t n = infer_desc_elements(d);
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

/* Cache maintenance between the CPU and a device that does not snoop
 * the CPU cache. DMA_BUF_IOCTL_SYNC is the portable way to ask for it,
 * and on arm64 it is real: END(write) cleans the lines a device will
 * read, START(read) invalidates the lines a device wrote. On x86 the
 * kernel treats every DMA master as coherent and the ioctl does nothing
 * -- but NVIDIA's open kernel module maps imported system memory into
 * the GPU without snooping (measured on a 2080 Ti / 610.43: whatever
 * barrier the GPU uses, it reads the lines as they were before the CPU
 * wrote; clflush, non-temporal stores or evicting the lines all make it
 * current), so on x86 this does by hand what the ioctl does on arm64.
 * clflush both writes dirty lines back and invalidates them, so the one
 * loop serves both directions: before a device reads what the CPU wrote,
 * and before the CPU reads what a device wrote (lines this CPU cached
 * from an earlier read of the same pages would otherwise shadow the
 * device's writes). A DMA producer -- a camera, a decoder -- never needs
 * the write side: its writes do not pass through the CPU cache. */
void infer_dmabuf_cpu_cache_sync(const void* p, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
    const char* line = (const char*)((uintptr_t)p & ~(uintptr_t)63);
    const char* end = (const char*)p + bytes;
    for (; line < end; line += 64) { _mm_clflush(line); }
    _mm_mfence();
#else
    (void)p;
    (void)bytes; /* DMA_BUF_IOCTL_SYNC did the maintenance */
#endif
}

int infer_dmabuf_alloc(const struct infer_desc* d, struct infer_tensor* t) {
    if (t->mem.dmabuf.map) { return 0; }
    int heap = open(DMA_HEAP_PATH, O_RDWR | O_CLOEXEC);
    INFER_CHECK(heap >= 0, "cannot open " DMA_HEAP_PATH);
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    /* the whole byte image (an importer maps img_h rows), page rounded */
    size_t size = (infer_desc_bytes_image(d) + page - 1) / page * page;
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
    return 0;
}

int infer_dmabuf_gen(const struct infer_desc* d, uint32_t seed, struct infer_tensor* t) {
    if (infer_dmabuf_alloc(d, t) < 0) { return -1; }
    size_t n = infer_desc_elements(d);
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
     * skips the bracket to measure that cost; the cache flush below is
     * not part of the bracket and stays, because without it a
     * non-snooping reader sees the previous iteration's bytes. */
    static int nosync = -1;
    if (nosync < 0) { nosync = getenv("INFER_DMABUF_NOSYNC") != NULL; }
    if (!nosync && infer_dmabuf_sync(t, 1, 1) < 0) { return -1; }
    float* f = t->mem.dmabuf.map;
    for (size_t i = 0; i < n; i++) { f[i] = infer_gen_value((uint32_t)i, seed); }
    infer_dmabuf_cpu_cache_sync(f, n * sizeof(float));
    if (!nosync && infer_dmabuf_sync(t, 0, 1) < 0) { return -1; }
    infer_sync_reset(&t->ready); /* CPU writes are complete on return */
    return 0;
}

int infer_dmabuf_readback(const struct infer_tensor* t, float* dst) {
    /* a device may have written it: its fence first, then the read
     * window (implicit sync, and on arm64 the invalidate) */
    if (t->ready.sync_fd >= 0) {
        INFER_CHECK(infer_wait_sync_fd(t->ready.sync_fd, 2000) == 0, "dmabuf: ready fence timed out");
    }
    size_t bytes = infer_desc_bytes_packed(&t->desc);
    if (infer_dmabuf_sync(t, 1, 0) < 0) { return -1; }
    infer_dmabuf_cpu_cache_sync(t->mem.dmabuf.map, bytes);
    memcpy(dst, t->mem.dmabuf.map, bytes);
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

static int domain_up(const struct infer_ctx* c, enum infer_domain d) {
    if (d != INFER_DOMAIN_CPU && !(c->have & (1u << d))) {
        fprintf(stderr, "domain %s is not initialised\n", infer_domain_name(d));
        return 0;
    }
    return 1;
}

int infer_tensor_alloc(struct infer_ctx* c,
                       enum infer_domain d,
                       const struct infer_desc* desc,
                       struct infer_tensor* out) {
    if (!domain_up(c, d)) { return -1; }
    switch (d) {
    case INFER_DOMAIN_CPU: return cpu_alloc(desc, out);
    case INFER_DOMAIN_GL: return infer_gl_alloc(&c->gl, desc, out);
    case INFER_DOMAIN_VK: return infer_vk_alloc(&c->vk, desc, out);
    case INFER_DOMAIN_WGPU: return infer_wgpu_alloc(&c->wgpu, desc, out);
    case INFER_DOMAIN_DMABUF: return infer_dmabuf_alloc(desc, out);
    default: return -1;
    }
}

int infer_tensor_gen(struct infer_ctx* c,
                     enum infer_domain d,
                     const struct infer_desc* desc,
                     uint32_t seed,
                     struct infer_tensor* out) {
    if (!domain_up(c, d)) { return -1; }
    switch (d) {
    case INFER_DOMAIN_CPU: return cpu_gen(desc, seed, out);
    case INFER_DOMAIN_GL: return infer_gl_gen(&c->gl, desc, seed, out);
    case INFER_DOMAIN_VK: return infer_vk_gen(&c->vk, desc, seed, out);
    case INFER_DOMAIN_WGPU: return infer_wgpu_gen(&c->wgpu, desc, seed, out);
    case INFER_DOMAIN_DMABUF: return infer_dmabuf_gen(desc, seed, out);
    default: return -1;
    }
}

int infer_tensor_wait_ready(struct infer_ctx* c, struct infer_tensor* t) {
    switch (t->domain) {
    case INFER_DOMAIN_CPU: return 0;
    case INFER_DOMAIN_WGPU:
        /* no exportable fence for a plain buffer: the queue's work-done
         * future is the token, and same-queue consumers need none */
        return infer_wgpu_wait_idle(&c->wgpu);
    case INFER_DOMAIN_DMABUF:
        if (t->ready.sync_fd < 0) {
            /* written by the CPU through the mmap hand-over (the CPU
             * engine): "ready" means the write window is closed and the
             * lines are clean, so a device reader sees the bytes -- the
             * mirror of infer_dmabuf_gen's bracket */
            infer_dmabuf_cpu_cache_sync(t->mem.dmabuf.map, infer_desc_bytes_packed(&t->desc));
            return infer_dmabuf_sync(t, 0, 1);
        }
        /* fallthrough: written by a device, its fence is the token */
    case INFER_DOMAIN_GL:
    case INFER_DOMAIN_VK:
        INFER_CHECK(infer_wait_sync_fd(t->ready.sync_fd, 2000) == 0,
                    "%s: ready fence timed out",
                    infer_domain_name(t->domain));
        return 0;
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
    if (t->priv && t->priv_free) { t->priv_free(t->priv); }
    t->priv = NULL;
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
