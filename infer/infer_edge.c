/* infer_edge.c -- the conversion-edge registry: one function per
 * (source domain, destination domain) pair, each tagged with its cost
 * class, each gated by a probe of the drivers involved. The registry is
 * what anira's planner intersects at prepare(); here it is a table the
 * hello-world prints.
 *
 * The two DEVICE_COPY edges (GL -> WebGPU, Vulkan -> WebGPU) share one
 * body: the source tensor's dma-buf is imported into Dawn as a byte
 * image and a WGSL pass writes the packed float buffer. Their sync is a
 * pair of sync files: the producer's "ready" goes in as a SharedFence,
 * Dawn's "done reading" comes back and the producer waits on it before
 * writing again.
 */
#include "infer.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "infer_util.h"

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

/* dst becomes a non-owning alias of src (identity edges) */
static int alias(struct infer_tensor* src, struct infer_tensor* dst) {
    dst->domain = src->domain;
    dst->desc = src->desc;
    dst->mem = src->mem;
    dst->owned = 0;
    dst->ready.sync_fd = -1;
    dst->released.sync_fd = -1;
    return 0;
}

static int ensure_cpu(struct infer_tensor* dst, const struct infer_desc* d) {
    if (!dst->mem.cpu.ptr) {
        dst->domain = INFER_DOMAIN_CPU;
        dst->desc = *d;
        dst->desc.row_pitch_bytes = d->img_w * 4;
        dst->mem.cpu.ptr = malloc(infer_desc_bytes_packed(d));
        dst->owned = 1;
        dst->ready.sync_fd = -1;
        dst->released.sync_fd = -1;
        INFER_CHECK(dst->mem.cpu.ptr, "out of memory");
    }
    return 0;
}

static int ensure_wgpu(struct infer_ctx* c, struct infer_tensor* dst, const struct infer_desc* d) {
    if (!dst->mem.wgpu.buffer) {
        dst->domain = INFER_DOMAIN_WGPU;
        dst->desc = *d;
        dst->desc.row_pitch_bytes = d->img_w * 4;
        dst->mem.wgpu.buffer = infer_wgpu_create_buffer(&c->wgpu, infer_desc_bytes_packed(d), 0);
        dst->owned = 1;
        dst->ready.sync_fd = -1;
        dst->released.sync_fd = -1;
        INFER_CHECK(dst->mem.wgpu.buffer, "wgpu: buffer creation failed");
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* edges                                                              */
/* ------------------------------------------------------------------ */

static int cpu_identity(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    (void)c;
    return alias(src, dst);
}

static int wgpu_identity(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    (void)c;
    return alias(src, dst);
}

static int wgpu_write_buffer(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    if (ensure_wgpu(c, dst, &src->desc) < 0) { return -1; }
    wgpuQueueWriteBuffer(c->wgpu.queue,
                         dst->mem.wgpu.buffer,
                         0,
                         src->mem.cpu.ptr,
                         infer_desc_bytes_packed(&src->desc));
    return 0;
}

static int readback_to_cpu(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    if (ensure_cpu(dst, &src->desc) < 0) { return -1; }
    return infer_tensor_readback(c, src, dst->mem.cpu.ptr);
}

/* DMABUF -> CPU: the mmap IS the tensor. The CPU engine reads the fd's
 * pages directly; the only work is the cache-coherence bracket. */
static int dmabuf_map(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    (void)c;
    if (dst->domain == INFER_DOMAIN_DMABUF && dst->edge_state_src == src) {
        return 0; /* window already open on this source */
    }
    if (infer_dmabuf_sync(src, 1, 0) < 0) { return -1; }
    dst->domain = INFER_DOMAIN_CPU;
    dst->desc = src->desc;
    dst->mem.cpu.ptr = src->mem.dmabuf.map;
    dst->owned = 0;
    dst->ready.sync_fd = -1;
    dst->released.sync_fd = -1;
    return 0;
}

/* GL/VK/DMABUF -> WGPU: import the source's dma-buf once, relayout every time */
static int dmabuf_to_wgpu(struct infer_ctx* c, struct infer_tensor* src, struct infer_tensor* dst) {
    int fd = src->domain == INFER_DOMAIN_GL     ? src->mem.gl.dmabuf_fd
             : src->domain == INFER_DOMAIN_VK   ? src->mem.vk.dmabuf_fd
                                                : src->mem.dmabuf.fd;
    if (ensure_wgpu(c, dst, &src->desc) < 0) { return -1; }
    if (dst->edge_state && dst->edge_state_src != src) {
        dst->edge_state_free(dst->edge_state);
        dst->edge_state = NULL;
    }
    if (!dst->edge_state) {
        dst->edge_state = infer_wgpu_import_create(&c->wgpu, fd, &src->desc);
        dst->edge_state_free = infer_wgpu_import_destroy;
        dst->edge_state_src = src;
        INFER_CHECK(dst->edge_state, "dma-buf import into WebGPU failed");
    }
    int release_fd = -1;
    int r = infer_wgpu_import_relayout(&c->wgpu,
                                       dst->edge_state,
                                       src->ready.sync_fd,
                                       dst->mem.wgpu.buffer,
                                       &release_fd);
    infer_sync_set(&src->released, release_fd);
    return r;
}

/* ------------------------------------------------------------------ */
/* probes                                                             */
/* ------------------------------------------------------------------ */

static int probe_wgpu(struct infer_ctx* c) {
    return (c->have & INFER_WANT_WGPU) != 0;
}
static int probe_gl(struct infer_ctx* c) {
    return (c->have & INFER_WANT_GL) != 0;
}
static int probe_vk(struct infer_ctx* c) {
    return (c->have & INFER_WANT_VK) != 0;
}
static int probe_gl_to_wgpu(struct infer_ctx* c) {
    return probe_gl(c) && probe_wgpu(c) && c->gl.has_dmabuf_import && c->wgpu.has_dmabuf;
}
static int probe_vk_to_wgpu(struct infer_ctx* c) {
    return probe_vk(c) && probe_wgpu(c) && c->vk.has_dmabuf_export && c->vk.has_drm_modifier &&
           c->wgpu.has_dmabuf;
}
static int probe_dmabuf(struct infer_ctx* c) {
    return (c->have & INFER_WANT_DMABUF) != 0;
}
static int probe_dmabuf_to_wgpu(struct infer_ctx* c) {
    return probe_dmabuf(c) && probe_wgpu(c) && c->wgpu.has_dmabuf;
}

/* ------------------------------------------------------------------ */
/* the table                                                          */
/* ------------------------------------------------------------------ */

void infer_registry_init(struct infer_registry* r) {
    static const struct infer_edge table[] = {
        {INFER_DOMAIN_CPU, INFER_DOMAIN_CPU, "cpu_identity", INFER_COST_ZERO_COPY, NULL, cpu_identity, 0},
        {INFER_DOMAIN_WGPU, INFER_DOMAIN_WGPU, "wgpu_identity", INFER_COST_ZERO_COPY, probe_wgpu, wgpu_identity, 0},
        {INFER_DOMAIN_GL, INFER_DOMAIN_WGPU, "gl_dmabuf_import", INFER_COST_DEVICE_COPY, probe_gl_to_wgpu, dmabuf_to_wgpu, 0},
        {INFER_DOMAIN_VK, INFER_DOMAIN_WGPU, "vk_dmabuf_import", INFER_COST_DEVICE_COPY, probe_vk_to_wgpu, dmabuf_to_wgpu, 0},
        {INFER_DOMAIN_CPU, INFER_DOMAIN_WGPU, "wgpu_write_buffer", INFER_COST_HOST_COPY, probe_wgpu, wgpu_write_buffer, 0},
        {INFER_DOMAIN_GL, INFER_DOMAIN_CPU, "gl_bo_map", INFER_COST_HOST_COPY, probe_gl, readback_to_cpu, 0},
        {INFER_DOMAIN_VK, INFER_DOMAIN_CPU, "vk_map", INFER_COST_HOST_COPY, probe_vk, readback_to_cpu, 0},
        {INFER_DOMAIN_WGPU, INFER_DOMAIN_CPU, "wgpu_map_read", INFER_COST_HOST_COPY, probe_wgpu, readback_to_cpu, 0},
        {INFER_DOMAIN_DMABUF, INFER_DOMAIN_WGPU, "dmabuf_import", INFER_COST_DEVICE_COPY, probe_dmabuf_to_wgpu, dmabuf_to_wgpu, 0},
        {INFER_DOMAIN_DMABUF, INFER_DOMAIN_CPU, "dmabuf_mmap", INFER_COST_ZERO_COPY, probe_dmabuf, dmabuf_map, 0},
    };
    memset(r, 0, sizeof *r);
    r->count = (int)(sizeof table / sizeof table[0]);
    memcpy(r->edges, table, sizeof table);
}

void infer_registry_probe(struct infer_registry* r, struct infer_ctx* c) {
    for (int i = 0; i < r->count; i++) {
        struct infer_edge* e = &r->edges[i];
        e->available = e->probe ? e->probe(c) : 1;
    }
}

const struct infer_edge* infer_registry_find(const struct infer_registry* r,
                                             enum infer_domain src,
                                             enum infer_domain dst) {
    for (int i = 0; i < r->count; i++) {
        if (r->edges[i].src == src && r->edges[i].dst == dst) { return &r->edges[i]; }
    }
    return NULL;
}

const struct infer_edge* infer_edge_convert(const struct infer_registry* r,
                                            struct infer_ctx* c,
                                            struct infer_tensor* src,
                                            enum infer_domain dst_domain,
                                            struct infer_tensor* dst) {
    const struct infer_edge* e = infer_registry_find(r, src->domain, dst_domain);
    if (!e || !e->available) { return NULL; }
    return e->convert(c, src, dst) == 0 ? e : NULL;
}
