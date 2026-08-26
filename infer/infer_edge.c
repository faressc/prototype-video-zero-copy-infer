/* infer_edge.c -- the conversion-edge registry: one function per
 * (from domain, to domain) pair, each tagged with its cost class, each
 * gated by a probe of the drivers involved. The registry is what
 * anira's planner intersects at prepare(); here it is a table the
 * hello-world prints. It serves both sides of the engine: producer ->
 * engine input, and engine output -> consumer.
 *
 * The DEVICE_COPY edges share one body each way. Into WebGPU: the
 * source tensor's dma-buf is imported into Dawn as a byte image and a
 * WGSL pass writes the packed float buffer the engine reads. Out of
 * WebGPU: the consumer's dma-buf is imported the same way and a WGSL
 * pass writes it from the packed float buffer the engine produced.
 * Their sync is a pair of sync files per tensor: the fence that gates
 * Dawn's access goes in as a SharedFence (the producer's "ready" for an
 * input, the consumer's "released" for an output), Dawn's fence comes
 * back (the producer's "released", the consumer's "ready").
 *
 * Imports are cached in the caller's infer_edge_cache, keyed by the
 * tensor whose memory was imported -- never on the tensors themselves.
 */
#include "infer.h"

#include <stdio.h>
#include <string.h>

#include "infer_util.h"

/* ------------------------------------------------------------------ */
/* the cache                                                          */
/* ------------------------------------------------------------------ */

void infer_edge_cache_init(struct infer_edge_cache* k) {
    memset(k, 0, sizeof *k);
}

void* infer_edge_cache_get(const struct infer_edge_cache* k, const struct infer_tensor* key) {
    for (int i = 0; i < k->count; i++) {
        if (k->slot[i].key == key) { return k->slot[i].state; }
    }
    return NULL;
}

int infer_edge_cache_put(struct infer_edge_cache* k,
                         const struct infer_tensor* key,
                         void* state,
                         void (*free_fn)(void* state)) {
    INFER_CHECK(k->count < INFER_EDGE_CACHE_SLOTS, "edge cache full");
    k->slot[k->count].key = key;
    k->slot[k->count].state = state;
    k->slot[k->count].free_fn = free_fn;
    k->count++;
    return 0;
}

void infer_edge_cache_fini(struct infer_edge_cache* k) {
    for (int i = 0; i < k->count; i++) {
        if (k->slot[i].state && k->slot[i].free_fn) { k->slot[i].free_fn(k->slot[i].state); }
    }
    memset(k, 0, sizeof *k);
}

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

/* dst becomes a non-owning alias of src (the identity hand-overs) */
static int alias(struct infer_tensor* src, struct infer_tensor* dst) {
    dst->domain = src->domain;
    dst->desc = src->desc;
    dst->mem = src->mem;
    dst->owned = 0;
    dst->ready.sync_fd = -1;
    dst->released.sync_fd = -1;
    return 0;
}

/* a copy edge's destination: allocated on first use, reused after --
 * or already there, when the consumer allocated it (bind_output) */
static int ensure(struct infer_ctx* c,
                  enum infer_domain dom,
                  struct infer_tensor* dst,
                  const struct infer_desc* d) {
    return infer_tensor_alloc(c, dom, d, dst);
}

static int dmabuf_fd_of(const struct infer_tensor* t) {
    switch (t->domain) {
    case INFER_DOMAIN_GL: return t->mem.gl.dmabuf_fd;
    case INFER_DOMAIN_VK: return t->mem.vk.dmabuf_fd;
    case INFER_DOMAIN_DMABUF: return t->mem.dmabuf.fd;
    default: return -1;
    }
}

/* the byte-image import of t, made once per (tensor, direction) and
 * kept in the cache */
static struct infer_wgpu_import* import_of(struct infer_ctx* c,
                                           struct infer_edge_cache* k,
                                           struct infer_tensor* t,
                                           int writable) {
    struct infer_wgpu_import* im = infer_edge_cache_get(k, t);
    if (im) { return im; }
    int fd = dmabuf_fd_of(t);
    if (fd < 0) {
        fprintf(stderr, "edge: %s tensor has no dma-buf to import\n", infer_domain_name(t->domain));
        return NULL;
    }
    im = infer_wgpu_import_create(&c->wgpu, fd, &t->desc, writable);
    if (!im) { return NULL; }
    if (infer_edge_cache_put(k, t, im, infer_wgpu_import_destroy) < 0) {
        infer_wgpu_import_destroy(im);
        return NULL;
    }
    return im;
}

/* the dma-buf's mmap IS the CPU tensor: a view, plus the CPU access
 * window the kernel wants opened (waits on the buffer's implicit-sync
 * fences, and invalidates on arm64). Left open across the engine's
 * access; the counterpart END comes from whoever finishes the access
 * (the producer's next write bracket for an input, infer_tensor_wait_
 * ready for an output). */
static int dmabuf_view(struct infer_tensor* src, struct infer_tensor* dst, int write) {
    if (infer_dmabuf_sync(src, 1, write) < 0) { return -1; }
    dst->domain = INFER_DOMAIN_CPU;
    dst->desc = src->desc;
    dst->mem.cpu.ptr = src->mem.dmabuf.map;
    dst->owned = 0;
    dst->ready.sync_fd = -1;
    dst->released.sync_fd = -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* edges                                                              */
/* ------------------------------------------------------------------ */

static int identity(struct infer_ctx* c,
                    struct infer_edge_cache* k,
                    struct infer_tensor* src,
                    struct infer_tensor* dst) {
    (void)c;
    (void)k;
    return alias(src, dst);
}

/* CPU -> WGPU (an input the WebGPU EP reads, or a CPU EP output a WebGPU consumer wants) */
static int wgpu_write_buffer(struct infer_ctx* c,
                             struct infer_edge_cache* k,
                             struct infer_tensor* src,
                             struct infer_tensor* dst) {
    (void)k;
    if (ensure(c, INFER_DOMAIN_WGPU, dst, &src->desc) < 0) { return -1; }
    wgpuQueueWriteBuffer(c->wgpu.queue,
                         dst->mem.wgpu.buffer,
                         0,
                         src->mem.cpu.ptr,
                         infer_desc_bytes_packed(&src->desc));
    return 0;
}

/* GL / VK / WGPU -> CPU */
static int readback_to_cpu(struct infer_ctx* c,
                           struct infer_edge_cache* k,
                           struct infer_tensor* src,
                           struct infer_tensor* dst) {
    (void)k;
    if (ensure(c, INFER_DOMAIN_CPU, dst, &src->desc) < 0) { return -1; }
    return infer_tensor_readback(c, src, dst->mem.cpu.ptr);
}

/* DMABUF -> CPU, hand-over: the CPU engine reads the fd's pages in place */
static int dmabuf_map_read(struct infer_ctx* c,
                           struct infer_edge_cache* k,
                           struct infer_tensor* src,
                           struct infer_tensor* dst) {
    (void)c;
    (void)k;
    return dmabuf_view(src, dst, 0);
}

/* CPU -> DMABUF, hand-over, called as convert(consumer, engine): the
 * CPU engine writes the consumer's pages in place */
static int dmabuf_map_write(struct infer_ctx* c,
                            struct infer_edge_cache* k,
                            struct infer_tensor* src,
                            struct infer_tensor* dst) {
    (void)c;
    (void)k;
    return dmabuf_view(src, dst, 1);
}

/* GL / VK / DMABUF -> WGPU: import the source's dma-buf once, relayout
 * every time. src->ready gates the pass; Dawn's fence becomes
 * src->released. */
static int dmabuf_to_wgpu(struct infer_ctx* c,
                          struct infer_edge_cache* k,
                          struct infer_tensor* src,
                          struct infer_tensor* dst) {
    if (ensure(c, INFER_DOMAIN_WGPU, dst, &src->desc) < 0) { return -1; }
    struct infer_wgpu_import* im = import_of(c, k, src, 0);
    INFER_CHECK(im, "dma-buf import into WebGPU failed");
    int release_fd = -1;
    int r = infer_wgpu_import_access(&c->wgpu,
                                     im,
                                     src->ready.sync_fd,
                                     INFER_WGPU_TEX_TO_BUF,
                                     dst->mem.wgpu.buffer,
                                     &release_fd);
    infer_sync_set(&src->released, release_fd);
    return r;
}

/* WGPU -> GL / VK / DMABUF: import the consumer's dma-buf once, write
 * it from the engine's buffer every time. dst->released (the consumer
 * is done reading the previous contents) gates the pass; Dawn's fence
 * becomes dst->ready. The consumer allocated dst. */
static int wgpu_to_dmabuf(struct infer_ctx* c,
                          struct infer_edge_cache* k,
                          struct infer_tensor* src,
                          struct infer_tensor* dst) {
    INFER_CHECK(dmabuf_fd_of(dst) >= 0, "output edge: the consumer's %s tensor is not allocated",
                infer_domain_name(dst->domain));
    struct infer_wgpu_import* im = import_of(c, k, dst, 1);
    INFER_CHECK(im, "dma-buf import into WebGPU (for writing) failed");
    int release_fd = -1;
    int r = infer_wgpu_import_access(&c->wgpu,
                                     im,
                                     dst->released.sync_fd,
                                     INFER_WGPU_BUF_TO_TEX,
                                     src->mem.wgpu.buffer,
                                     &release_fd);
    infer_sync_reset(&dst->released); /* consumed: Dawn holds its own dup */
    infer_sync_set(&dst->ready, release_fd);
    return r;
}

/* CPU -> VK / GL: host uploads into memory the consumer allocated */
static int cpu_to_vk(struct infer_ctx* c,
                     struct infer_edge_cache* k,
                     struct infer_tensor* src,
                     struct infer_tensor* dst) {
    (void)k;
    INFER_CHECK(dst->mem.vk.memory, "vk_upload: the consumer's tensor is not allocated");
    return infer_vk_upload(&c->vk, src->mem.cpu.ptr, dst);
}

static int cpu_to_gl(struct infer_ctx* c,
                     struct infer_edge_cache* k,
                     struct infer_tensor* src,
                     struct infer_tensor* dst) {
    (void)k;
    INFER_CHECK(dst->mem.gl.bo, "gl_upload: the consumer's tensor is not allocated");
    return infer_gl_upload(&c->gl, src->mem.cpu.ptr, dst);
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
static int probe_gl_wgpu(struct infer_ctx* c) {
    return probe_gl(c) && probe_wgpu(c) && c->gl.has_dmabuf_import && c->wgpu.has_dmabuf;
}
static int probe_vk_wgpu(struct infer_ctx* c) {
    return probe_vk(c) && probe_wgpu(c) && c->vk.has_dmabuf_export && c->vk.has_drm_modifier &&
           c->wgpu.has_dmabuf;
}
static int probe_dmabuf(struct infer_ctx* c) {
    return (c->have & INFER_WANT_DMABUF) != 0;
}
static int probe_dmabuf_wgpu(struct infer_ctx* c) {
    return probe_dmabuf(c) && probe_wgpu(c) && c->wgpu.has_dmabuf;
}

/* ------------------------------------------------------------------ */
/* the table                                                          */
/* ------------------------------------------------------------------ */

#define ROW(from, to, name, cost, handover, probe, fn) \
    {INFER_DOMAIN_##from, INFER_DOMAIN_##to, name, INFER_COST_##cost, handover, probe, fn, 0}

void infer_registry_init(struct infer_registry* r) {
    /* names say the mechanism, not the domains -- the (from, to) key
     * already does, and the table hello_inference prints has columns
     * for both */
    static const struct infer_edge table[] = {
        /* into the engine, and identities that serve both sides */
        ROW(CPU, CPU, "identity", ZERO_COPY, 1, NULL, identity),
        ROW(WGPU, WGPU, "identity", ZERO_COPY, 1, probe_wgpu, identity),
        ROW(GL, WGPU, "dmabuf_import", DEVICE_COPY, 0, probe_gl_wgpu, dmabuf_to_wgpu),
        ROW(VK, WGPU, "dmabuf_import", DEVICE_COPY, 0, probe_vk_wgpu, dmabuf_to_wgpu),
        ROW(DMABUF, WGPU, "dmabuf_import", DEVICE_COPY, 0, probe_dmabuf_wgpu, dmabuf_to_wgpu),
        ROW(CPU, WGPU, "write_buffer", HOST_COPY, 0, probe_wgpu, wgpu_write_buffer),
        ROW(GL, CPU, "read_pixels", HOST_COPY, 0, probe_gl, readback_to_cpu),
        ROW(VK, CPU, "map", HOST_COPY, 0, probe_vk, readback_to_cpu),
        ROW(WGPU, CPU, "map_read", HOST_COPY, 0, probe_wgpu, readback_to_cpu),
        ROW(DMABUF, CPU, "mmap", ZERO_COPY, 1, probe_dmabuf, dmabuf_map_read),
        /* out of the engine */
        ROW(WGPU, GL, "dmabuf_write", DEVICE_COPY, 0, probe_gl_wgpu, wgpu_to_dmabuf),
        ROW(WGPU, VK, "dmabuf_write", DEVICE_COPY, 0, probe_vk_wgpu, wgpu_to_dmabuf),
        ROW(WGPU, DMABUF, "dmabuf_write", DEVICE_COPY, 0, probe_dmabuf_wgpu, wgpu_to_dmabuf),
        ROW(CPU, GL, "upload", HOST_COPY, 0, probe_gl, cpu_to_gl),
        ROW(CPU, VK, "upload", HOST_COPY, 0, probe_vk, cpu_to_vk),
        ROW(CPU, DMABUF, "mmap", ZERO_COPY, 1, probe_dmabuf, dmabuf_map_write),
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

int infer_edge_apply(const struct infer_edge* e,
                     struct infer_ctx* c,
                     struct infer_edge_cache* k,
                     struct infer_tensor* src,
                     struct infer_tensor* dst) {
    return e->convert(c, k, src, dst);
}

const struct infer_edge* infer_edge_convert(const struct infer_registry* r,
                                            struct infer_edge_cache* k,
                                            struct infer_ctx* c,
                                            struct infer_tensor* src,
                                            enum infer_domain dst_domain,
                                            struct infer_tensor* dst) {
    const struct infer_edge* e = infer_registry_find(r, src->domain, dst_domain);
    if (!e || !e->available) { return NULL; }
    return e->convert(c, k, src, dst) == 0 ? e : NULL;
}
