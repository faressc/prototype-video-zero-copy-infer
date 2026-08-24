/* shm_present.c -- see shm_present.h. hello/shm.c with fill_gradient
 * cut out and a scene vtable put in its place. */

#define _GNU_SOURCE /* for memfd_create */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shm_present.h"

static void on_buffer_release(void* data, struct wl_buffer* buffer) {
    struct shm_slot* s = data;
    if (s->buffer == buffer) {
        s->busy = 0;
    } else {
        wl_buffer_destroy(buffer); /* orphan from a resize */
    }
}

static const struct wl_buffer_listener buffer_listener = {
    .release = on_buffer_release,
};

static void present_configure(struct app* a) {
    struct shm_presenter* p = (struct shm_presenter*)a;
    const int w = a->width;
    const int h = a->height;
    if (p->pool && w == p->buf_w && h == p->buf_h) { return; }

    for (int i = 0; i < SHM_SLOTS; i++) {
        if (p->slots[i].buffer && !p->slots[i].busy) { wl_buffer_destroy(p->slots[i].buffer); }
    }
    if (p->pool) {
        wl_shm_pool_destroy(p->pool);
        munmap(p->pool_data, p->pool_bytes);
    }

    const size_t slot_bytes = (size_t)w * 4 * h;
    p->pool_bytes = slot_bytes * SHM_SLOTS;

    int fd = memfd_create("hello-wayland", 0);
    if (fd < 0 || ftruncate(fd, (off_t)p->pool_bytes) < 0) {
        perror("shm alloc");
        exit(1);
    }
    p->pool_data = mmap(NULL, p->pool_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p->pool_data == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    p->pool = wl_shm_create_pool(a->shm, fd, (int32_t)p->pool_bytes);
    close(fd);

    for (int i = 0; i < SHM_SLOTS; i++) {
        p->slots[i].buffer = wl_shm_pool_create_buffer(p->pool,
                                                       (int32_t)(i * slot_bytes),
                                                       w,
                                                       h,
                                                       w * 4,
                                                       WL_SHM_FORMAT_XRGB8888);
        wl_buffer_add_listener(p->slots[i].buffer, &buffer_listener, &p->slots[i]);
        p->slots[i].busy = 0;
    }

    /* The depth buffer is not shared with anyone: plain memory, not in
     * the pool, not a wl_buffer. Its CPU-ness is the whole point. */
    if (p->flags & SHM_PRESENT_DEPTH) {
        free(p->depth);
        p->depth = malloc((size_t)w * (size_t)h * sizeof(float));
        if (!p->depth) {
            perror("depth alloc");
            exit(1);
        }
    }

    p->buf_w = w;
    p->buf_h = h;
    if (p->scene->resize) { p->scene->resize(p, w, h, p->user); }
}

static void present_redraw(struct app* a) {
    struct shm_presenter* p = (struct shm_presenter*)a;

    struct shm_slot* s = NULL;
    int slot = 0;
    for (int i = 0; i < SHM_SLOTS; i++) {
        if (!p->slots[i].busy) {
            s = &p->slots[i];
            slot = i;
            break;
        }
    }
    if (!s) { return; } /* both on loan: skip this frame */

    struct shm_target t = {
        .pixels = p->pool_data + (size_t)slot * (size_t)p->buf_w * (size_t)p->buf_h,
        .width = p->buf_w,
        .height = p->buf_h,
        .stride_px = p->buf_w,
        .depth = p->depth,
        .slot = slot,
    };
    p->scene->draw(p, &t, p->user);

    wl_surface_attach(a->surface, s->buffer, 0, 0);
    wl_surface_damage_buffer(a->surface, 0, 0, INT32_MAX, INT32_MAX);
    s->busy = 1;
    wl_surface_commit(a->surface);
}

static const struct app_backend present_backend = {
    .configure = present_configure,
    .redraw = present_redraw,
};

int shm_present_init(struct shm_presenter* p,
                     unsigned flags,
                     const struct shm_scene* scene,
                     void* user,
                     const char* title,
                     const char* app_id) {
    memset(p, 0, sizeof(*p));
    p->flags = flags;
    p->scene = scene;
    p->user = user;
    if (app_init(&p->app, &present_backend, title, app_id) < 0) { return -1; }
    p->scene->init(p, p->user); /* no context to wait for on the CPU */
    return 0;
}

void shm_present_run(struct shm_presenter* p) {
    app_run(&p->app);
}

void shm_present_fini(struct shm_presenter* p) {
    p->scene->fini(p, p->user);
    for (int i = 0; i < SHM_SLOTS; i++) {
        if (p->slots[i].buffer) { wl_buffer_destroy(p->slots[i].buffer); }
    }
    if (p->pool) {
        wl_shm_pool_destroy(p->pool);
        munmap(p->pool_data, p->pool_bytes);
    }
    free(p->depth);
    app_finish(&p->app);
}
