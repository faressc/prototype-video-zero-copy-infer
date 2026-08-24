/* shm.c -- the CPU backend: draw pixels into shared memory.
 *
 * wl_shm: we create a plain memory file (memfd), mmap it, draw into
 * it, and pass the fd over the socket. The compositor maps the same
 * file -> both sides see the same pixels, zero copy. Because of that
 * sharing, a committed buffer is ON LOAN: we may not draw into it
 * again until the compositor returns it with wl_buffer.release.
 * Hence SLOTS buffers: draw into a free one while another is on
 * screen. All protocol plumbing lives in common.c.
 */

#define _GNU_SOURCE /* for memfd_create */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common.h"

/* Double buffering: one buffer on screen, one being drawn. Bump to 3
 * (triple buffering) if a compositor ever holds buffers long enough
 * that redraw has to skip frames. */
enum { SLOTS = 2 };

struct slot {
    struct wl_buffer* buffer; /* this slot's CURRENT protocol object */
    int busy;                 /* committed and not yet released */
};

struct shm_app {
    struct app app; /* MUST be first: hooks cast app* back to shm_app* */

    /* ONE pool holding SLOTS buffers side by side, recreated only when
     * the size changes. Steady-state animation allocates nothing. */
    struct wl_shm_pool* pool;
    uint32_t* pool_data;
    size_t pool_bytes;
    struct slot slots[SLOTS];
    int buf_w, buf_h;
};

/* release: the compositor stopped reading this buffer. Still a slot's
 * current buffer -> free again; replaced by a resize meanwhile ->
 * orphan, destroy it now (not earlier: the pages were being read). */
static void on_buffer_release(void* data, struct wl_buffer* buffer) {
    struct slot* s = data;
    if (s->buffer == buffer) {
        s->busy = 0;
    } else {
        wl_buffer_destroy(buffer);
    }
}

static const struct wl_buffer_listener buffer_listener = {
    .release = on_buffer_release,
};

/* A resize gets a FRESH pool instead of growing the old one in place:
 * reusing offsets in a grown pool could overlap a buffer the
 * compositor still reads. Destroying the old pool object immediately
 * is explicitly legal -- each buffer holds a reference, the storage
 * lives until the last buffer dies. */
static void shm_configure(struct app* a) {
    struct shm_app* s = (struct shm_app*)a;
    const int w = a->width;
    const int h = a->height;
    if (s->pool && w == s->buf_w && h == s->buf_h) { return; }

    for (int i = 0; i < SLOTS; i++) {
        /* free slots die now; busy ones become orphans */
        if (s->slots[i].buffer && !s->slots[i].busy) { wl_buffer_destroy(s->slots[i].buffer); }
    }
    if (s->pool) {
        wl_shm_pool_destroy(s->pool);
        munmap(s->pool_data, s->pool_bytes);
    }

    const size_t slot_bytes = (size_t)w * 4 * h; /* XRGB8888 */
    s->pool_bytes = slot_bytes * SLOTS;

    int fd = memfd_create("hello-wayland", 0);
    if (fd < 0 || ftruncate(fd, (off_t)s->pool_bytes) < 0) {
        perror("shm alloc");
        exit(1);
    }
    s->pool_data = mmap(NULL, s->pool_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (s->pool_data == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    s->pool = wl_shm_create_pool(a->shm, fd, (int32_t)s->pool_bytes);
    close(fd); /* mappings + pool keep the pages alive */

    for (int i = 0; i < SLOTS; i++) {
        s->slots[i].buffer = wl_shm_pool_create_buffer(s->pool,
                                                       (int32_t)(i * slot_bytes),
                                                       w,
                                                       h,
                                                       w * 4,
                                                       WL_SHM_FORMAT_XRGB8888);
        wl_buffer_add_listener(s->slots[i].buffer, &buffer_listener, &s->slots[i]);
        s->slots[i].busy = 0;
    }
    s->buf_w = w;
    s->buf_h = h;
}

/* The "shader", CPU edition: the same per-pixel function egl.c and
 * vulkan.c run on the GPU, here as two nested loops. */
static void fill_gradient(uint32_t* px, int w, int h, uint32_t t, int cx, int cy) {
    const int shift = (int)(t / 8);
    for (int y = 0; y < h; y++) {
        const uint32_t g = (uint32_t)(((y - cy) * 255 / h) & 255);
        for (int x = 0; x < w; x++) {
            const uint32_t r = (uint32_t)((((x - cx) * 255 / w) + shift) & 255);
            const uint32_t b = 255 - r;
            px[(size_t)y * (size_t)w + (size_t)x] = (r << 16) | (g << 8) | b;
        }
    }
}

/* Draw one frame into a free slot and commit it. If every slot is on
 * loan, skip -- the next frame callback tries again. */
static void shm_redraw(struct app* a) {
    struct shm_app* s = (struct shm_app*)a;

    struct slot* free_slot = NULL;
    for (int i = 0; i < SLOTS; i++) {
        if (!s->slots[i].busy) {
            free_slot = &s->slots[i];
            break;
        }
    }
    if (!free_slot) { return; }

    uint32_t* px =
        s->pool_data + (size_t)(free_slot - s->slots) * (size_t)s->buf_w * (size_t)s->buf_h;
    fill_gradient(px, s->buf_w, s->buf_h, app_anim_time(a), (int)a->ptr_x, (int)a->ptr_y);

    wl_surface_attach(a->surface, free_slot->buffer, 0, 0);
    wl_surface_damage_buffer(a->surface, 0, 0, INT32_MAX, INT32_MAX);
    free_slot->busy = 1;
    wl_surface_commit(a->surface);
}

static const struct app_backend shm_backend = {
    .configure = shm_configure,
    .redraw = shm_redraw, /* non-NULL: common runs the frame loop */
};

int main(void) {
    struct shm_app s = {0};

    if (app_init(&s.app, &shm_backend, "hello wayland", "hello-wayland") < 0) { return 1; }
    app_run(&s.app);

    for (int i = 0; i < SLOTS; i++) {
        if (s.slots[i].buffer) { wl_buffer_destroy(s.slots[i].buffer); }
    }
    if (s.pool) {
        wl_shm_pool_destroy(s.pool);
        munmap(s.pool_data, s.pool_bytes);
    }
    app_finish(&s.app);
    return 0;
}
