/* shm_present.h -- the CPU "presenter": wl_shm pool + slots, extracted
 * from hello/shm.c, handing a scene a pointer to pixels.
 *
 * The seam is the same as the GPU presenters' (init / resize / draw /
 * fini), but the target is memory you can address: a row-major
 * XRGB8888 array the compositor maps too (README §4 step 8), plus --
 * opt-in -- a float depth buffer of the same size. The depth buffer is
 * ours alone (the compositor never reads depth), so it is a plain
 * malloc, the CPU twin of the GL depth renderbuffer / Vulkan depth
 * image the other presenters allocate.
 *
 * Presents with row 0 at the TOP: NDC y=-1 is the top row -- the same
 * y-down convention as the dmabuf FBOs and Vulkan (README §17 table),
 * so scenes build projections with y_down = 1.
 */
#ifndef HELLO_WAYLAND_SHM_PRESENT_H
#define HELLO_WAYLAND_SHM_PRESENT_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

enum {
    SHM_PRESENT_DEPTH = 1 << 0, /* targets carry a float depth buffer */
};

enum { SHM_SLOTS = 2 };

struct shm_target {
    uint32_t* pixels; /* XRGB8888, row-major, stride_px pixels per row */
    int width, height;
    int stride_px;
    float* depth; /* NULL without SHM_PRESENT_DEPTH; width*height floats */
    int slot;
};

struct shm_presenter;

struct shm_scene {
    void (*init)(struct shm_presenter* p, void* user);
    void (*resize)(struct shm_presenter* p, int w, int h, void* user); /* optional */
    void (*draw)(struct shm_presenter* p, const struct shm_target* t, void* user);
    void (*fini)(struct shm_presenter* p, void* user);
};

struct shm_slot {
    struct wl_buffer* buffer;
    int busy;
};

struct shm_presenter {
    struct app app; /* MUST be first */

    unsigned flags;
    const struct shm_scene* scene;
    void* user;

    struct wl_shm_pool* pool;
    uint32_t* pool_data;
    size_t pool_bytes;
    struct shm_slot slots[SHM_SLOTS];
    float* depth; /* one, shared: frames are drawn one after another */
    int buf_w, buf_h;
};

int shm_present_init(struct shm_presenter* p,
                     unsigned flags,
                     const struct shm_scene* scene,
                     void* user,
                     const char* title,
                     const char* app_id);
void shm_present_run(struct shm_presenter* p);
void shm_present_fini(struct shm_presenter* p);

#endif
