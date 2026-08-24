/* gles_present.h -- the GLES "presenter": everything between the
 * Wayland layer (common.c) and the scene that draws.
 *
 * Extracted from gles-dmabuf.c and gles-eglsurface.c, which stay as the
 * untouched single-file references. The presenter owns the EGL context
 * and the swapchain -- in one of two modes, selectable at init:
 *
 *   GLES_MODE_DMABUF      the hand-rolled swapchain: gbm_bo slots,
 *                         zwp_linux_dmabuf_v1 wl_buffers, busy/release
 *                         tracking, explicit sync via linux-drm-syncobj
 *                         when the compositor offers it (README §14/§15)
 *   GLES_MODE_EGLSURFACE  Mesa's swapchain behind wl_egl_window +
 *                         EGLSurface + eglSwapBuffers
 *
 * The scene never sees a wl_buffer, a slot, or a fence; the presenter
 * never sees a shader, a mesh, or a uniform. The seam is `draw(target)`:
 * the presenter binds a framebuffer, the scene fills it, the presenter
 * presents it.
 *
 * One property the scene MUST know: in dmabuf mode the target presents
 * with NDC y = -1 at the TOP row (FBO rendering cancels GL's usual
 * y-flip -- README §14); in eglsurface mode it presents GL-style, y up.
 * Read `p->y_down` and build the projection accordingly (mat4.h).
 */
#ifndef HELLO_WAYLAND_GLES_PRESENT_H
#define HELLO_WAYLAND_GLES_PRESENT_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "common.h"

struct gbm_bo;
struct gbm_device;
struct wl_egl_window;
struct zwp_linux_dmabuf_v1;
struct wp_linux_drm_syncobj_manager_v1;
struct wp_linux_drm_syncobj_surface_v1;
struct wp_linux_drm_syncobj_timeline_v1;

enum gles_mode {
    GLES_MODE_DMABUF,
    GLES_MODE_EGLSURFACE,
};

enum {
    GLES_PRESENT_DEPTH = 1 << 0, /* targets carry a depth buffer */
};

enum { GLES_SLOTS = 2 };

/* What a scene draws into, this frame. Already bound as GL_FRAMEBUFFER
 * when draw() is called. */
struct gles_target {
    GLuint fbo; /* 0 = the EGLSurface's default framebuffer (eglsurface mode) */
    int width, height;
    int slot; /* swapchain slot index, for per-slot scene resources */
};

struct gles_presenter;

struct gles_scene {
    /* Context is current. Compile programs, upload meshes. Called once. */
    void (*init)(struct gles_presenter* p, void* user);
    /* Optional: targets now have this size. */
    void (*resize)(struct gles_presenter* p, int w, int h, void* user);
    /* Fill the bound target. Do not present; the presenter does. */
    void (*draw)(struct gles_presenter* p, const struct gles_target* t, void* user);
    /* Context still current. Release GL objects. */
    void (*fini)(struct gles_presenter* p, void* user);
};

/* dmabuf mode: one swapchain slot (gles-dmabuf.c's struct egl_slot). */
struct gles_slot {
    struct gbm_bo* bo;
    EGLImage image;
    GLuint rbo, fbo;
    struct wl_buffer* buffer;
    int busy;

    uint32_t acquire_syncobj, release_syncobj;
    struct wp_linux_drm_syncobj_timeline_v1* acquire_timeline;
    struct wp_linux_drm_syncobj_timeline_v1* release_timeline;
    uint64_t point;
};

struct gles_presenter {
    struct app app; /* MUST be first */

    enum gles_mode mode;
    unsigned flags;
    const struct gles_scene* scene;
    void* user;
    int scene_ready;

    int y_down; /* 1 in dmabuf mode: NDC y=-1 is the top row on screen */

    /* the context factory (both modes) */
    EGLDisplay egl_display;
    EGLConfig egl_config;
    EGLContext egl_context;

    /* eglsurface mode */
    struct wl_egl_window* egl_window;
    EGLSurface egl_surface;

    /* dmabuf mode */
    int drm_fd;
    struct gbm_device* gbm;
    struct zwp_linux_dmabuf_v1* dmabuf;
    struct wp_linux_drm_syncobj_manager_v1* syncobj_manager;
    struct wp_linux_drm_syncobj_surface_v1* surface_sync;
    PFNEGLCREATESYNCKHRPROC create_sync;
    PFNEGLDESTROYSYNCKHRPROC destroy_sync;
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fence_fd;
    PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC image_target_rbo;
    struct gles_slot slots[GLES_SLOTS];
    GLuint depth_rbo; /* shared by all slots: one context, frames are sequential */

    int buf_w, buf_h;
};

/* Connect (app_init), bring up EGL in the chosen mode, and -- in dmabuf
 * mode, where the context is current immediately -- call scene->init.
 * In eglsurface mode scene->init runs on the first configure, when the
 * EGLSurface exists. Returns 0 on success. */
int gles_present_init(struct gles_presenter* p,
                      enum gles_mode mode,
                      unsigned flags,
                      const struct gles_scene* scene,
                      void* user,
                      const char* title,
                      const char* app_id);

/* app_run: the frame-callback loop, presenting one frame per tick. */
void gles_present_run(struct gles_presenter* p);

/* scene->fini, then tear down the swapchain, EGL, and the connection. */
void gles_present_fini(struct gles_presenter* p);

#endif
