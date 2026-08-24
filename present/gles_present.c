/* gles_present.c -- see gles_present.h. The code is gles-dmabuf.c and
 * gles-eglsurface.c with the gradient cut out and a scene vtable put
 * in its place; comments explaining the machinery live in those files
 * and in README §14/§15. Two additions: the depth buffer (a plain
 * renderbuffer shared by the slots in dmabuf mode, EGL_DEPTH_SIZE in
 * eglsurface mode) and the y_down flag the scene reads. */

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <drm_fourcc.h>
#include <gbm.h>
#include <wayland-egl.h>
#include <xf86drm.h>

#include "gles_present.h"
#include "linux-dmabuf-v1-client-protocol.h"
#include "linux-drm-syncobj-v1-client-protocol.h"

/* ------------------------------------------------------------------ */
/* dmabuf mode: the swapchain by hand (gles-dmabuf.c)                 */
/* ------------------------------------------------------------------ */

static void on_buffer_release(void* data, struct wl_buffer* buffer) {
    struct gles_slot* s = data;
    if (s->buffer == buffer) {
        s->busy = 0;
    } else {
        wl_buffer_destroy(buffer); /* orphan from a resize */
    }
}

static const struct wl_buffer_listener buffer_listener = {
    .release = on_buffer_release,
};

static void destroy_slot_gl(struct gles_presenter* p, struct gles_slot* s) {
    if (s->fbo) { glDeleteFramebuffers(1, &s->fbo); }
    if (s->rbo) { glDeleteRenderbuffers(1, &s->rbo); }
    if (s->image != EGL_NO_IMAGE) { eglDestroyImage(p->egl_display, s->image); }
    if (s->bo) { gbm_bo_destroy(s->bo); }
    s->fbo = 0;
    s->rbo = 0;
    s->image = EGL_NO_IMAGE;
    s->bo = NULL;
}

static struct wp_linux_drm_syncobj_timeline_v1* create_timeline(struct gles_presenter* p,
                                                                uint32_t* handle) {
    int fd = -1;
    if (drmSyncobjCreate(p->drm_fd, 0, handle) < 0 ||
        drmSyncobjHandleToFD(p->drm_fd, *handle, &fd) < 0) {
        perror("drmSyncobjCreate");
        exit(1);
    }
    struct wp_linux_drm_syncobj_timeline_v1* timeline =
        wp_linux_drm_syncobj_manager_v1_import_timeline(p->syncobj_manager, fd);
    close(fd);
    return timeline;
}

static void destroy_slot_sync(struct gles_presenter* p, struct gles_slot* s) {
    if (s->acquire_timeline) { wp_linux_drm_syncobj_timeline_v1_destroy(s->acquire_timeline); }
    if (s->release_timeline) { wp_linux_drm_syncobj_timeline_v1_destroy(s->release_timeline); }
    if (s->acquire_syncobj) { drmSyncobjDestroy(p->drm_fd, s->acquire_syncobj); }
    if (s->release_syncobj) { drmSyncobjDestroy(p->drm_fd, s->release_syncobj); }
    s->acquire_timeline = NULL;
    s->release_timeline = NULL;
    s->acquire_syncobj = 0;
    s->release_syncobj = 0;
    s->point = 0;
}

static int slot_is_free(struct gles_presenter* p, struct gles_slot* s) {
    if (s->busy && p->surface_sync) {
        uint64_t value = 0;
        if (drmSyncobjQuery(p->drm_fd, &s->release_syncobj, &value, 1) == 0 && value >= s->point) {
            s->busy = 0;
        }
    }
    return !s->busy;
}

static void dmabuf_configure(struct gles_presenter* p) {
    struct app* a = &p->app;
    const int w = a->width;
    const int h = a->height;
    if (p->slots[0].buffer && w == p->buf_w && h == p->buf_h) { return; }

    /* The depth buffer is NOT a dmabuf: the compositor never reads it,
     * so it is an ordinary driver-allocated renderbuffer. One is enough
     * for all slots -- a single context renders frames one after the
     * other, and GL orders its own commands. */
    if (p->flags & GLES_PRESENT_DEPTH) {
        if (p->depth_rbo) { glDeleteRenderbuffers(1, &p->depth_rbo); }
        glGenRenderbuffers(1, &p->depth_rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, p->depth_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, w, h);
    }

    for (int i = 0; i < GLES_SLOTS; i++) {
        struct gles_slot* s = &p->slots[i];
        destroy_slot_gl(p, s);
        if (p->surface_sync && s->busy && s->point) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            drmSyncobjTimelineWait(p->drm_fd,
                                   &s->release_syncobj,
                                   &s->point,
                                   1,
                                   (int64_t)now.tv_sec * 1000000000 + now.tv_nsec + 100000000,
                                   DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                   NULL);
            s->busy = 0;
        }
        destroy_slot_sync(p, s);
        if (s->buffer && !s->busy) { wl_buffer_destroy(s->buffer); }

        s->bo = gbm_bo_create(p->gbm,
                              (uint32_t)w,
                              (uint32_t)h,
                              DRM_FORMAT_XRGB8888,
                              GBM_BO_USE_RENDERING);
        if (!s->bo) {
            fprintf(stderr, "gbm_bo_create failed\n");
            exit(1);
        }
        int fd = gbm_bo_get_fd(s->bo);
        uint32_t stride = gbm_bo_get_stride(s->bo);
        uint64_t modifier = gbm_bo_get_modifier(s->bo);

        EGLAttrib attrs[] = {
            EGL_WIDTH,
            w,
            EGL_HEIGHT,
            h,
            EGL_LINUX_DRM_FOURCC_EXT,
            DRM_FORMAT_XRGB8888,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            (EGLAttrib)stride,
            EGL_NONE,
            EGL_NONE,
            EGL_NONE,
            EGL_NONE,
            EGL_NONE,
        };
        if (modifier != DRM_FORMAT_MOD_INVALID) {
            attrs[12] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
            attrs[13] = (EGLAttrib)(modifier & 0xffffffff);
            attrs[14] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
            attrs[15] = (EGLAttrib)(modifier >> 32);
        }
        s->image =
            eglCreateImage(p->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
        if (s->image == EGL_NO_IMAGE) {
            fprintf(stderr, "eglCreateImage failed\n");
            exit(1);
        }
        glGenRenderbuffers(1, &s->rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, s->rbo);
        p->image_target_rbo(GL_RENDERBUFFER, s->image);
        glGenFramebuffers(1, &s->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, s->rbo);
        if (p->flags & GLES_PRESENT_DEPTH) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                      GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER,
                                      p->depth_rbo);
        }
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "FBO incomplete\n");
            exit(1);
        }

        struct zwp_linux_buffer_params_v1* params = zwp_linux_dmabuf_v1_create_params(p->dmabuf);
        zwp_linux_buffer_params_v1_add(params,
                                       fd,
                                       0,
                                       0,
                                       stride,
                                       (uint32_t)(modifier >> 32),
                                       (uint32_t)(modifier & 0xffffffff));
        s->buffer = zwp_linux_buffer_params_v1_create_immed(params, w, h, DRM_FORMAT_XRGB8888, 0);
        zwp_linux_buffer_params_v1_destroy(params);
        if (p->surface_sync) {
            s->acquire_timeline = create_timeline(p, &s->acquire_syncobj);
            s->release_timeline = create_timeline(p, &s->release_syncobj);
        } else {
            wl_buffer_add_listener(s->buffer, &buffer_listener, s);
        }
        s->busy = 0;

        close(fd);
    }
    p->buf_w = w;
    p->buf_h = h;
}

static void dmabuf_redraw(struct gles_presenter* p) {
    struct app* a = &p->app;

    struct gles_slot* s = NULL;
    int slot = 0;
    for (int i = 0; i < GLES_SLOTS; i++) {
        if (slot_is_free(p, &p->slots[i])) {
            s = &p->slots[i];
            slot = i;
            break;
        }
    }
    if (!s) { return; } /* both on loan: skip this frame */

    glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
    struct gles_target t = {.fbo = s->fbo, .width = p->buf_w, .height = p->buf_h, .slot = slot};
    p->scene->draw(p, &t, p->user);

    if (p->surface_sync) {
        s->point++;
        EGLSyncKHR sync = p->create_sync(p->egl_display, EGL_SYNC_NATIVE_FENCE_ANDROID, NULL);
        glFlush();
        EGLint fence_fd = sync != EGL_NO_SYNC_KHR ? p->dup_fence_fd(p->egl_display, sync)
                                                  : EGL_NO_NATIVE_FENCE_FD_ANDROID;
        if (sync != EGL_NO_SYNC_KHR) { p->destroy_sync(p->egl_display, sync); }

        if (fence_fd >= 0) {
            uint32_t tmp = 0;
            if (drmSyncobjCreate(p->drm_fd, 0, &tmp) < 0 ||
                drmSyncobjImportSyncFile(p->drm_fd, tmp, fence_fd) < 0 ||
                drmSyncobjTransfer(p->drm_fd, s->acquire_syncobj, s->point, tmp, 0, 0) < 0) {
                fprintf(stderr, "syncobj import failed\n");
                exit(1);
            }
            drmSyncobjDestroy(p->drm_fd, tmp);
            close(fence_fd);
        } else {
            glFinish();
            drmSyncobjTimelineSignal(p->drm_fd, &s->acquire_syncobj, &s->point, 1);
        }
        wp_linux_drm_syncobj_surface_v1_set_acquire_point(p->surface_sync,
                                                          s->acquire_timeline,
                                                          (uint32_t)(s->point >> 32),
                                                          (uint32_t)(s->point & 0xffffffff));
        wp_linux_drm_syncobj_surface_v1_set_release_point(p->surface_sync,
                                                          s->release_timeline,
                                                          (uint32_t)(s->point >> 32),
                                                          (uint32_t)(s->point & 0xffffffff));
    } else {
        glFlush();
    }

    wl_surface_attach(a->surface, s->buffer, 0, 0);
    wl_surface_damage_buffer(a->surface, 0, 0, INT32_MAX, INT32_MAX);
    s->busy = 1;
    wl_surface_commit(a->surface);
}

/* Bind zwp_linux_dmabuf_v1 (+ the optional syncobj manager) on a
 * private queue, then hand the proxies to the default queue. */
static void on_dmabuf_global(void* data,
                             struct wl_registry* registry,
                             uint32_t name,
                             const char* interface,
                             uint32_t version) {
    struct gles_presenter* p = data;
    if (strcmp(interface, zwp_linux_dmabuf_v1_interface.name) == 0) {
        p->dmabuf = wl_registry_bind(registry,
                                     name,
                                     &zwp_linux_dmabuf_v1_interface,
                                     version < 3 ? version : 3);
    } else if (strcmp(interface, wp_linux_drm_syncobj_manager_v1_interface.name) == 0) {
        p->syncobj_manager =
            wl_registry_bind(registry, name, &wp_linux_drm_syncobj_manager_v1_interface, 1);
    }
}

static void on_dmabuf_global_remove(void* d, struct wl_registry* r, uint32_t n) {
    (void)d;
    (void)r;
    (void)n;
}

static const struct wl_registry_listener dmabuf_registry_listener = {
    .global = on_dmabuf_global,
    .global_remove = on_dmabuf_global_remove,
};

static void bind_dmabuf(struct gles_presenter* p) {
    struct wl_display* display = p->app.display;

    struct wl_event_queue* queue = wl_display_create_queue(display);
    struct wl_display* wrapped = wl_proxy_create_wrapper(display);
    wl_proxy_set_queue((struct wl_proxy*)wrapped, queue);

    struct wl_registry* registry = wl_display_get_registry(wrapped);
    wl_registry_add_listener(registry, &dmabuf_registry_listener, p);
    wl_display_roundtrip_queue(display, queue);

    wl_registry_destroy(registry);
    wl_proxy_wrapper_destroy(wrapped);
    if (!p->dmabuf) {
        fprintf(stderr, "compositor lacks zwp_linux_dmabuf_v1\n");
        exit(1);
    }
    wl_proxy_set_queue((struct wl_proxy*)p->dmabuf, NULL);
    if (p->syncobj_manager) { wl_proxy_set_queue((struct wl_proxy*)p->syncobj_manager, NULL); }
    wl_event_queue_destroy(queue);
}

static int dmabuf_init(struct gles_presenter* p) {
    bind_dmabuf(p);

    p->drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (p->drm_fd < 0) {
        perror("open render node");
        return -1;
    }
    p->gbm = gbm_create_device(p->drm_fd);

    p->egl_display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, p->gbm, NULL);
    if (p->egl_display == EGL_NO_DISPLAY || !eglInitialize(p->egl_display, NULL, NULL)) {
        fprintf(stderr, "EGL init failed\n");
        return -1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    const char* exts = eglQueryString(p->egl_display, EGL_EXTENSIONS);
    if (!strstr(exts, "EGL_KHR_surfaceless_context")) {
        fprintf(stderr, "need EGL_KHR_surfaceless_context\n");
        return -1;
    }

    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,
        0,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_NONE,
    };
    EGLint n = 0;
    if (!eglChooseConfig(p->egl_display, config_attribs, &p->egl_config, 1, &n) || n < 1) {
        fprintf(stderr, "no EGL config\n");
        return -1;
    }

    static const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    p->egl_context = eglCreateContext(p->egl_display, p->egl_config, EGL_NO_CONTEXT, ctx_attribs);
    if (p->egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        return -1;
    }
    eglMakeCurrent(p->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, p->egl_context);

    p->image_target_rbo = (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress(
        "glEGLImageTargetRenderbufferStorageOES");
    if (!p->image_target_rbo) {
        fprintf(stderr, "need GL_OES_EGL_image\n");
        return -1;
    }

    if (p->syncobj_manager && strstr(exts, "EGL_ANDROID_native_fence_sync")) {
        p->create_sync = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
        p->destroy_sync = (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
        p->dup_fence_fd =
            (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress("eglDupNativeFenceFDANDROID");
        if (p->create_sync && p->destroy_sync && p->dup_fence_fd) {
            p->surface_sync =
                wp_linux_drm_syncobj_manager_v1_get_surface(p->syncobj_manager, p->app.surface);
        }
    }
    if (!p->surface_sync) {
        fprintf(stderr, "explicit sync unavailable -- falling back to implicit sync\n");
    }

    /* surfaceless context is current now: the scene can build itself
     * before any configure fires */
    p->scene->init(p, p->user);
    p->scene_ready = 1;
    return 0;
}

static void dmabuf_fini(struct gles_presenter* p) {
    for (int i = 0; i < GLES_SLOTS; i++) {
        destroy_slot_gl(p, &p->slots[i]);
        destroy_slot_sync(p, &p->slots[i]);
        if (p->slots[i].buffer) { wl_buffer_destroy(p->slots[i].buffer); }
    }
    if (p->depth_rbo) { glDeleteRenderbuffers(1, &p->depth_rbo); }
    if (p->surface_sync) { wp_linux_drm_syncobj_surface_v1_destroy(p->surface_sync); }
    if (p->syncobj_manager) { wp_linux_drm_syncobj_manager_v1_destroy(p->syncobj_manager); }
    if (p->dmabuf) { zwp_linux_dmabuf_v1_destroy(p->dmabuf); }
    eglMakeCurrent(p->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (p->egl_context != EGL_NO_CONTEXT) { eglDestroyContext(p->egl_display, p->egl_context); }
    if (p->egl_display != EGL_NO_DISPLAY) { eglTerminate(p->egl_display); }
    if (p->gbm) { gbm_device_destroy(p->gbm); }
    if (p->drm_fd >= 0) { close(p->drm_fd); }
}

/* ------------------------------------------------------------------ */
/* eglsurface mode: Mesa's swapchain (gles-eglsurface.c)              */
/* ------------------------------------------------------------------ */

static void eglsurface_configure(struct gles_presenter* p) {
    struct app* a = &p->app;

    if (!p->egl_window) {
        p->egl_window = wl_egl_window_create(a->surface, a->width, a->height);
        p->egl_surface =
            eglCreatePlatformWindowSurface(p->egl_display, p->egl_config, p->egl_window, NULL);
        if (p->egl_surface == EGL_NO_SURFACE) {
            fprintf(stderr, "eglCreatePlatformWindowSurface failed\n");
            exit(1);
        }
        eglMakeCurrent(p->egl_display, p->egl_surface, p->egl_surface, p->egl_context);
        eglSwapInterval(p->egl_display, 0); /* common's frame callbacks pace us */
        /* this context style needs a current SURFACE -- only now can
         * the scene run GL */
        p->scene->init(p, p->user);
        p->scene_ready = 1;
    } else if (a->width != p->buf_w || a->height != p->buf_h) {
        wl_egl_window_resize(p->egl_window, a->width, a->height, 0, 0);
    }
    p->buf_w = a->width;
    p->buf_h = a->height;
}

static void eglsurface_redraw(struct gles_presenter* p) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    struct gles_target t = {.fbo = 0, .width = p->buf_w, .height = p->buf_h, .slot = 0};
    p->scene->draw(p, &t, p->user);
    eglSwapBuffers(p->egl_display, p->egl_surface);
}

static int eglsurface_init(struct gles_presenter* p) {
    p->egl_display = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, p->app.display, NULL);
    if (p->egl_display == EGL_NO_DISPLAY || !eglInitialize(p->egl_display, NULL, NULL)) {
        fprintf(stderr, "EGL init failed\n");
        return -1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    /* The depth buffer is part of the CONFIG here: Mesa allocates it
     * alongside each swapchain color buffer; we never see it. */
    const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,
        EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_DEPTH_SIZE,
        (p->flags & GLES_PRESENT_DEPTH) ? 16 : 0,
        EGL_NONE,
    };
    EGLint n = 0;
    if (!eglChooseConfig(p->egl_display, config_attribs, &p->egl_config, 1, &n) || n < 1) {
        fprintf(stderr, "no EGL config\n");
        return -1;
    }

    static const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    p->egl_context = eglCreateContext(p->egl_display, p->egl_config, EGL_NO_CONTEXT, ctx_attribs);
    if (p->egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        return -1;
    }
    return 0;
}

static void eglsurface_fini(struct gles_presenter* p) {
    eglMakeCurrent(p->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (p->egl_surface != EGL_NO_SURFACE) { eglDestroySurface(p->egl_display, p->egl_surface); }
    if (p->egl_window) { wl_egl_window_destroy(p->egl_window); }
    if (p->egl_context != EGL_NO_CONTEXT) { eglDestroyContext(p->egl_display, p->egl_context); }
    if (p->egl_display != EGL_NO_DISPLAY) { eglTerminate(p->egl_display); }
}

/* ------------------------------------------------------------------ */
/* The two hooks, dispatching on mode                                 */
/* ------------------------------------------------------------------ */

static void present_configure(struct app* a) {
    struct gles_presenter* p = (struct gles_presenter*)a;
    if (p->mode == GLES_MODE_DMABUF) {
        dmabuf_configure(p);
    } else {
        eglsurface_configure(p);
    }
    if (p->scene->resize) { p->scene->resize(p, a->width, a->height, p->user); }
}

static void present_redraw(struct app* a) {
    struct gles_presenter* p = (struct gles_presenter*)a;
    if (p->mode == GLES_MODE_DMABUF) {
        dmabuf_redraw(p);
    } else {
        eglsurface_redraw(p);
    }
}

static const struct app_backend present_backend = {
    .configure = present_configure,
    .redraw = present_redraw,
};

/* ------------------------------------------------------------------ */

int gles_present_init(struct gles_presenter* p,
                      enum gles_mode mode,
                      unsigned flags,
                      const struct gles_scene* scene,
                      void* user,
                      const char* title,
                      const char* app_id) {
    memset(p, 0, sizeof(*p));
    p->mode = mode;
    p->flags = flags;
    p->scene = scene;
    p->user = user;
    p->y_down = (mode == GLES_MODE_DMABUF);
    p->drm_fd = -1;
    p->egl_display = EGL_NO_DISPLAY;
    p->egl_context = EGL_NO_CONTEXT;
    p->egl_surface = EGL_NO_SURFACE;
    for (int i = 0; i < GLES_SLOTS; i++) { p->slots[i].image = EGL_NO_IMAGE; }

    if (app_init(&p->app, &present_backend, title, app_id) < 0) { return -1; }

    return mode == GLES_MODE_DMABUF ? dmabuf_init(p) : eglsurface_init(p);
}

void gles_present_run(struct gles_presenter* p) {
    app_run(&p->app);
}

void gles_present_fini(struct gles_presenter* p) {
    if (p->scene_ready) { p->scene->fini(p, p->user); }
    if (p->mode == GLES_MODE_DMABUF) {
        dmabuf_fini(p);
    } else {
        eglsurface_fini(p);
    }
    app_finish(&p->app);
}
