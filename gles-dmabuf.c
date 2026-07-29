/* gles-dmabuf.c -- raw GLES with a HAND-ROLLED swapchain.
 *
 * Sibling of gles-eglsurface.c, which renders the identical gradient
 * through the classic EGLSurface/eglSwapBuffers path -- diff the two
 * to see exactly what eglSwapBuffers does for you.
 *
 * EGL's two halves, separated at last:
 *
 *   KEPT:      the context factory -- display, config, context,
 *              MakeCurrent. GLES structurally cannot create its own
 *              context (no gl* call can run without one being current),
 *              so this rump of EGL stays. Surfaceless: we never create
 *              an EGLSurface.
 *
 *   DISCARDED: the window-system half. No wl_egl_window, no
 *              EGLSurface, no eglSwapBuffers. Instead WE do what Mesa
 *              did behind our back:
 *
 *                gbm_bo_create           allocate GPU memory
 *                gbm_bo_get_fd           export it as a dmabuf fd
 *                eglCreateImage + FBO    make GL render INTO it
 *                zwp_linux_dmabuf_v1     wrap the fd in a wl_buffer
 *                slots + busy + release  the swapchain (shm.c pattern!)
 *
 * Consequences you can verify with WAYLAND_DEBUG=1: no more
 * "{mesa egl surface queue}" lines -- Mesa is no longer a tenant on
 * the socket; every dmabuf message is ours, on the Default Queue.
 *
 * This is the architecture of weston-simple-dmabuf-egl and of
 * compositors themselves (wlroots renders exactly this way).
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h> /* glEGLImageTargetRenderbufferStorageOES */
#include <drm_fourcc.h>   /* DRM_FORMAT_XRGB8888, DRM_FORMAT_MOD_LINEAR */
#include <gbm.h>

#include "common.h"
#include "linux-dmabuf-v1-client-protocol.h" /* generated, bound by US now */

/* Back to explicit double buffering: WE are the swapchain again. */
enum { SLOTS = 2 };

struct egl_slot {
    struct gbm_bo* bo;        /* the GPU allocation */
    EGLImage image;           /* its handle inside EGL/GL */
    GLuint rbo, fbo;          /* GL render target plumbing */
    struct wl_buffer* buffer; /* its handle inside the compositor */
    int busy;                 /* committed, not yet released (shm.c verbatim) */
};

struct egl_app {
    struct app app; /* MUST be first */

    /* our own GPU access -- what eglInitialize did secretly before */
    int drm_fd;
    struct gbm_device* gbm;
    struct zwp_linux_dmabuf_v1* dmabuf;

    /* the surviving rump of EGL: pure context factory */
    EGLDisplay egl_display;
    EGLConfig egl_config;
    EGLContext egl_context;
    PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC image_target_rbo;

    GLuint program;
    GLint a_pos;
    GLint u_size, u_cursor, u_time;

    struct egl_slot slots[SLOTS];
    int buf_w, buf_h;
};

/* ------------------------------------------------------------------ */
/* Shaders (runtime-compiled -- still the GL way).                    */
/* ------------------------------------------------------------------ */

static const char* vert_src =
    "attribute vec2 a_pos;\n"
    "uniform vec2 u_size;\n"
    "varying vec2 v_px;\n"
    "void main() {\n"
    /* NO y-flip, unlike the window-surface version: rendering into an
     * FBO, GL's y=0 row sits at the START of the buffer, which scanout
     * reads as the TOP row -- the two inversions cancel. (Mesa's
     * window surfaces hide this by flipping for you.) */
    "    v_px = (a_pos * 0.5 + 0.5) * u_size;\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char* frag_src =
    /* highp: mediump is REAL fp16 on AGX; pixel-coordinate math
     * overflows 65504 -> banded garbage (see README §14). */
    "precision highp float;\n"
    "varying vec2 v_px;\n"
    "uniform vec2 u_size;\n"
    "uniform vec2 u_cursor;\n"
    "uniform float u_time;\n"
    "void main() {\n"
    "    float shift = u_time / 8.0;\n"
    "    float r = mod((v_px.x - u_cursor.x) * 255.0 / u_size.x + shift, 256.0) / 255.0;\n"
    "    float g = mod((v_px.y - u_cursor.y) * 255.0 / u_size.y, 256.0) / 255.0;\n"
    "    gl_FragColor = vec4(r, g, 1.0 - r, 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);

    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof(log), NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return sh;
}

static void init_gl(struct egl_app* e) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_src);

    e->program = glCreateProgram();
    glAttachShader(e->program, vs);
    glAttachShader(e->program, fs);
    glLinkProgram(e->program);

    GLint ok = 0;
    glGetProgramiv(e->program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(e->program, sizeof(log), NULL, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    e->a_pos = glGetAttribLocation(e->program, "a_pos");
    e->u_size = glGetUniformLocation(e->program, "u_size");
    e->u_cursor = glGetUniformLocation(e->program, "u_cursor");
    e->u_time = glGetUniformLocation(e->program, "u_time");
}

/* ------------------------------------------------------------------ */
/* The swapchain, by hand.                                            */
/* ------------------------------------------------------------------ */

/* Identical to shm.c: current buffer released -> slot free again;
 * orphan from a resize -> destroy now. */
static void on_buffer_release(void* data, struct wl_buffer* buffer) {
    struct egl_slot* s = data;
    if (s->buffer == buffer) {
        s->busy = 0;
    } else {
        wl_buffer_destroy(buffer);
    }
}

static const struct wl_buffer_listener buffer_listener = {
    .release = on_buffer_release,
};

/* Render-side resources can die IMMEDIATELY even if the compositor
 * still displays the buffer: the dmabuf pages are refcounted by the
 * kernel, and the compositor's import holds its own reference. Only
 * the wl_buffer object needs the orphan treatment. */
static void destroy_slot_gl(struct egl_app* e, struct egl_slot* s) {
    if (s->fbo) { glDeleteFramebuffers(1, &s->fbo); }
    if (s->rbo) { glDeleteRenderbuffers(1, &s->rbo); }
    if (s->image != EGL_NO_IMAGE) { eglDestroyImage(e->egl_display, s->image); }
    if (s->bo) { gbm_bo_destroy(s->bo); }
    s->fbo = 0;
    s->rbo = 0;
    s->image = EGL_NO_IMAGE;
    s->bo = NULL;
}

static void egl_configure(struct app* a) {
    struct egl_app* e = (struct egl_app*)a;
    const int w = a->width;
    const int h = a->height;
    if (e->slots[0].buffer && w == e->buf_w && h == e->buf_h) { return; }

    for (int i = 0; i < SLOTS; i++) {
        struct egl_slot* s = &e->slots[i];
        destroy_slot_gl(e, s);
        if (s->buffer && !s->busy) {
            wl_buffer_destroy(s->buffer);
        } /* busy -> orphan; on_buffer_release destroys it */

        /* 1. allocate GPU memory (what Mesa's swapchain did).
         *    LINEAR: universally shareable layout; a real app
         *    negotiates tiled modifiers via dmabuf feedback. */
        s->bo = gbm_bo_create(e->gbm,
                              (uint32_t)w,
                              (uint32_t)h,
                              DRM_FORMAT_XRGB8888,
                              GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
        if (!s->bo) {
            fprintf(stderr, "gbm_bo_create failed\n");
            exit(1);
        }
        int fd = gbm_bo_get_fd(s->bo); /* GPU memory, wearing an fd */
        uint32_t stride = gbm_bo_get_stride(s->bo);

        /* 2. let GL render INTO it: dmabuf -> EGLImage -> rbo -> FBO */
        const EGLAttrib attrs[] = {
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
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (EGLAttrib)(DRM_FORMAT_MOD_LINEAR & 0xffffffff),
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (EGLAttrib)(DRM_FORMAT_MOD_LINEAR >> 32),
            EGL_NONE,
        };
        s->image =
            eglCreateImage(e->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
        if (s->image == EGL_NO_IMAGE) {
            fprintf(stderr, "eglCreateImage failed\n");
            exit(1);
        }
        glGenRenderbuffers(1, &s->rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, s->rbo);
        e->image_target_rbo(GL_RENDERBUFFER, s->image);
        glGenFramebuffers(1, &s->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, s->rbo);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "FBO incomplete\n");
            exit(1);
        }

        /* 3. let the COMPOSITOR read it: dmabuf -> wl_buffer, over the
         *    protocol Mesa used to speak behind our back. */
        struct zwp_linux_buffer_params_v1* params = zwp_linux_dmabuf_v1_create_params(e->dmabuf);
        zwp_linux_buffer_params_v1_add(params,
                                       fd,
                                       0,
                                       0,
                                       stride,
                                       (uint32_t)(DRM_FORMAT_MOD_LINEAR >> 32),
                                       (uint32_t)(DRM_FORMAT_MOD_LINEAR & 0xffffffff));
        s->buffer = zwp_linux_buffer_params_v1_create_immed(params, w, h, DRM_FORMAT_XRGB8888, 0);
        zwp_linux_buffer_params_v1_destroy(params);
        wl_buffer_add_listener(s->buffer, &buffer_listener, s);
        s->busy = 0;

        close(fd); /* image + compositor import keep the memory alive */
    }
    e->buf_w = w;
    e->buf_h = h;
}

/* eglSwapBuffers, unrolled: find free slot, render into its FBO,
 * attach + damage + commit. The busy scan is back from shm.c. */
static void egl_redraw(struct app* a) {
    struct egl_app* e = (struct egl_app*)a;

    struct egl_slot* s = NULL;
    for (int i = 0; i < SLOTS; i++) {
        if (!e->slots[i].busy) {
            s = &e->slots[i];
            break;
        }
    }
    if (!s) { return; } /* both on loan: skip this frame */

    static const GLfloat verts[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

    glBindFramebuffer(GL_FRAMEBUFFER, s->fbo); /* render target = the dmabuf */
    glViewport(0, 0, e->buf_w, e->buf_h);
    glUseProgram(e->program);
    glUniform2f(e->u_size, (GLfloat)e->buf_w, (GLfloat)e->buf_h);
    glUniform2f(e->u_cursor, (GLfloat)a->ptr_x, (GLfloat)a->ptr_y);
    glUniform1f(e->u_time, (GLfloat)app_anim_time(a));
    glEnableVertexAttribArray((GLuint)e->a_pos);
    glVertexAttribPointer((GLuint)e->a_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Submit, don't wait: the kernel attaches the render job's fence
     * to the dmabuf (implicit sync) and the compositor waits on it
     * before reading. glFinish would be the CPU-stalling sledgehammer;
     * linux-drm-syncobj-v1 is the modern explicit alternative. */
    glFlush();

    wl_surface_attach(a->surface, s->buffer, 0, 0);
    wl_surface_damage_buffer(a->surface, 0, 0, INT32_MAX, INT32_MAX);
    s->busy = 1;
    wl_surface_commit(a->surface);
}

static const struct app_backend egl_backend = {
    .configure = egl_configure,
    .redraw = egl_redraw,
};

/* ------------------------------------------------------------------ */
/* Binding zwp_linux_dmabuf_v1 ourselves -- on a PRIVATE queue, the   */
/* Mesa pattern: the roundtrip must not dispatch the default queue    */
/* (that could fire configure before the GL context exists).          */
/* ------------------------------------------------------------------ */

static void on_dmabuf_global(void* data,
                             struct wl_registry* registry,
                             uint32_t name,
                             const char* interface,
                             uint32_t version) {
    struct egl_app* e = data;
    if (strcmp(interface, zwp_linux_dmabuf_v1_interface.name) == 0) {
        e->dmabuf = wl_registry_bind(registry,
                                     name,
                                     &zwp_linux_dmabuf_v1_interface,
                                     version < 3 ? version : 3);
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

static void bind_dmabuf(struct egl_app* e) {
    struct wl_display* display = e->app.display;

    struct wl_event_queue* queue = wl_display_create_queue(display);
    struct wl_display* wrapped = wl_proxy_create_wrapper(display);
    wl_proxy_set_queue((struct wl_proxy*)wrapped, queue);

    /* a second registry: anyone may ask for the menu again */
    struct wl_registry* registry = wl_display_get_registry(wrapped);
    wl_registry_add_listener(registry, &dmabuf_registry_listener, e);
    wl_display_roundtrip_queue(display, queue); /* pumps ONLY our queue */

    wl_registry_destroy(registry);
    wl_proxy_wrapper_destroy(wrapped);
    if (!e->dmabuf) {
        fprintf(stderr, "compositor lacks zwp_linux_dmabuf_v1\n");
        exit(1);
    }
    /* move the bound proxy to the default queue: from now on its
     * descendants (params, wl_buffers) dispatch in the main pump */
    wl_proxy_set_queue((struct wl_proxy*)e->dmabuf, NULL);
    wl_event_queue_destroy(queue);
}

int main(void) {
    struct egl_app e = {0};

    if (app_init(&e.app, &egl_backend, "hello wayland (GLES/dmabuf)", "hello-wayland-gles-dmabuf") <
        0) {
        return 1;
    }

    bind_dmabuf(&e);

    /* Our own GPU access: the render node (unprivileged -- no
     * modesetting rights, those stay with the compositor on card0). */
    e.drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (e.drm_fd < 0) {
        perror("open render node");
        return 1;
    }
    e.gbm = gbm_create_device(e.drm_fd);

    /* EGL as pure context factory: GBM platform, no window surface
     * ever. This is the compositor's own path (wlroots does this). */
    e.egl_display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, e.gbm, NULL);
    if (e.egl_display == EGL_NO_DISPLAY || !eglInitialize(e.egl_display, NULL, NULL)) {
        fprintf(stderr, "EGL init failed\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    const char* exts = eglQueryString(e.egl_display, EGL_EXTENSIONS);
    if (!strstr(exts, "EGL_KHR_surfaceless_context")) {
        fprintf(stderr, "need EGL_KHR_surfaceless_context\n");
        return 1;
    }

    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,
        0, /* we create NO EGL surfaces at all */
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
    if (!eglChooseConfig(e.egl_display, config_attribs, &e.egl_config, 1, &n) || n < 1) {
        fprintf(stderr, "no EGL config\n");
        return 1;
    }

    static const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    e.egl_context = eglCreateContext(e.egl_display, e.egl_config, EGL_NO_CONTEXT, ctx_attribs);
    if (e.egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        return 1;
    }

    /* current WITHOUT a surface -- rendering goes to our FBOs only */
    eglMakeCurrent(e.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, e.egl_context);

    e.image_target_rbo = (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress(
        "glEGLImageTargetRenderbufferStorageOES");
    if (!e.image_target_rbo) {
        fprintf(stderr, "need GL_OES_EGL_image\n");
        return 1;
    }

    /* context is current with no surface needed -> shaders compile now,
     * before any configure can fire */
    init_gl(&e);

    app_run(&e.app);

    for (int i = 0; i < SLOTS; i++) {
        destroy_slot_gl(&e, &e.slots[i]);
        if (e.slots[i].buffer) { wl_buffer_destroy(e.slots[i].buffer); }
    }
    zwp_linux_dmabuf_v1_destroy(e.dmabuf);
    eglMakeCurrent(e.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(e.egl_display, e.egl_context);
    eglTerminate(e.egl_display);
    gbm_device_destroy(e.gbm);
    close(e.drm_fd);

    app_finish(&e.app);
    return 0;
}
