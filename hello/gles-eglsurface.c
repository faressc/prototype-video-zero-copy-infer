/* gles-eglsurface.c -- GLES through the CLASSIC EGL window-system path.
 *
 * Sibling of gles-dmabuf.c: identical gradient, identical GL code,
 * opposite philosophy. Here EGL keeps BOTH of its halves:
 *
 *   context factory:  display, config, context (same as the sibling)
 *   window system:    wl_egl_window + EGLSurface + eglSwapBuffers
 *
 * Everything gles-dmabuf.c does by hand happens inside Mesa here:
 * gbm_bo allocation, dmabuf export, wl_buffer creation over
 * zwp_linux_dmabuf_v1 (bound by Mesa on a PRIVATE event queue -- watch
 * for "{mesa egl surface queue}" in WAYLAND_DEBUG output; the sibling
 * has none), release tracking, and the attach/damage/commit inside
 * eglSwapBuffers. This is how GTK, Qt and weston-simple-egl render.
 *
 * Diff the two files: the delta IS eglSwapBuffers, unrolled.
 */

#include <stdio.h>
#include <stdlib.h>

#include <EGL/egl.h>
#include <EGL/eglext.h> /* EGL_PLATFORM_WAYLAND_KHR */
#include <GLES2/gl2.h>
#include <wayland-egl.h> /* wl_egl_window: the 4-function mailbox */

#include "common.h"

struct egl_app {
    struct app app; /* MUST be first */

    EGLDisplay egl_display;
    EGLConfig egl_config;
    EGLContext egl_context;
    struct wl_egl_window* egl_window;
    EGLSurface egl_surface; /* Mesa's hidden swapchain lives behind this */

    GLuint program;
    GLint a_pos;
    GLint u_size, u_cursor, u_time;

    int buf_w, buf_h;
};

/* ------------------------------------------------------------------ */
/* Shaders: identical to gles-dmabuf.c EXCEPT the y-flip. A window    */
/* surface presents with GL's usual bottom-left origin and Mesa       */
/* orients the buffer for scanout -- so WE must flip v_px to get      */
/* top-left pixel coordinates. (FBO rendering in the sibling cancels  */
/* the flip instead.)                                                 */
/* ------------------------------------------------------------------ */

static const char* vert_src =
    "attribute vec2 a_pos;\n"
    "uniform vec2 u_size;\n"
    "varying vec2 v_px;\n"
    "void main() {\n"
    "    vec2 uv = a_pos * 0.5 + 0.5;\n"
    "    v_px = vec2(uv.x, 1.0 - uv.y) * u_size; /* GL bottom-left -> our top-left */\n"
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
/* The two hooks. Compare with gles-dmabuf.c: no slots, no busy       */
/* flags, no release listener, no dmabuf protocol -- Mesa keeps all   */
/* of that behind the EGLSurface.                                     */
/* ------------------------------------------------------------------ */

static void egl_configure(struct app* a) {
    struct egl_app* e = (struct egl_app*)a;

    if (!e->egl_window) {
        /* First configure: the GL-facing window handle at the acked
         * size, then Mesa's swapchain behind it. Only now can GL run
         * commands -- this context style needs a current SURFACE
         * (contrast the sibling's surfaceless context, current since
         * main). */
        e->egl_window = wl_egl_window_create(a->surface, a->width, a->height);
        e->egl_surface =
            eglCreatePlatformWindowSurface(e->egl_display, e->egl_config, e->egl_window, NULL);
        if (e->egl_surface == EGL_NO_SURFACE) {
            fprintf(stderr, "eglCreatePlatformWindowSurface failed\n");
            exit(1);
        }
        eglMakeCurrent(e->egl_display, e->egl_surface, e->egl_surface, e->egl_context);
        /* 0 = don't throttle in swap; common's frame callbacks pace us */
        eglSwapInterval(e->egl_display, 0);
        init_gl(e);
    } else if (a->width != e->buf_w || a->height != e->buf_h) {
        /* Two ints into the wl_egl_window mailbox; Mesa reallocates
         * its swapchain lazily on the next swap (its own orphan
         * handling -- the sibling does this dance explicitly). */
        wl_egl_window_resize(e->egl_window, a->width, a->height, 0, 0);
    }
    e->buf_w = a->width;
    e->buf_h = a->height;
}

static void egl_redraw(struct app* a) {
    struct egl_app* e = (struct egl_app*)a;

    /* A triangle 2x the screen: clipped part vanishes, rest covers
     * every pixel -- no diagonal seam like a two-triangle quad. */
    static const GLfloat verts[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

    glViewport(0, 0, e->buf_w, e->buf_h);
    glUseProgram(e->program);
    glUniform2f(e->u_size, (GLfloat)e->buf_w, (GLfloat)e->buf_h);
    glUniform2f(e->u_cursor, (GLfloat)a->ptr_x, (GLfloat)a->ptr_y);
    glUniform1f(e->u_time, (GLfloat)app_anim_time(a));
    glEnableVertexAttribArray((GLuint)e->a_pos);
    glVertexAttribPointer((GLuint)e->a_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* = the sibling's whole redraw tail: pick free swapchain buffer,
     * flush GL, attach, damage, commit, track release. One call. */
    eglSwapBuffers(e->egl_display, e->egl_surface);
}

static const struct app_backend egl_backend = {
    .configure = egl_configure,
    .redraw = egl_redraw,
};

/* ------------------------------------------------------------------ */

int main(void) {
    struct egl_app e = {0};

    if (app_init(&e.app,
                 &egl_backend,
                 "hello wayland (GLES/EGLSurface)",
                 "hello-wayland-gles-eglsurface") < 0) {
        return 1;
    }

    /* EGL bring-up on the WAYLAND platform: eglInitialize makes Mesa
     * a second tenant on our socket (private queue, dmabuf binding,
     * format negotiation -- all invisible to us). Display-level only;
     * no window needed yet. */
    e.egl_display = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, e.app.display, NULL);
    if (e.egl_display == EGL_NO_DISPLAY || !eglInitialize(e.egl_display, NULL, NULL)) {
        fprintf(stderr, "EGL init failed\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,
        EGL_WINDOW_BIT, /* window surfaces this time */
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

    app_run(&e.app);

    eglMakeCurrent(e.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (e.egl_surface != EGL_NO_SURFACE) { eglDestroySurface(e.egl_display, e.egl_surface); }
    if (e.egl_window) { wl_egl_window_destroy(e.egl_window); }
    eglDestroyContext(e.egl_display, e.egl_context);
    eglTerminate(e.egl_display);

    app_finish(&e.app);
    return 0;
}
