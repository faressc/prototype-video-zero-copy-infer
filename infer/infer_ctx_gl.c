/* infer_ctx_gl.c -- the OpenGL ES domain, headless: EGL on the GBM
 * render node with a surfaceless ES 3.1 context (the context factory
 * from present/gles_present.c, minus the window), a fragment shader
 * that fills the tensor, and the export path.
 *
 * GL cannot export its own buffer objects, so an exportable GL tensor
 * is a gbm_bo: linear dma-buf memory allocated by us, imported into GL
 * as an EGLImage-backed renderbuffer and rendered into (a fullscreen
 * triangle whose fragments are the tensor's bytes). The bo's fd is
 * what Vulkan or Dawn import; its stride is the byte image's row pitch.
 * Completion travels as a sync file via EGL_ANDROID_native_fence_sync
 * (falling back to glFinish when the driver lacks it).
 */
#include "infer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <drm_fourcc.h>

#include "infer_util.h"
#include "infer_hash_glsl.h"
#include "infer_gen_es_glsl.h"

static int has_ext(const char* list, const char* name) {
    return list && strstr(list, name) != NULL;
}

/* ------------------------------------------------------------------ */
/* bring-up                                                           */
/* ------------------------------------------------------------------ */

static int load_procs(struct infer_ctx_gl* c) {
    const char* exts = eglQueryString(c->dpy, EGL_EXTENSIONS);
    c->create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    c->destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    c->image_target_rbo = (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress(
        "glEGLImageTargetRenderbufferStorageOES");
    c->has_dmabuf_import = has_ext(exts, "EGL_EXT_image_dma_buf_import") && c->create_image &&
                           c->destroy_image && c->image_target_rbo &&
                           has_ext((const char*)glGetString(GL_EXTENSIONS), "GL_OES_EGL_image");
    c->has_modifiers = has_ext(exts, "EGL_EXT_image_dma_buf_import_modifiers");
    c->create_sync = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
    c->destroy_sync = (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
    c->wait_sync = (PFNEGLWAITSYNCKHRPROC)eglGetProcAddress("eglWaitSyncKHR");
    c->dup_fence_fd =
        (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress("eglDupNativeFenceFDANDROID");
    c->has_native_fence = has_ext(exts, "EGL_ANDROID_native_fence_sync") && c->create_sync &&
                          c->destroy_sync && c->wait_sync && c->dup_fence_fd;
    fprintf(stderr,
            "gl: %s; dma-buf import %s, modifiers %s, native fence %s\n",
            (const char*)glGetString(GL_VERSION),
            c->has_dmabuf_import ? "yes" : "NO",
            c->has_modifiers ? "yes" : "no",
            c->has_native_fence ? "yes" : "NO");
    return 0;
}

static GLuint compile(GLenum type, const char* const* sources, int n) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, n, sources, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "gl: shader:\n%s\n", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static int build_program(struct infer_ctx_gl* c) {
    /* a fullscreen triangle from gl_VertexID; the fragment shader does
     * the work */
    static const char* const vs_src[] = {
        "#version 310 es\n"
        "void main() {\n"
        "    vec2 v = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
        "    gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n"};
    /* ES needs default precisions declared before the shared generator
     * (which is plain GLSL) can mention `float` */
    const char* const fs_src[] = {"#version 310 es\nprecision highp float;\nprecision highp int;\n",
                                  infer_hash_glsl,
                                  infer_gen_es_glsl};
    GLuint vs = compile(GL_VERTEX_SHADER, vs_src, 1);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src, 3);
    INFER_CHECK(vs && fs, "gl: generator shaders failed");
    c->gen_program = glCreateProgram();
    glAttachShader(c->gen_program, vs);
    glAttachShader(c->gen_program, fs);
    glLinkProgram(c->gen_program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(c->gen_program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(c->gen_program, sizeof log, NULL, log);
        fprintf(stderr, "gl: link:\n%s\n", log);
        return -1;
    }
    return 0;
}

int infer_ctx_gl_init(struct infer_ctx_gl* c, const char* render_node) {
    memset(c, 0, sizeof *c);
    c->owned = 1;
    c->drm_fd = open(render_node ? render_node : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    INFER_CHECK(c->drm_fd >= 0, "gl: cannot open render node");
    c->gbm = gbm_create_device(c->drm_fd);
    INFER_CHECK(c->gbm, "gl: gbm_create_device failed");

    c->dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, c->gbm, NULL);
    INFER_CHECK(c->dpy != EGL_NO_DISPLAY && eglInitialize(c->dpy, NULL, NULL), "gl: EGL init failed");
    eglBindAPI(EGL_OPENGL_ES_API);
    INFER_CHECK(has_ext(eglQueryString(c->dpy, EGL_EXTENSIONS), "EGL_KHR_surfaceless_context"),
                "gl: need EGL_KHR_surfaceless_context");

    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, 0, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE,
    };
    EGLint n = 0;
    INFER_CHECK(eglChooseConfig(c->dpy, config_attribs, &c->cfg, 1, &n) && n >= 1, "gl: no EGL config");
    static const EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE,
    };
    c->ctx = eglCreateContext(c->dpy, c->cfg, EGL_NO_CONTEXT, ctx_attribs);
    INFER_CHECK(c->ctx != EGL_NO_CONTEXT, "gl: eglCreateContext (ES 3.1) failed");
    INFER_CHECK(eglMakeCurrent(c->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, c->ctx), "gl: make current");

    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    INFER_CHECK(major >= 3, "gl: need OpenGL ES 3.x");
    if (load_procs(c) < 0) { return -1; }
    return build_program(c);
}

int infer_ctx_gl_borrow(struct infer_ctx_gl* c, EGLDisplay dpy, EGLContext ctx, struct gbm_device* gbm) {
    memset(c, 0, sizeof *c);
    c->dpy = dpy;
    c->ctx = ctx;
    c->gbm = gbm;
    c->drm_fd = -1;
    INFER_CHECK(eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx), "gl: make current");
    if (load_procs(c) < 0) { return -1; }
    return build_program(c);
}

void infer_ctx_gl_fini(struct infer_ctx_gl* c) {
    if (c->gen_program) { glDeleteProgram(c->gen_program); }
    if (c->owned) {
        eglMakeCurrent(c->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (c->ctx != EGL_NO_CONTEXT) { eglDestroyContext(c->dpy, c->ctx); }
        if (c->dpy != EGL_NO_DISPLAY) { eglTerminate(c->dpy); }
        if (c->gbm) { gbm_device_destroy(c->gbm); }
        if (c->drm_fd >= 0) { close(c->drm_fd); }
    }
    memset(c, 0, sizeof *c);
}

/* ------------------------------------------------------------------ */
/* the tensor: a gbm_bo seen by GL as an RGBA8 renderbuffer            */
/* ------------------------------------------------------------------ */

/* A GL tensor has to be renderable by this driver AND importable by
 * whoever consumes it, and left to itself GBM picks a layout that is
 * only the former. On NVIDIA it hands back the block-linear
 * 0x300000000e08014, which EGL will happily render into but Vulkan --
 * and therefore Dawn -- does not list at all, so every dma-buf import
 * of a GL tensor fails. The two sets do overlap (the 0x3000000006060xx
 * family here); the fix is to allocate from the intersection instead of
 * from the driver's preference.
 *
 * EGL is asked what it can render into, Vulkan what it can import, and
 * the overlap is handed to gbm_bo_create_with_modifiers so the driver
 * still picks its favourite -- just from a set every consumer accepts.
 * If either query is unavailable, or nothing overlaps, fall back to
 * letting the driver choose: that is the old behaviour, and it is what
 * Apple GPUs want anyway (there the driver's tiled choice is the only
 * layout that renders correctly -- LINEAR makes Mesa shadow the shader
 * writes into a tiled copy it never flushes to the dma-buf). */
static void negotiate_modifier(struct infer_ctx_gl* c) {
    c->bo_negotiated = 1;
    c->bo_explicit = 0;

    EGLuint64KHR egl_mods[64];
    EGLBoolean external[64];
    EGLint egl_n = 0;
    PFNEGLQUERYDMABUFMODIFIERSEXTPROC query =
        (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
    if (!c->has_modifiers || !query ||
        !query(c->dpy, DRM_FORMAT_ABGR8888, 64, egl_mods, external, &egl_n) || egl_n <= 0) {
        fprintf(stderr, "gl: no modifier query; letting the driver choose the layout\n");
        return;
    }
    uint64_t vk_mods[64];
    int vk_n = infer_vk_importable_modifiers(vk_mods, 64);
    if (vk_n <= 0) {
        fprintf(stderr, "gl: no importer modifier list; letting the driver choose the layout\n");
        return;
    }

    uint64_t shared[64];
    int shared_n = 0;
    for (EGLint i = 0; i < egl_n && shared_n < 64; i++) {
        if (external[i]) { continue; } /* sampleable as external only, not renderable */
        for (int j = 0; j < vk_n; j++) {
            if ((uint64_t)egl_mods[i] == vk_mods[j]) {
                shared[shared_n++] = vk_mods[j];
                break;
            }
        }
    }
    if (!shared_n) {
        fprintf(stderr,
                "gl: no layout is both renderable and importable (%d renderable, %d importable);"
                " letting the driver choose\n",
                (int)egl_n,
                vk_n);
        return;
    }

    /* Prefer the finest granularity, not the driver's favourite. Handed
     * the whole list, NVIDIA's GBM picks a 64-row block height
     * (0x...606013) and GL and Vulkan then disagree about any image
     * shorter than one block: GL renders with 64-row blocks, Vulkan's
     * explicit-layout import walks the same bytes as if the blocks were
     * clamped to the image, and from the second GOB on the importer
     * reads GL's padding (a 96x32 tensor: element 16 onward is zero).
     * NVIDIA's block-linear modifiers keep log2(block height in GOBs)
     * in the low nibble, so the one-GOB layout (0x...606010, 8-row
     * blocks) sorts first; other vendors' modifiers stay in list order.
     * Try them one at a time so the choice is ours, not GBM's. */
    for (int i = 1; i < shared_n; i++) {
        uint64_t m = shared[i];
        int j = i;
        while (j > 0 && (shared[j - 1] >> 56) == 0x03 && (m >> 56) == 0x03 &&
               (shared[j - 1] & 0xf) > (m & 0xf)) {
            shared[j] = shared[j - 1];
            j--;
        }
        shared[j] = m;
    }
    for (int i = 0; i < shared_n; i++) {
        struct gbm_bo* bo = gbm_bo_create_with_modifiers(c->gbm, 64, 64, GBM_FORMAT_ABGR8888, &shared[i], 1);
        if (!bo) { continue; }
        c->bo_modifier = gbm_bo_get_modifier(bo);
        c->bo_explicit = 1;
        gbm_bo_destroy(bo);
        fprintf(stderr,
                "gl: layout 0x%llx (%d renderable, %d importable, %d shared; finest block first)\n",
                (unsigned long long)c->bo_modifier,
                (int)egl_n,
                vk_n,
                shared_n);
        return;
    }
    fprintf(stderr, "gl: none of the shared layouts can be allocated; letting the driver choose\n");
}

static int alloc_tensor(struct infer_ctx_gl* c, const struct infer_desc* d, struct infer_tensor* t) {
    t->domain = INFER_DOMAIN_GL;
    t->desc = *d;
    t->owned = 1;
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
    INFER_CHECK(c->has_dmabuf_import, "gl: dma-buf import unavailable");

    /* the layout every consumer can live with; see negotiate_modifier.
     * The modifier travels with the fd, so importers see the real layout. */
    if (!c->bo_negotiated) { negotiate_modifier(c); }
    t->mem.gl.bo =
        c->bo_explicit
            ? gbm_bo_create_with_modifiers(c->gbm, d->img_w, d->img_h, GBM_FORMAT_ABGR8888,
                                           &c->bo_modifier, 1)
            : gbm_bo_create(c->gbm, d->img_w, d->img_h, GBM_FORMAT_ABGR8888, GBM_BO_USE_RENDERING);
    INFER_CHECK(t->mem.gl.bo, "gl: gbm_bo_create failed");
    uint64_t mod = gbm_bo_get_modifier(t->mem.gl.bo);
    if (mod == DRM_FORMAT_MOD_INVALID) { mod = DRM_FORMAT_MOD_LINEAR; }
    t->desc.drm_modifier = mod;
    t->desc.planes = (uint32_t)gbm_bo_get_plane_count(t->mem.gl.bo);
    INFER_CHECK(t->desc.planes >= 1 && t->desc.planes <= 4, "gl: odd plane count");
    for (uint32_t p = 0; p < t->desc.planes; p++) {
        t->desc.plane_offset[p] = gbm_bo_get_offset(t->mem.gl.bo, (int)p);
        t->desc.plane_pitch[p] = gbm_bo_get_stride_for_plane(t->mem.gl.bo, (int)p);
    }
    t->desc.row_pitch_bytes = t->desc.plane_pitch[0];
    t->mem.gl.dmabuf_fd = gbm_bo_get_fd(t->mem.gl.bo);
    INFER_CHECK(t->mem.gl.dmabuf_fd >= 0, "gl: gbm_bo_get_fd failed");
    fprintf(stderr,
            "gl: tensor %ux%u, modifier 0x%llx, %u plane(s), pitch %u\n",
            d->img_w,
            d->img_h,
            (unsigned long long)mod,
            t->desc.planes,
            t->desc.row_pitch_bytes);

    static const EGLint plane_fd[4] = {EGL_DMA_BUF_PLANE0_FD_EXT, EGL_DMA_BUF_PLANE1_FD_EXT,
                                       EGL_DMA_BUF_PLANE2_FD_EXT, EGL_DMA_BUF_PLANE3_FD_EXT};
    static const EGLint plane_off[4] = {EGL_DMA_BUF_PLANE0_OFFSET_EXT, EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                                        EGL_DMA_BUF_PLANE2_OFFSET_EXT, EGL_DMA_BUF_PLANE3_OFFSET_EXT};
    static const EGLint plane_pitch[4] = {EGL_DMA_BUF_PLANE0_PITCH_EXT, EGL_DMA_BUF_PLANE1_PITCH_EXT,
                                          EGL_DMA_BUF_PLANE2_PITCH_EXT, EGL_DMA_BUF_PLANE3_PITCH_EXT};
    static const EGLint plane_mod_lo[4] = {
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
        EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_LO_EXT};
    static const EGLint plane_mod_hi[4] = {
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
        EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_HI_EXT};
    EGLint attrs[48];
    int i = 0;
    attrs[i++] = EGL_WIDTH;
    attrs[i++] = (EGLint)d->img_w;
    attrs[i++] = EGL_HEIGHT;
    attrs[i++] = (EGLint)d->img_h;
    attrs[i++] = EGL_LINUX_DRM_FOURCC_EXT;
    attrs[i++] = DRM_FORMAT_ABGR8888;
    for (uint32_t p = 0; p < t->desc.planes; p++) {
        attrs[i++] = plane_fd[p];
        attrs[i++] = t->mem.gl.dmabuf_fd;
        attrs[i++] = plane_off[p];
        attrs[i++] = (EGLint)t->desc.plane_offset[p];
        attrs[i++] = plane_pitch[p];
        attrs[i++] = (EGLint)t->desc.plane_pitch[p];
        if (c->has_modifiers) {
            attrs[i++] = plane_mod_lo[p];
            attrs[i++] = (EGLint)(mod & 0xffffffffu);
            attrs[i++] = plane_mod_hi[p];
            attrs[i++] = (EGLint)(mod >> 32);
        }
    }
    attrs[i++] = EGL_NONE;
    t->mem.gl.image = c->create_image(c->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    INFER_CHECK(t->mem.gl.image != EGL_NO_IMAGE_KHR, "gl: eglCreateImage(dma-buf) failed");

    glGenRenderbuffers(1, &t->mem.gl.rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, t->mem.gl.rbo);
    c->image_target_rbo(GL_RENDERBUFFER, t->mem.gl.image);
    glGenFramebuffers(1, &t->mem.gl.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t->mem.gl.fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, t->mem.gl.rbo);
    INFER_CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
                "gl: framebuffer over the EGLImage is incomplete");
    INFER_CHECK(glGetError() == GL_NO_ERROR, "gl: EGLImage renderbuffer setup failed");
    return 0;
}

/* the consumer's "done reading" fence, honoured on the GPU timeline */
static void wait_released(struct infer_ctx_gl* c, struct infer_tensor* t) {
    int fd = t->released.sync_fd;
    if (fd < 0) { return; }
    if (c->has_native_fence) {
        EGLint attrs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fd, EGL_NONE};
        EGLSyncKHR s = c->create_sync(c->dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, attrs);
        if (s != EGL_NO_SYNC_KHR) {
            c->wait_sync(c->dpy, s, 0); /* EGL owns fd now */
            c->destroy_sync(c->dpy, s);
            t->released.sync_fd = -1;
            return;
        }
    }
    infer_wait_sync_fd(fd, 1000);
    infer_sync_reset(&t->released);
}

int infer_gl_gen(struct infer_ctx_gl* c,
                 const struct infer_desc* d,
                 uint32_t seed,
                 struct infer_tensor* t) {
    if (!t->mem.gl.bo && alloc_tensor(c, d, t) < 0) { return -1; }
    wait_released(c, t);

    glBindFramebuffer(GL_FRAMEBUFFER, t->mem.gl.fbo);
    glViewport(0, 0, (GLsizei)t->desc.img_w, (GLsizei)t->desc.img_h);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glUseProgram(c->gen_program);
    glUniform4ui(glGetUniformLocation(c->gen_program, "p"), t->desc.img_w, t->desc.img_h, 0, seed);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLenum err = glGetError();
    INFER_CHECK(err == GL_NO_ERROR, "gl: draw failed (0x%x)", err);

    if (c->has_native_fence) {
        EGLSyncKHR s = c->create_sync(c->dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, NULL);
        glFlush(); /* the fence fd exists once the commands are submitted */
        int fd = s != EGL_NO_SYNC_KHR ? c->dup_fence_fd(c->dpy, s) : -1;
        if (s != EGL_NO_SYNC_KHR) { c->destroy_sync(c->dpy, s); }
        infer_sync_set(&t->ready, fd >= 0 ? fd : -1);
    } else {
        glFinish();
        infer_sync_set(&t->ready, -1);
    }
    return 0;
}

/* Read the tensor back through GL rather than by mapping the bo. No
 * layout is both renderable and CPU-mappable on every driver -- NVIDIA
 * renders only into block-linear and maps only linear, so gbm_bo_map
 * fails on exactly the buffers the generator can write. Reading through
 * the FBO asks the driver to do the detiling it alone knows how to do,
 * and it is exact: an RGBA8 renderbuffer read as GL_RGBA/GL_UNSIGNED_BYTE
 * is a byte copy, which is what the bit-exactness check demands. */
int infer_gl_readback(struct infer_ctx_gl* c, const struct infer_tensor* t, float* dst) {
    (void)c;
    INFER_CHECK(t->mem.gl.fbo, "gl: tensor has no framebuffer to read");
    glBindFramebuffer(GL_FRAMEBUFFER, t->mem.gl.fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 4); /* rows are img_w * 4 bytes */
    glReadPixels(0,
                 0,
                 (GLsizei)t->desc.img_w,
                 (GLsizei)t->desc.img_h,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 dst);
    GLenum err = glGetError();
    INFER_CHECK(err == GL_NO_ERROR, "gl: glReadPixels failed (0x%x)", err);
    return 0;
}

void infer_gl_release(struct infer_ctx_gl* c, struct infer_tensor* t) {
    if (!t->owned) { return; }
    if (t->mem.gl.fbo) { glDeleteFramebuffers(1, &t->mem.gl.fbo); }
    if (t->mem.gl.rbo) { glDeleteRenderbuffers(1, &t->mem.gl.rbo); }
    if (t->mem.gl.image) { c->destroy_image(c->dpy, t->mem.gl.image); }
    if (t->mem.gl.dmabuf_fd >= 0) { close(t->mem.gl.dmabuf_fd); }
    if (t->mem.gl.bo) { gbm_bo_destroy(t->mem.gl.bo); }
}
