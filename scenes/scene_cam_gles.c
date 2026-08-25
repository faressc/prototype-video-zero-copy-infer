/* scene_cam_gles.c -- the camera through GLES, zero-copy.
 *
 * The §15 gateway in the other direction: eglCreateImage imports the
 * camera's dmabuf (NV12, two planes in one fd) as an EGLImage, and
 * glEGLImageTargetTexture2DOES turns it into a GL_TEXTURE_EXTERNAL_OES
 * texture -- "external" because the driver, not GL, owns the layout,
 * and because sampling it does the YUV->RGB conversion in the texture
 * unit (the EGL_YUV_* hints say which matrix and range). Pixels never
 * touch the CPU: ISP -> dmabuf -> GPU sampler -> our FBO -> compositor.
 *
 * Every buffer is imported ONCE at startup (six textures); per frame
 * only the binding changes. After a draw that read buffer N, an EGL
 * fence is created; N goes back to V4L2 only when it has signaled --
 * V4L2 knows nothing about dmabuf fences.
 *
 * The effect chain (effect_chain.h) is a sequence of fullscreen passes
 * ping-ponging between two scratch textures (each an FBO to render
 * into and a texture to sample from); the last pass goes to the window
 * -- or, with C, onto the cube. Texture units are fixed: 0 camera,
 * 1 scratch A, 2 scratch B; u_src picks which one a pass reads.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <drm_fourcc.h>

#include "cam_effects_glsl.h" /* generated: the shared effect chain as a C string */
#include "cam_stream.h"
#include "cube_mesh.h"
#include "effect_chain.h"
#include "gles_present.h"
#include "mat4.h"

#ifndef CAM_GLES_MODE
#define CAM_GLES_MODE GLES_MODE_DMABUF
#endif

/* the uniforms every pass program has (fullscreen and cube alike) */
struct pass_uniforms {
    GLint u_cam, u_tmp0, u_tmp1, u_texel, u_time, u_effect, u_src;
};

struct cam_scene {
    struct cam_stream stream;
    uint32_t cw, ch;

    EGLImage images[CAM_MAX_BUFFERS];
    GLuint textures[CAM_MAX_BUFFERS]; /* GL_TEXTURE_EXTERNAL_OES, one per buffer */
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_tex;

    GLuint program; /* the fullscreen pass */
    GLint a_pos, u_flip;
    struct pass_uniforms pu;

    GLuint tmp_tex[2], tmp_fbo[2]; /* the chain's scratch, camera-sized */

    GLuint cube_program, vbo, ibo; /* the cube as the last pass */
    GLint c_pos, c_normal, c_uv, c_mvp, c_model, c_uv_scale;
    struct pass_uniforms cu;
};

/* ------------------------------------------------------------------ */
/* fences: EGL sync objects, polled with zero timeout                 */
/* ------------------------------------------------------------------ */

static int egl_fence_done(void* fence, void* user) {
    struct gles_presenter* p = user;
    return eglClientWaitSync(p->egl_display, (EGLSync)fence, 0, 0) == EGL_CONDITION_SATISFIED;
}

static void egl_fence_destroy(void* fence, void* user) {
    struct gles_presenter* p = user;
    eglDestroySync(p->egl_display, (EGLSync)fence);
}

static const struct cam_fence_ops egl_fence_ops = {.done = egl_fence_done,
                                                   .destroy = egl_fence_destroy};

/* ------------------------------------------------------------------ */
/* shaders                                                            */
/* ------------------------------------------------------------------ */

static const char* vert_src =
    "attribute vec2 a_pos;\n"
    "uniform float u_flip;\n" /* 1 = window surface (y up): flip v */
    "varying vec2 v_uv;\n"
    "void main() {\n"
    "    vec2 uv = a_pos * 0.5 + 0.5;\n"
    "    v_uv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, u_flip));\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

/* shared by both fragment shaders: the three inputs and the macros the
 * effect body expects; u_src selects the stage this pass reads */
static const char* frag_head =
    "#extension GL_OES_EGL_image_external : require\n"
    "precision highp float;\n"
    "varying vec2 v_uv;\n"
    "uniform samplerExternalOES u_cam;\n" /* the camera: YUV->RGB in the sampler */
    "uniform sampler2D u_tmp0;\n"         /* scratch A */
    "uniform sampler2D u_tmp1;\n"         /* scratch B */
    "uniform vec2 u_texel;\n"
    "uniform float u_time;\n"
    "uniform int u_effect;\n"
    "uniform int u_src;\n"
    "vec3 src_sample(vec2 uv) {\n"
    "    if (u_src == 1) { return texture2D(u_tmp0, uv).rgb; }\n"
    "    if (u_src == 2) { return texture2D(u_tmp1, uv).rgb; }\n"
    "    return texture2D(u_cam, uv).rgb;\n"
    "}\n"
    "#define SRC(uv) src_sample(uv)\n"
    "#define U_TEXEL u_texel\n"
    "#define U_EFFECT u_effect\n"
    "#define U_TIME u_time\n";

static const char* frag_tail = "void main() { gl_FragColor = vec4(apply_effect(v_uv), 1.0); }\n";

static const char* cube_vert_src =
    "attribute vec3 a_pos;\n"
    "attribute vec3 a_normal;\n"
    "attribute vec2 a_uv;\n"
    "uniform mat4 u_mvp;\n"
    "uniform mat4 u_model;\n"
    "uniform vec2 u_uv_scale;\n" /* aspect: the face is square, the camera is not */
    "varying vec3 v_normal;\n"
    "varying vec2 v_uv;\n"
    "void main() {\n"
    "    v_normal = mat3(u_model) * a_normal;\n"
    "    v_uv = 0.5 + (a_uv - 0.5) * u_uv_scale;\n"
    "    gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
    "}\n";

/* the cube's fragment shader = the same pass body with a lit main. v_uv
 * arrives already aspect-corrected, so the effects (and their texel
 * offsets) run in camera-texture space as in the fullscreen pass. */
static const char* cube_frag_tail =
    "varying vec3 v_normal;\n"
    "void main() {\n"
    "    vec3 l = normalize(vec3(0.4, 0.8, 1.0));\n"
    "    float d = max(dot(normalize(v_normal), l), 0.0);\n"
    "    gl_FragColor = vec4(apply_effect(v_uv) * (0.35 + 0.65 * d), 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return sh;
}

static GLuint link_program(const char* vs_src, const char* frag_tail_src) {
    size_t n = strlen(frag_head) + strlen(cam_effects_glsl) + strlen(frag_tail_src) + 1;
    char* frag = malloc(n);
    snprintf(frag, n, "%s%s%s", frag_head, cam_effects_glsl, frag_tail_src);
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag);
    free(frag);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), NULL, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static void lookup_pass_uniforms(GLuint prog, struct pass_uniforms* u) {
    u->u_cam = glGetUniformLocation(prog, "u_cam");
    u->u_tmp0 = glGetUniformLocation(prog, "u_tmp0");
    u->u_tmp1 = glGetUniformLocation(prog, "u_tmp1");
    u->u_texel = glGetUniformLocation(prog, "u_texel");
    u->u_time = glGetUniformLocation(prog, "u_time");
    u->u_effect = glGetUniformLocation(prog, "u_effect");
    u->u_src = glGetUniformLocation(prog, "u_src");
    /* samplers name texture UNITS, set once: 0 camera, 1 A, 2 B */
    glUseProgram(prog);
    glUniform1i(u->u_cam, 0);
    glUniform1i(u->u_tmp0, 1);
    glUniform1i(u->u_tmp1, 2);
}

static void set_pass_uniforms(const struct cam_scene* s,
                              const struct pass_uniforms* u,
                              int effect,
                              int src,
                              float time) {
    glUniform2f(u->u_texel, 1.0f / (float)s->cw, 1.0f / (float)s->ch);
    glUniform1f(u->u_time, time);
    glUniform1i(u->u_effect, effect);
    glUniform1i(u->u_src, src);
}

static void build_programs(struct cam_scene* s) {
    s->program = link_program(vert_src, frag_tail);
    s->a_pos = glGetAttribLocation(s->program, "a_pos");
    s->u_flip = glGetUniformLocation(s->program, "u_flip");
    lookup_pass_uniforms(s->program, &s->pu);

    s->cube_program = link_program(cube_vert_src, cube_frag_tail);
    s->c_pos = glGetAttribLocation(s->cube_program, "a_pos");
    s->c_normal = glGetAttribLocation(s->cube_program, "a_normal");
    s->c_uv = glGetAttribLocation(s->cube_program, "a_uv");
    s->c_mvp = glGetUniformLocation(s->cube_program, "u_mvp");
    s->c_model = glGetUniformLocation(s->cube_program, "u_model");
    s->c_uv_scale = glGetUniformLocation(s->cube_program, "u_uv_scale");
    lookup_pass_uniforms(s->cube_program, &s->cu);

    /* Center-crop the camera image to the square face ("cover"): use the
     * full extent of the short axis, a centered window of the long one.
     * Set once; the camera size is fixed. For letterboxing ("contain",
     * bars on the face) swap the two branches. */
    const float aspect = (float)s->cw / (float)s->ch;
    glUseProgram(s->cube_program);
    if (aspect > 1.0f) {
        glUniform2f(s->c_uv_scale, 1.0f / aspect, 1.0f);
    } else {
        glUniform2f(s->c_uv_scale, 1.0f, aspect);
    }

    glGenBuffers(1, &s->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, s->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
    glGenBuffers(1, &s->ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

/* ------------------------------------------------------------------ */
/* the import                                                         */
/* ------------------------------------------------------------------ */

static void import_camera_buffers(struct gles_presenter* p, struct cam_scene* s) {
    const struct camera* cam = &s->stream.cam;
    if (cam->format.pixelformat != V4L2_PIX_FMT_NV12) {
        fprintf(stderr, "cam_gles: only NV12 import is implemented\n");
        exit(1);
    }
    const char* gl_exts = (const char*)glGetString(GL_EXTENSIONS);
    if (!gl_exts || !strstr(gl_exts, "GL_OES_EGL_image_external")) {
        fprintf(stderr, "cam_gles: need GL_OES_EGL_image_external\n");
        exit(1);
    }
    s->image_target_tex =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!s->image_target_tex) {
        fprintf(stderr, "cam_gles: need glEGLImageTargetTexture2DOES\n");
        exit(1);
    }

    for (uint32_t i = 0; i < cam->buf_count; i++) {
        int fd = cam->bufs[i].dmabuf_fd;
        EGLAttrib stride = (EGLAttrib)camera_stride(cam);
        /* two planes, same fd, different offsets -- and the colorimetry
         * hints that tell the sampler which YUV->RGB matrix to apply */
        EGLAttrib attrs[] = {
            EGL_WIDTH,
            (EGLAttrib)cam->format.width,
            EGL_HEIGHT,
            (EGLAttrib)cam->format.height,
            EGL_LINUX_DRM_FOURCC_EXT,
            DRM_FORMAT_NV12,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            stride,
            EGL_DMA_BUF_PLANE1_FD_EXT,
            fd,
            EGL_DMA_BUF_PLANE1_OFFSET_EXT,
            (EGLAttrib)camera_uv_offset(cam),
            EGL_DMA_BUF_PLANE1_PITCH_EXT,
            stride,
            EGL_YUV_COLOR_SPACE_HINT_EXT,
            camera_is_bt709(cam) ? EGL_ITU_REC709_EXT : EGL_ITU_REC601_EXT,
            EGL_SAMPLE_RANGE_HINT_EXT,
            camera_is_full_range(cam) ? EGL_YUV_FULL_RANGE_EXT : EGL_YUV_NARROW_RANGE_EXT,
            EGL_NONE,
        };
        s->images[i] =
            eglCreateImage(p->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
        if (s->images[i] == EGL_NO_IMAGE) {
            fprintf(stderr,
                    "cam_gles: eglCreateImage(NV12) failed, EGL error 0x%x\n",
                    eglGetError());
            exit(1);
        }
        glGenTextures(1, &s->textures[i]);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, s->textures[i]);
        s->image_target_tex(GL_TEXTURE_EXTERNAL_OES, s->images[i]);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
    fprintf(stderr, "cam_gles: %u camera buffers imported as external textures\n", cam->buf_count);
}

/* Two ordinary RGBA textures we can both render into (FBO) and sample
 * from: the intermediates of the chain. */
static void create_scratch(struct cam_scene* s) {
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &s->tmp_tex[i]);
        glBindTexture(GL_TEXTURE_2D, s->tmp_tex[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     (GLsizei)s->cw,
                     (GLsizei)s->ch,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1, &s->tmp_fbo[i]);
        glBindFramebuffer(GL_FRAMEBUFFER, s->tmp_fbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               s->tmp_tex[i],
                               0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "cam_gles: scratch FBO incomplete\n");
            exit(1);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/* ------------------------------------------------------------------ */

static void on_cam_fd(struct app* a) {
    struct gles_presenter* p = (struct gles_presenter*)a;
    struct cam_scene* s = p->user;
    cam_stream_pump(&s->stream);
}

static void cam_init(struct gles_presenter* p, void* user) {
    struct cam_scene* s = user;
    if (cam_stream_open(&s->stream, &egl_fence_ops, p) < 0) { exit(1); }
    s->cw = s->stream.cam.format.width;
    s->ch = s->stream.cam.format.height;

    build_programs(s);
    import_camera_buffers(p, s);
    create_scratch(s);

    p->app.aux_fd = cam_stream_fd(&s->stream);
    p->app.on_aux_fd = on_cam_fd;
    p->app.effect_count = FX_COUNT;
}

/* One effect pass over the viewport rectangle: program, uniforms, one
 * oversized triangle. `flip` = 1 on a y-up window surface, 0 on an FBO. */
static void draw_fullscreen(struct cam_scene* s,
                            int x,
                            int y,
                            int w,
                            int h,
                            float flip,
                            int effect,
                            int src,
                            float time) {
    static const GLfloat verts[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    glViewport(x, y, w, h);
    glUseProgram(s->program);
    glUniform1f(s->u_flip, flip);
    set_pass_uniforms(s, &s->pu, effect, src, time);
    glEnableVertexAttribArray((GLuint)s->a_pos);
    glVertexAttribPointer((GLuint)s->a_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void draw_cube(struct gles_presenter* p,
                      struct cam_scene* s,
                      const struct gles_target* t,
                      int effect,
                      int src) {
    glViewport(0, 0, t->width, t->height);
    glClearDepthf(1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(p->y_down ? GL_CW : GL_CCW); /* see scene_cube_gles.c */

    mat4 model = cube_model_matrix(&p->app);
    mat4 mvp = mat4_mul(cube_view_proj(t->width, t->height, p->y_down, 0, p->app.zoom), model);

    glUseProgram(s->cube_program);
    glUniformMatrix4fv(s->c_mvp, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(s->c_model, 1, GL_FALSE, model.m);
    set_pass_uniforms(s, &s->cu, effect, src, (float)(p->app.anim_ms * 0.001));

    glBindBuffer(GL_ARRAY_BUFFER, s->vbo);
    const GLsizei stride = sizeof(struct cube_vertex);
    glEnableVertexAttribArray((GLuint)s->c_pos);
    glVertexAttribPointer((GLuint)s->c_pos,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, pos));
    glEnableVertexAttribArray((GLuint)s->c_normal);
    glVertexAttribPointer((GLuint)s->c_normal,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, normal));
    glEnableVertexAttribArray((GLuint)s->c_uv);
    glVertexAttribPointer((GLuint)s->c_uv,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, uv));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ibo);
    glDrawElements(GL_TRIANGLES, CUBE_INDEX_COUNT, GL_UNSIGNED_SHORT, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
}

static void cam_draw(struct gles_presenter* p, const struct gles_target* t, void* user) {
    struct cam_scene* s = user;
    cam_stream_reap(&s->stream);

    glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
    glViewport(0, 0, t->width, t->height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    int latest = cam_stream_latest(&s->stream);
    if (latest < 0) { return; }

    /* letterbox rectangle, in GL window coordinates */
    float sx = (float)t->width / (float)s->cw, sy = (float)t->height / (float)s->ch;
    float sc = sx < sy ? sx : sy;
    int dw = (int)((float)s->cw * sc), dh = (int)((float)s->ch * sc);
    int x0 = (t->width - dw) / 2, y0 = (t->height - dh) / 2;

    /* the three inputs, on their fixed units */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, s->textures[latest]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, s->tmp_tex[0]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, s->tmp_tex[1]);

    /* the chain: pass i reads stage `src` (0 = camera, 1/2 = scratch
     * A/B) and writes scratch i&1 -- except the last, which goes to
     * the window or the cube. A pass never reads what it writes. */
    int passes[EFFECT_MAX_PASSES];
    int n = effect_chain_passes(p->app.effect, passes);
    const float time = (float)(p->app.anim_ms * 0.001);
    int src = 0;
    for (int i = 0; i < n; i++) {
        const int last = i == n - 1;
        if (!last) {
            glBindFramebuffer(GL_FRAMEBUFFER, s->tmp_fbo[i & 1]);
            draw_fullscreen(s, 0, 0, (int)s->cw, (int)s->ch, 0.0f, passes[i], src, time);
            src = 1 + (i & 1);
        } else if (p->app.mode) {
            glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
            draw_cube(p, s, t, passes[i], src);
        } else {
            glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
            draw_fullscreen(s, x0, y0, dw, dh, p->y_down ? 0.0f : 1.0f, passes[i], src, time);
        }
    }

    /* the fence: signals when everything queued so far -- including
     * the reads of textures[latest] -- has executed on the GPU. The
     * presenter's glFlush after us submits it. */
    EGLSync fence = eglCreateSync(p->egl_display, EGL_SYNC_FENCE, NULL);
    cam_stream_rendered(&s->stream, latest, fence);
}

static void cam_fini(struct gles_presenter* p, void* user) {
    struct cam_scene* s = user;
    cam_stream_close(&s->stream);
    for (uint32_t i = 0; i < CAM_MAX_BUFFERS; i++) {
        if (s->textures[i]) { glDeleteTextures(1, &s->textures[i]); }
        if (s->images[i]) { eglDestroyImage(p->egl_display, s->images[i]); }
    }
    glDeleteFramebuffers(2, s->tmp_fbo);
    glDeleteTextures(2, s->tmp_tex);
    glDeleteProgram(s->program);
    glDeleteBuffers(1, &s->vbo);
    glDeleteBuffers(1, &s->ibo);
    glDeleteProgram(s->cube_program);
}

static const struct gles_scene cam_scene = {
    .init = cam_init,
    .resize = NULL,
    .draw = cam_draw,
    .fini = cam_fini,
};

int main(void) {
    struct cam_scene s = {0};
    struct gles_presenter p;
    const int dmabuf = CAM_GLES_MODE == GLES_MODE_DMABUF;
    if (gles_present_init(
            &p,
            CAM_GLES_MODE,
            GLES_PRESENT_DEPTH,
            &cam_scene,
            &s,
            dmabuf ? "camera (GLES/dmabuf)" : "camera (GLES/EGLSurface)",
            dmabuf ? "hello-wayland-cam-gles-dmabuf" : "hello-wayland-cam-gles-eglsurface") < 0) {
        return 1;
    }
    gles_present_run(&p);
    gles_present_fini(&p);
    return 0;
}
