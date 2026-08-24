/* scene_cube_gles.c -- a lit, spinning cube on the GLES presenter.
 *
 * One source, two binaries: CUBE_GLES_MODE selects the presenter mode
 * (dmabuf swapchain by hand, or Mesa's behind an EGLSurface). The scene
 * code is identical for both; the presenter absorbs the difference --
 * except for one clip-space fact it hands back as p->y_down, which
 * decides the projection's y sign and therefore the front-face winding.
 *
 * What is new versus the gradient: vertex + index buffers (GPU-side
 * copies of cube_mesh.h, uploaded once), three attributes per vertex,
 * mat4 uniforms, depth test, back-face culling, and a clear.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "cube_mesh.h"
#include "gles_present.h"
#include "mat4.h"

#ifndef CUBE_GLES_MODE
#define CUBE_GLES_MODE GLES_MODE_DMABUF
#endif

struct cube {
    GLuint program;
    GLuint vbo, ibo;
    GLint a_pos, a_normal, a_color;
    GLint u_mvp, u_model;
};

static const char* vert_src =
    "attribute vec3 a_pos;\n"
    "attribute vec3 a_normal;\n"
    "attribute vec3 a_color;\n"
    "uniform mat4 u_mvp;\n"
    "uniform mat4 u_model;\n"
    "varying vec3 v_normal;\n"
    "varying vec3 v_color;\n"
    "void main() {\n"
    "    v_normal = mat3(u_model) * a_normal;\n"
    "    v_color = a_color;\n"
    "    gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
    "}\n";

static const char* frag_src =
    "precision mediump float;\n" /* lighting math stays well inside fp16 */
    "varying vec3 v_normal;\n"
    "varying vec3 v_color;\n"
    "void main() {\n"
    "    vec3 l = normalize(vec3(0.4, 0.8, 1.0));\n"
    "    float d = max(dot(normalize(v_normal), l), 0.0);\n"
    "    gl_FragColor = vec4(v_color * (0.25 + 0.75 * d), 1.0);\n"
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

static void cube_init(struct gles_presenter* p, void* user) {
    (void)p;
    struct cube* c = user;

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    c->program = glCreateProgram();
    glAttachShader(c->program, vs);
    glAttachShader(c->program, fs);
    glLinkProgram(c->program);
    GLint ok = 0;
    glGetProgramiv(c->program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(c->program, sizeof(log), NULL, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    c->a_pos = glGetAttribLocation(c->program, "a_pos");
    c->a_normal = glGetAttribLocation(c->program, "a_normal");
    c->a_color = glGetAttribLocation(c->program, "a_color");
    c->u_mvp = glGetUniformLocation(c->program, "u_mvp");
    c->u_model = glGetUniformLocation(c->program, "u_model");

    /* The mesh moves to GPU memory once. GLES2's client-side arrays
     * (the gradient's verts[] pointer) would re-upload it every draw. */
    glGenBuffers(1, &c->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
    glGenBuffers(1, &c->ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, c->ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
}

static void cube_draw(struct gles_presenter* p, const struct gles_target* t, void* user) {
    struct cube* c = user;
    const struct app* a = &p->app;

    glViewport(0, 0, t->width, t->height);
    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    /* The mesh is CCW from outside. GL judges winding in ITS window
     * coordinates (y up). In dmabuf mode the projection flipped y so
     * the buffer scans out upright -- which makes the mesh CW in GL's
     * eyes. Same picture on screen, opposite setting. */
    glFrontFace(p->y_down ? GL_CW : GL_CCW);

    mat4 model = cube_model_matrix(a);
    mat4 mvp = mat4_mul(cube_view_proj(t->width, t->height, p->y_down, 0, a->zoom), model);

    glUseProgram(c->program);
    glUniformMatrix4fv(c->u_mvp, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(c->u_model, 1, GL_FALSE, model.m);

    glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
    const GLsizei stride = sizeof(struct cube_vertex);
    glEnableVertexAttribArray((GLuint)c->a_pos);
    glVertexAttribPointer((GLuint)c->a_pos,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, pos));
    glEnableVertexAttribArray((GLuint)c->a_normal);
    glVertexAttribPointer((GLuint)c->a_normal,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, normal));
    glEnableVertexAttribArray((GLuint)c->a_color);
    glVertexAttribPointer((GLuint)c->a_color,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (const void*)offsetof(struct cube_vertex, color));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, c->ibo);
    glDrawElements(GL_TRIANGLES, CUBE_INDEX_COUNT, GL_UNSIGNED_SHORT, 0);
}

static void cube_fini(struct gles_presenter* p, void* user) {
    (void)p;
    struct cube* c = user;
    glDeleteBuffers(1, &c->vbo);
    glDeleteBuffers(1, &c->ibo);
    glDeleteProgram(c->program);
}

static const struct gles_scene cube_scene = {
    .init = cube_init,
    .resize = NULL,
    .draw = cube_draw,
    .fini = cube_fini,
};

int main(void) {
    struct cube c = {0};
    struct gles_presenter p;

    const int dmabuf = CUBE_GLES_MODE == GLES_MODE_DMABUF;
    if (gles_present_init(
            &p,
            CUBE_GLES_MODE,
            GLES_PRESENT_DEPTH,
            &cube_scene,
            &c,
            dmabuf ? "cube (GLES/dmabuf)" : "cube (GLES/EGLSurface)",
            dmabuf ? "hello-wayland-cube-gles-dmabuf" : "hello-wayland-cube-gles-eglsurface") < 0) {
        return 1;
    }
    gles_present_run(&p);
    gles_present_fini(&p);
    return 0;
}
