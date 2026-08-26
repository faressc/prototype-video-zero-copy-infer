/* hand_overlay_gles.c -- the shared overlay triangles, drawn through
 * GLES2.
 *
 * The whole backend is one program, one streamed VBO and one
 * glDrawArrays, because hand_overlay.h already turned the skeleton into
 * triangles in frame-normalised space. All this has to decide is where
 * [0,1]^2 lands: a letterboxed rectangle of the window, or -- in cube
 * mode -- nothing at all, because there the overlay is drawn into the
 * effect chain's scratch at camera resolution and the cube samples it
 * along with the image.
 *
 * Shaders as runtime source strings, matching scene_cam_gles.c's style
 * rather than introducing a fourth shader build model for 30 lines.
 */
#include <stdio.h>
#include <stdlib.h>

#include "hand_overlay_gles.h"

/* One multiply-add, and the caller does all the thinking. Earlier this
 * had a u_flip uniform as well and that was the bug: two places deciding
 * the vertical convention, neither of them the place that knows it. The
 * caller does know -- it computed the letterbox rectangle and it knows
 * which way its target stores rows -- so it hands over a scale and an
 * offset that are already correct, and the shader has no opinion. */
static const char* vert_src =
    "attribute vec2 a_pos;\n" /* frame-normalised, [0,1] x [0,1], y DOWN */
    "attribute vec4 a_col;\n"
    "uniform vec4 u_rect;\n"  /* xy = scale, zw = offset, straight into NDC */
    "varying vec4 v_col;\n"
    "void main() {\n"
    "    v_col = a_col;\n"
    "    gl_Position = vec4(a_pos * u_rect.xy + u_rect.zw, 0.0, 1.0);\n"
    "}\n";

static const char* frag_src =
    "precision mediump float;\n"
    "varying vec4 v_col;\n"
    "void main() { gl_FragColor = v_col; }\n";

static GLuint compile(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "hand overlay shader failed:\n%s\n", log);
        exit(1);
    }
    return sh;
}

void hand_overlay_gles_init(struct hand_overlay_gles* g) {
    GLuint vs = compile(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile(GL_FRAGMENT_SHADER, frag_src);
    g->program = glCreateProgram();
    glAttachShader(g->program, vs);
    glAttachShader(g->program, fs);
    glLinkProgram(g->program);
    GLint ok = 0;
    glGetProgramiv(g->program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(g->program, sizeof log, NULL, log);
        fprintf(stderr, "hand overlay link failed:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    g->a_pos = glGetAttribLocation(g->program, "a_pos");
    g->a_col = glGetAttribLocation(g->program, "a_col");
    g->u_rect = glGetUniformLocation(g->program, "u_rect");
    glGenBuffers(1, &g->vbo);
}

void hand_overlay_gles_fini(struct hand_overlay_gles* g) {
    if (g->vbo) { glDeleteBuffers(1, &g->vbo); }
    if (g->program) { glDeleteProgram(g->program); }
}

void hand_overlay_gles_draw(struct hand_overlay_gles* g,
                            const struct hand_vert* v,
                            int n,
                            float sx,
                            float sy,
                            float ox,
                            float oy) {
    if (n <= 0) { return; }
    glUseProgram(g->program);
    glUniform4f(g->u_rect, sx, sy, ox, oy);
    glBindBuffer(GL_ARRAY_BUFFER, g->vbo);
    /* STREAM_DRAW: written once per frame, read once, never reused */
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)n * sizeof *v), v, GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g->a_pos);
    glVertexAttribPointer((GLuint)g->a_pos, 2, GL_FLOAT, GL_FALSE, sizeof *v, (const void*)0);
    glEnableVertexAttribArray((GLuint)g->a_col);
    /* 0xRRGGBBAA in memory is R,G,B,A byte order on little-endian, which
     * is what UNSIGNED_BYTE normalized expects -- so the colour constants
     * in hand_overlay.h read the way they are written */
    glVertexAttribPointer((GLuint)g->a_col, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof *v,
                          (const void*)offsetof(struct hand_vert, rgba));
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, n);
    glDisable(GL_BLEND);
    glDisableVertexAttribArray((GLuint)g->a_pos);
    glDisableVertexAttribArray((GLuint)g->a_col);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
