/* cube_mesh.h -- the one mesh all three cube backends draw.
 *
 * 24 vertices (4 per face, so each face has its own flat normal and
 * color) and 36 indices (2 triangles per face). Winding is
 * COUNTER-CLOCKWISE when viewed from outside, the GL convention; how
 * each API judges that winding after its own clip-space quirks is the
 * scenes' business (see mat4_perspective's y_down note).
 *
 * Also the shared animation math, so the three binaries show the
 * identical rotating cube: time spins it, the pointer tilts it.
 */
#ifndef HELLO_WAYLAND_CUBE_MESH_H
#define HELLO_WAYLAND_CUBE_MESH_H

#include <stdint.h>

#include "common.h"
#include "mat4.h"

struct cube_vertex {
    float pos[3];
    float normal[3];
    float color[3];
    float uv[2]; /* texture coordinates: (0,0) top-left of the face, as
                  * seen from outside -- camera row 0 lands at the top */
};

/* One face = 4 corners, listed CCW from outside: bottom-left,
 * bottom-right, top-right, top-left -- so the uv's are fixed. */
#define CUBE_FACE(nx, ny, nz, r, g, b, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3) \
    {{x0, y0, z0}, {nx, ny, nz}, {r, g, b}, {0, 1}},                                   \
        {{x1, y1, z1}, {nx, ny, nz}, {r, g, b}, {1, 1}},                               \
        {{x2, y2, z2}, {nx, ny, nz}, {r, g, b}, {1, 0}}, {                             \
        {x3, y3, z3}, {nx, ny, nz}, {r, g, b}, {0, 0}                                  \
    }

static const struct cube_vertex cube_vertices[24] = {
    /* +Z front, red */
    CUBE_FACE(0, 0, 1, 0.9f, 0.2f, 0.2f, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1),
    /* -Z back, green */
    CUBE_FACE(0, 0, -1, 0.2f, 0.8f, 0.3f, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1),
    /* +X right, blue */
    CUBE_FACE(1, 0, 0, 0.25f, 0.4f, 0.95f, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1),
    /* -X left, yellow */
    CUBE_FACE(-1, 0, 0, 0.95f, 0.85f, 0.2f, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1),
    /* +Y top, cyan */
    CUBE_FACE(0, 1, 0, 0.2f, 0.85f, 0.9f, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1),
    /* -Y bottom, magenta */
    CUBE_FACE(0, -1, 0, 0.9f, 0.3f, 0.85f, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1),
};

#undef CUBE_FACE

#define CUBE_INDEX_COUNT 36

static const uint16_t cube_indices[CUBE_INDEX_COUNT] = {
    0,  1,  2,  0,  2,  3,  /* +Z */
    4,  5,  6,  4,  6,  7,  /* -Z */
    8,  9,  10, 8,  10, 11, /* +X */
    12, 13, 14, 12, 14, 15, /* -X */
    16, 17, 18, 16, 18, 19, /* +Y */
    20, 21, 22, 20, 22, 23, /* -Y */
};

/* Model matrix: the animation clock (pausable, scroll-scaled, from
 * common.c) spins the cube; the pointer position tilts it. Rotation
 * only, so the same matrix also transforms normals. */
static inline mat4 cube_model_matrix(const struct app* a) {
    float t = (float)(a->anim_ms * 0.001);
    float w = a->width > 0 ? (float)a->width : 1.0f;
    float h = a->height > 0 ? (float)a->height : 1.0f;
    float tilt_y = ((float)a->ptr_x / w - 0.5f) * 1.2f;
    float tilt_x = ((float)a->ptr_y / h - 0.5f) * 1.2f;
    return mat4_mul(mat4_rotate_y(t * 0.8f + tilt_y), mat4_rotate_x(t * 0.5f + tilt_x));
}

/* View * projection for a target of w x h pixels. The two flags are the
 * ONLY per-backend difference in the whole transform chain. `zoom`
 * (the touchpad pinch, from common.c) dollies the camera: distance
 * 4.5 / zoom, so zoom 2 = twice as close. */
static inline mat4 cube_view_proj(int w, int h, int y_down, int z_zero_to_one, double zoom) {
    float aspect = (float)w / (float)(h > 0 ? h : 1);
    mat4 proj =
        mat4_perspective(80.0f * MAT4_PI / 180.0f, aspect, 0.1f, 100.0f, y_down, z_zero_to_one);
    mat4 view = mat4_translate(0.0f, 0.0f, -4.5f / (float)zoom);
    return mat4_mul(proj, view);
}

#endif
