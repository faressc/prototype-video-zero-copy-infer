/* mat4.h -- the smallest useful 4x4 matrix kit.
 *
 * Column-major float[16], i.e. GL/GLSL memory order: element (row r,
 * column c) lives at m[c*4 + r]. The same 64 bytes upload unchanged to
 * glUniformMatrix4fv(transpose = GL_FALSE), to a Vulkan push-constant
 * block, and into a GLSL `mat4`.
 *
 * Coordinate conventions (right-handed, camera looks down -Z):
 *   model  -> world  : mat4_rotate_* / mat4_translate
 *   world  -> camera : the view matrix (a translate here)
 *   camera -> clip   : mat4_perspective -- the ONLY place the three
 *                      backends' clip spaces differ, see the flags.
 */
#ifndef HELLO_WAYLAND_MAT4_H
#define HELLO_WAYLAND_MAT4_H

#include <math.h>
#include <string.h>

#define MAT4_PI 3.14159265358979323846f

typedef struct mat4 {
    float m[16];
} mat4;

static inline mat4 mat4_identity(void) {
    mat4 r;
    memset(r.m, 0, sizeof(r.m));
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

/* a * b: apply b first, then a (matrix convention, column vectors). */
static inline mat4 mat4_mul(mat4 a, mat4 b) {
    mat4 r;
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) { s += a.m[k * 4 + row] * b.m[c * 4 + k]; }
            r.m[c * 4 + row] = s;
        }
    }
    return r;
}

/* out = m * v, for a 4-component column vector -- what the GPU does per
 * vertex when the shader writes `gl_Position = mvp * vec4(pos, 1.0)`. */
static inline void mat4_mul_vec4(const mat4* m, const float v[4], float out[4]) {
    for (int row = 0; row < 4; row++) {
        out[row] = m->m[0 * 4 + row] * v[0] + m->m[1 * 4 + row] * v[1] + m->m[2 * 4 + row] * v[2] +
                   m->m[3 * 4 + row] * v[3];
    }
}

static inline mat4 mat4_translate(float x, float y, float z) {
    mat4 r = mat4_identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static inline mat4 mat4_rotate_x(float rad) {
    float c = cosf(rad), s = sinf(rad);
    mat4 r = mat4_identity();
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

static inline mat4 mat4_rotate_y(float rad) {
    float c = cosf(rad), s = sinf(rad);
    mat4 r = mat4_identity();
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

/* Perspective projection. fovy in radians.
 *
 * y_down:         negate clip-space y. GL window surfaces present with
 *                 y UP (NDC y=-1 = bottom row). FBO-rendered dmabufs and
 *                 Vulkan swapchain images present with y DOWN (NDC y=-1
 *                 = row 0 = the TOP row on screen) -- flipping y here is
 *                 how the same mesh lands upright in all three.
 * z_zero_to_one:  Vulkan's clip z range is [0,1]; GL's is [-1,1]. The
 *                 difference is one row of the matrix.
 *
 * Flipping y also flips the triangle winding the API observes; the
 * caller compensates with the front-face setting (see the scenes). */
static inline mat4 mat4_perspective(float fovy,
                                    float aspect,
                                    float near,
                                    float far,
                                    int y_down,
                                    int z_zero_to_one) {
    float f = 1.0f / tanf(fovy * 0.5f);
    mat4 r;
    memset(r.m, 0, sizeof(r.m));
    r.m[0] = f / aspect;
    r.m[5] = y_down ? -f : f;
    if (z_zero_to_one) {
        r.m[10] = far / (near - far);
        r.m[14] = (far * near) / (near - far);
    } else {
        r.m[10] = (far + near) / (near - far);
        r.m[14] = (2.0f * far * near) / (near - far);
    }
    r.m[11] = -1.0f; /* w = -z_camera: the perspective divide */
    return r;
}

#endif
