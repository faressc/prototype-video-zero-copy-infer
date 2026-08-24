/* scene_cube_shm.c -- the cube through a SOFTWARE rasterizer.
 *
 * Every fixed-function stage the GPU scenes configure with a struct is
 * a loop here, in pipeline order (README §15's diagram, executed):
 *
 *   vertex stage           run_vertex_shader()   the .vert, per vertex
 *   primitive assembly     the index loop         TRIANGLE_LIST
 *   clipping               clip_near()            why w must stay > 0
 *   perspective divide     to_screen()            x/w, y/w, z/w, keep 1/w
 *   viewport transform     to_screen()            NDC -> pixels
 *   face culling           signed area            frontFace / cullMode
 *   rasterization          raster_triangle()      edge functions = coverage
 *   interpolation          barycentrics           the tilted plane, per pixel,
 *                                                 perspective-CORRECT (÷w)
 *   depth test             LESS against depth[]   VkPipelineDepthStencilState
 *   fragment stage         shade()                the .frag, per pixel
 *   output                 pack to XRGB8888       the blend/write unit
 *
 * Same mesh, same matrices, same light as the GPU versions. Presents
 * y-down (row 0 = top), so the projection is built like Vulkan's
 * (y_down = 1) -- and, like Vulkan, this rasterizer judges winding AS
 * SEEN ON SCREEN, so the CCW mesh is front-facing with no flip.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cube_mesh.h"
#include "mat4.h"
#include "shm_present.h"

/* ------------------------------------------------------------------ */
/* Vertex stage                                                       */
/* ------------------------------------------------------------------ */

/* What the vertex shader emits: gl_Position (clip space, 4 floats --
 * NOT yet divided by w) plus the varyings. */
struct clip_vertex {
    float clip[4];
    float normal[3];
    float color[3];
};

/* cube.vert, as a C function. */
static void run_vertex_shader(const mat4* mvp,
                              const mat4* model,
                              const struct cube_vertex* in,
                              struct clip_vertex* out) {
    float pos[4] = {in->pos[0], in->pos[1], in->pos[2], 1.0f};
    mat4_mul_vec4(mvp, pos, out->clip);

    /* mat3(model) * normal: the upper-left 3x3, rotation only */
    for (int r = 0; r < 3; r++) {
        out->normal[r] = model->m[0 * 4 + r] * in->normal[0] + model->m[1 * 4 + r] * in->normal[1] +
                         model->m[2 * 4 + r] * in->normal[2];
    }
    memcpy(out->color, in->color, sizeof(out->color));
}

/* ------------------------------------------------------------------ */
/* Clipping (near plane only)                                         */
/* ------------------------------------------------------------------ */

/* The perspective divide below computes x/w. For a vertex BEHIND the
 * camera w is negative, and a vertex AT the camera has w = 0 -- the
 * projected point flips sides or flies to infinity, and a triangle
 * spanning the camera plane turns into garbage across the whole
 * screen. Pinch in far enough (zoom 5 puts the camera inside the cube)
 * and that happens every frame. So triangles are cut against the near
 * plane FIRST, in clip space, where everything is still linear.
 *
 * Near plane in clip space, for a [0,1] z range: z_clip >= 0.
 * Sutherland-Hodgman against one plane: walk the edges, keep inside
 * vertices, emit the intersection where an edge crosses. A triangle
 * yields 3 or 4 vertices (or 0). Attributes are interpolated linearly
 * along the edge -- legal in clip space, not after the divide.
 *
 * The GPU clips against all six planes (and uses a guard band for
 * x/y); here x/y/far are left to the bounding-box clamp and the depth
 * test, which is safe once w > 0 is guaranteed. */
static int clip_near(const struct clip_vertex in[3], struct clip_vertex out[4]) {
    int n = 0;
    for (int i = 0; i < 3; i++) {
        const struct clip_vertex* a = &in[i];
        const struct clip_vertex* b = &in[(i + 1) % 3];
        float da = a->clip[2]; /* signed distance to the plane */
        float db = b->clip[2];
        if (da >= 0.0f) { out[n++] = *a; }
        if ((da >= 0.0f) != (db >= 0.0f)) {
            float t = da / (da - db);
            struct clip_vertex* v = &out[n++];
            for (int k = 0; k < 4; k++) { v->clip[k] = a->clip[k] + t * (b->clip[k] - a->clip[k]); }
            for (int k = 0; k < 3; k++) {
                v->normal[k] = a->normal[k] + t * (b->normal[k] - a->normal[k]);
                v->color[k] = a->color[k] + t * (b->color[k] - a->color[k]);
            }
        }
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* Perspective divide + viewport transform                            */
/* ------------------------------------------------------------------ */

/* A vertex the rasterizer works with: pixel coordinates, depth, and
 * the varyings PRE-DIVIDED by w (plus 1/w itself), which is what makes
 * interpolating them across the screen perspective-correct. */
struct screen_vertex {
    float x, y;      /* pixels */
    float z;         /* depth in [0,1] */
    float inv_w;     /* 1/w */
    float normal[3]; /* normal / w */
    float color[3];  /* color / w */
};

static void to_screen(const struct clip_vertex* c, int w, int h, struct screen_vertex* s) {
    float inv_w = 1.0f / c->clip[3];
    float nx = c->clip[0] * inv_w; /* NDC: the perspective divide */
    float ny = c->clip[1] * inv_w;
    float nz = c->clip[2] * inv_w;

    /* the viewport lens: [-1,1] -> [0,w] x [0,h]. y is already "down"
     * because the projection flipped it (y_down = 1). */
    s->x = (nx * 0.5f + 0.5f) * (float)w;
    s->y = (ny * 0.5f + 0.5f) * (float)h;
    s->z = nz;
    s->inv_w = inv_w;
    for (int k = 0; k < 3; k++) {
        s->normal[k] = c->normal[k] * inv_w;
        s->color[k] = c->color[k] * inv_w;
    }
}

/* ------------------------------------------------------------------ */
/* Fragment stage                                                     */
/* ------------------------------------------------------------------ */

/* cube.frag, as a C function: one directional light, then pack to
 * XRGB8888 -- the "blend/write unit" with blending off. */
static uint32_t shade(const float normal[3], const float color[3]) {
    static const float light[3] = {0.4f / 1.3416408f, 0.8f / 1.3416408f, 1.0f / 1.3416408f};
    float len = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    float d = 0.0f;
    if (len > 0.0f) {
        d = (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]) / len;
        if (d < 0.0f) { d = 0.0f; }
    }
    float k = 0.25f + 0.75f * d;
    uint32_t r = (uint32_t)(fminf(color[0] * k, 1.0f) * 255.0f + 0.5f);
    uint32_t g = (uint32_t)(fminf(color[1] * k, 1.0f) * 255.0f + 0.5f);
    uint32_t b = (uint32_t)(fminf(color[2] * k, 1.0f) * 255.0f + 0.5f);
    return (r << 16) | (g << 8) | b;
}

/* ------------------------------------------------------------------ */
/* Rasterizer                                                         */
/* ------------------------------------------------------------------ */

/* Edge function: twice the signed area of triangle (a, b, p). Its sign
 * says which side of the directed edge a->b the point p lies on; the
 * three of them per pixel are the coverage test AND, normalized, the
 * barycentric weights. This is the "two for-loops cast in silicon"
 * from README §15, uncast. */
static float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

static void raster_triangle(const struct shm_target* t,
                            const struct screen_vertex* v0,
                            const struct screen_vertex* v1,
                            const struct screen_vertex* v2) {
    /* Face culling. In y-down pixel space a triangle that LOOKS
     * counter-clockwise has NEGATIVE signed area by this formula. Front
     * = CCW on screen (Vulkan's definition); the mesh is CCW from
     * outside and the projection flipped y, so front faces come out
     * CCW on screen. Positive area = back face -> discard. Zero =
     * degenerate (edge-on) -> nothing to draw either. */
    float area = edge(v0->x, v0->y, v1->x, v1->y, v2->x, v2->y);
    if (area >= 0.0f) { return; }
    float inv_area = 1.0f / area;

    /* Bounding box, clamped to the target: the cheap stand-in for x/y
     * clipping. Pixels are sampled at their centers (+0.5). */
    int x0 = (int)floorf(fminf(fminf(v0->x, v1->x), v2->x));
    int x1 = (int)ceilf(fmaxf(fmaxf(v0->x, v1->x), v2->x));
    int y0 = (int)floorf(fminf(fminf(v0->y, v1->y), v2->y));
    int y1 = (int)ceilf(fmaxf(fmaxf(v0->y, v1->y), v2->y));
    if (x0 < 0) { x0 = 0; }
    if (y0 < 0) { y0 = 0; }
    if (x1 > t->width) { x1 = t->width; }
    if (y1 > t->height) { y1 = t->height; }

    for (int y = y0; y < y1; y++) {
        float py = (float)y + 0.5f;
        for (int x = x0; x < x1; x++) {
            float px = (float)x + 0.5f;

            /* coverage: inside iff all three barycentrics are >= 0.
             * (Dividing by the signed area makes the test independent
             * of winding.) A pixel exactly on a shared edge passes for
             * both triangles; the depth test then resolves it. GPUs
             * use the "top-left rule" to make that a strict choice. */
            float l0 = edge(v1->x, v1->y, v2->x, v2->y, px, py) * inv_area;
            float l1 = edge(v2->x, v2->y, v0->x, v0->y, px, py) * inv_area;
            float l2 = edge(v0->x, v0->y, v1->x, v1->y, px, py) * inv_area;
            if (l0 < 0.0f || l1 < 0.0f || l2 < 0.0f) { continue; }

            /* Depth: z/w is affine in screen space, so plain barycentric
             * interpolation of the divided z is exact -- the same
             * property that makes v_px exact in §15. */
            float z = l0 * v0->z + l1 * v1->z + l2 * v2->z;
            size_t idx = (size_t)y * (size_t)t->stride_px + (size_t)x;

            /* the depth test: VK_COMPARE_OP_LESS, depthWriteEnable */
            if (z >= t->depth[idx]) { continue; }
            t->depth[idx] = z;

            /* Varyings: interpolate attribute/w and 1/w (both affine in
             * screen space), then divide -- perspective-correct. Skip
             * the divide and textures swim on rotating surfaces. */
            float inv_w = l0 * v0->inv_w + l1 * v1->inv_w + l2 * v2->inv_w;
            float w = 1.0f / inv_w;
            float normal[3], color[3];
            for (int k = 0; k < 3; k++) {
                normal[k] = (l0 * v0->normal[k] + l1 * v1->normal[k] + l2 * v2->normal[k]) * w;
                color[k] = (l0 * v0->color[k] + l1 * v1->color[k] + l2 * v2->color[k]) * w;
            }

            t->pixels[idx] = shade(normal, color);
        }
    }
}

/* ------------------------------------------------------------------ */
/* The frame                                                          */
/* ------------------------------------------------------------------ */

static void cube_init(struct shm_presenter* p, void* user) {
    (void)p;
    (void)user; /* nothing to compile, nothing to upload: the mesh is
                 * already in the only memory the CPU has */
}

static void cube_draw(struct shm_presenter* p, const struct shm_target* t, void* user) {
    (void)user;
    const struct app* a = &p->app;

    /* loadOp = CLEAR, for both attachments: (0.08, 0.08, 0.10) and far */
    const uint32_t clear = (20u << 16) | (20u << 8) | 26u;
    for (int y = 0; y < t->height; y++) {
        uint32_t* row = t->pixels + (size_t)y * (size_t)t->stride_px;
        for (int x = 0; x < t->width; x++) { row[x] = clear; }
    }
    for (size_t i = 0; i < (size_t)t->width * (size_t)t->height; i++) { t->depth[i] = 1.0f; }

    /* the "push constants" */
    mat4 model = cube_model_matrix(a);
    mat4 mvp = mat4_mul(cube_view_proj(t->width, t->height, 1, 1, a->zoom), model);

    /* vertex stage: 24 invocations, once per vertex -- indexed drawing
     * means shared vertices are transformed once, not per triangle */
    struct clip_vertex clipped[24];
    for (int i = 0; i < 24; i++) {
        run_vertex_shader(&mvp, &model, &cube_vertices[i], &clipped[i]);
    }

    /* primitive assembly: TRIANGLE_LIST over the index buffer */
    for (int i = 0; i < CUBE_INDEX_COUNT; i += 3) {
        struct clip_vertex tri[3] = {clipped[cube_indices[i]],
                                     clipped[cube_indices[i + 1]],
                                     clipped[cube_indices[i + 2]]};

        /* clip -> 0..4 vertices -> fan into triangles */
        struct clip_vertex poly[4];
        int n = clip_near(tri, poly);
        if (n < 3) { continue; }

        struct screen_vertex sv[4];
        for (int k = 0; k < n; k++) { to_screen(&poly[k], t->width, t->height, &sv[k]); }
        for (int k = 1; k + 1 < n; k++) { raster_triangle(t, &sv[0], &sv[k], &sv[k + 1]); }
    }
}

static void cube_fini(struct shm_presenter* p, void* user) {
    (void)p;
    (void)user;
}

static const struct shm_scene cube_scene = {
    .init = cube_init,
    .resize = NULL,
    .draw = cube_draw,
    .fini = cube_fini,
};

int main(void) {
    struct shm_presenter p;
    if (shm_present_init(&p,
                         SHM_PRESENT_DEPTH,
                         &cube_scene,
                         NULL,
                         "cube (shm, software)",
                         "hello-wayland-cube-shm") < 0) {
        return 1;
    }
    shm_present_run(&p);
    shm_present_fini(&p);
    return 0;
}
