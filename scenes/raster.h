/* raster.h -- the software rasterizer from scene_cube_shm.c, generalized:
 * a vertex carries RASTER_ATTRS floats of varyings (normal, color, uv,
 * whatever the scene wants) and a shade callback receives them
 * interpolated -- the "fragment shader" as a function pointer. Same
 * stages, same order, same comments as the teaching copy; read that
 * one first. Header-only so scenes can include it.
 *
 * Presents y-down (row 0 = top): build projections with y_down = 1,
 * z_zero_to_one = 1; the mesh's CCW-from-outside faces are front. */
#ifndef HELLO_WAYLAND_RASTER_H
#define HELLO_WAYLAND_RASTER_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mat4.h"

enum { RASTER_ATTRS = 8 };

struct raster_target {
    uint32_t* pixels;
    int width, height, stride_px;
    float* depth;
};

struct raster_vertex { /* vertex-stage output: clip position + varyings */
    float clip[4];
    float attr[RASTER_ATTRS];
};

/* the fragment shader: interpolated varyings -> packed 0x00RRGGBB */
typedef uint32_t (*raster_shade_fn)(const float attr[RASTER_ATTRS], void* user);

struct raster_screen_vertex {
    float x, y, z, inv_w;
    float attr[RASTER_ATTRS]; /* pre-divided by w */
};

static inline int raster_clip_near(const struct raster_vertex in[3], struct raster_vertex out[4]) {
    int n = 0;
    for (int i = 0; i < 3; i++) {
        const struct raster_vertex* a = &in[i];
        const struct raster_vertex* b = &in[(i + 1) % 3];
        float da = a->clip[2], db = b->clip[2];
        if (da >= 0.0f) { out[n++] = *a; }
        if ((da >= 0.0f) != (db >= 0.0f)) {
            float t = da / (da - db);
            struct raster_vertex* v = &out[n++];
            for (int k = 0; k < 4; k++) { v->clip[k] = a->clip[k] + t * (b->clip[k] - a->clip[k]); }
            for (int k = 0; k < RASTER_ATTRS; k++) {
                v->attr[k] = a->attr[k] + t * (b->attr[k] - a->attr[k]);
            }
        }
    }
    return n;
}

static inline void raster_to_screen(const struct raster_vertex* c,
                                    int w,
                                    int h,
                                    struct raster_screen_vertex* s) {
    float inv_w = 1.0f / c->clip[3];
    s->x = (c->clip[0] * inv_w * 0.5f + 0.5f) * (float)w;
    s->y = (c->clip[1] * inv_w * 0.5f + 0.5f) * (float)h;
    s->z = c->clip[2] * inv_w;
    s->inv_w = inv_w;
    for (int k = 0; k < RASTER_ATTRS; k++) { s->attr[k] = c->attr[k] * inv_w; }
}

static inline float raster_edge(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

static inline void raster_triangle(const struct raster_target* t,
                                   const struct raster_screen_vertex* v0,
                                   const struct raster_screen_vertex* v1,
                                   const struct raster_screen_vertex* v2,
                                   raster_shade_fn shade,
                                   void* user) {
    float area = raster_edge(v0->x, v0->y, v1->x, v1->y, v2->x, v2->y);
    if (area >= 0.0f) { return; } /* back-facing or degenerate */
    float inv_area = 1.0f / area;

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
            float l0 = raster_edge(v1->x, v1->y, v2->x, v2->y, px, py) * inv_area;
            float l1 = raster_edge(v2->x, v2->y, v0->x, v0->y, px, py) * inv_area;
            float l2 = raster_edge(v0->x, v0->y, v1->x, v1->y, px, py) * inv_area;
            if (l0 < 0.0f || l1 < 0.0f || l2 < 0.0f) { continue; }

            float z = l0 * v0->z + l1 * v1->z + l2 * v2->z;
            size_t idx = (size_t)y * (size_t)t->stride_px + (size_t)x;
            if (z >= t->depth[idx]) { continue; }
            t->depth[idx] = z;

            float w = 1.0f / (l0 * v0->inv_w + l1 * v1->inv_w + l2 * v2->inv_w);
            float attr[RASTER_ATTRS];
            for (int k = 0; k < RASTER_ATTRS; k++) {
                attr[k] = (l0 * v0->attr[k] + l1 * v1->attr[k] + l2 * v2->attr[k]) * w;
            }
            t->pixels[idx] = shade(attr, user);
        }
    }
}

/* One indexed draw: clip, divide, fan, rasterize. */
static inline void raster_draw_indexed(const struct raster_target* t,
                                       const struct raster_vertex* verts,
                                       const uint16_t* indices,
                                       int index_count,
                                       raster_shade_fn shade,
                                       void* user) {
    for (int i = 0; i + 2 < index_count; i += 3) {
        struct raster_vertex tri[3] = {verts[indices[i]],
                                       verts[indices[i + 1]],
                                       verts[indices[i + 2]]};
        struct raster_vertex poly[4];
        int n = raster_clip_near(tri, poly);
        if (n < 3) { continue; }
        struct raster_screen_vertex sv[4];
        for (int k = 0; k < n; k++) { raster_to_screen(&poly[k], t->width, t->height, &sv[k]); }
        for (int k = 1; k + 1 < n; k++) {
            raster_triangle(t, &sv[0], &sv[k], &sv[k + 1], shade, user);
        }
    }
}

#endif
