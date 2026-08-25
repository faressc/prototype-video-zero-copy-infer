/* scene_cam_shm.c -- the camera through the CPU.
 *
 * The copy path, on purpose: the ISP's buffer is mmap'd, converted
 * NV12 -> RGB into a scratch image (that IS the "sampler" the GPU
 * versions get for free), the effect chain runs as per-pixel C
 * functions over it (the "fragment shader", one pass per active
 * effect, ping-ponging between two scratch images), and the result
 * is scaled into the letterboxed window -- or mapped onto the cube.
 * Reference for colors and for what cam_effects.glsl computes; keep
 * the two in step.
 *
 * Buffer lifetime is trivial here: the CPU IS the reader, so a buffer
 * is free the moment draw() returns (fence "done" = always true).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cam_stream.h"
#include "cube_mesh.h"
#include "effect_chain.h"
#include "nv12_convert.h"
#include "raster.h"
#include "shm_present.h"

struct cam_scene {
    struct cam_stream stream;
    uint32_t cw, ch;      /* camera size */
    uint32_t* rgb;        /* converted camera frame: the "texture" */
    uint32_t* scratch[2]; /* the chain ping-pongs between these */
};

/* ------------------------------------------------------------------ */
/* the fence ops: the CPU reads synchronously                         */
/* ------------------------------------------------------------------ */

static int cpu_fence_done(void* fence, void* user) {
    (void)fence;
    (void)user;
    return 1;
}

static const struct cam_fence_ops cpu_fence_ops = {.done = cpu_fence_done, .destroy = NULL};

/* ------------------------------------------------------------------ */
/* "texture sampling" and the effects, in C                           */
/* ------------------------------------------------------------------ */

static inline float px_r(uint32_t p) {
    return (float)((p >> 16) & 255) / 255.0f;
}
static inline float px_g(uint32_t p) {
    return (float)((p >> 8) & 255) / 255.0f;
}
static inline float px_b(uint32_t p) {
    return (float)(p & 255) / 255.0f;
}
static inline uint32_t px_pack(float r, float g, float b) {
    return (nv12_clamp8(r * 255.0f) << 16) | (nv12_clamp8(g * 255.0f) << 8) |
           nv12_clamp8(b * 255.0f);
}
static inline float luma(uint32_t p) {
    return 0.2126f * px_r(p) + 0.7152f * px_g(p) + 0.0722f * px_b(p);
}

/* clamp-to-edge sampling, integer coordinates */
static inline uint32_t tex(const uint32_t* img, int w, int h, int x, int y) {
    if (x < 0) { x = 0; }
    if (y < 0) { y = 0; }
    if (x >= w) { x = w - 1; }
    if (y >= h) { y = h - 1; }
    return img[(size_t)y * (size_t)w + (size_t)x];
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

static uint32_t thermal(float y) {
    float t0 = clampf(y * 3.0f, 0, 1), t1 = clampf(y * 3.0f - 1.0f, 0, 1),
          t2 = clampf(y * 3.0f - 2.0f, 0, 1);
    float ar = 0.0f + (0.6f - 0.0f) * t0, ag = 0.0f, ab = 0.25f + (0.7f - 0.25f) * t0;
    float br = ar + (1.0f - ar) * t1, bg = ag + (0.85f - ag) * t1, bb = ab + (0.1f - ab) * t1;
    return px_pack(br + (1.0f - br) * t2, bg + (1.0f - bg) * t2, bb + (1.0f - bb) * t2);
}

/* One pass of the chain: `in` -> `out`, both camera-sized. Mirrors
 * apply_effect() in cam_effects.glsl, pass id for pass id. */
static void effect_pass(int pass, const uint32_t* in, uint32_t* out, int w, int h) {
    switch (pass) {
        case PASS_THERMAL:
            for (size_t i = 0; i < (size_t)w * (size_t)h; i++) { out[i] = thermal(luma(in[i])); }
            break;
        case PASS_SOBEL:
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float tl = luma(tex(in, w, h, x - 1, y - 1)),
                          tc = luma(tex(in, w, h, x, y - 1)),
                          tr = luma(tex(in, w, h, x + 1, y - 1)),
                          ml = luma(tex(in, w, h, x - 1, y)), mr = luma(tex(in, w, h, x + 1, y)),
                          bl = luma(tex(in, w, h, x - 1, y + 1)),
                          bc = luma(tex(in, w, h, x, y + 1)),
                          br = luma(tex(in, w, h, x + 1, y + 1));
                    float gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                    float gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
                    float g = clampf(sqrtf(gx * gx + gy * gy) * 2.0f, 0, 1);
                    out[(size_t)y * (size_t)w + (size_t)x] = px_pack(g, g, g);
                }
            }
            break;
        case PASS_MOSAIC: { /* 64 square cells across */
            float cell = (float)w / 64.0f;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int sx = (int)((floorf((float)x / cell) + 0.5f) * cell);
                    int sy = (int)((floorf((float)y / cell) + 0.5f) * cell);
                    out[(size_t)y * (size_t)w + (size_t)x] = tex(in, w, h, sx, sy);
                }
            }
            break;
        }
        case PASS_BLUR_H:
        case PASS_BLUR_V: { /* one direction of the separable gaussian */
            static const float wgt[5] = {0.2270f, 0.1945f, 0.1216f, 0.0540f, 0.0162f};
            const int horizontal = pass == PASS_BLUR_H;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float r = 0, g = 0, b = 0;
                    for (int k = -4; k <= 4; k++) {
                        uint32_t p =
                            horizontal ? tex(in, w, h, x + 2 * k, y) : tex(in, w, h, x, y + 2 * k);
                        float wk = wgt[k < 0 ? -k : k];
                        r += px_r(p) * wk;
                        g += px_g(p) * wk;
                        b += px_b(p) * wk;
                    }
                    out[(size_t)y * (size_t)w + (size_t)x] = px_pack(r, g, b);
                }
            }
            break;
        }
        default: memcpy(out, in, (size_t)w * (size_t)h * sizeof(uint32_t)); break;
    }
}

/* ------------------------------------------------------------------ */
/* the camera on the cube (C). The software rasterizer from raster.h  */
/* with a "fragment shader" that samples the chain's output at the    */
/* interpolated uv -- texture mapping, and the first place the        */
/* perspective-correct 1/w bookkeeping is visible: without it the     */
/* image would bend on tilted faces.                                  */
/* ------------------------------------------------------------------ */

struct cube_ctx {
    const uint32_t* img;
    int w, h;
};

static uint32_t cube_shade(const float attr[RASTER_ATTRS], void* user) {
    const struct cube_ctx* c = user;
    /* attr 0..2 = world normal, 3..4 = uv */
    int x = (int)(attr[3] * (float)c->w), y = (int)(attr[4] * (float)c->h);
    uint32_t p = tex(c->img, c->w, c->h, x, y); /* the texture lookup */

    float nx = attr[0], ny = attr[1], nz = attr[2];
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    float d = len > 0 ? (nx * 0.4f + ny * 0.8f + nz * 1.0f) / (len * 1.3416408f) : 0;
    float k = 0.35f + 0.65f * (d < 0 ? 0 : d);
    return px_pack(px_r(p) * k, px_g(p) * k, px_b(p) * k);
}

static void draw_cube(struct cam_scene* s,
                      const struct app* a,
                      const struct shm_target* t,
                      const uint32_t* img) {
    for (size_t i = 0; i < (size_t)t->width * (size_t)t->height; i++) { t->depth[i] = 1.0f; }

    mat4 model = cube_model_matrix(a);
    mat4 mvp = mat4_mul(cube_view_proj(t->width, t->height, 1, 1, a->zoom), model);

    /* Center-crop the camera image to the square face ("cover"): full
     * extent of the short axis, a centered window of the long one.
     * Linear, so applied per vertex; the rasterizer interpolates. */
    const float aspect = (float)s->cw / (float)s->ch;
    const float su = aspect > 1.0f ? 1.0f / aspect : 1.0f;
    const float sv = aspect > 1.0f ? 1.0f : aspect;

    /* vertex stage: position through mvp, normal through the rotation,
     * uv aspect-corrected */
    struct raster_vertex verts[24];
    for (int i = 0; i < 24; i++) {
        const struct cube_vertex* in = &cube_vertices[i];
        float pos[4] = {in->pos[0], in->pos[1], in->pos[2], 1.0f};
        mat4_mul_vec4(&mvp, pos, verts[i].clip);
        memset(verts[i].attr, 0, sizeof(verts[i].attr));
        for (int r = 0; r < 3; r++) {
            verts[i].attr[r] = model.m[0 * 4 + r] * in->normal[0] +
                               model.m[1 * 4 + r] * in->normal[1] +
                               model.m[2 * 4 + r] * in->normal[2];
        }
        verts[i].attr[3] = 0.5f + (in->uv[0] - 0.5f) * su;
        verts[i].attr[4] = 0.5f + (in->uv[1] - 0.5f) * sv;
    }

    struct raster_target rt = {t->pixels, t->width, t->height, t->stride_px, t->depth};
    struct cube_ctx ctx = {img, (int)s->cw, (int)s->ch};
    raster_draw_indexed(&rt, verts, cube_indices, CUBE_INDEX_COUNT, cube_shade, &ctx);
}

/* ------------------------------------------------------------------ */

static void on_cam_fd(struct app* a) {
    struct shm_presenter* p = (struct shm_presenter*)a;
    struct cam_scene* s = p->user;
    cam_stream_pump(&s->stream);
}

static void cam_init(struct shm_presenter* p, void* user) {
    struct cam_scene* s = user;
    if (cam_stream_open(&s->stream, &cpu_fence_ops, NULL) < 0) { exit(1); }
    const uint32_t pf = s->stream.cam.format.pixelformat;
    if (pf != V4L2_PIX_FMT_NV12 && pf != V4L2_PIX_FMT_YUYV) {
        fprintf(stderr, "cam_shm: only NV12 and YUYV are implemented on the CPU path\n");
        exit(1);
    }
    s->cw = s->stream.cam.format.width;
    s->ch = s->stream.cam.format.height;
    size_t n = (size_t)s->cw * (size_t)s->ch;
    s->rgb = malloc(n * sizeof(uint32_t));
    s->scratch[0] = malloc(n * sizeof(uint32_t));
    s->scratch[1] = malloc(n * sizeof(uint32_t));

    p->app.aux_fd = cam_stream_fd(&s->stream);
    p->app.on_aux_fd = on_cam_fd;
    p->app.effect_count = FX_COUNT;
}

static void cam_draw(struct shm_presenter* p, const struct shm_target* t, void* user) {
    struct cam_scene* s = user;
    cam_stream_reap(&s->stream);

    /* clear to black (the letterbox bars) */
    for (int y = 0; y < t->height; y++) {
        memset(t->pixels + (size_t)y * (size_t)t->stride_px, 0, (size_t)t->width * 4);
    }

    int latest = cam_stream_latest(&s->stream);
    if (latest < 0) { return; } /* no frame yet */

    /* "sample the camera": map + convert. The map is the ISP's own
     * pages -- zero-copy up to here; the conversion is the copy. */
    const uint8_t* base = camera_map(&s->stream.cam, (uint32_t)latest);
    if (!base) { return; }
    camera_frame_to_xrgb(&s->stream.cam, base, s->rgb);
    cam_stream_rendered(&s->stream, latest, NULL); /* CPU: done reading already */

    /* the chain: each pass reads the previous stage, writes the other
     * scratch. `img` ends up pointing at the final stage. */
    int passes[EFFECT_MAX_PASSES];
    int n = effect_chain_passes(p->app.effect, passes);
    const uint32_t* img = s->rgb;
    for (int i = 0; i < n; i++) {
        if (passes[i] == PASS_NONE) { break; } /* passthrough: nothing to do */
        uint32_t* dst = s->scratch[i & 1];
        effect_pass(passes[i], img, dst, (int)s->cw, (int)s->ch);
        img = dst;
    }

    if (p->app.mode) { /* C: the chain's output on the cube */
        draw_cube(s, &p->app, t, img);
        return;
    }

    /* letterbox: the largest rectangle with the camera's aspect that
     * fits the window, centered; nearest-neighbor scale into it */
    float sx = (float)t->width / (float)s->cw, sy = (float)t->height / (float)s->ch;
    float sc = sx < sy ? sx : sy;
    int dw = (int)((float)s->cw * sc), dh = (int)((float)s->ch * sc);
    int x0 = (t->width - dw) / 2, y0 = (t->height - dh) / 2;
    for (int y = 0; y < dh; y++) {
        uint32_t* row = t->pixels + (size_t)(y0 + y) * (size_t)t->stride_px + (size_t)x0;
        const uint32_t* src = img + (size_t)((float)y / sc) * (size_t)s->cw;
        for (int x = 0; x < dw; x++) { row[x] = src[(size_t)((float)x / sc)]; }
    }
}

static void cam_fini(struct shm_presenter* p, void* user) {
    (void)p;
    struct cam_scene* s = user;
    cam_stream_close(&s->stream);
    free(s->rgb);
    free(s->scratch[0]);
    free(s->scratch[1]);
}

static const struct shm_scene cam_scene = {
    .init = cam_init,
    .resize = NULL,
    .draw = cam_draw,
    .fini = cam_fini,
};

int main(void) {
    struct cam_scene s = {0};
    struct shm_presenter p;
    if (shm_present_init(&p,
                         SHM_PRESENT_DEPTH,
                         &cam_scene,
                         &s,
                         "camera (shm, CPU)",
                         "hello-wayland-cam-shm") < 0) {
        return 1;
    }
    shm_present_run(&p);
    shm_present_fini(&p);
    return 0;
}
