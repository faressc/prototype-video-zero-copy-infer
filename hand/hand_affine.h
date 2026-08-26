/* hand_affine.h -- the 2x3 map from a tensor pixel to a frame pixel,
 * and nothing else.
 *
 * Stage five needs two apparently different preprocessing steps: the
 * palm detector wants the whole frame letterboxed into 192x192, and the
 * landmark model wants a rotated, expanded crop resampled into 224x224.
 * They are the same operation:
 *
 *   for each DESTINATION pixel centre (u, v) of a [1, th, tw, 3] tensor,
 *   map it through an affine into SOURCE (frame) pixel coordinates,
 *   sample there, convert, normalise, store.
 *
 * Only the matrix differs. So there is one WGSL pass and one C loop
 * (hand_frame_wgpu.c / hand_frame_cpu.c), and because they are handed
 * the same six floats they compute the same thing -- which is what makes
 * "the CPU EP and the WebGPU EP see the same input" checkable instead of
 * hopeful.
 *
 * Direction convention, stated once because getting it backwards is the
 * classic bug: the matrix goes DESTINATION -> SOURCE. That is the useful
 * direction for a resampler (walk the output, gather the input), and it
 * is also the direction that maps a model's output coordinates back onto
 * the frame -- so decoding needs no inverse at all.
 *
 * Header-only, static inline, like scenes/mat4.h.
 */
#ifndef HELLO_WAYLAND_HAND_AFFINE_H
#define HELLO_WAYLAND_HAND_AFFINE_H

#include <math.h>
#include <stdint.h>

/* x = m[0]*u + m[1]*v + m[2]
 * y = m[3]*u + m[4]*v + m[5]   -- (u,v) tensor px, (x,y) frame px */
struct hand_affine {
    float m[6];
};

static inline void hand_affine_apply(struct hand_affine a, float u, float v, float* x, float* y) {
    *x = a.m[0] * u + a.m[1] * v + a.m[2];
    *y = a.m[3] * u + a.m[4] * v + a.m[5];
}

/* The uniform scale of a fit map (frame px per tensor px). Meaningful
 * only for the two fit constructors below, which have no rotation and no
 * shear -- which is exactly why detection maths is done in their space
 * (see hand_decode.c): a length there is a length in the frame times
 * this one number, so square boxes stay square. */
static inline float hand_affine_scale(struct hand_affine a) {
    return a.m[0];
}

/* Fit the whole frame inside the tensor, centred, with dead borders:
 * "contain". MediaPipe's ImageToTensor with keep_aspect_ratio and
 * BORDER_ZERO. Pixels the frame does not cover map outside it and the
 * samplers write the zero value there -- that IS the letterbox. */
static inline struct hand_affine hand_affine_letterbox(uint32_t fw,
                                                       uint32_t fh,
                                                       uint32_t tw,
                                                       uint32_t th) {
    float sx = (float)tw / (float)fw, sy = (float)th / (float)fh;
    float s = sx < sy ? sx : sy; /* tensor px per frame px */
    float off_x = ((float)tw - (float)fw * s) * 0.5f;
    float off_y = ((float)th - (float)fh * s) * 0.5f;
    struct hand_affine a = {{1.0f / s, 0.0f, -off_x / s, 0.0f, 1.0f / s, -off_y / s}};
    return a;
}

/* Fill the tensor from a centred crop of the frame: "cover". No borders
 * and full tensor resolution, at the price of the long axis' ends.
 * Identical code to the letterbox, max instead of min -- worth having
 * because letterboxing 640x360 into 192x192 leaves only 108 useful rows,
 * so a hand has to be large before the detector sees it. */
static inline struct hand_affine hand_affine_square_crop(uint32_t fw,
                                                         uint32_t fh,
                                                         uint32_t tw,
                                                         uint32_t th) {
    float sx = (float)tw / (float)fw, sy = (float)th / (float)fh;
    float s = sx > sy ? sx : sy;
    float off_x = ((float)tw - (float)fw * s) * 0.5f;
    float off_y = ((float)th - (float)fh * s) * 0.5f;
    struct hand_affine a = {{1.0f / s, 0.0f, -off_x / s, 0.0f, 1.0f / s, -off_y / s}};
    return a;
}

/* A rotated square window of the frame, resampled into the whole tensor:
 * the landmark stage's crop. (cx, cy) and `side` are FRAME pixels, `rot`
 * radians counter-clockwise in a y-down image. */
static inline struct hand_affine hand_affine_roi(float cx,
                                                 float cy,
                                                 float side,
                                                 float rot,
                                                 uint32_t tw,
                                                 uint32_t th) {
    float c = cosf(rot), s = sinf(rot);
    /* d = (u/tw - 0.5, v/th - 0.5) in [-0.5, 0.5]; frame = centre + side * R * d */
    float m0 = side * c / (float)tw, m1 = -side * s / (float)th;
    float m3 = side * s / (float)tw, m4 = side * c / (float)th;
    struct hand_affine a = {{m0,
                             m1,
                             cx - 0.5f * side * c + 0.5f * side * s,
                             m3,
                             m4,
                             cy - 0.5f * side * s - 0.5f * side * c}};
    return a;
}

/* The other direction: frame px -> tensor px. Not needed by the forward
 * pipeline (see the direction note above), but it is what turns a point
 * known in the frame into the tensor coordinates that produced it, and
 * the self-test round-trips through it. */
static inline struct hand_affine hand_affine_invert(struct hand_affine a) {
    float det = a.m[0] * a.m[4] - a.m[1] * a.m[3];
    float id = det != 0.0f ? 1.0f / det : 0.0f;
    struct hand_affine r = {{a.m[4] * id,
                             -a.m[1] * id,
                             (a.m[1] * a.m[5] - a.m[4] * a.m[2]) * id,
                             -a.m[3] * id,
                             a.m[0] * id,
                             (a.m[3] * a.m[2] - a.m[0] * a.m[5]) * id}};
    return r;
}

#endif
