/* hand_overlay_gles.h -- the GLES half of the overlay. Separate from
 * hand_overlay.h because that one is backend-free geometry and this one
 * needs a GL context; a scene includes whichever matches its presenter.
 */
#ifndef HELLO_WAYLAND_HAND_OVERLAY_GLES_H
#define HELLO_WAYLAND_HAND_OVERLAY_GLES_H

#include <stddef.h>

#include <GLES2/gl2.h>

#include "hand_overlay.h"

struct hand_overlay_gles {
    GLuint program, vbo;
    GLint a_pos, a_col, u_rect;
};

void hand_overlay_gles_init(struct hand_overlay_gles* g);
void hand_overlay_gles_fini(struct hand_overlay_gles* g);

/* Draw `n` triangle vertices. The transform is entirely the caller's:
 * gl_Position.xy = a_pos * (sx, sy) + (ox, oy), where a_pos is
 * frame-normalised [0,1]^2 with y DOWN and the result is NDC.
 *
 * Deriving it, because getting it wrong is invisible until it is
 * obviously wrong. a_pos spans [0,1], NOT [-1,1], so the offset is the
 * rectangle's near EDGE, not its centre -- offsetting by the centre
 * shifts everything right and down by half the rectangle. And the
 * vertical direction is not a free choice: it has to match wherever the
 * camera IMAGE put its first row, which is decided by the `u_flip`
 * draw_fullscreen was called with, not by the target's name.
 *
 * hand_overlay_gles_rect below works both out from the numbers the
 * caller already has, so no caller has to re-derive it. */
void hand_overlay_gles_draw(struct hand_overlay_gles* g,
                            const struct hand_vert* v,
                            int n,
                            float sx,
                            float sy,
                            float ox,
                            float oy);

/* Frame-normalised [0,1]^2 -> NDC, for an image occupying window pixels
 * x in [x0, x0+w), y in [y0, y0+h) of a `vw` x `vh` viewport.
 *
 * `image_row0_at_ndc_bottom` says where the camera image's FIRST row
 * ended up: 1 when draw_fullscreen was called with flip 0 (the y-down
 * targets -- the dmabuf swapchain's FBOs and the effect chain's
 * scratch), 0 when it was called with flip 1 (a y-up window surface).
 * That single bit is the whole vertical question. */
static inline void hand_overlay_gles_rect(int x0,
                                          int y0,
                                          int w,
                                          int h,
                                          int vw,
                                          int vh,
                                          int image_row0_at_ndc_bottom,
                                          float* sx,
                                          float* sy,
                                          float* ox,
                                          float* oy) {
    *sx = 2.0f * (float)w / (float)vw;
    *ox = 2.0f * (float)x0 / (float)vw - 1.0f; /* the LEFT edge */
    if (image_row0_at_ndc_bottom) {
        /* frame y grows downward and so does NDC y here: v = 0 at the
         * bottom of the rectangle */
        *sy = 2.0f * (float)h / (float)vh;
        *oy = 2.0f * (float)y0 / (float)vh - 1.0f;
    } else {
        /* v = 0 at the TOP: the scale is negative and the offset is the
         * far edge */
        *sy = -2.0f * (float)h / (float)vh;
        *oy = 2.0f * (float)(y0 + h) / (float)vh - 1.0f;
    }
}

#endif
