/* hand_overlay_cpu.h -- the shared overlay triangles, rasterised into
 * ordinary memory.
 *
 * The CPU twin of hand_overlay_gles.c / hand_overlay_vk.c, and the same
 * relationship scenes/raster.h has to the GPU cube: every stage the
 * hardware does for free, written out. It is deliberately NOT routed
 * through raster.h -- that one wants clip-space vertices and a depth
 * buffer, while this is forty lines of 2D edge functions and an alpha
 * blend, and the honest comparison is the short version.
 *
 * Pixel space, not NDC: the caller passes the rectangle in pixels, so
 * there is no y-up/y-down question to get wrong here (unlike the GLES
 * path, where it cost two bugs).
 */
#ifndef HELLO_WAYLAND_HAND_OVERLAY_CPU_H
#define HELLO_WAYLAND_HAND_OVERLAY_CPU_H

#include <stdint.h>

#include "hand_overlay.h"

/* Blend `n` triangle vertices into an XRGB buffer. Frame-normalised
 * (x, y) maps to pixels as (x * sx + ox, y * sy + oy), and drawing is
 * clipped to the whole buffer. */
void hand_overlay_cpu_draw(uint32_t* pixels,
                           int width,
                           int height,
                           int stride_px,
                           const struct hand_vert* v,
                           int n,
                           float sx,
                           float sy,
                           float ox,
                           float oy);

#endif
