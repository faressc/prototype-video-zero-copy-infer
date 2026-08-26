/* hand_overlay_cpu.c -- see hand_overlay_cpu.h. */
#include "hand_overlay_cpu.h"

static float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

/* src over dst, with the source's alpha. The target is XRGB
 * (0x00RRGGBB, as scene_cam_shm.c's px_pack makes it); the vertex colour
 * is hand_rgba's packing, R in the low byte. */
static void blend(uint32_t* dst, uint32_t rgba) {
    const unsigned a = (rgba >> 24) & 0xFFu;
    if (a == 0) { return; }
    const unsigned sr = rgba & 0xFFu, sg = (rgba >> 8) & 0xFFu, sb = (rgba >> 16) & 0xFFu;
    if (a == 0xFFu) {
        *dst = (sr << 16) | (sg << 8) | sb;
        return;
    }
    const unsigned d = *dst;
    const unsigned dr = (d >> 16) & 0xFFu, dg = (d >> 8) & 0xFFu, db = d & 0xFFu;
    const unsigned ia = 255u - a;
    const unsigned r = (sr * a + dr * ia + 127u) / 255u;
    const unsigned g = (sg * a + dg * ia + 127u) / 255u;
    const unsigned b = (sb * a + db * ia + 127u) / 255u;
    *dst = (r << 16) | (g << 8) | b;
}

void hand_overlay_cpu_draw(uint32_t* pixels,
                           int width,
                           int height,
                           int stride_px,
                           const struct hand_vert* v,
                           int n,
                           float sx,
                           float sy,
                           float ox,
                           float oy) {
    for (int t = 0; t + 2 < n; t += 3) {
        float x[3], y[3];
        for (int k = 0; k < 3; k++) {
            x[k] = v[t + k].x * sx + ox;
            y[k] = v[t + k].y * sy + oy;
        }
        /* the overlay's quads are emitted with a consistent winding, but
         * a stroke that turns can flip it, so take the absolute area and
         * test both signs rather than culling half the geometry */
        float area = edge(x[0], y[0], x[1], y[1], x[2], y[2]);
        if (area == 0.0f) { continue; }
        const float s = area < 0.0f ? -1.0f : 1.0f;

        int x0 = (int)floorf(fminf(fminf(x[0], x[1]), x[2]));
        int x1 = (int)ceilf(fmaxf(fmaxf(x[0], x[1]), x[2]));
        int y0 = (int)floorf(fminf(fminf(y[0], y[1]), y[2]));
        int y1 = (int)ceilf(fmaxf(fmaxf(y[0], y[1]), y[2]));
        if (x0 < 0) { x0 = 0; }
        if (y0 < 0) { y0 = 0; }
        if (x1 > width) { x1 = width; }
        if (y1 > height) { y1 = height; }

        for (int py = y0; py < y1; py++) {
            uint32_t* row = pixels + (size_t)py * (size_t)stride_px;
            const float fy = (float)py + 0.5f;
            for (int px = x0; px < x1; px++) {
                const float fx = (float)px + 0.5f;
                const float e0 = s * edge(x[0], y[0], x[1], y[1], fx, fy);
                const float e1 = s * edge(x[1], y[1], x[2], y[2], fx, fy);
                const float e2 = s * edge(x[2], y[2], x[0], y[0], fx, fy);
                if (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) {
                    blend(&row[px], v[t].rgba);
                }
            }
        }
    }
}
