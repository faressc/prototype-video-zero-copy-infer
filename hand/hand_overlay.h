/* hand_overlay.h -- results to triangles, once, for everybody.
 *
 * The four camera backends draw a line four different ways, so the only
 * way to keep the overlay honest across them is to make the GEOMETRY
 * shared and leave each backend only the job of pushing vertices. That
 * turns "the skeleton lands in the same place in all four windows" from
 * a hope into a checkable claim -- the same role cam_shm plays for
 * colour in stage three, one layer up.
 *
 * Triangles, not lines. GL_LINES wider than 1 px is not portable,
 * Vulkan's wideLines is an optional feature, and a 1 px skeleton is
 * invisible on a 640x360 image blown up to a window. Expanding the
 * strokes to quads on the host costs nothing and gives all four
 * backends -- the software rasterizer included -- one code path.
 *
 * Coordinates out are NORMALISED to the frame ([0,1] x [0,1], y down),
 * which is what every backend can map: to a letterbox rectangle in the
 * window, or to the cube's face through the same uv the camera image
 * uses.
 */
#ifndef HELLO_WAYLAND_HAND_OVERLAY_H
#define HELLO_WAYLAND_HAND_OVERLAY_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hand_tracker.h"

struct hand_vert {
    float x, y;    /* normalised to the frame */
    uint32_t rgba; /* see hand_rgba: R in the LOW byte */
};

/* Pack a colour so its BYTES, in memory, run R, G, B, A -- which is what
 * a GL_UNSIGNED_BYTE (or VK_FORMAT_R8G8B8A8_UNORM) vertex attribute
 * reads. Writing the constants as 0xRRGGBBAA hex literals instead looks
 * tidy and is wrong on a little-endian machine: 0x00FF00C0 lands in
 * memory as C0 00 FF 00, so an intended opaque green arrives as
 * transparent purple. It did -- the calibration border was invisible and
 * the ROI box came out lavender, which is how this was found. A function
 * makes the order explicit and endianness a non-question. */
static inline uint32_t hand_rgba(unsigned r, unsigned g, unsigned b, unsigned a) {
    return (r & 0xFFu) | ((g & 0xFFu) << 8) | ((b & 0xFFu) << 16) | ((a & 0xFFu) << 24);
}

/* HAND_MAX hands with box and ROI (~300 verts each), the calibration
 * frame, and a readout of up to HAND_READOUT_CHARS glyphs at 15 cells x
 * 6 verts each: ~2.2k. */
enum { HAND_OVERLAY_MAX_VERTS = 4096, HAND_READOUT_CHARS = 23 };

/* Colours picked so the skeleton reads over any camera image: the bones
 * bright, the joints brighter, the detector's box dim, and one hue per
 * provider so a screenshot says which EP drew it. */
enum {
    HAND_COL_UNUSED = 0, /* the palette is #defines below: see hand_rgba */
};

#define HAND_COL_BONE_CPU hand_rgba(0x30, 0xE0, 0xFF, 0xFF) /* cyan */
#define HAND_COL_BONE_GPU hand_rgba(0xFF, 0xC0, 0x20, 0xFF) /* amber */
#define HAND_COL_BONE_CUDA hand_rgba(0x76, 0xE6, 0x20, 0xFF) /* green */
#define HAND_COL_JOINT hand_rgba(0xFF, 0xFF, 0xFF, 0xFF)
#define HAND_COL_BOX hand_rgba(0x60, 0xFF, 0x60, 0xA0)
#define HAND_COL_KP hand_rgba(0xFF, 0x60, 0xFF, 0xFF)
#define HAND_COL_CALIB hand_rgba(0x00, 0xFF, 0x00, 0xC0)
#define HAND_COL_CALIB_TL hand_rgba(0xFF, 0x20, 0x20, 0xFF)
#define HAND_COL_TEXT hand_rgba(0xFF, 0xFF, 0xFF, 0xFF)
#define HAND_COL_TEXT_BG hand_rgba(0x00, 0x00, 0x00, 0xA0)

/* aspect: frame width / height, so a stroke asked for in units of frame
 * WIDTH comes out visually square rather than stretched */
struct hand_overlay_style {
    float stroke; /* bone half-width, in fractions of the frame width */
    float joint;  /* joint half-size, likewise */
    float aspect; /* frame_w / frame_h */
    int show_box; /* the detector's box and its 7 keypoints */
    int show_roi; /* the rotated crop the landmark stage was fed */
    /* A border on the frame's exact edges plus a centre cross, and one
     * red mark in the TOP-LEFT corner. Drawn by --overlay-test, because
     * "does the overlay line up with the image" is otherwise judged by
     * eye against a hand that may not be there: with this, the border
     * either hugs the picture or it does not, and the red corner says
     * which way up the mapping is. It found two real bugs on the first
     * try -- an offset given as the rectangle's centre when the geometry
     * spans [0,1], and an inverted vertical. */
    int show_calib;
    /* A line of text drawn top-left, on the picture, in the same
     * triangles as the skeleton -- so all four backends get a readout
     * without a font renderer, an atlas or a text pipeline. Meant for
     * the inference time, the number MediaPipe's web demo puts on its
     * canvas, so the two can be compared by eye. Empty = nothing drawn.
     * Uppercase, digits and . : / only (see hand_font). */
    char readout[HAND_READOUT_CHARS + 1];
    float text; /* one font cell, in fractions of the frame width */
};

static inline struct hand_overlay_style hand_overlay_default(float aspect) {
    /* show_box on by default, and not for decoration: when the detector
     * finds a palm the landmark model then rejects (presence below
     * threshold), the box is the ONLY thing there is to draw. Without it
     * such a frame renders nothing at all and looks like a dead overlay
     * rather than a disagreement between the two models. */
    struct hand_overlay_style s = {0.006f, 0.010f, aspect, 1, 0, 0, {0}, 0.006f};
    return s;
}

/* A 3x5 bitmap font: digits, the letters a provider name and "MS" need,
 * and three marks. Rows top to bottom, three bits each, MSB left. Small
 * because the point is that the readout is drawn with the same quads as
 * everything else; anything fancier would be a text pipeline, which is
 * exactly what the window title was chosen to avoid. */
struct hand_glyph {
    char ch;
    unsigned char rows[5];
};

static const struct hand_glyph hand_font[] = {
    {'0', {7, 5, 5, 5, 7}}, {'1', {2, 6, 2, 2, 7}}, {'2', {7, 1, 7, 4, 7}}, {'3', {7, 1, 7, 1, 7}},
    {'4', {5, 5, 7, 1, 1}}, {'5', {7, 4, 7, 1, 7}}, {'6', {7, 4, 7, 5, 7}}, {'7', {7, 1, 1, 1, 1}},
    {'8', {7, 5, 7, 5, 7}}, {'9', {7, 5, 7, 1, 7}}, {'.', {0, 0, 0, 0, 2}}, {':', {0, 2, 0, 2, 0}},
    {'/', {1, 1, 2, 4, 4}}, {'A', {7, 5, 7, 5, 5}}, {'B', {6, 5, 6, 5, 6}}, {'C', {7, 4, 4, 4, 7}},
    {'D', {6, 5, 5, 5, 6}}, {'E', {7, 4, 7, 4, 7}}, {'G', {7, 4, 5, 5, 7}}, {'M', {5, 7, 7, 5, 5}},
    {'P', {7, 5, 7, 4, 4}}, {'S', {7, 4, 7, 1, 7}}, {'T', {7, 2, 2, 2, 2}}, {'U', {5, 5, 5, 5, 7}},
    {'W', {5, 5, 5, 7, 5}},
};

static inline int hand_overlay_push(struct hand_vert* out,
                                    int n,
                                    int max,
                                    float x,
                                    float y,
                                    uint32_t c) {
    if (n < max) {
        out[n].x = x;
        out[n].y = y;
        out[n].rgba = c;
    }
    return n + 1;
}

/* A quad from a to b, `hw` wide, as two triangles. The width is applied
 * perpendicular to the stroke, corrected for aspect so it does not
 * thicken as the stroke turns. */
static inline int hand_overlay_stroke(struct hand_vert* out,
                                      int n,
                                      int max,
                                      float ax,
                                      float ay,
                                      float bx,
                                      float by,
                                      float hw,
                                      float aspect,
                                      uint32_t c) {
    /* work in a square space so the perpendicular is a real perpendicular */
    const float dy = (by - ay) / aspect, dx = bx - ax;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-6f) { return n; }
    const float px = -dy / len * hw, py = dx / len * hw * aspect;
    n = hand_overlay_push(out, n, max, ax + px, ay + py, c);
    n = hand_overlay_push(out, n, max, bx + px, by + py, c);
    n = hand_overlay_push(out, n, max, bx - px, by - py, c);
    n = hand_overlay_push(out, n, max, ax + px, ay + py, c);
    n = hand_overlay_push(out, n, max, bx - px, by - py, c);
    n = hand_overlay_push(out, n, max, ax - px, ay - py, c);
    return n;
}

static inline int hand_overlay_quad(struct hand_vert* out,
                                    int n,
                                    int max,
                                    float cx,
                                    float cy,
                                    float hw,
                                    float aspect,
                                    uint32_t c) {
    const float hy = hw * aspect;
    n = hand_overlay_push(out, n, max, cx - hw, cy - hy, c);
    n = hand_overlay_push(out, n, max, cx + hw, cy - hy, c);
    n = hand_overlay_push(out, n, max, cx + hw, cy + hy, c);
    n = hand_overlay_push(out, n, max, cx - hw, cy - hy, c);
    n = hand_overlay_push(out, n, max, cx + hw, cy + hy, c);
    n = hand_overlay_push(out, n, max, cx - hw, cy + hy, c);
    return n;
}

/* An axis-aligned rectangle, corners given directly in frame space. */
static inline int hand_overlay_rect(struct hand_vert* out,
                                    int n,
                                    int max,
                                    float x0,
                                    float y0,
                                    float x1,
                                    float y1,
                                    uint32_t c) {
    n = hand_overlay_push(out, n, max, x0, y0, c);
    n = hand_overlay_push(out, n, max, x1, y0, c);
    n = hand_overlay_push(out, n, max, x1, y1, c);
    n = hand_overlay_push(out, n, max, x0, y0, c);
    n = hand_overlay_push(out, n, max, x1, y1, c);
    n = hand_overlay_push(out, n, max, x0, y1, c);
    return n;
}

/* `s` with its top-left at (x, y), one lit font cell per `cell`-sized
 * quad, glyphs 3 cells wide on a 4-cell pitch. Unknown characters are
 * skipped, so a stray lowercase letter costs nothing but its space. */
static inline int hand_overlay_text(struct hand_vert* out,
                                    int n,
                                    int max,
                                    float x,
                                    float y,
                                    float cell,
                                    float aspect,
                                    const char* s,
                                    uint32_t c) {
    const float cy = cell * aspect;
    for (; *s; s++, x += 4.0f * cell) {
        const unsigned char* rows = NULL;
        for (size_t i = 0; i < sizeof hand_font / sizeof hand_font[0]; i++) {
            if (hand_font[i].ch == *s) {
                rows = hand_font[i].rows;
                break;
            }
        }
        if (!rows) { continue; }
        for (int r = 0; r < 5; r++) {
            for (int col = 0; col < 3; col++) {
                if (rows[r] & (4u >> col)) {
                    n = hand_overlay_quad(out,
                                          n,
                                          max,
                                          x + ((float)col + 0.5f) * cell,
                                          y + ((float)r + 0.5f) * cy,
                                          cell * 0.5f,
                                          aspect,
                                          c);
                }
            }
        }
    }
    return n;
}

/* Everything worth drawing about one set of results, as triangles.
 * Returns the vertex count written (clamped to `max`, and the count it
 * WOULD have needed is not reported -- HAND_OVERLAY_MAX_VERTS is sized
 * for HAND_MAX hands with the box and ROI on, plus the readout). */
static inline int hand_overlay_build(const struct hand_results* r,
                                     struct hand_overlay_style st,
                                     struct hand_vert* out,
                                     int max) {
    int n = 0;
    if (st.readout[0]) {
        /* top-left, on a dark backing so the digits read over any
         * picture. Drawn whether or not there is a hand: the time of a
         * cycle that found nothing is still a number worth seeing. */
        const float cell = st.text, pad = cell;
        const float x0 = 0.012f, y0 = 0.012f * st.aspect;
        const float w = (float)strlen(st.readout) * 4.0f * cell - cell + 2.0f * pad;
        const float h = (5.0f * cell + 2.0f * pad) * st.aspect;
        n = hand_overlay_rect(out, n, max, x0, y0, x0 + w, y0 + h, HAND_COL_TEXT_BG);
        n = hand_overlay_text(out,
                              n,
                              max,
                              x0 + pad,
                              y0 + pad * st.aspect,
                              cell,
                              st.aspect,
                              st.readout,
                              HAND_COL_TEXT);
    }
    if (st.show_calib) {
        /* the frame's exact edges, inset by half the stroke so the whole
         * quad is inside [0,1] and a correct mapping shows all four sides */
        const float w = st.stroke * 0.5f, e = w, f = 1.0f - w;
        const uint32_t g = HAND_COL_CALIB;
        n = hand_overlay_stroke(out, n, max, e, e, f, e, w, st.aspect, g);
        n = hand_overlay_stroke(out, n, max, f, e, f, f, w, st.aspect, g);
        n = hand_overlay_stroke(out, n, max, f, f, e, f, w, st.aspect, g);
        n = hand_overlay_stroke(out, n, max, e, f, e, e, w, st.aspect, g);
        /* the centre, so a half-rectangle offset error is obvious */
        n = hand_overlay_stroke(out, n, max, 0.45f, 0.5f, 0.55f, 0.5f, w, st.aspect, g);
        n = hand_overlay_stroke(out, n, max, 0.5f, 0.45f, 0.5f, 0.55f, w, st.aspect, g);
        /* and one asymmetric mark: TOP-LEFT, in red. If it shows up
         * bottom-left the vertical is inverted, which is exactly the bug
         * a symmetric border cannot reveal. */
        n = hand_overlay_quad(out, n, max, 0.06f, 0.06f, 0.02f, st.aspect, HAND_COL_CALIB_TL);
    }
    if (!r || r->count <= 0) { return n; }
    const uint32_t bone = r->ep == INFER_EP_WEBGPU ? HAND_COL_BONE_GPU
                          : r->ep == INFER_EP_CUDA ? HAND_COL_BONE_CUDA
                                                   : HAND_COL_BONE_CPU;

    for (int hi = 0; hi < r->count && hi < HAND_MAX; hi++) {
        const struct hand_result* h = &r->hand[hi];

        if (st.show_roi && h->roi.side > 0.0f) {
            /* the rotated square the landmark model actually saw */
            const float c = cosf(h->roi.rot), s = sinf(h->roi.rot);
            const float hs = h->roi.side * 0.5f;
            float cx[4], cy[4];
            const float sx[4] = {-hs, hs, hs, -hs}, sy[4] = {-hs, -hs, hs, hs};
            for (int i = 0; i < 4; i++) {
                cx[i] = h->roi.cx + sx[i] * c - sy[i] * s;
                cy[i] = h->roi.cy + (sx[i] * s + sy[i] * c) * st.aspect;
            }
            for (int i = 0; i < 4; i++) {
                n = hand_overlay_stroke(out,
                                        n,
                                        max,
                                        cx[i],
                                        cy[i],
                                        cx[(i + 1) & 3],
                                        cy[(i + 1) & 3],
                                        st.stroke * 0.5f,
                                        st.aspect,
                                        HAND_COL_BOX);
            }
        }
        /* landmarks: a hand the detector found but the landmark model
         * rejected (presence below threshold) has none */
        int have_lm = 0;
        for (int i = 0; i < HAND_LANDMARKS; i++) {
            if (h->lm[i].x != 0.0f || h->lm[i].y != 0.0f) {
                have_lm = 1;
                break;
            }
        }
        /* The detector's 7 palm keypoints are a FALLBACK, not an
         * addition: they exist only on the cycles the detector ran, so
         * drawing them alongside the skeleton makes them blink at the
         * re-anchor period while the hand sits still. Shown only when
         * there is no skeleton to show instead. */
        if (st.show_box && h->from_detector && !have_lm) {
            for (int k = 0; k < HAND_PALM_KP; k++) {
                n = hand_overlay_quad(out,
                                      n,
                                      max,
                                      h->palm_kp[k][0],
                                      h->palm_kp[k][1],
                                      st.joint * 0.6f,
                                      st.aspect,
                                      HAND_COL_KP);
            }
        }
        if (!have_lm) { continue; }
        for (int b = 0; b < HAND_BONES; b++) {
            const struct hand_point* a = &h->lm[hand_bones[b][0]];
            const struct hand_point* z = &h->lm[hand_bones[b][1]];
            n = hand_overlay_stroke(out,
                                    n,
                                    max,
                                    a->x,
                                    a->y,
                                    z->x,
                                    z->y,
                                    st.stroke,
                                    st.aspect,
                                    bone);
        }
        for (int i = 0; i < HAND_LANDMARKS; i++) {
            /* fingertips a touch larger, so the hand reads at a glance */
            const int tip = i == 4 || i == 8 || i == 12 || i == 16 || i == 20;
            n = hand_overlay_quad(out,
                                  n,
                                  max,
                                  h->lm[i].x,
                                  h->lm[i].y,
                                  st.joint * (tip ? 1.0f : 0.65f),
                                  st.aspect,
                                  HAND_COL_JOINT);
        }
    }
    return n < max ? n : max;
}

#endif
