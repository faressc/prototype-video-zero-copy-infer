/* hand_decode.c -- MediaPipe's hand arithmetic, ported. No ORT, no
 * Dawn, no camera: two arrays of floats in, geometry out, so it can be
 * asserted against known values without a GPU or a webcam
 * (hello_hand --self-test).
 *
 * It is all cheap -- 2016 anchors times a few flops is ~50 us -- so none
 * of it belongs on the GPU. Putting the decode in a shader would buy
 * nothing and cost a second readback.
 *
 * Every constant is quoted from a graph config; hand.h names the files.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hand.h"

/* MediaPipe's hand connections, and the order they are usually drawn:
 * thumb, index, middle, ring, little, then the palm arch that closes
 * the hand (0-17). */
const unsigned char hand_bones[HAND_BONES][2] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},                 /* thumb */
    {0, 5},   {5, 6},   {6, 7},   {7, 8},                 /* index */
    {5, 9},   {9, 10},  {10, 11}, {11, 12},               /* middle */
    {9, 13},  {13, 14}, {14, 15}, {15, 16},               /* ring */
    {13, 17}, {17, 18}, {18, 19}, {19, 20},               /* little */
    {0, 17},                                              /* the palm arch */
};

static float normalize_radians(float a) {
    /* into (-pi, pi]: MediaPipe's NormalizeRadians */
    return a - 2.0f * (float)M_PI * floorf((a + (float)M_PI) / (2.0f * (float)M_PI));
}

/* ------------------------------------------------------------------ */
/* anchors                                                            */
/* ------------------------------------------------------------------ */

int hand_palm_anchors(float* xy_out, int max_anchors) {
    /* palm_detection_cpu.pbtxt: num_layers 4, strides {8,16,16,16},
     * aspect_ratios {1.0}, anchor_offset 0.5, fixed_anchor_size true.
     * interpolated_scale_aspect_ratio defaults to 1.0, which adds a
     * second anchor per aspect ratio per layer. */
    static const int strides[4] = {8, 16, 16, 16};
    const int num_layers = 4;
    const int side = HAND_DETECT_SIDE;
    int n = 0;

    int layer = 0;
    while (layer < num_layers) {
        /* SsdAnchorsCalculator merges consecutive layers of EQUAL stride
         * and stacks their anchors on the same grid -- which is why three
         * declared stride-16 layers become six anchors per cell, not
         * three separate 12x12 grids. */
        int last = layer;
        int per_cell = 0;
        while (last < num_layers && strides[last] == strides[layer]) {
            per_cell += 2; /* one per aspect ratio (1.0) + the interpolated one */
            last++;
        }
        /* ceil(input / stride), as in the calculator */
        const int fm = (side + strides[layer] - 1) / strides[layer];
        for (int y = 0; y < fm; y++) {
            for (int x = 0; x < fm; x++) {
                for (int k = 0; k < per_cell; k++) {
                    if (n >= max_anchors) { return n; }
                    /* fixed_anchor_size makes w = h = 1 for every anchor,
                     * so only the centre is worth storing -- and
                     * min_scale/max_scale never enter the arithmetic */
                    xy_out[n * 2 + 0] = ((float)x + 0.5f) / (float)fm;
                    xy_out[n * 2 + 1] = ((float)y + 0.5f) / (float)fm;
                    n++;
                }
            }
        }
        layer = last;
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* detections                                                         */
/* ------------------------------------------------------------------ */

int hand_palm_decode(const float* regressors,
                     const float* scores,
                     const float* anchors_xy,
                     int anchor_count,
                     float score_thresh,
                     struct hand_det* out,
                     int max_out) {
    /* x/y/w/h_scale are all 192, and anchor w = h = 1, so every
     * regressor is simply "pixels of the detector input" */
    const float inv = 1.0f / (float)HAND_DETECT_SIDE;
    const float side = (float)HAND_DETECT_SIDE;
    int n = 0;
    for (int i = 0; i < anchor_count && n < max_out; i++) {
        /* sigmoid_score with score_clipping_thresh 100 */
        float raw = scores[i];
        if (raw < -100.0f) { raw = -100.0f; }
        if (raw > 100.0f) { raw = 100.0f; }
        const float score = 1.0f / (1.0f + expf(-raw));
        if (score < score_thresh) { continue; }

        const float* r = regressors + (size_t)i * HAND_PALM_COORDS;
        const float ax = anchors_xy[i * 2 + 0], ay = anchors_xy[i * 2 + 1];
        struct hand_det* d = &out[n++];
        d->score = score;
        /* reverse_output_order is TRUE: index 0 is dx, index 1 is dy.
         * With it false they would be swapped, and the boxes would land
         * transposed -- plausible-looking and completely wrong. */
        d->cx = (r[0] * inv + ax) * side;
        d->cy = (r[1] * inv + ay) * side;
        d->w = r[2] * inv * side;
        d->h = r[3] * inv * side;
        for (int k = 0; k < HAND_PALM_KP; k++) {
            /* keypoint_coord_offset 4, num_values_per_keypoint 2 */
            const float* kp = r + 4 + k * 2;
            d->kp[k][0] = (kp[0] * inv + ax) * side;
            d->kp[k][1] = (kp[1] * inv + ay) * side;
        }
    }
    return n;
}

static int by_score_desc(const void* a, const void* b) {
    float x = ((const struct hand_det*)a)->score, y = ((const struct hand_det*)b)->score;
    return (x < y) - (x > y);
}

static float iou(const struct hand_det* a, const struct hand_det* b) {
    const float ax0 = a->cx - a->w * 0.5f, ax1 = a->cx + a->w * 0.5f;
    const float ay0 = a->cy - a->h * 0.5f, ay1 = a->cy + a->h * 0.5f;
    const float bx0 = b->cx - b->w * 0.5f, bx1 = b->cx + b->w * 0.5f;
    const float by0 = b->cy - b->h * 0.5f, by1 = b->cy + b->h * 0.5f;
    const float ix = fminf(ax1, bx1) - fmaxf(ax0, bx0);
    const float iy = fminf(ay1, by1) - fmaxf(ay0, by0);
    if (ix <= 0.0f || iy <= 0.0f) { return 0.0f; }
    const float inter = ix * iy;
    const float uni = a->w * a->h + b->w * b->h - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

int hand_nms(struct hand_det* d, int n, float iou_thresh, int max_keep) {
    /* non_max_suppression_calculator.cc, algorithm WEIGHTED: the survivor
     * is not the top-scoring box but the score-weighted average -- corners
     * AND keypoints -- of every box that overlaps it. A palm covers ~15
     * anchors, and which one scores highest changes with every frame of
     * sensor noise; averaging them is what makes the re-detected rect
     * (and the rotation derived from its keypoints) sit still instead of
     * hopping between anchors. The survivor keeps the top box's score. */
    enum { CAP = 64 };
    if (n <= 0) { return 0; }
    if (n > CAP) { n = CAP; }
    qsort(d, (size_t)n, sizeof *d, by_score_desc);
    struct hand_det out[CAP];
    unsigned char rem[CAP], nxt[CAP];
    int n_rem = n, n_out = 0;
    for (int i = 0; i < n; i++) { rem[i] = (unsigned char)i; }
    while (n_rem > 0 && n_out < max_keep) {
        const struct hand_det top = d[rem[0]];
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0, kp[HAND_PALM_KP][2] = {{0}};
        float total = 0.0f;
        int n_nxt = 0;
        for (int i = 0; i < n_rem; i++) {
            const struct hand_det* c = &d[rem[i]];
            if (iou(c, &top) > iou_thresh) {
                const float w = c->score;
                total += w;
                xmin += (c->cx - c->w * 0.5f) * w;
                xmax += (c->cx + c->w * 0.5f) * w;
                ymin += (c->cy - c->h * 0.5f) * w;
                ymax += (c->cy + c->h * 0.5f) * w;
                for (int k = 0; k < HAND_PALM_KP; k++) {
                    kp[k][0] += c->kp[k][0] * w;
                    kp[k][1] += c->kp[k][1] * w;
                }
            } else {
                nxt[n_nxt++] = rem[i];
            }
        }
        struct hand_det wd = top; /* total > 0: top overlaps itself */
        wd.cx = (xmin + xmax) * 0.5f / total;
        wd.cy = (ymin + ymax) * 0.5f / total;
        wd.w = (xmax - xmin) / total;
        wd.h = (ymax - ymin) / total;
        for (int k = 0; k < HAND_PALM_KP; k++) {
            wd.kp[k][0] = kp[k][0] / total;
            wd.kp[k][1] = kp[k][1] / total;
        }
        out[n_out++] = wd;
        if (n_nxt == n_rem) { break; } /* MediaPipe's no-progress guard */
        memcpy(rem, nxt, (size_t)n_nxt);
        n_rem = n_nxt;
    }
    memcpy(d, out, (size_t)n_out * sizeof *d);
    return n_out;
}

/* ------------------------------------------------------------------ */
/* the region of interest                                             */
/* ------------------------------------------------------------------ */

/* RectTransformationCalculator, in its own order: shift the centre using
 * the ORIGINAL width and height (rotated by the rect's own rotation),
 * THEN square to the long side, THEN scale. Doing it in any other order
 * moves the crop. */
static struct hand_roi rect_transform(float cx,
                                      float cy,
                                      float w,
                                      float h,
                                      float rot,
                                      float shift_y,
                                      float scale) {
    const float c = cosf(rot), s = sinf(rot);
    /* shift_x is 0 in both graphs, so its two terms drop out */
    struct hand_roi r;
    r.cx = cx - shift_y * h * s;
    r.cy = cy + shift_y * h * c;
    r.side = (w > h ? w : h) * scale; /* square_long, then scale */
    r.rot = rot;
    return r;
}

struct hand_roi hand_roi_from_detection(const struct hand_det* d) {
    /* DetectionsToRects: keypoint 0 (palm base) to keypoint 2, brought to
     * a target angle of 90 degrees. The -(y1-y0) is not a typo: MediaPipe
     * measures the angle in a y-UP frame while the image is y-down. */
    const float x0 = d->kp[0][0], y0 = d->kp[0][1];
    const float x1 = d->kp[2][0], y1 = d->kp[2][1];
    const float rot = normalize_radians((float)M_PI * 0.5f - atan2f(-(y1 - y0), x1 - x0));
    return rect_transform(d->cx, d->cy, d->w, d->h, rot, -0.5f, 2.6f);
}

struct hand_roi hand_roi_from_landmarks(const struct hand_point* lm) {
    /* hand_landmarks_to_rect_calculator.cc first REDUCES the 21 landmarks
     * to the palm: wrist, the three thumb joints, and the MCP + PIP of
     * each finger -- no fingertips, no DIP joints. Every number below is
     * relative to that 12-point list, including the rotation constants:
     * kIndexFinger/kMiddleFinger/kRingFingerPIPJoint are 4/6/8 IN THE
     * PARTIAL LIST, which is full-set 5/9/13 -- the knuckle row, not the
     * PIP row their names claim ("Indices within the partial landmarks").
     *
     * The reduction is what keeps the tracking loop stable: wrist and
     * knuckles are the hand's rigid frame, so the rect fed back for the
     * next crop cannot be inflated by curling or spread fingers -- and
     * scale 2.0 on the PALM box is what the landmark model expects to
     * see. Bounding all 21 points instead is the difference between a
     * quiet track and a ~1 Hz flicker: the box breathes with the
     * fingers, one bad landmark set doubles it, and the next crop shows
     * a hand too small to score above the 0.5 presence gate. */
    static const unsigned char palm_idx[12] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
    const float x0 = lm[0].x, y0 = lm[0].y;
    float x1 = (lm[5].x + lm[13].x) * 0.5f, y1 = (lm[5].y + lm[13].y) * 0.5f;
    x1 = (x1 + lm[9].x) * 0.5f;
    y1 = (y1 + lm[9].y) * 0.5f;
    const float rot = normalize_radians((float)M_PI * 0.5f - atan2f(-(y1 - y0), x1 - x0));

    /* the extent of the palm points measured IN THE ROTATED frame, so the
     * box hugs the palm rather than its axis-aligned shadow */
    const float c = cosf(-rot), s = sinf(-rot);
    float min_x = 1e30f, max_x = -1e30f, min_y = 1e30f, max_y = -1e30f;
    for (int i = 0; i < 12; i++) {
        const struct hand_point* p = &lm[palm_idx[i]];
        const float rx = p->x * c - p->y * s;
        const float ry = p->x * s + p->y * c;
        if (rx < min_x) { min_x = rx; }
        if (rx > max_x) { max_x = rx; }
        if (ry < min_y) { min_y = ry; }
        if (ry > max_y) { max_y = ry; }
    }
    /* the centre back in the unrotated frame */
    const float mx = (min_x + max_x) * 0.5f, my = (min_y + max_y) * 0.5f;
    const float cx = mx * cosf(rot) - my * sinf(rot);
    const float cy = mx * sinf(rot) + my * cosf(rot);
    return rect_transform(cx, cy, max_x - min_x, max_y - min_y, rot, -0.1f, 2.0f);
}

/* ------------------------------------------------------------------ */
/* landmarks                                                          */
/* ------------------------------------------------------------------ */

void hand_landmarks_decode(const float* raw63, struct hand_roi roi, struct hand_point* out) {
    /* The ROI affine maps the landmark tensor's pixels straight back to
     * whatever space the ROI was expressed in -- so decoding needs no
     * inverse. That is the whole reason hand_affine runs
     * destination -> source (see hand_affine.h). */
    struct hand_affine a =
        hand_affine_roi(roi.cx, roi.cy, roi.side, roi.rot, HAND_LANDMARK_SIDE, HAND_LANDMARK_SIDE);
    for (int i = 0; i < HAND_LANDMARKS; i++) {
        hand_affine_apply(a, raw63[i * 3 + 0], raw63[i * 3 + 1], &out[i].x, &out[i].y);
        /* normalize_z 0.4 (hand_landmark_cpu.pbtxt); relative depth in
         * units of the ROI's side, which the overlay ignores */
        out[i].z = raw63[i * 3 + 2] / (float)HAND_LANDMARK_SIDE / 0.4f * roi.side;
    }
}
