/* hand.h -- stage five's vocabulary: what a detected hand is, and the
 * pure arithmetic that turns two models' output tensors into one.
 *
 * The pipeline is MediaPipe's, ported rather than invented, so every
 * constant here is quoted from a graph config and the file says which.
 * Verified 2026-08-26 against
 *   mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
 *   mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
 *   mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt
 *   mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt
 *   mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc
 * because a wrong one of them does not fail -- it quietly finds no hand,
 * or puts the crop on the forearm.
 *
 * Coordinate spaces, kept straight by never mixing them in one struct:
 *   detector space  pixels of the 192x192 palm-detector input. Isotropic
 *                   w.r.t. the frame (the fit affine is a uniform scale
 *                   plus a translation), so a square box stays square
 *                   and a rotation is a rotation. ALL detection and ROI
 *                   arithmetic happens here.
 *   frame space     pixels of the camera frame. One hand_affine_apply
 *                   away, and the last thing decoding does.
 *   normalised      frame space divided by (width, height). What the
 *                   overlay wants, because it is resolution-free.
 */
#ifndef HELLO_WAYLAND_HAND_H
#define HELLO_WAYLAND_HAND_H

#include "hand_affine.h"

enum {
    HAND_LANDMARKS = 21,
    HAND_PALM_KP = 7, /* num_keypoints */
    HAND_MAX = 2,     /* hands tracked at once */
    HAND_BONES = 21,
};

/* palm_detection_cpu.pbtxt: input 192x192, num_boxes 2016, num_coords 18 */
enum {
    HAND_DETECT_SIDE = 192,
    HAND_PALM_ANCHORS = 2016,
    HAND_PALM_COORDS = 18,
};
/* hand_landmark_cpu.pbtxt: ImageToTensor 224x224, num_landmarks 21 */
enum {
    HAND_LANDMARK_SIDE = 224,
};

/* Both models take pixels in [0,1]: palm_detection_cpu.pbtxt and
 * hand_landmark_cpu.pbtxt both say output_tensor_float_range { min: 0.0
 * max: 1.0 }. Checked rather than assumed -- the wrong range yields
 * near-zero scores, not an error. */
#define HAND_NORM_RANGE ((struct hand_norm){0.0f, 1.0f})

/* ------------------------------------------------------------------ */
/* the detector                                                       */
/* ------------------------------------------------------------------ */

/* One palm, in DETECTOR pixel space. */
struct hand_det {
    float score;
    float cx, cy, w, h;        /* the box */
    float kp[HAND_PALM_KP][2]; /* 7 palm keypoints */
};

/* A rotated square window, in whatever space the producer says. */
struct hand_roi {
    float cx, cy, side, rot; /* rot in radians */
};

/* The SSD anchor grid. Only the CENTRES matter: fixed_anchor_size is
 * true, so every anchor has w = h = 1 and min_scale/max_scale never
 * enter the arithmetic -- which is why this is twenty lines rather than
 * sixty. Layers of equal stride are merged, so the four declared layers
 * (strides 8,16,16,16) become 24*24*2 = 1152 plus 12*12*6 = 864 = 2016,
 * exactly the model's num_boxes. Writes 2 floats per anchor, NORMALISED
 * to [0,1]. Returns the count. */
int hand_palm_anchors(float* xy_out, int max_anchors);

/* raw regressors [2016][18] + raw scores [2016] -> detections in
 * DETECTOR pixel space, score-thresholded, unsorted. Returns the count.
 *
 * The subtlety that silently ruins everything if missed:
 * reverse_output_order is TRUE, so regressor 0 is dx and 1 is dy -- not
 * dy, dx. */
int hand_palm_decode(const float* regressors,
                     const float* scores,
                     const float* anchors_xy,
                     int anchor_count,
                     float score_thresh,
                     struct hand_det* out,
                     int max_out);

/* MediaPipe's WEIGHTED NMS (non_max_suppression_calculator.cc), in
 * place, highest score first. Each surviving detection is the
 * score-weighted average -- box and keypoints -- of every detection
 * overlapping it above iou_thresh, carrying the top score. Returns the
 * number kept (at most max_keep, which stands in for the tracking
 * graph's ClipDetectionVectorSizeCalculator). */
int hand_nms(struct hand_det* d, int n, float iou_thresh, int max_keep);

/* palm_detection_detection_to_roi.pbtxt: DetectionsToRects with
 * rotation_vector_start/end_keypoint_index 0 and 2 and
 * rotation_vector_target_angle_degrees 90, then RectTransformation with
 * scale 2.6, shift_y -0.5, square_long. Order matters: the shift uses
 * the ORIGINAL box size, then it is squared, then scaled. */
struct hand_roi hand_roi_from_detection(const struct hand_det* d);

/* ------------------------------------------------------------------ */
/* the landmark stage                                                 */
/* ------------------------------------------------------------------ */

struct hand_point {
    float x, y, z;
};

/* Identity [1,63] (21 * xyz in 224-input pixels) -> `out` in the space
 * `roi` is expressed in, by pushing each point through the ROI affine.
 * normalize_z 0.4 per hand_landmark_cpu.pbtxt; z stays relative and the
 * overlay ignores it. */
void hand_landmarks_decode(const float* raw63,
                           struct hand_roi roi,
                           struct hand_point* out /* HAND_LANDMARKS */);

/* hand_landmark_landmarks_to_roi.pbtxt + hand_landmarks_to_rect_calculator.cc:
 * the next frame's ROI from this frame's landmarks -- the whole reason
 * steady-state tracking costs one inference instead of two.
 *
 * Both the box and the rotation use only the PALM's 12 landmarks
 * (wrist, thumb chain, and each finger's MCP + PIP -- no tips, no DIPs):
 * the calculator reduces the 21 to that subset first, and its rotation
 * constants 4/6/8 index the SUBSET, i.e. full-set 5/9/13, the knuckle
 * row (weights 1/4, 1/2, 1/4), target angle 90 deg. The box is the
 * subset's axis-aligned extent IN THE ROTATED frame, then
 * RectTransformation with scale 2.0, shift_y -0.1, square_long.
 * Bounding all 21 points instead is not a smaller approximation, it is
 * the flicker: the fed-back ROI then breathes with the fingers and one
 * wide landmark set doubles it. */
struct hand_roi hand_roi_from_landmarks(const struct hand_point* lm);

/* The 21-point skeleton, for the overlay and for eyeballing a dump. */
extern const unsigned char hand_bones[HAND_BONES][2];

/* ------------------------------------------------------------------ */
/* one tracked hand, ready to draw                                    */
/* ------------------------------------------------------------------ */

struct hand_result {
    struct hand_point lm[HAND_LANDMARKS]; /* normalised to the frame */
    float palm_kp[HAND_PALM_KP][2];       /* normalised; detector frames only */
    struct hand_roi roi;                  /* normalised */
    float score, presence, handedness;
    int from_detector; /* the detector ran for this result */
};

#endif
