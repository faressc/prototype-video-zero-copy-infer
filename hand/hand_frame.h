/* hand_frame.h -- FrameToTensor: the pass README §20 named and stage
 * five owes.
 *
 * A camera frame is pixels with a colour space; a model input is
 * normalised NHWC floats. Turning one into the other is not an edge (an
 * edge preserves meaning and changes memory) -- it is a pass, and it
 * exists twice, once per execution provider:
 *
 *   CPU EP      hand_frame_to_tensor_cpu:  a C loop over the frame's mmap
 *   WebGPU EP   hand_frame_to_tensor_wgpu: one compute pass over the
 *               frame's dma-buf, imported into the shared Dawn device,
 *               writing the WGPUBuffer ORT has bound
 *
 * Both are handed the same struct hand_affine and the same YUV
 * coefficients (camera/nv12_convert.h's nv12_yuv_params), so they
 * compute the same tensor. hello_hand --compare-frame-to-tensor asserts
 * it, which is the check that makes an EP disagreement downstream
 * attributable.
 */
#ifndef HELLO_WAYLAND_HAND_FRAME_H
#define HELLO_WAYLAND_HAND_FRAME_H

#include "hand_affine.h"
#include "infer.h"

/* What the model wants its pixels scaled into: [0,1] or [-1,1]. Which
 * one a MediaPipe model expects is in its graph's
 * output_tensor_float_range, and getting it wrong yields near-zero
 * detection scores rather than an error -- so it is a field, not a
 * constant (see hello_hand --range). */
struct hand_norm {
    float lo, hi;
};

/* What to sample where the affine lands outside the frame. Not a detail:
 * MediaPipe sets it PER MODEL and the two hand graphs disagree, so
 * getting it from one global choice is wrong for one of them.
 *
 *   detector (192x192)  palm_detection_cpu.pbtxt says border_mode:
 *                       BORDER_ZERO explicitly -- the letterbox bars are
 *                       meant to be black, that is what letterboxing IS.
 *   landmark (224x224)  hand_landmark_cpu.pbtxt omits border_mode, and
 *                       ImageToTensorCalculator's documented default is
 *                       BORDER_REPLICATE.
 *
 * Using ZERO for the landmark crop costs nothing while the hand is small
 * and far away -- the 2.6x ROI stays inside the frame, so no border is
 * ever sampled. Bring the hand CLOSER and the ROI grows past the frame
 * edge, a black wedge appears in a crop the model has never seen like
 * that in training, and presence collapses to ~0.1 while the detector is
 * still reporting 0.95. That is a flicker that arrives with proximity
 * and nothing else, which is a strange enough signature to be worth
 * naming here. */
enum hand_border {
    HAND_BORDER_ZERO,     /* outside the frame reads as the normalised zero */
    HAND_BORDER_REPLICATE /* outside the frame reads as the nearest edge pixel */
};

/* Fill a tensor in INFER_DOMAIN_CPU. `t` must already be allocated with
 * a [1, th, tw, 3] float32 descriptor. The frame must carry a `map`. */
int hand_frame_to_tensor_cpu(const struct infer_frame* f,
                             struct hand_affine a,
                             struct hand_norm n,
                             enum hand_border border,
                             struct infer_tensor* t);

/* The WebGPU twin. `im` must be inside an access bracket
 * (infer_wgpu_frame_begin) and `t` must be a tensor in
 * INFER_DOMAIN_WGPU. Records one compute pass on the shared queue --
 * the same queue ORT replays on, so no fence is needed between them --
 * and returns once submitted, not once complete. */
struct hand_frame_pass;
struct hand_frame_pass* hand_frame_pass_create(struct infer_ctx_wgpu* c);
void hand_frame_pass_destroy(struct hand_frame_pass* p);
int hand_frame_to_tensor_wgpu(struct infer_ctx_wgpu* c,
                              struct hand_frame_pass* p,
                              struct infer_wgpu_frame_import* im,
                              const struct infer_frame* f,
                              struct hand_affine a,
                              struct hand_norm n,
                              enum hand_border border,
                              struct infer_tensor* t);

/* The descriptor a [1, th, tw, 3] float32 image tensor needs. Handy
 * because both stages want one and it must match what the model
 * reported. */
void hand_frame_tensor_desc(uint32_t tw, uint32_t th, struct infer_desc* d);

/* Build the import descriptor for a frame, applying the one driver
 * workaround this needs: a packed YUYV plane is imported as ABGR8888 at
 * half width (texel = Y0,U,Y1,V), because not every driver imports
 * DRM_FORMAT_YUYV -- the same substitution scene_cam_gles.c makes for
 * EGL. The returned frame is for the IMPORTER only; everything else
 * keeps the honest fourcc. */
struct infer_frame hand_frame_import_desc(const struct infer_frame* f);

/* One V4L2 buffer as a frame. `map` is filled only if the buffer has
 * already been mapped (camera_map is not called here: the WebGPU path
 * never needs it and mapping it would be a pointless page-table walk).
 * Lives here rather than in scene glue because both the scenes and
 * hello_hand need it, and hand already shares camera's colour
 * coefficients. */
struct camera;
void hand_frame_from_camera(struct infer_frame* f, const struct camera* cam, uint32_t index);

#endif
