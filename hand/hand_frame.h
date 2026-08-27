/* hand_frame.h -- FrameToTensor: the pass README §20 named and stage
 * five owes.
 *
 * A camera frame is pixels with a colour space; a model input is
 * normalised NHWC floats. Turning one into the other is not an edge (an
 * edge preserves meaning and changes memory) -- it is a pass, and it
 * exists once per execution provider:
 *
 *   CPU EP      hand_frame_to_tensor_cpu:  a C loop over the frame's mmap
 *   WebGPU EP   hand_frame_to_tensor_wgpu: one compute pass over the
 *               frame's dma-buf, imported into the shared Dawn device,
 *               writing the WGPUBuffer ORT has bound
 *   CUDA EP     hand_frame_to_tensor_vk:   the same pass on the headless
 *               Vulkan device (CUDA imports no dma-buf, Vulkan does),
 *               writing either a VK tensor's buffer or the opaque-fd
 *               buffer the CUDA EP reads through its mapping
 *
 * All are handed the same struct hand_affine and the same YUV
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

/* The Vulkan twin, for the CUDA EP (or any consumer of a packed-float
 * VkBuffer). `im` from infer_vk_frame_import_create; `dst` is whichever
 * STORAGE buffer the caller owns -- a VK tensor's alias, or the shared
 * opaque-fd buffer whose CUDA mapping the EP has bound. Submits one
 * compute pass and returns; completion is the pass's fence
 * (hand_frame_vk_wait) or the optional `signal` semaphore the CUDA
 * stream waits on. `release_fd` (may be NULL) receives the submit's
 * completion as a sync file: the "done reading this frame" fence.
 * The next apply on the same pass waits the previous one's fence, so
 * one pass object serialises itself. */
struct hand_frame_pass_vk;
struct hand_frame_pass_vk* hand_frame_pass_vk_create(struct infer_ctx_vk* c);
void hand_frame_pass_vk_destroy(struct infer_ctx_vk* c, struct hand_frame_pass_vk* p);
int hand_frame_to_tensor_vk(struct infer_ctx_vk* c,
                            struct hand_frame_pass_vk* p,
                            struct infer_vk_frame_import* im,
                            const struct infer_frame* f,
                            struct hand_affine a,
                            struct hand_norm n,
                            enum hand_border border,
                            uint32_t tw,
                            uint32_t th,
                            VkBuffer dst,
                            uint64_t dst_bytes,
                            VkSemaphore signal,
                            int* release_fd);
/* The same pass writing a VK TENSOR for an EXTERNAL reader -- Dawn,
 * through the registry's vk -> wgpu row: the WebGPU EP's feed on a
 * driver that refuses Dawn's own import of the frame. The destination
 * is the tensor's buffer alias; around the dispatch its image and
 * buffer are acquired from and released to VK_QUEUE_FAMILY_EXTERNAL the
 * way infer_vk_gen does it, the write waits the tensor's `released`
 * fence (the reader's "done"), and the tensor's `ready` becomes the
 * submit's completion. The tensor's own fence is the one submitted on,
 * so its readback and release wait for this pass like for any writer.
 * tw/th are the tensor's dims. One pass object per tensor. */
int hand_frame_to_tensor_vk_tensor(struct infer_ctx_vk* c,
                                   struct hand_frame_pass_vk* p,
                                   struct infer_vk_frame_import* im,
                                   const struct infer_frame* f,
                                   struct hand_affine a,
                                   struct hand_norm n,
                                   enum hand_border border,
                                   struct infer_tensor* t,
                                   int* release_fd);
/* block until the last submitted pass completed */
int hand_frame_vk_wait(struct infer_ctx_vk* c, struct hand_frame_pass_vk* p);

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
