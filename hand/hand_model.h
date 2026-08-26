/* hand_model.h -- one ONNX Runtime session plus the tensors bound to it,
 * for either execution provider.
 *
 * This is the unit both the headless harness and the windowed scenes
 * share, and it exists mostly to enforce one rule:
 *
 *   the tensors handed to the engine are created ONCE and never rotate.
 *
 * That is not tidiness. The WebGPU EP runs with graph capture, which
 * records the bind groups of the buffers it captured on and replays
 * them; rebinding a different buffer later means the replay reads the
 * captured one, and every other inference silently returns the previous
 * frame's answer (README §20 spent two sharpenings of hello_inference
 * finding that out). So the camera's six rotating buffers stay on the
 * PRODUCER side of the FrameToTensor pass, which writes the one fixed
 * input tensor this object owns -- exactly the doc's producer-side
 * multi-buffering, with a camera in the producer's place.
 *
 * Outputs go through the registry rather than a hand-rolled readback, so
 * the cost class is the one hello_inference prints: `identity` (a
 * hand-over -- the CPU EP writes host memory directly) or `map_read` (a
 * HOST_COPY) for the WebGPU EP.
 */
#ifndef HELLO_WAYLAND_HAND_MODEL_H
#define HELLO_WAYLAND_HAND_MODEL_H

#include "hand.h"
#include "hand_frame.h"

enum { HAND_MODEL_MAX_OUTPUTS = 8 };

struct hand_model;

/* `side` is the square input the model wants (192 detector, 224
 * landmark); it is checked against what the model reports. `border` is
 * how the model's graph says to fill outside the frame, and the two hand
 * models genuinely differ -- see enum hand_border. */
struct hand_model* hand_model_create(struct infer_ctx* ctx,
                                     const struct infer_registry* reg,
                                     enum infer_ep ep,
                                     const char* model_path,
                                     uint32_t side,
                                     enum hand_border border);
void hand_model_destroy(struct hand_model* m);

/* Fill the engine's input from a frame through `a`. Which one to call is
 * decided by the model's EP: the CPU EP wants the frame's mmap, the
 * WebGPU EP wants an import inside an open access bracket. */
int hand_model_feed_cpu(struct hand_model* m, const struct infer_frame* f, struct hand_affine a);
int hand_model_feed_wgpu(struct hand_model* m,
                         struct infer_wgpu_frame_import* im,
                         const struct infer_frame* f,
                         struct hand_affine a);

/* Bind and submit. Returns when the work is SUBMITTED -- for the WebGPU
 * EP the outputs are not readable yet, which is the whole point of the
 * next call being separate. */
int hand_model_run(struct hand_model* m);

/* Move the outputs into host memory and wait for them: the consumer's
 * ready token, taken the blocking way. The windowed WebGPU path wants a
 * polled version instead and will not use this. */
int hand_model_sync(struct hand_model* m);

/* Output i as host floats, valid after a successful sync. */
const float* hand_model_output(const struct hand_model* m, int i);
size_t hand_model_output_count(const struct hand_model* m);
size_t hand_model_output_elements(const struct hand_model* m, int i);
enum infer_ep hand_model_ep(const struct hand_model* m);
/* What the out-edge cost, for the timing line: "identity" or "map_read". */
const char* hand_model_out_edge(const struct hand_model* m);

#endif
