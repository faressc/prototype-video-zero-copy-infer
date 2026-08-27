/* hand_tracker.h -- the whole pipeline behind a non-blocking seam, so a
 * render loop can ask "any hands?" without ever waiting for an answer.
 *
 * Threading, and an honest note about what it does and does not buy.
 * Everything -- both ORT sessions per provider, the Dawn device, the
 * FrameToTensor passes, the decode -- lives on ONE worker thread. The
 * render thread only ever writes a request slot and reads a result slot,
 * both lock-free. Consequences worth stating plainly:
 *
 *   - The presenter's frame callback never blocks on an inference, which
 *     is the point: a 17 ms CPU-EP cycle cannot drop frames.
 *   - Dawn and ORT are touched from exactly one thread, so their thread
 *     safety never becomes a question to reason about.
 *   - It does NOT move the WebGPU EP's GPU work off the renderer's back.
 *     Dawn has one queue per device, so ~6.5 ms of inference lands on
 *     the same queue that drives the 60 Hz present no matter which CPU
 *     thread submitted it. The thread buys CPU isolation, not GPU
 *     isolation. The stats line prints the frame rate so the difference
 *     is visible rather than argued about.
 *
 * The result the overlay draws is therefore always a frame or two old.
 * That is what MediaPipe's own graph does anyway -- the tracking ROI
 * comes from the previous frame's landmarks -- and a hand does not move
 * far in 30 ms.
 *
 * Camera buffer lifetime: the tracker reads the camera's dma-buf
 * directly (no copy), so a buffer it is working on must not go back to
 * V4L2. The scene asks hand_tracker_holds() from its cam_fence_ops.done,
 * alongside its own GPU fence -- which is why camera/cam_stream.[ch]
 * needs no changes at all: the fence was always opaque to it.
 */
#ifndef HELLO_WAYLAND_HAND_TRACKER_H
#define HELLO_WAYLAND_HAND_TRACKER_H

#include "hand.h"
#include "hand_frame.h"

/* One cached Dawn import per camera buffer. Matches camera/v4l2_camera.h's
 * CAM_MAX_BUFFERS, spelled separately so hand/ does not have to include
 * the V4L2 header just for a bound. */
enum { HAND_MAX_FRAME_IMPORTS = 8 };

struct hand_options {
    enum infer_ep ep; /* the provider in use */
    int both;         /* build cpu + webgpu up front so P can switch instantly */
    int all;          /* build every provider that comes up */
    int crop;         /* square-crop the frame instead of letterboxing it */
    int detect_only;  /* skip the landmark stage */
    int num_hands;    /* MediaPipe's num_hands: gates the detector (see --hands) */
    int detect_every; /* re-run the detector at least this often (frames) */
    int verbose;
    /* Publish a fixed synthetic hand instead of running the models. The
     * overlay's geometry, its transform into the window, its blending and
     * its mapping onto the cube's face do not depend on whether a hand is
     * in front of the camera -- and a live camera is a poor way to test
     * them, because "nothing drawn" has two explanations. This leaves
     * one. Also what the README's screenshots are taken with. */
    int overlay_test;
    const char* palm_model;
    const char* lmk_model;
};

/* --ep cpu|webgpu|cuda|both|all, --crop, --detect-only, --detect-every N,
 * --palm/--landmark PATH, --full, --verbose. HAND_EP=... supplies the
 * default, matching the camera's CAM_DEVICE / CAM_SIZE knobs. Returns 0,
 * or -1 after printing usage. */
int hand_options_parse(int argc, char** argv, struct hand_options* o);

/* What one completed cycle produced. Coordinates are NORMALISED to the
 * frame, so the overlay is resolution-free and the four backends can be
 * checked against each other. */
struct hand_results {
    int count;
    struct hand_result hand[HAND_MAX];
    uint32_t frame_seq;
    enum infer_ep ep; /* which provider produced THIS result */
    int ran_detector;
    /* Wall time of the cycle that produced this result: FrameToTensor,
     * the detector when it ran, every landmark run, and the decode --
     * frame in to landmarks out. That is the same span MediaPipe's web
     * demo reports as "inference time" (one detectForVideo() call), so
     * the number the overlay draws is comparable to the one on screen
     * there. A cycle that ran the detector is honestly ~2x the others. */
    float cycle_ms;
};

/* The hand --overlay-test draws: a plausible splayed right hand at the
 * middle of the frame, slowly waving so motion is visible too. Lives
 * here so all four backends exercise the identical geometry, and so the
 * question "is the overlay drawn correctly" can be answered without a
 * hand, a camera or a model. */
void hand_results_synthetic(struct hand_results* out, double phase);

struct hand_tracker;

struct hand_tracker* hand_tracker_create(const struct hand_options* o,
                                         uint32_t frame_w,
                                         uint32_t frame_h);
void hand_tracker_destroy(struct hand_tracker* h);

/* Non-blocking. Takes the job only if the worker is idle -- the newest
 * frame wins and older ones are simply not tracked, the same policy
 * cam_stream_pump already applies to display. Returns 1 if taken.
 * `index` is the camera buffer, remembered so hand_tracker_holds can
 * answer; `seq` is cam_stream's latest_seq, echoed back in the result. */
int hand_tracker_submit(struct hand_tracker* h,
                        const struct infer_frame* f,
                        int index,
                        uint32_t seq);

/* Non-blocking. 1 and fills `out` when a cycle finished since the last
 * call; 0 otherwise. Never allocates. */
int hand_tracker_poll(struct hand_tracker* h, struct hand_results* out);

/* 1 while the worker may still be reading camera buffer `index`. */
int hand_tracker_holds(const struct hand_tracker* h, int index);

/* The live switch. Only meaningful when more than one provider stands
 * (`both` / `all`); takes effect on the next cycle, and the in-flight
 * one is finished on the old provider rather than abandoned. */
void hand_tracker_set_ep(struct hand_tracker* h, enum infer_ep ep);
enum infer_ep hand_tracker_ep(const struct hand_tracker* h);
/* how many providers stand ready (P cycles them when > 1), and the next
 * ready one after the current -- what the P key maps to */
int hand_tracker_ready_count(const struct hand_tracker* h);
enum infer_ep hand_tracker_next_ep(const struct hand_tracker* h);

/* One line of medians for stderr and the window title, in
 * hello_inference's vocabulary. Returns the length written. */
int hand_tracker_stats_line(const struct hand_tracker* h, char* buf, size_t n);

#endif
