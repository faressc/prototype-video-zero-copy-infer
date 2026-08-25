/* cam_stream.h -- frame pacing + buffer lifetime on top of v4l2_camera,
 * shared by all camera scenes.
 *
 * Two facts drive the design:
 *
 *  1. The camera runs at its own rate (30 fps) and the display at its
 *     own (60+); frames arrive in bursts and the compositor consumes
 *     at vsync. Only the NEWEST captured frame is worth showing --
 *     older ones the GPU never touched go straight back to the driver.
 *
 *  2. V4L2 ignores dmabuf fences. A buffer must not be requeued while
 *     the GPU may still be reading it, or the ISP scribbles over a
 *     texture mid-frame. So a rendered buffer is retired only after a
 *     backend-provided "done reading?" test says so -- an EGL fence,
 *     a Vulkan frame serial, or (CPU) trivially true.
 *
 * Usage per frame: cam_stream_pump() from the fd callback (dequeue
 * everything, pick the newest), cam_stream_latest() when drawing,
 * cam_stream_rendered(index, fence) after submitting a draw that read
 * it, cam_stream_reap() whenever convenient to return finished buffers.
 */
#ifndef HELLO_WAYLAND_CAM_STREAM_H
#define HELLO_WAYLAND_CAM_STREAM_H

#include "v4l2_camera.h"

struct cam_fence_ops {
    int (*done)(void* fence, void* user);     /* 1 = GPU finished reading */
    void (*destroy)(void* fence, void* user); /* may be NULL */
};

struct cam_stream {
    struct camera cam;
    struct cam_fence_ops ops;
    void* user;

    int latest;                    /* buffer index held for rendering, -1 = none yet */
    int latest_seq;                /* bumps per new frame: "is there something new?" */
    void* fence[CAM_MAX_BUFFERS];  /* per buffer: the last draw that read it */
    int retiring[CAM_MAX_BUFFERS]; /* replaced as latest, waiting on fence */

    unsigned frames_captured, frames_dropped; /* stats */
};

/* Opens the camera (env CAM_DEVICE=/dev/videoN, CAM_SIZE=WxH override
 * the defaults: first device, 640x360) and starts streaming. */
int cam_stream_open(struct cam_stream* s, const struct cam_fence_ops* ops, void* user);
void cam_stream_close(struct cam_stream* s);

int cam_stream_fd(const struct cam_stream* s);

/* Dequeue all ready frames; the newest becomes `latest`, the rest are
 * requeued immediately. The previous latest is retired (or requeued
 * at once if nothing ever rendered it). */
void cam_stream_pump(struct cam_stream* s);

/* Requeue retiring buffers whose fence has signaled. */
void cam_stream_reap(struct cam_stream* s);

/* Record that a draw reading `index` was submitted; `fence` answers
 * "done?" later. Replaces (and destroys) any older fence for it. */
void cam_stream_rendered(struct cam_stream* s, int index, void* fence);

static inline int cam_stream_latest(const struct cam_stream* s) {
    return s->latest;
}

#endif
