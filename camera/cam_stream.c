/* cam_stream.c -- see cam_stream.h. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cam_stream.h"

int cam_stream_open(struct cam_stream* s, const struct cam_fence_ops* ops, void* user) {
    memset(s, 0, sizeof(*s));
    s->ops = *ops;
    s->user = user;
    s->latest = -1;

    const char* dev = getenv("CAM_DEVICE");
    unsigned w = 640, h = 360;
    const char* size = getenv("CAM_SIZE");
    if (size && sscanf(size, "%ux%u", &w, &h) != 2) {
        fprintf(stderr, "CAM_SIZE: expected WxH\n");
        return -1;
    }
    if (camera_open(&s->cam, dev, w, h) < 0) { return -1; }
    return camera_start(&s->cam);
}

void cam_stream_close(struct cam_stream* s) {
    for (int i = 0; i < CAM_MAX_BUFFERS; i++) {
        if (s->fence[i] && s->ops.destroy) { s->ops.destroy(s->fence[i], s->user); }
        s->fence[i] = NULL;
    }
    camera_close(&s->cam);
    fprintf(stderr,
            "camera: %u frames captured, %u dropped (never rendered)\n",
            s->frames_captured,
            s->frames_dropped);
}

int cam_stream_fd(const struct cam_stream* s) {
    return s->cam.fd;
}

static void retire(struct cam_stream* s, int index) {
    if (s->fence[index]) {
        s->retiring[index] = 1; /* wait for the GPU */
    } else {
        camera_queue(&s->cam, (uint32_t)index); /* never read: back at once */
        s->frames_dropped++;
    }
}

void cam_stream_pump(struct cam_stream* s) {
    struct cam_frame f;
    int newest = -1;
    while (camera_dequeue(&s->cam, &f)) {
        s->frames_captured++;
        if (newest >= 0) {
            /* an older frame from the same burst: nobody will see it */
            camera_queue(&s->cam, (uint32_t)newest);
            s->frames_dropped++;
        }
        newest = (int)f.index;
    }
    if (newest < 0) { return; }
    if (s->latest >= 0) { retire(s, s->latest); }
    s->latest = newest;
    s->latest_seq++;
}

void cam_stream_rendered(struct cam_stream* s, int index, void* fence) {
    if (index < 0) { return; }
    if (s->fence[index] && s->ops.destroy) { s->ops.destroy(s->fence[index], s->user); }
    s->fence[index] = fence;
}

void cam_stream_reap(struct cam_stream* s) {
    for (int i = 0; i < CAM_MAX_BUFFERS; i++) {
        if (!s->retiring[i]) { continue; }
        if (s->fence[i] && !s->ops.done(s->fence[i], s->user)) { continue; }
        if (s->fence[i] && s->ops.destroy) { s->ops.destroy(s->fence[i], s->user); }
        s->fence[i] = NULL;
        s->retiring[i] = 0;
        camera_queue(&s->cam, (uint32_t)i);
    }
}
