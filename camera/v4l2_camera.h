/* v4l2_camera.h -- a raw V4L2 capture device, C port of
 * video-concepts/zero-copy-camera's V4l2Camera.
 *
 * Buffers are allocated by the driver (V4L2_MEMORY_MMAP) and exported
 * once as dmabuf fds (VIDIOC_EXPBUF) -- the memfd trick from README §4
 * step 8 in reverse: the KERNEL owns the pages (the ISP DMA-writes
 * into them) and hands us an fd that EGL and Vulkan can import
 * without a copy. The CPU backend maps them instead.
 *
 * The device fd is non-blocking and poll()-able: POLLIN means a
 * filled buffer can be dequeued. Buffers cycle QBUF (ours -> driver)
 * / DQBUF (driver -> ours); a buffer we hold is NOT written to.
 *
 * On this machine: the built-in camera is `apple-isp` (a plain V4L2
 * device -- all processing happens in ISP firmware, no libcamera
 * pipeline needed), NV12 only, BT.709 limited range.
 */
#ifndef HELLO_WAYLAND_V4L2_CAMERA_H
#define HELLO_WAYLAND_V4L2_CAMERA_H

#include <linux/videodev2.h>
#include <stdint.h>

enum { CAM_MAX_BUFFERS = 8 };

struct cam_buffer {
    int dmabuf_fd;   /* exported once; -1 if not */
    uint32_t length; /* bytes */
    void* map;       /* CPU mapping, made on demand by camera_map() */
};

struct cam_frame {
    uint32_t index; /* which buffer */
    uint32_t sequence;
    int64_t timestamp_us;
};

struct camera {
    int fd;
    char path[64];
    struct v4l2_pix_format format; /* what the driver actually gave us */
    struct cam_buffer bufs[CAM_MAX_BUFFERS];
    uint32_t buf_count;
    int streaming;
};

/* Opens `path` (NULL = first capture-capable /dev/video*), negotiates
 * NV12 (falling back to YUYV) at the requested size -- the driver may
 * adjust the size; read cam->format afterwards. Returns 0 on success. */
int camera_open(struct camera* cam, const char* path, uint32_t width, uint32_t height);

/* Allocates + exports the buffers, queues them all, starts streaming. */
int camera_start(struct camera* cam);

/* Non-blocking: returns 1 and fills *out if a filled buffer was
 * dequeued, 0 if none is ready. Call until it returns 0. */
int camera_dequeue(struct camera* cam, struct cam_frame* out);

/* Hand a buffer back to the driver for the next capture. */
void camera_queue(struct camera* cam, uint32_t index);

/* CPU access: mmap the buffer (cached; NULL on failure). */
void* camera_map(struct camera* cam, uint32_t index);

void camera_close(struct camera* cam);

/* Colorimetry, from what the driver reported. */
int camera_is_bt709(const struct camera* cam);
int camera_is_full_range(const struct camera* cam);

/* Plane geometry helpers for NV12: Y plane at offset 0, interleaved
 * UV plane right after it, both with the same stride. */
static inline uint32_t camera_stride(const struct camera* cam) {
    return cam->format.bytesperline;
}
static inline uint32_t camera_uv_offset(const struct camera* cam) {
    return cam->format.bytesperline * cam->format.height;
}

#endif
