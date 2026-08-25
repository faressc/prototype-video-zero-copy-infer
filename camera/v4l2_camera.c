/* v4l2_camera.c -- see v4l2_camera.h. */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "v4l2_camera.h"

enum { CAM_REQUEST_BUFFERS = 6 };

static int xioctl(int fd, unsigned long request, void* arg) {
    int ret;
    do { ret = ioctl(fd, request, arg); } while (ret < 0 && errno == EINTR);
    return ret;
}

static void fourcc_str(uint32_t fourcc, char out[5]) {
    out[0] = (char)(fourcc & 0xff);
    out[1] = (char)((fourcc >> 8) & 0xff);
    out[2] = (char)((fourcc >> 16) & 0xff);
    out[3] = (char)((fourcc >> 24) & 0xff);
    out[4] = 0;
}

/* First /dev/videoN that can capture + stream. The apple-isp exposes
 * /dev/video0 for capture; /dev/media0 is its media-controller node. */
static int find_first_capture(char out[64]) {
    for (int i = 0; i < 64; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/video%d", i);
        int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) { continue; }
        struct v4l2_capability cap = {0};
        int ok = xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0;
        close(fd);
        if (!ok) { continue; }
        uint32_t caps =
            (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
        if ((caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING)) {
            strncpy(out, path, 63);
            out[63] = 0;
            return 0;
        }
    }
    return -1;
}

static int negotiate_format(struct camera* cam, uint32_t width, uint32_t height) {
    const uint32_t preferred[] = {V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV};
    for (size_t i = 0; i < sizeof(preferred) / sizeof(preferred[0]); i++) {
        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = preferred[i];
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        if (xioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) { continue; }
        if (fmt.fmt.pix.pixelformat != preferred[i]) { continue; } /* driver switched formats */
        cam->format = fmt.fmt.pix;
        char fcc[5];
        fourcc_str(cam->format.pixelformat, fcc);
        fprintf(stderr,
                "camera: %s %ux%u stride %u, colorspace %u ycbcr_enc %u quant %u\n",
                fcc,
                cam->format.width,
                cam->format.height,
                cam->format.bytesperline,
                cam->format.colorspace,
                cam->format.ycbcr_enc,
                cam->format.quantization);
        if (cam->format.width != width || cam->format.height != height) {
            fprintf(stderr, "camera: driver adjusted size from %ux%u\n", width, height);
        }
        return 0;
    }
    fprintf(stderr, "camera: %s offers neither NV12 nor YUYV; available:", cam->path);
    struct v4l2_fmtdesc desc = {0};
    desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (desc.index = 0; xioctl(cam->fd, VIDIOC_ENUM_FMT, &desc) == 0; desc.index++) {
        char fcc[5];
        fourcc_str(desc.pixelformat, fcc);
        fprintf(stderr, " %s", fcc);
    }
    fprintf(stderr, "\n");
    return -1;
}

int camera_open(struct camera* cam, const char* path, uint32_t width, uint32_t height) {
    memset(cam, 0, sizeof(*cam));
    cam->fd = -1;
    for (int i = 0; i < CAM_MAX_BUFFERS; i++) { cam->bufs[i].dmabuf_fd = -1; }

    if (path) {
        strncpy(cam->path, path, sizeof(cam->path) - 1);
    } else if (find_first_capture(cam->path) < 0) {
        fprintf(stderr, "camera: no capture device found\n");
        return -1;
    }

    cam->fd = open(cam->path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (cam->fd < 0) {
        fprintf(stderr, "camera: cannot open %s: %s\n", cam->path, strerror(errno));
        return -1;
    }
    struct v4l2_capability cap = {0};
    if (xioctl(cam->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "camera: VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        return -1;
    }
    fprintf(stderr, "camera: %s (%s), driver %s\n", cam->path, cap.card, cap.driver);
    return negotiate_format(cam, width, height);
}

int camera_start(struct camera* cam) {
    struct v4l2_requestbuffers req = {0};
    req.count = CAM_REQUEST_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0) {
        if (errno == EBUSY) {
            fprintf(stderr,
                    "camera: %s is busy (another process holds its buffers -- PipeWire?)\n",
                    cam->path);
        } else {
            fprintf(stderr, "camera: VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        }
        return -1;
    }
    if (req.count < 3 || req.count > CAM_MAX_BUFFERS) {
        fprintf(stderr, "camera: driver granted %u buffers\n", req.count);
        return -1;
    }
    cam->buf_count = req.count;

    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(cam->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "camera: VIDIOC_QUERYBUF(%u) failed: %s\n", i, strerror(errno));
            return -1;
        }
        cam->bufs[i].length = buf.length;

        /* the export: driver pages -> an fd anyone can import */
        struct v4l2_exportbuffer exp = {0};
        exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        exp.index = i;
        exp.flags = O_CLOEXEC | O_RDWR;
        if (xioctl(cam->fd, VIDIOC_EXPBUF, &exp) < 0) {
            fprintf(stderr,
                    "camera: VIDIOC_EXPBUF(%u) failed: %s (no dmabuf export?)\n",
                    i,
                    strerror(errno));
            return -1;
        }
        cam->bufs[i].dmabuf_fd = exp.fd;
    }
    fprintf(stderr,
            "camera: %u buffers of %u bytes exported as dmabufs\n",
            req.count,
            cam->bufs[0].length);

    for (uint32_t i = 0; i < req.count; i++) { camera_queue(cam, i); }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam->fd, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "camera: VIDIOC_STREAMON failed: %s\n", strerror(errno));
        return -1;
    }
    cam->streaming = 1;
    return 0;
}

int camera_dequeue(struct camera* cam, struct cam_frame* out) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) { return 0; }
        fprintf(stderr, "camera: VIDIOC_DQBUF failed: %s\n", strerror(errno));
        return 0;
    }
    out->index = buf.index;
    out->sequence = buf.sequence;
    out->timestamp_us = (int64_t)buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
    return 1;
}

void camera_queue(struct camera* cam, uint32_t index) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (xioctl(cam->fd, VIDIOC_QBUF, &buf) < 0) {
        fprintf(stderr, "camera: VIDIOC_QBUF(%u) failed: %s\n", index, strerror(errno));
    }
}

void* camera_map(struct camera* cam, uint32_t index) {
    struct cam_buffer* b = &cam->bufs[index];
    if (!b->map) {
        /* the dmabuf fd is mmap-able too, but the classic V4L2 way is
         * to map through the device fd at the buffer's offset */
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = index;
        if (xioctl(cam->fd, VIDIOC_QUERYBUF, &buf) < 0) { return NULL; }
        void* m = mmap(NULL, buf.length, PROT_READ, MAP_SHARED, cam->fd, buf.m.offset);
        if (m == MAP_FAILED) {
            fprintf(stderr, "camera: mmap(%u) failed: %s\n", index, strerror(errno));
            return NULL;
        }
        b->map = m;
    }
    return b->map;
}

void camera_close(struct camera* cam) {
    if (cam->fd < 0) { return; }
    if (cam->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(cam->fd, VIDIOC_STREAMOFF, &type);
        cam->streaming = 0;
    }
    for (uint32_t i = 0; i < cam->buf_count; i++) {
        if (cam->bufs[i].map) { munmap(cam->bufs[i].map, cam->bufs[i].length); }
        if (cam->bufs[i].dmabuf_fd >= 0) { close(cam->bufs[i].dmabuf_fd); }
    }
    struct v4l2_requestbuffers req = {0};
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(cam->fd, VIDIOC_REQBUFS, &req);
    close(cam->fd);
    cam->fd = -1;
}

int camera_is_bt709(const struct camera* cam) {
    const struct v4l2_pix_format* f = &cam->format;
    if (f->ycbcr_enc == V4L2_YCBCR_ENC_709) { return 1; }
    if (f->ycbcr_enc == V4L2_YCBCR_ENC_601) { return 0; }
    /* default encoding follows the colorspace */
    return f->colorspace == V4L2_COLORSPACE_REC709 || f->colorspace == V4L2_COLORSPACE_SRGB;
}

int camera_is_full_range(const struct camera* cam) {
    const struct v4l2_pix_format* f = &cam->format;
    if (f->quantization == V4L2_QUANTIZATION_FULL_RANGE) { return 1; }
    if (f->quantization == V4L2_QUANTIZATION_LIM_RANGE) { return 0; }
    return f->colorspace == V4L2_COLORSPACE_JPEG; /* JPEG defaults to full range */
}
