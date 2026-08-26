/* nv12_convert.h -- NV12 / YUYV -> packed 0x00RRGGBB on the CPU.
 *
 * NV12: a full-resolution Y plane, then a half-resolution plane of
 * interleaved U,V pairs (4:2:0 -- one chroma sample per 2x2 luma
 * block). YUYV (what UVC webcams deliver): one packed plane, 4 bytes
 * per horizontal pixel PAIR: Y0 U Y1 V (4:2:2 -- one chroma sample
 * per 2 luma). YUV->RGB is a 3x3 matrix after removing the offsets;
 * the matrix (BT.601 vs BT.709) and the offsets/scales (limited: Y in
 * 16..235, C in 16..240; full: 0..255) come from the driver's
 * colorimetry. This is exactly what VkSamplerYcbcrConversion and
 * samplerExternalOES do inside the GPU's texture unit.
 */
#ifndef HELLO_WAYLAND_NV12_CONVERT_H
#define HELLO_WAYLAND_NV12_CONVERT_H

#include <stddef.h>
#include <stdint.h>

#include "v4l2_camera.h"

static inline uint32_t nv12_clamp8(float v) {
    return v <= 0.0f ? 0u : v >= 255.0f ? 255u : (uint32_t)(v + 0.5f);
}

/* The one place the numbers live. Four consumers read them: the two
 * converters below (the CPU camera scene), stage five's CPU
 * FrameToTensor, and the uniform stage five hands its WGSL pass -- so
 * "the GPU and the CPU compute the same colour" is a property of the
 * code, not a thing to re-check.
 *   coef  = {r<-v, g<-u, g<-v, b<-u}
 *   range = {y offset, y scale, chroma scale}, in 0..255 units */
static inline void nv12_yuv_params(int bt709, int full_range, float coef[4], float range[3]) {
    coef[0] = bt709 ? 1.5748f : 1.4020f;
    coef[1] = bt709 ? 0.1873f : 0.3441f;
    coef[2] = bt709 ? 0.4681f : 0.7141f;
    coef[3] = bt709 ? 1.8556f : 1.7720f;
    range[0] = full_range ? 0.0f : 16.0f;
    range[1] = 255.0f / (full_range ? 255.0f : 219.0f);
    range[2] = 255.0f / (full_range ? 255.0f : 224.0f);
}

static inline void nv12_to_xrgb(const uint8_t* y_plane,
                                const uint8_t* uv_plane,
                                uint32_t stride,
                                uint32_t width,
                                uint32_t height,
                                int bt709,
                                int full_range,
                                uint32_t* out /* width*height */) {
    /* R = y + rv*v; G = y - gu*u - gv*v; B = y + bu*u, with y,u,v
     * normalized to [0,1] / [-0.5,0.5] */
    float coef[4], range[3];
    nv12_yuv_params(bt709, full_range, coef, range);
    const float rv = coef[0], gu = coef[1], gv = coef[2], bu = coef[3];
    const float y_off = range[0], y_scale = range[1], c_scale = range[2];

    for (uint32_t row = 0; row < height; row++) {
        const uint8_t* y = y_plane + (size_t)row * stride;
        const uint8_t* uv = uv_plane + (size_t)(row / 2) * stride; /* 2 luma rows share one */
        uint32_t* dst = out + (size_t)row * width;
        for (uint32_t x = 0; x < width; x++) {
            float yy = ((float)y[x] - y_off) * y_scale;
            float u = ((float)uv[(x / 2) * 2] - 128.0f) * c_scale;
            float v = ((float)uv[(x / 2) * 2 + 1] - 128.0f) * c_scale;
            uint32_t r = nv12_clamp8(yy + rv * v);
            uint32_t g = nv12_clamp8(yy - gu * u - gv * v);
            uint32_t b = nv12_clamp8(yy + bu * u);
            dst[x] = (r << 16) | (g << 8) | b;
        }
    }
}

static inline void yuyv_to_xrgb(const uint8_t* plane,
                                uint32_t stride,
                                uint32_t width,
                                uint32_t height,
                                int bt709,
                                int full_range,
                                uint32_t* out /* width*height */) {
    float coef[4], range[3];
    nv12_yuv_params(bt709, full_range, coef, range);
    const float rv = coef[0], gu = coef[1], gv = coef[2], bu = coef[3];
    const float y_off = range[0], y_scale = range[1], c_scale = range[2];

    for (uint32_t row = 0; row < height; row++) {
        const uint8_t* src = plane + (size_t)row * stride;
        uint32_t* dst = out + (size_t)row * width;
        for (uint32_t x = 0; x < width; x += 2) {
            const uint8_t* q = src + (size_t)x * 2; /* Y0 U Y1 V */
            float u = ((float)q[1] - 128.0f) * c_scale;
            float v = ((float)q[3] - 128.0f) * c_scale;
            for (uint32_t k = 0; k < 2 && x + k < width; k++) {
                float yy = ((float)q[k * 2] - y_off) * y_scale;
                uint32_t r = nv12_clamp8(yy + rv * v);
                uint32_t g = nv12_clamp8(yy - gu * u - gv * v);
                uint32_t b = nv12_clamp8(yy + bu * u);
                dst[x + k] = (r << 16) | (g << 8) | b;
            }
        }
    }
}

/* Whatever the camera negotiated -> XRGB. Returns -1 for a format
 * neither converter handles. */
static inline int camera_frame_to_xrgb(const struct camera* cam,
                                       const uint8_t* base,
                                       uint32_t* out) {
    const int bt709 = camera_is_bt709(cam), full = camera_is_full_range(cam);
    switch (cam->format.pixelformat) {
        case V4L2_PIX_FMT_NV12:
            nv12_to_xrgb(base,
                         base + camera_uv_offset(cam),
                         camera_stride(cam),
                         cam->format.width,
                         cam->format.height,
                         bt709,
                         full,
                         out);
            return 0;
        case V4L2_PIX_FMT_YUYV:
            yuyv_to_xrgb(base,
                         camera_stride(cam),
                         cam->format.width,
                         cam->format.height,
                         bt709,
                         full,
                         out);
            return 0;
        default: return -1;
    }
}

#endif
