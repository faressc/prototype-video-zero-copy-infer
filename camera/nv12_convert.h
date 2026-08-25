/* nv12_convert.h -- NV12 -> packed 0x00RRGGBB on the CPU.
 *
 * NV12: a full-resolution Y plane, then a half-resolution plane of
 * interleaved U,V pairs (4:2:0 -- one chroma sample per 2x2 luma
 * block). YUV->RGB is a 3x3 matrix after removing the offsets; the
 * matrix (BT.601 vs BT.709) and the offsets/scales (limited: Y in
 * 16..235, C in 16..240; full: 0..255) come from the driver's
 * colorimetry. This is exactly what VkSamplerYcbcrConversion and
 * samplerExternalOES do inside the GPU's texture unit.
 */
#ifndef HELLO_WAYLAND_NV12_CONVERT_H
#define HELLO_WAYLAND_NV12_CONVERT_H

#include <stddef.h>
#include <stdint.h>

static inline uint32_t nv12_clamp8(float v) {
    return v <= 0.0f ? 0u : v >= 255.0f ? 255u : (uint32_t)(v + 0.5f);
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
    const float rv = bt709 ? 1.5748f : 1.4020f;
    const float gu = bt709 ? 0.1873f : 0.3441f;
    const float gv = bt709 ? 0.4681f : 0.7141f;
    const float bu = bt709 ? 1.8556f : 1.7720f;
    const float y_off = full_range ? 0.0f : 16.0f;
    const float y_scale = 255.0f / (full_range ? 255.0f : 219.0f);
    const float c_scale = 255.0f / (full_range ? 255.0f : 224.0f);

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

#endif
