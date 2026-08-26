/* hand_frame_cpu.c -- FrameToTensor on the host: the CPU EP's front end,
 * and the reference the WebGPU pass is checked against.
 *
 * Why this does not call camera/nv12_convert.h's converters, although it
 * shares their coefficients: nv12_to_xrgb converts the WHOLE frame at
 * full resolution into 8-bit XRGB and does no resampling. Routing
 * through it would pay for a 640x360 conversion and quantise to 8 bits
 * per channel, both before the resample that throws most of it away.
 * This reads only the ~110k source neighbourhoods the tensor actually
 * needs, straight out of the ISP's pages, in f32. What the two DO share
 * is nv12_yuv_params -- the numbers, not the loop.
 *
 * Sampling convention, matched exactly by hand_frame_to_tensor.wgsl:
 * continuous image coordinates where pixel k covers [k, k+1) and has its
 * centre at k+0.5, bilinear with clamp-to-edge, and chroma sampled at
 * half those coordinates ("centre siting"). The two paths agree because
 * they run the same formula, not because they were tuned to.
 */
#include <string.h>

#include <drm_fourcc.h>

#include "hand_frame.h"
#include "infer_util.h"
#include "nv12_convert.h"

void hand_frame_tensor_desc(uint32_t tw, uint32_t th, struct infer_desc* d) {
    memset(d, 0, sizeof *d);
    d->dtype = INFER_F32;
    d->ndim = 4;
    d->dims[0] = 1;
    d->dims[1] = (int64_t)th;
    d->dims[2] = (int64_t)tw;
    d->dims[3] = 3;
    infer_desc_set_image(d);
}

struct infer_frame hand_frame_import_desc(const struct infer_frame* f) {
    struct infer_frame d = *f;
    if (f->drm_format == DRM_FORMAT_YUYV) {
        /* one packed plane, 4 bytes per pixel PAIR. Not every driver
         * imports DRM_FORMAT_YUYV (NVIDIA does not), so the same bytes go
         * in as an RGBA8 image of half the width -- texel = (Y0,U,Y1,V) --
         * and the shader unpacks by texel parity. scene_cam_gles.c makes
         * the identical substitution for EGL. */
        d.drm_format = DRM_FORMAT_ABGR8888;
        d.width = f->width / 2;
        d.planes = 1;
    }
    return d;
}

void hand_frame_from_camera(struct infer_frame* f, const struct camera* cam, uint32_t index) {
    memset(f, 0, sizeof *f);
    const int nv12 = cam->format.pixelformat == V4L2_PIX_FMT_NV12;
    f->dmabuf_fd[0] = cam->bufs[index].dmabuf_fd;
    f->dmabuf_fd[1] = f->dmabuf_fd[2] = f->dmabuf_fd[3] = -1;
    f->offset[0] = 0;
    f->pitch[0] = camera_stride(cam);
    if (nv12) {
        /* two planes, one fd: the interleaved UV plane sits right after
         * the luma plane at the same stride */
        f->offset[1] = camera_uv_offset(cam);
        f->pitch[1] = camera_stride(cam);
        f->planes = 2;
        f->drm_format = DRM_FORMAT_NV12;
    } else {
        f->planes = 1;
        f->drm_format = DRM_FORMAT_YUYV; /* the honest fourcc; see hand_frame_import_desc */
    }
    f->drm_modifier = DRM_FORMAT_MOD_LINEAR; /* V4L2 MMAP buffers are linear */
    f->width = cam->format.width;
    f->height = cam->format.height;
    f->bt709 = (uint8_t)(camera_is_bt709(cam) != 0);
    f->full_range = (uint8_t)(camera_is_full_range(cam) != 0);
    f->map = cam->bufs[index].map; /* NULL unless camera_map() was called */
    f->ready.sync_fd = -1;         /* V4L2 hands out no fence */
}

/* Bilinear tap of one component of an interleaved 8-bit plane.
 * (x, y) are continuous image coordinates; the plane is w x h samples of
 * `comps` bytes each, `stride` bytes per row. Clamp to edge. */
static float tap(const uint8_t* base,
                 uint32_t stride,
                 int w,
                 int h,
                 int comps,
                 int comp,
                 float x,
                 float y) {
    float fx = x - 0.5f, fy = y - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - (float)x0, ay = fy - (float)y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) { x0 = 0; }
    if (y0 < 0) { y0 = 0; }
    if (x1 < 0) { x1 = 0; }
    if (y1 < 0) { y1 = 0; }
    if (x0 > w - 1) { x0 = w - 1; }
    if (y0 > h - 1) { y0 = h - 1; }
    if (x1 > w - 1) { x1 = w - 1; }
    if (y1 > h - 1) { y1 = h - 1; }
    const uint8_t* r0 = base + (size_t)y0 * stride;
    const uint8_t* r1 = base + (size_t)y1 * stride;
    float p00 = (float)r0[(size_t)x0 * comps + comp], p10 = (float)r0[(size_t)x1 * comps + comp];
    float p01 = (float)r1[(size_t)x0 * comps + comp], p11 = (float)r1[(size_t)x1 * comps + comp];
    return (p00 * (1.0f - ax) + p10 * ax) * (1.0f - ay) +
           (p01 * (1.0f - ax) + p11 * ax) * ay;
}

static float clamp01(float v) {
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

int hand_frame_to_tensor_cpu(const struct infer_frame* f,
                             struct hand_affine a,
                             struct hand_norm n,
                             enum hand_border border,
                             struct infer_tensor* t) {
    INFER_CHECK(t->domain == INFER_DOMAIN_CPU && t->mem.cpu.ptr, "hand: CPU tensor expected");
    INFER_CHECK(f->map, "hand: frame has no CPU mapping");
    INFER_CHECK(t->desc.ndim == 4 && t->desc.dims[3] == 3, "hand: tensor is not [1,H,W,3]");
    const int nv12 = f->drm_format == DRM_FORMAT_NV12;
    const int yuyv = f->drm_format == DRM_FORMAT_YUYV;
    INFER_CHECK(nv12 || yuyv, "hand: only NV12 and YUYV frames (fourcc 0x%08x)", f->drm_format);

    const uint32_t th = (uint32_t)t->desc.dims[1], tw = (uint32_t)t->desc.dims[2];
    const int fw = (int)f->width, fh = (int)f->height;
    const uint8_t* base = f->map;
    const uint8_t* luma = base + f->offset[0];
    const uint8_t* chroma = nv12 ? base + f->offset[1] : NULL;
    const uint32_t pitch = f->pitch[0];

    float coef[4], range[3];
    nv12_yuv_params(f->bt709, f->full_range, coef, range);
    const float span = n.hi - n.lo;
    float* dst = t->mem.cpu.ptr;

    for (uint32_t j = 0; j < th; j++) {
        for (uint32_t i = 0; i < tw; i++) {
            float x, y;
            hand_affine_apply(a, (float)i + 0.5f, (float)j + 0.5f, &x, &y);
            float* o = dst + ((size_t)j * tw + i) * 3;
            /* outside the frame: black for the detector's letterbox bars,
             * the nearest edge pixel for the landmark crop. tap() already
             * clamps, so REPLICATE is simply not taking this branch. */
            if (border == HAND_BORDER_ZERO &&
                (x < 0.0f || y < 0.0f || x >= (float)fw || y >= (float)fh)) {
                o[0] = o[1] = o[2] = n.lo;
                continue;
            }
            float yy, u, v;
            if (nv12) {
                yy = tap(luma, pitch, fw, fh, 1, 0, x, y);
                /* the chroma plane is half size and covers the same
                 * extent, so its coordinates are half ours */
                u = tap(chroma, pitch, fw / 2, fh / 2, 2, 0, x * 0.5f, y * 0.5f);
                v = tap(chroma, pitch, fw / 2, fh / 2, 2, 1, x * 0.5f, y * 0.5f);
            } else {
                /* packed (Y0,U,Y1,V) per pixel pair: luma is one byte per
                 * pixel at stride 2, chroma one pair per two pixels. The
                 * parity pick is nearest by construction -- interpolating
                 * across a pair boundary would mix Y0 with Y1. */
                int xi = (int)x;
                if (xi > fw - 1) { xi = fw - 1; }
                int yi = (int)y;
                if (yi > fh - 1) { yi = fh - 1; }
                const uint8_t* row = luma + (size_t)yi * pitch;
                const uint8_t* q = row + (size_t)(xi / 2) * 4; /* Y0 U Y1 V */
                yy = (float)q[(xi & 1) * 2];
                u = (float)q[1];
                v = (float)q[3];
            }
            yy = (yy - range[0]) * range[1];
            u = (u - 128.0f) * range[2];
            v = (v - 128.0f) * range[2];
            float r = clamp01((yy + coef[0] * v) * (1.0f / 255.0f));
            float g = clamp01((yy - coef[1] * u - coef[2] * v) * (1.0f / 255.0f));
            float b = clamp01((yy + coef[3] * u) * (1.0f / 255.0f));
            o[0] = n.lo + r * span;
            o[1] = n.lo + g * span;
            o[2] = n.lo + b * span;
        }
    }
    return 0;
}
