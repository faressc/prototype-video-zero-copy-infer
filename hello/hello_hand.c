/* hello_hand.c -- stage five headless: a camera frame (or a raw NV12
 * file) turned into a model input, run through the palm detector and the
 * hand-landmark model on either execution provider, decoded to 21
 * landmarks, and printed.
 *
 * Headless on purpose, and first. Everything that can be wrong about
 * hand tracking -- the affine, the chroma siting, the anchor layout, the
 * regressor order, the ROI rotation, the input range -- is wrong in a
 * way that a spinning cube reports as "no hand". So each of those gets
 * an assertion or a printed number here before any window exists.
 *
 *   hello_hand --self-test                     the pure arithmetic, no camera, no model
 *   hello_hand --probe                         which Dawn import path this driver allows
 *   hello_hand --stage frame --ppm /tmp/t.ppm  eyeball what the model is actually fed
 *   hello_hand --compare-frame-to-tensor       the CPU and WebGPU passes agree
 *
 * --strict turns the checks into the exit code, as in hello_inference.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <drm_fourcc.h>
#include <poll.h>

#include "hand.h"
#include "hand_frame.h"
#include "hand_model.h"
#include "infer.h"
#include "infer_util.h"
#include "v4l2_camera.h"

enum stage { STAGE_FRAME, STAGE_DETECT, STAGE_LANDMARK };

struct options {
    enum infer_ep ep;
    enum stage stage;
    const char* frame_path;
    const char* ppm_path;
    const char* palm_model;
    const char* lmk_model;
    const char* dump_path;
    uint32_t want_w, want_h;
    uint32_t tensor;
    int crop, self_test, probe, compare, strict, verbose, iters, warmup, all, require_hand, chain;
    struct hand_roi roi;
    int have_roi;
    struct hand_norm norm;
};

/* ------------------------------------------------------------------ */
/* the frame source: a live camera, or a raw NV12 file                 */
/* ------------------------------------------------------------------ */

/* A file-backed frame still has to be a dma-buf, or the WebGPU path
 * cannot see it. Stage four's dma-heap producer allocates one; its
 * descriptor only decides the SIZE (bytes_image = img_w * img_h * 4), so
 * a shape wide enough for the frame's planes is all that is needed --
 * the frame's own pitch/offset describe the real layout. */
struct frame_source {
    struct camera cam;
    int have_cam;
    struct infer_tensor heap; /* file-backed: the dma-buf holding the bytes */
    struct infer_frame frame;
};

static int source_open_file(struct frame_source* s, const struct options* o) {
    INFER_CHECK(o->want_w && o->want_h, "--frame needs --size WxH");
    size_t need = (size_t)o->want_w * o->want_h * 3 / 2; /* NV12, stride == width */
    size_t f_size = 0;
    void* bytes = infer_read_file(o->frame_path, &f_size);
    INFER_CHECK(bytes, "cannot read %s", o->frame_path);
    if (f_size < need) {
        free(bytes);
        INFER_CHECK(0, "%s is %zu bytes, NV12 %ux%u needs %zu", o->frame_path, f_size, o->want_w,
                    o->want_h, need);
    }
    struct infer_desc d;
    memset(&d, 0, sizeof d);
    d.dtype = INFER_F32;
    d.ndim = 1;
    d.dims[0] = (int64_t)((need + 3) / 4);
    d.img_w = 64;
    d.img_h = (uint32_t)((need + 255) / 256);
    d.row_pitch_bytes = d.img_w * 4;
    memset(&s->heap, 0, sizeof s->heap);
    s->heap.ready.sync_fd = -1;
    s->heap.released.sync_fd = -1;
    if (infer_dmabuf_alloc(&d, &s->heap) < 0) {
        free(bytes);
        INFER_CHECK(0, "no dma-heap: --frame needs /dev/dma_heap/system");
    }
    infer_dmabuf_sync(&s->heap, 1, 1);
    memcpy(s->heap.mem.dmabuf.map, bytes, need);
    infer_dmabuf_cpu_cache_sync(s->heap.mem.dmabuf.map, need);
    infer_dmabuf_sync(&s->heap, 0, 1);
    free(bytes);

    memset(&s->frame, 0, sizeof s->frame);
    s->frame.dmabuf_fd[0] = s->heap.mem.dmabuf.fd;
    s->frame.dmabuf_fd[1] = s->frame.dmabuf_fd[2] = s->frame.dmabuf_fd[3] = -1;
    s->frame.offset[0] = 0;
    s->frame.pitch[0] = o->want_w;
    s->frame.offset[1] = o->want_w * o->want_h;
    s->frame.pitch[1] = o->want_w;
    s->frame.planes = 2;
    s->frame.drm_format = DRM_FORMAT_NV12;
    s->frame.drm_modifier = DRM_FORMAT_MOD_LINEAR;
    s->frame.width = o->want_w;
    s->frame.height = o->want_h;
    s->frame.bt709 = 1; /* what convert.sh's source material is; --frame is a fixture */
    s->frame.full_range = 0;
    s->frame.map = s->heap.mem.dmabuf.map;
    s->frame.ready.sync_fd = -1;
    return 0;
}

static int source_open_camera(struct frame_source* s, const struct options* o) {
    uint32_t w = o->want_w ? o->want_w : 640, h = o->want_h ? o->want_h : 360;
    INFER_CHECK(camera_open(&s->cam, getenv("CAM_DEVICE"), w, h) == 0, "no camera");
    INFER_CHECK(camera_start(&s->cam) == 0, "camera will not stream");
    s->have_cam = 1;
    /* wait for one filled buffer -- the ISP takes a few frames to settle,
     * so take the fourth rather than the first */
    struct cam_frame got;
    int have = 0;
    for (int tries = 0; tries < 60 && have < 4; tries++) {
        struct pollfd p = {.fd = s->cam.fd, .events = POLLIN};
        if (poll(&p, 1, 1000) != 1) { break; }
        while (camera_dequeue(&s->cam, &got)) {
            have++;
            if (have < 4) { camera_queue(&s->cam, got.index); }
        }
    }
    INFER_CHECK(have >= 4, "camera delivered no frame");
    INFER_CHECK(camera_map(&s->cam, got.index), "cannot map the camera buffer");
    hand_frame_from_camera(&s->frame, &s->cam, got.index);
    fprintf(stderr,
            "camera: %s %ux%u %.4s, %s %s\n",
            s->cam.path,
            s->cam.format.width,
            s->cam.format.height,
            (const char*)&s->cam.format.pixelformat,
            camera_is_bt709(&s->cam) ? "BT.709" : "BT.601",
            camera_is_full_range(&s->cam) ? "full range" : "limited range");
    return 0;
}

static int source_open(struct frame_source* s, const struct options* o) {
    memset(s, 0, sizeof *s);
    return o->frame_path ? source_open_file(s, o) : source_open_camera(s, o);
}

static void source_close(struct frame_source* s) {
    if (s->have_cam) { camera_close(&s->cam); }
    if (s->heap.mem.dmabuf.map) { infer_dmabuf_release(&s->heap); }
}

/* ------------------------------------------------------------------ */
/* --self-test: the arithmetic that has no dependencies                */
/* ------------------------------------------------------------------ */

static int check(int ok, const char* what) {
    fprintf(stderr, "  %-52s %s\n", what, ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

static int near(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int self_test(void) {
    int bad = 0;
    fprintf(stderr, "affine:\n");

    /* letterbox 640x360 -> 192x192: scale 0.3, so the frame becomes
     * 192x108 with 42 dead rows top and bottom */
    struct hand_affine lb = hand_affine_letterbox(640, 360, 192, 192);
    float x, y;
    hand_affine_apply(lb, 96.0f, 96.0f, &x, &y); /* tensor centre -> frame centre */
    bad += check(near(x, 320.0f, 0.01f) && near(y, 180.0f, 0.01f), "letterbox centre -> frame centre");
    hand_affine_apply(lb, 0.0f, 42.0f, &x, &y); /* top-left of the covered band */
    bad += check(near(x, 0.0f, 0.01f) && near(y, 0.0f, 0.01f), "letterbox band origin -> frame origin");
    bad += check(near(hand_affine_scale(lb), 1.0f / 0.3f, 1e-4f), "letterbox scale is 1/0.3 frame px per tensor px");
    hand_affine_apply(lb, 96.0f, 10.0f, &x, &y);
    bad += check(y < 0.0f, "above the band maps outside the frame (the border)");

    /* square crop 640x360 -> 192x192: scale 192/360, the sides are lost */
    struct hand_affine cr = hand_affine_square_crop(640, 360, 192, 192);
    hand_affine_apply(cr, 96.0f, 96.0f, &x, &y);
    bad += check(near(x, 320.0f, 0.01f) && near(y, 180.0f, 0.01f), "square-crop centre -> frame centre");
    hand_affine_apply(cr, 0.0f, 0.0f, &x, &y);
    bad += check(x > 0.0f && near(y, 0.0f, 0.01f), "square-crop keeps the short axis whole");

    /* an unrotated ROI is a plain window: centre to centre, and the
     * corners land on the window's corners */
    struct hand_affine r0 = hand_affine_roi(100.0f, 200.0f, 80.0f, 0.0f, 224, 224);
    hand_affine_apply(r0, 112.0f, 112.0f, &x, &y);
    bad += check(near(x, 100.0f, 0.01f) && near(y, 200.0f, 0.01f), "roi centre -> roi centre");
    hand_affine_apply(r0, 0.0f, 0.0f, &x, &y);
    bad += check(near(x, 60.0f, 0.01f) && near(y, 160.0f, 0.01f), "roi origin -> window corner");

    /* a quarter turn sends the ROI's +u axis to the frame's +y axis */
    struct hand_affine r90 = hand_affine_roi(100.0f, 200.0f, 80.0f, (float)M_PI_2, 224, 224);
    hand_affine_apply(r90, 224.0f, 112.0f, &x, &y);
    bad += check(near(x, 100.0f, 0.01f) && near(y, 240.0f, 0.01f), "roi rotated 90 deg sends +u to +y");

    /* the inverse is the inverse, for a rotated map too */
    struct hand_affine inv = hand_affine_invert(r90);
    float u, v;
    hand_affine_apply(r90, 37.0f, 201.0f, &x, &y);
    hand_affine_apply(inv, x, y, &u, &v);
    bad += check(near(u, 37.0f, 1e-3f) && near(v, 201.0f, 1e-3f), "roi affine round-trips through its inverse");
    inv = hand_affine_invert(lb);
    hand_affine_apply(lb, 11.0f, 133.0f, &x, &y);
    hand_affine_apply(inv, x, y, &u, &v);
    bad += check(near(u, 11.0f, 1e-3f) && near(v, 133.0f, 1e-3f), "letterbox affine round-trips");

    /* The anchor grid, pinned by five identities. If the layer merging or
     * the emission order were wrong, the count could still come out right
     * while every box landed somewhere else -- so check the layout, not
     * just the total. */
    fprintf(stderr, "anchors:\n");
    static float anchors[HAND_PALM_ANCHORS * 2 + 64];
    const int n = hand_palm_anchors(anchors, HAND_PALM_ANCHORS + 32);
    bad += check(n == HAND_PALM_ANCHORS, "count is 2016, the model's num_boxes");
    const float a24 = 0.5f / 24.0f, a12 = 0.5f / 12.0f;
    bad += check(near(anchors[0], a24, 1e-6f) && near(anchors[1], a24, 1e-6f),
                 "anchor 0 is the stride-8 grid's first cell centre");
    bad += check(near(anchors[2], a24, 1e-6f) && near(anchors[3], a24, 1e-6f),
                 "anchor 1 shares it (2 per cell: aspect + interpolated)");
    bad += check(near(anchors[4], 1.5f / 24.0f, 1e-6f) && near(anchors[5], a24, 1e-6f),
                 "anchor 2 steps one cell in x, not in y");
    bad += check(near(anchors[1151 * 2 + 0], 23.5f / 24.0f, 1e-6f) &&
                     near(anchors[1151 * 2 + 1], 23.5f / 24.0f, 1e-6f),
                 "anchor 1151 closes the 24x24 grid (1152 = 24*24*2)");
    int six_alike = 1;
    for (int k = 0; k < 6; k++) {
        six_alike &= near(anchors[(1152 + k) * 2 + 0], a12, 1e-6f) &&
                     near(anchors[(1152 + k) * 2 + 1], a12, 1e-6f);
    }
    bad += check(six_alike, "anchors 1152..1157 share the 12x12 centre (3 merged layers)");

    /* An unrotated palm box must give an unrotated ROI, squared to the
     * long side and scaled 2.6, with the centre shifted along -y by half
     * the ORIGINAL height (shift before square, per RectTransformation). */
    fprintf(stderr, "roi:\n");
    struct hand_det d;
    memset(&d, 0, sizeof d);
    d.score = 1.0f;
    d.cx = 96.0f;
    d.cy = 96.0f;
    d.w = 40.0f;
    d.h = 20.0f;
    d.kp[0][0] = 96.0f; /* palm base below, middle-finger keypoint above: */
    d.kp[0][1] = 100.0f; /* the hand points straight up, so rotation is 0 */
    d.kp[2][0] = 96.0f;
    d.kp[2][1] = 60.0f;
    struct hand_roi roi = hand_roi_from_detection(&d);
    bad += check(near(roi.rot, 0.0f, 1e-5f), "a hand pointing up gives rotation 0");
    bad += check(near(roi.side, 40.0f * 2.6f, 1e-3f), "side is the long edge times 2.6");
    bad += check(near(roi.cx, 96.0f, 1e-3f) && near(roi.cy, 96.0f - 10.0f, 1e-3f),
                 "centre shifts -0.5 * original height along y");
    /* rotate the keypoints a quarter turn: the ROI must follow */
    d.kp[0][0] = 100.0f;
    d.kp[0][1] = 96.0f;
    d.kp[2][0] = 60.0f;
    d.kp[2][1] = 96.0f;
    roi = hand_roi_from_detection(&d);
    bad += check(near(fabsf(roi.rot), (float)M_PI_2, 1e-5f), "a hand pointing left rotates 90 deg");

    /* landmarks -> ROI -> landmarks: decoding pushes the model's 224-space
     * output through the ROI affine, so a point at the tensor centre must
     * come back at the ROI centre whatever the rotation. */
    float raw[HAND_LANDMARKS * 3];
    for (int i = 0; i < HAND_LANDMARKS * 3; i++) { raw[i] = 112.0f; }
    struct hand_point lm[HAND_LANDMARKS];
    struct hand_roi r2 = {50.0f, 70.0f, 90.0f, 0.7f};
    hand_landmarks_decode(raw, r2, lm);
    bad += check(near(lm[0].x, 50.0f, 1e-3f) && near(lm[0].y, 70.0f, 1e-3f),
                 "landmark at the tensor centre decodes to the ROI centre");

    fprintf(stderr, "%s\n", bad ? "self-test FAILED" : "self-test ok");
    return bad;
}

/* ------------------------------------------------------------------ */
/* the frame stage                                                     */
/* ------------------------------------------------------------------ */

static void write_ppm(const char* path,
                      const float* t,
                      uint32_t w,
                      uint32_t h,
                      struct hand_norm n) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    const float span = n.hi - n.lo;
    for (size_t i = 0; i < (size_t)w * h * 3; i++) {
        float v = span != 0.0f ? (t[i] - n.lo) / span : 0.0f;
        v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
        fputc((int)(v * 255.0f + 0.5f), f);
    }
    fclose(f);
    fprintf(stderr, "wrote %s (%ux%u)\n", path, w, h);
}

/* The WebGPU EP's input on a driver that refuses Dawn's frame import:
 * the Vulkan pass writes a VK tensor, the registry's vk -> wgpu row
 * moves it into a WGPU tensor (hand_model_use_vk_feed's route, replayed
 * here without a model), and what comes back is held to the same bar as
 * the direct import -- the edge is part of the input now, so it is part
 * of the check. Two rounds (the first pays the import and the pipeline),
 * the second timed; `out` receives the WGPU tensor's floats. */
static int frame_to_tensor_via_vk_feed(struct infer_ctx* ctx,
                                       const struct frame_source* s,
                                       const struct options* o,
                                       const struct infer_desc* d,
                                       struct hand_affine a,
                                       float* out,
                                       double* us) {
    struct infer_registry reg;
    infer_registry_init(&reg);
    infer_registry_probe(&reg, ctx);
    const struct infer_edge* e = infer_registry_find(&reg, INFER_DOMAIN_VK, INFER_DOMAIN_WGPU);
    INFER_CHECK(e && e->available, "no vk -> wgpu edge on this machine");
    struct infer_tensor t_v, t_g;
    memset(&t_v, 0, sizeof t_v);
    memset(&t_g, 0, sizeof t_g);
    t_v.ready.sync_fd = t_v.released.sync_fd = -1;
    t_g.ready.sync_fd = t_g.released.sync_fd = -1;
    const int tensors_ok = infer_tensor_alloc(ctx, INFER_DOMAIN_VK, d, &t_v) == 0 &&
                           t_v.mem.vk.buffer && infer_tensor_alloc(ctx, INFER_DOMAIN_WGPU, d, &t_g) == 0;
    struct infer_frame idesc = hand_frame_import_desc(&s->frame);
    struct infer_vk_frame_import* vim = infer_vk_frame_import_create(&ctx->vk, &idesc);
    struct hand_frame_pass_vk* vp = vim ? hand_frame_pass_vk_create(&ctx->vk) : NULL;
    struct infer_edge_cache cache;
    infer_edge_cache_init(&cache);
    int rc = -1;
    if (!vim) {
        fprintf(stderr, "vk frame import refused\n");
    } else if (vp && tensors_ok) {
        rc = 0;
        for (int it = 0; it < 2 && rc == 0; it++) {
            uint64_t t0 = infer_now_ns();
            if (hand_frame_to_tensor_vk_tensor(&ctx->vk, vp, vim, &s->frame, a, o->norm,
                                               HAND_BORDER_ZERO, &t_v, NULL) < 0 ||
                infer_edge_apply(e, ctx, &cache, &t_v, &t_g) < 0 ||
                infer_tensor_wait_ready(ctx, &t_g) < 0) {
                rc = -1;
            }
            *us = (double)(infer_now_ns() - t0) / 1e3;
        }
        if (rc == 0 && infer_tensor_readback(ctx, &t_g, out) < 0) { rc = -1; }
    } else {
        fprintf(stderr, "vk feed: pass/tensor creation failed\n");
    }
    hand_frame_vk_wait(&ctx->vk, vp);
    infer_edge_cache_fini(&cache); /* Dawn's import of t_v holds its fd */
    infer_tensor_release(ctx, &t_g);
    infer_tensor_release(ctx, &t_v);
    hand_frame_pass_vk_destroy(&ctx->vk, vp);
    infer_vk_frame_import_destroy(vim);
    return rc;
}

/* Both FrameToTensor paths on one frame, then the difference. This is
 * the check that makes every later "the two EPs disagree" question
 * answerable: if the inputs match, the disagreement is the model's. */
static int stage_frame(struct frame_source* s, const struct options* o) {
    struct infer_ctx ctx;
    const int want_gpu = o->ep == INFER_EP_WEBGPU || o->compare || o->probe;
    unsigned want = want_gpu ? INFER_WANT_WGPU : 0u;
    /* the Vulkan pass (the CUDA EP's feed) is held to the same bar, and
     * it is also what says whether Dawn's import may be attempted at all
     * (and what feeds the WebGPU EP when not); its domain is best-effort
     * -- absent, the section reports and moves on */
    if (want_gpu) { want |= INFER_WANT_VK; }
    infer_ctx_init_all(&ctx, want);
    if (want_gpu && !(ctx.have & INFER_WANT_WGPU)) {
        fprintf(stderr, "no WebGPU domain: cannot run the GPU frame pass\n");
        return 1;
    }

    struct infer_desc d;
    hand_frame_tensor_desc(o->tensor, o->tensor, &d);
    const size_t n = infer_desc_elements(&d);
    struct hand_affine a = o->crop
                               ? hand_affine_square_crop(s->frame.width, s->frame.height, o->tensor, o->tensor)
                               : hand_affine_letterbox(s->frame.width, s->frame.height, o->tensor, o->tensor);

    struct infer_tensor t_cpu;
    memset(&t_cpu, 0, sizeof t_cpu);
    t_cpu.ready.sync_fd = t_cpu.released.sync_fd = -1;
    int bad = 0;
    float* cpu = NULL;
    float* gpu = NULL;

    if (infer_tensor_alloc(&ctx, INFER_DOMAIN_CPU, &d, &t_cpu) < 0) {
        fprintf(stderr, "cpu tensor alloc failed\n");
        bad = 1;
        goto out;
    }
    {
        uint64_t t0 = infer_now_ns();
        if (hand_frame_to_tensor_cpu(&s->frame, a, o->norm, HAND_BORDER_ZERO, &t_cpu) < 0) {
            bad = 1;
            goto out;
        }
        double us = (double)(infer_now_ns() - t0) / 1e3;
        cpu = malloc(n * sizeof(float));
        memcpy(cpu, t_cpu.mem.cpu.ptr, n * sizeof(float));
        fprintf(stderr, "frame -> tensor, cpu:    %7.1f us\n", us);
    }

    /* the Vulkan pass, before the Dawn one: on a driver whose NV12
     * import Dawn cannot take, this is still comparable */
    if ((o->compare || o->probe) && (ctx.have & INFER_WANT_VK) && cpu) {
        struct infer_frame idesc = hand_frame_import_desc(&s->frame);
        struct infer_vk_frame_import* vim = infer_vk_frame_import_create(&ctx.vk, &idesc);
        if (!vim) {
            fprintf(stderr, "vk frame import: REFUSED -- the CUDA EP must be fed from the host\n");
        } else {
            struct hand_frame_pass_vk* vp = hand_frame_pass_vk_create(&ctx.vk);
            struct infer_tensor t_v;
            memset(&t_v, 0, sizeof t_v);
            t_v.ready.sync_fd = t_v.released.sync_fd = -1;
            float* vkq = NULL;
            if (!vp || infer_tensor_alloc(&ctx, INFER_DOMAIN_VK, &d, &t_v) < 0) {
                fprintf(stderr, "vk pass/tensor creation failed\n");
                bad = 1;
            } else if (!t_v.mem.vk.buffer) {
                fprintf(stderr, "vk tensor has no buffer alias; vk pass not comparable here\n");
            } else {
                int vbad = 0;
                double us = 0;
                /* warm up once (pipeline, first acquire), then time */
                for (int it = 0; it < 2 && !vbad; it++) {
                    uint64_t t0 = infer_now_ns();
                    if (hand_frame_to_tensor_vk(&ctx.vk, vp, vim, &s->frame, a, o->norm,
                                                HAND_BORDER_ZERO, o->tensor, o->tensor,
                                                t_v.mem.vk.buffer, infer_desc_bytes_packed(&d),
                                                VK_NULL_HANDLE, NULL) < 0 ||
                        hand_frame_vk_wait(&ctx.vk, vp) < 0) {
                        vbad = 1;
                    }
                    us = (double)(infer_now_ns() - t0) / 1e3;
                }
                if (!vbad) {
                    vkq = malloc(n * sizeof(float));
                    if (infer_tensor_readback(&ctx, &t_v, vkq) < 0) { vbad = 1; }
                }
                if (vbad) {
                    bad = 1;
                } else {
                    fprintf(stderr, "frame -> tensor, vk:     %7.1f us (submit + wait)\n", us);
                    double err = 0;
                    size_t worst = 0;
                    for (size_t i = 0; i < n; i++) {
                        double e = fabs((double)cpu[i] - (double)vkq[i]);
                        if (e > err) {
                            err = e;
                            worst = i;
                        }
                    }
                    const double tol = 1.0 / 255.0;
                    fprintf(stderr,
                            "cpu vs vk tensor:        max |d| %.3g at element %zu (cpu %.5f, vk %.5f) -- %s\n",
                            err, worst, (double)cpu[worst], (double)vkq[worst],
                            err <= tol ? "ok" : "MISMATCH");
                    if (o->verbose) {
                        /* which failure is it: all-zero planes, one bad
                         * plane, or a resampling disagreement */
                        size_t nbad = 0;
                        float vmin = vkq[0], vmax = vkq[0];
                        for (size_t i = 0; i < n; i++) {
                            if (fabs((double)cpu[i] - (double)vkq[i]) > tol) { nbad++; }
                            if (vkq[i] < vmin) { vmin = vkq[i]; }
                            if (vkq[i] > vmax) { vmax = vkq[i]; }
                        }
                        const size_t mid = n / 2 - (n / 2) % 3;
                        fprintf(stderr,
                                "  vk: %zu/%zu off; vk range [%g, %g]\n"
                                "  el 0..2  vk (%.3f %.3f %.3f) cpu (%.3f %.3f %.3f)\n"
                                "  mid      vk (%.3f %.3f %.3f) cpu (%.3f %.3f %.3f)\n",
                                nbad, n, (double)vmin, (double)vmax, (double)vkq[0],
                                (double)vkq[1], (double)vkq[2], (double)cpu[0], (double)cpu[1],
                                (double)cpu[2], (double)vkq[mid], (double)vkq[mid + 1],
                                (double)vkq[mid + 2], (double)cpu[mid], (double)cpu[mid + 1],
                                (double)cpu[mid + 2]);
                    }
                    if (err > tol) { bad = 1; }
                }
            }
            free(vkq);
            infer_tensor_release(&ctx, &t_v);
            hand_frame_pass_vk_destroy(&ctx.vk, vp);
            infer_vk_frame_import_destroy(vim);
        }
    }

    if (want_gpu) {
        struct infer_frame idesc = hand_frame_import_desc(&s->frame);
        /* Our Vulkan device is asked first whether Dawn's import may even
         * be attempted: a driver that refuses it (NVIDIA 610.43, every
         * linear NV12 layout) takes the process or the device down with
         * it, so the question cannot be put to Dawn. Refused, the WebGPU
         * EP's input comes through the vk pass and the vk -> wgpu row --
         * and that is what gets compared, edge included. */
        if (infer_vk_frame_multiplanar_importable(&ctx.vk, &idesc) == 0) {
            fprintf(stderr,
                    "frame import:            REFUSED by the driver (vk probe); the WebGPU EP is "
                    "fed by the vk pass + the vk -> wgpu edge\n");
            gpu = malloc(n * sizeof(float));
            double us = 0;
            if (frame_to_tensor_via_vk_feed(&ctx, s, o, &d, a, gpu, &us) < 0) {
                bad = 1;
            } else {
                fprintf(stderr,
                        "frame -> tensor, webgpu: %7.1f us (vk pass + dmabuf_import, submit + wait)\n",
                        us);
            }
        } else {
            struct infer_wgpu_frame_import* im = infer_wgpu_frame_import_create(&ctx.wgpu, &idesc);
            if (!im) {
                /* not a bug in us: this driver will not let a shader read
                 * the camera's frame, so the WebGPU EP has to be fed from
                 * the host (the cpu -> wgpu write_buffer row). Reported,
                 * not hidden. */
                fprintf(stderr,
                        "frame import: REFUSED -- the WebGPU EP must be fed from the host\n");
                bad = 1;
                goto out;
            }
            fprintf(stderr, "frame import:            plane views, sampled in place\n");
            struct hand_frame_pass* pass = hand_frame_pass_create(&ctx.wgpu);
            struct infer_tensor t_gpu;
            memset(&t_gpu, 0, sizeof t_gpu);
            t_gpu.ready.sync_fd = t_gpu.released.sync_fd = -1;
            if (!pass || infer_tensor_alloc(&ctx, INFER_DOMAIN_WGPU, &d, &t_gpu) < 0) {
                fprintf(stderr, "gpu pass/tensor creation failed\n");
                bad = 1;
            } else {
                gpu = malloc(n * sizeof(float));
                /* warm up once (shader compile, first import access), then time */
                double us = 0;
                for (int it = 0; it < 2; it++) {
                    int rel = -1;
                    uint64_t t0 = infer_now_ns();
                    if (infer_wgpu_frame_begin(&ctx.wgpu, im, -1) < 0 ||
                        hand_frame_to_tensor_wgpu(&ctx.wgpu, pass, im, &s->frame, a, o->norm,
                                                  HAND_BORDER_ZERO, &t_gpu) < 0 ||
                        infer_wgpu_frame_end(&ctx.wgpu, im, &rel) < 0) {
                        bad = 1;
                        break;
                    }
                    infer_wgpu_wait_idle(&ctx.wgpu);
                    us = (double)(infer_now_ns() - t0) / 1e3;
                    if (rel >= 0) { close(rel); }
                }
                if (!bad && infer_tensor_readback(&ctx, &t_gpu, gpu) < 0) {
                    fprintf(stderr, "gpu tensor readback failed\n");
                    bad = 1;
                }
                if (!bad) {
                    fprintf(stderr, "frame -> tensor, webgpu: %7.1f us (submit + wait)\n", us);
                }
            }
            infer_tensor_release(&ctx, &t_gpu);
            hand_frame_pass_destroy(pass);
            infer_wgpu_frame_import_destroy(im);
        }
    }

    if (!bad && cpu && gpu) {
        double err = 0;
        size_t worst = 0;
        for (size_t i = 0; i < n; i++) {
            double e = fabs((double)cpu[i] - (double)gpu[i]);
            if (e > err) {
                err = e;
                worst = i;
            }
        }
        /* the two paths run the same formula on the same bytes, so they
         * agree to f32 rounding -- 1/255 (one 8-bit step) is the loosest
         * bar worth calling a pass, and anything near it means the chroma
         * siting or the affine differ, not the arithmetic */
        const double tol = 1.0 / 255.0;
        fprintf(stderr,
                "cpu vs webgpu tensor:    max |d| %.3g at element %zu (cpu %.5f, gpu %.5f) -- %s\n",
                err, worst, (double)cpu[worst], (double)gpu[worst], err <= tol ? "ok" : "MISMATCH");
        if (err > tol) { bad = 1; }
    }
    if (!bad && o->ppm_path) {
        write_ppm(o->ppm_path, gpu && o->ep == INFER_EP_WEBGPU ? gpu : cpu, o->tensor, o->tensor,
                  o->norm);
    }

out:
    free(cpu);
    free(gpu);
    infer_tensor_release(&ctx, &t_cpu);
    infer_ctx_fini_all(&ctx);
    return bad;
}

/* ------------------------------------------------------------------ */
/* the detector, and the landmark stage on top of it                   */
/* ------------------------------------------------------------------ */
/* One model's turn in a cycle, timed the way hello_inference times a
 * cell: the FrameToTensor pass, the engine call, and the wait for the
 * consumer's token, SEPARATELY. Keeping run and wait apart is the whole
 * point on the WebGPU EP, where they are ~0.4 ms of host submission and
 * ~6 ms of GPU -- their sum would say nothing about which resource the
 * inference actually costs. */
struct model_time {
    double pre, run, wait;
};

struct cycle_time {
    struct model_time palm, lmk;
    double decode; /* anchors, sigmoid, NMS, ROI, landmark decode */
};

/* The EP decides where the input comes from; everything after that is
 * identical, so the cycle body is written once. */
static int feed(struct hand_model* m,
                struct infer_wgpu_frame_import* im,
                struct infer_vk_frame_import* vim,
                const struct infer_frame* f,
                struct hand_affine a) {
    if (hand_model_wants_vk_frame(m)) { return hand_model_feed_vk(m, vim, f, a); }
    if (hand_model_ep(m) == INFER_EP_WEBGPU) { return hand_model_feed_wgpu(m, im, f, a); }
    return hand_model_feed_cpu(m, f, a);
}

static int one_model(struct hand_model* m,
                     struct infer_wgpu_frame_import* im,
                     struct infer_vk_frame_import* vim,
                     const struct infer_frame* f,
                     struct hand_affine a,
                     struct model_time* t) {
    uint64_t t0 = infer_now_ns();
    if (feed(m, im, vim, f, a) < 0) { return -1; }
    uint64_t t1 = infer_now_ns();
    if (hand_model_run(m) < 0) { return -1; }
    uint64_t t2 = infer_now_ns();
    if (hand_model_sync(m) < 0) { return -1; }
    uint64_t t3 = infer_now_ns();
    t->pre = (double)(t1 - t0) / 1e3;
    t->run = (double)(t2 - t1) / 1e3;
    t->wait = (double)(t3 - t2) / 1e3;
    return 0;
}

/* A ROI measured in DETECTOR pixels, carried over to FRAME pixels. The
 * fit affine is a uniform scale plus a translation, so the rotation and
 * the squareness survive untouched -- which is the entire reason the
 * detection arithmetic is done in detector space (hand.h). */
static struct hand_roi roi_to_frame(struct hand_roi r, struct hand_affine fit) {
    struct hand_roi f = r;
    hand_affine_apply(fit, r.cx, r.cy, &f.cx, &f.cy);
    f.side = r.side * hand_affine_scale(fit);
    return f;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double median(double* v, int n) {
    if (n <= 0) { return 0; }
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct pipeline_out {
    int hands;
    struct hand_det det[HAND_MAX];
    struct hand_point lm[HAND_LANDMARKS];
    struct hand_roi roi_frame;
    float presence, handedness;
    int have_landmarks;
};

static int run_models(struct frame_source* s, const struct options* o, struct pipeline_out* res) {
    struct infer_ctx ctx;
    const int gpu = o->ep == INFER_EP_WEBGPU; /* the dma-buf frame-import path */
    const enum infer_domain ep_dom = infer_engine_ort_input_domain(o->ep);
    unsigned want = ep_dom == INFER_DOMAIN_CPU ? 0u : 1u << ep_dom;
    /* the CUDA EP's frame pass runs on the Vulkan device, and the WebGPU
     * EP wants it for the question that comes before Dawn's import (and
     * as the feed when the answer is no); best effort either way */
    if (o->ep == INFER_EP_CUDA || o->ep == INFER_EP_WEBGPU) { want |= INFER_WANT_VK; }
    infer_ctx_init_all(&ctx, want);
    if (ep_dom != INFER_DOMAIN_CPU && !(ctx.have & (1u << ep_dom))) {
        fprintf(stderr, "no %s domain\n", infer_domain_name(ep_dom));
        return 1;
    }
    struct infer_registry reg;
    infer_registry_init(&reg);
    infer_registry_probe(&reg, &ctx);

    int bad = 0;
    struct hand_model* palm = NULL;
    struct hand_model* lmk = NULL;
    struct infer_wgpu_frame_import* im = NULL;
    struct infer_vk_frame_import* vim = NULL;
    static float anchors[HAND_PALM_ANCHORS * 2];
    struct hand_det det[64];
    struct cycle_time* t = NULL;
    const int total = o->warmup + o->iters;
    struct hand_roi chain_roi = {0, 0, 0, 0};
    int have_chain = 0;

    memset(res, 0, sizeof *res);
    palm = hand_model_create(&ctx, &reg, o->ep, o->palm_model, HAND_DETECT_SIDE,
                             HAND_BORDER_ZERO);
    if (!palm) {
        bad = 1;
        goto out;
    }
    if (o->stage == STAGE_LANDMARK) {
        lmk = hand_model_create(&ctx, &reg, o->ep, o->lmk_model, HAND_LANDMARK_SIDE,
                                HAND_BORDER_REPLICATE);
        if (!lmk) {
            bad = 1;
            goto out;
        }
    }
    if (gpu) {
        struct infer_frame idesc = hand_frame_import_desc(&s->frame);
        /* the tracker's decision (hand_tracker.c, choose_wgpu_route): our
         * Vulkan device is asked first whether Dawn's import may even be
         * attempted; refused, the vk pass feeds the WebGPU EP instead */
        if (infer_vk_frame_multiplanar_importable(&ctx.vk, &idesc) == 0) {
            if (hand_model_use_vk_feed(palm) < 0 || (lmk && hand_model_use_vk_feed(lmk) < 0)) {
                fprintf(stderr, "frame import refused by the driver, and no vk feed: the WebGPU "
                                "EP cannot be fed\n");
                bad = 1;
                goto out;
            }
        } else {
            im = infer_wgpu_frame_import_create(&ctx.wgpu, &idesc);
            if (!im) {
                fprintf(stderr, "frame import refused: the WebGPU EP would need a host feed\n");
                bad = 1;
                goto out;
            }
        }
    }
    if (palm && hand_model_wants_vk_frame(palm)) {
        struct infer_frame idesc = hand_frame_import_desc(&s->frame);
        vim = infer_vk_frame_import_create(&ctx.vk, &idesc);
        if (!vim) {
            /* the model came up expecting the device feed; a frame the
             * driver will not import is a real failure, not a fallback */
            fprintf(stderr, "vk frame import refused\n");
            bad = 1;
            goto out;
        }
    }
    hand_palm_anchors(anchors, HAND_PALM_ANCHORS);

    const struct hand_affine fit =
        o->crop ? hand_affine_square_crop(s->frame.width, s->frame.height, HAND_DETECT_SIDE,
                                          HAND_DETECT_SIDE)
                : hand_affine_letterbox(s->frame.width, s->frame.height, HAND_DETECT_SIDE,
                                        HAND_DETECT_SIDE);
    t = calloc((size_t)total, sizeof *t);

    for (int it = 0; it < total && !bad; it++) {
        int rel = -1;
        /* ONE access bracket per cycle: both FrameToTensor passes read the
         * same frame, so they share it. That is why infer_wgpu_frame_begin
         * and _end are separate calls (infer.h). */
        if (gpu && im && infer_wgpu_frame_begin(&ctx.wgpu, im, -1) < 0) {
            bad = 1;
            break;
        }
        if (one_model(palm, im, vim, &s->frame, fit, &t[it].palm) < 0) { bad = 1; }

        uint64_t d0 = infer_now_ns();
        int nkept = 0;
        if (!bad) {
            const int n = hand_palm_decode(hand_model_output(palm, 0), hand_model_output(palm, 1),
                                           anchors, HAND_PALM_ANCHORS, 0.5f, det, 64);
            nkept = hand_nms(det, n, 0.3f, HAND_MAX);
        }
        uint64_t d1 = infer_now_ns();
        t[it].decode = (double)(d1 - d0) / 1e3;

        if (!bad && lmk && nkept > 0) {
            /* --roi replays one exact landmark call: the ROI the tracker
             * was using when it rejected the hand, on the frame it was
             * looking at. No detector opinion involved. */
            struct hand_roi fr = o->have_roi ? o->roi
                                : have_chain ? chain_roi
                                             : roi_to_frame(hand_roi_from_detection(&det[0]), fit);
            struct hand_affine crop = hand_affine_roi(fr.cx, fr.cy, fr.side, fr.rot,
                                                      HAND_LANDMARK_SIDE, HAND_LANDMARK_SIDE);
            /* The single most useful thing to look at when the detector is
             * confident and the landmark model then reports no hand: what
             * the landmark model was actually handed. A crop that does not
             * contain an upright hand explains a low presence score
             * instantly, and nothing else does. */
            if (o->ppm_path && it == o->warmup) {
                struct infer_desc cd;
                hand_frame_tensor_desc(HAND_LANDMARK_SIDE, HAND_LANDMARK_SIDE, &cd);
                struct infer_tensor ct;
                memset(&ct, 0, sizeof ct);
                ct.ready.sync_fd = ct.released.sync_fd = -1;
                if (infer_tensor_alloc(&ctx, INFER_DOMAIN_CPU, &cd, &ct) == 0 &&
                    hand_frame_to_tensor_cpu(&s->frame, crop, HAND_NORM_RANGE,
                                             HAND_BORDER_REPLICATE, &ct) == 0) {
                    write_ppm(o->ppm_path, ct.mem.cpu.ptr, HAND_LANDMARK_SIDE, HAND_LANDMARK_SIDE,
                              HAND_NORM_RANGE);
                    fprintf(stderr,
                            "  roi: detector px (%.1f, %.1f) side %.1f rot %.3f rad\n"
                            "       frame px    (%.1f, %.1f) side %.1f\n",
                            (double)hand_roi_from_detection(&det[0]).cx,
                            (double)hand_roi_from_detection(&det[0]).cy,
                            (double)hand_roi_from_detection(&det[0]).side,
                            (double)fr.rot, (double)fr.cx, (double)fr.cy, (double)fr.side);
                }
                infer_tensor_release(&ctx, &ct);
            }
            if (one_model(lmk, im, vim, &s->frame, crop, &t[it].lmk) < 0) {
                bad = 1;
            } else {
                uint64_t d2 = infer_now_ns();
                hand_landmarks_decode(hand_model_output(lmk, 0), fr, res->lm);
                res->presence = hand_model_output(lmk, 1)[0];
                res->handedness = hand_model_output(lmk, 2)[0];
                /* --chain: walk the TRACKING path, which hello_hand
                 * otherwise never exercises. Iteration 0 uses the
                 * detector's ROI; every one after that uses the ROI
                 * derived from the previous iteration's landmarks, which
                 * is what the windowed tracker does from its second frame
                 * onwards. If presence collapses at step 1, the fault is
                 * hand_roi_from_landmarks and not the detector. */
                if (o->chain) {
                    struct hand_roi next = hand_roi_from_landmarks(res->lm);
                    fprintf(stderr,
                            "  chain %d: presence %.4f  roi (%.1f, %.1f) side %.1f rot %+.3f"
                            "  -> next (%.1f, %.1f) side %.1f rot %+.3f\n",
                            it, (double)res->presence, (double)fr.cx, (double)fr.cy,
                            (double)fr.side, (double)fr.rot, (double)next.cx, (double)next.cy,
                            (double)next.side, (double)next.rot);
                    chain_roi = next;
                    have_chain = 1;
                }
                if (o->verbose && it == o->warmup) {
                    /* ALL of the landmark model's outputs, raw. Which one
                     * is "presence" is an assumption about how tf2onnx
                     * ordered the graph's Identity_N, and an assumption
                     * is exactly what a confidence an order of magnitude
                     * below MediaPipe's calls into question. */
                    fprintf(stderr, "  landmark model raw outputs:\n");
                    for (size_t k = 0; k < hand_model_output_count(lmk); k++) {
                        const float* v = hand_model_output(lmk, (int)k);
                        const size_t ne = hand_model_output_elements(lmk, (int)k);
                        fprintf(stderr, "    out %zu: %zu elem  ", k, ne);
                        for (size_t e = 0; e < (ne < 6 ? ne : 6); e++) {
                            fprintf(stderr, "%9.4f", (double)v[e]);
                        }
                        if (ne > 6) { fprintf(stderr, "  ..."); }
                        fprintf(stderr, "\n");
                    }
                }
                res->roi_frame = fr;
                res->have_landmarks = 1;
                t[it].decode += (double)(infer_now_ns() - d2) / 1e3;
            }
        }
        if (gpu && im) {
            infer_wgpu_frame_end(&ctx.wgpu, im, &rel);
            if (rel >= 0) { close(rel); }
        }
        res->hands = nkept;
        for (int i = 0; i < nkept && i < HAND_MAX; i++) { res->det[i] = det[i]; }
    }

    if (!bad) {
        /* medians over the post-warmup runs, as hello_inference does: the
         * first WebGPU run pays for shader compilation and the graph
         * capture, and averaging that in would flatter nothing */
        double* v = calloc((size_t)o->iters, sizeof *v);
        struct cycle_time med;
#define MED(field)                                                            \
    do {                                                                      \
        for (int i = 0; i < o->iters; i++) { v[i] = t[o->warmup + i].field; }  \
        med.field = median(v, o->iters);                                       \
    } while (0)
        MED(palm.pre);
        MED(palm.run);
        MED(palm.wait);
        MED(lmk.pre);
        MED(lmk.run);
        MED(lmk.wait);
        MED(decode);
#undef MED
        free(v);

        printf("%-9s %-7s %-9s %7s %7s %7s\n", "model", "ep", "out_edge", "pre", "run", "wait");
        printf("%-9s %-7s %-9s %7.0f %7.0f %7.0f\n", "palm", infer_ep_name(o->ep),
               hand_model_out_edge(palm), med.palm.pre, med.palm.run, med.palm.wait);
        if (lmk) {
            printf("%-9s %-7s %-9s %7.0f %7.0f %7.0f\n", "landmark", infer_ep_name(o->ep),
                   hand_model_out_edge(lmk), med.lmk.pre, med.lmk.run, med.lmk.wait);
        }
        printf("%-9s %-7s %-9s %7.0f\n", "decode", "host", "-", med.decode);
        printf("%-9s %-7s %-9s %7.0f  us per cycle (medians of %d, %d warmup)\n", "TOTAL", "", "",
               med.palm.pre + med.palm.run + med.palm.wait + med.lmk.pre + med.lmk.run +
                   med.lmk.wait + med.decode,
               o->iters, o->warmup);
        printf("\n(run is the engine call -- submission only on the WebGPU EP; wait is the\n"
               "consumer's ready token, the part the call returning does not cover.)\n");

        printf("\n%d hand(s) in a %ux%u frame, detector %s\n", res->hands, s->frame.width,
               s->frame.height, o->crop ? "square-crop" : "letterbox");
        for (int i = 0; i < res->hands; i++) {
            float cx, cy;
            hand_affine_apply(fit, res->det[i].cx, res->det[i].cy, &cx, &cy);
            const float sc = hand_affine_scale(fit);
            printf("  palm %d: score %.3f  box (%.3f, %.3f) %.3f x %.3f\n", i, res->det[i].score,
                   cx / (float)s->frame.width, cy / (float)s->frame.height,
                   res->det[i].w * sc / (float)s->frame.width,
                   res->det[i].h * sc / (float)s->frame.height);
        }
        if (res->have_landmarks) {
            printf("\n  presence %.3f  handedness %.3f (>0.5 = right, as seen)\n", res->presence,
                   res->handedness);
            if (o->verbose) {
                for (int i = 0; i < HAND_LANDMARKS; i++) {
                    printf("    lm %2d (%.3f, %.3f, %+.3f)\n", i,
                           res->lm[i].x / (float)s->frame.width,
                           res->lm[i].y / (float)s->frame.height,
                           res->lm[i].z / (float)s->frame.width);
                }
            }
        }
        /* "no hand" and "the decode is wrong" look identical from the
         * outside, so always report the score distribution: a broken input
         * or a transposed regressor leaves the max score in the noise
         * (~1e-3), while a hand that is merely small or clipped scores
         * respectably and just missed the threshold. */
        {
            const float* sc = hand_model_output(palm, 1);
            float best = -1e30f;
            int best_i = 0, over[3] = {0, 0, 0};
            for (int i = 0; i < HAND_PALM_ANCHORS; i++) {
                const float p = 1.0f / (1.0f + expf(-sc[i]));
                if (p > best) {
                    best = p;
                    best_i = i;
                }
                if (p > 0.1f) { over[0]++; }
                if (p > 0.3f) { over[1]++; }
                if (p > 0.5f) { over[2]++; }
            }
            printf("  scores: max %.4f at anchor %d (centre %.3f, %.3f); over 0.1/0.3/0.5: %d/%d/%d\n",
                   best, best_i, anchors[best_i * 2], anchors[best_i * 2 + 1], over[0], over[1],
                   over[2]);
        }
        if (res->hands == 0) {
            fprintf(stderr,
                    "\nno hand over threshold. If one was in shot: a 16:9 frame letterboxed into\n"
                    "192x192 leaves only 108 useful rows, so try --crop, or CAM_SIZE=640x480.\n"
                    "A max score in the noise (~1e-3) instead means the INPUT or the decode is\n"
                    "wrong, not the scene -- check --stage frame --ppm first.\n");
            /* Against the committed fixture, finding nothing IS the
             * failure -- that frame has a hand in it. Against a live
             * camera it means only that nobody waved. */
            if (o->strict && o->frame_path) { bad = 1; }
        }
        /* A detection the landmark model rejects is not a hand, and the
         * palm detector fires on faces readily -- so "something was
         * detected" is a much weaker claim than it looks. --require-hand
         * makes the two models having to AGREE part of the exit code; it
         * is off by default because the committed fixture deliberately
         * does not contain a reachable hand (see testdata/README.md), and
         * is still a perfectly good deterministic input for comparing the
         * two providers, which is what it is there for. */
        if (o->require_hand && res->hands > 0 && lmk &&
            !(res->have_landmarks && res->presence >= 0.5f)) {
            fprintf(stderr,
                    "\ndetected something at score %.3f, but presence is %.3f: the landmark\n"
                    "model says that is not a hand.\n",
                    (double)res->det[0].score, (double)res->presence);
            bad = 1;
        }
        /* Keep the frame that worked. A live camera shows something
         * different every run, which is fine for a demo and useless for a
         * test, so the fixture ctest uses is captured the moment a real
         * detection happens rather than staged. */
        if (o->dump_path && res->hands > 0 &&
            (!lmk || (res->have_landmarks && res->presence >= 0.5f)) && s->frame.map) {
            const size_t bytes = (size_t)s->frame.pitch[0] * s->frame.height * 3 / 2;
            FILE* fp = fopen(o->dump_path, "wb");
            if (fp) {
                fwrite(s->frame.map, 1, bytes, fp);
                fclose(fp);
                fprintf(stderr, "dumped %zu bytes of NV12 %ux%u (stride %u) to %s\n", bytes,
                        s->frame.width, s->frame.height, s->frame.pitch[0], o->dump_path);
            } else {
                fprintf(stderr, "cannot write %s\n", o->dump_path);
            }
        }
    }

out:
    free(t);
    infer_wgpu_frame_import_destroy(im);
    hand_model_destroy(lmk);
    hand_model_destroy(palm);
    infer_vk_frame_import_destroy(vim); /* after the models: their pass fences */
    infer_ctx_fini_all(&ctx);
    return bad;
}

/* ------------------------------------------------------------------ */

static void usage(void) {
    fprintf(stderr,
            "usage: hello_hand [--ep cpu|webgpu|cuda] [--all] [--stage frame|detect|landmark]\n"
            "                  [--self-test] [--probe] [--compare-frame-to-tensor]\n"
            "                  [--frame raw.nv12 --size WxH] [--size WxH] [--tensor N]\n"
            "                  [--crop] [--range 0|1] [--ppm out.ppm] [--iters N]\n"
            "                  [--strict] [--verbose]\n"
            "       --all runs the cpu EP and then every GPU EP on the same frame and\n"
            "       compares what they found (the acceptance test).\n"
            "       --frame reads a raw NV12 fixture (stride == width) instead of the\n"
            "       camera; --crop centre-crops instead of letterboxing; --range 1\n"
            "       feeds [-1,1] instead of [0,1]; --tensor sets the model input side.\n");
}

int main(int argc, char** argv) {
    struct options o = {
        .ep = INFER_EP_CPU,
        .stage = STAGE_FRAME,
        /* full, like the tracker (and MediaPipe's web demo); the ctest
         * entries pass --palm/--landmark explicitly so the committed
         * golden values stay pinned to the lite pair */
        .palm_model = "models/palm_detection_full.onnx",
        .lmk_model = "models/hand_landmark_full.onnx",
        .tensor = 192,
        .iters = 10,
        /* the WebGPU EP's first run pays for shader compilation and the
         * graph capture; two throwaways is enough to leave it behind */
        .warmup = 2,
        .norm = HAND_NORM_RANGE,
    };
    for (int i = 1; i < argc; i++) {
        const char* s = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(s, "--self-test")) {
            o.self_test = 1;
        } else if (!strcmp(s, "--probe")) {
            o.probe = 1;
            o.compare = 1;
        } else if (!strcmp(s, "--compare-frame-to-tensor")) {
            o.compare = 1;
        } else if (!strcmp(s, "--crop")) {
            o.crop = 1;
        } else if (!strcmp(s, "--roi") && v) {
            if (sscanf(v, "%f,%f,%f,%f", &o.roi.cx, &o.roi.cy, &o.roi.side, &o.roi.rot) != 4) {
                usage();
                return 2;
            }
            o.have_roi = 1;
            o.stage = STAGE_LANDMARK;
            i++;
        } else if (!strcmp(s, "--chain")) {
            o.chain = 1;
        } else if (!strcmp(s, "--require-hand")) {
            o.require_hand = 1;
        } else if (!strcmp(s, "--all")) {
            o.all = 1;
            if (o.stage == STAGE_FRAME) { o.stage = STAGE_LANDMARK; }
        } else if (!strcmp(s, "--strict")) {
            o.strict = 1;
        } else if (!strcmp(s, "--verbose")) {
            o.verbose = 1;
        } else if (!strcmp(s, "--ep") && v) {
            if (infer_ep_parse(v, &o.ep) < 0) {
                usage();
                return 2;
            }
            i++;
        } else if (!strcmp(s, "--stage") && v) {
            if (!strcmp(v, "frame")) {
                o.stage = STAGE_FRAME;
            } else if (!strcmp(v, "detect")) {
                o.stage = STAGE_DETECT;
            } else if (!strcmp(v, "landmark")) {
                o.stage = STAGE_LANDMARK;
            } else {
                usage();
                return 2;
            }
            i++;
        } else if (!strcmp(s, "--frame") && v) {
            o.frame_path = v;
            i++;
        } else if (!strcmp(s, "--ppm") && v) {
            o.ppm_path = v;
            i++;
        } else if (!strcmp(s, "--dump-frame") && v) {
            o.dump_path = v;
            i++;
        } else if (!strcmp(s, "--palm") && v) {
            o.palm_model = v;
            i++;
        } else if (!strcmp(s, "--landmark") && v) {
            o.lmk_model = v;
            i++;
        } else if (!strcmp(s, "--full")) {
            o.palm_model = "models/palm_detection_full.onnx";
            o.lmk_model = "models/hand_landmark_full.onnx";
        } else if (!strcmp(s, "--lite")) {
            o.palm_model = "models/palm_detection_lite.onnx";
            o.lmk_model = "models/hand_landmark_lite.onnx";
        } else if (!strcmp(s, "--size") && v) {
            if (sscanf(v, "%ux%u", &o.want_w, &o.want_h) != 2) {
                usage();
                return 2;
            }
            i++;
        } else if (!strcmp(s, "--tensor") && v) {
            o.tensor = (uint32_t)atoi(v);
            i++;
        } else if (!strcmp(s, "--iters") && v) {
            o.iters = atoi(v);
            i++;
        } else if (!strcmp(s, "--warmup") && v) {
            o.warmup = atoi(v);
            i++;
        } else if (!strcmp(s, "--range") && v) {
            o.norm = atoi(v) ? (struct hand_norm){-1.0f, 1.0f} : (struct hand_norm){0.0f, 1.0f};
            i++;
        } else {
            usage();
            return 2;
        }
    }
    infer_verbose = o.verbose;

    int bad = 0;
    if (o.self_test) { bad += self_test(); }

    /* --self-test alone needs no frame at all: it is the part of stage
     * five that has no dependencies, so it can run anywhere */
    if (!o.self_test || o.probe || o.compare || o.ppm_path || o.stage != STAGE_FRAME) {
        struct frame_source src;
        if (source_open(&src, &o) < 0) {
            source_close(&src);
            return o.strict ? 1 : 0; /* no camera is not a test failure */
        }
        if (o.stage == STAGE_FRAME) {
            bad += stage_frame(&src, &o);
        } else if (o.all) {
            /* The acceptance test: the same frame through the CPU
             * provider and every GPU provider must find the same hand.
             * fp32 kernels in a different order on a different device
             * will not agree bit for bit, but they must agree to far
             * less than a pixel -- and if they disagree about WHETHER
             * there is a hand, the threshold is sitting on a knife edge
             * and that is worth saying out loud. */
            struct options a = o;
            a.ep = INFER_EP_CPU;
            struct pipeline_out ra;
            printf("---- cpu EP ----\n");
            const int fa = run_models(&src, &a, &ra);
            bad += fa;
            for (int e = 0; e < INFER_EP_COUNT; e++) {
                if (e == INFER_EP_CPU) { continue; }
                if (e == INFER_EP_CUDA && !infer_cuda_compiled()) {
                    printf("\n---- cuda EP: built without CUDA, skipped ----\n");
                    continue;
                }
                struct options b = o;
                b.ep = (enum infer_ep)e;
                struct pipeline_out rb;
                printf("\n---- %s EP ----\n", infer_ep_name(b.ep));
                const int fb = run_models(&src, &b, &rb);
                bad += fb;
                printf("\n---- agreement: cpu vs %s ----\n", infer_ep_name(b.ep));
                if (fa || fb) {
                    printf("  one of the providers failed; nothing to compare\n");
                    continue;
                }
                if (ra.hands != rb.hands) {
                    printf("  DISAGREE on the hand count: cpu %d, %s %d\n", ra.hands,
                           infer_ep_name(b.ep), rb.hands);
                    bad++;
                    continue;
                }
                if (ra.hands == 0) {
                    printf("  both found no hand (nothing compared)\n");
                    continue;
                }
                double dbox = 0, dlm = 0;
                for (int i = 0; i < ra.hands; i++) {
                    dbox = fmax(dbox, fabs((double)ra.det[i].cx - rb.det[i].cx));
                    dbox = fmax(dbox, fabs((double)ra.det[i].cy - rb.det[i].cy));
                }
                if (ra.have_landmarks && rb.have_landmarks) {
                    for (int i = 0; i < HAND_LANDMARKS; i++) {
                        dlm = fmax(dlm, fabs((double)ra.lm[i].x - rb.lm[i].x));
                        dlm = fmax(dlm, fabs((double)ra.lm[i].y - rb.lm[i].y));
                    }
                }
                /* a pixel of the DETECTOR input is 3.3 frame pixels here,
                 * so a tenth of a detector pixel is a tight bar */
                const double box_tol = 0.1, lm_tol = 1.0;
                printf("  palm score   cpu %.4f  %s %.4f\n", ra.det[0].score, infer_ep_name(b.ep),
                       rb.det[0].score);
                printf("  box centre   max |d| %.4f detector px  (bar %.2f) -- %s\n", dbox, box_tol,
                       dbox <= box_tol ? "ok" : "MISMATCH");
                if (ra.have_landmarks) {
                    printf("  landmarks    max |d| %.4f frame px     (bar %.2f) -- %s\n", dlm,
                           lm_tol, dlm <= lm_tol ? "ok" : "MISMATCH");
                    printf("  presence     cpu %.4f  %s %.4f\n", ra.presence, infer_ep_name(b.ep),
                           rb.presence);
                }
                if (dbox > box_tol || (ra.have_landmarks && dlm > lm_tol)) { bad++; }
            }
        } else {
            struct pipeline_out res;
            bad += run_models(&src, &o, &res);
        }
        source_close(&src);
    }
    if (bad) { fprintf(stderr, "%d check(s) failed\n", bad); }
    return bad && o.strict ? 1 : 0;
}
