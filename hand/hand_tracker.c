/* hand_tracker.c -- see hand_tracker.h.
 *
 * The tracking state machine here is MediaPipe's, and it is the reason
 * this is affordable at all: the detector runs only when the track is
 * lost (or every detect_every frames, to notice a second hand), and the
 * steady state is one landmark inference on a region of interest derived
 * from the PREVIOUS frame's landmarks. Measured on this machine that is
 * ~6.7 ms instead of ~17 ms per cycle on the CPU EP, and ~6.5 ms instead
 * of ~14.5 ms of GPU on the WebGPU EP.
 */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hand_model.h"
#include "hand_tracker.h"
#include "infer_util.h"

enum { STAT_WINDOW = 32 };

/* ONE threshold, 0.5, and no hysteresis -- because that is what
 * MediaPipe does, checked rather than guessed:
 *
 *   hand_landmark_cpu.pbtxt:
 *     ThresholdingCalculator { threshold: 0.5 }  -> FLAG:hand_presence
 *   and three GateCalculators, so the landmarks, the handedness AND the
 *   world landmarks are all gated on that one flag.
 *
 * Below the threshold MediaPipe emits NOTHING for that hand: no
 * landmarks, therefore no ROI, therefore PreviousLoopbackCalculator has
 * nothing to loop and the next frame's GateCalculator lets palm
 * detection run again. "Lost" and "not confident" are the same state.
 *
 * An earlier version here had a second, lower bar to KEEP a track (0.3),
 * on the theory that MediaPipe's separate min_detection_confidence and
 * min_tracking_confidence meant hysteresis. They do not -- both feed
 * thresholds of 0.5 -- and the invention did real damage: it kept a
 * failing track alive while its ROI, re-derived each frame at 2.0x the
 * landmarks' extent, doubled away from the hand (measured at 300 -> 900
 * -> 1475 px on a 640x360 frame). A track that is not confident enough
 * to aim the next crop is not a track. */
#define HAND_PRESENCE_THRESHOLD 0.5f

/* Cycles to sit out after the landmark model rejects a detection.
 *
 * 0 = MediaPipe's behaviour, and the default for that reason: its
 * GateCalculator lets palm detection run on every frame that produced no
 * landmarks, so a static false positive costs a detector inference per
 * frame and MediaPipe simply pays it. The palm detector does fire on
 * faces (0.70 on one here, re-proposed and re-rejected identically to
 * four decimal places), so paying it is measurable -- ~10 ms CPU or ~8 ms
 * GPU per cycle for an answer that has not changed.
 *
 * Raising this trades re-acquisition latency for that: N cycles of
 * silence after a rejection, so a hand actually arriving is picked up
 * ~N/30 s late. Left at 0 because matching MediaPipe matters more here
 * than the saving, and because a slower re-acquire is itself visible as
 * a gap in the overlay. */
#define HAND_REJECT_COOLDOWN 0

/* One provider's pair of sessions. Built once; graph capture pins the
 * tensors inside, so nothing here may be rebuilt on an EP switch --
 * which is exactly why `both` builds both up front. */
struct provider {
    struct hand_model* palm;
    struct hand_model* lmk;
    int ready;
};

struct stats {
    double pre, run, wait, decode; /* rolling sums over `n` cycles */
    double cycle;
    int n;
    unsigned cycles, detector_runs, hands;
};

struct hand_tracker {
    struct hand_options opt;
    uint32_t fw, fh;
    struct hand_affine fit;

    struct infer_ctx ctx;
    struct infer_registry reg;
    struct provider prov[INFER_EP_COUNT];
    /* one import per camera buffer, cached for the camera's lifetime: the
     * import is the expensive part, and the six of them are the producer
     * side of the FrameToTensor pass (the engine's tensors stay fixed).
     * Two flavours, one per device pass: Dawn's (the WebGPU EP) and the
     * headless Vulkan one (the CUDA EP's feed). */
    struct infer_wgpu_frame_import* import[HAND_MAX_FRAME_IMPORTS];
    int import_fd[HAND_MAX_FRAME_IMPORTS];
    struct infer_vk_frame_import* vimport[HAND_MAX_FRAME_IMPORTS];
    int vimport_fd[HAND_MAX_FRAME_IMPORTS];
    /* how the WebGPU provider gets its frame: 0 = not decided yet (the
     * first frame decides), 1 = Dawn's import, 2 = the vk pass + the
     * vk -> wgpu row (choose_wgpu_route) */
    int wgpu_route;

    /* the request slot: written by the render thread, read by the worker */
    pthread_mutex_t lock;
    pthread_cond_t wake;
    struct infer_frame req_frame;
    int req_index;   /* -1 = empty */
    uint32_t req_seq;
    int busy_index;  /* the buffer the worker is reading, -1 = none */
    int quit;

    /* the result slots: a seqlock. The worker writes slot (gen>>1)&1 and
     * bumps `gen`; the reader retries while it moved or is odd. No mutex,
     * no allocation, nothing for the render thread to block on. */
    struct hand_results slot[2];
    _Atomic uint32_t gen;
    _Atomic uint32_t consumed;

    _Atomic int want_ep;
    _Atomic int holds_index;

    /* Tracking state, worker-private: MediaPipe's
     * prev_hand_rects_from_landmarks, the vector that
     * PreviousLoopbackCalculator carries from one frame to the next.
     * Its SIZE against num_hands is what gates the detector. */
    struct hand_roi prev[HAND_MAX];
    int prev_count;
    int since_detect;

    pthread_mutex_t stat_lock;
    struct stats st;

    pthread_t thread;
    int thread_started;
};

/* ------------------------------------------------------------------ */
/* options                                                            */
/* ------------------------------------------------------------------ */

static void options_usage(void) {
    fprintf(stderr,
            "  --ep cpu|webgpu|cuda|both|all\n"
            "                         which execution provider (HAND_EP sets the default);\n"
            "                         `both` builds cpu + webgpu, `all` every provider that\n"
            "                         comes up -- extras stand ready so P switches instantly\n"
            "  --crop / --letterbox   how the frame reaches the 192x192 detector. Crop is the\n"
            "                         default: MediaPipe letterboxes, but on a 16:9 camera that\n"
            "                         spends 84 of 192 rows on bars (score 0.26 vs 0.76 here)\n"
            "  --detect-only          palm boxes and keypoints, no landmark stage\n"
            "  --overlay-test         draw a synthetic hand instead of running the models,\n"
            "                         so the overlay's alignment can be checked without one\n"
            "  --hands N              how many hands to look for (default 1, max 2). This is\n"
            "                         MediaPipe's num_hands, and it gates the detector: while\n"
            "                         fewer than N are tracked the detector runs EVERY cycle,\n"
            "                         and the association keeps the tracked ROI over any\n"
            "                         detection that overlaps it. 2 therefore costs a detector\n"
            "                         plus TWO landmark runs per cycle and gives every false\n"
            "                         positive a landmark run; 1 stops detecting once a hand is\n"
            "                         held\n"
            "  --detect-every N       ALSO re-run the detector every N cycles while tracking.\n"
            "                         Default 0 = never: a re-anchor swaps a tight ROI for the\n"
            "                         detector's coarser one and can drop the track (that was a\n"
            "                         1 Hz blink), and with one track slot it finds no 2nd hand\n"
            "  --full / --lite        which model pair. Full is the default -- it is what\n"
            "                         MediaPipe's web demo runs, and the lite landmark model\n"
            "                         is noisier around the 0.5 presence gate\n"
            "  --palm PATH --landmark PATH\n"
            "  --verbose              a timing line on stderr once a second\n");
}

int hand_options_parse(int argc, char** argv, struct hand_options* o) {
    memset(o, 0, sizeof *o);
    o->ep = INFER_EP_CPU;
    /* Centre-crop by default, --letterbox to opt out. MediaPipe's own
     * preprocessing is the letterbox, and on a 4:3 phone camera that is
     * the right choice -- but letterboxing a 16:9 webcam into a square
     * 192x192 spends 84 of 192 rows on black bars, and a hand then has to
     * be very large before the detector sees it. Measured on this
     * machine, same hand, same frame: best score 0.26 letterboxed
     * (nothing detected) against 0.76 cropped. A default that usually
     * finds nothing is not a defensible default, however faithful. */
    o->crop = 1;
    /* 0 = never re-anchor: the detector runs ONLY when the track is lost,
     * which is what MediaPipe's graph does.
     *
     * This was 30, and that was a bug with a one-second period. A
     * re-anchor replaces a tight, landmark-derived ROI with the
     * detector's coarser box; the landmark model then does worse on that
     * crop, presence dips under the bar, and the track drops -- so the
     * skeleton blinked once a second with the hand held still, at
     * whatever rate detect_every divided by the cycle rate (~30/s here,
     * hence ~1 Hz).
     *
     * The re-anchor had no upside to pay for that: only one ROI is
     * tracked, so a periodic detection cannot pick up a second hand
     * anyway. Set it non-zero to experiment; a real multi-hand version
     * would need a second track slot before it means anything. */
    /* 1, not MediaPipe's usual 2, and the reason is worth stating.
     * num_hands gates the detector: while fewer than num_hands are
     * tracked it runs on EVERY frame, and every rect it proposes gets a
     * landmark inference. With 2 and one hand up, that is a detector plus
     * two landmark runs per cycle (~24 ms), and the palm detector's
     * favourite false positive -- a face, always in shot -- gets a
     * landmark run every frame. When one of those happens to pass while
     * the real hand's run does not, the overlay draws the wrong skeleton;
     * the picture then jumps between the hand, junk, and nothing, which
     * is worse than either. With 1 the detector stops the moment a hand
     * is held, no candidate but the tracked one is ever evaluated, and a
     * cycle is one ~7 ms inference. --hands 2 for two hands and that
     * cost. */
    o->num_hands = 1;
    o->detect_every = 0;
    /* The _full models, matching MediaPipe's own web demo (the legacy JS
     * solution defaults to modelComplexity 1 = full; the Tasks
     * hand_landmarker.task bundles the full landmark model too). The
     * _lite landmark model is noticeably noisier around the 0.5 presence
     * gate, which reads as flicker no tracking logic can remove;
     * comparing against the demo is only honest on the same weights.
     * --lite for the cheaper pair. */
    o->palm_model = "models/palm_detection_full.onnx";
    o->lmk_model = "models/hand_landmark_full.onnx";
    const char* env = getenv("HAND_EP");
    if (env) {
        if (!strcmp(env, "both")) {
            o->both = 1;
        } else if (!strcmp(env, "all")) {
            o->all = 1;
        } else if (infer_ep_parse(env, &o->ep) < 0) {
            fprintf(stderr, "HAND_EP must be cpu, webgpu, cuda, both or all\n");
            return -1;
        }
    }
    for (int i = 1; i < argc; i++) {
        const char* s = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(s, "--ep") && v) {
            if (!strcmp(v, "both")) {
                o->both = 1;
                o->ep = INFER_EP_CPU;
            } else if (!strcmp(v, "all")) {
                o->all = 1;
                o->ep = INFER_EP_CPU;
            } else if (infer_ep_parse(v, &o->ep) < 0) {
                options_usage();
                return -1;
            }
            i++;
        } else if (!strcmp(s, "--crop")) {
            o->crop = 1;
        } else if (!strcmp(s, "--letterbox")) {
            o->crop = 0;
        } else if (!strcmp(s, "--detect-only")) {
            o->detect_only = 1;
        } else if (!strcmp(s, "--overlay-test")) {
            o->overlay_test = 1;
        } else if (!strcmp(s, "--verbose")) {
            o->verbose = 1;
        } else if (!strcmp(s, "--hands") && v) {
            o->num_hands = atoi(v);
            if (o->num_hands < 1) { o->num_hands = 1; }
            if (o->num_hands > HAND_MAX) { o->num_hands = HAND_MAX; }
            i++;
        } else if (!strcmp(s, "--detect-every") && v) {
            o->detect_every = atoi(v);
            i++;
        } else if (!strcmp(s, "--full")) {
            o->palm_model = "models/palm_detection_full.onnx";
            o->lmk_model = "models/hand_landmark_full.onnx";
        } else if (!strcmp(s, "--lite")) {
            o->palm_model = "models/palm_detection_lite.onnx";
            o->lmk_model = "models/hand_landmark_lite.onnx";
        } else if (!strcmp(s, "--palm") && v) {
            o->palm_model = v;
            i++;
        } else if (!strcmp(s, "--landmark") && v) {
            o->lmk_model = v;
            i++;
        } else {
            fprintf(stderr, "unknown option: %s\n", s);
            options_usage();
            return -1;
        }
    }
    if (o->detect_every < 0) { o->detect_every = 0; } /* 0 = only when lost */
    return 0;
}

/* ------------------------------------------------------------------ */
/* the synthetic hand, for testing the overlay without a hand          */
/* ------------------------------------------------------------------ */

void hand_results_synthetic(struct hand_results* out, double phase) {
    memset(out, 0, sizeof *out);
    out->count = 1;
    out->ep = INFER_EP_CPU;
    out->ran_detector = 1;
    struct hand_result* h = &out->hand[0];
    h->score = 1.0f;
    h->presence = 1.0f;
    h->handedness = 1.0f;
    h->from_detector = 1;

    /* a splayed right hand, palm to camera: the wrist low and centred,
     * five fingers fanning upward. Written as (angle, length) per finger
     * so it stays recognisable rather than a table of magic numbers. */
    const float cx = 0.5f + 0.06f * (float)sin(phase);
    const float cy = 0.72f;
    const float s = 0.085f; /* one phalanx, in fractions of frame width */
    /* thumb, index, middle, ring, little: direction in radians (0 = right,
     * -pi/2 = up) and how long each bone is relative to s */
    static const float dir[5] = {-2.60f, -1.85f, -1.57f, -1.30f, -1.05f};
    static const float len[5] = {0.80f, 1.05f, 1.15f, 1.00f, 0.80f};

    h->lm[0].x = cx;
    h->lm[0].y = cy; /* wrist */
    for (int f = 0; f < 5; f++) {
        /* MCP joints sit on a shallow arc above the wrist */
        const float a = dir[f], l = len[f] * s;
        float px = cx + cosf(a) * l * 0.6f;
        float py = cy + sinf(a) * l * 0.6f;
        for (int j = 0; j < 4; j++) {
            const int idx = 1 + f * 4 + j;
            /* fingers curl very slightly as they extend */
            const float aj = a + 0.06f * (float)j;
            px += cosf(aj) * l * 0.55f;
            py += sinf(aj) * l * 0.55f;
            h->lm[idx].x = px;
            h->lm[idx].y = py;
            h->lm[idx].z = 0.0f;
        }
    }
    /* the palm keypoints the detector would have produced, and the ROI */
    for (int k = 0; k < HAND_PALM_KP; k++) {
        h->palm_kp[k][0] = cx + 0.05f * cosf((float)k * 1.2f);
        h->palm_kp[k][1] = cy - 0.05f - 0.04f * sinf((float)k * 1.2f);
    }
    h->roi = (struct hand_roi){cx, cy - 0.16f, 0.42f, 0.0f};
}

/* ------------------------------------------------------------------ */
/* the worker                                                         */
/* ------------------------------------------------------------------ */

static struct infer_wgpu_frame_import* import_for(struct hand_tracker* h,
                                                  const struct infer_frame* f,
                                                  int index) {
    if (index < 0 || index >= HAND_MAX_FRAME_IMPORTS) { return NULL; }
    if (h->import[index] && h->import_fd[index] == f->dmabuf_fd[0]) { return h->import[index]; }
    if (h->import[index]) { infer_wgpu_frame_import_destroy(h->import[index]); }
    struct infer_frame idesc = hand_frame_import_desc(f);
    h->import[index] = infer_wgpu_frame_import_create(&h->ctx.wgpu, &idesc);
    h->import_fd[index] = f->dmabuf_fd[0];
    return h->import[index];
}

static struct infer_vk_frame_import* import_for_vk(struct hand_tracker* h,
                                                   const struct infer_frame* f,
                                                   int index) {
    if (index < 0 || index >= HAND_MAX_FRAME_IMPORTS) { return NULL; }
    if (h->vimport[index] && h->vimport_fd[index] == f->dmabuf_fd[0]) { return h->vimport[index]; }
    if (h->vimport[index]) { infer_vk_frame_import_destroy(h->vimport[index]); }
    struct infer_frame idesc = hand_frame_import_desc(f);
    h->vimport[index] = infer_vk_frame_import_create(&h->ctx.vk, &idesc);
    h->vimport_fd[index] = f->dmabuf_fd[0];
    return h->vimport[index];
}

static int feed(struct hand_model* m,
                struct infer_wgpu_frame_import* im,
                struct infer_vk_frame_import* vim,
                const struct infer_frame* f,
                struct hand_affine a) {
    if (hand_model_wants_vk_frame(m)) { return hand_model_feed_vk(m, vim, f, a); }
    if (hand_model_ep(m) == INFER_EP_WEBGPU) { return hand_model_feed_wgpu(m, im, f, a); }
    return hand_model_feed_cpu(m, f, a);
}

/* Dawn's import of the frame, or the Vulkan pass with the registry's
 * vk -> wgpu row behind it? Decided at the first frame, because that is
 * when the frame's layout is known, and decided on OUR Vulkan device
 * first: Dawn's import of a frame this driver refuses (NVIDIA 610.43,
 * every linear NV12 layout) does not fail, it aborts the process or
 * loses the device. Where the probe cannot tell -- a single-plane frame,
 * no Vulkan domain -- Dawn's import is attempted as before. */
static int choose_wgpu_route(struct hand_tracker* h, struct provider* p, const struct infer_frame* f) {
    struct infer_frame idesc = hand_frame_import_desc(f);
    if (infer_vk_frame_multiplanar_importable(&h->ctx.vk, &idesc) != 0) { return 1; }
    if (hand_model_use_vk_feed(p->palm) < 0 || (p->lmk && hand_model_use_vk_feed(p->lmk) < 0)) {
        fprintf(stderr,
                "hand: the driver refuses Dawn's frame import and the vk feed is unavailable; "
                "attempting Dawn's import regardless\n");
        return 1;
    }
    fprintf(stderr, "hand: the driver refuses Dawn's frame import; the webgpu provider is fed by "
                    "the vk pass + vk -> wgpu edge\n");
    return 2;
}

/* detector pixels -> frame pixels. The fit affine has no rotation and no
 * shear, so the ROI's angle and squareness survive untouched -- which is
 * why the detection arithmetic is done in detector space (hand.h). */
static struct hand_roi roi_to_frame(struct hand_roi r, struct hand_affine fit) {
    struct hand_roi o = r;
    hand_affine_apply(fit, r.cx, r.cy, &o.cx, &o.cy);
    o.side = r.side * hand_affine_scale(fit);
    return o;
}

static void publish(struct hand_tracker* h, const struct hand_results* r) {
    const uint32_t g = atomic_load_explicit(&h->gen, memory_order_relaxed);
    const int slot = (int)((g >> 1) & 1u) ^ 1; /* write the one nobody reads */
    h->slot[slot] = *r;
    /* odd while in flux, even when settled: a reader that saw an odd
     * value, or a different one either side of its copy, retries */
    atomic_store_explicit(&h->gen, g + 1, memory_order_release);
    atomic_store_explicit(&h->gen, g + 2, memory_order_release);
}
/* ------------------------------------------------------------------ */
/* the cycle: MediaPipe's hand_landmark_tracking graph, in C           */
/* ------------------------------------------------------------------ */

/* This follows mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt
 * node for node, because an earlier approximation of it -- one tracked
 * ROI, detection only when that ROI was lost -- was subtly and
 * persistently less stable, and every attempt to fix it by adjusting
 * thresholds made something else worse. The graph's shape IS the
 * algorithm. Its nodes, in order, and what each is here:
 *
 *   NormalizedRectVectorHasMinSizeCalculator  prev_count >= num_hands
 *   GateCalculator (DISALLOW:has_enough)      run the detector or not
 *   PalmDetectionCpu                          the detector
 *   ClipDetectionVectorSizeCalculator          keep at most num_hands
 *   PalmDetectionDetectionToRoi                hand_roi_from_detection
 *   AssociationNormRectCalculator              associate(), below
 *   BeginLoop / HandLandmarkCpu / EndLoop      one landmark run per rect
 *   HandLandmarkLandmarksToRoi                 hand_roi_from_landmarks
 *   PreviousLoopbackCalculator                 h->prev for the next cycle
 *
 * Two consequences of the real shape that the approximation got wrong,
 * and both matter more than any threshold:
 *
 * 1. With num_hands = 2 and ONE hand visible, prev_count (1) is less
 *    than num_hands (2), so the detector runs EVERY cycle -- not only
 *    when the track is lost. MediaPipe pays a detector inference per
 *    frame for as long as it is still looking for another hand.
 *
 * 2. Which makes the association load-bearing. Its inputs are, in this
 *    order, the palm rects and then the previous landmark rects, and
 *    AssociationCalculator keeps the element from the LATER stream when
 *    two overlap ("the element that comes in from a later input stream
 *    is kept in the output"). So a detection that lands on a hand
 *    already being tracked is DISCARDED and the tracked ROI survives
 *    untouched; only a detection that overlaps nothing adds a hand.
 *    That is what lets detection run every frame without the tracked
 *    ROI ever being replaced by a coarser or jitterier one.
 */

/* IoU of two rotated squares, approximated by their axis-aligned bounds.
 * MediaPipe's AssociationNormRectCalculator uses OverlapSimilarity on
 * NormalizedRects, which likewise ignores the rotation. */
static float roi_iou(struct hand_roi a, struct hand_roi b) {
    const float ah = a.side * 0.5f, bh = b.side * 0.5f;
    const float ix = fminf(a.cx + ah, b.cx + bh) - fmaxf(a.cx - ah, b.cx - bh);
    const float iy = fminf(a.cy + ah, b.cy + bh) - fmaxf(a.cy - ah, b.cy - bh);
    if (ix <= 0.0f || iy <= 0.0f) { return 0.0f; }
    const float inter = ix * iy;
    const float uni = a.side * a.side + b.side * b.side - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* AssociationNormRectCalculator, min_similarity_threshold 0.5. Elements
 * are added in input-stream order and a new one EVICTS any it overlaps,
 * so passing the previous landmark rects last gives them priority.
 *
 * `max_out` must be big enough for num_hands detections PLUS every
 * previous rect, and that is not a detail: capping it at HAND_MAX during
 * accumulation lets two palm rects fill the buffer before the previous
 * landmark rects are even offered, so the actual TRACK gets dropped in
 * favour of two fresh detections. MediaPipe clips the DETECTIONS to
 * num_hands beforehand (ClipDetectionVectorSizeCalculator) and never
 * caps the association output, exactly so the tracked rect survives. */
static int associate(const struct hand_roi* palm,
                     int n_palm,
                     const struct hand_roi* prev,
                     int n_prev,
                     struct hand_roi* out,
                     int max_out) {
    int n = 0;
    for (int pass = 0; pass < 2; pass++) {
        const struct hand_roi* in = pass == 0 ? palm : prev;
        const int count = pass == 0 ? n_palm : n_prev;
        for (int i = 0; i < count; i++) {
            /* evict whatever this one overlaps: later wins */
            for (int j = 0; j < n;) {
                if (roi_iou(in[i], out[j]) >= 0.5f) {
                    out[j] = out[--n];
                } else {
                    j++;
                }
            }
            if (n < max_out) { out[n++] = in[i]; }
        }
    }
    return n;
}

static void one_cycle(struct hand_tracker* h, const struct infer_frame* f, int index, uint32_t seq) {
    const enum infer_ep ep = (enum infer_ep)atomic_load(&h->want_ep);
    struct provider* p = &h->prov[ep];
    if (!p->ready) { return; }
    static float anchors[HAND_PALM_ANCHORS * 2];
    static int anchors_done = 0;
    if (!anchors_done) {
        hand_palm_anchors(anchors, HAND_PALM_ANCHORS);
        anchors_done = 1;
    }

    struct infer_wgpu_frame_import* im = NULL;
    struct infer_vk_frame_import* vim = NULL;
    const int gpu = ep == INFER_EP_WEBGPU;
    if (gpu && h->wgpu_route == 0) { h->wgpu_route = choose_wgpu_route(h, p, f); }
    if (gpu && h->wgpu_route == 1) {
        im = import_for(h, f, index);
        if (!im) { return; }
        /* one access bracket for the cycle: every FrameToTensor pass in it
         * reads the same frame, which is why begin and end are separate */
        if (infer_wgpu_frame_begin(&h->ctx.wgpu, im, -1) < 0) { return; }
    }
    if (hand_model_wants_vk_frame(p->palm)) {
        vim = import_for_vk(h, f, index);
        if (!vim) { return; }
    }

    struct hand_results res;
    memset(&res, 0, sizeof res);
    res.frame_seq = seq;
    res.ep = ep;

    double t_pre = 0, t_run = 0, t_wait = 0, t_dec = 0;
    const uint64_t c0 = infer_now_ns();

    /* --- NormalizedRectVectorHasMinSizeCalculator + GateCalculator --- */
    const int want_detect = h->prev_count < h->opt.num_hands;

    struct hand_roi palm_rects[HAND_MAX];
    int n_palm = 0;
    if (want_detect) {
        struct hand_det det[64];
        const uint64_t a0 = infer_now_ns();
        if (feed(p->palm, im, vim, f, h->fit) < 0) { goto done; }
        const uint64_t a1 = infer_now_ns();
        if (hand_model_run(p->palm) < 0) { goto done; }
        const uint64_t a2 = infer_now_ns();
        if (hand_model_sync(p->palm) < 0) { goto done; }
        const uint64_t a3 = infer_now_ns();
        const int n = hand_palm_decode(hand_model_output(p->palm, 0), hand_model_output(p->palm, 1),
                                       anchors, HAND_PALM_ANCHORS, 0.5f, det, 64);
        /* ClipDetectionVectorSizeCalculator: at most num_hands */
        const int kept = hand_nms(det, n, 0.3f, h->opt.num_hands);
        for (int i = 0; i < kept && n_palm < HAND_MAX; i++) {
            palm_rects[n_palm++] = roi_to_frame(hand_roi_from_detection(&det[i]), h->fit);
        }
        t_pre += (double)(a1 - a0) / 1e3;
        t_run += (double)(a2 - a1) / 1e3;
        t_wait += (double)(a3 - a2) / 1e3;
        t_dec += (double)(infer_now_ns() - a3) / 1e3;
        res.ran_detector = 1;
        h->since_detect = 0;
    } else {
        h->since_detect++;
    }

    /* --- AssociationNormRectCalculator: prev last, so prev wins --- */
    struct hand_roi rects[HAND_MAX * 2];
    const uint64_t d0 = infer_now_ns();
    int n_rects = associate(palm_rects, n_palm, h->prev, h->prev_count, rects,
                            (int)(sizeof rects / sizeof rects[0]));
    t_dec += (double)(infer_now_ns() - d0) / 1e3;

    /* --- the landmark loop, once per rect --- */
    struct hand_roi next[HAND_MAX];
    int n_next = 0;
    if (!h->opt.detect_only) {
        for (int r = 0; r < n_rects; r++) {
            struct hand_affine crop = hand_affine_roi(rects[r].cx, rects[r].cy, rects[r].side,
                                                      rects[r].rot, HAND_LANDMARK_SIDE,
                                                      HAND_LANDMARK_SIDE);
            const uint64_t b0 = infer_now_ns();
            if (feed(p->lmk, im, vim, f, crop) < 0) { goto done; }
            const uint64_t b1 = infer_now_ns();
            if (hand_model_run(p->lmk) < 0) { goto done; }
            const uint64_t b2 = infer_now_ns();
            if (hand_model_sync(p->lmk) < 0) { goto done; }
            const uint64_t b3 = infer_now_ns();
            struct hand_point lm[HAND_LANDMARKS];
            hand_landmarks_decode(hand_model_output(p->lmk, 0), rects[r], lm);
            const float presence = hand_model_output(p->lmk, 1)[0];
            const float handedness = hand_model_output(p->lmk, 2)[0];
            t_pre += (double)(b1 - b0) / 1e3;
            t_run += (double)(b2 - b1) / 1e3;
            t_wait += (double)(b3 - b2) / 1e3;

            /* HandLandmarkCpu's three GateCalculators: below the
             * threshold NOTHING is emitted for this hand, so it
             * contributes no landmarks and no rect to loop back. */
            if (presence < HAND_PRESENCE_THRESHOLD) {
                if (h->opt.verbose) {
                    fprintf(stderr,
                            "hand: dropped (presence %.3f) roi (%.1f, %.1f) side %.1f rot %+.3f\n",
                            (double)presence, (double)rects[r].cx, (double)rects[r].cy,
                            (double)rects[r].side, (double)rects[r].rot);
                }
                t_dec += (double)(infer_now_ns() - b3) / 1e3;
                continue;
            }
            if (res.count < HAND_MAX) {
                struct hand_result* hr = &res.hand[res.count++];
                for (int i = 0; i < HAND_LANDMARKS; i++) {
                    hr->lm[i].x = lm[i].x / (float)h->fw;
                    hr->lm[i].y = lm[i].y / (float)h->fh;
                    hr->lm[i].z = lm[i].z / (float)h->fw;
                }
                hr->presence = presence;
                hr->handedness = handedness;
                hr->score = presence;
                hr->roi = (struct hand_roi){rects[r].cx / (float)h->fw, rects[r].cy / (float)h->fh,
                                            rects[r].side / (float)h->fw, rects[r].rot};
            }
            /* HandLandmarkLandmarksToRoi -> PreviousLoopbackCalculator */
            if (n_next < HAND_MAX) { next[n_next++] = hand_roi_from_landmarks(lm); }
            t_dec += (double)(infer_now_ns() - b3) / 1e3;
        }
    } else {
        /* detect-only: report the boxes, keep nothing to loop back */
        for (int r = 0; r < n_rects && res.count < HAND_MAX; r++) {
            struct hand_result* hr = &res.hand[res.count++];
            hr->from_detector = 1;
            hr->roi = (struct hand_roi){rects[r].cx / (float)h->fw, rects[r].cy / (float)h->fh,
                                        rects[r].side / (float)h->fw, rects[r].rot};
        }
    }

    /* PreviousLoopbackCalculator: what this cycle produced is what the
     * next one starts from. A hand whose landmarks were gated out simply
     * is not in here, so the next cycle's gate lets the detector look for
     * it again -- which is exactly how MediaPipe "loses" a hand. */
    for (int i = 0; i < n_next; i++) { h->prev[i] = next[i]; }
    h->prev_count = n_next;

    /* frame in to landmarks out, the span the web demo calls inference
     * time; the frame's release fence below is not part of it there
     * either */
    res.cycle_ms = (float)((double)(infer_now_ns() - c0) / 1e6);
    publish(h, &res);
    pthread_mutex_lock(&h->stat_lock);
    if (h->st.n >= STAT_WINDOW) { memset(&h->st, 0, sizeof h->st); }
    h->st.pre += t_pre;
    h->st.run += t_run;
    h->st.wait += t_wait;
    h->st.decode += t_dec;
    h->st.cycle += (double)(infer_now_ns() - c0) / 1e3;
    h->st.n++;
    h->st.cycles++;
    h->st.detector_runs += (unsigned)res.ran_detector;
    h->st.hands = (unsigned)res.count;
    pthread_mutex_unlock(&h->stat_lock);

done:
    if (gpu && im) {
        int rel = -1;
        infer_wgpu_frame_end(&h->ctx.wgpu, im, &rel);
        /* Dawn's "done reading the frame" fence. The scene composes it
         * into cam_stream's fence so V4L2 cannot reclaim the buffer while
         * the GPU is still in it -- but we have already published, so a
         * host wait here is the simplest correct thing and costs ~100 us
         * on an otherwise idle queue. */
        if (rel >= 0) {
            infer_wait_sync_fd(rel, 100);
            close(rel);
        }
    }
}


/* The first Run of a session pays what create() deferred: the CUDA EP's
 * per-thread cuBLAS/cuDNN context (created lazily on the running thread,
 * which is why this happens HERE and not in create()) and, on its third
 * run, the graph capture with its one hidden stream sync; the WebGPU
 * EP's shader compilation and capture likewise. Pay it before the first
 * frame -- the landmark model otherwise pays it the first time a hand
 * appears, as a one-frame freeze. The input is whatever the fresh
 * allocation contains; the models do not care and nothing reads the
 * outputs. */
static void warm_up(struct hand_tracker* h) {
    for (int e = 0; e < INFER_EP_COUNT; e++) {
        struct provider* p = &h->prov[e];
        if (!p->ready || e == INFER_EP_CPU) { continue; }
        for (int i = 0; i < 3; i++) {
            struct hand_model* mm[2] = {p->palm, p->lmk};
            for (int m = 0; m < 2; m++) {
                if (!mm[m]) { continue; }
                if (hand_model_run(mm[m]) < 0 || hand_model_sync(mm[m]) < 0) {
                    fprintf(stderr, "hand: %s warm-up failed\n", infer_ep_name((enum infer_ep)e));
                    return;
                }
            }
        }
    }
}

static void* worker(void* arg) {
    struct hand_tracker* h = arg;
    warm_up(h);
    for (;;) {
        struct infer_frame f;
        int index;
        uint32_t seq;
        pthread_mutex_lock(&h->lock);
        while (h->req_index < 0 && !h->quit) { pthread_cond_wait(&h->wake, &h->lock); }
        if (h->quit) {
            pthread_mutex_unlock(&h->lock);
            return NULL;
        }
        f = h->req_frame;
        index = h->req_index;
        seq = h->req_seq;
        h->req_index = -1;
        h->busy_index = index;
        atomic_store(&h->holds_index, index);
        pthread_mutex_unlock(&h->lock);

        one_cycle(h, &f, index, seq);

        pthread_mutex_lock(&h->lock);
        h->busy_index = -1;
        atomic_store(&h->holds_index, -1);
        pthread_mutex_unlock(&h->lock);
    }
}

/* ------------------------------------------------------------------ */
/* bring-up                                                           */
/* ------------------------------------------------------------------ */

static int build_provider(struct hand_tracker* h, enum infer_ep ep) {
    struct provider* p = &h->prov[ep];
    /* the two graphs disagree about the border on purpose: see enum hand_border */
    p->palm = hand_model_create(&h->ctx, &h->reg, ep, h->opt.palm_model, HAND_DETECT_SIDE,
                                HAND_BORDER_ZERO);
    if (!p->palm) { return -1; }
    if (!h->opt.detect_only) {
        p->lmk = hand_model_create(&h->ctx, &h->reg, ep, h->opt.lmk_model, HAND_LANDMARK_SIDE,
                                   HAND_BORDER_REPLICATE);
        if (!p->lmk) { return -1; }
    }
    p->ready = 1;
    fprintf(stderr, "hand: %s provider ready (out-edge %s)\n", infer_ep_name(ep),
            hand_model_out_edge(p->palm));
    return 0;
}

struct hand_tracker* hand_tracker_create(const struct hand_options* o,
                                         uint32_t frame_w,
                                         uint32_t frame_h) {
    struct hand_tracker* h = calloc(1, sizeof *h);
    h->opt = *o;
    h->fw = frame_w;
    h->fh = frame_h;
    h->req_index = -1;
    h->busy_index = -1;
    atomic_store(&h->holds_index, -1);
    atomic_store(&h->want_ep, (int)o->ep);
    for (int i = 0; i < HAND_MAX_FRAME_IMPORTS; i++) {
        h->import_fd[i] = -1;
        h->vimport_fd[i] = -1;
    }
    pthread_mutex_init(&h->lock, NULL);
    pthread_mutex_init(&h->stat_lock, NULL);
    pthread_cond_init(&h->wake, NULL);
    h->fit = o->crop ? hand_affine_square_crop(frame_w, frame_h, HAND_DETECT_SIDE, HAND_DETECT_SIDE)
                     : hand_affine_letterbox(frame_w, frame_h, HAND_DETECT_SIDE, HAND_DETECT_SIDE);

    /* Which providers are wanted, and which memory domains that needs.
     * The frame arrives as a dma-buf and the results leave as host
     * floats, so the presenter's EGL context and VkDevice are never
     * touched -- only the wanted EPs' own domains come up. */
    int need[INFER_EP_COUNT] = {0};
    need[o->ep] = 1;
    if (o->both) { need[INFER_EP_CPU] = need[INFER_EP_WEBGPU] = 1; }
    if (o->all) {
        for (int e = 0; e < INFER_EP_COUNT; e++) { need[e] = 1; }
    }
    unsigned want = 0;
    for (int e = 0; e < INFER_EP_COUNT; e++) {
        const enum infer_domain dom = infer_engine_ort_input_domain((enum infer_ep)e);
        if (need[e] && dom != INFER_DOMAIN_CPU) { want |= 1u << dom; }
        /* the CUDA EP's frame pass runs on the Vulkan device, and the
         * WebGPU EP needs that device too: to ask, before Dawn is handed
         * the frame, whether this driver imports it at all, and as the
         * feed when it does not (choose_wgpu_route). Best effort --
         * absent, CUDA falls back to the host feed and Dawn's import is
         * simply attempted */
        if (need[e] && (e == INFER_EP_CUDA || e == INFER_EP_WEBGPU)) { want |= INFER_WANT_VK; }
    }
    infer_ctx_init_all(&h->ctx, want);
    infer_registry_init(&h->reg);
    infer_registry_probe(&h->reg, &h->ctx);

    /* an EP whose domain did not come up is dropped, per EP; the CPU
     * provider is the floor and always can */
    for (int e = 0; e < INFER_EP_COUNT; e++) {
        const enum infer_domain dom = infer_engine_ort_input_domain((enum infer_ep)e);
        if (need[e] && dom != INFER_DOMAIN_CPU && !(h->ctx.have & (1u << dom))) {
            fprintf(stderr, "hand: no %s domain; dropping the %s provider\n",
                    infer_domain_name(dom), infer_ep_name((enum infer_ep)e));
            need[e] = 0;
        }
    }
    {
        int any = 0;
        for (int e = 0; e < INFER_EP_COUNT; e++) { any |= need[e]; }
        if (!any) { need[INFER_EP_CPU] = 1; }
    }

    /* Every wanted provider up front, because graph capture pins the
     * tensors inside a session: an EP switch cannot rebuild them, so the
     * only way to make P instant is to have them all already standing. */
    for (int e = 0; e < INFER_EP_COUNT; e++) {
        if (need[e] && build_provider(h, (enum infer_ep)e) < 0) {
            hand_tracker_destroy(h);
            return NULL;
        }
    }
    if (!h->prov[atomic_load(&h->want_ep)].ready) {
        /* asked for webgpu, only cpu came up (or vice versa) */
        for (int e = 0; e < INFER_EP_COUNT; e++) {
            if (h->prov[e].ready) { atomic_store(&h->want_ep, e); }
        }
    }
    if (!h->prov[atomic_load(&h->want_ep)].ready) {
        fprintf(stderr, "hand: no provider came up\n");
        hand_tracker_destroy(h);
        return NULL;
    }

    if (pthread_create(&h->thread, NULL, worker, h) != 0) {
        fprintf(stderr, "hand: cannot start the worker thread\n");
        hand_tracker_destroy(h);
        return NULL;
    }
    h->thread_started = 1;
    return h;
}

void hand_tracker_destroy(struct hand_tracker* h) {
    if (!h) { return; }
    if (h->thread_started) {
        pthread_mutex_lock(&h->lock);
        h->quit = 1;
        pthread_cond_signal(&h->wake);
        pthread_mutex_unlock(&h->lock);
        pthread_join(h->thread, NULL);
    }
    /* the imports hold dup'd fds and must go before the device */
    for (int i = 0; i < HAND_MAX_FRAME_IMPORTS; i++) {
        infer_wgpu_frame_import_destroy(h->import[i]);
        infer_vk_frame_import_destroy(h->vimport[i]);
    }
    for (int e = 0; e < INFER_EP_COUNT; e++) {
        hand_model_destroy(h->prov[e].lmk);
        hand_model_destroy(h->prov[e].palm);
    }
    infer_ctx_fini_all(&h->ctx);
    pthread_cond_destroy(&h->wake);
    pthread_mutex_destroy(&h->stat_lock);
    pthread_mutex_destroy(&h->lock);
    free(h);
}

int hand_tracker_submit(struct hand_tracker* h,
                        const struct infer_frame* f,
                        int index,
                        uint32_t seq) {
    int taken = 0;
    /* trylock, so a render thread never waits on the worker even for the
     * length of a slot write */
    if (pthread_mutex_trylock(&h->lock) != 0) { return 0; }
    if (h->req_index < 0 && h->busy_index < 0) {
        h->req_frame = *f;
        h->req_index = index;
        h->req_seq = seq;
        taken = 1;
        pthread_cond_signal(&h->wake);
    }
    pthread_mutex_unlock(&h->lock);
    return taken;
}

int hand_tracker_poll(struct hand_tracker* h, struct hand_results* out) {
    /* the seqlock read: copy, then check nothing moved underneath */
    for (int spin = 0; spin < 4; spin++) {
        const uint32_t g0 = atomic_load_explicit(&h->gen, memory_order_acquire);
        if (g0 == 0) { return 0; } /* nothing published yet */
        if (g0 & 1u) { continue; } /* a write is in flux */
        const struct hand_results r = h->slot[(g0 >> 1) & 1u];
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&h->gen, memory_order_relaxed) != g0) { continue; }
        if (atomic_load_explicit(&h->consumed, memory_order_relaxed) == g0) { return 0; }
        atomic_store_explicit(&h->consumed, g0, memory_order_relaxed);
        *out = r;
        return 1;
    }
    return 0;
}

int hand_tracker_holds(const struct hand_tracker* h, int index) {
    return index >= 0 && atomic_load(&h->holds_index) == index;
}

void hand_tracker_set_ep(struct hand_tracker* h, enum infer_ep ep) {
    if (ep < 0 || ep >= INFER_EP_COUNT || !h->prov[ep].ready) { return; }
    atomic_store(&h->want_ep, (int)ep);
    /* a provider switch invalidates the track: the ROI came from the
     * other one's landmarks, and starting from a detection is cleaner
     * than trusting it across the boundary */
    pthread_mutex_lock(&h->stat_lock);
    memset(&h->st, 0, sizeof h->st);
    pthread_mutex_unlock(&h->stat_lock);
}

enum infer_ep hand_tracker_ep(const struct hand_tracker* h) {
    return (enum infer_ep)atomic_load(&h->want_ep);
}

int hand_tracker_ready_count(const struct hand_tracker* h) {
    int n = 0;
    for (int e = 0; e < INFER_EP_COUNT; e++) { n += h->prov[e].ready; }
    return n;
}

enum infer_ep hand_tracker_next_ep(const struct hand_tracker* h) {
    const int cur = atomic_load(&h->want_ep);
    for (int i = 1; i <= INFER_EP_COUNT; i++) {
        const int e = (cur + i) % INFER_EP_COUNT;
        if (h->prov[e].ready) { return (enum infer_ep)e; }
    }
    return (enum infer_ep)cur;
}

int hand_tracker_stats_line(const struct hand_tracker* h, char* buf, size_t n) {
    struct hand_tracker* m = (struct hand_tracker*)h;
    pthread_mutex_lock(&m->stat_lock);
    const struct stats s = m->st;
    pthread_mutex_unlock(&m->stat_lock);
    const double k = s.n ? 1.0 / s.n : 0.0;
    return snprintf(buf, n,
                    "ep=%s pre %.2f run %.2f wait %.2f dec %.2f = %.1f ms | det %u/%u | hands %u",
                    infer_ep_name(hand_tracker_ep(h)), s.pre * k / 1e3, s.run * k / 1e3,
                    s.wait * k / 1e3, s.decode * k / 1e3, s.cycle * k / 1e3, s.detector_runs,
                    s.cycles, s.hands);
}
