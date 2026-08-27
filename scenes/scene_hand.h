/* scene_hand.h -- what a camera scene needs to grow hand tracking, and
 * nothing more.
 *
 * The three stage-three scenes are compiled twice: once as cam_* exactly
 * as they were, once with SCENE_HAND=1 as hand_*. So every hook here has
 * to be small enough that the #if blocks in scene_cam_{shm,gles,vk}.c
 * stay call sites rather than a second implementation -- about thirty
 * lines each. Everything substantial lives in hand/.
 *
 * Two things live here because they belong to neither side: filling a
 * struct infer_frame from a V4L2 buffer (hand/ should not have to know
 * about camera/'s structs beyond the colour coefficients it already
 * shares), and composing the tracker's "still reading it" answer with
 * the backend's own GPU fence -- which is the whole reason
 * camera/cam_stream.[ch] needs no changes: its fence was always opaque.
 */
#ifndef HELLO_WAYLAND_SCENE_HAND_H
#define HELLO_WAYLAND_SCENE_HAND_H

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "cam_stream.h"
#include "common.h"
#include "hand_overlay.h"
#include "hand_tracker.h"

struct hand_scene {
    struct hand_tracker* tracker;
    struct hand_options opt;
    struct hand_results latest;
    struct hand_overlay_style style;
    struct hand_vert verts[HAND_OVERLAY_MAX_VERTS];
    int vert_count;
    int last_ep;      /* to notice the P key */
    int logged_rect;  /* --overlay-test prints the transform once */
    int last_title;   /* what the title last said, so it is set only on a change */
    double last_line; /* ms of the last stderr/title report */
    float cycle_ema;  /* the readout's number: cycle_ms smoothed over ~10 cycles */
    uint32_t frames, submitted, dropped;
};

/* Bring the tracker up. Returns 0 even when it fails: a camera scene
 * without hand tracking is still a camera scene, and refusing to start
 * would be a worse trade than a warning. */
static inline int hand_scene_init(struct hand_scene* hs,
                                  const struct cam_stream* s,
                                  int argc,
                                  char** argv) {
    memset(hs, 0, sizeof *hs);
    if (hand_options_parse(argc, argv, &hs->opt) < 0) { return -1; }
    hs->style = hand_overlay_default((float)s->cam.format.width / (float)s->cam.format.height);
    hs->style.show_roi = hs->opt.verbose;
    if (hs->opt.overlay_test) {
        /* no models and no Dawn: this mode exists to test the drawing */
        hs->style.show_calib = 1;
        hs->style.show_box = 1;
        hs->style.show_roi = 1;
        fprintf(stderr,
                "hand: --overlay-test, drawing a synthetic hand plus a calibration frame\n"
                "      (the green border must hug the picture; the red mark is TOP-LEFT)\n");
        return 0;
    }
    hs->tracker = hand_tracker_create(&hs->opt, s->cam.format.width, s->cam.format.height);
    if (!hs->tracker) {
        fprintf(stderr, "hand: tracking disabled\n");
        return 0;
    }
    hs->last_ep = (int)hand_tracker_ep(hs->tracker);
    fprintf(stderr,
            "hand: %s into the detector; keys: H overlay, C cube, 1-4 effects, E clear%s\n"
            "      the window title carries the live timings (--verbose puts them here too)\n",
            hs->opt.crop ? "centre-crop" : "letterbox",
            hand_tracker_ready_count(hs->tracker) > 1 ? ", P provider" : "");
    return 0;
}

static inline void hand_scene_fini(struct hand_scene* hs) {
    hand_tracker_destroy(hs->tracker);
    hs->tracker = NULL;
}

/* Once per frame, before drawing: hand the newest camera buffer to the
 * worker if it is idle, pick up whatever it finished, and rebuild the
 * overlay geometry. `index` is the buffer the scene is about to read. */
static inline void hand_scene_update(struct hand_scene* hs,
                                     struct app* app,
                                     const struct cam_stream* s,
                                     int index) {
    if (hs->opt.overlay_test) {
        /* no models, no camera: a known hand, so "is the overlay drawn in
         * the right place" has exactly one possible answer */
        hand_results_synthetic(&hs->latest, app->anim_ms * 0.001);
        hs->vert_count =
            app->overlay
                ? hand_overlay_build(&hs->latest, hs->style, hs->verts, HAND_OVERLAY_MAX_VERTS)
                : 0;
        return;
    }
    if (!hs->tracker) { return; }
    hs->frames++;

    /* the P key: `app->infer_ep` is a plain counter common.c bumps, so
     * the scene decides what it means -- here, the next ready provider */
    if (hand_tracker_ready_count(hs->tracker) > 1 && app->infer_ep != hs->last_ep) {
        hs->last_ep = app->infer_ep;
        hand_tracker_set_ep(hs->tracker, hand_tracker_next_ep(hs->tracker));
    }

    if (index >= 0) {
        struct infer_frame f;
        hand_frame_from_camera(&f, &s->cam, (uint32_t)index);
        /* The CPU and CUDA providers read the frame through its mmap (the
         * C FrameToTensor); map it once and the pointer stays good for
         * the buffer's life. The WebGPU provider imports the fd and never
         * needs this. */
        if (hand_tracker_ep(hs->tracker) != INFER_EP_WEBGPU && !f.map) {
            f.map = camera_map((struct camera*)&s->cam, (uint32_t)index);
        }
        if (hand_tracker_submit(hs->tracker, &f, index, (uint32_t)s->latest_seq)) {
            hs->submitted++;
        } else {
            hs->dropped++;
        }
    }
    if (hand_tracker_poll(hs->tracker, &hs->latest)) {
        /* The number on the picture, in the web demo's terms: the wall
         * time of one cycle, frame in to landmarks out (the tracker
         * measures it; hand_results.cycle_ms says what it covers).
         * Smoothed with a 1/10 EMA so it can be read rather than watched
         * jitter -- the demo's counter is per call and dances -- and
         * labelled with the provider, since P changes it live. */
        if (hs->latest.cycle_ms > 0.0f) {
            hs->cycle_ema = hs->cycle_ema > 0.0f
                                ? hs->cycle_ema + 0.1f * (hs->latest.cycle_ms - hs->cycle_ema)
                                : hs->latest.cycle_ms;
        }
        char epname[8];
        const char* en = infer_ep_name(hs->latest.ep);
        size_t ei = 0;
        for (; en[ei] && ei + 1 < sizeof epname; ei++) {
            epname[ei] = (char)toupper((unsigned char)en[ei]);
        }
        epname[ei] = '\0';
        snprintf(hs->style.readout,
                 sizeof hs->style.readout,
                 "%.1f MS %s",
                 (double)hs->cycle_ema,
                 epname);
        hs->vert_count =
            hand_overlay_build(&hs->latest, hs->style, hs->verts, HAND_OVERLAY_MAX_VERTS);
    }
    if (!app->overlay) { hs->vert_count = 0; }

    /* Once a second: the same string into the window title ALWAYS, and
     * onto stderr with --verbose.
     *
     * The title is the trick that makes an EP comparison visible in a
     * screenshot without a font renderer, a texture atlas or a new
     * pipeline in three backends -- and it is unconditional because
     * "hands 0" in the title bar is the difference between a user
     * wondering whether the build is broken and knowing the detector
     * simply is not seeing a hand. It costs one request a second. */
    if (app->anim_ms - hs->last_line > 1000.0) {
        hs->last_line = app->anim_ms;
        char line[256], title[320];
        hand_tracker_stats_line(hs->tracker, line, sizeof line);
        const unsigned tried = hs->submitted + hs->dropped;
        snprintf(title,
                 sizeof title,
                 "hand: %s | fed %u/%u",
                 line,
                 hs->submitted,
                 tried ? tried : 1u);
        if (hs->opt.verbose) { fprintf(stderr, "%s\n", title); }
        /* The title carries the numbers, but it is set only when
         * something a person would notice CHANGES -- the hand count or
         * the provider -- not once a second with fresh timings.
         * xdg_toplevel.set_title is a round trip that can make a
         * compositor redecorate, and a 1 Hz window-state mutation is a
         * plausible source of exactly the periodic flicker this stage
         * spent a while chasing. Not worth re-introducing one to display
         * a median. */
        const int now = hs->latest.count * 4 + (int)hs->latest.ep;
        if (app->toplevel && now != hs->last_title) {
            hs->last_title = now;
            xdg_toplevel_set_title(app->toplevel, title);
        }
        hs->submitted = hs->dropped = 0;
    }
}

/* The tracker may still be reading a camera buffer after the renderer is
 * done with it, so a scene's cam_fence_ops.done must consult both. This
 * is the composition that keeps camera/ untouched. */
static inline int hand_scene_holds(const struct hand_scene* hs, int index) {
    return hs->tracker && hand_tracker_holds(hs->tracker, index);
}

#endif
