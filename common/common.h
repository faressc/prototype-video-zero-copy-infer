/* common.h -- the Wayland client boilerplate shared by all backends.
 *
 * One window, three renderers (shm.c, egl.c, vulkan.c). Everything
 * protocol-side lives in common.c: connect + registry, the xdg role
 * and configure/ack handshake, the full wl_seat (pointer with frame
 * batching, keyboard via xkbcommon, themed cursor), the pausable
 * animation clock, and the frame-callback render loop.
 *
 * A backend provides two hooks and embeds `struct app` as the FIRST
 * member of its own state struct, so hook callbacks can cast the
 * `struct app*` back to the outer type -- the classic container
 * pattern, first-member edition:
 *
 *     struct shm_app { struct app app;  ...backend fields... };
 *     static void shm_configure(struct app* a) {
 *         struct shm_app* s = (struct shm_app*)a;   // valid: first member
 *     }
 */
#ifndef HELLO_WAYLAND_COMMON_H
#define HELLO_WAYLAND_COMMON_H

#include <stdint.h>
#include <wayland-client.h>
#include <wayland-cursor.h>
#include <xkbcommon/xkbcommon.h>

#include "pointer-gestures-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"

/* fallback size when the compositor has no opinion (configure sends 0) */
enum { APP_DEFAULT_WIDTH = 640, APP_DEFAULT_HEIGHT = 360 };

struct app;

struct app_backend {
    /* React to a configure: a->width/height hold the (acked) new size.
     * Create or resize rendering resources here; do not draw. */
    void (*configure)(struct app* a);

    /* Draw + present exactly one frame (ending in a commit / swap /
     * present). If non-NULL, common runs the wl_surface.frame callback
     * loop and calls this each tick. If NULL, the backend drives its
     * own loop (vulkan.c: blocking FIFO present). */
    void (*redraw)(struct app* a);
};

/* One batched pointer transaction (seat v5+): event handlers STAGE
 * fields here, wl_pointer.frame APPLIES them atomically -- the same
 * stage/apply discipline as surface commit and configure/ack. */
enum {
    APP_PTR_MOTION = 1 << 0,
    APP_PTR_BUTTON = 1 << 1,
    APP_PTR_AXIS_V = 1 << 2,
};

struct app_pointer_event {
    uint32_t mask;
    double x, y;
    uint32_t button;
    uint32_t button_state;
    uint32_t serial;
    double axis_v;
};

struct app {
    const struct app_backend* backend;

    /* connection + introduction phase */
    struct wl_display* display;
    struct wl_registry* registry;
    struct wl_compositor* compositor;
    struct wl_shm* shm; /* bound for shm.c AND for the cursor theme */
    struct xdg_wm_base* wm_base;

    /* the window itself */
    struct wl_surface* surface;
    struct xdg_surface* xdg_surface;
    struct xdg_toplevel* toplevel;

    /* input: one seat = one user */
    struct wl_seat* seat;
    struct wl_pointer* pointer;
    struct wl_keyboard* keyboard;
    struct xkb_context* xkb_context;
    struct xkb_keymap* xkb_keymap;
    struct xkb_state* xkb_state;
    double ptr_x, ptr_y;                /* last APPLIED pointer position */
    struct app_pointer_event ptr_event; /* staged, not-yet-applied batch */

    /* the cursor: one more surface with one theme-provided buffer */
    struct wl_cursor_theme* cursor_theme;
    struct wl_surface* cursor_surface;
    struct wl_cursor_image* cursor_image;

    /* geometry: STAGED by xdg_toplevel.configure, APPLIED (into
     * width/height) when common acks xdg_surface.configure */
    int pending_w, pending_h;
    int width, height;
    int configured; /* first configure acked -- presenting is now legal */

    /* animation clock: pausable (Space), speed-scaled (scroll wheel) */
    uint32_t last_time;
    double anim_ms;
    double speed;
    int paused;
    int animating; /* frame-callback loop started? */

    /* touchpad pinch (zwp_pointer_gestures_v1): a gesture object per
     * wl_pointer, events begin/update/end; update carries the scale
     * relative to where the fingers STARTED, so zoom = zoom-at-begin *
     * scale. Scenes with a camera apply `zoom` (1.0 = default); the
     * gradients ignore it. NULL when the compositor lacks the global. */
    struct zwp_pointer_gestures_v1* gestures;
    struct zwp_pointer_gesture_pinch_v1* pinch;
    double zoom;
    double zoom_at_pinch_begin;

    /* a second fd in the event loop (the camera): app_run polls it next
     * to the display fd and calls on_aux_fd when it is readable. -1 =
     * none. Loop-owning backends call on_aux_fd themselves. */
    int aux_fd;
    void (*on_aux_fd)(struct app* a);

    /* effects, as a BITMASK: key N toggles bit N-1, E clears all --
     * active effects layer. A scene sets effect_count (bits in use). */
    int effect, effect_count;
    int mode; /* C toggles a scene-defined mode (the camera scenes: on the cube) */

    /* Stage five's two knobs, in the same spirit as `mode`: common bumps
     * them, the scene decides what they mean. P counts provider switches
     * (the hand scenes alternate CPU / WebGPU); H toggles the landmark
     * overlay, on by default. Scenes that know nothing about either
     * simply never read them. */
    int infer_ep;
    int overlay;

    int running;
};

/* Connect, bind globals, set up window + input + cursor, send the
 * empty first commit. Returns 0 on success. */
int app_init(struct app* a,
             const struct app_backend* backend,
             const char* title,
             const char* app_id);

/* The callback-driven event loop: blocks in wl_display_dispatch until
 * running goes 0. Backends with redraw == NULL write their own loop
 * instead (dispatch_pending + draw). */
void app_run(struct app* a);

/* Tear down everything app_init created. Backends destroy their own
 * rendering resources BEFORE calling this. */
void app_finish(struct app* a);

/* Request the next wl_surface.frame callback; the request rides the
 * next commit. Common calls this itself inside its frame loop. */
void app_schedule_frame(struct app* a);

/* Advance the animation clock to `now_ms` (any monotonic ms source:
 * frame-callback timestamps, CLOCK_MONOTONIC, ...). Honors pause and
 * scroll speed. Common calls this in its frame loop; loop-owning
 * backends call it once per frame themselves. */
void app_advance_clock(struct app* a, uint32_t now_ms);

/* The clock as a ms value for shaders / fill loops, wrapped into
 * [0, 2048) -- the period of the shared shift = (t/8) mod 256 fill
 * math. Small on purpose: huge values (e.g. a negative clock cast to
 * uint32_t) lose float32 precision on the GPU and band the gradient. */
uint32_t app_anim_time(const struct app* a);

#endif
