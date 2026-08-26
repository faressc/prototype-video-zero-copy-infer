/* common.c -- everything that is Wayland, not rendering.
 *
 * The annotated deep-dives for each block live in README.md:
 *   registry/introduction (§4), configure/ack (§5), frame loop (§6),
 *   wl_seat (§7), cursor (§8), keyboard (§9).
 */

#include "common.h"

#include <errno.h>
#include <linux/input-event-codes.h> /* BTN_LEFT */
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Registry: read the menu, bind what every backend needs.            */
/* ------------------------------------------------------------------ */

static void on_global(void* data,
                      struct wl_registry* registry,
                      uint32_t name,
                      const char* interface,
                      uint32_t version) {
    struct app* a = data;

    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        /* v4: needed for wl_surface.damage_buffer */
        a->compositor = wl_registry_bind(registry, name, &wl_compositor_interface, 4);
    } else if (strcmp(interface, wl_shm_interface.name) == 0) {
        a->shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        a->wm_base = wl_registry_bind(registry, name, &xdg_wm_base_interface, 1);
    } else if (strcmp(interface, wl_seat_interface.name) == 0) {
        /* negotiate min(server, what we implement); v5 = pointer frames */
        a->seat = wl_registry_bind(registry, name, &wl_seat_interface, version < 5 ? version : 5);
    } else if (strcmp(interface, zwp_pointer_gestures_v1_interface.name) == 0) {
        /* optional: pinch exists since v1, that is all we need */
        a->gestures = wl_registry_bind(registry, name, &zwp_pointer_gestures_v1_interface, 1);
    }
}

static void on_global_remove(void* data, struct wl_registry* registry, uint32_t name) {
    (void)data;
    (void)registry;
    (void)name;
}

static const struct wl_registry_listener registry_listener = {
    .global = on_global,
    .global_remove = on_global_remove,
};

/* The liveness check: compositor pings, we must pong or be declared dead. */
static void on_wm_base_ping(void* data, struct xdg_wm_base* wm_base, uint32_t serial) {
    (void)data;
    xdg_wm_base_pong(wm_base, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
    .ping = on_wm_base_ping,
};

/* ------------------------------------------------------------------ */
/* Keyboard: raw scancodes + fd-shared keymap -> xkbcommon -> keysyms. */
/* ------------------------------------------------------------------ */

static void on_keyboard_keymap(void* data,
                               struct wl_keyboard* keyboard,
                               uint32_t format,
                               int32_t fd,
                               uint32_t size) {
    (void)keyboard;
    struct app* a = data;

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
        close(fd);
        return;
    }
    char* map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return;
    }
    /* replace, don't create-once: layout switches re-send this event */
    xkb_state_unref(a->xkb_state);
    xkb_keymap_unref(a->xkb_keymap);
    a->xkb_keymap = xkb_keymap_new_from_string(a->xkb_context,
                                               map,
                                               XKB_KEYMAP_FORMAT_TEXT_V1,
                                               XKB_KEYMAP_COMPILE_NO_FLAGS);
    a->xkb_state = a->xkb_keymap ? xkb_state_new(a->xkb_keymap) : NULL;
    munmap(map, size);
    close(fd);
}

static void on_keyboard_key(void* data,
                            struct wl_keyboard* keyboard,
                            uint32_t serial,
                            uint32_t time,
                            uint32_t key,
                            uint32_t key_state) {
    (void)keyboard;
    (void)serial;
    (void)time;
    struct app* a = data;

    if (key_state != WL_KEYBOARD_KEY_STATE_PRESSED || !a->xkb_state) { return; }

    /* evdev scancode + 8 = xkb keycode (fossilized X11 offset) */
    xkb_keysym_t sym = xkb_state_key_get_one_sym(a->xkb_state, key + 8);
    switch (sym) {
        case XKB_KEY_Escape:
        case XKB_KEY_q: a->running = 0; break;
        case XKB_KEY_space: a->paused = !a->paused; break;
        case XKB_KEY_e:
        case XKB_KEY_E: a->effect = 0; break; /* all off */
        case XKB_KEY_c:
        case XKB_KEY_C: a->mode = !a->mode; break;
        /* stage five: P switches execution provider, H hides the overlay.
         * A counter rather than a flag for P, so a scene with more than
         * two providers could cycle them. */
        case XKB_KEY_p:
        case XKB_KEY_P: a->infer_ep++; break;
        case XKB_KEY_h:
        case XKB_KEY_H: a->overlay = !a->overlay; break;
        default:
            if (sym >= XKB_KEY_1 && sym <= XKB_KEY_9) {
                int n = (int)(sym - XKB_KEY_1);
                if (n < a->effect_count) { a->effect ^= 1 << n; } /* toggle: effects layer */
            }
            break;
    }
}

static void on_keyboard_modifiers(void* data,
                                  struct wl_keyboard* keyboard,
                                  uint32_t serial,
                                  uint32_t mods_depressed,
                                  uint32_t mods_latched,
                                  uint32_t mods_locked,
                                  uint32_t group) {
    (void)keyboard;
    (void)serial;
    struct app* a = data;
    /* mirrored, not deduced: the compositor owns modifier truth */
    if (a->xkb_state) {
        xkb_state_update_mask(a->xkb_state, mods_depressed, mods_latched, mods_locked, 0, 0, group);
    }
}

static void on_keyboard_enter(void* d,
                              struct wl_keyboard* k,
                              uint32_t s,
                              struct wl_surface* su,
                              struct wl_array* keys) {
    (void)d;
    (void)k;
    (void)s;
    (void)su;
    (void)keys; /* keys already held at focus -- a real app processes these */
}
static void on_keyboard_leave(void* d, struct wl_keyboard* k, uint32_t s, struct wl_surface* su) {
    (void)d;
    (void)k;
    (void)s;
    (void)su; /* leave implies ALL keys released */
}
static void on_keyboard_repeat_info(void* d, struct wl_keyboard* k, int32_t r, int32_t de) {
    (void)d;
    (void)k;
    (void)r;
    (void)de; /* repeat is the client's job; we skip it */
}

static const struct wl_keyboard_listener keyboard_listener = {
    .keymap = on_keyboard_keymap,
    .enter = on_keyboard_enter,
    .leave = on_keyboard_leave,
    .key = on_keyboard_key,
    .modifiers = on_keyboard_modifiers,
    .repeat_info = on_keyboard_repeat_info,
};

/* ------------------------------------------------------------------ */
/* Pointer: handlers STAGE into a->ptr_event; frame APPLIES.          */
/* ------------------------------------------------------------------ */

static void on_pointer_enter(void* data,
                             struct wl_pointer* pointer,
                             uint32_t serial,
                             struct wl_surface* surface,
                             wl_fixed_t sx,
                             wl_fixed_t sy) {
    (void)surface;
    struct app* a = data;

    a->ptr_event.mask |= APP_PTR_MOTION;
    a->ptr_event.x = wl_fixed_to_double(sx);
    a->ptr_event.y = wl_fixed_to_double(sy);

    /* set_cursor is the protocol REPLY to this enter (like pong to
     * ping), so it does not wait for the frame event. Hotspot = which
     * pixel of the image is the logical tip. */
    if (a->cursor_surface) {
        wl_pointer_set_cursor(pointer,
                              serial,
                              a->cursor_surface,
                              (int32_t)a->cursor_image->hotspot_x,
                              (int32_t)a->cursor_image->hotspot_y);
    } else {
        wl_pointer_set_cursor(pointer, serial, NULL, 0, 0);
    }
}

static void on_pointer_motion(void* data,
                              struct wl_pointer* pointer,
                              uint32_t time,
                              wl_fixed_t sx,
                              wl_fixed_t sy) {
    (void)pointer;
    (void)time;
    struct app* a = data;
    a->ptr_event.mask |= APP_PTR_MOTION;
    a->ptr_event.x = wl_fixed_to_double(sx);
    a->ptr_event.y = wl_fixed_to_double(sy);
}

static void on_pointer_button(void* data,
                              struct wl_pointer* pointer,
                              uint32_t serial,
                              uint32_t time,
                              uint32_t button,
                              uint32_t button_state) {
    (void)pointer;
    (void)time;
    struct app* a = data;
    a->ptr_event.mask |= APP_PTR_BUTTON;
    a->ptr_event.button = button;
    a->ptr_event.button_state = button_state;
    a->ptr_event.serial = serial;
}

static void on_pointer_axis(void* data,
                            struct wl_pointer* pointer,
                            uint32_t time,
                            uint32_t axis,
                            wl_fixed_t value) {
    (void)pointer;
    (void)time;
    struct app* a = data;
    if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
        a->ptr_event.mask |= APP_PTR_AXIS_V;
        a->ptr_event.axis_v += wl_fixed_to_double(value); /* sum over the frame */
    }
}

/* APPLY one batched transaction atomically, then clear the staging. */
static void on_pointer_frame(void* data, struct wl_pointer* pointer) {
    (void)pointer;
    struct app* a = data;
    struct app_pointer_event* ev = &a->ptr_event;

    if (ev->mask & APP_PTR_MOTION) {
        a->ptr_x = ev->x;
        a->ptr_y = ev->y;
    }
    if ((ev->mask & APP_PTR_BUTTON) && ev->button == BTN_LEFT &&
        ev->button_state == WL_POINTER_BUTTON_STATE_PRESSED) {
        /* the serial proves FRESH user intent -- the compositor honors
         * move only against a recent input serial */
        xdg_toplevel_move(a->toplevel, a->seat, ev->serial);
    }
    if (ev->mask & APP_PTR_AXIS_V) {
        a->speed -= ev->axis_v / 20.0; /* through zero into reverse */
        if (a->speed > 10.0) { a->speed = 10.0; }
        if (a->speed < -10.0) { a->speed = -10.0; }
    }
    memset(ev, 0, sizeof(*ev));
}

static void on_pointer_leave(void* d, struct wl_pointer* p, uint32_t s, struct wl_surface* su) {
    (void)d;
    (void)p;
    (void)s;
    (void)su;
}
static void on_pointer_axis_source(void* d, struct wl_pointer* p, uint32_t s) {
    (void)d;
    (void)p;
    (void)s;
}
static void on_pointer_axis_stop(void* d, struct wl_pointer* p, uint32_t t, uint32_t ax) {
    (void)d;
    (void)p;
    (void)t;
    (void)ax;
}
static void on_pointer_axis_discrete(void* d, struct wl_pointer* p, uint32_t ax, int32_t di) {
    (void)d;
    (void)p;
    (void)ax;
    (void)di;
}

static const struct wl_pointer_listener pointer_listener = {
    .enter = on_pointer_enter,
    .leave = on_pointer_leave,
    .motion = on_pointer_motion,
    .button = on_pointer_button,
    .axis = on_pointer_axis,
    .frame = on_pointer_frame,
    .axis_source = on_pointer_axis_source,
    .axis_stop = on_pointer_axis_stop,
    .axis_discrete = on_pointer_axis_discrete,
};

/* ------------------------------------------------------------------ */
/* Touchpad pinch: libinput recognizes the gesture, the compositor    */
/* forwards it as a small begin/update/end transaction on a gesture   */
/* object hanging off the wl_pointer. `scale` is CUMULATIVE since     */
/* begin (2.0 = fingers twice as far apart as at the start), so the   */
/* zoom is anchored to its value at begin rather than integrated.     */
/* ------------------------------------------------------------------ */

static void on_pinch_begin(void* data,
                           struct zwp_pointer_gesture_pinch_v1* pinch,
                           uint32_t serial,
                           uint32_t time,
                           struct wl_surface* surface,
                           uint32_t fingers) {
    (void)pinch;
    (void)serial;
    (void)time;
    (void)surface;
    (void)fingers;
    struct app* a = data;
    a->zoom_at_pinch_begin = a->zoom;
}

static void on_pinch_update(void* data,
                            struct zwp_pointer_gesture_pinch_v1* pinch,
                            uint32_t time,
                            wl_fixed_t dx,
                            wl_fixed_t dy,
                            wl_fixed_t scale,
                            wl_fixed_t rotation) {
    (void)pinch;
    (void)time;
    (void)dx;
    (void)dy;
    (void)rotation;
    struct app* a = data;
    double zoom = a->zoom_at_pinch_begin * wl_fixed_to_double(scale);
    if (zoom < 0.2) { zoom = 0.2; }
    if (zoom > 5.0) { zoom = 5.0; }
    a->zoom = zoom;
}

static void on_pinch_end(void* data,
                         struct zwp_pointer_gesture_pinch_v1* pinch,
                         uint32_t serial,
                         uint32_t time,
                         int32_t cancelled) {
    (void)pinch;
    (void)serial;
    (void)time;
    struct app* a = data;
    if (cancelled) { a->zoom = a->zoom_at_pinch_begin; }
}

static const struct zwp_pointer_gesture_pinch_v1_listener pinch_listener = {
    .begin = on_pinch_begin,
    .update = on_pinch_update,
    .end = on_pinch_end,
};

/* Capabilities are DYNAMIC: unplug the mouse and this fires again. */
static void on_seat_capabilities(void* data, struct wl_seat* seat, uint32_t caps) {
    struct app* a = data;

    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !a->pointer) {
        a->pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(a->pointer, &pointer_listener, a);
        if (a->gestures) {
            /* the gesture object is a child of THIS pointer; it goes
             * away with it below */
            a->pinch = zwp_pointer_gestures_v1_get_pinch_gesture(a->gestures, a->pointer);
            zwp_pointer_gesture_pinch_v1_add_listener(a->pinch, &pinch_listener, a);
        }
    } else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && a->pointer) {
        if (a->pinch) {
            zwp_pointer_gesture_pinch_v1_destroy(a->pinch);
            a->pinch = NULL;
        }
        wl_pointer_release(a->pointer);
        a->pointer = NULL;
    }
    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !a->keyboard) {
        a->keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(a->keyboard, &keyboard_listener, a);
    } else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && a->keyboard) {
        wl_keyboard_release(a->keyboard);
        a->keyboard = NULL;
    }
}

static void on_seat_name(void* d, struct wl_seat* s, const char* n) {
    (void)d;
    (void)s;
    (void)n;
}

static const struct wl_seat_listener seat_listener = {
    .capabilities = on_seat_capabilities,
    .name = on_seat_name,
};

/* ------------------------------------------------------------------ */
/* Frame loop (for backends with a redraw hook).                      */
/* ------------------------------------------------------------------ */

static void on_frame_done(void* data, struct wl_callback* cb, uint32_t time_ms);

static const struct wl_callback_listener frame_listener = {
    .done = on_frame_done,
};

void app_schedule_frame(struct app* a) {
    struct wl_callback* cb = wl_surface_frame(a->surface);
    wl_callback_add_listener(cb, &frame_listener, a);
}

void app_advance_clock(struct app* a, uint32_t now_ms) {
    if (a->last_time != 0 && !a->paused) {
        a->anim_ms += (double)(now_ms - a->last_time) * a->speed;
    }
    a->last_time = now_ms;
}

uint32_t app_anim_time(const struct app* a) {
    /* All three fills compute shift = (t/8) mod 256, so the clock's
     * period is 8*256 = 2048 ms. Wrap into [0, 2048) BEFORE the
     * cast: a raw uint32_t cast of a negative clock (scroll into
     * reverse) lands near 2^32, and a float that large has an ulp of
     * hundreds -- the GPU gradients quantize into wide bands (the
     * README §14 fp16 overflow, one precision tier up). */
    double t = fmod(a->anim_ms, 2048.0);
    if (t < 0) { t += 2048.0; }
    return (uint32_t)t;
}

static void on_frame_done(void* data, struct wl_callback* cb, uint32_t time_ms) {
    struct app* a = data;
    wl_callback_destroy(cb); /* one-shot */

    app_advance_clock(a, time_ms);
    app_schedule_frame(a); /* the request rides the commit in redraw */
    a->backend->redraw(a);
}

/* ------------------------------------------------------------------ */
/* Configure/ack: common acks and applies the size; the backend hook  */
/* resizes resources; redraw (if any) maps / repaints.                */
/* ------------------------------------------------------------------ */

static void on_xdg_surface_configure(void* data, struct xdg_surface* xdg_surface, uint32_t serial) {
    struct app* a = data;

    xdg_surface_ack_configure(xdg_surface, serial);
    a->width = a->pending_w;
    a->height = a->pending_h;
    a->configured = 1;

    if (a->backend->configure) { a->backend->configure(a); }

    if (a->backend->redraw) {
        if (!a->animating) {
            a->animating = 1;
            app_schedule_frame(a);
        }
        a->backend->redraw(a);
    }
}

static const struct xdg_surface_listener xdg_surface_listener = {
    .configure = on_xdg_surface_configure,
};

static void on_toplevel_configure(void* data,
                                  struct xdg_toplevel* toplevel,
                                  int32_t w,
                                  int32_t h,
                                  struct wl_array* states) {
    (void)toplevel;
    (void)states;
    struct app* a = data;
    /* STAGE only; 0 means "you pick" */
    a->pending_w = w > 0 ? w : APP_DEFAULT_WIDTH;
    a->pending_h = h > 0 ? h : APP_DEFAULT_HEIGHT;
}

static void on_toplevel_close(void* data, struct xdg_toplevel* toplevel) {
    (void)toplevel;
    struct app* a = data;
    a->running = 0;
}

static const struct xdg_toplevel_listener toplevel_listener = {
    .configure = on_toplevel_configure,
    .close = on_toplevel_close,
};

/* ------------------------------------------------------------------ */

static void init_cursor(struct app* a) {
    const char* size_env = getenv("XCURSOR_SIZE");
    int cursor_size = size_env ? atoi(size_env) : 0;
    if (cursor_size <= 0) { cursor_size = 24; }

    a->cursor_theme = wl_cursor_theme_load(NULL, cursor_size, a->shm);
    struct wl_cursor* cursor =
        a->cursor_theme ? wl_cursor_theme_get_cursor(a->cursor_theme, "default") : NULL;
    if (!cursor && a->cursor_theme) {
        cursor = wl_cursor_theme_get_cursor(a->cursor_theme, "left_ptr");
    }
    if (cursor && cursor->image_count > 0) {
        a->cursor_image = cursor->images[0]; /* frame 0 only; no animation */
        a->cursor_surface = wl_compositor_create_surface(a->compositor);
        wl_surface_attach(a->cursor_surface, wl_cursor_image_get_buffer(a->cursor_image), 0, 0);
        wl_surface_damage_buffer(a->cursor_surface, 0, 0, INT32_MAX, INT32_MAX);
        wl_surface_commit(a->cursor_surface); /* role-less: dormant until set_cursor */
    } else {
        fprintf(stderr, "no cursor theme found -- cursor will be hidden\n");
    }
}

int app_init(struct app* a,
             const struct app_backend* backend,
             const char* title,
             const char* app_id) {
    a->backend = backend;
    a->running = 1;
    a->speed = 1.0;
    a->zoom = 1.0;
    a->overlay = 1; /* H hides it; a hand overlay nobody asked to see is the point */
    a->aux_fd = -1;
    a->pending_w = APP_DEFAULT_WIDTH;
    a->pending_h = APP_DEFAULT_HEIGHT;

    a->display = wl_display_connect(NULL);
    if (!a->display) {
        fprintf(stderr, "no Wayland display (is WAYLAND_DISPLAY set?)\n");
        return -1;
    }

    a->registry = wl_display_get_registry(a->display);
    wl_registry_add_listener(a->registry, &registry_listener, a);
    wl_display_roundtrip(a->display);

    if (!a->compositor || !a->shm || !a->wm_base) {
        fprintf(stderr, "missing wl_compositor, wl_shm or xdg_wm_base\n");
        return -1;
    }
    xdg_wm_base_add_listener(a->wm_base, &wm_base_listener, a);

    if (a->seat) {
        wl_seat_add_listener(a->seat, &seat_listener, a);
        a->xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
        init_cursor(a);
    }

    a->surface = wl_compositor_create_surface(a->compositor);
    a->xdg_surface = xdg_wm_base_get_xdg_surface(a->wm_base, a->surface);
    xdg_surface_add_listener(a->xdg_surface, &xdg_surface_listener, a);
    a->toplevel = xdg_surface_get_toplevel(a->xdg_surface);
    xdg_toplevel_add_listener(a->toplevel, &toplevel_listener, a);
    xdg_toplevel_set_title(a->toplevel, title);
    xdg_toplevel_set_app_id(a->toplevel, app_id);

    /* the empty first commit: "role assigned, please configure me" */
    wl_surface_commit(a->surface);
    return 0;
}

/* The event loop, multi-fd edition. wl_display_dispatch() hides a loop
 * that can only wait on the Wayland socket; to also wait on the camera
 * fd, poll() has to be ours -- and then libwayland's read must be
 * split into its documented three steps so no thread/queue can race
 * the socket read (§11): prepare_read (claim the socket; fails if the
 * queue still has undispatched events -- drain first), poll, then
 * read_events (into every queue) or cancel_read. Mesa's queues on the
 * same socket are why this dance exists: whoever reads delivers to
 * ALL queues. */
void app_run(struct app* a) {
    struct pollfd fds[2] = {
        {.fd = wl_display_get_fd(a->display), .events = POLLIN},
        {.fd = -1, .events = POLLIN},
    };

    while (a->running) {
        while (wl_display_prepare_read(a->display) != 0) {
            if (wl_display_dispatch_pending(a->display) == -1) { return; }
        }
        wl_display_flush(a->display);

        /* re-read every turn: a scene may register its fd from inside
         * a configure callback (the EGLSurface presenter's init) */
        const int nfds = a->aux_fd >= 0 ? 2 : 1;
        fds[1].fd = a->aux_fd;
        fds[0].revents = 0;
        fds[1].revents = 0;
        if (poll(fds, (nfds_t)nfds, -1) < 0) {
            wl_display_cancel_read(a->display);
            if (errno == EINTR) { continue; }
            return;
        }
        if (fds[0].revents & (POLLERR | POLLHUP)) {
            wl_display_cancel_read(a->display);
            return;
        }
        if (fds[0].revents & POLLIN) {
            if (wl_display_read_events(a->display) == -1) { return; }
        } else {
            wl_display_cancel_read(a->display);
        }
        if (wl_display_dispatch_pending(a->display) == -1) { return; }

        if (nfds == 2 && (fds[1].revents & POLLIN) && a->on_aux_fd) { a->on_aux_fd(a); }
    }
}

void app_finish(struct app* a) {
    if (a->cursor_surface) { wl_surface_destroy(a->cursor_surface); }
    if (a->cursor_theme) { wl_cursor_theme_destroy(a->cursor_theme); }
    if (a->pinch) { zwp_pointer_gesture_pinch_v1_destroy(a->pinch); }
    if (a->gestures) { zwp_pointer_gestures_v1_destroy(a->gestures); }
    if (a->pointer) { wl_pointer_release(a->pointer); }
    if (a->keyboard) { wl_keyboard_release(a->keyboard); }
    if (a->seat) { wl_seat_destroy(a->seat); }
    xkb_state_unref(a->xkb_state);
    xkb_keymap_unref(a->xkb_keymap);
    xkb_context_unref(a->xkb_context);
    xdg_toplevel_destroy(a->toplevel);
    xdg_surface_destroy(a->xdg_surface);
    wl_surface_destroy(a->surface);
    xdg_wm_base_destroy(a->wm_base);
    wl_shm_destroy(a->shm);
    wl_compositor_destroy(a->compositor);
    wl_registry_destroy(a->registry);
    wl_display_disconnect(a->display);
}
