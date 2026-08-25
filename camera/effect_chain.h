/* effect_chain.h -- which effects are on (a bitmask, one bit per number
 * key) and the sequence of passes that realizes them.
 *
 * Effects layer: the active ones run one after another, each pass
 * reading the previous stage. The order is fixed here, not by the
 * order keys were pressed, because it is the order that makes sense:
 * quantize (mosaic) first, then smooth (blur), then detect (Sobel),
 * then recolor (thermal). Blur is two passes (horizontal, vertical).
 *
 * Every backend renders the same pass list: the camera -> scratch A
 * -> scratch B -> A ... -> the window (or the cube). With no effects
 * on, a single passthrough pass.
 */
#ifndef HELLO_WAYLAND_EFFECT_CHAIN_H
#define HELLO_WAYLAND_EFFECT_CHAIN_H

enum { /* app.effect bits: key N toggles bit N-1 */
       FX_THERMAL = 1 << 0,
       FX_SOBEL = 1 << 1,
       FX_MOSAIC = 1 << 2,
       FX_BLUR = 1 << 3,
       FX_COUNT = 4,
};

enum { /* U_EFFECT values, one per pass -- must match cam_effects.glsl */
       PASS_NONE = 0,
       PASS_THERMAL = 1,
       PASS_SOBEL = 2,
       PASS_MOSAIC = 3,
       PASS_BLUR_H = 4,
       PASS_BLUR_V = 5,
};

enum { EFFECT_MAX_PASSES = 8 };

static inline int effect_chain_passes(int mask, int out[EFFECT_MAX_PASSES]) {
    int n = 0;
    if (mask & FX_MOSAIC) { out[n++] = PASS_MOSAIC; }
    if (mask & FX_BLUR) {
        out[n++] = PASS_BLUR_H;
        out[n++] = PASS_BLUR_V;
    }
    if (mask & FX_SOBEL) { out[n++] = PASS_SOBEL; }
    if (mask & FX_THERMAL) { out[n++] = PASS_THERMAL; }
    if (n == 0) { out[n++] = PASS_NONE; }
    return n;
}

#endif
