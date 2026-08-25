/* cam_effects.glsl -- the effect chain, shared by every backend.
 *
 * Effects LAYER: each active one is a separate pass that reads the
 * previous stage (the camera for the first pass, a scratch texture
 * afterwards) and writes the next. This file is the body of ONE pass;
 * the C side (camera/effect_chain.h) decides the sequence.
 *
 * Written in the intersection of GLSL ES 1.00 (the GLES scenes) and
 * GLSL 4.50 (Vulkan): no arrays of constants, no integer bit ops, no
 * texture() vs texture2D() -- the includer provides:
 *
 *   SRC(uv)   vec3   this pass's input stage
 *   U_TEXEL   vec2   1 / camera size in pixels, for neighborhood taps
 *   U_EFFECT  int    which pass (PASS_* in effect_chain.h)
 *   U_TIME    float  seconds
 *
 * The CPU scene (scene_cam_shm.c) reimplements these in C; keep them
 * in step when editing.
 */

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722)); /* BT.709 */
}

/* 1: luma through a false-color ramp (black-blue-magenta-yellow-white) */
vec3 thermal(float y) {
    vec3 a = mix(vec3(0.0, 0.0, 0.25), vec3(0.6, 0.0, 0.7), clamp(y * 3.0, 0.0, 1.0));
    vec3 b = mix(a, vec3(1.0, 0.85, 0.1), clamp(y * 3.0 - 1.0, 0.0, 1.0));
    return mix(b, vec3(1.0), clamp(y * 3.0 - 2.0, 0.0, 1.0));
}

/* 2: Sobel edge magnitude -- two 3x3 kernels over the luma of the 8
 * neighbors. A hand-unrolled convolution; the NN stage is this with
 * learned weights and many channels. */
vec3 sobel(vec2 uv) {
    float tx = U_TEXEL.x;
    float ty = U_TEXEL.y;
    float tl = luma(SRC(uv + vec2(-tx, -ty)));
    float tc = luma(SRC(uv + vec2(0.0, -ty)));
    float tr = luma(SRC(uv + vec2(tx, -ty)));
    float ml = luma(SRC(uv + vec2(-tx, 0.0)));
    float mr = luma(SRC(uv + vec2(tx, 0.0)));
    float bl = luma(SRC(uv + vec2(-tx, ty)));
    float bc = luma(SRC(uv + vec2(0.0, ty)));
    float br = luma(SRC(uv + vec2(tx, ty)));
    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    float g = clamp(sqrt(gx * gx + gy * gy) * 2.0, 0.0, 1.0);
    return vec3(g);
}

/* 3: mosaic -- snap uv to a grid of square cells before sampling */
vec3 pixelate(vec2 uv) {
    float cells_x = 64.0;
    float cells_y = cells_x * U_TEXEL.x / U_TEXEL.y; /* keep cells square */
    vec2 cells = vec2(cells_x, cells_y);
    vec2 snapped = (floor(uv * cells) + 0.5) / cells;
    return SRC(snapped);
}

/* 4/5: separable Gaussian, 9 taps, horizontal then vertical -- two
 * passes of the chain, 18 reads per pixel instead of 81 for the 2D
 * kernel. The reason multi-pass rendering exists. */
vec3 blur(vec2 uv, vec2 d) {
    vec3 c = SRC(uv) * 0.2270;
    c += (SRC(uv + d * 1.0) + SRC(uv - d * 1.0)) * 0.1945;
    c += (SRC(uv + d * 2.0) + SRC(uv - d * 2.0)) * 0.1216;
    c += (SRC(uv + d * 3.0) + SRC(uv - d * 3.0)) * 0.0540;
    c += (SRC(uv + d * 4.0) + SRC(uv - d * 4.0)) * 0.0162;
    return c;
}

vec3 apply_effect(vec2 uv) {
    if (U_EFFECT == 1) { return thermal(luma(SRC(uv))); }
    if (U_EFFECT == 2) { return sobel(uv); }
    if (U_EFFECT == 3) { return pixelate(uv); }
    if (U_EFFECT == 4) { return blur(uv, vec2(U_TEXEL.x * 2.0, 0.0)); }
    if (U_EFFECT == 5) { return blur(uv, vec2(0.0, U_TEXEL.y * 2.0)); }
    return SRC(uv);
}
