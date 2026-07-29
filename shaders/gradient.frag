/* main.c's fill_gradient() inner loop, as a fragment shader: executed
 * once per pixel, in parallel, on the GPU. mod() replaces & 255.
 * Push constants are Vulkan's "small uniforms": up to 128 bytes passed
 * directly in the command buffer, no descriptor sets needed. */
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(push_constant) uniform PC {
    vec2 size;   /* window size in pixels */
    vec2 cursor; /* pointer position, surface-local */
    float time;  /* animation clock, ms */
} pc;

void main() {
    vec2 px = v_uv * pc.size;
    float shift = pc.time / 8.0;
    float r = mod((px.x - pc.cursor.x) * 255.0 / pc.size.x + shift, 256.0) / 255.0;
    float g = mod((px.y - pc.cursor.y) * 255.0 / pc.size.y, 256.0) / 255.0;
    o_color = vec4(r, g, 1.0 - r, 1.0);
}
