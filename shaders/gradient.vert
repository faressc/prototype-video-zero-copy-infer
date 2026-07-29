/* Fullscreen triangle without any vertex buffer: gl_VertexIndex 0,1,2
 * generates (0,0), (2,0), (0,2) -> clip-space (-1,-1), (3,-1), (-1,3).
 * The part outside the screen is clipped; the rest covers every pixel.
 * Vulkan clip space has y pointing DOWN, so uv (0,0) lands at the
 * top-left -- matching the pixel coordinates of main.c's gradient. */
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
