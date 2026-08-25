/* cam.vert -- the fullscreen triangle again (gradient.vert), with the
 * uv passed on for texture sampling. Vulkan clip space is y-down, so
 * uv (0,0) is the top-left -- and camera row 0 is the top: no flip. */
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
