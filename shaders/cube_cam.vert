/* cube_cam.vert -- the cube with texture coordinates: cube.vert plus a
 * uv attribute passed through as a varying. The rasterizer's
 * perspective-correct interpolation of v_uv is what keeps the camera
 * image from bending on faces seen at an angle.
 *
 * Lighting is computed HERE, in model space: the CPU pushes the light
 * direction already rotated by the inverse model rotation, so
 * normal . light needs no model matrix -- that keeps the vertex push
 * constants at 80 bytes and leaves room for the fragment stage's
 * effect parameters within the 128-byte minimum. Exact for this cube:
 * its normals are constant per face, so per-vertex = per-pixel. */
#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
layout(location = 3) in vec2 a_uv;

layout(location = 0) out float v_diffuse;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 light_model; /* xyz: light direction in model space */
} pc;

void main() {
    v_diffuse = max(dot(a_normal, pc.light_model.xyz), 0.0);
    v_uv = a_uv;
    gl_Position = pc.mvp * vec4(a_pos, 1.0);
}
