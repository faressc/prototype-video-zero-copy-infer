/* cube.vert -- a real vertex stage at last: positions fetched from a
 * vertex buffer (see the filled-in VkPipelineVertexInputStateCreateInfo
 * in scene_cube_vk.c), transformed by matrices from push constants.
 *
 * mvp   = projection * view * model   (128-byte push budget: exactly
 * model = rotation only, so it also    two mat4s -- the guaranteed
 *         transforms normals            minimum, fully spent)
 *
 * Vulkan clip space: y DOWN, z in [0,1]. The projection matrix was
 * built for that (mat4_perspective y_down=1, z_zero_to_one=1), so the
 * GL-convention mesh lands upright without any flip here. */
#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_color;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
} pc;

void main() {
    v_normal = mat3(pc.model) * a_normal;
    v_color = a_color;
    gl_Position = pc.mvp * vec4(a_pos, 1.0);
}
