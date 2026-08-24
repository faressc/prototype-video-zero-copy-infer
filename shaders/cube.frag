/* cube.frag -- one directional light, Lambert diffuse plus ambient.
 * v_normal arrives interpolated (flat across a face here, since all
 * four corners share the normal), renormalized to be safe. */
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;

layout(location = 0) out vec4 o_color;

const vec3 LIGHT_DIR = normalize(vec3(0.4, 0.8, 1.0));

void main() {
    float diffuse = max(dot(normalize(v_normal), LIGHT_DIR), 0.0);
    o_color = vec4(v_color * (0.25 + 0.75 * diffuse), 1.0);
}
