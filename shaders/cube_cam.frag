/* cube_cam.frag -- the camera as the cube's surface, as the LAST pass
 * of the effect chain: same bindings and pass parameters as cam.frag
 * (the push constants sit after the vertex stage's, at offset 96),
 * sampled at the mesh uv, then lit. */
#version 450

layout(location = 0) in float v_diffuse;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform sampler2D u_cam;
layout(set = 0, binding = 1) uniform sampler2D u_tmp0;
layout(set = 0, binding = 2) uniform sampler2D u_tmp1;

layout(push_constant) uniform PC {
    layout(offset = 96) vec2 texel;
    float time;
    int effect;
    int src;
} pc;

#define SRC(uv) \
    ((pc.src == 1) ? texture(u_tmp0, uv).rgb \
                   : (pc.src == 2) ? texture(u_tmp1, uv).rgb : texture(u_cam, uv).rgb)
#define U_TEXEL pc.texel
#define U_EFFECT pc.effect
#define U_TIME pc.time

#include "cam_effects.glsl"

void main() {
    o_color = vec4(apply_effect(v_uv) * (0.35 + 0.65 * v_diffuse), 1.0);
}
