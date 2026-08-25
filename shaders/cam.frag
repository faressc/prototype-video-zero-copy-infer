/* cam.frag -- Vulkan wrapper around the shared effect chain: ONE pass.
 *
 * binding 0: the camera. With VkSamplerYcbcrConversion the sampler
 *            itself converts NV12 -> RGB; the shader sees a plain
 *            sampler2D. (Fallback path: an RGBA image the CPU filled.)
 * binding 1, 2: the two scratch textures the chain ping-pongs through.
 *
 * pc.src says which of the three this pass reads; pc.effect which
 * effect it applies. Descriptor sets are the "which textures" channel
 * push constants can't be; the per-pass scalars still ride them. */
#version 450
/* glslc enables #include implicitly; glslangValidator needs it spelled out */
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform sampler2D u_cam;
layout(set = 0, binding = 1) uniform sampler2D u_tmp0;
layout(set = 0, binding = 2) uniform sampler2D u_tmp1;

layout(push_constant) uniform PC {
    vec2 texel;
    float time;
    int effect;
    int src; /* 0 camera, 1 scratch A, 2 scratch B */
} pc;

#define SRC(uv) \
    ((pc.src == 1) ? texture(u_tmp0, uv).rgb \
                   : (pc.src == 2) ? texture(u_tmp1, uv).rgb : texture(u_cam, uv).rgb)
#define U_TEXEL pc.texel
#define U_EFFECT pc.effect
#define U_TIME pc.time

#include "cam_effects.glsl"

void main() {
    o_color = vec4(apply_effect(v_uv), 1.0);
}
