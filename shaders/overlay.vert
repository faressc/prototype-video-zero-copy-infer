#version 450
/* overlay.vert -- the shared overlay triangles, Vulkan side.
 *
 * The same one multiply-add the GLES overlay does, and the same division
 * of labour: the CALLER computed the rectangle and knows which way its
 * target stores rows, so it hands over a scale and an offset that are
 * already correct and this shader has no opinion about orientation.
 * (Two bugs came from splitting that decision between the caller and the
 * shader; see hand/hand_overlay_gles.h.)
 */

layout(location = 0) in vec2 a_pos; // frame-normalised, [0,1] x [0,1], y DOWN
layout(location = 1) in vec4 a_col; // R8G8B8A8_UNORM, so R is the low byte

layout(push_constant) uniform Push {
    vec4 rect; // xy = scale, zw = offset, straight into NDC
}
push;

layout(location = 0) out vec4 v_col;

void main() {
    v_col = a_col;
    gl_Position = vec4(a_pos * push.rect.xy + push.rect.zw, 0.0, 1.0);
}
