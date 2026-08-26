#version 450
/* overlay.frag -- flat colour, blended by the pipeline's blend state. */

layout(location = 0) in vec4 v_col;
layout(location = 0) out vec4 o_col;

void main() {
    o_col = v_col;
}
