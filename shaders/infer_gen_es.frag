/* infer_gen_es.frag -- GLES: fill the tensor's byte image by RENDERING
 * into it (an EGLImage-backed renderbuffer on a gbm_bo). Each fragment
 * is one texel = the four bytes of one float; unpackUnorm4x8 makes the
 * unorm8 store exact. The C side prepends "#version 310 es" and
 * infer_hash.glsl. (imageStore into an imported linear texture goes
 * through a shadow copy on Mesa/agx; rendering does not.)
 */
precision highp float;
precision highp int;

uniform uvec4 p; /* img_w, img_h, unused, seed */

out vec4 frag;

void main() {
    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);
    frag = unpackUnorm4x8(floatBitsToUint(infer_gen(y * p.x + x, p.w)));
}
