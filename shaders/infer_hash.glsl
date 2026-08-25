/* infer_hash.glsl -- the deterministic tensor generator, shared by the
 * Vulkan (#include, GLSL 4.50) and GLES (string concatenation, ES 3.10)
 * compute shaders; the WGSL and C versions are copies of the same two
 * lines. uint32 wraparound, one exact 16-bit int->float conversion, a
 * power-of-two divide: bit-identical on every backend.
 */
uint infer_hash(uint i, uint seed) { return ((i + seed) * 2654435761u) >> 8; }
float infer_gen(uint i, uint seed) { return float(infer_hash(i, seed) & 0xFFFFu) / 65536.0 - 0.5; }
