// infer_gen.cu -- CUDA: fill a packed float buffer with the deterministic
// generator (the same two lines as infer_hash.glsl / infer_gen.wgsl /
// infer_gen_value() in infer_tensor.c: uint32 wraparound arithmetic and
// one exact int -> float conversion, so every domain agrees bit for bit).
//
// Compiled to PTX by nvcc at configure time and embedded as a C string
// (CMakeLists.txt, section 8); loaded at run time with cudaLibraryLoadData.
// Written out rather than #including infer_hash.glsl: nvcc has no `uint`.

extern "C" __global__ void infer_gen(float* out, unsigned n, unsigned seed) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const unsigned h = ((i + seed) * 2654435761u) >> 8;
    out[i] = (float)(h & 0xFFFFu) / 65536.0f - 0.5f;
}
