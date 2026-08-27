/* infer_ctx_cuda.c -- the CUDA domain: cudaMalloc'd device memory on
 * one non-blocking stream that ONNX Runtime's CUDA EP is handed as its
 * user_compute_stream. Everything that touches the memory -- the
 * generator, the copies, the EP's kernels -- is ordered on that stream,
 * so the stream IS the completion token (cudaStreamSynchronize), the
 * way the WGPU domain's queue is for WebGPU.
 *
 * Runtime API only, on purpose. The driver API would need a current
 * context on every calling thread, and linking libcuda.so.1 would make
 * every binary refuse to load on a machine without the NVIDIA driver;
 * libcudart resolves the driver lazily and reports "no device" instead.
 * The generator arrives as PTX (shaders/infer_gen.cu, compiled by nvcc
 * at configure time, embedded as a string) and is loaded with the
 * runtime's library API -- no NVRTC in the process.
 *
 * Without INFER_HAVE_CUDA (no toolkit, cuDNN or CUDA-enabled ORT found
 * at configure time) every function here is a stub: init fails once,
 * audibly, and the domain never comes up.
 *
 * Threading: the CUDA EP captures its graph in cudaStreamCaptureModeGlobal,
 * under which a CUDA call from ANY thread that is unsafe during capture
 * (an allocation, a synchronisation) invalidates the capture. Nothing
 * here runs from a second thread today -- the hand tracker's worker is
 * the only CUDA caller -- and that is a rule to keep, not an accident.
 */
#include "infer.h"

#include <stdio.h>
#include <string.h>

#include "infer_util.h"

#ifdef INFER_HAVE_CUDA

#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "infer_gen_ptx.h" /* generated: static const char infer_gen_ptx[] */

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        cudaError_t cuda_check_r = (expr);                                                      \
        if (cuda_check_r != cudaSuccess) {                                                      \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,                       \
                    cudaGetErrorString(cuda_check_r));                                          \
            return -1;                                                                          \
        }                                                                                       \
    } while (0)

enum { GEN_BLOCK = 256 };

static cudaStream_t stream_of(const struct infer_ctx_cuda* c) {
    return (cudaStream_t)c->stream;
}

/* The runtime's current device is per THREAD, and a new thread starts
 * on device 0. Cheap when it is already current, so every entry point
 * pays it rather than trusting the caller's thread. */
static int use(const struct infer_ctx_cuda* c) {
    CUDA_CHECK(cudaSetDevice(c->device));
    return 0;
}

int infer_cuda_compiled(void) {
    return 1;
}

int infer_ctx_cuda_init(struct infer_ctx_cuda* c) {
    memset(c, 0, sizeof *c);
    const char* env = getenv("INFER_CUDA_DEVICE");
    c->device = env ? atoi(env) : 0;

    int count = 0;
    cudaError_t r = cudaGetDeviceCount(&count);
    if (r != cudaSuccess || count <= c->device) {
        fprintf(stderr, "cuda: no device %d (%s)\n", c->device,
                r == cudaSuccess ? "none present" : cudaGetErrorString(r));
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaFree(0)); /* create the primary context now, not on the first copy */

    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, c->device));
    snprintf(c->name, sizeof c->name, "%s", prop.name);
    c->cc_major = prop.major;
    c->cc_minor = prop.minor;

    cudaStream_t s = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    c->stream = s;

    /* the generator: PTX for the compute capability CMake was told
     * (INFER_CUDA_ARCH), JIT-compiled by the driver for this GPU */
    cudaLibrary_t lib = NULL;
    r = cudaLibraryLoadData(&lib, infer_gen_ptx, NULL, NULL, 0, NULL, NULL, 0);
    if (r != cudaSuccess) {
        fprintf(stderr, "cuda: loading the generator PTX failed: %s\n", cudaGetErrorString(r));
        cudaStreamDestroy(s);
        c->stream = NULL;
        return -1;
    }
    cudaKernel_t k = NULL;
    r = cudaLibraryGetKernel(&k, lib, "infer_gen");
    if (r != cudaSuccess) {
        fprintf(stderr, "cuda: infer_gen not in the PTX: %s\n", cudaGetErrorString(r));
        cudaLibraryUnload(lib);
        cudaStreamDestroy(s);
        c->stream = NULL;
        return -1;
    }
    c->library = lib;
    c->kernel = k;
    c->owned = 1;

    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    fprintf(stderr,
            "cuda: device %d %s (sm_%d%d), runtime %d.%d, driver %d.%d, one non-blocking stream\n",
            c->device, c->name, c->cc_major, c->cc_minor, rt / 1000, rt % 1000 / 10, drv / 1000,
            drv % 1000 / 10);
    return 0;
}

void infer_ctx_cuda_fini(struct infer_ctx_cuda* c) {
    if (!c->owned) { return; }
    cudaSetDevice(c->device);
    if (c->library) { cudaLibraryUnload((cudaLibrary_t)c->library); }
    if (c->stream) { cudaStreamDestroy(stream_of(c)); }
    /* no cudaDeviceReset: ORT's EP shares the primary context */
    memset(c, 0, sizeof *c);
}

int infer_cuda_alloc(struct infer_ctx_cuda* c, const struct infer_desc* d, struct infer_tensor* t) {
    if (t->mem.cuda.ptr) { return 0; } /* ensure() re-calls this on every copy edge */
    if (use(c) < 0) { return -1; }
    const size_t bytes = infer_desc_bytes_packed(d);
    void* p = NULL;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    t->domain = INFER_DOMAIN_CUDA;
    t->desc = *d;
    t->desc.row_pitch_bytes = d->img_w * 4; /* packed */
    t->mem.cuda.ptr = p;
    t->mem.cuda.size = bytes;
    t->owned = 1;
    t->ready.sync_fd = -1;
    t->released.sync_fd = -1;
    if (infer_verbose) { fprintf(stderr, "cuda: tensor %zu bytes at %p\n", bytes, p); }
    return 0;
}

int infer_cuda_gen(struct infer_ctx_cuda* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t) {
    if (infer_cuda_alloc(c, d, t) < 0) { return -1; }
    if (use(c) < 0) { return -1; }
    unsigned n = (unsigned)infer_desc_elements(d);
    void* out = t->mem.cuda.ptr;
    unsigned s = seed;
    void* args[] = {&out, &n, &s};
    dim3 grid = {(n + GEN_BLOCK - 1) / GEN_BLOCK, 1, 1};
    dim3 block = {GEN_BLOCK, 1, 1};
    /* a cudaKernel_t from cudaLibraryGetKernel is what cudaLaunchKernel
     * takes as `func` (CUDA 12+; the cast is the documented spelling) */
    CUDA_CHECK(cudaLaunchKernel((const void*)c->kernel, grid, block, args, 0, stream_of(c)));
    infer_sync_reset(&t->ready); /* the stream is the token */
    return 0;
}

int infer_cuda_upload(struct infer_ctx_cuda* c, const float* src, struct infer_tensor* t) {
    INFER_CHECK(t->mem.cuda.ptr, "cuda: upload into an unallocated tensor");
    if (use(c) < 0) { return -1; }
    /* pageable source: the runtime stages it before returning, so the
     * caller may reuse `src` at once; the device side is ordered on the
     * stream ahead of whatever runs next (the EP's kernels) */
    CUDA_CHECK(cudaMemcpyAsync(t->mem.cuda.ptr, src, infer_desc_bytes_packed(&t->desc),
                               cudaMemcpyHostToDevice, stream_of(c)));
    infer_sync_reset(&t->ready);
    return 0;
}

int infer_cuda_readback(struct infer_ctx_cuda* c, const struct infer_tensor* t, float* dst) {
    INFER_CHECK(t->mem.cuda.ptr, "cuda: readback of an unallocated tensor");
    if (use(c) < 0) { return -1; }
    /* NOT cudaMemcpy: the stream is non-blocking, so a legacy-stream copy
     * would run ahead of the kernels that write this tensor */
    CUDA_CHECK(cudaMemcpyAsync(dst, t->mem.cuda.ptr, infer_desc_bytes_packed(&t->desc),
                               cudaMemcpyDeviceToHost, stream_of(c)));
    CUDA_CHECK(cudaStreamSynchronize(stream_of(c)));
    return 0;
}

int infer_cuda_wait_idle(struct infer_ctx_cuda* c) {
    if (use(c) < 0) { return -1; }
    CUDA_CHECK(cudaStreamSynchronize(stream_of(c)));
    return 0;
}

void infer_cuda_release(struct infer_ctx_cuda* c, struct infer_tensor* t) {
    if (!t->owned || !t->mem.cuda.ptr) { return; }
    cudaSetDevice(c->device);
    cudaError_t r = cudaFree(t->mem.cuda.ptr);
    if (r != cudaSuccess) { fprintf(stderr, "cuda: cudaFree: %s\n", cudaGetErrorString(r)); }
    t->mem.cuda.ptr = NULL;
}

#else /* !INFER_HAVE_CUDA: the stubs */

int infer_cuda_compiled(void) {
    return 0;
}

int infer_ctx_cuda_init(struct infer_ctx_cuda* c) {
    memset(c, 0, sizeof *c);
    fprintf(stderr,
            "cuda: built without the CUDA domain (no toolkit, cuDNN or CUDA-enabled onnxruntime "
            "at configure time)\n");
    return -1;
}

void infer_ctx_cuda_fini(struct infer_ctx_cuda* c) {
    (void)c;
}

int infer_cuda_alloc(struct infer_ctx_cuda* c, const struct infer_desc* d, struct infer_tensor* t) {
    (void)c;
    (void)d;
    (void)t;
    INFER_CHECK(0, "cuda: built without CUDA");
    return -1;
}

int infer_cuda_gen(struct infer_ctx_cuda* c,
                   const struct infer_desc* d,
                   uint32_t seed,
                   struct infer_tensor* t) {
    (void)seed;
    return infer_cuda_alloc(c, d, t);
}

int infer_cuda_upload(struct infer_ctx_cuda* c, const float* src, struct infer_tensor* t) {
    (void)c;
    (void)src;
    (void)t;
    INFER_CHECK(0, "cuda: built without CUDA");
    return -1;
}

int infer_cuda_readback(struct infer_ctx_cuda* c, const struct infer_tensor* t, float* dst) {
    (void)c;
    (void)t;
    (void)dst;
    INFER_CHECK(0, "cuda: built without CUDA");
    return -1;
}

int infer_cuda_wait_idle(struct infer_ctx_cuda* c) {
    (void)c;
    return -1;
}

void infer_cuda_release(struct infer_ctx_cuda* c, struct infer_tensor* t) {
    (void)c;
    (void)t;
}

#endif
