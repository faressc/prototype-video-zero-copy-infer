/* infer_engine_ort.c -- ONNX Runtime behind the engine seam: one
 * session, one IO binding, inputs and outputs both bound as tensors
 * the CALLER owns in whatever domain the EP works in (host pointers for
 * the CPU EP, WGPUBuffers for the WebGPU EP). Nothing is allocated or
 * copied here: the engine reads its input in place and writes its
 * outputs in place, and completion is the caller's token, not this
 * adapter returning.
 *
 * The WebGPU EP is given OUR Dawn device (deviceId >= 1 selects the
 * user-supplied instance/device; ORT prefixes the option keys itself).
 * A WebGPU tensor is an OrtValue whose data pointer IS the WGPUBuffer
 * handle, on memory info "WebGPU_Buffer" with OrtDevice id 0 -- the id
 * the EP's allocator reports, NOT the deviceId option (see below).
 */
#include "infer.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_c_api.h>

#include "infer_util.h"

/* Two different "device ids". The provider option deviceId >= 1 tells
 * the WebGPU EP to use the Dawn device we hand it. The OrtDevice id on a
 * memory info is something else: the EP's allocator reports its buffers
 * as OrtDevice{GPU, DEFAULT, NONE, 0} (webgpu/allocator.h, a constexpr)
 * whatever deviceId was. A tensor labelled with any other OrtDevice is,
 * to the session, on a foreign device: it copies it into an EP-owned
 * buffer before every run and copies outputs back after -- a hidden
 * device copy under a row that says ZERO_COPY, and one that a captured
 * graph does not replay, so replays read a stale copy. Measured before
 * the ids were told apart: every capture-on cell STALE. */
enum { ORT_WEBGPU_DEVICE_ID = 1, ORT_WEBGPU_MEM_DEVICE_ID = 0, MAX_IO = INFER_ENGINE_MAX_OUTPUTS };

struct infer_engine_ort {
    const OrtApi* api;
    OrtEnv* env;
    OrtSessionOptions* so;
    OrtSession* session;
    OrtIoBinding* binding;
    OrtMemoryInfo* cpu_mem;
    OrtMemoryInfo* gpu_mem;
    OrtAllocator* alloc;
    OrtRunOptions* run_opts;
    enum infer_ep ep;
    char* input_name;
    char* output_names[MAX_IO];
    size_t output_count;
    struct infer_desc input_desc;
    struct infer_desc output_desc[MAX_IO];
    OrtValue* input_value;
    OrtValue* output_values[MAX_IO];
};

#define ORT_CHECK(e, expr)                                                       \
    do {                                                                         \
        OrtStatus* st_ = (expr);                                                 \
        if (st_) {                                                               \
            fprintf(stderr, "ORT: %s: %s\n", #expr, (e)->api->GetErrorMessage(st_)); \
            (e)->api->ReleaseStatus(st_);                                        \
            return -1;                                                           \
        }                                                                        \
    } while (0)

enum infer_domain infer_engine_ort_input_domain(enum infer_ep ep) {
    return ep == INFER_EP_WEBGPU ? INFER_DOMAIN_WGPU : INFER_DOMAIN_CPU;
}

enum infer_domain infer_engine_ort_output_domain(enum infer_ep ep) {
    /* the kernels write where they read: the same device */
    return infer_engine_ort_input_domain(ep);
}

/* input 0 or output i: float32, dims with dynamic entries forced to 1 */
static int query_desc(struct infer_engine_ort* e, int is_input, size_t i, struct infer_desc* d) {
    const OrtApi* api = e->api;
    OrtTypeInfo* ti = NULL;
    if (is_input) {
        ORT_CHECK(e, api->SessionGetInputTypeInfo(e->session, i, &ti));
    } else {
        ORT_CHECK(e, api->SessionGetOutputTypeInfo(e->session, i, &ti));
    }
    const OrtTensorTypeAndShapeInfo* ts = NULL;
    ORT_CHECK(e, api->CastTypeInfoToTensorInfo(ti, &ts));
    ONNXTensorElementDataType et;
    ORT_CHECK(e, api->GetTensorElementType(ts, &et));
    size_t nd = 0;
    ORT_CHECK(e, api->GetDimensionsCount(ts, &nd));
    int64_t dims[INFER_MAX_DIMS] = {0};
    if (nd > INFER_MAX_DIMS) { nd = INFER_MAX_DIMS; }
    ORT_CHECK(e, api->GetDimensions(ts, dims, nd));
    api->ReleaseTypeInfo(ti);
    INFER_CHECK(et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "%s %zu is not float32",
                is_input ? "input" : "output",
                i);
    memset(d, 0, sizeof *d);
    d->dtype = INFER_F32;
    d->ndim = (int)nd;
    for (size_t j = 0; j < nd; j++) { d->dims[j] = dims[j] > 0 ? dims[j] : 1; }
    infer_desc_set_image(d);
    return 0;
}

static int create(struct infer_engine_ort* e, struct infer_ctx* c, const char* model_path) {
    const OrtApi* api = e->api;
    ORT_CHECK(e, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hello_inference", &e->env));
    ORT_CHECK(e, api->CreateSessionOptions(&e->so));
    ORT_CHECK(e, api->SetSessionGraphOptimizationLevel(e->so, ORT_ENABLE_ALL));
    ORT_CHECK(e, api->SetSessionLogSeverityLevel(e->so, getenv("INFER_ORT_VERBOSE") ? 0 : 2));

    if (e->ep == INFER_EP_WEBGPU) {
        INFER_CHECK(c->have & INFER_WANT_WGPU, "WebGPU EP needs the WebGPU context");
        char inst[32], dev[32], id[8];
        snprintf(inst, sizeof inst, "%" PRIuPTR, (uintptr_t)c->wgpu.instance);
        snprintf(dev, sizeof dev, "%" PRIuPTR, (uintptr_t)c->wgpu.device);
        snprintf(id, sizeof id, "%d", ORT_WEBGPU_DEVICE_ID);
        const char* keys[16] = {"deviceId", "webgpuInstance", "webgpuDevice", "preferredLayout",
                                "dawnBackendType"};
        const char* vals[16] = {id, inst, dev, "NHWC", "Vulkan"};
        size_t nkeys = 5;
#ifdef INFER_ORT_EXTERNAL_DAWN
        /* ORT built with onnxruntime_USE_EXTERNAL_DAWN: it links no Dawn
         * implementation, only dawn_proc thunks, and must be handed the
         * proc table of OUR libwebgpu_dawn.so (the one the device was
         * created on). dawn::native::GetProcs() is a C++ symbol; resolve
         * it by its Itanium-mangled name -- it returns a reference, which
         * is a pointer in the ABI. Installed once per process by ORT. */
        static char procs[32];
        typedef const void* (*get_procs_fn)(void);
        get_procs_fn get_procs = (get_procs_fn)dlsym(RTLD_DEFAULT, "_ZN4dawn6native8GetProcsEv");
        INFER_CHECK(get_procs != NULL, "dawn::native::GetProcs not found in the loaded Dawn");
        snprintf(procs, sizeof procs, "%" PRIuPTR, (uintptr_t)get_procs());
        keys[nkeys] = "dawnProcTable";
        vals[nkeys] = procs;
        nkeys++;
#endif
        /* experiments: INFER_ORT_OPTS="validationMode=disabled,enableGraphCapture=1" */
        static char extra[256];
        const char* env = getenv("INFER_ORT_OPTS");
        int capture_set = 0;
        if (env) {
            snprintf(extra, sizeof extra, "%s", env);
            for (char* tok = strtok(extra, ","); tok && nkeys < 15; tok = strtok(NULL, ",")) {
                char* eq = strchr(tok, '=');
                if (!eq) { continue; }
                *eq = '\0';
                if (strcmp(tok, "enableGraphCapture") == 0) { capture_set = 1; }
                keys[nkeys] = tok;
                vals[nkeys] = eq + 1;
                nkeys++;
            }
        }
        /* Graph capture: ORT records its command buffers once and replays
         * them -- on the palm detector 3.3 ms -> 1.1 ms of host time per
         * run, GPU time unchanged. Three things had to be true first,
         * each found by hello_inference's stale check: outputs bound on
         * the device with completion taken from the caller's token (with
         * host-bound outputs Run() returned ~8 ms early and every
         * iteration read the previous outputs); the OrtDevice id above
         * (a replay re-reads ORT's private copy of a "foreign" input);
         * and the same buffers bound every run (a replay re-dispatches
         * the bind groups it captured, so rotating slots alternate stale
         * outputs) -- which is the contract of this adapter. */
        if (!capture_set) {
            keys[nkeys] = "enableGraphCapture";
            vals[nkeys] = "1";
            nkeys++;
        }
        ORT_CHECK(e, api->SessionOptionsAppendExecutionProvider(e->so, "WebGPU", keys, vals, nkeys));
        ORT_CHECK(e,
                  api->CreateMemoryInfo("WebGPU_Buffer",
                                        OrtDeviceAllocator,
                                        ORT_WEBGPU_MEM_DEVICE_ID,
                                        OrtMemTypeDefault,
                                        &e->gpu_mem));
    }
    ORT_CHECK(e, api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &e->cpu_mem));
    ORT_CHECK(e, api->CreateSession(e->env, model_path, e->so, &e->session));
    ORT_CHECK(e, api->GetAllocatorWithDefaultOptions(&e->alloc));
    ORT_CHECK(e, api->CreateRunOptions(&e->run_opts));

    size_t n_in = 0;
    ORT_CHECK(e, api->SessionGetInputCount(e->session, &n_in));
    INFER_CHECK(n_in == 1, "model must have exactly one input (has %zu)", n_in);
    ORT_CHECK(e, api->SessionGetInputName(e->session, 0, e->alloc, &e->input_name));
    ORT_CHECK(e, api->SessionGetOutputCount(e->session, &e->output_count));
    INFER_CHECK(e->output_count <= MAX_IO, "too many outputs");
    for (size_t i = 0; i < e->output_count; i++) {
        ORT_CHECK(e, api->SessionGetOutputName(e->session, i, e->alloc, &e->output_names[i]));
    }
    if (query_desc(e, 1, 0, &e->input_desc) < 0) { return -1; }
    for (size_t i = 0; i < e->output_count; i++) {
        if (query_desc(e, 0, i, &e->output_desc[i]) < 0) { return -1; }
    }

    /* the binding is empty until the caller binds its own tensors --
     * every output must be bound before a run */
    ORT_CHECK(e, api->CreateIoBinding(e->session, &e->binding));
    return 0;
}

struct infer_engine_ort* infer_engine_ort_create(struct infer_ctx* c,
                                                 enum infer_ep ep,
                                                 const char* model_path) {
    struct infer_engine_ort* e = calloc(1, sizeof *e);
    e->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    e->ep = ep;
    if (!e->api) {
        fprintf(stderr, "ORT: API version %d not available\n", ORT_API_VERSION);
        free(e);
        return NULL;
    }
    if (create(e, c, model_path) < 0) {
        infer_engine_ort_destroy(e);
        return NULL;
    }
    return e;
}

int infer_engine_ort_input_desc(const struct infer_engine_ort* e, struct infer_desc* d) {
    *d = e->input_desc;
    return 0;
}

size_t infer_engine_ort_output_count(const struct infer_engine_ort* e) {
    return e->output_count;
}

int infer_engine_ort_output_desc(const struct infer_engine_ort* e, size_t i, struct infer_desc* d) {
    INFER_CHECK(i < e->output_count, "output %zu does not exist", i);
    *d = e->output_desc[i];
    return 0;
}

/* wrap a caller-owned tensor: ORT neither copies nor frees it */
static int wrap(struct infer_engine_ort* e,
                const struct infer_tensor* t,
                const struct infer_desc* shape,
                OrtValue** out) {
    const OrtApi* api = e->api;
    void* data = t->domain == INFER_DOMAIN_WGPU ? (void*)t->mem.wgpu.buffer : (void*)t->mem.cpu.ptr;
    const OrtMemoryInfo* mi = t->domain == INFER_DOMAIN_WGPU ? e->gpu_mem : e->cpu_mem;
    INFER_CHECK(data, "tensor has no memory");
    if (*out) {
        api->ReleaseValue(*out);
        *out = NULL;
    }
    ORT_CHECK(e,
              api->CreateTensorWithDataAsOrtValue(mi,
                                                  data,
                                                  infer_desc_bytes_packed(shape),
                                                  shape->dims,
                                                  (size_t)shape->ndim,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  out));
    return 0;
}

int infer_engine_ort_bind_input(struct infer_engine_ort* e, const struct infer_tensor* t) {
    enum infer_domain want = infer_engine_ort_input_domain(e->ep);
    INFER_CHECK(t->domain == want,
                "input tensor is in %s, the %s EP reads %s",
                infer_domain_name(t->domain),
                infer_ep_name(e->ep),
                infer_domain_name(want));
    if (wrap(e, t, &t->desc, &e->input_value) < 0) { return -1; }
    ORT_CHECK(e, e->api->BindInput(e->binding, e->input_name, e->input_value));
    return 0;
}

int infer_engine_ort_bind_output(struct infer_engine_ort* e, size_t i, const struct infer_tensor* t) {
    INFER_CHECK(i < e->output_count, "output %zu does not exist", i);
    enum infer_domain want = infer_engine_ort_output_domain(e->ep);
    INFER_CHECK(t->domain == want,
                "output tensor is in %s, the %s EP writes %s",
                infer_domain_name(t->domain),
                infer_ep_name(e->ep),
                infer_domain_name(want));
    /* a pre-bound output must be exactly the shape the kernel produces */
    const struct infer_desc* want_shape = &e->output_desc[i];
    INFER_CHECK(t->desc.ndim == want_shape->ndim &&
                    memcmp(t->desc.dims, want_shape->dims, sizeof(int64_t) * (size_t)want_shape->ndim) == 0,
                "output %zu: bound tensor has the wrong shape",
                i);
    if (wrap(e, t, want_shape, &e->output_values[i]) < 0) { return -1; }
    ORT_CHECK(e, e->api->BindOutput(e->binding, e->output_names[i], e->output_values[i]));
    return 0;
}

int infer_engine_ort_run(struct infer_engine_ort* e) {
    for (size_t i = 0; i < e->output_count; i++) {
        INFER_CHECK(e->output_values[i], "output %zu is not bound", i);
    }
    ORT_CHECK(e, e->api->RunWithBinding(e->session, e->run_opts, e->binding));
    return 0;
}

void infer_engine_ort_destroy(struct infer_engine_ort* e) {
    if (!e) { return; }
    const OrtApi* api = e->api;
    if (e->input_value) { api->ReleaseValue(e->input_value); }
    for (size_t i = 0; i < MAX_IO; i++) {
        if (e->output_values[i]) { api->ReleaseValue(e->output_values[i]); }
    }
    if (e->binding) { api->ReleaseIoBinding(e->binding); }
    if (e->input_name) { api->ReleaseStatus(api->AllocatorFree(e->alloc, e->input_name)); }
    for (size_t i = 0; i < e->output_count; i++) {
        if (e->output_names[i]) {
            api->ReleaseStatus(api->AllocatorFree(e->alloc, e->output_names[i]));
        }
    }
    if (e->run_opts) { api->ReleaseRunOptions(e->run_opts); }
    if (e->session) { api->ReleaseSession(e->session); }
    if (e->so) { api->ReleaseSessionOptions(e->so); }
    if (e->cpu_mem) { api->ReleaseMemoryInfo(e->cpu_mem); }
    if (e->gpu_mem) { api->ReleaseMemoryInfo(e->gpu_mem); }
    if (e->env) { api->ReleaseEnv(e->env); }
    free(e);
}
