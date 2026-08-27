/* infer_engine_ort.c -- ONNX Runtime behind the engine seam: one
 * session, one IO binding, inputs and outputs both bound as tensors
 * the CALLER owns in whatever domain the EP works in (host pointers for
 * the CPU EP, WGPUBuffers for the WebGPU EP, device pointers on our
 * stream for the CUDA EP). Nothing is allocated or copied here: the
 * engine reads its input in place and writes its outputs in place, and
 * completion is the caller's token, not this adapter returning.
 *
 * The WebGPU EP is given OUR Dawn device (deviceId >= 1 selects the
 * user-supplied instance/device; ORT prefixes the option keys itself).
 * A WebGPU tensor is an OrtValue whose data pointer IS the WGPUBuffer
 * handle, on memory info "WebGPU_Buffer" with OrtDevice id 0 -- the id
 * the EP's allocator reports, NOT the deviceId option (see below).
 *
 * The CUDA EP is given OUR stream (user_compute_stream, through the
 * opaque V2 options -- the string-keyed generic append does not take
 * "CUDA" in ORT 1.29) and told not to synchronise it at the end of Run
 * (disable_synchronize_execution_providers), so Run is submission there
 * too. A CUDA tensor is an OrtValue whose data pointer is the device
 * pointer, on memory info "Cuda" with the device ordinal as its
 * OrtDevice id -- for this EP the two ids ARE the same number
 * (allocator.cc maps "Cuda" to OrtDevice{GPU, DEFAULT, NVIDIA, id}, the
 * device its allocator reports), so no foreign-device copy arises.
 */
#include "infer.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_run_options_config_keys.h>

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
enum {
    ORT_WEBGPU_DEVICE_ID = 1,
    ORT_WEBGPU_MEM_DEVICE_ID = 0,
    MAX_IO = INFER_ENGINE_MAX_OUTPUTS,
    MAX_OPTS = 24,
};

struct infer_engine_ort {
    const OrtApi* api;
    OrtEnv* env;
    OrtSessionOptions* so;
    OrtSession* session;
    OrtIoBinding* binding;
    OrtMemoryInfo* cpu_mem;
    OrtMemoryInfo* gpu_mem; /* the GPU EP's memory info: WebGPU_Buffer or Cuda */
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
    switch (ep) {
    case INFER_EP_WEBGPU: return INFER_DOMAIN_WGPU;
    case INFER_EP_CUDA: return INFER_DOMAIN_CUDA;
    default: return INFER_DOMAIN_CPU;
    }
}

enum infer_domain infer_engine_ort_output_domain(enum infer_ep ep) {
    /* the kernels write where they read: the same device */
    return infer_engine_ort_input_domain(ep);
}

/* the graph-capture option, in each EP's own spelling */
static const char* capture_key(enum infer_ep ep) {
    return ep == INFER_EP_CUDA ? "enable_cuda_graph" : "enableGraphCapture";
}

/* ------------------------------------------------------------------ */
/* provider options: defaults, then INFER_ORT_OPTS overrides            */
/* ------------------------------------------------------------------ */

struct opts {
    const char* keys[MAX_OPTS];
    const char* vals[MAX_OPTS];
    size_t n;
};

/* set, or override an earlier key of the same name */
static int opts_set(struct opts* o, const char* key, const char* val) {
    for (size_t i = 0; i < o->n; i++) {
        if (strcmp(o->keys[i], key) == 0) {
            o->vals[i] = val;
            return 0;
        }
    }
    INFER_CHECK(o->n < MAX_OPTS, "too many EP options");
    o->keys[o->n] = key;
    o->vals[o->n] = val;
    o->n++;
    return 0;
}

/* experiments: INFER_ORT_OPTS="validationMode=disabled,enableGraphCapture=0"
 * (WebGPU) or "enable_cuda_graph=0,cudnn_conv_algo_search=EXHAUSTIVE,
 * prefer_nhwc=1" (CUDA). The keys and values point into one static
 * buffer: ORT copies them on append, and one session is created at a
 * time. */
static int opts_from_env(struct opts* o) {
    static char extra[256];
    const char* env = getenv("INFER_ORT_OPTS");
    if (!env) { return 0; }
    snprintf(extra, sizeof extra, "%s", env);
    for (char* tok = strtok(extra, ","); tok; tok = strtok(NULL, ",")) {
        char* eq = strchr(tok, '=');
        if (!eq) { continue; }
        *eq = '\0';
        if (opts_set(o, tok, eq + 1) < 0) { return -1; }
    }
    return 0;
}

/* did the environment take a position on graph capture? */
static int capture_forced(enum infer_ep ep) {
    const char* env = getenv("INFER_ORT_OPTS");
    return env && strstr(env, capture_key(ep)) != NULL;
}

/* ------------------------------------------------------------------ */
/* the session                                                          */
/* ------------------------------------------------------------------ */

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

/* Fresh session options with the EP appended, graph capture as asked.
 * Fresh every time: an EP cannot be removed from options once appended,
 * so the capture fallback below builds them again from nothing. Fills
 * e->so and, for a GPU EP, e->gpu_mem. */
static int session_options_create(struct infer_engine_ort* e, struct infer_ctx* c, int capture) {
    const OrtApi* api = e->api;
    ORT_CHECK(e, api->CreateSessionOptions(&e->so));
    ORT_CHECK(e, api->SetSessionGraphOptimizationLevel(e->so, ORT_ENABLE_ALL));
    ORT_CHECK(e, api->SetSessionLogSeverityLevel(e->so, getenv("INFER_ORT_VERBOSE") ? 0 : 2));
    /* Two sessions in one process (stage five runs a detector and a
     * landmark model) do NOT overlap in time, and still halve each
     * other's speed with the default settings: ORT's intra-op pool
     * spin-waits after its work is done, so the idle session's threads
     * burn the cores the busy one wants. Measured on the palm detector:
     * 11.8 ms alone, 23.4 ms with an idle landmark session beside it,
     * back to ~12 ms with spinning off. One session sees no difference,
     * which is why stage four never noticed.
     * INFER_ORT_SPIN=1 restores the default for comparison. */
    if (!getenv("INFER_ORT_SPIN")) {
        ORT_CHECK(e, api->AddSessionConfigEntry(e->so, "session.intra_op.allow_spinning", "0"));
    }

    if (e->ep == INFER_EP_WEBGPU) {
        INFER_CHECK(c->have & INFER_WANT_WGPU, "WebGPU EP needs the WebGPU context");
        char inst[32], dev[32], id[8];
        snprintf(inst, sizeof inst, "%" PRIuPTR, (uintptr_t)c->wgpu.instance);
        snprintf(dev, sizeof dev, "%" PRIuPTR, (uintptr_t)c->wgpu.device);
        snprintf(id, sizeof id, "%d", ORT_WEBGPU_DEVICE_ID);
        struct opts o = {{0}, {0}, 0};
        opts_set(&o, "deviceId", id);
        opts_set(&o, "webgpuInstance", inst);
        opts_set(&o, "webgpuDevice", dev);
        opts_set(&o, "preferredLayout", "NHWC");
        opts_set(&o, "dawnBackendType", "Vulkan");
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
        opts_set(&o, "dawnProcTable", procs);
#endif
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
        opts_set(&o, capture_key(e->ep), capture ? "1" : "0");
        if (opts_from_env(&o) < 0) { return -1; }
        ORT_CHECK(e, api->SessionOptionsAppendExecutionProvider(e->so, "WebGPU", o.keys, o.vals, o.n));
        ORT_CHECK(e,
                  api->CreateMemoryInfo("WebGPU_Buffer",
                                        OrtDeviceAllocator,
                                        ORT_WEBGPU_MEM_DEVICE_ID,
                                        OrtMemTypeDefault,
                                        &e->gpu_mem));
    } else if (e->ep == INFER_EP_CUDA) {
        INFER_CHECK(c->have & INFER_WANT_CUDA,
                    "CUDA EP needs the CUDA context (built without CUDA, or no device)");
        char dev[8], stream[32];
        snprintf(dev, sizeof dev, "%d", c->cuda.device);
        snprintf(stream, sizeof stream, "%" PRIuPTR, (uintptr_t)c->cuda.stream);
        struct opts o = {{0}, {0}, 0};
        opts_set(&o, "device_id", dev);
        /* our stream: every kernel the EP launches is ordered behind our
         * H2D copy and ahead of our D2H copy / stream wait, with no
         * cross-stream events to get wrong */
        opts_set(&o, "has_user_compute_stream", "1");
        opts_set(&o, "user_compute_stream", stream);
        /* EXHAUSTIVE (ORT's default) runs cudnnFind for every conv shape
         * the first time a session sees it -- seconds per session, and
         * hello_inference --all creates 36 CUDA sessions. HEURISTIC asks
         * cuDNN's model instead; INFER_ORT_OPTS=cudnn_conv_algo_search=
         * EXHAUSTIVE for the perf numbers. */
        opts_set(&o, "cudnn_conv_algo_search", "HEURISTIC");
        /* CUDA graphs: the kernel launches of one Run are recorded (on the
         * third run) and replayed thereafter -- the same host-time saving
         * the WebGPU EP's capture buys, under the same rule: the same
         * device addresses bound every run. ORT refuses it for a graph
         * with any node off the CUDA EP; create() then retries without. */
        opts_set(&o, capture_key(e->ep), capture ? "1" : "0");
        if (opts_from_env(&o) < 0) { return -1; }
        OrtCUDAProviderOptionsV2* co = NULL;
        ORT_CHECK(e, api->CreateCUDAProviderOptions(&co));
        OrtStatus* st = api->UpdateCUDAProviderOptions(co, o.keys, o.vals, o.n);
        if (!st) { st = api->SessionOptionsAppendExecutionProvider_CUDA_V2(e->so, co); }
        api->ReleaseCUDAProviderOptions(co);
        if (st) {
            fprintf(stderr, "ORT: CUDA EP: %s\n", api->GetErrorMessage(st));
            api->ReleaseStatus(st);
            return -1;
        }
        ORT_CHECK(e,
                  api->CreateMemoryInfo("Cuda",
                                        OrtDeviceAllocator,
                                        c->cuda.device,
                                        OrtMemTypeDefault,
                                        &e->gpu_mem));
    }
    return 0;
}

static int create(struct infer_engine_ort* e, struct infer_ctx* c, const char* model_path) {
    const OrtApi* api = e->api;
    ORT_CHECK(e, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hello_inference", &e->env));

    /* capture on for both GPU EPs unless INFER_ORT_OPTS says otherwise */
    if (session_options_create(e, c, 1) < 0) { return -1; }
    OrtStatus* st = api->CreateSession(e->env, model_path, e->so, &e->session);
    if (st && e->ep == INFER_EP_CUDA && !capture_forced(e->ep) &&
        strstr(api->GetErrorMessage(st), "have not been partitioned")) {
        /* a node the CUDA EP has no kernel for stays on the CPU EP, and
         * ORT will not capture a graph that crosses devices. Say so, and
         * run the session without capture rather than not at all. */
        fprintf(stderr, "ORT: CUDA graph capture refused (%s); running without\n",
                api->GetErrorMessage(st));
        api->ReleaseStatus(st);
        st = NULL;
        api->ReleaseSessionOptions(e->so);
        e->so = NULL;
        if (e->gpu_mem) {
            api->ReleaseMemoryInfo(e->gpu_mem);
            e->gpu_mem = NULL;
        }
        if (session_options_create(e, c, 0) < 0) { return -1; }
        st = api->CreateSession(e->env, model_path, e->so, &e->session);
    }
    if (st) {
        fprintf(stderr, "ORT: CreateSession: %s\n", api->GetErrorMessage(st));
        api->ReleaseStatus(st);
        return -1;
    }
    ORT_CHECK(e, api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &e->cpu_mem));
    ORT_CHECK(e, api->GetAllocatorWithDefaultOptions(&e->alloc));
    ORT_CHECK(e, api->CreateRunOptions(&e->run_opts));
    if (e->ep == INFER_EP_CUDA) {
        /* Run returns at submission: no cudaStreamSynchronize at the end
         * of the run (nor after a graph replay). The consumer's wait on
         * its tensor is the completion, as for WebGPU. Without this the
         * `run` column would silently include the GPU time. */
        ORT_CHECK(e, api->AddRunConfigEntry(e->run_opts,
                                           kOrtRunOptionsConfigDisableSynchronizeExecutionProviders,
                                           "1"));
    }

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
    void* data;
    const OrtMemoryInfo* mi;
    switch (t->domain) {
    case INFER_DOMAIN_WGPU:
        data = (void*)t->mem.wgpu.buffer;
        mi = e->gpu_mem;
        break;
    case INFER_DOMAIN_CUDA:
        data = t->mem.cuda.ptr;
        mi = e->gpu_mem;
        break;
    default:
        data = (void*)t->mem.cpu.ptr;
        mi = e->cpu_mem;
        break;
    }
    INFER_CHECK(data, "tensor has no memory");
    INFER_CHECK(mi, "no memory info for the %s domain on the %s EP", infer_domain_name(t->domain),
                infer_ep_name(e->ep));
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
