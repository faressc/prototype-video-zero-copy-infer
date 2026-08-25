/* infer_engine_ort.c -- ONNX Runtime behind the engine seam: one
 * session, one IO binding, inputs bound as whatever domain the EP
 * consumes (host pointer for the CPU EP, WGPUBuffer for the WebGPU EP),
 * outputs bound to the host.
 *
 * The WebGPU EP is given OUR Dawn device (deviceId >= 1 selects the
 * user-supplied instance/device; ORT prefixes the option keys itself).
 * A WebGPU tensor is an OrtValue whose data pointer IS the WGPUBuffer
 * handle, on memory info "WebGPU_Buffer" with device id == deviceId.
 */
#include "infer.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_c_api.h>

#include "infer_util.h"

enum { ORT_WEBGPU_DEVICE_ID = 1, MAX_IO = 8 };

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
    OrtValue* input_value;
    float* output;
    size_t output_elems;
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

static int query_input(struct infer_engine_ort* e) {
    const OrtApi* api = e->api;
    OrtTypeInfo* ti = NULL;
    ORT_CHECK(e, api->SessionGetInputTypeInfo(e->session, 0, &ti));
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
    INFER_CHECK(et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "input is not float32");
    e->input_desc.dtype = INFER_F32;
    e->input_desc.ndim = (int)nd;
    for (size_t i = 0; i < nd; i++) { e->input_desc.dims[i] = dims[i] > 0 ? dims[i] : 1; }
    infer_desc_set_image(&e->input_desc);
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
        /* experiments: INFER_ORT_OPTS="validationMode=disabled,enableGraphCapture=0" */
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
        /* Graph capture (record ORT's command buffers once, replay them)
         * is OFF by default. Measured on this machine (palm detector,
         * outputs bound to host memory): with capture on, Run() returns
         * ~8 ms before the GPU finishes and every iteration reads back
         * the PREVIOUS iteration's outputs -- hello_inference's stale
         * check flags all of them. The "4x faster" it seemed to give was
         * that missing wait. Opt in with INFER_ORT_OPTS=enableGraphCapture=1
         * to reproduce; using it correctly needs GPU-bound outputs and an
         * explicit completion wait, which is anira's output_ready token. */
        if (!capture_set) {
            keys[nkeys] = "enableGraphCapture";
            vals[nkeys] = "0";
            nkeys++;
        }
        ORT_CHECK(e, api->SessionOptionsAppendExecutionProvider(e->so, "WebGPU", keys, vals, nkeys));
        ORT_CHECK(e,
                  api->CreateMemoryInfo("WebGPU_Buffer",
                                        OrtDeviceAllocator,
                                        ORT_WEBGPU_DEVICE_ID,
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
    if (query_input(e) < 0) { return -1; }

    ORT_CHECK(e, api->CreateIoBinding(e->session, &e->binding));
    for (size_t i = 0; i < e->output_count; i++) {
        ORT_CHECK(e, api->BindOutputToDevice(e->binding, e->output_names[i], e->cpu_mem));
    }
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

int infer_engine_ort_bind_input(struct infer_engine_ort* e, const struct infer_tensor* t) {
    const OrtApi* api = e->api;
    enum infer_domain want = infer_engine_ort_input_domain(e->ep);
    INFER_CHECK(t->domain == want,
                "input tensor is in %s, the %s EP wants %s",
                infer_domain_name(t->domain),
                infer_ep_name(e->ep),
                infer_domain_name(want));
    void* data = t->domain == INFER_DOMAIN_WGPU ? (void*)t->mem.wgpu.buffer : (void*)t->mem.cpu.ptr;
    const OrtMemoryInfo* mi = t->domain == INFER_DOMAIN_WGPU ? e->gpu_mem : e->cpu_mem;
    if (e->input_value) {
        api->ReleaseValue(e->input_value);
        e->input_value = NULL;
    }
    ORT_CHECK(e,
              api->CreateTensorWithDataAsOrtValue(mi,
                                                  data,
                                                  infer_desc_bytes_packed(&t->desc),
                                                  t->desc.dims,
                                                  (size_t)t->desc.ndim,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  &e->input_value));
    ORT_CHECK(e, api->BindInput(e->binding, e->input_name, e->input_value));
    return 0;
}

int infer_engine_ort_run(struct infer_engine_ort* e) {
    const OrtApi* api = e->api;
    ORT_CHECK(e, api->RunWithBinding(e->session, e->run_opts, e->binding));

    OrtValue** outs = NULL;
    size_t n = 0;
    ORT_CHECK(e, api->GetBoundOutputValues(e->binding, e->alloc, &outs, &n));
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        OrtTensorTypeAndShapeInfo* ts = NULL;
        ORT_CHECK(e, api->GetTensorTypeAndShape(outs[i], &ts));
        size_t cnt = 0;
        ORT_CHECK(e, api->GetTensorShapeElementCount(ts, &cnt));
        api->ReleaseTensorTypeAndShapeInfo(ts);
        total += cnt;
    }
    if (total != e->output_elems) {
        free(e->output);
        e->output = malloc(total * sizeof(float));
        e->output_elems = total;
    }
    size_t off = 0;
    for (size_t i = 0; i < n; i++) {
        OrtTensorTypeAndShapeInfo* ts = NULL;
        ORT_CHECK(e, api->GetTensorTypeAndShape(outs[i], &ts));
        size_t cnt = 0;
        ORT_CHECK(e, api->GetTensorShapeElementCount(ts, &cnt));
        api->ReleaseTensorTypeAndShapeInfo(ts);
        void* p = NULL;
        ORT_CHECK(e, api->GetTensorMutableData(outs[i], &p));
        memcpy(e->output + off, p, cnt * sizeof(float));
        off += cnt;
        api->ReleaseValue(outs[i]);
    }
    ORT_CHECK(e, api->AllocatorFree(e->alloc, outs));
    return 0;
}

const float* infer_engine_ort_output(const struct infer_engine_ort* e, size_t* count) {
    *count = e->output_elems;
    return e->output;
}

void infer_engine_ort_destroy(struct infer_engine_ort* e) {
    if (!e) { return; }
    const OrtApi* api = e->api;
    if (e->input_value) { api->ReleaseValue(e->input_value); }
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
    free(e->output);
    free(e);
}
