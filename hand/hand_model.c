/* hand_model.c -- see hand_model.h. The whole file is stage four's API
 * used as intended: the engine says which domain it reads and writes,
 * the registry supplies the conversion to the consumer, and the tensors
 * on both sides belong to us.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hand_model.h"
#include "infer_util.h"

struct hand_model {
    struct infer_ctx* ctx;
    enum infer_ep ep;
    uint32_t side;
    enum hand_border border;
    struct infer_engine_ort* eng;
    struct hand_frame_pass* pass; /* WebGPU EP only */

    struct infer_desc in_desc;
    struct infer_tensor t_in; /* fixed for the session's life: graph capture */

    size_t nout;
    struct infer_desc odesc[HAND_MODEL_MAX_OUTPUTS];
    struct infer_tensor t_out[HAND_MODEL_MAX_OUTPUTS];  /* engine domain */
    struct infer_tensor t_host[HAND_MODEL_MAX_OUTPUTS]; /* consumer domain: CPU */
    const struct infer_edge* out_edge;
    struct infer_edge_cache cache;
};

static void tensor_init(struct infer_tensor* t) {
    memset(t, 0, sizeof *t);
    t->ready.sync_fd = -1; /* fd 0 is a valid descriptor: "none" must be explicit */
    t->released.sync_fd = -1;
}

struct hand_model* hand_model_create(struct infer_ctx* ctx,
                                     const struct infer_registry* reg,
                                     enum infer_ep ep,
                                     const char* model_path,
                                     uint32_t side,
                                     enum hand_border border) {
    struct hand_model* m = calloc(1, sizeof *m);
    m->ctx = ctx;
    m->ep = ep;
    m->side = side;
    m->border = border;
    tensor_init(&m->t_in);
    for (int i = 0; i < HAND_MODEL_MAX_OUTPUTS; i++) {
        tensor_init(&m->t_out[i]);
        tensor_init(&m->t_host[i]);
    }
    infer_edge_cache_init(&m->cache);

    m->eng = infer_engine_ort_create(ctx, ep, model_path);
    if (!m->eng) {
        fprintf(stderr, "hand: cannot load %s\n", model_path);
        hand_model_destroy(m);
        return NULL;
    }
    struct infer_desc reported;
    infer_engine_ort_input_desc(m->eng, &reported);
    if (reported.ndim != 4 || reported.dims[1] != (int64_t)side ||
        reported.dims[2] != (int64_t)side || reported.dims[3] != 3) {
        fprintf(stderr,
                "hand: %s wants [1,%lld,%lld,%lld], not [1,%u,%u,3]\n",
                model_path,
                (long long)reported.dims[1],
                (long long)reported.dims[2],
                (long long)reported.dims[3],
                side,
                side);
        hand_model_destroy(m);
        return NULL;
    }
    /* our own descriptor rather than the model's: it carries the byte
     * image the dma-buf edges need, and it must agree on the shape */
    hand_frame_tensor_desc(side, side, &m->in_desc);

    const enum infer_domain in_dom = infer_engine_ort_input_domain(ep);
    const enum infer_domain out_dom = infer_engine_ort_output_domain(ep);
    if (infer_tensor_alloc(ctx, in_dom, &m->in_desc, &m->t_in) < 0) {
        fprintf(stderr, "hand: input tensor alloc failed\n");
        hand_model_destroy(m);
        return NULL;
    }

    m->out_edge = infer_registry_find(reg, out_dom, INFER_DOMAIN_CPU);
    if (!m->out_edge || !m->out_edge->available) {
        fprintf(stderr, "hand: no %s -> cpu edge\n", infer_domain_name(out_dom));
        hand_model_destroy(m);
        return NULL;
    }
    m->nout = infer_engine_ort_output_count(m->eng);
    if (m->nout > HAND_MODEL_MAX_OUTPUTS) { m->nout = HAND_MODEL_MAX_OUTPUTS; }
    for (size_t i = 0; i < m->nout; i++) {
        infer_engine_ort_output_desc(m->eng, i, &m->odesc[i]);
        if (infer_tensor_alloc(ctx, INFER_DOMAIN_CPU, &m->odesc[i], &m->t_host[i]) < 0) {
            fprintf(stderr, "hand: host output alloc failed\n");
            hand_model_destroy(m);
            return NULL;
        }
        /* a copy edge needs its own tensor in the engine's domain; a
         * hand-over edge makes the engine's output a view of the host
         * tensor instead, and that view is made per run below */
        if (!m->out_edge->handover &&
            infer_tensor_alloc(ctx, out_dom, &m->odesc[i], &m->t_out[i]) < 0) {
            fprintf(stderr, "hand: engine output alloc failed\n");
            hand_model_destroy(m);
            return NULL;
        }
    }

    if (ep == INFER_EP_WEBGPU) {
        m->pass = hand_frame_pass_create(&ctx->wgpu);
        if (!m->pass) {
            hand_model_destroy(m);
            return NULL;
        }
    }
    return m;
}

void hand_model_destroy(struct hand_model* m) {
    if (!m) { return; }
    /* order matters, as in hello_inference: the engine's OrtValues wrap
     * our tensors, and the edge cache's imports hold their fds */
    if (m->eng) { infer_engine_ort_destroy(m->eng); }
    infer_edge_cache_fini(&m->cache);
    hand_frame_pass_destroy(m->pass);
    for (int i = 0; i < HAND_MODEL_MAX_OUTPUTS; i++) {
        infer_tensor_release(m->ctx, &m->t_out[i]);
        infer_tensor_release(m->ctx, &m->t_host[i]);
    }
    infer_tensor_release(m->ctx, &m->t_in);
    free(m);
}

int hand_model_feed_cpu(struct hand_model* m, const struct infer_frame* f, struct hand_affine a) {
    INFER_CHECK(m->ep == INFER_EP_CPU, "hand: feed_cpu on a WebGPU model");
    return hand_frame_to_tensor_cpu(f, a, HAND_NORM_RANGE, m->border, &m->t_in);
}

int hand_model_feed_wgpu(struct hand_model* m,
                         struct infer_wgpu_frame_import* im,
                         const struct infer_frame* f,
                         struct hand_affine a) {
    INFER_CHECK(m->ep == INFER_EP_WEBGPU, "hand: feed_wgpu on a CPU model");
    return hand_frame_to_tensor_wgpu(&m->ctx->wgpu, m->pass, im, f, a, HAND_NORM_RANGE,
                                    m->border, &m->t_in);
}

int hand_model_run(struct hand_model* m) {
    /* A hand-over output edge runs BEFORE the engine and with the call
     * inverted -- the engine's output becomes a view of the CONSUMER's
     * tensor, so the kernels write host memory in place. */
    if (m->out_edge->handover) {
        for (size_t i = 0; i < m->nout; i++) {
            INFER_CHECK(infer_edge_apply(m->out_edge, m->ctx, &m->cache, &m->t_host[i],
                                         &m->t_out[i]) == 0,
                        "hand: out-edge hand-over failed");
        }
    }
    INFER_CHECK(infer_engine_ort_bind_input(m->eng, &m->t_in) == 0, "hand: bind_input failed");
    for (size_t i = 0; i < m->nout; i++) {
        INFER_CHECK(infer_engine_ort_bind_output(m->eng, i, &m->t_out[i]) == 0,
                    "hand: bind_output %zu failed", i);
    }
    return infer_engine_ort_run(m->eng);
}

int hand_model_sync(struct hand_model* m) {
    if (!m->out_edge->handover) {
        for (size_t i = 0; i < m->nout; i++) {
            INFER_CHECK(infer_edge_apply(m->out_edge, m->ctx, &m->cache, &m->t_out[i],
                                         &m->t_host[i]) == 0,
                        "hand: out-edge copy failed");
        }
    }
    for (size_t i = 0; i < m->nout; i++) {
        INFER_CHECK(infer_tensor_wait_ready(m->ctx, &m->t_host[i]) == 0, "hand: ready wait failed");
    }
    return 0;
}

const float* hand_model_output(const struct hand_model* m, int i) {
    return i >= 0 && (size_t)i < m->nout ? m->t_host[i].mem.cpu.ptr : NULL;
}

size_t hand_model_output_count(const struct hand_model* m) {
    return m->nout;
}

size_t hand_model_output_elements(const struct hand_model* m, int i) {
    return i >= 0 && (size_t)i < m->nout ? infer_desc_elements(&m->odesc[i]) : 0;
}

enum infer_ep hand_model_ep(const struct hand_model* m) {
    return m->ep;
}

const char* hand_model_out_edge(const struct hand_model* m) {
    return m->out_edge ? m->out_edge->name : "-";
}
