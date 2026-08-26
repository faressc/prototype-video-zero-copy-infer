/* hello_inference.c -- the hello world of inference: a tensor produced
 * on {cpu, gl, vk, wgpu, dmabuf}, moved along a registry edge into the
 * domain the {cpu, webgpu} execution provider of ONNX Runtime reads,
 * run, and its outputs moved along a second edge into the domain a
 * consumer on {cpu, gl, vk, wgpu, dmabuf} wants them in. Every cell
 * reports which two edges it took and what each costs, whether the
 * bytes that reached the engine are bit-identical to the CPU generator,
 * how far the outputs the consumer sees are from the CPU reference, and
 * how long each step took -- the producer, the input edge, the engine
 * call, the output edge, and the wait for the consumer's ready token,
 * which is the part of an inference the engine call returning does not
 * cover.
 *
 *   hello_inference --src vk --ep webgpu --dst vk --model models/palm_detection_lite.onnx
 *   hello_inference --all --strict --model models/synthetic_conv.onnx
 *
 * --strict makes the expected cost class of both edges an assertion:
 * the harness exists to notice when a "zero-copy" path silently became
 * a copy, so an unavailable DEVICE_COPY edge is a failure, not a
 * fallback.
 *
 * One tensor per side, bound to the engine every run: this loop is
 * synchronous (it waits for the consumer's token before producing the
 * next input), so rotating buffers would hide nothing -- measured
 * within noise on every fence-gated producer -- and the WebGPU EP's
 * captured graph needs the same buffers bound every run.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer.h"
#include "infer_util.h"

struct options {
    enum infer_domain src, dst;
    enum infer_ep ep;
    const char* model;
    int iters, warmup, all, strict, verbose;
    uint32_t seed;
};

struct cell_result {
    enum infer_domain src, dst;
    enum infer_ep ep;
    const char *in_edge, *out_edge;
    enum infer_cost in_cost, out_cost;
    int input_exact;
    double max_abs_err;
    double gen_us, in_us, run_us, out_us, wait_us;
    double total_us; /* median of per-iteration gen+in+run+out+wait: the frame latency */
    const char* status; /* "ok", or why not */
};

static int is_dmabuf_domain(enum infer_domain d) {
    return d == INFER_DOMAIN_GL || d == INFER_DOMAIN_VK || d == INFER_DOMAIN_DMABUF;
}

/* What an edge between two domains must cost, from first principles --
 * independent of the registry, so --strict can catch the registry (or
 * a driver) quietly downgrading a path. Symmetric: the price of moving
 * a into b is the price of moving b into a. */
static enum infer_cost expected_cost(enum infer_domain a, enum infer_domain b) {
    if (a == b) { return INFER_COST_ZERO_COPY; }
    if ((a == INFER_DOMAIN_WGPU && is_dmabuf_domain(b)) || (b == INFER_DOMAIN_WGPU && is_dmabuf_domain(a))) {
        return INFER_COST_DEVICE_COPY;
    }
    if ((a == INFER_DOMAIN_DMABUF && b == INFER_DOMAIN_CPU) ||
        (a == INFER_DOMAIN_CPU && b == INFER_DOMAIN_DMABUF)) {
        return INFER_COST_ZERO_COPY;
    }
    return INFER_COST_HOST_COPY;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double median(double* v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void tensor_init(struct infer_tensor* t) {
    memset(t, 0, sizeof *t);
    t->ready.sync_fd = -1; /* fd 0 is a valid descriptor: "none" must be explicit */
    t->released.sync_fd = -1;
}

enum { MAX_OUT = INFER_ENGINE_MAX_OUTPUTS };

/* one (source, EP, destination) cell; reference == NULL computes the reference */
static int run_cell(struct infer_ctx* ctx,
                    const struct infer_registry* reg,
                    const struct options* o,
                    enum infer_domain src,
                    enum infer_ep ep,
                    enum infer_domain dst,
                    const float* reference,
                    size_t reference_count,
                    float** out_copy,
                    size_t* out_count,
                    struct cell_result* res) {
    memset(res, 0, sizeof *res);
    res->src = src;
    res->ep = ep;
    res->dst = dst;
    res->in_edge = res->out_edge = "-";
    res->in_cost = res->out_cost = INFER_COST_UNAVAILABLE;
    res->status = "failed";
    res->max_abs_err = NAN;

    /* the engine says where it reads and where it writes; the registry
     * says how to get there from src and from there to dst */
    enum infer_domain in_dom = infer_engine_ort_input_domain(ep);
    enum infer_domain out_dom = infer_engine_ort_output_domain(ep);
    const struct infer_edge* in_e = infer_registry_find(reg, src, in_dom);
    if (!in_e || !in_e->available) {
        res->status = "no in-edge";
        return -1;
    }
    res->in_edge = in_e->name;
    res->in_cost = in_e->cost;
    const struct infer_edge* out_e = infer_registry_find(reg, out_dom, dst);
    if (!out_e || !out_e->available) {
        res->status = "no out-edge";
        return -1;
    }
    res->out_edge = out_e->name;
    res->out_cost = out_e->cost;

    struct infer_engine_ort* eng = infer_engine_ort_create(ctx, ep, o->model);
    if (!eng) {
        res->status = "engine";
        return -1;
    }
    struct infer_desc idesc;
    infer_engine_ort_input_desc(eng, &idesc);
    size_t n_in = infer_desc_elements(&idesc);
    size_t nout = infer_engine_ort_output_count(eng);
    struct infer_desc odesc[MAX_OUT];
    size_t n_out = 0;
    for (size_t i = 0; i < nout; i++) {
        infer_engine_ort_output_desc(eng, i, &odesc[i]);
        n_out += infer_desc_elements(&odesc[i]);
    }

    struct infer_tensor t_src, t_in, t_out[MAX_OUT], t_dst[MAX_OUT];
    tensor_init(&t_src);
    tensor_init(&t_in);
    for (int i = 0; i < MAX_OUT; i++) {
        tensor_init(&t_out[i]);
        tensor_init(&t_dst[i]);
    }
    /* what the edges keep between iterations (the dma-buf imports),
     * owned here -- the plan's, in anira -- never by the tensors */
    struct infer_edge_cache cache;
    infer_edge_cache_init(&cache);

    float* check = malloc(n_in * sizeof(float));
    float* out_buf = malloc(n_out * sizeof(float));
    float* prev_out = malloc(n_out * sizeof(float));
    int have_prev = 0;
    int total = o->warmup + o->iters;
    double* gen_t = calloc((size_t)total, sizeof(double));
    double* in_t = calloc((size_t)total, sizeof(double));
    double* run_t = calloc((size_t)total, sizeof(double));
    double* out_t = calloc((size_t)total, sizeof(double));
    double* wait_t = calloc((size_t)total, sizeof(double));
    double* total_t = calloc((size_t)total, sizeof(double));
    int stale = 0; /* outputs identical to the previous iteration's although the input changed */
    int rc = -1;

    /* The consumer's side of every output: allocated in the consumer's
     * domain, by the consumer (the doc's bind_output -- a user-owned
     * destination). For a hand-over edge the engine writes into it
     * directly; for a copy edge the engine writes its own tensor in its
     * native domain and the edge moves the result. */
    for (size_t i = 0; i < nout; i++) {
        if (infer_tensor_alloc(ctx, dst, &odesc[i], &t_dst[i]) < 0) {
            res->status = "consumer alloc";
            goto out;
        }
        if (!out_e->handover && infer_tensor_alloc(ctx, out_dom, &odesc[i], &t_out[i]) < 0) {
            res->status = "engine alloc";
            goto out;
        }
    }

    for (int it = 0; it < total; it++) {
        /* a different input every iteration: an engine that returned a
         * previous iteration's outputs (asynchronous replay, stale
         * readback) would otherwise be indistinguishable from a correct one */
        uint32_t seed_it = o->seed + (uint32_t)it;
        uint64_t t0 = infer_now_ns();
        if (infer_tensor_gen(ctx, src, &idesc, seed_it, &t_src) < 0) {
            res->status = "producer";
            goto out;
        }
        uint64_t t1 = infer_now_ns();
        /* into the engine: the engine's input becomes a view of the
         * producer's tensor (hand-over) or a copy of it */
        if (infer_edge_apply(in_e, ctx, &cache, &t_src, &t_in) < 0) {
            res->status = "in-edge";
            goto out;
        }
        uint64_t t2 = infer_now_ns();
        /* a hand-over output: the engine's output becomes a view of the
         * consumer's tensor, before the run. Every iteration, because for
         * a dma-buf the view also opens the CPU write window. */
        if (out_e->handover) {
            for (size_t i = 0; i < nout; i++) {
                if (infer_edge_apply(out_e, ctx, &cache, &t_dst[i], &t_out[i]) < 0) {
                    res->status = "out-edge";
                    goto out;
                }
            }
        }
        uint64_t t2b = infer_now_ns();
        int bound = infer_engine_ort_bind_input(eng, &t_in) == 0;
        for (size_t i = 0; bound && i < nout; i++) {
            bound = infer_engine_ort_bind_output(eng, i, &t_out[i]) == 0;
        }
        if (!bound || infer_engine_ort_run(eng) < 0) {
            res->status = "run";
            goto out;
        }
        uint64_t t3 = infer_now_ns();
        /* a copy output: move what the engine wrote into the consumer's
         * tensor, after the run (submits, for a GPU consumer) */
        if (!out_e->handover) {
            for (size_t i = 0; i < nout; i++) {
                if (infer_edge_apply(out_e, ctx, &cache, &t_out[i], &t_dst[i]) < 0) {
                    res->status = "out-edge";
                    goto out;
                }
            }
        }
        uint64_t t4 = infer_now_ns();
        /* the consumer's token: the outputs are usable in dst once this
         * returns. Everything above only submitted. */
        for (size_t i = 0; i < nout; i++) {
            if (infer_tensor_wait_ready(ctx, &t_dst[i]) < 0) {
                res->status = "ready";
                goto out;
            }
        }
        uint64_t t5 = infer_now_ns();
        gen_t[it] = (double)(t1 - t0) / 1e3;
        in_t[it] = (double)(t2 - t1) / 1e3;
        run_t[it] = (double)(t3 - t2b) / 1e3;
        out_t[it] = (double)((t2b - t2) + (t4 - t3)) / 1e3;
        wait_t[it] = (double)(t5 - t4) / 1e3;
        total_t[it] = (double)(t5 - t0) / 1e3;

        /* diagnostic: the consumer's token signaled, so the GPU has
         * finished everything submitted before the inference --
         * including the input edge's release signal. If that fence is
         * still unsignaled now, its signal is being deferred. */
        if (o->verbose && it == o->warmup && t_src.released.sync_fd >= 0) {
            uint64_t w0 = infer_now_ns();
            int already = infer_wait_sync_fd(t_src.released.sync_fd, 0) == 0;
            int ok = already || infer_wait_sync_fd(t_src.released.sync_fd, 2000) == 0;
            fprintf(stderr,
                    "  release fence after ready: %s (%.1f us more)\n",
                    already ? "already signaled" : ok ? "signaled later" : "TIMEOUT",
                    ok ? (double)(infer_now_ns() - w0) / 1e3 : -1.0);
        }

        /* what the consumer sees this iteration (untimed, after the
         * token): all outputs, concatenated */
        {
            size_t off = 0;
            for (size_t i = 0; i < nout; i++) {
                if (infer_tensor_readback(ctx, &t_dst[i], out_buf + off) < 0) {
                    res->status = "readback";
                    goto out;
                }
                off += infer_desc_elements(&odesc[i]);
            }
            if (have_prev && memcmp(prev_out, out_buf, n_out * sizeof(float)) == 0) { stale++; }
            memcpy(prev_out, out_buf, n_out * sizeof(float));
            have_prev = 1;
        }

        if (it == 0) {
            /* what reached the engine, compared bit for bit with the generator */
            if (infer_tensor_readback(ctx, &t_in, check) < 0) {
                res->status = "readback";
                goto out;
            }
            res->input_exact = 1;
            for (size_t i = 0; i < n_in; i++) {
                float want = infer_gen_value((uint32_t)i, o->seed);
                if (memcmp(&check[i], &want, sizeof(float)) != 0) {
                    res->input_exact = 0;
                    if (o->verbose) {
                        fprintf(stderr, "  input[%zu] = %g, generator says %g\n", i, check[i], want);
                    }
                    break;
                }
            }
            if (reference) {
                double err = 0;
                if (n_out != reference_count) {
                    err = INFINITY;
                } else {
                    for (size_t i = 0; i < n_out; i++) {
                        double d = fabs((double)out_buf[i] - (double)reference[i]);
                        if (d > err) { err = d; }
                    }
                }
                res->max_abs_err = err;
            } else {
                res->max_abs_err = 0;
                if (out_copy) {
                    *out_copy = malloc(n_out * sizeof(float));
                    memcpy(*out_copy, out_buf, n_out * sizeof(float));
                }
                if (out_count) { *out_count = n_out; }
            }
        }
    }
    if (stale) {
        fprintf(stderr, "  STALE: %d iteration(s) returned the previous iteration's outputs\n", stale);
    }
    res->gen_us = median(gen_t + o->warmup, o->iters);
    res->in_us = median(in_t + o->warmup, o->iters);
    res->run_us = median(run_t + o->warmup, o->iters);
    res->out_us = median(out_t + o->warmup, o->iters);
    res->wait_us = median(wait_t + o->warmup, o->iters);
    res->total_us = median(total_t + o->warmup, o->iters);
    res->status = stale ? "STALE" : "ok";
    rc = stale ? -1 : 0;
out:
    infer_engine_ort_destroy(eng); /* its OrtValues wrap our tensors */
    infer_edge_cache_fini(&cache); /* the imports hold the tensors' fds */
    for (int i = 0; i < MAX_OUT; i++) {
        infer_tensor_release(ctx, &t_out[i]);
        infer_tensor_release(ctx, &t_dst[i]);
    }
    infer_tensor_release(ctx, &t_in);
    infer_tensor_release(ctx, &t_src);
    free(check);
    free(out_buf);
    free(prev_out);
    free(gen_t);
    free(in_t);
    free(run_t);
    free(out_t);
    free(wait_t);
    free(total_t);
    return rc;
}

/* one letter per cost class, so a row fits a terminal */
static char cost_code(enum infer_cost c) {
    switch (c) {
    case INFER_COST_ZERO_COPY: return 'Z';
    case INFER_COST_DEVICE_COPY: return 'D';
    case INFER_COST_HOST_COPY: return 'H';
    default: return '-';
    }
}

static void print_header(void) {
    printf("%-6s %-6s %-6s %-13s %s %-13s %s %-4s %-7s %5s %5s %5s %5s %5s %6s  %s\n",
           "src", "ep", "dst", "in_edge", "c", "out_edge", "c", "inp", "err",
           "gen", "in", "run", "out", "wait", "total", "status");
}

static void print_result(const struct cell_result* r) {
    char err[24];
    if (isnan(r->max_abs_err)) {
        snprintf(err, sizeof err, "-");
    } else {
        snprintf(err, sizeof err, "%.2g", r->max_abs_err);
    }
    printf("%-6s %-6s %-6s %-13s %c %-13s %c %-4s %-7s %5.0f %5.0f %5.0f %5.0f %5.0f %6.0f  %s\n",
           infer_domain_name(r->src),
           infer_ep_name(r->ep),
           infer_domain_name(r->dst),
           r->in_edge,
           cost_code(r->in_cost),
           r->out_edge,
           cost_code(r->out_cost),
           r->status[0] == 'o' ? (r->input_exact ? "ok" : "DIFF") : "-",
           err,
           r->gen_us,
           r->in_us,
           r->run_us,
           r->out_us,
           r->wait_us,
           r->total_us,
           r->status);
}

static void usage(void) {
    fprintf(stderr,
            "usage: hello_inference [--src D] [--ep cpu|webgpu] [--dst D] [--model path]\n"
            "                       [--iters N] [--warmup N] [--seed S] [--all] [--strict] [--verbose]\n"
            "       D = cpu|gl|vk|wgpu|dmabuf: --src is where the input is produced, --dst is\n"
            "       where a consumer wants the outputs; --all sweeps every (src, ep, dst).\n");
}

int main(int argc, char** argv) {
    struct options o = {
        .src = INFER_DOMAIN_CPU,
        .dst = INFER_DOMAIN_CPU,
        .ep = INFER_EP_CPU,
        .model = "models/synthetic_conv.onnx",
        .iters = 20,
        .warmup = 3,
        .seed = 1,
    };
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--all")) {
            o.all = 1;
        } else if (!strcmp(a, "--strict")) {
            o.strict = 1;
        } else if (!strcmp(a, "--verbose")) {
            o.verbose = 1;
        } else if (!strcmp(a, "--src") && v) {
            if (infer_domain_parse(v, &o.src) < 0) { usage(); return 2; }
            i++;
        } else if (!strcmp(a, "--dst") && v) {
            if (infer_domain_parse(v, &o.dst) < 0) { usage(); return 2; }
            i++;
        } else if (!strcmp(a, "--ep") && v) {
            if (infer_ep_parse(v, &o.ep) < 0) { usage(); return 2; }
            i++;
        } else if (!strcmp(a, "--model") && v) {
            o.model = v;
            i++;
        } else if (!strcmp(a, "--iters") && v) {
            o.iters = atoi(v);
            i++;
        } else if (!strcmp(a, "--warmup") && v) {
            o.warmup = atoi(v);
            i++;
        } else if (!strcmp(a, "--seed") && v) {
            o.seed = (uint32_t)strtoul(v, NULL, 0);
            i++;
        } else {
            usage();
            return 2;
        }
    }
    if (o.iters < 1) { o.iters = 1; }
    infer_verbose = o.verbose;

    unsigned want = 0;
    if (o.all) {
        want = INFER_WANT_GL | INFER_WANT_VK | INFER_WANT_WGPU | INFER_WANT_DMABUF;
    } else {
        if (o.src != INFER_DOMAIN_CPU) { want |= 1u << o.src; }
        if (o.dst != INFER_DOMAIN_CPU) { want |= 1u << o.dst; }
        if (o.ep == INFER_EP_WEBGPU) { want |= INFER_WANT_WGPU; }
    }
    struct infer_ctx ctx;
    infer_ctx_init_all(&ctx, want); /* best effort; cells report what is missing */
    struct infer_registry reg;
    infer_registry_init(&reg);
    infer_registry_probe(&reg, &ctx);

    if (o.verbose) {
        fprintf(stderr, "edges:\n");
        for (int i = 0; i < reg.count; i++) {
            const struct infer_edge* e = &reg.edges[i];
            fprintf(stderr,
                    "  %-6s -> %-6s %-13s %-11s %-9s %s\n",
                    infer_domain_name(e->src),
                    infer_domain_name(e->dst),
                    e->name,
                    infer_cost_name(e->cost),
                    e->handover ? "hand-over" : "copy",
                    e->available ? "available" : "unavailable");
        }
    }

    /* the reference: cpu -> cpu EP -> cpu */
    float* reference = NULL;
    size_t ref_count = 0;
    struct cell_result ref_res;
    if (run_cell(&ctx, &reg, &o, INFER_DOMAIN_CPU, INFER_EP_CPU, INFER_DOMAIN_CPU, NULL, 0, &reference,
                 &ref_count, &ref_res) < 0) {
        fprintf(stderr, "reference cell failed: %s\n", ref_res.status);
        infer_ctx_fini_all(&ctx);
        return 1;
    }

    int failures = 0;
    print_header();
    if (!o.all) {
        struct cell_result r;
        if (o.src == INFER_DOMAIN_CPU && o.ep == INFER_EP_CPU && o.dst == INFER_DOMAIN_CPU) {
            r = ref_res;
        } else {
            run_cell(&ctx, &reg, &o, o.src, o.ep, o.dst, reference, ref_count, NULL, NULL, &r);
        }
        print_result(&r);
        if (strcmp(r.status, "ok") != 0 || !r.input_exact) { failures++; }
        if (o.strict && (r.in_cost != expected_cost(o.src, infer_engine_ort_input_domain(o.ep)) ||
                         r.out_cost != expected_cost(infer_engine_ort_output_domain(o.ep), o.dst))) {
            failures++;
        }
    } else {
        double tol = strstr(o.model, "synthetic") ? 1e-4 : 1e-3;
        for (int ep = 0; ep < INFER_EP_COUNT; ep++) {
            enum infer_domain in_dom = infer_engine_ort_input_domain((enum infer_ep)ep);
            enum infer_domain out_dom = infer_engine_ort_output_domain((enum infer_ep)ep);
            for (int dst = 0; dst < INFER_DOMAIN_COUNT; dst++) {
                for (int src = 0; src < INFER_DOMAIN_COUNT; src++) {
                    struct cell_result r;
                    if (src == INFER_DOMAIN_CPU && ep == INFER_EP_CPU && dst == INFER_DOMAIN_CPU) {
                        r = ref_res;
                    } else {
                        run_cell(&ctx, &reg, &o, (enum infer_domain)src, (enum infer_ep)ep,
                                 (enum infer_domain)dst, reference, ref_count, NULL, NULL, &r);
                    }
                    print_result(&r);
                    int bad = strcmp(r.status, "ok") != 0 || !r.input_exact ||
                              !(r.max_abs_err <= tol);
                    if (o.strict && (r.in_cost != expected_cost((enum infer_domain)src, in_dom) ||
                                     r.out_cost != expected_cost(out_dom, (enum infer_domain)dst))) {
                        bad = 1;
                    }
                    if (bad) { failures++; }
                }
            }
        }
        printf("\nc: Z = ZERO_COPY (hand-over), D = DEVICE_COPY (one GPU pass), H = HOST_COPY\n"
               "(through host memory), - = unavailable. inp: the input the engine read, bit for\n"
               "bit against the generator. err: max |out - reference|. Times are medians in us.\n"
               "in_edge moves the input from src into the EP's input domain; out_edge moves the\n"
               "outputs from the EP's output domain into dst. GPU producers and edges only\n"
               "submit: run is the engine call (WebGPU EP: submission), wait the wait for the\n"
               "consumer's ready token -- the time the engine call returning does not cover.\n"
               "total is the median of per-iteration gen+in+run+out+wait: wall-clock latency\n"
               "from starting to produce the input to the outputs being ready in dst.\n");
    }
    free(reference);
    infer_ctx_fini_all(&ctx);
    if (failures) {
        fprintf(stderr, "%d cell(s) failed\n", failures);
        return o.strict ? 1 : 0;
    }
    return 0;
}
