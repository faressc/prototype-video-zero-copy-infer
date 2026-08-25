/* hello_inference.c -- the hello world of inference: a tensor produced
 * on {cpu, gl, vk, wgpu}, moved along a registry edge into the domain
 * the {cpu, webgpu} execution provider of ONNX Runtime wants, run, and
 * checked. Every cell reports which edge it took, what that edge costs,
 * whether the bytes that reached the engine are bit-identical to the
 * CPU generator, how far the output is from the CPU reference, and how
 * long the producer, the edge and the run took.
 *
 *   hello_inference --src vk --ep webgpu --model models/palm_detection_lite.onnx
 *   hello_inference --all --strict --model models/synthetic_conv.onnx
 *
 * --strict makes the expected cost class of every cell an assertion:
 * the harness exists to notice when a "zero-copy" path silently became
 * a copy, so an unavailable DEVICE_COPY edge is a failure, not a
 * fallback.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer.h"
#include "infer_util.h"

struct options {
    enum infer_domain src;
    enum infer_ep ep;
    const char* model;
    int iters, warmup, all, strict, verbose;
    int buffers; /* producer-side multi-buffering depth: 2 (default) or 1 (--single) */
    uint32_t seed;
};

struct cell_result {
    enum infer_domain src;
    enum infer_ep ep;
    const char* edge;
    enum infer_cost cost;
    int input_exact;
    double max_abs_err;
    double gen_us, edge_us, run_us;
    double total_us; /* median of per-iteration gen+edge+run: the frame latency */
    const char* status; /* "ok", or why not */
};

static enum infer_cost expected_cost(enum infer_domain src, enum infer_ep ep) {
    enum infer_domain dst = infer_engine_ort_input_domain(ep);
    if (src == dst) { return INFER_COST_ZERO_COPY; }
    if (dst == INFER_DOMAIN_WGPU &&
        (src == INFER_DOMAIN_GL || src == INFER_DOMAIN_VK || src == INFER_DOMAIN_DMABUF)) {
        return INFER_COST_DEVICE_COPY;
    }
    if (dst == INFER_DOMAIN_CPU && src == INFER_DOMAIN_DMABUF) { return INFER_COST_ZERO_COPY; }
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

/* one (source, EP) cell; reference == NULL computes the reference */
static int run_cell(struct infer_ctx* ctx,
                    const struct infer_registry* reg,
                    const struct options* o,
                    enum infer_domain src,
                    enum infer_ep ep,
                    const float* reference,
                    size_t reference_count,
                    float** out_copy,
                    struct cell_result* res) {
    memset(res, 0, sizeof *res);
    res->src = src;
    res->ep = ep;
    res->edge = "-";
    res->cost = INFER_COST_UNAVAILABLE;
    res->status = "failed";
    res->max_abs_err = NAN;

    enum infer_domain dst = infer_engine_ort_input_domain(ep);
    const struct infer_edge* edge = infer_registry_find(reg, src, dst);
    if (!edge || !edge->available) {
        res->status = "no edge";
        return -1;
    }
    res->edge = edge->name;

    struct infer_engine_ort* eng = infer_engine_ort_create(ctx, ep, o->model);
    if (!eng) {
        res->status = "engine";
        return -1;
    }
    struct infer_desc desc;
    infer_engine_ort_input_desc(eng, &desc);
    size_t n = infer_desc_elements(&desc);

    /* Producer-side multi-buffering (the doc's "rotate descriptors, reuse
     * a slot when its input_released token signals"): NBUF source tensors
     * in rotation, each with its own converted tensor so the edge's cached
     * dma-buf import stays valid per slot. With one slot the producer
     * must wait for the consumer's release fence before every write and
     * the pipeline serialises; with two it writes slot B while the GPU
     * still reads slot A. */
    enum { MAX_BUF = 2 };
    int nbuf = o->buffers < 1 ? 1 : o->buffers > MAX_BUF ? MAX_BUF : o->buffers;
    struct infer_tensor t_src[MAX_BUF], t_in[MAX_BUF];
    for (int b = 0; b < MAX_BUF; b++) {
        memset(&t_src[b], 0, sizeof t_src[b]);
        memset(&t_in[b], 0, sizeof t_in[b]);
        t_src[b].ready.sync_fd = t_src[b].released.sync_fd = -1;
        t_in[b].ready.sync_fd = t_in[b].released.sync_fd = -1;
    }
    float* check = malloc(n * sizeof(float));
    int total = o->warmup + o->iters;
    double* gen_t = calloc((size_t)total, sizeof(double));
    double* edge_t = calloc((size_t)total, sizeof(double));
    double* run_t = calloc((size_t)total, sizeof(double));
    double* total_t = calloc((size_t)total, sizeof(double));
    double release_lat_us = -1; /* verbose: when did the consumer's release fence signal? */
    double* drain_t = calloc((size_t)total, sizeof(double)); /* verbose: GPU drain after Run */
    float* prev_out = NULL;
    size_t prev_cnt = 0;
    int stale = 0; /* outputs identical to the previous iteration's although the input changed */
    int rc = -1;

    for (int it = 0; it < total; it++) {
        int b = it % nbuf;
        /* a different input every iteration: an engine that returned a
         * previous iteration's outputs (asynchronous replay, stale
         * readback) would otherwise be indistinguishable from a correct one */
        uint32_t seed_it = o->seed + (uint32_t)it;
        uint64_t t0 = infer_now_ns();
        if (infer_tensor_gen(ctx, src, &desc, seed_it, &t_src[b]) < 0) {
            res->status = "producer";
            goto out;
        }
        uint64_t t1 = infer_now_ns();
        const struct infer_edge* taken = infer_edge_convert(reg, ctx, &t_src[b], dst, &t_in[b]);
        if (!taken) {
            res->status = "edge";
            goto out;
        }
        res->cost = taken->cost;
        uint64_t t2 = infer_now_ns();
        if (infer_engine_ort_bind_input(eng, &t_in[b]) < 0 || infer_engine_ort_run(eng) < 0) {
            res->status = "run";
            goto out;
        }
        uint64_t t3 = infer_now_ns();
        gen_t[it] = (double)(t1 - t0) / 1e3;
        edge_t[it] = (double)(t2 - t1) / 1e3;
        run_t[it] = (double)(t3 - t2) / 1e3;
        total_t[it] = (double)(t3 - t0) / 1e3;

        /* Run returned with the outputs on the host. Did the GPU actually
         * finish, or is the engine replaying asynchronously? (verbose) */
        if (o->verbose && ep == INFER_EP_WEBGPU && (ctx->have & INFER_WANT_WGPU)) {
            uint64_t d0 = infer_now_ns();
            infer_wgpu_wait_idle(&ctx->wgpu);
            drain_t[it] = (double)(infer_now_ns() - d0) / 1e3;
        }
        {
            size_t cnt;
            const float* out = infer_engine_ort_output(eng, &cnt);
            if (prev_out && cnt == prev_cnt && memcmp(prev_out, out, cnt * sizeof(float)) == 0) {
                stale++;
            }
            if (!prev_out || cnt != prev_cnt) {
                free(prev_out);
                prev_out = malloc(cnt * sizeof(float));
                prev_cnt = cnt;
            }
            memcpy(prev_out, out, cnt * sizeof(float));
        }

        /* diagnostic: the run returned with the outputs on the host, so the
         * GPU has finished everything submitted before the inference --
         * including the edge's release signal. If that fence is still
         * unsignaled now, its signal is being deferred past the inference. */
        if (o->verbose && it == o->warmup && t_src[b].released.sync_fd >= 0) {
            uint64_t w0 = infer_now_ns();
            int already = infer_wait_sync_fd(t_src[b].released.sync_fd, 0) == 0;
            int ok = already || infer_wait_sync_fd(t_src[b].released.sync_fd, 2000) == 0;
            release_lat_us = ok ? (double)(infer_now_ns() - w0) / 1e3 : -2;
            fprintf(stderr,
                    "  release fence after run: %s (%.1f us more)\n",
                    already ? "already signaled" : ok ? "signaled later" : "TIMEOUT",
                    release_lat_us);
        }

        if (it == 0) {
            /* what reached the engine, compared bit for bit with the generator */
            if (infer_tensor_readback(ctx, &t_in[b], check) < 0) {
                res->status = "readback";
                goto out;
            }
            res->input_exact = 1;
            for (size_t i = 0; i < n; i++) {
                float want = infer_gen_value((uint32_t)i, o->seed);
                if (memcmp(&check[i], &want, sizeof(float)) != 0) {
                    res->input_exact = 0;
                    if (o->verbose) {
                        fprintf(stderr, "  input[%zu] = %g, generator says %g\n", i, check[i], want);
                    }
                    break;
                }
            }
            size_t cnt;
            const float* out = infer_engine_ort_output(eng, &cnt);
            if (reference) {
                double err = 0;
                if (cnt != reference_count) {
                    err = INFINITY;
                } else {
                    for (size_t i = 0; i < cnt; i++) {
                        double d = fabs((double)out[i] - (double)reference[i]);
                        if (d > err) { err = d; }
                    }
                }
                res->max_abs_err = err;
            } else {
                res->max_abs_err = 0;
                if (out_copy) {
                    *out_copy = malloc(cnt * sizeof(float));
                    memcpy(*out_copy, out, cnt * sizeof(float));
                }
            }
        }
    }
    if (o->verbose && ep == INFER_EP_WEBGPU) {
        fprintf(stderr,
                "  GPU drain after Run (median): %.1f us  -- the part of the inference Run did not wait for\n",
                median(drain_t + o->warmup, o->iters));
    }
    if (stale) {
        fprintf(stderr, "  STALE: %d iteration(s) returned the previous iteration's outputs\n", stale);
    }
    res->gen_us = median(gen_t + o->warmup, o->iters);
    res->edge_us = median(edge_t + o->warmup, o->iters);
    res->run_us = median(run_t + o->warmup, o->iters);
    res->total_us = median(total_t + o->warmup, o->iters);
    res->status = stale ? "STALE" : "ok";
    rc = stale ? -1 : 0;
out:
    (void)release_lat_us;
    free(drain_t);
    free(prev_out);
    infer_engine_ort_destroy(eng);
    for (int b = 0; b < MAX_BUF; b++) {
        infer_tensor_release(ctx, &t_in[b]);
        infer_tensor_release(ctx, &t_src[b]);
    }
    free(check);
    free(gen_t);
    free(edge_t);
    free(run_t);
    free(total_t);
    return rc;
}

static void print_header(void) {
    printf("%-6s %-7s %-18s %-12s %-6s %-11s %10s %10s %10s %10s  %s\n",
           "src", "ep", "edge", "cost", "input", "max_abs_err", "gen_us", "edge_us", "run_us", "total_us",
           "status");
}

static void print_result(const struct cell_result* r) {
    char err[24];
    if (isnan(r->max_abs_err)) {
        snprintf(err, sizeof err, "-");
    } else {
        snprintf(err, sizeof err, "%.3g", r->max_abs_err);
    }
    printf("%-6s %-7s %-18s %-12s %-6s %-11s %10.1f %10.1f %10.1f %10.1f  %s\n",
           infer_domain_name(r->src),
           infer_ep_name(r->ep),
           r->edge,
           infer_cost_name(r->cost),
           r->status[0] == 'o' ? (r->input_exact ? "exact" : "DIFF") : "-",
           err,
           r->gen_us,
           r->edge_us,
           r->run_us,
           r->total_us,
           r->status);
}

static void usage(void) {
    fprintf(stderr,
            "usage: hello_inference [--src cpu|gl|vk|wgpu|dmabuf] [--ep cpu|webgpu] [--model path]\n"
            "                       [--iters N] [--warmup N] [--seed S] [--all] [--strict] [--verbose]\n"
            "                       [--single]   (one source buffer instead of two in rotation)\n");
}

int main(int argc, char** argv) {
    struct options o = {
        .src = INFER_DOMAIN_CPU,
        .ep = INFER_EP_CPU,
        .model = "models/synthetic_conv.onnx",
        .iters = 20,
        .warmup = 3,
        .buffers = 2,
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
        } else if (!strcmp(a, "--single")) {
            o.buffers = 1;
        } else if (!strcmp(a, "--src") && v) {
            if (infer_domain_parse(v, &o.src) < 0) { usage(); return 2; }
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

    unsigned want = 0;
    if (o.all) {
        want = INFER_WANT_GL | INFER_WANT_VK | INFER_WANT_WGPU | INFER_WANT_DMABUF;
    } else {
        if (o.src != INFER_DOMAIN_CPU) { want |= 1u << o.src; }
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
                    "  %-5s -> %-5s %-18s %-12s %s\n",
                    infer_domain_name(e->src),
                    infer_domain_name(e->dst),
                    e->name,
                    infer_cost_name(e->cost),
                    e->available ? "available" : "unavailable");
        }
    }

    /* the reference: cpu -> cpu EP */
    float* reference = NULL;
    struct cell_result ref_res;
    if (run_cell(&ctx, &reg, &o, INFER_DOMAIN_CPU, INFER_EP_CPU, NULL, 0, &reference, &ref_res) < 0) {
        fprintf(stderr, "reference cell failed: %s\n", ref_res.status);
        infer_ctx_fini_all(&ctx);
        return 1;
    }
    size_t ref_count = 0;
    {
        /* count = the engine's output size; recompute cheaply */
        struct infer_engine_ort* e = infer_engine_ort_create(&ctx, INFER_EP_CPU, o.model);
        struct infer_desc d;
        infer_engine_ort_input_desc(e, &d);
        struct infer_tensor t = {.ready.sync_fd = -1, .released.sync_fd = -1};
        infer_tensor_gen(&ctx, INFER_DOMAIN_CPU, &d, o.seed, &t);
        infer_engine_ort_bind_input(e, &t);
        infer_engine_ort_run(e);
        infer_engine_ort_output(e, &ref_count);
        infer_tensor_release(&ctx, &t);
        infer_engine_ort_destroy(e);
    }

    int failures = 0;
    print_header();
    if (!o.all) {
        struct cell_result r;
        if (o.src == INFER_DOMAIN_CPU && o.ep == INFER_EP_CPU) {
            r = ref_res;
        } else {
            run_cell(&ctx, &reg, &o, o.src, o.ep, reference, ref_count, NULL, &r);
        }
        print_result(&r);
        if (strcmp(r.status, "ok") != 0 || !r.input_exact) { failures++; }
        if (o.strict && r.cost != expected_cost(o.src, o.ep)) { failures++; }
    } else {
        double tol = strstr(o.model, "synthetic") ? 1e-4 : 1e-3;
        for (int ep = 0; ep < INFER_EP_COUNT; ep++) {
            for (int src = 0; src < INFER_DOMAIN_COUNT; src++) {
                struct cell_result r;
                if (src == INFER_DOMAIN_CPU && ep == INFER_EP_CPU) {
                    r = ref_res;
                } else {
                    run_cell(&ctx, &reg, &o, (enum infer_domain)src, (enum infer_ep)ep, reference,
                             ref_count, NULL, &r);
                }
                print_result(&r);
                int bad = strcmp(r.status, "ok") != 0 || !r.input_exact ||
                          !(r.max_abs_err <= tol);
                if (o.strict && r.cost != expected_cost((enum infer_domain)src, (enum infer_ep)ep)) {
                    bad = 1;
                }
                if (bad) { failures++; }
            }
        }
        printf("\ncost classes: ZERO_COPY = handle hand-over, DEVICE_COPY = one GPU pass, "
               "HOST_COPY = through host memory.\n"
               "run_us includes ONNX Runtime's copy of the outputs to the host; GPU producers and\n"
               "edges only submit work, so run_us also absorbs what they left in flight.\n"
               "total_us is the median of per-iteration gen+edge+run: the wall-clock latency from\n"
               "starting to produce the input to having the outputs on the host.\n");
    }
    free(reference);
    infer_ctx_fini_all(&ctx);
    if (failures) {
        fprintf(stderr, "%d cell(s) failed\n", failures);
        return o.strict ? 1 : 0;
    }
    return 0;
}
