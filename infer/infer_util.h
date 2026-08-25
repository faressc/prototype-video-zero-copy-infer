/* infer_util.h -- the small things every infer/ file needs and nothing
 * in the repo shares yet: fail-fast checks, a monotonic clock, a whole-
 * file reader, alignment, and a host-side wait on a sync file.
 *
 * Header-only, static inline, like scenes/mat4.h.
 */
#ifndef HELLO_WAYLAND_INFER_UTIL_H
#define HELLO_WAYLAND_INFER_UTIL_H

#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline void infer_die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

/* Return -1 from the enclosing function with a located message. */
#define INFER_CHECK(cond, ...)                              \
    do {                                                    \
        if (!(cond)) {                                      \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                   \
            fputc('\n', stderr);                            \
            return -1;                                      \
        }                                                   \
    } while (0)

#define VK_CHECK(expr)                                                  \
    do {                                                                \
        VkResult vk_check_r = (expr);                                   \
        if (vk_check_r != VK_SUCCESS) {                                 \
            fprintf(stderr, "%s failed: %d\n", #expr, (int)vk_check_r); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

static inline uint64_t infer_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline size_t infer_align(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

/* Whole file into a malloc'd buffer (SPIR-V, models). NULL on failure. */
static inline void* infer_read_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

/* Block until a sync file (dma-fence fd) signals. -1 = nothing to wait
 * for. Returns 0 when signaled, -1 on timeout/error. */
static inline int infer_wait_sync_fd(int fd, int timeout_ms) {
    if (fd < 0) { return 0; }
    struct pollfd p = {.fd = fd, .events = POLLIN};
    int r = poll(&p, 1, timeout_ms);
    return r == 1 ? 0 : -1;
}

#endif
