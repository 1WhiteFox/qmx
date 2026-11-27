#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _DEFAULT_SOURCE
        #define _DEFAULT_SOURCE
    #endif
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include "../include/qmx_runtime.h"
#include "../include/qmx_kernels.h"
#include <string.h>

#if defined(QMX_ENABLE_THREADS) && !defined(QMX_NO_THREADS) && !defined(QMX_NO_MALLOC) && \
    !defined(QMX_NO_STDIO) && !defined(__DJGPP__) && !defined(QMX_WIN98_COMPAT)
    #define QMX_HAS_AUTOTUNE 1
    #include <stdlib.h>
    #include <stdio.h>
    #if defined(_WIN32)
        #include <windows.h>
    #elif defined(__APPLE__)
        #include <sys/types.h>
        #include <sys/sysctl.h>
        #include <sys/time.h>
        #include <unistd.h>
    #elif defined(__linux__) || defined(__unix__)
        #include <sys/time.h>
        #include <unistd.h>
    #endif
#else
    #define QMX_HAS_AUTOTUNE 0
#endif

#if QMX_HAS_AUTOTUNE

static double qmx_autotune_now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, now;
    if (QueryPerformanceFrequency(&freq) && freq.QuadPart > 0 && QueryPerformanceCounter(&now)) {
        return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
    }
    return (double)GetTickCount();
#else
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec * 0.001;
    }
    return 0.0;
#endif
}

int qmx_detect_physical_cores(void) {
    int cores = 0;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len > 0) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc((size_t)len);
        if (buf) {
            if (GetLogicalProcessorInformation(buf, &len)) {
                DWORD cnt = len / (DWORD)sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
                for (DWORD i = 0; i < cnt; i++) {
                    if (buf[i].Relationship == RelationProcessorCore) cores++;
                }
            }
            free(buf);
        }
    }
    if (cores <= 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        cores = (int)si.dwNumberOfProcessors;
        if (cores > 2) cores /= 2;
    }
#elif defined(__APPLE__)
    int val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &val, &sz, NULL, 0) == 0 && val > 0) {
        cores = val;
    } else if (sysctlbyname("hw.physicalcpu", &val, &sz, NULL, 0) == 0 && val > 0) {
        cores = val;
    }
    if (cores <= 0) {
        long onln = sysconf(_SC_NPROCESSORS_ONLN);
        cores = (onln > 0) ? (int)onln : 1;
    }
#elif defined(__linux__)
    FILE *fp = fopen("/proc/cpuinfo", "rb");
    if (fp) {
        int seen_phys[256];
        int seen_core[256];
        int unique_cnt = 0;
        int cur_phys = -1;
        int cur_core = -1;
        char line[256];
        while (fgets(line, sizeof(line), fp) != NULL) {
            if (line[0] == '\n' || line[0] == '\r') {
                if (cur_phys >= 0 && cur_core >= 0 && unique_cnt < 256) {
                    int dup = 0;
                    for (int k = 0; k < unique_cnt; k++) {
                        if (seen_phys[k] == cur_phys && seen_core[k] == cur_core) {
                            dup = 1;
                            break;
                        }
                    }
                    if (!dup) {
                        seen_phys[unique_cnt] = cur_phys;
                        seen_core[unique_cnt] = cur_core;
                        unique_cnt++;
                    }
                }
                cur_phys = -1;
                cur_core = -1;
            } else if (strncmp(line, "physical id", 11) == 0) {
                const char *colon = strchr(line, ':');
                if (colon) cur_phys = (int)strtol(colon + 1, NULL, 10);
            } else if (strncmp(line, "core id", 7) == 0) {
                const char *colon = strchr(line, ':');
                if (colon) cur_core = (int)strtol(colon + 1, NULL, 10);
            }
        }
        if (cur_phys >= 0 && cur_core >= 0 && unique_cnt < 256) {
            int dup = 0;
            for (int k = 0; k < unique_cnt; k++) {
                if (seen_phys[k] == cur_phys && seen_core[k] == cur_core) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) unique_cnt++;
        }
        fclose(fp);
        cores = unique_cnt;
    }
    if (cores <= 0) {
        long onln = sysconf(_SC_NPROCESSORS_ONLN);
        cores = (onln > 0) ? (int)onln : 1;
        if (cores > 2) cores /= 2;
    }
#else
    long onln = sysconf(_SC_NPROCESSORS_ONLN);
    cores = (onln > 0) ? (int)onln : 1;
#endif
    if (cores < 1) cores = 1;
    if (cores > QMX_MAX_THREADS) cores = QMX_MAX_THREADS;
    return cores;
}

uint64_t qmx_detect_avail_ram_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = (DWORD)sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        return (uint64_t)ms.ullAvailPhys;
    }
    return 0u;
#elif defined(__APPLE__)
    uint64_t mem = 0u;
    size_t sz = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &sz, NULL, 0) == 0) {
        return (mem * 3u) / 4u;
    }
    return 0u;
#elif defined(__linux__)
    FILE *fp = fopen("/proc/meminfo", "rb");
    if (!fp) return 0u;
    uint64_t avail_kb = 0u;
    uint64_t free_kb = 0u;
    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            avail_kb = (uint64_t)strtoull(line + 13, NULL, 10);
            break;
        } else if (strncmp(line, "MemFree:", 8) == 0) {
            free_kb = (uint64_t)strtoull(line + 8, NULL, 10);
        }
    }
    fclose(fp);
    uint64_t kb = avail_kb ? avail_kb : free_kb;
    return kb * 1024ULL;
#else
    return 0u;
#endif
}

int qmx_recommend_threads(const QmxModel *model) {
    int phys = qmx_detect_physical_cores();
    if (!model) return phys;
    uint64_t ffn_elems = (uint64_t)model->header.hidden_dim * (uint64_t)model->header.intermediate_dim;
    if (ffn_elems < 65536ULL) return 1;
    if (ffn_elems < 524288ULL) return (phys > 4) ? 4 : phys;
    return phys;
}

static double qmx_bench_candidate_ms(QmxBackend b, uint32_t rows, uint32_t cols, int iters,
                                     const uint8_t *w_data, const float *w_scales,
                                     const float *x_in, float *out,
                                     int8_t *xq, int16_t *xq16, int8_t *xq_eo,
                                     int32_t *x_sums, float *x_scales) {
    const QmxBackendOps *ops = qmx_get_backend_ops(b);
    if (!ops || ops == &qmx_ops_scalar) return 1e30;

    QmxTensor w;
    memset(&w, 0, sizeof(w));
    w.data       = w_data;
    w.scales     = w_scales;
    w.rows       = rows;
    w.cols       = cols;
    w.quant_type = QMX_QUANT_Q4_ROW32;
    w.tile_rows  = QMX_TILE_ROWS;

    ops->prepare_act(x_in, xq, xq16, x_scales, cols);
    if (ops->prepare_act_eo) ops->prepare_act_eo(xq, xq_eo, x_sums, cols);

    /* Warmup */
    for (int i = 0; i < 4; i++) {
        if (ops->matmul_tiled_eo) ops->matmul_tiled_eo(out, xq, xq_eo, x_sums, x_scales, &w, 0);
        else ops->matmul_tiled(out, xq, xq16, x_scales, &w, 0);
    }

    double best_ms = 1e30;
    volatile float sink = 0.0f;
    for (int r = 0; r < 3; r++) {
        double t0 = qmx_autotune_now_ms();
        for (int i = 0; i < iters; i++) {
            if (ops->matmul_tiled_eo) ops->matmul_tiled_eo(out, xq, xq_eo, x_sums, x_scales, &w, 0);
            else ops->matmul_tiled(out, xq, xq16, x_scales, &w, 0);
            sink += out[0];
        }
        double dt = qmx_autotune_now_ms() - t0;
        if (dt > 0.0 && dt < best_ms) best_ms = dt;
    }
    (void)sink;
    return best_ms;
}

QmxBackend qmx_autotune_backend(uint32_t dim, uint32_t inter_dim) {
    (void)inter_dim;
    QmxBackend def_b = qmx_detect_backend();
    QmxBackend cand_a = QMX_BACKEND_AUTO;
    QmxBackend cand_b = QMX_BACKEND_AUTO;

    if (def_b == QMX_BACKEND_AVX512 && qmx_cpu_has_avx2() &&
        qmx_get_backend_ops(QMX_BACKEND_AVX2) != &qmx_ops_scalar) {
        cand_a = QMX_BACKEND_AVX512;
        cand_b = QMX_BACKEND_AVX2;
    } else if (def_b == QMX_BACKEND_NEON_DOT && qmx_cpu_has_neon() &&
               qmx_get_backend_ops(QMX_BACKEND_NEON) != &qmx_ops_scalar) {
        cand_a = QMX_BACKEND_NEON_DOT;
        cand_b = QMX_BACKEND_NEON;
    } else {
        return def_b;
    }

    int bucket = (dim >= 1024u) ? 1 : 0;
    static int cached_valid[2] = { 0, 0 };
    static QmxBackend cached_best[2] = { QMX_BACKEND_AUTO, QMX_BACKEND_AUTO };
    if (cached_valid[bucket]) return cached_best[bucket];

    uint32_t rows = 256u;
    uint32_t cols = bucket ? 1024u : 512u;
    int iters     = bucket ? 32 : 64;
    uint32_t n_tr = rows / QMX_TILE_ROWS;
    uint32_t n_tc = cols / QMX_TILE_COLS;
    uint32_t nt   = n_tr * n_tc;

    uint8_t  *w_data   = (uint8_t *)malloc((size_t)nt * 256u);
    uint16_t *w_scales = (uint16_t *)malloc((size_t)nt * QMX_TILE_ROWS * sizeof(uint16_t));
    float    *x_in     = (float *)malloc((size_t)cols * sizeof(float));
    float    *out      = (float *)malloc((size_t)rows * sizeof(float));
    int8_t   *xq       = (int8_t *)malloc((size_t)cols);
    int16_t  *xq16     = (int16_t *)malloc((size_t)cols * sizeof(int16_t));
    int8_t   *xq_eo    = (int8_t *)malloc((size_t)cols);
    int32_t  *x_sums   = (int32_t *)malloc((size_t)n_tc * sizeof(int32_t));
    float    *x_scales = (float *)malloc((size_t)n_tc * sizeof(float));

    if (!w_data || !w_scales || !x_in || !out || !xq || !xq16 || !xq_eo || !x_sums || !x_scales) {
        free(w_data); free(w_scales); free(x_in); free(out);
        free(xq); free(xq16); free(xq_eo); free(x_sums); free(x_scales);
        return def_b;
    }

    uint32_t rng = 0x517CC1B7u;
    for (size_t i = 0; i < (size_t)nt * 256u; i++) {
        rng = rng * 1664525u + 1013904223u;
        w_data[i] = (uint8_t)(rng >> 24u);
    }
    for (size_t i = 0; i < (size_t)nt * QMX_TILE_ROWS; i++) {
        w_scales[i] = 0x3400u; /* 0.25f */
    }
    for (uint32_t i = 0; i < cols; i++) {
        rng = rng * 1664525u + 1013904223u;
        x_in[i] = ((float)((int32_t)(rng >> 8u) - 8388608)) / 8388608.0f;
    }

    double ms_a = qmx_bench_candidate_ms(cand_a, rows, cols, iters, w_data, (const float *)(const void *)w_scales,
                                         x_in, out, xq, xq16, xq_eo, x_sums, x_scales);
    double ms_b = qmx_bench_candidate_ms(cand_b, rows, cols, iters, w_data, (const float *)(const void *)w_scales,
                                         x_in, out, xq, xq16, xq_eo, x_sums, x_scales);

    free(w_data); free(w_scales); free(x_in); free(out);
    free(xq); free(xq16); free(xq_eo); free(x_sums); free(x_scales);

    QmxBackend best = (ms_b * 1.02 < ms_a) ? cand_b : cand_a;
    cached_best[bucket]  = best;
    cached_valid[bucket] = 1;
    return best;
}

#else

int qmx_detect_physical_cores(void) {
    return 1;
}

uint64_t qmx_detect_avail_ram_bytes(void) {
    return 0u;
}

int qmx_recommend_threads(const QmxModel *model) {
    (void)model;
    return 1;
}

QmxBackend qmx_autotune_backend(uint32_t dim, uint32_t inter_dim) {
    (void)dim;
    (void)inter_dim;
    return qmx_detect_backend();
}

#endif
