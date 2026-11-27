#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _DEFAULT_SOURCE
        #define _DEFAULT_SOURCE
    #endif
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include "../include/qmx_plugin.h"
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
    #include <windows.h>
#elif !defined(__DJGPP__) && !defined(QMX_NO_STDIO)
    #include <sys/time.h>
#endif

#define QMX_PROF_MAX_LAYERS 128

typedef struct {
    uint64_t step_count;
    uint64_t token_count;
    uint64_t layer_calls[QMX_PROF_MAX_LAYERS];
    double   layer_ms[QMX_PROF_MAX_LAYERS];
    double   cur_start_ms;
    uint32_t max_layer_seen;
} QmxLayerProfilerState;

static QmxLayerProfilerState g_prof_state;

static double prof_now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    if (QueryPerformanceFrequency(&f) && f.QuadPart > 0 && QueryPerformanceCounter(&t)) {
        return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
    }
    return 0.0;
#elif !defined(__DJGPP__) && !defined(QMX_NO_STDIO)
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec * 0.001;
    }
    return 0.0;
#else
    return 0.0;
#endif
}

static void prof_on_event(QmxPluginEventKind ev, uint32_t layer_idx, int batch_size, int pos, int token_id, void *ctx) {
    QmxLayerProfilerState *st = (QmxLayerProfilerState *)ctx;
    (void)batch_size;
    (void)pos;
    (void)token_id;
    if (!st) return;
    if (ev == QMX_HOOK_LAYER_BEGIN) {
        st->cur_start_ms = prof_now_ms();
    } else if (ev == QMX_HOOK_LAYER_END) {
        if (layer_idx < QMX_PROF_MAX_LAYERS) {
            double dt = prof_now_ms() - st->cur_start_ms;
            if (dt > 0.0) st->layer_ms[layer_idx] += dt;
            st->layer_calls[layer_idx]++;
            if (layer_idx + 1u > st->max_layer_seen) st->max_layer_seen = layer_idx + 1u;
        }
    } else if (ev == QMX_HOOK_STEP_END) {
        st->step_count++;
    } else if (ev == QMX_HOOK_TOKEN_SAMPLED) {
        st->token_count++;
    }
}

static void prof_destroy(void *ctx) {
    QmxLayerProfilerState *st = (QmxLayerProfilerState *)ctx;
    if (st) memset(st, 0, sizeof(*st));
}

static QmxPluginDesc g_prof_desc = {
    QMX_PLUGIN_ABI_VERSION,
    (uint32_t)sizeof(QmxPluginDesc),
    QMX_PLUGIN_HOOK,
    "layer_profiler",
    "1.3b",
    &g_prof_state,
    prof_destroy,
    { .hook = { prof_on_event } }
};

const QmxPluginDesc *qmx_plugin_profiler_desc(void) {
    memset(&g_prof_state, 0, sizeof(g_prof_state));
    return &g_prof_desc;
}

uint64_t qmx_plugin_profiler_steps(void) {
    return g_prof_state.step_count;
}

uint64_t qmx_plugin_profiler_tokens(void) {
    return g_prof_state.token_count;
}

uint64_t qmx_plugin_profiler_layer_calls(uint32_t layer_idx) {
    return (layer_idx < QMX_PROF_MAX_LAYERS) ? g_prof_state.layer_calls[layer_idx] : 0u;
}

#ifdef QMX_PLUGIN_SHARED
QMX_PLUGIN_EXPORT const QmxPluginDesc *qmx_plugin_entry(uint32_t host_abi_version) {
    if (host_abi_version != QMX_PLUGIN_ABI_VERSION) return NULL;
    return qmx_plugin_profiler_desc();
}
#endif
