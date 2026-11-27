#include "../include/qmx_plugin.h"
#include <stdint.h>
#include <string.h>

#define QMX_BAN_MAX_TOKENS 64

typedef struct {
    int ids[QMX_BAN_MAX_TOKENS];
    int count;
    uint64_t calls;
} QmxTokenBanState;

static QmxTokenBanState g_ban_state;

static void ban_filter_logits(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt, void *ctx) {
    QmxTokenBanState *st = (QmxTokenBanState *)ctx;
    (void)recent;
    (void)recent_cnt;
    if (!st || !logits) return;
    st->calls++;
    for (int i = 0; i < st->count; i++) {
        int id = st->ids[i];
        if (id >= 0 && (uint32_t)id < vocab_size) {
            logits[id] = -1e30f;
        }
    }
}

static void ban_destroy(void *ctx) {
    QmxTokenBanState *st = (QmxTokenBanState *)ctx;
    if (st) memset(st, 0, sizeof(*st));
}

static QmxPluginDesc g_ban_desc = {
    QMX_PLUGIN_ABI_VERSION,
    (uint32_t)sizeof(QmxPluginDesc),
    QMX_PLUGIN_SAMPLER,
    "token_ban",
    "1.3b",
    &g_ban_state,
    ban_destroy,
    { .sampler = { ban_filter_logits } }
};

void qmx_plugin_token_ban_reset(void) {
    memset(&g_ban_state, 0, sizeof(g_ban_state));
}

int qmx_plugin_token_ban_add(int token_id) {
    if (token_id < 0 || g_ban_state.count >= QMX_BAN_MAX_TOKENS) return -1;
    g_ban_state.ids[g_ban_state.count++] = token_id;
    return 0;
}

uint64_t qmx_plugin_token_ban_calls(void) {
    return g_ban_state.calls;
}

const QmxPluginDesc *qmx_plugin_token_ban_desc(void) {
    return &g_ban_desc;
}

#ifdef QMX_PLUGIN_SHARED
QMX_PLUGIN_EXPORT const QmxPluginDesc *qmx_plugin_entry(uint32_t host_abi_version) {
    if (host_abi_version != QMX_PLUGIN_ABI_VERSION) return NULL;
    qmx_plugin_token_ban_reset();
    qmx_plugin_token_ban_add(0);
    return &g_ban_desc;
}
#endif
